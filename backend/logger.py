"""SQLite-backed routing decision log for LocalMind.

Every query (single-route or decomposed) is persisted as one row in the
``query_log`` table so history, aggregate stats, and per-expert utilisation
survive restarts and can be exported for demos. Aggregates are computed with SQL
(``COUNT``/``AVG``/``SUM``), not Python loops.

Concurrency: FastAPI's async handlers (and, historically, thread-pool workers)
may touch the log from different threads, so each operation opens its own
short-lived connection with ``check_same_thread=False`` and a process-wide lock
serialises writes. This keeps the standard-library ``sqlite3`` module safe here
without pulling in SQLAlchemy or a connection pool.
"""

from __future__ import annotations

import json
import os
import sqlite3
from threading import Lock

from log_config import get_logger

logger = get_logger("logger")

# Database file lives next to the backend by default; overridable for tests/CI
# via LOCALMIND_DB (e.g. a temp file) so they never touch the real database.
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "localmind.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  query TEXT NOT NULL,
  decomposed INTEGER NOT NULL,
  subtask_count INTEGER NOT NULL,
  experts_activated TEXT NOT NULL,
  vision_activated INTEGER NOT NULL,
  combined_response TEXT,
  combiner_skipped INTEGER NOT NULL,
  total_latency_ms INTEGER NOT NULL,
  sparsity_ratio REAL NOT NULL
)
"""


class DecisionLog:
    """Durable, thread-safe query log over SQLite ``query_log``."""

    def __init__(self, db_path: str | None = None) -> None:
        """Create the log, ensuring the database file and schema exist."""
        self._db_path = db_path or os.environ.get("LOCALMIND_DB", DEFAULT_DB_PATH)
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)
        logger.info("Decision log ready at %s", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection usable from any thread, rows as dict-likes."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, decision: dict) -> None:
        """Insert one row summarising a completed query.

        ``decision`` carries: ``timestamp``, ``query``, ``decomposed`` (bool),
        ``subtask_count``, ``experts_activated`` (list of expert tags),
        ``vision_activated`` (bool), ``combined_response``, ``combiner_skipped``
        (bool), ``total_latency_ms``, ``sparsity_ratio``. Missing keys fall back
        to sensible defaults so either flow can log without bespoke shaping.
        """
        experts = decision.get("experts_activated", []) or []
        row = (
            decision.get("timestamp", ""),
            decision.get("query", ""),
            int(bool(decision.get("decomposed", False))),
            int(decision.get("subtask_count", len(experts) or 1)),
            json.dumps(experts),
            int(bool(decision.get("vision_activated", False))),
            decision.get("combined_response", ""),
            int(bool(decision.get("combiner_skipped", True))),
            int(decision.get("total_latency_ms", 0)),
            float(decision.get("sparsity_ratio", 0.0)),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_log (
                  timestamp, query, decomposed, subtask_count, experts_activated,
                  vision_activated, combined_response, combiner_skipped,
                  total_latency_ms, sparsity_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Turn a query_log row into the JSON-friendly dict the API returns."""
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "query": row["query"],
            "decomposed": bool(row["decomposed"]),
            "subtask_count": row["subtask_count"],
            "experts_activated": json.loads(row["experts_activated"]),
            "vision_activated": bool(row["vision_activated"]),
            "combined_response": row["combined_response"],
            "combiner_skipped": bool(row["combiner_skipped"]),
            "total_latency_ms": row["total_latency_ms"],
            "sparsity_ratio": row["sparsity_ratio"],
        }

    def get_history(self, n: int = 50) -> list[dict]:
        """Return the most recent ``n`` query rows, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Compute aggregate statistics over every row via SQL aggregates."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  COUNT(*)                          AS total_queries,
                  COALESCE(SUM(decomposed), 0)      AS decomposed_queries,
                  COALESCE(SUM(vision_activated), 0) AS vision_queries,
                  COALESCE(AVG(total_latency_ms), 0) AS avg_total_latency_ms,
                  COALESCE(AVG(sparsity_ratio), 0)   AS avg_sparsity_ratio,
                  COALESCE(AVG(subtask_count), 0)    AS avg_subtask_count
                FROM query_log
                """
            ).fetchone()
        total = row["total_queries"]
        return {
            "total_queries": total,
            "decomposed_queries": row["decomposed_queries"],
            "single_route_queries": total - row["decomposed_queries"],
            "vision_queries": row["vision_queries"],
            "avg_total_latency_ms": round(row["avg_total_latency_ms"], 1),
            "avg_sparsity_ratio": round(row["avg_sparsity_ratio"], 3),
            "avg_subtask_count": round(row["avg_subtask_count"], 2),
        }

    def get_expert_activation_stats(self) -> dict:
        """Tally per-expert activations by parsing the experts_activated column.

        Returns ``{"total_activations": int, "experts": {name: {"count": int,
        "pct": float}}}``; each expert is counted once per query that activated
        it (the distinct set stored per row).
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT experts_activated FROM query_log").fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            for expert in json.loads(r["experts_activated"]):
                counts[expert] = counts.get(expert, 0) + 1
        total = sum(counts.values())
        experts = {
            name: {
                "count": count,
                "pct": round(100.0 * count / total, 1) if total else 0.0,
            }
            for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        }
        return {"total_activations": total, "experts": experts}

    def export_json(self, path: str) -> None:
        """Write every logged query to ``path`` as a JSON array (oldest first)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM query_log ORDER BY id ASC").fetchall()
        records = [self._row_to_dict(r) for r in rows]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        logger.info("Exported %d query rows to %s", len(records), path)


# Module-level singleton shared across the router and the API layer.
decision_log = DecisionLog()
