"""In-memory routing decision log with computed statistics.

LocalMind keeps its routing history in a bounded, thread-safe in-memory buffer
rather than a database. This keeps the system dependency-free and instantly
inspectable for a demo while still supporting rich aggregate statistics. The
buffer holds at most ``MAX_ENTRIES`` decisions and evicts the oldest first
(FIFO) once full.
"""

from __future__ import annotations

from collections import Counter, deque
from statistics import mean
from threading import Lock

MAX_ENTRIES = 200


class DecisionLog:
    """A bounded, thread-safe FIFO log of routing decisions."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        """Create a decision log holding at most ``max_entries`` decisions."""
        self._entries: deque[dict] = deque(maxlen=max_entries)
        # Sub-task decisions from the decomposed (MoE) flow are kept in a
        # separate buffer so the existing single-route /history and /stats
        # endpoints are completely unaffected by their differing shape.
        # Unified reporting across both is a later upgrade.
        self._subtask_entries: deque[dict] = deque(maxlen=max_entries)
        # Lifetime tally of expert activations across every decomposed query.
        # Unlike the bounded buffers above this is a running count that is never
        # evicted, so it reflects total usage since process start — the basis
        # for the per-expert utilisation reported by /expert-stats.
        self._expert_activations: Counter[str] = Counter()
        self._lock = Lock()

    def log(self, decision: dict) -> None:
        """Append a single-route routing decision, evicting oldest if full."""
        with self._lock:
            self._entries.append(dict(decision))

    def log_subtask(self, decision: dict) -> None:
        """Append a decomposed-query sub-task decision to the sub-task buffer.

        Stored separately from single-route decisions so it cannot perturb the
        existing /stats and /history results. Accepts the per-sub-task decision
        dict produced by ``router.route_decomposed``.
        """
        with self._lock:
            self._subtask_entries.append(dict(decision))
            # One logged sub-task == one expert activation. Count it here so the
            # tally stays consistent with what actually ran.
            expert = decision.get("expert")
            if expert:
                self._expert_activations[expert] += 1

    def get_expert_activation_stats(self) -> dict:
        """Return per-expert activation counts and their share of the total.

        Aggregates the lifetime activation tally across all decomposed queries.
        Returns ``{"total_activations": int, "experts": {name: {"count": int,
        "pct": float}}}`` where ``pct`` is each expert's percentage of all
        activations (0.0 when nothing has run yet).
        """
        with self._lock:
            counts = dict(self._expert_activations)
        total = sum(counts.values())
        experts = {
            name: {
                "count": count,
                "pct": round(100.0 * count / total, 1) if total else 0.0,
            }
            for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        }
        return {"total_activations": total, "experts": experts}

    def get_subtask_history(self, n: int = 50) -> list[dict]:
        """Return the most recent ``n`` sub-task decisions, newest first."""
        with self._lock:
            items = list(self._subtask_entries)
        return list(reversed(items[-n:]))

    def get_history(self, n: int = 50) -> list[dict]:
        """Return the most recent ``n`` decisions, newest first."""
        with self._lock:
            items = list(self._entries)
        return list(reversed(items[-n:]))

    def get_stats(self) -> dict:
        """Compute aggregate statistics over the entire log.

        Statistics are derived from every retained decision (up to
        ``MAX_ENTRIES``), not just the slice returned by ``get_history``.
        Returns the stats dict consumed by the ``/stats`` endpoint.
        """
        with self._lock:
            items = list(self._entries)

        total = len(items)
        mistral = [e for e in items if e.get("route") == "mistral"]
        deepseek = [e for e in items if e.get("route") == "deepseek"]

        mistral_latencies = [
            e["latency_ms"] for e in mistral if e.get("latency_ms")
        ]
        deepseek_latencies = [
            e["latency_ms"] for e in deepseek if e.get("latency_ms")
        ]
        complexities = [e["complexity"] for e in items if "complexity" in e]
        privacies = [e["privacy"] for e in items if "privacy" in e]
        total_saved = sum(int(e.get("compute_saved_ms", 0)) for e in items)

        def pct(count: int) -> float:
            return round(100.0 * count / total, 1) if total else 0.0

        def avg(values: list[float], digits: int = 1) -> float:
            return round(mean(values), digits) if values else 0.0

        return {
            "total_queries": total,
            "mistral_count": len(mistral),
            "deepseek_count": len(deepseek),
            "mistral_pct": pct(len(mistral)),
            "deepseek_pct": pct(len(deepseek)),
            "total_compute_saved_ms": total_saved,
            "avg_latency_mistral_ms": avg(mistral_latencies),
            "avg_latency_deepseek_ms": avg(deepseek_latencies),
            "avg_complexity": avg(complexities, 3),
            "avg_privacy": avg(privacies, 3),
        }


# Module-level singleton shared across the router and the API layer.
decision_log = DecisionLog()
