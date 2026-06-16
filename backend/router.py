"""Routing orchestration for LocalMind.

Ties the pieces together for a single query: classify it, dispatch it to the
chosen local model, estimate the compute saved versus always using the heavier
model, log the decision, and return a unified response dict.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from statistics import mean

from classifier import classify
from logger import decision_log
from models import deepseek_client, mistral_client

# Default baseline used before any DeepSeek R1 latency has been observed. The
# compute-saved metric compares a Mistral run against what DeepSeek R1 would
# likely have cost; until we have real data we assume 8 seconds.
DEFAULT_DEEPSEEK_LATENCY_MS = 8000

# Rolling window of recent DeepSeek R1 latencies used to refine the baseline.
_recent_deepseek_latencies: deque[int] = deque(maxlen=20)


def _baseline_latency_ms() -> int:
    """Estimate how long this query would have taken on DeepSeek R1.

    Uses the rolling average of recently observed DeepSeek R1 latencies, or a
    sensible default if none have been recorded yet.
    """
    if _recent_deepseek_latencies:
        return int(mean(_recent_deepseek_latencies))
    return DEFAULT_DEEPSEEK_LATENCY_MS


def handle_query(query: str) -> dict:
    """Classify, route, execute, and log a single query.

    Returns the unified router response dict containing the model output, the
    chosen route, the classification scores and reasoning, latency, the
    estimated compute saved, and a UTC timestamp. If the underlying model call
    fails, an ``error`` key is included and the response text is empty.
    """
    classification = classify(query)
    route = classification["route"]

    if route == "mistral":
        result = mistral_client.generate(query)
    else:
        result = deepseek_client.generate(query)

    latency_ms = int(result.get("latency_ms", 0))
    failed = "error" in result

    # Feed successful DeepSeek R1 latencies back into the rolling baseline.
    if route == "deepseek" and not failed and latency_ms > 0:
        _recent_deepseek_latencies.append(latency_ms)

    # Compute saved is only meaningful when we used the lighter model.
    if route == "mistral" and not failed:
        compute_saved_ms = max(0, _baseline_latency_ms() - latency_ms)
    else:
        compute_saved_ms = 0

    decision = {
        "query": query,
        "response": result.get("response", ""),
        "route": route,
        "reasoning": classification["reasoning"],
        "complexity": classification["complexity"],
        "privacy": classification["privacy"],
        "latency_ms": latency_ms,
        "model": result.get("model", route),
        "compute_saved_ms": compute_saved_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if failed:
        decision["error"] = result["error"]

    decision_log.log(decision)
    return decision
