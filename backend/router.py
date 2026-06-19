"""Routing orchestration for LocalMind.

Ties the pieces together for a single query. Two flows live here:

* ``handle_query`` — the original single-route flow: classify a query, dispatch
  it to one chosen model, estimate compute saved, log the decision, and return a
  unified response dict. Left behaviourally unchanged (now ``async`` because the
  model clients are async).
* ``route_decomposed`` — the tiered, Mixture-of-Experts-inspired flow: ask the
  gate to decompose the query into sub-tasks (recursively, where a sub-task is
  itself compound), score each sub-task to an expert, run the selected experts
  **concurrently** with ``asyncio.gather``, and finally **combine** the
  per-sub-task answers (see ``combiner.py``) into one unified response.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from statistics import mean

import combiner
import gate
from classifier import classify
from logger import decision_log
from models import deepseek_client, llama32_client, llava_client, mistral_client

# Maps a gate-assigned expert name to the client coroutine that serves it. Only
# the LLaVA client takes an image, so it is dispatched separately below.
_TEXT_EXPERT_CLIENTS = {
    "llama3.2": llama32_client.generate,
    "mistral": mistral_client.generate,
    "deepseek-r1:7b": deepseek_client.generate,
}

# Total number of distinct experts in the tiered system. Used to compute the
# sparsity ratio (how few of the available experts a given query activates).
EXPERTS_AVAILABLE = 4

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


async def handle_query(query: str) -> dict:
    """Classify, route, execute, and log a single query.

    Returns the unified router response dict containing the model output, the
    chosen route, the classification scores and reasoning, latency, the
    estimated compute saved, and a UTC timestamp. If the underlying model call
    fails, an ``error`` key is included and the response text is empty.
    """
    classification = classify(query)
    route = classification["route"]

    if route == "mistral":
        result = await mistral_client.generate(query)
    else:
        result = await deepseek_client.generate(query)

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


async def _run_expert(expert: str, subtask: str, image_base64: str | None) -> dict:
    """Execute one expert on one sub-task, returning ``{response, latency_ms}``.

    Dispatches to the client coroutine for ``expert`` (the vision expert
    additionally receives ``image_base64``). The underlying clients never raise;
    if a call returns an error it is surfaced as a clear string in ``response``
    so a single failing sub-task cannot crash the whole decomposed query.
    """
    if expert == "llava":
        result = await llava_client.generate(subtask, image_base64)
    else:
        client = _TEXT_EXPERT_CLIENTS.get(expert)
        if client is None:
            return {
                "response": f"[error] No client registered for expert '{expert}'.",
                "latency_ms": 0,
            }
        result = await client(subtask)

    if "error" in result:
        return {"response": f"[error] {result['error']}", "latency_ms": result.get("latency_ms", 0)}
    return {"response": result.get("response", ""), "latency_ms": int(result.get("latency_ms", 0))}


async def route_decomposed(query: str, image_base64: str | None) -> dict:
    """Decompose a query, gate each sub-task to an expert, run them, combine.

    Implements the full tiered, MoE-inspired flow:

    1. **Decompose** — the gate splits ``query`` into sub-tasks (recursively,
       where a sub-task is itself compound).
    2. **Gate** — each sub-task is scored to exactly one expert (sparse
       activation: only the chosen experts run).
    3. **Execute concurrently** — the selected experts run together via
       ``asyncio.gather`` (``return_exceptions=True`` so one failing expert never
       cancels the others); result order matches the sub-task order.
    4. **Combine** — ``combiner.combine`` fuses the answers into one unified
       response (skipped when there is only one sub-task).

    Each sub-task's routing decision is logged. Returns a dict with the original
    query, whether decomposition happened, the per-sub-task list, the legacy
    ``synthesis`` object (for the dashboard), the combiner fields
    (``combined_response``/``combiner_skipped``/``combiner_latency_ms``), the
    ``sparsity`` metric, the wall-clock ``total_latency_ms`` for the parallel
    batch plus combiner, and a timestamp.
    """
    has_image = image_base64 is not None
    sub_tasks = await gate.decompose(query, has_image)

    # "Decomposed" is False only when the gate produced a single sub-task that is
    # just the original query (the simple/fallback path).
    decomposed = not (
        len(sub_tasks) == 1 and sub_tasks[0]["subtask"].strip() == query.strip()
    )
    timestamp = datetime.now(timezone.utc).isoformat()

    # Score every sub-task first (cheap, pure-CPU heuristic), then dispatch the
    # expert calls concurrently. gate_score is deterministic and order-stable, so
    # zipping the gathered results back to the scores preserves sub-task order.
    scores = [
        gate.gate_score(st["subtask"], st["depends_on_image"]) for st in sub_tasks
    ]

    # Wall-clock timing starts here: it spans the whole parallel expert batch
    # plus the combiner, so total_latency_ms reflects what the caller actually
    # waited. With true parallelism it tracks the SLOWEST expert (+ combiner),
    # not the sum of the experts.
    batch_start = time.perf_counter()
    gathered = await asyncio.gather(
        *(
            _run_expert(score["expert"], st["subtask"], image_base64)
            for st, score in zip(sub_tasks, scores)
        ),
        return_exceptions=True,
    )
    # return_exceptions=True means a crashing coroutine yields an Exception
    # object instead of propagating; normalise those into error results so one
    # bad expert cannot break the response. (The clients never raise, so this is
    # belt-and-suspenders.)
    executions = [
        ex
        if not isinstance(ex, BaseException)
        else {"response": f"[error] expert crashed: {ex}", "latency_ms": 0}
        for ex in gathered
    ]

    subtask_results: list[dict] = []
    for sub_task, score, execution in zip(sub_tasks, scores, executions):
        entry = {
            "subtask": sub_task["subtask"],
            "expert": score["expert"],
            "complexity": score["complexity"],
            "privacy": score["privacy"],
            "reasoning": score["reasoning"],
            "hard_routed": score["hard_routed"],
            "response": execution["response"],
            "latency_ms": execution["latency_ms"],
            "depth": sub_task.get("depth", 0),
        }
        subtask_results.append(entry)
        decision_log.log_subtask({"query": query, "timestamp": timestamp, **entry})

    combiner_result = await combiner.combine(query, subtask_results)
    total_latency_ms = int((time.perf_counter() - batch_start) * 1000)

    # Sparsity: how few of the available experts this query actually activated.
    activated = sorted({entry["expert"] for entry in subtask_results})
    sparsity = {
        "experts_activated": len(activated),
        "experts_available": EXPERTS_AVAILABLE,
        "sparsity_ratio": round(len(activated) / EXPERTS_AVAILABLE, 3),
        "vision_activated": "llava" in activated,
        "activated_expert_names": activated,
    }

    # Legacy dashboard shape: DecomposedPanel renders `synthesis` when present.
    # It is None whenever the combiner was skipped.
    synthesis = (
        None
        if combiner_result["skipped"]
        else {
            "response": combiner_result["combined_response"],
            "model": combiner_result["combiner_model"],
            "latency_ms": combiner_result["combiner_latency_ms"],
        }
    )

    return {
        "query": query,
        "decomposed": decomposed,
        "subtasks": subtask_results,
        "synthesis": synthesis,
        "combined_response": combiner_result["combined_response"],
        "combiner_skipped": combiner_result["skipped"],
        "combiner_latency_ms": combiner_result["combiner_latency_ms"],
        "sparsity": sparsity,
        "total_latency_ms": total_latency_ms,
        "timestamp": timestamp,
    }
