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
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from statistics import mean

import combiner
import gate
from classifier import classify
from log_config import get_logger
from logger import decision_log
from models import deepseek_client, llama32_client, llava_client, mistral_client

logger = get_logger("router")

# Maps an expert name to its streaming client coroutine. The vision client also
# takes an image; all four share the ``stream(prompt, image_base64=None)`` shape.
_STREAM_EXPERT_CLIENTS = {
    "llama3.2": llama32_client.stream,
    "mistral": mistral_client.stream,
    "deepseek-r1:7b": deepseek_client.stream,
    "llava": llava_client.stream,
}

# Maps a gate-assigned expert name to the client coroutine that serves it. Only
# the vision client takes an image, so it is dispatched separately below.
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
    logger.info(
        "Single-route query (complexity=%.2f, privacy=%.2f) -> %s",
        classification["complexity"],
        classification["privacy"],
        route,
    )

    if route == "mistral":
        result = await mistral_client.generate(query)
    else:
        result = await deepseek_client.generate(query)

    latency_ms = int(result.get("latency_ms", 0))
    failed = "error" in result
    if failed:
        logger.warning("Single-route %s call failed: %s", route, result.get("error"))

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

    # Persist a query_log row (single-route maps to a one-expert, non-decomposed
    # query so it shows up in history/stats alongside the decomposed flow).
    expert = "mistral" if route == "mistral" else "deepseek-r1:7b"
    decision_log.log(
        {
            "timestamp": decision["timestamp"],
            "query": query,
            "decomposed": False,
            "subtask_count": 1,
            "experts_activated": [expert],
            "vision_activated": False,
            "combined_response": decision["response"],
            "combiner_skipped": True,
            "total_latency_ms": latency_ms,
            "sparsity_ratio": round(1 / EXPERTS_AVAILABLE, 3),
        }
    )
    logger.info("Single-route response from %s in %d ms", expert, latency_ms)
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
    logger.info(
        "Decomposed query into %d sub-task(s) (decomposed=%s, image=%s)",
        len(sub_tasks),
        decomposed,
        has_image,
    )

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

    combiner_result = await combiner.combine(query, subtask_results)
    total_latency_ms = int((time.perf_counter() - batch_start) * 1000)

    # Sparsity: how few of the available experts this query actually activated.
    activated = sorted({entry["expert"] for entry in subtask_results})
    sparsity = _sparsity(set(activated))
    logger.info(
        "Decomposed query complete: experts=%s, total_latency=%d ms",
        activated,
        total_latency_ms,
    )
    decision_log.log(
        {
            "timestamp": timestamp,
            "query": query,
            "decomposed": decomposed,
            "subtask_count": len(subtask_results),
            "experts_activated": activated,
            "vision_activated": sparsity["vision_activated"],
            "combined_response": combiner_result["combined_response"],
            "combiner_skipped": combiner_result["skipped"],
            "total_latency_ms": total_latency_ms,
            "sparsity_ratio": sparsity["sparsity_ratio"],
        }
    )

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


def _sparsity(activated_experts: set[str]) -> dict:
    """Build the sparsity metric from the set of experts that actually ran."""
    activated = sorted(activated_experts)
    return {
        "experts_activated": len(activated),
        "experts_available": EXPERTS_AVAILABLE,
        "sparsity_ratio": round(len(activated) / EXPERTS_AVAILABLE, 3),
        "vision_activated": "llava" in activated,
        "activated_expert_names": activated,
    }


async def stream_decomposed(
    query: str, image_base64: str | None
) -> AsyncGenerator[dict, None]:
    """Decompose, gate, then stream every expert concurrently, then combine.

    The streaming counterpart of ``route_decomposed``. Yields transport-agnostic
    event dicts ``{"event": str, "data": dict}`` that the API layer formats as
    SSE. Event order:

    1. ``gate_complete`` — the per-sub-task routing decisions, emitted *before*
       any expert runs so the UI can render the routing immediately.
    2. ``expert_token`` — one per streamed token, tagged with ``subtask_index``;
       these **interleave** across experts because all experts stream at once.
    3. ``expert_done`` — once per expert when its stream finishes.
    4. ``sparsity`` — emitted after every expert has finished.
    5. ``combiner_token`` — the synthesis, streamed token-by-token (skipped when
       the combiner is skipped).
    6. ``done`` — terminal summary.

    Concurrency uses a fan-in pattern: one producer task per expert pushes its
    chunks onto a single shared ``asyncio.Queue``, and this generator drains the
    queue, emitting each chunk the instant it arrives in whatever order the
    experts produce them. A per-producer sentinel signals completion. A failing
    expert is isolated — its error surfaces as a token and ``expert_done`` but
    never cancels the others.
    """
    has_image = image_base64 is not None
    sub_tasks = await gate.decompose(query, has_image)
    scores = [
        gate.gate_score(st["subtask"], st["depends_on_image"]) for st in sub_tasks
    ]

    yield {
        "event": "gate_complete",
        "data": {
            "subtasks": [
                {
                    "subtask_index": i,
                    "subtask": st["subtask"],
                    "expert": sc["expert"],
                    "complexity": sc["complexity"],
                    "privacy": sc["privacy"],
                    "reasoning": sc["reasoning"],
                    "hard_routed": sc["hard_routed"],
                    "depth": st.get("depth", 0),
                }
                for i, (st, sc) in enumerate(zip(sub_tasks, scores))
            ]
        },
    }

    timestamp = datetime.now(timezone.utc).isoformat()
    batch_start = time.perf_counter()
    queue: asyncio.Queue = asyncio.Queue()

    async def _produce(idx: int, expert: str, subtask: str) -> None:
        """Stream one expert, pushing tagged chunks then a sentinel onto the queue."""
        start = time.perf_counter()
        collected: list[str] = []
        errored: str | None = None
        client = _STREAM_EXPERT_CLIENTS.get(expert)
        try:
            if client is None:
                errored = f"No streaming client for expert '{expert}'."
            else:
                async for chunk in client(subtask, image_base64):
                    if chunk.get("error"):
                        errored = chunk["error"]
                        await queue.put(
                            {
                                "kind": "token",
                                "idx": idx,
                                "expert": expert,
                                "token": f"[error] {chunk['error']}",
                                "is_thinking": False,
                            }
                        )
                    else:
                        token = chunk.get("token", "")
                        # The final answer excludes DeepSeek's reasoning trace.
                        if not chunk.get("is_thinking", False):
                            collected.append(token)
                        await queue.put(
                            {
                                "kind": "token",
                                "idx": idx,
                                "expert": expert,
                                "token": token,
                                "is_thinking": bool(chunk.get("is_thinking", False)),
                            }
                        )
                    if chunk.get("done"):
                        break
        except Exception as exc:  # noqa: BLE001 — isolate one expert's failure
            errored = str(exc)
            await queue.put(
                {
                    "kind": "token",
                    "idx": idx,
                    "expert": expert,
                    "token": f"[error] {exc}",
                    "is_thinking": False,
                }
            )
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            full = f"[error] {errored}" if errored else "".join(collected)
            await queue.put(
                {
                    "kind": "done",
                    "idx": idx,
                    "expert": expert,
                    "full_response": full,
                    "latency_ms": latency_ms,
                }
            )
            await queue.put({"kind": "sentinel"})

    producers = [
        asyncio.create_task(_produce(i, sc["expert"], st["subtask"]))
        for i, (st, sc) in enumerate(zip(sub_tasks, scores))
    ]

    full_responses: dict[int, str] = {}
    latencies: dict[int, int] = {}
    try:
        remaining = len(producers)
        while remaining > 0:
            item = await queue.get()
            kind = item["kind"]
            if kind == "sentinel":
                remaining -= 1
            elif kind == "token":
                yield {
                    "event": "expert_token",
                    "data": {
                        "subtask_index": item["idx"],
                        "expert": item["expert"],
                        "token": item["token"],
                        "is_thinking": item["is_thinking"],
                    },
                }
            elif kind == "done":
                full_responses[item["idx"]] = item["full_response"]
                latencies[item["idx"]] = item["latency_ms"]
                yield {
                    "event": "expert_done",
                    "data": {
                        "subtask_index": item["idx"],
                        "expert": item["expert"],
                        "full_response": item["full_response"],
                        "latency_ms": item["latency_ms"],
                    },
                }
    finally:
        # If the consumer is abandoned (client disconnect), don't leak producers.
        for task in producers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producers, return_exceptions=True)

    # Assemble the per-sub-task results (now that every expert has finished) for
    # logging, the combiner, and the sparsity metric.
    subtask_results: list[dict] = []
    for i, (st, sc) in enumerate(zip(sub_tasks, scores)):
        entry = {
            "subtask": st["subtask"],
            "expert": sc["expert"],
            "complexity": sc["complexity"],
            "privacy": sc["privacy"],
            "reasoning": sc["reasoning"],
            "hard_routed": sc["hard_routed"],
            "response": full_responses.get(i, ""),
            "latency_ms": latencies.get(i, 0),
            "depth": st.get("depth", 0),
        }
        subtask_results.append(entry)

    sparsity = _sparsity({entry["expert"] for entry in subtask_results})
    yield {"event": "sparsity", "data": sparsity}

    # Combiner: stream the synthesis (or carry the lone answer when skipped).
    combined_parts: list[str] = []
    combined_response = ""
    combiner_skipped = False
    combiner_latency_ms = 0
    async for chunk in combiner.combine_stream(query, subtask_results):
        if chunk["done"]:
            combiner_skipped = bool(chunk.get("skipped", False))
            combiner_latency_ms = int(chunk.get("combiner_latency_ms", 0))
            if combiner_skipped:
                combined_response = chunk["token"]
            else:
                if chunk["token"]:
                    combined_parts.append(chunk["token"])
                    yield {"event": "combiner_token", "data": {"token": chunk["token"]}}
                combined_response = "".join(combined_parts)
            break
        combined_parts.append(chunk["token"])
        yield {"event": "combiner_token", "data": {"token": chunk["token"]}}

    total_latency_ms = int((time.perf_counter() - batch_start) * 1000)

    decomposed = not (
        len(sub_tasks) == 1 and sub_tasks[0]["subtask"].strip() == query.strip()
    )
    decision_log.log(
        {
            "timestamp": timestamp,
            "query": query,
            "decomposed": decomposed,
            "subtask_count": len(subtask_results),
            "experts_activated": sparsity["activated_expert_names"],
            "vision_activated": sparsity["vision_activated"],
            "combined_response": combined_response,
            "combiner_skipped": combiner_skipped,
            "total_latency_ms": total_latency_ms,
            "sparsity_ratio": sparsity["sparsity_ratio"],
        }
    )
    logger.info(
        "Streamed query complete: experts=%s, total_latency=%d ms",
        sparsity["activated_expert_names"],
        total_latency_ms,
    )

    yield {
        "event": "done",
        "data": {
            "combined_response": combined_response,
            "total_latency_ms": total_latency_ms,
            "combiner_skipped": combiner_skipped,
            "combiner_latency_ms": combiner_latency_ms,
        },
    }
