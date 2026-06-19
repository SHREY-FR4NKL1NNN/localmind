"""Routing orchestration for LocalMind.

Ties the pieces together for a single query. Two flows live here:

* ``handle_query`` — the original single-route flow: classify a query, dispatch
  it to one chosen model, estimate compute saved, log the decision, and return a
  unified response dict. Left untouched for backward compatibility.
* ``route_decomposed`` — the tiered, Mixture-of-Experts-inspired flow: ask the
  gate to decompose the query into sub-tasks (recursively, where a sub-task is
  itself compound), score each sub-task to an expert, run the selected experts
  **in parallel**, and finally **synthesize** the per-sub-task answers into one
  unified response.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from statistics import mean

import gate
from classifier import classify
from logger import decision_log
from models import deepseek_client, llama32_client, llava_client, mistral_client

# Maps a gate-assigned expert name to the client callable that serves it. Only
# the LLaVA client takes an image, so it is dispatched separately below.
_TEXT_EXPERT_CLIENTS = {
    "llama3.2": llama32_client.generate,
    "mistral": mistral_client.generate,
    "deepseek-r1:7b": deepseek_client.generate,
}

# Default baseline used before any DeepSeek R1 latency has been observed. The
# compute-saved metric compares a Mistral run against what DeepSeek R1 would
# likely have cost; until we have real data we assume 8 seconds.
DEFAULT_DEEPSEEK_LATENCY_MS = 8000

# Rolling window of recent DeepSeek R1 latencies used to refine the baseline.
_recent_deepseek_latencies: deque[int] = deque(maxlen=20)

# Upper bound on concurrent expert calls in the decomposed flow. The model
# clients are blocking I/O (HTTP to Ollama), so threads overlap their network
# waits well. The cap keeps us from hammering Ollama with more simultaneous
# generations than it can usefully serve. It is deliberately conservative: on a
# memory-constrained GPU the experts are *different* models that do not all fit
# in VRAM at once, so Ollama serialises them by swapping models in and out.
# Issuing many concurrent calls then just deepens that queue and pushes the
# slowest queued call past its client timeout (observed live), without buying
# real parallelism. A small cap keeps the queue shallow while still overlapping
# the cases that genuinely can run together.
MAX_PARALLEL_EXPERTS = 2

# The synthesis/combiner step is served by Mistral: the general-purpose expert
# is a good fit for fusing several short answers into one coherent reply without
# the latency of the reasoning model.
SYNTHESIS_MODEL = "mistral"


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


def _run_expert(expert: str, subtask: str, image_base64: str | None) -> dict:
    """Execute one expert on one sub-task, returning ``{response, latency_ms}``.

    Dispatches to the client for ``expert`` (the vision expert additionally
    receives ``image_base64``). The underlying clients never raise; if a call
    returns an error it is surfaced as a clear string in ``response`` so a single
    failing sub-task cannot crash the whole decomposed query.
    """
    if expert == "llava":
        result = llava_client.generate(subtask, image_base64)
    else:
        client = _TEXT_EXPERT_CLIENTS.get(expert)
        if client is None:
            return {
                "response": f"[error] No client registered for expert '{expert}'.",
                "latency_ms": 0,
            }
        result = client(subtask)

    if "error" in result:
        return {"response": f"[error] {result['error']}", "latency_ms": result.get("latency_ms", 0)}
    return {"response": result.get("response", ""), "latency_ms": int(result.get("latency_ms", 0))}


def _synthesize(query: str, subtask_results: list[dict]) -> dict | None:
    """Combine per-sub-task answers into one unified response via Mistral.

    The combiner step in the MoE-inspired flow: it fuses the individual expert
    answers into a single coherent reply to the user's original request, without
    referring to the sub-tasks or experts. Returns ``{"response", "model",
    "latency_ms"}`` (the ``response`` carries an ``[error]`` prefix if the
    synthesis call itself failed), or ``None`` when synthesis is not warranted —
    i.e. there are fewer than two successfully-answered sub-tasks, in which case
    the lone answer already is the response and combining would add nothing.
    """
    usable = [r for r in subtask_results if not r["response"].startswith("[error]")]
    if len(usable) < 2:
        return None

    answers = "\n\n".join(
        f"Sub-task {i}: {r['subtask']}\nAnswer: {r['response']}"
        for i, r in enumerate(usable, start=1)
    )
    prompt = (
        "Several sub-tasks of a single user request were answered separately "
        "below. Combine them into ONE coherent, well-organised reply to the "
        "original request. Do not mention that the work was split up, and do "
        "not refer to 'sub-tasks' or which model produced what — just give the "
        "unified answer.\n\n"
        f"Original request: {query}\n\n{answers}\n\nUnified answer:"
    )
    result = mistral_client.generate(prompt)
    latency_ms = int(result.get("latency_ms", 0))
    if "error" in result:
        return {
            "response": f"[error] {result['error']}",
            "model": SYNTHESIS_MODEL,
            "latency_ms": latency_ms,
        }
    return {
        "response": result.get("response", ""),
        "model": SYNTHESIS_MODEL,
        "latency_ms": latency_ms,
    }


def route_decomposed(query: str, image_base64: str | None) -> dict:
    """Decompose a query, gate each sub-task to an expert, run them, synthesize.

    Implements the full tiered, MoE-inspired flow:

    1. **Decompose** — the gate splits ``query`` into sub-tasks (recursively,
       where a sub-task is itself compound).
    2. **Gate** — each sub-task is scored to exactly one expert (sparse
       activation: only the chosen experts run).
    3. **Execute in parallel** — the selected experts run concurrently on a
       thread pool (bounded by ``MAX_PARALLEL_EXPERTS``), overlapping their
       Ollama network waits; result order matches the sub-task order.
    4. **Synthesize** — when more than one sub-task was answered, a combiner
       step fuses the answers into one unified response.

    Each sub-task's routing decision is logged. Returns a dict with the original
    query, whether decomposition actually happened, a per-sub-task list of
    ``{subtask, expert, complexity, privacy, reasoning, hard_routed, response,
    latency_ms, depth}``, the ``synthesis`` result (or ``None``), and a
    timestamp.
    """
    has_image = image_base64 is not None
    sub_tasks = gate.decompose(query, has_image)

    # "Decomposed" is False only when the gate produced a single sub-task that is
    # just the original query (the simple/fallback path).
    decomposed = not (
        len(sub_tasks) == 1 and sub_tasks[0]["subtask"].strip() == query.strip()
    )
    timestamp = datetime.now(timezone.utc).isoformat()

    # Score every sub-task first (cheap, pure-CPU heuristic), then dispatch the
    # expert calls concurrently. gate_score is deterministic and order-stable, so
    # zipping the futures back to the scores preserves sub-task order.
    scores = [
        gate.gate_score(st["subtask"], st["depends_on_image"]) for st in sub_tasks
    ]
    with ThreadPoolExecutor(
        max_workers=min(MAX_PARALLEL_EXPERTS, len(sub_tasks))
    ) as pool:
        futures = [
            pool.submit(_run_expert, score["expert"], st["subtask"], image_base64)
            for st, score in zip(sub_tasks, scores)
        ]
        executions = [f.result() for f in futures]

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

    synthesis = _synthesize(query, subtask_results) if decomposed else None

    return {
        "query": query,
        "decomposed": decomposed,
        "subtasks": subtask_results,
        "synthesis": synthesis,
        "timestamp": timestamp,
    }
