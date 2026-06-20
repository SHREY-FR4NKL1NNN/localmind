"""Combiner / synthesis step for LocalMind's decomposed (MoE-inspired) flow.

After the gate routes each sub-task to an expert and the experts run in
parallel, this module fuses the individual expert answers into one coherent
reply to the user's original request.

This is a **TEXT SYNTHESIS** step, *not* a literal Mixture-of-Experts
weighted-sum. A real MoE layer combines its experts' outputs as numeric vectors
(a gate-weighted sum in latent space) and is trained end-to-end. That is
impossible here: our "experts" are full language models that emit complete
natural-language text, not comparable hidden-state vectors, so there is nothing
to weight-sum. Instead we ask a small model to read the sub-answers and write a
single unified response. This is the one deliberate, unavoidable deviation from
textbook MoE in LocalMind's design — the routing/sparsity behaviour is faithful;
the combination is necessarily textual.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator

from log_config import get_logger
from models import llama32_client

logger = get_logger("combiner")

# The combiner is served by Llama 3.2: the smallest/fastest model is enough to
# stitch a handful of short answers into one reply, and keeping it off the
# heavier models keeps the extra synthesis latency low.
COMBINER_MODEL = "llama3.2"


def _skip(combined_response: str) -> dict:
    """Build a combiner result for the case where no fusion is performed."""
    return {
        "combined_response": combined_response,
        "combiner_latency_ms": 0,
        "combiner_model": COMBINER_MODEL,
        "skipped": True,
    }


def _build_prompt(original_query: str, usable: list[dict]) -> str:
    """Build the synthesis prompt fed to Llama 3.2.

    Each usable sub-answer is labelled; results from the vision expert are
    labelled ``Image analysis:`` so the combiner knows they describe an attached
    image rather than answering a text sub-task. Shared by both ``combine`` and
    ``combine_stream`` so the two paths fuse answers identically.
    """
    blocks = []
    for i, r in enumerate(usable, start=1):
        if r.get("expert") == "llava":
            blocks.append(f"Image analysis: {r['response']}")
        else:
            blocks.append(f"Sub-task {i}: {r['subtask']}\nAnswer: {r['response']}")
    answers = "\n\n".join(blocks)
    return (
        "Several sub-tasks of a single user request were answered separately "
        "below. Combine them into ONE coherent, well-organised reply to the "
        "original request. Do not mention that the work was split up, and do "
        "not refer to 'sub-tasks' or which model produced what — just give the "
        "unified answer.\n\n"
        f"Original request: {original_query}\n\n{answers}\n\nUnified answer:"
    )


async def combine(original_query: str, subtask_results: list[dict]) -> dict:
    """Fuse per-sub-task answers into one unified response via Llama 3.2.

    ``subtask_results`` is the list of per-sub-task entries produced by the
    router (each carries at least ``subtask`` and ``response``). Returns
    ``{"combined_response", "combiner_latency_ms", "combiner_model", "skipped"}``.

    The combiner is skipped (``skipped=True``, zero latency) when there is
    nothing to fuse: exactly one sub-task result (its lone answer already *is*
    the reply), or fewer than two usable (non-``[error]``) answers. Otherwise
    Llama 3.2 writes a single coherent reply to ``original_query`` and
    ``skipped`` is False. Like the model clients, this never raises — a failed
    synthesis call surfaces an ``[error]`` prefix in ``combined_response``.
    """
    # Exactly one sub-task: its answer is the whole response, nothing to combine.
    if len(subtask_results) <= 1:
        only = subtask_results[0]["response"] if subtask_results else ""
        return _skip(only)

    usable = [r for r in subtask_results if not r["response"].startswith("[error]")]
    if len(usable) < 2:
        # Not enough successful answers to fuse — return the lone usable answer
        # (or, if every expert failed, the first error) without a combiner call.
        lone = usable[0]["response"] if usable else subtask_results[0]["response"]
        return _skip(lone)

    prompt = _build_prompt(original_query, usable)

    start = time.perf_counter()
    result = await llama32_client.generate(prompt)
    combiner_latency_ms = int((time.perf_counter() - start) * 1000)

    if "error" in result:
        combined_response = f"[error] {result['error']}"
    else:
        combined_response = result.get("response", "")

    return {
        "combined_response": combined_response,
        "combiner_latency_ms": combiner_latency_ms,
        "combiner_model": COMBINER_MODEL,
        "skipped": False,
    }


async def combine_stream(
    original_query: str, subtask_results: list[dict]
) -> AsyncGenerator[dict, None]:
    """Stream the combiner's unified answer token-by-token via Llama 3.2.

    The streaming counterpart of ``combine``. Yields ``{"token", "done"}`` dicts;
    the terminal chunk (``done=True``) additionally carries ``skipped`` and
    ``combiner_latency_ms``.

    When there is nothing to fuse — exactly one sub-task result, or fewer than
    two usable (non-``[error]``) answers — Llama 3.2 is **not** called: a single
    terminal chunk is yielded carrying the lone answer as its ``token`` with
    ``skipped=True``. Otherwise the sub-answers are synthesised and the model's
    chunks are forwarded as they arrive. Like the clients, this never raises — a
    failed synthesis surfaces an ``[error]`` prefix in the terminal token.
    """
    if len(subtask_results) <= 1:
        only = subtask_results[0]["response"] if subtask_results else ""
        yield {"token": only, "done": True, "skipped": True, "combiner_latency_ms": 0}
        return

    usable = [r for r in subtask_results if not r["response"].startswith("[error]")]
    if len(usable) < 2:
        lone = usable[0]["response"] if usable else subtask_results[0]["response"]
        yield {"token": lone, "done": True, "skipped": True, "combiner_latency_ms": 0}
        return

    prompt = _build_prompt(original_query, usable)
    start = time.perf_counter()
    async for chunk in llama32_client.stream(prompt):
        if chunk.get("error"):
            combiner_latency_ms = int((time.perf_counter() - start) * 1000)
            yield {
                "token": f"[error] {chunk['error']}",
                "done": True,
                "skipped": False,
                "combiner_latency_ms": combiner_latency_ms,
            }
            return
        if chunk["done"]:
            combiner_latency_ms = int((time.perf_counter() - start) * 1000)
            yield {
                "token": chunk["token"],
                "done": True,
                "skipped": False,
                "combiner_latency_ms": combiner_latency_ms,
            }
            return
        yield {"token": chunk["token"], "done": False}
