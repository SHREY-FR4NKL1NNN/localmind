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

from models import llama32_client

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
        f"Original request: {original_query}\n\n{answers}\n\nUnified answer:"
    )

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
