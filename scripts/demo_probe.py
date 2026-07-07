#!/usr/bin/env python3
"""Probe LocalMind's decomposed SSE endpoint to find queries that genuinely fan
out across multiple experts — the ones worth using in a demo (so the router looks
like a router, not a plain chatbot lighting a single expert).

For each query it POSTs to ``/query/decomposed/stream``, parses the SSE events,
and reports: subtask count, the distinct experts activated, the sparsity ratio,
and whether the combiner ran. It flags "fan-out winners" (sparsity >= 0.5, the
combiner ran, and >= 2 distinct experts).

Usage:
    python scripts/demo_probe.py                      # probe local (127.0.0.1:8000)
    python scripts/demo_probe.py --target <url>       # probe a deployment
    python scripts/demo_probe.py --query "..."        # probe one custom query
    LOCALMIND_URL=<url> python scripts/demo_probe.py   # target via env

Note: this needs LIVE inference (local Ollama, or a hosted provider on the
target). Multi-domain prompts (code + reasoning, or a trivial + an analytical
ask, or text + an image) tend to fan out; a single-topic prompt usually does not.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

import httpx

DEFAULT_TARGET = os.environ.get("LOCALMIND_URL", "http://127.0.0.1:8000")

# Candidates biased toward multi-domain asks that should split across experts.
CANDIDATE_QUERIES = [
    "What is 2+2, and also compare transformer architectures to state space "
    "models for edge inference?",
    "Write a Python function to reverse a linked list, and explain the "
    "time-complexity tradeoffs of doing it iteratively versus recursively.",
    "Explain how RSA encryption works, write pseudocode for key generation, and "
    "note the main security caveats.",
    "Design a REST API for a todo app, and analyze the tradeoffs between SQL and "
    "NoSQL for storing the todos.",
    "Give me the one-word capital of France, and separately reason step by step "
    "through the Monty Hall problem.",
]


def _parse_sse_block(raw: str):
    event = "message"
    data_lines: list[str] = []
    for line in raw.split("\n"):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    try:
        return event, json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None


async def probe(client: httpx.AsyncClient, target: str, query: str):
    subtask_count = 0
    experts: list[str] = []
    sparsity_ratio = 0.0
    combiner_ran = False
    async with client.stream(
        "POST",
        f"{target}/query/decomposed/stream",
        json={"query": query, "image_base64": None},
        timeout=240,
    ) as resp:
        resp.raise_for_status()
        buffer = ""
        async for chunk in resp.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                parsed = _parse_sse_block(block)
                if not parsed:
                    continue
                event, data = parsed
                if event == "gate_complete":
                    subtask_count = len(data.get("subtasks", []))
                elif event == "sparsity":
                    experts = data.get("activated_expert_names", [])
                    sparsity_ratio = float(data.get("sparsity_ratio", 0.0))
                elif event == "combiner_token":
                    combiner_ran = True  # only emitted when the combiner runs
                elif event == "done":
                    return subtask_count, experts, sparsity_ratio, combiner_ran
    return subtask_count, experts, sparsity_ratio, combiner_ran


async def main() -> None:
    ap = argparse.ArgumentParser(description="Find fan-out demo queries for LocalMind.")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="backend base URL")
    ap.add_argument(
        "--query", action="append", help="probe a specific query (repeatable)"
    )
    args = ap.parse_args()
    queries = args.query or CANDIDATE_QUERIES

    print(f"Probing {args.target}\n")
    print(f"{'':4}{'sub':>3} {'sparsity':>8} {'combiner':>8}  experts · query")
    print("-" * 88)
    winners = []
    async with httpx.AsyncClient() as client:
        for q in queries:
            try:
                subs, experts, ratio, comb = await probe(client, args.target, q)
            except Exception as exc:  # noqa: BLE001 — probe should never crash the run
                print(f"ERR  {exc}  · {q[:56]}")
                continue
            distinct = sorted(set(experts))
            win = ratio >= 0.5 and comb and len(distinct) >= 2
            if win:
                winners.append((q, subs, distinct, ratio))
            mark = "WIN " if win else "    "
            print(
                f"{mark}{subs:>3} {ratio:>8.2f} {str(comb):>8}  "
                f"{','.join(distinct) or '-'} · {q[:52]}"
            )

    print("\n=== fan-out winners (sparsity >= 0.5, combiner ran, >= 2 experts) ===")
    if not winners:
        print("(none this run — try more multi-domain prompts, or run vs live inference)")
    for q, subs, experts, ratio in winners:
        print(f"- sparsity {ratio:.2f}, {subs} subtasks, experts={experts}\n    {q}")


if __name__ == "__main__":
    asyncio.run(main())
