"""Async client for the Llama 3.2 model served locally by Ollama.

Llama 3.2 plays two roles in LocalMind's tiered, Mixture-of-Experts-inspired
design: it is the fast/trivial *expert* for the simplest sub-tasks, and it is
also the *gating network* itself — the gate (``gate.py``) calls this client to
decompose an incoming query into sub-tasks. It is the smallest and fastest of
the four models, hence the short timeout. Like the other clients, this never
raises — connection problems and timeouts are caught and returned as a
structured error dict so the router/gate stay resilient.

HTTP is performed with ``httpx.AsyncClient`` so the decomposed flow can run
several experts concurrently with ``asyncio.gather`` (see ``router.py``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator

import httpx

# Streaming reads can legitimately run much longer than a single non-streaming
# call (the connection stays open for the whole generation), so streaming uses a
# generous fixed timeout rather than the short per-call budget above.
STREAM_TIMEOUT = httpx.Timeout(120.0)

# Default to the IPv4 loopback explicitly: on Windows, "localhost" can resolve
# to IPv6 (::1) first and stall for seconds before falling back to IPv4, while
# Ollama listens on 127.0.0.1. Override via OLLAMA_BASE_URL if needed.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL_NAME = "llama3.2"
DISPLAY_NAME = "Llama 3.2"
# Llama 3.2 is the smallest model, but it is the busiest one in the decomposed
# flow: it serves as both the gate (decomposition) AND the fast expert, so a
# single decomposed query can issue several llama3.2 calls that Ollama queues
# behind one another. On a memory-constrained GPU Ollama also swaps models in
# and out, adding cold-load time to a queued call. A 20s budget was too tight
# under that contention (observed timeouts on trivial sub-tasks); 45s gives the
# queued/cold calls headroom while still failing fast on a genuinely stuck call.
TIMEOUT_SECONDS = 45


async def generate(
    prompt: str,
    options: dict | None = None,
    response_format: dict | str | None = None,
) -> dict:
    """Generate a completion from Llama 3.2 via Ollama.

    Sends a non-streaming request to the local Ollama ``/api/generate``
    endpoint using ``httpx.AsyncClient``. On success returns
    ``{"response", "latency_ms", "model"}``. On any connection error or timeout
    returns the same shape with an additional ``error`` key and an empty
    ``response`` — this coroutine never raises.

    Two optional knobs support the gate's structured-decomposition use:
    ``options`` is forwarded as Ollama generation options (e.g.
    ``{"temperature": 0}`` for deterministic output), and ``response_format`` is
    forwarded as Ollama's ``format`` field — either ``"json"`` or a JSON schema
    dict to constrain the output. Both default to ``None`` so ordinary expert
    calls behave exactly like the other model clients.
    """
    payload: dict = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if options is not None:
        payload["options"] = options
    if response_format is not None:
        payload["format"] = response_format

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(GENERATE_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "response": (data.get("response") or "").strip(),
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
        }
    except httpx.TimeoutException:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "response": "",
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
            "error": f"{DISPLAY_NAME} request timed out after {TIMEOUT_SECONDS}s.",
        }
    except httpx.HTTPError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "response": "",
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
            "error": f"Could not reach Ollama for {DISPLAY_NAME}: {exc}",
        }


async def stream(
    prompt: str, image_base64: str | None = None
) -> AsyncGenerator[dict, None]:
    """Stream a completion from Llama 3.2 token-by-token via Ollama.

    Opens a streaming ``POST`` to Ollama's ``/api/generate`` with
    ``stream: true`` and yields one ``{"token", "done", "model"}`` dict per
    decoded chunk. ``image_base64`` is accepted for signature parity with the
    vision client but ignored here (Llama 3.2 is text-only). Empty/garbled lines
    are skipped silently. On a connection error or timeout this yields exactly
    one terminal error chunk ``{"token": "", "done": True, "model", "error"}``
    and stops — it never raises.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
    try:
        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
            async with client.stream("POST", GENERATE_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    done = bool(data.get("done", False))
                    yield {
                        "token": data.get("response", ""),
                        "done": done,
                        "model": MODEL_NAME,
                    }
                    if done:
                        return
    except httpx.TimeoutException:
        yield {
            "token": "",
            "done": True,
            "model": MODEL_NAME,
            "error": f"{DISPLAY_NAME} stream timed out after 120s.",
        }
    except httpx.HTTPError as exc:
        yield {
            "token": "",
            "done": True,
            "model": MODEL_NAME,
            "error": f"Could not reach Ollama for {DISPLAY_NAME}: {exc}",
        }
