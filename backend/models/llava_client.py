"""Async client for the LLaVA vision model served locally by Ollama.

LLaVA is LocalMind's *vision expert*: any sub-task that depends on an image is
hard-routed here by the gate. It accepts an optional base64-encoded image which
is passed to Ollama under the multimodal ``images`` field. Its timeout is the
most generous of the four models because vision inference is the heaviest. Like
the other clients, this never raises — failures are returned as a structured
error dict so the router/gate degrade gracefully. HTTP is performed with
``httpx.AsyncClient`` so it can be awaited alongside the text experts.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator

import httpx

# Streaming keeps the connection open for the whole (heavy) vision generation,
# so it uses a generous fixed timeout rather than the non-streaming budget above.
STREAM_TIMEOUT = httpx.Timeout(120.0)

# Default to the IPv4 loopback explicitly: on Windows, "localhost" can resolve
# to IPv6 (::1) first and stall for seconds before falling back to IPv4, while
# Ollama listens on 127.0.0.1. Override via OLLAMA_BASE_URL if needed.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
# Vision expert: MiniCPM-V 2.6 (8B). Upgraded from llava:7b (Q4_0) for much
# stronger fine-grained recognition / logo+text reading. (Llama 3.2 Vision was
# evaluated but its `mllama` architecture won't load on this Ollama runner.)
# Shared by both generate() and stream().
MODEL_NAME = "minicpm-v"
DISPLAY_NAME = "MiniCPM-V"
TIMEOUT_SECONDS = 45


async def generate(prompt: str, image_base64: str | None = None) -> dict:
    """Generate a completion from LLaVA via Ollama, optionally over an image.

    Sends a non-streaming request to the local Ollama ``/api/generate``
    endpoint using ``httpx.AsyncClient``. When ``image_base64`` is provided it is
    attached under Ollama's multimodal ``images`` field (a list of
    base64-encoded strings). On success returns
    ``{"response", "latency_ms", "model"}``. On any connection error or timeout
    returns the same shape with an additional ``error`` key and an empty
    ``response`` — this coroutine never raises.
    """
    payload: dict = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if image_base64:
        payload["images"] = [image_base64]

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
    """Stream a vision completion from LLaVA token-by-token via Ollama.

    Opens a streaming ``POST`` to Ollama's ``/api/generate`` with
    ``stream: true``; when ``image_base64`` is provided it is attached under
    Ollama's multimodal ``images`` field. Yields one ``{"token", "done",
    "model"}`` dict per decoded chunk. Empty/garbled lines are skipped silently.
    On a connection error or timeout this yields exactly one terminal error
    chunk and stops — it never raises.
    """
    payload: dict = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
    if image_base64:
        payload["images"] = [image_base64]
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
