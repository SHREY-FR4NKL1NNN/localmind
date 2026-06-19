"""Async client for the Mistral 7B model served locally by Ollama.

Mistral 7B is LocalMind's fast, lightweight path: it handles simple and
moderately complex queries with low latency and modest compute. This client
never raises — connection problems and timeouts are caught and returned as a
structured error dict so the router stays resilient. HTTP is performed with
``httpx.AsyncClient`` so expert calls can be awaited concurrently.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator

import httpx

# Streaming keeps the connection open for the whole generation, so it uses a
# generous fixed timeout rather than the short non-streaming budget above.
STREAM_TIMEOUT = httpx.Timeout(120.0)

# Default to the IPv4 loopback explicitly: on Windows, "localhost" can resolve
# to IPv6 (::1) first and stall for seconds before falling back to IPv4, while
# Ollama listens on 127.0.0.1. Override via OLLAMA_BASE_URL if needed.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL_NAME = "mistral"
DISPLAY_NAME = "Mistral 7B"
TIMEOUT_SECONDS = 30


async def generate(prompt: str) -> dict:
    """Generate a completion from Mistral 7B via Ollama.

    Sends a non-streaming request to the local Ollama ``/api/generate``
    endpoint using ``httpx.AsyncClient``. On success returns
    ``{"response", "latency_ms", "model"}``. On any connection error or timeout
    returns the same shape with an additional ``error`` key and an empty
    ``response`` — this coroutine never raises.
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(
                GENERATE_URL,
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            )
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
    """Stream a completion from Mistral 7B token-by-token via Ollama.

    Opens a streaming ``POST`` to Ollama's ``/api/generate`` with
    ``stream: true`` and yields one ``{"token", "done", "model"}`` dict per
    decoded chunk. ``image_base64`` is accepted for signature parity with the
    vision client but ignored here (Mistral is text-only). Empty/garbled lines
    are skipped silently. On a connection error or timeout this yields exactly
    one terminal error chunk and stops — it never raises.
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
