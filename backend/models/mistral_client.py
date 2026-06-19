"""Async client for the Mistral 7B model served locally by Ollama.

Mistral 7B is LocalMind's fast, lightweight path: it handles simple and
moderately complex queries with low latency and modest compute. This client
never raises — connection problems and timeouts are caught and returned as a
structured error dict so the router stays resilient. HTTP is performed with
``httpx.AsyncClient`` so expert calls can be awaited concurrently.
"""

from __future__ import annotations

import os
import time

import httpx

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
