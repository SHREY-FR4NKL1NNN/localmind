"""Client for the Llama 3.2 model served locally by Ollama.

Llama 3.2 plays two roles in LocalMind's tiered, Mixture-of-Experts-inspired
design: it is the fast/trivial *expert* for the simplest sub-tasks, and it is
also the *gating network* itself — the gate (``gate.py``) calls this client to
decompose an incoming query into sub-tasks. It is the smallest and fastest of
the four models, hence the short timeout. Like the other clients, this never
raises — connection problems and timeouts are caught and returned as a
structured error dict so the router/gate stay resilient.
"""

from __future__ import annotations

import os
import time

import requests

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


def generate(
    prompt: str,
    options: dict | None = None,
    response_format: dict | str | None = None,
) -> dict:
    """Generate a completion from Llama 3.2 via Ollama.

    Sends a non-streaming request to the local Ollama ``/api/generate``
    endpoint. On success returns ``{"response", "latency_ms", "model"}``. On
    any connection error or timeout returns the same shape with an additional
    ``error`` key and an empty ``response`` — this function never raises.

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
        resp = requests.post(GENERATE_URL, json=payload, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "response": (data.get("response") or "").strip(),
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
        }
    except requests.exceptions.Timeout:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "response": "",
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
            "error": f"{DISPLAY_NAME} request timed out after {TIMEOUT_SECONDS}s.",
        }
    except requests.exceptions.RequestException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "response": "",
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
            "error": f"Could not reach Ollama for {DISPLAY_NAME}: {exc}",
        }
