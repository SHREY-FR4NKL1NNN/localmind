"""Client for the LLaVA vision model served locally by Ollama.

LLaVA is LocalMind's *vision expert*: any sub-task that depends on an image is
hard-routed here by the gate. It accepts an optional base64-encoded image which
is passed to Ollama under the multimodal ``images`` field. Its timeout is the
most generous of the four models because vision inference is the heaviest. Like
the other clients, this never raises — failures are returned as a structured
error dict so the router/gate degrade gracefully.
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
MODEL_NAME = "llava:latest"
DISPLAY_NAME = "LLaVA"
TIMEOUT_SECONDS = 45


def generate(prompt: str, image_base64: str | None = None) -> dict:
    """Generate a completion from LLaVA via Ollama, optionally over an image.

    Sends a non-streaming request to the local Ollama ``/api/generate``
    endpoint. When ``image_base64`` is provided it is attached under Ollama's
    multimodal ``images`` field (a list of base64-encoded strings). On success
    returns ``{"response", "latency_ms", "model"}``. On any connection error or
    timeout returns the same shape with an additional ``error`` key and an
    empty ``response`` — this function never raises.
    """
    payload: dict = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if image_base64:
        payload["images"] = [image_base64]

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
