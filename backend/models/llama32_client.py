"""Client for LocalMind's fast/router model (Llama 3.2 locally).

Llama 3.2 plays two roles in LocalMind's tiered, Mixture-of-Experts-inspired
design: it is the fast/trivial *expert* for the simplest sub-tasks, and it is
also the *gating network* itself — the gate (``gate.py``) calls this client to
decompose an incoming query into sub-tasks (with structured output). Both map to
the provider's ``router`` role.

Transport is delegated to ``provider`` (OpenAI-compatible: Ollama in dev, a
hosted model in prod); this module keeps the historical ``generate``/``stream``
API and the ``model`` identity so the router, gate, and tests are unchanged. Like
the other clients it never raises — failures come back as a structured error dict.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import provider

MODEL_NAME = "llama3.2"
DISPLAY_NAME = "Llama 3.2"
ROLE = "router"

# Llama 3.2 is the busiest model in the decomposed flow (gate AND fast expert),
# so a single query can queue several calls; keep a generous non-streaming budget.
TIMEOUT_SECONDS = 45
STREAM_TIMEOUT = 120


async def generate(
    prompt: str,
    options: dict | None = None,
    response_format: dict | str | None = None,
) -> dict:
    """Non-streaming completion. ``options`` (e.g. ``{"temperature": 0}``) and
    ``response_format`` (the gate's decomposition schema) are forwarded to the
    provider; both default to ``None`` so ordinary expert calls behave like the
    other clients. Returns ``{"response", "latency_ms", "model"}`` (+ ``error``)."""
    return await provider.complete(
        ROLE,
        prompt,
        model_label=MODEL_NAME,
        options=options,
        response_format=response_format,
        timeout=TIMEOUT_SECONDS,
    )


async def stream(
    prompt: str, image_base64: str | None = None
) -> AsyncGenerator[dict, None]:
    """Stream token-by-token, yielding ``{"token", "done", "model"}`` per chunk.
    ``image_base64`` is accepted for signature parity but ignored (text-only)."""
    async for chunk in provider.stream_tokens(
        ROLE, prompt, model_label=MODEL_NAME, timeout=STREAM_TIMEOUT
    ):
        yield chunk
