"""Client for LocalMind's general expert (Mistral locally).

Mistral handles everyday, general-purpose sub-tasks — the provider's ``general``
role. Transport is delegated to ``provider`` (OpenAI-compatible: Ollama in dev,
a hosted model in prod); this module keeps the historical ``generate``/``stream``
API and ``model`` identity so the router and tests are unchanged. Never raises —
failures come back as a structured error dict.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import provider

MODEL_NAME = "mistral"
DISPLAY_NAME = "Mistral 7B"
ROLE = "general"

TIMEOUT_SECONDS = 45
STREAM_TIMEOUT = 120


async def generate(prompt: str) -> dict:
    """Non-streaming completion. Returns ``{"response", "latency_ms", "model"}``
    (plus ``"error"`` on failure)."""
    return await provider.complete(
        ROLE, prompt, model_label=MODEL_NAME, timeout=TIMEOUT_SECONDS
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
