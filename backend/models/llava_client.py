"""Client for LocalMind's vision expert (the provider's ``vision`` role).

Any sub-task that depends on an image is hard-routed here by the gate. Locally
the model is MiniCPM-V (served by Ollama); in prod it maps to a hosted
multimodal model. The module/expert-key name "llava" is retained as the
vision-slot identifier.

Transport is delegated to ``provider`` (OpenAI-compatible), which attaches the
image as an ``image_url`` content part. If the active provider has **no** vision
model configured (``MODEL_MAP[...]["vision"] is None``), the provider returns a
clear "vision unavailable in this deployment" error dict rather than failing —
so a text-only prod provider degrades gracefully. Never raises.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import provider

# Vision expert; locally MiniCPM-V 2.6. "llava" remains the vision-slot key.
MODEL_NAME = "minicpm-v"
DISPLAY_NAME = "MiniCPM-V"
ROLE = "vision"

TIMEOUT_SECONDS = 45
STREAM_TIMEOUT = 120


async def generate(prompt: str, image_base64: str | None = None) -> dict:
    """Non-streaming vision completion. When ``image_base64`` is provided it is
    attached to the message as an image. Returns ``{"response", "latency_ms",
    "model"}`` (plus ``"error"`` on failure or when vision is unavailable)."""
    return await provider.complete(
        ROLE,
        prompt,
        model_label=MODEL_NAME,
        image_base64=image_base64,
        timeout=TIMEOUT_SECONDS,
    )


async def stream(
    prompt: str, image_base64: str | None = None
) -> AsyncGenerator[dict, None]:
    """Stream a vision completion token-by-token, yielding ``{"token", "done",
    "model"}`` per chunk. When vision is unavailable on the active provider, a
    single terminal error chunk is yielded."""
    async for chunk in provider.stream_tokens(
        ROLE,
        prompt,
        model_label=MODEL_NAME,
        image_base64=image_base64,
        timeout=STREAM_TIMEOUT,
    ):
        yield chunk
