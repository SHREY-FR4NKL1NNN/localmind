"""Client for LocalMind's reasoning expert (DeepSeek R1).

DeepSeek R1 is LocalMind's high-capability reasoning path — the provider's
``reasoning`` role. Transport is delegated to ``provider`` (OpenAI-compatible:
Ollama in dev, a hosted model in prod); this module keeps the historical
``generate``/``stream`` API and ``model`` identity so the router and tests are
unchanged. Never raises — failures come back as a structured error dict.

R1 wraps its chain-of-thought in literal ``<think>...</think>`` tags inside the
streamed text. ``stream`` strips those tags — tracking the ``inside_think`` state
*across* chunks with a carry buffer, since a tag may be split across two chunks
(e.g. ``"<thi"`` then ``"nk>"``) — and reports ``is_thinking`` per chunk so the
reasoning trace can be surfaced separately from the final answer. This works
regardless of transport as long as R1 emits the tags inline; a hosted provider
that instead surfaces reasoning in a separate field would need handling here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import provider

MODEL_NAME = "deepseek-r1:7b"
DISPLAY_NAME = "DeepSeek R1"
ROLE = "reasoning"

TIMEOUT_SECONDS = 60
STREAM_TIMEOUT = 120

# DeepSeek R1 wraps its chain-of-thought in these literal tags inside the
# streamed text. They are detected and stripped so the reasoning can be surfaced
# separately from the final answer (see ``stream``).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _is_partial_tag(text: str) -> bool:
    """Return True if ``text`` is an incomplete prefix of a think tag.

    A streamed token may end in the middle of ``<think>`` or ``</think>`` (e.g.
    ``"<thi"``), with the rest arriving in the next chunk. Such a fragment is a
    *proper* prefix of one of the tags and must be held back in a carry buffer
    rather than emitted as literal text.
    """
    return (
        text != _THINK_OPEN
        and text != _THINK_CLOSE
        and (_THINK_OPEN.startswith(text) or _THINK_CLOSE.startswith(text))
    )


def _strip_think(buffer: str, inside_think: bool) -> tuple[str, bool, str]:
    """Strip think tags from ``buffer``, tracking state across chunks.

    Scans ``buffer`` (the carry-over from the previous chunk already prepended)
    left to right, removing literal ``<think>``/``</think>`` tags and flipping
    ``inside_think`` as each is crossed. Returns ``(visible_text, inside_think,
    carry)`` where ``carry`` is a trailing partial tag to prepend to the next
    chunk. Both complete-in-one-chunk thoughts and tags split across two chunks
    are handled.
    """
    out: list[str] = []
    carry = ""
    i = 0
    n = len(buffer)
    while i < n:
        if buffer[i] == "<":
            remaining = buffer[i:]
            if remaining.startswith(_THINK_OPEN):
                inside_think = True
                i += len(_THINK_OPEN)
                continue
            if remaining.startswith(_THINK_CLOSE):
                inside_think = False
                i += len(_THINK_CLOSE)
                continue
            if _is_partial_tag(remaining):
                # Possible tag completing in the next chunk — hold it back.
                carry = remaining
                break
        out.append(buffer[i])
        i += 1
    return "".join(out), inside_think, carry


async def generate(prompt: str) -> dict:
    """Non-streaming completion. Returns ``{"response", "latency_ms", "model"}``
    (plus ``"error"`` on failure)."""
    return await provider.complete(
        ROLE, prompt, model_label=MODEL_NAME, timeout=TIMEOUT_SECONDS
    )


async def stream(
    prompt: str, image_base64: str | None = None
) -> AsyncGenerator[dict, None]:
    """Stream token-by-token, separating R1's reasoning trace.

    Yields one ``{"token", "done", "model", "is_thinking"}`` dict per chunk,
    where ``token`` has the ``<think>``/``</think>`` tags removed and
    ``is_thinking`` reflects the state *after* that chunk's tags were processed.
    ``image_base64`` is accepted for signature parity but ignored (text-only). A
    provider/transport error yields exactly one terminal error chunk.
    """
    inside_think = False
    carry = ""
    async for chunk in provider.stream_tokens(
        ROLE, prompt, model_label=MODEL_NAME, timeout=STREAM_TIMEOUT
    ):
        if chunk.get("error"):
            yield {
                "token": "",
                "done": True,
                "model": MODEL_NAME,
                "is_thinking": False,
                "error": chunk["error"],
            }
            return

        # Reasoning delivered in a separate field (e.g. OpenRouter R1): it is
        # already isolated, so surface it as a thinking token directly — no tag
        # stripping needed.
        if chunk.get("reasoning"):
            yield {
                "token": chunk.get("token", ""),
                "done": False,
                "model": MODEL_NAME,
                "is_thinking": True,
            }
            continue

        done = bool(chunk.get("done", False))
        visible, inside_think, carry = _strip_think(
            carry + chunk.get("token", ""), inside_think
        )
        if done and carry:
            # Stream ended on a dangling partial tag — it was never a real tag,
            # so flush it as literal text.
            visible += carry
            carry = ""
        yield {
            "token": visible,
            "done": done,
            "model": MODEL_NAME,
            "is_thinking": inside_think,
        }
        if done:
            return
