"""Async client for the DeepSeek R1 model served locally by Ollama.

DeepSeek R1 is LocalMind's high-capability reasoning path: it handles complex,
multi-step, and technically demanding queries at the cost of higher latency and
compute. Like the Mistral client, this never raises — failures are returned as a
structured error dict so the router can degrade gracefully. HTTP is performed
with ``httpx.AsyncClient`` so expert calls can be awaited concurrently.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator

import httpx

# Streaming keeps the connection open for the whole (long) reasoning generation,
# so it uses a generous fixed timeout rather than the non-streaming budget above.
STREAM_TIMEOUT = httpx.Timeout(120.0)

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

# Default to the IPv4 loopback explicitly: on Windows, "localhost" can resolve
# to IPv6 (::1) first and stall for seconds before falling back to IPv4, while
# Ollama listens on 127.0.0.1. Override via OLLAMA_BASE_URL if needed.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL_NAME = "deepseek-r1:7b"
DISPLAY_NAME = "DeepSeek R1"
TIMEOUT_SECONDS = 60


async def generate(prompt: str) -> dict:
    """Generate a completion from DeepSeek R1 via Ollama.

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
    """Stream a completion from DeepSeek R1, separating its reasoning trace.

    Opens a streaming ``POST`` to Ollama's ``/api/generate`` with
    ``stream: true``. DeepSeek R1 wraps its chain-of-thought in literal
    ``<think>...</think>`` tags within the streamed text; these tags are stripped
    and the running ``inside_think`` state is tracked **across** chunks (a tag may
    be split, e.g. ``"<thi"`` then ``"nk>"``, which a carry buffer reassembles).

    Yields one ``{"token", "done", "model", "is_thinking"}`` dict per decoded
    chunk, where ``token`` has the tag text removed and ``is_thinking`` reflects
    the ``inside_think`` state *after* that chunk's tags were processed.
    ``image_base64`` is accepted for signature parity but ignored (text-only).
    Empty/garbled lines are skipped silently; a connection error or timeout
    yields exactly one terminal error chunk and stops — it never raises.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
    inside_think = False
    carry = ""
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
                    visible, inside_think, carry = _strip_think(
                        carry + data.get("response", ""), inside_think
                    )
                    if done and carry:
                        # Stream ended on a dangling partial tag — it was never a
                        # real tag, so flush it as literal text.
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
    except httpx.TimeoutException:
        yield {
            "token": "",
            "done": True,
            "model": MODEL_NAME,
            "is_thinking": False,
            "error": f"{DISPLAY_NAME} stream timed out after 120s.",
        }
    except httpx.HTTPError as exc:
        yield {
            "token": "",
            "done": True,
            "model": MODEL_NAME,
            "is_thinking": False,
            "error": f"Could not reach Ollama for {DISPLAY_NAME}: {exc}",
        }
