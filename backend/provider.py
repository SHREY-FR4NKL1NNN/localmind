"""OpenAI-compatible provider abstraction for LocalMind.

LocalMind's four model clients (``llama32``/``deepseek``/``mistral``/``llava``)
used to each POST to Ollama's native ``/api/generate``. This module lets them all
speak to *any* OpenAI-compatible endpoint instead — Ollama in dev, Groq /
OpenRouter / Azure in prod — by translating LocalMind's existing call and return
shapes onto the OpenAI Chat Completions API. The clients keep their public
``generate()`` / ``stream()`` API and their logical identity (the ``model`` field
and the router's expert keys); only the transport underneath changes.

Roles decouple the code from concrete model names:

- ``router``    — the small/fast model (Llama 3.2 locally): gate decomposition
                  *and* the fast expert.
- ``reasoning`` — DeepSeek R1 for hard / analytical sub-tasks.
- ``general``   — Mistral for everyday sub-tasks.
- ``vision``    — the multimodal expert (image hard-route). May be ``None`` on a
                  provider that has no vision model → the client degrades with a
                  clear "vision unavailable in this deployment" message.

The provider never raises: connection errors, timeouts, and an unconfigured role
all come back as the same structured error dicts the router/gate already expect,
so LocalMind's resilience and the SSE ``asyncio.Queue`` fan-in are unchanged.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator

from openai import AsyncAzureOpenAI, AsyncOpenAI

# Kept for parity with the old clients: on Windows "localhost" can resolve to
# IPv6 first and stall, so default Ollama to the IPv4 loopback. In OpenAI-compat
# mode we talk to Ollama's ``/v1`` sibling of the native API.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ``ollama`` (default) keeps dev entirely local & free. Prod sets this to
# ``groq`` / ``openrouter`` / ``azure``.
PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# Logical role -> concrete model, per provider. ``None`` means the role is not
# available on that provider (used for ``vision`` -> graceful disable). Hosted
# providers rotate/deprecate model ids often — verify against provider docs
# before relying on any string below.
MODEL_MAP: dict[str, dict[str, str | None]] = {
    "ollama": {
        "router": "llama3.2",
        "reasoning": "deepseek-r1:7b",
        "general": "mistral",
        "vision": "minicpm-v",
    },
    "groq": {
        "router": "llama-3.3-70b-versatile",
        "reasoning": "deepseek-r1-distill-llama-70b",
        "general": "llama-3.1-8b-instant",
        # Groq's multimodal availability is uneven; disable vision by default and
        # let the image hard-route degrade gracefully. Prefer OpenRouter for a
        # single provider that also serves vision.
        "vision": None,
    },
    "openrouter": {
        "router": "meta-llama/llama-3.3-70b-instruct",
        "reasoning": "deepseek/deepseek-r1",
        "general": "meta-llama/llama-3.1-8b-instruct",
        "vision": "meta-llama/llama-3.2-11b-vision-instruct",
    },
    "azure": {  # these are Azure *deployment* names, not model names
        "router": os.getenv("AZURE_ROUTER_DEPLOYMENT", "gpt-4o-mini"),
        "reasoning": os.getenv("AZURE_REASONING_DEPLOYMENT", "gpt-4o"),
        "general": os.getenv("AZURE_GENERAL_DEPLOYMENT", "gpt-4o-mini"),
        "vision": os.getenv("AZURE_VISION_DEPLOYMENT") or None,
    },
}


def _make_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    if PROVIDER == "ollama":
        return AsyncOpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    if PROVIDER == "groq":
        return AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
    if PROVIDER == "openrouter":
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    if PROVIDER == "azure":
        return AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER!r}")


_client = _make_client()


def model_for(role: str) -> str | None:
    """Concrete model/deployment for a logical role on the active provider."""
    return MODEL_MAP.get(PROVIDER, {}).get(role)


def _build_messages(prompt: str, image_base64: str | None) -> list[dict]:
    if image_base64:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]
    return [{"role": "user", "content": prompt}]


def _translate_options(options: dict | None) -> dict:
    """Map the small set of Ollama generation options LocalMind uses onto the
    OpenAI chat kwargs."""
    kwargs: dict = {}
    if not options:
        return kwargs
    if "temperature" in options:
        kwargs["temperature"] = options["temperature"]
    if "top_p" in options:
        kwargs["top_p"] = options["top_p"]
    if "num_predict" in options:
        kwargs["max_tokens"] = options["num_predict"]
    return kwargs


def _translate_response_format(response_format: dict | str | None) -> dict | None:
    """Ollama's native API accepts a full JSON *schema* (or ``"json"``) in its
    ``format`` field. json_schema support across OpenAI-compatible providers is
    uneven, but ``{"type": "json_object"}`` is universal — and combined with
    temperature 0, the gate's schema-describing prompt, and the gate's robust
    JSON extraction it keeps decomposition deterministic. We therefore collapse
    any requested format to a JSON object."""
    if response_format is None:
        return None
    return {"type": "json_object"}


async def complete(
    role: str,
    prompt: str,
    *,
    model_label: str,
    options: dict | None = None,
    response_format: dict | str | None = None,
    image_base64: str | None = None,
    timeout: float = 45,
) -> dict:
    """Non-streaming completion. Returns the client contract
    ``{"response", "latency_ms", "model"}`` (plus ``"error"`` on failure). Never
    raises. ``model_label`` is the caller's logical identity, preserved so
    routing/logging/sparsity keys stay stable regardless of the concrete model.
    """
    model = model_for(role)
    if model is None:
        return {
            "response": "",
            "latency_ms": 0,
            "model": model_label,
            "error": (
                f"{model_label}: the '{role}' capability is not available in "
                f"this deployment (provider={PROVIDER})."
            ),
        }

    kwargs = _translate_options(options)
    rf = _translate_response_format(response_format)
    if rf is not None:
        kwargs["response_format"] = rf

    start = time.perf_counter()
    try:
        resp = await _client.chat.completions.create(
            model=model,
            messages=_build_messages(prompt, image_base64),
            timeout=timeout,
            **kwargs,
        )
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        return {
            "response": content.strip(),
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "model": model_label,
        }
    except Exception as exc:  # openai raises APIError/APITimeoutError/etc.
        return {
            "response": "",
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "model": model_label,
            "error": f"{model_label} via {PROVIDER} failed: {exc}",
        }


async def stream_tokens(
    role: str,
    prompt: str,
    *,
    model_label: str,
    image_base64: str | None = None,
    timeout: float = 120,
) -> AsyncGenerator[dict, None]:
    """Streaming completion. Yields ``{"token", "done", "model"}`` per content
    delta, then a terminal ``done=True`` chunk. On error yields exactly one
    terminal error chunk. Never raises. Callers that need extra fields (e.g.
    DeepSeek's ``is_thinking``) wrap this generator."""
    model = model_for(role)
    if model is None:
        yield {
            "token": "",
            "done": True,
            "model": model_label,
            "error": (
                f"{model_label}: the '{role}' capability is not available in "
                f"this deployment (provider={PROVIDER})."
            ),
        }
        return

    try:
        stream = await _client.chat.completions.create(
            model=model,
            messages=_build_messages(prompt, image_base64),
            stream=True,
            timeout=timeout,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Some providers (e.g. OpenRouter's DeepSeek-R1) surface the model's
            # reasoning in a separate ``reasoning`` field instead of inline
            # ``<think>`` tags. Emit it as a reasoning-flagged chunk so the
            # reasoning-aware client can route it to the thinking trace.
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                yield {
                    "token": reasoning,
                    "done": False,
                    "model": model_label,
                    "reasoning": True,
                }
            if delta.content:
                yield {"token": delta.content, "done": False, "model": model_label}
        yield {"token": "", "done": True, "model": model_label}
    except Exception as exc:
        yield {
            "token": "",
            "done": True,
            "model": model_label,
            "error": f"{model_label} via {PROVIDER} failed: {exc}",
        }
