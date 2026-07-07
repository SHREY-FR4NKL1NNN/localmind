"""Tests for the OpenAI-compatible provider abstraction and the DeepSeek
think-trace stripping — the two things a provider/model swap could silently
break. All offline: no network, no Ollama.
"""

import pytest

import provider
from models import deepseek_client
from openai import AsyncAzureOpenAI, AsyncOpenAI


def test_make_client_selects_endpoint_per_provider(monkeypatch):
    monkeypatch.setattr(provider, "PROVIDER", "ollama")
    c = provider._make_client()
    assert isinstance(c, AsyncOpenAI)
    assert "11434/v1" in str(c.base_url)

    monkeypatch.setattr(provider, "PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    c = provider._make_client()
    assert isinstance(c, AsyncOpenAI)
    assert "openrouter.ai/api/v1" in str(c.base_url)

    monkeypatch.setattr(provider, "PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert "api.groq.com/openai/v1" in str(provider._make_client().base_url)

    monkeypatch.setattr(provider, "PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    assert isinstance(provider._make_client(), AsyncAzureOpenAI)

    monkeypatch.setattr(provider, "PROVIDER", "bogus")
    with pytest.raises(ValueError):
        provider._make_client()


def test_model_for_maps_roles_per_provider(monkeypatch):
    monkeypatch.setattr(provider, "PROVIDER", "openrouter")
    assert provider.model_for("reasoning") == "deepseek/deepseek-r1"
    assert provider.model_for("vision") == "meta-llama/llama-3.2-11b-vision-instruct"

    monkeypatch.setattr(provider, "PROVIDER", "groq")
    assert provider.model_for("vision") is None  # graceful-disable target


def test_response_format_collapses_to_json_object():
    assert provider._translate_response_format(None) is None
    assert provider._translate_response_format("json") == {"type": "json_object"}
    assert provider._translate_response_format({"type": "array"}) == {
        "type": "json_object"
    }


def test_translate_options_maps_known_keys():
    assert provider._translate_options(None) == {}
    out = provider._translate_options(
        {"temperature": 0, "num_predict": 128, "top_p": 0.9}
    )
    assert out == {"temperature": 0, "max_tokens": 128, "top_p": 0.9}


async def test_complete_graceful_disable_when_role_unavailable(monkeypatch):
    # Groq has no vision model -> complete() must return a structured error dict,
    # never raising or hitting the network.
    monkeypatch.setattr(provider, "PROVIDER", "groq")
    r = await provider.complete("vision", "describe this", model_label="minicpm-v")
    assert r["response"] == ""
    assert "not available" in r["error"]
    assert r["model"] == "minicpm-v"


async def test_stream_graceful_disable_yields_one_error_chunk(monkeypatch):
    monkeypatch.setattr(provider, "PROVIDER", "groq")
    chunks = [
        c
        async for c in provider.stream_tokens(
            "vision", "describe", model_label="minicpm-v"
        )
    ]
    assert len(chunks) == 1
    assert chunks[0]["done"] is True
    assert "not available" in chunks[0]["error"]


async def test_deepseek_stream_routes_separate_reasoning_field(monkeypatch):
    # Simulate a provider (e.g. OpenRouter R1) that streams reasoning in a
    # separate field, then the answer as content. The client must flag the
    # reasoning as is_thinking=True and the answer as is_thinking=False.
    async def fake_stream_tokens(role, prompt, *, model_label, image_base64=None, timeout=120):
        yield {"token": "let me think", "done": False, "model": model_label, "reasoning": True}
        yield {"token": "42", "done": False, "model": model_label}
        yield {"token": "", "done": True, "model": model_label}

    monkeypatch.setattr(provider, "stream_tokens", fake_stream_tokens)
    chunks = [c async for c in deepseek_client.stream("7*6?")]
    thinking = [c["token"] for c in chunks if c.get("is_thinking")]
    answer = "".join(c["token"] for c in chunks if not c.get("is_thinking"))
    assert thinking == ["let me think"]
    assert "42" in answer


def test_strip_think_removes_tags_and_tracks_state():
    # _strip_think removes the literal <think>/</think> tags and tracks the
    # inside_think state; the inner reasoning text is kept in the token (the
    # frontend separates it via is_thinking during streaming).
    visible, inside, carry = deepseek_client._strip_think(
        "<think>weighing options</think>The answer is 4.", False
    )
    assert "<think>" not in visible and "</think>" not in visible
    assert "The answer is 4." in visible
    assert inside is False
    assert carry == ""


def test_strip_think_handles_tag_split_across_chunks():
    # First chunk ends mid-tag; the partial must be carried, never emitted raw.
    v1, inside, carry = deepseek_client._strip_think("hello <thi", False)
    assert v1 == "hello "
    assert carry == "<thi"
    assert inside is False
    # Next chunk completes the tag: it's reassembled and stripped (not literal),
    # and we're now inside the reasoning region.
    v2, inside, carry = deepseek_client._strip_think(carry + "nk>secret", inside)
    assert "<think>" not in v2
    assert inside is True
    assert carry == ""
