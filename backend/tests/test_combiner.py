"""Tests for combiner.combine() with the Llama 3.2 synthesis call mocked."""

from unittest.mock import AsyncMock

import combiner
from models import llama32_client


async def test_single_subtask_skips_combiner(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(llama32_client, "generate", mock)

    results = [{"subtask": "q", "expert": "mistral", "response": "the answer"}]
    out = await combiner.combine("q", results)

    assert out["skipped"] is True
    assert out["combined_response"] == "the answer"
    mock.assert_not_called()


async def test_multiple_subtasks_calls_combiner(monkeypatch):
    mock = AsyncMock(return_value={"response": "unified answer"})
    monkeypatch.setattr(llama32_client, "generate", mock)

    results = [
        {"subtask": "a", "expert": "mistral", "response": "answer a"},
        {"subtask": "b", "expert": "llama3.2", "response": "answer b"},
    ]
    out = await combiner.combine("original request", results)

    assert out["skipped"] is False
    assert out["combined_response"] == "unified answer"
    mock.assert_awaited_once()


async def test_llava_subtask_labeled_as_image(monkeypatch):
    mock = AsyncMock(return_value={"response": "unified"})
    monkeypatch.setattr(llama32_client, "generate", mock)

    results = [
        {"subtask": "describe image", "expert": "llava", "response": "a red circle"},
        {"subtask": "what is 2+2", "expert": "llama3.2", "response": "4"},
    ]
    await combiner.combine("original", results)

    args, kwargs = mock.call_args
    prompt = args[0] if args else kwargs.get("prompt", "")
    assert "Image analysis:" in prompt
