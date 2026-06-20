"""Tests for gate.decompose() with the Llama 3.2 gate call mocked."""

import json
from unittest.mock import AsyncMock

import gate
from models import llama32_client


async def test_short_query_skips_decomposition(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(llama32_client, "generate", mock)

    result = await gate.decompose("what is the capital of france", False)

    assert len(result) == 1
    assert result[0]["subtask"] == "what is the capital of france"
    mock.assert_not_called()  # the heuristic short-circuit avoids the model


async def test_long_query_decomposes(monkeypatch):
    payload = json.dumps(
        [
            {"subtask": "Summarize the article", "depends_on_image": False},
            {"subtask": "Translate the summary to French", "depends_on_image": False},
        ]
    )
    monkeypatch.setattr(
        llama32_client, "generate", AsyncMock(return_value={"response": payload})
    )

    result = await gate.decompose(
        "Summarize the article and translate the summary to French", False
    )

    assert len(result) == 2
    subtasks = [r["subtask"] for r in result]
    assert any("Summarize" in s for s in subtasks)
    assert any("French" in s for s in subtasks)


async def test_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr(
        llama32_client,
        "generate",
        AsyncMock(return_value={"response": "this is not valid json"}),
    )

    # Compound (so it isn't short-circuited) but carries no analysis verb, so it
    # genuinely reaches the model — whose garbage output must degrade gracefully.
    query = "Summarize this article and also draft a reply and then list next steps"
    result = await gate.decompose(query, False)

    assert len(result) == 1
    assert result[0]["subtask"] == query


async def test_image_flag_adds_vision_subtask(monkeypatch):
    payload = json.dumps([{"subtask": "Describe the scene", "depends_on_image": False}])
    monkeypatch.setattr(
        llama32_client, "generate", AsyncMock(return_value={"response": payload})
    )

    query = "Tell me about this and summarize the key points"
    result = await gate.decompose(query, True)

    assert any(st["depends_on_image"] for st in result)
