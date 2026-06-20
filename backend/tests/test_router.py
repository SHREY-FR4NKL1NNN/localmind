"""Tests for router.route_decomposed() with model clients and gate mocked."""

import gate
import router


async def _two_subtasks(query, has_image, _depth=0):
    """Deterministic 2-sub-task decomposition routing to two distinct experts."""
    return [
        {"subtask": "hello there friend", "depends_on_image": False, "depth": 0},
        {
            "subtask": (
                "Compare and analyze the architectural tradeoffs between "
                "transformers and state space models for edge inference workloads"
            ),
            "depends_on_image": False,
            "depth": 0,
        },
    ]


async def test_route_decomposed_parallel(monkeypatch, mock_clients):
    monkeypatch.setattr(gate, "decompose", _two_subtasks)

    result = await router.route_decomposed("two part query", None)

    # Both sub-tasks ran and both responses came back (gather fan-out).
    assert len(result["subtasks"]) == 2
    assert all(st["response"] == "mock response" for st in result["subtasks"])
    # Two distinct experts (llama3.2 + deepseek-r1:7b) were activated.
    assert result["sparsity"]["experts_activated"] == 2
    assert mock_clients["llama32"].await_count >= 1
    assert mock_clients["deepseek"].await_count >= 1


async def test_route_decomposed_returns_correct_shape(monkeypatch, mock_clients):
    monkeypatch.setattr(gate, "decompose", _two_subtasks)

    result = await router.route_decomposed("q", None)

    for key in (
        "query",
        "decomposed",
        "subtasks",
        "combined_response",
        "combiner_skipped",
        "combiner_latency_ms",
        "sparsity",
        "total_latency_ms",
        "timestamp",
    ):
        assert key in result


async def test_expert_error_does_not_crash(monkeypatch, mock_clients):
    monkeypatch.setattr(gate, "decompose", _two_subtasks)
    # One expert blows up mid-call; the other must still succeed.
    mock_clients["deepseek"].side_effect = RuntimeError("boom")

    result = await router.route_decomposed("q", None)

    assert len(result["subtasks"]) == 2
    responses = [st["response"] for st in result["subtasks"]]
    assert any(r.startswith("[error]") for r in responses)
    assert "mock response" in responses
