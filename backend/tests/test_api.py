"""Integration tests for the FastAPI app via httpx ASGITransport.

The endpoints are async, so we drive them with ``httpx.AsyncClient`` over an
``ASGITransport`` rather than the sync ``TestClient``. Ollama is mocked, so these
run in CI without it.
"""

import httpx
from httpx import ASGITransport

import main


def _client():
    transport = ASGITransport(app=main.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health_endpoint():
    async with _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "models" in body


async def test_expert_stats_endpoint():
    async with _client() as client:
        resp = await client.get("/expert-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_activations" in body
    assert "experts" in body


async def test_query_decomposed_endpoint_shape(mock_clients):
    async with _client() as client:
        resp = await client.post("/query/decomposed", json={"query": "What is 2+2?"})
    assert resp.status_code == 200
    body = resp.json()
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
        assert key in body
