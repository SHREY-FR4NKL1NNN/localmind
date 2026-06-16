"""FastAPI application exposing the LocalMind routing layer.

Endpoints:

* ``POST /query``   — classify and route a query, returning the full decision.
* ``GET  /history`` — the most recent routing decisions.
* ``GET  /stats``   — aggregate statistics across all retained decisions.
* ``GET  /health``  — service health plus live Ollama reachability.

All inference happens locally via Ollama; this service makes no external API
calls. CORS is enabled for the Vite dev server at http://localhost:5173.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load environment configuration before importing modules that read it at
# import time (the model clients capture OLLAMA_BASE_URL on import).
load_dotenv()

import requests  # noqa: E402  (imported after dotenv on purpose)
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import router as router_module  # noqa: E402
from logger import decision_log  # noqa: E402

# IPv4 loopback by default — see note in the model clients about IPv6 stalls.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Models LocalMind depends on, by their Ollama tag.
REQUIRED_MODELS = ("mistral", "deepseek-r1:7b")


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str = Field(..., min_length=1, description="The user query to route.")


class RouterResponse(BaseModel):
    """Unified response returned by the router for a single query."""

    query: str
    response: str
    route: str = Field(..., description='Selected route: "mistral" or "deepseek".')
    reasoning: str
    complexity: float
    privacy: float
    latency_ms: int
    model: str
    compute_saved_ms: int
    timestamp: str
    error: str | None = None


class StatsResponse(BaseModel):
    """Aggregate routing statistics returned by ``GET /stats``."""

    total_queries: int
    mistral_count: int
    deepseek_count: int
    mistral_pct: float
    deepseek_pct: float
    total_compute_saved_ms: int
    avg_latency_mistral_ms: float
    avg_latency_deepseek_ms: float
    avg_complexity: float
    avg_privacy: float


class HealthResponse(BaseModel):
    """Service health returned by ``GET /health``."""

    status: str
    ollama: str = Field(..., description='"reachable" or "unreachable".')
    models: list[str]


app = FastAPI(
    title="LocalMind",
    description="Smart local LLM routing layer over Ollama.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=RouterResponse)
def post_query(request: QueryRequest) -> dict:
    """Classify and route a query, returning the full routing decision."""
    return router_module.handle_query(request.query)


@app.get("/history", response_model=list[RouterResponse])
def get_history() -> list[dict]:
    """Return the 50 most recent routing decisions, newest first."""
    return decision_log.get_history(50)


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> dict:
    """Return aggregate statistics computed across all retained decisions."""
    return decision_log.get_stats()


@app.get("/health", response_model=HealthResponse)
def get_health() -> dict:
    """Report service health and live Ollama reachability.

    Pings Ollama's ``/api/tags`` endpoint to confirm it is running and reports
    which of the required models are currently available.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m.get("name", "") for m in resp.json().get("models", [])]
        present = [
            required
            for required in REQUIRED_MODELS
            if any(name == required or name.startswith(required) for name in available)
        ]
        return {"status": "ok", "ollama": "reachable", "models": present}
    except requests.exceptions.RequestException:
        return {"status": "ok", "ollama": "unreachable", "models": []}
