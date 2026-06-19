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

import httpx  # noqa: E402  (imported after dotenv on purpose)
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import router as router_module  # noqa: E402
from logger import decision_log  # noqa: E402

# IPv4 loopback by default — see note in the model clients about IPv6 stalls.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Models LocalMind depends on, by their Ollama tag. The tiered MoE-inspired
# system uses all four: Llama 3.2 (fast expert + gate), Mistral (general),
# DeepSeek R1 (reasoning), and LLaVA (vision).
REQUIRED_MODELS = ("llama3.2", "mistral", "deepseek-r1:7b", "llava")


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str = Field(..., min_length=1, description="The user query to route.")


class DecomposedQueryRequest(BaseModel):
    """Request body for ``POST /query/decomposed``."""

    query: str = Field(..., min_length=1, description="The user query to decompose and route.")
    image_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded image; sub-tasks needing it hard-route to LLaVA.",
    )


class SubtaskDecision(BaseModel):
    """One sub-task's expert assignment and result within a decomposed query."""

    subtask: str
    expert: str = Field(..., description="Assigned expert model tag.")
    complexity: float
    privacy: float
    reasoning: str
    hard_routed: bool = Field(..., description="True only when hard-routed to LLaVA for an image.")
    response: str
    latency_ms: int
    depth: int = Field(..., description="Recursion depth; 0 for top-level sub-tasks, 1+ for nested ones.")


class SynthesisResult(BaseModel):
    """The combiner step's unified answer over all sub-task responses."""

    response: str
    model: str = Field(..., description="Model that produced the synthesis.")
    latency_ms: int


class SparsityInfo(BaseModel):
    """How few of the available experts a decomposed query activated."""

    experts_activated: int = Field(..., description="Distinct experts that actually ran.")
    experts_available: int = Field(..., description="Total experts in the tiered system (4).")
    sparsity_ratio: float = Field(..., description="experts_activated / experts_available.")
    vision_activated: bool = Field(..., description="True if the LLaVA vision expert ran.")
    activated_expert_names: list[str] = Field(..., description="Sorted distinct expert tags that ran.")


class DecomposedResponse(BaseModel):
    """Response returned by ``POST /query/decomposed``."""

    query: str
    decomposed: bool = Field(..., description="False if the query was a single, non-decomposed sub-task.")
    subtasks: list[SubtaskDecision]
    synthesis: SynthesisResult | None = Field(
        default=None,
        description="Legacy combiner view for the dashboard; null when the combiner was skipped.",
    )
    combined_response: str = Field(..., description="The unified reply (or the lone sub-task answer when skipped).")
    combiner_skipped: bool = Field(..., description="True when the combiner did not run (≤1 usable answer).")
    combiner_latency_ms: int = Field(..., description="Combiner call latency; 0 when skipped.")
    sparsity: SparsityInfo
    total_latency_ms: int = Field(
        ...,
        description="Wall-clock for the parallel expert batch + combiner; tracks the slowest expert, not the sum.",
    )
    timestamp: str


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
async def post_query(request: QueryRequest) -> dict:
    """Classify and route a query, returning the full routing decision."""
    return await router_module.handle_query(request.query)


@app.post("/query/decomposed", response_model=DecomposedResponse)
async def post_query_decomposed(request: DecomposedQueryRequest) -> dict:
    """Decompose a query into sub-tasks and route each to its expert.

    Runs the full tiered, Mixture-of-Experts-inspired flow: the gate splits the
    query into sub-tasks (recursively where a sub-task is itself compound),
    scores each to one expert, executes the selected experts concurrently with
    ``asyncio.gather``, and combines their answers into one unified response.
    Simple queries return a single, non-decomposed sub-task with the combiner
    skipped.
    """
    return await router_module.route_decomposed(request.query, request.image_base64)


@app.get("/history", response_model=list[RouterResponse])
def get_history() -> list[dict]:
    """Return the 50 most recent routing decisions, newest first."""
    return decision_log.get_history(50)


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> dict:
    """Return aggregate statistics computed across all retained decisions."""
    return decision_log.get_stats()


@app.get("/expert-stats")
def get_expert_stats() -> dict:
    """Return per-expert activation counts and their share of all activations.

    Reflects the tiered (decomposed) flow's lifetime expert utilisation since
    process start — a window into which experts the router actually exercises.
    """
    return decision_log.get_expert_activation_stats()


@app.get("/health", response_model=HealthResponse)
async def get_health() -> dict:
    """Report service health and live Ollama reachability.

    Pings Ollama's ``/api/tags`` endpoint to confirm it is running and reports
    which of the required models are currently available.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            available = [m.get("name", "") for m in resp.json().get("models", [])]
        present = [
            required
            for required in REQUIRED_MODELS
            if any(name == required or name.startswith(required) for name in available)
        ]
        return {"status": "ok", "ollama": "reachable", "models": present}
    except httpx.HTTPError:
        return {"status": "ok", "ollama": "unreachable", "models": []}
