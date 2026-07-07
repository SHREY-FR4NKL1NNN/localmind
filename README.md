# LocalMind

[![CI](https://github.com/SHREY-FR4NKL1NNN/localmind/actions/workflows/ci.yml/badge.svg)](https://github.com/SHREY-FR4NKL1NNN/localmind/actions/workflows/ci.yml)

**Smart local LLM routing — the right model for the right query, entirely on your own machine.**

🔗 **[Live demo → localmind-theta.vercel.app](https://localmind-theta.vercel.app)**

> 📐 See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full design: the MoE
> analogy (and where it deviates), routing rules, the streaming fan-in, and an
> honest list of limitations.

LocalMind is a routing layer that sits in front of several locally-served
language models and decides, per query, which one should answer. It runs in two
modes:

- **Single-route** (`/query`) — the original flow. A query is classified on
  complexity and privacy, then dispatched to one model: **Mistral 7B** for the
  simple and **DeepSeek R1** for the hard.
- **Tiered / Mixture-of-Experts-inspired** (`/query/decomposed`) — a query is
  *decomposed* into sub-tasks, each sub-task is *gated* to one of four experts,
  the chosen experts run **in parallel**, and their answers are **synthesized**
  into one unified reply.

Every decision is classified, logged, and visualised in a live dashboard.

## Why intelligent local routing matters

It's tempting to assume that "everything runs locally" makes optimisation
irrelevant — there's no API bill, after all. But the constraints just move:

- **Compute.** A 7B reasoning model spinning up for "what is 2+2?" wastes GPU
  cycles, heats your laptop, and blocks the queue. Routing trivial queries to a
  lighter model keeps the heavy model free for work that actually needs it.
- **Latency.** Local inference is not free latency. Sending a one-line question
  through a slow reasoning model can mean an 8-second wait for a one-word
  answer. Routing it to Mistral returns it in a fraction of the time.
- **Privacy.** Even locally, minimising how sensitive data flows through larger,
  more capable models is good hygiene. LocalMind factors privacy sensitivity
  into its routing policy.
- **Cost.** The routing pattern is exactly what you'd use to cut spend in a
  hosted setup: send trivial work to a cheap model, reserve the expensive one for
  queries that need it. LocalMind proves the decision logic out locally, then runs
  the *same* logic unchanged against a hosted provider in production.

LocalMind is **model-agnostic**. The routing logic is decoupled from the inference
backend by a small provider abstraction (`backend/provider.py`), so the same
router drives **Ollama locally** (fully offline, no API keys) or a **hosted
OpenAI-compatible provider** (OpenRouter in production) with no code change — only
the `LLM_PROVIDER` env var and a key differ. See
[Deployment & Architecture](#deployment--architecture).

## The four experts

The tiered flow routes among four role-specialised models. Llama 3.2 does double
duty: it is both the fast expert *and* the gate that decomposes queries (an
LLM-driven decompose plus a heuristic per-sub-task score — not a learned gating
network; see below).

| Expert          | Ollama tag       | Role                                                          |
| --------------- | ---------------- | ------------------------------------------------------------- |
| **Llama 3.2**   | `llama3.2`       | Fast tier for trivial sub-tasks **+** the decomposition gate. |
| **Mistral 7B**  | `mistral`        | General-purpose tier **+** the synthesis/combiner step.       |
| **DeepSeek R1** | `deepseek-r1:7b` | Reasoning tier for complex / multi-step sub-tasks.            |
| **MiniCPM-V**   | `minicpm-v`      | Vision tier; image-bearing sub-tasks are hard-routed here.    |

> The vision expert's internal routing key is `llava` (a stable slot id kept from
> the original vision model, hence `llava_client.py`); it's now backed by the
> `minicpm-v` model and shown as "MiniCPM-V" in the UI.

> **A note on "MoE".** This is a rule-based gate *inspired by* Mixture-of-Experts
> routing — it reproduces the **behaviour** (sparse, input-dependent expert
> selection: only the chosen experts run) using transparent, explainable rules.
> It is **not** a learned/trained gate. That trade is deliberate: every routing
> decision carries a human-readable reason, which matters for a system whose
> whole premise is trustworthy local inference.

## Architecture

### Single-route flow (`/query`)

```
                         ┌──────────────────────────────────────┐
   Query  ──────────────▶│              Classifier              │
                         │  complexity score  ·  privacy score  │
                         └───────────────────┬──────────────────┘
                                             ▼
                         ┌──────────────────────────────────────┐
                         │                Router                │
                         │   apply policy · call model · time   │
                         └──────┬──────────────────────┬────────┘
                  complexity<0.4│         complexity≥0.7│
                                ▼                      ▼
                     ┌────────────────┐     ┌────────────────────┐
                     │   Mistral 7B   │     │    DeepSeek R1     │
                     └───────┬────────┘     └─────────┬──────────┘
                             └───────────┬────────────┘
                                         ▼
                         ┌──────────────────────────────────────┐
                         │     Response  +  Decision Log         │
                         └───────────────────┬──────────────────┘
                                             ▼
                                   React Dashboard (live)
```

### Tiered MoE-inspired flow (`/query/decomposed`)

```
                         ┌──────────────────────────────────────┐
   Query  ──────────────▶│   Gate · decompose (Llama 3.2)       │
   (+ optional image)    │   recursive: compound → sub-tasks    │
                         └───────────────────┬──────────────────┘
                                             ▼
                         ┌──────────────────────────────────────┐
                         │   Gate · score each sub-task → expert │
                         │   (heuristic + image hard-route)      │
                         └───────────────────┬──────────────────┘
                                             ▼  (run in parallel)
            ┌──────────────┬─────────────────┼─────────────────┐
            ▼              ▼                 ▼                 ▼
      ┌──────────┐  ┌────────────┐   ┌──────────────┐   ┌──────────┐
      │ Llama 3.2│  │ Mistral 7B │   │ DeepSeek R1  │   │MiniCPM-V │
      │  (fast)  │  │ (general)  │   │ (reasoning)  │   │ (vision) │
      └────┬─────┘  └─────┬──────┘   └──────┬───────┘   └────┬─────┘
           └──────────────┴────────┬────────┴────────────────┘
                                   ▼
                         ┌──────────────────────────────────────┐
                         │   Synthesis / combiner (Mistral 7B)   │
                         │   fuse sub-answers → unified reply     │
                         └───────────────────┬──────────────────┘
                                             ▼
                              Unified answer  +  per-sub-task trace
```

## Routing logic

### Single-route classifier

The classifier produces a **complexity** score and a **privacy** score, each in
`[0, 1]`. The router then applies this policy:

| Complexity        | Privacy      | Route        | Reason                                                             |
| ----------------- | ------------ | ------------ | ----------------------------------------------------------------- |
| `< 0.4`           | any          | Mistral 7B   | Short, direct query — the lightweight model is plenty.            |
| `0.4 – 0.7`       | `> 0.6`      | Mistral 7B   | Moderate but privacy-sensitive — keep it on the smaller path.     |
| `0.4 – 0.7`       | `≤ 0.6`      | DeepSeek R1  | Moderate and non-sensitive — worth the stronger reasoning.        |
| `≥ 0.7`           | any          | DeepSeek R1  | Complex / multi-step — full capability is warranted.              |

**Complexity** is a weighted blend of query length, multi-step indicator phrases
(`compare`, `analyze`, `explain why`, `design`, …), technical-jargon density,
code-related terms, and clause count. **Privacy** accumulates weighted evidence
from structured PII (phone numbers, emails, SSN-like patterns, full names) and
sensitive vocabulary (financial, health, address terms).

### Tiered gate (decompose → score → run → synthesize)

1. **Decompose** (`gate.decompose`, Llama 3.2). The query is split into at most
   four sub-tasks. Llama 3.2 is prompted with Ollama **structured outputs** (a
   JSON-array schema) at `temperature 0`, so the gate's output is deterministic
   and always parseable — a free-text prompt to a 3B model is not. Trivially
   simple queries skip the model call entirely; an unparseable result degrades
   gracefully to a single sub-task. **Recursive decomposition:** a sub-task that
   still reads as a compound request (joined by `and` / `also` / `then` /
   `as well as`) is decomposed one level deeper, bounded by a depth cap
   (`MAX_DEPTH`) and a leaf cap (`MAX_LEAVES`) so it always terminates. An
   analytical ask carrying a comparison/analysis verb (*"compare X to Y and
   evaluate the tradeoffs"*) **skips decomposition entirely** and is kept whole:
   the 3B gate otherwise tends to shred one analytical sentence into syntactic
   fragments, which the prompt alone cannot reliably prevent. The intact ask is
   then routed to the reasoning expert via its complexity boost (below).

2. **Score** (`gate.gate_score`). Each sub-task is routed to exactly one expert:

   | Condition                                       | Expert           |
   | ----------------------------------------------- | ---------------- |
   | depends on an image (**hard rule**)             | `llava`          |
   | effective complexity `≥ 0.6`                    | `deepseek-r1:7b` |
   | privacy `> 0.6` and complexity `< 0.6`          | `mistral`        |
   | complexity `< 0.3`                              | `llama3.2`       |
   | otherwise                                       | `mistral`        |

   A comparison/analysis verb (`compare`, `analyze`, `evaluate`, `contrast`,
   `tradeoff`, …) adds a fixed `+0.30` boost to complexity *before* thresholding,
   because isolating a short analytical clause otherwise under-scores it on the
   length-weighted heuristic. The boost is shown in the reasoning string.

3. **Run in parallel** (`router.route_decomposed`). The selected experts execute
   concurrently via `asyncio.gather` over `httpx.AsyncClient` coroutines — the
   awaited Ollama calls overlap on the event loop — while results keep sub-task
   order (`return_exceptions=True`, so one failing expert never cancels the
   rest). **Sparse activation:** only the chosen experts run, never all four; the
   response's `sparsity` block reports how few were activated. Generous per-client
   timeouts absorb the model-swap queueing a memory-constrained GPU imposes when
   the distinct experts don't all fit in VRAM at once. The whole batch's
   wall-clock is returned as `total_latency_ms` — close to the *slowest* expert,
   not the sum, which is what proves the calls actually ran in parallel.

4. **Combine** (`combiner.combine`, Llama 3.2). When more than one sub-task was
   answered, the combiner fuses the individual answers into one coherent reply to
   the original request. With a single sub-task the lone answer is returned as-is
   and the combiner is skipped (`combiner_skipped: true`). This is a **text
   synthesis** step, *not* a literal MoE weighted-sum — see the note in
   `combiner.py`.

## API

| Method & path        | Purpose                                                                     |
| -------------------- | --------------------------------------------------------------------------- |
| `POST /query`        | Single-route: classify and dispatch to one model. Returns the full decision. |
| `POST /query/decomposed` | Tiered flow: decompose, gate, run in parallel (`asyncio.gather`), combine. Returns per-sub-task trace, `sparsity`, and `total_latency_ms`. Optional `image_base64`. |
| `GET  /history`      | The 50 most recent single-route decisions, newest first.                    |
| `GET  /stats`        | Aggregate statistics across retained single-route decisions.                |
| `GET  /expert-stats` | Lifetime per-expert activation counts and each expert's share of all activations. |
| `GET  /health`       | Service health + live Ollama reachability + which of the four models are present. |

Interactive docs are served at `/docs`.

## Setup & run

### Prerequisites

- [Ollama](https://ollama.com) installed and serving on `http://localhost:11434`
- The four models pulled:

  ```bash
  ollama pull llama3.2
  ollama pull mistral
  ollama pull deepseek-r1:7b
  ollama pull minicpm-v   # vision expert
  ```

- Python 3.12+ and Node.js 18+

### Backend

```bash
cd localmind/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000` (interactive docs at `/docs`).
Optionally copy `.env.example` to `.env` to point at a non-default Ollama URL.

### Frontend

```bash
cd localmind/frontend
npm install
npm run dev
```

The dashboard is now at `http://localhost:5173`. Toggle **Decompose (MoE)** next
to the Submit button to run the tiered flow and see the per-sub-task trace plus
the synthesized answer.

### Demo queries that fan out

A single-topic prompt usually lights just one expert (it looks like a plain
chatbot). These prompts genuinely fan out across experts — verified locally with
`scripts/demo_probe.py`, which reports subtask count, activated experts, sparsity
ratio, and whether the combiner ran:

| Query | Experts | Sparsity |
| ----- | ------- | -------- |
| *"What is 2+2, and also compare transformer architectures to state space models for edge inference?"* | Llama 3.2 + DeepSeek R1 | 0.50 |
| *"What is 10 times 10, and also analyze the tradeoffs between microservices and monoliths for a small startup."* | Llama 3.2 + DeepSeek R1 | 0.50 |
| *"Describe in a few sentences how photosynthesis works, and separately compare it to cellular respiration and evaluate which releases more usable energy."* | Llama 3.2 + Mistral | 0.50 |

The pattern that works: a **trivial front clause** (arithmetic, a one-word fact)
joined by *"and also / and separately"* to a **second ask of a different
complexity**, so the two land in different tiers and the combiner fuses them. Add
an **image** to the query to pull in the vision expert for a higher-sparsity,
3–4-expert demo.

> These were probed against local Ollama. **TODO:** once the OpenRouter production
> credit posts, re-run `python scripts/demo_probe.py --target <azure-url>` to
> confirm the picks on the hosted models (the router scoring is identical, so they
> should hold).

## Deployment & Architecture

LocalMind deploys as two pieces, with the routing logic deliberately decoupled
from the inference backend:

```
Vercel (static SPA)  ──►  Azure Container Apps (FastAPI, stable HTTPS)
                               │
                               ▼
                     provider abstraction (backend/provider.py)
                       ├─ dev:  Ollama       (LLM_PROVIDER=ollama)
                       └─ prod: OpenRouter    (LLM_PROVIDER=openrouter)
```

**Provider abstraction.** Every model client talks to an OpenAI-compatible
endpoint through `backend/provider.py`, chosen by the `LLM_PROVIDER` env var
(`ollama` | `openrouter` | `groq` | `azure`). A logical role→model map (`router`,
`reasoning`, `general`, `vision`) decouples the code from concrete model names, so
the same routing logic runs on local Ollama or a hosted provider unchanged. The
gate's structured-output request maps to each provider's JSON mode
(`response_format: json_object`), keeping decomposition deterministic everywhere.
DeepSeek R1's reasoning is captured whether a provider emits inline `<think>` tags
(Ollama) or a separate `reasoning` field (OpenRouter).

### Frontend (Vercel)

1. Import the repo to Vercel; set the **root directory** to `frontend`.
2. Set `VITE_API_URL` to the backend's public URL (the Container Apps FQDN).
3. Deploy. (`frontend/vercel.json` sets the build command, `dist` output, and the
   SPA rewrite.)

### Backend (Azure Container Apps)

The backend runs as a container on **Azure Container Apps** with a stable HTTPS
URL — no tunnel. In production it runs `LLM_PROVIDER=openrouter`; the API key
lives in the **Container Apps secret store**, never in the image or in git.

ACR Tasks (server-side build) is unavailable on some subscriptions (e.g. Azure for
Students), so build the image locally and push it, then deploy:

```bash
# one-time: register providers + resource group
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
az group create -n localmind-rg -l centralindia

# build locally and push to your ACR
az acr login -n <registry>
docker build -t <registry>.azurecr.io/localmind:latest backend
docker push  <registry>.azurecr.io/localmind:latest

# deploy the image
az containerapp up -n localmind-api -g localmind-rg \
  --image <registry>.azurecr.io/localmind:latest \
  --ingress external --target-port 8000 \
  --registry-server <registry>.azurecr.io \
  --env-vars LLM_PROVIDER=openrouter

# attach the API key as a secret (never plaintext env)
az containerapp secret set -n localmind-api -g localmind-rg --secrets openrouter-key=<KEY>
az containerapp update     -n localmind-api -g localmind-rg \
  --set-env-vars OPENROUTER_API_KEY=secretref:openrouter-key
```

`backend/Dockerfile` builds from `backend/` (non-root, `uvicorn main:app` on
8000). `GET /health` reports the active provider and its configured models and
does **not** depend on Ollama when a hosted provider is selected.

**CORS.** `main.py` allows the Vite dev origin (`localhost:5173` / `127.0.0.1:5173`)
and any `*.vercel.app` subdomain (preview + production deploys). The backend's own
Azure origin is not a browser origin, so it is intentionally not listed.

## Testing

The backend ships with a `pytest` suite that runs **without Ollama** — every
model client is mocked via a fixture in `tests/conftest.py`, so the suite (and
CI) need no GPU and no running models.

```bash
cd localmind/backend
pip install -r requirements.txt   # includes httpx + the openai client
pip install pytest pytest-asyncio
pytest tests/ -v
```

What's covered:

- **`test_classifier.py`** — the routing scorer (`gate.gate_score`): trivial →
  `llama3.2`, moderate → `mistral`, complex → `deepseek-r1:7b`, the image
  hard-route to `llava`, the privacy override, and the analysis-verb boost.
- **`test_gate.py`** — `gate.decompose` with the Llama 3.2 call mocked: the
  short-query short-circuit (no model call), valid-JSON decomposition, graceful
  fallback on invalid JSON, and the auto-added vision sub-task for images.
- **`test_combiner.py`** — `combiner.combine`: single-sub-task skip, multi
  synthesis, and the `Image analysis:` labelling of vision-expert results.
- **`test_router.py`** — `router.route_decomposed`: parallel fan-out + sparsity,
  the full return shape, and one expert erroring without crashing the others.
- **`test_api.py`** — FastAPI integration via `httpx.AsyncClient` + `ASGITransport`
  for `/health`, `/expert-stats`, and `/query/decomposed`.
- **`test_provider.py`** — the provider abstraction: role→model selection per
  `LLM_PROVIDER`, the graceful "vision unavailable" path, response-format/option
  translation, and DeepSeek R1 `<think>` / separate-`reasoning`-field extraction.

CI runs the same suite plus `ruff` on every push to `main` / `feature/*` and on
PRs (see `.github/workflows/ci.yml`).

## Data persistence

Every query (single-route and decomposed) is persisted to a local SQLite database
at **`backend/localmind.db`** (table `query_log`) using the standard-library
`sqlite3` module — no ORM, no extra dependency. History, aggregate stats, and
per-expert utilisation are computed with SQL aggregates and survive restarts. The
database file is git-ignored.

> **Deployment tradeoff.** On Azure Container Apps the container filesystem is
> ephemeral, so `localmind.db` **resets on every restart / new revision** — the
> decision log is not durable in production. This is an accepted tradeoff for the
> demo; durable storage via an Azure Files mount (pointing `LOCALMIND_DB` at the
> share) is a documented next step.

Export the full log to JSON for a demo or for sharing:

```python
from logger import decision_log
decision_log.export_json("localmind_log.export.json")
```

Override the database location (e.g. for tests or an ephemeral run) with the
`LOCALMIND_DB` environment variable.

## Technical decisions

- **Rule-based classifier and gate instead of learned ones.** The routing
  signals (length, keyword cues, PII patterns) are cheap to compute and,
  crucially, *fully explainable* — every decision carries a human-readable
  reason. A trained gate would add latency, a training-data requirement, and an
  opaque decision surface for marginal benefit at this scale. The rule-based
  core is also the right baseline to benchmark a future learned gate against.
- **Structured outputs for decomposition.** A 3B model emitting free-text JSON
  is flaky; constraining Llama 3.2 with an Ollama JSON-array schema at
  `temperature 0` makes the gate deterministic and always parseable, with a
  graceful single-sub-task fallback if anything still goes wrong.
- **Parallel experts via `asyncio` + `httpx`.** Every model client is an
  `async def` over `httpx.AsyncClient`, and `route_decomposed` fans the chosen
  experts out with `asyncio.gather(..., return_exceptions=True)`. The awaited
  Ollama calls overlap on a single event loop — no thread pool — so the batch's
  wall-clock tracks the slowest expert rather than the sum, and one failing
  expert cannot cancel the others. Generous per-client timeouts absorb the
  server-side queueing Ollama does when the distinct experts don't co-reside in
  VRAM.
- **Fully local, no external APIs.** Privacy (no data leaves the machine),
  latency (no network round-trip), cost (no per-token billing), and offline
  capability all follow directly. It also makes the project trivially
  reproducible — clone, pull the models, run.
- **FastAPI + Vite/React instead of a monolith.** A clean HTTP boundary lets the
  routing logic be tested and reused independently of any UI, gives us free
  interactive API docs via FastAPI, and lets the React dashboard iterate with
  hot-reload.
- **SQLite persistence via the standard library.** Every query is written to a
  `query_log` table (`backend/localmind.db`) using built-in `sqlite3` — no ORM,
  no extra dependency. History/stats/expert-utilisation are SQL aggregates and
  survive restarts, while a per-operation connection with `check_same_thread=False`
  plus a write lock keeps it safe under FastAPI's async handlers. `export_json()`
  dumps the whole log for demos.
- **Structured logging to stdout.** All modules log through a shared
  `localmind.*` logger (`log_config.py`) at INFO (or DEBUG with `LOCALMIND_DEBUG=1`),
  so routing decisions, latencies, and warnings are captured by systemd/Docker/CI.

## Scope expansion

The original brief was a **Mixture-of-Experts-inspired routing layer**: decompose
a query, gate each sub-task to one expert, run the experts in parallel, and
combine their answers. That core is implemented faithfully. Two pieces of this
project were built **beyond** that core spec as deliberate extensions, and are
called out here so the boundary is explicit:

- **Recursive decomposition** (`gate.decompose`, bounded by `MAX_DEPTH` /
  `MAX_LEAVES`). The core spec only required a single decomposition pass. Splitting
  a still-compound sub-task one level deeper is an enhancement to decomposition
  quality, not part of the original routing design.
- **The React dashboard** (`frontend/`). The core deliverable is the routing
  backend and its HTTP API; the Vite/React UI — live feed, stats bar, per-sub-task
  trace, and the **Decompose (MoE)** toggle — is an extension built to make the
  routing behaviour visible and demoable.

Both are intentional additions that go past the baseline; neither is required for
the core decompose → gate → parallel → combine flow to function.

## What I'd build next

(Consistent with [ARCHITECTURE.md](ARCHITECTURE.md).)

- **A learned gate** — fine-tune a small classifier on real query→route data
  collected from this system, with the rule-based gate as both baseline and
  labeller.
- **Streaming decomposition** — start routing and running sub-tasks before the
  full decomposition completes.
- **Multi-GPU support** — true parallel expert inference with one model per GPU,
  removing the single-GPU VRAM time-slicing bottleneck.
- **Persistent conversation context** carried across sub-tasks and turns.
- **Additional experts** — code-specialized (CodeLlama), multilingual (Qwen),
  math-specialized (Mathstral).

## Project layout

```
localmind/
├── ARCHITECTURE.md        # full design: MoE analogy, routing rules, streaming, limitations
├── .github/workflows/
│   └── ci.yml             # GitHub Actions: pytest + ruff (no Ollama needed)
├── backend/
│   ├── main.py            # FastAPI app: /query /query/decomposed(/stream) /history /stats /expert-stats /health
│   ├── router.py          # single-route + decomposed (decompose→gate→asyncio.gather→combine) + SSE stream
│   ├── gate.py            # MoE-inspired gate: decompose (recursive) + per-sub-task scoring
│   ├── combiner.py        # async text-synthesis combiner (Llama 3.2); not a literal MoE weighted-sum
│   ├── classifier.py      # complexity & privacy scoring + single-route policy
│   ├── models/                # all async httpx.AsyncClient clients (generate + stream)
│   │   ├── llama32_client.py   # fast expert + decomposition gate (structured outputs) + combiner
│   │   ├── mistral_client.py   # general expert
│   │   ├── deepseek_client.py  # reasoning expert (+ <think> trace stripping)
│   │   └── llava_client.py     # vision expert (multimodal) — MiniCPM-V
│   ├── logger.py          # SQLite query_log: history/stats/expert-stats/export (SQL aggregates)
│   ├── log_config.py      # shared structured logging (localmind.* → stdout)
│   ├── tests/             # pytest suite (classifier, gate, combiner, router, api); Ollama mocked
│   ├── requirements.txt
│   ├── .gitignore         # ignores localmind.db
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── components/
│   │   │   ├── QueryInput.jsx        # query box + Decompose (MoE) toggle
│   │   │   ├── ResponsePanel.jsx     # single-route result
│   │   │   ├── DecomposedPanel.jsx   # sub-task trace + synthesized answer
│   │   │   ├── LiveFeed.jsx
│   │   │   └── StatsBar.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
└── README.md
```
