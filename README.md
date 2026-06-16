# LocalMind

**Smart local LLM routing — the right model for the right query, entirely on your own machine.**

LocalMind is a routing layer that sits in front of two locally-served language
models and decides, per query, which one should answer. Simple, low-stakes
queries go to **Mistral 7B** — fast and cheap on compute. Complex, multi-step,
or technically demanding queries go to **DeepSeek R1** — slower but far more
capable. Every decision is classified, logged, and visualised in a live
dashboard.

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
- **Cost (by analogy).** The routing pattern here is exactly what you'd use to
  cut spend in a hosted setup — LocalMind proves out the decision logic with
  zero external dependencies and zero data leaving the machine.

Everything is **fully local**: all inference goes through [Ollama](https://ollama.com)
on `localhost`. No external API calls, no API keys, works offline.

## Architecture

```
                         ┌──────────────────────────────────────┐
   Query  ──────────────▶│              Classifier              │
                         │  complexity score  ·  privacy score  │
                         └───────────────────┬──────────────────┘
                                             │
                                             ▼
                         ┌──────────────────────────────────────┐
                         │                Router                │
                         │   apply policy · call model · time   │
                         └──────┬──────────────────────┬────────┘
                                │                      │
                  complexity<0.4│         complexity≥0.7│
                  (or sensitive)│      (or moderate+open)│
                                ▼                      ▼
                     ┌────────────────┐     ┌────────────────────┐
                     │   Mistral 7B   │     │    DeepSeek R1     │
                     │  (Ollama,fast) │     │  (Ollama, capable) │
                     └───────┬────────┘     └─────────┬──────────┘
                             │                        │
                             └───────────┬────────────┘
                                         ▼
                         ┌──────────────────────────────────────┐
                         │     Response  +  Decision Log         │
                         │   latency · scores · compute saved    │
                         └───────────────────┬──────────────────┘
                                             ▼
                                   React Dashboard (live)
```

## Routing logic

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

## Setup & run

### Prerequisites

- [Ollama](https://ollama.com) installed and serving on `http://localhost:11434`
- The two models pulled:

  ```bash
  ollama pull mistral
  ollama pull deepseek-r1:7b
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

The dashboard is now at `http://localhost:5173`.

## Technical decisions

- **Rule-based classifier instead of an ML classifier.** The routing signals
  (length, keyword cues, PII patterns) are cheap to compute and, crucially,
  *fully explainable* — every decision carries a human-readable reason. A trained
  classifier would add latency, a training-data requirement, and an opaque
  decision surface for marginal benefit at this scale. The rule-based core is
  also the right baseline to benchmark a future ML model against.
- **Fully local, no external APIs.** Privacy (no data leaves the machine),
  latency (no network round-trip), cost (no per-token billing), and offline
  capability all follow directly. It also makes the project trivially
  reproducible — clone, pull two models, run.
- **FastAPI + Vite/React instead of a monolith.** A clean HTTP boundary lets the
  routing logic be tested and reused independently of any UI, gives us free
  interactive API docs via FastAPI, and lets the React dashboard iterate with
  hot-reload. Separation of concerns over a single templated server.
- **In-memory logging instead of a database.** Routing history is ephemeral
  demo/observability data, not a system of record. An in-memory ring buffer
  keeps the project dependency-free, makes stats computation a simple list
  comprehension, and starts instantly with nothing to migrate or clean up.

## What I'd build next

- **Streaming responses via SSE** so tokens render as they're generated instead
  of waiting for the full completion.
- **A fine-tuned complexity classifier** trained on real query/route data, with
  the current rule-based router as both the baseline and the labeller.
- **Persistent SQLite logging with export** for longitudinal analysis of routing
  quality and compute savings.
- **Support for additional Ollama models** (DeepSeek R1 is in; next Phi-3 and
  others) with a routing policy that picks among more than two tiers.
- **Automatic model benchmarking on startup** to measure each model's real
  latency profile and calibrate the compute-saved baseline dynamically.

## Project layout

```
localmind/
├── backend/
│   ├── main.py            # FastAPI app: /query /history /stats /health
│   ├── router.py          # classify → route → call model → log
│   ├── classifier.py      # complexity & privacy scoring + routing policy
│   ├── models/
│   │   ├── mistral_client.py
│   │   └── deepseek_client.py
│   ├── logger.py          # in-memory decision log + stats
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── components/
│   │   │   ├── QueryInput.jsx
│   │   │   ├── ResponsePanel.jsx
│   │   │   ├── LiveFeed.jsx
│   │   │   └── StatsBar.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
└── README.md
```
