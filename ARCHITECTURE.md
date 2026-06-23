# LocalMind — Architecture

## Overview

LocalMind is a Mixture-of-Experts–*inspired* routing layer over a set of local
language models served by [Ollama](https://ollama.com). Instead of sending every
query to one large model, it **decomposes** a request into sub-tasks, **routes**
each to the cheapest expert that can handle it, runs the chosen experts **in
parallel**, and **synthesizes** their answers into one reply. The problem it
solves is intelligent local LLM routing: matching each piece of work to the
right-sized model so simple asks stay fast and cheap while hard asks still get a
capable reasoner. Because everything runs locally, this buys **privacy** (no data
leaves the machine), **latency** (no network round-trip), and **cost** (no
per-token billing) — all without a single external API call.

## MoE Analogy (and where it deviates)

LocalMind borrows the *behavioural* shape of a Mixture-of-Experts layer.

**What's faithful:**
- **Sparse activation** — only the experts a query needs actually run; the rest
  stay idle. A trivial arithmetic sub-task never wakes the reasoning model.
- **Input-dependent expert selection** — the gate inspects each sub-task's
  characteristics (length, reasoning cues, jargon, privacy signals, image
  presence) and picks an expert from that.
- **Heterogeneous experts** — the experts are genuinely different models with
  different strengths: a fast generalist (Llama 3.2), a balanced generalist
  (Mistral), a heavy reasoner (DeepSeek R1), and a vision model (MiniCPM-V).
- **A gating function** that routes based on input, not a fixed pipeline.

**What deliberately deviates:**
- **The gate is rule-based, not learned end-to-end.** A real MoE trains its gate
  jointly with the experts via backprop. LocalMind's gate is a transparent set of
  heuristics (complexity/privacy scoring + an analysis-verb boost + an image hard
  rule).
- **The combiner does text synthesis, not weighted vector summation.** A real MoE
  combines expert outputs as vectors in latent space (a gate-weighted sum). Our
  experts emit *full natural-language text*, which has no comparable hidden-state
  vector to sum — so a small model reads the sub-answers and writes one unified
  reply instead. This is the single unavoidable structural deviation.
- **Sub-task decomposition is a heuristic** approximation of the learned,
  token-level routing a trained MoE performs implicitly.

> This system approximates MoE routing behavior using interpretable rules rather
> than trained parameters. The tradeoff is explainability over optimality — every
> routing decision has a human-readable reason.

## System Architecture

```
Query + Image?
     │
     ▼
┌─────────────┐
│   Gate      │  llama3.2 (fast)
│  (decompose │  ─ short query heuristic
│  + classify)│  ─ complexity/privacy scorer
└──────┬──────┘
       │
  ┌────┴─────────────────┐
  │                      │
  ▼                      ▼
Subtask 1           Subtask 2 (+ image?)
  │                      │
  ▼                      ▼
Expert A            Expert B
(mistral /          (deepseek /
llama3.2)           minicpm-v)
  │                      │
  └────────┬─────────────┘
           │  asyncio.gather()
           ▼
     ┌──────────┐
     │ Combiner │  llama3.2 synthesizes
     └──────────┘
           │
           ▼
    Final Response
    + Sparsity Metric
    + Routing Log
```

## Expert Roster

| Expert | Model | Domain | Capability | Activation trigger |
| ------ | ----- | ------ | ---------- | ------------------ |
| Fast / Gate | `llama3.2` (3.2B) | General + decomposition + combiner | Quick answers, JSON decomposition, synthesis | effective complexity `< 0.3`; also the gate and combiner |
| Generalist | `mistral` (7B) | General-purpose | Balanced quality/latency | `0.3 ≤ complexity < 0.6`, or privacy-override |
| Reasoner | `deepseek-r1:7b` (7B) | Multi-step reasoning | Comparison, analysis, technical depth | effective complexity `≥ 0.6` |
| Vision | `minicpm-v` (MiniCPM-V 2.6, 8B) | Multimodal | Image understanding, logo/text reading | any sub-task with an attached image (hard rule) |

> The vision expert's **internal routing key is `llava`** (the gate emits
> `expert: "llava"`, and the client module is `llava_client.py`) — a stable
> slot identifier retained from the original vision model. It is now **backed by
> the `minicpm-v` Ollama model**; the UI displays it as "MiniCPM-V".

## Routing Rules

`gate.gate_score(subtask, depends_on_image)` decides one expert per sub-task.
Scores come from `classifier.score_complexity` / `score_privacy`; an analysis-verb
boost is added to complexity *before* thresholding; images are a hard override.

| Condition (evaluated in order) | Expert | Why |
| ------------------------------ | ------ | --- |
| `depends_on_image` is true | `llava` | **Hard rule** — vision work can only go to the vision model, regardless of any score. `hard_routed = true`. |
| effective complexity `≥ 0.6` | `deepseek-r1:7b` | Reasoning-heavy; needs the strongest model. |
| privacy `> 0.6` (and complexity `< 0.6`) | `mistral` | **Privacy override** — sensitive content gets the balanced model rather than the weakest one, even when complexity is low. |
| complexity `< 0.3` | `llama3.2` | Trivial; the fast expert is enough. |
| otherwise (`0.3 ≤ complexity < 0.6`) | `mistral` | Moderate; the generalist fits. |

**Keyword (analysis-verb) boost:** if the sub-task contains a comparison/analysis
verb (`compare`, `analyze`, `analyse`, `evaluate`, `contrast`, `tradeoff`,
`trade-off`), a fixed `+0.30` is added to complexity before thresholding. This
exists because the length-weighted complexity heuristic under-scores a short
analytical clause once decomposition isolates it (e.g. "Compare X to Y" is only a
few words); the boost lifts such asks back onto the reasoning path. The reported
complexity is the post-boost (effective) value, and the reasoning string shows
the base score, the matched verbs, and the boost for full explainability.

## Streaming Architecture

The streaming endpoint (`POST /query/decomposed/stream`) runs the same pipeline
but emits Server-Sent Events as work happens. Concurrency uses an **`asyncio.Queue`
fan-in**: one producer task per expert streams its client and pushes tagged
chunks onto a single shared queue; a single consumer drains the queue and emits
each chunk the instant it arrives — so tokens from different experts **interleave**
in whatever order the experts produce them. A per-producer sentinel signals
completion; a failing expert is isolated (its error surfaces as a token + an
`expert_done`, never cancelling the others). The combiner only starts after every
`expert_done` has fired.

```
Client                    Server (stream_decomposed)
  │   POST /…/stream          │
  │ ────────────────────────► │
  │                           │ gate.decompose + gate_score
  │ ◄──── event: gate_complete│  (routing known before any token)
  │                           │
  │                           │ spawn 1 producer task / expert ─┐
  │                           │                                 │ asyncio.Queue
  │ ◄──── event: expert_token │  idx=0  ◄───────────────────────┤  (interleaved
  │ ◄──── event: expert_token │  idx=1  ◄───────────────────────┤   fan-in)
  │ ◄──── event: expert_token │  idx=0  ◄───────────────────────┤
  │ ◄──── event: expert_token │  idx=1  ◄───────────────────────┘
  │ ◄──── event: expert_done  │  idx=0
  │ ◄──── event: expert_done  │  idx=1
  │ ◄──── event: sparsity     │  (after all experts done)
  │ ◄──── event: combiner_token│ llama3.2 synthesis, streamed
  │ ◄──── event: combiner_token│
  │ ◄──── event: done         │  combined_response + total_latency_ms
```

Event order is always: `gate_complete → expert_token* (interleaved) →
expert_done (×N) → sparsity → combiner_token* → done`.

## Limitations and Honest Caveats

- **Rule-based gate.** Optimal routing really wants a *trained* gate; ours uses
  heuristics that can misclassify edge cases. Concretely, the length-weighted
  complexity score under-rates short-but-hard asks: an isolated "Compare
  transformer architectures to state space models" clause scores low until the
  analysis-verb boost rescues it, and a single analytical sentence is otherwise
  prone to being shredded into fragments by the 3B decomposition model (handled
  with a dedicated short-circuit, but illustrative of the heuristic's limits).
- **Single-GPU VRAM contention.** True parallel inference needs the models to
  co-reside in VRAM. On ~8 GB with ~4.5 GB models, Ollama can't hold two large
  models at once, so it **time-slices** them — meaning the `asyncio.gather` /
  queue dispatch is genuinely concurrent at the code level but functionally
  **sequential at the GPU level**. `total_latency_ms` still tracks the slowest
  expert (+ combiner), not the sum, but the wall-clock win is bounded by the
  hardware, not the routing code.
- **DeepSeek thinking traces.** This Ollama build of `deepseek-r1:7b` (Q4_K_M)
  does **not** emit `<think>…</think>` tags in its streamed output. The detection
  code (including the cross-chunk carry buffer for split tags) is implemented and
  unit-tested, but is currently inactive against this model — the reasoning trace
  panel stays empty and no literal tags leak.
- **Vision model accuracy.** The vision expert was upgraded from `llava` (7B,
  Q4_0) to **MiniCPM-V 2.6 (8B)** for much stronger fine-grained recognition:
  llava misidentified an NVIDIA RTX workstation card as "a PlayStation console,"
  whereas MiniCPM-V reads the on-device "NVIDIA" logo and correctly identifies it
  as an NVIDIA RTX-series GPU. It still can't always pin the exact SKU (e.g. it
  guessed "RTX 3080" for an RTX 6000) — reading the small model wordmark is hard
  when the vision encoder downsamples a wide marketing banner to a low-resolution
  square. (Llama 3.2 Vision was evaluated as the upgrade but its `mllama`
  architecture won't load on this Ollama build's `llama-server` runner — not a
  VRAM issue; it fails at architecture parsing even on CPU.)
- **Text combiner vs vector combination.** As above — a deliberate, documented
  deviation from textbook MoE, unavoidable when experts emit full text.

## What I'd Build Next

- **Learned gate:** fine-tune a small classifier on real query→route data
  collected from this system, with the rule-based gate as both baseline and
  labeller.
- **Streaming decomposition:** start routing and running sub-tasks before the full
  decomposition has completed.
- **Multi-GPU support:** true parallel expert inference with one model per GPU,
  removing the VRAM time-slicing bottleneck.
- **Persistent conversation context** carried across sub-tasks and turns.
- **Additional experts:** code-specialized (CodeLlama), multilingual (Qwen),
  math-specialized (Mathstral).

## Scope Expansion Note

Two pieces were built **beyond** the original MoE routing spec as deliberate
extensions: **recursive decomposition** (re-splitting a still-compound sub-task
one level deeper, bounded by depth and leaf caps) and the **full frontend
streaming UI** (the animated gating diagram, live per-expert token panels, the
DeepSeek reasoning-trace panel, and the sparsity/combiner views). Neither is
required for the core decompose → gate → parallel → combine flow to function.
