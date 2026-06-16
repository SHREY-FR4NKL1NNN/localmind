"""Query classifier for LocalMind.

Scores an incoming query on two independent axes — complexity and privacy
sensitivity — using transparent, rule-based heuristics, then applies the
routing policy to decide whether the query should be served by the fast,
lightweight model (Mistral 7B) or the higher-capability reasoning model
(DeepSeek R1).

The classifier is intentionally deterministic and explainable: every routing
decision can be traced back to the individual signals that produced it, which
matters for a system whose whole premise is trustworthy local inference.
"""

from __future__ import annotations

import re

# Phrases that signal a query asks for multi-step reasoning or generation
# rather than a simple lookup or arithmetic answer.
MULTI_STEP_INDICATORS = (
    "compare", "analyze", "analyse", "explain why", "write a", "design",
    "critique", "debate", "summarize", "summarise", "contrast", "evaluate",
    "derive", "prove", "step by step", "trade-off", "tradeoff", "pros and cons",
)

# Technical / domain jargon whose presence correlates with harder queries.
TECHNICAL_TERMS = (
    "algorithm", "architecture", "latency", "throughput", "concurrency",
    "asymptotic", "complexity", "transformer", "gradient", "tensor",
    "kubernetes", "distributed", "quantization", "inference", "embedding",
    "optimization", "optimisation", "regression", "neural", "topology",
    "state space", "mamba", "attention",
)

# Code-related terms that indicate a programming question.
CODE_TERMS = (
    "function", "class", "variable", "compile", "runtime", "regex", "python",
    "javascript", "rust", "sql", "recursion", "pointer", "async", "stack trace",
    "exception", "refactor", "syntax", "endpoint", " api ",
)

# Privacy-sensitive vocabulary, grouped by category with per-category weights.
FINANCIAL_TERMS = (
    "salary", "debt", "bank", "loan", "mortgage", "credit", "income", "tax",
    "invoice", "payment", "account number", "net worth",
)
HEALTH_TERMS = (
    "diagnosis", "prescription", "symptom", "medication", "disease", "patient",
    "therapy", "mental health", "doctor", "blood pressure",
)
ADDRESS_TERMS = (
    "street", "avenue", "zip code", "postal code", "apartment", "address",
)


def _clamp(value: float) -> float:
    """Constrain a score to the inclusive range [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    """Return how many of the given terms appear as substrings in ``text``."""
    return sum(1 for term in terms if term in text)


def score_complexity(query: str) -> float:
    """Estimate query complexity on a 0–1 scale.

    Combines five normalised signals — overall length, multi-step reasoning
    indicators, technical jargon density, code-related terms, and clause
    count — into a single weighted score. Returns a float in [0.0, 1.0].
    """
    text = query.lower()
    words = re.findall(r"[a-zA-Z']+", text)
    word_count = len(words)

    # Longer queries tend to be more involved; saturates around 60 words.
    length_score = _clamp(word_count / 60.0)

    # Two or more multi-step cues are enough to max out this signal.
    indicator_score = _clamp(_count_hits(text, MULTI_STEP_INDICATORS) / 2.0)

    # Three or more jargon / code hits saturate their respective signals.
    technical_score = _clamp(_count_hits(text, TECHNICAL_TERMS) / 3.0)
    code_score = _clamp(_count_hits(text, CODE_TERMS) / 3.0)

    # Approximate the number of clauses via punctuation and conjunctions.
    clause_markers = len(
        re.findall(r"[,;:]|\band\b|\bor\b|\bbut\b|\bwhich\b|\bbecause\b", text)
    )
    clause_score = _clamp(clause_markers / 5.0)

    complexity = (
        0.30 * length_score
        + 0.25 * indicator_score
        + 0.20 * technical_score
        + 0.10 * code_score
        + 0.15 * clause_score
    )
    return _clamp(complexity)


def score_privacy(query: str) -> float:
    """Estimate the privacy sensitivity of a query on a 0–1 scale.

    Detects structured PII (phone numbers, emails, SSN-like patterns, full
    names) and sensitive vocabulary (financial, health, and address terms),
    accumulating weighted evidence. Returns a float in [0.0, 1.0].
    """
    text = query.lower()
    score = 0.0

    # US-style SSN pattern is a strong privacy signal.
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", query):
        score += 0.5
    # A run of 7+ digits (optionally spaced/dashed) looks like a phone number.
    if re.search(r"\b(?:\+?\d[\s-]?){7,}\d\b", query):
        score += 0.4
    # Email addresses.
    if re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", query):
        score += 0.4
    # Two consecutive capitalised words often denote a person's full name.
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", query):
        score += 0.2

    for terms, weight in (
        (FINANCIAL_TERMS, 0.3),
        (HEALTH_TERMS, 0.3),
        (ADDRESS_TERMS, 0.2),
    ):
        if _count_hits(text, terms):
            score += weight

    return _clamp(score)


def classify(query: str) -> dict:
    """Classify a query and select a model route.

    Applies the LocalMind routing policy:

    * complexity < 0.4                      → Mistral 7B (lightweight is enough)
    * 0.4 <= complexity < 0.7, privacy > 0.6 → Mistral 7B (keep sensitive data
      on the smaller, faster path)
    * 0.4 <= complexity < 0.7, otherwise     → DeepSeek R1
    * complexity >= 0.7                      → DeepSeek R1 (needs full capability)

    Returns a dict with ``complexity``, ``privacy``, ``route``
    ("mistral" | "deepseek"), and a human-readable ``reasoning`` string.
    """
    complexity = round(score_complexity(query), 3)
    privacy = round(score_privacy(query), 3)

    if complexity < 0.4:
        route = "mistral"
        reasoning = (
            f"Low complexity ({complexity:.2f} < 0.40): the query is short and "
            f"direct, so the fast, lightweight Mistral 7B path is sufficient."
        )
    elif complexity < 0.7:
        if privacy > 0.6:
            route = "mistral"
            reasoning = (
                f"Moderate complexity ({complexity:.2f}) but high privacy "
                f"sensitivity ({privacy:.2f} > 0.60): keeping this query on the "
                f"smaller, faster Mistral 7B path to minimise exposure."
            )
        else:
            route = "deepseek"
            reasoning = (
                f"Moderate complexity ({complexity:.2f}) with low privacy "
                f"sensitivity ({privacy:.2f}): routing to DeepSeek R1 for "
                f"stronger reasoning."
            )
    else:
        route = "deepseek"
        reasoning = (
            f"High complexity ({complexity:.2f} >= 0.70): this looks like a "
            f"multi-step or technically demanding query, so DeepSeek R1's full "
            f"capability is warranted."
        )

    return {
        "complexity": complexity,
        "privacy": privacy,
        "route": route,
        "reasoning": reasoning,
    }
