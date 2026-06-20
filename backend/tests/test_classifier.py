"""Tests for the rule-based routing scorer.

Note: the spec calls this "classifier.py gate_score()", but ``gate_score`` lives
in ``gate.py`` (it composes ``classifier.score_complexity``/``score_privacy``
with the analysis-verb boost and the image hard-route). We test the real
function. Queries are calibrated to this classifier's actual calibration, which
scores conservatively.
"""

import gate


def test_trivial_query_routes_llama32():
    result = gate.gate_score("hi", False)
    assert result["expert"] == "llama3.2"
    assert result["complexity"] < 0.3


def test_moderate_query_routes_mistral():
    # A multi-clause generation/code task lands in the mistral band [0.3, 0.6).
    query = (
        "Write a Python function that parses a CSV file, validates each column, "
        "and handles malformed rows gracefully"
    )
    result = gate.gate_score(query, False)
    assert result["expert"] == "mistral"
    assert 0.3 <= result["complexity"] < 0.6


def test_complex_query_routes_deepseek():
    query = (
        "Compare and analyze the architectural tradeoffs between transformer "
        "attention mechanisms and state space models for real-time edge "
        "inference workloads"
    )
    result = gate.gate_score(query, False)
    assert result["expert"] == "deepseek-r1:7b"
    assert result["complexity"] >= 0.7


def test_image_subtask_hard_routes_llava():
    # The image hard-route ignores complexity/privacy entirely.
    result = gate.gate_score("anything at all", True)
    assert result["expert"] == "llava"
    assert result["hard_routed"] is True


def test_privacy_boost():
    # Low-complexity but privacy-heavy: the privacy override sends it to mistral
    # rather than the weakest llama3.2 expert.
    query = (
        "My salary is 90000, my bank account number is 12345678, and my "
        "mortgage statement shows my home address"
    )
    result = gate.gate_score(query, False)
    assert result["privacy"] > 0.6
    assert result["complexity"] < 0.3
    assert result["expert"] == "mistral"


def test_keyword_boost_for_comparison():
    # The analysis-verb ("compare") boost lifts effective complexity to >= 0.4
    # even for an otherwise trivial query.
    result = gate.gate_score("Compare cats and dogs", False)
    assert result["complexity"] >= 0.4
