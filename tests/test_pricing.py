"""Tests for the hosted-equivalent pricing arithmetic."""

import pytest

from underwriting.pricing import HOSTED_RATES_PER_MTOK, case_cost, span_cost


def test_rates_are_sonnet_class_per_mtok():
    """The hosted rate card is pinned to the Sonnet-class per-Mtok numbers."""
    assert HOSTED_RATES_PER_MTOK == {"input": 3.0, "output": 15.0}


def test_span_cost_simple_arithmetic():
    """span_cost scales each side by its rate and adds the two halves."""
    assert span_cost(1_000_000, 0) == pytest.approx(3.0)
    assert span_cost(0, 1_000_000) == pytest.approx(15.0)
    assert span_cost(500_000, 250_000) == pytest.approx(1.5 + 3.75)
    assert span_cost(0, 0) == 0.0


def test_case_cost_sums_over_spans():
    """case_cost sums the hand-worked span costs across a trace's spans."""
    # Third span worked out by hand: 1,000,000 input at $3.00/Mtok = $3.00,
    # 100,000 output at $15.00/Mtok = $1.50, so that span costs $4.50.
    spans = [
        {"usage": {"input_tokens": 1_000_000, "output_tokens": 0, "model": "x"}},
        {"usage": {"input_tokens": 0, "output_tokens": 1_000_000, "model": "x"}},
        {"usage": {"input_tokens": 1_000_000, "output_tokens": 100_000, "model": "x"}},
    ]
    expected = 3.0 + 15.0 + 4.5
    assert case_cost(spans) == pytest.approx(expected)


def test_case_cost_tolerates_missing_or_empty_usage():
    """Missing usage keys, null usage, and an empty span list all cost zero."""
    assert case_cost([]) == 0.0
    assert case_cost([{}, {"usage": None}, {"usage": {}}]) == 0.0
