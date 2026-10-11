"""Tests for the hosted-equivalent pricing arithmetic."""

import pytest

from underwriting.pricing import HOSTED_RATES_PER_MTOK, case_cost, span_cost


def test_rates_are_sonnet_class_per_mtok():
    assert HOSTED_RATES_PER_MTOK == {"input": 3.0, "output": 15.0}


def test_span_cost_simple_arithmetic():
    assert span_cost(1_000_000, 0) == pytest.approx(3.0)
    assert span_cost(0, 1_000_000) == pytest.approx(15.0)
    assert span_cost(500_000, 250_000) == pytest.approx(1.5 + 3.75)
    assert span_cost(0, 0) == 0.0


def test_case_cost_sums_over_spans():
    spans = [
        {"usage": {"input_tokens": 1_000_000, "output_tokens": 0, "model": "x"}},
        {"usage": {"input_tokens": 0, "output_tokens": 1_000_000, "model": "x"}},
        {"usage": {"input_tokens": 120, "output_tokens": 80, "model": "x"}},
    ]
    expected = 3.0 + 15.0 + span_cost(120, 80)
    assert case_cost(spans) == pytest.approx(expected)


def test_case_cost_tolerates_missing_or_empty_usage():
    assert case_cost([]) == 0.0
    assert case_cost([{}, {"usage": None}, {"usage": {}}]) == 0.0
