"""Tests that the checked-in benchmark governance record remains valid."""

from benchmarks.check_scientific_drift import check


def test_frozen_benchmark_baseline_and_corpus_hash_pass_guardrails():
    assert check() == []
