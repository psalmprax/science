"""Focused tests for repeated-seed MMLM benchmark reporting."""

import pytest

from apedp_mmlm.models import Milestone1BenchmarkSuite


def test_repeated_seed_summary_reports_raw_metrics_and_variability():
    summary = Milestone1BenchmarkSuite().run_repeated_milestone_1([31, 47])

    assert summary["protocol_summary"]["replicates"] == 2
    assert summary["protocol_summary"]["seeds"] == [31, 47]
    assert summary["protocol_summary"]["variability_scope"] == (
        "model initialization on one fixed synthetic split"
    )
    assert summary["protocol_summary"]["external_or_real_world_evidence"] is False
    assert [run["protocol_summary"]["seed"] for run in summary["runs"]] == [31, 47]

    perplexity = summary["metric_summary"]["B0"]["held_out_perplexity"]
    assert len(perplexity["values"]) == 2
    assert perplexity["mean"] == pytest.approx(sum(perplexity["values"]) / 2)
    assert perplexity["sample_std"] >= 0.0
    assert summary["hypothesis_verdict_frequencies"]["H2"] == {"NOT_ASSESSED": 1.0}


def test_repeated_seed_summary_requires_multiple_unique_integer_seeds():
    suite = Milestone1BenchmarkSuite()

    with pytest.raises(ValueError, match="at least two seeds"):
        suite.run_repeated_milestone_1([31])
    with pytest.raises(ValueError, match="unique"):
        suite.run_repeated_milestone_1([31, 31])
    with pytest.raises(TypeError, match="integers"):
        suite.run_repeated_milestone_1([31, 47.0])  # type: ignore[list-item]
