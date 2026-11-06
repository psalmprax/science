"""Scope and evaluation-budget checks for the JetFlow integrator sweep."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from benchmarks.run_jetflow_integrator_sweep import run_sweep


def test_sweep_compares_jet2_and_euler_at_equal_evaluation_counts():
    report = run_sweep(seeds=(13,), step_counts=(1, 2), reference_steps=64)

    assert "not evidence of" in report["scope"]
    assert report["protocol"]["reference"] == "classical fixed-step RK4"
    assert len(report["rows"]) == 3 * 1 * 2 * 2
    for row in report["rows"]:
        assert row["velocity_evaluations"] == 2 * row["requested_steps"]
        assert math.isfinite(row["rmse_vs_rk4"])
        assert math.isfinite(row["relative_l2_vs_rk4"])
        assert row["runtime_s"] >= 0.0
