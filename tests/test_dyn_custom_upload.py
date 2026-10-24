"""Tests for custom dataset upload and universal equation discovery."""
from __future__ import annotations

import numpy as np

from apedp_dyn.sandbox.custom_upload import parse_csv_data, run_upload_discovery


def test_parse_csv_data_basic():
    csv_text = """t,x,y
0.0, 1.0, 2.0
0.1, 1.2, 2.4
0.2, 1.5, 3.0
0.3, 1.8, 3.6
0.4, 2.0, 4.0
0.5, 2.2, 4.4
"""
    parsed = parse_csv_data(csv_text)
    assert parsed["time_col"] == "t"
    assert parsed["target_col"] == "y"
    assert parsed["feature_cols"] == ["x"]
    assert len(parsed["numeric_cols"]["y"]) == 6


def test_multivariable_upload_discovery():
    # Synthetic physical data: y = 2.0 * x1 + 0.5 * (x2**2)
    rng = np.random.default_rng(42)
    n = 60
    x1 = rng.uniform(1.0, 5.0, size=n)
    x2 = rng.uniform(0.5, 3.0, size=n)
    y = 2.0 * x1 + 0.5 * (x2**2) + rng.normal(0, 0.01, size=n)

    rows = ["x1,x2,y"]
    for i in range(n):
        rows.append(f"{x1[i]:.4f},{x2[i]:.4f},{y[i]:.4f}")
    csv_text = "\n".join(rows)

    adj, cands, _series_info, extra = run_upload_discovery(csv_text, target_col="y")
    assert adj.status.value in ("unique_support", "multi_hypothesis", "insufficient_evidence")
    assert len(cands) > 0
    assert extra["target_variable"] == "y"
    assert "x1" in extra["feature_variables"] and "x2" in extra["feature_variables"]


def test_1d_series_upload_discovery():
    # 1D trajectory: damped sine wave
    t = np.linspace(0, 10, 80)
    prices = 100.0 * np.exp(-0.02 * t) * (1.0 + 0.1 * np.sin(1.2 * t))
    rows = ["date,price"]
    for i, p in enumerate(prices):
        rows.append(f"2023-01-{i%28+1:02d},{p:.4f}")
    csv_text = "\n".join(rows)

    adj, cands, series_info, extra = run_upload_discovery(csv_text, target_col="price")
    assert adj.status.value in ("unique_support", "multi_hypothesis", "insufficient_evidence")
    assert len(cands) > 0
    assert extra["dataset_type"] == "1D Dynamical Time Series"
    assert len(series_info["actual"]) > 10
