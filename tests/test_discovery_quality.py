# Databricks notebook source
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp.discovery_quality import multi_seed_discovery, screen_candidate


def test_screen_candidate_rejects_nonfinite_extrapolation():
    x = {"x": np.array([0.5, 0.75, 0.9])}
    quality = screen_candidate(1 / (1 - sp.Symbol("x")), x, np.array([2.0, 4.0, 10.0]))
    assert quality.finite is True
    assert quality.output_bounds[1] > 10.0


def test_multiseed_report_exposes_stability():
    class FakeEngine:
        def run_discovery(self, x_data, y_data, generations, pop_size, seed):
            return sp.Symbol("x"), 0.1, []

    report = multi_seed_discovery(lambda seed: FakeEngine(), {"x": np.arange(5.0)}, np.arange(5.0))
    assert report["seed_count"] == 5
    assert report["successful_seed_count"] == 5
    assert report["stability_rate"] == 1.0
