# Databricks notebook source
from __future__ import annotations

import numpy as np

from apedp.regime_diagnostics import assess_regime_coverage


def test_monotonic_training_data_cannot_claim_unseen_turnover():
    result = assess_regime_coverage(
        np.array([12.0, 20.0, 50.0, 200.0, 1000.0]),
        np.array([980.0, 910.0, 820.0, 650.0, 380.0]),
        extrapolation_factor=1.25,
    )
    assert result.status == "UNDERDETERMINED"
    assert result.observed_turns == 0


def test_observed_turn_allows_regime_analysis():
    result = assess_regime_coverage(
        np.array([3.0, 5.0, 8.0, 12.0, 20.0]),
        np.array([620.0, 780.0, 920.0, 980.0, 910.0]),
        extrapolation_factor=1.25,
    )
    assert result.status == "COVERED"
    assert result.observed_turns >= 1