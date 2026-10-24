# Databricks notebook source
"""Tests for pathology-benchmark generators (review §23)."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.pathology_benchmarks import CASES, PathologyCase, generate


def test_all_cases_deterministic_and_shaped():
    assert len(CASES) == 6
    for case in CASES:
        x, y = generate(case)
        x2, y2 = generate(case)
        assert x.shape == (case.n_points,) and y.shape == (case.n_points,)
        assert np.array_equal(x, x2) and np.array_equal(y, y2)
        assert np.all(np.isfinite(y))


def test_oscillatory_truth_flagged_and_oscillates():
    case = next(c for c in CASES if c.case_id == "F-oscillatory-truth")
    assert case.oscillatory_truth is True and case.smooth_truth is False
    _, y = generate(case)
    assert int(np.sum(np.diff(np.sign(np.diff(y))) != 0)) >= 10


def test_smooth_cases_flagged_non_oscillatory():
    for case in CASES:
        if case.case_id == "F-oscillatory-truth":
            continue
        assert case.smooth_truth is True and case.oscillatory_truth is False


def test_unknown_case_rejected():
    with pytest.raises(KeyError):
        generate(PathologyCase("Z", "none", False, False, (0.0, 1.0), 10, 0.01, 0))
