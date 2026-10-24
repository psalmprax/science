# Databricks notebook source
import numpy as np
import sympy as sp

from apedp.decay_models import fit_shifted_power_decay


def test_fit_shifted_power_decay_recovers_offset_power_law():
    day = np.arange(1, 61, dtype=float)
    target = 40.0 / (day + 0.5) ** 1.2 + 0.3
    expression = fit_shifted_power_decay(day, target, variable="day")
    predict = sp.lambdify(sp.Symbol("day"), expression, "numpy")
    assert np.mean((predict(day) - target) ** 2) < 1e-3
