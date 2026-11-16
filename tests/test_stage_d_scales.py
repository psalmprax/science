"""Stage D refines in two unit-free x scales and keeps the better accepted result.

Until 2026-11-16 it used ``x / max|x|`` alone. On TIER_B_01 (Planck in
wavelength) that frame needs an amplitude and a denominator coefficient more
than ten times their caps, every start railed, the refinement was rejected, and
the synthesizer abstained at R-squared 0.940 on a problem the benchmark report
records above 0.999. ``tests/test_tier_bc_benchmark.py`` failed for that reason
from the first commit in this repository's history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from apedp.model_synthesis import AutonomousModelSynthesizer
from apedp.model_synthesis_helpers import _stage_d_x_scales

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tier_bc_benchmark import load_tier_dataset

PLANCK_GRID = np.linspace(0.4, 3.5, 50)


def test_the_first_scale_is_the_largest_magnitude():
    """Order is the tie rule: an equal second result must not displace the first."""
    scales = _stage_d_x_scales(PLANCK_GRID)
    assert scales[0] == pytest.approx(3.5)
    assert len(scales) == 2


def test_the_second_scale_puts_samples_on_both_sides_of_one():
    _, centre = _stage_d_x_scales(PLANCK_GRID)
    normalized = PLANCK_GRID / centre
    assert normalized.min() < 1.0 < normalized.max()
    assert centre == pytest.approx(float(np.exp(np.mean(np.log(PLANCK_GRID)))))


@pytest.mark.parametrize("unit", [1e-9, 1e-3, 1e3, 1e6])
def test_both_scales_follow_a_change_of_x_units(unit):
    """Equivariance is what makes the normalized problem the same problem."""
    base = _stage_d_x_scales(PLANCK_GRID)
    scaled = _stage_d_x_scales(PLANCK_GRID * unit)
    assert scaled == pytest.approx(tuple(s * unit for s in base), rel=1e-12)


def test_sign_does_not_enter_the_scale():
    assert _stage_d_x_scales(-PLANCK_GRID) == pytest.approx(_stage_d_x_scales(PLANCK_GRID))


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([]), (1.0,)),
        (np.array([0.0, 0.0]), (1.0,)),
        (np.array([np.nan, np.inf]), (1.0,)),
        (np.array([2.0, 2.0, 2.0]), (2.0,)),
        (np.array([0.0, 4.0]), (4.0,)),
    ],
)
def test_degenerate_inputs_give_one_usable_scale(values, expected):
    """No zero, NaN or duplicate scale reaches the optimizer."""
    assert _stage_d_x_scales(values) == pytest.approx(expected)


def test_refinement_keeps_the_better_of_the_scales(monkeypatch):
    """``refine_stage_d`` is a minimum over accepted results, earliest first on a tie."""
    synthesizer = AutonomousModelSynthesizer()
    results = {3.0: 0.5, 1.5: 0.2}

    class _Refined:
        def __init__(self, scale):
            self.scale, self.val_mse = scale, results[scale]

    monkeypatch.setattr("apedp.model_synthesis._stage_d_x_scales", lambda x: (3.0, 1.5))
    monkeypatch.setattr(
        synthesizer, "_refine_stage_d_at",
        lambda model, xt, yt, xv, yv, scale: _Refined(scale) if results[scale] is not None else None,
    )
    assert synthesizer.refine_stage_d(None, PLANCK_GRID, None, None, None).scale == 1.5

    results.update({3.0: 0.2, 1.5: 0.2})
    assert synthesizer.refine_stage_d(None, PLANCK_GRID, None, None, None).scale == 3.0

    results.update({3.0: None, 1.5: 0.9})
    assert synthesizer.refine_stage_d(None, PLANCK_GRID, None, None, None).scale == 1.5

    results.update({3.0: None, 1.5: None})
    assert synthesizer.refine_stage_d(None, PLANCK_GRID, None, None, None) is None


@pytest.mark.parametrize("unit", [1.0, 1e-3, 1e2])
def test_planck_is_refined_and_not_refused_in_any_x_unit(unit):
    """The measured case, end to end, and in units the caps were not written for."""
    _, (x_tr, y_tr), (x_val, y_val) = load_tier_dataset("tier_b", "TIER_B_01")
    top = AutonomousModelSynthesizer().synthesize(x_tr * unit, y_tr, x_val * unit, y_val)[0]
    assert top.abstention_reason is None
    assert top.structural_family == "rational_exponential_envelope"
    assert top.diagnostics.get("stage_d_refined") is True
    assert 1.0 - top.val_mse / np.var(y_val) > 0.999


def test_the_single_scale_refinement_still_rails_on_planck():
    """The defect, kept reproducible: ``x / max|x|`` alone accepts nothing here.

    If this starts passing a refinement, the caps or the form changed and the
    second scale may no longer be what carries TIER_B_01.
    """
    _, (x_tr, y_tr), (x_val, y_val) = load_tier_dataset("tier_b", "TIER_B_01")
    synthesizer = AutonomousModelSynthesizer()
    coarse = synthesizer.fit_candidate(
        "(A * x**2 / (1.0 + B * x**2.5)) * exp(-C * x**1.5)", "rational_exponential_envelope",
        x_tr, y_tr, x_val, y_val,
    )
    assert coarse is not None
    largest, centre = _stage_d_x_scales(x_tr)
    assert synthesizer._refine_stage_d_at(coarse, x_tr, y_tr, x_val, y_val, largest) is None
    refined = synthesizer._refine_stage_d_at(coarse, x_tr, y_tr, x_val, y_val, centre)
    assert refined is not None and refined.val_mse < 0.05 * coarse.val_mse
