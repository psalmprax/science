"""Null-calibrated whiteness flag: fix verification + two-layer FPR.

All measured, fixed seeds.

The frozen Guard 2 compares |rho1| of holdout residuals against a single
constant (0.5). That miscalibrates in both directions:
  * OVER-refusal on persistent targets: a faithful model of an AR(2) target
    leaves AR(2)-shaped residuals (|rho1| up to ~0.9); the frozen 0.5 refuses
    them, penalising a good model for its target's persistence.
  * UNDER-refusal on white targets: the calibrated flag on a white target is
    ~0.15, far tighter than 0.5, so the frozen flag admits AR(1)-shaped misfit
    (|rho1| ~ 0.3) that a calibrated guard refuses.

The calibrated flag is derived from the target's own serial dependence by
block-demeaning (``whiteness_calibration.calibrate_flag``), so it is tight
where the target is tight and loose where the target is loose.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.whiteness_calibration import (
    calibrate_flag,
    check_whiteness,
    screen_structured_null_worlds,
)

FROZEN_FLAG = 0.5


def _rho1(arr: np.ndarray) -> float:
    c = arr - arr.mean()
    return float(np.dot(c[:-1], c[1:]) / float(np.dot(c, c)))


def _target(kind: str, n: int, seed: int) -> np.ndarray:
    r = np.random.RandomState(seed)
    if kind == "white":
        return r.randn(n)
    if kind == "ar2":
        e = np.zeros(n)
        for t in range(2, n):
            e[t] = 1.4 * e[t - 1] - 0.45 * e[t - 2] + 0.5 * r.randn()
        return e
    raise ValueError(kind)


def _mixed_residual(kind: str, n: int, seed: int, w_ar: float) -> np.ndarray:
    """Residual = w_ar*(target-shaped AR) + (1-w_ar)*white — intermediate |rho1|."""
    r = np.random.RandomState(seed)
    ar = _target(kind, n, seed + 100)
    resid = w_ar * ar + (1.0 - w_ar) * r.randn(n)
    return resid / np.std(resid)


def test_flag_is_monotone_in_target_persistence():
    flags = {}
    for kind in ("white", "ar2"):
        calib = calibrate_flag(_target(kind, 600, seed=7))
        assert calib is not None
        flags[kind] = calib
        if kind == "white":
            assert abs(calib.target_rho1) < 0.3, calib.target_rho1
        else:
            assert calib.target_rho1 > 0.8, calib.target_rho1
    assert flags["ar2"].flag > flags["white"].flag, (flags["white"].flag, flags["ar2"].flag)
    print("\n[flag] white: target_rho1=%.3f flag=%.3f | ar2: target_rho1=%.3f flag=%.3f"
          % (flags["white"].target_rho1, flags["white"].flag,
             flags["ar2"].target_rho1, flags["ar2"].flag))


def test_over_refusal_fixed_on_persistent_target():
    """AR(2) target + AR-shaped residual: frozen refuses, calibrated accepts."""
    y = _target("ar2", 600, seed=7)
    calib = calibrate_flag(y)
    assert calib is not None
    resid = _mixed_residual("ar2", 600, seed=200, w_ar=0.5)
    rho = abs(_rho1(resid))
    frozen_ok, _ = check_whiteness(resid, FROZEN_FLAG)
    calib_ok, why = check_whiteness(resid, calib.flag, calib)
    assert FROZEN_FLAG < rho < calib.flag, (rho, calib.flag)  # in the fixed band
    assert not frozen_ok and calib_ok, (rho, calib.flag, why)
    print("\n[over-refusal] |rho1|=%.3f: frozen 0.5 refuses, calibrated %.3f accepts"
          % (rho, calib.flag))


def test_under_refusal_fixed_on_white_target():
    """White target + AR-shaped misfit: frozen admits, calibrated refuses."""
    y = np.random.RandomState(23).randn(200)
    calib = calibrate_flag(y)
    assert calib is not None
    r = np.random.RandomState(23)  # AR(1) misfit of moderate magnitude
    resid = np.zeros(200)
    for t in range(1, 200):
        resid[t] = 0.42 * resid[t - 1] + r.randn()
    resid *= 0.4 / np.std(resid)
    rho = abs(_rho1(resid))
    frozen_ok, _ = check_whiteness(resid, FROZEN_FLAG)
    calib_ok, why = check_whiteness(resid, calib.flag, calib)
    assert calib.flag < rho < FROZEN_FLAG, (rho, calib.flag)  # in the fixed band
    assert frozen_ok and not calib_ok, (rho, calib.flag, why)
    print("\n[under-refusal] |rho1|=%.3f: frozen 0.5 admits, calibrated %.3f refuses"
          % (rho, calib.flag))


def test_layered_structured_null_fpr_reported_separately():
    """Proposer leaks on structured nulls; the calibrated guard stops all."""
    from apedp_dyn.proposers.rime_proposer import RimeProposer

    def engine(frame):
        return RimeProposer(feature_cols=["x"], target_col="target",
                            max_depth=3, budget=200).fit_and_propose(frame)

    rep = screen_structured_null_worlds(engine, n_worlds=24)
    print("\n[layered-fpr] %s" % rep)
    assert rep["proposer_promotions"] > 0, rep
    assert rep["system_promotions"] == 0, rep
    # Blocked promotions are strongly autocorrelated: the guard is catching
    # structure, not noise.
    assert rep["median_blocked_rho1"] > 0.5, rep


def test_unusable_target_yields_refusal_not_pass():
    """A flag that cannot be resolved must never read as whiteness."""
    ok, why = check_whiteness(np.arange(50, dtype=float), float("nan"))
    assert not ok, why
    assert calibrate_flag(np.array([1.0, 2.0])) is None          # too short
    assert calibrate_flag(np.full(100, 3.0)) is not None         # constant target
    ok_const, why_const = check_whiteness(np.ones(10), calib_ok_flag())
    assert not ok_const, why_const


def calib_ok_flag() -> float:
    return 0.5
