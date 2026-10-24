"""Frozen E4 protocol suite (doc S71): active disambiguation on synthetic ambiguity.

Synthetic truth (hidden from selector, P2): y = x^2 + 0.1 x^3, observed on
[-1,1] (noise 0.1) where H1=x^2, H2=x^2+0.1x^3, H3=x^2-0.1x^3 are
indistinguishable; candidate measurements query extrapolation points where
the cubic diverges. No trading data (per protocol: synthetic first).
"""

import numpy as np
import pytest

from apedp_dyn.epistemic.active_disambiguation import (
    compare_policies,
    discrimination_table,
    predict_all,
    run_trial,
    select_measurement,
)
from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

H1 = lambda x: x**2
H2 = lambda x: x**2 + 0.1 * x**3
H3 = lambda x: x**2 - 0.1 * x**3
HYPS = {"H1": H1, "H2": H2, "H3": H3}

from apedp_dyn.epistemic.active_disambiguation.policy import Measurement

MEASUREMENTS = [Measurement(f"x{x:+}", float(x), 0.1) for x in (-2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 0.3, -0.3)]
FLAT_MEASUREMENTS = [Measurement("near+0.3", 0.3, 0.1), Measurement("near-0.3", -0.3, 0.1)]

X_PRE = np.linspace(-1, 1, 500)


def _pre_scored(rng):
    y = H2(X_PRE) + rng.normal(0, 0.1, len(X_PRE))
    out = []
    for hid, fn in HYPS.items():
        resid = y - fn(X_PRE)
        c = CandidateDynamics(lhs="y", terms=(Term(hid, "a0"),), candidate_id=hid)
        out.append(ScoredCandidate(c, {"gate4_status": {"passed": True, "holdout_residuals": list(resid),
                                                       "holdout_rmse": float(np.sqrt(np.mean(resid**2))),
                                                       "holdout_obs_rms": 1.0}},
                                   train_loss=float(np.mean(resid**2)),
                                   holdout_rmse=float(np.sqrt(np.mean(resid**2)))))
    out.sort(key=lambda s: float(s.holdout_rmse))
    return out


def test_P1_input_genuinely_multi():
    adj = adjudicate(_pre_scored(np.random.default_rng(3)), bootstrap=False, data_origin="sim")
    assert "multi" in str(getattr(adj, "status", adj)).lower()


def test_P2P3_selector_blind_to_truth():
    a = select_measurement(HYPS, MEASUREMENTS)
    b = select_measurement({"H1": H1, "H2": H2, "H3": H3}, MEASUREMENTS)
    assert not a.abstained and a.selected.mid == b.selected.mid == "x-2.5"
    assert a.table == b.table  # determined by survivors + observation model only


def test_P4_noise_normalized():
    preds = predict_all(HYPS, MEASUREMENTS)
    hi = discrimination_table(preds, {m.mid: 0.05 for m in MEASUREMENTS})
    lo = discrimination_table(preds, {m.mid: 0.20 for m in MEASUREMENTS})
    assert hi[0].d_sigma == pytest.approx(4.0 * lo[0].d_sigma)


def test_P5_active_advantage():
    res = compare_policies(HYPS, MEASUREMENTS, X_PRE,
                           lambda rng: H2(X_PRE) + rng.normal(0, 0.1, len(X_PRE)), H2)
    assert res["oracle"]["p_unique"] >= res["active"]["p_unique"] > res["random"]["p_unique"]
    assert res["active"]["p_unique"] >= res["worst"]["p_unique"]


def test_P6_readjudication_recovers_truth():
    rng = np.random.default_rng(11)
    r = run_trial("active", HYPS, MEASUREMENTS, X_PRE,
                  H2(X_PRE) + rng.normal(0, 0.1, len(X_PRE)), H2, rng)
    assert "unique_support" in r.status_post.lower()
    assert r.winner == "H2"
    assert r.rounds <= 3


def test_P7_abstention_when_nothing_discriminates():
    sel = select_measurement(HYPS, FLAT_MEASUREMENTS)
    assert sel.abstained and sel.selected is None
    rng = np.random.default_rng(11)
    r = run_trial("active", HYPS, FLAT_MEASUREMENTS, X_PRE,
                  H2(X_PRE) + rng.normal(0, 0.1, len(X_PRE)), H2, rng)
    assert r.status_post == "INSUFFICIENT_DISCRIMINATION"


def test_P8_frozen_reproducibility():
    kw = {"hypotheses": HYPS, "measurements": MEASUREMENTS, "x_pre": X_PRE,
            "y_pre_fn": lambda rng: H2(X_PRE) + rng.normal(0, 0.1, len(X_PRE)), "truth_fn": H2}
    assert compare_policies(**kw) == compare_policies(**kw)
    assert select_measurement(HYPS, MEASUREMENTS).selected.mid == "x-2.5"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
