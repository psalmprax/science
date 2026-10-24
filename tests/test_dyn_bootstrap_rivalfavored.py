"""Regression: bootstrap 'rival-favored' must refuse promotion.

Spec (bootstrap.py): CI entirely below 0 -> MULTI_HYPOTHESIS with
rival-favored note. The live tree previously fell through to UNIQUE_SUPPORT
while printing "confirmed by the paired block bootstrap (branch
'rival-favored')" -- promoting the point-winner against resampled evidence.

Via genuine squared-error inputs this branch is nearly unreachable
(mean(d) < 0 implies point Delta < 0, which takes the early MULTI exit),
so the path is pinned with a stubbed tiebreak; branch semantics are pinned
directly against tiebreak with constant series.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic import bootstrap as boot_mod
from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _pair():
    ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x", "x3", "v")))
    tb = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "v", "abs_v_v")))
    ca = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="m1")
    cb = CandidateDynamics(lhs="xddot", terms=tb, candidate_id="m2")
    g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.1,
          "holdout_n_samples": 100}
    sa = ScoredCandidate(ca, {"gate4_status": g4}, train_loss=0.5, holdout_rmse=0.20)
    sb = ScoredCandidate(cb, {"gate4_status": g4}, train_loss=0.52, holdout_rmse=0.50)
    return sa, sb


def test_tiebreak_rival_favored_branch_semantics():
    """Constant series with rival everywhere better -> 'rival-favored'."""
    top = np.full(100, 1.0)
    riv = np.full(100, 0.01)
    tb = boot_mod.tiebreak(0.5, 0.05, top, riv, n_boot=200, seed=0)
    assert tb["branch"] == "rival-favored"
    assert tb["hi"] < 0


def test_rival_favored_refuses_promotion(monkeypatch):
    """A bootstrap verdict against the point-winner must not promote."""
    def fake_tiebreak(delta_point, tau_epi, resid_top, resid_rival,
                      n_boot=2000, seed=0):
        return {"lo": -0.5, "hi": -0.1, "mean": -0.3,
                "delta_point": float(delta_point), "tau_epi": float(tau_epi),
                "branch": "rival-favored", "n": len(resid_top), "n_boot": n_boot}

    monkeypatch.setattr(boot_mod, "tiebreak", fake_tiebreak)
    sa, sb = _pair()
    adj = adjudicate([sa, sb], tau_epi=0.05, bootstrap=True,
                     resid_best=[0.04] * 100, resid_rival=[0.25] * 100)
    assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS
    assert adj.promoted is None
    assert adj.bootstrap_branch == "rival-favored"
    assert "rival" in adj.reason


def test_multi_branch_still_refuses(monkeypatch):
    """The pre-existing 'multi' path is unchanged by the fix."""
    def fake_tiebreak(delta_point, tau_epi, resid_top, resid_rival,
                      n_boot=2000, seed=0):
        return {"lo": -0.1, "hi": 0.4, "mean": 0.15,
                "delta_point": float(delta_point), "tau_epi": float(tau_epi),
                "branch": "multi", "n": len(resid_top), "n_boot": n_boot}

    monkeypatch.setattr(boot_mod, "tiebreak", fake_tiebreak)
    sa, sb = _pair()
    adj = adjudicate([sa, sb], tau_epi=0.05, bootstrap=True,
                     resid_best=[0.04] * 100, resid_rival=[0.25] * 100)
    assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS
    assert adj.bootstrap_branch == "multi"
