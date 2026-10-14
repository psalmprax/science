"""Referee resolution-floor sensitivity (#7): read-only sweep, default untouched.

REL_FLOOR=0.05 (adjudicate.py) is documented against two anchors — an
estimator-noise ceiling (0.0121, artifact below) and a Duffing-pair bracket
(0.0245 must stay MULTI, 0.700 must stay UNIQUE) — but the constant itself
is NOT spec-pinned (absent from canonical_spec + JSON). This suite does not
change it. It passes explicit tau_epi (= f * obs_rms) to adjudicate() and
measures where verdicts flip, on three planted rival pairs:

  * near-tie (RMSE ratio ~0.03, inside the bracket): UNIQUE at small floors,
    MULTI at large floors; f=0.05 must read MULTI.
  * decisive (ratio ~0.7): UNIQUE at f=0.05 AND f=0.1 (below upper bound).
  * identical (ratio 0.0): MULTI at every floor — never promote identicals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

ROOT = Path(__file__).resolve().parent.parent


def _sc(cid: str, resid: np.ndarray, terms=("slope",), n_train: int = 200) -> ScoredCandidate:
    resid = np.asarray(resid, dtype=float)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    g4 = {"passed": True, "holdout_obs_rms": 3.0, "holdout_noise_sigma": float(np.std(resid)),
          "holdout_n_samples": len(resid), "holdout_residuals": resid.tolist(),
          "holdout_sq_error": (resid ** 2).tolist()}
    cand = CandidateDynamics(lhs="y", terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                           candidate_id=cid)
    fit = FitResult(param_values={"beta": 1.0, "alpha": 0.0}, train_loss=rmse,
                    nfev=1, success=True, message="OLS")
    return ScoredCandidate(cand, fit=fit, train_loss=rmse, holdout_rmse=rmse,
                           provenance={"gate4_status": g4})


def _adj(s1, s2, r1, r2, tau_f):
    return adjudicate(
        [s1, s2], resid_best=np.asarray(r1, dtype=float).tolist(),
        resid_rival=np.asarray(r2, dtype=float).tolist(),
        bootstrap=False, data_origin=DataOrigin.SIMULATION.value,
        benchmark_level=BenchmarkLevel.L0_SELECTION.value,
        tau_epi=float(tau_f * 3.0))


def test_anchor_artifact_present():
    # The docstring's lower-bound anchor must exist on disk (AGENTS.md 1.4:
    # never claim files unverified). Content check is light: problems key.
    p = ROOT / "outputs" / "d2a_ab_report.json"
    assert p.exists(), "anchor artifact missing — lower bound unverifiable"
    d = json.loads(p.read_text())
    assert "problems" in d, list(d.keys())[:8]


def test_floor_sweep_near_tie_pair():
    rng = np.random.RandomState(8)
    n = 300
    e1 = rng.randn(n) * 0.3
    e2 = e1 + 0.257 * rng.randn(n)
    r1 = float(np.sqrt(np.mean(e1 ** 2)))
    r2 = float(np.sqrt(np.mean(e2 ** 2)))
    ratio = (r2 - r1) / 3.0
    assert 0.015 < ratio < 0.06, ratio  # inside the documented bracket
    s1, s2 = _sc("best", e1, ("slope",)), _sc("rival", e2, ("slope", "curve"))
    verdicts = {}
    for f in (0.005, 0.02, 0.05, 0.15):
        a = _adj(s1, s2, e1, e2, f)
        verdicts[f] = a.status.value
    print(f"\n[floor-sweep] ratio={ratio:.4f}: " + ", ".join(f"{k}->{v}" for k, v in verdicts.items()))
    assert verdicts[0.005] == EpistemicStatus.UNIQUE_SUPPORT.value
    assert verdicts[0.05] == EpistemicStatus.MULTI_HYPOTHESIS.value
    assert verdicts[0.15] == EpistemicStatus.MULTI_HYPOTHESIS.value
    seq = [verdicts[f] for f in (0.005, 0.02, 0.05, 0.15)]
    assert seq == sorted(seq, key=lambda v: 0 if v == "unique_support" else 1), seq  # monotone


def test_floor_sweep_decisive_and_identical_pairs():
    rng = np.random.RandomState(9)
    n = 300
    e1 = rng.randn(n) * 0.3
    e2 = e1 + 2.4 * rng.randn(n)  # decisive gap
    r1 = float(np.sqrt(np.mean(e1 ** 2)))
    r2 = float(np.sqrt(np.mean(e2 ** 2)))
    ratio = (r2 - r1) / 3.0
    assert ratio > 0.5, ratio
    s1, s2 = _sc("best", e1, ("slope",)), _sc("rival", e2, ("slope", "curve"))
    for f in (0.05, 0.1):
        a = _adj(s1, s2, e1, e2, f)
        assert a.status == EpistemicStatus.UNIQUE_SUPPORT.value, (f, a.status)
    # identical residuals: never promote, at any floor
    for f in (0.01, 0.05, 0.1):
        a = _adj(_sc("ratio", e1, ("one", "pratio")), _sc("quad", e1, ("one", "p2")), e1, e1, f)
        assert a.status == EpistemicStatus.MULTI_HYPOTHESIS.value, (f, a.status)
    print(f"\n[floor-pairs] decisive ratio={ratio:.3f} UNIQUE at 0.05+0.1; identical never promoted")
