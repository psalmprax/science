"""Referee whiteness-guard sensitivity: read-only sweep over sole_rho_flag.

Does NOT touch the frozen default (0.5): every call passes an explicit
flag; one case asserts the default call matches the flag=0.5 outcome
(default intact). Planted sole survivor with good fidelity (RMSE/obs 0.10)
but AR-structured residuals (squared-error rho1 ~= 0.41): the verdict must
flip between flag 0.3 (refuse) and 0.5/0.7 (promote), locating the gate's
operating point on this case. This measures #7 sensitivity; it does not
re-preregister anything.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _sole_ar_case():
    rng = np.random.RandomState(4)
    n = 400
    x = rng.randn(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.55 * e[t - 1] + rng.randn()
    e = e / np.std(e) * 0.3
    y = 3.0 * x + e
    ntr = 200
    w, *_ = np.linalg.lstsq(np.column_stack([np.ones(ntr), x[:ntr]]), y[:ntr], rcond=None)
    resid = y[ntr:] - (w[0] + w[1] * x[ntr:])
    obs_rms = float(np.sqrt(np.mean(y[ntr:] ** 2)))
    sq = resid ** 2
    c = sq - sq.mean()
    rho_sq = float(np.dot(c[:-1], c[1:]) / np.dot(c, c))

    def mk():
        g4 = {"passed": True, "holdout_obs_rms": obs_rms, "holdout_noise_sigma": float(np.std(resid)),
              "holdout_n_samples": len(resid), "holdout_residuals": resid.tolist(),
              "holdout_sq_error": (resid ** 2).tolist()}
        cand = CandidateDynamics(lhs="y", terms=(Term("slope", "beta"),), candidate_id="linsole")
        tr = float(np.mean((y[:ntr] - (w[0] + w[1] * x[:ntr])) ** 2))
        fit = FitResult(param_values={"beta": float(w[1]), "alpha": float(w[0])},
                        train_loss=tr, nfev=1, success=True, message="OLS")
        return ScoredCandidate(cand, fit=fit, train_loss=tr,
                               holdout_rmse=float(np.sqrt(np.mean(resid ** 2))),
                               provenance={"gate4_status": g4})

    return mk, resid, rho_sq


def test_whiteness_guard_operating_point():
    mk, resid, rho_sq = _sole_ar_case()
    assert 0.3 < rho_sq < 0.5, rho_sq  # straddles the sweep by construction
    kw = {"resid_sole": resid.tolist(), "sole_survivor_rule": True,
              "data_origin": DataOrigin.SIMULATION.value,
              "benchmark_level": BenchmarkLevel.L0_SELECTION.value}
    v03 = adjudicate([mk()], sole_rho_flag=0.3, **kw)
    v05 = adjudicate([mk()], sole_rho_flag=0.5, **kw)
    v07 = adjudicate([mk()], sole_rho_flag=0.7, **kw)
    assert v03.status == EpistemicStatus.INSUFFICIENT_EVIDENCE, v03.status
    assert v03.promoted is None
    assert v05.status == EpistemicStatus.UNIQUE_SUPPORT, v05.status
    assert v07.status == EpistemicStatus.UNIQUE_SUPPORT, v07.status
    print(f"\n[rho-sweep] rho1_sqerr={rho_sq:.3f}: 0.3->refuse, 0.5->promote, 0.7->promote")


def test_frozen_default_untouched():
    mk, resid, _ = _sole_ar_case()
    kw = {"resid_sole": resid.tolist(), "sole_survivor_rule": True,
              "data_origin": DataOrigin.SIMULATION.value,
              "benchmark_level": BenchmarkLevel.L0_SELECTION.value}
    default = adjudicate([mk()], **kw)
    explicit = adjudicate([mk()], sole_rho_flag=0.5, **kw)
    assert default.status == explicit.status == EpistemicStatus.UNIQUE_SUPPORT
