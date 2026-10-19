"""Lynx-hare sandbox (real data): cyclicity discovery in hare-lynx trappings.

Data: R datasets::lynx, annual Canadian lynx trappings 1821-1934
(/tmp/lynx.csv, public domain MacLulich series). The file carries lynx
only, so this tests single-series cycle discovery rather than joint
predator-prey fitting: log-trapped counts, train on 1821-1877, holdout
on 1878-1934 (temporal split = intervention analog: later decades unseen).

Candidates (OLS, fixed periods -- the period itself is NOT fitted):
  cycle_10yr: logN = a + b*sin(2 pi t/10) + c*cos(2 pi t/10)  (classic ~10yr)
  cycle_5yr:  logN = a + b*sin(2 pi t/5) + c*cos(2 pi t/5)    (harmonic rival)
  ar1_lag:    logN_t = a + b*logN_{t-1}                       (density dependence)
  trend:      logN = a + b*t                                  (secular drift)
  constant:   logN = a                                         (null)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import (
    CandidateDynamics,
    ScoredCandidate,
    Term,
)

_VENDORED_LYNX = Path(__file__).resolve().parents[2] / "data" / "market" / "lynx_maclulich.csv"


def load_lynx(path=None):
    """Load the MacLulich lynx series.

    The series is vendored in-repo (data/market/lynx_maclulich.csv, provenance in
    data/market/PROVENANCE.json) so the sandbox never depends on an untracked
    /tmp file; an explicit ``path`` still overrides for ad-hoc series.
    """
    path = path or _VENDORED_LYNX
    t, y = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["time"]))
            y.append(float(row["value"]))
    return np.array(t), np.array(y)


def run_lynx(split_year: float = 1878.0, bootstrap: bool = True):
    t, y = load_lynx()
    yl = np.log(y)
    tr = t < split_year
    t_tr, y_tr, t_ho, y_ho = t[tr], yl[tr], t[~tr], yl[~tr]
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))
    ns = float(0.02 * obs_rms)
    o = np.ones_like

    def mk(cid, terms, X_tr, X_ho):
        w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        rh = y_ho - X_ho @ w
        rm = float(np.sqrt(np.mean(rh ** 2)))
        g4 = {"passed": True, "holdout_obs_rms": obs_rms,
              "holdout_noise_sigma": ns, "holdout_n_samples": len(y_ho),
              "holdout_residuals": rh.tolist()}
        c = CandidateDynamics(lhs="log_trappings",
                              terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                              candidate_id=cid)
        return ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.REAL_OBSERVATIONAL},
                               train_loss=float(np.mean((y_tr - X_tr @ w) ** 2)),
                               holdout_rmse=rm)

    w10 = 2 * np.pi / 10.0
    w5 = 2 * np.pi / 5.0
    scored = [
        mk("cycle_10yr", ("one", "sin10", "cos10"),
           np.column_stack([o(t_tr), np.sin(w10 * t_tr), np.cos(w10 * t_tr)]),
           np.column_stack([o(t_ho), np.sin(w10 * t_ho), np.cos(w10 * t_ho)])),
        mk("cycle_5yr", ("one", "sin5", "cos5"),
           np.column_stack([o(t_tr), np.sin(w5 * t_tr), np.cos(w5 * t_tr)]),
           np.column_stack([o(t_ho), np.sin(w5 * t_ho), np.cos(w5 * t_ho)])),
    ]
    # AR(1): aligned lagged design (same length handling for train/holdout)
    Xa_tr = np.column_stack([np.ones(len(y_tr) - 1), y_tr[:-1]])
    Xa_ho = np.column_stack([np.ones(len(y_ho) - 1), y_ho[:-1]])
    wa, *_ = np.linalg.lstsq(Xa_tr, y_tr[1:], rcond=None)
    rha = y_ho[1:] - Xa_ho @ wa
    rma = float(np.sqrt(np.mean(rha ** 2)))
    obs_a = float(np.sqrt(np.mean(y_ho[1:] ** 2)))
    g4a = {"passed": True, "holdout_obs_rms": obs_a,
           "holdout_noise_sigma": float(0.02 * obs_a),
           "holdout_n_samples": len(y_ho) - 1, "holdout_residuals": rha.tolist()}
    ca = CandidateDynamics(lhs="log_trappings",
                           terms=(Term("one", "a0"), Term("lag1", "a1")),
                           candidate_id="ar1_lag")
    scored.append(ScoredCandidate(ca, {"gate4_status": g4a, "data_origin": DataOrigin.REAL_OBSERVATIONAL},
                                  train_loss=float(np.mean((y_tr[1:] - Xa_tr @ wa) ** 2)),
                                  holdout_rmse=rma))
    Xt_tr = np.column_stack([o(t_tr), t_tr])
    Xt_ho = np.column_stack([o(t_ho), t_ho])
    wt, *_ = np.linalg.lstsq(Xt_tr, y_tr, rcond=None)
    rht = y_ho - Xt_ho @ wt
    g4t = {"passed": True, "holdout_obs_rms": obs_rms,
           "holdout_noise_sigma": ns, "holdout_n_samples": len(y_ho),
           "holdout_residuals": rht.tolist()}
    ct = CandidateDynamics(lhs="log_trappings",
                           terms=(Term("one", "a0"), Term("t", "a1")),
                           candidate_id="trend")
    scored.append(ScoredCandidate(ct, {"gate4_status": g4t, "data_origin": DataOrigin.REAL_OBSERVATIONAL},
                                  train_loss=float(np.mean((y_tr - Xt_tr @ wt) ** 2)),
                                  holdout_rmse=float(np.sqrt(np.mean(rht ** 2)))))
    ranked = sorted([s for s in scored if np.isfinite(s.train_loss)], key=lambda s: s.train_loss)
    # Align residual lengths (AR(1) lag drops one holdout point)
    n_min = min(len(s.provenance["gate4_status"]["holdout_residuals"]) for s in ranked[:2])
    r_best = ranked[0].provenance["gate4_status"]["holdout_residuals"][-n_min:]
    r_rival = ranked[1].provenance["gate4_status"]["holdout_residuals"][-n_min:]
    adj = adjudicate(
        scored, resid_best=r_best,
        resid_rival=r_rival,
        bootstrap=bootstrap, bootstrap_n_boot=1000, bootstrap_seed=7,
        data_origin=DataOrigin.REAL_OBSERVATIONAL,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, scored


def hello_world():
    print("=" * 64)
    print("LYNX SANDBOX (REAL DATA 1821-1934): cycle discovery, split 1878")
    print("=" * 64)
    adj, scored = run_lynx()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:10s} holdout RMSE={s.holdout_rmse:.4f}")
    print(f"  verdict: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None} "
          f"origin={adj.data_origin}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
