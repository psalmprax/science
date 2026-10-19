"""Kuramoto sandbox: synchronization transition law discovery.

Truth (hidden): N phase oscillators dti/dt = wi + (K/N)*sum_j sin(tj-ti),
Gaussian natural frequencies. Order parameter r(K) undergoes a
second-order transition at Kc: r ~ sqrt(1 - Kc/K) above threshold.

Candidates for r_inf(K) (OLS-friendly, Kc NOT supplied):
  sqrt_K:    r = a*sqrt(K) + b
  linear_K:  r = a*K + b
  inv_K:     r = a + b/K
  constant:  r = b
Train = weak-to-moderate coupling; holdout = strong coupling
(intervention analog: coupling cranked up). None matches the true
sqrt(1-Kc/K) form exactly -- the question is which shape transfers and
whether the verdict overclaims.
"""
from __future__ import annotations

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


def order_param(K, N=40, T=40.0, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, N)
    th = rng.uniform(-np.pi, np.pi, N)
    n = int(T / dt)
    for _ in range(n):
        th = th + (w + (K / N) * np.sum(np.sin(th[None, :] - th[:, None]), axis=1)) * dt
    r = float(np.abs(np.exp(1j * th).mean()))
    return r


def run_kuramoto(K_tr=None, K_ho=None, N=40, seed=42):
    K_tr = np.linspace(0.5, 2.5, 9) if K_tr is None else np.asarray(K_tr)
    K_ho = np.linspace(3.0, 5.0, 6) if K_ho is None else np.asarray(K_ho)
    r_tr = np.array([order_param(K, N=N, seed=seed) for K in K_tr])
    r_ho = np.array([order_param(K, N=N, seed=seed + 1) for K in K_ho])
    obs_rms = float(np.sqrt(np.mean(r_ho ** 2)))
    ns = float(0.02 * obs_rms + 1e-9)
    o = np.ones_like

    def mk(cid, terms, ftr, fho):
        X_tr = np.column_stack([o(K_tr), ftr])
        X_ho = np.column_stack([o(K_ho), fho])
        w, *_ = np.linalg.lstsq(X_tr, r_tr, rcond=None)
        rh = r_ho - X_ho @ w
        rm = float(np.sqrt(np.mean(rh ** 2)))
        g4 = {"passed": True, "holdout_obs_rms": obs_rms,
              "holdout_noise_sigma": ns, "holdout_n_samples": len(K_ho),
              "holdout_residuals": rh.tolist()}
        c = CandidateDynamics(lhs="order_param",
                              terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                              candidate_id=cid)
        return ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                               train_loss=float(np.mean((r_tr - X_tr @ w) ** 2)),
                               holdout_rmse=rm)

    scored = [
        mk("sqrt_K", ("one", "sqrtK"), np.sqrt(K_tr), np.sqrt(K_ho)),
        mk("linear_K", ("one", "K"), K_tr, K_ho),
        mk("inv_K", ("one", "invK"), 1 / K_tr, 1 / K_ho),
    ]
    Xc_tr = o(K_tr)[:, None]
    Xc_ho = o(K_ho)[:, None]
    wc, *_ = np.linalg.lstsq(Xc_tr, r_tr, rcond=None)
    rhc = r_ho - Xc_ho @ wc
    g4c = {"passed": True, "holdout_obs_rms": obs_rms,
           "holdout_noise_sigma": ns, "holdout_n_samples": len(K_ho),
           "holdout_residuals": rhc.tolist()}
    cc = CandidateDynamics(lhs="order_param", terms=(Term("one", "b"),),
                           candidate_id="constant")
    scored.append(ScoredCandidate(cc, {"gate4_status": g4c, "data_origin": DataOrigin.SIMULATION},
                                  train_loss=float(np.mean((r_tr - Xc_tr @ wc) ** 2)),
                                  holdout_rmse=float(np.sqrt(np.mean(rhc ** 2)))))
    ranked = sorted([s for s in scored if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)], key=lambda s: s.holdout_rmse)
    adj = adjudicate(
        scored, resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
        resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7,
        data_origin=DataOrigin.SIMULATION,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, scored, (K_tr, r_tr, K_ho, r_ho)


def hello_world():
    print("=" * 64)
    print("KURAMOTO SANDBOX: sync transition r(K), train K<=2.5, holdout K>=3")
    print("=" * 64)
    adj, scored, _ = run_kuramoto()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:10s} holdout RMSE={s.holdout_rmse:.4f}")
    print(f"  verdict: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
