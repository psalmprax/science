"""Finance L3 with reachability constraints (redo of a failed execution).

A first attempt probed the unconstrained argmax (r_prev=-0.06, vol=0.005),
where the true law compounds x3/step -- a dynamically invalid regime -- and
returned a formally-valid but meaningless UNIQUE on astronomical RMSEs.
That failure is recorded here as the reason for the constraint below.

Constraint: |signal| = |0.015 * r_prev / vol| <= 0.05 (no explosive
compounding; reachable market dynamics only). The probe maximizes
|AR(1) - constant| divergence INSIDE the reachable set. Fresh synthetic
observations come from the true vol-scaled-momentum law at probe
conditions (new seed); Stage-1 fits stay frozen (transfer pattern).
Tiebreak contracts squared-error series, never signed residuals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import (
    CandidateDynamics,
    ScoredCandidate,
    Term,
)
from apedp_dyn.sandbox.finance_l2_demo import (
    generate_market_regime_data,
)

MAX_SIGNAL = 0.05  # reachability: per-step drift bounded (no explosive corners)


def _mk(cid, terms, rmse, resid, obs_rms, ns):
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": float(np.std(resid)),
          "holdout_n_samples": len(resid),
          "holdout_residuals": [float(v) for v in resid]}
    c = CandidateDynamics(lhs="r_t",
                          terms=tuple(Term(t, f"a{i}") for i, t in enumerate(terms)),
                          candidate_id=cid)
    return ScoredCandidate(candidate=c, provenance={"gate4_status": g4},
                           train_loss=rmse, holdout_rmse=rmse)


def run_finance_l3():
    _, returns = generate_market_regime_data(n_points=150,
                                             regime="volatility_scaled_momentum",
                                             seed=42)
    n = len(returns)
    ntr = n // 2
    y_tr, y_ho = returns[:ntr], returns[ntr:]
    x_tr, x_ho = returns[:ntr - 1], returns[ntr - 1:-1]
    w_ar1, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x_tr), x_tr]),
                                y_tr[1:], rcond=None)
    w_c = float(y_tr.mean())
    obs0 = float(np.sqrt(np.mean(y_ho ** 2)))
    pa0 = w_ar1[0] + w_ar1[1] * x_ho
    ra0 = float(np.sqrt(np.mean((y_ho - pa0) ** 2)))
    rc0 = float(np.sqrt(np.mean((y_ho - w_c) ** 2)))
    adj0 = adjudicate(
        [_mk("ar1", ("one", "r_prev"), ra0, y_ho - pa0, obs0, 0.015),
         _mk("const", ("one",), rc0, y_ho - w_c, obs0, 0.015)],
        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7)
    print(f"Stage 1: {adj0.status.value}  ar1={ra0:.4f} const={rc0:.4f}")

    # Reachable probe: maximize divergence subject to |signal| <= MAX_SIGNAL
    a0, a1 = w_ar1
    rp = np.linspace(-0.06, 0.06, 241)
    vl = np.linspace(0.005, 0.05, 181)
    R, V = np.meshgrid(rp, vl)
    reachable = np.abs(0.015 * R / np.maximum(V, 1e-9)) <= MAX_SIGNAL
    D = np.where(reachable, np.abs(a0 + a1 * R - w_c), -np.inf)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    rp_star, vl_star = float(R[i, j]), float(V[i, j])
    print(f"L3 reachable probe: r_prev*={rp_star:+.4f} vol*={vl_star:.4f} "
          f"div={D.max():.4f} signal={0.015 * rp_star / vl_star:+.4f}")

    # Execute inside the reachable set: fresh synthetic near probe, new seed
    rng = np.random.default_rng(777)
    r = np.zeros(60)
    r[0] = rp_star
    for t in range(1, 60):
        sig = vl_star
        signal = 0.015 * (r[t - 1] / max(sig, 1e-6))
        r[t] = signal + float(rng.normal(0.0, sig))
    print(f"  max|signal| during execution: "
          f"{float(np.max(np.abs(0.015 * r[:-1] / vl_star))):.4f} (bound {MAX_SIGNAL})")
    y, x = r[1:], r[:-1]
    pa = w_ar1[0] + w_ar1[1] * x
    pc = np.full_like(y, w_c)
    ra = float(np.sqrt(np.mean((y - pa) ** 2)))
    rc = float(np.sqrt(np.mean((y - pc) ** 2)))
    obs = float(np.sqrt(np.mean(y ** 2)))
    se_a = ((y - pa) ** 2).tolist()
    se_c = ((y - pc) ** 2).tolist()
    adj1 = adjudicate(
        [_mk("ar1", ("one", "r_prev"), ra, y - pa, obs, vl_star),
         _mk("const", ("one",), rc, y - pc, obs, vl_star)],
        resid_best=se_a, resid_rival=se_c,
        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7)
    print(f"Stage 3: {adj1.status.value}  ar1={ra:.4f} const={rc:.4f} "
          f"promoted={adj1.promoted.candidate.candidate_id if adj1.promoted else None}")
    print("reason:", adj1.reason[:140])
    return adj0, adj1


if __name__ == "__main__":
    run_finance_l3()
