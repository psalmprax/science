"""Creep sandbox: Norton power-law discovery (steady-state creep rate).

Truth (hidden): eps_dot = A * sigma^n, A=1e-9, n=4.5 (dislocation-climb
regime power law). Candidates: power law (log-log linear), linear in
stress, exponential in stress, constant. Train = low-to-mid stress;
holdout = high stress (extrapolation = intervention analog: does the
power law transfer beyond the fitted regime?).

Follows M4/L1 conventions: OLS, gate4 provenance, bootstrap, native
data_origin labeling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import (  # noqa: E402
    BenchmarkLevel,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import (  # noqa: E402
    CandidateDynamics,
    ScoredCandidate,
    Term,
)

CREEP_A = 1e-9
CREEP_N = 4.5


def generate_creep_data(n_points: int = 40, noise_sigma: float = 0.03,
                        s_min: float = 50.0, s_max: float = 300.0,
                        seed: int = 42):
    """Steady-state creep rates over engineering stress range (MPa)."""
    rng = np.random.default_rng(seed)
    sigma = np.logspace(np.log10(s_min), np.log10(s_max), n_points)
    truth = CREEP_A * sigma ** CREEP_N
    # Relative (lognormal-ish) noise: absolute Gaussian noise on log-spanned
    # data destroys small values (clipped at 1e-15 -> log outliers -> slope
    # explosion to n_hat~20). Creep measurements are relative precision.
    obs = truth * (1.0 + rng.normal(0, noise_sigma, n_points))
    obs = np.maximum(obs, 1e-15)
    return sigma, np.maximum(obs, 1e-15)


def run_creep(noise_sigma: float = 0.03, n_train: int = 24,
              bootstrap: bool = True, seed: int = 42):
    sigma, obs = generate_creep_data(noise_sigma=noise_sigma, seed=seed)
    s_tr, y_tr, s_ho, y_ho = sigma[:n_train], obs[:n_train], sigma[n_train:], obs[n_train:]
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))
    ns = float(noise_sigma * obs_rms + 1e-18)
    o = np.ones_like

    def mk(cid, terms, X_tr, X_ho):
        w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        rh = y_ho - X_ho @ w
        rm = float(np.sqrt(np.mean(rh ** 2)))
        g4 = {"passed": True, "holdout_obs_rms": obs_rms,
              "holdout_noise_sigma": ns, "holdout_n_samples": len(y_ho),
              "holdout_residuals": rh.tolist()}
        c = CandidateDynamics(lhs="creep_rate",
                              terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                              candidate_id=cid)
        return ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                               train_loss=float(np.mean((y_tr - X_tr @ w) ** 2)),
                               holdout_rmse=rm)

    ltr, lho = np.log(s_tr), np.log(s_ho)
    scored = [
        # power law via log-log (recovers n as slope)
        mk("power_law", ("one", "logS"),
           np.column_stack([o(ltr), ltr]), np.column_stack([o(lho), lho])),
        mk("linear_stress", ("one", "S"),
           np.column_stack([o(s_tr), s_tr]), np.column_stack([o(s_ho), s_ho])),
        mk("exp_stress", ("one", "expS"),
           np.column_stack([o(s_tr), np.exp(s_tr / 300.0)]),
           np.column_stack([o(s_ho), np.exp(s_ho / 300.0)])),
        mk("constant", ("one",), o(s_tr)[:, None], o(s_ho)[:, None]),
    ]
    # NOTE: power_law fits log(rate); its RMSE is on log scale -- rescale to
    # rate space for a fair holdout comparison below.
    pw = scored[0]
    Xl_tr = np.column_stack([o(ltr), ltr])
    Xl_ho = np.column_stack([o(lho), lho])
    wl, *_ = np.linalg.lstsq(Xl_tr, np.log(np.maximum(y_tr, 1e-15)), rcond=None)
    pred_rate = np.exp(Xl_ho @ wl)
    rh_rate = y_ho - pred_rate
    rm_rate = float(np.sqrt(np.mean(rh_rate ** 2)))
    g4 = dict(pw.provenance["gate4_status"])
    g4["holdout_residuals"] = rh_rate.tolist()
    c = CandidateDynamics(lhs="creep_rate",
                          terms=(Term("one", "a0"), Term("logS", "n")),
                          candidate_id="power_law")
    scored[0] = ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                                train_loss=pw.train_loss, holdout_rmse=rm_rate)
    ranked = sorted([s for s in scored if np.isfinite(s.train_loss)], key=lambda s: s.train_loss)
    adj = adjudicate(
        scored, resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
        resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
        bootstrap=bootstrap, bootstrap_n_boot=1000, bootstrap_seed=7,
        data_origin=DataOrigin.SIMULATION,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    # recovered exponent for the record
    return adj, scored, float(wl[1])


def hello_world():
    print("=" * 64)
    print("CREEP SANDBOX: Norton law eps=A*s^n (A=1e-9, n=4.5), extrapolate high-S")
    print("=" * 64)
    adj, scored, n_hat = run_creep()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:14s} holdout RMSE={s.holdout_rmse:.4g}")
    print(f"  recovered exponent n_hat={n_hat:.2f} (truth 4.5)")
    print(f"  verdict: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
