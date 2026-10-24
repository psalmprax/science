"""Game-theory sandbox: which game was played, recovered from adaptation paths.

Hidden truth: 2-strategy replicator dynamics dx/dt = x(1-x)(f_C(x) - f_D(x))
with payoff matrices the candidates never see:
  PD:        R=3 S=0 T=5 P=1  (defection dominates; x -> 0)
  Stag Hunt: R=4 T=3 P=2 S=0  (bistable; separatrix inside (0,1))

Candidates are polynomial forms in x (replicator dynamics are cubic):
constant, linear, quadratic, cubic. Train trajectories start from one set
of initial cooperator fractions; holdout trajectories start from unseen
fractions -- the intervention analog (new initial populations).

Two questions:
  A. Form recovery: does the cubic form win on PD data?
  B. Game identification: PD-fitted cubic vs SH-fitted cubic, both scored
     on PD holdout from unseen ICs -- which game generated the path?
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

PD = {"R": 3.0, "S": 0.0, "T": 5.0, "P": 1.0}
SH = {"R": 4.0, "S": 0.0, "T": 3.0, "P": 2.0}


def replicator_rhs(x, game):
    """dx/dt for cooperator fraction x under game payoffs."""
    fC = game["R"] * x + game["S"] * (1 - x)
    fD = game["T"] * x + game["P"] * (1 - x)
    return x * (1 - x) * (fC - fD)


def simulate_game(game, x0, t_end=20.0, dt=0.1, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(t_end / dt) + 1
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = float(np.clip(x[t - 1] + replicator_rhs(x[t - 1], game) * dt, 0.0, 1.0))
    if noise > 0:
        x = np.clip(x + rng.normal(0, noise, n), 0.0, 1.0)
    return np.arange(n) * dt, x


def _mk(cid, terms, X_tr, y_tr, X_ho, y_ho, obs_rms, ns):
    w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    rh = y_ho - X_ho @ w
    rm = float(np.sqrt(np.mean(rh ** 2)))
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": ns, "holdout_n_samples": len(y_ho),
          "holdout_residuals": rh.tolist()}
    c = CandidateDynamics(lhs="coop_fraction",
                          terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                          candidate_id=cid)
    return ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                           train_loss=float(np.mean((y_tr - X_tr @ w) ** 2)),
                           holdout_rmse=rm), rh


def _poly_cols(x, degree):
    return np.column_stack([x ** k for k in range(degree + 1)])


def run_form_recovery(game=PD, game_name="PD", noise=0.005, seed=42):
    """A: cubic form vs lower orders, unseen initial conditions at holdout."""
    tr_ics = (0.2, 0.4, 0.6, 0.8)
    ho_ics = (0.1, 0.3, 0.5, 0.7, 0.9)
    # Regress x_next on polynomial of x (discrete-time map form), pooled
    # over trajectories; holdout uses unseen initial fractions.
    Xa_tr = np.concatenate([simulate_game(game, x0, noise=noise, seed=seed)[1][:-1] for x0 in tr_ics])
    Ya_tr = np.concatenate([simulate_game(game, x0, noise=noise, seed=seed)[1][1:] for x0 in tr_ics])
    Xa_ho = np.concatenate([simulate_game(game, x0, noise=noise, seed=seed + 1)[1][:-1] for x0 in ho_ics])
    Ya_ho = np.concatenate([simulate_game(game, x0, noise=noise, seed=seed + 1)[1][1:] for x0 in ho_ics])
    obs_rms = float(np.sqrt(np.mean(Ya_ho ** 2)))
    ns = float(noise * obs_rms + 1e-6)
    defs = [("constant", ("one",), 0), ("linear", ("one", "x"), 1),
            ("quadratic", ("one", "x", "x2"), 2), ("cubic", ("one", "x", "x2", "x3"), 3)]
    scored = []
    for cid, terms, deg in defs:
        sc, _ = _mk(cid, terms, _poly_cols(Xa_tr, deg), Ya_tr,
                    _poly_cols(Xa_ho, deg), Ya_ho, obs_rms, ns)
        scored.append(sc)
    ranked = sorted([s for s in scored if np.isfinite(s.train_loss)], key=lambda s: s.train_loss)
    adj = adjudicate(scored,
                     resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
                     resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
                     bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7,
                     data_origin=DataOrigin.SIMULATION,
                     benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, scored


def run_game_identification(noise=0.005, seed=42):
    """B: PD-fitted cubic vs SH-fitted cubic, scored on PD holdout (unseen ICs)."""
    ho_ics = (0.15, 0.35, 0.55, 0.75, 0.95)
    tr_ics = (0.2, 0.4, 0.6, 0.8)
    Xa_pd = np.concatenate([simulate_game(PD, x0, noise=noise, seed=seed)[1][:-1] for x0 in tr_ics])
    Ya_pd = np.concatenate([simulate_game(PD, x0, noise=noise, seed=seed)[1][1:] for x0 in tr_ics])
    Xa_sh = np.concatenate([simulate_game(SH, x0, noise=noise, seed=seed)[1][:-1] for x0 in tr_ics])
    Ya_sh = np.concatenate([simulate_game(SH, x0, noise=noise, seed=seed)[1][1:] for x0 in tr_ics])
    Xa_ho = np.concatenate([simulate_game(PD, x0, noise=noise, seed=seed + 1)[1][:-1] for x0 in ho_ics])
    Ya_ho = np.concatenate([simulate_game(PD, x0, noise=noise, seed=seed + 1)[1][1:] for x0 in ho_ics])
    obs_rms = float(np.sqrt(np.mean(Ya_ho ** 2)))
    ns = float(noise * obs_rms + 1e-6)
    w_pd, *_ = np.linalg.lstsq(_poly_cols(Xa_pd, 3), Ya_pd, rcond=None)
    w_sh, *_ = np.linalg.lstsq(_poly_cols(Xa_sh, 3), Ya_sh, rcond=None)
    out = []
    for cid, w, Xa_f, Ya_f in (("cubic_from_PD", w_pd, Xa_pd, Ya_pd),
                               ("cubic_from_SH", w_sh, Xa_sh, Ya_sh)):
        pred = _poly_cols(Xa_ho, 3) @ w
        rh = Ya_ho - pred
        rm = float(np.sqrt(np.mean(rh ** 2)))
        g4 = {"passed": True, "holdout_obs_rms": obs_rms,
              "holdout_noise_sigma": ns, "holdout_n_samples": len(Ya_ho),
              "holdout_residuals": rh.tolist()}
        c = CandidateDynamics(lhs="coop_fraction",
                              terms=(Term("one", "a0"), Term("x", "a1"),
                                     Term("x2", "a2"), Term("x3", "a3")),
                              candidate_id=cid)
        out.append(ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                                   train_loss=float(np.mean((Ya_f - _poly_cols(Xa_f, 3) @ w) ** 2)),
                                   holdout_rmse=rm))
    adj = adjudicate(out,
                     resid_best=out[0].provenance["gate4_status"]["holdout_residuals"],
                     resid_rival=out[1].provenance["gate4_status"]["holdout_residuals"],
                     bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7,
                     data_origin=DataOrigin.SIMULATION,
                     benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, out


def hello_world():
    print("=" * 64)
    print("GAME THEORY SANDBOX: which game was played?")
    print("=" * 64)
    adj, scored = run_form_recovery()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:10s} holdout RMSE={s.holdout_rmse:.6f}")
    print(f"  A form recovery: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    adj2, out = run_game_identification()
    for s in sorted(out, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:14s} holdout RMSE={s.holdout_rmse:.6f}")
    print(f"  B game identification: {adj2.status.value} "
          f"promoted={adj2.promoted.candidate.candidate_id if adj2.promoted else None}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
