"""SIR epidemiology sandbox: R0 threshold + vaccination intervention.

Truth (hidden): SIR with beta=0.3, gamma=0.12 (R0 = 2.5). At t_vax,
a vaccination campaign cuts beta by 70% (beta -> 0.09, R0 -> 0.75).
Train = pre-intervention outbreak; holdout = post-intervention decay.
The intervention analog is real: a policy change between splits.

Candidates (new-infection incidence forms): mass-action beta*S*I,
linear-in-I, exponential-in-t, constant. Question A: which form wins
pre/post break? Question B (the threshold one): per-window R0 estimates
above/below 1 vs observed growth/decline -- does the data support the
threshold up to adjudication?
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

BETA, GAMMA = 0.3, 0.12
VAX_CUT = 0.7
T_VAX = 60.0


def simulate_sir(beta=BETA, gamma=GAMMA, t_vax=T_VAX, vax_cut=VAX_CUT,
                 t_end=160.0, dt=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(t_end / dt) + 1
    S = np.empty(n)
    Inf = np.empty(n)
    S[0], Inf[0] = 0.99, 0.01
    for t in range(1, n):
        b = beta * (1 - vax_cut) if t * dt >= t_vax else beta
        dS = -b * S[t - 1] * Inf[t - 1] * dt
        dI = (b * S[t - 1] * Inf[t - 1] - gamma * Inf[t - 1]) * dt
        S[t] = S[t - 1] + dS
        Inf[t] = max(0.0, Inf[t - 1] + dI)
    inc = np.maximum(0.0, -np.diff(S, prepend=S[0]))
    if noise > 0:
        inc = np.maximum(0.0, inc + rng.normal(0, noise, n))
    return np.arange(n) * dt, S, Inf, inc


def _mk(cid, terms, X_tr, y_tr, X_ho, y_ho, obs_rms, ns):
    w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    rh = y_ho - X_ho @ w
    rm = float(np.sqrt(np.mean(rh ** 2)))
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": ns, "holdout_n_samples": len(y_ho),
          "holdout_residuals": rh.tolist()}
    c = CandidateDynamics(lhs="incidence",
                          terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                          candidate_id=cid)
    return ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                           train_loss=float(np.mean((y_tr - X_tr @ w) ** 2)),
                           holdout_rmse=rm)


def run_sir_forms(noise=0.02, seed=42):
    """A: incidence-form recovery across the vaccination break."""
    t, S, Inf, inc = simulate_sir(noise=noise, seed=seed)
    ntr = int(T_VAX)  # train strictly pre-intervention
    Str, Itr, ytr = S[:ntr], Inf[:ntr], inc[:ntr]
    Sho, Iho, yho = S[ntr:], Inf[ntr:], inc[ntr:]
    obs_rms = float(np.sqrt(np.mean(yho ** 2)))
    ns = float(noise * obs_rms + 1e-9)
    o = np.ones_like
    defs = [
        ("mass_action", ("one", "SI"), Str * Itr, Sho * Iho),
        ("linear_I", ("one", "I"), Itr, Iho),
        ("exponential_t", ("one", "t"), np.arange(ntr), np.arange(ntr, len(t))),
        ("constant", ("one",), None, None),
    ]
    scored = []
    for cid, terms, ftr, fho in defs:
        X_tr = np.column_stack([o(Str), ftr]) if ftr is not None else o(Str)[:, None]
        X_ho = np.column_stack([o(Sho), fho]) if fho is not None else o(Sho)[:, None]
        scored.append(_mk(cid, terms, X_tr, ytr, X_ho, yho, obs_rms, ns))
    ranked = sorted([s for s in scored if np.isfinite(s.train_loss)], key=lambda s: s.train_loss)
    adj = adjudicate(
        scored, resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
        resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7,
        data_origin=DataOrigin.SIMULATION,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, scored


def run_r0_threshold(noise=0.02, seed=42):
    """B: windowed R0 estimates vs observed growth sign -- threshold support."""
    t, S, Inf, _inc = simulate_sir(noise=noise, seed=seed)
    wlen = 14
    ests, grows = [], []
    for s in range(0, len(t) - wlen, 7):
        seg = Inf[s:s + wlen]
        if seg[0] <= 1e-9:
            continue
        g = float(np.polyfit(np.arange(wlen), np.log(np.maximum(seg, 1e-9)), 1)[0])
        Sbar = float(S[s:s + wlen].mean())
        R0 = (g / GAMMA + 1) / max(Sbar, 1e-9)
        ests.append(R0)
        grows.append(g)
    ests, grows = np.array(ests), np.array(grows)
    agree = float(np.mean((ests > 1) == (grows > 0)))
    return {"n_windows": len(ests), "agreement": agree,
            "est_range": (float(ests.min()), float(ests.max()))}


def hello_world():
    print("=" * 64)
    print("SIR SANDBOX: R0=2.5 outbreak, vaccination cuts beta 70% at t=60")
    print("=" * 64)
    adj, scored = run_sir_forms()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:14s} holdout RMSE={s.holdout_rmse:.6f}")
    print(f"  A forms: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    r = run_r0_threshold()
    print(f"  B threshold: {r['n_windows']} windows, "
          f"sign(R0-1)==sign(growth): {r['agreement']:.0%}, R0 range {r['est_range'][0]:.2f}..{r['est_range'][1]:.2f}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
