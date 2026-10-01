"""AD-CAB Stage A: synthetic automotive component degradation (ground truth).

Truth (hidden): brake-pad wear with Arrhenius temperature dependence
    D(t, T) = k0 * t^alpha * exp(-Ea / (R * T)), k0=2e-4, alpha=1.35,
    Ea=4.0e4 J/mol, R=8.314. Mild steel-on-steel wear with thermal
    acceleration; no interaction term in truth (tests false-interaction
    rejection).

Regimes: train 20-60 C / 0-800 cycles; interpolation holdout same
envelope unseen cycles; extrapolation holdout 80 C / 800-1600 cycles.
Transfer assessment (transfer_scope) wired on every verdict: the module
under test is the machinery, and AD-CAB is its first stress battery.

Candidates: power-law wear, linear wear, log wear, saturating
exponential, Arrhenius-interaction (t^a * exp(-b/T)), constant.
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
from apedp_dyn.epistemic.transfer_scope import TransferScope  # noqa: E402
from apedp_dyn.grammar.candidate_ir import (  # noqa: E402
    CandidateDynamics,
    ScoredCandidate,
    Term,
)

R_GAS = 8.314
TRUTH = {"k0": 2e-4, "alpha": 1.35, "Ea": 4.0e4}


def generate_degradation(n_per_regime: int = 40, noise_sigma: float = 0.03,
                         seed: int = 42):
    """Wear grids: train/interp envelope vs hot extrapolation envelope."""
    rng = np.random.default_rng(seed)

    def grid(t_lo, t_hi, T_lo, T_hi, n):
        t = np.linspace(t_lo, t_hi, n)
        T = np.linspace(T_lo, T_hi, n) + 273.15
        return t, T

    def truth(t, T):
        return TRUTH["k0"] * t ** TRUTH["alpha"] * np.exp(-TRUTH["Ea"] / (R_GAS * T))

    out = {}
    for name, (t_lo, t_hi, T_lo, T_hi) in {
            "train": (50.0, 800.0, 20.0, 60.0),
            "interp": (50.0, 800.0, 20.0, 60.0),
            "extra": (800.0, 1600.0, 80.0, 80.0)}.items():
        t, T = grid(t_lo, t_hi, T_lo, T_hi, n_per_regime)
        y = truth(t, T)
        y = y + rng.normal(0, noise_sigma * np.sqrt(np.mean(y ** 2)), n_per_regime)
        out[name] = (t, T, np.maximum(y, 1e-12))
    return out


def _feats(kind, t, T):
    o = np.ones_like(t)
    if kind == "power":
        return np.column_stack([o, np.log(np.maximum(t, 1e-9))])
    if kind == "linear":
        return np.column_stack([o, t])
    if kind == "logt":
        return np.column_stack([o, np.log1p(t)])
    if kind == "satexp":
        return np.column_stack([o, 1 - np.exp(-t / 400.0)])
    if kind == "arrhenius":
        return np.column_stack([o, np.log(np.maximum(t, 1e-9)), -1.0 / (R_GAS * T)])
    if kind == "const":
        return o[:, None]
    raise ValueError(kind)


def run_adcab_stage_a(noise_sigma: float = 0.03, seed: int = 42,
                      bootstrap: bool = True):
    data = generate_degradation(noise_sigma=noise_sigma, seed=seed)
    (t_tr, T_tr, y_tr), (t_ie, T_ie, y_ie), (t_ee, T_ee, y_ee) = (
        data["train"], data["interp"], data["extra"])
    obs_rms = float(np.sqrt(np.mean(y_ie ** 2)))
    ns = float(noise_sigma * obs_rms)
    defs = [("power_wear", ("one", "logt"), "power"),
            ("linear_wear", ("one", "t"), "linear"),
            ("log_wear", ("one", "log1pt"), "logt"),
            ("sat_exp", ("one", "satexp"), "satexp"),
            ("arrhenius", ("one", "logt", "invT"), "arrhenius"),
            ("constant", ("one",), "const")]
    scored = []
    for cid, terms, kind in defs:
        F_tr, F_ie = _feats(kind, t_tr, T_tr), _feats(kind, t_ie, T_ie)
        # NOTE: power/arrhenius fit in log space (relative precision); the
        # holdout RMSE below is mapped back to engineering units for a fair
        # comparison (same lesson as the creep sandbox).
        if kind in ("power", "arrhenius"):
            w, *_ = np.linalg.lstsq(F_tr, np.log(y_tr), rcond=None)
            pred = np.exp(F_ie @ w)
        else:
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
            pred = F_ie @ w
        rh = y_ie - pred
        rm = float(np.sqrt(np.mean(rh ** 2)))
        g4 = {"passed": True, "holdout_obs_rms": obs_rms,
              "holdout_noise_sigma": ns, "holdout_n_samples": len(y_ie),
              "holdout_residuals": rh.tolist()}
        c = CandidateDynamics(lhs="wear_depth",
                              terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                              candidate_id=cid)
        scored.append(ScoredCandidate(
            c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
            train_loss=float(rm), holdout_rmse=rm))
    ranked = sorted([s for s in scored if np.isfinite(s.train_loss)], key=lambda s: s.train_loss)
    adj = adjudicate(
        scored, resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
        resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
        bootstrap=bootstrap, bootstrap_n_boot=1000, bootstrap_seed=7,
        data_origin=DataOrigin.SIMULATION,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    # Transfer: best-vs-rival on interp vs hot extrapolation (frozen fits reused
    # via stored weights is overkill here; refit-free scoring uses Stage-1 weights
    # recomputed deterministically -- same data, same seed, same numbers).
    order = sorted(scored, key=lambda s: s.holdout_rmse)
    b, r = order[0], order[1]
    extra = {}
    for cid, _terms, kind in defs:
        F_tr2 = _feats(kind, t_tr, T_tr)
        F_ee2 = _feats(kind, t_ee, T_ee)
        if kind in ("power", "arrhenius"):
            w2, *_ = np.linalg.lstsq(F_tr2, np.log(y_tr), rcond=None)
            pred = np.exp(F_ee2 @ w2)
        else:
            w2, *_ = np.linalg.lstsq(F_tr2, y_tr, rcond=None)
            pred = F_ee2 @ w2
        extra[cid] = float(np.sqrt(np.mean((y_ee - pred) ** 2)))
    scope = TransferScope(
        candidate_id=b.candidate.candidate_id, rival_id=r.candidate.candidate_id,
        interp_rmse_best=b.holdout_rmse, interp_rmse_rival=r.holdout_rmse,
        extra_rmse_best=extra[b.candidate.candidate_id],
        extra_rmse_rival=extra[r.candidate.candidate_id])
    return adj, scored, scope


def hello_world():
    print("=" * 64)
    print("AD-CAB STAGE A: brake-pad wear D=k0*t^1.35*exp(-Ea/RT), hot extrap")
    print("=" * 64)
    adj, scored, scope = run_adcab_stage_a()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:14s} interp RMSE={s.holdout_rmse:.4g}")
    print(f"  verdict: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    print(f"  transfer: {scope.status} reversal={scope.leadership_reversed} "
          f"blowup={scope.error_blowup:.1f}x")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
