"""QEC threshold-law sandbox: surface-code logical error scaling.

Truth (hidden): p_L(p) = C * (p / p_th)^((d+1)/2), d=3, C=0.1, p_th=0.01.
Physical error rates span the threshold (p in [5e-4, 2e-2]) so the data
covers both suppression (below p_th) and blowup (above) regimes.

Includes:
  Stage 1: four fixed forms (threshold-ratio, plain quadratic, linear,
      constant) -> expected MULTI between the algebraically identical
      ratio and quadratic forms.
  L3: probe between threshold-power and linear over reachable p,
      fresh true-law observations at probe, frozen-fit re-adjudication.
  Scalability: exponent recovery per distance via the regularized
      saturating form p_L,soft (linear-space weighted fit, multi-seed).
      The naive log-log slope is documented as biased and NOT used.

Conventions follow the M1/M2/M3 L1 sandboxes: OLS fits, gate4 provenance
with residuals, bootstrap tiebreak, native data_origin labeling.
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

QEC_C = 0.1
QEC_P_TH = 0.01
QEC_D = 3
QEC_EXP = (QEC_D + 1) / 2


def generate_qec_threshold_data(
    n_points: int = 60,
    noise_sigma: float = 0.02,
    seed: int = 42,
    distance: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic logical error rates spanning the threshold.

    Parameters
    ----------
    n_points:
        Number of log-spaced points across the threshold window.
    noise_sigma:
        Observation noise relative to the RMS signal amplitude.
    seed:
        PRNG seed (uses numpy.random.default_rng).
    distance:
        Code distance d (default 3 -> exponent e = (d+1)/2 = 2.0).
        For d=5 -> exponent e = 3.0; for d=7 -> exponent e = 4.0.
    """
    rng = np.random.default_rng(seed)
    p = np.logspace(-3.3, -1.7, n_points)
    exp = (distance + 1) / 2.0
    truth = QEC_C * (p / QEC_P_TH) ** exp
    obs = truth + rng.normal(0, noise_sigma * np.sqrt(np.mean(truth ** 2)), n_points)
    return p, obs


def generate_qec_saturating_data(
    n_points: int = 80,
    noise_sigma: float = 0.02,
    seed: int = 42,
    distance: int = 3,
    model: str = "pade",
    p_sat: float = 0.5,
    p_min: float = 0.001,
    p_max: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic saturating logical error rates spanning threshold into saturation.

    Parameters
    ----------
    n_points:
        Number of points linearly spaced across [p_min, p_max].
    noise_sigma:
        Observation noise relative to RMS signal amplitude.
    seed:
        PRNG seed (uses numpy.random.default_rng).
    distance:
        Code distance d (default 3 -> exponent e = 2.0).
    model:
        Saturating envelope model:
        - "pade": Rational saturation p_L = u / (1 + (C/p_sat)*u), where u = C*(p/p_th)^e.
        - "exponential": Soft exponential p_L = p_sat * (1 - exp(-(C/p_sat)*(p/p_th)^e)).
    p_sat:
        Asymptotic saturation ceiling (default 0.5 for depolarizing noise).
    p_min, p_max:
        Physical error rate interval.
    """
    rng = np.random.default_rng(seed)
    p = np.linspace(p_min, p_max, n_points)
    exp = (distance + 1) / 2.0
    u = QEC_C * (p / QEC_P_TH) ** exp
    if model == "pade":
        truth = u / (1.0 + (QEC_C / p_sat) * (p / QEC_P_TH) ** exp)
    elif model == "exponential":
        truth = p_sat * (1.0 - np.exp(-(QEC_C / p_sat) * (p / QEC_P_TH) ** exp))
    else:
        raise ValueError(f"unknown saturating model: {model}; expected 'pade' or 'exponential'")
    obs = truth + rng.normal(0, noise_sigma * np.sqrt(np.mean(truth ** 2)), n_points)
    return p, obs


def _mk(cid, terms, X_tr, y_tr, X_ho, y_ho, obs_rms, noise_est,
        data_origin=DataOrigin.SIMULATION):
    w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    resid = y_ho - X_ho @ w
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": noise_est, "holdout_n_samples": len(y_ho),
          "holdout_residuals": resid.tolist()}
    c = CandidateDynamics(lhs="logical_error",
                          terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                          candidate_id=cid)
    return ScoredCandidate(c, {"gate4_status": g4, "data_origin": data_origin},
                           train_loss=float(np.mean((y_tr - X_tr @ w) ** 2)),
                           holdout_rmse=rmse)


def run_qec_threshold_stage1(noise_sigma: float = 0.02, seed: int = 42,
                             bootstrap: bool = True):
    """Stage 1: four fixed forms on threshold-spanning data."""
    p, obs = generate_qec_threshold_data(noise_sigma=noise_sigma, seed=seed)
    ntr = len(p) // 2
    ptr, ytr, pho, yho = p[:ntr], obs[:ntr], p[ntr:], obs[ntr:]
    obs_rms = float(np.sqrt(np.mean(yho ** 2)))
    ns = float(noise_sigma * obs_rms)
    o = np.ones_like
    infos = [
        ("threshold_ratio", ("one", "pratio"), (ptr / QEC_P_TH) ** QEC_EXP, (pho / QEC_P_TH) ** QEC_EXP),
        ("plain_quad", ("one", "p2"), ptr ** 2, pho ** 2),
        ("linear_p", ("one", "p"), ptr, pho),
        ("constant", ("one",), None, None),
    ]
    scored, resid = [], {}
    for cid, terms, ftr, fho in infos:
        X_tr = np.column_stack([o(ptr), ftr]) if ftr is not None else o(ptr)[:, None]
        X_ho = np.column_stack([o(pho), fho]) if fho is not None else o(pho)[:, None]
        sc = _mk(cid, terms, X_tr, ytr, X_ho, yho, obs_rms, ns)
        scored.append(sc)
        resid[cid] = np.asarray(
            sc.provenance["gate4_status"]["holdout_residuals"], dtype=float)
    ranked = sorted([s for s in scored if np.isfinite(s.train_loss)],
                    key=lambda s: s.train_loss)
    adj = adjudicate(
        scored, resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
        resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
        bootstrap=bootstrap, bootstrap_n_boot=1000, bootstrap_seed=7,
        data_origin=DataOrigin.SIMULATION,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, scored


def run_qec_l3(seed: int = 42, probe_seed: int = 777):
    """L3: probe threshold-power vs linear over reachable p, fresh data, frozen fits."""
    p, obs = generate_qec_threshold_data(seed=seed)
    ntr = len(p) // 2
    ptr, ytr = p[:ntr], obs[:ntr]
    o = np.ones_like
    Xte = np.column_stack([o(ptr), (ptr / QEC_P_TH) ** QEC_EXP])
    wte, *_ = np.linalg.lstsq(Xte, ytr, rcond=None)
    Xli = np.column_stack([o(ptr), ptr])
    wli, *_ = np.linalg.lstsq(Xli, ytr, rcond=None)

    grid = np.linspace(0.005, 0.03, 300)
    div = np.abs((wte[0] + wte[1] * (grid / QEC_P_TH) ** QEC_EXP)
                 - (wli[0] + wli[1] * grid))
    p_star = float(grid[np.argmax(div)])

    rng = np.random.default_rng(probe_seed)
    pp = np.linspace(max(0.002, p_star - 0.004), p_star + 0.004, 30)
    tp = QEC_C * (pp / QEC_P_TH) ** QEC_EXP
    op = tp + rng.normal(0, 0.02 * np.sqrt(np.mean(tp ** 2)), 30)
    o_rms = float(np.sqrt(np.mean(op ** 2)))

    def frozen(cid, terms, w, feat):
        pred = w[0] + w[1] * feat(pp)
        rm = float(np.sqrt(np.mean((op - pred) ** 2)))
        g4 = {"passed": True, "holdout_obs_rms": o_rms,
              "holdout_noise_sigma": 0.02 * o_rms,
              "holdout_n_samples": len(op), "holdout_residuals": (op - pred).tolist()}
        c = CandidateDynamics(lhs="logical_error",
                              terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                              candidate_id=cid)
        return ScoredCandidate(c, {"gate4_status": g4, "data_origin": DataOrigin.SIMULATION},
                               train_loss=rm, holdout_rmse=rm), pred

    s1, p1 = frozen("threshold", ("one", "pratio"), wte, lambda e: (e / QEC_P_TH) ** QEC_EXP)
    s2, p2 = frozen("linear", ("one", "p"), wli, lambda e: e)
    # resid_best/resid_rival are SIGNED residuals; adjudicate squares them for
    # the tiebreak, which differences per-timepoint squared errors.
    #
    # This call site used to pre-square -- with a comment diagnosing exactly the
    # defect ("tiebreak contracts squared-error series, not signed residuals.
    # Passing (op - pred) here decouples the branch label from the RMSE
    # margin"). That was a correct diagnosis applied at the wrong layer: it
    # repaired this one caller while leaving every other caller (materials,
    # lynx, qec_threshold, the sweeps) passing signed residuals into a
    # tiebreak that differences them. The conversion now lives in adjudicate,
    # so pre-squaring here would double-square.
    adj = adjudicate([s1, s2], resid_best=(op - p1).tolist(),
                     resid_rival=(op - p2).tolist(),
                     bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7,
                     data_origin=DataOrigin.SIMULATION,
                     benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return {"p_star": p_star, "divergence": float(div.max())}, adj, [s1, s2]


def recover_exponent_per_distance(distances=(3, 5, 7), seeds=(11, 22, 33, 44, 55)):
    """Valid scalability estimator: saturating form, linear-space weighted fit.

    NOTE: the naive log-log polyfit is biased by the noise floor and must NOT
    be used (it returned 5.9/5.4/3.5 for true 2/3/4). This uses the
    repo's regularized form p_L,soft = p_sat*(1-exp(-(C/p_sat)*(p/p_th)^e)).
    """
    from scipy.optimize import curve_fit

    p_sat = 0.5
    p = np.logspace(-3.0, -1.7, 80)
    out = {}
    for dd in distances:
        e_true = (dd + 1) / 2
        ests = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            t = p_sat * (1.0 - np.exp(-(QEC_C / p_sat) * (p / QEC_P_TH) ** e_true))
            sig = 0.02 * t
            o = t + rng.normal(0, sig)
            try:
                popt, _ = curve_fit(
                    lambda pp, e: p_sat * (1 - np.exp(-(QEC_C / p_sat) * (pp / QEC_P_TH) ** e)),
                    p, o, p0=[2.0], sigma=sig, bounds=(0.5, 8.0), maxfev=5000)
                ests.append(float(popt[0]))
            except RuntimeError:
                ests.append(float("nan"))
        ests = np.asarray(ests)
        out[dd] = {"true": e_true, "mean": float(np.nanmean(ests)),
                   "std": float(np.nanstd(ests))}
    return out


def hello_world():
    print("=" * 64)
    print("QEC THRESHOLD SANDBOX: p_L = C*(p/p_th)^((d+1)/2), d=3")
    print("=" * 64)
    adj, scored = run_qec_threshold_stage1()
    for s in sorted(scored, key=lambda c: c.holdout_rmse):
        print(f"  {s.candidate.candidate_id:15s} holdout RMSE={s.holdout_rmse:.6f}")
    print(f"  verdict: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    probe, adj3, ss = run_qec_l3()
    print(f"  L3 probe p*={probe['p_star']:.4f}; "
          f"frozen threshold RMSE={ss[0].holdout_rmse:.4f} vs linear RMSE={ss[1].holdout_rmse:.4f}")
    print(f"  L3 verdict: {adj3.status.value} "
          f"promoted={adj3.promoted.candidate.candidate_id if adj3.promoted else None}")
    for dd, r in recover_exponent_per_distance().items():
        print(f"  d={dd}: true={r['true']:.1f} recovered={r['mean']:.2f}+/-{r['std']:.2f}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
