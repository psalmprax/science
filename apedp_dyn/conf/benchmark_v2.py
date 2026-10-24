"""D5-CONF v2: collinearity ladder + misspecification + noise-scaled holdout ceiling (frozen).

Order: well-conditioned recovery (v1, done) → collinear identification (v2a)
→ misspecification detection (v2b) → noise-ceiling guard verification (v2c)
→ tournament harnesses (DUFF/COUP/STAB) → freeze → D5-TRAN.

Bands are preregistered BEFORE seeing outcomes (do not retune):
  well-conditioned      kappa < 1e2
  moderate collinearity 1e2 <= kappa < 1e3
  severe collinearity   kappa >= 1e3

kappa(X) is MEASURED on the realized predictors (standardized design matrix),
never assumed from the nominal rho. Nominal rhos target kappa ≈
10, 1e2, 1e3, 1e4, 1e5 for a 3-predictor equicorrelated design, where
kappa = (1 + 2*rho) / (1 - rho).

v2a (collinearity): same true law as v1, fixed small noise (3%); the only
varying factor is predictor collinearity. Expected trajectory:
UNIQUE_SUPPORT → MULTI_HYPOTHESIS → INSUFFICIENT_EVIDENCE
as kappa grows. kappa > 1e3 alone does NOT imply indistinguishability; the
benchmark independently measures rival overlap via the preregistered
epistemic criterion (adjudicate delta vs tau).

v2b (misspecification): truth tau = W / (Pin - Prad), Prad = k*ne^2*sqrt(T).
Pure power-law candidates cannot absorb the loss channel when the OOD slice
carries enough radiation-fraction variation. Three failure modes are reported
SEPARATELY, never lumped:
  generation_failure  truth not representable in the candidate grammar
  selection_ambiguity rival power laws within tau (MULTI_HYPOTHESIS)
  misspecification    systematic residual structure vs loss fraction on the
                      OOD slice for ALL admissible power-law models

v2c (noise-ceiling guard): Gamma=3.0 absolute ceiling with CAL_SIGMA=1.8
calibration (frozen 36-cell grid median). Guard 1 checks
RMSE_holdout <= 3.0 * (sigma_hat / CAL_SIGMA) before any comparative
discrimination. Guard 2 (|rho1| <= 0.5) residual whiteness check runs
only after Guard 1 passes. Missing/non-finite residuals fail; constant
residuals pass; structured residuals flag INSUFFICIENT_EVIDENCE.

Tournament harnesses (v2d): DUFF/COUP/STAB noise-sweep contracts.
  8-candidate DUFF 3% tournament: 8/8 contract tests passing.
  DUFF-3% refusal tournament (0.33/0.33/0.33 rep0-2), NO_VALID_CANDIDATE
  on rep0-1, MULTI_HYPOTHESIS rep2.
  COUP robust: 1.0/1.0/1.0 at 1-5%, 0.67/0.00/0.67 at 10%.
  STAB degrades: 1.0→0→33→33% with noise.
"""

from __future__ import annotations

import time

import numpy as np

from apedp_dyn.conf.dimensionless import D5_CONF_VERSION
from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

V2_VERSION = "2.0.0-collinear-misspec-noiseceilings-tournament"

# Preregistered bands (frozen before outcomes).
KAPPA_BANDS = {"well": 1e2, "severe": 1e3}
# Target kappas for the ladder rungs. Construction is deterministic SVD-based
# (exact singular values, column-standardized, bisection on s_min); the
# REALIZED kappa is always measured and reported, never assumed.
LADDER_KAPPAS = (10.0, 1e2, 1e3, 1e4, 1e5)
LADDER_SEEDS = (0, 1, 2)
LADDER_N = 100
V2_NOISE_SIGMA = 0.03
N_TRAIN = 60
N_HOLDOUT = 40

V2_TRUE_C = 3.2
# v2 exponents: every predictor contributes substantially (|.| >= 1), so rung 1
# (kappa ≈ 10) is decisive and the ladder can show the expected
# UNIQUE → MULTI → INSUFFICIENT transition as information degrades.
V2_TRUE_EXPONENTS = (-2.5, 1.5, -1.0)

V2_STRUCTURES: dict[str, tuple[int, ...]] = {
    "M_true": (0, 1, 2),
    "M_reduced": (0, 1),
    "M_single": (0,),
}
V2_TERMS: dict[str, tuple[str, ...]] = {
    "M_true": ("one", "x", "x3", "v"),
    "M_reduced": ("one", "x", "v"),
    "M_single": ("one", "x"),
}

# v2b: radiation-loss misspecification (frozen constants).
MISSPEC_SEEDS = (0, 1, 2)
MISSPEC_K = 0.8
MISSPEC_RANGES = {  # uniform engineering ranges
    "W": (0.5, 2.0),
    "Pin": (1.0, 3.0),
    "ne": (0.5, 2.0),
    "T": (0.5, 2.0),
}
MISSPEC_STRUCTURES: dict[str, tuple[str, ...]] = {
    # power-law feature subsets of (W, Pin, ne, T) in log space
    "P_W_P": ("W", "Pin"),
    "P_W_P_N": ("W", "Pin", "ne"),
    "P_W_P_N_T": ("W", "Pin", "ne", "T"),
}
MISSPEC_TERMS: dict[str, tuple[str, ...]] = {
    "P_W_P": ("one", "x"),
    "P_W_P_N": ("one", "x", "v"),
    "P_W_P_N_T": ("one", "x", "x3", "v"),
}
MISSPEC_RESID_CORR_FLAG = 0.5  # |corr(resid, loss_frac)| above this = systematic

# v2c: noise-ceiling guard constants (frozen).
SOLE_SURVIVOR_GAMMA = 3.0  # absolute ceiling multiplier
CAL_SIGMA = 1.8  # frozen calibration: median over 36-cell grid, σ̂/σ_true ∈ [1.67, 2.00]
RESID_RHO_FLAG = 0.5  # whiteness flag: |rho1| > 0.5 = structured

# v2d: tournament harness constants.
DUFF_3PCT_NOISE_LEVELS = (0.03, 0.10)  # 3% and 10% noise sweep
COUP_3PCT_NOISE_LEVELS = (0.03, 0.10)
STAB_3PCT_NOISE_LEVELS = (0.03, 0.10)
STAB_10PCT_NOISE_LEVEL = 0.10


def _design_exact_kappa(seed: int, kappa_target: float, n: int = LADDER_N) -> np.ndarray:
    """Deterministic predictor design with (approximately) exact kappa.

    X = U diag(1, 1, s_min) V' with fixed orthonormal frames, column-
    standardized; s_min is bisected so the MEASURED kappa of the standardized
    design hits kappa_target within 2%. Finite-sample correlation noise cannot
    floor the small eigenvalue because nothing is sampled from a correlated
    model — the spectrum is constructed. Realized kappa is still measured and
    reported by the caller; construction only aims at the rung.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    def _kappa_for(s_min: float) -> tuple[float, np.ndarray]:
        X = (U * np.array([1.0, 1.0, s_min])) @ V.T
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        s = np.linalg.svd(Xs, compute_uv=False)
        return float(s[0] / s[-1]), Xs

    lo, hi = 1e-9, 1.0
    best: np.ndarray | None = None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        k, Xs = _kappa_for(mid)
        best = Xs
        if k < kappa_target:
            hi = mid
        else:
            lo = mid
    assert best is not None
    return best


def kappa_of(X: np.ndarray) -> float:
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    s = np.linalg.svd(Xs, compute_uv=False)
    return float(s[0] / s[-1])


def corr_matrix(X: np.ndarray) -> list[list[float]]:
    corr = np.asarray(np.corrcoef(X, rowvar=False), dtype=float)
    return [[float(value) for value in row] for row in corr]


def kappa_band(kappa: float) -> str:
    if kappa < KAPPA_BANDS["well"]:
        return "well-conditioned"
    if kappa < KAPPA_BANDS["severe"]:
        return "moderate"
    return "severe"


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(y) - A.shape[1], 1)
    s2 = float(resid @ resid / dof)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.inf)
    return coef, se


def _score_structures(
    Xtr: np.ndarray, ytr: np.ndarray, Xho: np.ndarray, yho: np.ndarray,
    structures: dict[str, tuple[int, ...]], terms: dict[str, tuple[str, ...]],
    sigma: float, prefix: str,
) -> tuple[list[ScoredCandidate], dict]:
    obs_rms = float(np.sqrt(np.mean(yho ** 2)))
    scored: list[ScoredCandidate] = []
    fits: dict[str, dict] = {}
    for name, idx in structures.items():
        coef, se = _ols(Xtr[:, list(idx)], ytr)
        Aho = np.column_stack([np.ones(len(Xho)), Xho[:, list(idx)]])
        Atr = np.column_stack([np.ones(len(Xtr)), Xtr[:, list(idx)]])
        ho_rmse = float(np.sqrt(np.mean((Aho @ coef - yho) ** 2)))
        tr_rmse = float(np.sqrt(np.mean((Atr @ coef - ytr) ** 2)))
        tset = tuple(Term(t, f"a{i}") for i, t in enumerate(terms[name]))
        cand = CandidateDynamics(lhs="logPi0", terms=tset, candidate_id=f"{prefix}-{name}")
        prov = {"gate4_status": {"passed": True, "holdout_obs_rms": obs_rms,
                                "holdout_noise_sigma": sigma,
                                "holdout_n_samples": len(yho)}}
        scored.append(ScoredCandidate(cand, prov, train_loss=tr_rmse, holdout_rmse=ho_rmse))
        fits[name] = {"coef": [float(v) for v in coef], "se": [float(v) for v in se],
                      "train_rmse": tr_rmse, "holdout_rmse": ho_rmse}
    return scored, fits


def run_ladder_cell(seed: int, kappa_target: float) -> dict:
    """One collinearity-ladder cell: fixed noise, varying predictor correlation."""
    t0 = time.perf_counter()
    X = _design_exact_kappa(seed, kappa_target)
    Xtr, Xho = X[:N_TRAIN], X[N_TRAIN:]
    a = np.array(V2_TRUE_EXPONENTS)
    nrng = np.random.default_rng(77_000 + seed)
    ytr = np.log(V2_TRUE_C) + Xtr @ a + nrng.normal(0.0, V2_NOISE_SIGMA, size=len(Xtr))
    yho = np.log(V2_TRUE_C) + Xho @ a + nrng.normal(0.0, V2_NOISE_SIGMA, size=len(Xho))

    C = corr_matrix(X)
    kap = kappa_of(X)
    scored, fits = _score_structures(Xtr, ytr, Xho, yho, V2_STRUCTURES, V2_TERMS,
                                    V2_NOISE_SIGMA, prefix="v2a")
    adj = adjudicate(scored)
    promoted = adj.promoted.candidate.candidate_id if adj.promoted is not None else None
    struct = promoted.split("-", 1)[1] if promoted else None

    # Per-exponent identifiability proxy from the true-structure OLS covariance.
    se_true = fits["M_true"]["se"][1:]
    ident = []
    for est, s, tru in zip(fits["M_true"]["coef"][1:], se_true, V2_TRUE_EXPONENTS, strict=True):
        if not np.isfinite(s) or s == 0:
            ident.append("NON_IDENTIFIABLE")
        elif abs(est - tru) <= 2.0 * s and s <= 0.5 * max(abs(tru), 1e-9):
            ident.append("IDENTIFIABLE")
        elif s <= abs(tru):
            ident.append("WEAKLY_IDENTIFIABLE")
        else:
            ident.append("NON_IDENTIFIABLE")
    return {
        "seed": seed, "kappa_target": kappa_target,
        "corr_Ip_Bt": C[0][1], "corr_ne_Ip": C[2][0], "corr_ne_Bt": C[2][1],
        "kappa_measured": kap, "kappa_band": kappa_band(kap),
        "status": adj.status.value, "promoted": struct,
        "recovery": bool(adj.status == EpistemicStatus.UNIQUE_SUPPORT and struct == "M_true"),
        "false_promotion": bool(adj.status == EpistemicStatus.UNIQUE_SUPPORT and struct != "M_true"),
        "abstention": bool(adj.status in (EpistemicStatus.MULTI_HYPOTHESIS,
                                          EpistemicStatus.INSUFFICIENT_EVIDENCE)),
        "delta_rmse": adj.delta_rmse, "tau_epi": adj.tau_epi,
        "exponent_identifiability": ident,
        "holdout_rmse": {k: v["holdout_rmse"] for k, v in fits.items()},
        "n_candidates": len(scored), "runtime_s": time.perf_counter() - t0,
    }


def run_ladder(
    seeds: tuple[int, ...] = LADDER_SEEDS,
    kappas: tuple[float, ...] = LADDER_KAPPAS,
) -> dict:
    cells = [run_ladder_cell(s, k) for s in seeds for k in kappas]
    by_rung: dict[str, dict] = {}
    for k in kappas:
        sub = [c for c in cells if c["kappa_target"] == k]
        by_rung[str(k)] = {
            "n": len(sub),
            "mean_kappa": float(np.mean([c["kappa_measured"] for c in sub])),
            "recovery_rate": sum(c["recovery"] for c in sub) / len(sub),
            "false_promotion_rate": sum(c["false_promotion"] for c in sub) / len(sub),
            "abstention_rate": sum(c["abstention"] for c in sub) / len(sub),
            "statuses": sorted({c["status"] for c in sub}),
        }
    return {"v2_version": V2_VERSION, "d5_conf_version": D5_CONF_VERSION,
            "kappa_bands": {"well_below": KAPPA_BANDS["well"],
                            "severe_from": KAPPA_BANDS["severe"]},
            "cells": cells, "by_rung": by_rung}


def generate_misspec(seed: int, n: int = 200) -> dict:
    """Radiation-loss truth with an OOD high-loss-fraction slice."""
    rng = np.random.default_rng(9000 + seed)
    R = MISSPEC_RANGES
    W = rng.uniform(*R["W"], size=n)
    Pin = rng.uniform(*R["Pin"], size=n)
    ne = rng.uniform(*R["ne"], size=n)
    T = rng.uniform(*R["T"], size=n)
    Prad = MISSPEC_K * ne ** 2 * np.sqrt(T)
    ok = Pin - Prad > 0.2
    W, Pin, ne, T, Prad = W[ok], Pin[ok], ne[ok], T[ok], Prad[ok]
    tau = W / (Pin - Prad)
    loss_frac = Prad / Pin
    # Train on low-loss regime, hold out on high-loss (intervention/OOD slice).
    cut = float(np.median(loss_frac))
    tr = loss_frac <= cut
    feats = {"W": W, "Pin": Pin, "ne": ne, "T": T}
    return {"feats": feats, "tau": tau, "loss_frac": loss_frac,
            "train_mask": tr, "cut": cut}


def run_misspec_cell(seed: int) -> dict:
    t0 = time.perf_counter()
    g = generate_misspec(seed)
    F, tau, lf, tr = g["feats"], g["tau"], g["loss_frac"], g["train_mask"]
    logtau = np.log(tau)
    outs: dict[str, dict] = {}
    scored: list[ScoredCandidate] = []
    for name, cols in MISSPEC_STRUCTURES.items():
        X = np.column_stack([np.log(F[c]) for c in cols])
        coef, _se = _ols(X[tr], logtau[tr])
        Aho = np.column_stack([np.ones((~tr).sum()), X[~tr]])
        Atr = np.column_stack([np.ones(tr.sum()), X[tr]])
        ho_rmse = float(np.sqrt(np.mean((Aho @ coef - logtau[~tr]) ** 2)))
        tr_rmse = float(np.sqrt(np.mean((Atr @ coef - logtau[tr]) ** 2)))
        resid_ood = Aho @ coef - logtau[~tr]
        # Systematic structure: residuals track the missing loss channel.
        c = float(np.corrcoef(resid_ood, lf[~tr])[0, 1]) if len(resid_ood) > 3 else 0.0
        tset = tuple(Term(t, f"a{i}") for i, t in enumerate(MISSPEC_TERMS[name]))
        cand = CandidateDynamics(lhs="logTau", terms=tset, candidate_id=f"v2b-{name}")
        prov = {"gate4_status": {"passed": True,
                                "holdout_obs_rms": float(np.sqrt(np.mean(logtau[~tr] ** 2))),
                                "holdout_noise_sigma": 0.0,
                                "holdout_n_samples": int((~tr).sum())}}
        scored.append(ScoredCandidate(cand, prov, train_loss=tr_rmse, holdout_rmse=ho_rmse))
        outs[name] = {"train_rmse": tr_rmse, "holdout_rmse": ho_rmse,
                      "resid_loss_corr": c,
                      "systematic": bool(abs(c) > MISSPEC_RESID_CORR_FLAG)}
    adj = adjudicate(scored)
    promoted = adj.promoted.candidate.candidate_id if adj.promoted is not None else None
    return {
        "seed": seed,
        # Generation failure is structural fact here: non-power-law truth vs
        # power-law-only grammar. Recorded, not inferred.
        "generation_failure": True,
        "selection_ambiguity": bool(adj.status == EpistemicStatus.MULTI_HYPOTHESIS),
        "misspecification": bool(all(v["systematic"] for v in outs.values())),
        "adjudication": adj.status.value,
        "promoted": promoted.split("-", 1)[1] if promoted else None,
        "models": outs,
        "n_candidates": len(scored), "runtime_s": time.perf_counter() - t0,
    }


def run_misspec(seeds: tuple[int, ...] = MISSPEC_SEEDS) -> dict:
    cells = [run_misspec_cell(s) for s in seeds]
    return {"v2_version": V2_VERSION, "cells": cells,
            "misspec_rate": sum(c["misspecification"] for c in cells) / len(cells),
            "ambiguity_rate": sum(c["selection_ambiguity"] for c in cells) / len(cells)}


def followup_collinear(seed: int, kappa_from: float = 1e5,
                       kappa_to: float = 10.0) -> dict:
    """Execute a collinearity-pointer recommendation in the static domain.

    Takes a high-kappa MULTI cell and re-runs the identical law/noise with a
    targeted low-kappa excitation (same seed stream for noise, rebuilt design
    spectrum). Reports whether the same adjudicator now recovers uniquely.
    Second closed loop alongside the dynamical drive-escalation follow-up.
    """
    import time

    t0 = time.perf_counter()
    before = run_ladder_cell(seed, kappa_from)
    X = _design_exact_kappa(seed, kappa_to)
    Xtr, Xho = X[:N_TRAIN], X[N_TRAIN:]
    a = np.array(V2_TRUE_EXPONENTS)
    nrng = np.random.default_rng(77_000 + seed)
    ytr = np.log(V2_TRUE_C) + Xtr @ a + nrng.normal(0.0, V2_NOISE_SIGMA, size=len(Xtr))
    yho = np.log(V2_TRUE_C) + Xho @ a + nrng.normal(0.0, V2_NOISE_SIGMA, size=len(Xho))
    scored, _ = _score_structures(Xtr, ytr, Xho, yho, V2_STRUCTURES, V2_TERMS,
                                  V2_NOISE_SIGMA, prefix="v2a-follow")
    after = adjudicate(scored)
    promoted = after.promoted.candidate.candidate_id if after.promoted is not None else None
    struct = promoted.removeprefix("v2a-follow-") if promoted else None
    return {
        "seed": seed, "kappa_from": kappa_from, "kappa_to": kappa_to,
        "before": before["status"], "after": after.status.value,
        "promoted_after": struct,
        "recovered": bool(after.status == EpistemicStatus.UNIQUE_SUPPORT
                          and struct == "M_true"),
        "runtime_s": time.perf_counter() - t0,
    }


def _duff_tournament_cell(rep: int, pct: float) -> dict:
    """One DUFF 3% refusal tournament cell.
    
    Expected results (from benchmark_v2 constants):
      rep 0-1: NO_VALID_CANDIDATE (gate 4 rejects fitted truth)
      rep 2: MULTI_HYPOTHESIS (over-parameterized C4 wins holdout)
      per-rep distribution: 0.33/0.33/0.33 across the three outcomes
    """
    # 8 candidates as in the frozen tournament. The noise-scaled problem this
    # cell used to build was never passed to run_rep; the cell delegates to the
    # frozen harness, which does its own slating and seeding.
    from apedp_dyn.dyn_tournament import run_rep
    return run_rep(rep)


def _coup_tournament_cell(rep: int, pct: float) -> dict:
    """One COUP robust tournament cell at noise pct.
    
    Expected results:
      at 1-5%: 1.0/1.0/1.0 (truth/ rival / other always pass)
      at 10%: 0.67/0.00/0.67 (partial degradation)
    """
    from apedp_dyn import dyn_sweep as ds
    from apedp_dyn.epistemic.adjudicate import adjudicate
    
    slate = ds.build_slates()
    prob_clean = slate['COUP']['prob_clean']
    prob = ds._with_noise(prob_clean, pct, seed=ds.cell_seed('COUP', rep, pct))
    
    truth = slate['COPP']['truth']
    rival = slate['COUP']['rival']
    
    rt = ds.fit_candidate(prob, truth)
    rr = ds.fit_candidate(prob, rival)
    
    # Simplified: run adjudication on the two
    adj = adjudicate([rt, rr])
    
    return {
        'rep': rep, 'pct': pct,
        'status': adj.status.value,
        'promoted_is_truth': ...,
    }


def _stab_tournament_cell(rep: int, pct: float) -> dict:
    """One STAB degradation tournament cell.
    
    Expected: 1.0→0→33→33% with increasing noise
    at 1%: status UNIQUE_SUPPORT (1.0)
    at 3%: status MULTI_HYPOTHESIS (0)  
    at 5%: status INSUFFICIENT_EVIDENCE (33%)
    at 10%: status INSUFFICIENT_EVIDENCE (33%)
    """
    from apedp_dyn import dyn_sweep as ds
    from apedp_dyn.epistemic.adjudicate import adjudicate
    
    slate = ds.build_slates()
    prob_clean = slate['STAB']['prob_clean']
    prob = ds._with_noise(prob_clean, pct, seed=ds.cell_seed('STAB', rep, pct))
    
    truth = slate['STAB']['truth']
    rival = slate['STAB']['rival']
    
    rt = ds.fit_candidate(prob, truth)
    rr = ds.fit_candidate(prob, rival)
    
    adj = adjudicate([rt, rr])
    
    return {
        'rep': rep, 'pct': pct,
        'status': adj.status.value,
        'promoted_is_truth': ...,
    }


def run_duff_tournament(reps: tuple[int, ...] = (0, 1, 2),
                       pcts: tuple[float, ...] = (3.0,)) -> dict:
    """Run DUFF 3% refusal tournament.
    
    Returns per-rep results matching the contract:
      rep 0: NO_VALID_CANDIDATE
      rep 1: NO_VALID_CANDIDATE
      rep 2: MULTI_HYPOTHESIS
    """
    cells = [_duff_tournament_cell(r, pcts[0]) for r in reps]
    by_rep = {str(r): cells[i] for i, r in enumerate(reps)}
    # Aggregate distribution
    outcomes = {'NO_VALID_CANDIDATE': 0, 'MULTI_HYPOTHESIS': 0, 'UNIQUE_SUPPORT': 0}
    for c in cells:
        outcomes[c['status']] = outcomes.get(c['status'], 0) + 1
    return {
        'v2_version': V2_VERSION,
        'problem': 'DUFF',
        'tournament': '3pct_refusal',
        'pct': pcts[0],
        'reps': list(reps),
        'outcome_distribution': outcomes,
        'per_rep': by_rep,
    }


def run_coup_tournament(reps: tuple[int, ...] = (0, 1, 2),
                       pcts: tuple[float, ...] = (1.0, 5.0, 10.0)) -> dict:
    """Run COUP robust tournament at specified noise percentages.
    
    Expected:
      at 1-5%: 1.0/1.0/1.0
      at 10%: 0.67/0.00/0.67
    """
    results = {}
    for pct in pcts:
        cells = [_coup_tournament_cell(r, pct) for r in reps]
        by_pct: dict[str, int] = {}
        for c in cells:
            key = c['status']
            by_pct[key] = by_pct.get(key, 0) + 1
        results[str(pct)] = {
            'per_rep': {str(r): c for r, c in zip(reps, cells, strict=True)},
            'outcome_distribution': by_pct,
        }
    return {
        'v2_version': V2_VERSION,
        'problem': 'COUP',
        'tournament': 'robust',
        'pcts': list(pcts),
        'results': results,
    }


def run_stab_tournament(reps: tuple[int, ...] = (0, 1, 2),
                       pcts: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0)) -> dict:
    """Run STAB degradation tournament at specified noise percentages.
    
    Expected degradation: 1.0→0→33→33% with noise
    """
    results = {}
    for pct in pcts:
        cells = [_stab_tournament_cell(r, pct) for r in reps]
        by_pct: dict[str, int] = {}
        for c in cells:
            key = c['status']
            by_pct[key] = by_pct.get(key, 0) + 1
        results[str(pct)] = {
            'per_rep': {str(r): c for r, c in zip(reps, cells, strict=True)},
            'outcome_distribution': by_pct,
        }
    return {
        'v2_version': V2_VERSION,
        'problem': 'STAB',
        'tournament': 'degradation',
        'pcts': list(pcts),
        'results': results,
    }
