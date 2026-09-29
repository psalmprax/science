"""D5-CONF frozen noise-sweep benchmark (paired datasets).

Frozen benchmark matrix (do not tune to hit a target false-promotion rate;
false promotion is a REPORTED outcome):

  true law:  log Pi0 = log C + a log Pi1 + b log Pi2 + c log Pi3
  C = 3.2, (a, b, c) = (-2.7, -0.8, -0.2)
  predictors: log-uniform, independent (well-conditioned by design)
  n_train = 60, n_holdout = 40
  seeds = (0, 1, 2, 3, 4); base X fixed per seed across noise levels (paired)
  noise: multiplicative lognormal, sigma in {0.01, 0.03, 0.05, 0.10}
  candidates: structures = subsets of {Pi1, Pi2, Pi3} fit by OLS in log space
  adjudication: APEDP adjudicate() with holdout_scale from the frozen record

Recorded per (seed, sigma): structural recovery, false promotion,
calibrated abstention, parameter error (true-structure fit), holdout RMSEs,
candidate count, runtime.
"""
from __future__ import annotations

import time

import numpy as np

from apedp_dyn.conf.dimensionless import D5_CONF_VERSION
from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

TRUE_C = 3.2
TRUE_EXPONENTS = (-2.7, -0.8, -0.2)
N_TRAIN = 60
N_HOLDOUT = 40
SEEDS = (0, 1, 2, 3, 4)
NOISE_SIGMAS = (0.01, 0.03, 0.05, 0.10)
# log-uniform predictor ranges (Pi1, Pi2, Pi3)
LOG_RANGES = ((-0.7, 0.7), (-1.1, 1.1), (-1.6, 1.6))

CANDIDATE_STRUCTURES: dict[str, tuple[int, ...]] = {
    # name -> indices into (Pi1, Pi2, Pi3)
    "M_true": (0, 1, 2),
    "M_nobeta": (0, 2),
    "M_bohm": (0,),
    "M_nu_only": (2,),
}
# Distinct grammar term sets per structure so adjudication _identity
# treats them as structurally distinct rivals.
STRUCTURE_TERMS: dict[str, tuple[str, ...]] = {
    "M_true": ("one", "x", "x3", "v"),
    "M_nobeta": ("one", "x", "v"),
    "M_bohm": ("one", "x"),
    "M_nu_only": ("one", "v"),
}


def generate_base_predictors(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic base predictors (shared across noise levels = paired)."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_HOLDOUT
    cols = [
        rng.uniform(lo, hi, size=n) for (lo, hi) in LOG_RANGES
    ]
    X = np.column_stack(cols)
    return X[:N_TRAIN], X[N_TRAIN:]


def synthesize(seed: int, sigma: float) -> dict:
    """Generate one paired (train, holdout) dataset at noise level sigma."""
    Xtr, Xho = generate_base_predictors(seed)
    rng = np.random.default_rng(10_000 * seed + round(sigma * 1000))
    a = np.array(TRUE_EXPONENTS)
    ytr = np.log(TRUE_C) + Xtr @ a + rng.normal(0.0, sigma, size=len(Xtr))
    yho = np.log(TRUE_C) + Xho @ a + rng.normal(0.0, sigma, size=len(Xho))
    return {"Xtr": Xtr, "ytr": ytr, "Xho": Xho, "yho": yho}


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.asarray(coef, dtype=float)


def run_cell(seed: int, sigma: float) -> dict:
    """Fit all candidate structures on one dataset; adjudicate; score."""
    t0 = time.perf_counter()
    d = synthesize(seed, sigma)
    Xtr, ytr, Xho, yho = d["Xtr"], d["ytr"], d["Xho"], d["yho"]
    obs_rms = float(np.sqrt(np.mean(yho**2)))

    scored: list[ScoredCandidate] = []
    fits: dict[str, dict] = {}
    for name, idx in CANDIDATE_STRUCTURES.items():
        coef = _fit_ols(Xtr[:, list(idx)], ytr)
        pred_tr = np.column_stack([np.ones(len(Xtr)), Xtr[:, list(idx)]]) @ coef
        pred_ho = np.column_stack([np.ones(len(Xho)), Xho[:, list(idx)]]) @ coef
        sq_err = (pred_ho - yho) ** 2
        train_rmse = float(np.sqrt(np.mean((pred_tr - ytr) ** 2)))
        ho_rmse = float(np.sqrt(np.mean(sq_err)))
        terms = tuple(Term(t, f"a{i}") for i, t in enumerate(STRUCTURE_TERMS[name]))
        cand = CandidateDynamics(lhs="logPi0", terms=terms, candidate_id=f"conf-{name}")
        prov = {
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": sigma,
                "holdout_n_samples": len(yho),
                # Per-point holdout squared error. This benchmark fits by OLS
                # rather than through fit_candidate, so no gate runs and there
                # is no gate-recorded series to inherit. Recording it here puts
                # adjudication's residual-whiteness note on the same footing as
                # every other caller: the evidence is computed once, at the
                # point the error is, and travels with the candidate. tolist()
                # because these records are json-dumped into the report.
                "holdout_sq_error": sq_err.tolist(),
            },
        }
        scored.append(ScoredCandidate(cand, prov, train_loss=train_rmse, holdout_rmse=ho_rmse))
        fits[name] = {"coef": [float(v) for v in coef], "train_rmse": train_rmse,
                      "holdout_rmse": ho_rmse}

    adj = adjudicate(scored)
    promoted = adj.promoted.candidate.candidate_id if adj.promoted is not None else None
    promoted_structure = promoted.replace("conf-", "") if promoted else None
    recovery = adj.status == EpistemicStatus.UNIQUE_SUPPORT and promoted_structure == "M_true"
    false_promotion = adj.status == EpistemicStatus.UNIQUE_SUPPORT and promoted_structure != "M_true"
    abstention = adj.status in (EpistemicStatus.MULTI_HYPOTHESIS, EpistemicStatus.INSUFFICIENT_EVIDENCE)

    # Parameter error of the true-structure fit vs truth.
    true_coef = fits["M_true"]["coef"]  # [logC, a, b, c]
    param_err = float(np.linalg.norm(
        np.array(true_coef[1:]) - np.array(TRUE_EXPONENTS)))

    return {
        "seed": seed,
        "sigma": sigma,
        "status": adj.status.value,
        "promoted": promoted_structure,
        "structural_recovery": bool(recovery),
        "false_promotion": bool(false_promotion),
        "calibrated_abstention": bool(abstention),
        "delta_rmse": adj.delta_rmse,
        "tau_epi": adj.tau_epi,
        "param_error_true_structure": param_err,
        "holdout_rmse": {k: v["holdout_rmse"] for k, v in fits.items()},
        "n_candidates": len(scored),
        "runtime_s": time.perf_counter() - t0,
    }


def run_sweep(
    seeds: tuple[int, ...] = SEEDS,
    sigmas: tuple[float, ...] = NOISE_SIGMAS,
) -> dict:
    cells = [run_cell(s, sig) for s in seeds for sig in sigmas]
    by_sigma: dict[str, dict] = {}
    for sig in sigmas:
        sub = [c for c in cells if c["sigma"] == sig]
        by_sigma[str(sig)] = {
            "n": len(sub),
            "structural_recovery_rate": sum(c["structural_recovery"] for c in sub) / len(sub),
            # False promotion is REPORTED, never tuned against.
            "false_promotion_rate": sum(c["false_promotion"] for c in sub) / len(sub),
            "abstention_rate": sum(c["calibrated_abstention"] for c in sub) / len(sub),
            "mean_param_error": float(np.mean([c["param_error_true_structure"] for c in sub])),
            "mean_runtime_s": float(np.mean([c["runtime_s"] for c in sub])),
        }
    return {
        "d5_conf_version": D5_CONF_VERSION,
        "true_C": TRUE_C,
        "true_exponents": list(TRUE_EXPONENTS),
        "n_train": N_TRAIN,
        "n_holdout": N_HOLDOUT,
        "seeds": list(seeds),
        "noise_sigmas": list(sigmas),
        "candidate_structures": {k: list(v) for k, v in CANDIDATE_STRUCTURES.items()},
        "cells": cells,
        "by_sigma": by_sigma,
    }
