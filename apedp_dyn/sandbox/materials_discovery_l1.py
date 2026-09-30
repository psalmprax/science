"""APEDP Level 1 Constrained Materials Discovery Benchmark.

Demonstrates L1 constrained equation discovery and epistemic adjudication
on physical metallurgy strengthening mechanisms (Hall-Petch grain boundary
strengthening, Orowan precipitation hardening, and thermal/strain-rate effects).

Taxonomy:
  BenchmarkLevel: L1_CONSTRAINED_DISCOVERY
  Tests candidate generation from restricted physical grammar, parameter
  fitting on train splits, holdout validation, and epistemic adjudication.

Benchmarks:
  M1 (Single Mechanism):
      True: sigma_y = sigma_0 + k_hp * d^(-1/2)
      Goal: APEDP isolates the Hall-Petch term over linear, inverse, and null.
      Outcome: UNIQUE_SUPPORT for hp_single.

  M2 (Competing Independent Mechanisms):
      True: sigma_y = sigma_0 + k_hp * d^(-1/2) + A_precip * Phi^(1/2)
      Goal: APEDP discovers dual mechanism when grain size d and particle
            fraction Phi vary independently.
      Outcome: UNIQUE_SUPPORT for dual_hp_precip.

  M3 (Mechanism Overlap / Collinear Coupling):
      True: sigma_y = sigma_0 + k_hp * d^(-1/2) + A_precip * Phi^(1/2)
      Condition: Microstructural Zener pinning couples Phi ~ 1/d, making
                 d^(-1/2) and Phi^(1/2) observationally collinear.
      Goal: Test epistemic restraint under structural unidentifiability.
      Outcome: MULTI_HYPOTHESIS (refuses UNIQUE_SUPPORT despite point-margin).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")


import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    BenchmarkLevel,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

SOLE_SURVIVOR_GAMMA = 3.0
CAL_SIGMA = 1.8
RESID_RHO_FLAG = 0.5


# ── Data Generators for M1, M2, M3 ───────────────────────────────────────

def generate_m1_data(
    n_points: int = 60,
    noise_sigma: float = 0.02,
    sigma_0: float = 100.0,
    k_hp: float = 0.5,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate M1 single-mechanism data: sigma_y = sigma_0 + k_hp * d^(-1/2) + noise.

    Physical metallurgy parameters (e.g. low-carbon steel):
      sigma_0: friction stress (100 MPa)
      k_hp: Hall-Petch slope (0.5 MPa·m^(1/2))
      d: grain size from 1 um (1e-6 m) to 100 um (1e-4 m)
    """
    rng = np.random.RandomState(seed)
    d = np.logspace(-6, -4, n_points)
    T = np.full(n_points, 300.0)
    Phi = rng.uniform(0.02, 0.08, n_points)
    eps_dot = np.full(n_points, 1e-3)

    sigma_clean = sigma_0 + k_hp * (d ** -0.5)
    noise = rng.normal(0, noise_sigma * sigma_0, n_points)
    sigma_obs = sigma_clean + noise

    features = {"d": d, "T": T, "Phi": Phi, "eps_dot": eps_dot}
    return features, sigma_obs


def generate_m2_data(
    n_points: int = 60,
    noise_sigma: float = 0.02,
    sigma_0: float = 100.0,
    k_hp: float = 0.5,
    a_precip: float = 300.0,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate M2 competing-mechanism data: independent grain size d and precipitate Phi.

    Physical metallurgy parameters:
      sigma_0: 100 MPa, k_hp: 0.5 MPa·m^(1/2)
      a_precip: Orowan precipitate strengthening coefficient (300 MPa)
      Phi: particle volume fraction (0.01 to 0.15) varying independently of d
    """
    rng = np.random.RandomState(seed)
    d = np.logspace(-6, -4, n_points)
    T = np.full(n_points, 300.0)
    Phi = rng.uniform(0.01, 0.15, n_points)
    eps_dot = np.full(n_points, 1e-3)

    sigma_clean = sigma_0 + k_hp * (d ** -0.5) + a_precip * (Phi ** 0.5)
    noise = rng.normal(0, noise_sigma * sigma_0, n_points)
    sigma_obs = sigma_clean + noise

    features = {"d": d, "T": T, "Phi": Phi, "eps_dot": eps_dot}
    return features, sigma_obs


def generate_m3_data(
    n_points: int = 60,
    noise_sigma: float = 0.03,
    sigma_0: float = 100.0,
    k_hp: float = 0.5,
    a_precip: float = 300.0,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate M3 mechanism-overlap data: Zener pinning couples Phi strongly to d.

    Microstructural coupling:
      Second-phase particles pin grain boundaries such that equilibrium grain
      size satisfies Zener's limit: d ~ r_p / Phi, i.e., Phi ~ c / d.
      This makes d^(-1/2) and Phi^(1/2) observationally collinear.
    """
    rng = np.random.RandomState(seed)
    d = np.logspace(-6, -4, n_points)
    T = np.full(n_points, 300.0)
    eps_dot = np.full(n_points, 1e-3)

    # Zener pinning relation: Phi = c / d + small perturbation
    c_zener = 2.5e-7
    Phi = np.clip(c_zener / d + rng.normal(0, 0.002, n_points), 0.005, 0.25)

    sigma_clean = sigma_0 + k_hp * (d ** -0.5) + a_precip * (Phi ** 0.5)
    noise = rng.normal(0, noise_sigma * sigma_0, n_points)
    sigma_obs = sigma_clean + noise

    features = {"d": d, "T": T, "Phi": Phi, "eps_dot": eps_dot}
    return features, sigma_obs


# ── L1 Constrained Grammar Candidate Fitting ─────────────────────────────

def fit_l1_materials_candidates(
    features: dict[str, np.ndarray],
    sigma_obs: np.ndarray,
    n_train: int = 30,
    noise_sigma: float = 0.02,
    data_origin: str = DataOrigin.SIMULATION,
) -> list[ScoredCandidate]:
    """Generate and fit candidate models from the restricted metallurgy grammar.

    Grammar Vocabulary:
      1:            Baseline yield strength (sigma_0)
      d^(-1/2):     Hall-Petch grain boundary strengthening
      Phi^(1/2):    Orowan precipitate / dislocation hardening
      d:            Naive linear grain size
      d^(-1):       Dislocation cell size / subgrain strengthening
      log(T/T_0):   Thermal softening
      log(eps_dot): Strain rate sensitivity

    All candidates fitted via OLS on train split; scored on held-out split.
    """
    d = features["d"]
    T = features["T"]
    Phi = features["Phi"]
    eps_dot = features["eps_dot"]

    d_tr, d_ho = d[:n_train], d[n_train:]
    T_tr, T_ho = T[:n_train], T[n_train:]
    Phi_tr, Phi_ho = Phi[:n_train], Phi[n_train:]
    eps_tr, eps_ho = eps_dot[:n_train], eps_dot[n_train:]
    y_tr, y_ho = sigma_obs[:n_train], sigma_obs[n_train:]

    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))
    noise_est = float(noise_sigma * 150.0)

    # Precompute basis columns
    ones_tr, ones_ho = np.ones_like(d_tr), np.ones_like(d_ho)
    hp_tr, hp_ho = d_tr ** -0.5, d_ho ** -0.5
    precip_tr, precip_ho = Phi_tr ** 0.5, Phi_ho ** 0.5
    d_lin_tr, d_lin_ho = d_tr, d_ho
    d_inv_tr, d_inv_ho = d_tr ** -1.0, d_ho ** -1.0
    therm_tr, therm_ho = np.log(np.maximum(T_tr / 300.0, 1e-4)), np.log(np.maximum(T_ho / 300.0, 1e-4))
    rate_tr, rate_ho = np.log(np.maximum(eps_tr / 1e-3, 1e-4)), np.log(np.maximum(eps_ho / 1e-3, 1e-4))

    # Candidate definitions: (candidate_id, lhs, terms, X_tr, X_ho)
    definitions = [
        (
            "hp_single",
            (Term("one", "sigma_0"), Term("d_inv_half", "k_hp")),
            np.column_stack([ones_tr, hp_tr]),
            np.column_stack([ones_ho, hp_ho]),
        ),
        (
            "precip_single",
            (Term("one", "sigma_0"), Term("phi_half", "a_precip")),
            np.column_stack([ones_tr, precip_tr]),
            np.column_stack([ones_ho, precip_ho]),
        ),
        (
            "dual_hp_precip",
            (Term("one", "sigma_0"), Term("d_inv_half", "k_hp"), Term("phi_half", "a_precip")),
            np.column_stack([ones_tr, hp_tr, precip_tr]),
            np.column_stack([ones_ho, hp_ho, precip_ho]),
        ),
        (
            "hp_thermal",
            (Term("one", "sigma_0"), Term("d_inv_half", "k_hp"), Term("log_temp", "b_therm")),
            np.column_stack([ones_tr, hp_tr, therm_tr]),
            np.column_stack([ones_ho, hp_ho, therm_ho]),
        ),
        (
            "hp_strain_rate",
            (Term("one", "sigma_0"), Term("d_inv_half", "k_hp"), Term("log_rate", "c_rate")),
            np.column_stack([ones_tr, hp_tr, rate_tr]),
            np.column_stack([ones_ho, hp_ho, rate_ho]),
        ),
        (
            "linear_grain",
            (Term("one", "sigma_0"), Term("d_linear", "k_lin")),
            np.column_stack([ones_tr, d_lin_tr]),
            np.column_stack([ones_ho, d_lin_ho]),
        ),
        (
            "subgrain_inv",
            (Term("one", "sigma_0"), Term("d_inv", "k_inv")),
            np.column_stack([ones_tr, d_inv_tr]),
            np.column_stack([ones_ho, d_inv_ho]),
        ),
        (
            "constant_baseline",
            (Term("one", "sigma_0"),),
            np.column_stack([ones_tr]),
            np.column_stack([ones_ho]),
        ),
    ]

    candidates: list[ScoredCandidate] = []
    for cand_id, terms, X_tr, X_ho in definitions:
        w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        pred_tr = X_tr @ w
        loss_tr = float(np.mean((y_tr - pred_tr) ** 2))

        pred_ho = X_ho @ w
        resid_ho = y_ho - pred_ho
        rmse_ho = float(np.sqrt(np.mean(resid_ho ** 2)))

        c = CandidateDynamics(lhs="yield_strength", terms=terms, candidate_id=cand_id)
        candidates.append(ScoredCandidate(
            c,
            provenance={
                "data_origin": data_origin,
                "benchmark_level": BenchmarkLevel.L1_CONSTRAINED_DISCOVERY,
                "gate4_status": {
                    "passed": True,
                    "holdout_obs_rms": obs_rms,
                    "holdout_noise_sigma": noise_est,
                    "holdout_n_samples": len(y_ho),
                    "rmse": rmse_ho,
                    "holdout_residuals": resid_ho.tolist(),
                    "holdout_sq_error": (resid_ho ** 2).tolist(),
                },
            },
            train_loss=loss_tr,
            holdout_rmse=rmse_ho,
        ))

    return candidates


# ── Benchmark Discovery Runners ──────────────────────────────────────────

def run_l1_materials_benchmark(
    benchmark_id: str = "M1",
    noise_sigma: float = 0.02,
    n_points: int = 60,
    n_train: int = 30,
    candidate_pool: str = "auto",
    candidate_ids: list[str] | None = None,
    bootstrap: bool = True,
    data_origin: str = DataOrigin.SIMULATION,
    seed: int = 42,
) -> tuple[Adjudication, list[ScoredCandidate]]:
    """Execute one of the L1 metallurgy discovery benchmarks (M1, M2, or M3).

    candidate_pool='auto':
      M1: Compares single-mechanism hypotheses (Hall-Petch vs linear grain vs inverse grain vs precipitate vs constant).
      M2: Compares multi-mechanism hypotheses (dual grain+precipitate vs single grain vs single precipitate vs thermal vs constant).
      M3: Compares mechanism overlap where Zener pinning makes d^(-1/2) and Phi^(1/2) collinear.
    """
    if benchmark_id.upper() == "M1":
        features, sigma_obs = generate_m1_data(
            n_points=n_points, noise_sigma=noise_sigma, seed=seed
        )
    elif benchmark_id.upper() == "M2":
        features, sigma_obs = generate_m2_data(
            n_points=n_points, noise_sigma=noise_sigma, seed=seed
        )
    elif benchmark_id.upper() == "M3":
        features, sigma_obs = generate_m3_data(
            n_points=n_points, noise_sigma=noise_sigma, seed=seed
        )
    else:
        raise ValueError(f"Unknown benchmark_id: {benchmark_id}. Must be 'M1', 'M2', or 'M3'.")

    candidates = fit_l1_materials_candidates(
        features=features,
        sigma_obs=sigma_obs,
        n_train=n_train,
        noise_sigma=noise_sigma,
        data_origin=data_origin,
    )

    if candidate_ids is not None:
        candidates = [c for c in candidates if c.candidate.candidate_id in candidate_ids]
    elif candidate_pool == "auto":
        if benchmark_id.upper() == "M1":
            active_ids = {"hp_single", "precip_single", "linear_grain", "subgrain_inv", "constant_baseline"}
        elif benchmark_id.upper() == "M2":
            active_ids = {"dual_hp_precip", "hp_single", "precip_single", "hp_thermal", "constant_baseline"}
        elif benchmark_id.upper() == "M3":
            active_ids = {"dual_hp_precip", "hp_single", "precip_single", "constant_baseline"}
        else:
            active_ids = None
        if active_ids:
            candidates = [c for c in candidates if c.candidate.candidate_id in active_ids]

    ranked = sorted(
        [s for s in candidates if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: s.train_loss,
    )
    r_best = None
    r_rival = None
    if len(ranked) >= 2:
        r_best = ranked[0].provenance.get("gate4_status", {}).get("holdout_residuals")
        r_rival = ranked[1].provenance.get("gate4_status", {}).get("holdout_residuals")

    adj = adjudicate(
        candidates,
        resid_best=r_best,
        resid_rival=r_rival,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        bootstrap=bootstrap,
        bootstrap_n_boot=1000,
        bootstrap_seed=seed,
        data_origin=data_origin,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY,
    )
    return adj, candidates
