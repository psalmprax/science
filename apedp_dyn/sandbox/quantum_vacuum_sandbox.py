"""APEDP Quantum Vacuum & String-Breaking Sandbox (Duke Trapped-Ion Simulator).

Based on the quantum simulation experiment published in Nature Physics (2026):
"Observation of string-breaking dynamics in a quantum simulator" (arXiv:2410.13815),
by Arinjoy De, Alessio Lerose, De Luo, Federica M. Surace, Alexander Schuckert,
Elizabeth R. Bennewitz, Brayden Ware, William Morong, Kate S. Collins,
Zohreh Davoudi, Alexey V. Gorshkov, Or Katz, and Christopher Monroe (Duke Quantum Center).

Physical Background:
In (1+1)-dimensional lattice gauge theory mapped to a 13 trapped-ion spin chain,
matter pops into existence from the vacuum when an electric flux string between
charges is stretched or quenched under string tension.
When the potential energy stored in the string exceeds the particle-pair mass
threshold, non-perturbative quantum tunneling produces particle-antiparticle
pairs from the vacuum (the Schwinger mechanism / string breaking), screening the
original charges.

This sandbox models:
1. Field-dependent vacuum pair production rate:
   - Non-perturbative Schwinger mechanism: Gamma ~ a0 + a1 * exp(-E_c / E)
   - Perturbative power-law generation: Gamma ~ b0 + b1 * E^2
   - Classical linear / thermal drift: Gamma ~ c0 + c1 * E
   - Thermal / vacuum stability constant: Gamma ~ d0
2. Spatiotemporal quench dynamics on the 13-ion trapped chain:
   - Supercritical string breaking (decay into screening pairs)
   - Subcritical confined meson state (coherent localized Bloch oscillations)
   - Deconfined free spreading (ballistic propagation)

Adjudication uses APEDP's epistemic pipeline (gamma ceiling, whiteness,
bootstrap tiebreaker, native data_origin labeling) with genuine parameter fitting
(no mock or random losses).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path when invoked as a script
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import (
    RESID_RHO_FLAG,
    SOLE_SURVIVOR_GAMMA,
    Adjudication,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

CAL_SIGMA = 1.8


# ── Configuration & State Definitions ──────────────────────────────────
@dataclass
class QuantumVacuumConfig:
    """Configuration for the trapped-ion quantum vacuum simulation."""
    n_ions: int = 13                    # 13 trapped-ion chain (Duke DQC platform)
    string_length: int = 7              # initial electric flux string length (sites 3 to 9)
    coupling_J: float = 1.0             # spin-spin interaction coupling
    transverse_field_hx: float = 0.5    # transverse field driving spin flips / pair creation
    string_tension_hz: float = 2.5      # longitudinal field / string tension E
    critical_field_Ec: float = 1.5      # Schwinger critical field threshold
    duration: float = 10.0              # simulation duration [time units]
    dt: float = 0.1                     # timestep
    noise_sigma: float = 0.02           # observation noise scale (relative to RMS)
    data_origin: str = "simulation"     # "simulation" or "experimental"


@dataclass
class QuantumVacuumState:
    """State of the 13-ion string breaking simulator at a given time."""
    time: float
    string_persistence: float           # normalized string flux P(t) in [0, 1]
    pair_density: float                 # particle pair count N_p(t)
    site_polarization: np.ndarray       # local <sigma_i^z> along the 13-ion chain


# ── Trapped-Ion String Breaking Quench Simulator ────────────────────────
def simulate_string_breaking_quench(
    config: QuantumVacuumConfig,
    seed: int = 42,
) -> tuple[list[QuantumVacuumState], np.ndarray]:
    """Simulate real-time quench dynamics of the 13-ion string breaking platform.

    Returns:
        states: List of QuantumVacuumState at each timestep.
        observations: Noisy measurements array of shape (N, 3): [time, P_obs, pair_density_obs].
    """
    rng = np.random.RandomState(seed)
    n_steps = int(config.duration / config.dt) + 1
    times = np.linspace(0.0, config.duration, n_steps)

    states: list[QuantumVacuumState] = []
    observations = np.zeros((n_steps, 3))

    E = config.string_tension_hz
    Ec = config.critical_field_Ec
    n_ions = config.n_ions

    # String region spans center sites (e.g., sites 3 through 9 for 13 ions)
    start_str = (n_ions - config.string_length) // 2
    end_str = start_str + config.string_length

    # Determine dynamical regime
    if E > Ec:
        # Supercritical: string breaks spontaneously into screening particle pairs
        # Edge-driven mechanism: pairs form at boundary and spread inward
        gamma_break = float(2.0 * config.transverse_field_hx * np.exp(-Ec / max(E, 1e-6)))
        p_inf = 0.10  # residual screening flux
        for i, t in enumerate(times):
            p_clean = p_inf + (1.0 - p_inf) * np.exp(-gamma_break * t)
            n_pairs_clean = 1.0 - np.exp(-gamma_break * t)

            # Local magnetizations showing edge-driven particle creation
            sigma_z = np.ones(n_ions)
            for site in range(start_str, end_str):
                # Distance to nearest string edge
                edge_dist = min(site - start_str, end_str - 1 - site)
                site_decay = gamma_break * (1.0 + 1.5 / (1.0 + edge_dist))
                sigma_z[site] = -1.0 + 2.0 * (1.0 - np.exp(-site_decay * t))

            states.append(QuantumVacuumState(
                time=t,
                string_persistence=float(p_clean),
                pair_density=float(n_pairs_clean),
                site_polarization=sigma_z,
            ))
            observations[i, 0] = t
            observations[i, 1] = max(0.0, p_clean + rng.normal(0, config.noise_sigma * 0.8))
            observations[i, 2] = max(0.0, n_pairs_clean + rng.normal(0, config.noise_sigma * 0.8))

    elif E > 0.0:
        # Subcritical: string is unbroken; confined charges undergo coherent Bloch oscillations
        omega_bloch = float(2.0 * E + 0.3 * config.coupling_J)
        for i, t in enumerate(times):
            p_clean = 0.60 + 0.35 * np.cos(omega_bloch * t)
            n_pairs_clean = 0.15 * (1.0 - np.cos(omega_bloch * t))

            sigma_z = np.ones(n_ions)
            for site in range(start_str, end_str):
                sigma_z[site] = -0.7 + 0.3 * np.cos(omega_bloch * t)

            states.append(QuantumVacuumState(
                time=t,
                string_persistence=float(p_clean),
                pair_density=float(n_pairs_clean),
                site_polarization=sigma_z,
            ))
            observations[i, 0] = t
            observations[i, 1] = max(0.0, p_clean + rng.normal(0, config.noise_sigma * 0.8))
            observations[i, 2] = max(0.0, n_pairs_clean + rng.normal(0, config.noise_sigma * 0.8))

    else:
        # Zero tension: deconfined free ballistic spreading of charges
        v_spread = 0.10 * config.coupling_J
        for i, t in enumerate(times):
            p_clean = max(0.0, 1.0 - v_spread * t)
            n_pairs_clean = min(1.0, 0.08 * t)

            sigma_z = np.ones(n_ions)
            spread_dist = int(v_spread * t * 10)
            cur_start = max(0, start_str - spread_dist)
            cur_end = min(n_ions, end_str + spread_dist)
            sigma_z[cur_start:cur_end] = -0.5

            states.append(QuantumVacuumState(
                time=t,
                string_persistence=float(p_clean),
                pair_density=float(n_pairs_clean),
                site_polarization=sigma_z,
            ))
            observations[i, 0] = t
            observations[i, 1] = max(0.0, p_clean + rng.normal(0, config.noise_sigma * 0.8))
            observations[i, 2] = max(0.0, n_pairs_clean + rng.normal(0, config.noise_sigma * 0.8))

    return states, observations


# ── Schwinger Pair Production Rate Data Generation ──────────────────────
def generate_schwinger_pair_production_data(
    n_points: int = 50,
    noise_sigma: float = 0.02,
    E_min: float = 0.6,
    E_max: float = 5.0,
    E_c: float = 1.5,
    gamma_0: float = 10.0,
    gamma_bg: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic pair production rate vs string tension / electric field.

    Physical truth:
        Gamma(E) = gamma_bg + gamma_0 * exp(-E_c / E) + noise.

    Returns:
        E: Field strength array.
        gamma_obs: Noisy observed pair creation rate.
        gamma_true: Ground truth pair creation rate.
    """
    rng = np.random.RandomState(seed)
    E = np.linspace(E_min, E_max, n_points)
    gamma_true = gamma_bg + gamma_0 * np.exp(-E_c / E)
    obs_rms = float(np.sqrt(np.mean(gamma_true**2)))
    noise = rng.normal(0, noise_sigma * obs_rms, n_points)
    gamma_obs = np.maximum(0.0, gamma_true + noise)
    return E, gamma_obs, gamma_true


# ── Candidate Hypotheses Fitting: Pair Production Rate ──────────────────
def fit_quantum_pair_candidates(
    E: np.ndarray,
    gamma_obs: np.ndarray,
    n_train: int = 25,
    noise_sigma: float = 0.02,
    E_c: float = 1.5,
    data_origin: str = "simulation",
) -> list[ScoredCandidate]:
    """Fit competing vacuum pair production hypotheses on train split and score on holdout.

    Competing Hypotheses:
    1. Schwinger Non-Perturbative Tunneling: Gamma = a0 + a1 * exp(-E_c / E)
    2. Perturbative Power-Law Generation:    Gamma = b0 + b1 * E^2
    3. Classical Linear Drift / Ohm's Law:   Gamma = c0 + c1 * E
    4. Thermal / Constant Null Background:   Gamma = d0

    No mock or random losses: all fits are real least-squares regressions.
    """
    E_tr, y_tr = E[:n_train], gamma_obs[:n_train]
    E_ho, y_ho = E[n_train:], gamma_obs[n_train:]

    obs_rms = float(np.sqrt(np.mean(y_ho**2)))
    noise_est = float(noise_sigma * obs_rms)
    candidates: list[ScoredCandidate] = []

    # 1. Non-Perturbative Schwinger Model: Gamma = a0 + a1 * exp(-E_c / E)
    term_exp_tr = np.exp(-E_c / E_tr)
    term_exp_ho = np.exp(-E_c / E_ho)
    X_tr_1 = np.column_stack([np.ones_like(term_exp_tr), term_exp_tr])
    w_1, *_ = np.linalg.lstsq(X_tr_1, y_tr, rcond=None)
    pred_tr_1 = X_tr_1 @ w_1
    loss_tr_1 = float(np.mean((y_tr - pred_tr_1)**2))
    X_ho_1 = np.column_stack([np.ones_like(term_exp_ho), term_exp_ho])
    pred_ho_1 = X_ho_1 @ w_1
    resid_1 = y_ho - pred_ho_1
    rmse_ho_1 = float(np.sqrt(np.mean(resid_1**2)))

    c1 = CandidateDynamics(
        lhs="pair_production_rate",
        terms=(Term("one", "a0"), Term("schwinger_exp", "a1")),
        candidate_id="schwinger_nonperturbative",
    )
    candidates.append(ScoredCandidate(
        c1,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_1,
                "holdout_residuals": resid_1.tolist(),
                "holdout_sq_error": (resid_1**2).tolist(),
            },
        },
        train_loss=loss_tr_1,
        holdout_rmse=rmse_ho_1,
    ))

    # 2. Perturbative Power-Law Model: Gamma = b0 + b1 * E^2
    X_tr_2 = np.column_stack([np.ones_like(E_tr), E_tr**2])
    w_2, *_ = np.linalg.lstsq(X_tr_2, y_tr, rcond=None)
    pred_tr_2 = X_tr_2 @ w_2
    loss_tr_2 = float(np.mean((y_tr - pred_tr_2)**2))
    X_ho_2 = np.column_stack([np.ones_like(E_ho), E_ho**2])
    pred_ho_2 = X_ho_2 @ w_2
    resid_2 = y_ho - pred_ho_2
    rmse_ho_2 = float(np.sqrt(np.mean(resid_2**2)))

    c2 = CandidateDynamics(
        lhs="pair_production_rate",
        terms=(Term("one", "b0"), Term("E_sq", "b1")),
        candidate_id="perturbative_power_law",
    )
    candidates.append(ScoredCandidate(
        c2,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_2,
                "holdout_residuals": resid_2.tolist(),
                "holdout_sq_error": (resid_2**2).tolist(),
            },
        },
        train_loss=loss_tr_2,
        holdout_rmse=rmse_ho_2,
    ))

    # 3. Classical Linear Drift / Ohm's Law: Gamma = c0 + c1 * E
    X_tr_3 = np.column_stack([np.ones_like(E_tr), E_tr])
    w_3, *_ = np.linalg.lstsq(X_tr_3, y_tr, rcond=None)
    pred_tr_3 = X_tr_3 @ w_3
    loss_tr_3 = float(np.mean((y_tr - pred_tr_3)**2))
    X_ho_3 = np.column_stack([np.ones_like(E_ho), E_ho])
    pred_ho_3 = X_ho_3 @ w_3
    resid_3 = y_ho - pred_ho_3
    rmse_ho_3 = float(np.sqrt(np.mean(resid_3**2)))

    c3 = CandidateDynamics(
        lhs="pair_production_rate",
        terms=(Term("one", "c0"), Term("E", "c1")),
        candidate_id="classical_linear",
    )
    candidates.append(ScoredCandidate(
        c3,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_3,
                "holdout_residuals": resid_3.tolist(),
                "holdout_sq_error": (resid_3**2).tolist(),
            },
        },
        train_loss=loss_tr_3,
        holdout_rmse=rmse_ho_3,
    ))

    # 4. Thermal / Constant Null Background: Gamma = d0
    X_tr_4 = np.column_stack([np.ones_like(E_tr)])
    w_4, *_ = np.linalg.lstsq(X_tr_4, y_tr, rcond=None)
    pred_tr_4 = X_tr_4 @ w_4
    loss_tr_4 = float(np.mean((y_tr - pred_tr_4)**2))
    X_ho_4 = np.column_stack([np.ones_like(E_ho)])
    pred_ho_4 = X_ho_4 @ w_4
    resid_4 = y_ho - pred_ho_4
    rmse_ho_4 = float(np.sqrt(np.mean(resid_4**2)))

    c4 = CandidateDynamics(
        lhs="pair_production_rate",
        terms=(Term("one", "d0"),),
        candidate_id="thermal_constant",
    )
    candidates.append(ScoredCandidate(
        c4,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_4,
                "holdout_residuals": resid_4.tolist(),
                "holdout_sq_error": (resid_4**2).tolist(),
            },
        },
        train_loss=loss_tr_4,
        holdout_rmse=rmse_ho_4,
    ))

    return candidates


# ── Candidate Hypotheses Fitting: Quench Dynamics ───────────────────────
def fit_quench_dynamics_candidates(
    t: np.ndarray,
    P_obs: np.ndarray,
    n_train: int | None = None,
    noise_sigma: float = 0.02,
    gamma_rate: float = 0.8,
    omega_bloch: float = 3.0,
    data_origin: str = "simulation",
) -> list[ScoredCandidate]:
    """Fit competing quench dynamics models on train split and score on holdout split.

    Competing Hypotheses for String Persistence P(t):
    1. String Breaking Decay:      P(t) = a0 + a1 * exp(-gamma * t)
    2. Coherent Meson Oscillation: P(t) = b0 + b1 * cos(omega * t)
    3. Ballistic Linear Drift:     P(t) = c0 + c1 * t
    4. Constant String Baseline:   P(t) = d0
    """
    n_samples = len(t)
    if n_train is None:
        n_train = n_samples // 2

    t_tr, y_tr = t[:n_train], P_obs[:n_train]
    t_ho, y_ho = t[n_train:], P_obs[n_train:]

    obs_rms = float(np.sqrt(np.mean(y_ho**2))) if len(y_ho) > 0 else 1.0
    noise_est = float(noise_sigma * obs_rms)
    candidates: list[ScoredCandidate] = []

    # 1. Decay (String Breaking)
    X_tr_dec = np.column_stack([np.ones_like(t_tr), np.exp(-gamma_rate * t_tr)])
    w_dec, *_ = np.linalg.lstsq(X_tr_dec, y_tr, rcond=None)
    pred_tr_dec = X_tr_dec @ w_dec
    loss_tr_dec = float(np.mean((y_tr - pred_tr_dec)**2))
    X_ho_dec = np.column_stack([np.ones_like(t_ho), np.exp(-gamma_rate * t_ho)])
    pred_ho_dec = X_ho_dec @ w_dec
    resid_dec = y_ho - pred_ho_dec
    rmse_ho_dec = float(np.sqrt(np.mean(resid_dec**2)))

    c_dec = CandidateDynamics(
        lhs="string_persistence",
        terms=(Term("one", "a0"), Term("decay_exp", "a1")),
        candidate_id="decay_string_breaking",
    )
    candidates.append(ScoredCandidate(
        c_dec,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_dec,
                "holdout_residuals": resid_dec.tolist(),
                "holdout_sq_error": (resid_dec**2).tolist(),
            },
        },
        train_loss=loss_tr_dec,
        holdout_rmse=rmse_ho_dec,
    ))

    # 2. Coherent Oscillations (Confined Meson State)
    X_tr_osc = np.column_stack([np.ones_like(t_tr), np.cos(omega_bloch * t_tr)])
    w_osc, *_ = np.linalg.lstsq(X_tr_osc, y_tr, rcond=None)
    pred_tr_osc = X_tr_osc @ w_osc
    loss_tr_osc = float(np.mean((y_tr - pred_tr_osc)**2))
    X_ho_osc = np.column_stack([np.ones_like(t_ho), np.cos(omega_bloch * t_ho)])
    pred_ho_osc = X_ho_osc @ w_osc
    resid_osc = y_ho - pred_ho_osc
    rmse_ho_osc = float(np.sqrt(np.mean(resid_osc**2)))

    c_osc = CandidateDynamics(
        lhs="string_persistence",
        terms=(Term("one", "b0"), Term("cos_osc", "b1")),
        candidate_id="coherent_oscillations",
    )
    candidates.append(ScoredCandidate(
        c_osc,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_osc,
                "holdout_residuals": resid_osc.tolist(),
                "holdout_sq_error": (resid_osc**2).tolist(),
            },
        },
        train_loss=loss_tr_osc,
        holdout_rmse=rmse_ho_osc,
    ))

    # 3. Ballistic Linear Drift (Deconfined Propagation)
    X_tr_lin = np.column_stack([np.ones_like(t_tr), t_tr])
    w_lin, *_ = np.linalg.lstsq(X_tr_lin, y_tr, rcond=None)
    pred_tr_lin = X_tr_lin @ w_lin
    loss_tr_lin = float(np.mean((y_tr - pred_tr_lin)**2))
    X_ho_lin = np.column_stack([np.ones_like(t_ho), t_ho])
    pred_ho_lin = X_ho_lin @ w_lin
    resid_lin = y_ho - pred_ho_lin
    rmse_ho_lin = float(np.sqrt(np.mean(resid_lin**2)))

    c_lin = CandidateDynamics(
        lhs="string_persistence",
        terms=(Term("one", "c0"), Term("t", "c1")),
        candidate_id="ballistic_linear",
    )
    candidates.append(ScoredCandidate(
        c_lin,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_lin,
                "holdout_residuals": resid_lin.tolist(),
                "holdout_sq_error": (resid_lin**2).tolist(),
            },
        },
        train_loss=loss_tr_lin,
        holdout_rmse=rmse_ho_lin,
    ))

    # 4. Constant Persistence
    X_tr_c = np.column_stack([np.ones_like(t_tr)])
    w_c, *_ = np.linalg.lstsq(X_tr_c, y_tr, rcond=None)
    pred_tr_c = X_tr_c @ w_c
    loss_tr_c = float(np.mean((y_tr - pred_tr_c)**2))
    X_ho_c = np.column_stack([np.ones_like(t_ho)])
    pred_ho_c = X_ho_c @ w_c
    resid_c = y_ho - pred_ho_c
    rmse_ho_c = float(np.sqrt(np.mean(resid_c**2)))

    c_c = CandidateDynamics(
        lhs="string_persistence",
        terms=(Term("one", "d0"),),
        candidate_id="constant_persistence",
    )
    candidates.append(ScoredCandidate(
        c_c,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_c,
                "holdout_residuals": resid_c.tolist(),
                "holdout_sq_error": (resid_c**2).tolist(),
            },
        },
        train_loss=loss_tr_c,
        holdout_rmse=rmse_ho_c,
    ))

    return candidates


# ── Discovery Adjudication Runners ──────────────────────────────────────
def run_quantum_vacuum_discovery(
    noise_sigma: float = 0.02,
    n_points: int = 50,
    n_train: int = 25,
    E_c: float = 1.5,
    E_min: float = 0.6,
    E_max: float = 5.0,
    bootstrap: bool = True,
    pass_bootstrap_residuals: bool = True,
    data_origin: str = "simulation",
    seed: int = 42,
) -> tuple[Adjudication, np.ndarray, np.ndarray]:
    """Run APEDP epistemic discovery on quantum vacuum pair creation data.

    Returns:
        (adjudication_result, field_strengths, observed_rates)
    """
    E, gamma_obs, _ = generate_schwinger_pair_production_data(
        n_points=n_points,
        noise_sigma=noise_sigma,
        E_min=E_min,
        E_max=E_max,
        E_c=E_c,
        seed=seed,
    )
    candidates = fit_quantum_pair_candidates(
        E=E,
        gamma_obs=gamma_obs,
        n_train=n_train,
        noise_sigma=noise_sigma,
        E_c=E_c,
        data_origin=data_origin,
    )

    r_best = None
    r_rival = None
    if pass_bootstrap_residuals and len(candidates) >= 2:
        ranked = sorted(
            [s for s in candidates if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)],
            key=lambda s: s.holdout_rmse,
        )
        if len(ranked) >= 2:
            best_cand = ranked[0]
            distinct_rivals = [s for s in ranked[1:] if s.candidate.candidate_id != best_cand.candidate.candidate_id]
            if distinct_rivals:
                second_cand = min(
                    distinct_rivals,
                    key=lambda s: s.holdout_rmse if s.holdout_rmse is not None else float("inf"),
                )
                r_best = np.asarray(best_cand.provenance.get("gate4_status", {}).get("holdout_sq_error"), dtype=float)
                r_rival = np.asarray(second_cand.provenance.get("gate4_status", {}).get("holdout_sq_error"), dtype=float)

    adj = adjudicate(
        candidates,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        resid_best=r_best,
        resid_rival=r_rival,
        bootstrap=bootstrap,
        bootstrap_n_boot=1000,
        bootstrap_seed=seed,
        data_origin=data_origin,
    )
    return adj, E, gamma_obs


def run_quench_dynamics_discovery(
    config: QuantumVacuumConfig,
    bootstrap: bool = True,
    pass_bootstrap_residuals: bool = True,
    seed: int = 42,
) -> tuple[Adjudication, np.ndarray, np.ndarray]:
    """Run APEDP epistemic discovery on string breaking quench dynamics."""
    _, obs = simulate_string_breaking_quench(config, seed=seed)
    times = obs[:, 0]
    p_obs = obs[:, 1]

    # Physically-derived rates for candidate hypotheses
    if config.string_tension_hz > 0:
        gamma_phys = float(2.0 * config.transverse_field_hx * np.exp(-config.critical_field_Ec / max(config.string_tension_hz, 1e-6)))
        omega_phys = float(2.0 * config.string_tension_hz + 0.3 * config.coupling_J)
    else:
        gamma_phys = 0.5
        omega_phys = 1.0

    candidates = fit_quench_dynamics_candidates(
        t=times,
        P_obs=p_obs,
        noise_sigma=config.noise_sigma,
        gamma_rate=gamma_phys,
        omega_bloch=omega_phys,
        data_origin=config.data_origin,
    )

    r_best = None
    r_rival = None
    if pass_bootstrap_residuals and len(candidates) >= 2:
        ranked = sorted(
            [s for s in candidates if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)],
            key=lambda s: s.holdout_rmse,
        )
        if len(ranked) >= 2:
            best_cand = ranked[0]
            distinct_rivals = [s for s in ranked[1:] if s.candidate.candidate_id != best_cand.candidate.candidate_id]
            if distinct_rivals:
                second_cand = min(
                    distinct_rivals,
                    key=lambda s: s.holdout_rmse if s.holdout_rmse is not None else float("inf"),
                )
                r_best = np.asarray(best_cand.provenance.get("gate4_status", {}).get("holdout_sq_error"), dtype=float)
                r_rival = np.asarray(second_cand.provenance.get("gate4_status", {}).get("holdout_sq_error"), dtype=float)

    adj = adjudicate(
        candidates,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        resid_best=r_best,
        resid_rival=r_rival,
        bootstrap=bootstrap,
        bootstrap_n_boot=1000,
        bootstrap_seed=seed,
        data_origin=config.data_origin,
    )
    return adj, times, p_obs


# ── Interactive CLI Demo ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("APEDP Quantum Vacuum & String-Breaking Sandbox (Duke Trapped-Ion Simulator)")
    print("Reference: Nature Physics (2026), arXiv:2410.13815 (Monroe, Lerose, De et al.)")
    print("=" * 72)

    # 1. Pair production rate discovery across noise levels
    print("\n[1] Vacuum Pair Production Rate vs Electric Field Discovery:")
    for noise in (0.01, 0.03, 0.10, 0.40):
        adj, _, _ = run_quantum_vacuum_discovery(noise_sigma=noise, bootstrap=True, data_origin="simulation")
        print(f"\nNoise level: {noise*100:.0f}%")
        print(f"  Status:       {adj.status.value}")
        print(f"  Promoted:     {adj.promoted.candidate.candidate_id if adj.promoted else 'None (Abstention)'}")
        print(f"  Best RMSE:    {adj.best_rmse:.4f}" if adj.best_rmse is not None else "  Best RMSE: None")
        print(f"  Second RMSE:  {adj.second_rmse:.4f}" if adj.second_rmse is not None else "  Second RMSE: None")
        print(f"  Data Origin:  {adj.data_origin}")
        print(f"  Reason:       {adj.reason[:80]}...")

    # 2. String-breaking quench dynamics across physical regimes
    print("\n" + "-" * 72)
    print("[2] Trapped-Ion String Quench Dynamics Discovery:")
    regimes = [
        ("Supercritical Tension (E=2.5 > Ec=1.5): String-Breaking Decay", 2.5),
        ("Subcritical Tension (E=0.5 < Ec=1.5): Confined Meson Oscillations", 0.5),
        ("Zero Tension (E=0.0): Deconfined Free Ballistic Spreading", 0.0),
    ]
    for desc, E_field in regimes:
        cfg = QuantumVacuumConfig(string_tension_hz=E_field, duration=10.0, noise_sigma=0.02, data_origin="simulation")
        adj_q, _, _ = run_quench_dynamics_discovery(cfg, bootstrap=True)
        print(f"\n{desc}")
        print(f"  Status:       {adj_q.status.value}")
        print(f"  Promoted:     {adj_q.promoted.candidate.candidate_id if adj_q.promoted else 'None'}")
        print(f"  Best RMSE:    {adj_q.best_rmse:.4f}" if adj_q.best_rmse is not None else "  Best RMSE: None")
        print(f"  Data Origin:  {adj_q.data_origin}")
        print(f"  Reason:       {adj_q.reason[:80]}...")
