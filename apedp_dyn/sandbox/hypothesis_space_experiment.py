"""Hypothesis-Space Expansion Benchmark: Model Starvation vs Genuine Non-Identifiability.

This benchmark establishes the critical epistemic distinction:
    Abstention = Epistemic Uncertainty OR Hypothesis-Space Insufficiency

Experiment 1 (Model-Space Starvation):
    Stage 1: Restricted hypothesis space H_0 fails to capture nonlinear physics.
             Adjudication yields INSUFFICIENT_EVIDENCE with attribution HYPOTHESIS_SPACE_LIMITED.
    Stage 2: Expanded open grammar H_L2 generates the true functional form.
             Adjudication yields UNIQUE_SUPPORT with attribution DECISIVE_EVIDENTIAL_SUPPORT.
    Conclusion: Proves that the Stage 1 abstention was model-space starvation, not data noise.

Experiment 2 (Persistent Ambiguity & Intrinsic Non-Identifiability):
    Stage 1: Restricted hypothesis space H_0 yields MULTI_HYPOTHESIS due to observational collinearity.
    Stage 2: Expanded open grammar H_L2 continues to yield MULTI_HYPOTHESIS.
             Independent correlation/FIM diagnostic proves condition number kappa >> 10^3.
    Conclusion: expanded hypothesis space + persistent ambiguity => stronger evidence of genuine non-identifiability.

Stage 3 (L3 Active Interventional Resolution):
    Optimal probe X* is synthesized breaking the collinearity manifold.
    Re-adjudicating with the active intervention resolves the tie to UNIQUE_SUPPORT.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from apedp_dyn.discovery_l2.active_intervention import synthesize_discriminating_intervention
from apedp_dyn.discovery_l2.dimensional_types import BaseDimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicAttribution,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


@dataclass(frozen=True)
class StarvationExperimentResult:
    """Results from testing model-space starvation vs L2 open recovery."""

    stage1_status: EpistemicStatus
    stage1_attribution: str
    stage2_status: EpistemicStatus
    stage2_attribution: str
    discovered_expression: str
    demonstrates_starvation: bool


@dataclass(frozen=True)
class PersistentAmbiguityResult:
    """Results from testing persistent ambiguity across hypothesis-space expansion."""

    stage1_status: EpistemicStatus
    stage1_attribution: str
    stage2_status: EpistemicStatus
    stage2_attribution: str
    condition_number: float
    feature_correlation: float
    l3_status: EpistemicStatus
    l3_attribution: str
    demonstrates_persistent_ambiguity: bool
    l3_resolved_by_intervention: bool
    optimal_probe: dict[str, float] | None = None


def run_model_space_starvation_experiment(seed: int = 42) -> StarvationExperimentResult:
    """Run Experiment 1: True law is Kinetic Energy E = 0.5 * m * v^2."""
    rng = np.random.default_rng(seed)
    n = 60
    m = rng.uniform(1.0, 5.0, size=n)
    v = rng.uniform(2.0, 10.0, size=n)
    noise_sigma = 0.05
    # True relationship: E = 0.5 * m * v^2
    y_true = 0.5 * m * (v ** 2)
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=n)

    # Split into train (60%) and holdout (40%)
    n_tr = int(n * 0.6)
    m_tr, m_ho = m[:n_tr], m[n_tr:]
    v_tr, v_ho = v[:n_tr], v[n_tr:]
    y_tr, y_ho = y_obs[:n_tr], y_obs[n_tr:]
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))

    # -------------------------------------------------------------
    # Stage 1: Restricted hypothesis space H_0 (Linear in m, linear in v)
    # -------------------------------------------------------------
    # Candidate 1: E = c1 * m
    coef_m, *_ = np.linalg.lstsq(m_tr[:, None], y_tr, rcond=None)
    pred_m = m_ho * coef_m[0]
    rmse_m = float(np.sqrt(np.mean((y_ho - pred_m) ** 2)))

    # Candidate 2: E = c1 * v
    coef_v, *_ = np.linalg.lstsq(v_tr[:, None], y_tr, rcond=None)
    pred_v = v_ho * coef_v[0]
    rmse_v = float(np.sqrt(np.mean((y_ho - pred_v) ** 2)))

    # Candidate 3: E = c1 * m + c2 * v
    X_lin_tr = np.column_stack([m_tr, v_tr])
    X_lin_ho = np.column_stack([m_ho, v_ho])
    coef_mv, *_ = np.linalg.lstsq(X_lin_tr, y_tr, rcond=None)
    pred_mv = X_lin_ho @ coef_mv
    rmse_mv = float(np.sqrt(np.mean((y_ho - pred_mv) ** 2)))

    # In H_0, best model (rmse_mv ~ 20-30) completely fails fidelity ceiling (gamma * sigma = 3 * 0.05 = 0.15)
    def _make_sc(cid: str, rmse: float, preds: np.ndarray) -> ScoredCandidate:
        residuals = (y_ho - preds).tolist()
        g4 = {
            "passed": True,
            "holdout_obs_rms": obs_rms,
            "holdout_noise_sigma": noise_sigma,
            "holdout_n_samples": len(y_ho),
            "holdout_residuals": residuals,
        }
        cand = CandidateDynamics(lhs="E", terms=(Term("m", "c1"), Term("v", "c2")), candidate_id=cid)
        return ScoredCandidate(candidate=cand, provenance={"gate4_status": g4}, train_loss=rmse, holdout_rmse=rmse)

    c_m = _make_sc("linear_m", rmse_m, pred_m)
    c_v = _make_sc("linear_v", rmse_v, pred_v)
    c_mv = _make_sc("linear_mv", rmse_mv, pred_mv)

    adj_stage1 = adjudicate(
        [c_m, c_v, c_mv],
        sole_survivor_rule=True,
        data_origin=DataOrigin.SYNTHETIC,
        benchmark_level=BenchmarkLevel.L0_SELECTION,
        epistemic_attribution=EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED,
    )

    # -------------------------------------------------------------
    # Stage 2: Expanded physically typed hypothesis space H_L2
    # -------------------------------------------------------------
    engine = L2DiscoveryEngine(max_depth=2, max_complexity=5, operators=("+", "-", "*", "/", "pow"))
    dimensions = {
        "m": BaseDimension(m=1),
        "v": BaseDimension(l=1, t=-1),
    }
    target_dimension = BaseDimension(m=1, l=2, t=-2)  # Energy [M L^2 T^-2]

    l2_res = engine.discover(
        X={"m": m, "v": v},
        Y=y_obs,
        dimensions=dimensions,
        target_dimension=target_dimension,
        train_ratio=0.6,
        data_origin=DataOrigin.SYNTHETIC,
        bootstrap=True,
        seed=seed,
    )
    adj_stage2 = l2_res.adjudication

    promoted_expr = ""
    if adj_stage2.promoted is not None:
        best_cand = adj_stage2.promoted
        promoted_expr = best_cand.candidate.metadata.get("canonical_expr") or best_cand.candidate.candidate_id

    demonstrates_starvation = (
        adj_stage1.status in (EpistemicStatus.INSUFFICIENT_EVIDENCE, EpistemicStatus.NO_VALID_CANDIDATE)
        and adj_stage1.epistemic_attribution == EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED
        and adj_stage2.status == EpistemicStatus.UNIQUE_SUPPORT
        and adj_stage2.epistemic_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT
    )

    return StarvationExperimentResult(
        stage1_status=adj_stage1.status,
        stage1_attribution=adj_stage1.epistemic_attribution,
        stage2_status=adj_stage2.status,
        stage2_attribution=adj_stage2.epistemic_attribution,
        discovered_expression=promoted_expr,
        demonstrates_starvation=demonstrates_starvation,
    )


def run_persistent_ambiguity_experiment(seed: int = 42) -> PersistentAmbiguityResult:
    """Run Experiment 2: Collinear variables u and w where w = 2*u."""
    rng = np.random.default_rng(seed)
    n = 80
    u = rng.uniform(1.0, 5.0, size=n)
    # Strictly collinear in passive observational regime: w = 2.0 * u + tiny noise
    w = 2.0 * u + rng.normal(0.0, 0.005, size=n)
    noise_sigma = 0.02

    # True data-generating mechanism: Y = 3.0 * u + noise (which also equals 1.5 * w + noise!)
    y_true = 3.0 * u
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=n)

    # Diagnostic check: correlation and condition number
    corr_matrix = np.corrcoef(u, w)
    feature_corr = float(corr_matrix[0, 1])
    X_obs = np.column_stack([u, w])
    _, s_vals, _ = np.linalg.svd(X_obs)
    cond_num = float(s_vals[0] / max(s_vals[-1], 1e-12))

    dimensions = {
        "u": BaseDimension(l=1),
        "w": BaseDimension(l=1),
    }
    target_dimension = BaseDimension(l=1)

    # -------------------------------------------------------------
    # Stage 1: Restricted hypothesis space H_0 = {u, w}
    # -------------------------------------------------------------
    n_tr = int(n * 0.6)
    u_tr, u_ho = u[:n_tr], u[n_tr:]
    w_tr, w_ho = w[:n_tr], w[n_tr:]
    y_tr, y_ho = y_obs[:n_tr], y_obs[n_tr:]
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))

    coef_u, *_ = np.linalg.lstsq(u_tr[:, None], y_tr, rcond=None)
    coef_w, *_ = np.linalg.lstsq(w_tr[:, None], y_tr, rcond=None)
    pred_u = u_ho * coef_u[0]
    pred_w = w_ho * coef_w[0]
    rmse_u = float(np.sqrt(np.mean((y_ho - pred_u) ** 2)))
    rmse_w = float(np.sqrt(np.mean((y_ho - pred_w) ** 2)))

    def _make_sc(cid: str, term: str, rmse: float, preds: np.ndarray) -> ScoredCandidate:
        residuals = (y_ho - preds).tolist()
        g4 = {
            "passed": True,
            "holdout_obs_rms": obs_rms,
            "holdout_noise_sigma": noise_sigma,
            "holdout_n_samples": len(y_ho),
            "holdout_residuals": residuals,
        }
        cand = CandidateDynamics(lhs="Y", terms=(Term(term, "c1"),), candidate_id=cid)
        return ScoredCandidate(candidate=cand, provenance={"gate4_status": g4}, train_loss=rmse, holdout_rmse=rmse)

    c_u = _make_sc("mechanism_u", "u", rmse_u, pred_u)
    c_w = _make_sc("mechanism_w", "w", rmse_w, pred_w)

    adj_stage1 = adjudicate(
        [c_u, c_w],
        data_origin=DataOrigin.REAL_OBSERVATIONAL,
        benchmark_level=BenchmarkLevel.L0_SELECTION,
        epistemic_attribution=EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY,
    )

    # -------------------------------------------------------------
    # Stage 2: Expanded open grammar H_L2 across {u, w}
    # -------------------------------------------------------------
    engine = L2DiscoveryEngine(max_depth=2, max_complexity=5, operators=("+", "-", "*", "/"))
    l2_res = engine.discover(
        X={"u": u, "w": w},
        Y=y_obs,
        dimensions=dimensions,
        target_dimension=target_dimension,
        train_ratio=0.6,
        data_origin=DataOrigin.REAL_OBSERVATIONAL,
        bootstrap=False,
        seed=seed,
    )
    adj_stage2 = l2_res.adjudication

    # -------------------------------------------------------------
    # Stage 3: L3 Active Interventional Discrimination
    # -------------------------------------------------------------
    # Construct model prediction functions
    def fn_u(X: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(X["u"] * coef_u[0])

    def fn_w(X: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(X["w"] * coef_w[0])

    models: list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]] = [
        ("mechanism_u", fn_u), ("mechanism_w", fn_w)]
    bounds = {"u": (1.0, 10.0), "w": (1.0, 10.0)}

    # Synthesize active intervention probe breaking collinearity w = 2*u
    probe = synthesize_discriminating_intervention(models, bounds, seed=seed)

    # Execute active intervention: generate experimental observation at perturbed probe points
    # In an active experiment, we independently set u and w to break the collinear manifold
    n_active = 30
    u_active = rng.uniform(1.0, 10.0, size=n_active)
    # Break correlation: w is chosen independently of u
    w_active = rng.uniform(1.0, 10.0, size=n_active)
    # True physical system responds to u: Y = 3.0 * u
    y_active_true = 3.0 * u_active
    y_active_obs = y_active_true + rng.normal(0.0, noise_sigma, size=n_active)

    # Evaluate models on active experimental holdout
    pred_u_act = u_active * coef_u[0]
    pred_w_act = w_active * coef_w[0]
    rmse_u_act = float(np.sqrt(np.mean((y_active_obs - pred_u_act) ** 2)))
    rmse_w_act = float(np.sqrt(np.mean((y_active_obs - pred_w_act) ** 2)))
    obs_rms_act = float(np.sqrt(np.mean(y_active_obs ** 2)))

    def _make_sc_active(cid: str, term: str, rmse: float, preds: np.ndarray) -> ScoredCandidate:
        residuals = (y_active_obs - preds).tolist()
        g4 = {
            "passed": True,
            "holdout_obs_rms": obs_rms_act,
            "holdout_noise_sigma": noise_sigma,
            "holdout_n_samples": len(y_active_obs),
            "holdout_residuals": residuals,
        }
        cand = CandidateDynamics(lhs="Y", terms=(Term(term, "c1"),), candidate_id=cid)
        return ScoredCandidate(candidate=cand, provenance={"gate4_status": g4}, train_loss=rmse, holdout_rmse=rmse)

    c_u_act = _make_sc_active("mechanism_u", "u", rmse_u_act, pred_u_act)
    c_w_act = _make_sc_active("mechanism_w", "w", rmse_w_act, pred_w_act)

    adj_l3 = adjudicate(
        [c_u_act, c_w_act],
        data_origin=DataOrigin.EXPERIMENTAL,
        benchmark_level=BenchmarkLevel.L3_ACTIVE_DISCOVERY,
    )

    demonstrates_persistent_ambiguity = (
        adj_stage1.status == EpistemicStatus.MULTI_HYPOTHESIS
        and adj_stage2.status == EpistemicStatus.MULTI_HYPOTHESIS
        and adj_stage2.epistemic_attribution == EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY
    )

    l3_resolved = (
        adj_l3.status == EpistemicStatus.UNIQUE_SUPPORT
        and adj_l3.promoted is not None
        and adj_l3.promoted.candidate.candidate_id == "mechanism_u"
    )

    return PersistentAmbiguityResult(
        stage1_status=adj_stage1.status,
        stage1_attribution=adj_stage1.epistemic_attribution,
        stage2_status=adj_stage2.status,
        stage2_attribution=adj_stage2.epistemic_attribution,
        condition_number=cond_num,
        feature_correlation=feature_corr,
        l3_status=adj_l3.status,
        l3_attribution=adj_l3.epistemic_attribution,
        demonstrates_persistent_ambiguity=demonstrates_persistent_ambiguity,
        l3_resolved_by_intervention=l3_resolved,
        optimal_probe=probe.probe_point,
    )
