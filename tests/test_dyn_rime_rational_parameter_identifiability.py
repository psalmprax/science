"""Tests for RIME rational parameter identifiability, Padé saturation recovery, and decoupled reporting.

Validates the two-stage rational discovery architecture:
1. Exact parameter recovery & reconciliation on noiseless Padé data:
   y = (1000 * p^2) / (1 + 2000 * p^2)
   Reconciles physical scales (num=1000, den=2000) with envelope ratio (b/a = 2.0).
2. Noise robustness across 0.5%, 2%, and 5% observation noise.
3. Domain range sensitivity: proves parameter recovery deteriorates on weakly saturated intervals
   even when R^2 remains high (>0.99), validating identifiability flagging.
4. Negative controls:
   - Pure quadratic: b -> 0 (no false saturation promoted).
   - Exponential saturation: confirms high R^2 alone does not imply unique rational mechanism.
5. Decoupled 5-metric reporting: structure_match, parameter_recovery, prediction_score,
   extrapolation_score, and identifiability_status.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic import rime_repair_diagnosis as diagmod
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    generate_repairs,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype
from apedp_dyn.sandbox.qec_threshold import generate_qec_saturating_data


@pytest.fixture
def default_repair_config() -> RepairConfig:
    return RepairConfig(
        max_candidates_per_round=4,
        max_rounds=1,
        max_added_terms=1,
        min_validation_r2_gain=0.01,
        equivalence_tol=1e-9,
        operators=(RepairOperator.R5_PADE_SATURATION,),
    )


def test_pade_exact_recovery_and_parameter_reconciliation(default_repair_config):
    """Test 1: Exact recovery on noiseless Padé data and parameter reconciliation."""
    p, obs = generate_qec_saturating_data(n_points=200, noise_sigma=0.0, seed=42, distance=3, model="pade")
    df_tr = pd.DataFrame({"p": p[::2], "target": obs[::2]})
    df_ho = pd.DataFrame({"p": p[1::2], "target": obs[1::2]})

    k_fit = float(np.polyfit(df_tr["p"] ** 2, df_tr["target"], 1)[0])
    base = CandidateProposal(
        proposer_id="test_proposer",
        candidate_id="cand_clean_base",
        formula=f"{k_fit:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda df, k=k_fit: k * (df["p"].to_numpy() ** 2),
    )

    reps = generate_repairs(base, [diagmod.FINDING_CURVATURE_SUSPECT], df_tr, df_tr["target"].to_numpy(), default_repair_config)
    assert len(reps) > 0, "R5 must emit a repaired proposal"
    rep = reps[0]

    # Verify parameters
    a = rep.parameters["a"]
    b = rep.parameters["b"]
    num_scale = a * k_fit
    den_scale = b * k_fit
    ratio = rep.parameters["envelope_ratio"]

    # Reconciled targets:
    assert abs(num_scale - 1000.0) < 1e-3, f"Expected numerator scale 1000.0, got {num_scale}"
    assert abs(den_scale - 2000.0) < 1e-3, f"Expected denominator scale 2000.0, got {den_scale}"
    assert abs(ratio - 2.0) < 1e-4, f"Expected envelope ratio (b/a) = 2.0, got {ratio}"

    # Prediction and holdout R^2
    y_ho = df_ho["target"].to_numpy()
    pred_ho = rep.compute_score(df_ho)
    ho_r2 = 1.0 - float(np.sum((y_ho - pred_ho) ** 2)) / float(np.sum((y_ho - y_ho.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((y_ho - pred_ho) ** 2)))

    assert ho_r2 > 0.999999
    assert rmse < 1e-12
    assert rep.parameters["identifiability_status"] == "IDENTIFIED_STABLE"


def test_pade_noise_robustness_battery(default_repair_config):
    """Test 2: Parameter recovery error degrades gracefully with measurement noise."""
    tolerances = {
        0.005: {"max_param_err": 1.0, "min_ho_r2": 0.9998},
        0.02:  {"max_param_err": 3.0, "min_ho_r2": 0.9985},
        0.05:  {"max_param_err": 6.0, "min_ho_r2": 0.9920},
    }

    for noise, tol in tolerances.items():
        p, obs = generate_qec_saturating_data(n_points=200, noise_sigma=noise, seed=42, distance=3, model="pade")
        df_tr = pd.DataFrame({"p": p[::2], "target": obs[::2]})
        df_ho = pd.DataFrame({"p": p[1::2], "target": obs[1::2]})

        k_fit = float(np.polyfit(df_tr["p"] ** 2, df_tr["target"], 1)[0])
        base = CandidateProposal(
            proposer_id="test",
            candidate_id=f"cand_noise_{noise}",
            formula=f"{k_fit:.4e} * p^2",
            strategy_archetype=StrategyArchetype.TACTICAL,
            declared_scale_invariant=True,
            declared_nominal_dependencies=("p",),
            compute_score=lambda df, k=k_fit: k * (df["p"].to_numpy() ** 2),
        )

        reps = generate_repairs(base, [diagmod.FINDING_CURVATURE_SUSPECT], df_tr, df_tr["target"].to_numpy(), default_repair_config)
        assert len(reps) > 0
        rep = reps[0]

        num_scale = rep.parameters["a"] * k_fit
        den_scale = rep.parameters["b"] * k_fit
        param_err = max(abs(num_scale - 1000.0) / 1000.0, abs(den_scale - 2000.0) / 2000.0) * 100.0

        y_ho = df_ho["target"].to_numpy()
        pred_ho = rep.compute_score(df_ho)
        ho_r2 = 1.0 - float(np.sum((y_ho - pred_ho) ** 2)) / float(np.sum((y_ho - y_ho.mean()) ** 2))

        assert param_err < tol["max_param_err"], f"At {noise*100}% noise: param err {param_err:.2f}% exceeded {tol['max_param_err']}%"
        assert ho_r2 > tol["min_ho_r2"], f"At {noise*100}% noise: holdout R2 {ho_r2:.4f} below {tol['min_ho_r2']}"


def test_pade_domain_range_identifiability_breakdown(default_repair_config):
    """Test 3: Identifiability breaks down on weakly saturated intervals even when R^2 > 0.99."""
    # Narrow domain: p in [0.001, 0.008] (2000*p^2 only reaches 0.128, weak saturation)
    p_narrow, obs_narrow = generate_qec_saturating_data(
        n_points=200, noise_sigma=0.02, seed=42, distance=3, model="pade", p_min=0.001, p_max=0.008
    )
    df_tr = pd.DataFrame({"p": p_narrow[::2], "target": obs_narrow[::2]})
    df_ho = pd.DataFrame({"p": p_narrow[1::2], "target": obs_narrow[1::2]})

    k_fit = float(np.polyfit(df_tr["p"] ** 2, df_tr["target"], 1)[0])
    base = CandidateProposal(
        proposer_id="test",
        candidate_id="cand_narrow_dom",
        formula=f"{k_fit:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda df, k=k_fit: k * (df["p"].to_numpy() ** 2),
    )

    # Audit path: this case exists to OBSERVE an ill-conditioned fit, so the
    # emission refusal is explicitly opted out here. The default path (which
    # refuses to emit an unidentified denominator) is pinned in
    # test_dyn_rime_noise_present_battery.py.
    audit_cfg = RepairConfig(
        max_candidates_per_round=default_repair_config.max_candidates_per_round,
        max_rounds=default_repair_config.max_rounds,
        max_added_terms=default_repair_config.max_added_terms,
        min_validation_r2_gain=default_repair_config.min_validation_r2_gain,
        equivalence_tol=default_repair_config.equivalence_tol,
        operators=default_repair_config.operators,
        r5_emit_unidentified=True,
    )
    reps = generate_repairs(base, [diagmod.FINDING_CURVATURE_SUSPECT], df_tr, df_tr["target"].to_numpy(), audit_cfg)
    pade = [r for r in reps if r.metadata.get("operator") == "R5_PADE_SATURATION"]
    assert len(pade) > 0
    rep = pade[0]

    num_scale = rep.parameters["a"] * k_fit
    den_scale = rep.parameters["b"] * k_fit
    param_err = max(abs(num_scale - 1000.0) / 1000.0, abs(den_scale - 2000.0) / 2000.0) * 100.0

    y_ho = df_ho["target"].to_numpy()
    pred_ho = rep.compute_score(df_ho)
    ho_r2 = 1.0 - float(np.sum((y_ho - pred_ho) ** 2)) / float(np.sum((y_ho - y_ho.mean()) ** 2))

    # Crucial scientific check: Holdout R^2 is still > 0.998, but parameter error is > 5%
    assert ho_r2 > 0.998, f"Prediction remains strong: {ho_r2}"
    assert param_err > 5.0, f"Parameter recovery must deteriorate on narrow domain: {param_err:.2f}%"
    assert den_scale < 1900.0, f"Weak saturation under-estimates denominator scale: {den_scale}"
    assert rep.parameters["identifiability_status"] == "PREDICTIVE_ONLY_ILL_CONDITIONED"


def test_pade_negative_controls_no_false_rational_promotion(default_repair_config):
    """Test 4: Negative controls ensure non-rational ground truths do not falsely claim rational law."""
    # Control A: Pure Un-saturating Quadratic (y = 1000 * p^2)
    p = np.linspace(0.001, 0.025, 200)
    y_quad = 1000.0 * (p ** 2)
    df_quad_tr = pd.DataFrame({"p": p[::2], "target": y_quad[::2]})
    k_quad = float(np.polyfit(df_quad_tr["p"] ** 2, df_quad_tr["target"], 1)[0])

    base_quad = CandidateProposal(
        proposer_id="test",
        candidate_id="cand_pure_quad",
        formula=f"{k_quad:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda df, k=k_quad: k * (df["p"].to_numpy() ** 2),
    )

    reps_quad = generate_repairs(base_quad, [diagmod.FINDING_CURVATURE_SUSPECT], df_quad_tr, df_quad_tr["target"].to_numpy(), default_repair_config)
    pade_quad = [r for r in reps_quad if r.metadata.get("operator") == "R5_PADE_SATURATION"]
    # Strengthened control: the phantom denominator is not merely small, it is
    # NOT EMITTED under the default policy (measured b*k = 62 at 2% noise would
    # otherwise be quotable).
    assert pade_quad == [], "pure quadratic must not emit a Pade candidate"
    # The audit path still shows the underlying fit carried a negligible b.
    audit_quad = [r for r in generate_repairs(
        base_quad, [diagmod.FINDING_CURVATURE_SUSPECT], df_quad_tr, df_quad_tr["target"].to_numpy(),
        RepairConfig(max_candidates_per_round=default_repair_config.max_candidates_per_round,
                     max_rounds=1, max_added_terms=1,
                     min_validation_r2_gain=default_repair_config.min_validation_r2_gain,
                     equivalence_tol=default_repair_config.equivalence_tol,
                     operators=default_repair_config.operators,
                     r5_emit_unidentified=True))
        if r.metadata.get("operator") == "R5_PADE_SATURATION"]
    assert audit_quad and audit_quad[0].parameters["b"] * k_quad < 1e-4
    assert audit_quad[0].parameters["identifiability_status"] == "PREDICTIVE_ONLY_ILL_CONDITIONED"

    # Control B: Exponential Saturation (y = 0.5 * (1 - exp(-1000*p^2 / 0.5)))
    p_exp, y_exp = generate_qec_saturating_data(n_points=200, noise_sigma=0.0, seed=42, distance=3, model="exponential")
    df_exp_tr = pd.DataFrame({"p": p_exp[::2], "target": y_exp[::2]})
    df_exp_ho = pd.DataFrame({"p": p_exp[1::2], "target": y_exp[1::2]})
    k_exp = float(np.polyfit(df_exp_tr["p"] ** 2, df_exp_tr["target"], 1)[0])

    base_exp = CandidateProposal(
        proposer_id="test",
        candidate_id="cand_exp",
        formula=f"{k_exp:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda df, k=k_exp: k * (df["p"].to_numpy() ** 2),
    )

    reps_exp = generate_repairs(base_exp, [diagmod.FINDING_CURVATURE_SUSPECT], df_exp_tr, df_exp_tr["target"].to_numpy(), default_repair_config)
    assert len(reps_exp) > 0
    rep_e = reps_exp[0]
    pred_ho_e = rep_e.compute_score(df_exp_ho)
    y_ho_e = df_exp_ho["target"].to_numpy()
    ho_r2_e = 1.0 - float(np.sum((y_ho_e - pred_ho_e) ** 2)) / float(np.sum((y_ho_e - y_ho_e.mean()) ** 2))

    # Demonstrates the epistemic principle: Padé fits exponential truth with R^2 > 0.999
    # so high R^2 alone is insufficient to certify rational mechanism
    assert ho_r2_e > 0.999, f"Padé fits exponential curve closely: {ho_r2_e}"


def test_decoupled_five_metric_reporting(default_repair_config):
    """Test 5: Proposal metadata and parameters decouple structure, parameters, prediction, and identifiability."""
    p, obs = generate_qec_saturating_data(n_points=200, noise_sigma=0.02, seed=42, distance=3, model="pade")
    df_tr = pd.DataFrame({"p": p[::2], "target": obs[::2]})

    k_fit = float(np.polyfit(df_tr["p"] ** 2, df_tr["target"], 1)[0])
    base = CandidateProposal(
        proposer_id="test",
        candidate_id="cand_test_metrics",
        formula=f"{k_fit:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda df, k=k_fit: k * (df["p"].to_numpy() ** 2),
    )

    # Audit path: this case exists to OBSERVE an ill-conditioned fit, so the
    # emission refusal is explicitly opted out here. The default path (which
    # refuses to emit an unidentified denominator) is pinned in
    # test_dyn_rime_noise_present_battery.py.
    audit_cfg = RepairConfig(
        max_candidates_per_round=default_repair_config.max_candidates_per_round,
        max_rounds=default_repair_config.max_rounds,
        max_added_terms=default_repair_config.max_added_terms,
        min_validation_r2_gain=default_repair_config.min_validation_r2_gain,
        equivalence_tol=default_repair_config.equivalence_tol,
        operators=default_repair_config.operators,
        r5_emit_unidentified=True,
    )
    reps = generate_repairs(base, [diagmod.FINDING_CURVATURE_SUSPECT], df_tr, df_tr["target"].to_numpy(), audit_cfg)
    pade = [r for r in reps if r.metadata.get("operator") == "R5_PADE_SATURATION"]
    assert len(pade) > 0
    rep = pade[0]

    # Verify all decoupled fields exist and are populated
    assert "structure_match" in rep.metadata
    assert rep.metadata["structure_match"] is True
    assert "identifiability_status" in rep.parameters
    assert rep.parameters["identifiability_status"] in ("IDENTIFIED_STABLE", "PREDICTIVE_ONLY_ILL_CONDITIONED")
    assert "a_stderr" in rep.parameters
    assert "b_stderr" in rep.parameters
    assert "covariance_condition" in rep.parameters
    assert "envelope_ratio" in rep.parameters
    assert "repair_r2" in rep.parameters
