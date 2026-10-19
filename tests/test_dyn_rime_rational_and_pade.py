"""Tests for RIME rational division, Padé saturation repair, and saturating transforms.

Validates the three implementation pathways for rational and saturating discovery:
1. Option 1: Native Rust SafeDiv rational operations in RIME search graph & rewrites.
2. Option 2: Epistemic Repair Engine Operator R5 (Padé saturation envelope).
3. Option 3: Normalized algebraic transforms (sat_norm, sat_sq_norm) in RimeProposer.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic import rime_repair_diagnosis as diagmod
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    generate_repairs,
)
from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    StrategyArchetype,
)
from apedp_dyn.proposers.rime_proposer import _RIME_AVAILABLE, RimeProposer


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="apedp_kernel with RIME required")
def test_option_1_native_rust_rational_ops_safediv_discovery():
    """Option 1: Verify RimeProposer with rational_ops=True discovers SafeDiv graphs."""
    np.random.seed(42)
    # Generate clean rational target y = x / (1 + x)
    x = np.linspace(0.1, 5.0, 100)
    y = x / (1.0 + x)

    df = pd.DataFrame({"x": x, "y": y})

    # Run RIME with rational_ops=True
    proposer = RimeProposer(
        feature_cols=["x"],
        target_col="y",
        max_depth=1,
        beam=4,
        budget=50,
        static_mode=True,
        rational_ops=True,
        min_r2=0.95,
    )
    proposals = proposer.fit_and_propose(df)

    assert len(proposals) > 0, "RIME should discover candidates with rational_ops=True"
    best = proposals[0]
    assert best.metadata.get("rational_ops") is True
    # Verify the formula or equation contains safediv
    eq = best.metadata.get("equation", "")
    assert "safediv" in eq, f"Expected safediv in equation, got: {eq}"

    # Verify score evaluation matches ground truth
    y_pred = best.compute_score(df)
    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r2 > 0.99, f"Discovered SafeDiv form should achieve R^2 > 0.99, got {r2:.4f}"


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="apedp_kernel with RIME required")
def test_option_1_rational_ops_disabled_by_default():
    """Verify rational_ops defaults to False and preserves frozen grammar."""
    proposer = RimeProposer()
    assert proposer._rational_ops is False


def test_option_2_epistemic_repair_engine_pade_saturation():
    """Option 2: Verify R5_PADE_SATURATION operator on saturating QEC threshold data."""
    np.random.seed(42)
    # Saturating QEC-style physical data: p_L = (p / p_th)^2 / (1 + (p / p_th)^2)
    p_th = 0.01
    p = np.linspace(0.001, 0.025, 100)
    u = (p / p_th) ** 2
    p_L = u / (1.0 + u)

    df = pd.DataFrame({"p": p, "p_L": p_L})
    # Split: 60 rows repair partition, 40 rows holdout test partition
    df_repair = df.iloc[:60].reset_index(drop=True)
    df_test = df.iloc[60:].reset_index(drop=True)

    # Initial naive proposal: fits un-saturated textbook power law y ~ k * p^2
    # which has high curvature error near threshold
    k_fit = float(np.polyfit(df_repair["p"] ** 2, df_repair["p_L"], 1)[0])
    base_prop = CandidateProposal(
        proposer_id="naive_power_law",
        candidate_id="cand_qec_d3_base",
        formula=f"{k_fit:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda frame: k_fit * (frame["p"].to_numpy(dtype=float) ** 2),
    )

    config = RepairConfig(
        max_candidates_per_round=4,
        max_rounds=1,
        max_added_terms=1,
        min_validation_r2_gain=0.02,
        equivalence_tol=1e-9,
        operators=(RepairOperator.R5_PADE_SATURATION,),
    )

    findings = [diagmod.FINDING_CURVATURE_SUSPECT]
    repairs = generate_repairs(
        base_prop,
        findings,
        df_repair,
        df_repair["p_L"].to_numpy(),
        config,
    )

    assert len(repairs) > 0, "Padé saturation repair should be generated"
    best_rep = repairs[0]
    assert best_rep.metadata.get("operator") == RepairOperator.R5_PADE_SATURATION.value
    assert "pade_sat" in best_rep.metadata.get("term", "")
    assert best_rep.parameters["repair_r2"] > 0.99, f"Repaired R^2 should be > 0.99, got {best_rep.parameters['repair_r2']:.4f}"

    # Verify generalization on holdout test frame
    y_test_pred = best_rep.compute_score(df_test)
    y_test_true = df_test["p_L"].to_numpy()
    ss_res = np.sum((y_test_true - y_test_pred) ** 2)
    ss_tot = np.sum((y_test_true - np.mean(y_test_true)) ** 2)
    test_r2 = 1.0 - (ss_res / ss_tot)
    assert test_r2 > 0.99, f"Holdout test R^2 should exceed 0.99, got {test_r2:.4f}"


def test_option_2_epistemic_repair_engine_end_to_end_repair():
    """Option 2: Verify full repair() pipeline with partitions and sealed evaluation."""
    from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
    from apedp_dyn.epistemic.rime_repair_engine import RepairStatus, repair

    np.random.seed(42)
    p_th = 0.01
    p = np.linspace(0.001, 0.025, 1000)
    u = (p / p_th) ** 2
    p_L = u / (1.0 + u)

    df = pd.DataFrame({"p": p, "target": p_L})
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())

    k_fit = float(np.polyfit(parts.discovery_df["p"] ** 2, parts.discovery_y, 1)[0])
    base_prop = CandidateProposal(
        proposer_id="naive_power_law",
        candidate_id="cand_qec_d3_base",
        formula=f"{k_fit:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda frame: k_fit * (frame["p"].to_numpy(dtype=float) ** 2),
    )

    config = RepairConfig(
        max_candidates_per_round=4,
        max_rounds=1,
        max_added_terms=1,
        min_validation_r2_gain=0.02,
        equivalence_tol=1e-9,
        operators=(RepairOperator.R5_PADE_SATURATION,),
    )
    # Mechanism assertions run with the cross-slice invariance probe disabled:
    # this frame is a single-variable narrow-band grid (p in [0.001, 0.025]),
    # and contiguous slices re-propose structurally different originals (state
    # vs lag forms) by construction, so the gate resolves AMBIGUOUS regardless
    # of the R5 fit quality. The gate's refusal behaviour is asserted
    # explicitly below and covered by
    # tests/test_dyn_rime_repair_v012_sensitivity.py.
    mech_config = replace(config, invariance_slices=0)

    cuts = {
        "rho_flag": RESID_RHO_FLAG,
        "bias_ratio_cut": 0.25,
        "variance_ratio_cut": 4.0,
        "curvature_corr_cut": 0.3,
        "interaction_corr_cut": 0.3,
        "min_rows": 30,
    }

    out = repair(base_prop, parts, mech_config, **cuts)
    assert out.status == RepairStatus.IMPROVED, f"Repair failed: {out.submit_note}"
    assert out.winner is not None
    assert out.winner.metadata["operator"] == RepairOperator.R5_PADE_SATURATION.value
    assert out.sealed is not None
    assert out.sealed["winner_r2"] > 0.99
    assert out.sealed["sealed_r2_gain"] > 0

    # With the default pair-mode invariance probe the same frame resolves
    # AMBIGUOUS: independent dev slices cannot re-discover the pooled
    # structure, so the identification claim is withheld (fail-closed).
    gated = repair(base_prop, parts, config, **cuts)
    assert gated.status == RepairStatus.AMBIGUOUS
    assert "cross-slice invariance" in gated.submit_note


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="apedp_kernel with RIME required")
def test_option_3_normalized_algebraic_transforms():
    """Option 3: Verify RimeProposer with sat_sq_norm transform on saturating QEC data."""
    np.random.seed(42)
    p_th = 0.01
    p = np.linspace(0.001, 0.025, 80)
    u = (p / p_th) ** 2
    p_L = u / (1.0 + u)
    df = pd.DataFrame({"p": p, "p_L": p_L})

    proposer = RimeProposer(
        feature_cols=["p"],
        target_col="p_L",
        max_depth=0,
        budget=20,
        static_mode=True,
        algebraic_transforms=["sat_sq_norm"],
        min_r2=0.90,
    )
    proposals = proposer.fit_and_propose(df)

    assert len(proposals) > 0, "RimeProposer should discover sat_sq_norm candidate"
    best = proposals[0]
    assert "sat_sq_norm(p)" in best.formula
    assert best.parameters["dev_r2"] > 0.97, f"Expected dev_r2 > 0.97, got {best.parameters['dev_r2']}"

    # Verify score evaluation on new test frame
    p_test = np.linspace(0.002, 0.020, 30)
    u_test = (p_test / p_th) ** 2
    y_test_true = u_test / (1.0 + u_test)
    df_test = pd.DataFrame({"p": p_test, "p_L": y_test_true})

    y_test_pred = best.compute_score(df_test)
    ss_res = np.sum((y_test_true - y_test_pred) ** 2)
    ss_tot = np.sum((y_test_true - np.mean(y_test_true)) ** 2)
    test_r2 = 1.0 - (ss_res / ss_tot)
    assert test_r2 > 0.97, f"Test R^2 should be > 0.97, got {test_r2:.4f}"


def test_options_123_under_2pct_and_10pct_noise():
    """Pin discovery performance across Options 1, 2, and 3 under 0%, 2%, and 10% noise.

    Empirical stress-test benchmarks:
    - Option 1 (SafeDiv): 1.0000 (clean) -> 0.9997 (2% noise) -> 0.9919 (10% noise).
    - Option 2 (R5 Padé): 1.0000 (clean) -> 0.9068 (2% noise) -> NO_IMPROVEMENT / gate abstains (10% noise).
    - Option 3 (sat_sq_norm): 0.9753 (clean) -> 0.9750 (2% noise) -> 0.9752 (10% noise).
    """
    from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
    from apedp_dyn.epistemic.rime_repair_engine import (
        RepairConfig,
        RepairOperator,
        RepairPartitions,
        RepairStatus,
        repair,
    )

    # 1. Option 1: SafeDiv under noise
    x = np.linspace(0.1, 5.0, 100)
    y_clean = x / (1.0 + x)
    rng = np.random.RandomState(42)

    for noise, min_expected_r2 in [(0.0, 0.9999), (0.02, 0.999), (0.10, 0.990)]:
        sigma = noise * float(np.std(y_clean))
        y = y_clean + rng.normal(0, sigma, len(x)) if noise > 0 else y_clean
        prop = RimeProposer(
            feature_cols=["x"], target_col="y", max_depth=1, budget=50,
            static_mode=True, rational_ops=True, min_r2=0.0,
        )
        res = prop.fit_and_propose(pd.DataFrame({"x": x, "y": y}))
        assert len(res) > 0
        pred = res[0].compute_score(pd.DataFrame({"x": x}))
        r2 = 1.0 - np.sum((y_clean - pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2)
        assert r2 >= min_expected_r2, f"Option 1 noise={noise}: got R2={r2:.4f}"

    # 2. Option 2: R5 Padé repair under noise
    p_th = 0.01
    p = np.linspace(0.001, 0.025, 1000)
    u = (p / p_th) ** 2
    p_L_clean = u / (1.0 + u)
    sigma_p = float(np.std(p_L_clean))

    config = RepairConfig(
        max_candidates_per_round=4, max_rounds=1, max_added_terms=1,
        min_validation_r2_gain=0.01, equivalence_tol=1e-9,
        operators=(RepairOperator.R5_PADE_SATURATION,),
        # Same narrow-band-grid rationale as test_option_2: this test pins the
        # R5 sealed-R2 behaviour under noise. The cross-slice invariance gate
        # is a separate identification criterion, covered by
        # tests/test_dyn_rime_repair_v012_sensitivity.py.
        invariance_slices=0,
    )
    cuts = {
        "rho_flag": RESID_RHO_FLAG, "bias_ratio_cut": 0.25, "variance_ratio_cut": 4.0,
        "curvature_corr_cut": 0.3, "interaction_corr_cut": 0.3, "min_rows": 30,
    }

    # 2a. 0% noise: exact recovery
    parts_0 = RepairPartitions.from_frame(pd.DataFrame({"p": p, "target": p_L_clean}), p_L_clean)
    k_fit0 = float(np.polyfit(parts_0.discovery_df["p"] ** 2, parts_0.discovery_y, 1)[0])
    base0 = CandidateProposal(
        proposer_id="p", candidate_id="b0", formula=f"{k_fit0}*p^2",
        strategy_archetype=StrategyArchetype.TACTICAL, declared_scale_invariant=True,
        compute_score=lambda fr: k_fit0 * (fr["p"].to_numpy(dtype=float) ** 2),
    )
    out0 = repair(base0, parts_0, config, **cuts)
    assert out0.status == RepairStatus.IMPROVED
    assert out0.sealed["winner_r2"] > 0.9999

    # 2b. 2% noise: sealed winner R^2 ~ 0.9068
    rng2 = np.random.RandomState(42)
    p_L_2 = p_L_clean + rng2.normal(0, 0.02 * sigma_p, len(p))
    parts_2 = RepairPartitions.from_frame(pd.DataFrame({"p": p, "target": p_L_2}), p_L_2)
    k_fit2 = float(np.polyfit(parts_2.discovery_df["p"] ** 2, parts_2.discovery_y, 1)[0])
    base2 = CandidateProposal(
        proposer_id="p", candidate_id="b2", formula=f"{k_fit2}*p^2",
        strategy_archetype=StrategyArchetype.TACTICAL, declared_scale_invariant=True,
        compute_score=lambda fr: k_fit2 * (fr["p"].to_numpy(dtype=float) ** 2),
    )
    out2 = repair(base2, parts_2, config, **cuts)
    assert out2.status == RepairStatus.IMPROVED
    assert out2.sealed["winner_r2"] == pytest.approx(0.9068, abs=0.01)

    # 2c. 10% noise: sealed winner R^2 drops to ~0.2024 due to high noise distortion
    rng10 = np.random.RandomState(42)
    p_L_10 = p_L_clean + rng10.normal(0, 0.10 * sigma_p, len(p))
    parts_10 = RepairPartitions.from_frame(pd.DataFrame({"p": p, "target": p_L_10}), p_L_10)
    k_fit10 = float(np.polyfit(parts_10.discovery_df["p"] ** 2, parts_10.discovery_y, 1)[0])
    base10 = CandidateProposal(
        proposer_id="p", candidate_id="b10", formula=f"{k_fit10}*p^2",
        strategy_archetype=StrategyArchetype.TACTICAL, declared_scale_invariant=True,
        compute_score=lambda fr: k_fit10 * (fr["p"].to_numpy(dtype=float) ** 2),
    )
    # v0.2.1: at 10% noise the denominator is no longer identified, so the
    # DEFAULT policy refuses to emit the Pade candidate and the run resolves
    # NO_IMPROVEMENT (the test's own docstring anticipated this as "gate
    # abstains"). The degraded fit is still observable on the audit path.
    out10 = repair(base10, parts_10, config, **cuts)
    assert out10.status in (RepairStatus.IMPROVED, RepairStatus.NO_IMPROVEMENT), out10.status
    if out10.status == RepairStatus.NO_IMPROVEMENT:
        assert out10.winner is None and out10.sealed is None
        print("\n[r5-noise] 10%% noise: default policy refuses (denominator unidentified)")
    else:
        assert out10.sealed["winner_r2"] == pytest.approx(0.2024, abs=0.01)

    # 3. Option 3: sat_sq_norm under noise (consistently ~ 0.975)
    p_sub = np.linspace(0.001, 0.025, 80)
    u_sub = (p_sub / p_th) ** 2
    y_sub_clean = u_sub / (1.0 + u_sub)
    sigma_sub = float(np.std(y_sub_clean))

    for noise in [0.0, 0.02, 0.10]:
        y_noisy = y_sub_clean + rng.normal(0, noise * sigma_sub, len(p_sub)) if noise > 0 else y_sub_clean
        prop3 = RimeProposer(
            feature_cols=["p"], target_col="y", max_depth=0, budget=20,
            static_mode=True, algebraic_transforms=["sat_sq_norm"], min_r2=0.0,
        )
        res3 = prop3.fit_and_propose(pd.DataFrame({"p": p_sub, "y": y_noisy}))
        assert len(res3) > 0
        pred3 = res3[0].compute_score(pd.DataFrame({"p": p_sub}))
        r2_3 = 1.0 - np.sum((y_sub_clean - pred3) ** 2) / np.sum((y_sub_clean - np.mean(y_sub_clean)) ** 2)
        # Measured per-level (shared rng advances across levels): ~0.975 at
        # zero noise, ~0.996 at 2%/10% since square_seeds became default-on
        # under the operator release gate. Floor-asserted rather than pinned
        # to one number, because the rng is shared and the curve is the point.
        print(f"\n[sat_sq_norm] noise={noise:.2f} R2={r2_3:.4f}")
        assert r2_3 >= 0.95, (noise, r2_3)
