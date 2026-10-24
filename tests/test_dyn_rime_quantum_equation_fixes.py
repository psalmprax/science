"""Pinned regression tests for RIME quantum equation capability enhancements.

Covers the three verified fixes:
1. Limitation 1 (Negative Powers): Brus quantum dot confinement:
   E_g(R) = 1.74 + 2.45*R^(-2) - 0.28*R^(-1)
   Verifies that RIME discovers negative power scaling laws without bounded proxy
   divergence on extrapolation.
2. Limitation 2 (Continuous Denominator Scale in Rational Operator):
   Detuned Rabi Lorentzian lineshape:
   P_max(Delta) = Omega_R^2 / (Omega_R^2 + Delta^2) = 1 / (1 + 0.04*Delta^2)
   Verifies that R5_PADE_SATURATION fits Form 2 (resonance) lineshapes, identifying
   Omega_R^2 = 25.0 under clean and noisy regimes with covariance conditioning.
3. Limitation 3 (Harmonic Frequency Multipliers):
   Resonant Rabi pulse dynamics:
   P_e(t) = (1 - cos(2*t)) / 2 = sin^2(t)
   Verifies that harmonic_seeds=True discovers cos(t + t) = cos(2t) natively with
   holdout extrapolation R^2 = 1.000000.
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
from apedp_dyn.proposers.rime_proposer import RimeProposer, is_rime_available


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel unavailable")
def test_brus_quantum_dot_confinement_negative_powers():
    """Verify negative power discovery on Brus quantum dot confinement equation."""
    np.random.seed(42)
    # E_g(R) = 1.74 + 2.45 * R^(-2) - 0.28 * R^(-1)
    r_train = np.linspace(2.0, 5.0, 100)
    e_train = 1.74 + 2.45 * (r_train**-2) - 0.28 * (r_train**-1)
    df_train = pd.DataFrame({"R": r_train, "E_g": e_train})

    r_extrap = np.linspace(1.2, 8.0, 100)
    e_extrap = 1.74 + 2.45 * (r_extrap**-2) - 0.28 * (r_extrap**-1)
    df_extrap = pd.DataFrame({"R": r_extrap, "E_g": e_extrap})

    # 1. Native kernel power_ops with negative exponents
    prop_native = RimeProposer(
        feature_cols=["R"],
        target_col="E_g",
        static_mode=True,
        power_ops=True,
        additive_terms=2,
        budget=200,
    )
    proposals_native = prop_native.fit_and_propose(df_train)
    assert len(proposals_native) > 0, "Native RIME should produce proposals"
    best_native = proposals_native[0]
    assert "R^-2" in best_native.formula or "R^-1" in best_native.formula

    pred_extrap_native = best_native.compute_score(df_extrap)
    ss_tot = np.sum((e_extrap - np.mean(e_extrap)) ** 2)
    r2_native = 1.0 - np.sum((e_extrap - pred_extrap_native) ** 2) / ss_tot
    assert r2_native > 0.95, f"Native negative power extrapolation R^2 should be > 0.95, got {r2_native:.4f}"

    # 2. Algebraic transforms ['inv', 'inv_sq']
    prop_trans = RimeProposer(
        feature_cols=["R"],
        target_col="E_g",
        static_mode=True,
        algebraic_transforms=["inv", "inv_sq"],
        additive_terms=2,
        budget=200,
    )
    proposals_trans = prop_trans.fit_and_propose(df_train)
    assert len(proposals_trans) > 0, "Transforms RIME should produce proposals"
    best_trans = proposals_trans[0]
    assert "inv" in best_trans.formula

    pred_extrap_trans = best_trans.compute_score(df_extrap)
    r2_trans = 1.0 - np.sum((e_extrap - pred_extrap_trans) ** 2) / ss_tot
    assert r2_trans > 0.95, f"Transforms extrapolation R^2 should be > 0.95, got {r2_trans:.4f}"


def test_detuned_rabi_lorentzian_resonance_clean_and_noisy():
    """Verify continuous denominator scale fitting for Lorentzian lineshapes in R5."""
    # P_max(Delta) = Omega_R^2 / (Omega_R^2 + Delta^2) with Omega_R = 5.0 => b = 1/25 = 0.04
    omega_sq_true = 25.0
    delta_train = np.linspace(-15.0, 15.0, 100)
    z_train = delta_train**2
    y_clean = omega_sq_true / (omega_sq_true + z_train)

    df_train_clean = pd.DataFrame({"Delta": delta_train, "target": y_clean})

    base_prop = CandidateProposal(
        proposer_id="rime_symbolic",
        candidate_id="cand_rabi_resonance_base",
        formula="Delta^2",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("Delta",),
        compute_score=lambda f: f["Delta"].to_numpy(dtype=float) ** 2,
    )

    config = RepairConfig(
        max_candidates_per_round=4,
        max_rounds=1,
        max_added_terms=1,
        min_validation_r2_gain=0.01,
        equivalence_tol=1e-9,
        operators=(RepairOperator.R5_PADE_SATURATION,),
    )
    findings = [diagmod.FINDING_CURVATURE_SUSPECT]

    # Clean verification
    repairs_clean = generate_repairs(base_prop, findings, df_train_clean, y_clean, config)
    assert len(repairs_clean) > 0, "R5 repair should generate resonance candidate"
    best_clean = repairs_clean[0]
    assert best_clean.parameters.get("form") == "resonance"
    assert best_clean.parameters.get("identifiability_status") == "IDENTIFIED_STABLE"
    assert np.isclose(best_clean.parameters.get("a"), 1.0, atol=1e-3)
    assert np.isclose(best_clean.parameters.get("b"), 0.04, atol=1e-4)

    omega_sq_rec_clean = 1.0 / best_clean.parameters.get("b")
    assert np.isclose(omega_sq_rec_clean, 25.0, atol=1e-2)

    delta_test = np.linspace(-30.0, 30.0, 200)
    y_test = omega_sq_true / (omega_sq_true + delta_test**2)
    df_test = pd.DataFrame({"Delta": delta_test})
    pred_test = best_clean.compute_score(df_test)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_clean = 1.0 - np.sum((y_test - pred_test) ** 2) / ss_tot
    assert r2_clean > 0.9999, f"Clean test R^2 should exceed 0.9999, got {r2_clean:.6f}"

    # Noisy verification (2% noise)
    np.random.seed(42)
    y_noisy = y_clean + 0.02 * np.random.normal(0, 1, size=len(y_clean))
    df_train_noisy = pd.DataFrame({"Delta": delta_train, "target": y_noisy})
    repairs_noisy = generate_repairs(base_prop, findings, df_train_noisy, y_noisy, config)
    assert len(repairs_noisy) > 0, "R5 repair should generate resonance candidate with 2% noise"
    best_noisy = repairs_noisy[0]
    assert best_noisy.parameters.get("identifiability_status") == "IDENTIFIED_STABLE"
    omega_sq_rec_noisy = 1.0 / best_noisy.parameters.get("b")
    # Parameter within 2% error of ground truth
    rel_err = abs(omega_sq_rec_noisy - 25.0) / 25.0
    assert rel_err < 0.02, f"Recovered Omega^2 error should be < 2%, got {rel_err*100:.2f}%"

    pred_test_noisy = best_noisy.compute_score(df_test)
    r2_noisy = 1.0 - np.sum((y_test - pred_test_noisy) ** 2) / ss_tot
    assert r2_noisy > 0.999, f"Noisy test R^2 should exceed 0.999, got {r2_noisy:.6f}"


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel unavailable")
def test_resonant_rabi_oscillations_harmonic_seeds():
    """Verify frequency multiplier recovery via harmonic_seeds on Rabi oscillations."""
    # P_e(t) = sin^2(t) = 0.5 - 0.5 * cos(2 * t)
    t_train = np.linspace(0.0, 4.0 * np.pi, 200)
    y_train = np.sin(t_train) ** 2
    df_train = pd.DataFrame({"t": t_train, "P_e": y_train})

    t_test = np.linspace(4.0 * np.pi, 8.0 * np.pi, 200)
    y_test = np.sin(t_test) ** 2
    df_test = pd.DataFrame({"t": t_test, "P_e": y_test})

    # 1. Native kernel harmonic search
    prop_native = RimeProposer(
        feature_cols=["t"],
        target_col="P_e",
        static_mode=True,
        stem_ops=True,
        harmonic_seeds=True,
        budget=300,
        beam=16,
        max_depth=3,
    )
    proposals_native = prop_native.fit_and_propose(df_train)
    assert len(proposals_native) > 0, "Native harmonic RIME should produce proposals"
    best_native = proposals_native[0]
    assert "cos((t + t))" in best_native.formula or "cos(2" in best_native.formula

    pred_native = best_native.compute_score(df_test)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_native = 1.0 - np.sum((y_test - pred_native) ** 2) / ss_tot
    assert r2_native > 0.9999, f"Native harmonic test R^2 should exceed 0.9999, got {r2_native:.6f}"

    # 2. STEM transforms cos2
    prop_trans = RimeProposer(
        feature_cols=["t"],
        target_col="P_e",
        static_mode=True,
        stem_ops=True,
        algebraic_transforms=["cos2", "sin2"],
        budget=200,
    )
    proposals_trans = prop_trans.fit_and_propose(df_train)
    assert len(proposals_trans) > 0, "Transforms RIME should produce proposals"
    best_trans = proposals_trans[0]
    assert "cos2(t)" in best_trans.formula

    pred_trans = best_trans.compute_score(df_test)
    r2_trans = 1.0 - np.sum((y_test - pred_trans) ** 2) / ss_tot
    assert r2_trans > 0.9999, f"Transforms test R^2 should exceed 0.9999, got {r2_trans:.6f}"


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel unavailable")
def test_landauer_conductance_staircase_quantization_and_refusal():
    """Verify Landauer Conductance Staircase: classical limit, referee refusal, and floor discovery."""
    from apedp_dyn.epistemic.adjudicate import _sole_survivor_whiteness

    g0 = 1.0

    def landauer_staircase(vg: np.ndarray) -> np.ndarray:
        g = np.zeros_like(vg)
        for n in range(1, 5):
            g += g0 * (vg >= float(n)).astype(float)
        return g

    np.random.seed(42)
    vg_train = np.sort(np.random.uniform(0.1, 3.9, 200))
    g_train = landauer_staircase(vg_train)

    vg_extrap = np.linspace(4.1, 5.9, 100)
    g_extrap = np.zeros_like(vg_extrap)
    for n in range(1, 7):
        g_extrap += g0 * (vg_extrap >= float(n)).astype(float)

    df_train = pd.DataFrame({"Vg": vg_train, "G": g_train})
    df_extrap = pd.DataFrame({"Vg": vg_extrap, "G": g_extrap})

    # 1. Unextended RIME discovers macroscopic Ohm's law limit, but Epistemic Referee refuses
    prop_smooth = RimeProposer(
        feature_cols=["Vg"],
        target_col="G",
        static_mode=True,
        budget=200,
    )
    props_smooth = prop_smooth.fit_and_propose(df_train)
    assert len(props_smooth) > 0
    best_smooth = props_smooth[0]
    pred_train_smooth = best_smooth.compute_score(df_train)
    resid_smooth = g_train - pred_train_smooth

    # Referee whiteness must fail on the smooth linear model (sawtooth residual)
    white_pass, note = _sole_survivor_whiteness(resid_smooth, 0.5)
    assert not white_pass, f"Epistemic referee should refuse smooth model: {note}"

    # 2. RIME with floor transform discovers exact quantized conductance staircase
    prop_quant = RimeProposer(
        feature_cols=["Vg"],
        target_col="G",
        static_mode=True,
        algebraic_transforms=["floor"],
        budget=200,
    )
    props_quant = prop_quant.fit_and_propose(df_train)
    assert len(props_quant) > 0
    best_quant = props_quant[0]
    assert "floor(Vg)" in best_quant.formula

    # Check scale factor recovers G_0 = 1.0
    assert np.isclose(best_quant.parameters["scale"], 1.0, atol=1e-3)

    # Check forward extrapolation across unseen plateaus 4 and 5
    pred_ext = best_quant.compute_score(df_extrap)
    ss_tot = np.sum((g_extrap - np.mean(g_extrap)) ** 2)
    r2_ext = 1.0 - np.sum((g_extrap - pred_ext) ** 2) / ss_tot
    assert r2_ext > 0.999999, f"Quantized extrapolation R^2 should be ~1.0, got {r2_ext:.8f}"


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel unavailable")
def test_schwinger_vacuum_pair_production_discovery():
    """Verify non-perturbative Schwinger vacuum pair production discovery and deep extrapolation."""
    from apedp_dyn.epistemic.adjudicate import _sole_survivor_whiteness

    c0_true = 10.0

    def schwinger_rate(x: np.ndarray) -> np.ndarray:
        safe_x = np.maximum(1e-12, x)
        return c0_true * (safe_x**2) * np.exp(-1.0 / safe_x)

    np.random.seed(42)
    x_tr = np.sort(np.random.uniform(0.12, 0.40, 100))
    y_tr = schwinger_rate(x_tr)

    x_ex = np.linspace(0.04, 0.10, 100)
    y_ex = schwinger_rate(x_ex)

    df_tr = pd.DataFrame({"x": x_tr, "y": y_tr})
    df_ex = pd.DataFrame({"x": x_ex, "y": y_ex})

    prop = RimeProposer(
        feature_cols=["x"],
        target_col="y",
        static_mode=True,
        algebraic_transforms=["arrh", "sq"],
        budget=200,
    )
    props = prop.fit_and_propose(df_tr)
    assert len(props) > 0, "RIME should discover Schwinger rate"
    best = props[0]
    assert "arrh" in best.formula

    # Check prefactor recovers C0 = 10.0
    assert np.isclose(best.parameters["scale"], 10.0, atol=1e-3)

    # Check deep tunneling extrapolation across 7 orders of magnitude
    pred_ex = best.compute_score(df_ex)
    ss_tot = np.sum((y_ex - np.mean(y_ex)) ** 2)
    r2_ex = 1.0 - np.sum((y_ex - pred_ex) ** 2) / ss_tot
    assert r2_ex > 0.999999, f"Schwinger deep extrapolation R^2 should be ~1.0, got {r2_ex:.8f}"

    # Clean fit precision check (within machine float precision)
    pred_tr = best.compute_score(df_tr)
    assert np.allclose(pred_tr, y_tr, atol=1e-6)

    # Residual whiteness check on realistic noisy data (2% observation noise)
    y_noisy = y_tr * (1.0 + 0.02 * np.random.normal(0, 1, size=len(y_tr)))
    df_noisy = pd.DataFrame({"x": x_tr, "y": y_noisy})
    props_noisy = prop.fit_and_propose(df_noisy)
    assert len(props_noisy) > 0
    pred_noisy = props_noisy[0].compute_score(df_noisy)
    resid_noisy = y_noisy - pred_noisy
    white_pass, note = _sole_survivor_whiteness(resid_noisy, 0.5)
    assert white_pass, f"Schwinger true model residuals should pass whiteness: {note}"


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel unavailable")
def test_planck_blackbody_radiance_discovery():
    """Verify Planck blackbody distribution discovery and dual asymptotic extrapolation."""
    from apedp_dyn.epistemic.adjudicate import _sole_survivor_whiteness

    c0_true = 1.0

    def planck_radiance(x: np.ndarray) -> np.ndarray:
        safe_x = np.maximum(1e-12, x)
        return c0_true * (safe_x**3) / (np.exp(np.clip(safe_x, -50, 50)) - 1.0)

    np.random.seed(42)
    x_tr = np.sort(np.random.uniform(0.5, 6.0, 100))
    y_tr = planck_radiance(x_tr)

    x_ex_lo = np.linspace(0.05, 0.40, 50)
    y_ex_lo = planck_radiance(x_ex_lo)
    x_ex_hi = np.linspace(7.0, 12.0, 50)
    y_ex_hi = planck_radiance(x_ex_hi)

    df_tr = pd.DataFrame({"x": x_tr, "y": y_tr})
    df_ex_lo = pd.DataFrame({"x": x_ex_lo, "y": y_ex_lo})
    df_ex_hi = pd.DataFrame({"x": x_ex_hi, "y": y_ex_hi})

    prop = RimeProposer(
        feature_cols=["x"],
        target_col="y",
        static_mode=True,
        algebraic_transforms=["planck", "bose"],
        budget=200,
    )
    props = prop.fit_and_propose(df_tr)
    assert len(props) > 0, "RIME should discover Planck distribution"
    best = props[0]
    assert "planck" in best.formula or "bose" in best.formula

    # Check Rayleigh-Jeans low-frequency extrapolation
    pred_lo = best.compute_score(df_ex_lo)
    ss_tot_lo = np.sum((y_ex_lo - np.mean(y_ex_lo)) ** 2)
    r2_lo = 1.0 - np.sum((y_ex_lo - pred_lo) ** 2) / ss_tot_lo
    assert r2_lo > 0.999999, f"Rayleigh-Jeans extrapolation R^2 should be ~1.0, got {r2_lo:.8f}"

    # Check Wien high-frequency exponential tail extrapolation
    pred_hi = best.compute_score(df_ex_hi)
    ss_tot_hi = np.sum((y_ex_hi - np.mean(y_ex_hi)) ** 2)
    r2_hi = 1.0 - np.sum((y_ex_hi - pred_hi) ** 2) / ss_tot_hi
    assert r2_hi > 0.999999, f"Wien extrapolation R^2 should be ~1.0, got {r2_hi:.8f}"

    # Clean fit precision check
    pred_tr = best.compute_score(df_tr)
    assert np.allclose(pred_tr, y_tr, atol=1e-6)

    # Residual whiteness check on realistic noisy data (2% observation noise)
    y_noisy = y_tr * (1.0 + 0.02 * np.random.normal(0, 1, size=len(y_tr)))
    df_noisy = pd.DataFrame({"x": x_tr, "y": y_noisy})
    props_noisy = prop.fit_and_propose(df_noisy)
    assert len(props_noisy) > 0
    pred_noisy = props_noisy[0].compute_score(df_noisy)
    resid_noisy = y_noisy - pred_noisy
    white_pass, note = _sole_survivor_whiteness(resid_noisy, 0.5)
    assert white_pass, f"Planck true model residuals should pass whiteness: {note}"

