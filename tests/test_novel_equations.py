"""Tests demonstrating generation, regularized formulation, and validation of novel equations

addressing real-world breakdown regimes in:
1. Aerodynamics (Transonic compressibility breakdown at M -> 1)
2. Materials Science (Hall-Petch breakdown at nanoscale grain sizes d -> 0)
3. Nanotechnology (Casimir / van der Waals stiction divergence at gap z -> 0)
4. Quantum Computing (Knill-Laflamme condition and Quantum Singleton coding bounds)
"""
import math

import numpy as np
import sympy as sp

from apedp.applied_defects import (
    AppliedMaterialScienceDefectAuditor,
    AppliedNanotechnologyDefectAuditor,
    AppliedPhysicsDefectAuditor,
    UniversalAppliedDefectEngine,
)
from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.qec import (
    KnillLaflammeAuditor,
    QuantumCodeValidator,
    SurfaceCodeThresholdEngine,
)
from apedp.repair import RegularizationAuditor


def test_aerodynamics_transonic_regularization():
    """1. Aerodynamics: Prandtl-Glauert singularity vs. Van Dyke Transonic Regularization.

    Original: Cp = Cp_0 / sqrt(1 - M^2) -> diverges as M -> 1.
    Regularized: Cp_reg = Cp_0 / ((1 - M^2)^2 + eps^2)^(1/4).
    Calibrated: eps = 0.04 reduces operational distortion at M=0.8 from 3.9% to <0.2%,
    and at M=0.9 from 11.4% to 1.1%, while keeping transonic peak finite (-2.5).
    """
    engine = UniversalAppliedDefectEngine()

    # Original equation failure: Mach 1.0 divergence detected
    res_mach1 = engine.audit_applied_equation("Aerodynamics", "Prandtl_Glauert", {"mach_number": 1.0})
    assert res_mach1["defect_detected"] is True
    assert any("TRANSONIC_SINGULARITY" in d for d in res_mach1["defects"])

    # Verify regularized equation is finite and continuous through Mach 1
    cp_0 = -0.5
    eps_calibrated = 0.04  # Calibrated boundary layer relief parameter
    m_values = [0.0, 0.5, 0.8, 0.95, 1.0, 1.05, 1.2]
    for m in m_values:
        denom = ((1.0 - m**2)**2 + eps_calibrated**2)**0.25
        cp_reg = cp_0 / denom
        assert np.isfinite(cp_reg)
        assert abs(cp_reg) > 0

    # At M=1, Cp is smoothly regularized to Cp_0 / sqrt(eps)
    cp_at_1 = cp_0 / math.sqrt(eps_calibrated)
    res_at_1 = AppliedPhysicsDefectAuditor.audit_prandtl_glauert_compressibility(
        1.0, cp_incompressible=cp_0, boundary_relief_eps=eps_calibrated
    )
    assert abs(res_at_1["regularized_cp"] - cp_at_1) < 1e-9
    assert abs(res_at_1["regularized_cp"] - (-2.5)) < 1e-9

    # Verify calibrated operational distortion (<0.5% at M=0.8 vs 3.9% with eps=0.15)
    assert res_at_1["distortion_at_m08"] < 0.005
    assert res_at_1["distortion_at_m09"] < 0.015  # ~1.1% distortion at M=0.9

    # Verify airfoil thickness ratio scaling: tau = 0.08 -> eps = tau^(2/3) ~ 0.1856
    res_airfoil = AppliedPhysicsDefectAuditor.audit_prandtl_glauert_compressibility(
        1.0, cp_incompressible=cp_0, airfoil_thickness_ratio=0.08
    )
    expected_eps = 0.08 ** (2.0 / 3.0)
    assert abs(res_airfoil["boundary_relief_eps"] - expected_eps) < 1e-6


def test_materials_science_inverse_hall_petch_regularization():
    """2. Materials Science: Hall-Petch divergence vs. Inverse Hall-Petch Softening.

    Original: sigma_y = sigma_0 + k_HP / sqrt(d) -> diverges to infinity as d -> 0.
    Regularized: sigma_reg = sigma_0 + (k_HP / sqrt(d + d0)) * [1 / (1 + (dc / d)^2)].
    """
    engine = UniversalAppliedDefectEngine()

    # Original equation failure: Sub-10nm grain size triggers inverse Hall-Petch breakdown
    res_nano = engine.audit_applied_equation("Materials", "Hall_Petch", {"grain_size_meters": 3e-9})
    assert res_nano["defect_detected"] is True
    assert any("INVERSE_HALL_PETCH_BREAKDOWN" in d for d in res_nano["defects"])

    # Test dimensional assurance on the regularized formula
    dim_engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(dim_engine)
    dim_engine.register_variable("d_grain", "meter")
    dim_engine.register_variable("sigma_0", "pascal")
    dim_engine.register_variable("k_hp", "pascal * meter**0.5")
    dim_engine.register_variable("d_c", "meter")

    # Regularized Hall-Petch formula
    sigma_reg_sym = sp.Symbol("sigma_0") + (sp.Symbol("k_hp") / sp.sqrt(sp.Symbol("d_grain"))) * (
        1 / (1 + (sp.Symbol("d_c") / sp.Symbol("d_grain"))**2)
    )
    platform.register_model_spec("Regularized_Hall_Petch", sigma_reg_sym, "pascal")

    # Evaluate at macroscopic d (10 um) -> standard Hall-Petch behaviour
    val_macro = platform.evaluate_model(
        "Regularized_Hall_Petch",
        {"sigma_0": 70e6, "k_hp": 0.74e6, "d_grain": 10e-6, "d_c": 10e-9}
    )
    assert 3e8 < val_macro < 3.1e8  # ~304 MPa

    # Evaluate at sub-nanoscale d (1 nm) -> softening arrests divergence below theoretical shear (3 GPa)
    val_subnano = platform.evaluate_model(
        "Regularized_Hall_Petch",
        {"sigma_0": 70e6, "k_hp": 0.74e6, "d_grain": 1e-9, "d_c": 10e-9}
    )
    assert val_subnano < 1e9  # Under 1 GPa, preventing unphysical infinite strength

    # Concrete material profile verification: Ferritic Steel, Annealed Copper, Pure Titanium
    steel_res = AppliedMaterialScienceDefectAuditor.audit_hall_petch_nanoscale_breakdown(
        grain_size_meters=1e-9, material_name="ferritic_steel"
    )
    assert steel_res["material_profile"] == "ferritic_steel"
    assert steel_res["regularized_yield_stress_pa"] < 1.0e9  # ~301 MPa
    assert steel_res["theoretical_limit_respected"] is True

    copper_res = AppliedMaterialScienceDefectAuditor.audit_hall_petch_nanoscale_breakdown(
        grain_size_meters=1e-9, material_name="copper_annealed"
    )
    assert copper_res["regularized_yield_stress_pa"] < 1.5e9
    assert copper_res["theoretical_limit_respected"] is True

    ti_res = AppliedMaterialScienceDefectAuditor.audit_hall_petch_nanoscale_breakdown(
        grain_size_meters=1e-9, material_name="titanium_pure"
    )
    assert ti_res["regularized_yield_stress_pa"] < 4.0e9
    assert ti_res["theoretical_limit_respected"] is True


def test_nanotechnology_casimir_stiction_regularization():
    """3. Nanotechnology: Casimir/vdW continuum blowup vs. Born-Repulsion Regularization.

    Original: F_vdw = (A_H * Area) / (6 * pi * z^3) -> diverges as z -> 0.
    Regularized: F_reg = [A_H * Area / (6 * pi * (z + z0)^3)] * [1 - (z0 / (z + z0))^6].
    """
    engine = UniversalAppliedDefectEngine()

    # Original equation failure: Sub-nanometer separation causes infinite stiction
    res_stiction = engine.audit_applied_equation(
        "Nanotechnology", "Casimir_Stiction", {"separation_distance_meters": 1e-10}
    )
    assert res_stiction["defect_detected"] is True
    assert any("CASIMIR_STICTION_DIVERGENCE" in d for d in res_stiction["defects"])

    # Evaluate regularized Lennard-Jones/Born attractive-repulsive balance
    a_h = 1e-19      # Hamaker constant (Joules)
    area = 1e-8      # 100 um x 100 um plate (m^2)
    z0 = 3e-10       # Interatomic equilibrium spacing (0.3 nm)

    def f_regularized(z: float) -> float:
        z_eff = z + z0
        attraction = (a_h * area) / (6.0 * math.pi * (z_eff**3))
        repulsion_balance = 1.0 - (z0 / z_eff)**6
        return attraction * repulsion_balance

    # At macroscopic distance z = 100 nm, approaches classical 1/z^3 attraction
    f_far = f_regularized(100e-9)
    f_classical = (a_h * area) / (6.0 * math.pi * (100e-9)**3)
    assert abs(f_far - f_classical) / f_classical < 0.01

    # At physical contact z = 0, force smoothly drops to 0 (stable atomic equilibrium)
    f_contact = f_regularized(0.0)
    assert abs(f_contact) < 1e-15  # Zero net stiction at equilibrium contact!

    # Verify auditor produces zero net force at contact z = 0 and positive potential curvature
    contact_res = AppliedNanotechnologyDefectAuditor.audit_casimir_nems_stiction_divergence(
        separation_distance_meters=0.0
    )
    assert abs(contact_res["regularized_force_n"]) < 1e-15
    assert contact_res["equilibrium_contact_arrest"] is True
    assert contact_res["contact_mechanical_stability"] is True
    assert contact_res["potential_curvature_d2u_dz2"] > 0.0



def test_quantum_computing_knill_laflamme_and_coding_bounds():
    r"""4. Quantum Computing: Quantum Singleton Bound & Knill-Laflamme Algebraic Verification.

    Bounds: [[n, k, d]] code requires n - k >= 2*(d - 1).
    Conditions: P E_i^\dagger E_j P = alpha_ij P.
    """
    # 1. Quantum Singleton bound test:
    # An invalid code [[4, 1, 3]] violates Singleton: 4 - 1 = 3 < 2*(3 - 1) = 4
    singleton_bad, _, _ = QuantumCodeValidator.check_quantum_singleton_bound(4, 1, 3)
    assert singleton_bad is False

    # The 5-qubit code [[5, 1, 3]] saturates the bound: 5 - 1 = 4 == 2*(3 - 1)
    code_513 = QuantumCodeValidator.validate_code_parameters("FiveQubitCode", 5, 1, 3)
    assert code_513["physically_admissible"] is True
    assert code_513["singleton_bound"]["is_qmds_code"] is True

    # 2. Knill-Laflamme conditions verification for 3-qubit bit-flip code:
    # Codespace: span(|000>, |111>)
    ket_000 = np.zeros(8, dtype=complex)
    ket_000[0] = 1.0
    ket_111 = np.zeros(8, dtype=complex)
    ket_111[7] = 1.0

    # Projector P = |000><000| + |111><111|
    P = np.outer(ket_000, ket_000.conj()) + np.outer(ket_111, ket_111.conj())

    # Error set: Identity, X_1 (bit flip on qubit 1), X_2, X_3
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    E0 = np.kron(I2, np.kron(I2, I2))
    E1 = np.kron(X, np.kron(I2, I2))
    E2 = np.kron(I2, np.kron(X, I2))
    E3 = np.kron(I2, np.kron(I2, X))

    kl_res = KnillLaflammeAuditor.verify_knill_laflamme(P, [E0, E1, E2, E3])
    assert kl_res["is_correctable"] is True
    assert kl_res["is_degenerate_code"] is False

    # 3. Surface code regularized threshold model (single logical qubit k=1 -> p_sat=0.5)
    threshold_engine = SurfaceCodeThresholdEngine()
    audit_res = threshold_engine.register_and_audit_surface_code(
        d_val=3, C_prefactor=0.1, p_th_nominal=0.01, k_logical_qubits=1
    )
    assert audit_res["raw_flaw_detected"] is True
    assert audit_res["raw_model_evaluation"] > 1.0  # Raw model unphysical: 250% probability
    assert audit_res["regularized_within_physical_bounds"] is True  # Softened model guaranteed <= 0.5
    assert audit_res["depolarizing_saturation_ceiling"] == 0.5
    assert audit_res["status"] == "REPAIRED_AND_BOUNDED"

    # Multi-qubit depolarizing ceiling (k=2 -> p_sat = 1 - 1/4 = 0.75)
    audit_k2 = threshold_engine.register_and_audit_surface_code(
        d_val=3, C_prefactor=0.1, p_th_nominal=0.01, k_logical_qubits=2
    )
    assert audit_k2["depolarizing_saturation_ceiling"] == 0.75
    assert audit_k2["regularized_within_physical_bounds"] is True
    assert 0.0 <= audit_k2["regularized_evaluation"] <= 0.75


def test_four_tier_regularization_framework():
    """5. Methodological Optimization: Four-Tier Assurance Framework Verification."""
    # Tier 1: Mathematical Limit Verification
    m_sym = sp.Symbol("M", positive=True)
    cp0 = -0.5
    eps = 0.04
    reg_expr = cp0 / ((1.0 - m_sym**2)**2 + eps**2)**0.25
    t1 = RegularizationAuditor.audit_tier1_mathematical_limit(reg_expr, m_sym, 1.0)
    assert t1["singularity_removed"] is True
    assert abs(t1["limit_value"] - (-0.5 / math.sqrt(0.04))) < 1e-6

    # Tier 2: Numerical Stability Scan across 0 <= M <= 1.2
    grid = np.linspace(0.0, 1.2, 50)

    def reg_func(m: float) -> float:
        return cp0 / (((1.0 - m**2)**2 + eps**2) ** 0.25)

    t2 = RegularizationAuditor.audit_tier2_numerical_stability(reg_func, grid)
    assert t2["numerically_stable"] is True
    assert not t2["nan_detected"]
    assert not t2["inf_detected"]

    # Tier 3: Asymptotic Operational Distortion Profile
    def class_func(m: float) -> float:
        return cp0 / math.sqrt(1.0 - m**2)

    subsonic_grid = [0.1, 0.3, 0.5, 0.7, 0.8]
    t3 = RegularizationAuditor.audit_tier3_asymptotic_distortion(
        class_func, reg_func, subsonic_grid, distortion_threshold=0.01
    )
    assert t3["within_tolerance"] is True
    assert t3["max_distortion"] < 0.005  # Less than 0.5% distortion up to M=0.8

    # Tier 4: Parameter Sweep & Calibration Optimization
    def factory(e: float):
        return lambda m: cp0 / (((1.0 - m**2)**2 + e**2) ** 0.25)

    candidates = [0.01, 0.04, 0.10, 0.15, 0.20]
    sweep = RegularizationAuditor.audit_tier4_parameter_sweep(
        regularized_factory=factory,
        classical_func=class_func,
        param_candidates=candidates,
        operational_grid=subsonic_grid,
        boundary_point=1.0,
        max_boundary_allowance=10.0  # Max allowable |Cp| at M=1
    )
    assert sweep["optimal_parameter"] is not None
    assert len(sweep["sweep_results"]) == 5
    assert sweep["optimal_distortion"] < 0.005
