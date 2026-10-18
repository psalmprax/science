"""Factory functions for the physical benchmark contracts."""
from __future__ import annotations

from .contracts import (
    ScientificBenchmarkContract,
)

# =========================================================================
# Canonical Benchmark Contracts for the 5 Frontier Domain Repairs
# =========================================================================

def create_hall_petch_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Nanoscale Hall-Petch Yield Strength Softening."""
    return ScientificBenchmarkContract(
        model_id="HP-REF-001",
        baseline_model="sigma_y = sigma_0 + k_HP / sqrt(d)",
        refined_model="sigma_y = min(sigma_0 + (k_HP / sqrt(d + d0)) / (1 + (dc / d)^2), sigma_theoretical)",
        mechanism="Dislocation pileup exhaustion coupled to grain boundary sliding threshold and atomic shear limit",
        validated_domain={"grain_size_m": (3e-9, 100e-6)},
        predicted_observables=("yield_strength_pa",),
        prediction_intervals={"yield_strength_pa": (50e6, 3.0e9)},
        falsification_thresholds={"max_relative_error": 0.08},
        locked_test_data_hash="fals_hall_petch_sealed_data_hash",
        data_generation_method="Molecular dynamics + micropillar compression compilation",
        noise_model="Gaussian sensor noise (sigma=3%) + dislocation distribution variance",
        expected_failure_modes=("High-temperature Coble/Nabarro-Herring creep (T > 0.4 Tm)",),
        expected_limits={"d_to_zero": "sigma_0", "d_to_infinity": "sigma_0"},
        expected_invariants=("Non-negative yield stress: sigma_y > 0", "sigma_y <= sigma_theoretical"),
        pass_criteria=("Relative prediction error < 8% across 3–100 nm", "No divergence at d -> 0"),
        fail_criteria=("Yield stress exceeds 3.0 GPa", "Unbounded divergence below 10 nm"),
        domain_restriction_criteria=("Restrict domain to d > 3 nm if amorphous intergranular phase develops",),
    )


def create_casimir_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Casimir NEMS Stiction Divergence / Steric Repulsion."""
    return ScientificBenchmarkContract(
        model_id="CAS-REF-001",
        baseline_model="F_vdW = (A_H * Area) / (6 * pi * z^3)",
        # Parentheses rather than square brackets: grouping punctuation has to be in the
        # subset ``safe_parse_expr`` accepts, or the model string cannot be audited.
        refined_model="F_net = (A_H * Area / (6 * pi * (z + z0)^3)) * (1 - (z0 / (z + z0))^6)",
        mechanism="Integrated Lennard-Jones retarded dispersion + Born steric electronic Pauli repulsion",
        validated_domain={"separation_z_m": (0.0, 10e-6)},
        predicted_observables=("attractive_force_n", "contact_stiffness_n_m"),
        prediction_intervals={"attractive_force_n": (0.0, 1.0e-3)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="fals_casimir_sealed_data_hash",
        data_generation_method="AFM cantilever force-distance curves in ultra-high vacuum",
        noise_model="Thermal cantilever Brownian fluctuation + capacitive bridge drift",
        expected_failure_modes=("Patch potential electrostatic fluctuations at z > 2 um",),
        expected_limits={"z_to_zero": "0.0 N (equilibrium mechanical contact balance)", "z_to_infinity": "0.0 N"},
        expected_invariants=("F(0) = 0 restoring equilibrium", "d^2U/dz^2 > 0 at z=0"),
        pass_criteria=("Net force finite on [0, inf)", "Contact stiffness k_eff > 0"),
        fail_criteria=("Infinite force divergence at z -> 0", "Unstable contact equilibrium"),
        domain_restriction_criteria=("Restrict to z < 2 um when electrostatic patch potentials dominate",),
    )


def create_butler_volmer_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Butler-Volmer Mass-Transport Overpotential Saturation."""
    return ScientificBenchmarkContract(
        model_id="BV-REF-001",
        baseline_model="j = j_0 * (exp(alpha_a * f * eta) - exp(-alpha_c * f * eta))",
        refined_model="j_reg = (j_BV * j_lim) / (j_lim + abs(j_BV))",
        mechanism="Mixed activation-diffusion concentration polarization (Koutecky-Levich boundary layer)",
        validated_domain={"overpotential_volts": (-2.0, 2.0)},
        predicted_observables=("current_density_a_m2",),
        prediction_intervals={"current_density_a_m2": (-5000.0, 5000.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="fals_butler_volmer_sealed_data_hash",
        data_generation_method="Rotating disk electrode (RDE) voltammetry",
        noise_model="Potentiostat ohmic drop iR-compensation residual noise",
        expected_failure_modes=("Electrode passivation and gas evolution bubble blanketing",),
        expected_limits={"eta_to_plus_inf": "j_lim", "eta_to_minus_inf": "-j_lim"},
        expected_invariants=("Asymptotic current saturation |j| <= j_lim", "Zero current at zero overpotential j(0)=0"),
        pass_criteria=("Current saturates strictly at j_lim", "Monotonic current-voltage response"),
        fail_criteria=("Exponential current explosion exceeding j_lim", "Negative differential resistance"),
        domain_restriction_criteria=("Restrict domain to |eta| < 1.5 V if solvent electrolysis occurs",),
    )


def create_prandtl_glauert_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Prandtl-Glauert Transonic Compressibility Regularization."""
    return ScientificBenchmarkContract(
        model_id="PG-REF-001",
        baseline_model="Cp = Cp_0 / sqrt(1 - M^2)",
        refined_model="Cp_reg = Cp_0 / ((1 - M^2)^2 + eps^2)^(1/4)",
        mechanism="Van Dyke / transonic small-disturbance non-linear wave compression relief",
        validated_domain={"mach_number": (0.0, 0.98)},
        predicted_observables=("pressure_coefficient_cp",),
        prediction_intervals={"pressure_coefficient_cp": (-10.0, 0.0)},
        falsification_thresholds={"max_relative_error": 0.06},
        locked_test_data_hash="fals_prandtl_glauert_sealed_data_hash",
        data_generation_method="Transonic wind tunnel surface pressure taps",
        noise_model="Transonic shock buffet boundary layer unsteady noise",
        expected_failure_modes=("Strong detached normal shock waves (M > 1.15)",),
        expected_limits={"M_to_one": "Cp_0 / sqrt(eps)", "M_to_zero": "Cp_0"},
        expected_invariants=("Finite Cp across all M <= 1", "Subsonic recovery to Cp_0 at M=0"),
        pass_criteria=("Finite transonic peak Cp", "Relative pre-shock distortion < 1.5% at M=0.85"),
        fail_criteria=("Division by zero at M = 1.0", "Unphysical positive pressure inversion"),
        domain_restriction_criteria=("Restrict domain to M <= 0.95 for thick airfoils (tau > 0.12)",),
    )


def create_gibbs_thomson_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Gibbs-Thomson Nanoparticle Melting Depression."""
    return ScientificBenchmarkContract(
        model_id="GT-REF-001",
        baseline_model="Delta_Tm / Tm0 = (2 * gamma_sl) / (rho_s * L_f * r)",
        refined_model="Tm = Tm0 * exp(-C / (r + r0))",
        mechanism="Discrete atomic cluster coordination shell saturation preserving Third Law of Thermodynamics",
        validated_domain={"nanoparticle_radius_m": (0.5e-9, 100e-9)},
        predicted_observables=("melting_temperature_k",),
        prediction_intervals={"melting_temperature_k": (50.0, 1500.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="fals_gibbs_thomson_sealed_data_hash",
        data_generation_method="Nanocalorimetry (differential scanning calorimetry) of monodisperse clusters",
        noise_model="Substrate interaction wetting heat correction variance",
        expected_failure_modes=("Non-spherical faceted icosahedral cluster reconstructions",),
        expected_limits={"r_to_zero": "Tm0 * exp(-C/r0) > 0 K", "r_to_infinity": "Tm0"},
        expected_invariants=("Strict physical positivity Tm > 0 K", "Monotonic melting point depression with 1/r"),
        pass_criteria=("No negative Kelvin temperatures", "Asymptotic approach to bulk melting point"),
        fail_criteria=("Prediction of Tm <= 0 K", "Non-monotonic melting curve"),
        domain_restriction_criteria=("Restrict to r > 0.8 nm for strongly faceted noble metals",),
    )
