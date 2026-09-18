"""Tests for APEDP External Benchmark Ingestion & Empirical Ground-Truth Adapters."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.epistemic import (
    EvidenceMaturityLevel,
    LifecycleManager,
    ModelEvidenceRecord,
    ModelInterventionClass,
)
from apedp.gated_assurance import FailureSeverity, GatedScientificAssuranceEngine
from apedp.ingestion import (
    AIFeynmanBenchmarkSuite,
    FeynmanBenchmarkCase,
    NACA0012TransonicBenchmark,
    NanocrystallineMicrocompressionBenchmark,
)


def test_ai_feynman_suite_schema_and_dimensional_validation():
    """Validates that AI Feynman benchmark cases have valid schemas and exact dimensional consistency."""
    cases = AIFeynmanBenchmarkSuite.list_cases()
    assert len(cases) >= 5
    assert "I.6.20a" in cases
    assert "I.12.1" in cases
    assert "I.15.3x" in cases
    assert "I.27.6" in cases
    assert "I.34.8" in cases

    for case_id in cases:
        case = AIFeynmanBenchmarkSuite.get_case(case_id)
        assert isinstance(case, FeynmanBenchmarkCase)
        assert case.equation_id == case_id
        assert len(case.variables) > 0
        assert len(case.variable_ranges) == len(case.variables)
        assert len(case.variable_units) == len(case.variables)

        # Dimensional homogeneity verification
        is_dimensionally_valid = AIFeynmanBenchmarkSuite.audit_case_dimensions(case_id)
        assert is_dimensionally_valid is True, f"Feynman case '{case_id}' failed dimensional validation."


def test_ai_feynman_data_generation_with_noise():
    """Validates reproducible data generation with clean and noisy signals."""
    case = AIFeynmanBenchmarkSuite.get_case("I.12.1")

    # Clean dataset (sigma = 0.0)
    features_clean, targets_clean, hash_clean = case.generate_data(n_samples=100, noise_sigma=0.0, seed=42)
    assert features_clean.shape == (100, 2)
    assert targets_clean.shape == (100,)
    assert len(hash_clean) == 64

    # Exact physical evaluation check (F = mu * N)
    expected_f = features_clean[:, 0] * features_clean[:, 1]
    np.testing.assert_allclose(targets_clean, expected_f, rtol=1e-6)

    # Noisy dataset (sigma = 0.05, 5% noise)
    _features_noisy, targets_noisy, hash_noisy = case.generate_data(n_samples=100, noise_sigma=0.05, seed=42)
    assert hash_clean != hash_noisy
    assert np.mean(np.abs(targets_noisy - targets_clean)) > 0.0


def test_naca0012_transonic_wind_tunnel_benchmark():
    """Validates empirical falsification against real NASA / AGARD wind-tunnel measurements."""
    mach, cp, unc, data_hash = NACA0012TransonicBenchmark.get_dataset()
    assert len(mach) == 11
    assert len(cp) == 11
    assert len(unc) == 11
    assert len(data_hash) == 64

    # The transonic suction peak is finite (-1.38), NOT infinite
    assert np.all(cp < 0.0)
    assert np.min(cp) >= -2.0

    lm = LifecycleManager()
    engine = GatedScientificAssuranceEngine(lifecycle_manager=lm)

    record = ModelEvidenceRecord(
        model_id="PG-EXP-001",
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=("PG_Classical", "PG-EXP-001"),
        intervention_class=ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION,
        evidence_maturity=EvidenceMaturityLevel.EMPIRICALLY_SUPPORTED,
        domain="Applied Aerodynamics",
        mechanism_family="transonic_small_disturbance",
        equation_str="Cp = Cp_0 / ((1 - M^2)^2 + (0.70 * eps * M^14)^2)^(1/4)",
        variables=("M",),
        parameters={"Cp_0": -0.52, "eps": 0.12 ** (2.0 / 3.0)},
        derivation_basis="Van Dyke transonic non-linear wave compression relief with critical Mach scaling",
        # Every quantity here is a dimensionless coefficient or Mach number, so the
        # homogeneity check has a definite answer rather than an assumption.
        mathematical_assurance={
            "variable_units": {"Cp": "dimensionless", "Cp_0": "dimensionless", "eps": "dimensionless", "M": "dimensionless"},
            "expected_output_unit": "dimensionless",
        },
        mathematical_guarantees=("Strictly finite across all Mach M <= 1.0",),
        numerical_guarantees=("No division by zero at M = 1.0",),
        physical_constraints=("Cp < 0 (suction surface)",),
        assumptions=("Attached boundary layer prior to strong shock detachment",),
        known_failure_modes=("Strong detached normal shocks at M > 1.15",),
        validated_domain={"mach_number": (0.0, 0.98)},
        extrapolation_limits={"mach_number": (0.0, 1.20)},
        falsification_target="Transonic shock singularity",
        falsification_tests=("NASA AGARD wind tunnel taps",),
        failure_interpretation="Van Dyke transonic softening failure",
        # The declared asymptote must state the limit *this* equation attains. The
        # softening term here carries the coefficient 0.70, so M -> 1 gives
        # Cp_0 / sqrt(0.70 * eps) = -1.2600, not Cp_0 / sqrt(eps) = -1.0542. The
        # assertions at the end of test_naca0012_transonic_wind_tunnel_benchmark
        # measure that limit against the model and hold the two together.
        symbolic_asymptotics={"M_to_one": "Cp_0 / sqrt(0.70 * eps)"},
    )

    bench_result = NACA0012TransonicBenchmark.run_benchmark_audit(
        engine=engine,
        candidate_record=record,
        thickness_ratio=0.12,
    )

    assert bench_result.model_id == "PG-EXP-001"
    assert bench_result.prediction_coverage >= 0.90
    assert bench_result.worst_severity == FailureSeverity.PASS
    assert bench_result.epistemic_consequence == "CONTRACT_VALIDATED"
    assert lm.get_state("PG-EXP-001") == "Contract_Validated"

    # The declared transonic limit is measured against the model, and the two must
    # agree: the gate evaluates M along an approach sequence to 1.0 and reports the
    # residual. A declaration that its own equation does not attain is a finding,
    # not a formality -- this fixture previously declared Cp_0 / sqrt(eps), which
    # omits the 0.70 coefficient and is 19.5% (1/sqrt(0.70)) away from the model.
    assert bench_result.asymptotic_constraints == "PASS"
    asymptotes = bench_result.details["measurement_evidence"]["asymptote_checks"]
    assert [check["status"] for check in asymptotes] == ["CONSISTENT"]
    declared, measured = asymptotes[0]["declared_limit"], asymptotes[0]["measured_limit"]
    assert declared == pytest.approx(measured, rel=0.05)
    assert measured == pytest.approx(-1.2600, abs=5e-3)
    assert bench_result.improvement_metric.i_physicality == 1.0


def test_nanocrystalline_microcompression_benchmark():
    """Validates empirical falsification against real experimental micropillar compression data."""
    d, sigma, unc, data_hash = NanocrystallineMicrocompressionBenchmark.get_dataset()
    assert len(d) == 11
    assert len(sigma) == 11
    assert len(unc) == 11
    assert len(data_hash) == 64

    # Real experimental inverse Hall-Petch softening: sigma at 3 nm is lower than sigma at 12 nm!
    idx_12nm = 7  # 12 nm: 980 MPa
    idx_3nm = 10  # 3 nm: 620 MPa
    assert sigma[idx_3nm] < sigma[idx_12nm]

    lm = LifecycleManager()
    engine = GatedScientificAssuranceEngine(lifecycle_manager=lm)

    record = ModelEvidenceRecord(
        model_id="HP-EXP-001",
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=("HP_Classical", "HP-EXP-001"),
        intervention_class=ModelInterventionClass.MECHANISTIC_REPLACEMENT,
        evidence_maturity=EvidenceMaturityLevel.EMPIRICALLY_SUPPORTED,
        domain="Nanocrystalline Metallurgy",
        mechanism_family="grain_boundary_sliding",
        # The evaluator uses ``k_HP`` and clamps at ``sigma_theoretical``; the
        # declaration now says the same thing in the parser's own syntax instead of
        # referring to an undeclared ``k`` and a square-bracketed grouping.
        equation_str="sigma = min(sigma_0 + (k_HP / sqrt(d)) / (1 + (dc/d)^q), sigma_theoretical)",
        variables=("d",),
        parameters={"sigma_0": 90.0e6, "k_HP": 0.26867e6, "dc": 25.0e-9, "q": 0.95, "sigma_theoretical": 1.5e9},
        derivation_basis="Dislocation loop exhaustion coupled to grain boundary sliding threshold",
        mathematical_assurance={
            "variable_units": {
                "sigma": "pascal",
                "sigma_0": "pascal",
                "k_HP": "Pa*m**0.5",
                "d": "meter",
                "dc": "meter",
                "q": "dimensionless",
                "sigma_theoretical": "pascal",
            },
            "expected_output_unit": "pascal",
        },
        mathematical_guarantees=("Strictly finite yield stress on d in [0, inf)",),
        numerical_guarantees=("No division by zero as d -> 0",),
        physical_constraints=("Non-negative yield stress: sigma_y > 0", "sigma_y <= sigma_theoretical"),
        assumptions=("Grain boundary sliding dominant below dc ~ 25 nm",),
        known_failure_modes=("High-temperature Coble creep at T > 0.4 Tm",),
        validated_domain={"grain_size_m": (3e-9, 100e-6)},
        extrapolation_limits={"grain_size_m": (1e-9, 1e-3)},
        falsification_target="Nanoscale softening inversion",
        falsification_tests=("Micropillar compression",),
        failure_interpretation="Constitutive softening failure",
        symbolic_asymptotics={"d_to_zero": "sigma_0", "d_to_inf": "sigma_0"},
    )

    bench_result = NanocrystallineMicrocompressionBenchmark.run_benchmark_audit(
        engine=engine,
        candidate_record=record,
    )

    assert bench_result.model_id == "HP-EXP-001"
    assert bench_result.prediction_coverage >= 0.90
    assert bench_result.worst_severity == FailureSeverity.PASS
    assert bench_result.epistemic_consequence == "CONTRACT_VALIDATED"
    assert lm.get_state("HP-EXP-001") == "Contract_Validated"
