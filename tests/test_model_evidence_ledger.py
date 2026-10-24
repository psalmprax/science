from dataclasses import FrozenInstanceError

import pytest
import sympy as sp

from apedp.epistemic import (
    EvidenceMaturityLevel,
    LifecycleManager,
    ModelEvidenceRecord,
    ModelInterventionClass,
    RefinementOperator,
)


def test_model_evidence_record_schema_and_serialization():
    """Validates the persistent ModelEvidenceRecord schema, guarantees, and assurance partitions."""
    record = ModelEvidenceRecord(
        model_id="M_HP_02",
        model_version="2.0.0",
        parent_model_id="M_HP_01",
        model_lineage=["M_HP_00_Classical", "M_HP_01_Heuristic_Cutoff", "M_HP_02"],
        intervention_class=ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION,
        evidence_maturity=EvidenceMaturityLevel.FALSIFICATION_SURVIVED,
        domain="Nanocrystalline Metallurgy",
        mechanism_family="dislocation_pileup_exhaustion_and_boundary_sliding",
        equation_str="sigma_y(d) = min(sigma_0 + (k_HP / sqrt(d + d0)) / (1 + (dc / d)^2), sigma_theoretical)",
        variables=["d"],
        parameters={"sigma_0": 70e6, "k_HP": 0.74e6, "dc": 10e-9, "d0": 1e-10, "sigma_theoretical": 3e9},
        derivation_basis="Rational softening continuation coupling Hall-Petch pileup to grain boundary sliding threshold",
        mathematical_guarantees=[
            "Strictly finite on d in [0, inf)",
            "Recovers classical d^(-1/2) scaling at large d (d >> dc)",
            "Upper bounded by lattice theoretical shear limit sigma_theoretical"
        ],
        numerical_guarantees=["No floating point overflow/underflow or division by zero on d >= 0"],
        physical_constraints=["Non-negative yield stress: sigma_y > 0 for all physical d"],
        assumptions=[
            "Grain boundary sliding becomes active below dc ~ 10 nm",
            "Homogeneous grain morphology without preferred texture orientation"
        ],
        known_failure_modes=["High-temperature diffusional Coble/Nabarro-Herring creep regimes"],
        validated_domain={"d_meters": (3e-9, 100e-6)},
        extrapolation_limits={"d_meters": (1e-9, 1e-3)},
        falsification_target="Yield stress inversion slope and saturation peak at nanoscale d in [3 nm, 30 nm]",
        falsification_tests=["In-situ micropillar compression testing under TEM across nanograin sizes"],
        failure_interpretation="Candidate transition function inadequate; requires discrete dislocation-dynamics model",
        software_assurance={"pytest_status": "PASSED", "ruff_linter": "CLEAN", "deterministic_replay": True},
        # The declaration claimed ``dimensional_homogeneity: PASSED`` as prose with no
        # units to check it against; the units below let the gate measure it.
        mathematical_assurance={
            "symbolic_limit_d_zero": "finite",
            "dimensional_homogeneity": "PASSED",
            "variable_units": {
                "sigma_y": "pascal",
                "sigma_0": "pascal",
                "k_HP": "Pa*m**0.5",
                "d": "meter",
                "d0": "meter",
                "dc": "meter",
                "sigma_theoretical": "pascal",
            },
            "expected_output_unit": "pascal",
        },
        scientific_assurance={"out_of_sample_tested": False, "competing_models_evaluated": 3},
        symbolic_asymptotics={"d -> 0": "sigma_0 + 0", "d -> inf": "k_HP / sqrt(d)"},
        uncertainty_budget={"combined_uncertainty_pa": 4.2e6, "coverage_factor_k": 2.0},
        epistemic_status="SUPPORTED_WITH_DOMAIN_RESTRICTION"
    )

    data = record.to_dict()
    assert data["model_id"] == "M_HP_02"
    assert data["intervention_class"] == "PHENOMENOLOGICAL_EXTENSION"
    assert data["evidence_maturity"] == "FALSIFICATION_SURVIVED"
    assert "sigma_theoretical" in data["equation_str"]
    assert len(data["mathematical_guarantees"]) == 3
    assert data["falsification_target"] != ""
    assert data["failure_interpretation"] != ""
    assert data["epistemic_status"] == "SUPPORTED_WITH_DOMAIN_RESTRICTION"


def test_lifecycle_manager_refined_scientific_transitions():
    """Validates the nuanced scientific lifecycle: Validated -> Challenged -> Domain_Restricted."""
    lifecycle = LifecycleManager()
    model_name = "Model_Turbulence_Dissipation_v1"

    # 1. Register & promote to Candidate
    lifecycle.register_model(model_name, "Draft")
    lifecycle.promote(model_name, "Candidate", checks_passed=True, evidence_id="TOURNAMENT_ENTRY")
    assert lifecycle.get_state(model_name) == "Candidate"

    # 2. Promote to Validated
    lifecycle.promote(model_name, "Validated", checks_passed=True, evidence_id="NUMERICAL_STRESS_PASSED")
    assert lifecycle.get_state(model_name) == "Validated"

    # 3. New boundary evidence challenges the model
    lifecycle.promote(model_name, "Challenged", checks_passed=True, evidence_id="DNS_WALL_ANOMALY")
    assert lifecycle.get_state(model_name) == "Challenged"

    # 4. Model is not discarded, but refined with domain restriction
    lifecycle.promote(model_name, "Domain_Restricted", checks_passed=True, evidence_id="RESTRICT_Y_PLUS_GT_5")
    assert lifecycle.get_state(model_name) == "Domain_Restricted"


def test_symbolic_asymptotics_audit():
    """Audits symbolic asymptotic limits for competing candidate continuations."""
    d, dc, k, s0 = sp.symbols("d dc k s0", positive=True)

    # Candidate 1: Rational transition
    expr_rational = s0 + (k / sp.sqrt(d)) / (1 + (dc / d)**2)
    lim_d0_rat = sp.limit(expr_rational, d, 0)
    lim_dinf_rat = sp.limit(expr_rational, d, sp.oo)

    assert lim_d0_rat == s0  # Smoothly saturates to friction stress at d -> 0!
    assert lim_dinf_rat == s0  # At inf, k/sqrt(d) vanishes, leaving s0

    # Candidate 2: Exponential transition
    expr_exp = s0 + (k / sp.sqrt(d)) * (1 - sp.exp(-(d / dc)**2))
    lim_d0_exp = sp.limit(expr_exp, d, 0)
    assert lim_d0_exp == s0  # Also finite at d -> 0

    # Classical unregularized: divergent
    expr_class = s0 + k / sp.sqrt(d)
    lim_d0_class = sp.limit(expr_class, d, 0)
    assert lim_d0_class == sp.oo


def test_refinement_operators_enum():
    """Validates formal refinement operators."""
    assert RefinementOperator.REGULARIZE.value == "R_regularize"
    assert RefinementOperator.MECHANISTIC.value == "R_mechanistic"
    assert RefinementOperator.SELECT.value == "R_select"
    assert RefinementOperator.GROUND.value == "R_ground"
    assert RefinementOperator.FALSIFY.value == "R_falsify"


def test_immutable_record_hashing_and_freeze():
    """Validates dataclass immutability and deterministic cryptographic hashing."""
    from apedp.epistemic import (
        EvidenceMaturityLevel,
        EvidenceVector,
        FalsificationContract,
        ModelEvidenceRecord,
        ModelInterventionClass,
        ScientificProvenanceChain,
        UncertaintyDecomposition,
    )

    vector = EvidenceVector(
        analytical_score=0.95,
        numerical_score=1.0,
        adversarial_score=1.0,
        empirical_score=0.88,
        independent_replication_score=0.0,
        derived_maturity_level=EvidenceMaturityLevel.FALSIFICATION_SURVIVED,
    )

    uncertainty = UncertaintyDecomposition(
        measurement_uncertainty=0.02,
        parameter_uncertainty=0.015,
        model_discrepancy=0.03,
        structural_uncertainty=0.01,
        extrapolation_uncertainty=0.08,
        epistemic_total=0.055,
        aleatoric_total=0.02,
        combined_expanded_uncertainty=0.11,
    )

    contract = FalsificationContract(
        falsification_id="FALS-HP-001",
        target_observable="d(sigma)/d(1/sqrt(d))",
        expected_interval=(-0.5, 0.0),
        domain_interval=(3e-9, 15e-9),
        decision_threshold=0.0,
        null_hypothesis="Slope remains positive indicating continued hardening",
        alternative_hypothesis="Slope inverts negative indicating grain-boundary sliding softening",
        required_sample_size=20,
        statistical_test="Student's two-tailed t-test with alpha=0.01",
        preregistered=True,
        failure_interpretation="Candidate transition function inadequate under tested conditions",
        target_unit="pascal * meter^0.5",
    )

    provenance = ScientificProvenanceChain(
        code_commit="git:8f3a9e2",
        dataset_hashes=("sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",),
        environment_hash="sha256:d8b209",
        random_seed=42,
        execution_id="EXEC-2026-09-18-001",
        timestamp="2026-09-18T00:00:00Z",
        agent_id="APEDP_SDI_ENGINE",
        parent_record_hash="sha256:0000000000000000000000000000000000000000000000000000000000000000",
    )

    record = ModelEvidenceRecord(
        model_id="M_HP_03",
        model_version="3.0.0",
        parent_model_id="M_HP_02",
        model_lineage=("M_HP_00", "M_HP_01", "M_HP_02", "M_HP_03"),
        intervention_class=ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION,
        secondary_intervention_classes=(ModelInterventionClass.NUMERICAL_REGULARIZATION,),
        evidence_maturity=EvidenceMaturityLevel.FALSIFICATION_SURVIVED,
        domain="Nanocrystalline Metallurgy",
        mechanism_family="dislocation_pileup_exhaustion_and_boundary_sliding",
        equation_str="sigma_y(d) = min(sigma_0 + (k_HP / sqrt(d + d0)) / (1 + (dc / d)^2), sigma_theoretical)",
        variables=("d",),
        parameters={"sigma_0": 70e6, "k_HP": 0.74e6, "dc": 10e-9, "d0": 1e-10, "sigma_theoretical": 3e9},
        derivation_basis="Rational softening continuation coupling Hall-Petch pileup to grain boundary sliding threshold",
        mathematical_guarantees=("Strictly finite on d in [0, inf)", "Upper bounded by sigma_theoretical"),
        numerical_guarantees=("No floating point overflow",),
        physical_constraints=("sigma_y > 0",),
        assumptions=("Grain boundary sliding dominant below dc ~ 10 nm",),
        known_failure_modes=("Coble creep at T > 0.4 Tm",),
        validated_domain={"d_meters": (3e-9, 100e-6)},
        extrapolation_limits={"d_meters": (1e-9, 1e-3)},
        falsification_target="Nanoscale softening inversion",
        falsification_tests=("TEM in-situ compression",),
        failure_interpretation="Candidate transition function inadequate",
        evidence_vector=vector,
        uncertainty_decomposition=uncertainty,
        falsification_contract=contract,
        provenance_chain=provenance,
        mathematical_assurance={
            "variable_units": {
                "sigma_y": "pascal",
                "sigma_0": "pascal",
                "k_HP": "Pa*m**0.5",
                "d": "meter",
                "d0": "meter",
                "dc": "meter",
                "sigma_theoretical": "pascal",
            },
            "expected_output_unit": "pascal",
        },
    )

    # 1. Immutability verification: attempting to overwrite attribute raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        record.model_id = "MUTATED"

    # 2. Cryptographic content hash computation is deterministic
    hash1 = record.compute_record_hash()
    hash2 = record.compute_record_hash()
    assert isinstance(hash1, str) and len(hash1) == 64
    assert hash1 == hash2


def test_anti_self_confirmation_vault_isolation():
    """Validates that DataPartitionVault blocks training/selection access to locked falsification data."""
    from apedp.epistemic import DataPartitionVault

    vault = DataPartitionVault(
        development_data_hash="dev_hash_12345",
        selection_data_hash="sel_hash_67890",
        falsification_data_hash="fals_hash_abcde",
        locked_external_test_hash="ext_hash_fghij",
        is_falsification_locked=True,
    )

    # Legitimate access passes
    assert vault.verify_isolation("dev_hash_12345", "development") is True
    assert vault.verify_isolation("sel_hash_67890", "selection") is True
    assert vault.verify_isolation("fals_hash_abcde", "falsification") is True

    # Attempting to access falsification data during training or selection raises PermissionError
    with pytest.raises(PermissionError) as exc_info:
        vault.verify_isolation("fals_hash_abcde", "development")
    assert "ANTI-SELF-CONFIRMATION VIOLATION" in str(exc_info.value)


def test_prior_art_auditor_symbolic_isomorphism():
    """Validates prior-art detection on mathematically identical but syntactically distinct forms."""
    from apedp.epistemic import PriorArtAuditor

    # Test 1: y = a*x / (b + x) vs y = a / (1 + b/x)
    is_iso, desc = PriorArtAuditor.check_symbolic_equivalence(
        "a * x / (b + x)",
        "a / (1 + b / x)"
    )
    assert is_iso is True
    assert "IDENTICAL_PRIOR_ART" in desc

    # Test 2: Genuinely distinct functional forms
    is_distinct, desc2 = PriorArtAuditor.check_symbolic_equivalence(
        "a * x / (b + x)",
        "a * exp(-b * x)"
    )
    assert is_distinct is False
    assert "DISTINCT" in desc2

