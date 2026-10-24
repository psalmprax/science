"""Tests for APEDP Gated Scientific Assurance & Blind Falsification Harness."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from apedp.epistemic import (
    DataPartitionVault,
    DiscoveryMode,
    EvidenceMaturityLevel,
    LifecycleManager,
    ModelEpistemicStatus,
    ModelEvidenceRecord,
    ModelInterventionClass,
    TestableModelPackage,
    VerificationGate,
)
from apedp.gated_assurance import (
    BlindChallengeHarness,
    DiagnosticReasoningBenchmark,
    EpistemicActionStance,
    FailureSeverity,
    GatedScientificAssuranceEngine,
    ScientificBenchmarkContract,
    ScientificFeasibilityConfusionMatrix,
    ScientificImprovementMetric,
    SpecificationFeasibility,
    create_butler_volmer_benchmark_contract,
    create_casimir_benchmark_contract,
    create_gibbs_thomson_benchmark_contract,
    create_hall_petch_benchmark_contract,
    create_prandtl_glauert_benchmark_contract,
    map_feasibility_to_stance,
)


def test_canonical_benchmark_contracts_schema():
    """Validates the 5 frontier domain benchmark contracts schemas and serialization."""
    contracts = [
        create_hall_petch_benchmark_contract(),
        create_casimir_benchmark_contract(),
        create_butler_volmer_benchmark_contract(),
        create_prandtl_glauert_benchmark_contract(),
        create_gibbs_thomson_benchmark_contract(),
    ]

    expected_ids = ["HP-REF-001", "CAS-REF-001", "BV-REF-001", "PG-REF-001", "GT-REF-001"]

    for contract, exp_id in zip(contracts, expected_ids, strict=True):
        assert isinstance(contract, ScientificBenchmarkContract)
        assert contract.model_id == exp_id
        assert len(contract.baseline_model) > 0
        assert len(contract.refined_model) > 0
        assert len(contract.mechanism) > 0
        assert len(contract.validated_domain) > 0
        assert len(contract.predicted_observables) > 0
        assert len(contract.prediction_intervals) > 0
        assert len(contract.falsification_thresholds) > 0
        assert len(contract.expected_failure_modes) > 0
        assert len(contract.pass_criteria) > 0
        assert len(contract.fail_criteria) > 0
        assert len(contract.domain_restriction_criteria) > 0

        # Serialization to dictionary
        d = contract.to_dict()
        assert d["model_id"] == exp_id
        assert "locked_test_data_hash" in d


def test_failure_severity_distinctions():
    """Validates failure severity categories and scientific meaning."""
    assert FailureSeverity.PASS.value == "PASS"
    assert FailureSeverity.WARNING.value == "WARNING"
    assert FailureSeverity.DOMAIN_VIOLATION.value == "DOMAIN_VIOLATION"
    assert FailureSeverity.PREDICTIVE_FAILURE.value == "PREDICTIVE_FAILURE"
    assert FailureSeverity.PHYSICAL_INVARIANT_VIOLATION.value == "PHYSICAL_INVARIANT_VIOLATION"
    assert FailureSeverity.MECHANISTIC_CONTRADICTION.value == "MECHANISTIC_CONTRADICTION"
    assert FailureSeverity.IMPLEMENTATION_FAILURE.value == "IMPLEMENTATION_FAILURE"


def test_multidimensional_scientific_improvement_metric():
    """Validates that multidimensional scientific improvement refuses models that sacrifice physics for fit."""
    # Case 1: Excellent fit, but broken physicality and stability -> REJECTED
    unphysical_improvement = ScientificImprovementMetric(
        i_fit=0.99,
        i_stability=0.20,     # Singular pole introduced
        i_physicality=0.30,   # Violates energy conservation
        i_generality=0.50,
        i_complexity=0.10,   # Massive overparameterization
        i_falsifiability=0.40,
    )
    assert unphysical_improvement.is_net_positive is False

    # Case 2: Genuine scientific refinement (balanced, physically bounded)
    genuine_improvement = ScientificImprovementMetric(
        i_fit=0.88,
        i_stability=0.95,
        i_physicality=1.00,
        i_generality=0.80,
        i_complexity=0.85,
        i_falsifiability=0.90,
    )
    assert genuine_improvement.is_net_positive is True
    assert "is_net_positive" in genuine_improvement.to_dict()


def test_diagnostic_reasoning_benchmark():
    """Validates benchmarking of scientific reasoning behind a model failure and repair."""
    diag = DiagnosticReasoningBenchmark(
        initial_failure="contact equilibrium condition insufficient",
        expected_diagnosis="first-order force condition does not establish local stability",
        actual_diagnosis="first-order force condition does not establish local stability",
        expected_refinement="potential construction + curvature analysis",
        actual_refinement="potential construction + curvature analysis",
        diagnosis_correctness=True,
        refinement_correctness=True,
        reasoning_notes="Curvature k_eff = d^2U/dz^2 > 0 proves stable contact restoring equilibrium.",
    )
    assert diag.diagnosis_correctness is True
    assert diag.refinement_correctness is True
    d = diag.to_dict()
    assert d["diagnosis_correctness"] is True
    assert "k_eff" in d["reasoning_notes"]


def test_blind_challenge_harness_leak_prevention_and_evaluation():
    """Validates anti-self-confirmation isolation and blind scoring."""
    vault = DataPartitionVault(
        development_data_hash="dev_hash_1",
        selection_data_hash="sel_hash_2",
        falsification_data_hash="locked_challenge_hash_3",
        locked_external_test_hash="ext_hash_4",
        is_falsification_locked=True,
    )
    harness = BlindChallengeHarness(vault=vault)

    features = np.linspace(1.0, 10.0, 20).reshape(-1, 1)
    targets = 2.0 * features.flatten()

    harness.register_sealed_challenge(
        dataset_hash="locked_challenge_hash_3",
        features=features,
        targets=targets,
        metadata={"domain": "linear_scaling"},
    )

    contract = ScientificBenchmarkContract(
        model_id="LIN-001",
        baseline_model="y = 2*x",
        refined_model="y = 2*x",
        mechanism="linear proportionality",
        validated_domain={"x": (1.0, 10.0)},
        predicted_observables=("y",),
        prediction_intervals={"y": (0.0, 25.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="locked_challenge_hash_3",
        data_generation_method="synthetic",
        noise_model="none",
        expected_failure_modes=(),
        expected_limits={},
        expected_invariants=(),
        pass_criteria=(),
        fail_criteria=(),
        domain_restriction_criteria=(),
    )

    # 1. Anti-self-confirmation leak protection: dev phase cannot access locked challenge
    with pytest.raises(PermissionError) as exc_info:
        harness.evaluate_blind(
            candidate_evaluator=lambda x: 2.0 * x.flatten(),
            dataset_hash="locked_challenge_hash_3",
            contract=contract,
            caller_phase="development",
        )
    assert "ANTI-SELF-CONFIRMATION VIOLATION" in str(exc_info.value)

    # 2. Legitimate falsification evaluation succeeds blindly
    result = harness.evaluate_blind(
        candidate_evaluator=lambda x: 2.0 * x.flatten(),
        dataset_hash="locked_challenge_hash_3",
        contract=contract,
        caller_phase="falsification",
    )
    assert result["status"] == FailureSeverity.PASS
    assert result["coverage"] == 1.0
    assert result["mean_relative_error"] == 0.0

    # 2b. The blind evaluation must not hand the sealed labels back to the caller.
    assert "predictions" not in result
    assert "targets" not in result

    # 2c. The sealed partition is a copy: mutating the caller's arrays afterwards
    #     cannot retroactively change what was scored.
    features[:] = 0.0
    targets[:] = 0.0
    after_mutation = harness.evaluate_blind(
        candidate_evaluator=lambda x: 2.0 * x.flatten(),
        dataset_hash="locked_challenge_hash_3",
        contract=contract,
        caller_phase="falsification",
    )
    assert after_mutation["coverage"] == 1.0
    assert after_mutation["mean_relative_error"] == 0.0

    # 3. Candidate evaluation crash triggers IMPLEMENTATION_FAILURE
    def faulty_evaluator(x: np.ndarray) -> np.ndarray:
        raise ValueError("Division by zero in model subroutine")

    crash_result = harness.evaluate_blind(
        candidate_evaluator=faulty_evaluator,
        dataset_hash="locked_challenge_hash_3",
        contract=contract,
        caller_phase="falsification",
    )
    assert crash_result["status"] == FailureSeverity.IMPLEMENTATION_FAILURE
    assert "crashed" in crash_result["error"]


def test_operational_gate_verifies_recomputed_content_hash():
    """The operational gate must recompute the digest, not trust a 64-char string."""
    from apedp.gated_assurance import OperationalGateAuditor

    def make_record(record_hash: str) -> ModelEvidenceRecord:
        return ModelEvidenceRecord(
            model_id="HASH-001",
            model_version="1.0.0",
            parent_model_id=None,
            model_lineage=("HASH-001",),
            intervention_class=ModelInterventionClass.NUMERICAL_REGULARIZATION,
            evidence_maturity=EvidenceMaturityLevel.NUMERICALLY_STRESS_TESTED,
            domain="Numerical Analysis",
            mechanism_family="stabilized_kernel",
            equation_str="y = sin(x)",
            variables=("x",),
            parameters={},
            derivation_basis="Sine approximation",
            mathematical_guarantees=(),
            numerical_guarantees=(),
            physical_constraints=(),
            assumptions=(),
            known_failure_modes=(),
            validated_domain={"x": (0.0, 10.0)},
            extrapolation_limits={"x": (-10.0, 20.0)},
            falsification_target="",
            falsification_tests=(),
            failure_interpretation="",
            record_hash=record_hash,
        )

    sample_inputs = np.linspace(0.0, 10.0, 50).reshape(-1, 1)

    def audit(record: ModelEvidenceRecord):
        return OperationalGateAuditor.audit_readiness(
            candidate_record=record,
            eval_fn=lambda x: np.sin(x),
            sample_inputs=sample_inputs,
        )

    # A correctly sealed record passes the reproducibility check.
    sealed = make_record("")
    sealed = replace(sealed, record_hash=sealed.compute_record_hash())
    assert sealed.verify_record_hash() is True
    assert audit(sealed).reproducibility_hash_verified is True

    # A plausible-looking 64-character string is not a hash of anything.
    forged = audit(make_record("a" * 64))
    assert forged.reproducibility_hash_verified is False
    assert any("Content hash mismatch" in req for req in forged.unmet_requirements)

    # An unsealed record is reported as missing rather than silently passing.
    unsealed = audit(make_record(""))
    assert unsealed.reproducibility_hash_verified is False
    assert any("hash missing" in req for req in unsealed.unmet_requirements)

    # Tampering with content after sealing invalidates the sealed digest.
    tampered = replace(sealed, equation_str="y = cos(x)")
    assert tampered.verify_record_hash() is False
    assert audit(tampered).reproducibility_hash_verified is False


def test_blind_challenge_registration_separated_from_evaluation():
    """Registration is a distinct setup step; evaluation never seals implicitly."""
    harness = BlindChallengeHarness()
    features = np.linspace(1.0, 10.0, 20).reshape(-1, 1)
    targets = 2.0 * features.flatten()

    # Sealing happens during setup and returns an opaque content hash.
    challenge_hash = harness.seal_challenge(features, targets, {"domain": "linear_scaling"})
    assert len(challenge_hash) == 64
    assert harness.has_challenge(challenge_hash) is True
    assert harness.has_challenge("deadbeef" * 8) is False

    # A partition sealed from different data has a different hash.
    other_hash = harness.seal_challenge(features + 1.0, targets + 1.0, {})
    assert other_hash != challenge_hash

    contract = ScientificBenchmarkContract(
        model_id="LIN-SEAL-001",
        baseline_model="y = 2*x",
        refined_model="y = 2*x",
        mechanism="linear proportionality",
        validated_domain={"x": (1.0, 10.0)},
        predicted_observables=("y",),
        prediction_intervals={"y": (0.0, 25.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash=challenge_hash,
        data_generation_method="synthetic",
        noise_model="none",
        expected_failure_modes=(),
        expected_limits={},
        expected_invariants=(),
        pass_criteria=(),
        fail_criteria=(),
        domain_restriction_criteria=(),
    )

    def make_record(model_id: str) -> ModelEvidenceRecord:
        return ModelEvidenceRecord(
            model_id=model_id,
            model_version="1.0.0",
            parent_model_id=None,
            model_lineage=(model_id,),
            intervention_class=ModelInterventionClass.MATHEMATICAL_REGULARIZATION,
            evidence_maturity=EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
            domain="Numerical Analysis",
            mechanism_family="linear_proportionality",
            equation_str="y = 2*x",
            variables=("x",),
            parameters={},
            derivation_basis="linear",
            mathematical_guarantees=(),
            numerical_guarantees=(),
            physical_constraints=(),
            assumptions=(),
            known_failure_modes=(),
            validated_domain={"x": (1.0, 10.0)},
            extrapolation_limits={"x": (0.5, 20.0)},
            falsification_target="",
            falsification_tests=(),
            failure_interpretation="",
            symbolic_asymptotics={"x_to_inf": "unbounded"},
        )

    def run(record: ModelEvidenceRecord, **kwargs):
        # Share the harness so every run is measured against the same sealed partition.
        engine = GatedScientificAssuranceEngine()
        engine.harness = harness
        return engine.run_assurance_gate(
            candidate_record=record,
            contract=contract,
            eval_fn=lambda x: 2.0 * x.flatten(),
            baseline_eval_fn=lambda x: 2.0 * x.flatten(),
            features=features,
            ground_truth=targets,
            **kwargs,
        )

    # 1. An unsealed hash cannot be scored against: evaluation never seals implicitly.
    with pytest.raises(KeyError):
        run(make_record("LIN-UNSEALED"), challenge_hash="0" * 64)

    # 2. A hash that does not describe the supplied partition is refused, so metrics
    #    cannot be reported for one partition while scored against another.
    with pytest.raises(ValueError):
        run(make_record("LIN-MISMATCH"), challenge_hash=other_hash)

    # 3. Pre-sealed evaluation uses the fixed partition and records the mode.
    pre_sealed = run(make_record("LIN-PRESEALED"), challenge_hash=challenge_hash)
    assert pre_sealed.prediction_coverage == 1.0
    assert pre_sealed.details["challenge_mode"] == "pre_sealed"
    assert "predictions" not in pre_sealed.details
    assert "targets" not in pre_sealed.details

    # 4. The convenience path self-seals and says so.
    self_sealed = run(make_record("LIN-SELFSEALED"))
    assert self_sealed.details["challenge_mode"] == "self_sealed"
    assert "predictions" not in self_sealed.details
    assert "targets" not in self_sealed.details


def test_gated_scientific_assurance_end_to_end_hall_petch():
    """End-to-end execution of GatedScientificAssuranceEngine on Hall-Petch Nanoscale Softening."""
    contract = create_hall_petch_benchmark_contract()
    lm = LifecycleManager()

    record = ModelEvidenceRecord(
        model_id=contract.model_id,
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=("M_HP_Classical", contract.model_id),
        intervention_class=ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION,
        evidence_maturity=EvidenceMaturityLevel.FALSIFICATION_SURVIVED,
        domain="Nanocrystalline Metallurgy",
        mechanism_family="dislocation_pileup_exhaustion",
        equation_str=contract.refined_model,
        variables=("d",),
        parameters={"sigma_0": 70e6, "k_HP": 0.74e6, "dc": 10e-9, "d0": 1e-10, "sigma_theoretical": 3.0e9},
        derivation_basis="Rational softening continuation coupling Hall-Petch pileup to grain boundary sliding",
        # Unit metadata, so the gate measures dimensional homogeneity instead of
        # reporting UNVERIFIED. k_HP carries Pa*m**0.5: the law is sigma_0 + k/sqrt(d).
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
        mathematical_guarantees=("Strictly finite on d in [0, inf)", "sigma_y <= sigma_theoretical"),
        numerical_guarantees=("No division by zero or NaN",),
        physical_constraints=("Non-negative yield stress: sigma_y > 0",),
        assumptions=("Grain boundary sliding dominant below dc ~ 10 nm",),
        known_failure_modes=("High-temperature creep at T > 0.4 Tm",),
        validated_domain={"grain_size_m": (3e-9, 100e-6)},
        extrapolation_limits={"grain_size_m": (1e-9, 1e-3)},
        falsification_target="Nanoscale softening inversion",
        falsification_tests=("TEM in-situ compression",),
        failure_interpretation="Candidate softening function inadequate",
        symbolic_asymptotics={"d_to_zero": "sigma_0", "d_to_inf": "sigma_0"},
    )

    engine = GatedScientificAssuranceEngine(lifecycle_manager=lm)

    # Synthetic grain size grid: 3 nm to 50 nm
    d_grid = np.geomspace(3e-9, 50e-9, 30).reshape(-1, 1)

    # Refined model evaluator
    def refined_evaluator(d: np.ndarray) -> np.ndarray:
        d_val = np.maximum(d.flatten(), 1e-11)
        softening = 1.0 / (1.0 + (10e-9 / d_val) ** 2)
        sigma = 70e6 + (0.74e6 / np.sqrt(d_val + 1e-10)) * softening
        return np.minimum(sigma, 3.0e9)

    # Baseline classical evaluator (blows up at small d)
    def baseline_evaluator(d: np.ndarray) -> np.ndarray:
        d_val = np.maximum(d.flatten(), 1e-11)
        return 70e6 + 0.74e6 / np.sqrt(d_val)

    # Ground truth data (with 1% realistic measurement variance)
    ground_truth = refined_evaluator(d_grid) * (1.0 + 0.01 * np.sin(np.arange(len(d_grid))))

    diagnostic = DiagnosticReasoningBenchmark(
        initial_failure="Hall-Petch classical model predicts infinite yield stress as grain size d -> 0",
        expected_diagnosis="Dislocation pileup cannot form in grains smaller than dislocation loop nucleation radius",
        actual_diagnosis="Dislocation pileup cannot form in grains smaller than dislocation loop nucleation radius",
        expected_refinement="Rational crossover to grain boundary sliding with upper atomic shear ceiling",
        actual_refinement="Rational crossover to grain boundary sliding with upper atomic shear ceiling",
        diagnosis_correctness=True,
        refinement_correctness=True,
    )

    bench_result = engine.run_assurance_gate(
        candidate_record=record,
        contract=contract,
        eval_fn=refined_evaluator,
        baseline_eval_fn=baseline_evaluator,
        features=d_grid,
        ground_truth=ground_truth,
        diagnostic=diagnostic,
    )

    # Validations on the machine-readable result
    assert bench_result.model_id == "HP-REF-001"
    assert bench_result.prediction_coverage >= 0.95
    assert bench_result.asymptotic_constraints == "PASS"
    assert bench_result.physical_bounds == "PASS"
    # This record derives equation_str from the Hall-Petch contract, so this also pins
    # the contract's model string to the parser's syntax and to declared units.
    assert bench_result.dimensional_consistency == "PASS"
    assert bench_result.out_of_sample_error == "PASS"
    assert bench_result.critical_failure_count == 0
    assert bench_result.worst_severity == FailureSeverity.PASS
    assert bench_result.epistemic_consequence == "CONTRACT_VALIDATED"
    assert bench_result.improvement_metric.is_net_positive is True

    # Check report formatting matches requested specification
    report = bench_result.format_report()
    assert "FALSIFICATION_RESULT" in report
    assert "Model: HP-REF-001" in report
    assert "Prediction coverage:" in report
    assert "Epistemic consequence:" in report
    assert "CONTRACT_VALIDATED" in report

    # Lifecycle state machine is promoted to Contract_Validated (NOT directly Operational)
    assert lm.get_state("HP-REF-001") == "Contract_Validated"


def test_operational_gate_auditor_separation():
    """Validates that Contract_Validated models require passing OperationalGateAuditor before Operational promotion."""

    lm = LifecycleManager()
    engine = GatedScientificAssuranceEngine(lifecycle_manager=lm)

    # Register candidate and promote to Contract_Validated
    lm.register_model("OP-TEST-001", initial_state="Draft")
    lm.promote("OP-TEST-001", "Candidate", checks_passed=True)
    lm.promote("OP-TEST-001", "Contract_Validated", checks_passed=True)

    record = ModelEvidenceRecord(
        model_id="OP-TEST-001",
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=("OP-TEST-001",),
        intervention_class=ModelInterventionClass.NUMERICAL_REGULARIZATION,
        evidence_maturity=EvidenceMaturityLevel.NUMERICALLY_STRESS_TESTED,
        domain="Numerical Analysis",
        mechanism_family="stabilized_kernel",
        equation_str="y = sin(x)",
        variables=("x",),
        parameters={},
        derivation_basis="Sine approximation",
        mathematical_guarantees=(),
        numerical_guarantees=(),
        physical_constraints=(),
        assumptions=(),
        known_failure_modes=(),
        validated_domain={"x": (0.0, 10.0)},
        extrapolation_limits={"x": (-10.0, 20.0)},
        falsification_target="",
        falsification_tests=(),
        failure_interpretation="",
    )
    # The operational gate recomputes the digest from the record content, so the
    # record must actually be sealed rather than carrying a plausible-looking string.
    record = replace(record, record_hash=record.compute_record_hash())

    sample_inputs = np.linspace(0.0, 10.0, 50).reshape(-1, 1)

    # 1. Unmet requirements (e.g. no drift monitoring, no clamp) -> rejected
    with pytest.raises(ValueError) as exc_info:
        engine.promote_to_operational(
            candidate_record=record,
            eval_fn=lambda x: np.sin(x),
            sample_inputs=sample_inputs,
            drift_monitoring_configured=False,
            fail_safe_clamp_defined=False,
        )
    assert "Operational Gate Rejected" in str(exc_info.value)
    assert lm.get_state("OP-TEST-001") == "Contract_Validated"

    # 2. All operational requirements satisfied -> promoted to Operational
    op_report = engine.promote_to_operational(
        candidate_record=record,
        eval_fn=lambda x: np.sin(x),
        sample_inputs=sample_inputs,
        drift_monitoring_configured=True,
        fail_safe_clamp_defined=True,
    )
    assert op_report.is_operational_ready is True
    assert lm.get_state("OP-TEST-001") == "Operational"


def test_mechanistic_adjudication_layer():
    """Validates adjudication separating DOMAIN_FAILURE from MODEL_FAILURE and EVIDENCE_PROBLEM."""
    from apedp.gated_assurance import (
        AdjudicationVerdict,
        MechanisticAdjudicationEngine,
        ReplicationIndependenceVector,
    )

    contract = create_hall_petch_benchmark_contract()
    record = ModelEvidenceRecord(
        model_id="ADJ-001",
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=("ADJ-001",),
        intervention_class=ModelInterventionClass.PHENOMENOLOGICAL_EXTENSION,
        evidence_maturity=EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
        domain="Nanocrystalline Metallurgy",
        mechanism_family="dislocation_pileup",
        # A concrete Hall-Petch form rather than the placeholder "sigma = f(d)":
        # an opaque function name is not a machine-checkable declaration, and the
        # gate rightly refuses to verify what it cannot parse.
        equation_str="sigma = sigma_0 + k_HP / sqrt(d)",
        variables=("d",),
        parameters={"sigma_0": 70.0e6, "k_HP": 0.74e6},
        derivation_basis="test",
        mathematical_assurance={
            "variable_units": {"sigma": "pascal", "sigma_0": "pascal", "k_HP": "Pa*m**0.5", "d": "meter"},
            "expected_output_unit": "pascal",
        },
        mathematical_guarantees=(),
        numerical_guarantees=(),
        physical_constraints=(),
        assumptions=(),
        known_failure_modes=(),
        validated_domain={"d": (10.0, 100.0)},
        extrapolation_limits={"d": (5.0, 200.0)},
        falsification_target="",
        falsification_tests=(),
        failure_interpretation="",
    )

    features = np.linspace(10.0, 100.0, 20).reshape(-1, 1)

    # Case A: Boundary Concentration (failures strictly at edge) -> DOMAIN_FAILURE
    obs = np.ones(20) * 100.0
    preds_boundary_fail = obs.copy()
    preds_boundary_fail[0] = 500.0  # boundary low failure
    preds_boundary_fail[-1] = 500.0 # boundary high failure

    rep_a = MechanisticAdjudicationEngine.adjudicate(
        candidate_record=record,
        contract=contract,
        features=features,
        predictions=preds_boundary_fail,
        observations=obs,
    )
    assert rep_a.verdict == AdjudicationVerdict.DOMAIN_FAILURE
    assert rep_a.recommended_lifecycle_action == "DOMAIN_RESTRICTED"

    # Case B: High measurement noise / low replication independence -> EVIDENCE_PROBLEM
    preds_mid_fail = obs.copy()
    preds_mid_fail[10] = 115.0  # violation in middle (15% error > 5% threshold)
    rep_b = MechanisticAdjudicationEngine.adjudicate(
        candidate_record=record,
        contract=contract,
        features=features,
        predictions=preds_mid_fail,
        observations=obs,
        estimated_sensor_noise_rel=0.08,
        independence=ReplicationIndependenceVector(data_independence=0.1),
    )
    assert rep_b.verdict == AdjudicationVerdict.EVIDENCE_PROBLEM
    assert rep_b.recommended_lifecycle_action == "CHALLENGED"

    # Case C: Pervasive core in-domain failure with high-fidelity independent data -> MODEL_FAILURE
    preds_core_fail = np.zeros(20) # massive violation everywhere
    rep_c = MechanisticAdjudicationEngine.adjudicate(
        candidate_record=record,
        contract=contract,
        features=features,
        predictions=preds_core_fail,
        observations=obs,
        independence=ReplicationIndependenceVector(
            data_independence=0.9, instrument_independence=0.9, operator_independence=0.9,
            site_independence=0.9, implementation_independence=0.9, analysis_independence=0.9,
            temporal_independence=0.9
        ),
    )
    assert rep_c.verdict == AdjudicationVerdict.MODEL_FAILURE
    assert rep_c.recommended_lifecycle_action == "REFUTED"


def test_zero_discrepancy_adjudication_never_refutes():
    """Regression: predictions that match the observations must never be adjudicated MODEL_FAILURE.

    Adjudication is triggered by non-empirical signals too (a violated physical
    bound, boundary probe failures). When the empirical channel is clean it must
    abstain via ``NO_CONTRADICTION``: ``MODEL_FAILURE`` maps downstream to
    MECHANISTIC_CONTRADICTION, which the gate turns into
    REFUTED_BY_EMPIRICAL_DATA -- i.e. refuting a model on data that agree with it.
    """
    from apedp.gated_assurance import AdjudicationVerdict, MechanisticAdjudicationEngine

    contract = create_casimir_benchmark_contract()
    record = ModelEvidenceRecord(
        model_id="CAS-REF-001",
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=("CAS-REF-001",),
        intervention_class=ModelInterventionClass.MATHEMATICAL_REGULARIZATION,
        evidence_maturity=EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
        domain="Nanoscale Electromechanics",
        mechanism_family="steric_repulsion",
        # ``abs(z)`` rather than ``|z|``: the repository's parser (by design) accepts
        # only its allowlisted function spelling, and the declaration has to be
        # machine-checkable for the gate to measure anything against it.
        equation_str="F(z) = -abs(z) - 1",
        variables=("z",),
        parameters={},
        derivation_basis="test",
        mathematical_assurance={
            "variable_units": {"F": "dimensionless", "z": "dimensionless"},
            "expected_output_unit": "dimensionless",
        },
        mathematical_guarantees=(),
        numerical_guarantees=(),
        # Deliberately violated by the evaluator below, which still fits the data.
        physical_constraints=("F > 0",),
        assumptions=(),
        known_failure_modes=(),
        validated_domain={"z": (1.0, 5.0)},
        extrapolation_limits={"z": (0.5, 10.0)},
        falsification_target="",
        falsification_tests=(),
        failure_interpretation="",
    )

    features = np.arange(1.0, 6.0).reshape(-1, 1)
    observations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # 1. The adjudicator abstains when every prediction is within threshold.
    report = MechanisticAdjudicationEngine.adjudicate(
        candidate_record=record,
        contract=contract,
        features=features,
        predictions=observations.copy(),
        observations=observations,
    )
    assert report.verdict == AdjudicationVerdict.NO_CONTRADICTION
    assert report.recommended_lifecycle_action != "REFUTED"
    assert report.confidence == 0.0

    # 2. End-to-end: an exact fit that breaks a declared bound is challenged on
    #    the invariant, not refuted by the empirical data.
    lm = LifecycleManager()
    engine = GatedScientificAssuranceEngine(lifecycle_manager=lm)

    def violating_eval(x: np.ndarray) -> np.ndarray:
        return -np.abs(x[:, 0]) - 1.0  # strictly negative => "F > 0" violated, exactly

    bench_result = engine.run_assurance_gate(
        candidate_record=record,
        contract=contract,
        eval_fn=violating_eval,
        baseline_eval_fn=lambda x: np.zeros(len(x)),
        features=features,
        ground_truth=violating_eval(features),  # predictions == observations
    )

    assert bench_result.prediction_coverage == 1.0
    assert bench_result.adjudication_report is not None
    assert bench_result.adjudication_report.verdict == AdjudicationVerdict.NO_CONTRADICTION
    assert bench_result.worst_severity != FailureSeverity.MECHANISTIC_CONTRADICTION
    assert bench_result.epistemic_consequence != "REFUTED_BY_EMPIRICAL_DATA"
    assert not bench_result.epistemic_consequence.startswith("REFUTED")

    # The invariant violation still blocks Contract_Validated.
    assert bench_result.physical_bounds == "FAIL"
    assert bench_result.worst_severity == FailureSeverity.PHYSICAL_INVARIANT_VIOLATION
    assert bench_result.epistemic_consequence == "CHALLENGED_INVARIANT_VIOLATION"
    assert lm.get_state("CAS-REF-001") == "Challenged"


def test_assurance_gate_probe_counts_and_metric_scores_are_measured():
    """Probe counts and the complexity/falsifiability sub-scores must be measured.

    Regression: the gate used to report a fixed 8/8 adversarial probe ratio and
    constant ``i_complexity=0.85`` / ``i_falsifiability=0.95`` for every candidate.
    """
    contract = ScientificBenchmarkContract(
        model_id="PROBE-001",
        baseline_model="y = 0",
        refined_model="y = 2*x",
        mechanism="linear proportionality",
        validated_domain={"x": (1.0, 10.0)},
        predicted_observables=("y",),
        # Brackets the candidate's predictions so the interval criterion is met.
        prediction_intervals={"y": (0.0, 100.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="probe_sealed_partition",
        data_generation_method="synthetic",
        noise_model="none",
        expected_failure_modes=(),
        expected_limits={},
        expected_invariants=(),
        pass_criteria=(),
        fail_criteria=(),
        domain_restriction_criteria=(),
    )
    features = np.linspace(1.0, 10.0, 10).reshape(-1, 1)

    def make_record(model_id: str, parameters: dict) -> ModelEvidenceRecord:
        return ModelEvidenceRecord(
            model_id=model_id,
            model_version="1.0.0",
            parent_model_id=None,
            model_lineage=(model_id,),
            intervention_class=ModelInterventionClass.MATHEMATICAL_REGULARIZATION,
            evidence_maturity=EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
            domain="Numerical Analysis",
            mechanism_family="linear_proportionality",
            equation_str="2 * x",
            variables=("x",),
            parameters=parameters,
            derivation_basis="linear",
            mathematical_guarantees=(),
            numerical_guarantees=(),
            physical_constraints=(),
            assumptions=(),
            known_failure_modes=(),
            validated_domain={"x": (1.0, 10.0)},
            extrapolation_limits={"x": (0.5, 20.0)},
            falsification_target="",
            falsification_tests=(),
            failure_interpretation="",
        )

    def run(record: ModelEvidenceRecord, eval_fn):
        return GatedScientificAssuranceEngine().run_assurance_gate(
            candidate_record=record,
            contract=contract,
            eval_fn=eval_fn,
            baseline_eval_fn=lambda x: np.zeros(len(x)),
            features=features,
            ground_truth=eval_fn(features),
        )

    # One declared interval -> exactly four probes: two endpoints plus a 10%
    # extrapolation past each end. The denominator is derived, not a literal.
    clean = run(make_record("PROBE-CLEAN", {"a": 1.0}), lambda x: 2.0 * x[:, 0])
    assert clean.adversarial_probes_total == 4
    assert clean.adversarial_probes_passed == 4
    assert clean.critical_failure_count == 0

    # Two intervals -> eight probes.
    doubled = run(
        replace(make_record("PROBE-DOUBLE", {}), validated_domain={"x": (1.0, 10.0), "y": (1.0, 10.0)}),
        lambda x: 2.0 * x[:, 0],
    )
    assert doubled.adversarial_probes_total == 8
    assert doubled.adversarial_probes_passed == 8

    # A probe that returns NaN past the high end is counted as a critical failure,
    # and passed + failed always equals the probes actually run.
    def nan_past_high(x: np.ndarray) -> np.ndarray:
        v = x[:, 0]
        return np.where(v > 10.5, np.nan, 2.0 * v)

    failing = run(make_record("PROBE-FAIL", {"a": 1.0}), nan_past_high)
    assert failing.adversarial_probes_total == 4
    assert failing.adversarial_probes_passed == 3
    assert failing.critical_failure_count == 1
    assert (
        failing.adversarial_probes_passed + failing.critical_failure_count
        == failing.adversarial_probes_total
    )

    # i_complexity is the parameter/sample-support ratio: 1 free parameter against
    # 10 sealed observations.
    assert clean.improvement_metric.i_complexity == pytest.approx(1.0 - 1 / 10)
    assert clean.improvement_metric.i_falsifiability == 1.0
    # ...and it responds to the record rather than being the old literal 0.85/0.95.
    assert (clean.improvement_metric.i_complexity, clean.improvement_metric.i_falsifiability) != (0.85, 0.95)

    # Ten parameters against ten sealed observations leaves no residual degrees of
    # freedom, so parsimony is zero.
    over_parameterized = run(
        make_record("PROBE-OVERPARAM", {f"p{i}": 1.0 for i in range(10)}),
        lambda x: 2.0 * x[:, 0],
    )
    assert over_parameterized.improvement_metric.i_complexity == pytest.approx(0.0)

    # Predictions outside the contract's declared interval lower i_falsifiability.
    off_interval = run(make_record("PROBE-OFFINTERVAL", {}), lambda x: 2.0 * x[:, 0] + 200.0)
    assert off_interval.improvement_metric.i_falsifiability < clean.improvement_metric.i_falsifiability


def test_multidimensional_pareto_dominance():
    """Validates Pareto dominance without collapsing metrics into a scalar."""
    from apedp.gated_assurance import ParetoFrontierTracker, ScientificImprovementMetric

    m1 = ScientificImprovementMetric(
        i_fit=0.80, i_stability=0.90, i_physicality=0.90,
        i_generality=0.70, i_complexity=0.80, i_falsifiability=0.80,
    )
    # m2 is strictly better in fit, equal or better in all others
    m2 = ScientificImprovementMetric(
        i_fit=0.92, i_stability=0.90, i_physicality=0.90,
        i_generality=0.75, i_complexity=0.80, i_falsifiability=0.85,
    )
    # m3 improves fit but breaks stability/physicality
    m3 = ScientificImprovementMetric(
        i_fit=0.99, i_stability=0.30, i_physicality=0.40,
        i_generality=0.70, i_complexity=0.50, i_falsifiability=0.80,
    )

    assert m2.pareto_dominates(m1) is True
    assert m1.pareto_dominates(m2) is False
    assert m3.pareto_dominates(m1) is False

    tracker = ParetoFrontierTracker()
    assert tracker.consider_candidate("M1", m1) is True
    assert tracker.consider_candidate("M3", m3) is False  # rejected by hard physical constraints
    assert tracker.consider_candidate("M2", m2) is True   # dominates and prunes M1
    frontier = tracker.get_frontier()
    assert "M2" in frontier
    assert "M1" not in frontier  # pruned!


def test_blind_diagnostic_benchmark_anti_leakage():
    """Validates BlindDiagnosticBenchmark cryptographic sealing to prevent answer leakage."""
    from apedp.gated_assurance import BlindDiagnosticBenchmark

    symptoms = {
        "divergence_type": "contact_singularity",
        "coordinate": "z -> 0",
        "residual_derivative": "dF/dz -> inf",
    }
    expected_diagnosis = "first-order force condition does not establish local contact stability"

    sealed_bench = BlindDiagnosticBenchmark.create_sealed_challenge(
        challenge_id="CHAL-001",
        symptom_signature=symptoms,
        expected_diagnosis=expected_diagnosis,
    )
    assert len(sealed_bench.sealed_diagnosis_hash) == 64
    assert sealed_bench.inferred_diagnosis == ""

    # Candidate infers diagnosis blindly from symptoms
    candidate_inference = "first-order force condition does not establish local mechanical stability"

    # Evaluation unlocks the seal
    evaluated = sealed_bench.evaluate_blind_submission(
        inferred_diagnosis=candidate_inference,
        expected_diagnosis=expected_diagnosis,
    )
    assert evaluated.category_match is True
    assert "matched" in evaluated.notes

    # Tampered expected diagnosis fails cryptographic verification
    with pytest.raises(ValueError) as exc_info:
        sealed_bench.evaluate_blind_submission(
            inferred_diagnosis=candidate_inference,
            expected_diagnosis="tampered wrong diagnosis",
        )
    assert "SEAL COMPROMISED" in str(exc_info.value)


def test_replication_independence_model():
    """Validates 7-axis independence model for experimental replications."""
    from apedp.gated_assurance import ReplicationIndependenceVector

    # 1. Nominal repetition (same lab, same operator, same code)
    nominal = ReplicationIndependenceVector(
        data_independence=0.1, instrument_independence=0.05, operator_independence=0.0,
        site_independence=0.0, implementation_independence=0.0, analysis_independence=0.05,
        temporal_independence=0.1,
    )
    assert nominal.replication_tier == "NOMINAL_REPETITION"
    assert nominal.composite_strength < 0.20

    # 2. Gold standard independent replication
    gold = ReplicationIndependenceVector(
        data_independence=0.95, instrument_independence=0.90, operator_independence=0.95,
        site_independence=1.00, implementation_independence=0.90, analysis_independence=0.85,
        temporal_independence=0.90,
    )
    assert gold.replication_tier == "GOLD_STANDARD_INDEPENDENT"
    assert gold.composite_strength >= 0.85


def test_epistemic_graph_scientific_evidence_edges_and_summary():
    """Validates EpistemicGraph with rich scientific evidence edges and summary generation."""
    from apedp.epistemic import EpistemicGraph

    graph = EpistemicGraph()

    # Create nodes: Model, Observation, Prediction, Experiment
    graph.add_node("Model_HP_02", "Model", {"name": "Hall-Petch Refined"})
    graph.add_node("Model_HP_00", "Model", {"name": "Classical Hall-Petch"})
    graph.add_node("Obs_TEM_01", "Observation", {"microscopy": "in-situ TEM"})
    graph.add_node("Pred_Yield_01", "Prediction", {"grain_size": "5nm"})
    graph.add_node("Exp_Compression_01", "Experiment", {"pillar_diameter": "100nm"})

    # Connect edges
    graph.add_edge("Model_HP_02", "Model_HP_00", "DERIVED_FROM")
    graph.add_edge("Obs_TEM_01", "Model_HP_02", "SUPPORTS")
    graph.add_edge("Model_HP_02", "Pred_Yield_01", "PREDICTS")
    graph.add_edge("Model_HP_02", "Exp_Compression_01", "TESTED_BY")

    summary = graph.get_model_evidence_summary("Model_HP_02")
    assert summary["model_id"] == "Model_HP_02"
    assert summary["supporting_evidence_count"] == 1
    assert summary["contradicting_evidence_count"] == 0
    assert summary["predictions_count"] == 1
    assert summary["tests_count"] == 1
    assert "Obs_TEM_01" in summary["supporting_nodes"]
    assert "Pred_Yield_01" in summary["prediction_nodes"]
    assert "Exp_Compression_01" in summary["test_nodes"]
    assert "Model_HP_00" in summary["derived_from_nodes"]


def test_gated_scientific_assurance_mechanistic_contradiction():
    """Validates that a candidate violating physical admissibility is refuted after adjudication."""
    contract = create_gibbs_thomson_benchmark_contract()
    lm = LifecycleManager()

    record = ModelEvidenceRecord(
        model_id=contract.model_id,
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=(contract.model_id,),
        intervention_class=ModelInterventionClass.MATHEMATICAL_REGULARIZATION,
        evidence_maturity=EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
        domain="Nanotechnology",
        mechanism_family="thermodynamic_melting_point",
        equation_str="Tm = Tm0 * (1 - C/r)",
        variables=("r",),
        # ``C`` is the capillary length the evaluator below uses (0.5e-9 m). It was
        # referenced by the equation and declared nowhere, which made the record
        # symbolically open.
        parameters={"Tm0": 1337.0, "C": 0.5e-9},
        derivation_basis="Gibbs-Thomson classical capillary relation",
        mathematical_assurance={
            "variable_units": {"Tm": "kelvin", "Tm0": "kelvin", "C": "meter", "r": "meter"},
            "expected_output_unit": "kelvin",
        },
        mathematical_guarantees=(),
        numerical_guarantees=(),
        physical_constraints=("Strict physical positivity: Tm > 0 K",),
        assumptions=(),
        known_failure_modes=(),
        validated_domain={"nanoparticle_radius_m": (0.5e-9, 10e-9)},
        extrapolation_limits={"nanoparticle_radius_m": (0.2e-9, 20e-9)},
        falsification_target="Negative temperature",
        falsification_tests=(),
        failure_interpretation="Third law violation",
    )

    engine = GatedScientificAssuranceEngine(lifecycle_manager=lm)

    # Nanoparticle radius grid down to 0.1 nm (causes negative Kelvin predictions across domain)
    r_grid = np.linspace(0.1e-9, 5e-9, 20).reshape(-1, 1)

    def flawed_evaluator(r: np.ndarray) -> np.ndarray:
        return 1337.0 * (1.0 - 0.5e-9 / r.flatten())

    def baseline_evaluator(r: np.ndarray) -> np.ndarray:
        return flawed_evaluator(r)

    ground_truth = np.full(len(r_grid), 800.0)

    bench_result = engine.run_assurance_gate(
        candidate_record=record,
        contract=contract,
        eval_fn=flawed_evaluator,
        baseline_eval_fn=baseline_evaluator,
        features=r_grid,
        ground_truth=ground_truth,
    )

    assert bench_result.physical_bounds == "FAIL"
    assert bench_result.worst_severity == FailureSeverity.MECHANISTIC_CONTRADICTION
    assert bench_result.epistemic_consequence == "REFUTED_BY_EMPIRICAL_DATA"
    assert lm.get_state("GT-REF-001") == "Refuted"


def test_testable_model_package_and_discrimination():
    """Validates TestableModelPackage construction, maximal discrimination point, and epistemic ladder."""
    # Two competing admissible hypotheses for high-field carrier velocity saturation:
    # f1 (Caughey-Thomas quadratic): v1(E) = mu * E / (1 + (mu*E / vs)^2)^(1/2)
    # f2 (Exponential transition):  v2(E) = vs * (1 - exp(-mu * E / vs))
    mu = 0.14
    vs = 1.0e5
    e_grid = np.linspace(1e4, 5e6, 500)
    v1 = mu * e_grid / np.sqrt(1.0 + (mu * e_grid / vs) ** 2)
    v2 = vs * (1.0 - np.exp(-mu * e_grid / vs))
    delta = np.abs(v1 - v2)
    idx_max = int(np.argmax(delta))
    e_star = float(e_grid[idx_max])

    package = TestableModelPackage(
        model_id="APEDP-SEMI-001",
        equation_str="v(E) = mu * E / (1 + (mu*E / vs)^2)^(1/2)",
        parameters={"mu": mu, "vs": vs},
        domain={"electric_field_v_m": (0.0, 1e7)},
        constraints=("Strictly positive for E > 0", "Monotonically increasing: dv/dE >= 0", "Asymptote <= vs"),
        limiting_behavior={"E_to_zero": "mu * E (Ohmic)", "E_to_inf": "vs (Saturation)"},
        recovered_limits={"low_field": "Drude linear conduction"},
        predictions=("Drift velocity v_d", "Differential mobility dv/dE"),
        required_data={"variable": "E_field", "range": (1e4, 5e6), "target_uncertainty": 0.02},
        holdout_test={"partition": "high_field_cryogenic_holdout", "points": 50},
        falsification_criteria=("dv/dE < 0", "v_d > 1.2 * vs"),
        competing_models=("vs * (1 - exp(-mu * E / vs))",),
        maximal_discrimination_point={"E_field_v_m": e_star, "max_discrepancy_m_s": float(delta[idx_max])},
        mechanistic_status="phenomenological_continuation",
        epistemic_status=ModelEpistemicStatus.TESTABLE_HYPOTHESIS,
    )

    data = package.to_dict()
    assert data["model_id"] == "APEDP-SEMI-001"
    assert data["epistemic_status"] == "TESTABLE_HYPOTHESIS"
    assert "E_field_v_m" in data["maximal_discrimination_point"]
    assert data["maximal_discrimination_point"]["E_field_v_m"] > 0.0
    assert len(data["competing_models"]) == 1


def test_feasibility_confusion_matrix_and_five_way_taxonomy():
    """Validates 5-way response taxonomy and ScientificFeasibilityConfusionMatrix metrics."""
    matrix = ScientificFeasibilityConfusionMatrix(
        true_solvable=95,
        false_unsolvable=0,
        true_contradictory=5,
        false_solvable=0,
    )

    assert matrix.total_cases == 100
    assert matrix.solve_recall == 1.0
    assert matrix.refusal_specificity == 1.0

    # Counts are not pre-seeded: an unpopulated matrix claims nothing.
    empty = ScientificFeasibilityConfusionMatrix()
    assert empty.total_cases == 0
    assert empty.solve_recall == 0.0
    assert empty.refusal_specificity == 0.0


def test_feasibility_confusion_matrix_computed_from_real_classification():
    """The confusion matrix must be tabulated from real classification runs."""
    from apedp.gated_assurance import classify_specification

    def package(model_id, equation, constraints, predictions, criteria):
        return TestableModelPackage(
            model_id=model_id,
            equation_str=equation,
            parameters={},
            domain={"x": (0.1, 10.0)},
            constraints=constraints,
            limiting_behavior={},
            recovered_limits={},
            predictions=predictions,
            required_data={},
            holdout_test={},
            falsification_criteria=criteria,
        )

    solvable = package(
        "SPEC-SOLVABLE", "2.0 / x + 1.0", (), ("f_value",), ("f <= 0",)
    )
    inconsistent = package(
        "SPEC-INCONSISTENT",
        "1 / x",
        (
            "contact_zero: lim_{x->0+} f(x) = 0",
            "strict_lower_bound: f(x) >= k/x**2 for all x > 0",
        ),
        ("f_value",),
        ("f(0) != 0",),
    )
    untestable = package("SPEC-UNTESTABLE", "x + 1.0", (), (), ())

    # The platform's own tests decide the labels.
    assert classify_specification(inconsistent) == SpecificationFeasibility.INCONSISTENT
    assert classify_specification(untestable) == SpecificationFeasibility.UNTESTABLE
    assert classify_specification(solvable) == SpecificationFeasibility.SOLVABLE

    matrix = ScientificFeasibilityConfusionMatrix.evaluate_specifications([
        (solvable, SpecificationFeasibility.SOLVABLE),
        (inconsistent, SpecificationFeasibility.INCONSISTENT),
        (untestable, SpecificationFeasibility.UNTESTABLE),
    ])
    assert matrix.total_cases == 3
    assert matrix.true_solvable == 1
    assert matrix.false_unsolvable == 0
    assert matrix.true_contradictory == 2
    assert matrix.false_solvable == 0
    assert matrix.solve_recall == 1.0
    assert matrix.refusal_specificity == 1.0

    # A misclassification is recorded as an error rather than smoothed away.
    wrong = ScientificFeasibilityConfusionMatrix.from_classifications([
        (SpecificationFeasibility.INCONSISTENT, SpecificationFeasibility.SOLVABLE),
    ])
    assert wrong.false_solvable == 1
    assert wrong.refusal_specificity == 0.0

    # Ensure 5-way classification maps to distinct epistemic actions
    actions = {
        SpecificationFeasibility.SOLVABLE: EpistemicActionStance.SOLVE,
        SpecificationFeasibility.INCONSISTENT: EpistemicActionStance.REFUSE,
        SpecificationFeasibility.UNDERDETERMINED: EpistemicActionStance.AGNOSTIC,
        SpecificationFeasibility.OUT_OF_DOMAIN: EpistemicActionStance.ABSTAIN,
        SpecificationFeasibility.UNTESTABLE: EpistemicActionStance.FLAG_UNTESTABLE,
    }

    for feasibility, stance in actions.items():
        assert map_feasibility_to_stance(feasibility) == stance


def test_heteroscedastic_uncertainty_weighted_discrimination():
    """Validates that uncertainty-weighting shifts maximal discrimination point to highest SNR."""
    # Model 1: f1(x) = x
    # Model 2: f2(x) = x + 0.1 * x^2
    # Grid: x in [0.1, 10.0]
    # Raw diff = 0.1 * x^2, which monotonically increases and peaks at x = 10.0 (diff = 10.0)
    # However, detector noise sigma(x) = 0.05 * x^3 (blows up cubically at high x)
    # SNR = (0.1 * x^2) / (0.05 * x^3) = 2.0 / x, which peaks at lowest x (x = 0.1)!
    x_grid = np.linspace(0.1, 10.0, 100)

    def f1(x: np.ndarray) -> np.ndarray:
        return x

    def f2(x: np.ndarray) -> np.ndarray:
        return x + 0.1 * (x ** 2)

    def sigma(x: np.ndarray) -> np.ndarray:
        return 0.05 * (x ** 3)

    # Without uncertainty weighting: peaks at maximum raw discrepancy (x = 10.0)
    x_raw, _, raw_val = TestableModelPackage.compute_uncertainty_weighted_discrimination(
        eval_f1=f1,
        eval_f2=f2,
        x_grid=x_grid,
        uncertainty_fn=None,
    )
    assert np.isclose(x_raw, 10.0)
    assert np.isclose(raw_val, 10.0)

    # With heteroscedastic uncertainty weighting: peaks at maximum SNR (x = 0.1)
    x_snr, snr_val, _ = TestableModelPackage.compute_uncertainty_weighted_discrimination(
        eval_f1=f1,
        eval_f2=f2,
        x_grid=x_grid,
        uncertainty_fn=sigma,
    )
    assert np.isclose(x_snr, 0.1)
    assert snr_val > 15.0  # 2.0 / 0.1 = 20.0


def test_generation_provenance_and_discovery_modes():
    """Validates generation provenance traceability and discovery modes."""
    provenance = {
        "discovery_mode": DiscoveryMode.SYNTHESIZE.value,
        "parent_models": [],
        "pi_groups": ["Pi_1 = v / c", "Pi_2 = sigma / E"],
        "invariants_satisfied": ["Energy conservation", "Lorentz frame invariance"],
        "search_space_size": 10000,
        "candidates_evaluated": 342,
    }

    pkg = TestableModelPackage(
        model_id="APEDP-SYNTH-001",
        equation_str="Pi_1 = 0.5 * Pi_2",
        parameters={"coupling": 0.5},
        domain={"x": (0.0, 1.0)},
        constraints=("Pi_1 >= 0",),
        limiting_behavior={"x_to_0": "0"},
        recovered_limits={"non_relativistic": "Galilean limit"},
        predictions=("Relativistic deflection",),
        required_data={"observable": "angle", "resolution": 1e-6},
        holdout_test={"partition": "synchrotron_beam", "points": 100},
        falsification_criteria=("deflection < 0",),
        epistemic_status=ModelEpistemicStatus.GENERATED,
        generation_provenance=provenance,
    )

    data = pkg.to_dict()
    assert data["generation_provenance"]["discovery_mode"] == "SYNTHESIZE"
    assert len(data["generation_provenance"]["pi_groups"]) == 2
    assert len(VerificationGate) == 6
    assert VerificationGate.PHYSICAL_ADMISSIBILITY.value == "PHYSICAL_ADMISSIBILITY"





# ---------------------------------------------------------------------------
# Measured sub-scores: the gate's metric literals and presence tests
# ---------------------------------------------------------------------------

def _measured_gate_contract(model_id: str) -> ScientificBenchmarkContract:
    return ScientificBenchmarkContract(
        model_id=model_id,
        baseline_model="y = 0",
        refined_model="y = 2*x",
        mechanism="probe",
        validated_domain={"x": (1.0, 10.0)},
        predicted_observables=("y",),
        prediction_intervals={"y": (-1e9, 1e9)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="w3_sealed_partition",
        data_generation_method="synthetic",
        noise_model="none",
        expected_failure_modes=(),
        expected_limits={},
        expected_invariants=(),
        pass_criteria=(),
        fail_criteria=(),
        domain_restriction_criteria=(),
    )


def _measured_gate_record(model_id: str, **overrides) -> ModelEvidenceRecord:
    fields: dict = {
        "model_version": "1.0.0",
        "parent_model_id": None,
        "model_lineage": (model_id,),
        "intervention_class": ModelInterventionClass.MATHEMATICAL_REGULARIZATION,
        "evidence_maturity": EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
        "domain": "Numerical Analysis",
        "mechanism_family": "linear_proportionality",
        "equation_str": "y = 2*x",
        "variables": ("x",),
        "parameters": {},
        "derivation_basis": "linear",
        "mathematical_guarantees": (),
        "numerical_guarantees": (),
        "physical_constraints": (),
        "assumptions": (),
        "known_failure_modes": (),
        "validated_domain": {"x": (1.0, 10.0)},
        "extrapolation_limits": {"x": (0.5, 20.0)},
        "falsification_target": "",
        "falsification_tests": (),
        "failure_interpretation": "",
    }
    fields.update(overrides)
    return ModelEvidenceRecord(model_id=model_id, **fields)


def _run_measured_gate(record: ModelEvidenceRecord, eval_fn, features: np.ndarray):
    return GatedScientificAssuranceEngine().run_assurance_gate(
        candidate_record=record,
        contract=_measured_gate_contract(record.model_id),
        eval_fn=eval_fn,
        baseline_eval_fn=lambda x: np.zeros(len(x)),
        features=features,
        ground_truth=2.0 * features[:, 0],
    )


def test_gate_sub_scores_are_measured_not_severity_lookups():
    """i_stability / i_physicality / i_generality must respond to the model.

    Regression: the gate reported ``1.0 if critical_failures == 0 else 0.5``,
    ``1.0 if physical_bounds == "PASS" else 0.0`` and ``0.90/0.40`` -- three constants
    dressed as continuous scores, none of which could move with the candidate.
    """
    features = np.linspace(1.0, 10.0, 12).reshape(-1, 1)
    clean = _run_measured_gate(_measured_gate_record("W3-CLEAN"), lambda v: 2.0 * v[:, 0], features)

    # i_physicality is the fraction of evaluable claims the model actually satisfies.
    partial = _run_measured_gate(
        _measured_gate_record("W3-PARTIAL", physical_constraints=("y > 0", "y <= 25", "y < 0")),
        lambda v: 2.0 * v[:, 0],
        features,
    )
    assert partial.physical_bounds == "FAIL"  # the third claim is contradicted by the model
    assert partial.improvement_metric.i_physicality == pytest.approx(2 / 3, abs=1e-4)  # rounded to 4dp
    assert (partial.improvement_metric.i_physicality, clean.improvement_metric.i_physicality) != (0.0, 1.0)

    # i_stability is probe survival discounted by measured local amplification.
    nan_past_high = lambda v: np.where(v[:, 0] > 10.5, np.nan, 2.0 * v[:, 0])
    unstable = _run_measured_gate(_measured_gate_record("W3-NAN"), nan_past_high, features)
    stability_evidence = unstable.details["measurement_evidence"]["i_stability"]
    assert stability_evidence["probe_survival"] == pytest.approx(0.75)
    assert unstable.improvement_metric.i_stability == pytest.approx(
        0.75 * stability_evidence["conditioning"], rel=1e-6
    )
    assert unstable.improvement_metric.i_stability < clean.improvement_metric.i_stability

    amplifying = lambda v: 2.0 * v[:, 0] + 1000.0 * np.sign(np.sin(50.0 * v[:, 0]))
    noisy = _run_measured_gate(_measured_gate_record("W3-AMPLIFY"), amplifying, features)
    assert noisy.details["measurement_evidence"]["i_stability"]["probe_survival"] == 1.0
    assert noisy.improvement_metric.i_stability < unstable.improvement_metric.i_stability

    # i_generality compares interior and outer-fringe error on the sealed partition.
    boundary_degraded = lambda v: np.where(v[:, 0] > 7.0, 2.0 * v[:, 0] + 50.0, 2.0 * v[:, 0])
    boundary = _run_measured_gate(_measured_gate_record("W3-BOUNDARY"), boundary_degraded, features)
    generality_evidence = boundary.details["measurement_evidence"]["i_generality"]
    assert generality_evidence["boundary_mae"] > generality_evidence["interior_mae"]
    assert boundary.improvement_metric.i_generality < clean.improvement_metric.i_generality
    assert clean.details["measurement_evidence"]["i_generality"]["reason"].startswith("both partitions fit")


def test_gate_constraints_are_parsed_and_evaluated_not_substring_matched():
    """Constraints must be evaluated, not searched for the substring ``"> 0"``.

    Regression: the gate only recognised a non-negativity claim, so any other
    relation -- including the suction-surface inequality below -- was a free pass.
    """
    features = np.linspace(1.0, 10.0, 12).reshape(-1, 1)
    positive = lambda v: 2.0 * v[:, 0]

    suction = _run_measured_gate(
        _measured_gate_record("W3-SUCTION", physical_constraints=("Cp < 0 (suction surface)",)), positive, features
    )
    assert suction.physical_bounds == "FAIL"  # old gate: no "> 0" substring -> PASS
    assert suction.details["measurement_evidence"]["constraint_checks"][0]["status"] == "VIOLATED"

    ceiling = _run_measured_gate(
        _measured_gate_record(
            "W3-CEILING",
            parameters={"y_ceiling": 25.0},
            physical_constraints=("Strict positivity: y > 0 K", "y <= y_ceiling"),
        ),
        positive,
        features,
    )
    assert ceiling.physical_bounds == "PASS"  # parameter threshold resolved from the record

    opaque = _run_measured_gate(
        _measured_gate_record("W3-OPAQUE", physical_constraints=("The model behaves well",)), positive, features
    )
    assert opaque.physical_bounds == "WARNING"
    assert opaque.details["measurement_evidence"]["constraint_checks"][0]["status"] == "UNMEASURABLE"
    assert opaque.improvement_metric.i_physicality == 0.0  # no evidence is not credit


def test_gate_asymptotes_are_measured_against_the_model():
    """Asymptotes must be measured, not credited because the record declares any.

    Regression: ``asymptotics = "PASS" if len(symbolic_asymptotics) > 0 else "WARNING"``.
    """
    features = np.linspace(1.0, 10.0, 12).reshape(-1, 1)
    linear = lambda v: 2.0 * v[:, 0]
    declared_divergent = {"x_to_inf": "unbounded"}

    grows = _run_measured_gate(
        _measured_gate_record("W3-GROW", symbolic_asymptotics=declared_divergent), linear, features
    )
    assert grows.asymptotic_constraints == "PASS"

    bounded = _run_measured_gate(
        _measured_gate_record("W3-BOUNDED", symbolic_asymptotics=declared_divergent),
        lambda v: np.full(len(v), 2.0),
        features,
    )
    assert bounded.asymptotic_constraints == "FAIL"  # declares divergence, is bounded

    pole = _run_measured_gate(
        _measured_gate_record("W3-POLE", symbolic_asymptotics={"x_to_zero": "0"}),
        lambda v: 2.0 / v[:, 0],
        features,
    )
    assert pole.asymptotic_constraints == "FAIL"  # declares a finite limit, has a pole

    finite = _run_measured_gate(
        _measured_gate_record("W3-FINITE", symbolic_asymptotics={"x_to_zero": "0"}), linear, features
    )
    assert finite.asymptotic_constraints == "PASS"

    unknown = _run_measured_gate(
        _measured_gate_record("W3-UNKNOWN", symbolic_asymptotics={"q_to_inf": "unbounded"}), linear, features
    )
    assert unknown.asymptotic_constraints == "WARNING"
    assert "not one of the record's declared inputs" in (
        unknown.details["measurement_evidence"]["asymptote_checks"][0]["detail"]
    )

    silent = _run_measured_gate(_measured_gate_record("W3-SILENT"), linear, features)
    assert silent.asymptotic_constraints == "WARNING"
    assert silent.improvement_metric.i_physicality == 0.0


def test_gate_dimensional_consistency_is_measured_not_inferred_from_derivation_basis():
    """Dimensional consistency must be checked, not asserted via a non-empty field.

    Regression: ``dim_consistency = "PASS" if candidate_record.derivation_basis else "WARNING"``
    reported PASS for any record with a prose derivation basis, regardless of the equation.
    """
    features = np.linspace(1.0, 10.0, 12).reshape(-1, 1)
    linear = lambda v: 2.0 * v[:, 0]

    undeclared = _run_measured_gate(
        _measured_gate_record("W3-DIM-UNDECLARED", equation_str="y = 2*x + k_mystery"), linear, features
    )
    assert undeclared.dimensional_consistency == "FAIL"
    assert undeclared.details["measurement_evidence"]["dimensional_consistency"]["undeclared_symbols"] == ["k_mystery"]

    homogeneous = _run_measured_gate(
        _measured_gate_record(
            "W3-DIM-OK",
            mathematical_assurance={"variable_units": {"x": "meter", "y": "meter"}, "expected_output_unit": "meter"},
        ),
        linear,
        features,
    )
    assert homogeneous.dimensional_consistency == "PASS"

    inhomogeneous = _run_measured_gate(
        _measured_gate_record(
            "W3-DIM-BAD",
            equation_str="y = 2*x + z",
            parameters={"z": 1.0},
            mathematical_assurance={"variable_units": {"x": "meter", "z": "second", "y": "meter"}},
        ),
        linear,
        features,
    )
    assert inhomogeneous.dimensional_consistency == "FAIL"  # meter + second

    unverifiable = _run_measured_gate(_measured_gate_record("W3-DIM-UNVERIFIED"), linear, features)
    assert unverifiable.dimensional_consistency == "UNVERIFIED"
    assert "homogeneity is unverified" in unverifiable.details["measurement_evidence"]["dimensional_consistency"]["reason"]
