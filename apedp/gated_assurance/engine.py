"""The gated scientific assurance engine tying the stages together."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from apedp.epistemic import LifecycleManager, ModelEvidenceRecord

from .checks import (
    _check_asymptote,
    _check_constraint,
    _clamp01,
    _column_index,
    _input_columns,
    _measure_dimensional_consistency,
    _measure_generality,
    _measure_stability,
)
from .contracts import (
    AdjudicationReport,
    AdjudicationVerdict,
    BlindDiagnosticBenchmark,
    FailureSeverity,
    FalsificationBenchResult,
    KnownDiagnosticBenchmark,
    ParetoFrontierTracker,
    ReplicationIndependenceVector,
    ScientificBenchmarkContract,
    ScientificImprovementMetric,
)
from .mechanistic import (
    MechanisticAdjudicationEngine,
)
from .operational import (
    BlindChallengeHarness,
    OperationalGateAuditor,
    OperationalReadinessReport,
)


class GatedScientificAssuranceEngine:
    """Executable Scientific Control Plane governing the complete model assurance lifecycle."""

    def __init__(self, lifecycle_manager: LifecycleManager | None = None):
        self.lifecycle_manager = lifecycle_manager or LifecycleManager()
        self.harness = BlindChallengeHarness()
        self.frontier_tracker = ParetoFrontierTracker()

    def run_assurance_gate(
        self,
        candidate_record: ModelEvidenceRecord,
        contract: ScientificBenchmarkContract,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        baseline_eval_fn: Callable[[np.ndarray], np.ndarray],
        features: np.ndarray,
        ground_truth: np.ndarray,
        diagnostic: KnownDiagnosticBenchmark | BlindDiagnosticBenchmark | None = None,
        independence: ReplicationIndependenceVector | None = None,
        challenge_hash: str | None = None,
    ) -> FalsificationBenchResult:
        """Executes full gated assurance: Numeric, Asymptotic, Invariant, and Adversarial checks.

        Blinding: registration and evaluation are separate steps. Seal the
        holdout once during setup with ``BlindChallengeHarness.seal_challenge``
        and pass its hash as ``challenge_hash`` so every candidate is scored
        against the same fixed partition. When ``challenge_hash`` is omitted the
        partition is sealed here from the supplied arrays -- a self-sealing
        convenience for single-shot use that carries no independence guarantee;
        the mode is recorded in ``details["challenge_mode"]``.
        """
        # 1. Resolve the sealed challenge partition (registration happens separately)
        if challenge_hash is None:
            data_hash = self.harness.seal_challenge(
                features=features,
                targets=ground_truth,
                metadata={"model_id": candidate_record.model_id},
            )
            challenge_mode = "self_sealed"
        else:
            data_hash = challenge_hash
            if not self.harness.has_challenge(data_hash):
                raise KeyError(
                    f"Sealed challenge '{data_hash}' is not registered. Seal the partition with "
                    "BlindChallengeHarness.seal_challenge() before evaluating candidates against it."
                )
            if self.harness.content_hash(features, ground_truth) != data_hash:
                raise ValueError(
                    "challenge_hash does not match the supplied evaluation partition; refusing "
                    "to report metrics computed against a different partition."
                )
            challenge_mode = "pre_sealed"

        # 2. Evaluate blind empirical performance (aggregate statistics only)
        blind_result = self.harness.evaluate_blind(
            candidate_evaluator=eval_fn,
            dataset_hash=data_hash,
            contract=contract,
            caller_phase="falsification",
        )
        blind_result["challenge_mode"] = challenge_mode

        # 3. Numeric & Physical Invariants Tests
        preds = eval_fn(features)
        monotonic = "PASS"
        if len(preds) > 1 and np.any(np.diff(features[:, 0]) > 0):
            diffs = np.diff(preds)
            if not (np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6)):
                monotonic = "FAIL"

        # Physical constraints and declared asymptotics are evaluated against the
        # model, not read off the record's prose. See "Measured assurance
        # primitives" above for the conventions these measurements follow: an
        # unevaluable claim is reported as unmeasured, never as satisfied.
        input_columns = _input_columns(candidate_record, features)
        constraint_checks = [
            _check_constraint(constraint, preds, input_columns, candidate_record.parameters)
            for constraint in candidate_record.physical_constraints
        ]
        constraints_satisfied = sum(1 for check in constraint_checks if check.status == "SATISFIED")
        constraints_violated = sum(1 for check in constraint_checks if check.status == "VIOLATED")
        constraints_unmeasurable = sum(1 for check in constraint_checks if check.status == "UNMEASURABLE")
        if not constraint_checks:
            physical_bounds = "PASS"
        elif constraints_violated:
            physical_bounds = "FAIL"
        elif constraints_unmeasurable:
            physical_bounds = "WARNING"
        else:
            physical_bounds = "PASS"

        domain_bounds = list(candidate_record.validated_domain.values())
        base_vector = np.array([0.5 * (low + high) for low, high in domain_bounds], dtype=float)
        asymptote_checks = [
            _check_asymptote(
                candidate_record,
                eval_fn,
                name,
                declared,
                base_vector,
                domain_bounds,
                lambda symbol: _column_index(candidate_record, symbol),
            )
            for name, declared in candidate_record.symbolic_asymptotics.items()
        ]
        asymptotes_consistent = sum(1 for check in asymptote_checks if check.status == "CONSISTENT")
        asymptotes_contradicted = sum(1 for check in asymptote_checks if check.status == "CONTRADICTED")
        asymptotes_measured = asymptotes_consistent + asymptotes_contradicted
        if not asymptote_checks or asymptotes_measured == 0:
            asymptotics = "WARNING"
        elif asymptotes_contradicted:
            asymptotics = "FAIL"
        else:
            asymptotics = "PASS"

        dim_consistency, dim_fraction, dim_evidence = _measure_dimensional_consistency(candidate_record)

        # 4. Adversarial Boundary Stress-Testing
        #    Each declared interval contributes four probe points: both endpoints
        #    plus a 10% extrapolation past each end. Counts are tallied from the
        #    probes actually executed -- there is no fixed denominator.
        adversarial_total = 0
        adversarial_passed = 0
        critical_failures = 0

        for low, high in candidate_record.validated_domain.values():
            for point in (low, high, low * 0.9, high * 1.1):
                adversarial_total += 1
                probe_ok = False
                try:
                    probe_res = eval_fn(np.array([[point]]))
                    probe_ok = not (np.any(np.isnan(probe_res)) or np.any(np.isinf(probe_res)))
                except (RuntimeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError, OverflowError):
                    probe_ok = False
                if probe_ok:
                    adversarial_passed += 1
                else:
                    critical_failures += 1

        # 5. Adjudication Layer for Contradictions
        adjudication_report: AdjudicationReport | None = None
        if physical_bounds == "FAIL" or blind_result["coverage"] < 0.90 or critical_failures > 0:
            adjudication_report = MechanisticAdjudicationEngine.adjudicate(
                candidate_record=candidate_record,
                contract=contract,
                features=features,
                predictions=preds,
                observations=ground_truth,
                independence=independence,
            )

        # 6. Determine Failure Severity with Adjudication
        worst_severity = FailureSeverity.PASS
        if blind_result["status"] == FailureSeverity.IMPLEMENTATION_FAILURE or critical_failures > 2:
            worst_severity = FailureSeverity.IMPLEMENTATION_FAILURE
        elif adjudication_report is not None and adjudication_report.verdict != AdjudicationVerdict.NO_CONTRADICTION:
            if adjudication_report.verdict == AdjudicationVerdict.DOMAIN_FAILURE:
                worst_severity = FailureSeverity.DOMAIN_VIOLATION
            elif adjudication_report.verdict == AdjudicationVerdict.EVIDENCE_PROBLEM:
                worst_severity = FailureSeverity.PREDICTIVE_FAILURE
            elif adjudication_report.verdict == AdjudicationVerdict.MODEL_FAILURE:
                worst_severity = FailureSeverity.MECHANISTIC_CONTRADICTION
            elif adjudication_report.verdict == AdjudicationVerdict.IMPLEMENTATION_FAILURE:
                worst_severity = FailureSeverity.IMPLEMENTATION_FAILURE
        elif physical_bounds == "FAIL":
            # Adjudication returned NO_CONTRADICTION: the data license no
            # mechanistic refutation. Report the non-empirical signal that
            # triggered adjudication instead of a fabricated contradiction.
            worst_severity = FailureSeverity.PHYSICAL_INVARIANT_VIOLATION
        elif critical_failures > 0 or monotonic == "FAIL":
            worst_severity = FailureSeverity.WARNING

        # 7. Compute Multidimensional Scientific Improvement Metric
        base_preds = baseline_eval_fn(features)
        base_err = float(np.mean(np.abs(base_preds - ground_truth)))
        cand_err = float(np.mean(np.abs(preds - ground_truth)))
        err_reduction = max(0.0, min(1.0, (base_err - cand_err) / (base_err + 1e-9)))

        # i_complexity: parameter parsimony measured as the free-parameter count
        # against the sealed observations available to support it. A model with as
        # many parameters as sealed samples has no residual degrees of freedom and
        # scores 0.0. ``parameters`` may include fixed physical ceilings, so this is
        # a conservative (pessimistic) proxy rather than an exact count.
        n_parameters = len(candidate_record.parameters)
        n_sealed_samples = int(blind_result["n_samples"])
        i_complexity = _clamp01(1.0 - n_parameters / max(n_sealed_samples, 1))

        # i_falsifiability: the fraction of the contract's machine-checkable
        # falsification criteria that this candidate actually satisfies on the
        # sealed partition. A contract that declares no prediction intervals still
        # contributes the residual-coverage criterion; an empty check list cannot
        # occur, and an empty contract cannot score above that single criterion.
        falsification_criteria_met: list[bool] = [
            bool(np.all(preds >= low) and np.all(preds <= high))
            for low, high in contract.prediction_intervals.values()
        ]
        falsification_criteria_met.append(bool(blind_result["coverage"] >= 0.90))
        i_falsifiability = _clamp01(float(np.mean(falsification_criteria_met)))

        # i_stability and i_generality are measurements, not severity lookups: the
        # gate previously reported ``1.0/0.5`` and ``0.90/0.40``, which were two
        # constants dressed as continuous scores. Each measurement's basis is
        # published in ``details["measurement_evidence"]`` so a reader can audit
        # how the number was obtained.
        i_stability, stability_evidence = _measure_stability(preds, adversarial_passed, adversarial_total)
        i_generality, generality_evidence = _measure_generality(features, preds, ground_truth)

        physicality_components: dict[str, float] = {}
        measurable_constraints = constraints_satisfied + constraints_violated
        if measurable_constraints:
            physicality_components["constraints"] = constraints_satisfied / measurable_constraints
        if asymptotes_measured:
            physicality_components["asymptotics"] = asymptotes_consistent / asymptotes_measured
        if dim_fraction is not None:
            physicality_components["dimensional"] = dim_fraction
        i_physicality = (
            _clamp01(float(np.mean(list(physicality_components.values())))) if physicality_components else 0.0
        )

        improvement = ScientificImprovementMetric(
            i_fit=round(err_reduction, 4),
            i_stability=round(i_stability, 4),
            i_physicality=round(i_physicality, 4),
            i_generality=round(i_generality, 4),
            i_complexity=round(i_complexity, 4),
            i_falsifiability=round(i_falsifiability, 4),
        )

        blind_result["measurement_evidence"] = {
            "i_stability": stability_evidence,
            "i_generality": generality_evidence,
            "i_physicality": {
                "components": {name: round(value, 4) for name, value in physicality_components.items()},
                "reason": "" if physicality_components else "no declared claim was evaluable against the model",
                "declared_constraints": len(constraint_checks),
                "unmeasurable_constraints": constraints_unmeasurable,
                "declared_asymptotes": len(asymptote_checks),
                "unmeasurable_asymptotes": len(asymptote_checks) - asymptotes_measured,
            },
            "dimensional_consistency": dim_evidence,
            "constraint_checks": [check.to_dict() for check in constraint_checks],
            "asymptote_checks": [check.to_dict() for check in asymptote_checks],
        }

        # Track in Pareto frontier
        self.frontier_tracker.consider_candidate(candidate_record.model_id, improvement)

        # 8. Map to Epistemic Lifecycle Consequence
        if worst_severity == FailureSeverity.PASS:
            epistemic_consequence = "CONTRACT_VALIDATED"
        elif worst_severity == FailureSeverity.DOMAIN_VIOLATION:
            epistemic_consequence = "DOMAIN_RESTRICTION_REQUIRED"
        elif worst_severity == FailureSeverity.MECHANISTIC_CONTRADICTION:
            epistemic_consequence = "REFUTED_BY_EMPIRICAL_DATA"
        elif worst_severity == FailureSeverity.PREDICTIVE_FAILURE:
            epistemic_consequence = "CHALLENGED_PREDICTION_ERROR"
        elif worst_severity == FailureSeverity.PHYSICAL_INVARIANT_VIOLATION:
            epistemic_consequence = "CHALLENGED_INVARIANT_VIOLATION"
        else:
            epistemic_consequence = "RETAINED_IN_CANDIDATE_FOR_FIX"

        # 9. Update LifecycleManager State
        model_name = candidate_record.model_id
        current_state = self.lifecycle_manager.get_state(model_name)
        if current_state == "Draft":
            self.lifecycle_manager.promote(model_name, "Candidate", checks_passed=True, evidence_id="INIT")

        if worst_severity == FailureSeverity.PASS:
            # Promote to Contract_Validated (NOT directly Operational)
            self.lifecycle_manager.promote(model_name, "Contract_Validated", checks_passed=True, evidence_id=data_hash[:8])
        elif worst_severity == FailureSeverity.DOMAIN_VIOLATION:
            if self.lifecycle_manager.get_state(model_name) == "Candidate":
                self.lifecycle_manager.promote(model_name, "Challenged", checks_passed=True, evidence_id=data_hash[:8])
            self.lifecycle_manager.promote(model_name, "Domain_Restricted", checks_passed=True, evidence_id=data_hash[:8])
        elif worst_severity == FailureSeverity.MECHANISTIC_CONTRADICTION:
            self.lifecycle_manager.promote(model_name, "Refuted", checks_passed=True, evidence_id=data_hash[:8])
        elif worst_severity in (
            FailureSeverity.PREDICTIVE_FAILURE,
            FailureSeverity.PHYSICAL_INVARIANT_VIOLATION,
        ) and self.lifecycle_manager.get_state(model_name) == "Candidate":
            self.lifecycle_manager.promote(model_name, "Challenged", checks_passed=True, evidence_id=data_hash[:8])

        return FalsificationBenchResult(
            model_id=candidate_record.model_id,
            domain=candidate_record.domain,
            prediction_coverage=blind_result["coverage"],
            monotonicity=monotonic,
            asymptotic_constraints=asymptotics,
            dimensional_consistency=dim_consistency,
            physical_bounds=physical_bounds,
            out_of_sample_error="PASS" if blind_result["coverage"] >= 0.90 else "FAIL",
            adversarial_probes_passed=adversarial_passed,
            adversarial_probes_total=adversarial_total,
            critical_failure_count=critical_failures,
            worst_severity=worst_severity,
            epistemic_consequence=epistemic_consequence,
            improvement_metric=improvement,
            diagnostic_benchmark=diagnostic,
            adjudication_report=adjudication_report,
            details=blind_result,
        )

    def promote_to_operational(
        self,
        candidate_record: ModelEvidenceRecord,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        sample_inputs: np.ndarray,
        max_latency_ms: float = 50.0,
        drift_monitoring_configured: bool = True,
        fail_safe_clamp_defined: bool = True,
    ) -> OperationalReadinessReport:
        """Evaluates operational gate: Contract_Validated -> Operational."""
        model_name = candidate_record.model_id
        current_state = self.lifecycle_manager.get_state(model_name)
        if current_state not in ("Contract_Validated", "Validated"):
            raise ValueError(
                f"Cannot promote '{model_name}' to Operational: current state is '{current_state}', must be 'Contract_Validated' or 'Validated'."
            )

        report = OperationalGateAuditor.audit_readiness(
            candidate_record=candidate_record,
            eval_fn=eval_fn,
            sample_inputs=sample_inputs,
            max_latency_ms=max_latency_ms,
            drift_monitoring_configured=drift_monitoring_configured,
            fail_safe_clamp_defined=fail_safe_clamp_defined,
        )

        if not report.is_operational_ready:
            raise ValueError(f"Operational Gate Rejected for '{model_name}': {report.unmet_requirements}")

        self.lifecycle_manager.promote(
            model_name,
            "Operational",
            checks_passed=True,
            evidence_id=f"OP_GATE_{candidate_record.record_hash[:8]}",
        )
        return report
