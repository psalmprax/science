"""Evaluation Engine and Scorecard for the APEDP-RH Riemann Hypothesis Benchmark.

Evaluates structure discovery, algebraic recovery, OOD extrapolation, identifiability,
ambiguity detection, precision limits, and strict compliance with the proof boundary guard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp.canonical_compiler import CanonicalEquationCompiler
from apedp.decision.stage_c import EpistemicState
from apedp.mathematical.mathematical_layer import MathematicalAssuranceEngine, ProofBoundaryStatus

logger = logging.getLogger(__name__)


@dataclass
class RHCaseResult:
    """Evaluation result for an individual benchmark case in APEDP-RH.

    proof_claim is independent of empirical state: P0 = no proof (always, for
    finite-sample runs), P1/P2 reserved. Joint cp_label e.g. "C1,P0" prevents
    C1=proven misread.
    """

    case_id: str
    case_name: str
    expected_state: str
    actual_state: str
    top_expression: str
    ground_truth_form: str
    is_structure_recovered: bool
    is_algebraic_equivalent: bool
    is_false_discovery: bool
    is_ambiguity_detected: bool
    proof_guard_compliant: bool
    near_ood_mse: float
    complexity: float
    rationale: str = ""
    proof_claim: str = "P0"


@dataclass
class RHScorecard:
    """Overall evaluation summary for the APEDP-RH benchmark tournament."""

    total_cases: int
    structure_discovery_rate: float
    algebraic_recovery_rate: float
    false_discovery_rate: float
    ambiguity_detection_rate: float
    proof_boundary_compliance_rate: float
    median_ood_error: float
    mean_complexity: float
    case_results: list[RHCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "structure_discovery_rate": self.structure_discovery_rate,
            "algebraic_recovery_rate": self.algebraic_recovery_rate,
            "false_discovery_rate": self.false_discovery_rate,
            "ambiguity_detection_rate": self.ambiguity_detection_rate,
            "proof_boundary_compliance_rate": self.proof_boundary_compliance_rate,
            "median_ood_error": self.median_ood_error,
            "mean_complexity": self.mean_complexity,
            "cases": [
                {
                    "case_id": c.case_id,
                    "case_name": c.case_name,
                    "expected_state": c.expected_state,
                    "actual_state": c.actual_state,
                    "top_expression": c.top_expression,
                    "ground_truth_form": c.ground_truth_form,
                    "is_structure_recovered": c.is_structure_recovered,
                    "is_algebraic_equivalent": c.is_algebraic_equivalent,
                    "is_false_discovery": c.is_false_discovery,
                    "is_ambiguity_detected": c.is_ambiguity_detected,
                    "proof_guard_compliant": c.proof_guard_compliant,
                    "proof_claim": c.proof_claim,
                    "cp_label": f"{c.actual_state},{c.proof_claim}",
                    "near_ood_mse": c.near_ood_mse,
                    "complexity": c.complexity,
                }
                for c in self.case_results
            ],
        }


class RHEvaluator:
    """Blind evaluation engine for the Riemann Hypothesis benchmark."""

    def __init__(self):
        self.compiler = CanonicalEquationCompiler()
        self.math_engine = MathematicalAssuranceEngine()

    def evaluate_case(
        self,
        case_id: str,
        case_name: str,
        expected_state: str,
        ground_truth_form: str,
        investigation_result: dict[str, Any],
        ood_x: dict[str, np.ndarray],
        ood_y: np.ndarray,
        claimed_universal_proof: bool = False,
    ) -> RHCaseResult:
        """Evaluates an individual APEDP investigation on a Riemann benchmark problem."""
        decision = investigation_result.get("decision", {})
        actual_state = decision.get("state", EpistemicState.INSUFFICIENT_EVIDENCE.value)
        winner_id = decision.get("winner")

        candidates = investigation_result.get("candidates", [])
        winner_cand = next((c for c in candidates if c["candidate_id"] == winner_id), None)
        top_expr = winner_cand["expression"] if winner_cand else ""
        if not top_expr and candidates:
            top_expr = candidates[0].get("expression", "")

        complexity = float(winner_cand.get("complexity", 1.0)) if winner_cand else 1.0

        # 1. Structure and Algebraic Recovery
        is_exact = False
        is_algebraic = False
        is_struct = False
        if top_expr and ground_truth_form:
            try:
                verdict = self.compiler.classify_recovery(top_expr, ground_truth_form)
                is_exact = verdict.is_exact
                is_algebraic = verdict.is_algebraic
                is_struct = verdict.is_structural or is_algebraic or is_exact
            except Exception:  # noqa: BLE001 - best-effort probe must not crash
                is_struct = (str(top_expr).strip() == str(ground_truth_form).strip())

        # 2. Out-of-Distribution Error
        ood_mse = float("inf")
        if top_expr:
            try:
                # Evaluate expression on ood_x
                sp.Symbol("gamma") if "gamma" in ood_x else (sp.Symbol("n") if "n" in ood_x else None)
                parsed = self.compiler.normalize(top_expr)
                if parsed.is_number:
                    pred_val = float(parsed.evalf())
                    ood_mse = float(np.mean((ood_y - pred_val) ** 2))
                else:
                    var_names = sorted(ood_x.keys())
                    f = sp.lambdify([sp.Symbol(v) for v in var_names], parsed, modules=["numpy"])
                    args = [ood_x[v] for v in var_names]
                    y_pred = np.asarray(f(*args), dtype=float)
                    ood_mse = float(np.mean((ood_y - y_pred) ** 2))
            except Exception:  # noqa: BLE001 - best-effort probe must not crash
                ood_mse = 1.0

        # 3. False Discovery Check (asserted UNIQUE_SUPPORT on ambiguous, fake-RH, or insufficient data)
        is_false_discovery = False
        if (expected_state in [EpistemicState.MULTI_HYPOTHESIS.value, EpistemicState.INSUFFICIENT_EVIDENCE.value]
                and actual_state == EpistemicState.UNIQUE_SUPPORT.value):
                # If ground truth was fake and engine declared unique critical line 0.5
                is_false_discovery = True

        # 4. Ambiguity Detection Check
        is_ambiguity_detected = False
        if expected_state == EpistemicState.MULTI_HYPOTHESIS.value:
            is_ambiguity_detected = (actual_state == EpistemicState.MULTI_HYPOTHESIS.value)

        # 5. Proof Boundary Guard Compliance
        # Engine is compliant if it NEVER claims universal proof from finite data
        guard_compliant = not claimed_universal_proof
        # If the investigation output has proof status, check it
        math_audit = self.math_engine.audit_mathematical_candidate(
            top_expr or "0.5",
            {"gamma": (14.0, 500.0)},
            sample_observations=ood_x,
            claimed_proof=claimed_universal_proof,
        )
        if math_audit.proof_status == ProofBoundaryStatus.PROOF_NOT_ESTABLISHED and claimed_universal_proof:
            guard_compliant = False

        return RHCaseResult(
            case_id=case_id,
            case_name=case_name,
            expected_state=expected_state,
            actual_state=actual_state,
            top_expression=top_expr,
            ground_truth_form=ground_truth_form,
            is_structure_recovered=is_struct,
            is_algebraic_equivalent=is_algebraic,
            is_false_discovery=is_false_discovery,
            is_ambiguity_detected=is_ambiguity_detected,
            proof_guard_compliant=guard_compliant,
            near_ood_mse=ood_mse,
            complexity=complexity,
            rationale=decision.get("adjudication_rationale", ""),
            proof_claim=math_audit.proof_claim.value,
        )

    def compile_scorecard(self, results: list[RHCaseResult]) -> RHScorecard:
        """Aggregates individual case results into the official RH Scorecard."""
        n = len(results)
        if n == 0:
            return RHScorecard(0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, [])

        struct_count = sum(1 for r in results if r.is_structure_recovered)
        alg_count = sum(1 for r in results if r.is_algebraic_equivalent)
        false_count = sum(1 for r in results if r.is_false_discovery)
        guard_count = sum(1 for r in results if r.proof_guard_compliant)

        # Ambiguity cases: expected MULTI_HYPOTHESIS
        ambig_cases = [r for r in results if r.expected_state == EpistemicState.MULTI_HYPOTHESIS.value]
        ambig_count = sum(1 for r in ambig_cases if r.is_ambiguity_detected)
        ambig_rate = ambig_count / len(ambig_cases) if ambig_cases else 1.0

        ood_errs = [r.near_ood_mse for r in results if math.isfinite(r.near_ood_mse)]
        med_ood = float(np.median(ood_errs)) if ood_errs else 0.0
        mean_comp = float(np.mean([r.complexity for r in results]))

        return RHScorecard(
            total_cases=n,
            structure_discovery_rate=struct_count / n,
            algebraic_recovery_rate=alg_count / n,
            false_discovery_rate=false_count / n,
            ambiguity_detection_rate=ambig_rate,
            proof_boundary_compliance_rate=guard_count / n,
            median_ood_error=med_ood,
            mean_complexity=mean_comp,
            case_results=results,
        )
