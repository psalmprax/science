"""Specification-feasibility classification and the three-gate validator."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy as sp

from apedp.epistemic import ModelEvidenceRecord, TestableModelPackage
from apedp.falsification import (
    ContradictoryConstraintEngine,
    FalsificationAxis,
    IndependentAdversarialFalsifier,
)
from apedp.safety import safe_parse_expr

from .checks import (
    _PROBE_ERRORS,
)


class SpecificationFeasibility(str, Enum):
    """5-way scientific feasibility classification of proposed problem specifications."""
    SOLVABLE = "SOLVABLE"                # Admissible model space is non-empty and regularizable
    INCONSISTENT = "INCONSISTENT"        # Contradictory/impossible constraints (provably empty)
    UNDERDETERMINED = "UNDERDETERMINED"  # Multiple distinct admissible models indistinguishable by data
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"      # Request falls outside physics/ontology envelope
    UNTESTABLE = "UNTESTABLE"            # Model/hypothesis makes no empirically testable or distinguishing predictions


class EpistemicActionStance(str, Enum):
    """Scientific control plane actions corresponding to the 5-way feasibility classification."""
    SOLVE = "SOLVE"                      # Generate/refine admissible candidate model
    REFUSE = "REFUSE"                    # Issue non-existence proof certificate and reject specification
    AGNOSTIC = "AGNOSTIC"                # Report underdetermination and identify maximal-discrimination experiment
    ABSTAIN = "ABSTAIN"                  # Abstain from ungrounded promotion due to domain invalidity
    FLAG_UNTESTABLE = "FLAG_UNTESTABLE"  # Flag untestable hypothesis and request observable consequences


def map_feasibility_to_stance(feasibility: SpecificationFeasibility) -> EpistemicActionStance:
    """Map a 5-way specification feasibility classification to its epistemic action stance."""
    mapping = {
        SpecificationFeasibility.SOLVABLE: EpistemicActionStance.SOLVE,
        SpecificationFeasibility.INCONSISTENT: EpistemicActionStance.REFUSE,
        SpecificationFeasibility.UNDERDETERMINED: EpistemicActionStance.AGNOSTIC,
        SpecificationFeasibility.OUT_OF_DOMAIN: EpistemicActionStance.ABSTAIN,
        SpecificationFeasibility.UNTESTABLE: EpistemicActionStance.FLAG_UNTESTABLE,
    }
    return mapping[feasibility]


def classify_specification(package: TestableModelPackage) -> SpecificationFeasibility:
    """Classify a specification by running the platform's real feasibility tests.

    Decision order, most specific first:

    1. ``INCONSISTENT`` -- :class:`ContradictoryConstraintEngine` proves the declared
       constraints are mutually unsatisfiable.
    2. ``UNTESTABLE`` -- the specification declares no observable predictions or no
       falsification criteria, so nothing could refute it.
    3. ``OUT_OF_DOMAIN`` -- the mathematical/boundary axes of the
       :class:`IndependentAdversarialFalsifier` find a singularity inside the
       declared domain (best effort: prose or unparseable equations skip this test).
    4. ``SOLVABLE`` otherwise.

    ``UNDERDETERMINED`` is deliberately never returned: distinguishing competing
    models requires observational data and their discrimination point, which a
    static specification does not carry.
    """
    certificate = ContradictoryConstraintEngine.check_incompatibility(list(package.constraints))
    if certificate is not None and certificate.is_infeasible:
        return SpecificationFeasibility.INCONSISTENT

    if not package.predictions or not package.falsification_criteria:
        return SpecificationFeasibility.UNTESTABLE

    if _has_singularity_inside_declared_domain(package):
        return SpecificationFeasibility.OUT_OF_DOMAIN

    return SpecificationFeasibility.SOLVABLE


def _has_singularity_inside_declared_domain(package: TestableModelPackage) -> bool:
    """Real falsifier probe for a pole/divergence inside the declared domain.

    Returns False when the equation cannot be parsed (prose forms), since the
    check is simply not applicable rather than passed.
    """
    try:
        expr = safe_parse_expr(
            package.equation_str,
            extra_symbols={name: sp.Symbol(name) for name in package.domain},
        )
    except (ValueError, TypeError):
        return False
    report = IndependentAdversarialFalsifier().falsify(
        formula=expr,
        variable_domains=dict(package.domain),
    )
    return any(
        c.axis in (FalsificationAxis.MATHEMATICAL, FalsificationAxis.BOUNDARY)
        for c in report.counterexamples
    )


@dataclass(frozen=True)
class ScientificFeasibilityConfusionMatrix:
    """Confusion matrix comparing feasibility classifications against ground truth.

    Counts are never pre-seeded. A matrix built without observations reports zero
    cases and a score of 0.0 -- it does not claim a perfect record by default.
    Populate it from real runs with :meth:`from_classifications` (label pairs) or
    :meth:`evaluate_specifications`, which runs :func:`classify_specification` over
    labelled :class:`TestableModelPackage` inputs.

    Binarisation: ground truth is "solvable" vs "not solvable"; the system's
    decision is "its stance is SOLVE" vs "it refuses to solve".
    """
    true_solvable: int = 0
    false_unsolvable: int = 0
    true_contradictory: int = 0
    false_solvable: int = 0

    @classmethod
    def from_classifications(
        cls,
        observations: Iterable[tuple[SpecificationFeasibility, SpecificationFeasibility]],
    ) -> ScientificFeasibilityConfusionMatrix:
        """Tabulate ``(ground_truth, predicted)`` feasibility label pairs."""
        true_solvable = false_unsolvable = true_contradictory = false_solvable = 0
        for truth, predicted in observations:
            truth_solves = map_feasibility_to_stance(truth) == EpistemicActionStance.SOLVE
            predicted_solves = map_feasibility_to_stance(predicted) == EpistemicActionStance.SOLVE
            if truth_solves and predicted_solves:
                true_solvable += 1
            elif truth_solves and not predicted_solves:
                false_unsolvable += 1
            elif not truth_solves and not predicted_solves:
                true_contradictory += 1
            else:
                false_solvable += 1
        return cls(
            true_solvable=true_solvable,
            false_unsolvable=false_unsolvable,
            true_contradictory=true_contradictory,
            false_solvable=false_solvable,
        )

    @classmethod
    def evaluate_specifications(
        cls,
        labelled_specifications: Iterable[tuple[TestableModelPackage, SpecificationFeasibility]],
    ) -> ScientificFeasibilityConfusionMatrix:
        """Run the real feasibility classifier over labelled specifications.

        The caller supplies ground-truth labels (established independently, e.g.
        from a curated benchmark); the system's own verdict is what is scored.
        """
        return cls.from_classifications(
            (truth, classify_specification(package))
            for package, truth in labelled_specifications
        )

    @property
    def total_cases(self) -> int:
        return self.true_solvable + self.false_unsolvable + self.true_contradictory + self.false_solvable

    @property
    def solve_recall(self) -> float:
        """Recall on solvable models: TP / (TP + FN)."""
        denom = self.true_solvable + self.false_unsolvable
        return float(self.true_solvable / denom) if denom > 0 else 0.0

    @property
    def refusal_specificity(self) -> float:
        """Specificity on contradictory models: TN / (TN + FP)."""
        denom = self.true_contradictory + self.false_solvable
        return float(self.true_contradictory / denom) if denom > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "true_solvable_tp": self.true_solvable,
            "false_unsolvable_fn": self.false_unsolvable,
            "true_contradictory_tn": self.true_contradictory,
            "false_solvable_fp": self.false_solvable,
            "solve_recall": self.solve_recall,
            "refusal_specificity": self.refusal_specificity,
        }


# Three-Gate Validation with Proper Dimensional Analysis

def validate_three_gates(candidate_expr: str, record: ModelEvidenceRecord) -> dict:
    """
    Validate a candidate expression against the three-gate validation system.
    
    Gate 1: Dimensional Plausibility
    - Uses DimensionalAssuranceEngine to check that the expression has well-defined dimensions
    - Passes if: expression has valid dimensions compatible with a physical quantity
    - Fails if: expression is dimensionally invalid or references undeclared symbols
    
    Gate 2: Anomaly Check
    - Checks for critical anomalies (division by zero, dimensional violations)
    - Passes if: no critical anomalies detected
    - Fails if: division/dimensional anomalies present
    
    Gate 3: Verdict Acceptability
    - Checks that the verdict is not REJECTED in the new 8-state classification
    - Passes if: classification is ESTABLISHED, DERIVED, PHYSICALLY_PLAUSIBLE, 
      CANDIDATE, or HEURISTIC
    - Fails if: classification is IDENTITY, DIMENSIONALLY_INVALID, or PHYSICALLY_UNSUPPORTED
    
    Returns:
        {
            "gate1": {"pass": bool, "detail": str},
            "gate2": {"pass": bool, "detail": str},
            "gate3": {"pass": bool, "detail": str},
            "overall": {"pass": bool, "classification": Classification},
            "recommendation": str
        }
    """
    import sympy as sp

    from apedp.classification_ontology import Classification
    from apedp.gated_assurance import _measure_dimensional_consistency
    from apedp.identity_detection import detect_identity
    
    result: dict[str, Any] = {
        "gate1": {"pass": False, "detail": ""},
        "gate2": {"pass": False, "detail": ""},
        "gate3": {"pass": False, "detail": ""},
        "overall": {"pass": False, "classification": Classification.UNASSIGNED},
        "recommendation": "",
    }
    
    # Gate 1: Dimensional Plausibility
    try:
        dim_result = _measure_dimensional_consistency(record)
        if dim_result[0] == "PASS":
            result["gate1"] = {"pass": True, "detail": "Expression has valid dimensional consistency"}
        elif dim_result[0] == "UNVERIFIED":
            result["gate1"] = {"pass": False, "detail": "Dimensional consistency unverified - no unit metadata"}
        else:
            result["gate1"] = {"pass": False, "detail": f"Dimensional inconsistency: {dim_result[1]}"}
    except _PROBE_ERRORS as e:
        result["gate1"] = {"pass": False, "detail": f"Dimensional analysis error: {e!s}"}
    
    # Gate 2: Anomaly Check
    try:
        expr = sp.sympify(candidate_expr)
        is_identity, _, _ = detect_identity(expr)
        if is_identity:
            result["gate2"] = {"pass": False, "detail": "Expression is a mathematical identity, not a physical law"}
        else:
            result["gate2"] = {"pass": True, "detail": "No critical anomalies detected"}
    except _PROBE_ERRORS as e:
        result["gate2"] = {"pass": False, "detail": f"Anomaly check error: {e!s}"}
    
    # Gate 3: Verdict Acceptability
    try:
        # Check the record's classification/verdict
        if hasattr(record, 'status') and record.status:
            status = str(record.status)
            # Map old statuses to new classification
            status_map = {
                "APPROVED": Classification.ESTABLISHED,
                "REJECTED": Classification.DIMENSIONALLY_INVALID,
                "REPAIRED": Classification.ESTABLISHED,
                "UNDETERMINED": Classification.CANDIDATE,
            }
            classification = status_map.get(status, Classification.CANDIDATE)
            
            # Check if classification is acceptable
            prohibited = [Classification.IDENTITY, Classification.DIMENSIONALLY_INVALID, 
                         Classification.PHYSICALLY_UNSUPPORTED]
            if classification in prohibited:
                result["gate3"] = {"pass": False, "detail": f"Verdict {classification.name} not acceptable"}
            else:
                result["gate3"] = {"pass": True, "detail": f"Verdict {classification.name} acceptable"}
        else:
            result["gate3"] = {"pass": True, "detail": "No verdict to check - new expression"}
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        result["gate3"] = {"pass": False, "detail": f"Verdict check error: {e!s}"}
    
    # Overall assessment
    all_pass = all(g["pass"] for g in [result["gate1"], result["gate2"], result["gate3"]])
    result["overall"]["pass"] = all_pass
    
    if all_pass:
        # Determine the best classification from the results
        classification = result["gate3"]["detail"].split(":")[-1].strip() if ":" in result["gate3"]["detail"] else Classification.CANDIDATE
        result["overall"]["classification"] = classification
        result["recommendation"] = "Proceed with validation and numerical testing"
    else:
        prohibited_reasons = []
        if not result["gate1"]["pass"]:
            prohibited_reasons.append(result["gate1"]["detail"])
        if not result["gate2"]["pass"]:
            prohibited_reasons.append(result["gate2"]["detail"])
        if not result["gate3"]["pass"]:
            prohibited_reasons.append(result["gate3"]["detail"])
        result["recommendation"] = f"Fix failed gates: {'; '.join(prohibited_reasons)}"
    
    return result
