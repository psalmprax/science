"""Independent adversarial falsifier: the active counterexample search."""
from __future__ import annotations

import logging
from typing import Any

import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.safety import safe_parse_expr

from .constraints import (
    ContradictoryConstraintEngine,
)
from .counterexamples import (
    Counterexample,
    unrepairable_counterexamples,
)
from .probe_support import _ProbeSupportMixin
from .probes_boundary import _BoundaryProbesMixin
from .probes_empirical import _EmpiricalProbesMixin
from .probes_mathematical import _MathematicalProbesMixin
from .probes_numerical import _NumericalProbesMixin
from .probes_structural import _StructuralProbesMixin
from .report import (
    FalsificationReport,
)
from .taxonomy import (
    FailureClass,
    FalsificationAxis,
)

logger = logging.getLogger("apedp.falsification")

class IndependentAdversarialFalsifier(
    _ProbeSupportMixin,
    _MathematicalProbesMixin,
    _BoundaryProbesMixin,
    _NumericalProbesMixin,
    _StructuralProbesMixin,
    _EmpiricalProbesMixin,
):
    """Adversarial Referee with zero access to generator biases or preferred assumptions."""

    def __init__(self, dimensional_engine: DimensionalAssuranceEngine | None = None):
        self.dim_engine = dimensional_engine or DimensionalAssuranceEngine()


    def falsify(
        self,
        formula: str | sp.Expr,
        variable_domains: dict[str, tuple[float, float]],
        physical_invariants: list[str] | None = None,
        critical_boundaries: dict[str, list[float]] | None = None,
        empirical_data: dict[str, Any] | None = None,
        variable_units: dict[str, str] | None = None,
        expected_output_unit: str | None = None,
        allow_repair_attempt: bool = True,
        constraints: list[str] | None = None
    ) -> FalsificationReport:
        """Executes an exhaustive, 5-axis adversarial falsification probe against the candidate formula."""
        formula_str = str(formula)
        try:
            expr = safe_parse_expr(formula) if isinstance(formula, str) else formula
        except (ValueError, TypeError) as e:
            ce = Counterexample(
                axis=FalsificationAxis.MATHEMATICAL,
                probe_name="SympifyParseProbe", failure_class=FailureClass.SYMBOLIC_PARSE_FAILURE,
                counterexample_point=formula_str,
                violation_description=f"Formula could not be parsed as valid mathematical expression: {e}",
                severity="FATAL"
            )
            return FalsificationReport(
                formula_str=formula_str,
                verdict="FALSIFIED",
                survived=False,
                cannot_repair=True,
                counterexamples=[ce],
                refusal_reason="SYNTAX_UNPARSEABLE: Formula is mathematically ill-formed.",
                axes_tested=[FalsificationAxis.MATHEMATICAL.value],
                epistemic_qualification="Unparseable mathematical expression."
            )

        # Check for mutually contradictory physical or mathematical constraints
        if constraints:
            infeas_cert = ContradictoryConstraintEngine.check_incompatibility(constraints)
            if infeas_cert and infeas_cert.is_infeasible:
                ce = Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="ContradictoryConstraintProbe", failure_class=FailureClass.CONTRADICTORY_CONSTRAINTS,
                    counterexample_point={"conflicts": infeas_cert.conflicting_constraints},
                    violation_description=infeas_cert.incompatibility_proof,
                    severity="FATAL"
                )
                return FalsificationReport(
                    formula_str=formula_str,
                    verdict="REFUSED_INCOMPATIBLE_CONSTRAINTS",
                    survived=False,
                    cannot_repair=True,
                    counterexamples=[ce],
                    refusal_reason=infeas_cert.incompatibility_proof,
                    axes_tested=[FalsificationAxis.STRUCTURAL_INVARIANT.value],
                    detailed_metrics={"infeasibility_certificate": infeas_cert.to_dict()},
                    epistemic_qualification="Incompatible constraints: Proof of non-existence established."
                )

        counterexamples: list[Counterexample] = []
        axes_tested: list[str] = [a.value for a in FalsificationAxis]
        metrics: dict[str, Any] = {}

        sym_names = [str(s) for s in expr.free_symbols]
        for v in sym_names:
            if v not in variable_domains:
                variable_domains[v] = (0.001, 100.0)

        # 1. Axis 1: Mathematical Counterexamples
        math_ces, math_m = self._probe_mathematical_counterexamples(expr, variable_domains, critical_boundaries)
        counterexamples.extend(math_ces)
        metrics["mathematical"] = math_m

        # 2. Axis 2: Boundary Counterexamples
        bound_ces, bound_m = self._probe_boundary_counterexamples(expr, variable_domains, critical_boundaries)
        counterexamples.extend(bound_ces)
        metrics["boundary"] = bound_m

        # 3. Axis 3: Numerical Stability Counterexamples
        num_ces, num_m = self._probe_numerical_counterexamples(expr, variable_domains)
        counterexamples.extend(num_ces)
        metrics["numerical"] = num_m

        # 4. Axis 4: Structural & Physical Invariants
        invariants = physical_invariants or []
        inv_ces, inv_m = self._probe_structural_invariants(
            expr, variable_domains, invariants, variable_units, expected_output_unit
        )
        counterexamples.extend(inv_ces)
        metrics["invariants"] = inv_m

        # 5. Axis 5: Empirical Validation (if empirical data provided)
        if empirical_data is not None and "x" in empirical_data and "y" in empirical_data:
            emp_ces, emp_m = self._probe_empirical_fidelity(expr, empirical_data)
            counterexamples.extend(emp_ces)
            metrics["empirical"] = emp_m

        # Every probe that could not decide is published in one flat list, tagged
        # with its axis, so a caller does not have to know the per-axis layout.
        probe_errors: list[dict[str, Any]] = [
            {"axis": axis, **error}
            for axis, axis_metrics in metrics.items()
            if isinstance(axis_metrics, dict)
            for error in axis_metrics.get("probe_errors", [])
        ]
        metrics["probe_errors"] = probe_errors
        metrics["probe_error_count"] = len(probe_errors)
        # Errors with a measured fallback are degradations; the rest are holes.
        # An error that does not carry the flag is treated as a hole, not a degradation.
        unresolved_errors = [e for e in probe_errors if e.get("unresolved", True)]
        metrics["unresolved_probe_count"] = len(unresolved_errors)

        # Evaluate verdict and refusal logic
        fatal_count = sum(1 for c in counterexamples if c.severity == "FATAL")
        survived = (len(counterexamples) == 0)

        if survived and unresolved_errors:
            # Nothing was falsified, but not everything was measured either. A
            # clean bill of health is only issued for a fully evaluated candidate.
            unresolved = sorted({f"{e['axis']}:{e['probe']}" for e in unresolved_errors})
            return FalsificationReport(
                formula_str=formula_str,
                verdict="INCONCLUSIVE_PROBE_ERROR",
                survived=False,
                cannot_repair=False,
                counterexamples=[],
                refusal_reason="PROBE_ERROR_NOT_EVALUABLE: " + ", ".join(unresolved),
                axes_tested=axes_tested,
                detailed_metrics=metrics,
                epistemic_qualification=(
                    f"{len(unresolved_errors)} probe(s) could not decide, so no counterexample "
                    "was found but no clean bill of health is warranted: an unevaluable "
                    "check is not a passed check."
                ),
            )

        if survived:
            domain_summary = ", ".join(f"{k} in [{v[0]:.2e}, {v[1]:.2e}]" for k, v in variable_domains.items())
            epistemic_note = (
                f"No counterexample found within tested domain ({domain_summary}), "
                f"relative precision tol=1e-6, active invariants, and 5 orthogonal falsification axes. "
                f"Provisional validity established within audited envelope (not universal proof)."
            )
            return FalsificationReport(
                formula_str=formula_str,
                verdict="SURVIVED_PROVISIONAL",
                survived=True,
                cannot_repair=False,
                counterexamples=[],
                refusal_reason=None,
                axes_tested=axes_tested,
                detailed_metrics=metrics,
                epistemic_qualification=epistemic_note
            )

        # Check whether the flaws indicate an unrepairable / ill-posed physical
        # violation. The decision is made on the structured failure class of each
        # counterexample; the descriptions are carried into the refusal text for a
        # human reader but are never matched against.
        metrics["failure_classes"] = sorted({c.failure_class.value for c in counterexamples})
        unrepairable_reasons = [
            c.violation_description for c in unrepairable_counterexamples(counterexamples)
        ]

        if unrepairable_reasons or (not allow_repair_attempt and fatal_count > 0):
            reason_text = "NO_DEFENSIBLE_REGULARIZATION_FOUND: " + " | ".join(unrepairable_reasons[:2]) if unrepairable_reasons else "FATAL_FLAW_DETECTED"
            return FalsificationReport(
                formula_str=formula_str,
                verdict="REFUSED_CANNOT_REPAIR",
                survived=False,
                cannot_repair=True,
                counterexamples=counterexamples,
                refusal_reason=reason_text,
                axes_tested=axes_tested,
                detailed_metrics=metrics,
                epistemic_qualification="Model suffers from unrepairable physical/mathematical violation."
            )

        return FalsificationReport(
            formula_str=formula_str,
            verdict="FALSIFIED",
            survived=False,
            cannot_repair=False,
            counterexamples=counterexamples,
            refusal_reason=None,
            axes_tested=axes_tested,
            detailed_metrics=metrics,
            epistemic_qualification="Counterexamples discovered within audited domain."
        )


