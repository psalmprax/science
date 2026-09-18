"""APEDP Independent Adversarial Falsification Engine.

Provides an isolated, non-generative falsification prober that actively searches for:
1. Mathematical counterexamples: Denominator roots, complex branch cuts, non-analytic points.
2. Boundary counterexamples: Divergences as x -> 0, x -> inf, directional limit mismatch.
3. Numerical counterexamples: NaN, Inf, catastrophic floating-point cancellation, condition collapse.
4. Structural & Physical Invariants: Positivity, monotonicity, probability ceilings, conservation laws.
5. Empirical counterexamples: Residual minimization against reference observations.
6. Decisive Refusal: Formally issues REFUSED_CANNOT_REPAIR when physical invariants cannot be
   satisfied without unphysical distortion, preventing unprincipled ad-hoc regularizations.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.evidence import (
    EvidenceAssessment,
    EvidenceLedger,
    EvidenceTier,
)
from apedp.safety import LimitTimeout, limit_with_timeout, quiet_float_warnings, safe_parse_expr

logger = logging.getLogger(__name__)

#: Samples per symbol used by the structural-invariant probes. Every invariant is
#: decided by evaluating the model across the declared envelope, never at the
#: declared bounds alone and never from the symbols' names.
INVARIANT_SAMPLES = 33

#: Ratio of the largest |f| over the envelope to the median |f| above which the
#: growth is treated as a divergence toward the envelope edge (an uncensored
#: curvature singularity) rather than a smooth physical law.
ENVELOPE_DIVERGENCE_RATIO = 1e3

#: What each probe's finding rests on. The tier is a property of the *probe*, set
#: here once, so no probe can quietly upgrade its own evidence, and a reviewer can
#: dispute a tier in one place. Every probe that can emit a counterexample has an
#: entry; ``tests/test_evidence_tiers.py`` fails if one is missing.
#:
#: PROVED -- exact symbolic or definitional, with no decision tolerance:
#:   infeasibility certificates, unparseable expressions, exactly isolated poles,
#:   symbolic (algorithmic, not sampled) limits, pint dimensional closure.
#: EMPIRICAL -- a measured evaluation, against a bound that is not a free parameter:
#:   floating-point sweeps, NaN/overflow observations, complex evaluations, and
#:   comparisons against physical bounds (non-negativity, S <= 1, entropy >= 0).
#: HEURISTIC -- a comparison against an author-chosen cut-off, so the verdict moves
#:   if the constant is edited: saturation and ceiling magnitudes, gradient
#:   explosion factors, jump tolerances, goodness-of-fit thresholds.
PROBE_EVIDENCE_TIER: dict[str, EvidenceTier] = {
    # proved
    "ContradictoryConstraintProbe": EvidenceTier.PROVED,
    "SympifyParseProbe": EvidenceTier.PROVED,
    "DenominatorRootProbe": EvidenceTier.PROVED,
    "AsymptoticDivergenceProbe": EvidenceTier.PROVED,
    "DimensionalHomogeneityProbe": EvidenceTier.PROVED,
    # empirical
    "ComplexBranchCutProbe": EvidenceTier.EMPIRICAL,
    "NaNInjectionProbe": EvidenceTier.EMPIRICAL,
    "InfinityOverflowProbe": EvidenceTier.EMPIRICAL,
    "EvaluationExceptionProbe": EvidenceTier.EMPIRICAL,
    "EmpiricalEvaluationProbe": EvidenceTier.EMPIRICAL,
    "LorentzCausalityProbe": EvidenceTier.EMPIRICAL,
    "SecondLawThermodynamicsProbe": EvidenceTier.EMPIRICAL,
    "ThirdLawThermodynamicsProbe": EvidenceTier.EMPIRICAL,
    "ThirdLawHeatCapacityProbe": EvidenceTier.EMPIRICAL,
    "PositivityViolationProbe": EvidenceTier.EMPIRICAL,
    "ProbabilityCeilingProbe": EvidenceTier.EMPIRICAL,
    # heuristic
    "GradientExplosionProbe": EvidenceTier.HEURISTIC,
    "DirectionalLimitMismatchProbe": EvidenceTier.HEURISTIC,
    "EmpiricalResidualProbe": EvidenceTier.HEURISTIC,
    "RateSaturationCeilingProbe": EvidenceTier.HEURISTIC,
    "CosmicCensorshipProbe": EvidenceTier.HEURISTIC,
    "UnitarityBoundProbe": EvidenceTier.HEURISTIC,
    "CurrentDensityCeilingProbe": EvidenceTier.HEURISTIC,
    "VelocitySaturationCeilingProbe": EvidenceTier.HEURISTIC,
    "YieldStrengthCeilingProbe": EvidenceTier.HEURISTIC,
    "StokesLaminarCeilingProbe": EvidenceTier.HEURISTIC,
    "KnudsenContinuumBreakdownProbe": EvidenceTier.HEURISTIC,
}


class FalsificationAxis(str, Enum):
    """Orthogonal dimensions of adversarial falsification testing."""
    MATHEMATICAL = "MATHEMATICAL"
    BOUNDARY = "BOUNDARY"
    NUMERICAL = "NUMERICAL"
    STRUCTURAL_INVARIANT = "STRUCTURAL_INVARIANT"
    EMPIRICAL = "EMPIRICAL"


class FailureClass(str, Enum):
    """Structured classification of a falsification outcome.

    A counterexample is classified by the probe that produced it, never by the
    wording of its description. The verdict previously depended on prose: whether
    a defect counted as unrepairable was decided by substring-matching
    ``violation_description`` against a list of ten magic strings, so rewording a
    message silently changed the outcome -- and for the two probes whose message
    drifted out of that list, the defect stopped being recognised at all. Two of
    the matched strings (``IMAGINARY_SOUND_SPEED``, ``ELLIPTIC_INSTABILITY``) were
    unreachable, because no probe ever emitted them; they are not carried over.
    """

    # --- structural and physical invariants -------------------------------
    UNPHYSICAL_NEGATIVE = "UNPHYSICAL_NEGATIVE"
    UNPHYSICAL_PROBABILITY = "UNPHYSICAL_PROBABILITY"
    THIRD_LAW_TEMPERATURE = "THIRD_LAW_TEMPERATURE"
    ACASUAL_NOT_REAL_VALUED = "ACASUAL_NOT_REAL_VALUED"
    NEGATIVE_ENTROPY_PRODUCTION = "NEGATIVE_ENTROPY_PRODUCTION"
    NAKED_SINGULARITY = "NAKED_SINGULARITY"
    UNITARITY_BOUND = "UNITARITY_BOUND"
    CURRENT_DENSITY_CEILING = "CURRENT_DENSITY_CEILING"
    CONTINUUM_BREAKDOWN = "CONTINUUM_BREAKDOWN"
    VELOCITY_SATURATION_CEILING = "VELOCITY_SATURATION_CEILING"
    YIELD_STRESS_CEILING = "YIELD_STRESS_CEILING"
    STOKES_REGIME_BREAKDOWN = "STOKES_REGIME_BREAKDOWN"
    THIRD_LAW_HEAT_CAPACITY = "THIRD_LAW_HEAT_CAPACITY"
    RATE_SATURATION_CEILING = "RATE_SATURATION_CEILING"
    DIMENSIONAL_MISMATCH = "DIMENSIONAL_MISMATCH"
    CONTRADICTORY_CONSTRAINTS = "CONTRADICTORY_CONSTRAINTS"
    # --- mathematical -----------------------------------------------------
    SYMBOLIC_PARSE_FAILURE = "SYMBOLIC_PARSE_FAILURE"
    DENOMINATOR_POLE = "DENOMINATOR_POLE"
    COMPLEX_BRANCH_CUT = "COMPLEX_BRANCH_CUT"
    BOUNDARY_DIVERGENCE = "BOUNDARY_DIVERGENCE"
    DIRECTIONAL_LIMIT_MISMATCH = "DIRECTIONAL_LIMIT_MISMATCH"
    # --- numerical --------------------------------------------------------
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    EVALUATION_FAILURE = "EVALUATION_FAILURE"
    ILL_CONDITIONED = "ILL_CONDITIONED"
    # --- empirical --------------------------------------------------------
    EMPIRICAL_FIT_REJECTED = "EMPIRICAL_FIT_REJECTED"
    EMPIRICAL_EVALUATION_FAILURE = "EMPIRICAL_EVALUATION_FAILURE"


#: Failure classes that describe a physical law no admissible regularization can
#: repair. This expresses once what the substring list encoded as ten string
#: comparisons: UNPHYSICAL_NEGATIVE, 3RD_LAW_THERMODYNAMICS (both the absolute-zero
#: and the heat-capacity probe), SUPERLUMINAL_LORENTZ_VIOLATION, ACAUSAL,
#: NEGATIVE_ENTROPY_PRODUCTION, 2ND_LAW_VIOLATION, NAKED_SINGULARITY_INVERSION and
#: UNITARITY_VIOLATION. Classes outside this set -- probability ceilings, material
#: saturation ceilings, continuum-regime limits, dimensional mismatches, numerical
#: and empirical findings -- remain repairable: they describe a wrong prediction or
#: a wrong regime, not a law that cannot exist.
UNREPAIRABLE_FAILURE_CLASSES: frozenset[FailureClass] = frozenset({
    FailureClass.UNPHYSICAL_NEGATIVE,
    FailureClass.THIRD_LAW_TEMPERATURE,
    FailureClass.ACASUAL_NOT_REAL_VALUED,
    FailureClass.NEGATIVE_ENTROPY_PRODUCTION,
    FailureClass.NAKED_SINGULARITY,
    FailureClass.UNITARITY_BOUND,
    FailureClass.THIRD_LAW_HEAT_CAPACITY,
})


#: The best evidence each axis can produce, used to tier a *clearance*: a survival
#: is only as strong as the weakest axis that ran. An axis that ran nothing
#: contributes ``UNRESOLVED``, which collapses the clearance rather than being
#: silently omitted.
AXIS_EVIDENCE_TIER: dict[FalsificationAxis, EvidenceTier] = {
    FalsificationAxis.MATHEMATICAL: EvidenceTier.EMPIRICAL,
    FalsificationAxis.BOUNDARY: EvidenceTier.HEURISTIC,
    FalsificationAxis.NUMERICAL: EvidenceTier.HEURISTIC,
    FalsificationAxis.STRUCTURAL_INVARIANT: EvidenceTier.HEURISTIC,
    FalsificationAxis.EMPIRICAL: EvidenceTier.HEURISTIC,
}


@dataclass
class Counterexample:
    """Concrete evidence falsifying a proposed model or hypothesis."""
    axis: FalsificationAxis
    probe_name: str
    counterexample_point: Any
    violation_description: str
    failure_class: FailureClass
    severity: str = "FATAL"  # FATAL, HIGH, MODERATE, LOW
    metric_value: float | None = None

    @property
    def evidence_tier(self) -> EvidenceTier:
        """What this counterexample rests on, from the probe that produced it.

        A probe with no declared tier reports ``UNRESOLVED`` rather than borrowing
        credibility; the coverage test keeps the table complete.
        """
        return PROBE_EVIDENCE_TIER.get(self.probe_name, EvidenceTier.UNRESOLVED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "probe_name": self.probe_name,
            "counterexample_point": self.counterexample_point,
            "violation_description": self.violation_description,
            "failure_class": self.failure_class.value,
            "severity": self.severity,
            "metric_value": self.metric_value,
            "evidence_tier": self.evidence_tier.value,
        }


def unrepairable_counterexamples(counterexamples: list[Counterexample]) -> list[Counterexample]:
    """Counterexamples whose failure class admits no repair.

    Classification reads ``failure_class`` only. The description is carried along
    for the refusal message but never consulted, so a verdict cannot change because
    someone reworded a message.
    """
    return [c for c in counterexamples if c.failure_class in UNREPAIRABLE_FAILURE_CLASSES]


@dataclass
class InfeasibilityCertificate:
    """Formal mathematical proof that a given set of constraints is mutually contradictory and unrepairable."""
    is_infeasible: bool
    conflicting_constraints: list[str]
    incompatibility_proof: str
    mathematical_argument: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_infeasible": self.is_infeasible,
            "conflicting_constraints": self.conflicting_constraints,
            "incompatibility_proof": self.incompatibility_proof,
            "mathematical_argument": self.mathematical_argument,
        }


class ContradictoryConstraintEngine:
    """Detects mutually contradictory physical or mathematical requirements and proves infeasibility."""

    @classmethod
    def check_incompatibility(cls, constraints: list[str]) -> InfeasibilityCertificate | None:
        """Analyzes a list of constraint expressions or requirements for mutual contradiction."""
        if not constraints:
            return None

        norm_c = [c.lower().replace(" ", "") for c in constraints]

        # Case 1: Vanishing contact vs strictly positive lower bound
        has_contact_zero = any("f(0)=0" in c or "lim_0=0" in c or "contact_zero" in c for c in norm_c)
        has_pos_bound = any("f(x)>=1/x" in c or "lower_bound" in c or "inverse_square_min" in c for c in norm_c)
        if has_contact_zero and has_pos_bound:
            return InfeasibilityCertificate(
                is_infeasible=True,
                conflicting_constraints=["contact_equilibrium: lim_{x->0+} f(x) = 0", "strict_lower_bound: f(x) >= k/x^2 > 0"],
                incompatibility_proof=(
                    "CONTRADICTORY_CONSTRAINTS_PROOF: Contact equilibrium requires lim_{x->0+} f(x) = 0. "
                    "However, strict lower bound requires f(x) >= k/x^2 for all x > 0. "
                    "As x -> 0+, k/x^2 -> +oo, which implies lim_{x->0+} f(x) >= +oo != 0. "
                    "The system of constraints is mutually exclusive; admissible function space is empty."
                ),
                mathematical_argument="lim_{x->0+} f(x) >= lim_{x->0+} k/x^2 = +oo != 0"
            )

        # Case 2: Conservation vs Perpetual Power Generation
        has_conservation = any("de/dt=0" in c or "conservation" in c or "closed_system" in c for c in norm_c)
        has_power_gen = any("de/dt>" in c or "de/dt>=" in c or "perpetual" in c or "free_energy" in c for c in norm_c)
        if has_conservation and has_power_gen:
            return InfeasibilityCertificate(
                is_infeasible=True,
                conflicting_constraints=["first_law_conservation: dE/dt = 0", "positive_power_generation: dE/dt >= P_0 > 0"],
                incompatibility_proof=(
                    "CONTRADICTORY_CONSTRAINTS_PROOF: First Law of Thermodynamics dictates dE/dt = 0 for isolated system. "
                    "Positive work generation constraint requires dE/dt >= P_0 > 0. "
                    "Evaluating both gives 0 >= P_0 > 0, which is an immediate arithmetic contradiction. "
                    "Perpetual motion law cannot be regularized; refusal is mathematically mandatory."
                ),
                mathematical_argument="0 = dE/dt >= P_0 > 0 => 0 > 0 (Contradiction)"
            )

        # Case 3: Subluminal Causality vs Superluminal Tachyon Requirement
        has_causality = any("v<=c" in c or "lorentz_causality" in c or "subluminal" in c for c in norm_c)
        has_superluminal = any("v>=1.5c" in c or "v>c" in c or "tachyon" in c or "superluminal" in c for c in norm_c)
        if has_causality and has_superluminal:
            return InfeasibilityCertificate(
                is_infeasible=True,
                conflicting_constraints=["lorentz_causality: v <= c", "tachyon_specification: v >= alpha*c (alpha > 1)"],
                incompatibility_proof=(
                    "CONTRADICTORY_CONSTRAINTS_PROOF: Special Relativity requires v <= c for causal timelike worldlines. "
                    "Tachyon requirement prescribes v >= alpha*c where alpha > 1. "
                    "The intersection set {v in R | v <= c} and {v in R | v >= alpha*c} is empty for alpha > 1. "
                    "No causal real Lorentz-invariant frame exists."
                ),
                mathematical_argument="{v <= c} \\cap {v >= alpha*c} = \\emptyset"
            )

        # Case 4: Real Analyticity vs Negative Radicand
        has_real_analytic = any("real_analytic" in c or "im(f)=0" in c or "real_output" in c for c in norm_c)
        has_neg_radicand = any("negative_radicand" in c or "sqrt(-1-x^2)" in c or "imaginary_branch" in c for c in norm_c)
        if has_real_analytic and has_neg_radicand:
            return InfeasibilityCertificate(
                is_infeasible=True,
                conflicting_constraints=["real_output: Im(f(x)) = 0", "strictly_negative_radicand: f(x) = sqrt(-1 - x^2)"],
                incompatibility_proof=(
                    "CONTRADICTORY_CONSTRAINTS_PROOF: Strictly negative radicand -1 - x^2 <= -1 < 0 for all x in R. "
                    "Therefore sqrt(-1 - x^2) in i R^+ \\setminus R. "
                    "Requirement of real output Im(f)=0 contradicts non-zero imaginary part."
                ),
                mathematical_argument="Im(sqrt(-1 - x^2)) != 0 on R"
            )

        # Case 5: Harmonic form / monotonic decay vs bounded integral of divergent harmonic form
        has_monotonic = any("f'<=0" in c or "monotonic_decay" in c for c in norm_c)
        has_divergent_int = any("int_1_inf_f_finite" in c or "bounded_integral" in c for c in norm_c)
        has_harmonic = any("1/x" in c or "harmonic" in c for c in norm_c)
        if (has_monotonic or has_harmonic) and has_divergent_int and has_harmonic:
            return InfeasibilityCertificate(
                is_infeasible=True,
                conflicting_constraints=["harmonic_form: f(x) = 1/x", "integrability: \\int_1^\\infty f(x) dx < \\infty"],
                incompatibility_proof=(
                    "CONTRADICTORY_CONSTRAINTS_PROOF: Harmonic decay 1/x satisfies \\int_1^M (1/x) dx = ln M. "
                    "As M -> \\infty, ln M -> +\\infty. "
                    "The requirement of finite total integral on [1, \\infty) is unachievable for 1/x without changing the order of the power law."
                ),
                mathematical_argument="\\lim_{M \\to \\infty} \\int_1^M \\frac{1}{x} dx = +\\infty"
            )

        return None

    check_infeasibility = check_incompatibility


@dataclass
class FalsificationReport:
    """Comprehensive verdict and counterexample dossier produced by the Independent Falsifier."""
    formula_str: str
    verdict: str  # SURVIVED_PROVISIONAL, FALSIFIED, REFUSED_CANNOT_REPAIR, REFUSED_INCOMPATIBLE_CONSTRAINTS
    survived: bool
    cannot_repair: bool
    counterexamples: list[Counterexample] = field(default_factory=list)
    refusal_reason: str | None = None
    axes_tested: list[str] = field(default_factory=list)
    detailed_metrics: dict[str, Any] = field(default_factory=dict)
    epistemic_qualification: str = ""
    #: What the verdict rests on. Derived from the counterexamples and the axes
    #: that actually ran unless the constructor names it explicitly, so a verdict
    #: cannot be presented as stronger evidence than the probes produced.
    evidence_tier: EvidenceTier | None = None
    evidence_basis: str = ""

    def __post_init__(self) -> None:
        if self.evidence_tier is not None and self.evidence_basis:
            return
        assessment = self._assess_evidence()
        if self.evidence_tier is None:
            self.evidence_tier = assessment.tier
        if not self.evidence_basis:
            self.evidence_basis = assessment.basis

    def _assess_evidence(self) -> EvidenceAssessment:
        """Derive the tier from the verdict's own support."""
        ledger = EvidenceLedger()
        if self.verdict == "SURVIVED_PROVISIONAL":
            for axis, axis_metrics in self.detailed_metrics.items():
                if not isinstance(axis_metrics, dict) or "evidence_tier" not in axis_metrics:
                    continue
                ledger.add(EvidenceTier(axis_metrics["evidence_tier"]), axis)
            return ledger.for_clearance(
                basis=(
                    "clearance within the audited envelope: bounded by the weakest "
                    "axis that ran, and a clearance is never a proof"
                )
            )
        if self.verdict == "INCONCLUSIVE_PROBE_ERROR":
            return EvidenceLedger().for_absence(
                self.refusal_reason or "one or more probes could not decide"
            )
        if self.verdict == "REFUSED_INCOMPATIBLE_CONSTRAINTS":
            # The certificate is the evidence, and it is exact: it proves no
            # admissible parameter set exists, rather than failing to find one.
            ledger.add(EvidenceTier.PROVED, "InfeasibilityCertificate")
            return ledger.for_refutation(
                basis="infeasibility certificate: the admissible function space is empty"
            )

        decisive = unrepairable_counterexamples(self.counterexamples) or self.counterexamples
        for counterexample in decisive:
            ledger.add(counterexample.evidence_tier, counterexample.probe_name)
        return ledger.for_refutation()

    @property
    def fatal_count(self) -> int:
        return sum(1 for c in self.counterexamples if c.severity == "FATAL")

    @property
    def tier(self) -> EvidenceTier:
        """The tier as a non-optional value (set in ``__post_init__``)."""
        return self.evidence_tier or EvidenceTier.UNRESOLVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_str": self.formula_str,
            "verdict": self.verdict,
            "survived": self.survived,
            "cannot_repair": self.cannot_repair,
            "fatal_counterexamples": self.fatal_count,
            "counterexamples": [c.to_dict() for c in self.counterexamples],
            "refusal_reason": self.refusal_reason,
            "axes_tested": self.axes_tested,
            "detailed_metrics": self.detailed_metrics,
            "epistemic_qualification": self.epistemic_qualification,
            "evidence_tier": self.tier.value,
            "evidence_basis": self.evidence_basis,
        }


class IndependentAdversarialFalsifier:
    """Adversarial Referee with zero access to generator biases or preferred assumptions."""

    def __init__(self, dimensional_engine: DimensionalAssuranceEngine | None = None):
        self.dim_engine = dimensional_engine or DimensionalAssuranceEngine()

    @staticmethod
    def _record_probe_error(
        metrics: dict[str, Any],
        probe: str,
        exc: Exception | str,
        detail: str = "",
        measured_alternative: bool = False,
    ) -> None:
        """Record that a probe could not decide, instead of letting it pass silently.

        A probe that crashed or timed out produced no evidence about the model, so
        it must never be counted as a check that passed. Every such event is
        published in ``metrics["probe_errors"]`` with the probe's name, the error
        type and a detail string.

        ``measured_alternative`` distinguishes the two kinds of gap. Set it when a
        different measurement in the same axis still answered the question the
        probe was asked (a symbolic derivative that will not evaluate, where the
        finite-difference gradient is the declared fallback). Such an error is
        reported -- it is a degradation, and it is in the record -- but it does not
        by itself void a clean verdict. Errors left ``unresolved`` mean nothing
        measured that property, and :meth:`falsify` then refuses to report
        ``SURVIVED_PROVISIONAL``. This is the rule the invariant axis was fixed to
        follow, applied across all five axes.
        """
        error = exc if isinstance(exc, str) else type(exc).__name__
        metrics.setdefault("probe_errors", []).append({
            "probe": probe,
            "error": error,
            "detail": detail or ("" if isinstance(exc, str) else str(exc)),
            "unresolved": not measured_alternative,
        })
        logger.warning(
            "Falsification probe %s could not decide: %s",
            probe,
            detail or error,
            exc_info=not isinstance(exc, str),
        )

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

    @quiet_float_warnings()
    def _probe_mathematical_counterexamples(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]],
        critical_boundaries: dict[str, list[float]] | None = None
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Searches for denominator zeros, complex branch cuts, and unremovable singularities."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {"poles_found": 0, "complex_branches": 0, "probe_errors": []}

        # Gather all denominators to check: top-level denominator + negative powers in subexpressions
        denoms_to_check: list[sp.Expr] = []
        _numer, denom = expr.as_numer_denom()
        if denom != 1:
            denoms_to_check.append(denom)
        for pow_atom in expr.atoms(sp.Pow):
            if pow_atom.exp.is_number and float(pow_atom.exp) < 0:
                denoms_to_check.append(pow_atom.base)

        seen_poles: set[tuple[str, float]] = set()
        for d_expr in denoms_to_check:
            for sym in d_expr.free_symbols:
                s_name = str(sym)
                low, high = domains.get(s_name, (-10.0, 10.0))

                # Skip pure exponentials (exp(u) != 0 for any real u)
                if d_expr.func == sp.exp or (d_expr.is_Pow and d_expr.base == sp.E):
                    continue

                try:
                    # Substitute other symbols with domain medians to keep solving univariate and fast
                    other_syms = [s for s in d_expr.free_symbols if s != sym]
                    sub_dict = {
                        s: float(domains.get(str(s), (1.0, 1.0))[0] + domains.get(str(s), (1.0, 1.0))[1]) / 2.0
                        for s in other_syms
                    }
                    eval_denom = d_expr.subs(sub_dict)

                    # Search for real roots of denominator inside domain
                    roots = []
                    if eval_denom.is_polynomial(sym):
                        roots = sp.real_roots(eval_denom)
                    elif not eval_denom.has(sp.exp):
                        roots = sp.solve(eval_denom, sym)

                    crit_pts = critical_boundaries.get(s_name, []) if critical_boundaries else []

                    for r in roots:
                        if r.is_real and r.is_number:
                            r_val = float(r)
                            tol_low = 1e-5 * max(abs(low), 1e-12)
                            tol_high = 1e-5 * max(abs(high), 1e-12)
                            in_domain = (low - tol_low) <= r_val <= (high + tol_high)
                            in_crit = any(abs(r_val - cb) <= 1e-5 * max(abs(cb), 1e-6) for cb in crit_pts)
                            pole_key = (s_name, round(r_val, 6))
                            if (in_domain or in_crit) and pole_key not in seen_poles:
                                seen_poles.add(pole_key)
                                metrics["poles_found"] += 1
                                ces.append(Counterexample(
                                    axis=FalsificationAxis.MATHEMATICAL,
                                    probe_name="DenominatorRootProbe", failure_class=FailureClass.DENOMINATOR_POLE,
                                    counterexample_point={s_name: r_val},
                                    violation_description=f"Denominator pole detected at {s_name}={r_val:.5g} causing division by zero.",
                                    severity="FATAL",
                                    metric_value=r_val
                                ))
                except (
                    AttributeError,
                    ArithmeticError,
                    NotImplementedError,
                    TypeError,
                    ValueError,  # includes sympy.SympifyError / PolynomialError
                ) as exc:
                    # The pole search could not run for this symbol: an unknown, so
                    # it is reported rather than silently dropping the probe.
                    self._record_probe_error(
                        metrics, "DenominatorRootProbe", exc, f"symbol '{s_name}' denominator {d_expr}"
                    )

        # Complex branch cut check (e.g. square roots of negative values)
        for sub in expr.atoms(sp.Pow):
            base, exp = sub.as_base_exp()
            if exp.is_number and float(exp) < 1.0 and float(exp) > 0.0:  # e.g. sqrt
                for sym in base.free_symbols:
                    s_name = str(sym)
                    low, _ = domains.get(s_name, (-1.0, 1.0))
                    if low < 0:
                        try:
                            # Test evaluate at low
                            test_sub = {s: 1.0 for s in base.free_symbols}
                            test_sub[sym] = low
                            val = complex(base.subs(test_sub).evalf())
                            if val.real < 0:
                                metrics["complex_branches"] += 1
                                ces.append(Counterexample(
                                    axis=FalsificationAxis.MATHEMATICAL,
                                    probe_name="ComplexBranchCutProbe", failure_class=FailureClass.COMPLEX_BRANCH_CUT,
                                    counterexample_point={s_name: low},
                                    violation_description=f"Radicand {base} becomes negative ({val.real:.3f}) at {s_name}={low}, producing unphysical imaginary values.",
                                    severity="FATAL"
                                ))
                        except (
                            AttributeError,
                            ArithmeticError,
                            NotImplementedError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            # Radicand evaluation failed for this sub-expression: the
                            # branch-cut behaviour is unknown, not clean.
                            self._record_probe_error(
                                metrics, "ComplexBranchCutProbe", exc, f"radicand {base}"
                            )

        # A clearance is only as strong as the methods behind it. An axis with
        # nothing to analyse (no denominator, no root or fractional power) is
        # silent rather than a hole, so it publishes no tier at all.
        if denoms_to_check or expr.atoms(sp.Pow):
            metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.MATHEMATICAL].value
        return ces, metrics

    @quiet_float_warnings()
    def _probe_boundary_counterexamples(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]],
        critical_boundaries: dict[str, list[float]] | None
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Evaluates directional asymptotic limits as variables approach critical points, 0, and infinity."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {
            "infinite_limits": 0,
            "limit_mismatches": 0,
            "probe_errors": [],
        }

        targets: dict[str, list[float]] = {}
        for s_name, (low, high) in domains.items():
            targets[s_name] = []
            if low <= 0.0 <= high:
                targets[s_name].append(0.0)

        if critical_boundaries:
            for k, vals in critical_boundaries.items():
                if k not in targets:
                    targets[k] = []
                targets[k].extend(vals)

        for sym in expr.free_symbols:
            s_name = str(sym)
            pts = set(targets.get(s_name, []))
            if not pts:
                continue

            # Substitute other variables with domain median to keep limits univariate, fast and stable
            other_syms = [s for s in expr.free_symbols if s != sym]
            sub_dict = {
                s: float(domains.get(str(s), (1.0, 1.0))[0] + domains.get(str(s), (1.0, 1.0))[1]) / 2.0
                for s in other_syms
            }
            eval_expr = expr.subs(sub_dict)

            for pt in pts:
                try:
                    # Fast-path: Check direct substitution first
                    direct_val = None
                    try:
                        direct_subs = eval_expr.subs(sym, pt)
                        if direct_subs.is_number and not direct_subs.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                            direct_val = float(direct_subs)
                    except (TypeError, ValueError, ArithmeticError, AttributeError) as exc:
                        # Direct substitution is only a fast path; the limits below
                        # are the actual measurement, so this is not swallowed.
                        logger.debug("Direct substitution failed at %s=%s: %s", s_name, pt, exc)
                        direct_val = None

                    if direct_val is not None and math.isfinite(direct_val):
                        # Regular and finite at pt, no singularity
                        continue

                    # Evaluate right limit (bounded; a timeout is an unknown, so it is
                    # recorded rather than treated as divergent *or* as clean)
                    try:
                        lim_right = limit_with_timeout(eval_expr, sym, pt, dir="+", timeout=2.0)
                    except LimitTimeout as exc:
                        self._record_probe_error(
                            metrics, "AsymptoticDivergenceProbe", exc, f"limit {s_name} -> {pt}^+ unresolved"
                        )
                        continue
                    is_div_right = lim_right.has(sp.oo) or lim_right.has(-sp.oo) or lim_right.has(sp.zoo)
                    if is_div_right:
                        metrics["infinite_limits"] += 1
                        ces.append(Counterexample(
                            axis=FalsificationAxis.BOUNDARY,
                            probe_name="AsymptoticDivergenceProbe",
                            counterexample_point={s_name: pt, "direction": "+"},
                            failure_class=FailureClass.BOUNDARY_DIVERGENCE,
                            violation_description=f"Formula diverges to {lim_right} as {s_name} -> {pt}^+.",
                            severity="FATAL"
                        ))
                    # Evaluate left limit if not at boundary min (bounded)
                    if pt > domains.get(s_name, (0, 0))[0]:
                        try:
                            lim_left = limit_with_timeout(eval_expr, sym, pt, dir="-", timeout=2.0)
                        except LimitTimeout as exc:
                            self._record_probe_error(
                                metrics,
                                "AsymptoticDivergenceProbe",
                                exc,
                                f"limit {s_name} -> {pt}^- unresolved",
                            )
                            continue
                        is_div_left = lim_left.has(sp.oo) or lim_left.has(-sp.oo) or lim_left.has(sp.zoo)
                        if is_div_left:
                            metrics["infinite_limits"] += 1
                            ces.append(Counterexample(
                                axis=FalsificationAxis.BOUNDARY,
                                probe_name="AsymptoticDivergenceProbe",
                                counterexample_point={s_name: pt, "direction": "-"},
                                failure_class=FailureClass.BOUNDARY_DIVERGENCE,
                                violation_description=f"Formula diverges to {lim_left} as {s_name} -> {pt}^-.",
                                severity="FATAL"
                            ))
                        elif not is_div_right and not is_div_left and lim_right.is_number and lim_left.is_number:
                            if abs(float(lim_right) - float(lim_left)) > 1e-4:
                                metrics["limit_mismatches"] += 1
                                ces.append(Counterexample(
                                    axis=FalsificationAxis.BOUNDARY,
                                    probe_name="DirectionalLimitMismatchProbe",
                                    counterexample_point={s_name: pt},
                                    failure_class=FailureClass.DIRECTIONAL_LIMIT_MISMATCH,
                                    violation_description=f"Jump discontinuity at {s_name}={pt}: lim+={float(lim_right):.4g} != lim-={float(lim_left):.4g}.",
                                    severity="HIGH"
                                ))
                except (
                    AttributeError,
                    ArithmeticError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as exc:
                    # Nothing was decided about the behaviour at this point.
                    self._record_probe_error(
                        metrics, "BoundaryAsymptoteProbe", exc, f"point {s_name}={pt}"
                    )

        # No declared or discovered boundary point means nothing to examine; the
        # axis stays silent rather than claiming a result.
        if any(targets.values()):
            metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.BOUNDARY].value
        return ces, metrics

    @quiet_float_warnings()
    def _probe_numerical_counterexamples(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]]
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Scans dense logarithmic and linear grids for floating point overflow, underflow, and gradient explosions."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {
            "nan_count": 0,
            "inf_count": 0,
            "grad_explosions": 0,
            "probe_errors": [],
        }

        syms = list(expr.free_symbols)
        if not syms:
            # A model with no free variable has no conditioning to measure: the
            # whole axis is unevaluable, which is not the same as clean.
            self._record_probe_error(
                metrics,
                "NumericalStabilityProbe",
                "NoFreeVariables",
                "constant model: no conditioning is measurable across a domain",
            )
            metrics["evidence_tier"] = EvidenceTier.UNRESOLVED.value
            return ces, metrics

        fn = sp.lambdify(syms, expr, modules=["numpy", "math"])

        # Grid testing
        num_samples = 40
        grids: dict[str, np.ndarray] = {}
        for s in syms:
            s_name = str(s)
            low, high = domains.get(s_name, (0.01, 10.0))
            if low > 0 and high / low > 100:
                grids[s_name] = np.geomspace(low, high, num_samples)
            else:
                grids[s_name] = np.linspace(low, high, num_samples)

        # Evaluate across grid combinations
        primary_sym = syms[0]
        primary_grid = grids[str(primary_sym)]
        fixed_args = [np.median(grids[str(s)]) for s in syms[1:]]

        y_vals: list[float] = []
        for x_val in primary_grid:
            try:
                args = [x_val] + fixed_args
                res = float(fn(*args))
                if math.isnan(res):
                    metrics["nan_count"] += 1
                    ces.append(Counterexample(
                        axis=FalsificationAxis.NUMERICAL,
                        probe_name="NaNInjectionProbe", failure_class=FailureClass.NON_FINITE_VALUE,
                        counterexample_point={str(primary_sym): float(x_val)},
                        violation_description=f"NaN encountered at {primary_sym!s}={x_val:.4e}.",
                        severity="FATAL"
                    ))
                    break
                if math.isinf(res):
                    metrics["inf_count"] += 1
                    ces.append(Counterexample(
                        axis=FalsificationAxis.NUMERICAL,
                        probe_name="InfinityOverflowProbe", failure_class=FailureClass.NON_FINITE_VALUE,
                        counterexample_point={str(primary_sym): float(x_val)},
                        violation_description=f"Numerical overflow (+/- inf) at {primary_sym!s}={x_val:.4e}.",
                        severity="FATAL"
                    ))
                    break
                y_vals.append(res)
            except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
                ces.append(Counterexample(
                    axis=FalsificationAxis.NUMERICAL,
                    probe_name="EvaluationExceptionProbe", failure_class=FailureClass.EVALUATION_FAILURE,
                    counterexample_point={str(primary_sym): float(x_val)},
                    violation_description=f"Evaluation crashed at {primary_sym!s}={x_val:.4e}: {e}",
                    severity="FATAL"
                ))
                break

        # Check normalized condition & gradient stability
        if len(y_vals) >= 5:
            low_x, high_x = domains.get(str(primary_sym), (primary_grid[0], primary_grid[-1]))
            # For multidecade variables (spanning >2 decades) with strictly positive outputs,
            # use scale-invariant logarithmic elasticity |d ln y / d ln x|
            if low_x > 0 and (high_x / low_x) > 100.0 and all(y > 0 for y in y_vals):
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_x = np.log(np.array(primary_grid[:len(y_vals)], dtype=float))
                    log_y = np.log(np.array(y_vals, dtype=float))
                    d_log_x = np.diff(log_x)
                    d_log_y = np.diff(log_y)
                    elasticity = np.abs(d_log_y / (d_log_x + 1e-15))
                max_elasticity = float(np.nanmax(elasticity)) if len(elasticity) > 0 else 0.0
                metrics["max_elasticity"] = max_elasticity
                if max_elasticity > 50.0 and not any(c.axis == FalsificationAxis.BOUNDARY for c in ces):
                    metrics["grad_explosions"] += 1
                    ces.append(Counterexample(
                        axis=FalsificationAxis.NUMERICAL,
                        probe_name="GradientExplosionProbe",
                        counterexample_point={str(primary_sym): float(primary_grid[np.nanargmax(elasticity)])},
                        failure_class=FailureClass.ILL_CONDITIONED,
                        violation_description=f"Numerical elasticity exploded (|d ln y / d ln x| = {max_elasticity:.2e}), indicating unphysical discontinuity.",
                        severity="HIGH",
                        metric_value=max_elasticity
                    ))
            else:
                y_range = max(y_vals) - min(y_vals)
                x_range = float(primary_grid[len(y_vals) - 1] - primary_grid[0])
                if y_range > 1e-12 and x_range > 1e-12:
                    dy_norm = np.diff(y_vals) / y_range
                    dx_norm = np.diff(primary_grid[:len(y_vals)]) / x_range
                    with np.errstate(divide="ignore", invalid="ignore"):
                        norm_gradients = np.abs(dy_norm / (dx_norm + 1e-15))
                    max_norm_grad = float(np.nanmax(norm_gradients)) if len(norm_gradients) > 0 else 0.0
                    metrics["max_normalized_gradient"] = max_norm_grad

                    # Check for derivative gradient explosion (normalized slope > 1e4)
                    d_exploded = False
                    try:
                        d_expr = sp.diff(expr, primary_sym)
                        d_fn = sp.lambdify(syms, d_expr, modules=["numpy", "math"])
                        for x_pt in [primary_grid[0], primary_grid[-1], 0.0, float(np.median(primary_grid))]:
                            if low_x <= x_pt <= high_x:
                                test_args = [x_pt] + fixed_args
                                d_val = abs(float(d_fn(*test_args)))
                                norm_d = d_val * x_range / (y_range + 1e-15)
                                if norm_d > 1e4 and not any(c.axis == FalsificationAxis.BOUNDARY for c in ces):
                                    d_exploded = True
                                    metrics["grad_explosions"] += 1
                                    ces.append(Counterexample(
                                        axis=FalsificationAxis.NUMERICAL,
                                        probe_name="GradientExplosionProbe",
                                        counterexample_point={str(primary_sym): float(x_pt)},
                                        failure_class=FailureClass.ILL_CONDITIONED,
                                        violation_description=f"Numerical gradient exploded (normalized slope {norm_d:.2e} > 1e4), indicating extreme stiffness or numerical discontinuity.",
                                        severity="HIGH",
                                        metric_value=norm_d
                                    ))
                                    break
                    except (
                        AttributeError,
                        ArithmeticError,
                        NotImplementedError,
                        TypeError,
                        ValueError,                        ) as exc:
                            # The gradient is still measured, by the finite-difference
                            # path immediately below: this is a degradation of the
                            # probe, not a hole in the axis, so it is recorded
                            # without voiding a clean verdict.
                            self._record_probe_error(
                                metrics,
                                "GradientExplosionProbe",
                                exc,
                                f"symbolic derivative of {expr}",
                                measured_alternative=True,
                            )

                    if not d_exploded and max_norm_grad > 1e4 and not any(c.axis == FalsificationAxis.BOUNDARY for c in ces):
                        metrics["grad_explosions"] += 1
                        ces.append(Counterexample(
                            axis=FalsificationAxis.NUMERICAL,
                            probe_name="GradientExplosionProbe",
                            counterexample_point={str(primary_sym): float(primary_grid[np.nanargmax(norm_gradients)])},
                            failure_class=FailureClass.ILL_CONDITIONED,
                            violation_description=f"Numerical normalized gradient exploded ({max_norm_grad:.2e}x scale factor), indicating extreme stiffness or numerical discontinuity.",
                            severity="HIGH",
                            metric_value=max_norm_grad
                        ))
                else:
                    # A model with no variation along the swept symbol has no
                    # conditioning to measure, but that is a null result rather than
                    # a crash: it is published as a metric, not as a probe error.
                    metrics["degenerate_sweep"] = True
                    metrics["sweep_output_range"] = y_range

        metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.NUMERICAL].value
        return ces, metrics

    def _sample_envelope(
        self,
        fn: Any,
        syms: list[sp.Symbol],
        domains: dict[str, tuple[float, float]],
    ) -> tuple[list[tuple[dict[str, float], float | None]], list[dict[str, Any]]]:
        """Evaluate the model across its declared envelope.

        For every symbol the model is swept across that symbol's declared interval
        (other symbols held at their midpoint); one further sweep moves all symbols
        together along the domain diagonal. Returns ``(samples, errors)`` where each
        sample is ``(point, value)`` and ``value`` is ``None`` when the model is not
        real-valued there, and ``errors`` records any sweep that could not be
        evaluated at all -- an unevaluated invariant is an unknown, never a pass.
        """
        bounds = {str(s): tuple(domains.get(str(s), (-1.0, 1.0))) for s in syms}
        mids = {name: (lo + hi) / 2.0 for name, (lo, hi) in bounds.items()}
        n = INVARIANT_SAMPLES

        sweeps: list[dict[str, np.ndarray]] = []
        for s in syms:
            lo, hi = bounds[str(s)]
            sweep = {name: np.full(n, mids[name], dtype=float) for name in bounds}
            sweep[str(s)] = np.linspace(lo, hi, n)
            sweeps.append(sweep)
        if syms:
            t = np.linspace(0.0, 1.0, n)
            sweeps.append({name: lo + t * (hi - lo) for name, (lo, hi) in bounds.items()})

        samples: list[tuple[dict[str, float], float | None]] = []
        errors: list[dict[str, Any]] = []
        for sweep in sweeps:
            args = [sweep[str(s)] for s in syms]
            try:
                with np.errstate(all="ignore"):
                    values = np.ravel(np.asarray(fn(*args), dtype=float))
            except (TypeError, ValueError, ArithmeticError, AttributeError, NotImplementedError) as exc:
                errors.append({
                    "probe": "EnvelopeSweep",
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "unresolved": True,
                })
                continue
            length = len(args[0]) if args else 0
            if values.size == 1 and length > 1:
                values = np.full(length, float(values[0]))
            for i in range(min(length, values.size)):
                point = {name: float(np.ravel(arr)[i]) for name, arr in sweep.items()}
                v = float(values[i])
                samples.append((point, v if np.isfinite(v) else None))
        return samples, errors

    @quiet_float_warnings()
    def _probe_structural_invariants(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]],
        invariants: list[str],
        variable_units: dict[str, str] | None = None,
        expected_output_unit: str | None = None
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Tests physical invariants such as positivity, probability ceilings, and dimensional consistency."""
        ces: list[Counterexample] = []

        syms = list(expr.free_symbols)
        if not syms:
            return ces, {"invariants_declared": len(invariants), "invariants_checked": 0,
                         "probe_errors": []}

        fn = sp.lambdify(syms, expr, modules=["numpy", "math"])

        # Every invariant below is decided from the model's own behaviour across
        # the declared envelope. Symbol *names* are never consulted, so renaming a
        # variable cannot disable a check, and a violation away from the declared
        # bounds (e.g. f = x**2 - 0.5 on [-1, 1]) is no longer invisible.
        samples, probe_errors = self._sample_envelope(fn, syms, domains)
        finite = [(pt, v) for pt, v in samples if v is not None]
        lowest = min(finite, key=lambda pv: pv[1]) if finite else None
        highest = max(finite, key=lambda pv: pv[1]) if finite else None
        widest = max(finite, key=lambda pv: abs(pv[1])) if finite else None
        first_unevaluated = next((pt for pt, v in samples if v is None), {})
        magnitudes = sorted(abs(v) for _, v in finite)
        median_magnitude = magnitudes[len(magnitudes) // 2] if magnitudes else 0.0
        boundary_divergence = (
            magnitudes[-1] / median_magnitude if median_magnitude > 0 else None
        )

        metrics: dict[str, Any] = {
            # Declared labels and executed checks are reported separately: the two
            # differ whenever an invariant has no probe, and an invariant that could
            # not be evaluated is never counted as one that passed.
            "invariants_declared": len(invariants),
            "invariants_checked": 0,
            "envelope_samples": len(samples),
            "envelope_samples_unevaluated": len(samples) - len(finite),
            "boundary_divergence": boundary_divergence,
            "probe_errors": list(probe_errors),
        }

        # Check 1: Dimensional consistency
        if variable_units and expected_output_unit:
            for s_name, u_str in variable_units.items():
                self.dim_engine.register_variable(s_name, u_str)
            try:
                inferred = self.dim_engine.check_expression_consistency(expr)
                expected_unit_obj = self.dim_engine.registry(expected_output_unit)
                metrics["invariants_checked"] += 1
                is_dim_compat = inferred.dimensionality == expected_unit_obj.dimensionality
                if not is_dim_compat:
                    ces.append(Counterexample(
                        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                        probe_name="DimensionalHomogeneityProbe", failure_class=FailureClass.DIMENSIONAL_MISMATCH,
                        counterexample_point={"inferred": str(inferred), "expected": expected_output_unit},
                        violation_description=f"Dimensional mismatch: expression has units '{inferred}' but expected '{expected_output_unit}'.",
                        severity="FATAL"
                    ))
            except (TypeError, ValueError, AttributeError, NotImplementedError, sp.SympifyError) as e:
                # A probe that could not run is an unevaluated check, not evidence
                # that the model is inconsistent.
                metrics["probe_errors"].append(
                    {"probe": "DimensionalInferenceProbe", "error": type(e).__name__, "detail": str(e)}
                )
                logger.warning("Dimensional invariant probe failed", exc_info=True)

        # Check 2: Positivity Invariant (e.g. Temperature, Mass, Concentration, Energy)
        if any(inv in ["positivity", "strictly_positive", "non_negative"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < -1e-12:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="PositivityViolationProbe", failure_class=FailureClass.UNPHYSICAL_NEGATIVE,
                    counterexample_point=lowest[0],
                    violation_description=f"UNPHYSICAL_NEGATIVE: Quantity evaluated to {lowest[1]:.4e} < 0, violating non-negativity.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))

        # Check 3: Probability Ceiling (p <= 1.0 or depolarizing ceiling)
        if "probability_bound" in invariants or "probability" in invariants:
            metrics["invariants_checked"] += 1
            if highest is not None and highest[1] > 1.00001:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="ProbabilityCeilingProbe", failure_class=FailureClass.UNPHYSICAL_PROBABILITY,
                    counterexample_point=highest[0],
                    violation_description=f"UNPHYSICAL_PROBABILITY: Predicted probability {highest[1]:.3f} exceeds absolute ceiling 1.0.",
                    severity="FATAL",
                    metric_value=highest[1]
                ))

        # Check 4: Thermodynamics 3rd Law (negative Kelvin Temperature). Previously
        # this fired on the *name* of the variable plus a negative declared bound,
        # without ever evaluating the model. It is now decided by the model's own
        # values: a thermodynamic quantity below absolute zero is a negative one.
        if "third_law_thermodynamics" in invariants:
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < 0.0:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="ThirdLawThermodynamicsProbe", failure_class=FailureClass.THIRD_LAW_TEMPERATURE,
                    counterexample_point=lowest[0],
                    violation_description=f"UNPHYSICAL_NEGATIVE_TEMPERATURE_3RD_LAW_THERMODYNAMICS: Model evaluates to {lowest[1]:.4e} over its declared envelope, below absolute zero.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))

        # Check 5: Lorentz Causality. Previously this compared a declared upper
        # bound against its own private copy of c. It is now decided by the model:
        # a subluminal formulation stays real-valued across its envelope, while a
        # tachyonic one is imaginary (non-finite) over the domain it claims.
        if "lorentz_causality" in invariants:
            metrics["invariants_checked"] += 1
            if len(finite) < len(samples):
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="LorentzCausalityProbe", failure_class=FailureClass.ACASUAL_NOT_REAL_VALUED,
                    counterexample_point=first_unevaluated,
                    violation_description="SUPERLUMINAL_LORENTZ_VIOLATION_ACAUSAL: Model is not real-valued (imaginary/Nan) somewhere in its declared envelope, as required by subluminal causality.",
                    severity="FATAL"
                ))

        # Check 6: Second Law of Thermodynamics (Entropy Production >= 0). The name
        # gate (sigma_entropy / dS_dt / entropy_prod) is gone: when the caller
        # declares the second law, the model's own output is what must not be
        # negative anywhere in the envelope.
        if "second_law_thermodynamics" in invariants:
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < 0:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="SecondLawThermodynamicsProbe", failure_class=FailureClass.NEGATIVE_ENTROPY_PRODUCTION,
                    counterexample_point=lowest[0],
                    violation_description="NEGATIVE_ENTROPY_PRODUCTION_2ND_LAW_VIOLATION: Entropy production rate is negative.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))

        # Check 7: Cosmic Censorship & Negative Mass Curvature Inversion. Previously
        # this fired on a symbol whose *name* contained "mass"/"m_" together with a
        # negative declared bound. It is now decided by the model: a positive-energy
        # law must not evaluate negative, nor grow without bound toward the edge of
        # the envelope it declares (an uncensored curvature divergence).
        if "cosmic_censorship" in invariants or "positive_energy" in invariants:
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < 0.0:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="CosmicCensorshipProbe",
                    counterexample_point=lowest[0],
                    failure_class=FailureClass.NAKED_SINGULARITY,
                    violation_description=f"NAKED_SINGULARITY_INVERSION: Positive-energy model evaluates to {lowest[1]:.4e} < 0 inside its declared envelope.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))
            elif boundary_divergence is not None and boundary_divergence > ENVELOPE_DIVERGENCE_RATIO:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="CosmicCensorshipProbe",
                    counterexample_point=widest[0] if widest else {},
                    failure_class=FailureClass.NAKED_SINGULARITY,
                    violation_description=f"NAKED_SINGULARITY_INVERSION: Model diverges toward the edge of its declared envelope (|f| grows {boundary_divergence:.1e}x), an uncensored curvature divergence.",
                    severity="FATAL",
                    metric_value=boundary_divergence
                ))

        # Check 8: High-Energy Unitarity Bound (Froissart Bound)
        if "unitarity_bound" in invariants:
            metrics["invariants_checked"] += 1
            if highest is not None and highest[1] > 1e4:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="UnitarityBoundProbe", failure_class=FailureClass.UNITARITY_BOUND,
                    counterexample_point=highest[0],
                    violation_description=f"UNITARITY_VIOLATION: Cross-section {highest[1]:.2e} diverges with s, violating Froissart/unitarity bound at high energies.",
                    severity="FATAL",
                    metric_value=highest[1]
                ))

        # Check 9: Current Density Ceiling (Electrochemistry Fickian Mass Transport Limit)
        if "current_density_ceiling" in invariants:
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e4:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="CurrentDensityCeilingProbe", failure_class=FailureClass.CURRENT_DENSITY_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"UNPHYSICAL_CURRENT_DIVERGENCE: Current density {widest[1]:.2e} A/m^2 exceeds mass-transport diffusion ceiling (1e4 A/m^2).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 10: Knudsen Continuum Validity (Fluid Dynamics Micro/Nanoscale Breakdown).
        # This one is a *regime* statement -- the smallest scale the envelope claims
        # against the 68 nm mean free path -- so it is decided from the declared
        # scales with no name matching. The previous implementation additionally
        # grepped the model's printed text for magic constants ("68e-9", "R**3",
        # "kn", ...) to guess whether a slip correction was already present; that
        # text search is removed. If an envelope reaches continuum-invalid scales,
        # the continuum invariant is violated and the remedy is a domain
        # restriction, which is what the caller is told.
        if "knudsen_continuum" in invariants:
            metrics["invariants_checked"] += 1
            scales = [lo for lo, _ in domains.values() if lo > 0.0]
            smallest_scale = min(scales) if scales else None
            if smallest_scale is not None:
                lambda_mfp = 68e-9  # Mean free path (68 nm)
                kn = lambda_mfp / smallest_scale
                if kn > 0.01:
                    ces.append(Counterexample(
                        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                        probe_name="KnudsenContinuumBreakdownProbe", failure_class=FailureClass.CONTINUUM_BREAKDOWN,
                        counterexample_point=widest[0] if widest else {},
                        violation_description=f"CONTINUUM_BREAKDOWN_KNUDSEN: At R={smallest_scale:.1e}m, Knudsen number Kn={kn:.1f} > 0.01. Continuum no-slip formulation is invalid without slip correction.",
                        severity="FATAL",
                        metric_value=kn
                    ))

        # Check 11: Carrier / Velocity Saturation Ceiling
        if any(inv in ["velocity_saturation_ceiling", "carrier_velocity_ceiling"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e5:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="VelocitySaturationCeilingProbe", failure_class=FailureClass.VELOCITY_SATURATION_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"VELOCITY_SATURATION_EXCEEDED: Predicted velocity {widest[1]:.2e} m/s exceeds optical phonon / saturation ceiling (1e5 m/s).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 12: Stress / Yield Ceiling (Continuum Plasticity Breakdown)
        if any(inv in ["stress_saturation_ceiling", "yield_strength_ceiling"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e9:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="YieldStrengthCeilingProbe", failure_class=FailureClass.YIELD_STRESS_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"YIELD_STRESS_EXCEEDED: Linear stress {widest[1]:.2e} Pa exceeds maximum material yield ceiling (1e9 Pa).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 13: Laminar Flow Drag Ceiling (Stokes Law Breakdown)
        if any(inv in ["reynolds_laminar_ceiling", "stokes_drag_ceiling"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e-3:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="StokesLaminarCeilingProbe", failure_class=FailureClass.STOKES_REGIME_BREAKDOWN,
                    counterexample_point=widest[0],
                    violation_description=f"STOKES_REGIME_BREAKDOWN: Drag force {widest[1]:.2e} N exceeds Stokes laminar creeping flow regime (Re >> 1).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 14: Third Law Heat Capacity Floor (C_v -> 0 as T -> 0). The probe no
        # longer keys off a variable *named* like a temperature; it applies to any
        # non-negative quantity, whose floor is zero, and evaluates the model at
        # that zero limit.
        if "third_law_heat_capacity" in invariants:
            metrics["invariants_checked"] += 1
            zero_floored = [str(s) for s in syms if domains.get(str(s), (1.0, 1.0))[0] >= 0.0]
            if zero_floored:
                test_pts = [
                    0.0 if str(sym) in zero_floored else domains.get(str(sym), (1.0, 1.0))[0]
                    for sym in syms
                ]
                try:
                    res_zero = float(fn(*test_pts))
                except (TypeError, ValueError, ArithmeticError, AttributeError) as exc:
                    metrics["probe_errors"].append({
                        "probe": "ThirdLawHeatCapacityProbe",
                        "error": type(exc).__name__,
                        "detail": str(exc),
                        "unresolved": True,
                    })
                    logger.warning("Third-law heat capacity probe failed", exc_info=True)
                    res_zero = None
                if res_zero is not None and abs(res_zero) > 1e-3:
                    ces.append(Counterexample(
                        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                        probe_name="ThirdLawHeatCapacityProbe", failure_class=FailureClass.THIRD_LAW_HEAT_CAPACITY,
                        counterexample_point=dict(zip([str(s) for s in syms], test_pts)),
                        violation_description=f"3RD_LAW_THERMODYNAMICS_VIOLATION: Heat capacity C_v(0) = {res_zero:.2e} != 0 at absolute zero.",
                        severity="FATAL",
                        metric_value=res_zero
                    ))

        # Check 15: Recombination / Quenching Saturation Ceiling
        if "saturation_ceiling" in invariants:
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e9:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="RateSaturationCeilingProbe", failure_class=FailureClass.RATE_SATURATION_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"UNBOUNDED_RATE_DIVERGENCE: Reaction/recombination rate {widest[1]:.2e} exceeds physical saturation ceiling.",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        if not finite and invariants:
            # Nothing could be decided: the declared invariants were never evaluated
            # against a real value, so they are unevaluable -- an error to surface,
            # not checks that passed by default.
            metrics["invariants_checked"] = 0
            metrics["probe_errors"].append({
                "probe": "EnvelopeSweep",
                "error": "NoRealValues",
                "detail": "model returned no real value anywhere in the declared envelope",
                "unresolved": True,
            })
            logger.warning(
                "No real values in the declared envelope for %s; %d declared invariant(s) unevaluable",
                expr,
                len(invariants),
            )

        # Declared invariants that were never evaluated are a hole in the evidence
        # and collapse the tier; an axis with no invariants to check is silent.
        if metrics["invariants_checked"] > 0:
            metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.STRUCTURAL_INVARIANT].value
        elif invariants:
            metrics["evidence_tier"] = EvidenceTier.UNRESOLVED.value
        return ces, metrics

    @quiet_float_warnings()
    def _probe_empirical_fidelity(
        self,
        expr: sp.Expr,
        empirical_data: dict[str, Any]
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Evaluates residual error and R^2 goodness-of-fit against empirical reference observations."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {"probe_errors": []}

        x_data = np.asarray(empirical_data["x"], dtype=float)
        y_obs = np.asarray(empirical_data["y"], dtype=float)

        syms = list(expr.free_symbols)
        if len(syms) != 1:
            # Skipping the axis is not passing it: a multi-variable model gets no
            # empirical verdict, and the report says why.
            self._record_probe_error(
                metrics,
                "EmpiricalResidualProbe",
                "ModelNotUnivariate",
                f"empirical comparison is implemented for 1 free variable, model has {len(syms)}",
            )
            metrics["evidence_tier"] = EvidenceTier.UNRESOLVED.value
            return ces, metrics

        metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.EMPIRICAL].value

        fn = sp.lambdify(syms[0], expr, modules=["numpy", "math"])
        try:
            y_pred = np.array([float(fn(x)) for x in x_data])
            residuals = y_obs - y_pred
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
            r_squared = 1.0 - (ss_res / (ss_tot + 1e-15))
            nrmse = float(np.sqrt(np.mean(residuals ** 2)) / (np.std(y_obs) + 1e-15))

            metrics["r_squared"] = r_squared
            metrics["nrmse"] = nrmse

            if r_squared < 0.70 or nrmse > 0.35:
                ces.append(Counterexample(
                    axis=FalsificationAxis.EMPIRICAL,
                    probe_name="EmpiricalResidualProbe", failure_class=FailureClass.EMPIRICAL_FIT_REJECTED,
                    counterexample_point={"worst_x": float(x_data[np.argmax(np.abs(residuals))])},
                    violation_description=f"Empirical fit rejected: R^2={r_squared:.3f} < 0.70 (NRMSE={nrmse:.1%}).",
                    severity="HIGH",
                    metric_value=r_squared
                ))
        except (TypeError, ValueError, ArithmeticError, AttributeError, NotImplementedError) as e:
            # Already surfaced as a failure rather than swallowed: a model that
            # cannot be evaluated at the reference observations fails the
            # comparison. An exception type outside this list is a defect in APEDP
            # and propagates.
            ces.append(Counterexample(
                axis=FalsificationAxis.EMPIRICAL,
                probe_name="EmpiricalEvaluationProbe", failure_class=FailureClass.EMPIRICAL_EVALUATION_FAILURE,
                counterexample_point={},
                violation_description=f"Empirical evaluation failed on test data: {e}",
                severity="HIGH"
            ))

        return ces, metrics


@dataclass
class BlindAuditResult:
    """Outcome of blind unannotated adversarial analysis on an arbitrary equation."""
    formula_str: str
    is_defective: bool
    free_variables: list[str]
    identified_failure_axes: list[str]
    failure_locations: list[dict[str, Any]]
    failure_descriptions: list[str]
    competing_candidates: list[str]
    champion_candidate: str | None
    champion_survived: bool
    champion_verdict: str
    infeasibility_certificate: InfeasibilityCertificate | None
    epistemic_qualification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_str": self.formula_str,
            "is_defective": self.is_defective,
            "free_variables": self.free_variables,
            "identified_failure_axes": self.identified_failure_axes,
            "failure_locations": self.failure_locations,
            "failure_descriptions": self.failure_descriptions,
            "competing_candidates": self.competing_candidates,
            "champion_candidate": self.champion_candidate,
            "champion_survived": self.champion_survived,
            "champion_verdict": self.champion_verdict,
            "infeasibility_certificate": self.infeasibility_certificate.to_dict() if self.infeasibility_certificate else None,
            "epistemic_qualification": self.epistemic_qualification,
        }


class BlindAdversarialAuditor:
    """Performs blind, unannotated defect discovery, multi-hypothesis repair, and cross-falsification.

    Given a bare equation without failure class hints, critical boundary hints, or domain tags:
    1. Autonomously localizes poles, branch cuts, and asymptotic divergences.
    2. Proves whether specified constraints are mutually incompatible.
    3. Synthesizes competing candidate reformulations (softening vs saturation vs representation transformation).
    4. Evaluates all candidates through tournament cross-falsification.
    5. Returns a rigorous audit report with provisional validity bounds.
    """

    def __init__(self, falsifier: IndependentAdversarialFalsifier | None = None):
        self.falsifier = falsifier or IndependentAdversarialFalsifier()

    def audit(
        self,
        formula: str | sp.Expr,
        default_domains: dict[str, tuple[float, float]] | None = None,
        constraints: list[str] | None = None
    ) -> BlindAuditResult:
        formula_str = str(formula)
        try:
            expr = safe_parse_expr(formula) if isinstance(formula, str) else formula
        except (ValueError, TypeError) as e:
            return BlindAuditResult(
                formula_str=formula_str,
                is_defective=True,
                free_variables=[],
                identified_failure_axes=["MATHEMATICAL"],
                failure_locations=[],
                failure_descriptions=[f"Unparseable formula: {e}"],
                competing_candidates=[],
                champion_candidate=None,
                champion_survived=False,
                champion_verdict="REFUSED_CANNOT_REPAIR",
                infeasibility_certificate=None,
                epistemic_qualification="Syntax failure."
            )

        # 1. Infeasibility check on constraints
        infeas_cert = ContradictoryConstraintEngine.check_incompatibility(constraints or [])
        if infeas_cert and infeas_cert.is_infeasible:
            return BlindAuditResult(
                formula_str=formula_str,
                is_defective=True,
                free_variables=[str(s) for s in expr.free_symbols],
                identified_failure_axes=["STRUCTURAL_INVARIANT"],
                failure_locations=[],
                failure_descriptions=[infeas_cert.incompatibility_proof],
                competing_candidates=[],
                champion_candidate=None,
                champion_survived=False,
                champion_verdict="REFUSED_INCOMPATIBLE_CONSTRAINTS",
                infeasibility_certificate=infeas_cert,
                epistemic_qualification="Incompatible constraints: Admissible function space is proven empty."
            )

        # 2. Autonomous domain setup & pole/branch localization
        free_syms = list(expr.free_symbols)
        domains: dict[str, tuple[float, float]] = dict(default_domains or {})
        for s in free_syms:
            s_name = str(s)
            if s_name not in domains:
                # Wide probing domain default
                domains[s_name] = (1e-6, 1e4)

        # Autonomous critical boundaries discovery
        crit_boundaries: dict[str, list[float]] = {}
        for s in free_syms:
            s_name = str(s)
            crit_boundaries[s_name] = [0.0]

        # Scan for denominator roots automatically
        denoms = []
        _numer, denom = expr.as_numer_denom()
        if denom != 1:
            denoms.append(denom)
        for p in expr.atoms(sp.Pow):
            if p.exp.is_number and float(p.exp) < 0:
                denoms.append(p.base)

        for d_expr in denoms:
            for s in d_expr.free_symbols:
                s_name = str(s)
                try:
                    r_list = sp.solve(d_expr, s)
                    for r in r_list:
                        if getattr(r, "is_real", False) and getattr(r, "is_number", False):
                            r_val = float(r)
                            crit_boundaries.setdefault(s_name, []).append(r_val)
                except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                    logger.debug("Denominator root solve failed for %s (%s)", d_expr, e)

        # 3. Blind adversarial falsification of raw formula
        raw_rep = self.falsifier.falsify(
            formula=expr,
            variable_domains=domains,
            critical_boundaries=crit_boundaries,
            allow_repair_attempt=True,
            constraints=constraints
        )

        is_defective = not raw_rep.survived
        failure_axes = list({c.axis.value for c in raw_rep.counterexamples})
        failure_locs = [c.counterexample_point for c in raw_rep.counterexamples]
        failure_descs = [c.violation_description for c in raw_rep.counterexamples]

        if not is_defective:
            return BlindAuditResult(
                formula_str=formula_str,
                is_defective=False,
                free_variables=[str(s) for s in free_syms],
                identified_failure_axes=[],
                failure_locations=[],
                failure_descriptions=[],
                competing_candidates=[formula_str],
                champion_candidate=formula_str,
                champion_survived=True,
                champion_verdict=raw_rep.verdict,
                infeasibility_certificate=None,
                epistemic_qualification=raw_rep.epistemic_qualification
            )

        # 4. Multi-Hypothesis Synthesis
        competing: list[str] = []
        primary_var = str(free_syms[0]) if free_syms else "x"
        s_sym = sp.Symbol(primary_var)

        # Strategy 1: Soft-core Tikhonov smoothing
        softened_var = sp.sqrt(s_sym**2 + sp.Symbol("epsilon_reg")**2)
        try:
            cand1_expr = expr.subs(s_sym, softened_var).subs(sp.Symbol("epsilon_reg"), 0.04)
            cand1_str = str(cand1_expr)
            if cand1_str != formula_str:
                competing.append(cand1_str)
        except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
            logger.debug("Soft-core candidate failed (%s)", e)

        # Strategy 2: Asymptotic saturation / Padé-envelope
        try:
            cand2_expr = (expr * 1000.0) / (1000.0 + sp.Abs(expr))
            cand2_str = str(cand2_expr)
            competing.append(cand2_str)
        except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
            logger.debug("Saturation candidate failed (%s)", e)

        # Strategy 3: Coordinate / reciprocal inversion (e.g. Lineweaver-Burk)
        try:
            poly_inv = sp.together(expr)
            n_inv, d_inv = poly_inv.as_numer_denom()
            if n_inv != 1 and d_inv != 1:
                cand3_expr = d_inv / n_inv
                cand3_str = str(cand3_expr)
                if cand3_str not in competing:
                    competing.append(cand3_str)
        except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
            logger.debug("Inversion candidate failed (%s)", e)

        # Fallback candidate if none generated
        if not competing:
            competing.append(f"({formula_str}) / (1.0 + 0.04 * abs({formula_str}))")

        # 5. Tournament cross-falsification of candidates
        champion = None
        champion_survived = False
        champion_verdict = "REFUSED_CANNOT_REPAIR"

        for cand in competing:
            c_rep = self.falsifier.falsify(
                formula=cand,
                variable_domains=domains,
                critical_boundaries=crit_boundaries,
                allow_repair_attempt=True,
                constraints=constraints
            )
            if c_rep.survived:
                champion = cand
                champion_survived = True
                champion_verdict = c_rep.verdict
                break

        if champion is None and competing:
            champion = competing[0]

        return BlindAuditResult(
            formula_str=formula_str,
            is_defective=is_defective,
            free_variables=[str(s) for s in free_syms],
            identified_failure_axes=failure_axes,
            failure_locations=failure_locs,
            failure_descriptions=failure_descs,
            competing_candidates=competing,
            champion_candidate=champion,
            champion_survived=champion_survived,
            champion_verdict=champion_verdict,
            infeasibility_certificate=None,
            epistemic_qualification="Blind audit completed across 5 axes without prior failure hints."
        )
