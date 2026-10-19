"""Axes, failure classes and the evidence tier each probe's finding rests on."""
from __future__ import annotations

from enum import Enum

from apedp.evidence import EvidenceTier

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
