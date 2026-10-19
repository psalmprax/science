"""Trust-score weights, novelty/category enums and measured-metric helpers."""
from __future__ import annotations

from enum import Enum

# Weights of the system trust score. Refusal accuracy carries the most weight
# because falsely "repairing" an unrepairable law is the most damaging failure
# mode; false-repair rate enters as its complement. Weights sum to 1.0.
TRUST_WEIGHT_DEFECT_DETECTION = 0.25


TRUST_WEIGHT_SOLVABLE_REPAIR = 0.25


#: Weight on finding an unrepairable law at all. This replaced a weight on the
#: refusal *rate*, which the case label controlled rather than the engine.
TRUST_WEIGHT_PATHOLOGICAL_DETECTION = 0.35


TRUST_WEIGHT_NO_FALSE_REPAIR = 0.15


class UnmeasuredMetricError(RuntimeError):
    """Raised when a metric is required but the suite contains nothing to measure.

    Subgroup entries report ``None`` when the relevant cases are absent, and
    aggregates refuse to be computed from them. ``None`` is used rather than
    ``nan`` deliberately: ``nan >= 0.9`` returns ``False`` and ``nan * w`` returns
    ``nan``, so an unmeasured figure used to flow through comparisons and sums
    without complaint. Any arithmetic or ordering comparison against ``None``
    raises immediately instead.
    """


def _rate(numerator: int, denominator: int) -> float | None:
    """Rate of ``numerator`` over ``denominator``, or ``None`` if unmeasurable.

    ``None`` means "the suite contains no cases of this kind", which is a coverage
    fact rather than a failure: it must not be reported as a rate at all, and it
    must not be mistaken for 0.0 or 1.0 (both of which it used to resemble).
    """
    return float(numerator / denominator) if denominator > 0 else None


def _require(value: float | None, name: str) -> float:
    """Narrow a possibly-unmeasured metric, refusing to invent a value for it.

    Used where a number is genuinely needed. Substituting 0.0 or 1.0 here is what
    previously let a suite that measured nothing still report a conformance score.
    """
    if value is None:
        raise UnmeasuredMetricError(
            f"{name} is unmeasurable: the suite contains no cases of that kind"
        )
    return value


class NoveltyTier(str, Enum):
    """Three-tier stratification of benchmark problem novelty."""
    TIER_A_KNOWN_REPAIR = "TIER_A_KNOWN_REPAIR"
    TIER_B_HIDDEN_REPAIR = "TIER_B_HIDDEN_REPAIR"
    TIER_C_NOVEL_OPEN = "TIER_C_NOVEL_OPEN"


class BenchmarkCategory(str, Enum):
    """The 10 canonical pathology categories forming the 100-case suite."""
    KNOWN_MATHEMATICAL_SINGULARITIES = "KNOWN_MATHEMATICAL_SINGULARITIES"
    PHYSICAL_DIVERGENCES = "PHYSICAL_DIVERGENCES"
    DOMAIN_OF_VALIDITY_FAILURES = "DOMAIN_OF_VALIDITY_FAILURES"
    CONSERVATION_INVARIANT_VIOLATIONS = "CONSERVATION_INVARIANT_VIOLATIONS"
    BOUNDARY_CONDITION_FAILURES = "BOUNDARY_CONDITION_FAILURES"
    NUMERICAL_PATHOLOGIES = "NUMERICAL_PATHOLOGIES"
    REPRESENTATION_MODEL_FORM_FAILURES = "REPRESENTATION_MODEL_FORM_FAILURES"
    DELIBERATELY_INCOMPATIBLE_CONSTRAINTS = "DELIBERATELY_INCOMPATIBLE_CONSTRAINTS"
    HIDDEN_KNOWN_REPAIRS = "HIDDEN_KNOWN_REPAIRS"
    NOVEL_OPEN_REPAIRS = "NOVEL_OPEN_REPAIRS"
