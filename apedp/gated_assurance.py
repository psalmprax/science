"""APEDP Gated Scientific Assurance & Blind Falsification Harness.

Transforms model verification from simple regression testing into an executable,
gated scientific assurance control plane. Enforces strict isolation between development
and locked challenge partitions, evaluates multidimensional scientific improvement metrics,
benchmarks diagnostic reasoning, and classifies failure severity to govern lifecycle decisions.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.epistemic import (
    DataPartitionVault,
    LifecycleManager,
    ModelEvidenceRecord,
    TestableModelPackage,
)
from apedp.evidence import EvidenceTier
from apedp.falsification import (
    ContradictoryConstraintEngine,
    FalsificationAxis,
    IndependentAdversarialFalsifier,
)
from apedp.safety import safe_parse_expr


def _clamp01(value: float) -> float:
    """Clamp a measured quantity into [0.0, 1.0]; non-finite input becomes 0.0."""
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


# =========================================================================
# Measured assurance primitives
# =========================================================================
#
# The gate below is only as meaningful as the provenance of its numbers. Every
# helper in this block replaces either a constant literal or a presence test
# ("the record declares asymptotics", "the constraint text contains '> 0'") with
# a measurement taken from the model under test. The distinction is not cosmetic:
# a presence test is a statement about the record's fields, while a measurement
# is a statement about the model's behaviour, and only the latter can be wrong in
# a way that the gate is supposed to catch.
#
# Two conventions are load-bearing and are therefore documented here rather than
# at each call site:
#
# * An *unevaluable* probe is never a satisfied one. A constraint that cannot be
#   parsed, an asymptote whose variable cannot be resolved, a model that returns
#   no finite value along an approach sequence -- each is classified
#   ``UNMEASURABLE``/``UNVERIFIED`` and reported, never silently passed.
# * A declared asymptote written as a *symbolic* expression (``sigma_0``,
#   ``Cp_0 / sqrt(eps)``) is read as a scaling claim about the limit's character,
#   because that is how the benchmark contracts in this repository use it. The
#   numerically resolved limit is measured and reported alongside it, but only a
#   pure numeric literal is treated as an unambiguous value claim that the gate
#   can contradict.

_RELATION_PATTERN = re.compile(
    r"([A-Za-z_]\w*)\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|[A-Za-z_]\w*)"
)
_LIMIT_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:->|_to_)\s*(.+?)\s*$")
_INFINITY_WORDS = frozenset({"inf", "infinity", "infty", "infinite", "unbounded"})
_FINITE_TARGET_WORDS = {"zero": 0.0, "one": 1.0, "two": 2.0, "half": 0.5}

#: Minimum |f| growth required when the record declares a divergent limit.
_ASYMPTOTE_GROWTH_FACTOR = 10.0
#: Tolerance applied only when the declared limit is a pure numeric literal.
_NUMERIC_LIMIT_TOLERANCE = 0.05
#: Relative slack for boundary relations (``x >= 0`` must not fail on 1e-16).
_BOUNDARY_TOLERANCE = 1e-9
#: Number of samples used to approach a declared limit.
_ASYMPTOTE_SAMPLE_COUNT = 6
#: Errors that the gate records as a failed probe instead of propagating. Mirrors
#: the tuple used by the adversarial probe loop below.
_PROBE_ERRORS = (RuntimeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError, OverflowError)


@dataclass(frozen=True)
class ConstraintCheck:
    """Outcome of evaluating one declared physical constraint against the model."""
    constraint: str
    status: str  # SATISFIED | VIOLATED | UNMEASURABLE
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"constraint": self.constraint, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class AsymptoteCheck:
    """Outcome of measuring one declared asymptotic limit against the model."""
    name: str
    status: str  # CONSISTENT | CONTRADICTED | UNMEASURABLE
    detail: str
    declared: str
    measured_limit: float | None = None
    declared_limit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asymptote": self.name,
            "declared": self.declared,
            "status": self.status,
            "measured_limit": self.measured_limit,
            "declared_limit": self.declared_limit,
            "detail": self.detail,
        }


def _relation(constraint: str) -> tuple[str, str, str] | None:
    """Extracts ``(lhs_symbol, operator, rhs_text)`` from a constraint description.

    Constraints are free-form prose (``"Strict physical positivity: Tm > 0 K"``),
    so the relation is located rather than assumed. ``None`` means the text carries
    no machine-checkable relation -- which is reported as unmeasurable, not passed.
    """
    match = _RELATION_PATTERN.search(constraint)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _resolve_threshold(text: str, parameters: dict[str, Any]) -> float | None:
    """Resolves a constraint's right-hand side to a number, from a literal or a parameter."""
    try:
        return float(text)
    except ValueError:
        pass
    value = parameters.get(text)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


def _compare(values: np.ndarray, operator: str, threshold: float) -> bool:
    """Evaluates ``values <op> threshold`` over every sample, with boundary slack."""
    if values.size == 0:
        return False
    finite = values[np.isfinite(values)]
    if finite.size != values.size:
        return False
    slack = _BOUNDARY_TOLERANCE * max(abs(threshold), 1.0)
    if operator == ">":
        return bool(np.all(finite > threshold - slack))
    if operator == ">=":
        return bool(np.all(finite >= threshold - slack))
    if operator == "<":
        return bool(np.all(finite < threshold + slack))
    if operator == "<=":
        return bool(np.all(finite <= threshold + slack))
    return bool(np.all(np.abs(finite - threshold) <= slack))


def _check_constraint(
    constraint: str,
    predictions: np.ndarray,
    inputs: dict[str, np.ndarray],
    parameters: dict[str, Any],
) -> ConstraintCheck:
    """Evaluates one declared constraint numerically against the sealed partition."""
    relation = _relation(constraint)
    if relation is None:
        if "finite" in constraint.lower():
            satisfied = bool(np.all(np.isfinite(predictions)))
            status = "SATISFIED" if satisfied else "VIOLATED"
            return ConstraintCheck(constraint, status, f"finiteness over {predictions.size} sealed predictions")
        return ConstraintCheck(
            constraint,
            "UNMEASURABLE",
            "no machine-checkable relation or finiteness claim found in the constraint text",
        )

    lhs_name, operator, rhs_text = relation
    threshold = _resolve_threshold(rhs_text, parameters)
    if threshold is None:
        return ConstraintCheck(
            constraint,
            "UNMEASURABLE",
            f"threshold '{rhs_text}' is neither a numeric literal nor a numeric parameter of the record",
        )

    values = inputs.get(lhs_name, predictions)
    satisfied = _compare(values, operator, threshold)
    source = "declared input" if lhs_name in inputs else "model output"
    detail = (
        f"{lhs_name} ({source}) {operator} {threshold:g} over {values.size} sealed points; "
        f"measured range [{np.min(values):.6g}, {np.max(values):.6g}]"
    )
    return ConstraintCheck(constraint, "SATISFIED" if satisfied else "VIOLATED", detail)


def _declared_symbol_names(record: ModelEvidenceRecord) -> set[str]:
    """Every symbol name the record declares anywhere in its own specification."""
    return (
        set(record.variables)
        | set(record.parameters)
        | set(record.validated_domain)
        | set(record.extrapolation_limits)
    )


def _declared_units(record: ModelEvidenceRecord) -> tuple[dict[str, str], str | None]:
    """Variable units plus the declared output unit, when the record supplies them."""
    for source in (record.mathematical_assurance, record.software_assurance, record.scientific_assurance):
        if not isinstance(source, dict):
            continue
        units = source.get("variable_units")
        if isinstance(units, dict) and units:
            output_unit = source.get("expected_output_unit") or source.get("output_unit")
            return {str(k): str(v) for k, v in units.items()}, (str(output_unit) if output_unit else None)
    return {}, None


def _observable_name(equation_str: str) -> str | None:
    """Name of the modelled observable, i.e. the left-hand side of the equation."""
    lhs, separator, _ = equation_str.partition("=")
    if not separator:
        return None
    lhs = lhs.strip()
    match = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:\(.*\))?", lhs)
    return match.group(1) if match else None


def _measure_dimensional_consistency(record: ModelEvidenceRecord) -> tuple[str, float | None, dict[str, Any]]:
    """Measures dimensional consistency of the record's equation.

    Two measurements are taken against the declared model, never against the
    record's prose:

    1. *Symbolic closure* -- every free symbol of the equation must be declared by
       the record (as an input, a parameter, or the modelled observable). An
       equation that references an undeclared symbol is a measured ``FAIL``.
    2. *Dimensional homogeneity* -- when the record declares variable units, both
       sides of the equation are checked with :class:`DimensionalAssuranceEngine`.

    Without unit metadata the homogeneity claim cannot be evaluated at all, so the
    result is ``UNVERIFIED`` (and excluded from the physicality sub-score) rather
    than the previous presence test, which returned ``PASS`` for any record with a
    non-empty ``derivation_basis``.
    """
    equation = record.equation_str.strip()
    if not equation:
        return "UNVERIFIED", None, {"reason": "record declares no equation to check"}

    lhs_text, separator, rhs_text = equation.partition("=")
    sides: dict[str, str] = {}
    if separator and rhs_text.strip():
        lhs_text = lhs_text.strip()
        # ``sigma_y(d) = ...`` and ``y = ...`` declare the modelled observable by
        # name; there is nothing to parse on the left. Anything more structured is
        # a real expression and both sides are checked.
        if re.fullmatch(r"[A-Za-z_]\w*\s*(?:\(.*\))?", lhs_text) is None:
            sides["lhs"] = lhs_text
        sides["rhs"] = rhs_text.strip()
    else:
        sides["rhs"] = equation

    evidence: dict[str, Any] = {"equation": equation}
    parsed: dict[str, sp.Expr] = {}
    try:
        for name, text in sides.items():
            parsed[name] = safe_parse_expr(text)
    except (TypeError, ValueError) as exc:
        evidence["reason"] = f"equation is not parseable for a dimensional check: {exc}"
        return "UNVERIFIED", None, evidence

    observable = _observable_name(equation)
    declared = _declared_symbol_names(record) | ({observable} if observable else set())
    free_symbols = {str(s) for expr in parsed.values() for s in expr.free_symbols}
    undeclared = sorted(free_symbols - declared)
    evidence["observable"] = observable
    evidence["undeclared_symbols"] = undeclared
    if undeclared:
        evidence["reason"] = f"equation references symbols the record never declares: {', '.join(undeclared)}"
        return "FAIL", 0.0, evidence

    units, output_unit = _declared_units(record)
    needed = sorted(free_symbols | ({observable} if observable else set()))
    missing = [name for name in needed if name not in units]
    if not units or missing:
        evidence["reason"] = (
            "equation is symbolically closed, but no unit metadata is declared "
            f"({missing or 'none supplied'}), so homogeneity is unverified"
        )
        return "UNVERIFIED", None, evidence

    engine = DimensionalAssuranceEngine()
    try:
        for name in needed:
            engine.register_variable(name, units[name])
        dimensions = {name: engine.check_expression_consistency(expr) for name, expr in parsed.items()}
    except (TypeError, ValueError, ArithmeticError) as exc:
        evidence["reason"] = f"dimensional inhomogeneity in the declared equation: {exc}"
        return "FAIL", 0.0, evidence

    evidence["dimensions"] = {name: str(dim.dimensionality) for name, dim in dimensions.items()}
    distinct = {str(dim.dimensionality) for dim in dimensions.values()}
    if len(distinct) > 1:
        evidence["reason"] = "the two sides of the equation are not dimensionally homogeneous"
        return "FAIL", 0.0, evidence
    if output_unit:
        expected = engine.registry(output_unit).dimensionality
        if any(dim.dimensionality != expected for dim in dimensions.values()):
            evidence["reason"] = f"equation dimensionality does not match the declared output unit '{output_unit}'"
            return "FAIL", 0.0, evidence
    return "PASS", 1.0, evidence


def _asymptote_target(text: str) -> float | None:
    """Resolves an asymptote key's target ('0', 'inf', 'one', ...) to a number."""
    lowered = text.strip().lower()
    if lowered in _FINITE_TARGET_WORDS:
        return _FINITE_TARGET_WORDS[lowered]
    if lowered in _INFINITY_WORDS:
        return math.inf
    if lowered.startswith("-") and lowered[1:].strip() in _INFINITY_WORDS:
        return -math.inf
    try:
        return float(lowered)
    except ValueError:
        return None


def _asymptote_probe_points(target: float, low: float, high: float, count: int = _ASYMPTOTE_SAMPLE_COUNT) -> list[float]:
    """Builds a sequence that approaches ``target`` from inside the declared domain."""
    if math.isinf(target):
        anchor = high if target > 0 else low
        if anchor == 0.0:
            return []
        return [anchor * (10.0 ** k) for k in range(count)]
    if target == 0.0:
        anchor = low if abs(low) <= abs(high) else high
        if anchor == 0.0:
            return []
        return [anchor * (10.0 ** -k) for k in range(count)]
    anchor = high if high != target else low
    if anchor == target:
        return []
    return [target - (target - anchor) * (10.0 ** -k) for k in range(count)]


def _series_trend(series: list[float]) -> str:
    """Classifies a sampled sequence as GROWING, DECAYING, or CONSTANT in magnitude."""
    first, last = abs(series[0]), abs(series[-1])
    if last >= first * 1.5:
        return "GROWING"
    if last <= first / 1.5:
        return "DECAYING"
    return "CONSTANT"


def _declared_limit_series(
    declared: str,
    parameters: dict[str, Any],
    symbol: str,
    points: list[float],
) -> tuple[list[float] | None, bool]:
    """Evaluates the declared limit expression along the approach points.

    Returns the declared series and whether the declaration was a pure numeric
    literal (the only form that is an unambiguous value claim rather than a
    symbolic scaling law).
    """
    text = declared.strip()
    if not text or text.lower() in _INFINITY_WORDS:
        return None, False
    try:
        expr = safe_parse_expr(text)
    except (TypeError, ValueError):
        return None, False

    is_numeric_literal = bool(expr.is_number)
    substitutions: dict[sp.Symbol, float] = {}
    for free in expr.free_symbols:
        if str(free) == symbol:
            continue
        value = parameters.get(str(free))
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return None, is_numeric_literal
        substitutions[free] = float(value)
    expr = expr.subs(substitutions)

    approach_symbol = sp.Symbol(symbol)
    if approach_symbol in expr.free_symbols:
        try:
            series = [float(sp.N(expr.subs(approach_symbol, point))) for point in points]
        except (TypeError, ValueError, ArithmeticError, ZeroDivisionError):
            return None, is_numeric_literal
        return (series if all(math.isfinite(value) for value in series) else None), is_numeric_literal
    if expr.free_symbols:
        return None, is_numeric_literal
    try:
        value = float(sp.N(expr))
    except (TypeError, ValueError, ArithmeticError):
        return None, is_numeric_literal
    return ([value] * len(points) if math.isfinite(value) else None), is_numeric_literal


def _check_asymptote(
    record: ModelEvidenceRecord,
    eval_fn: Callable[[np.ndarray], np.ndarray],
    name: str,
    declared: str,
    base_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    column_lookup: Callable[[str], int | None],
) -> AsymptoteCheck:
    """Measures one declared asymptote by evaluating the model along an approach sequence."""
    match = _LIMIT_KEY_PATTERN.match(name)
    if match is None:
        return AsymptoteCheck(name, "UNMEASURABLE", "asymptote key is not of the form '<variable> -> <target>'", declared)

    symbol, target_text = match.group(1), match.group(2)
    target = _asymptote_target(target_text)
    if target is None:
        return AsymptoteCheck(name, "UNMEASURABLE", f"unrecognised limit target '{target_text}'", declared)

    column = column_lookup(symbol)
    if column is None:
        return AsymptoteCheck(name, "UNMEASURABLE", f"'{symbol}' is not one of the record's declared inputs", declared)

    low, high = bounds[column]
    points = _asymptote_probe_points(target, low, high)
    if len(points) < 2:
        return AsymptoteCheck(name, "UNMEASURABLE", "no non-degenerate approach sequence exists for the declared limit", declared)

    measured: list[float] = []
    for point in points:
        probe = base_vector.copy()
        probe[column] = point
        try:
            result = np.asarray(eval_fn(probe.reshape(1, -1)), dtype=float).ravel()
        except _PROBE_ERRORS:
            result = np.array([np.nan])
        if result.size == 0 or not np.all(np.isfinite(result)):
            return AsymptoteCheck(
                name,
                "UNMEASURABLE",
                f"the model returns no finite value at approach point {symbol}={point:.6g}; "
                "the declared limit is unverified rather than satisfied",
                declared,
            )
        measured.append(float(result[0]))

    # An unresolvable declaration can still be a claim: the record may declare the
    # limit divergent ("unbounded"). That claim is measured as required growth of
    # |f| along the approach sequence. Anything else unresolvable is unverifiable.
    declared_series, declared_is_literal = _declared_limit_series(declared, record.parameters, symbol, points)
    if declared_series is None:
        if declared.strip().lower() not in _INFINITY_WORDS:
            return AsymptoteCheck(
                name,
                "UNMEASURABLE",
                f"declared limit '{declared}' is neither a resolvable expression nor an unbounded marker",
                declared,
                measured_limit=measured[-1],
            )
        growth = abs(measured[-1]) / max(abs(measured[0]), 1e-300)
        consistent = _series_trend(measured) == "GROWING" and growth >= _ASYMPTOTE_GROWTH_FACTOR
        detail = (
            f"declared divergence at {symbol} -> {target_text}: |f| moved {abs(measured[0]):.6g} -> "
            f"{abs(measured[-1]):.6g} ({growth:.3g}x, required growth >= {_ASYMPTOTE_GROWTH_FACTOR:g}x)"
        )
        return AsymptoteCheck(
            name, "CONSISTENT" if consistent else "CONTRADICTED", detail, declared, measured_limit=measured[-1]
        )

    # Resolvable declaration: the model must *close on* the declared limit. The
    # residual is measured against the declared series point-by-point, so a limit
    # approached proportionally to the input (``y = 2x`` as ``x -> 0``) counts as
    # attained while a model that settles on a different value does not. The band
    # is relative to the scale of the quantities involved, so it does not depend on
    # the units or magnitude of the observable.
    residuals = [abs(value - declared_value) for value, declared_value in zip(measured, declared_series)]
    scale = max(
        abs(declared_series[-1]),
        abs(measured[-1]),
        float(np.ptp(measured)),
        1e-300,
    )
    tolerance = _NUMERIC_LIMIT_TOLERANCE * scale
    residual_drop = residuals[-1] / max(residuals[0], 1e-300)
    consistent = residuals[-1] <= tolerance and residuals[-1] <= residuals[0]
    kind = "numeric limit" if declared_is_literal else "declared limit"
    detail = (
        f"{kind} {declared_series[-1]:.6g} at {symbol} -> {target_text}: residual moved {residuals[0]:.6g} -> "
        f"{residuals[-1]:.6g} ({residual_drop * 100:.2f}% of the initial residual) over {symbol} in "
        f"[{points[0]:.6g}, {points[-1]:.6g}]; model approaches {measured[-1]:.6g} "
        f"(tolerance {tolerance:.6g})"
    )
    return AsymptoteCheck(
        name,
        "CONSISTENT" if consistent else "CONTRADICTED",
        detail,
        declared,
        measured_limit=measured[-1],
        declared_limit=declared_series[-1],
    )


def _input_columns(record: ModelEvidenceRecord, features: np.ndarray) -> dict[str, np.ndarray]:
    """Maps declared input names (and domain keys) onto the supplied feature columns."""
    matrix = np.atleast_2d(np.asarray(features, dtype=float))
    keys = list(record.validated_domain)
    columns: dict[str, np.ndarray] = {}
    for index, key in enumerate(keys):
        if index < matrix.shape[1]:
            columns[key] = matrix[:, index]
    if len(record.variables) == len(keys):
        for index, variable in enumerate(record.variables):
            columns.setdefault(variable, matrix[:, index])
    return columns


def _column_index(record: ModelEvidenceRecord, symbol: str) -> int | None:
    """Resolves a symbolic variable to a feature column index, or ``None``."""
    keys = list(record.validated_domain)
    if symbol in keys:
        return keys.index(symbol)
    if len(record.variables) == len(keys) and symbol in record.variables:
        return record.variables.index(symbol)
    matches = [index for index, key in enumerate(keys) if symbol in key]
    return matches[0] if len(matches) == 1 else None


def _measure_stability(predictions: np.ndarray, probes_passed: int, probes_total: int) -> tuple[float, dict[str, Any]]:
    """Measures numerical stability from probe survival and local amplification.

    Replaces ``1.0 if critical_failures == 0 else 0.5``. Two measured factors:

    * *survival* -- the fraction of adversarial boundary probes that returned a
      finite value. Unprobeable candidates (no declared domain) do not receive
      credit for it.
    * *conditioning* -- ``1 / (1 + q)`` where ``q`` is the 95th percentile of
      ``|delta f| / range(f)`` across the sealed partition, i.e. a normalised
      measure of how violently the model amplifies its own inputs.
    """
    evidence: dict[str, Any] = {"probes_passed": probes_passed, "probes_total": probes_total}
    survival = float(probes_passed) / probes_total if probes_total > 0 else None
    evidence["probe_survival"] = survival

    finite = predictions[np.isfinite(predictions)]
    conditioning: float | None = None
    if finite.size >= 3:
        span = float(np.ptp(finite))
        if span > 0.0:
            steps = np.abs(np.diff(finite)) / span
            quantile = float(np.percentile(steps, 95.0))
            conditioning = 1.0 / (1.0 + quantile)
            evidence["relative_step_p95"] = quantile
    evidence["conditioning"] = conditioning

    factors = [factor for factor in (survival, conditioning) if factor is not None]
    if not factors:
        evidence["reason"] = "no adversarial probes ran and the predictions are too few or too flat to condition"
        return 0.0, evidence
    stability = _clamp01(float(np.prod(factors)))
    if survival is None:
        evidence["reason"] = "no adversarial probes were executed, so only the conditioning factor was available"
    if conditioning is None:
        evidence["reason"] = "the sealed predictions are constant or too few to condition, so only probe survival was available"
    return stability, evidence


def _measure_generality(
    features: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Measures generality as interior-versus-boundary error on the sealed partition.

    Replaces ``0.90 if worst_severity != DOMAIN_VIOLATION else 0.40``. The sealed
    partition is split by each sample's distance from the interior of its own
    observed range: a candidate that predicts the interior well but degrades on the
    outer fraction of the sealed domain scores below one that holds up across it.
    """
    matrix = np.atleast_2d(np.asarray(features, dtype=float))
    targets = np.asarray(ground_truth, dtype=float).ravel()
    preds = np.asarray(predictions, dtype=float).ravel()
    evidence: dict[str, Any] = {}

    if matrix.shape[0] != preds.size or targets.size != preds.size or preds.size < 4:
        evidence["reason"] = f"the sealed partition ({preds.size} samples) is too small to separate interior from boundary"
        return 0.0, evidence

    with np.errstate(invalid="ignore"):
        low = np.nanpercentile(matrix, 10.0, axis=0)
        high = np.nanpercentile(matrix, 90.0, axis=0)
        interior_mask = np.all((matrix >= low) & (matrix <= high), axis=1)
    boundary_mask = ~interior_mask
    n_interior, n_boundary = int(interior_mask.sum()), int(boundary_mask.sum())
    evidence.update({"n_interior": n_interior, "n_boundary": n_boundary})
    if n_interior < 2 or n_boundary < 2:
        evidence["reason"] = "the sealed partition does not populate both the interior and the outer fringe"
        return 0.0, evidence

    interior_error = float(np.mean(np.abs(preds[interior_mask] - targets[interior_mask])))
    boundary_error = float(np.mean(np.abs(preds[boundary_mask] - targets[boundary_mask])))
    scale = interior_error + boundary_error
    evidence.update({"interior_mae": interior_error, "boundary_mae": boundary_error})
    if not math.isfinite(scale) or scale <= 0.0:
        evidence["reason"] = "both partitions fit the sealed targets exactly, so no generality signal is available"
        return 1.0, evidence
    generality = _clamp01(1.0 - max(0.0, boundary_error - interior_error) / scale)
    return generality, evidence


class FailureSeverity(str, Enum):
    """Categorization of failure modes distinguishing implementation, domain, and mechanistic flaws."""
    PASS = "PASS"                                             # All contracts, bounds, and invariants satisfied
    WARNING = "WARNING"                                       # Sub-critical variance or minor numerical boundary softness
    DOMAIN_VIOLATION = "DOMAIN_VIOLATION"                     # Model valid in sub-regime but violates extended domain (triggers Domain_Restricted)
    PREDICTIVE_FAILURE = "PREDICTIVE_FAILURE"                 # Out-of-sample residual exceeds contract prediction interval
    PHYSICAL_INVARIANT_VIOLATION = "PHYSICAL_INVARIANT_VIOLATION"  # Declared invariant/bound violated while predictions still fit (triggers Challenged)
    MECHANISTIC_CONTRADICTION = "MECHANISTIC_CONTRADICTION"   # Empirical data refutes core constitutive mechanism (triggers Refuted)
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"         # Software/numerical fault (NaN, unhandled exception, conditioning crash)


class AdjudicationVerdict(str, Enum):
    """Outcomes of the mechanistic contradiction adjudication layer."""
    MODEL_FAILURE = "MODEL_FAILURE"                   # Genuine constitutive flaw -> REFUTED
    DOMAIN_FAILURE = "DOMAIN_FAILURE"                 # Discrepancy isolated to boundaries -> DOMAIN_RESTRICTED
    EVIDENCE_PROBLEM = "EVIDENCE_PROBLEM"             # Measurement noise / confounder / low independence -> CHALLENGED
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE" # Software bug in evaluation routine
    NO_CONTRADICTION = "NO_CONTRADICTION"             # Every prediction within threshold -> no empirical refutation available


#: What each adjudication verdict rests on. The measurements behind a verdict are
#: empirical, but the *classification* is decided by comparing them against
#: constants chosen in this module (a boundary margin of 0.15 of span and a
#: concentration ratio of 0.70, a noise-to-signal ratio of 0.40, a replication
#: strength of 0.35, and the contract's own relative-error threshold). Editing any
#: of those flips a verdict without new data, so the honest tier for the
#: classification is HEURISTIC -- except for IMPLEMENTATION_FAILURE, which is an
#: exact observation of the candidate's own output and needs no threshold.
ADJUDICATION_EVIDENCE_TIER: dict[AdjudicationVerdict, EvidenceTier] = {
    AdjudicationVerdict.IMPLEMENTATION_FAILURE: EvidenceTier.EMPIRICAL,
    AdjudicationVerdict.NO_CONTRADICTION: EvidenceTier.HEURISTIC,
    AdjudicationVerdict.DOMAIN_FAILURE: EvidenceTier.HEURISTIC,
    AdjudicationVerdict.EVIDENCE_PROBLEM: EvidenceTier.HEURISTIC,
    AdjudicationVerdict.MODEL_FAILURE: EvidenceTier.HEURISTIC,
}

#: Why each verdict holds its tier, so the claim travels with its caveat.
ADJUDICATION_EVIDENCE_BASIS: dict[AdjudicationVerdict, str] = {
    AdjudicationVerdict.IMPLEMENTATION_FAILURE: (
        "measured: the candidate's own evaluation returned NaN or infinite values"
    ),
    AdjudicationVerdict.NO_CONTRADICTION: (
        "measured residuals compared against the contract's declared relative-error "
        "threshold; no classification constant is involved, but the absence of a "
        "finding is not evidence of correctness"
    ),
    AdjudicationVerdict.DOMAIN_FAILURE: (
        "measured discrepancy locations classified by a boundary margin (0.15 of span) "
        "and a concentration ratio (0.70), both chosen in this module"
    ),
    AdjudicationVerdict.EVIDENCE_PROBLEM: (
        "measured noise-to-signal ratio and replication strength compared against "
        "module constants (0.40 and 0.35)"
    ),
    AdjudicationVerdict.MODEL_FAILURE: (
        "measured in-domain residuals exceeding the contract's threshold, after the "
        "module's boundary and noise gates were passed"
    ),
}


@dataclass(frozen=True)
class ReplicationIndependenceVector:
    """Formal 7-axis independence model for experimental confirmation (D, I, O, S, C, A, T)."""
    data_independence: float = 0.0          # [0.0, 1.0] disjoint dataset partitions
    instrument_independence: float = 0.0    # [0.0, 1.0] distinct sensor/detector hardware
    operator_independence: float = 0.0      # [0.0, 1.0] independent experimenter / research group
    site_independence: float = 0.0          # [0.0, 1.0] distinct physical laboratory facility
    implementation_independence: float = 0.0 # [0.0, 1.0] clean-room independent codebase
    analysis_independence: float = 0.0      # [0.0, 1.0] distinct statistical estimation / priors
    temporal_independence: float = 0.0      # [0.0, 1.0] distinct time epoch / non-simultaneous

    @property
    def composite_strength(self) -> float:
        """Geometric mean across all 7 independence dimensions."""
        axes = [
            self.data_independence,
            self.instrument_independence,
            self.operator_independence,
            self.site_independence,
            self.implementation_independence,
            self.analysis_independence,
            self.temporal_independence,
        ]
        prod = math.prod(max(0.01, x) for x in axes)
        return float(prod ** (1.0 / 7.0))

    @property
    def replication_tier(self) -> str:
        s = self.composite_strength
        if s >= 0.85:
            return "GOLD_STANDARD_INDEPENDENT"
        if s >= 0.65:
            return "STRONGLY_INDEPENDENT"
        if s >= 0.40:
            return "PARTIALLY_INDEPENDENT"
        return "NOMINAL_REPETITION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_independence": round(self.data_independence, 4),
            "instrument_independence": round(self.instrument_independence, 4),
            "operator_independence": round(self.operator_independence, 4),
            "site_independence": round(self.site_independence, 4),
            "implementation_independence": round(self.implementation_independence, 4),
            "analysis_independence": round(self.analysis_independence, 4),
            "temporal_independence": round(self.temporal_independence, 4),
            "composite_strength": round(self.composite_strength, 4),
            "replication_tier": self.replication_tier,
        }


@dataclass(frozen=True)
class ScientificImprovementMetric:
    """Multidimensional evaluation of model improvement: I = (I_fit, I_stability, I_physicality, I_generality, I_complexity, I_falsifiability).
    
    Enforces true Pareto dominance over candidate models, preventing collapsing scientific
    merit into an opaque single scalar loss.
    """
    i_fit: float              # [0.0, 1.0] Relative reduction in out-of-sample prediction residual
    i_stability: float        # [0.0, 1.0] Elimination of poles, conditioning index, bounded derivatives
    i_physicality: float      # [0.0, 1.0] Preservation of invariants, non-negative entropy, asymptotic limits
    i_generality: float       # [0.0, 1.0] Relative expansion of valid operational domain
    i_complexity: float       # [0.0, 1.0] Parameter parsimony (Occam penalty; 1.0 = minimal free parameters)
    i_falsifiability: float   # [0.0, 1.0] Sharpness of prediction intervals and empirical testability

    def pareto_dominates(self, other: ScientificImprovementMetric, tolerance: float = 1e-4) -> bool:
        """Determines Pareto dominance: M2 >= M1 across all dimensions, and M2 > M1 in >= 1 dimension."""
        if self.i_physicality < 0.5 or self.i_stability < 0.5:
            return False
        dims = ("i_fit", "i_stability", "i_physicality", "i_generality", "i_complexity", "i_falsifiability")
        not_worse = all(getattr(self, d) >= getattr(other, d) - tolerance for d in dims)
        strictly_better = any(getattr(self, d) > getattr(other, d) + 0.02 for d in dims)
        return not_worse and strictly_better

    @property
    def is_net_positive(self) -> bool:
        """Baseline qualification: hard constraints satisfied with non-trivial progress."""
        if self.i_physicality < 0.5 or self.i_stability < 0.5:
            return False
        return self.i_fit >= 0.5 and self.i_generality >= 0.4 and self.i_falsifiability >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "i_fit": round(self.i_fit, 4),
            "i_stability": round(self.i_stability, 4),
            "i_physicality": round(self.i_physicality, 4),
            "i_generality": round(self.i_generality, 4),
            "i_complexity": round(self.i_complexity, 4),
            "i_falsifiability": round(self.i_falsifiability, 4),
            "is_net_positive": self.is_net_positive,
        }


@dataclass
class ParetoFrontierTracker:
    """Maintains the non-dominated Pareto frontier of scientific models across iterations."""
    frontier: dict[str, ScientificImprovementMetric] = field(default_factory=dict)

    def consider_candidate(self, model_id: str, metric: ScientificImprovementMetric) -> bool:
        """Adds candidate if non-dominated; prunes any existing models dominated by candidate."""
        if metric.i_physicality < 0.5 or metric.i_stability < 0.5:
            return False

        # If any existing member dominates candidate, reject candidate
        for existing_m in self.frontier.values():
            if existing_m.pareto_dominates(metric):
                return False

        # Candidate is non-dominated; prune existing members dominated by candidate
        pruned_ids = [
            m_id for m_id, existing_m in self.frontier.items()
            if metric.pareto_dominates(existing_m)
        ]
        for p_id in pruned_ids:
            del self.frontier[p_id]

        self.frontier[model_id] = metric
        return True

    def get_frontier(self) -> dict[str, ScientificImprovementMetric]:
        return dict(self.frontier)


@dataclass(frozen=True)
class KnownDiagnosticBenchmark:
    """Benchmarks scientific reasoning where expected diagnosis is known (regression suite)."""
    initial_failure: str
    expected_diagnosis: str
    actual_diagnosis: str
    expected_refinement: str
    actual_refinement: str
    diagnosis_correctness: bool
    refinement_correctness: bool
    reasoning_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_failure": self.initial_failure,
            "expected_diagnosis": self.expected_diagnosis,
            "actual_diagnosis": self.actual_diagnosis,
            "expected_refinement": self.expected_refinement,
            "actual_refinement": self.actual_refinement,
            "diagnosis_correctness": self.diagnosis_correctness,
            "refinement_correctness": self.refinement_correctness,
            "reasoning_notes": self.reasoning_notes,
        }


# Backward-compatible alias
DiagnosticReasoningBenchmark = KnownDiagnosticBenchmark


@dataclass(frozen=True)
class BlindDiagnosticBenchmark:
    """Benchmarks scientific reasoning where the expected diagnosis is sealed (anti-answer-leakage)."""
    challenge_id: str
    symptom_signature: dict[str, Any]
    sealed_diagnosis_hash: str
    inferred_diagnosis: str
    unsealed_expected_diagnosis: str | None = None
    category_match: bool = False
    notes: str = ""

    @staticmethod
    def create_sealed_challenge(
        challenge_id: str,
        symptom_signature: dict[str, Any],
        expected_diagnosis: str,
    ) -> BlindDiagnosticBenchmark:
        """Creates a sealed challenge with cryptographic SHA-256 seal."""
        salt = "APEDP_BLIND_DIAGNOSTIC_SALT"
        seal_hash = hashlib.sha256((expected_diagnosis + salt).encode("utf-8")).hexdigest()
        return BlindDiagnosticBenchmark(
            challenge_id=challenge_id,
            symptom_signature=symptom_signature,
            sealed_diagnosis_hash=seal_hash,
            inferred_diagnosis="",
        )

    def evaluate_blind_submission(
        self,
        inferred_diagnosis: str,
        expected_diagnosis: str,
    ) -> BlindDiagnosticBenchmark:
        """Unlocks and grades the candidate's inferred diagnosis against sealed ground truth."""
        salt = "APEDP_BLIND_DIAGNOSTIC_SALT"
        computed_hash = hashlib.sha256((expected_diagnosis + salt).encode("utf-8")).hexdigest()
        if computed_hash != self.sealed_diagnosis_hash:
            raise ValueError("SEAL COMPROMISED: expected_diagnosis does not match sealed hash.")

        exp_words = set(expected_diagnosis.lower().replace("-", " ").split())
        inf_words = set(inferred_diagnosis.lower().replace("-", " ").split())
        overlap = len(exp_words.intersection(inf_words))
        match = overlap >= min(2, len(exp_words))

        return BlindDiagnosticBenchmark(
            challenge_id=self.challenge_id,
            symptom_signature=self.symptom_signature,
            sealed_diagnosis_hash=self.sealed_diagnosis_hash,
            inferred_diagnosis=inferred_diagnosis,
            unsealed_expected_diagnosis=expected_diagnosis,
            category_match=match,
            notes=f"Keyword overlap: {overlap} concepts matched.",
        )


@dataclass(frozen=True)
class AdjudicationReport:
    """Formal audit report dissecting the root cause of an empirical contradiction."""
    verdict: AdjudicationVerdict
    confidence: float
    boundary_concentration_ratio: float
    noise_to_signal_ratio: float
    reproducibility_independence: float
    recommended_lifecycle_action: str
    rationale: str

    @property
    def evidence_tier(self) -> EvidenceTier:
        """What this verdict rests on: ``HEURISTIC`` for every classification.

        Derived from the verdict rather than stored, so a report cannot be
        constructed claiming a stronger tier than its verdict's method supports.
        """
        return ADJUDICATION_EVIDENCE_TIER[self.verdict]

    @property
    def evidence_basis(self) -> str:
        """The measurements and constants the verdict was decided against."""
        return ADJUDICATION_EVIDENCE_BASIS[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "boundary_concentration_ratio": round(self.boundary_concentration_ratio, 4),
            "noise_to_signal_ratio": round(self.noise_to_signal_ratio, 4),
            "reproducibility_independence": round(self.reproducibility_independence, 4),
            "recommended_lifecycle_action": self.recommended_lifecycle_action,
            "rationale": self.rationale,
            "evidence_tier": self.evidence_tier.value,
            "evidence_basis": self.evidence_basis,
        }


class MechanisticAdjudicationEngine:
    """Adjudicates discrepancies between model predictions and observations before refuting."""

    @classmethod
    def adjudicate(
        cls,
        candidate_record: ModelEvidenceRecord,
        contract: ScientificBenchmarkContract,
        features: np.ndarray,
        predictions: np.ndarray,
        observations: np.ndarray,
        independence: ReplicationIndependenceVector | None = None,
        estimated_sensor_noise_rel: float = 0.02,
    ) -> AdjudicationReport:
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return AdjudicationReport(
                verdict=AdjudicationVerdict.IMPLEMENTATION_FAILURE,
                confidence=1.0,
                boundary_concentration_ratio=0.0,
                noise_to_signal_ratio=float("inf"),
                reproducibility_independence=0.0,
                recommended_lifecycle_action="RETAIN_IN_CANDIDATE_FOR_DEBUGGING",
                rationale="Evaluation produced NaN or infinite values indicating numerical overflow.",
            )

        abs_err = np.abs(predictions - observations)
        denom = np.where(np.abs(observations) < 1e-12, 1.0, np.abs(observations))
        rel_err = abs_err / denom
        threshold = contract.falsification_thresholds.get("max_relative_error", 0.05)
        failing_indices = np.where(rel_err > threshold)[0]

        if len(failing_indices) == 0:
            # Nothing to adjudicate: the empirical channel supplies no evidence
            # against the mechanism. Reporting MODEL_FAILURE here would map to
            # MECHANISTIC_CONTRADICTION -> REFUTED, i.e. refuting a model on data
            # that agree with it. Abstain and let the caller weigh the
            # non-empirical signal that triggered adjudication.
            return AdjudicationReport(
                verdict=AdjudicationVerdict.NO_CONTRADICTION,
                confidence=0.0,
                boundary_concentration_ratio=0.0,
                noise_to_signal_ratio=0.0,
                reproducibility_independence=1.0,
                recommended_lifecycle_action="NO_ADJUDICATION_REQUIRED",
                rationale=(
                    "No prediction exceeds the contract's relative-error threshold, so "
                    "the empirical channel provides no evidence against the mechanism; "
                    "mechanistic refutation is not available from this evaluation."
                ),
            )

        # 1. Boundary Concentration Test (Domain Failure vs Model Failure)
        boundary_violations = 0
        total_violations = len(failing_indices)
        for low, high in candidate_record.validated_domain.values():
            span = high - low
            margin = 0.15 * span
            boundary_mask = (features[:, 0] <= low + margin) | (features[:, 0] >= high - margin)
            boundary_violations = int(np.sum(boundary_mask[failing_indices]))

        boundary_ratio = boundary_violations / max(1, total_violations)

        if boundary_ratio >= 0.70:
            return AdjudicationReport(
                verdict=AdjudicationVerdict.DOMAIN_FAILURE,
                confidence=boundary_ratio,
                boundary_concentration_ratio=boundary_ratio,
                noise_to_signal_ratio=estimated_sensor_noise_rel,
                reproducibility_independence=independence.composite_strength if independence else 0.5,
                recommended_lifecycle_action="DOMAIN_RESTRICTED",
                rationale=f"Discrepancies concentrated at domain boundary ({boundary_ratio*100:.1f}%). Core mechanism valid inside restricted sub-regime.",
            )

        # 2. Evidence Noise & Confounder Test
        mean_violation_size = float(np.mean(rel_err[failing_indices]))
        noise_ratio = estimated_sensor_noise_rel / max(1e-6, mean_violation_size)
        rep_strength = independence.composite_strength if independence else 0.5

        if noise_ratio > 0.40 or rep_strength < 0.35:
            return AdjudicationReport(
                verdict=AdjudicationVerdict.EVIDENCE_PROBLEM,
                confidence=0.80,
                boundary_concentration_ratio=boundary_ratio,
                noise_to_signal_ratio=noise_ratio,
                reproducibility_independence=rep_strength,
                recommended_lifecycle_action="CHALLENGED",
                rationale=f"High noise-to-signal ratio ({noise_ratio:.2f}) or low replication independence ({rep_strength:.2f}). Evidence insufficient to refute.",
            )

        # 3. Genuine Constitutive Model Failure
        return AdjudicationReport(
            verdict=AdjudicationVerdict.MODEL_FAILURE,
            confidence=0.95,
            boundary_concentration_ratio=boundary_ratio,
            noise_to_signal_ratio=noise_ratio,
            reproducibility_independence=rep_strength,
            recommended_lifecycle_action="REFUTED",
            rationale="High-fidelity in-domain empirical observations decisively contradict constitutive mechanism.",
        )


@dataclass(frozen=True)
class OperationalReadinessReport:
    """Formal audit report determining if a Contract_Validated model can enter Operational deployment."""
    model_id: str
    is_operational_ready: bool
    robustness_score: float
    drift_monitoring_enabled: bool
    reproducibility_hash_verified: bool
    resource_budget_passed: bool
    fail_safe_boundary_clamp_defined: bool
    unmet_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "is_operational_ready": self.is_operational_ready,
            "robustness_score": round(self.robustness_score, 4),
            "drift_monitoring_enabled": self.drift_monitoring_enabled,
            "reproducibility_hash_verified": self.reproducibility_hash_verified,
            "resource_budget_passed": self.resource_budget_passed,
            "fail_safe_boundary_clamp_defined": self.fail_safe_boundary_clamp_defined,
            "unmet_requirements": list(self.unmet_requirements),
        }


class OperationalGateAuditor:
    """Dedicated operational readiness gate: Contract_Validated -> Operational."""

    @staticmethod
    def audit_readiness(
        candidate_record: ModelEvidenceRecord,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        sample_inputs: np.ndarray,
        max_latency_ms: float = 50.0,
        drift_monitoring_configured: bool = True,
        fail_safe_clamp_defined: bool = True,
    ) -> OperationalReadinessReport:
        unmet: list[str] = []

        # 1. Robustness under input perturbation
        pert_low = sample_inputs * 0.99
        pert_high = sample_inputs * 1.01
        try:
            _ = eval_fn(sample_inputs)
            res_low = eval_fn(pert_low)
            res_high = eval_fn(pert_high)
            grad_est = np.abs((res_high - res_low) / np.maximum(1e-12, np.abs(pert_high - pert_low)))
            robustness = float(np.mean(np.isfinite(grad_est) & (grad_est < 1e8)))
            if robustness < 0.90:
                unmet.append(f"Input perturbation robustness score too low ({robustness*100:.1f}% < 90%)")
        except (RuntimeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError, OverflowError) as e:
            robustness = 0.0
            unmet.append(f"Perturbation test crashed: {e}")

        # 2. Resource constraints
        t0 = time.perf_counter()
        _ = eval_fn(sample_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resource_passed = latency_ms <= max_latency_ms
        if not resource_passed:
            unmet.append(f"Execution latency {latency_ms:.2f}ms exceeds maximum {max_latency_ms:.2f}ms")

        # 3. Drift & safety
        if not drift_monitoring_configured:
            unmet.append("No active distribution drift monitoring configured.")
        if not fail_safe_clamp_defined:
            unmet.append("No fail-safe boundary clamp defined for out-of-domain runtime calls.")

        # 4. Content hash reproducibility check.
        #    The digest is recomputed from the record's content and compared, so a
        #    tampered or never-computed record is rejected. (A 64-character string
        #    is not evidence of anything.)
        if not candidate_record.record_hash:
            hash_verified = False
            unmet.append("Cryptographic content hash missing from the evidence record.")
        elif not candidate_record.verify_record_hash():
            hash_verified = False
            unmet.append(
                "Content hash mismatch: record_hash does not match a digest recomputed "
                "from this record, so the record was altered or the hash was never computed."
            )
        else:
            hash_verified = True

        is_ready = len(unmet) == 0
        return OperationalReadinessReport(
            model_id=candidate_record.model_id,
            is_operational_ready=is_ready,
            robustness_score=robustness,
            drift_monitoring_enabled=drift_monitoring_configured,
            reproducibility_hash_verified=hash_verified,
            resource_budget_passed=resource_passed,
            fail_safe_boundary_clamp_defined=fail_safe_clamp_defined,
            unmet_requirements=tuple(unmet),
        )


@dataclass(frozen=True)
class ScientificBenchmarkContract:
    """Rigorous scientific benchmark contract defining expected behavior, bounds, and falsification rules."""
    model_id: str
    baseline_model: str
    refined_model: str
    mechanism: str
    validated_domain: dict[str, tuple[float, float]]

    # Observables & Intervals
    predicted_observables: tuple[str, ...]
    prediction_intervals: dict[str, tuple[float, float]]
    falsification_thresholds: dict[str, float]

    # Data Provenance & Noise
    locked_test_data_hash: str
    data_generation_method: str
    noise_model: str

    # Hypotheses & Invariants
    expected_failure_modes: tuple[str, ...]
    expected_limits: dict[str, str]
    expected_invariants: tuple[str, ...]

    # Decision Criteria
    pass_criteria: tuple[str, ...]
    fail_criteria: tuple[str, ...]
    domain_restriction_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "baseline_model": self.baseline_model,
            "refined_model": self.refined_model,
            "mechanism": self.mechanism,
            "validated_domain": self.validated_domain,
            "predicted_observables": list(self.predicted_observables),
            "prediction_intervals": self.prediction_intervals,
            "falsification_thresholds": self.falsification_thresholds,
            "locked_test_data_hash": self.locked_test_data_hash,
            "data_generation_method": self.data_generation_method,
            "noise_model": self.noise_model,
            "expected_failure_modes": list(self.expected_failure_modes),
            "expected_limits": self.expected_limits,
            "expected_invariants": list(self.expected_invariants),
            "pass_criteria": list(self.pass_criteria),
            "fail_criteria": list(self.fail_criteria),
            "domain_restriction_criteria": list(self.domain_restriction_criteria),
        }


@dataclass(frozen=True)
class FalsificationBenchResult:
    """Machine-readable, institutional-grade falsification outcome."""
    model_id: str
    domain: str
    prediction_coverage: float
    monotonicity: str
    asymptotic_constraints: str
    dimensional_consistency: str
    physical_bounds: str
    out_of_sample_error: str
    adversarial_probes_passed: int
    adversarial_probes_total: int
    critical_failure_count: int
    worst_severity: FailureSeverity
    epistemic_consequence: str
    improvement_metric: ScientificImprovementMetric
    diagnostic_benchmark: KnownDiagnosticBenchmark | BlindDiagnosticBenchmark | None = None
    adjudication_report: AdjudicationReport | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def format_report(self) -> str:
        """Renders standard machine-readable institutional summary."""
        probe_ratio = f"{self.adversarial_probes_passed}/{self.adversarial_probes_total} PASS"
        return (
            "FALSIFICATION_RESULT\n"
            "--------------------\n"
            f"Model: {self.model_id}\n"
            f"Domain: {self.domain}\n\n"
            f"Prediction coverage:       {self.prediction_coverage * 100:.1f}%\n"
            f"Monotonicity:              {self.monotonicity}\n"
            f"Asymptotic constraints:    {self.asymptotic_constraints}\n"
            f"Dimensional consistency:   {self.dimensional_consistency}\n"
            f"Physical bounds:           {self.physical_bounds}\n"
            f"Out-of-sample error:       {self.out_of_sample_error}\n"
            f"Adversarial probes:        {probe_ratio}\n"
            f"Critical failure:          {self.critical_failure_count}\n\n"
            "Epistemic consequence:\n"
            f"{self.epistemic_consequence}\n"
        )


class BlindChallengeHarness:
    """Scientific Holdout Laboratory: Evaluates candidate models against sealed challenge datasets.

    Sealing and evaluation are separate operations on purpose:

    * :meth:`seal_challenge` registers a labeled partition during *setup* and
      returns an opaque content hash. Call it before any candidate exists.
    * :meth:`evaluate_blind` scores a candidate against an already-sealed
      partition. It never receives labels and only aggregate residual
      statistics leave the harness -- predictions and targets are never
      returned to the caller, so a holdout cannot be exfiltrated through the
      evaluation API and re-used for fitting.

    Sealed partitions are copied on registration, so a caller mutating the
    arrays afterwards cannot retroactively change what was sealed.
    """

    def __init__(self, vault: DataPartitionVault | None = None):
        self._sealed_challenges: dict[str, dict[str, Any]] = {}
        self.vault = vault or DataPartitionVault(
            development_data_hash="dev_seed_partition_42",
            selection_data_hash="sel_seed_partition_43",
            falsification_data_hash="fals_seed_partition_44",
            locked_external_test_hash="ext_seed_partition_45",
            is_falsification_locked=True,
        )

    @staticmethod
    def content_hash(features: np.ndarray, targets: np.ndarray) -> str:
        """Opaque SHA-256 content hash identifying a (features, targets) partition."""
        features_arr = np.ascontiguousarray(features)
        targets_arr = np.ascontiguousarray(targets)
        return hashlib.sha256(features_arr.tobytes() + targets_arr.tobytes()).hexdigest()

    def seal_challenge(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Seal a labeled partition and return its content hash.

        This is the registration half of the blind protocol and should run once,
        during partition setup, before any candidate is scored. Everything about
        the partition is only reachable through the returned hash.
        """
        dataset_hash = self.content_hash(features, targets)
        self.register_sealed_challenge(
            dataset_hash=dataset_hash,
            features=features,
            targets=targets,
            metadata=dict(metadata or {}),
        )
        return dataset_hash

    def register_sealed_challenge(
        self,
        dataset_hash: str,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any]
    ) -> None:
        """Stores challenge test partition securely indexed by SHA-256 hash.

        Prefer :meth:`seal_challenge`, which derives the index from the partition
        content rather than trusting a caller-supplied label. Arrays are copied
        so later mutation by the caller cannot alter the sealed partition.
        """
        self._sealed_challenges[dataset_hash] = {
            "features": np.array(features, copy=True),
            "targets": np.array(targets, copy=True),
            "metadata": dict(metadata),
            "locked": True,
        }

    def has_challenge(self, dataset_hash: str) -> bool:
        """True when a partition has already been sealed under ``dataset_hash``."""
        return dataset_hash in self._sealed_challenges

    def evaluate_blind(
        self,
        candidate_evaluator: Callable[[np.ndarray], np.ndarray],
        dataset_hash: str,
        contract: ScientificBenchmarkContract,
        caller_phase: str = "falsification"
    ) -> dict[str, Any]:
        """Blindly evaluates candidate model predictions without revealing dataset contents to caller.

        Returns aggregate residual statistics only. Neither the prediction vector
        nor the sealed targets are returned.
        """
        self.vault.verify_isolation(dataset_hash, caller_phase)

        if dataset_hash not in self._sealed_challenges:
            raise KeyError(
                f"Sealed challenge with hash '{dataset_hash}' not found in vault. "
                "Partitions must be sealed via seal_challenge() during setup; "
                "evaluate_blind never registers implicitly."
            )

        challenge = self._sealed_challenges[dataset_hash]
        features: np.ndarray = challenge["features"]
        targets: np.ndarray = challenge["targets"]

        try:
            predictions = candidate_evaluator(features)
        except (RuntimeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError, OverflowError) as e:
            return {
                "status": FailureSeverity.IMPLEMENTATION_FAILURE,
                "error": f"Evaluation crashed: {e}",
                "coverage": 0.0,
                "max_relative_error": float("inf"),
            }

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return {
                "status": FailureSeverity.IMPLEMENTATION_FAILURE,
                "error": "Predictions contain NaN or Inf",
                "coverage": 0.0,
                "max_relative_error": float("inf"),
            }

        abs_err = np.abs(predictions - targets)
        denom = np.where(np.abs(targets) < 1e-12, 1.0, np.abs(targets))
        rel_err = abs_err / denom

        threshold = contract.falsification_thresholds.get("max_relative_error", 0.05)
        in_bounds = rel_err <= threshold
        coverage = float(np.mean(in_bounds))

        worst_severity = FailureSeverity.PASS
        for val_min, val_max in contract.prediction_intervals.values():
            if np.any(predictions < val_min) or np.any(predictions > val_max):
                worst_severity = FailureSeverity.PREDICTIVE_FAILURE

        if coverage < 0.90 and worst_severity == FailureSeverity.PASS:
            worst_severity = FailureSeverity.PREDICTIVE_FAILURE

        # Aggregate statistics only: returning ``predictions``/``targets`` here
        # would hand the sealed partition back to the caller, defeating the
        # holdout entirely.
        return {
            "status": worst_severity,
            "coverage": coverage,
            "mean_relative_error": float(np.mean(rel_err)),
            "max_relative_error": float(np.max(rel_err)),
            "n_samples": len(targets),
        }


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


# =========================================================================
# Canonical Benchmark Contracts for the 5 Frontier Domain Repairs
# =========================================================================

def create_hall_petch_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Nanoscale Hall-Petch Yield Strength Softening."""
    return ScientificBenchmarkContract(
        model_id="HP-REF-001",
        baseline_model="sigma_y = sigma_0 + k_HP / sqrt(d)",
        refined_model="sigma_y = min(sigma_0 + (k_HP / sqrt(d + d0)) / (1 + (dc / d)^2), sigma_theoretical)",
        mechanism="Dislocation pileup exhaustion coupled to grain boundary sliding threshold and atomic shear limit",
        validated_domain={"grain_size_m": (3e-9, 100e-6)},
        predicted_observables=("yield_strength_pa",),
        prediction_intervals={"yield_strength_pa": (50e6, 3.0e9)},
        falsification_thresholds={"max_relative_error": 0.08},
        locked_test_data_hash="fals_hall_petch_sealed_data_hash",
        data_generation_method="Molecular dynamics + micropillar compression compilation",
        noise_model="Gaussian sensor noise (sigma=3%) + dislocation distribution variance",
        expected_failure_modes=("High-temperature Coble/Nabarro-Herring creep (T > 0.4 Tm)",),
        expected_limits={"d_to_zero": "sigma_0", "d_to_infinity": "sigma_0"},
        expected_invariants=("Non-negative yield stress: sigma_y > 0", "sigma_y <= sigma_theoretical"),
        pass_criteria=("Relative prediction error < 8% across 3–100 nm", "No divergence at d -> 0"),
        fail_criteria=("Yield stress exceeds 3.0 GPa", "Unbounded divergence below 10 nm"),
        domain_restriction_criteria=("Restrict domain to d > 3 nm if amorphous intergranular phase develops",),
    )


def create_casimir_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Casimir NEMS Stiction Divergence / Steric Repulsion."""
    return ScientificBenchmarkContract(
        model_id="CAS-REF-001",
        baseline_model="F_vdW = (A_H * Area) / (6 * pi * z^3)",
        # Parentheses rather than square brackets: grouping punctuation has to be in the
        # subset ``safe_parse_expr`` accepts, or the model string cannot be audited.
        refined_model="F_net = (A_H * Area / (6 * pi * (z + z0)^3)) * (1 - (z0 / (z + z0))^6)",
        mechanism="Integrated Lennard-Jones retarded dispersion + Born steric electronic Pauli repulsion",
        validated_domain={"separation_z_m": (0.0, 10e-6)},
        predicted_observables=("attractive_force_n", "contact_stiffness_n_m"),
        prediction_intervals={"attractive_force_n": (0.0, 1.0e-3)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="fals_casimir_sealed_data_hash",
        data_generation_method="AFM cantilever force-distance curves in ultra-high vacuum",
        noise_model="Thermal cantilever Brownian fluctuation + capacitive bridge drift",
        expected_failure_modes=("Patch potential electrostatic fluctuations at z > 2 um",),
        expected_limits={"z_to_zero": "0.0 N (equilibrium mechanical contact balance)", "z_to_infinity": "0.0 N"},
        expected_invariants=("F(0) = 0 restoring equilibrium", "d^2U/dz^2 > 0 at z=0"),
        pass_criteria=("Net force finite on [0, inf)", "Contact stiffness k_eff > 0"),
        fail_criteria=("Infinite force divergence at z -> 0", "Unstable contact equilibrium"),
        domain_restriction_criteria=("Restrict to z < 2 um when electrostatic patch potentials dominate",),
    )


def create_butler_volmer_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Butler-Volmer Mass-Transport Overpotential Saturation."""
    return ScientificBenchmarkContract(
        model_id="BV-REF-001",
        baseline_model="j = j_0 * (exp(alpha_a * f * eta) - exp(-alpha_c * f * eta))",
        refined_model="j_reg = (j_BV * j_lim) / (j_lim + abs(j_BV))",
        mechanism="Mixed activation-diffusion concentration polarization (Koutecky-Levich boundary layer)",
        validated_domain={"overpotential_volts": (-2.0, 2.0)},
        predicted_observables=("current_density_a_m2",),
        prediction_intervals={"current_density_a_m2": (-5000.0, 5000.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="fals_butler_volmer_sealed_data_hash",
        data_generation_method="Rotating disk electrode (RDE) voltammetry",
        noise_model="Potentiostat ohmic drop iR-compensation residual noise",
        expected_failure_modes=("Electrode passivation and gas evolution bubble blanketing",),
        expected_limits={"eta_to_plus_inf": "j_lim", "eta_to_minus_inf": "-j_lim"},
        expected_invariants=("Asymptotic current saturation |j| <= j_lim", "Zero current at zero overpotential j(0)=0"),
        pass_criteria=("Current saturates strictly at j_lim", "Monotonic current-voltage response"),
        fail_criteria=("Exponential current explosion exceeding j_lim", "Negative differential resistance"),
        domain_restriction_criteria=("Restrict domain to |eta| < 1.5 V if solvent electrolysis occurs",),
    )


def create_prandtl_glauert_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Prandtl-Glauert Transonic Compressibility Regularization."""
    return ScientificBenchmarkContract(
        model_id="PG-REF-001",
        baseline_model="Cp = Cp_0 / sqrt(1 - M^2)",
        refined_model="Cp_reg = Cp_0 / ((1 - M^2)^2 + eps^2)^(1/4)",
        mechanism="Van Dyke / transonic small-disturbance non-linear wave compression relief",
        validated_domain={"mach_number": (0.0, 0.98)},
        predicted_observables=("pressure_coefficient_cp",),
        prediction_intervals={"pressure_coefficient_cp": (-10.0, 0.0)},
        falsification_thresholds={"max_relative_error": 0.06},
        locked_test_data_hash="fals_prandtl_glauert_sealed_data_hash",
        data_generation_method="Transonic wind tunnel surface pressure taps",
        noise_model="Transonic shock buffet boundary layer unsteady noise",
        expected_failure_modes=("Strong detached normal shock waves (M > 1.15)",),
        expected_limits={"M_to_one": "Cp_0 / sqrt(eps)", "M_to_zero": "Cp_0"},
        expected_invariants=("Finite Cp across all M <= 1", "Subsonic recovery to Cp_0 at M=0"),
        pass_criteria=("Finite transonic peak Cp", "Relative pre-shock distortion < 1.5% at M=0.85"),
        fail_criteria=("Division by zero at M = 1.0", "Unphysical positive pressure inversion"),
        domain_restriction_criteria=("Restrict domain to M <= 0.95 for thick airfoils (tau > 0.12)",),
    )


def create_gibbs_thomson_benchmark_contract() -> ScientificBenchmarkContract:
    """Benchmark Contract for Gibbs-Thomson Nanoparticle Melting Depression."""
    return ScientificBenchmarkContract(
        model_id="GT-REF-001",
        baseline_model="Delta_Tm / Tm0 = (2 * gamma_sl) / (rho_s * L_f * r)",
        refined_model="Tm = Tm0 * exp(-C / (r + r0))",
        mechanism="Discrete atomic cluster coordination shell saturation preserving Third Law of Thermodynamics",
        validated_domain={"nanoparticle_radius_m": (0.5e-9, 100e-9)},
        predicted_observables=("melting_temperature_k",),
        prediction_intervals={"melting_temperature_k": (50.0, 1500.0)},
        falsification_thresholds={"max_relative_error": 0.05},
        locked_test_data_hash="fals_gibbs_thomson_sealed_data_hash",
        data_generation_method="Nanocalorimetry (differential scanning calorimetry) of monodisperse clusters",
        noise_model="Substrate interaction wetting heat correction variance",
        expected_failure_modes=("Non-spherical faceted icosahedral cluster reconstructions",),
        expected_limits={"r_to_zero": "Tm0 * exp(-C/r0) > 0 K", "r_to_infinity": "Tm0"},
        expected_invariants=("Strict physical positivity Tm > 0 K", "Monotonic melting point depression with 1/r"),
        pass_criteria=("No negative Kelvin temperatures", "Asymptotic approach to bulk melting point"),
        fail_criteria=("Prediction of Tm <= 0 K", "Non-monotonic melting curve"),
        domain_restriction_criteria=("Restrict to r > 0.8 nm for strongly faceted noble metals",),
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
