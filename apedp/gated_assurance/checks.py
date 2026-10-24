"""Constraint, asymptote, stability and generality probes shared by the engines."""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.epistemic import ModelEvidenceRecord
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
    residuals = [abs(value - declared_value) for value, declared_value in zip(measured, declared_series, strict=False)]
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
