# Databricks notebook source
"""Standalone dimensional admissibility layer enforced during generation.

Unlike post-hoc audits, :func:`filter_candidates` rejects dimensionally
invalid forms *before* any fitting happens, so a numerically competitive but
physically inadmissible expression (e.g. ``sqrt(T)`` for a temperature-like
target, or ``dx**2 + dy**2`` for a length target) can never win on error
alone. :func:`check_units` additionally verifies that the result dimension
matches the declared target dimension -- internal consistency is necessary
but not sufficient.

Conventions (documented, not hidden):
  * Fitted numeric coefficients are plain ``Float`` nodes and therefore
    dimensionless; the coefficient is understood to carry whatever units the
    form requires. Rejection applies to *structural* unit errors, never to
    fitted scales.
  * Forms containing undeclared symbols (typically free fit coefficients)
    are ``UNEVALUATED`` -- never blocked -- with one exception: a
    transcendental applied to an argument with declared non-dimensionless
    units and no undeclared symbols (``sin(T)`` hiding among coefficients)
    is ``FAIL``, because no coefficient value can repair it. An undeclared
    symbol inside the argument (``sin(p*T)``) preserves the benefit of the
    doubt, since ``p`` may carry inverse units.
  * Statuses mirror ``physics_validity_engine``: ``PASS`` / ``FAIL`` /
    ``UNEVALUATED`` (missing units, unparseable expression, engine limit).
    Only ``PASS`` admits; ``FAIL`` is decisive; ``UNEVALUATED`` earns nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.discovery_admission import AdmissionDecision, admit_candidate
from apedp.safety import safe_parse_expr


@dataclass(frozen=True)
class UnitVerdict:
    """Outcome of checking one expression against declared units."""

    status: str
    reason: str | None
    result_dimensionality: str | None
    all_symbols_declared: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "result_dimensionality": self.result_dimensionality,
            "all_symbols_declared": self.all_symbols_declared,
        }


# Transcendentals that require a dimensionless argument. ``sqrt``/``Abs``/
# ``Min``/``Max`` are deliberately excluded: they transform dimensions
# instead of requiring dimensionlessness.
_TRANSCENDENTAL_NAMES = frozenset({
    "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "asin", "acos", "atan", "asinh", "acosh", "atanh",
    "erf", "erfc", "gamma", "zeta",
})


def _dimensionality_name(unit: Any) -> str:
    try:
        return str(unit.dimensionality)
    except (AttributeError, TypeError, ValueError):
        return str(unit)


def _transcendental_violation(
    expression: sp.Expr,
    declared_dimensionality: dict[str, str | None],
) -> str | None:
    """Catch dimensioned transcendental arguments the engine never reaches.

    When free (undeclared) symbols exist, the full-engine check bails out
    as UNEVALUATED -- which previously let ``sin(T)``-type terms hide behind
    neighboring fit coefficients. A transcendental whose argument contains a
    declared non-dimensionless symbol and *no* undeclared symbol is a genuine
    violation regardless of any coefficients elsewhere in the expression:
    no coefficient value can make ``sin(kelvin)`` meaningful.
    """
    dimensionless = "dimensionless"
    for node in sp.preorder_traversal(expression):
        if not isinstance(node, sp.Function):
            continue
        name = getattr(type(node), "__name__", "")
        if name not in _TRANSCENDENTAL_NAMES or not node.args:
            continue
        argument = node.args[0]
        free = {str(symbol) for symbol in argument.free_symbols}
        if any(name not in declared_dimensionality for name in free):
            continue  # an undeclared symbol may carry inverse units: cannot judge
        dims = {declared_dimensionality[name] for name in free if declared_dimensionality.get(name)}
        if any(dim is not None and dim != dimensionless for dim in dims):
            return (
                f"transcendental {name} applied to dimensioned argument "
                f"({node.args[0]}); no coefficient can repair this"
            )
        if not free:
            # Fully numeric argument: still must be dimensionless in form.
            # A bare number is dimensionless, so this passes by construction.
            continue
    return None


def _declared_dimensionalities(
    expression: sp.Expr,
    variable_units: Mapping[str, str],
    engine: DimensionalAssuranceEngine,
) -> dict[str, str | None]:
    dimensionalities: dict[str, str | None] = {}
    for symbol in expression.free_symbols:
        name = str(symbol)
        if name not in variable_units:
            continue
        try:
            dimensionalities[name] = _dimensionality_name(engine.registry(str(variable_units[name])))
        except (ValueError, TypeError, AttributeError):
            dimensionalities[name] = None
    return dimensionalities


def _normalize_cbrt(expression: str) -> str:
    """Rewrite ``cbrt(arg)`` as ``(arg)**(1/3)`` with paren matching.

    ``safe_parse_expr`` has no ``cbrt`` token; the rewrite is unit-exact and
    stays inside this layer (the original string still flows to admission).
    """
    out: list[str] = []
    index = 0
    while index < len(expression):
        if expression.startswith("cbrt(", index):
            depth = 1
            cursor = index + len("cbrt(")
            while cursor < len(expression) and depth:
                if expression[cursor] == "(":
                    depth += 1
                elif expression[cursor] == ")":
                    depth -= 1
                cursor += 1
            if depth:
                raise ValueError("unbalanced cbrt() parentheses")
            inner = expression[index + len("cbrt(") : cursor - 1]
            out.append(f"({inner})**(1/3)")
            index = cursor
        else:
            out.append(expression[index])
            index += 1
    return "".join(out)


def check_units(
    expression: sp.Expr | str,
    variable_units: Mapping[str, str],
    target_unit: str | None = None,
) -> UnitVerdict:
    """Check structural unit validity and, when given, target-dimension match."""
    try:
        expr = safe_parse_expr(_normalize_cbrt(expression)) if isinstance(expression, str) else expression
    except (TypeError, ValueError) as exc:
        return UnitVerdict("UNEVALUATED", f"unparseable expression: {exc}", None, False)
    if expr is None:
        return UnitVerdict("UNEVALUATED", "missing expression", None, False)
    symbols = sorted(str(symbol) for symbol in expr.free_symbols)
    missing = [name for name in symbols if name not in variable_units]
    engine = DimensionalAssuranceEngine()
    violation = _transcendental_violation(expr, _declared_dimensionalities(expr, variable_units, engine))
    if violation is not None:
        # Decisive even with undeclared symbols present: no coefficient can
        # repair a transcendental of a dimensioned quantity (e.g. a sin(T)
        # term hiding among fit coefficients).
        return UnitVerdict("FAIL", violation, None, not missing)
    if missing:
        return UnitVerdict("UNEVALUATED", f"no declared unit for {', '.join(missing)}", None, False)
    if not symbols:
        # A constant is dimensionally homogeneous by construction.
        return UnitVerdict("PASS", None, "dimensionless", True)
    try:
        for name in symbols:
            engine.register_variable(name, variable_units[name])
        result_unit = engine.check_expression_consistency(expr)
    except TypeError as exc:
        # Genuine structural violation (mixed sums, dimensioned transcendentals).
        return UnitVerdict("FAIL", str(exc), None, True)
    except (ValueError, KeyError, AttributeError, ArithmeticError, NotImplementedError) as exc:
        return UnitVerdict("UNEVALUATED", f"engine limit: {exc}", None, True)
    result_name = _dimensionality_name(result_unit)
    if target_unit is not None:
        try:
            target_dim = _dimensionality_name(engine.registry(str(target_unit)))
        except (ValueError, TypeError, AttributeError) as exc:
            return UnitVerdict("UNEVALUATED", f"bad target unit: {exc}", result_name, True)
        if target_dim != result_name:
            return UnitVerdict(
                "FAIL",
                f"result dimension {result_name} does not match target {target_dim}",
                result_name,
                True,
            )
    return UnitVerdict("PASS", None, result_name, True)


@dataclass(frozen=True)
class DimFilterResult:
    """Generation-time enforcement outcome over a candidate grammar."""

    admissible: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    unevaluated: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "admissible": list(self.admissible),
            "rejected": [{"expression": expr, "reason": reason} for expr, reason in self.rejected],
            "unevaluated": list(self.unevaluated),
        }


def filter_candidates(
    expressions: Iterable[str],
    variable_units: Mapping[str, str],
    target_unit: str | None = None,
) -> DimFilterResult:
    """Split a grammar into admissible / rejected / unevaluated *before* fitting."""
    admissible: list[str] = []
    rejected: list[tuple[str, str]] = []
    unevaluated: list[str] = []
    for expression in expressions:
        verdict = check_units(expression, variable_units, target_unit)
        if verdict.status == "PASS":
            admissible.append(expression)
        elif verdict.status == "FAIL":
            rejected.append((expression, verdict.reason or "dimensional violation"))
        else:
            unevaluated.append(expression)
    return DimFilterResult(tuple(admissible), tuple(rejected), tuple(unevaluated))


def admit_with_units(
    expression: sp.Expr | str | None,
    *,
    variable_units: Mapping[str, str],
    target_unit: str | None = None,
    **admit_kwargs: Any,
) -> AdmissionDecision:
    """Wrapper computing ``dimensionally_consistent`` instead of trusting the caller."""
    verdict = check_units(expression, variable_units, target_unit) if expression is not None else None
    consistent = verdict is not None and verdict.status == "PASS"
    decision = admit_candidate(expression, dimensionally_consistent=consistent, **admit_kwargs)
    return AdmissionDecision(
        decision.status,
        decision.reasons,
        {**decision.evidence, "unit_verdict": verdict.to_dict() if verdict else None},
    )


__all__ = ["DimFilterResult", "UnitVerdict", "admit_with_units", "check_units", "filter_candidates"]
