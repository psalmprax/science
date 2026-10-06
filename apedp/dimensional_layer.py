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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

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


def _carries_explicit_coefficient(term: sp.Expr) -> bool:
    """True when the term has a numeric factor that may carry units.

    ``10.0*L**2`` -> True (the 10.0 can be a joule/meter**2); ``L**2`` ->
    False (its dimension is fixed by the declared variable units); ``-0.5``
    -> True (a bare constant is synthesizable by construction).

    The sign is not a dimensional coefficient: sympy factors ``-M**3`` as
    coeff ``-1``, so the magnitude is compared, otherwise a difference of
    bare terms would look synthesizable and slip through as admissible.
    """
    coeff, rest = term.as_coeff_Mul()
    return bool(abs(coeff) != 1) or not rest.free_symbols


def _term_dimension(term: sp.Expr, engine: DimensionalAssuranceEngine) -> str:
    return _dimensionality_name(engine.check_expression_consistency(term))


def audit_dimensional_admissibility(
    expression: sp.Expr | str,
    variable_units: Mapping[str, str],
    target_unit: str | None = None,
) -> dict[str, Any]:
    """Audit an expression under *dimensional constant synthesis*.

    Answers a different question from :func:`check_units`. ``check_units``
    asks "does this expression already have the target dimension?". This asks
    "is this expression structurally admissible, given that its numeric
    coefficients may carry units?" -- the question a discovery pipeline must
    ask before fitting, since the fitted constants are exactly what would
    supply those units.

    The synthesis rule, stated rather than assumed:

    * A term with an **explicit numeric coefficient** has a synthesizable
      dimension: the coefficient is fitted, so it may carry whatever units
      the term requires. ``10.0*L**2 + 5.0*T**2`` is therefore admissible
      (both terms become joules) while the coefficient-free ``L**2 + T**2``
      is not.
    * A **bare** term -- variables and powers only, no numeric factor -- has
      a dimension fixed by the declared variable units. Two bare terms in one
      sum must agree; if they do not, no coefficient can repair the sum, so
      the form is rejected. ``L + v*T`` is admissible because both sides are
      lengths; ``M + L`` is not.
    * A **bare dimensioned transcendental argument** (``sin(L)``,
      ``exp(L/T)``) is unrepairable. A parameterized one is not: in
      ``exp(-0.8*r)`` the 0.8 can carry inverse length, making the argument
      dimensionless. ``sin(v*T/L)`` needs no coefficient at all -- its
      argument is already dimensionless.
    * A mismatch between the expression dimension and ``target_unit`` is
      **repairable** by one overall multiplicative dimensional constant, so it
      does not make a form inadmissible. ``a**1.5`` is the canonical case:
      Kepler's ``T ~ a**1.5`` holds only because the constant carries units.

    Returns a dict with ``admissible`` (bool), ``status`` (``"VALID"`` /
    ``"INVALID"``), ``reason``, ``result_dimensionality`` and
    ``requires_synthesis_constant``.
    """
    try:
        expr = safe_parse_expr(_normalize_cbrt(expression)) if isinstance(expression, str) else expression
    except (TypeError, ValueError) as exc:
        return {"admissible": False, "status": "INVALID",
                "reason": f"unparseable expression: {exc}",
                "result_dimensionality": None, "requires_synthesis_constant": False}
    if expr is None:
        return {"admissible": False, "status": "INVALID", "reason": "missing expression",
                "result_dimensionality": None, "requires_synthesis_constant": False}

    engine = DimensionalAssuranceEngine()
    symbols = sorted(str(symbol) for symbol in expr.free_symbols)
    missing = [name for name in symbols if name not in variable_units]

    def _invalid(reason: str) -> dict[str, Any]:
        # Unassessable earns nothing: an expression we cannot audit is not
        # admissible, matching check_units' UNEVALUATED-is-not-PASS rule.
        return {"admissible": False, "status": "INVALID", "reason": reason,
                "result_dimensionality": None, "requires_synthesis_constant": False}

    if missing:
        return _invalid(f"no declared unit for {', '.join(missing)}")

    for name in symbols:
        try:
            engine.register_variable(name, variable_units[name])
        except (KeyError, ValueError, TypeError) as exc:
            return _invalid(f"cannot register unit for {name}: {exc}")

    # 1. Transcendentals. A bare dimensioned argument is unrepairable; an
    #    argument carrying an explicit numeric coefficient can be made
    #    dimensionless by that coefficient, and an already-dimensionless
    #    argument is fine as-is.
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, sp.Function) or not node.args:
            continue
        name = getattr(type(node), "__name__", "")
        if name not in _TRANSCENDENTAL_NAMES:
            continue
        argument = node.args[0]
        if _carries_explicit_coefficient(argument):
            continue
        try:
            argument_dim = _term_dimension(argument, engine)
        except (TypeError, ValueError, KeyError, AttributeError,
                ArithmeticError, NotImplementedError):
            continue  # cannot judge; benefit of the doubt
        if argument_dim != "dimensionless":
            return _invalid(
                f"Transcendental {name} applied to bare dimensioned "
                f"expression ({argument} has dimension {argument_dim}); "
                "no coefficient is present to supply inverse units"
            )

    # 2. Dimensioned exponents are unrepairable: no coefficient fixes x**t.
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and not node.exp.free_symbols:
            try:
                exp_dim = _term_dimension(node.exp, engine)
            except (TypeError, ValueError, KeyError, AttributeError,
                    ArithmeticError, NotImplementedError):
                continue
            if exp_dim != "dimensionless":
                return _invalid(
                    f"exponent must be dimensionless, got {exp_dim}; "
                    "no coefficient can repair a dimensioned exponent"
                )

    # 3. Additive terms. Coefficient-free terms have fixed dimensions and must
    #    agree; coefficient-bearing terms are free to carry the required units.
    terms = sp.Add.make_args(sp.expand(expr)) or (expr,)
    fixed_dims: dict[str, str] = {}
    has_synthesizable_term = False
    for term in terms:
        if _carries_explicit_coefficient(term):
            has_synthesizable_term = True
            continue
        try:
            dim = _term_dimension(term, engine)
        except TypeError as exc:
            return _invalid(f"Dimensional Error: {exc}")
        except (ValueError, KeyError, AttributeError, ArithmeticError,
                NotImplementedError) as exc:
            return _invalid(f"engine limit: {exc}")
        if fixed_dims and dim not in fixed_dims:
            return _invalid(
                "Addition of incompatible bare variable terms: "
                f"{sorted(fixed_dims)} and [{dim}] cannot be summed; "
                "no coefficient can repair a sum of fixed dimensions"
            )
        fixed_dims[dim] = str(term)

    result_dim = next(iter(fixed_dims)) if fixed_dims else None
    if result_dim is None:
        # Every term is synthesizable, so the expression has no fixed
        # dimension of its own -- the target is reached by the coefficients.
        result_dim = "synthesizable"
        requires = True
    else:
        requires = has_synthesizable_term or (
            target_unit is not None and result_dim != target_unit
        )
    return {
        "admissible": True,
        "status": "VALID",
        "reason": None,
        "result_dimensionality": result_dim,
        "requires_synthesis_constant": bool(requires),
    }


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
