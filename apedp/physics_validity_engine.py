# Physics Validity Engine (PVE) for APEDP
# Separates physics validity from trust score
# Implements layered assessment with hard requirements

"""
Physics Validity Engine (PVE)

The PVE computes a physics validity score (S_physics) that is strictly SEPARATE
from the trust score (S_trust). Layered assessment:

1. Hard requirements (decisive when they can be evaluated)
2. Weighted average over the components that *could* be evaluated
3. Category classification

Honesty rule, matching ``apedp/evidence.py``: an unevaluable check is not a
passed check. A requirement that cannot be evaluated (no declared units, an
unparseable expression, no empirical evidence field) is recorded as
``UNEVALUATED`` and its weight is left *unearned* -- the score is a fraction of
the full weighted scale, so skipping checks lowers the score instead of earning
free credit for them. ``breakdown["_unevaluated"]`` lists what was skipped and
``breakdown["_coverage"]`` reports the fraction of total weight that could
evaluated; a HIGH category therefore requires that the dimensional requirement
was actually checked, not merely absent.

The dimensional requirement delegates to ``apedp.core.DimensionalAssuranceEngine``
(pint-backed), so the project has one dimensional engine rather than a second,
weaker presence-only check here.
"""

import logging

import sympy as sp

from apedp.identity_detection import IDENTITY_PATTERNS, detect_identity, is_identity_safe

logger = logging.getLogger(__name__)

# PVE Component Weights. ``unit`` and ``dimensional`` are the two halves of the
# single dimensional requirement: "every symbol carried a declared unit" and
# "the declared units are homogeneous". They are evaluated together, so in
# practice they are both present or both absent.
PVE_WEIGHTS = {
    "dimensional": 0.20,
    "unit": 0.15,
    "known_law": 0.25,
    "limiting": 0.20,
    "empirical": 0.15,
    "derivation": 0.05,
}

#: Weighted-average thresholds for the category label.
_HIGH_THRESHOLD = 0.8
_SECONDARY_THRESHOLD = 0.5


def _probe_limiting(expr_str: str) -> float | None:
    """Probe limiting behaviour on a small grid.

    Returns ``1.0`` when finite everywhere on the probe grid, ``0.0`` when a
    non-finite value appears, and ``None`` when the expression cannot be probed
    at all -- ``None`` means "not evaluated", which the caller must not treat as
    a pass.
    """
    try:
        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expr_str)
        syms = sorted(expr.free_symbols, key=str)
        if not syms:
            return 1.0
        import itertools
        points = [0.5, 1.0, 2.0]
        for combo in itertools.product(points, repeat=min(len(syms), 3)):
            subs = {s: v for s, v in zip(syms, combo)}
            try:
                val = complex(expr.subs(subs).evalf())
            except (TypeError, ValueError, ArithmeticError) as exc:
                logger.debug("Limiting probe unevaluable at %s for %r: %s", subs, expr_str, exc)
                return None
            if val is None or str(val) in ("nan", "nan+nanj"):
                return 0.0
            import math
            if math.isnan(val.real) or math.isnan(val.imag):
                return 0.0
            if math.isinf(val.real) or math.isinf(val.imag):
                return 0.0
        return 1.0
    except (RuntimeError, TypeError, ValueError, ArithmeticError, KeyError) as exc:
        logger.debug("Limiting probe unavailable for %r: %s", expr_str, exc)
        return None


def _score_empirical(audit) -> float | None:
    """Score declared empirical evidence, or ``None`` when none is declared.

    ``None`` (no evidence field present) is deliberately distinct from ``0.0``
    (declared unsupported): the first is "not evaluated", the second is a
    measured negative. Collapsing the two into a 0.5 default made an absent
    field indistinguishable from a real mid-strength result.
    """
    if "empirical_error" in audit:
        try:
            return max(0.0, min(1.0, 1.0 - float(audit["empirical_error"])))
        except (TypeError, ValueError) as exc:
            logger.warning("empirical_error unreadable (%s); treating as unevaluated", exc)
            return None
    if "coverage" in audit:
        try:
            return max(0.0, min(1.0, float(audit["coverage"])))
        except (TypeError, ValueError) as exc:
            logger.warning("coverage unreadable (%s); treating as unevaluated", exc)
            return None
    if audit.get("empirically_supported") is True:
        return 1.0
    if audit.get("empirically_supported") is False:
        return 0.0
    return None


def _dimensional_check(audit) -> tuple[str, str | None, bool]:
    """Dimensional requirement: ``(verdict, reason, all_symbols_have_units)``.

    ``verdict`` is ``"PASS"``, ``"FAIL"`` (a real inconsistency), or
    ``"UNEVALUATED"`` (not enough information -- missing units, unparseable
    expression, or an engine error). Only ``"PASS"`` earns the dimensional
    weight; ``"FAIL"`` is decisive; ``"UNEVALUATED"`` earns nothing.
    """
    expr_str = audit.get("expression") or audit.get("equation") or ""
    if not expr_str:
        return "UNEVALUATED", "no expression to check", False
    try:
        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expr_str)
    except (ImportError, TypeError, ValueError) as exc:
        return "UNEVALUATED", f"unparseable expression: {exc}", False

    declared = audit.get("declared_units") or {}
    symbols = sorted(str(s) for s in expr.free_symbols)
    missing = [name for name in symbols if not declared.get(name)]
    if missing:
        return "UNEVALUATED", f"no declared unit for {', '.join(missing)}", False
    if not symbols:
        # A constant is dimensionally homogeneous by construction.
        return "PASS", None, True

    try:
        from apedp.core import DimensionalAssuranceEngine
        engine = DimensionalAssuranceEngine()
        for name in symbols:
            engine.register_variable(name, declared[name])
        engine.check_expression_consistency(expr)
        return "PASS", None, True
    except TypeError as exc:
        # Incompatible dimensions in a sum/product/power: a genuine failure.
        return "FAIL", str(exc), True
    except (ValueError, KeyError, AttributeError, ArithmeticError,
            NotImplementedError) as exc:
        logger.debug("Dimensional engine could not evaluate %r: %s", expr_str, exc)
        return "UNEVALUATED", f"engine error: {exc}", True


def check_hard_requirements(audit, standard_vars, known_laws) -> tuple:
    """Evaluate the hard requirements, then the weighted validity score.

    A requirement that can be evaluated is decisive: a dimensional
    inconsistency or a mathematical identity fails immediately with
    ``S_physics = 0``. A requirement that cannot be evaluated is recorded in
    ``errors`` as ``*_UNEVALUATED`` and its weight is dropped from the average
    instead of being awarded a full score. If nothing at all can be evaluated
    the score is ``0.0`` and ``NO_EVALUABLE_COMPONENTS`` is reported.

    Returns:
        ``(S_physics, breakdown, category, errors)``. ``breakdown`` maps each
        component to its score or ``None`` when unevaluated, and additionally
        carries ``_unevaluated`` (sorted component names) and ``_coverage``
        (fraction of total weight that could be evaluated).
    """
    errors: list[str] = []
    dim_verdict, dim_reason, units_declared = _dimensional_check(audit)
    if dim_verdict == "FAIL":
        errors.append(f"DIMENSIONAL_INCONSISTENCY: {dim_reason}")
        return 0.0, {}, "LOW", errors
    if dim_verdict == "UNEVALUATED":
        errors.append(f"DIMENSIONAL_UNEVALUATED: {dim_reason}")

    # Hard requirement: a mathematical identity is not a physical discovery.
    expr_str = audit.get("expression", "")
    if not expr_str:
        errors.append("IDENTITY_UNEVALUATED: no expression to check")
    else:
        try:
            from apedp.gated_assurance import safe_parse_expr
            is_id, _, _ = detect_identity(safe_parse_expr(expr_str))
        except (ImportError, TypeError, ValueError, ArithmeticError) as exc:
            errors.append(f"IDENTITY_UNEVALUATED: {exc!s}")
        else:
            if is_id:
                errors.append("MATHEMATICAL_IDENTITY")
                return 0.0, {}, "LOW", errors

    known_law_match = bool(audit.get("known_law_match", False))
    derivation_valid = audit.get("derivation_valid")
    components: dict[str, float | None] = {
        "unit": 1.0 if units_declared else None,
        "dimensional": 1.0 if dim_verdict == "PASS" else None,
        "known_law": 1.0 if known_law_match else 0.5,
        "limiting": _probe_limiting(expr_str) if expr_str else None,
        "empirical": _score_empirical(audit),
        "derivation": float(derivation_valid) if derivation_valid is not None else None,
    }

    evaluated = {k: v for k, v in components.items() if v is not None}
    total_weight = sum(PVE_WEIGHTS.values())
    breakdown: dict = dict(components)
    breakdown["_unevaluated"] = sorted(k for k, v in components.items() if v is None)
    breakdown["_coverage"] = round(sum(PVE_WEIGHTS[k] for k in evaluated) / total_weight, 4)
    if not evaluated:
        errors.append("NO_EVALUABLE_COMPONENTS")
        return 0.0, breakdown, "LOW", errors

    # Unevaluated components earn nothing: their weight stays in the denominator
    # as unearned, rather than being renormalised away (which would let a
    # candidate score 1.0 off the one or two checks that happened to run).
    S_physics = sum(PVE_WEIGHTS[k] * v for k, v in evaluated.items()) / total_weight

    if S_physics >= _HIGH_THRESHOLD:
        category = "HIGH"
    elif S_physics >= _SECONDARY_THRESHOLD:
        category = "SECONDARY"
    else:
        category = "LOW"

    return S_physics, breakdown, category, errors


# NOTE (2026-09-23): the unreferenced "benchmark calibration framework"
# (``BENCHMARK_DATASET`` / ``run_benchmark`` / ``calibrate_thresholds_from_benchmarks``)
# and the dead ``compute_physics_validity*`` / ``conservation_checklist`` wrappers
# were removed. The former printed authoritative-looking ROC AUC and "optimal
# threshold" numbers while deriving those thresholds by percentile over the very
# scores its own labels defined, so it validated itself. See
# docs/INDEPENDENT_REVIEW_2026-09-23.md §4.1.


def canonical_form(expression: str, extra_symbols: dict | None = None) -> str:
    """Normalize equation rearrangements to a canonical residual string.

    F=ma, a=F/m, ma-F=0 all map to the same residual (modulo sympy simplification).
    Returns '' when unparseable. extra_symbols is forwarded to safe_parse_expr
    (e.g. {"I": Symbol("I")} so current is not read as imaginary).
    """
    try:
        from apedp.gated_assurance import safe_parse_expr
    except ImportError as exc:
        logger.debug("canonical_form: parser import failed (%s); returning empty form", exc)
        return ""
    try:
        def parse(s: str) -> sp.Expr:
            if extra_symbols:
                return safe_parse_expr(s, extra_symbols)
            return safe_parse_expr(s)
        if "=" in expression:
            lhs, rhs = expression.split("=", 1)
            residual = parse(lhs) - parse(rhs)
        else:
            residual = parse(expression)
        return str(sp.simplify(residual))
    except (TypeError, ValueError, ArithmeticError) as exc:
        logger.debug("canonical_form: %r unparseable (%s); returning empty form", expression, exc)
        return ""


def same_relationship(expr_a: str, expr_b: str, extra_symbols: dict | None = None) -> bool:
    """True when two strings are the same relationship up to rearrangement.

    Compares cleared-numerator residuals up to a nonzero constant factor, so
    F=ma, a=F/m, and ma-F=0 all match. Returns False when unparseable.
    """
    ca, cb = canonical_form(expr_a, extra_symbols), canonical_form(expr_b, extra_symbols)
    if not ca or not cb:
        return False
    try:
        from apedp.gated_assurance import safe_parse_expr

        def parse(s: str) -> sp.Expr:
            if extra_symbols:
                return safe_parse_expr(s, extra_symbols)
            return safe_parse_expr(s)
        ra = parse(ca)
        rb = parse(cb)
        if bool(sp.simplify(ra - rb) == 0):
            return True
        na = sp.together(ra).as_numer_denom()[0]
        nb = sp.together(rb).as_numer_denom()[0]
        if bool(na == 0) or bool(nb == 0):
            return bool(sp.simplify(na - nb) == 0)
        ratio = sp.simplify(na / nb)
        return bool(ratio.is_Number and ratio != 0)
    except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
        logger.debug(
            "same_relationship: residual comparison failed (%s); exact string match: %s",
            exc, ca == cb,
        )
        return ca == cb


__all__ = [
    "IDENTITY_PATTERNS",
    "PVE_WEIGHTS",
    "canonical_form",
    "check_hard_requirements",
    "detect_identity",
    "is_identity_safe",
    "same_relationship",
]
