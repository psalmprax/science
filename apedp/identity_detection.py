# Identity Detection for APEDP
# Automatically catches mathematical identities that should not be discoveries

import logging

import sympy as sp

logger = logging.getLogger(__name__)

# Known identity patterns and their canonical forms
# These check after sympy simplification where applicable
IDENTITY_PATTERNS = {
    "x_over_x": {
        "pattern": lambda expr: bool(expr.simplify() == sp.Integer(1)),
        "canonical": sp.Integer(1),
        "description": "x/x = 1",
    },
    "x_minus_x": {
        "pattern": lambda expr: bool(expr.simplify() == sp.Integer(0)),
        "canonical": sp.Integer(0),
        "description": "x - x = 0",
    },
    "x_pow_zero": {
        "pattern": lambda expr: bool(expr.is_Pow and expr.exp == sp.Integer(0)),
        "canonical": sp.Integer(1),
        "description": "x^0 = 1",
    },
    # NOTE: no generic 2-term-Add pattern here. a+(-a) auto-simplifies to 0
    # at construction, so any len(args)==2 check is a false-positive factory
    # (it once flagged every binomial residual as an identity).
}


def detect_identity(expr: sp.Expr) -> tuple:
    """Detect if an expression is a mathematical identity.

    Returns:
        (is_identity, canonical_form, description)

    Examples:
        >>> detect_identity(sp.Symbol('x') / sp.Symbol('x'))
        (True, sp.Integer(1), "x/x = 1")
        >>> detect_identity(sp.Symbol('x') - sp.Symbol('x'))
        (True, sp.Integer(0), "x - x = 0")
        >>> detect_identity(sp.Symbol('x') ** sp.Integer(0))
        (True, sp.Integer(1), "x^0 = 1")
        >>> detect_identity(2 * sp.Symbol('x'))
        (False, None, None)
    """
    # First try simplification-based check
    try:
        simplified = expr.simplify()
        if simplified.is_Number:
            if simplified == sp.Integer(1):
                return True, sp.Integer(1), "simplifies to 1"
            elif simplified == sp.Integer(0):
                return True, sp.Integer(0), "simplifies to 0"
    except (TypeError, ValueError, ArithmeticError, AttributeError) as exc:
        logger.debug("Simplification failed for %s; falling back to patterns: %s", expr, exc)

    # Then try pattern matching
    for pattern_info in IDENTITY_PATTERNS.values():
        if pattern_info["pattern"](expr):
            return True, pattern_info["canonical"], pattern_info["description"]

    return False, None, None


def is_identity_safe(expr: sp.Expr) -> bool:
    """Check if expression is a safe identity.

    Returns True if the expression should be classified as IDENTITY
    and not treated as a physical discovery.
    """
    is_id, _, _ = detect_identity(expr)
    return bool(is_id)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
