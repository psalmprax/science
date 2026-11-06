"""Format a coefficient onto an operand without splitting a non-product operand.

Status: production

A term is emitted as ``coefficient*operand``. That text is a faithful statement
of the term only while ``operand`` is a single syntactic unit; for a sum it says
something else entirely. ``2*(a + b)`` written as ``2*a + b`` gives the second
addend a coefficient of 1, and nobody reading the string can tell that the
emitter meant otherwise.

The repo has two ways to get this right, and both are in use:

* build the expression as a sympy object (``sp.Mul(coeff, part)``), which is
  structurally immune — what ``apedp/discovery/_core.py`` does;
* or build the operand as a string, in which case the operand must be
  parenthesised whenever it is not already atomic.

This module is the second option, so that a caller-supplied name — a library
term, a DataFrame column, a fitted feature label — cannot smuggle a sum into a
coefficient slot. It is deliberately dependency-free and returns the operand
untouched when it is already atomic, so existing emitted strings do not churn
(and candidate ids hashed from those strings do not move).
"""
from __future__ import annotations

_OPENERS = frozenset("([")


def is_atomic(operand: str) -> bool:
    """True when ``operand`` needs no parentheses to multiply by a coefficient.

    Atomic here means: no binary ``+`` or ``-`` at bracket depth zero, and no
    leading sign. A product, a quotient, a power, a call, or an
    already-parenthesised sum is atomic; a sum or a signed operand is not.
    """
    text = operand.strip()
    if not text:
        return False
    if text[0] in "+-":
        # `2*-x` parses, but the operand is a signed expression rather than a
        # unit; wrapping it keeps the emitted text unambiguous.
        return False
    depth = 0
    for char in text:
        if char in _OPENERS:
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                # Unbalanced input is not something this can vouch for; the
                # caller's own parser should refuse it, not this formatter.
                return False
        elif depth == 0 and char in "+-":
            return False
    return True


def atom(operand: str) -> str:
    """``operand`` as one syntactic unit, parenthesised only when it is not."""
    return operand if is_atomic(operand) else f"({operand})"


def scaled(coefficient: str, operand: str) -> str:
    """``<coefficient>*<operand>`` with the operand bound as a unit."""
    return f"{coefficient}*{atom(operand)}"


def powered(operand: str, exponent: str) -> str:
    """``<operand>**<exponent>`` with the base bound as a unit.

    The sibling of :func:`scaled`. ``**`` binds tighter than ``+``, so
    ``a + b**2`` raises only ``b``: a name that may contain a sum has to be
    wrapped before the power is applied to it.
    """
    return f"{atom(operand)}**{exponent}"


def over(numerator: str, denominator: str) -> str:
    """``<numerator>/<denominator>`` with both sides bound as units.

    Division binds tighter than addition too, so ``1/a + b`` puts ``b`` outside
    the reciprocal entirely. Both sides are wrapped, because either may be a
    name the caller supplied.
    """
    return f"{atom(numerator)}/{atom(denominator)}"


__all__ = ["atom", "is_atomic", "over", "powered", "scaled"]
