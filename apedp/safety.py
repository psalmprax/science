"""Safe formula parsing and bounded symbolic limits, for the ``apedp`` package.

This module re-exports :mod:`apedp_dyn.safety`. Until 27 November 2026 it was a second copy of that file. The two had
drifted, and a weakness an outside audit found that day (a formula written as an f-string with look-alike letters
reached Python's builtins on Python 3.11 and earlier; ``.5`` and ``Piecewise`` were refused; an f-string or a
comprehension ended in an ``AttributeError`` instead of the ``ValueError`` callers handle) was in both. There is one
parser now, and every string-in entry point of this package uses it through the names below.

See ``apedp_dyn/safety.py`` for the threat model and the checks.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

from apedp_dyn import safety as _shared
from apedp_dyn.safety import (
    _SAFE_FUNCS,
    MAX_FORMULA_LENGTH,
    MAX_NAME_LENGTH,
    MAX_NUMERIC_EXPONENT,
    MAX_NUMERIC_POWER_BITS,
    MAX_SPECIAL_FUNCTION_ARGUMENT,
    SAFE_FUNCTIONS,
    FormulaResourceError,
    LimitTimeout,
    limit_with_timeout,
    plain_names,
    safe_parse_expr,
    safe_parse_pde,
    sp,
)


@contextlib.contextmanager
def quiet_float_warnings() -> Iterator[None]:
    """:func:`apedp_dyn.safety.quiet_float_warnings`, with its type stated here.

    This package's type check does not follow imports into ``apedp_dyn``, so the shared function imported by name
    is untyped to it, and every function of this package it decorates would become untyped as well. The body is the
    shared one.
    """
    with _shared.quiet_float_warnings():
        yield


__all__ = [
    "MAX_FORMULA_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_NUMERIC_EXPONENT",
    "MAX_NUMERIC_POWER_BITS",
    "MAX_SPECIAL_FUNCTION_ARGUMENT",
    "SAFE_FUNCTIONS",
    "_SAFE_FUNCS",
    "FormulaResourceError",
    "LimitTimeout",
    "limit_with_timeout",
    "plain_names",
    "quiet_float_warnings",
    "safe_parse_expr",
    "safe_parse_pde",
    "sp",
]
