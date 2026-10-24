"""APEDP Safety: hardened string-in parsing and bounded symbolic limits.

Threat model (string-in paths):
- Attacker controls a formula string via REST API (``/repair/anomalies``,
  ``/metrology/propagate``, ``/discover/*`` template priors), CSV content,
  or epistemic tournament inputs.
- Raw ``sympy.parse_expr`` / ``sympify`` on such input allows attribute
  access (``__class__``, ``__import__``) and dunder traversal, plus
  denial-of-service via pathological ``limit`` / integration calls that
  never return.
- Mitigations here: strict allowlist pre-checks, length caps, no attribute
  access, restricted ``local_dict``, and wall-clock timeouts on ``limit``.

This module is deliberately dependency-free (stdlib + sympy only).
"""
from __future__ import annotations

import contextlib
import logging
import re
import threading
import warnings
from collections.abc import Iterator

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

logger = logging.getLogger(__name__)

MAX_FORMULA_LENGTH = 2000

#: Rejected wherever they appear: dunder access, attribute access on the modules
#: that lead to execution, calls to IO/execution primitives, statement separators
#: and shell/backtick forms. These are unambiguous prefixes of an attack, so a
#: substring test is the right test for them.
@contextlib.contextmanager
def quiet_float_warnings() -> Iterator[None]:
    """Silence numpy's floating-point warnings around a *deliberate* probe.

    The falsifier and the boundary auditor evaluate models **at** their singular
    points on purpose (x -> 0, M -> 1, v -> c, eps -> 0). numpy reports every
    resulting divide-by-zero / invalid / overflow through the warnings machinery,
    while the caller turns the non-finite value into a finding, so at those call
    sites the warning is pure duplication -- and roughly 125 of them per test run
    made a *real* numerical warning impossible to spot.

    Narrowing the silence to the probe is the point: it keeps the noise out of the
    suite without hiding numerical problems in an ordinary fit elsewhere.
    Usable both as a context manager and as a decorator.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


_FORBIDDEN_SUBSTRINGS = (
    "__",
    "os.",
    "sys.",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "class ",
    "def ",
    ";",
    "\n",
    "\r",
    "`",
)

#: Rejected only as whole words. Matching these as substrings rejected legitimate
#: engineering names -- ``importance``, ``getattr_index``, ``hasattr_flag`` were
#: refused with "Forbidden token", which is a false positive on real work rather
#: than a defence. A word-boundary test keeps the defence and drops the noise; the
#: executed-expression protections are the symbol allowlist below plus the dunder
#: and call forms above, not this list.
_FORBIDDEN_IDENTIFIERS = (
    "import",
    "getattr",
    "setattr",
    "hasattr",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "lambda",
    "class",
    "def",
)

_SAFE_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "sign": sp.sign,
    # Lowercase aliases, for the same reason every other function here has one:
    # ``min(...)`` is the natural way to write a bounded physical law (the canonical
    # benchmark contracts all use it), and without the alias the parser resolves
    # ``min`` to a plain Symbol, so ``min(a, b)`` fails with "'Symbol' object is not
    # callable". The allowlist below governs what survives parsing either way, so
    # this closes an ergonomics gap rather than opening a surface.
    "min": sp.Min,
    "max": sp.Max,
    "Min": sp.Min,
    "Max": sp.Max,
    # ``ln`` is the engineering spelling of the natural log and shows up in
    # hand-written priors and in laws the search returns. Without the alias the
    # parser resolves it to a plain Symbol, so ``ln(x)`` fails with "'Symbol'
    # object is not callable" -- the same ergonomics gap as min/max above, and
    # closed the same way: the allowlist bounds what survives parsing, so adding
    # a name here widens the accepted vocabulary, not the executable surface.
    "ln": sp.log,
    "Heaviside": sp.Heaviside,
    "re": sp.re,
    "im": sp.im,
    "conjugate": sp.conjugate,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "zeta": sp.zeta,
    "besselj": sp.besselj,
    "bessely": sp.bessely,
    "besseli": sp.besseli,
    "besselk": sp.besselk,
    "Piecewise": sp.Piecewise,
    "piecewise": sp.Piecewise,
    "pi": sp.pi,
    "E": sp.E,
    "EulerGamma": sp.EulerGamma,
}

_TRANSFORMATIONS = (*standard_transformations, convert_xor)


def _rejects_attribute_access(s: str) -> bool:
    """Return True if the string contains attribute access beyond float literals.

    Floats like ``3.14`` contain a dot between digits and are allowed.
    Anything else with ``.`` (e.g. ``x.__class__``) is rejected.
    """
    # Remove digit.digit float patterns, then any remaining dot is suspicious.
    without_floats = re.sub(r"\d\.\d", "", s)
    return "." in without_floats


#: Names the parser itself needs to resolve; never shadowed with Symbols.
_PARSER_INTERNALS = frozenset({"Integer", "Float", "Rational", "Symbol", "I", "oo", "nan", "zoo"})


def _validate_formula_string(formula_str: str, max_length: int) -> str:
    """Apply the shared string-in policy and return the stripped formula.

    Single source of truth for the pre-parse checks: type, non-emptiness,
    length cap, forbidden substrings/identifiers, and attribute access. Both
    :func:`safe_parse_expr` and :func:`safe_parse_pde` call this, so the two
    allow-lists cannot drift apart -- the duplication is the risk, not the
    policy, which is deliberately identical for both entry points.

    Raises:
        ValueError: on empty input, over-length input, forbidden tokens, or
            attribute access.
        TypeError: if ``formula_str`` is not a string.
    """
    if not isinstance(formula_str, str):
        raise TypeError(f"Formula must be a string, got {type(formula_str)}")
    s = formula_str.strip()
    if not s:
        raise ValueError("Empty formula string")
    if len(s) > max_length:
        raise ValueError(f"Formula exceeds max length {max_length} (got {len(s)})")
    lowered = s.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token.lower() in lowered:
            raise ValueError(f"Forbidden token in formula: {token!r}")
    for identifier in _FORBIDDEN_IDENTIFIERS:
        if re.search(rf"\b{identifier}\b", lowered):
            raise ValueError(f"Forbidden identifier in formula: {identifier!r}")
    if _rejects_attribute_access(s):
        raise ValueError("Attribute access ('.') is not allowed in formulas")
    return s


def safe_parse_expr(
    formula_str: str,
    extra_symbols: dict[str, sp.Symbol] | None = None,
    max_length: int = MAX_FORMULA_LENGTH,
) -> sp.Expr:
    """Parse a user-supplied formula string with an allowlist policy.

    Raises:
        ValueError: on empty input, over-length input, forbidden tokens,
            attribute access, or unparseable content.
        TypeError: if ``formula_str`` is not a string.
    """
    s = _validate_formula_string(formula_str, max_length)
    # Power operator normalization: allow ^ as alias for **
    # (convert_xor transformation handles this too, kept explicit for clarity)
    # Shadow sympy globals (e.g. beta/gamma/zeta functions) with plain
    # Symbols so variable names like ``beta`` parse as symbols, not as
    # sympy special functions. Parser internals must keep resolving.
    local_dict: dict = dict(_SAFE_FUNCS)
    if extra_symbols:
        local_dict.update(extra_symbols)
    for ident in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)):
        if ident not in local_dict and ident not in _PARSER_INTERNALS:
            local_dict[ident] = sp.Symbol(ident)
    try:
        expr = sp.parse_expr(s, local_dict=local_dict, transformations=_TRANSFORMATIONS)
    except Exception as e:
        raise ValueError(f"Unparseable formula {s!r}: {e}") from e
    # Post-check: no unexpected function applications survived (allowlist by name).
    for atom in expr.atoms(sp.Function):
        func_type = type(atom)
        if func_type not in {type(v) for v in _SAFE_FUNCS.values() if isinstance(v, sp.FunctionClass | type)}:
            # Allowlist by name instead of fragile type comparison.
            name = getattr(func_type, "__name__", str(func_type))
            if name not in {
                "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                "exp", "log", "sqrt", "Abs", "sign", "Min", "Max", "min", "max",
                "Heaviside", "re", "im", "conjugate", "sinh", "cosh", "tanh",
                "erf", "erfc", "gamma", "zeta",
                "besselj", "bessely", "besseli", "besselk",
            }:
                raise ValueError(f"Function {name!r} is not in the allowlist")
    return expr


_PDE_FUNCS = ("Derivative", "Integral", "Function")


def safe_parse_pde(
    formula_str: str,
    max_length: int = MAX_FORMULA_LENGTH,
) -> sp.Expr:
    """Parse a PDE/ODE residual string (``= 0`` form) with a tight allowlist.

    Same forbidden-token, length-cap, and no-attribute-access policy as
    :func:`safe_parse_expr`, plus ``Derivative`` / ``Integral`` / named
    ``Function`` heads. Unknown callables (``T(x, t)``) become undefined
    functions; bare identifiers become symbols. Anything else that survives
    as a function application must be in the scalar allowlist or be an
    undefined function / Derivative / Integral, else ``ValueError``.
    """
    s = _validate_formula_string(formula_str, max_length)
    local_dict: dict = dict(_SAFE_FUNCS)
    local_dict.update({"Derivative": sp.Derivative, "Integral": sp.Integral, "Function": sp.Function})
    called = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", s))
    for ident in called:
        if ident in ("E", "pi", "EulerGamma", "I"):
            # Field call form (E(x,t), B(x,t)) beats the scalar constant.
            local_dict[ident] = sp.Function(ident)
        elif ident not in local_dict and ident not in _PARSER_INTERNALS:
            local_dict[ident] = sp.Function(ident)
    for ident in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)):
        if ident not in local_dict and ident not in _PARSER_INTERNALS:
            local_dict[ident] = sp.Symbol(ident)
    try:
        expr = sp.parse_expr(s, local_dict=local_dict, transformations=_TRANSFORMATIONS)
    except Exception as e:
        raise ValueError(f"Unparseable PDE formula {s!r}: {e}") from e
    for atom in expr.atoms(sp.Function):
        func_type = type(atom)
        name = getattr(func_type, "__name__", str(func_type))
        if name in {
            "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
            "exp", "log", "sqrt", "Abs", "sign", "Min", "Max", "min", "max",
            "Heaviside", "re", "im", "conjugate", "sinh", "cosh", "tanh",
            "erf", "erfc", "gamma", "zeta",
            "besselj", "bessely", "besseli", "besselk", "Derivative",
        }:
            continue
        # Undefined functions (T, u, psi, ...) are the point of PDE input.
        from sympy.core.function import AppliedUndef

        if isinstance(atom, AppliedUndef):
            continue
        raise ValueError(f"Function {name!r} is not in the PDE allowlist")
    return expr


class LimitTimeout(Exception):
    """Raised when a symbolic limit exceeds its wall-clock budget."""


#: Bound on concurrent/abandoned symbolic-limit workers. ``sp.limit`` cannot
#: be cancelled, so each timeout abandons a daemon thread until SymPy returns.
#: Without a bound, a burst of adversarial timeouts exhausts threads. When the
#: bound is reached the call fails fast with :class:`LimitTimeout` instead of
#: spawning another worker.
_MAX_LIMIT_WORKERS = 32
_LIMIT_SEMAPHORE = threading.Semaphore(_MAX_LIMIT_WORKERS)


def limit_with_timeout(
    expr: sp.Expr,
    sym: sp.Symbol,
    value: float | sp.Expr,
    timeout: float = 2.0,
    dir: str | None = None,
) -> sp.Expr:
    """Compute ``sp.limit`` with a wall-clock timeout.

    The symbolic attempt runs on a daemon thread and the caller waits at most
    ``timeout`` seconds. ``sp.limit`` cannot be cancelled cooperatively, so on
    expiry the worker thread is abandoned rather than joined: the caller is
    guaranteed to regain control at ``timeout`` no matter how long the symbolic
    engine would have run. (Joining an un-cancellable worker is precisely how a
    timeout stops bounding anything -- ``ThreadPoolExecutor.__exit__`` calls
    ``shutdown(wait=True)`` and blocks on the hung thread.)

    An abandoned worker keeps consuming a thread until SymPy returns. Per call
    that is bounded either way; the abandoned thread cannot block process exit
    because it is a daemon.

    On expiry a short numeric probe is attempted; :class:`LimitTimeout` is
    raised only when both the symbolic path and the numeric probe fail to
    produce a value within budget.

    Args:
        expr: expression to take the limit of.
        sym: symbol approaching ``value``.
        value: limit point (may be ``sp.oo``).
        timeout: wall-clock seconds for the symbolic attempt.
        dir: optional one-sided direction ``"+"`` / ``"-"``.
    """
    def _symbolic() -> sp.Expr:
        if dir is None:
            return sp.limit(expr, sym, value)
        return sp.limit(expr, sym, value, dir=dir)

    box: dict[str, sp.Expr] = {}
    failure: list[BaseException] = []

    def _worker() -> None:
        try:
            box["result"] = _symbolic()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            failure.append(exc)
        finally:
            _LIMIT_SEMAPHORE.release()

    if not _LIMIT_SEMAPHORE.acquire(blocking=False):
        raise LimitTimeout(
            f"limit({expr}, {sym} -> {value}) refused: too many symbolic-limit "
            f"workers (>= {_MAX_LIMIT_WORKERS}) in flight"
        )
    worker = threading.Thread(target=_worker, name="apedp-symbolic-limit", daemon=True)
    try:
        worker.start()
    except Exception:
        _LIMIT_SEMAPHORE.release()
        raise
    worker.join(timeout)

    if not worker.is_alive():
        if failure:
            raise failure[0]
        return box["result"]

    # Timed out. Do not join: the symbolic worker has no cancellation hook.
    logger.warning("sp.limit timed out after %.1fs for %s -> %s", timeout, sym, value)
    # Numeric fallback: probe approaching the point (a handful of evaluations).
    try:
        import math

        f = sp.lambdify(sym, expr, modules=["numpy"])

        if value in (sp.oo, -sp.oo, sp.zoo):
            probes = [1e6, 1e7] if value == sp.oo else [-1e6, -1e7]
        else:
            v = float(value)
            probes = [v + d for d in (1e-6, -1e-6, 1e-7, -1e-7)]
        vals = []
        for p in probes:
            try:
                # Deliberately at the singularity; the non-finite results are
                # handled below, so numpy's warning about them is noise here.
                with quiet_float_warnings():
                    vals.append(float(f(p)))
            except (ValueError, TypeError, ArithmeticError):
                continue

        finite = [x for x in vals if math.isfinite(x)]
        if finite:
            # FLAW-11: symmetric probe pairs cancel across odd poles —
            # f(x) = 1/x at x -> 0 averages (+1e6, -1e6) to exactly 0.0 and
            # the fallback reported a smooth limit across a vertical
            # asymptote. Detect sign opposition across the approach point:
            # probing on both sides (finite value less than +/- oo -> both
            # directions present) with non-negligible opposite-signed
            # magnitudes is asymptote evidence, not a finite two-sided limit.
            if value not in (sp.oo, -sp.oo, sp.zoo) and len(finite) >= 2:
                left = [x for x, p in zip(vals, probes, strict=False) if math.isfinite(x) and p < float(value)]
                right = [x for x, p in zip(vals, probes, strict=False) if math.isfinite(x) and p > float(value)]
                if left and right:
                    l_val, r_val = left[-1], right[-1]
                    if (l_val < 0.0) != (r_val < 0.0) and min(abs(l_val), abs(r_val)) > 1e-6:
                        raise LimitTimeout(
                            f"limit({expr}, {sym} -> {value}) timed out; numerically "
                            f"detected opposed one-sided behaviour (f(v-)={l_val:.3e} "
                            f"vs f(v+)={r_val:.3e}) — refusing to average across a "
                            "probable pole/asymptote"
                        )
            return sp.Float(sum(finite) / len(finite))
    except (ValueError, TypeError, ArithmeticError):
        logger.warning("Numeric limit fallback failed", exc_info=True)
    raise LimitTimeout(f"limit({expr}, {sym} -> {value}) timed out after {timeout}s")
