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
import io
import keyword
import logging
import re
import threading
import tokenize
import unicodedata
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_FORMULA_LENGTH",
    "MAX_NAME_LENGTH",
    "FormulaResourceError",
    "LimitTimeout",
    "limit_with_timeout",
    "plain_names",
    "quiet_float_warnings",
    "safe_parse_expr",
    "safe_parse_pde",
]

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

#: Public name for the allowlist above. ``apedp.safety`` is a re-export
#: façade, and ``apedp.epistemic`` needs to know which identifiers the parser
#: resolves to functions so it can shadow every *other* identifier with a
#: Symbol (physics ``I`` is current, not sqrt(-1)). Importing ``_SAFE_FUNCS``
#: from the façade made a private name part of a cross-package contract, and
#: the façade never declared it, so the import failed. Same dict object, not a
#: copy: extending one name extends both.
SAFE_FUNCTIONS = _SAFE_FUNCS

_TRANSFORMATIONS = (*standard_transformations, convert_xor)

#: Characters that have no place in a formula. Quotes make text literals and, with a prefix, f-strings, whose contents
#: Python evaluates as code and which the tokenizer of Python 3.11 and earlier hands to the parser as one opaque token,
#: so no check on identifiers ever sees inside. Braces and brackets make sets, dictionaries, lists and comprehensions.
#: A colon makes a lambda, a slice or an assignment expression. A backslash continues a line.
_FORBIDDEN_CHARACTERS = frozenset("'\"{}[]\\:#@$?")

#: What the parser's rewritten source may name besides the caller's symbols and the functions of the allowlist: the
#: constructors SymPy's own token and tree rewriting emit (also with ``evaluate=False``). Nothing else, and no builtin.
_PARSER_NAMESPACE: dict[str, Any] = {
    name: getattr(sp, name) for name in (
        "Integer", "Float", "Rational", "Number", "Symbol", "Function", "Lambda", "I", "oo", "nan", "zoo",
        "factorial", "factorial2", "Add", "Mul", "Pow", "And", "Or", "Not", "Eq", "Ne", "Lt", "Le", "Gt", "Ge")
}


def _parser_globals() -> dict[str, Any]:
    """The namespace a formula is evaluated in: SymPy's constructors, and no Python builtin.

    ``sympy.parse_expr`` runs the rewritten formula through ``eval``. Left to itself it does that in a namespace into
    which it has copied every builtin function (``eval``, ``open``, ``getattr`` and the rest), and ``eval`` adds
    ``__builtins__`` to any namespace that lacks it. A formula that got a call past the checks above would reach them.
    Here ``__builtins__`` is empty and the only names are the ones the rewriting needs. A new dictionary each time:
    ``eval`` may write to it.
    """
    return {"__builtins__": {}, **_PARSER_NAMESPACE}


# A number such as ``1e999999999`` is twelve characters and held a worker for more than ten minutes: sympy builds the
# power of ten exactly (found by this module's own adversarial sweep, 28 November 2026; it was so before the audit's
# repairs too). A literal's decimal exponent is bounded like any other numeric exponent here; the limit is on work.
_LITERAL_EXPONENT_LIMIT = 9999
_MAX_EXPONENT_DIGITS = len(str(_LITERAL_EXPONENT_LIMIT))
_DECIMAL_EXPONENT = re.compile(r"[eE]([+-]?[0-9_]+)[jJ]?\Z")


_NOT_PART_OF_THE_FORMULA = frozenset({tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
                                      tokenize.ENDMARKER})

#: SymPy's number constructors, which a formula may spell out: ``x**Rational(1, 2)`` is one of the ways a square
#: root is written here. Each with the most arguments it takes, whether they must be whole numbers, and the words
#: for a refusal. They are allowed on plain numbers only, which makes them another way of writing a literal and
#: nothing more: ``Float(1, 5000000)`` asks for five million digits, ``Rational(9**9**9, 1)`` for the power, and
#: ``Integer(1e9999)`` for a whole number far longer than the formula.
_NUMBER_CONSTRUCTORS: dict[str, tuple[int, bool, str]] = {
    "Rational": (2, True, "one or two whole numbers"),
    "Integer": (1, True, "one whole number"),
    "Float": (1, False, "one plain number"),
}
_WHOLE_NUMBER = re.compile(r"[0-9]+\Z")


def _written_as_plain_numbers(tokens: Sequence[tokenize.TokenInfo], opening: int, most: int, whole: bool) -> bool:
    """Whether the call whose bracket opens at ``opening`` holds one to ``most`` numbers, each with at most a sign."""
    count, position = 0, opening + 1
    while position < len(tokens):
        if tokens[position].type == tokenize.OP and tokens[position].string in ("+", "-"):
            position += 1
        if position >= len(tokens) or tokens[position].type != tokenize.NUMBER:
            return False
        text = tokens[position].string
        if text[-1] in "jJ" or (whole and not _WHOLE_NUMBER.match(text)):
            return False
        count, position = count + 1, position + 1
        if position >= len(tokens) or tokens[position].type != tokenize.OP:
            return False
        if tokens[position].string == ")":
            return 1 <= count <= most
        if tokens[position].string != ",":
            return False
        position += 1
    return False


def _token_objection(s: str, may_call: Callable[[str], bool] | None = None) -> str | None:
    """Why the tokens of ``s`` are not a formula's, or ``None``.

    Read with Python's own tokenizer, so ``.5`` and ``1.e-5`` are numbers and ``x.real`` is attribute access; a check
    on the text cannot tell these apart, and the earlier one refused the two numbers. Text literals of every kind are
    refused here as well as by character, whatever their prefix.

    With ``may_call``, every call in the formula is checked as well: only a name may be called, and only one that
    ``may_call`` passes, or one of SymPy's number constructors on plain numbers. Without this, ``Float(1, 5000000)``
    reached SymPy's own constructor and made a number of five million digits, and a call of a name that is no
    function failed only after its arguments had been worked out, however large (an outside review of 28 November
    2026 reproduced both).
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(s).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        return f"Unparseable formula {s!r}: {exc}"
    tokens = [token for token in tokens if token.type not in _NOT_PART_OF_THE_FORMULA]
    before: tokenize.TokenInfo | None = None
    for index, token in enumerate(tokens):
        if may_call is not None and token.type == tokenize.OP and token.string == "(" and before is not None:
            if before.type == tokenize.NAME and not keyword.iskeyword(before.string):
                # worded as the ordinary parse worded a call it could not make: callers match on the first words
                if before.string in _NUMBER_CONSTRUCTORS:
                    most, whole, takes = _NUMBER_CONSTRUCTORS[before.string]
                    if not _written_as_plain_numbers(tokens, index, most, whole):
                        return f"Unparseable formula {s!r}: {before.string!r} may be given {takes} only"
                elif not may_call(before.string):
                    return f"Unparseable formula {s!r}: {before.string!r} is not a function a formula may call"
            elif before.type == tokenize.NUMBER or before.string in (")", "!", "!!"):
                return f"Unparseable formula {s!r}: a formula may call a named function only"
        before = token
        kind = tokenize.tok_name.get(token.type, "")
        if token.type == tokenize.OP and token.string in (".", "..."):
            return "Attribute access ('.') is not allowed in formulas"
        if token.type == tokenize.STRING or kind.startswith("FSTRING") or kind.startswith("TSTRING"):
            return "Text literals are not allowed in formulas"
        if token.type == tokenize.NUMBER and token.string[:2].lower() not in ("0x", "0o", "0b"):
            exponent = _DECIMAL_EXPONENT.search(token.string)
            if exponent is not None and len(exponent.group(1).replace("_", "").lstrip("+-0")) > _MAX_EXPONENT_DIGITS:
                return (f"The number {token.string[:24]!r} has a decimal exponent beyond 1e{_LITERAL_EXPONENT_LIMIT}; "
                        "sympy would build it digit by digit")
    return None


def _rejects_attribute_access(s: str) -> bool:
    """Return True if the string contains attribute access. Floats such as ``3.14``, ``.5`` and ``1.e-5`` do not."""
    return _token_objection(s) == "Attribute access ('.') is not allowed in formulas"


MAX_NAME_LENGTH = 64
_PLAIN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def plain_names(names: Iterable[Any]) -> list[str]:
    """``names`` as a list of strings, if every one is a plain name; otherwise ``ValueError`` naming the first that is not.

    **Why.** The name of a column ends up inside formulas, and the discovery routes read formulas with
    ``sympy.sympify``, which evaluates them as Python. A "name" that is more than a name (``a+b``, ``f(1)``,
    ``x.real``) is therefore run as code. The route modules that do this are frozen by preregistrations and cannot be
    changed, so the rule is enforced where names come in: every place that takes column names from outside calls this
    first.

    **A plain name** is ASCII letters, digits and underscores, starting with a letter, at most ``MAX_NAME_LENGTH``
    long, not a Python keyword (``lambda``, ``in``, ``is``: a formula cannot use those as symbols), without a double
    underscore, and not used twice. A leading underscore is refused because the system's own fitted parameters are
    named that way (``_c0``, ``_k1``), and a column of that name would be taken for one of them.
    """
    out: list[str] = []
    for raw in names:
        name = raw if isinstance(raw, str) else str(raw)
        if not _PLAIN_NAME.match(name) or len(name) > MAX_NAME_LENGTH or "__" in name or keyword.iskeyword(name):
            raise ValueError(
                f"column name {name[:40]!r} is not a plain name: use letters, digits and underscores, starting with a "
                "letter (not a Python keyword, no double underscore)")
        if name in out:
            raise ValueError(f"column name {name!r} is used twice")
        out.append(name)
    return out


#: Function heads a parsed formula may contain. ``Piecewise`` is in the table of names a formula may use above; it
#: was missing here, so every formula that used it was refused after it had parsed.
_ALLOWED_FUNCTION_NAMES = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "log", "sqrt", "Abs", "sign", "Min", "Max", "min", "max",
    "Heaviside", "re", "im", "conjugate", "sinh", "cosh", "tanh",
    "erf", "erfc", "gamma", "zeta",
    "besselj", "bessely", "besseli", "besselk", "Piecewise",
})

#: Names the parser itself needs to resolve; never shadowed with Symbols.
_PARSER_INTERNALS = frozenset({"Integer", "Float", "Rational", "Symbol", "I", "oo", "nan", "zoo"})

#: The names a formula may call: the functions of the allowlist, and nothing the parser's own rewriting uses.
_CALLABLE_NAMES = frozenset(name for name, value in _SAFE_FUNCS.items() if callable(value))


def _may_call_in_a_formula(name: str) -> bool:
    return name in _CALLABLE_NAMES


def _may_call_in_a_pde(name: str) -> bool:
    """Any name but the parser's own: an unknown one is the equation's own function, such as ``T(x, t)``."""
    return name == "I" or name not in _PARSER_INTERNALS


def _validate_formula_string(formula_str: str, max_length: int, may_call: Callable[[str], bool]) -> str:
    """Apply the shared string-in policy and return the stripped formula.

    Single source of truth for the pre-parse checks: type, non-emptiness,
    length cap, forbidden substrings/identifiers, and attribute access. Both
    :func:`safe_parse_expr` and :func:`safe_parse_pde` call this, so the two
    allow-lists cannot drift apart -- the duplication is the risk, not the
    policy, which is deliberately identical for both entry points.

    Raises:
        ValueError: on empty input, over-length input, forbidden tokens,
            attribute access, or a call of anything ``may_call`` does not pass.
        TypeError: if ``formula_str`` is not a string.
    """
    if not isinstance(formula_str, str):
        raise TypeError(f"Formula must be a string, got {type(formula_str)}")
    s = formula_str.strip()
    if not s:
        raise ValueError("Empty formula string")
    if len(s) > max_length:
        raise ValueError(f"Formula exceeds max length {max_length} (got {len(s)})")
    # Python reads identifiers in normal form NFKC: full-width or mathematical letters are the plain ones to the
    # compiler and other letters to every check on the text. A formula that changes under that normal form is refused,
    # not rewritten: ``x²`` must not quietly become a symbol named ``x2``.
    if unicodedata.normalize("NFKC", s) != s:
        raise ValueError("Formula contains look-alike characters (it is not in Unicode normal form NFKC)")
    bad = sorted(set(s) & _FORBIDDEN_CHARACTERS)
    if bad:
        raise ValueError(f"Forbidden character in formula: {bad[0]!r}")
    lowered = s.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token.lower() in lowered:
            raise ValueError(f"Forbidden token in formula: {token!r}")
    for identifier in _FORBIDDEN_IDENTIFIERS:
        if re.search(rf"\b{identifier}\b", lowered):
            raise ValueError(f"Forbidden identifier in formula: {identifier!r}")
    objection = _token_objection(s, may_call)
    if objection is not None:
        raise ValueError(objection)
    return s


#: Bounds on what the parser is allowed to *evaluate*. SymPy folds numeric
#: sub-expressions while parsing, so ``9**9**9**9`` or ``zeta(10**7)`` is not a
#: large expression but an unbounded computation, and the allowlist above never
#: sees it because it runs after the parse returns. These are far above anything
#: a physical law uses (exponents are single digits; special-function arguments
#: are order one) and far below what costs time or memory.
MAX_NUMERIC_EXPONENT = 1000
MAX_NUMERIC_POWER_BITS = 20_000
MAX_SPECIAL_FUNCTION_ARGUMENT = 1000

#: Functions whose value at a large numeric argument stays symbolic or is cheap.
#: Every other function head (gamma, zeta, the Bessel family, factorial written
#: as ``n!``) is held to ``MAX_SPECIAL_FUNCTION_ARGUMENT``.
_CHEAP_AT_LARGE_ARGUMENT = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log",
    "Abs", "sign", "Heaviside", "re", "im", "conjugate",
    "sinh", "cosh", "tanh", "erf", "erfc",
})


class FormulaResourceError(ValueError):
    """A formula whose evaluation would exceed the parser's resource bounds."""


#: Function heads that are not held to that limit though they are not in the set above: ``sqrt`` is a power, which
#: the rule for powers bounds, and the others compare or choose among their arguments.
_NOT_HELD_TO_THE_LIMIT = _CHEAP_AT_LARGE_ARGUMENT | {"sqrt", "Min", "Max", "Piecewise"}

#: Functions the parser's own rewriting calls (``n!``), by the name it calls them.
_REWRITING_CALLS: dict[str, Any] = {"factorial": sp.factorial, "factorial2": sp.factorial2}


def _guarded_rebuild(node: sp.Basic, real: Mapping[str, Any] | None = None) -> sp.Basic:
    """Evaluate an unevaluated parse tree bottom-up, refusing unbounded steps.

    Children are evaluated first, so the check sees what SymPy would actually
    compute: ``9**(9**(9 + x - x))`` reaches the outer power with the exponent
    already folded to an integer, which a check on the source text would miss.

    ``real`` maps the name of a stand-in function (see :func:`_assert_bounded_evaluation`) to the function it stands
    for. A stand-in whose arguments have all come out as numbers within the bounds is then worked out with the real
    function, so that ``gamma(gamma(8))`` reaches the outer call as ``gamma(5040)`` and is refused there.
    """
    if not node.args:
        return node
    args = [_guarded_rebuild(arg, real) for arg in node.args]
    if isinstance(node, sp.Pow) and len(args) == 2:
        base, exponent = args
        if exponent.is_Number:
            if exponent.is_finite is not True or abs(exponent) > MAX_NUMERIC_EXPONENT:
                raise FormulaResourceError(
                    f"Numeric exponent exceeds the limit of {MAX_NUMERIC_EXPONENT}"
                )
            if base.is_Rational:
                bits = max(abs(int(base.p)).bit_length(), int(base.q).bit_length())
                if bits * (int(abs(exponent)) + 1) > MAX_NUMERIC_POWER_BITS:
                    raise FormulaResourceError(
                        "Numeric power is too large to evaluate "
                        f"(over {MAX_NUMERIC_POWER_BITS} bits)"
                    )
    elif isinstance(node, sp.Function):
        name = type(node).__name__
        stands_for = (real or {}).get(name)
        known_as = getattr(stands_for, "__name__", name) if stands_for is not None else name
        if known_as not in _NOT_HELD_TO_THE_LIMIT:
            for arg in args:
                if arg.is_Number and (
                    arg.is_finite is not True or abs(arg) > MAX_SPECIAL_FUNCTION_ARGUMENT
                ):
                    raise FormulaResourceError(
                        f"Numeric argument to {name!r} exceeds the limit of "
                        f"{MAX_SPECIAL_FUNCTION_ARGUMENT}"
                    )
        if stands_for is not None and all(arg.is_Number for arg in args):
            value = stands_for(*args)
            return value if isinstance(value, sp.Basic) else node.func(*args)
    return node.func(*args)


def _assert_bounded_evaluation(s: str, local_dict: dict[str, object], what: str = "formula") -> None:
    """Raise :class:`FormulaResourceError` if parsing ``s`` would run unbounded, ``ValueError`` if it cannot be parsed.

    Parses once with ``evaluate=False`` and replays the evaluation under :func:`_guarded_rebuild`. The caller then
    performs its ordinary evaluated parse, so accepted formulas produce exactly the expression they did before this
    guard existed.

    **Nothing is worked out during that first parse.** ``evaluate=False`` stops SymPy's arithmetic and not its
    functions: ``gamma(1001)``, ``1001!``, ``zeta(100000)`` and ``Min(x, 9**9**9)`` were all worked out while the
    tree was being built, before the bounds were looked at, and the last two held a worker (an outside review of 28
    November 2026 reproduced each). Every function is therefore replaced for this parse by a stand-in of the same
    name that does nothing, and the rebuild works a stand-in out only once its arguments are known to be in bounds.

    **A formula this parse cannot read is refused here.** It used to be left to the ordinary parse to report. That
    parse works the arguments of a call out before it finds that the call itself cannot be made, so ``foo(9**9**9)``
    failed only after the power had been attempted.
    """
    functions = {name: value for name, value in local_dict.items() if callable(value)}
    inert: dict[str, Any] = {**local_dict, **{name: sp.Function(name) for name in functions}}
    try:
        unevaluated = sp.parse_expr(
            s, local_dict=inert, transformations=_TRANSFORMATIONS, evaluate=False,
            global_dict={**_parser_globals(), **{name: sp.Function(name) for name in _REWRITING_CALLS}},
        )
        if isinstance(unevaluated, sp.Basic):
            _guarded_rebuild(unevaluated, {**functions, **_REWRITING_CALLS})
    except FormulaResourceError:
        raise
    except RecursionError as exc:
        raise FormulaResourceError("Formula is nested too deeply to evaluate safely") from exc
    except Exception as exc:
        raise ValueError(f"Unparseable {what} {s!r}: {exc}") from exc


def _a_safe_function_by_name(parsed: Any) -> bool:
    """True if the formula was the bare name of an allowed function, such as ``gamma`` with no argument.

    The parser has always handed back the function itself for that, and one caller reads it:
    ``apedp/mathematical/mathematical_layer.py`` treats it as a candidate with nothing to check. Nothing was called
    and the function is one of the allowed ones, so it is returned as before. The first version of the 27 November
    repair refused it with every other non-expression, and that caller's test failed on the stage branch.
    """
    return isinstance(parsed, sp.FunctionClass) and any(parsed is value for value in _SAFE_FUNCS.values())


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
    s = _validate_formula_string(formula_str, max_length, _may_call_in_a_formula)
    # Power operator normalization: allow ^ as alias for **
    # (convert_xor transformation handles this too, kept explicit for clarity)
    # Shadow sympy globals (e.g. beta/gamma/zeta functions) with plain
    # Symbols so variable names like ``beta`` parse as symbols, not as
    # sympy special functions. Parser internals must keep resolving.
    local_dict: dict[str, Any] = dict(_SAFE_FUNCS)
    if extra_symbols:
        local_dict.update(extra_symbols)
    for ident in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)):
        if ident not in local_dict and ident not in _PARSER_INTERNALS:
            local_dict[ident] = sp.Symbol(ident)
    _assert_bounded_evaluation(s, local_dict)
    try:
        expr = sp.parse_expr(s, local_dict=local_dict, global_dict=_parser_globals(), transformations=_TRANSFORMATIONS)
    except Exception as e:
        raise ValueError(f"Unparseable formula {s!r}: {e}") from e
    if _a_safe_function_by_name(expr):
        return expr
    if not isinstance(expr, sp.Basic):
        # A tuple, a truth value, a number of Python's own: evaluated, and not a formula. Without this check the
        # caller got an AttributeError from the next line instead of the ValueError it handles.
        raise ValueError(f"Unparseable formula {s!r}: it does not evaluate to an expression")
    # Post-check: no unexpected function applications survived (allowlist by name).
    for atom in expr.atoms(sp.Function):
        func_type = type(atom)
        if func_type not in {type(v) for v in _SAFE_FUNCS.values() if isinstance(v, sp.FunctionClass | type)}:
            # Allowlist by name instead of fragile type comparison.
            name = getattr(func_type, "__name__", str(func_type))
            if name not in _ALLOWED_FUNCTION_NAMES:
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
    s = _validate_formula_string(formula_str, max_length, _may_call_in_a_pde)
    local_dict: dict[str, Any] = dict(_SAFE_FUNCS)
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
    _assert_bounded_evaluation(s, local_dict, "PDE formula")
    try:
        expr = sp.parse_expr(s, local_dict=local_dict, global_dict=_parser_globals(), transformations=_TRANSFORMATIONS)
    except Exception as e:
        raise ValueError(f"Unparseable PDE formula {s!r}: {e}") from e
    if _a_safe_function_by_name(expr):
        return expr
    if not isinstance(expr, sp.Basic):
        raise ValueError(f"Unparseable PDE formula {s!r}: it does not evaluate to an expression")
    for atom in expr.atoms(sp.Function):
        func_type = type(atom)
        name = getattr(func_type, "__name__", str(func_type))
        if name in _ALLOWED_FUNCTION_NAMES or name == "Derivative":
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
