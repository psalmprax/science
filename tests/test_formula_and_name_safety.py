"""What comes in from outside is a name or a formula and nothing more (outside audit of 27 November 2026).

Three things were found that day. The route endpoint took column names as they came, and the routes evaluate formulas
that contain them, so a "name" such as ``q.subs(q, 7)`` was run as code. The formula parser kept builtins within reach
and could not see inside an f-string on Python 3.11 and earlier, so look-alike letters got a call through. And it
refused ``.5`` and ``Piecewise`` and ended in an ``AttributeError`` on anything that evaluated to something other than
an expression.
"""
from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest
import sympy as sp

import apedp.safety
from apedp_dyn import safety
from apedp_dyn.ui import route_endpoint

X = sp.Symbol("x")
NOT_NAMES = ["q.subs(q, 7)", "a+b", "f(1)", "x y", "a;b", "lambda", "import", "_c0", "a__b", "1x", "", "x'", "tempé",
             "ｘ", "x" * 65]


def test_a_column_name_is_a_name_and_nothing_more() -> None:
    assert safety.plain_names(["a", "x0", "Temp_K", "E", "gamma", "pi", "type"]) == ["a", "x0", "Temp_K", "E", "gamma", "pi", "type"]
    assert safety.plain_names({"a": [1.0], "b": [2.0]}) == ["a", "b"]                    # a mapping's keys, in order
    for bad in NOT_NAMES:
        with pytest.raises(ValueError, match="not a plain name"):
            safety.plain_names(["a", bad])
    with pytest.raises(ValueError, match="used twice"):
        safety.plain_names(["a", "a"])


@pytest.mark.parametrize("time_ordered", [False, True])
@pytest.mark.parametrize("name", ["q.subs(q, 7)", "abs(b)", "b)+(1", "ｂ"])
def test_the_route_endpoint_refuses_a_table_whose_column_name_is_code(name: str, time_ordered: bool,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("a route was run on a table whose column name is not a name")

    monkeypatch.setattr(route_endpoint.vouched_route_v2, "discover_law", never)
    monkeypatch.setattr(route_endpoint.series_route, "discover_predictive_law", never)
    rng = np.random.default_rng(0)
    a, b = rng.uniform(1, 5, 400), rng.uniform(1, 5, 400)
    out = route_endpoint.route_request({"columns": {"a": a.tolist(), name: b.tolist()}, "target": (2 * a + 3 * b).tolist(),
                                        "time_ordered": time_ordered})
    assert out["action"] == "not_run" and out["law"] is None and out["reported"] is False
    assert "is not a plain name" in out["error"]


def test_a_plain_table_is_read_as_before() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.uniform(1, 5, 400), rng.uniform(1, 5, 400)
    out = route_endpoint.route_request({"columns": {"a": a.tolist(), "b": b.tolist()}, "target": (2 * a + 3 * b).tolist()})
    assert out.get("error") is None and out["law_as_confirmed"] is not None
    law = sp.lambdify([sp.Symbol("a"), sp.Symbol("b")], safety.safe_parse_expr(out["law_as_confirmed"]), "numpy")
    assert np.allclose(law(a, b), 2 * a + 3 * b, rtol=1e-6)


HOSTILE = {
    "an f-string": 'f"{1+2}"',
    "an f-string that calls something": "f\"{len('ab')}\"",
    "the same in full-width letters": "f\"{ｌｅｎ('ab')}\"",
    "text": "'x'",
    "a comprehension": "[x for x in [1, 2]]",
    "a subscript": "x[0]",
    "a set": "{x}",
    "a lambda": "lambda x: x",
    "an assignment expression": "(y := x)",
    "attribute access": "x.real",
    "a method call": "x.subs(x, 1)",
    "a dunder": "x.__class__",
    "a look-alike letter": "ｘ + 1",
    "a superscript": "x²",
    "a backslash": "x \\\n + 1",
    "an import": "__import__('os')",
}


@pytest.mark.parametrize("module", [safety, apedp.safety], ids=["apedp_dyn.safety", "apedp.safety"])
@pytest.mark.parametrize("label", sorted(HOSTILE))
def test_what_is_not_a_formula_is_refused_with_a_value_error(module: Any, label: str) -> None:
    for parse in (module.safe_parse_expr, module.safe_parse_pde):
        with pytest.raises(ValueError):
            parse(HOSTILE[label])


def test_a_formula_is_evaluated_without_python_s_builtins() -> None:
    namespace = safety._parser_globals()
    assert namespace["__builtins__"] == {}
    assert not [name for name, value in namespace.items() if isinstance(value, types.BuiltinFunctionType)]
    assert safety._parser_globals() is not namespace                       # a new one each time: eval may write to it
    # even with every check before it skipped, a builtin's name is only a symbol there
    got = sp.parse_expr("len(x) + open(x)", local_dict={"x": X}, global_dict=safety._parser_globals(),
                        transformations=safety._TRANSFORMATIONS)
    assert {type(f).__name__ for f in got.atoms(sp.Function)} == {"len", "open"}


def test_numbers_with_a_bare_dot_and_piecewise_parse() -> None:
    assert safety.safe_parse_expr(".5*x", {"x": X}) == 0.5 * X
    assert safety.safe_parse_expr("1.e-5*x + 3.", {"x": X}) == 1.0e-5 * X + 3.0
    assert safety.safe_parse_expr("2.5e3*x", {"x": X}) == 2500.0 * X
    piecewise = safety.safe_parse_expr("Piecewise((x, x > 0), (0, True))", {"x": X})
    assert isinstance(piecewise, sp.Piecewise) and piecewise.subs(X, 2) == 2 and piecewise.subs(X, -2) == 0
    assert safety.safe_parse_expr("x^2 + sin(x)/exp(x)", {"x": X}) == X ** 2 + sp.sin(X) / sp.exp(X)
    assert safety.safe_parse_expr("θ + 1") == sp.Symbol("θ") + 1          # a letter that is its own normal form is a symbol


@pytest.mark.parametrize("text", ["True", "(x, 2)", "x, 2", "x > 1 > 0"])
def test_what_does_not_evaluate_to_an_expression_is_a_value_error_not_a_crash(text: str) -> None:
    with pytest.raises(ValueError):
        safety.safe_parse_expr(text, {"x": X})


def test_the_two_packages_share_one_parser() -> None:
    assert apedp.safety.safe_parse_expr is safety.safe_parse_expr and apedp.safety.safe_parse_pde is safety.safe_parse_pde
    assert apedp.safety.plain_names is safety.plain_names and apedp.safety._SAFE_FUNCS is safety._SAFE_FUNCS
    assert apedp.safety.limit_with_timeout is safety.limit_with_timeout


def test_the_bounded_evaluation_guard_still_sees_every_formula() -> None:
    # the guard parses without evaluating, in the same namespace without builtins; if a name were missing there it
    # would be skipped without a word and an unbounded power would be evaluated
    for text in ("9**9**9**9", "2**(10**9)", "gamma(10**7)"):
        with pytest.raises(safety.FormulaResourceError):
            safety.safe_parse_expr(text)
    for text in ("x < 1", "-x/3 + 2", "sqrt(x)*Abs(x)", "x**-2", "(x > 0) & (x < 2)"):
        unevaluated = sp.parse_expr(text, local_dict={"x": X, **safety._SAFE_FUNCS}, global_dict=safety._parser_globals(),
                                    transformations=safety._TRANSFORMATIONS, evaluate=False)
        assert isinstance(unevaluated, sp.Basic)


@pytest.mark.parametrize("text", ["1e999999999", "1e-999999999", "1E+99999", "2.5e10000*x", "x*1e1_0_0_0_0", "1e99999j",
                                  "sin(1e123456)", "x**1e55555"])
def test_a_number_whose_decimal_exponent_would_take_minutes_to_build_is_refused_at_once(text: str) -> None:
    """``1e999999999`` is twelve characters and held a worker for over ten minutes (sympy builds the power of ten
    exactly). Found by an adversarial sweep of this parser on 28 November 2026, after the audit's repairs; the
    parser had always done it."""
    import time

    started = time.monotonic()
    for parse in (safety.safe_parse_expr, safety.safe_parse_pde):
        with pytest.raises(ValueError, match="decimal exponent"):
            parse(text)
    assert time.monotonic() - started < 2.0


def test_ordinary_exponents_and_other_number_forms_still_parse() -> None:
    assert safety.safe_parse_expr("6.022e23*x", {"x": X}) == sp.Float("6.022e23") * X
    assert safety.safe_parse_expr("1e-9999*x", {"x": X}).has(X)
    assert float(safety.safe_parse_expr("1e00000000000000005")) == 100000.0
    assert float(safety.safe_parse_expr("1e1_0_0")) == 1e100
    assert safety.safe_parse_expr("0x1e9") == 489 and safety.safe_parse_expr("0xFFe55555") == 0xFFE55555   # hex digits, not an exponent


# ---------------------------------------------------------------------------
# A sweep the parser was not developed against (28 November 2026)
# ---------------------------------------------------------------------------

_WRAPPERS = [
    "{}", 'f"{{{}}}"', "f'{{{}}}'", 'f"""{{{}}}"""', 'F"{{{}}}"', 'rf"{{{}}}"', "[{}]", "({},)", "{{{}}}", "[{} for _ in (1,)]",
    "({} for _ in (1,))", "x + {}", "x*({})", "({}) if x else 0", "lambda: {}", "(lambda: {})()", "(y := {})", "{} or x", "not {}",
    "-{}", "x**{}", "Piecewise(({}, True))", "sin({})", "Lambda(x, {})", "Function({})", "Symbol({})", "Integer({})", "Float({})",
    "Eq(x, {})", "x.subs(x, {})", "x.__class__({})", "getattr(x, {})", "__import__({})", "eval({})", "exec({})", "open({})",
    "print({})", "type({})", "vars({})", "globals()[{}]", "__builtins__[{}]", "().__class__.__bases__[0].__subclasses__()[{}]",
    "x if {} else x", "{}\n", "\n{}", "{}#", "#\n{}", "{};", "x;{}", "x\\\n+{}", "'%s' % {}", '"{{}}".format({})', "`{}`", "$({})",
    "{}@x", "x@{}", "x[{}]", "x[{}:]", "{{x: {}}}", "*{}", "**{}", "await {}", "yield {}", "x\x0c+{}", "x\t+\t{}", "x +{}",
    "x {}", "x​{}", "﻿{}", "{}\x00", "\x00{}", "{}\r", "(\n{}\n)",
]


def _full_width(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if c.isascii() and c.isalpha() else c for c in text)


def _mathematical_bold(text: str) -> str:
    return "".join(chr(0x1D41A + ord(c) - 97) if "a" <= c <= "z" else chr(0x1D400 + ord(c) - 65) if "A" <= c <= "Z" else c
                   for c in text)


_SPELLINGS = [
    lambda s: s, _full_width, _mathematical_bold,
    lambda s: s.replace("CANARY", "C‍ANARY"),            # a joiner inside the name
    lambda s: s.replace("CANARY", "CANARÝ"),            # a combining mark after it
    lambda s: s.replace("CANARY", "ＣANARY"),             # one look-alike letter
    lambda s: s.replace("A", "А"),                       # Cyrillic: a different name, which may stand as a symbol
]


def test_no_spelling_of_a_call_reaches_a_builtin_and_every_refusal_is_a_value_error() -> None:
    """504 ways to write a call of a function placed in Python's builtins, through both parsers.

    Wrappers (f-strings of every prefix, comprehensions, lambdas, attribute and subscript routes, statement
    separators, odd white space, control characters) crossed with spellings of the name. The function must never be
    called, a refusal must be the ``ValueError`` callers handle, and whatever is returned must be an expression.
    The same sweep was run on Python 3.11, where an f-string is one opaque token to sympy, with the same outcome.
    """
    import builtins

    called: list[Any] = []
    builtins.CANARY = lambda *a, **k: called.append(a) or 7  # type: ignore[attr-defined]
    findings: list[tuple[str, str]] = []
    try:
        for wrapper in _WRAPPERS:
            for spell in _SPELLINGS:
                text = spell(wrapper.format("CANARY(1)"))
                for parse in (safety.safe_parse_expr, safety.safe_parse_pde):
                    before = len(called)
                    try:
                        got = parse(text)
                        if not isinstance(got, sp.Basic):
                            findings.append((f"returned {type(got).__name__}", text))
                    except ValueError:
                        pass
                    except Exception as exc:  # noqa: BLE001 - any other exception is the finding
                        findings.append((type(exc).__name__, text))
                    if len(called) != before:
                        findings.append(("the builtin was called", text))
    finally:
        del builtins.CANARY  # type: ignore[attr-defined]
    assert not findings, [(kind, ascii(text)) for kind, text in findings[:8]]
    assert not called


@pytest.mark.parametrize("text", ["(" * 900 + "x" + ")" * 900, "-" * 1900 + "x", "9.9**9.9**9.9**9.9", "factorial(10**8)",
                                  "factorial(factorial(factorial(9)))", "+".join(["x"] * 999), "(1/3)**(10**9)",
                                  "(x+1)**(10**6)", "gamma(10**9)", "x+" * 1200 + "x"],
                         ids=["deep-parentheses", "deep-unary", "tower-of-floats", "huge-factorial", "nested-factorial",
                              "many-terms", "big-rational-power", "binomial", "gamma", "over-length"])
def test_a_formula_built_to_take_long_is_refused_at_once(text: str) -> None:
    import time

    started = time.monotonic()
    with pytest.raises(ValueError):
        safety.safe_parse_expr(text)
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("name", ["x\n", "x\r", "x ", " x", "x\t", "x\x00", "x,y", "x=1", 'x"', "x`", "x$", "x#", "x@", "x[0]",
                                  "x{", "x\\", "x:y", "x‍", "x́", "х", "x__", "None", "True", "class", 5, None,
                                  1.5, b"x", ("x",)], ids=lambda n: ascii(n)[:12])
def test_more_things_that_are_not_names(name: Any) -> None:
    with pytest.raises(ValueError):
        safety.plain_names(["a", name])


def test_the_bare_name_of_an_allowed_function_comes_back_as_that_function_as_it_always_did() -> None:
    """``apedp/mathematical/mathematical_layer.py`` is handed ``"gamma"`` as a candidate and reads the function the
    parser returns for it. The first version of the repair refused it with every other non-expression, and
    ``tests/test_rh_benchmark.py`` failed on the stage branch; I had run only the tests that import the parser
    themselves."""
    assert safety.safe_parse_expr("gamma") is sp.gamma and safety.safe_parse_expr("sin") is sp.sin
    assert safety.safe_parse_expr("gamma", {"gamma": sp.Symbol("gamma")}) == sp.Symbol("gamma")      # a symbol if the caller says so
    assert safety.safe_parse_expr("gamma(x) + 1", {"x": X}) == sp.gamma(X) + 1
    for not_a_formula in ("(x, 2)", "True", "Symbol", "Integer"):      # the parser's own helpers are not handed out
        with pytest.raises(ValueError):
            safety.safe_parse_expr(not_a_formula, {"x": X})
    assert safety.safe_parse_expr("Function") == sp.Symbol("Function")  # a name the parser does not know is a symbol


# ---------------------------------------------------------------------------
# A cold review of the repaired parser by an outside reviewer (28 November 2026)
#
# Three ways were left to hold a worker. Each formula here once ran for seconds, minutes or without end, so they are
# parsed in one separate process under a time limit: if a repair is undone the test fails, and does not hang.

#: formula: the kind of refusal expected
HELD_A_WORKER = {
    "Eq(x, 9**9**9)": "call", "foo(9**9**9)": "call", "f(x)": "call", "factorial(10**8)": "call",       # a call of no function
    "Float(1, 5000000)": "plain", "sin(Float(1, 5000))": "plain", "sin(Float(1, 5000000))": "plain",     # SymPy's own constructors
    "Rational(9**9**9, 1)": "plain", "Integer(x)": "plain", "Rational(x, 2)": "plain", "Symbol(x)": "call",
    "(sin)(x)": "named", "2(3)": "named", "(x + 1)(x - 1)": "named", "3!(2)": "named",                   # a call of no name
    "gamma(1001)": "bound", "1001!": "bound", "100000!": "bound", "zeta(100000)": "bound",               # worked out while the tree was built
    "Min(x, 9**9**9)": "bound", "max(9**9**9, x)": "bound", "gamma(gamma(8))": "bound", "(7!)!": "bound",
    "gamma(2001/2)": "bound", "besselj(0, 1001)": "bound", "zeta(-100001)": "bound", "9**(9**(9 + x - x))": "bound",
    "Piecewise((x, x > 9**9**9), (0, True))": "bound", "sqrt(9**9**9)": "bound", "abs(1001!)": "bound",
}

_IN_ANOTHER_PROCESS = """
import json, sys
from apedp_dyn import safety
out = {}
for text in json.loads(sys.argv[1]):
    try:
        safety.safe_parse_expr(text)
        out[text] = "accepted"
    except safety.FormulaResourceError:
        out[text] = "bound"
    except ValueError as exc:
        words = {"is not a function a formula may call": "call", "named function only": "named", " may be given ": "plain"}
        out[text] = next((kind for said, kind in words.items() if said in str(exc)), "other")
print(json.dumps(out))
"""


def test_what_a_cold_review_found_to_hold_a_worker_is_refused_at_once() -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    done = subprocess.run([sys.executable, "-c", _IN_ANOTHER_PROCESS, json.dumps(sorted(HELD_A_WORKER))], capture_output=True,
                          text=True, timeout=120, cwd=Path(__file__).resolve().parents[1], check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == HELD_A_WORKER


def test_a_function_is_worked_out_only_once_its_arguments_are_known_to_be_in_bounds() -> None:
    a, b, y = sp.symbols("a b y")
    parse = safety.safe_parse_expr
    assert parse("gamma(7)") == 720 and parse("5!") == 120 and parse("3!!") == 3 and parse("gamma(gamma(4))") == 120
    assert parse("x**3/3! - x**5/5!", {"x": X}) == X ** 3 / 6 - X ** 5 / 120
    assert parse("gamma(999)") == sp.factorial(998) and parse("gamma(1999/2)") == sp.gamma(sp.Rational(1999, 2))
    assert parse("zeta(2)") == sp.pi ** 2 / 6 and parse("zeta(1000)") == sp.zeta(1000)
    assert parse("sqrt(2000)") == 20 * sp.sqrt(5) and parse("sqrt(10**900)") == sp.Integer(10) ** 450       # a power, not held to 1000
    assert parse("min(a, b)") == sp.Min(a, b) and parse("Min(3, 5) + max(a, 2)") == sp.Max(2, a) + 3
    assert parse("abs(x) + ln(y)", {"x": X}) == sp.Abs(X) + sp.log(y) and parse("exp(10**900)") == sp.exp(sp.Integer(10) ** 900)
    assert parse("Piecewise((x, x > 0), (0, True))", {"x": X}) == sp.Piecewise((X, X > 0), (0, True))
    assert parse("besselj(0, x) + erf(2000)", {"x": X}) == sp.besselj(0, X) + sp.erf(2000)
    assert parse("gamma(x)*zeta(y)", {"x": X}) == sp.gamma(X) * sp.zeta(y)
    assert parse("gamma") is sp.gamma                                           # the bare name, as before


def test_the_parser_for_equations_keeps_its_own_functions_and_the_same_bounds() -> None:
    t, T, u = sp.Symbol("t"), sp.Function("T"), sp.Function("u")
    parsed = safety.safe_parse_pde("Derivative(T(x, t), t) - 0.5*Derivative(T(x, t), x, 2)")
    assert parsed == sp.Derivative(T(X, t), t) - 0.5 * sp.Derivative(T(X, t), X, 2)
    assert safety.safe_parse_pde("u(x) + E(x)*I(t)") == u(X) + sp.Function("E")(X) * sp.Function("I")(t)   # field call forms
    assert safety.safe_parse_pde("Derivative(Rational(1, 3)*T(x), x)") == sp.Derivative(T(X) / 3, X)
    assert safety.safe_parse_pde("Integral(T(x), (x, 0, 2000))") == sp.Integral(T(X), (X, 0, 2000))
    # Function(T)(x) parsed before the repair of 28 November 2026 and is refused since: an equation writes T(x)
    for text in ("Float(1, 5000000)", "Derivative(Rational(x, 3)*T(x), x)", "Symbol(x) + T(x)", "(T)(x)", "2(x)", "Function(T)(x)"):
        with pytest.raises(ValueError, match=r"may call|named function only|may be given"):
            safety.safe_parse_pde(text)
    # the last parsed before that repair as well: an order of differentiation is held to the limit like any argument
    for text in ("Derivative(T(x, t), t) - gamma(1001)", "u(9**9**9)", "T(x) + 1001!", "Min(u(x), 9**9**9)", "Derivative(T(x), x, 2000)"):
        with pytest.raises(safety.FormulaResourceError):
            safety.safe_parse_pde(text)


def test_a_formula_the_first_parse_cannot_read_is_refused_and_not_handed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # it used to be handed to the ordinary parse, which works a call's arguments out before it finds the call cannot be made
    real, seen = sp.parse_expr, []

    def parse_expr(text: str, **options: Any) -> Any:
        seen.append(options.get("evaluate", True))
        if options.get("evaluate", True) is False:
            raise TypeError("a quirk of the unevaluated parse")
        return real(text, **options)

    monkeypatch.setattr(safety.sp, "parse_expr", parse_expr)
    for parse, what in ((safety.safe_parse_expr, "formula"), (safety.safe_parse_pde, "PDE formula")):
        seen.clear()
        with pytest.raises(ValueError, match=rf"Unparseable {what} 'x \+ 1': a quirk"):
            parse("x + 1")
        assert seen == [False]                                                  # the ordinary parse was never reached


def test_only_the_functions_of_the_allowlist_may_be_called() -> None:
    assert safety._CALLABLE_NAMES == {name for name, value in safety._SAFE_FUNCS.items() if callable(value)}
    assert {"sin", "ln", "min", "Piecewise", "gamma"} <= safety._CALLABLE_NAMES
    assert not {"pi", "E", "EulerGamma", "Float", "Integer", "Rational", "Symbol", "factorial"} & safety._CALLABLE_NAMES
    assert safety._token_objection("not (x > 0)", safety._may_call_in_a_formula) is None       # a keyword is not a call
    assert safety._token_objection("x*(y + 1) - (z)", safety._may_call_in_a_formula) is None   # nor is a bracket
    assert safety._token_objection("foo(1)") is None                            # without a rule for calls, only the old checks
    assert "may call" in (safety._token_objection("pi(2)", safety._may_call_in_a_formula) or "")
    # every refusal of a call begins as the ordinary parse's did: callers match on those words
    for text in ("totally_unknown_function(a)", "(a)(b)", "Float(1, 2)"):
        assert (safety._token_objection(text, safety._may_call_in_a_formula) or "").startswith(f"Unparseable formula {text!r}: ")


def test_a_number_may_be_spelt_with_sympy_s_constructors_on_plain_numbers_only() -> None:
    # x**Rational(1, 2) is one of the spellings of a square root this repository uses and writes itself
    parse = safety.safe_parse_expr
    assert parse("x**Rational(1, 2)", {"x": X}) == sp.sqrt(X) and parse("Rational(1, 3)*x", {"x": X}) == X / 3
    assert parse("Integer(10)**3") == 1000 and parse("Rational(-1, 2)") == sp.Rational(-1, 2) and parse("Rational(3)") == 3
    assert parse("Float(2.5)*x", {"x": X}) == 2.5 * X and parse("Float(-2.5e3)") == -2500.0
    assert parse("x**Rational(-1, 2)", {"x": X}) == 1 / sp.sqrt(X) and parse("Float(.5) + Float(1.)") == 1.5
    refused = {"one plain number": ("Float(1, 30)", "Float()", "Float(1j)", "Float(x)", "Float(--1)", "Float(1,)"),
               "one whole number": ("Integer(2.5)", "Integer((5))", "Integer(-x)", "Integer(1e9999)", "Integer(1, 2)", "Integer(0x10)"),
               "one or two whole numbers": ("Rational(1, 2, 3)", "Rational(1/2)", "Rational(2**3, 5)", "Rational(1,)", "Rational(1.5)",
                                            "Rational(p=1, q=2)", "Rational()")}
    for takes, texts in refused.items():
        for text in texts:
            with pytest.raises(ValueError, match=f"may be given {takes} only"):
                parse(text)
