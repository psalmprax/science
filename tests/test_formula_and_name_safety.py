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
