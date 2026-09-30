"""Security regression tests for the pure-Python fallback evaluator.

``RustKernelBridge._fallback_eval`` used to call ``eval(expr_str, {"__builtins__": {}}, env)``.
Passing an empty ``__builtins__`` is not a sandbox: attribute traversal still reaches every
live class in the interpreter, so an expression string arriving from the REST layer or from
CSV ingestion could execute. The evaluator now parses through the shared allowlist in
:mod:`apedp.safety` and compiles with ``sympy.lambdify``.

These tests assert the *behaviour* -- no execution, and correct numerics -- so a different
implementation that is equally safe satisfies them.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.rust_bridge import RustKernelBridge


@pytest.fixture
def bridge() -> RustKernelBridge:
    return RustKernelBridge()


@pytest.fixture
def x() -> np.ndarray:
    return np.linspace(0.1, 3.0, 7)


def _all_nan(values: np.ndarray) -> bool:
    return bool(np.all(~np.isfinite(values)))


# Payloads that a ``{"__builtins__": {}}`` sandbox does NOT stop, because attribute
# access and dunder traversal are enough on their own.
ESCAPE_PAYLOADS = [
    "(1).__class__.__mro__[1].__subclasses__()",
    "().__class__.__bases__[0].__subclasses__()",
    "[].__class__.__mro__",
    "x.__class__.__mro__",
    "''.__class__.__mro__[1].__subclasses__()",
]

# Payloads that are blocked by the name/identifier allowlist alone.
BANNED_NAME_PAYLOADS = [
    "__import__('os').system('id')",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "exec('x=1')",
    "globals()",
    "os.system('id')",
]


@pytest.mark.parametrize("payload", ESCAPE_PAYLOADS)
def test_attribute_traversal_escapes_do_not_execute(bridge: RustKernelBridge, x, payload):
    """The subclass-traversal family that defeats an empty ``__builtins__``."""
    assert _all_nan(bridge._fallback_eval(payload, ["x"], [x])), payload


@pytest.mark.parametrize("payload", BANNED_NAME_PAYLOADS)
def test_banned_identifiers_are_rejected(bridge: RustKernelBridge, x, payload):
    assert _all_nan(bridge._fallback_eval(payload, ["x"], [x])), payload


def test_no_dynamic_execution_in_the_bridge():
    """Guard the fix itself: no ``eval``/``exec``/``compile`` call anywhere in the bridge.

    Checked on the AST rather than the text, so a legitimate call to a function whose
    *name ends in* ``eval`` (``_fallback_eval``) is not mistaken for a dynamic-execution
    call, and a call hidden inside a nested scope cannot slip past.
    """
    import ast
    import inspect

    import apedp.rust_bridge as rb

    tree = ast.parse(inspect.getsource(rb))
    # ``kernel.compile(...)`` is the Rust extension's own public API (source ->
    # compiled program), not Python's ``compile()`` builtin, so attribute calls are
    # only screened for the unambiguously dangerous names.
    banned_names = {"eval", "exec", "compile", "__import__"}
    banned_attrs = {"eval", "exec", "__import__"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in banned_names:
            offenders.append(f"{func.id}() at line {node.lineno}")
        elif isinstance(func, ast.Attribute) and func.attr in banned_attrs:
            offenders.append(f".{func.attr}() at line {node.lineno}")
    assert not offenders, f"dynamic execution reintroduced in rust_bridge: {offenders}"


@pytest.mark.parametrize(
    ("expr", "reference"),
    [
        ("2.0*x**1.3/(1.0+0.7*x**2.1)", lambda v: 2.0 * v**1.3 / (1.0 + 0.7 * v**2.1)),
        ("x**2 + 3.5", lambda v: v**2 + 3.5),
        ("sqrt(x)*sin(x)", lambda v: np.sqrt(v) * np.sin(v)),
        ("exp(-x) + tanh(x)", lambda v: np.exp(-v) + np.tanh(v)),
        ("ln(x)", lambda v: np.log(v)),
        ("log(x) + x**2", lambda v: np.log(v) + v**2),
        ("min(x, 2.0)", lambda v: np.minimum(v, 2.0)),
        ("max(x, 1.0)", lambda v: np.maximum(v, 1.0)),
        ("abs(x - 1.5)", lambda v: np.abs(v - 1.5)),
        ("x**3 / (0.5 + 0.1*x**2)", lambda v: v**3 / (0.5 + 0.1 * v**2)),
    ],
)
def test_legitimate_expressions_match_numpy(bridge: RustKernelBridge, x, expr, reference):
    """The hardening must not cost numerical correctness."""
    result = bridge._fallback_eval(expr, ["x"], [x])
    assert np.all(np.isfinite(result)), expr
    np.testing.assert_allclose(result, reference(x), rtol=1e-10, atol=1e-12)


def test_multivariable_expression(bridge: RustKernelBridge, x):
    result = bridge._fallback_eval("a*b + 1", ["a", "b"], [x, x * 2.0])
    np.testing.assert_allclose(result, x * (x * 2.0) + 1.0, rtol=1e-12, atol=1e-12)


def test_scalar_constant_is_broadcast_to_the_data_shape(bridge: RustKernelBridge, x):
    result = bridge._fallback_eval("3.14", ["x"], [x])
    assert result.shape == x.shape
    np.testing.assert_allclose(result, np.full(x.shape, 3.14))


def test_malformed_expression_yields_nan_rather_than_raising(bridge: RustKernelBridge, x):
    for bad in ["", "   ", "x +", "((", "2 **"]:
        result = bridge._fallback_eval(bad, ["x"], [x])
        assert result.shape == x.shape
        assert _all_nan(result), bad


def test_non_string_input_is_rejected(bridge: RustKernelBridge, x):
    """A non-string must not reach the parser as something coercible."""
    assert _all_nan(bridge._fallback_eval(None, ["x"], [x]))  # type: ignore[arg-type]


def test_over_length_expression_is_rejected(bridge: RustKernelBridge, x):
    long_expr = "x" + "+x" * 5000
    assert _all_nan(bridge._fallback_eval(long_expr, ["x"], [x]))
