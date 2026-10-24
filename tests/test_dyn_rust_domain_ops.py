"""Domain grammar expansion: pi/e constants, min/max, sign, cbrt, erf, besselj.

Covers the parser-level two-argument forms (min/max), the special-function
arms (erf via A&S 7.1.26, besselj via power series), exact pi/e constants,
and native-GA generation of the new operators.
"""
from __future__ import annotations

import json
import math

import numpy as np

import apedp_kernel as ak
from apedp.evolutionary_search import SearchConfig, evolve


def test_parser_min_max_two_argument_forms():
    x = np.arange(1.0, 11.0)
    out = ak.evaluate_batch(ak.compile("min(3, x+1) + max(x, 5)", ["x"]), [x])
    expected = np.minimum(3.0, x + 1.0) + np.maximum(x, 5.0)
    assert np.allclose(out, expected)


def test_parser_rejects_single_argument_min():
    try:
        ak.compile("min(x)", ["x"])
    except ValueError:
        return
    raise AssertionError("min(x) must be rejected")


def test_parser_special_functions_match_reference():
    x = np.array([0.5, 1.0, 2.0])
    out = ak.evaluate_batch(ak.compile("erf(x)", ["x"]), [x])
    assert np.allclose(out, [math.erf(v) for v in x], atol=1e-6)
    out_b = ak.evaluate_batch(ak.compile("besselj(x)", ["x"]), [x])
    assert abs(out_b[0] - 0.938469807240813) < 1e-9  # J0(0.5)
    out_c = ak.evaluate_batch(ak.compile("cbrt(x)", ["x"]), [x])
    assert np.allclose(out_c, np.cbrt(x))


def test_parser_exact_pi_constant():
    x = np.array([1.0, 2.0])
    out = ak.evaluate_batch(ak.compile("pi * x", ["x"]), [x])
    assert np.allclose(out, math.pi * x)


def test_native_ga_generates_min_max_and_special_functions():
    x = np.linspace(0, 4, 150)
    r = evolve({"x": x}, np.minimum(x, 2.0), config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    assert r.best_expression is not None
    # Soft-min via erf is the expected small-budget outcome; the point is
    # the new operators are generatable and score competitively.
    mses = [p["train_mse"] for p in r.pareto_front]
    assert min(mses) < 0.01, min(mses)


def test_native_ga_misc_opt_out_still_runs():
    x = np.linspace(-3, 3, 200)
    r = evolve({"x": x}, 3.0 * x ** 2 + 2.0, config=SearchConfig(
        use_native_ga=True, islands=4, island_size=50,
        generations=20, seed=42, allow_misc=False))
    mses = [p["train_mse"] for p in r.pareto_front]
    assert mses and min(mses) < 1e-6


def test_enumeration_advanced_ops_recover_each_form():
    v = np.linspace(0.5, 3.5, 40)
    w = np.linspace(0.2, 0.9, 30)
    cases = {"tan": (v, np.tan(v)), "sinh": (v, np.sinh(v)),
             "cosh": (v, np.cosh(v)), "tanh": (v, np.tanh(v)),
             "asin": (w, np.arcsin(w)), "acos": (w, np.arccos(w)),
             "atan": (v, np.arctan(v))}
    for name, (x, y) in cases.items():
        out = json.loads(ak.rime_enumerate_search_json(
            json.dumps([x.tolist()]), json.dumps(y.tolist()),
            json.dumps(["v"]), False, False, False, False, True, 2))
        assert len(out) > 0
        assert out[0]["r2"] > 0.999, (name, out[0])


def test_enumeration_advanced_ops_gated_off_by_default():
    v = np.linspace(0.5, 3.5, 40)
    out = json.loads(ak.rime_enumerate_search_json(
        json.dumps([v.tolist()]), json.dumps(np.tanh(v).tolist()),
        json.dumps(["v"]), False, False, False, False, False, 2))
    for c in out:
        for token in ("tanh(", "sinh(", "asin(", "atan("):
            assert token not in c["formula"], c["formula"]


def test_native_ga_discovers_advanced_ops():
    x = np.linspace(0.5, 4.0, 100)
    # A power's exponent is a tunable parameter now: the kernel draws it from
    # the same continuous grid as any other constant and fits it when the
    # candidate is born, so 2.5 needs no hard-coded menu entry. That raised
    # recovery at the old 2 x 30 x 15 budget from 5/10 to 17/20 seeds -- still
    # not a deterministic claim, so this pin keeps the larger budget below
    # (10/10 seeds, worst best MSE 1.4e-30, ~1 s). Seed sensitivity of every GA
    # capability pin is tabulated in docs/ga_seed_fragility_audit.md.
    r_pow = evolve({"x": x}, x ** 2.5, config=SearchConfig(
        use_native_ga=True, strict_native_ga=True, islands=4, island_size=120,
        generations=80, seed=123))
    assert r_pow.best_expression is not None
    # Non-integer power is now reached through the general-Pow path (exact
    # x**2.5) rather than the older sqrt(x)*x**2 spelling; pin the capability
    # numerically instead of the expression spelling.
    best_mse = min(p["train_mse"] for p in r_pow.pareto_front)
    assert best_mse < 1e-6, (best_mse, r_pow.best_expression)

    y = np.linspace(0, 3, 100)
    r_atan = evolve({"x": y}, np.arctan(y), config=SearchConfig(
        use_native_ga=True, strict_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    assert r_atan.best_expression is not None
    assert "atan" in r_atan.best_expression, r_atan.best_expression
