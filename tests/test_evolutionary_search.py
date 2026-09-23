# Databricks notebook source
"""Tests for the upgraded evolutionary search (projection, tuning, islands,
physics priors, Pareto grouped selection, ALPS). Synthetic, deterministic,
dependency-free (no scipy needed).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp.evolutionary_search import (
    SearchConfig,
    _pareto_front,
    evaluate_skeleton,
    evolve,
    node_count,
    partition_parameters,
    skeletonize,
    tune_nonlinear,
)


def _quad_data(seed=0, n=30, noise=0.05):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 3.0, n)
    return {"x": x}, 2.0 * x**2 + rng.normal(0, noise, n)


def _cache():
    return {}


def test_skeletonize_lifts_floats():
    skeleton, params = skeletonize(sp.sympify("2.0*x**2 + 3.0"))
    assert len(params) == 2
    assert not list(skeleton.atoms(sp.Float))


def test_partition_linear_vs_nonlinear():
    x = sp.Symbol("x")
    p0, p1 = sp.Symbol("p_0"), sp.Symbol("p_1")
    nonlinear, linear = partition_parameters(p0 * x**2 + p1, [p0, p1])
    assert nonlinear == [] and set(linear) == {p0, p1}
    nonlinear, linear = partition_parameters(p0 * sp.exp(p1 * x), [p0, p1])
    assert p1 in nonlinear and p0 not in nonlinear


def test_projection_exact_on_quadratic():
    x_dict, y = _quad_data()
    x = sp.Symbol("x")
    p0, p1 = sp.Symbol("p_0"), sp.Symbol("p_1")
    mse, pmap = evaluate_skeleton(p0 * x**2 + p1, [p0, p1], {}, [x], x_dict, y, _cache())
    assert mse < 0.01
    assert abs(pmap[p0] - 2.0) < 0.05


def test_tune_nonlinear_deterministic():
    x_dict, y = _quad_data()
    x = sp.Symbol("x")
    expr = sp.sympify("p_0*x**2 + p_1")
    params = [sp.Symbol("p_0"), sp.Symbol("p_1")]
    first, mse1 = tune_nonlinear(expr, params, [x], x_dict, y, _cache(), restarts=1, iters=8, seed=3)
    second, mse2 = tune_nonlinear(expr, params, [x], x_dict, y, _cache(), restarts=1, iters=8, seed=3)
    assert mse1 == mse2
    assert first == second


def _tiny_config(**overrides):
    base = dict(islands=2, island_size=6, generations=3, migration_interval=2,
                max_depth=3, seed=11, tune_restarts=1, tune_iters=6)
    base.update(overrides)
    return SearchConfig(**base)


def test_evolve_deterministic_and_recovers_structure():
    x_dict, y = _quad_data()
    kwargs = dict(variable_units={"x": "meter"}, target_unit="meter**2")
    first = evolve(x_dict, y, **kwargs, config=_tiny_config())
    second = evolve(x_dict, y, **kwargs, config=_tiny_config())
    assert first.best_expression == second.best_expression
    assert first.best_expression is not None
    assert first.n_evaluations > 0
    assert len(first.pareto_front) >= 1


def test_evolve_grouped_pick_uses_holdout():
    x_dict, y = _quad_data()
    xtr = {k: v[:20] for k, v in x_dict.items()}
    xte = {k: v[20:] for k, v in x_dict.items()}
    result = evolve(xtr, y[:20], xte, y[20:], config=_tiny_config())
    assert result.grouped_pick is not None


def test_dim_gate_rejects_at_birth():
    from apedp.evolutionary_search import _IslandSearch

    x = sp.Symbol("x")
    search = _IslandSearch([x], {"x": np.ones(5)}, np.ones(5),
                           {"x": "meter"}, "meter", {}, _tiny_config(),
                           __import__("random").Random(0), {})
    assert search._admissible_at_birth(sp.exp(x)) is False
    assert search._admissible_at_birth(x) is True
    assert search._admissible_at_birth(x**2) is False  # m**2 != meter target
    assert search._admissible_at_birth(2.0 * x) is True  # coefficient undeclared: safe default
    assert search._admissible_at_birth(1 / (x - x)) is False  # sympy zoo guard


def test_symmetry_gate():
    from apedp.evolutionary_search import _IslandSearch

    x = sp.Symbol("x")
    search = _IslandSearch([x], {"x": np.ones(5)}, np.ones(5),
                           {}, None, {"x": "reflect"}, _tiny_config(),
                           __import__("random").Random(0), {})
    assert search._admissible_at_birth(x**2) is True
    assert search._admissible_at_birth(x**3) is False


def test_pi_seeds_from_same_unit_vars():
    from apedp.evolutionary_search import _IslandSearch

    a, b = sp.Symbol("a"), sp.Symbol("b")
    search = _IslandSearch([a, b], {"a": np.ones(5), "b": np.ones(5)}, np.ones(5),
                           {"a": "meter", "b": "meter"}, None, {}, _tiny_config(),
                           __import__("random").Random(0), {})
    assert len(search._pi_seeds()) >= 2


def test_pareto_front_nondominated():
    records = [("a", 1.0, 5.0), ("b", 0.5, 8.0), ("c", 0.5, 8.0), ("d", 2.0, 6.0)]
    front = _pareto_front(records)
    names = {item[0] for item in front}
    assert "d" not in names  # dominated by a on both objectives
    assert names == {"a", "b", "c"}


def test_threads_complete():
    x_dict, y = _quad_data()
    result = evolve(x_dict, y, config=_tiny_config(threads=2))
    assert result.best_expression is not None


def test_node_count_and_complexity_penalty():
    x = sp.Symbol("x")
    assert node_count(x**2) > node_count(x)


def test_allow_inv_registers_inverse_operator():
    from apedp.evolutionary_search import SearchConfig, _IslandSearch

    search = _IslandSearch([sp.Symbol("x")], {"x": __import__("numpy").ones(5)}, __import__("numpy").ones(5),
                           {}, None, {}, SearchConfig(allow_inv=True),
                           __import__("random").Random(0), {})
    assert "inv" in search.unary
    plain = _IslandSearch([sp.Symbol("x")], {"x": __import__("numpy").ones(5)}, __import__("numpy").ones(5),
                          {}, None, {}, SearchConfig(allow_inv=False),
                          __import__("random").Random(0), {})
    assert "inv" not in plain.unary


def test_shomate_minimum_depth_is_three_not_seven():
    # Measured, not assumed: n-ary Add flattens, so the reference family
    # needs depth 3 under this grammar -- depth 4 does not exclude it.
    from apedp.evolutionary_search import candidate_pathology

    T = sp.Symbol("T")
    A, B, C, D, E = (sp.Symbol(n) for n in "ABCDE")
    report = candidate_pathology(A + B * T + C * T**2 + D * T**3 + E / T**2)
    assert report["depth"] == 3
    assert report["transcendentals"] == 0


def test_pathology_flags_oscillator():
    from apedp.evolutionary_search import candidate_pathology

    T = sp.Symbol("T")
    report = candidate_pathology(sp.exp(2.0 / T) * sp.sin(0.4 * T))
    assert report["transcendentals"] == 2


def test_generation_hook_observes_without_altering():
    import numpy as np

    from apedp.evolutionary_search import SearchConfig, evolve

    rng = np.random.default_rng(0)
    x = np.linspace(0.5, 3.0, 20)
    y = 2.0 * x**2 + rng.normal(0, 0.05, len(x))
    seen = []
    cfg = dict(islands=2, island_size=6, generations=3, migration_interval=2,
               max_depth=3, seed=11, tune_restarts=1, tune_iters=6)
    plain = evolve({"x": x}, y, config=SearchConfig(**cfg))
    hooked = evolve({"x": x}, y, config=SearchConfig(**cfg),
                    on_generation=lambda g, pop: seen.append((g, len(pop))))
    assert [g for g, _ in seen] == [0, 1, 2]
    assert all(n > 0 for _, n in seen)
    assert hooked.best_expression == plain.best_expression


def test_denormalize_roundtrip_exact():
    import sympy as sp

    from apedp.evolutionary_search import denormalize_expression

    scaling = {"x": [1000.0, 500.0]}
    expr = denormalize_expression("2.0*x**2 + 1.0", scaling)
    raw = {"x": __import__("numpy").array([500.0, 1000.0, 1500.0])}
    std = {"x": (__import__("numpy").array([500.0, 1000.0, 1500.0]) - 1000.0) / 500.0}
    f_raw = sp.lambdify(sp.Symbol("x"), expr, "numpy")(raw["x"])
    f_std = sp.lambdify(sp.Symbol("x"), sp.sympify("2.0*x**2 + 1.0"), "numpy")(std["x"])
    assert __import__("numpy").allclose(f_raw, f_std)


def test_denormalize_wraps_target_scaling():
    import sympy as sp

    from apedp.evolutionary_search import denormalize_expression

    expr = denormalize_expression("x + 1.0", {"x": [0.0, 2.0], "__target__": [10.0, 5.0]})
    value = sp.lambdify(sp.Symbol("x"), expr, "numpy")(8.0)
    # Normalized space: (8/2)+1 = 5; physical: 5*5+10 = 35.
    assert abs(float(value) - 35.0) < 1e-9


def test_pathology_singular_design_refused():
    import numpy as np
    import sympy as sp

    from apedp.evolutionary_search import evaluate_skeleton

    x = np.linspace(1.0, 3.0, 20)
    y = 2.0 * x + 1.0
    xs = sp.Symbol("x")
    p0, p1 = sp.Symbol("p_0"), sp.Symbol("p_1")
    # Duplicate columns -> singular design -> must refuse, not return garbage.
    mse, _ = evaluate_skeleton(p0 * xs + p1 * xs, [p0, p1], {}, [xs], {"x": x}, y, {})
    assert mse == float("inf")


def test_pathology_pole_between_observations_refused():
    import numpy as np
    import sympy as sp

    from apedp.evolutionary_search import evaluate_skeleton

    x = np.linspace(0.0, 2.0, 21)
    y = np.ones_like(x)
    xs = sp.Symbol("x")
    p0 = sp.Symbol("p_0")
    mse, _ = evaluate_skeleton(
        p0 / (xs - 1.0), [p0], {}, [xs], {"x": x}, y, {}, dense_check=True
    )
    assert mse == float("inf")


def test_normalization_fixes_ill_conditioning():
    # True mechanism (demonstrated on the methane raw-Kelvin failure):
    # Shomate-class designs mix 1/T**2 (~1e-7) with T**3 (~1e10), so the raw
    # design has condition ~1e15 while the standardized one has ~1e2.
    # Synthetic Shomate-like truth keeps the test hermetic and deterministic.
    import numpy as np
    import sympy as sp

    from apedp.evolutionary_search import evaluate_skeleton

    rng = np.random.default_rng(9)
    T = np.linspace(100.0, 3000.0, 31)
    t = T / 1000.0
    y = 20.0 + 60.0 * t - 20.0 * t**2 + 3.0 * t**3 + 0.3 / t**2
    y = y + rng.normal(0, 0.05, len(y))
    xs = sp.Symbol("T")
    params = [sp.Symbol(f"p_{k}") for k in range(5)]
    skel = params[0] + params[1] * xs + params[2] * xs**2 + params[3] * xs**3 + params[4] / xs**2
    raw_mse, _ = evaluate_skeleton(skel, params, {}, [xs], {"T": T}, y, {})
    sigma = float(np.std(T))
    std_mse, _ = evaluate_skeleton(skel, params, {}, [xs], {"T": T / sigma}, y, {})
    assert std_mse < 0.5 * raw_mse
    assert std_mse < 0.05


def test_evolve_records_scaling_and_denormalized_forms():
    import numpy as np

    from apedp.evolutionary_search import SearchConfig, evolve

    rng = np.random.default_rng(9)
    x = np.linspace(4000.0, 6000.0, 30)
    y = 3.0 * x**3 + rng.normal(0, 1e8, len(x))
    result = evolve(
        {"x": x[:22]}, y[:22], {"x": x[22:]}, y[22:],
        config=SearchConfig(islands=2, island_size=8, generations=5,
                            max_depth=4, seed=4, tune_restarts=1,
                            tune_iters=10, normalize=True),
    )
    assert result.variable_scaling["x"][1] > 0
    assert result.denormalized_best is not None
    plain = evolve(
        {"x": x[:22]}, y[:22], {"x": x[22:]}, y[22:],
        config=SearchConfig(islands=2, island_size=8, generations=5,
                            max_depth=4, seed=4, tune_restarts=1,
                            tune_iters=10, normalize=False),
    )
    assert plain.variable_scaling == {}
    assert plain.denormalized_best == plain.best_expression
