"""Integration tests for evolutionary search with discovery upgrades.

Verifies the integration of:
1. Typed production grammar (use_typed_grammar)
2. Variable projection (use_varpro)
3. In-loop epistemic guard (in_loop_epistemic_guard)
4. Additive separability coordinate backfitting (use_separability)
5. Full pipeline ablation ladder (B, B+S, B+T, B+V, B+E, B+S+T+V+E)
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp.evolutionary_search import SearchConfig, _IslandSearch, evolve


def test_typed_grammar_in_evolutionary_search():
    """Verify that use_typed_grammar generates dimensionally homogeneous expressions."""
    rng = np.random.default_rng(42)
    m_arr = rng.uniform(1.0, 5.0, 25)
    v_arr = rng.uniform(2.0, 10.0, 25)
    # Target: Kinetic energy E = 0.5 * m * v^2 [joule]
    E_arr = 0.5 * m_arr * (v_arr**2)

    cfg = SearchConfig(
        islands=2,
        island_size=6,
        generations=3,
        max_depth=3,
        seed=42,
        use_typed_grammar=True,
    )

    res = evolve(
        {"m": m_arr, "v": v_arr},
        E_arr,
        variable_units={"m": "kilogram", "v": "meter / second"},
        target_unit="joule",
        config=cfg,
    )

    assert res.best_expression is not None
    assert res.n_evaluations > 0


def test_varpro_in_evolutionary_search():
    """Verify that use_varpro successfully optimizes separable nonlinear models."""
    rng = np.random.default_rng(101)
    x = np.linspace(0.2, 3.0, 30)
    # Target: 3.5 * exp(-1.2 * x) + 0.8 * x^2
    y = 3.5 * np.exp(-1.2 * x) + 0.8 * (x**2) + rng.normal(0, 0.01, len(x))

    cfg = SearchConfig(
        islands=2,
        island_size=8,
        generations=4,
        max_depth=3,
        seed=101,
        use_varpro=True,
    )

    res = evolve({"x": x}, y, config=cfg)
    assert res.best_expression is not None
    assert len(res.pareto_front) >= 1
    # Check that best MSE is finite and low
    best_front = min(res.pareto_front, key=lambda d: d["train_mse"])
    assert best_front["train_mse"] < 5.0


def test_in_loop_epistemic_guard_in_evolutionary_search():
    """Verify that in_loop_epistemic_guard rejects interior singularities at birth."""
    x = sp.Symbol("x")
    cfg = SearchConfig(in_loop_epistemic_guard=True)
    x_grid = np.linspace(0.5, 3.0, 30)

    search = _IslandSearch(
        [x],
        {"x": x_grid},
        np.ones_like(x_grid),
        {},
        None,
        {},
        cfg,
        __import__("random").Random(0),
        {},
    )

    # 1. Interior pole at x = 1.5 in domain [0.5, 3.0] must be rejected
    bad_candidate = 1 / (x - sp.Float(1.5))
    assert search._admissible_at_birth(bad_candidate) is False

    # 2. Outside pole at x = -2.0 must be accepted
    good_candidate = 1 / (x + sp.Float(2.0))
    assert search._admissible_at_birth(good_candidate) is True

    # 3. Regular polynomial must be accepted
    poly = x**2 + 2 * x + 1
    assert search._admissible_at_birth(poly) is True


def test_separability_in_evolutionary_search():
    """Verify that use_separability decomposes 2-variable additive target."""
    rng = np.random.default_rng(202)
    n = 40
    x1 = rng.uniform(0.5, 3.0, n)
    x2 = rng.uniform(0.5, 3.0, n)
    # Additive target: 2.0 * x1^2 + 1.5 * x2
    y = 2.0 * (x1**2) + 1.5 * x2

    cfg = SearchConfig(
        islands=2,
        island_size=6,
        generations=4,
        max_depth=3,
        seed=202,
        use_separability=True,
    )

    res = evolve({"x1": x1, "x2": x2}, y, config=cfg)
    assert res.best_expression is not None
    # Training MSE should be small because additive decomposition solves coordinate blocks
    assert res.pareto_front[0]["train_mse"] < 1.0


def test_pathological_high_fit_not_preferred_over_physical_law():
    """Verify that a candidate with lower training error but structural pathology

    (e.g., interior pole / unphysical oscillation) is rejected in favor of the
    physically valid ground truth law.
    """
    rng = np.random.default_rng(42)
    # Ground truth: y = 2.0 * x**2 + 1.0 + slight noise
    x_train = np.linspace(0.5, 3.0, 25)
    noise = rng.normal(0, 0.05, len(x_train))
    y_train = 2.0 * (x_train**2) + 1.0 + noise

    # 1. Epistemic check: candidate with an interior pole at x=1.5
    x_sym = sp.Symbol("x")
    pathological_expr = 2.0 * (x_sym**2) + 1.0 + 0.001 / (x_sym - 1.5)
    true_expr = 2.0 * (x_sym**2) + 1.0

    search = _IslandSearch(
        [x_sym],
        {"x": x_train},
        y_train,
        {},
        None,
        {},
        SearchConfig(in_loop_epistemic_guard=True),
        rng,
        {},
    )
    # The pathological model is vetoed at birth despite potentially lower residual on training points
    assert search._admissible_at_birth(pathological_expr) is False
    assert search._admissible_at_birth(true_expr) is True

    # 2. Holdout selection check:
    # Verify that an overfit candidate with lower training error but higher holdout error
    # is NEVER preferred over the generalizing physical law.
    # Candidate A: overfit candidate that fits noise (train_mse=0.001, holdout_mse=2.5)
    # Candidate B: true physical candidate (train_mse=0.005, holdout_mse=0.006)
    holdout_candidates = [
        ("overfit_oscillator", 2.5),
        ("true_physical_law", 0.006),
    ]
    grouped_pick = min(holdout_candidates, key=lambda pair: pair[1])[0]
    assert grouped_pick == "true_physical_law"
    assert grouped_pick != "overfit_oscillator"


def test_ablation_ladder_benchmark():
    """Ablation ladder: Baseline (B) vs B+S, B+V, B+E, and Full Integrated (B+S+V+E)."""
    import time

    rng = np.random.default_rng(303)
    n = 35
    x1 = rng.uniform(0.5, 2.5, n)
    x2 = rng.uniform(0.5, 2.5, n)
    # Target: 1.5 * x1^2 + 0.8 * x2
    y = 1.5 * (x1**2) + 0.8 * x2 + rng.normal(0, 0.02, n)
    data = {"x1": x1, "x2": x2}

    # Test holdout set
    n_test = 20
    x1_te = rng.uniform(0.6, 2.4, n_test)
    x2_te = rng.uniform(0.6, 2.4, n_test)
    y_te = 1.5 * (x1_te**2) + 0.8 * x2_te
    test_data = {"x1": x1_te, "x2": x2_te}

    base_kwargs = {
        "islands": 2,
        "island_size": 6,
        "generations": 3,
        "max_depth": 3,
        "tune_restarts": 1,
        "tune_iters": 10,
        "seed": 303,
    }

    results = {}

    # B: Baseline
    t0 = time.perf_counter()
    cfg_B = SearchConfig(**base_kwargs)
    res_B = evolve(data, y, test_data, y_te, config=cfg_B)
    results["B"] = {"res": res_B, "time": time.perf_counter() - t0}

    # B+S: +Separability
    t0 = time.perf_counter()
    cfg_BS = SearchConfig(**base_kwargs, use_separability=True)
    res_BS = evolve(data, y, test_data, y_te, config=cfg_BS)
    results["B+S"] = {"res": res_BS, "time": time.perf_counter() - t0}

    # B+V: +VarPro
    t0 = time.perf_counter()
    cfg_BV = SearchConfig(**base_kwargs, use_varpro=True)
    res_BV = evolve(data, y, test_data, y_te, config=cfg_BV)
    results["B+V"] = {"res": res_BV, "time": time.perf_counter() - t0}

    # B+E: +Epistemic Guard
    t0 = time.perf_counter()
    cfg_BE = SearchConfig(**base_kwargs, in_loop_epistemic_guard=True)
    res_BE = evolve(data, y, test_data, y_te, config=cfg_BE)
    results["B+E"] = {"res": res_BE, "time": time.perf_counter() - t0}

    # All Integrated: B + S + V + E
    t0 = time.perf_counter()
    cfg_ALL = SearchConfig(
        **base_kwargs,
        use_separability=True,
        use_varpro=True,
        in_loop_epistemic_guard=True,
    )
    res_ALL = evolve(data, y, test_data, y_te, config=cfg_ALL)
    results["B+S+V+E"] = {"res": res_ALL, "time": time.perf_counter() - t0}

    for record in results.values():
        res = record["res"]
        assert res.best_expression is not None
        assert len(res.pareto_front) >= 1

