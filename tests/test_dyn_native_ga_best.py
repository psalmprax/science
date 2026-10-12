"""Native GA best-selection pins: best means minimum MSE, never index 0.

Regression test for the complexity-ordered frontier bug: the Rust engine
returns the Pareto frontier sorted by complexity, so taking element 0
reported the simplest candidate (the mean constant) instead of the best.
Fast (native path, no scipy).
"""

import numpy as np

from apedp.evolutionary_search import SearchConfig, evolve


def test_native_best_is_minimum_mse_recovery():
    x = np.linspace(-3, 3, 200)
    y = 3.0 * x ** 2 + 2.0
    res = evolve({"x": x}, y, config=SearchConfig(
        use_native_ga=True, islands=4, island_size=50,
        generations=20, seed=42))
    mses = [p["train_mse"] for p in res.pareto_front]
    assert mses, "empty Pareto frontier"
    best_mse = min(mses)
    assert best_mse < 1e-6, f"native GA failed to recover x^2 (best MSE {best_mse})"
    # best_expression must be the min-MSE entry, not frontier position 0.
    entry = next(p for p in res.pareto_front if p["train_mse"] == best_mse)
    assert res.best_expression == entry["expression"]
    assert "x" in res.best_expression
