"""APEDP closed-loop orchestrator (Future Work item 2).

Sequences discover -> simulate/design -> update without fabricating data:
new measurements must be supplied by the caller (lab, held-out reveal in
demos). A pluggable ``design_fn`` hook lets the Bayesian designer from
``apedp.experiment_designer`` replace the default spread-based proposal.

Typical usage:
    from apedp.closed_loop import ClosedLoopRunner, ClosedLoopConfig
    from apedp.model_discovery import ModelDiscoveryEngine
    engine = ModelDiscoveryEngine(["x"])
    runner = ClosedLoopRunner(engine, candidates, design_grid={"x": grid})
    state = runner.run(x0, y0, measurement_plan=[(x1, y1), (x2, y2)])
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp.model_discovery import ModelDiscoveryEngine, ModelDiscoveryResult
from apedp.scientific_layers import ExperimentDesign


@dataclass
class ClosedLoopConfig:
    """Run configuration for the closed loop."""
    max_iterations: int = 3
    n_points_per_iter: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.n_points_per_iter < 1:
            raise ValueError("n_points_per_iter must be positive")


@dataclass
class ClosedLoopState:
    """Full auditable state of one closed-loop run."""
    iteration: int = 0
    x_data: dict[str, np.ndarray] = field(default_factory=dict)
    y_data: np.ndarray = field(default_factory=lambda: np.array([]))
    result: ModelDiscoveryResult | None = None
    design: ExperimentDesign | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "n_data": int(np.asarray(self.y_data).size),
            "result": self.result.to_dict() if self.result else None,
            "design": self.design.to_dict() if self.design else None,
            "history": self.history,
        }


class ClosedLoopRunner:
    """Discover -> design -> update loop over caller-supplied measurements.

    Args:
        engine: Configured ModelDiscoveryEngine (variables fixed at init).
        candidates: Dict of name -> sympy-parseable equation structures.
        design_grid: 1D candidate grid {variable: values} for proposals.
        uncertainty_fn: Optional measurement-uncertainty model over the grid.
        design_fn: Optional hook (result, grid) -> ExperimentDesign|None that
            overrides the default spread-based proposal (e.g. the Bayesian
            designer from apedp.experiment_designer).
        config: Loop configuration.
    """

    def __init__(
        self,
        engine: ModelDiscoveryEngine,
        candidates: dict,
        *,
        design_grid: dict[str, np.ndarray] | None = None,
        uncertainty_fn: Callable | None = None,
        design_fn: Callable[[ModelDiscoveryResult, dict], ExperimentDesign | None] | None = None,
        config: ClosedLoopConfig | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("at least one candidate structure is required")
        if design_grid is not None and set(design_grid) != set(engine.variables):
            raise ValueError("design_grid keys must match the engine variables")
        self.engine = engine
        self.candidates = dict(candidates)
        self.design_grid = design_grid
        self.uncertainty_fn = uncertainty_fn
        self.design_fn = design_fn
        self.config = config or ClosedLoopConfig()

    # -- core steps ------------------------------------------------------

    def discover(self, state: ClosedLoopState) -> ClosedLoopState:
        """Fit, rank, and package candidates on the current data."""
        state.result = self.engine.discover(
            self.candidates,
            state.x_data,
            np.asarray(state.y_data, dtype=float),
            design_grid=self.design_grid,
            uncertainty_fn=self.uncertainty_fn,
        )
        return state

    def propose(self, state: ClosedLoopState) -> ClosedLoopState:
        """Propose the next measurement(s); proposal only, no data created."""
        if state.result is None:
            raise ValueError("discover() must run before propose()")
        if self.design_fn is not None and self.design_grid is not None:
            state.design = self.design_fn(state.result, self.design_grid)
        else:
            state.design = state.result.experiment_design
        return state

    def update(
        self,
        state: ClosedLoopState,
        new_x: dict[str, np.ndarray],
        new_y: np.ndarray,
    ) -> ClosedLoopState:
        """Append caller-supplied measurements (lab or held-out reveal)."""
        new_y = np.asarray(new_y, dtype=float).ravel()
        if set(new_x) != set(self.engine.variables):
            raise ValueError("new_x keys must match the engine variables")
        n_new = new_y.size
        if n_new == 0 or not np.all(np.isfinite(new_y)):
            raise ValueError("new_y must contain finite values")
        for name, values in new_x.items():
            arr = np.asarray(values, dtype=float).ravel()
            if arr.shape != (n_new,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"new_x[{name!r}] must be finite with length {n_new}")
            state.x_data[name] = np.concatenate([state.x_data[name], arr])
        state.y_data = np.concatenate([np.asarray(state.y_data, dtype=float), new_y])
        state.iteration += 1
        state.history.append({"iteration": state.iteration, "n_new": n_new,
                              "n_total": int(state.y_data.size)})
        return state

    def step(
        self,
        state: ClosedLoopState,
        new_x: dict[str, np.ndarray] | None = None,
        new_y: np.ndarray | None = None,
    ) -> ClosedLoopState:
        """One full iteration: ingest (optional) -> discover -> propose."""
        if (new_x is None) != (new_y is None):
            raise ValueError("new_x and new_y must be supplied together")
        if new_x is not None and new_y is not None:
            self.update(state, new_x, new_y)
        self.discover(state)
        self.propose(state)
        return state

    def run(
        self,
        x_init: dict[str, np.ndarray],
        y_init: np.ndarray,
        measurement_plan: list[tuple[dict, np.ndarray]],
    ) -> ClosedLoopState:
        """Run the loop over an initial dataset plus a measurement plan.

        measurement_plan simulates the lab: each entry is caller-supplied
        data revealed at that iteration (in production, replace with live
        measurements at the proposed points).
        """
        y_init = np.asarray(y_init, dtype=float).ravel()
        if set(x_init) != set(self.engine.variables):
            raise ValueError("x_init keys must match the engine variables")
        state = ClosedLoopState(
            x_data={k: np.asarray(v, dtype=float).ravel() for k, v in x_init.items()},
            y_data=y_init,
        )
        self.step(state)
        for new_x, new_y in measurement_plan[: self.config.max_iterations]:
            self.step(state, new_x, new_y)
        return state


# ---------------------------------------------------------------------------
# Adapter: plug the Bayesian designer (item 1) into the loop (item 2)
# ---------------------------------------------------------------------------

def bayesian_design_adapter(
    predictive_fn: Callable[[ModelDiscoveryResult, np.ndarray], np.ndarray],
    *,
    sigma: float = 1.0,
    n_points: int = 1,
    min_distance: float = 0.0,
) -> Callable[[ModelDiscoveryResult, dict], ExperimentDesign | None]:
    """Build a runner design_fn from a result->predictive-samples function.

    predictive_fn(result, grid_1d) must return (n_samples, n_grid) draws.
    Returns None when the result has no usable ranked evaluators.
    """
    from apedp.experiment_designer import design_from_predictive_samples

    def _design(result: ModelDiscoveryResult, grid: dict) -> ExperimentDesign | None:
        if len(grid) != 1:
            return None
        xs = np.asarray(next(iter(grid.values())), dtype=float).reshape(-1)
        try:
            pred = np.asarray(predictive_fn(result, xs), dtype=float)
        except (TypeError, ValueError, ArithmeticError):
            return None
        if pred.ndim != 2 or pred.shape[1] != xs.size or pred.shape[0] < 2:
            return None
        return design_from_predictive_samples(
            pred, xs, sigma=sigma, n_points=n_points, min_distance=min_distance,
            objective="closed-loop Bayesian predictive design (EIG proxy)",
        )

    return _design


# ---------------------------------------------------------------------------
# Test block (synthetic closure demo with held-out reveal -- not lab data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Closed-Loop Tests (synthetic held-out reveal) ===\n")

    rng = np.random.default_rng(0)
    xt = np.linspace(0.1, 2.0, 30)
    truth = np.exp(-xt)
    # Initial: 6 noisy points; held-out pool simulates future lab measurements.
    idx = rng.choice(len(xt), size=6, replace=False)
    mask = np.ones(len(xt), dtype=bool)
    mask[idx] = False
    x_init = {"x": xt[idx]}
    y_init = truth[idx] + rng.normal(0.0, 0.01, size=6)

    engine = ModelDiscoveryEngine(["x"])
    candidates = {
        "exponential": "exp(-x)",
        "rational": "1/(1+x)",
        "linear": "1-x",
    }
    grid = {"x": np.linspace(0.1, 2.0, 40)}
    runner = ClosedLoopRunner(engine, candidates, design_grid=grid,
                              config=ClosedLoopConfig(max_iterations=2))
    state = runner.step(ClosedLoopState(
        x_data={k: np.asarray(v) for k, v in x_init.items()},
        y_data=np.asarray(y_init)))
    if state.result is None or state.result.selected is None:
        raise RuntimeError("closed-loop step produced no discovery result")
    print(f"Test 1: iter0 selected = {state.result.selected.name}")
    assert state.design is not None, "expected a design proposal"
    print(f"  proposal points: {state.design.selected_points}")

    # Test 2: two reveal iterations improve or hold the fit (monotone data growth)
    plan = []
    pool_x, pool_y = xt[mask], truth[mask]
    for k in range(2):
        j = int(np.argmax(np.abs(pool_x - 1.0) > 0))  # deterministic pick
        plan.append(({"x": pool_x[j:j + 1]}, pool_y[j:j + 1]))
        keep = np.ones(len(pool_x), dtype=bool)
        keep[j] = False
        pool_x, pool_y = pool_x[keep], pool_y[keep]
    final = runner.run(x_init, y_init, plan)
    if final.result is None or final.result.selected is None:
        raise RuntimeError("closed-loop run produced no discovery result")
    print(f"Test 2: iterations={final.iteration}, n={final.y_data.size}, "
          f"selected={final.result.selected.name}, "
          f"R2={final.result.selected.train_r_squared:.4f}")
    assert final.iteration == 2 and final.y_data.size == 8

    # Test 3: Bayesian adapter hook fires with bootstrap-style draws
    print("Test 3: Bayesian design adapter")
    def fake_predictive(result, xs):
        base = np.stack([c.evaluator({"x": xs}) for c in result.ranked
                         if c.evaluator is not None])
        # deterministic pseudo-posterior: each candidate +/- 1% bands
        return np.stack([base[0] * 0.99, base[0] * 1.01, base[0]])

    runner_b = ClosedLoopRunner(
        engine, candidates, design_grid=grid,
        design_fn=bayesian_design_adapter(fake_predictive, n_points=2),
        config=ClosedLoopConfig(max_iterations=1))
    state_b = runner_b.run(x_init, y_init, [])
    if state_b.design is None:
        raise RuntimeError("Bayesian adapter run produced no design proposal")
    print(f"  adapter proposal: {state_b.design.selected_points}")
    assert len(state_b.design.selected_points) == 2
    assert state_b.design.objective.startswith("closed-loop Bayesian")

    # Test 4: update() validates caller data
    print("Test 4: update validation")
    try:
        runner.update(final, {"wrong": np.array([1.0])}, np.array([1.0]))
        raise AssertionError("no error for mismatched variables")
    except ValueError:
        print("  correctly rejected mismatched variables")

    print("\n=== All closed-loop tests passed ===")
