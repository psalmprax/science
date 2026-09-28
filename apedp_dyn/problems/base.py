"""D1 benchmark problems: shared scaffolding.

A DynProblem bundles the contract objects (states/inputs/parameters), the
ground-truth equations, a train regime, and a held-out intervention regime
with high-accuracy ground-truth trajectories. Discovery sees train only;
gate 4 (intervention_holdout) scores against holdout truth.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable


def _vector_signal(components: list[Callable[[float], np.ndarray]]) -> Callable[[float], np.ndarray]:
    """Concatenate per-input callables into one length-n_inputs vector."""
    def u(t: float) -> np.ndarray:
        return np.concatenate([np.atleast_1d(np.asarray(c(t), dtype=float)).ravel()
                               for c in components])
    return u


def make_signal(spec: Any) -> Callable[[float], np.ndarray]:
    """Materialize a serializable input-signal spec to u(t).

    One input: a dict spec. Several inputs: a list/tuple of specs, or
    ``{"kind": "vector", "signals": [...]}``. The result is always a vector of
    length n_inputs, so multi-DOF problems describe excitation declaratively
    instead of hand-rolling lambdas — which also keeps a problem spec
    serializable, so a run can be replayed from its recorded experiment design.
    """
    if isinstance(spec, (list, tuple)):
        return _vector_signal([make_signal(s) for s in spec])
    if not isinstance(spec, dict):
        raise TypeError(
            f"signal spec must be a dict or a list of dicts, got {type(spec).__name__}")
    kind = spec["kind"]
    if kind == "vector":
        subs = spec.get("signals")
        if not isinstance(subs, (list, tuple)) or not subs:
            raise ValueError("vector signal needs a non-empty 'signals' list")
        return _vector_signal([make_signal(s) for s in subs])
    if kind == "zero":
        # 'n' lets a multi-input problem declare an all-zero drive of the
        # right arity without repeating the spec per channel.
        n = int(spec.get("n", 1))
        return lambda t: np.zeros(n)
    if kind == "sine":
        a, w, ph = spec["amplitude"], spec["frequency"], spec.get("phase", 0.0)
        return lambda t: np.array([a * np.sin(w * t + ph)])
    if kind == "step":
        a, t0 = spec["amplitude"], spec.get("t0", 1.0)
        return lambda t: np.array([a if t >= t0 else 0.0])
    if kind == "multisine":
        # Persistent excitation: sum of separated sinusoids (ported from
        # apedp/dyn experiment design — a single tone near resonance leaves
        # x(t) and u(t) nearly collinear and the regression ill-conditioned).
        comps = [(c[0], c[1], c[2] if len(c) > 2 else 0.0)
                 for c in spec["components"]]
        return lambda t: np.array([sum(a * np.sin(w * t + ph) for a, w, ph in comps)])
    raise ValueError(f"unknown signal kind {kind!r}")


@dataclass(frozen=True)
class DynProblem:
    problem_id: str
    name: str
    states: tuple[StateVariable, ...]
    inputs: tuple[InputVariable, ...]
    parameters: tuple[Parameter, ...]
    truth_equations: tuple[DynamicalEquation, ...]
    train: dict[str, Any] = field(compare=False)
    holdout: dict[str, Any] = field(compare=False)


def _truth_rhs(truth: tuple[DynamicalEquation, ...], order: list[str],
               inames: list[str], params: dict[str, float]):
    import sympy as sp

    compiled = [(e.state, sp.lambdify(
        [sp.Symbol(TIME_SYMBOL_)] + [sp.Symbol(n) for n in order]
        + [sp.Symbol(n) for n in inames] + [sp.Symbol(k) for k in params],
        e.sympy_expr, modules=["numpy"])) for e in truth]
    return compiled


TIME_SYMBOL_ = "t"


def generate_truth(problem: DynProblem, regime: str,
                   n_eval: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """High-accuracy ground-truth trajectory for the train or holdout regime."""
    from scipy.integrate import solve_ivp

    reg = problem.train if regime == "train" else problem.holdout
    order = [s.name for s in problem.states]
    inames = [u.name for u in problem.inputs]
    missing = [p.name for p in problem.parameters if p.value is None]
    if missing:
        raise ValueError(f"unidentified parameters: {missing}")
    params = {p.name: float(p.value) for p in problem.parameters}  # type: ignore[arg-type]
    compiled = _truth_rhs(problem.truth_equations, order, inames, params)
    u_of_t = reg["u_of_t"]
    x0 = np.asarray(reg["x0"], dtype=float)
    t_eval = np.linspace(reg["t_span"][0], reg["t_span"][1], n_eval)

    def f(t, x):
        u = np.atleast_1d(np.asarray(u_of_t(t), dtype=float))
        args = (t, *x, *u, *(params[k] for k in params))
        return np.array([float(c(*args)) for _, c in compiled])

    sol = solve_ivp(f, reg["t_span"], x0, method="DOP853",
                    rtol=1e-11, atol=1e-13, t_eval=t_eval)
    assert sol.success and np.all(np.isfinite(sol.y)), f"truth generation failed: {problem.problem_id}/{regime}"
    return t_eval, sol.y.T


def finalize(problem: DynProblem, n_eval: int = 400) -> DynProblem:
    """Attach ground-truth trajectories (train + holdout) to a problem spec."""
    t_tr, y_tr = generate_truth(problem, "train", n_eval)
    t_ho, y_ho = generate_truth(problem, "holdout", n_eval)
    train = dict(problem.train)
    train.update({"truth_t": t_tr, "truth_y": y_tr})
    holdout = dict(problem.holdout)
    holdout.update({"truth_t": t_ho, "truth_y": y_ho})
    return DynProblem(problem.problem_id, problem.name, problem.states,
                      problem.inputs, problem.parameters, problem.truth_equations,
                      train, holdout)
