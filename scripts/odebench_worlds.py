#!/usr/bin/env python3
"""Trajectories of the ODEBench systems, for PREREG-ODE-MAIN-SEARCH-2026-11.

The systems, their parameter values and two initial conditions each are
ODEBench's (``apedp_mmlm.law_corpora`` fetches and pins the file). The sampling
is ours: every system is integrated from four initial conditions over ten time
units at 10 000 samples. Two are the published ones; the other two lie between
them, one third and two thirds of the way, so that any convex constraint the
published pair satisfies holds for them too.

    development rows   the two published initial conditions
    pool               the initial condition one third of the way
    held back          the initial condition two thirds of the way; no engine sees it

A law is scored against the true vector field on 400 states: every hundredth
sample of each of the four clean trajectories. Half of those states come from
initial conditions the search never had, and all of them carry the true rate,
not a measured one. (Scoring on the held-back trajectory alone was rejected
during development: where that trajectory sits at a fixed point the true rate
hardly varies, and R^2 against it is decided by the fourth decimal of a
coefficient.)

Measurement noise is Gaussian with standard deviation ``level`` times the
standard deviation of each variable over the four clean trajectories.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.ode.window_rows import WindowRows, window_rows  # noqa: E402
from apedp_mmlm import law_corpora  # noqa: E402

T_END = 10.0
SAMPLES = 10_000
WINDOW = 100
BOUND = 1e6  # a trajectory that leaves this range is treated as diverged
TEST_ROWS_PER_TRAJECTORY = 100


def systems() -> list[dict[str, Any]]:
    """ODEBench's systems: id, dimension, one SymPy expression per component in ``x0, x1, ...``, initial conditions."""
    source = law_corpora._odebench_source()
    inits: dict[int, list[list[float]]] = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and getattr(statement.targets[0], "id", "") == "equations":
            assert isinstance(statement.value, ast.List)
            for item in statement.value.elts:
                assert isinstance(item, ast.Dict)
                fields = {key.value: value for key, value in zip(item.keys, item.values, strict=True) if isinstance(key, ast.Constant)}
                inits[int(law_corpora._number(fields["id"]))] = [
                    [law_corpora._number(v) for v in row.elts] for row in fields["init"].elts]  # type: ignore[attr-defined]
    out = []
    for entry in law_corpora.odebench_entries(source):
        values = {f"c_{i}": value for i, value in enumerate(entry["consts"])}
        rename = {sp.Symbol(f"x_{i}"): sp.Symbol(f"x{i}") for i in range(entry["dim"])}
        out.append({"id": entry["id"], "dim": entry["dim"], "texts": entry["components"],
                    "components": [law_corpora.parse_expression(text, values).xreplace(rename) for text in entry["components"]],
                    "inits": inits[entry["id"]]})
    return out


def vector_field(system: dict[str, Any]) -> Any:
    symbols = [sp.Symbol(f"x{i}") for i in range(system["dim"])]
    fns = [sp.lambdify(symbols, component, "numpy") for component in system["components"]]

    def rate(states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        with np.errstate(all="ignore"):
            out: np.ndarray = np.column_stack([np.asarray(fn(*states.T), dtype=float) * np.ones(len(states)) for fn in fns])
        return out

    return rate


def initial_conditions(system: dict[str, Any]) -> dict[str, np.ndarray]:
    a, b = (np.asarray(v, dtype=float) for v in system["inits"][:2])
    return {"dev_a": a, "dev_b": b, "pool": (2 * a + b) / 3, "test": (a + 2 * b) / 3}


def simulate(system: dict[str, Any], init: np.ndarray) -> np.ndarray | None:
    """Samples by variables, or ``None`` if the solver fails or the state leaves ``BOUND``."""
    rate = vector_field(system)
    t = np.linspace(0.0, T_END, SAMPLES)
    for method in ("DOP853", "LSODA"):
        with np.errstate(all="ignore"):
            solution = solve_ivp(lambda _t, x: rate(x)[0], (0.0, T_END), init, t_eval=t, method=method, rtol=1e-12, atol=1e-14)
        if solution.success and solution.y.shape[1] == SAMPLES and np.all(np.isfinite(solution.y)) and np.abs(solution.y).max() < BOUND:
            states: np.ndarray = solution.y.T
            return states
    return None


def world(system: dict[str, Any], level: float, tag: str = "PREREG-ODE-MAIN-SEARCH-2026-11") -> dict[str, Any]:
    """Rows for one system at one noise level, or ``{"excluded": reason}``."""
    clean = {}
    for name, init in initial_conditions(system).items():
        states = simulate(system, init)
        if states is None:
            return {"excluded": f"trajectory {name} failed to integrate or diverged"}
        clean[name] = states
    spread = np.vstack(list(clean.values())).std(axis=0)
    if not np.all(spread > 0):
        return {"excluded": "a variable does not vary"}
    t = np.linspace(0.0, T_END, SAMPLES)
    key = hashlib.sha256(f"{tag}|{system['id']}|{level}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    rows: dict[str, WindowRows] = {}
    for name in ("dev_a", "dev_b", "pool"):
        rows[name] = window_rows(t, clean[name] + level * spread * rng.standard_normal(clean[name].shape), window=WINDOW)
    dev = WindowRows(np.vstack([rows["dev_a"].states, rows["dev_b"].states]), np.vstack([rows["dev_a"].rates, rows["dev_b"].rates]),
                     np.concatenate([rows["dev_a"].times, rows["dev_b"].times]), rows["dev_a"].degree,
                     np.maximum(rows["dev_a"].fit_rms, rows["dev_b"].fit_rms), np.maximum(rows["dev_a"].rate_sd, rows["dev_b"].rate_sd))
    test_states = np.vstack([states[:: SAMPLES // TEST_ROWS_PER_TRAJECTORY] for states in clean.values()])
    return {"dev": dev, "pool": rows["pool"], "test_states": test_states, "test_rates": vector_field(system)(test_states),
            "spread": spread, "degrees": {name: rows[name].degree for name in rows}}


if __name__ == "__main__":
    for system in systems():
        made = world(system, 0.0)
        print(system["id"], system["dim"], made.get("excluded") or ("ok", made["degrees"]["dev_a"]))
