#!/usr/bin/env python3
"""A second set of dynamical systems nobody here has looked at, for testing additions to the ODE route.

The 98 chaotic polynomial systems of PREREG-ODE-BLIND-2026-11 were unseen once. Their outcomes
are known now, and three additions to the route came from looking at them. Those additions need
other systems. This script writes them, by two mechanical rules, with no discovery run on any.

**Part A, invented polynomial systems.** A seeded generator draws three-variable systems

    dx_i/dt = (A x)_i + (a few quadratic terms) - c_i * x_i**3

with ``A`` a diagonal of mixed signs plus a skew part. The cubic terms keep every trajectory
bounded; an unstable, rotating ``A`` keeps it from settling. A draw is kept if, started near the
origin and left for 200 time units, it neither leaves a bound nor comes to rest (every variable
still varies over the next 100) and its strongest oscillation has a period between 0.2 and 50.
Like the invented laws, these resemble nothing in particular: they are a test of the route.

**Part B, rational systems from the literature.** The systems of ``dysts`` 1.0 that the first
blind set left out for one reason only, "not a polynomial": autonomous flows in three or four
variables with a known period whose right-hand sides are quotients of polynomials in the state.
The route had 5.1 % wrong laws on ODEBench, where right-hand sides like these occur, and has not
been measured on such systems since its repairs.

Both parts are written in units of their own dominant period, with the first blind set's second
start (2.5 periods after the first) and its screen (a system whose rows the frozen world builder
does not produce in 120 s is listed and not run).

    python scripts/build_ode_second_blind_systems.py            # write benchmarks/ode_second_blind_systems.json
    python scripts/build_ode_second_blind_systems.py --check    # the file is what the rules give
    python scripts/build_ode_second_blind_systems.py --screen   # also list the systems whose rows cannot be built
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_dysts_polynomial_systems as first  # noqa: E402

OUT = ROOT / "benchmarks" / "ode_second_blind_systems.json"
SEED = 20261124
INVENTED = 30
CONSTANTS = (0.3, 0.45, 0.7, 1.3, 1.7, 2.3, 2.9, 3.7)
START = (0.5, -0.3, 0.2)
SETTLE, WATCH, WATCH_SAMPLES = 200.0, 100.0, 20001
BOUND = 1e3
STILL = 0.05                    # a variable whose spread over the watch is below this has come to rest
PERIODS = (0.2, 50.0)
MAX_DRAWS = 5000


def draw(rng: np.random.Generator) -> list[sp.Expr] | None:
    """One invented field, or ``None`` if its linear part does not push away from the origin while turning."""
    x = sp.symbols("x0:3")
    A = np.zeros((3, 3))
    for i in range(3):
        A[i, i] = float(rng.choice(CONSTANTS)) * float(rng.choice([-1, 1, 1]))
    for i, j in itertools.combinations(range(3), 2):
        if rng.random() < 0.6:
            value = 2.0 * float(rng.choice(CONSTANTS)) * float(rng.choice([-1, 1]))
            A[i, j], A[j, i] = value, -value
    eigenvalues = np.linalg.eigvals(A)
    if np.max(eigenvalues.real) <= 0.05 or not np.any(np.abs(eigenvalues.imag) > 0.1):
        return None
    products = [x[j] * x[k] for j, k in itertools.combinations_with_replacement(range(3), 2)]
    field = []
    for i in range(3):
        component = sum((sp.Float(A[i, j]) * x[j] for j in range(3) if A[i, j] != 0), sp.Integer(0))
        for _ in range(int(rng.integers(0, 3))):
            sign = float(rng.choice([-1, 1]))
            component += sp.Float(float(rng.choice(CONSTANTS)) * sign) * products[int(rng.integers(0, len(products)))]
        component -= sp.Float(float(rng.choice(CONSTANTS))) * x[i] ** 3
        field.append(sp.expand(component))
    return field


def watched(field: list[sp.Expr]) -> tuple[float, list[float]] | None:
    """The dominant period and a state on the motion, if the system stays bounded and does not come to rest."""
    x = sp.symbols("x0:3")
    functions = [sp.lambdify(x, component, "numpy") for component in field]
    with np.errstate(all="ignore"):
        solution = solve_ivp(lambda _t, s: [float(f(*s)) for f in functions], (0.0, SETTLE + WATCH), START, method="LSODA",
                             rtol=1e-9, atol=1e-11, t_eval=np.linspace(SETTLE, SETTLE + WATCH, WATCH_SAMPLES))
    if not solution.success or not np.all(np.isfinite(solution.y)) or np.abs(solution.y).max() > BOUND:
        return None
    y = solution.y
    if float(np.min(y.std(axis=1))) < STILL:
        return None
    signal = y[int(np.argmax(y.std(axis=1)))]
    power = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    frequency = np.fft.rfftfreq(len(signal), d=WATCH / (WATCH_SAMPLES - 1))
    period = 1.0 / float(frequency[1 + int(np.argmax(power[1:]))])
    if not PERIODS[0] < period < PERIODS[1]:
        return None
    return period, [float(v) for v in y[:, -1]]


def invented() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    symbols = list(sp.symbols("x0:3"))
    systems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(MAX_DRAWS):
        if len(systems) == INVENTED:
            break
        field = draw(rng)
        if field is None:
            continue
        seen_key = str(field)
        found = watched(field)
        if found is None or seen_key in seen:
            continue
        seen.add(seen_key)
        period, start = found
        scaled = [sp.expand(period * component) for component in field]
        second = first.second_start(scaled, np.asarray(start))
        if second is None:
            continue
        systems.append({"id": 2000 + len(systems), "name": f"Invented{len(systems):02d}", "part": "invented", "dim": 3,
                        "period": period, "texts": [first.text_of(c, symbols) for c in scaled], "inits": [start, second]})
    if len(systems) != INVENTED:
        raise SystemExit(f"only {len(systems)} invented systems after {MAX_DRAWS} draws")
    return systems


def rational_text(expression: sp.Expr) -> str:
    """A quotient of polynomials as text the frozen world builder can read, numbers in full."""
    return str(expression.xreplace({f: sp.Float(float(f), 17) for f in expression.atoms(sp.Float)}))


def rational() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The ``dysts`` systems left out of the first set only for not being polynomial, if they are rational."""
    import inspect

    import dysts.flows as flows

    systems: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    names = sorted(n for n, c in vars(flows).items() if inspect.isclass(c) and c.__module__ == "dysts.flows")
    for name in names:
        try:
            flow = getattr(flows, name)()
        except Exception as exc:  # noqa: BLE001 - a class the packaged metadata cannot build is listed as excluded
            excluded[name] = f"not constructible: {type(exc).__name__}"
            continue
        _, reason = first.eligible(name, flow)
        if reason != "not a polynomial":
            continue                                    # in the first set, or left out of it for another reason
        field = first.symbolic_field(flow)
        symbols = [sp.Symbol(f"x{i}") for i in range(len(flow.ic))]
        if field is None or not all(component.is_rational_function(*symbols) for component in field):
            excluded[name] = "not a quotient of polynomials"
            continue
        period = float(flow.period)
        scaled = [sp.Float(period) * sp.together(component) for component in field]
        start = [float(v) for v in np.asarray(flow.ic, dtype=float)]
        second = first.second_start(scaled, np.asarray(start))
        if second is None:
            excluded[name] = "the second initial condition could not be integrated"
            continue
        systems.append({"id": 2100 + len(systems), "name": name, "part": "rational", "dim": len(start), "period": period,
                        "texts": [rational_text(c) for c in scaled], "inits": [start, second]})
    return systems, excluded


def build() -> dict[str, Any]:
    quotients, excluded = rational()
    return {"note": "Written by scripts/build_ode_second_blind_systems.py by two mechanical rules; no discovery was run "
                    "on any of these systems before the preregistration that uses them was frozen.",
            "seed": SEED, "time_unit": "one dominant period of each system",
            "second_start_periods": first.SECOND_START, "systems": [*invented(), *quotients],
            "rational_excluded": dict(sorted(excluded.items()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args(argv)
    made = build()
    recorded = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.screen:
        made["not_buildable"] = first.screen(made["systems"])
        made["screen_seconds"] = first.SCREEN_SECONDS
    elif "not_buildable" in recorded:
        made["not_buildable"], made["screen_seconds"] = recorded["not_buildable"], recorded["screen_seconds"]
    text = json.dumps(made, indent=1, sort_keys=True) + "\n"
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.name} is not what the rules give")
            return 1
        print(f"{OUT.name}: {len(made['systems'])} systems, as the rules give them")
        return 0
    OUT.write_text(text, encoding="utf-8")
    parts = {part: sum(s["part"] == part for s in made["systems"]) for part in ("invented", "rational")}
    print(f"wrote {OUT.relative_to(ROOT)}: {parts}; not buildable: {sorted(made.get('not_buildable', {}))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
