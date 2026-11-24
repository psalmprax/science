#!/usr/bin/env python3
"""Write ``benchmarks/dysts_polynomial_systems.json``: chaotic systems the ODE routes were not built on.

Status: audit-only

The systems come from the ``dysts`` database of chaotic flows (W. Gilpin, 2021), the pool that
Kaptanoglu et al. (2023) drew the polynomial benchmark for sparse identification from. The rule,
applied mechanically and fixed before any discovery run on them:

- a flow of ``dysts`` 1.0 that is autonomous, has three or four variables, an initial condition on
  its attractor and a dominant period;
- whose right-hand side, evaluated on symbols with its published parameters, is a polynomial in
  the state of total degree at most ``MAX_DEGREE`` with at most ``MAX_TERMS`` terms per component;
- that is not one of ODEBench's systems (``IN_ODEBENCH``), on which the routes were built and
  measured, and does not have the same terms as one under a renaming of the variables
  (``SAME_TERMS_AS_ODEBENCH``: the same form with other coefficients).

Each system is written in units of its own dominant period, so that the fixed horizon of
``scripts/odebench_worlds.py`` (ten time units) covers ten periods of every system. The second
initial condition is where the first one arrives after ``SECOND_START`` periods.

The frozen world builder also starts trajectories between the two initial conditions, off the
attractor, and has no time limit. A few systems leave their attractor from there and the solver
then crawls. ``--screen`` builds every system's rows once, without noise, and lists under
``not_buildable`` those that do not finish within ``SCREEN_SECONDS``; the runner skips them. No
discovery is run on any system by this file.

    pip install dysts==1.0
    python scripts/build_dysts_polynomial_systems.py            # rewrite the file (keeps the screening)
    python scripts/build_dysts_polynomial_systems.py --screen   # ... and screen again (slow)
    python scripts/build_dysts_polynomial_systems.py --check    # fail if the rule gives other systems
"""
from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "dysts_polynomial_systems.json"
DYSTS_VERSION = "1.0"
MAX_DEGREE = 4
MAX_TERMS = 12
SECOND_START = 2.5  # periods after the first initial condition
SCREEN_SECONDS = 120.0  # a system whose rows take longer than this to build is not run (typical: two to twenty)
#: ODEBench has the Lorenz, Rössler, Aizawa and Chen-Lee systems (ids 54 to 61).
IN_ODEBENCH = ("Lorenz", "Rossler", "Aizawa", "ChenLee")
#: Found by comparing every eligible system's monomials with those of ODEBench's polynomial systems
#: of three or four variables, under every renaming: Chen and UnifiedChaotic have Lorenz's terms,
#: Langford has Aizawa's, LiuChen has Chen-Lee's, VallisElNino has the Maxwell-Bloch system's.
SAME_TERMS_AS_ODEBENCH = ("Chen", "Langford", "LiuChen", "UnifiedChaotic", "VallisElNino")


def python_function(function: Any) -> Any:
    """The plain function behind a compiled one."""
    return getattr(function, "py_func", function)


def symbolic_field(flow: Any) -> list[sp.Expr] | None:
    """The right-hand side on symbols ``x0, x1, ...``, or ``None`` if it is not plain arithmetic."""
    symbols = [sp.Symbol(f"x{i}") for i in range(len(flow.ic))]
    try:
        value = python_function(type(flow)._rhs)(*symbols, sp.Symbol("t"), **flow.params)
    except Exception:  # noqa: BLE001 - whatever a numerical field raises on symbols, it is not plain arithmetic
        return None
    try:
        return [sp.expand(sp.sympify(component)) for component in value]
    except (sp.SympifyError, TypeError):
        return None


def eligible(name: str, flow: Any) -> tuple[list[sp.Expr], str] | tuple[None, str]:
    """The field if the system passes the rule, and otherwise why it does not."""
    if name in IN_ODEBENCH:
        return None, "in ODEBench"
    if name in SAME_TERMS_AS_ODEBENCH:
        return None, "same terms as an ODEBench system"
    if getattr(flow, "nonautonomous", False):
        return None, "not autonomous"
    ic = np.asarray(getattr(flow, "ic", []), dtype=float)
    if ic.ndim != 1 or len(ic) not in (3, 4) or not np.all(np.isfinite(ic)):
        return None, "not three or four variables"
    period = getattr(flow, "period", None)
    if period is None or not np.isfinite(period) or period <= 0:
        return None, "no dominant period"
    field = symbolic_field(flow)
    if field is None or len(field) != len(ic):
        return None, "not plain arithmetic"
    symbols = [sp.Symbol(f"x{i}") for i in range(len(ic))]
    for component in field:
        if component.free_symbols - set(symbols):
            return None, "depends on something other than the state"
        try:
            poly = sp.Poly(component, *symbols)
        except sp.PolynomialError:
            return None, "not a polynomial"
        if poly.total_degree() > MAX_DEGREE:
            return None, f"degree above {MAX_DEGREE}"
        if len(poly.terms()) > MAX_TERMS:
            return None, f"more than {MAX_TERMS} terms in a component"
        if not all(sp.sympify(c).is_number for c in poly.coeffs()):
            return None, "a coefficient is not a number"
    if all(component == 0 for component in field):
        return None, "empty field"
    return field, "eligible"


def text_of(expression: sp.Expr, symbols: list[sp.Symbol]) -> str:
    """The polynomial as text: coefficients to 17 significant digits, powers as whole numbers."""
    parts = []
    for powers, coefficient in sp.Poly(expression, *symbols).terms():
        factors = [f"{symbol}**{power}" if power > 1 else str(symbol)
                   for symbol, power in zip(symbols, powers, strict=True) if power]
        parts.append("*".join([repr(float(coefficient)), *factors]))
    return " + ".join(parts) if parts else "0.0"


def second_start(field: list[sp.Expr], first: np.ndarray) -> list[float] | None:
    """Where ``first`` is after ``SECOND_START`` time units of ``field``, or ``None`` if integration fails."""
    symbols = [sp.Symbol(f"x{i}") for i in range(len(first))]
    functions = [sp.lambdify(symbols, component, "numpy") for component in field]
    with np.errstate(all="ignore"):
        solution = solve_ivp(lambda _t, x: [float(f(*x)) for f in functions], (0.0, SECOND_START), first,
                             method="DOP853", rtol=1e-12, atol=1e-14)
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        return None
    return [float(v) for v in solution.y[:, -1]]


def build() -> dict[str, Any]:
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
        field, reason = eligible(name, flow)
        if field is None:
            excluded[name] = reason
            continue
        period = float(flow.period)
        scaled = [sp.expand(period * component) for component in field]
        first = [float(v) for v in np.asarray(flow.ic, dtype=float)]
        second = second_start(scaled, np.asarray(first))
        if second is None:
            excluded[name] = "the second initial condition could not be integrated"
            continue
        symbols = [sp.Symbol(f"x{i}") for i in range(len(first))]
        systems.append({"id": 1000 + len(systems), "name": name, "dim": len(first), "period": period,
                        "texts": [text_of(component, symbols) for component in scaled], "inits": [first, second]})
    return {"source": f"dysts {DYSTS_VERSION} (W. Gilpin, 2021); rule in scripts/build_dysts_polynomial_systems.py",
            "time_unit": "one dominant period of each system", "max_degree": MAX_DEGREE, "max_terms": MAX_TERMS,
            "second_start_periods": SECOND_START, "in_odebench": list(IN_ODEBENCH),
            "same_terms_as_odebench": list(SAME_TERMS_AS_ODEBENCH),
            "systems": systems, "excluded": dict(sorted(excluded.items()))}


def _build_rows(connection: Any, system: dict[str, Any]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import odebench_worlds as worlds

    made = worlds.world({**system, "components": [sp.sympify(text) for text in system["texts"]]}, 0.0, tag="screen")
    connection.send(made.get("excluded"))
    connection.close()


def not_buildable(system: dict[str, Any], seconds: float = SCREEN_SECONDS) -> str | None:
    """Why the frozen world builder gives no rows for ``system`` in time, or ``None`` if it does."""
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_build_rows, args=(child, system))
    process.start()
    child.close()
    reason: str | None = f"its rows are not built within {seconds:.0f} s"
    if parent.poll(seconds):
        try:
            excluded = parent.recv()
            reason = str(excluded) if excluded else None
        except EOFError:
            reason = "the world builder ended without a result"
    process.join(1.0)
    if process.is_alive():
        process.kill()
        process.join()
    return reason


def screen(systems: list[dict[str, Any]], workers: int = 2) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:   # each thread starts one process per system and waits for it
        reasons = list(pool.map(not_buildable, systems))
    return {s["name"]: reason for s, reason in zip(systems, reasons, strict=True) if reason}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file instead of writing it")
    parser.add_argument("--screen", action="store_true", help="build every system's rows once and list those too slow")
    args = parser.parse_args(argv)
    made = build()
    recorded = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.check:
        same = all(recorded.get(key) == made[key] for key in made)
        print(f"{OUT.relative_to(ROOT)} {'matches' if same else f'differs from what dysts {DYSTS_VERSION} gives'}")
        return 0 if same else 1
    made["screen_seconds"] = SCREEN_SECONDS
    made["not_buildable"] = screen(made["systems"]) if args.screen else recorded.get("not_buildable", {})
    OUT.write_text(json.dumps(made, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(made['systems'])} systems by the rule, {len(made['not_buildable'])} of them not buildable, "
          f"{len(made['excluded'])} excluded -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
