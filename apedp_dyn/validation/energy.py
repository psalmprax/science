"""D5-stage validators: energy conservation/balance laws.

Principle (preserved from the original): check the APPROPRIATE balance — never
"everything must be conserved". Undamped free oscillation conserves; damped
decay obeys dE/dt = F·v − c·v²; forced intervals obey the power balance. These
validators discriminate STRUCTURES (anti-damped impostors detected hard) but
cannot detect wrongly-valued parameters (a 2x-stiffness oscillator conserves
its own energy exactly) — parameter accuracy stays with trajectory validation
(gate 4).

The conserved quantity is DERIVED from the model's own equations instead of
being hardcoded for a mass-spring, so the same validators cover the
cubic-restoring Duffing oscillator and multi-DOF coupled systems:

    E = ½ Σ v_i² + U(q),        F_i(q) = −∂U/∂q_i

F_i(q) is the position-only part of each acceleration law (velocities, inputs
and time zeroed; parameters substituted numerically). U is recovered by
integrating that 1-form along the axis path, which requires the restoring
1-form to be exact (∂F_i/∂q_j = ∂F_j/∂q_i) — a genuine physical property, and a
failure is reported rather than guessed around. Kinetic energy assumes the
grammar's unit-mass normalization (an RHS is force per unit mass), so no mass
appears in E.

The symbolic derivation is BOUNDED. Exact integration has no natural stopping
point, so an exotic restoring force could hang the gate instead of failing it;
a validator that hangs is worse than one that refuses. Only restoring forces
that are polynomial in the positions and a whitelist of elementary functions
are admitted (which is every mechanical form the grammar emits: polynomial
stiffness, polynomial-plus-cubic Duffing, linear coupling, sin θ pendulum),
with size and degree budgets and a wall-clock budget across the DOF loop.
Anything else returns ``conservative=False`` with the reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.model.variables import TIME_SYMBOL

RESTORING_FUNCTIONS = ("sin", "cos", "tan", "sinh", "cosh", "tanh")
"""Elementary functions a restoring force may use and still be integrated in
closed form along the axis path. Anything else (exp, erf, Bessel, ...) is
refused rather than integrated on a hope."""

RESTORING_DEGREE_BUDGET = 8
RESTORING_OPS_BUDGET = 120
DERIVATION_SECONDS = 5.0
"""Derivation budgets. Recorded heuristics, not derived bounds: measured
restoring forces here sit at degree ≤ 4 and ~20 ops, so these clear them by
a wide margin while keeping expand/simplify/integrate cheap."""


@dataclass(frozen=True)
class EnergyLaw:
    """Conserved-quantity law derived from a model's own equations.

    ``conservative`` is False when the model is not a second-order mechanical
    form or its restoring force is not a gradient field; ``potential`` is then
    None and ``detail`` says why. Indices address ``model.states`` order.
    """

    conservative: bool
    detail: str
    potential: Any = None
    position_indices: tuple[int, ...] = ()
    velocity_indices: tuple[int, ...] = ()
    position_symbols: tuple[Any, ...] = ()


def _pairs(model) -> list[tuple[int, int, sp.Symbol, sp.Symbol]]:
    """Position/velocity state pairs from kinematic pass-throughs (q' == v)."""
    order = [s.name for s in model.states]
    index = {n: i for i, n in enumerate(order)}
    eq = {e.state: e.sympy_expr for e in model.equations}
    pairs: list[tuple[int, int, sp.Symbol, sp.Symbol]] = []
    for name in order:
        expr = eq.get(name)
        if isinstance(expr, sp.Symbol):
            partner = str(expr)
            if partner in index and partner != name:
                pairs.append((index[name], index[partner],
                              sp.Symbol(name), sp.Symbol(partner)))
    return pairs


def _restoring_admissible(expr: Any, qs: tuple[Any, ...]) -> str | None:
    """None if `expr` is inside the bounded-derivation class, else the reason.

    Replacing the positions and any elementary function atoms by fresh symbols
    must leave a POLYNOMIAL. Polynomial stiffness (MSD, coupled), polynomial
    nonlinearity (Duffing) and sin θ (pendulum) all qualify; a non-elementary or
    rational integrand does not, and is refused instead of integrated.
    """
    funcs = sorted(expr.atoms(sp.Function), key=str)
    for f in funcs:
        if f.func.__name__ not in RESTORING_FUNCTIONS:
            return (f"restoring force uses {f.func.__name__}(), outside the "
                    f"derivable class {RESTORING_FUNCTIONS}")
    ops = int(sp.count_ops(expr))
    if ops > RESTORING_OPS_BUDGET:
        return (f"restoring force complexity {ops} exceeds budget "
                f"{RESTORING_OPS_BUDGET}")
    symbols = ([sp.Symbol(f"_q{i}") for i in range(len(qs))]
               + [sp.Symbol(f"_f{i}") for i in range(len(funcs))])
    probe = expr.subs({**{q: sp.Symbol(f"_q{i}") for i, q in enumerate(qs)},
                       **{f: sp.Symbol(f"_f{i}") for i, f in enumerate(funcs)}})
    try:
        degree = sp.Poly(sp.expand(probe), *symbols).total_degree()
    except (sp.PolynomialError, sp.GeneratorsNeeded, TypeError):
        return ("restoring force is not polynomial in the positions and their "
                "elementary functions; the potential would need a "
                "non-elementary integral")
    if degree > RESTORING_DEGREE_BUDGET:
        return (f"restoring force degree {degree} exceeds budget "
                f"{RESTORING_DEGREE_BUDGET}")
    return None


def derive_energy_law(model) -> EnergyLaw:
    """Derive E = ½ Σ v² + U(q) from the model's own acceleration laws."""
    pairs = _pairs(model)
    if not pairs:
        return EnergyLaw(False, "no position/velocity state pair (q' == v) "
                                "found; not a second-order mechanical form")
    pos = tuple(p[0] for p in pairs)
    vel = tuple(p[1] for p in pairs)
    qs = tuple(p[2] for p in pairs)
    vs = tuple(p[3] for p in pairs)
    eq = {e.state: e.sympy_expr for e in model.equations}

    try:
        values = model.param_values()
    except ValueError as e:
        return EnergyLaw(False, f"cannot derive energy: {e}", None, pos, vel, qs)

    subs: dict[Any, Any] = {v: sp.Integer(0) for v in vs}
    subs.update({sp.Symbol(u.name): sp.Integer(0) for u in model.inputs})
    subs[sp.Symbol(TIME_SYMBOL)] = sp.Integer(0)
    subs.update({sp.Symbol(k): sp.Float(v) for k, v in values.items()})
    forces = [sp.expand(eq[str(v)].subs(subs)) for v in vs]

    allowed = set(qs)
    stray = sorted({s for f in forces for s in f.free_symbols} - allowed)
    if stray:
        return EnergyLaw(
            False,
            "restoring part depends on non-position symbols: "
            f"{[str(s) for s in stray]}", None, pos, vel, qs)

    # Refuse anything the bounded derivation cannot handle, BEFORE integrating.
    for i, force in enumerate(forces):
        why = _restoring_admissible(force, qs)
        if why is not None:
            return EnergyLaw(False, f"DOF {i}: {why}", None, pos, vel, qs)

    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            gap = sp.simplify(sp.diff(forces[i], qs[j]) - sp.diff(forces[j], qs[i]))
            if gap != 0:
                return EnergyLaw(
                    False,
                    f"restoring 1-form is not exact (DOF {i},{j}): {gap}",
                    None, pos, vel, qs)

    # Integrate the (now exact) 1-form along the axis path, one DOF at a time:
    # later coordinates are held at 0 while q_i sweeps 0 -> q_i. Cheaper and
    # better conditioned than a radial integral, which produces sign-split
    # Piecewise results for trigonometric restoring forces like the pendulum's.
    potential = sp.Integer(0)
    deadline = time.monotonic() + DERIVATION_SECONDS
    for i, q in enumerate(qs):
        if time.monotonic() > deadline:
            return EnergyLaw(
                False,
                f"potential integration exceeded the {DERIVATION_SECONDS:g}s "
                f"budget at DOF {i}/{len(qs)}", None, pos, vel, qs)
        earlier_zeroed = forces[i].subs({qs[k]: sp.Integer(0)
                                        for k in range(i + 1, len(qs))})
        sweep = sp.Symbol(f"_sweep{i}", real=True)
        potential -= sp.integrate(earlier_zeroed.subs(q, sweep), (sweep, 0, q))
    potential = sp.simplify(sp.expand(potential))
    leftover = [str(s) for s in potential.free_symbols
                if str(s).startswith("_sweep")]
    if potential.has(sp.Integral) or leftover:
        return EnergyLaw(False, f"potential integral did not close: {leftover}",
                         None, pos, vel, qs)
    return EnergyLaw(True, f"conservative over {len(qs)} DOF",
                     potential, pos, vel, qs)


def total_energy(law: EnergyLaw, Y: np.ndarray) -> np.ndarray:
    """Unit-mass mechanical energy along a trajectory (samples × states)."""
    Y = np.asarray(Y, dtype=float)
    E = 0.5 * sum(Y[:, i] ** 2 for i in law.velocity_indices)
    if law.potential is not None:
        f = sp.lambdify(list(law.position_symbols), law.potential, modules=["numpy"])
        cols = [Y[:, i] for i in law.position_indices]
        E = E + np.asarray(f(*cols), dtype=float)
    return np.atleast_1d(np.asarray(E, dtype=float))


def _relative_drift(E: np.ndarray) -> float:
    """Cumulative relative drift max|E − E0| / max|E|.

    Cumulative, not per-step: sustained exponential growth has an almost
    constant per-step relative change, which a per-step metric cannot see.
    """
    e_max = float(np.max(np.abs(E)))
    if e_max == 0.0:
        raise ValueError("degenerate energy trajectory")
    return float(np.max(np.abs(E - E[0])) / e_max)


def _dissipation_verdict(E: np.ndarray, min_decay_fraction: float,
                         slack: float) -> tuple[bool, dict]:
    """Monotone non-increase plus real decay, for an unforced interval."""
    info: dict = {"decay_fraction": 1.0 - float(E[-1]) / float(E[0])}
    if np.any(np.diff(E) > slack * float(np.max(np.abs(E)))):
        return False, {**info, "reason": "energy increased in unforced interval"}
    if info["decay_fraction"] <= min_decay_fraction:
        return False, {**info, "reason": "damping did not act like damping"}
    return True, info


def energy_drift_from_law(law: EnergyLaw, Y: np.ndarray) -> float:
    """Cumulative relative drift of a derived law's energy along a trajectory."""
    return _relative_drift(total_energy(law, Y))


def simulate_trajectory(model, t: np.ndarray, u_of_t, x0,
                        *, rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    """Simulate a fitted model on a fixed grid; return (samples, states).

    Integrator-precision tolerances: conservation is a structural property and
    must sit at solver noise, far below any parameter tolerance.
    """
    from apedp_dyn.simulation import simulate

    t = np.asarray(t, dtype=float)
    res = simulate(model, (float(t[0]), float(t[-1])), u_of_t,
                   np.asarray(x0, dtype=float), n_eval=len(t),
                   rtol=rtol, atol=atol)
    if not res.success:
        raise ValueError(f"conservation sim failed: {res.detail}")
    return np.asarray(res.y.T, dtype=float)


def check_conservative(model, Y: np.ndarray,
                       tol: float = 1e-6) -> tuple[bool, float]:
    """Unforced, undamped motion must conserve the model's own energy.

    Returns (False, inf) when no energy law can be derived: an underivable
    balance is not evidence of conservation.
    """
    law = derive_energy_law(model)
    if not law.conservative:
        return False, float("inf")
    drift = energy_drift_from_law(law, Y)
    return drift <= tol, drift


def check_dissipative(model, Y: np.ndarray, min_decay_fraction: float = 0.3,
                      slack: float = 0.02) -> tuple[bool, dict]:
    """Unforced dissipative interval: monotone non-increase + real decay."""
    law = derive_energy_law(model)
    if not law.conservative:
        return False, {"reason": law.detail}
    return _dissipation_verdict(total_energy(law, Y), min_decay_fraction, slack)


# --- Analytic 1-DOF helpers (explicit mass/stiffness, no model required) -----


def mechanical_energy(x: np.ndarray, v: np.ndarray, m: float, k: float) -> np.ndarray:
    """E = m v^2/2 + k x^2/2."""
    return 0.5 * m * np.asarray(v, dtype=float) ** 2 + 0.5 * k * np.asarray(x, dtype=float) ** 2


def energy_drift(x: np.ndarray, v: np.ndarray, m: float, k: float) -> float:
    """Cumulative relative drift of the analytic 1-DOF law (legacy signature)."""
    return _relative_drift(mechanical_energy(x, v, m, k))


def simulate_pair(model, m: float, k: float, t: np.ndarray,
                  u_of_t, x0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate and return the first two states (legacy 1-DOF helper).

    ``m`` and ``k`` are accepted for backward compatibility and unused; use
    :func:`simulate_trajectory` for the full state vector.
    """
    Y = simulate_trajectory(model, t, u_of_t, x0)
    return Y[:, 0], Y[:, 1]


def check_conservative_core(x: np.ndarray, v: np.ndarray, m: float, k: float,
                            tol: float = 1e-6) -> tuple[bool, float]:
    """Undamped free oscillation must conserve to integrator precision."""
    drift = energy_drift(x, v, m, k)
    return drift <= tol, drift


def check_damped_decay(x: np.ndarray, v: np.ndarray, m: float, k: float,
                       min_decay_fraction: float = 0.3,
                       slack: float = 0.02) -> tuple[bool, dict]:
    """Unforced dissipative interval: monotone non-increase + real decay."""
    return _dissipation_verdict(mechanical_energy(x, v, m, k),
                                min_decay_fraction, slack)
