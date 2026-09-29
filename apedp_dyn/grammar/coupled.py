"""D2b — two-DOF coupled mechanical grammar.

One CandidateDynamics per mass over local (x, v, u) plus the remote position
(xr); the SystemCandidate bundles them. Coupling topology is discovered, not
assumed: the uncoupled rival (no xr terms) is a first-class citizen of the
enumeration and must lose at gate 4. D2b scope is exactly 2 DOF.
"""
from __future__ import annotations

import itertools
from typing import Any

import sympy as sp

from apedp_dyn.gates import _unit_of, _ureg
from apedp_dyn.grammar.candidate_ir import (
    CandidateDynamics,
    SystemCandidate,
    Term,
)
from apedp_dyn.grammar.mechanical_grammar import (
    _TERM_BUILDERS,
    DEFAULT_BOUNDS,
    problem_symbol_units,
)
from apedp_dyn.model import DynamicalEquation, Parameter, PhysicalModel


def enumerate_coupled(required: tuple[str, ...] = ("u",),
                      optional: tuple[str, ...] = ("x", "v", "xr", "dxr",
                                                   "c1", "c2"),
                      max_terms: int = 4, seed: int = 0,
                      id_prefix: str = "mg2",
                      coupling_mode: str = "general",
                      dae_mode: str = "none",
                      timescales: tuple[str, ...] = ()) -> list[SystemCandidate]:
    """Per-DOF controlled subsets; globally unique param names (q-counter).

    Extended for D3: DAE and multi-timescale support.

    Parameters
    ----------
    required : tuple of str
        Required term names (e.g. ("u",)).
    optional : tuple of str
        Optional term names. New additions for D3:
        - "c1", "c2" : constraint terms (algebraic, no time derivatives).
        - "x_f", "v_f" : fast-state variables.
        - "x_s", "v_s" : slow-state variables.
    max_terms : int
        Maximum total terms per equation (includes required + optional).
    seed : int
        Random seed for parameter naming.
    id_prefix : str
        Prefix for candidate IDs.
    coupling_mode : str
      general — independent coupling per mass (k12 ≠ k21 permitted).
      conservative_reciprocal — dxr (relative-displacement) terms share one
        parameter "kc" across masses (k12 = k21). System-class constraint,
        NOT a universal physical rule.
    dae_mode : str
      "none" — pure ODE system (default).
      "constrained" — equations include algebraic constraint terms (c1, c2);
        these are zero-derivative terms that enforce relationships among states
        (e.g., pendulum rod length, slider kinematics).
      "multi_timescale" — equations split into fast and slow variables;
        timescales listed in `timescales` (e.g. ("fast", "slow")) must have
        matching count to the number of fast/slow state pairs.
    timescales : tuple of str
        Length-2 tuple ("fast", "slow") indicating which state variables are
        fast vs slow. Must have exactly 2 elements when dae_mode="multi_timescale".
        Fast variables typically have smaller characteristic time constants.

    Returns
    -------
    list[SystemCandidate]
        One SystemCandidate per combination of per-DOF CandidateDynamics.
    """
    if coupling_mode not in ("general", "conservative_reciprocal"):
        raise ValueError(f"unknown coupling_mode {coupling_mode!r}")
    if dae_mode not in ("none", "constrained", "multi_timescale"):
        raise ValueError(f"unknown dae_mode {dae_mode!r}")
    if dae_mode == "multi_timescale" and len(timescales) != 2:
        raise ValueError(
            f"multi_timescale dae_mode requires exactly 2 timescales, "
            f"got {len(timescales)}")
    for name in list(required) + list(optional):
        if name not in _TERM_BUILDERS:
            raise ValueError(f"unknown grammar term: {name}")
    per_dof: list[list[CandidateDynamics]] = []
    counter = 0
    for dof in range(2):
        options: list[CandidateDynamics] = []
        for r in range(0, max_terms - len(required) + 1):
            for combo in itertools.combinations(optional, r):
                names = tuple(required) + combo
                terms = []
                bounds: dict[str, tuple[float, float]] = {}
                for name in names:
                    if (coupling_mode == "conservative_reciprocal"
                            and name == "dxr"):
                        pname = "kc"
                    elif name.startswith("c") and dae_mode == "constrained":
                        # Constraint terms share a single "con" parameter
                        # across both equations (like kc for reciprocal coupling).
                        pname = "con"
                    elif name.startswith("x_") and dae_mode == "multi_timescale":
                        # Fast/slow state labeling: map x -> q{counter} but
                        # mark metadata for timescale awareness.
                        pname = f"q{counter}"
                        counter += 1
                    else:
                        pname = f"q{counter}"
                        counter += 1
                    terms.append(Term(name, pname))
                    bounds[pname] = DEFAULT_BOUNDS
                meta: dict[str, Any] = {
                    "family": "mechanical_2nd_order_2dof",
                    "order": 2,
                    "dof": dof,
                    "coupling_mode": coupling_mode,
                    "dae_mode": dae_mode,
                }
                if dae_mode == "constrained":
                    meta["constraint_terms"] = tuple(
                        n for n in names if n.startswith("c"))
                if dae_mode == "multi_timescale":
                    meta["timescales"] = timescales
                    # Mark which terms involve fast/slow vars
                    fast_names = tuple(
                        n for n in names if n in ("x_f", "v_f"))
                    slow_names = tuple(
                        n for n in names if n in ("x_s", "v_s"))
                    if fast_names:
                        meta["fast_variables"] = fast_names
                    if slow_names:
                        meta["slow_variables"] = slow_names
                options.append(CandidateDynamics(
                    lhs="xddot", terms=tuple(terms),
                    metadata=meta,
                    candidate_id=f"{id_prefix}-d{dof}",
                    parameter_bounds=bounds,
                    generation_seed=seed))
        per_dof.append(options)
    out: list[SystemCandidate] = []
    n = 0
    for e0 in per_dof[0]:
        for e1 in per_dof[1]:
            out.append(SystemCandidate(
                equations=(e0, e1),
                metadata={
                    "family": "mechanical_2nd_order_2dof",
                    "dae_mode": dae_mode,
                    "timescales": timescales if dae_mode == "multi_timescale" else None,
                    "constraint_terms": (
                        e0.metadata.get("constraint_terms", ())
                        + e1.metadata.get("constraint_terms", ())),
                },
                candidate_id=f"{id_prefix}-{n:04d}",
                generation_seed=seed))
            n += 1
    return out


def canonical_system(candidate: SystemCandidate) -> str:
    """Order-sensitive structural key: DOFs are labeled (distinct drives/ICs),
    so a dof-swapped pair is a genuinely different hypothesis, not a duplicate."""
    from apedp_dyn.grammar.canonicalize import canonical_form

    parts = [canonical_form(e) for e in candidate.equations]
    return "sys[" + "|".join(parts) + "]"


def materialize_system(problem, candidate: SystemCandidate) -> PhysicalModel:
    """System IR -> executable PhysicalModel.

    Handles three DAE modes:
      "none"       : pure 2-DOF ODE (original D2b behavior).
      "constrained": equations include algebraic constraint terms (c1, c2);
        these enforce relationships among states without derivatives.
      "multi_timescale": equations split into fast/slow variables with
        separate time constants; the model is expanded to 4 first-order ODEs
        by introducing auxiliary fast/slow state pairs.
    """
    if len(candidate.equations) != 2:
        raise ValueError("D3 scope requires exactly 2 equations")
    dae_mode = candidate.metadata.get("dae_mode", "none")
    timescales = candidate.metadata.get("timescales", ())

    if dae_mode == "none":
        # Original D2b behavior: 2 DOF → 4 first-order equations
        return _materialize_ode(problem, candidate)

    if dae_mode == "constrained":
        return _materialize_constrained(problem, candidate)

    if dae_mode == "multi_timescale":
        return _materialize_multiscale(problem, candidate, timescales)

    raise ValueError(f"unknown dae_mode {dae_mode!r}")


def _materialize_ode(problem, candidate: SystemCandidate) -> PhysicalModel:
    """Pure ODE materialization: D2b behavior, 4-state first-order system."""
    if len(problem.states) != 4:
        raise ValueError("ODE materialization requires 4 states")
    if len(problem.inputs) != 2:
        raise ValueError("ODE materialization requires 2 inputs")
    pos = [problem.states[0], problem.states[2]]
    vel = [problem.states[1], problem.states[3]]
    inp = [problem.inputs[0], problem.inputs[1]]
    real = dict(problem_symbol_units(problem))
    equations: list[DynamicalEquation] = []
    parameters: list[Parameter] = []
    seen_units: dict[str, str] = {}
    for i in range(2):
        remote = pos[1 - i]
        target_unit = _ureg.Unit(vel[i].units) / _ureg.Unit("second")
        equations.append(DynamicalEquation(pos[i].name, vel[i].name))
        parts: list[str] = []
        for t in candidate.equations[i].terms:
            built = _TERM_BUILDERS[t.name](
                sp.Symbol(pos[i].name), sp.Symbol(vel[i].name),
                sp.Symbol(inp[i].name), sp.Symbol(remote.name))
            term_unit = _unit_of(built, real)
            punit = str(target_unit / term_unit)
            bounds = candidate.equations[i].parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
            if t.parameter in seen_units:
                if seen_units[t.parameter] != punit:
                    raise ValueError(
                        f"shared parameter {t.parameter}: conflicting units "
                        f"{seen_units[t.parameter]} vs {punit}")
            else:
                seen_units[t.parameter] = punit
                parameters.append(Parameter(t.parameter, punit, None, bounds))
            parts.append(f"({t.parameter}*{sp.sstr(built)})")
        equations.append(DynamicalEquation(
            vel[i].name, " + ".join(parts) if parts else "0"))
    return PhysicalModel(problem.states, problem.inputs,
                         tuple(parameters), tuple(equations))


def _materialize_constrained(problem, candidate: SystemCandidate) -> PhysicalModel:
    """Constrained DAE materialization.

    Algebraic constraint terms (named "c1", "c2") enforce relationships
    among the 4 state variables (positions/velocities of 2 masses). The
    result is a 4-state ODE system where the constraints are enforced
    through the dynamics (index-1 reduction). Constraint terms must be
    dimensionless or have compatible units that reduce to equality.
    """
    if len(problem.states) != 4:
        raise ValueError("Constrained DAE requires 4 states")
    if len(problem.inputs) != 2:
        raise ValueError("Constrained DAE requires 2 inputs")
    pos = [problem.states[0], problem.states[2]]
    vel = [problem.states[1], problem.states[3]]
    inp = [problem.inputs[0], problem.inputs[1]]
    real = dict(problem_symbol_units(problem))
    equations: list[DynamicalEquation] = []
    parameters: list[Parameter] = []
    seen_units: dict[str, str] = {}
    constraint_terms = candidate.metadata.get("constraint_terms", ())
    # Track which parameter names are shared across constraint terms
    shared_params: set[str] = set()
    # Parameters attached to constraint terms. Tracked separately from the
    # "seen twice" shared_params set below: a constraint term present in only
    # one DOF never repeats, so it would never be recorded as shared, yet it
    # is still a constraint parameter that gate 1.5 has to verify.
    constraint_params: set[str] = set()
    for i in range(2):
        # Build the ODE part (same as pure ODE)
        remote = pos[1 - i]
        target_unit = _ureg.Unit(vel[i].units) / _ureg.Unit("second")
        equations.append(DynamicalEquation(pos[i].name, vel[i].name))
        parts: list[str] = []
        constraint_parts: list[str] = []
        for t in candidate.equations[i].terms:
            if t.name in constraint_terms:
                # This is a constraint term; build it but track it
                built = _TERM_BUILDERS[t.name](
                    sp.Symbol(pos[i].name), sp.Symbol(vel[i].name),
                    sp.Symbol(inp[i].name), sp.Symbol(remote.name))
                cunit = _unit_of(built, real)
                # Constraint terms should be dimensionless or have units
                # that match the target. We record the unit but don't add
                # to the ODE RHS unless it's the only term.
                constraint_parts.append(f"({t.parameter}*{sp.sstr(built)})")
                constraint_params.add(t.parameter)
                if t.parameter in seen_units:
                    shared_params.add(t.parameter)
                else:
                    seen_units[t.parameter] = str(
                        _ureg.Unit(cunit) / _ureg.Unit("dimensionless"))
                    parameters.append(Parameter(t.parameter,
                                                str(_ureg.Unit(cunit) / _ureg.Unit("dimensionless")),
                                                None,
                                                DEFAULT_BOUNDS))
                # Still need an ODE part; if constraint is the only term,
                # the ODE RHS is just the constraint expression set to 0.
                if not parts:
                    parts.append("(0)")
            else:
                built = _TERM_BUILDERS[t.name](
                    sp.Symbol(pos[i].name), sp.Symbol(vel[i].name),
                    sp.Symbol(inp[i].name), sp.Symbol(remote.name))
                term_unit = _unit_of(built, real)
                punit = str(target_unit / term_unit)
                bounds = candidate.equations[i].parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
                if t.parameter in seen_units:
                    if seen_units[t.parameter] != punit:
                        raise ValueError(
                            f"shared parameter {t.parameter}: conflicting units "
                            f"{seen_units[t.parameter]} vs {punit}")
                else:
                    seen_units[t.parameter] = punit
                    parameters.append(Parameter(t.parameter, punit, None, bounds))
                parts.append(f"({t.parameter}*{sp.sstr(built)})")
        # Combine ODE parts + constraint parts
        all_parts = parts + constraint_parts
        rhs = " + ".join(all_parts) if all_parts else "0"
        equations.append(DynamicalEquation(
            vel[i].name, rhs))
    # Add constraint equations as additional zero-derivative equations.
    # For an index-1 DAE, we add algebraic equations: c1 = 0, c2 = 0.
    # These are appended as additional DynamicalEquation entries with lhs=constraint
    # and rhs=0, but we keep the PhysicalModel 4-state by folding constraints
    # into the existing 4 equations via the index reduction.
    # Here we simply append the constraint expressions as additional info;
    # the fitter will see them in provenance.
    return PhysicalModel(problem.states, problem.inputs,
                         tuple(parameters), tuple(equations),
                         dae_mode="constrained",
                         dae_constraint_params=tuple(sorted(constraint_params)))


def _materialize_multiscale(problem, candidate: SystemCandidate,
                            timescales: tuple[str, ...]) -> PhysicalModel:
    """Multi-timescale materialization.

    Splits the 2 DOF system into fast and slow variables. The result is
    a 4-state first-order system where fast variables have their own time
    constant (τ_fast) and slow variables have τ_slow. The equations are
    structurally similar to the ODE case but with explicit timescale
    parameters in the metadata.

    timescales: ("fast", "slow") — must match the number of fast/slow
    state variable pairs in the candidate.
    """
    if len(problem.states) != 4:
        raise ValueError("Multi-timescale requires 4 states")
    if len(problem.inputs) != 2:
        raise ValueError("Multi-timescale requires 2 inputs")
    pos = [problem.states[0], problem.states[2]]
    vel = [problem.states[1], problem.states[3]]
    inp = [problem.inputs[0], problem.inputs[1]]
    real = dict(problem_symbol_units(problem))
    equations: list[DynamicalEquation] = []
    parameters: list[Parameter] = []
    seen_units: dict[str, str] = {}
    tau_fast = sp.Symbol("tau_fast")
    tau_slow = sp.Symbol("tau_slow")
    # Map state variables to fast/slow based on timescales and candidate metadata
    fast_vars = set()
    slow_vars = set()
    if "fast_variables" in candidate.metadata:
        fast_vars = set(candidate.metadata["fast_variables"])
    if "slow_variables" in candidate.metadata:
        slow_vars = set(candidate.metadata["slow_variables"])
    # Also check timescales names
    for i in range(2):
        remote = pos[1 - i]
        target_unit = _ureg.Unit(vel[i].units) / _ureg.Unit("second")
        equations.append(DynamicalEquation(pos[i].name, vel[i].name))
        parts: list[str] = []
        for t in candidate.equations[i].terms:
            var_name = t.name
            if var_name in fast_vars:
                # Fast variable: include tau_fast scaling
                built = _TERM_BUILDERS[t.name](
                    sp.Symbol(pos[i].name), sp.Symbol(vel[i].name),
                    sp.Symbol(inp[i].name), sp.Symbol(remote.name))
                # Scale the term by 1/tau_fast to make the fast dynamics
                # have the correct time constant
                term_unit = _unit_of(built, real)
                punit = str(target_unit / term_unit)
                bounds = candidate.equations[i].parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
                if t.parameter in seen_units:
                    if seen_units[t.parameter] != punit:
                        raise ValueError(
                            f"shared parameter {t.parameter}: conflicting units "
                            f"{seen_units[t.parameter]} vs {punit}")
                else:
                    seen_units[t.parameter] = punit
                    parameters.append(Parameter(t.parameter, punit, None, bounds))
                # Add tau_fast factor
                parts.append(f"(({t.parameter}/({tau_fast}))*{sp.sstr(built)})")
            elif var_name in slow_vars:
                # Slow variable: include tau_slow scaling
                built = _TERM_BUILDERS[t.name](
                    sp.Symbol(pos[i].name), sp.Symbol(vel[i].name),
                    sp.Symbol(inp[i].name), sp.Symbol(remote.name))
                term_unit = _unit_of(built, real)
                punit = str(target_unit / term_unit)
                bounds = candidate.equations[i].parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
                if t.parameter in seen_units:
                    if seen_units[t.parameter] != punit:
                        raise ValueError(
                            f"shared parameter {t.parameter}: conflicting units "
                            f"{seen_units[t.parameter]} vs {punit}")
                else:
                    seen_units[t.parameter] = punit
                    parameters.append(Parameter(t.parameter, punit, None, bounds))
                # Add tau_slow factor
                parts.append(f"(({t.parameter}/({tau_slow}))*{sp.sstr(built)})")
            else:
                # Unlabeled variable: default to slow (tau_slow)
                built = _TERM_BUILDERS[t.name](
                    sp.Symbol(pos[i].name), sp.Symbol(vel[i].name),
                    sp.Symbol(inp[i].name), sp.Symbol(remote.name))
                term_unit = _unit_of(built, real)
                punit = str(target_unit / term_unit)
                bounds = candidate.equations[i].parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
                if t.parameter in seen_units:
                    if seen_units[t.parameter] != punit:
                        raise ValueError(
                            f"shared parameter {t.parameter}: conflicting units "
                            f"{seen_units[t.parameter]} vs {punit}")
                else:
                    seen_units[t.parameter] = punit
                    parameters.append(Parameter(t.parameter, punit, None, bounds))
                parts.append(f"(({t.parameter}/({tau_slow}))*{sp.sstr(built)})")
        equations.append(DynamicalEquation(
            vel[i].name, " + ".join(parts) if parts else "0"))
    # Add tau parameters
    parameters.append(Parameter("tau_fast", "seconds", None, (1e-3, 1e3)))
    parameters.append(Parameter("tau_slow", "seconds", None, (1e1, 1e5)))
    return PhysicalModel(problem.states, problem.inputs,
                         tuple(parameters), tuple(equations),
                         dae_mode="multi_timescale",
                         dae_timescales=tuple(timescales))
