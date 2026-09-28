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
    ScoredCandidate,
    SystemCandidate,
    Term,
)
from apedp_dyn.grammar.mechanical_grammar import (
    DEFAULT_BOUNDS,
    _TERM_BUILDERS,
    problem_symbol_units,
)
from apedp_dyn.model import DynamicalEquation, Parameter, PhysicalModel


def enumerate_coupled(required: tuple[str, ...] = ("u",),
                      optional: tuple[str, ...] = ("x", "v", "xr", "dxr"),
                      max_terms: int = 4, seed: int = 0,
                      id_prefix: str = "mg2",
                      coupling_mode: str = "general") -> list[SystemCandidate]:
    """Per-DOF controlled subsets; globally unique param names (q-counter).

    coupling_mode:
      general — independent coupling per mass (k12 ≠ k21 permitted).
      conservative_reciprocal — dxr (relative-displacement) terms share one
        parameter "kc" across masses (k12 = k21). This is a system-class
        constraint for the reciprocal spring, NOT a universal physical rule.
    """
    if coupling_mode not in ("general", "conservative_reciprocal"):
        raise ValueError(f"unknown coupling_mode {coupling_mode!r}")
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
                    else:
                        pname = f"q{counter}"
                        counter += 1
                    terms.append(Term(name, pname))
                    bounds[pname] = DEFAULT_BOUNDS
                options.append(CandidateDynamics(
                    lhs="xddot", terms=tuple(terms),
                    metadata={"family": "mechanical_2nd_order_2dof", "order": 2,
                              "dof": dof, "coupling_mode": coupling_mode,
                              "state_variables": ["x", "v"],
                              "input_variables": ["u"]},
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
                metadata={"family": "mechanical_2nd_order_2dof",
                          "coupling_mode": coupling_mode},
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
    """System IR -> executable 4-state first-order PhysicalModel."""
    if len(candidate.equations) != 2 or len(problem.states) != 4:
        raise ValueError("D2b scope is exactly 2 DOF / 4 states")
    if len(problem.inputs) != 2:
        raise ValueError("D2b scope needs exactly 2 inputs")
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
                # Shared parameter (conservative_reciprocal kc): units must agree.
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
