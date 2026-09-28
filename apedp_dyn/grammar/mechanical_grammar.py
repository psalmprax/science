"""D1.1 — mechanical grammar: second-order single-DOF systems xddot = f(x, v, u).

Controlled topology, not indiscriminate combinations. The default template is

    xddot = a0 + a1*x + a2*x^3 + a3*v + a4*|v|v + a5*sin(x) + a6*u

(x2, v2, x*v exist in the vocabulary for targeted studies but are excluded
from the default topology.) Subset enumeration + prune_admissible() realize
the D1.4 pipeline: dimension filter -> domain/operator filter (via gates 1-2)
-> duplicate canonicalization -> structural constraints, before any integration.
"""
from __future__ import annotations

import itertools
from typing import Any

import sympy as sp

from apedp_dyn.gates import (
    _unit_of,
    _ureg,
    dimensional_validity,
    equation_admissibility_structure,
)
from apedp_dyn.grammar.candidate_ir import (
    GRAMMAR_VERSION,
    CandidateDynamics,
    ScoredCandidate,
    Term,
)
from apedp_dyn.grammar.canonicalize import canonical_form, deduplicate
from apedp_dyn.model import DynamicalEquation, Parameter, PhysicalModel

# Grammar vocabulary: label -> builder over generic (X, V, U) symbols.
_TERM_BUILDERS = {
    "one": lambda X, V, U: sp.Integer(1),
    "x": lambda X, V, U: X,
    "x2": lambda X, V, U: X ** 2,
    "x3": lambda X, V, U: X ** 3,
    "v": lambda X, V, U: V,
    "abs_v_v": lambda X, V, U: sp.Abs(V) * V,
    "v2": lambda X, V, U: V ** 2,
    "u": lambda X, V, U: U,
    "x_v": lambda X, V, U: X * V,
    "sin_x": lambda X, V, U: sp.sin(X),
}

DEFAULT_TOPOLOGY = ("one", "x", "x3", "v", "abs_v_v", "sin_x", "u")

DEFAULT_BOUNDS = (-1e6, 1e6)


def validate(candidate: CandidateDynamics) -> None:
    """Structural validation of the IR itself (unknown terms, dup params)."""
    unknown = [t.name for t in candidate.terms if t.name not in _TERM_BUILDERS]
    if unknown:
        raise ValueError(f"unknown grammar terms: {unknown}")
    params = candidate.parameter_names()
    if len(set(params)) != len(params):
        raise ValueError(f"duplicate parameter names: {params}")


def full_topology_candidate(seed: int = 0) -> CandidateDynamics:
    """The full 7-term template; the fitter (not the grammar) kills terms."""
    terms = tuple(Term(name, f"a{i}") for i, name in enumerate(DEFAULT_TOPOLOGY))
    c = CandidateDynamics(
        lhs="xddot", terms=terms,
        metadata={"family": "mechanical_2nd_order", "order": 2,
                  "state_variables": ["x", "v"], "input_variables": ["u"]},
        candidate_id="mg1-full", parameter_bounds={f"a{i}": DEFAULT_BOUNDS for i in range(7)},
        generation_seed=seed)
    validate(c)
    return c


def enumerate_candidates(required: tuple[str, ...] = ("u",),
                         optional: tuple[str, ...] = ("one", "x", "x3", "v",
                                                      "abs_v_v", "sin_x"),
                         max_terms: int = 3, seed: int = 0,
                         id_prefix: str = "mg1") -> list[CandidateDynamics]:
    """Controlled subset enumeration containing `required`, up to max_terms."""
    for name in list(required) + list(optional):
        if name not in _TERM_BUILDERS:
            raise ValueError(f"unknown grammar term: {name}")
    out: list[CandidateDynamics] = []
    n = 0
    for r in range(0, max_terms - len(required) + 1):
        for combo in itertools.combinations(optional, r):
            names = tuple(required) + combo
            terms = tuple(Term(name, f"a{i}") for i, name in enumerate(names))
            c = CandidateDynamics(
                lhs="xddot", terms=terms,
                metadata={"family": "mechanical_2nd_order", "order": 2,
                          "state_variables": ["x", "v"], "input_variables": ["u"]},
                candidate_id=f"{id_prefix}-{n:04d}",
                parameter_bounds={f"a{i}": DEFAULT_BOUNDS for i in range(len(names))},
                generation_seed=seed)
            out.append(c)
            n += 1
    return out


def materialize(problem, candidate: CandidateDynamics) -> PhysicalModel:
    """IR -> executable first-order PhysicalModel (adds the kinematic equation).

    Generic grammar symbols map position -> states[0], velocity -> states[1],
    input -> inputs[0]. Parameter units derive as target-rate/term units, so
    gate 1 can still fail (and reject) genuinely inconsistent proposals.
    """
    validate(candidate)
    pos, vel = problem.states[0], problem.states[1]
    inp = problem.inputs[0] if problem.inputs else None
    X, V = sp.Symbol("gx"), sp.Symbol("gv")
    U = sp.Symbol("gu")
    real: dict[str, str] = dict(problem_symbol_units(problem))
    target_unit = _ureg.Unit(real[vel.name]) / _ureg.Unit("second")
    parts: list[str] = []
    parameters: list[Parameter] = []
    for t in candidate.terms:
        built = _TERM_BUILDERS[t.name](X, V, U)
        gu_sym = sp.Symbol(inp.name) if inp is not None else U
        rexpr = built.subs({X: sp.Symbol(pos.name), V: sp.Symbol(vel.name), U: gu_sym})
        if inp is None and U in rexpr.atoms(sp.Symbol):
            raise ValueError("candidate uses input but problem has none")
        term_unit = _unit_of(rexpr, real)
        punit = str(target_unit / term_unit)
        bounds = candidate.parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
        parameters.append(Parameter(t.parameter, punit, None, bounds))
        parts.append(f"({t.parameter}*{sp.sstr(rexpr)})")
    accel = " + ".join(parts) if parts else "0"
    equations = (DynamicalEquation(pos.name, vel.name),
                 DynamicalEquation(vel.name, accel))
    return PhysicalModel(problem.states, problem.inputs, tuple(parameters), equations)


def problem_symbol_units(problem) -> dict[str, str]:
    out = {s.name: s.units for s in problem.states}
    out.update({u.name: u.units for u in problem.inputs})
    return out


def prune_admissible(problem, candidates: list[CandidateDynamics]
                     ) -> list[ScoredCandidate]:
    """D1.4 pruning before expensive integration.

    materialize -> gate 1 (dimension) -> gate 2 (admissibility) -> dedup.
    Survivors carry gate1/2 provenance; rejects carry rejection_reason.
    Only survivors may proceed to the fitter (enforced in fit_candidate).
    """
    scored: list[ScoredCandidate] = []
    for c in candidates:
        prov: dict[str, Any] = ScoredCandidate(candidate=c, provenance={}).base_provenance()
        prov["canonical_form"] = canonical_form(c)
        try:
            model = materialize(problem, c)
        except (TypeError, ValueError) as e:
            prov["rejection_reason"] = f"materialize: {e}"
            scored.append(ScoredCandidate(c, prov))
            continue
        g1 = dimensional_validity(model)
        prov["gate1_status"] = {"passed": g1.passed, "detail": g1.detail}
        if not g1.passed:
            prov["rejection_reason"] = f"gate1: {g1.detail}"
            scored.append(ScoredCandidate(c, prov))
            continue
        g2 = equation_admissibility_structure(model)
        prov["gate2_status"] = {"passed": g2.passed, "detail": g2.detail}
        if not g2.passed:
            prov["rejection_reason"] = f"gate2: {g2.detail}"
            scored.append(ScoredCandidate(c, prov))
            continue
        prov["dimension_signature"] = str(
            (_ureg.Unit(problem.states[1].units) / _ureg.Unit("second")).dimensionality)
        scored.append(ScoredCandidate(c, prov, gates=(g1, g2)))
    survivors = [s for s in scored if "rejection_reason" not in s.provenance]
    # Deduplicate survivors by canonical form (stable order).
    seen: set[str] = set()
    unique: list[ScoredCandidate] = []
    for s in survivors:
        key = s.provenance["canonical_form"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def select_by_train_loss(fitted: list[ScoredCandidate]) -> ScoredCandidate:
    """Select lowest training loss; parsimony tiebreak within 1% (no weights)."""
    if not fitted:
        raise ValueError("nothing to select from")
    ordered = sorted(fitted, key=lambda s: (s.train_loss if s.train_loss is not None
                                            else float("inf")))
    best_loss = ordered[0].train_loss
    tied = [s for s in ordered
            if s.train_loss is not None and best_loss is not None
            and s.train_loss <= best_loss * 1.01]
    return min(tied, key=lambda s: len(s.candidate.terms))


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_TOPOLOGY",
    "deduplicate",
    "enumerate_candidates",
    "full_topology_candidate",
    "materialize",
    "problem_symbol_units",
    "prune_admissible",
    "select_by_train_loss",
    "validate",
]
