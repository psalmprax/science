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
    CandidateDynamics,
    ScoredCandidate,
    Term,
)
from apedp_dyn.grammar.canonicalize import canonical_form, deduplicate
from apedp_dyn.model import DynamicalEquation, Parameter, PhysicalModel

# Grammar vocabulary: label -> builder over generic (X, V, U, XR) symbols,
# where XR is the remote position (2-DOF coupling). Single-DOF materialize
# leaves XR unmapped so xr-terms die loudly at gate 1, never silently.
_TERM_BUILDERS = {
    "one": lambda X, V, U, XR, SX: sp.Integer(1),
    "x": lambda X, V, U, XR, SX: X,
    "x2": lambda X, V, U, XR, SX: X ** 2,
    "x3": lambda X, V, U, XR, SX: X ** 3,
    "v": lambda X, V, U, XR, SX: V,
    "abs_v_v": lambda X, V, U, XR, SX: sp.Abs(V) * V,
    "v2": lambda X, V, U, XR, SX: V ** 2,
    "u": lambda X, V, U, XR, SX: U,
    "x_v": lambda X, V, U, XR, SX: X * V,
    "sin_x": lambda X, V, U, XR, SX: sp.sin(X),
    "cos_x": lambda X, V, U, XR, SX: sp.cos(X),
    "tanh_x": lambda X, V, U, XR, SX: sp.tanh(X),
    "sign_x": lambda X, V, U, XR, SX: sp.sign(X),
    "xr": lambda X, V, U, XR, SX: XR,
    "dxr": lambda X, V, U, XR, SX: XR - X,  # typed relative displacement Δx
    "c1": lambda X, V, U, XR, SX: X ** 0,  # constraint term 1 (dimensionless)
    "c2": lambda X, V, U, XR, SX: X ** 0,  # constraint term 2 (dimensionless)
    "x_f": lambda X, V, U, XR, SX: X,  # fast-state variable
    "v_f": lambda X, V, U, XR, SX: V,  # fast velocity
    "x_s": lambda X, V, U, XR, SX: X,  # slow-state variable
    "v_s": lambda X, V, U, XR, SX: V,  # slow velocity
    "grad_x": lambda X, V, U, XR, SX: sp.symbols("grad_x"),  # spatial gradient ∇x
    "laplacian_x": lambda X, V, U, XR, SX: sp.symbols("laplacian_x"),  # ∇²x
    "sat_x": lambda X, V, U, XR, SX: X / (1 + sp.Abs(X)),
    "sat_v": lambda X, V, U, XR, SX: V / (1 + sp.Abs(V)),
    "sat_u": lambda X, V, U, XR, SX: U / (1 + sp.Abs(U)),
    "inv_x": lambda X, V, U, XR, SX: 1 / (1 + sp.Abs(X)),
    "inv_x2": lambda X, V, U, XR, SX: 1 / (1 + X ** 2),
    "u_relax": lambda X, V, U, XR, SX: U - V,  # relative relaxation impulse
    "log_abs_x": lambda X, V, U, XR, SX: sp.log(1 + sp.Abs(X)),
    "sqrt_abs_x": lambda X, V, U, XR, SX: sp.sqrt(1 + sp.Abs(X)),
    "abs_x": lambda X, V, U, XR, SX: sp.Abs(X),
    "abs_u": lambda X, V, U, XR, SX: sp.Abs(U),
    "x_u": lambda X, V, U, XR, SX: X * U,
    "v_u": lambda X, V, U, XR, SX: V * U,
    "abs_x_v": lambda X, V, U, XR, SX: sp.Abs(X) * V,
    "abs_v_u": lambda X, V, U, XR, SX: sp.Abs(V) * U,
    # Trading/finance oriented terms
    "ret_x": lambda X, V, U, XR, SX: X / (1 + X),  # approximate return-like transform
    "zscore_x": lambda X, V, U, XR, SX: (X - sp.symbols("mu_x")) / (sp.symbols("sigma_x") + 1e-12),
    "ma_x": lambda X, V, U, XR, SX: (X + sp.symbols("x_lag1")) / 2,  # moving average placeholder
    "mom_x": lambda X, V, U, XR, SX: X - sp.symbols("x_lag1"),  # momentum
    "rsi_like": lambda X, V, U, XR, SX: sp.tanh(X - sp.symbols("x_ma")),  # RSI-like bounded momentum
    "bollinger": lambda X, V, U, XR, SX: (X - sp.symbols("x_ma")) / (sp.symbols("x_std") + 1e-12),
    "macd_like": lambda X, V, U, XR, SX: (sp.symbols("x_ema_fast") - sp.symbols("x_ema_slow")),
    # Saturating binding laws need a reference scale with the state's units:
    # `X/(L+|X|)` is Langmuir/Michaelis-Menten occupancy and `X^2/(L^2+X^2)` its
    # cooperative (Hill n=2) form. L arrives as the term's `scale_parameter`;
    # without it an Add of `1` to a dimensional X is a gate-1 violation.
    "langmuir_bind": lambda X, V, U, XR, SX: X / (SX + sp.Abs(X)),
    "hill2_bind": lambda X, V, U, XR, SX: X ** 2 / (SX ** 2 + X ** 2),
    "hill3_bind": lambda X, V, U, XR, SX: X ** 3 / (SX ** 3 + X ** 3),
}

DEFAULT_TOPOLOGY = ("one", "x", "x3", "v", "abs_v_v", "sin_x", "u")

DEFAULT_BOUNDS = (-1e6, 1e6)
# A saturation reference scale is positive (it is a half-saturation level).
SATURATION_SCALE_BOUNDS = (1e-9, 1e9)
_SCALED_TERMS = frozenset({"langmuir_bind", "hill2_bind"})


def validate(candidate: CandidateDynamics) -> None:
    """Structural validation of the IR itself (unknown terms, dup params)."""
    unknown = [t.name for t in candidate.terms if t.name not in _TERM_BUILDERS]
    if unknown:
        raise ValueError(f"unknown grammar terms: {unknown}")
    for t in candidate.terms:
        if t.name in _SCALED_TERMS and not t.scale_parameter:
            raise ValueError(f"term {t.name!r} requires a scale_parameter")
        if t.scale_parameter is not None and t.name not in _SCALED_TERMS:
            raise ValueError(f"term {t.name!r} does not take a scale_parameter")
    params = candidate.parameter_names()
    if len(set(params)) != len(params):
        raise ValueError(f"duplicate parameter names: {params}")


def saturation_candidate(
    terms: tuple[str, ...] = ("langmuir_bind",),
    seed: int = 0,
) -> CandidateDynamics:
    """Typed saturation hypothesis: xddot = sum_i a_i * binding_i(x; s_i).

    Each term carries its own positive reference scale `s_i` in the state's
    units, so the candidate is dimensionally well-formed and gate 1 can still
    reject it on a genuinely inconsistent target.
    """
    for name in terms:
        if name not in _SCALED_TERMS:
            raise ValueError(f"not a saturation term: {name!r}")
    sat = tuple(
        Term(name, f"a{i}", scale_parameter=f"s{i}")
        for i, name in enumerate(terms)
    )
    bounds: dict[str, tuple[float, float]] = {}
    for i in range(len(terms)):
        bounds[f"a{i}"] = DEFAULT_BOUNDS
        bounds[f"s{i}"] = SATURATION_SCALE_BOUNDS
    candidate = CandidateDynamics(
        lhs="xddot", terms=sat,
        metadata={"family": "mechanical_2nd_order", "order": 2,
                  "state_variables": ["x", "v"], "input_variables": ["u"],
                  "saturation_law": "langmuir" if terms == ("langmuir_bind",) else "mixed"},
        candidate_id="mg1-saturation",
        parameter_bounds=bounds,
        generation_seed=seed,
    )
    validate(candidate)
    return candidate


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
    XR = sp.Symbol("gxr")
    SX = sp.Symbol("gL")
    real: dict[str, str] = dict(problem_symbol_units(problem))
    target_unit = _ureg.Unit(real[vel.name]) / _ureg.Unit("second")
    parts: list[str] = []
    parameters: list[Parameter] = []
    for t in candidate.terms:
        built = _TERM_BUILDERS[t.name](X, V, U, XR, SX)
        gu_sym = sp.Symbol(inp.name) if inp is not None else U
        rexpr = built.subs({X: sp.Symbol(pos.name), V: sp.Symbol(vel.name),
                            U: gu_sym, XR: sp.Symbol("gxr_remote")})
        if t.scale_parameter is not None:
            # The scale carries the state's unit; register it before the unit
            # check so a dimensional Add is proven compatible, not assumed.
            real[t.scale_parameter] = real[pos.name]
            rexpr = rexpr.subs({SX: sp.Symbol(t.scale_parameter)})
        if inp is None and U in rexpr.atoms(sp.Symbol):
            raise ValueError("candidate uses input but problem has none")
        term_unit = _unit_of(rexpr, real)
        punit = str(target_unit / term_unit)
        bounds = candidate.parameter_bounds.get(t.parameter, DEFAULT_BOUNDS)
        parameters.append(Parameter(t.parameter, punit, None, bounds))
        if t.scale_parameter is not None:
            sbounds = candidate.parameter_bounds.get(
                t.scale_parameter, SATURATION_SCALE_BOUNDS)
            parameters.append(Parameter(t.scale_parameter, real[pos.name], None,
                                        sbounds, role="scale"))
        parts.append(f"({t.parameter}*{sp.sstr(rexpr)})")
    accel = " + ".join(parts) if parts else "0"
    equations = (DynamicalEquation(pos.name, vel.name),
                 DynamicalEquation(vel.name, accel))
    return PhysicalModel(problem.states, problem.inputs, tuple(parameters), equations)


def problem_symbol_units(problem) -> dict[str, str]:
    out = {s.name: s.units for s in problem.states}
    out.update({u.name: u.units for u in problem.inputs})
    return out


def prune_admissible(problem, candidates: list) -> list[ScoredCandidate]:
    """D1.4 pruning before expensive integration.

    materialize -> gate 1 (dimension) -> gate 2 (admissibility) -> dedup.
    Accepts single-DOF CandidateDynamics and D2b SystemCandidate.
    Survivors carry gate1/2 provenance; rejects carry rejection_reason.
    Only survivors may proceed to the fitter (enforced in fit_candidate).
    """
    from apedp_dyn.grammar.candidate_ir import SystemCandidate
    from apedp_dyn.grammar.coupled import canonical_system, materialize_system

    scored: list[ScoredCandidate] = []
    for c in candidates:
        prov: dict[str, Any] = ScoredCandidate(candidate=c, provenance={}).base_provenance()
        prov["canonical_form"] = (canonical_system(c) if isinstance(c, SystemCandidate)
                                  else canonical_form(c))
        try:
            model = (materialize_system(problem, c) if isinstance(c, SystemCandidate)
                     else materialize(problem, c))
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

    def n_terms(s: ScoredCandidate) -> int:
        # Both IR forms expose parameter_names() by contract; probing for it
        # left the other arm (candidate.terms) untypeable and unreachable.
        return len(s.candidate.parameter_names())

    ordered = sorted(fitted, key=lambda s: (s.train_loss if s.train_loss is not None
                                            else float("inf")))
    best_loss = ordered[0].train_loss
    tied = [s for s in ordered
            if s.train_loss is not None and best_loss is not None
            and s.train_loss <= best_loss * 1.01]
    return min(tied, key=n_terms)


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
