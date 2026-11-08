"""D1.1/D1.7 — typed candidate IR.

The grammar never emits raw symbolic trees. It emits CandidateDynamics:
a left-hand side, a list of family-tagged Terms with named parameters, and
metadata. Everything downstream (gates, fitter, simulator) consumes this IR.
Provenance (D1.7) travels in ScoredCandidate.provenance from day one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

GRAMMAR_VERSION = "mechanical-2nd-order v1"

# Term families. Restoring/dissipative/forcing tags drive physical
# interpretation and future constraints; the fitter never sees them.
TERM_FAMILIES: dict[str, tuple[str, ...]] = {
    "restoring": ("x", "x2", "x3", "sin_x", "grad_x", "laplacian_x"),
    "dissipative": ("v", "abs_v_v", "v2"),
    "forcing": ("u",),
    "coupling": ("x_v", "xr", "dxr"),
    "constant": ("one",),
    # `langmuir_bind`/`hill2_bind` are dimensionally scaled forms: the reference
    # level travels in `Term.scale_parameter` (state units), never as a bare
    # `1 + X` add. `sat_*`/`inv_x2` remain dimensionless-only vocabulary.
    "saturation": ("sat_x", "sat_v", "sat_u", "inv_x2", "u_relax",
                   "langmuir_bind", "hill2_bind"),
}

TERM_FAMILY_OF: dict[str, str] = {
    term: fam for fam, terms in TERM_FAMILIES.items() for term in terms
}


@dataclass(frozen=True)
class Term:
    """One candidate term: grammar-vocabulary name + owning parameter name.

    `scale_parameter` names a second fitted parameter with the state's units,
    used only by dimensionally-scaled saturation terms (`langmuir_bind`,
    `hill2_bind`). It is part of the candidate's parameter set: the fitter
    estimates it and provenance records it.
    """

    name: str
    parameter: str
    scale_parameter: str | None = None

    @property
    def family(self) -> str:
        return TERM_FAMILY_OF[self.name]


@dataclass(frozen=True)
class CandidateDynamics:
    """Frozen structural proposal. Numbers come later, from the fitter."""

    lhs: str
    terms: tuple[Term, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_id: str = ""
    grammar_version: str = GRAMMAR_VERSION
    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    generation_seed: int = 0
    parent_candidates: tuple[str, ...] = ()

    def term_set(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.terms)

    def parameter_names(self) -> tuple[str, ...]:
        out: list[str] = []
        for t in self.terms:
            out.append(t.parameter)
            if t.scale_parameter is not None:
                out.append(t.scale_parameter)
        return tuple(out)


@dataclass(frozen=True)
class SystemCandidate:
    """Multi-DOF proposal: one single-DOF CandidateDynamics per mass (D2b).

    Parameter names must be globally unique across equations (the generator
    guarantees this); the fitter estimates the union jointly.
    """

    equations: tuple[CandidateDynamics, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_id: str = ""
    grammar_version: str = GRAMMAR_VERSION
    generation_seed: int = 0
    parent_candidates: tuple[str, ...] = ()

    def term_sets(self) -> tuple[tuple[str, ...], ...]:
        return tuple(e.term_set() for e in self.equations)

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p for e in self.equations for p in e.parameter_names())


@runtime_checkable
class CandidatePayload(Protocol):
    """What a scored record may read without assuming a term grammar.

    ``CandidateDynamics`` and ``SystemCandidate`` both satisfy it, and so does a
    free-form proposal reference: it has an identity and provenance metadata, but
    no term set to canonicalise. The referee can therefore carry either payload
    without being told that an ungrammatical proposal is a grammar candidate.
    """

    candidate_id: str
    metadata: dict[str, Any]
    grammar_version: str
    generation_seed: int
    parent_candidates: tuple[str, ...]

    def parameter_names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate plus its full audit trail (D1.7 provenance)."""

    candidate: CandidateDynamics | SystemCandidate | CandidatePayload
    provenance: dict[str, Any] = field(default_factory=dict)
    gates: tuple[Any, ...] = ()
    fit: Any = None
    train_loss: float | None = None
    holdout_rmse: float | None = None

    def base_provenance(self) -> dict[str, Any]:
        c = self.candidate
        if isinstance(c, SystemCandidate):
            term_set: Any = [list(e.term_set()) for e in c.equations]
            bounds = {k: v for e in c.equations for k, v in e.parameter_bounds.items()}
        elif isinstance(c, CandidateDynamics):
            term_set = list(c.term_set())
            bounds = dict(c.parameter_bounds)
        else:
            # A free-form reference has no term set: recording none is the honest
            # provenance, and pretending it had one would let a gate rank a grammar
            # the proposal never came from.
            term_set = []
            bounds = {}
        return {
            "candidate_id": c.candidate_id,
            "grammar_version": c.grammar_version,
            "term_set": term_set,
            "parameter_names": list(c.parameter_names()),
            "parameter_bounds": bounds,
            "generation_seed": c.generation_seed,
            "parent_candidates": list(c.parent_candidates),
            "metadata": dict(c.metadata),
        }
