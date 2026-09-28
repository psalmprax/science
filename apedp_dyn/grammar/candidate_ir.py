"""D1.1/D1.7 — typed candidate IR.

The grammar never emits raw symbolic trees. It emits CandidateDynamics:
a left-hand side, a list of family-tagged Terms with named parameters, and
metadata. Everything downstream (gates, fitter, simulator) consumes this IR.
Provenance (D1.7) travels in ScoredCandidate.provenance from day one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GRAMMAR_VERSION = "mechanical-2nd-order v1"

# Term families. Restoring/dissipative/forcing tags drive physical
# interpretation and future constraints; the fitter never sees them.
TERM_FAMILIES: dict[str, tuple[str, ...]] = {
    "restoring": ("x", "x2", "x3", "sin_x"),
    "dissipative": ("v", "abs_v_v", "v2"),
    "forcing": ("u",),
    "coupling": ("x_v", "xr", "dxr"),
    "constant": ("one",),
}

TERM_FAMILY_OF: dict[str, str] = {
    term: fam for fam, terms in TERM_FAMILIES.items() for term in terms
}


@dataclass(frozen=True)
class Term:
    """One candidate term: grammar-vocabulary name + owning parameter name."""

    name: str
    parameter: str

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
        return tuple(t.parameter for t in self.terms)


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


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate plus its full audit trail (D1.7 provenance)."""

    candidate: CandidateDynamics | SystemCandidate
    provenance: dict[str, Any] = field(default_factory=dict)
    gates: tuple = ()
    fit: Any = None
    train_loss: float | None = None
    holdout_rmse: float | None = None

    def base_provenance(self) -> dict[str, Any]:
        c = self.candidate
        if isinstance(c, SystemCandidate):
            term_set: Any = [list(e.term_set()) for e in c.equations]
            bounds = {k: v for e in c.equations for k, v in e.parameter_bounds.items()}
        else:
            term_set = list(c.term_set())
            bounds = dict(c.parameter_bounds)
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
