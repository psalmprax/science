"""Candidate-proposer seam (v1).

"Let other systems generate possibilities; let APEDP determine what deserves
belief." The seam is the IR, not the search: any proposer — APEDP's grammar,
PySR, SINDy, an LLM — delivers CandidateDynamics/SystemCandidate objects, and
everything downstream (prune -> fit -> gates -> adjudicate) is unchanged.

This module defines the protocol and adapts APEDP's own enumerators as the
reference implementation. External adapters translate foreign expressions
into IR terms; untranslatable constructs must raise rather than approximate.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from apedp_dyn.grammar.candidate_ir import CandidateDynamics, SystemCandidate, Term
from apedp_dyn.grammar.coupled import enumerate_coupled
from apedp_dyn.grammar.mechanical_grammar import (
    _TERM_BUILDERS,
    enumerate_candidates,
)

PROPOSER_VERSION = "1.0.0-ir-seam"
STRIDGE_REL_TOL = 0.05
STRIDGE_ITERS = 10

CandidateIR = CandidateDynamics | SystemCandidate


@runtime_checkable
class CandidateProposer(Protocol):
    """Anything that turns a problem into structural hypotheses."""

    @property
    def proposer_id(self) -> str:
        ...

    def propose(self, problem) -> Sequence[CandidateIR]:
        ...


class GrammarProposer:
    """Reference implementation: APEDP's own controlled enumeration."""

    def __init__(self, required=("u",), optional=("x", "x3", "v"),
                 max_terms: int = 4, coupled: bool = False,
                 proposer_id: str = "grammar-v1"):
        self._required = tuple(required)
        self._optional = tuple(optional)
        self._max_terms = max_terms
        self._coupled = coupled
        self._proposer_id = proposer_id

    @property
    def proposer_id(self) -> str:
        return self._proposer_id

    def propose(self, problem) -> Sequence[CandidateIR]:
        if self._coupled:
            return enumerate_coupled(required=self._required,
                                     optional=self._optional,
                                     max_terms=self._max_terms)
        return enumerate_candidates(required=self._required,
                                    optional=self._optional,
                                    max_terms=self._max_terms)


def _library_matrix(problem, term_names: Sequence[str]):
    """Numeric library columns for grammar terms on train data.

    Single-DOF only; terms needing the remote symbol are skipped (reported).
    Target convention matches the fitter's warm start: spline-estimated
    acceleration of states[1] over train.
    """
    import sympy as sp
    from scipy.interpolate import CubicSpline

    from apedp_dyn.grammar.mechanical_grammar import DEFAULT_BOUNDS

    t = np.asarray(problem.train["truth_t"], dtype=float)
    Y = np.asarray(problem.train["truth_y"], dtype=float)
    order = [s.name for s in problem.states]
    inames = [u.name for u in problem.inputs]
    U = np.array([problem.train["u_of_t"](ti) for ti in t],
                 dtype=float).reshape(len(t), -1)
    X, V, Gu, XR = sp.Symbol("gx"), sp.Symbol("gv"), sp.Symbol("gu"), sp.Symbol("gxr")
    cols: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    syms = [sp.Symbol(n) for n in order] + [sp.Symbol(n) for n in inames]
    args = [Y[:, i] for i in range(len(order))] + [U[:, i] for i in range(len(inames))]
    subs = {X: sp.Symbol(order[0]), V: sp.Symbol(order[1]),
            Gu: (sp.Symbol(inames[0]) if inames else Gu)}
    for name in term_names:
        built = _TERM_BUILDERS[name](X, V, Gu, XR)
        if XR in built.atoms(sp.Symbol):
            skipped.append(name)
            continue
        rexpr = built.subs(subs)
        col = np.asarray(sp.lambdify(syms, rexpr, modules=["numpy"])(*args),
                         dtype=float).ravel()
        if col.size == 1 and len(t) > 1:
            col = np.full(len(t), col[0])
        cols[name] = col
    target = CubicSpline(t, Y[:, 1]).derivative()(t)
    _ = DEFAULT_BOUNDS
    return cols, target, skipped


class LibraryProposer:
    """Second implementation: SINDy-flavored sparse regression proposer.

    Builds the grammar-term library on train data, iterative-thresholds
    (STRidge, frozen STRIDGE_REL_TOL/STRIDGE_ITERS), emits the surviving
    structure as IR. A genuinely different search algorithm behind the same
    IR seam: selection by sparsity, not enumeration + trajectory fit.
    """

    def __init__(self, vocabulary: Sequence[str] | None = None,
                 proposer_id: str = "library-stridge-v1"):
        # DAE auxiliaries, field terms (need spatial grids), and remote
        # symbols cannot be evaluated on single-DOF trajectory data.
        # x_v stays: evaluable here, rejected or kept on the merits.
        _nontrajectory = ("c1", "c2", "x_f", "v_f", "x_s", "v_s",
                            "grad_x", "laplacian_x", "xr", "dxr")
        self._vocabulary = tuple(vocabulary) if vocabulary is not None else tuple(
            k for k in _TERM_BUILDERS if k not in _nontrajectory)
        self._proposer_id = proposer_id

    @property
    def proposer_id(self) -> str:
        return self._proposer_id

    def propose(self, problem) -> Sequence[CandidateIR]:
        cols, target, _ = _library_matrix(problem, self._vocabulary)
        names = list(cols)
        A = np.column_stack([cols[n] for n in names])
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        for _ in range(STRIDGE_ITERS):
            keep = np.abs(coef) >= STRIDGE_REL_TOL * np.max(np.abs(coef))
            if int(np.sum(keep)) == 0:
                break
            coef_new, *_ = np.linalg.lstsq(A[:, keep], target, rcond=None)
            full = np.zeros_like(coef)
            full[keep] = coef_new
            if np.array_equal((full != 0), (coef != 0)):
                coef = full
                break
            coef = full
        kept = [n for n, c in zip(names, coef, strict=True) if c != 0]
        terms = tuple(Term(n, f"a{i}") for i, n in enumerate(kept))
        return [CandidateDynamics(lhs="xddot", terms=terms,
                                  metadata={"family": "mechanical_2nd_order",
                                            "proposer": self._proposer_id},
                                  candidate_id=f"{self._proposer_id}-0")]
