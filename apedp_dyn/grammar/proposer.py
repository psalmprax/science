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
from typing import Any, Protocol, runtime_checkable

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

    def propose(self, problem: Any) -> Sequence[CandidateIR]:
        ...


class GrammarProposer:
    """Reference implementation: APEDP's own controlled enumeration."""

    def __init__(self, required:tuple[str, ...]=("u",), optional:tuple[str, ...]=("x", "x3", "v"),
                 max_terms: int = 4, coupled: bool = False,
                 proposer_id: str = "grammar-v1") -> None:
        self._required = tuple(required)
        self._optional = tuple(optional)
        self._max_terms = max_terms
        self._coupled = coupled
        self._proposer_id = proposer_id

    @property
    def proposer_id(self) -> str:
        return self._proposer_id

    def propose(self, problem: Any) -> Sequence[CandidateIR]:
        if self._coupled:
            return enumerate_coupled(required=self._required,
                                     optional=self._optional,
                                     max_terms=self._max_terms)
        return enumerate_candidates(required=self._required,
                                    optional=self._optional,
                                    max_terms=self._max_terms)


def _library_matrix(problem: Any, term_names: Sequence[str]) -> Any:
    """Numeric library columns for grammar terms on train data.

    Single-DOF only; terms needing the remote symbol are skipped (reported).
    Target convention matches the fitter's warm start: spline-estimated
    acceleration of states[1] over train.
    """
    import sympy as sp
    from scipy.interpolate import CubicSpline

    from apedp_dyn.grammar.mechanical_grammar import DEFAULT_BOUNDS, UnavailableGrammarTerm

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
        try:
            built = _TERM_BUILDERS[name](X, V, Gu, XR, sp.Symbol("gL"))
        except UnavailableGrammarTerm as error:
            # A term whose builder refuses (no series to read) is a column this
            # library cannot carry, not a reason to abandon the whole library.
            skipped.append(f"{name}: {error}")
            continue
        if XR in built.atoms(sp.Symbol):
            skipped.append(name)
            continue
        rexpr = built.subs(subs)
        # A scaled law carries its reference level as a fitted parameter. There
        # is no fitted value at library-build time, so the column cannot be
        # evaluated; emitting it anyway made `gL` reach lambdify unbound.
        unbound = sorted(str(s) for s in rexpr.atoms(sp.Symbol) if s not in syms)
        if unbound:
            skipped.append(f"{name}: unbound {', '.join(unbound)} "
                           "(no fitted value at library-build time)")
            continue
        col = np.asarray(sp.lambdify(syms, rexpr, modules=["numpy"])(*args),
                         dtype=float).ravel()
        if col.size == 1 and len(t) > 1:
            col = np.full(len(t), col[0])
        cols[name] = col
    target = CubicSpline(t, Y[:, 1]).derivative()(t)
    _ = DEFAULT_BOUNDS
    return cols, target, skipped


#: Two library columns that correlate above this on the training trajectory are
#: one direction of the regression, not two.
COLLINEARITY_LIMIT = 0.999


def drop_collinear_columns(
    cols: dict[str, np.ndarray], limit: float = COLLINEARITY_LIMIT
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Keep the first of each group of near-duplicate columns; report the rest.

    A thresholded sparse regression cannot choose between two columns that are
    the same vector up to scale: least squares splits the amplitude between
    them arbitrarily, and the threshold then removes whatever genuine term was
    left smallest. On the Duffing training trajectory ``sin_x`` and ``tanh_x``
    correlate with ``x`` at 0.99999 and ``cos_x`` with ``x2`` at -1.00000 (the
    state stays small, so each is its own first Taylor term), and the library
    selected ``{one, abs_x, log_abs_x, sqrt_abs_x}`` instead of the plant
    ``{u, x, x3, v}``.

    Order decides which survives, so a primitive listed before its
    re-parameterisations is the one kept. A constant column has no correlation
    and is never dropped by this rule. Returns the surviving columns in their
    original order and one line per dropped column naming what it duplicated.

    CHANGELOG and docs/SCAFFOLD_INVENTORY.md described this function on
    2026-11-03 and 2026-11-06; it was not in the repository until 2026-11-16,
    and ``tests/test_dyn_grammar_scope.py`` failed for that reason.
    """
    kept: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for name, column in cols.items():
        duplicate_of: tuple[str, float] | None = None
        if float(np.std(column)) > 0.0:
            for other, other_column in kept.items():
                if float(np.std(other_column)) == 0.0:
                    continue
                correlation = float(np.corrcoef(column, other_column)[0, 1])
                if abs(correlation) > limit:
                    duplicate_of = (other, correlation)
                    break
        if duplicate_of is None:
            kept[name] = column
        else:
            dropped.append(
                f"{name}: collinear with {duplicate_of[0]} on the training trajectory "
                f"(r = {duplicate_of[1]:+.5f}, limit {limit})"
            )
    return kept, dropped


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
        # Terms that re-parameterize primitives already in the library rather
        # than adding a new shape: sat_x~x, sat_v~v, sat_u~u, inv_x2~const,
        # and u_relax == u - v exactly. Threshold-based STRidge is unstable
        # under that kind of collinearity -- on the Duffing recovery it kept
        # u_relax and dropped v, because u_relax and u are near-duplicates on
        # trajectory data. They stay in _TERM_BUILDERS (crypto FIN_TERMS, the
        # IR family map and the UI all reference them) and remain reachable by
        # passing an explicit vocabulary; they are simply not part of the
        # *default* sparse library, matching DEFAULT_TOPOLOGY, which excludes
        # them for the same reason.
        # `langmuir_bind`/`hill2_bind` need their reference scale *fitted* per
        # candidate, which a fixed library column cannot carry; they are
        # reachable through `saturation_candidate()` / an explicit vocabulary,
        # not the default sparse library.
        _composite = ("u_relax", "sat_x", "sat_v", "sat_u", "inv_x2",
                      "langmuir_bind", "hill2_bind")
        # Terms whose builder declares itself unavailable on single-DOF trajectory
        # data (no mu_x/x_lag1/x_ma series in a problem): including them in the
        # default library only produces a refusal after the column was requested.
        _unavailable = ("zscore_x", "ma_x", "mom_x", "rsi_like", "bollinger",
                        "macd_like")
        self._vocabulary = tuple(vocabulary) if vocabulary is not None else tuple(
            k for k in _TERM_BUILDERS
            if k not in _nontrajectory and k not in _composite
            and k not in _unavailable)
        self._proposer_id = proposer_id
        self.last_skipped: list[str] = []

    @property
    def proposer_id(self) -> str:
        return self._proposer_id

    def propose(self, problem: Any) -> Sequence[CandidateIR]:
        cols, target, skipped = _library_matrix(problem, self._vocabulary)
        cols, collinear = drop_collinear_columns(cols)
        #: Why each requested term contributed no column, for a caller that asks.
        self.last_skipped = [*skipped, *collinear]
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
