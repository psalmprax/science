"""Proposers that wrap or combine other proposers: physical units, a polynomial library, several at once.

Status: experimental

The main search path (``apedp_dyn.discovery_l2.main_search``) takes one
proposer and decides nothing about what it searches. Three results asked for
more than a bare search stage, and each is answered here by composition, with
the path itself untouched:

* **Units** (``UnitsReducingProposer``). The benchmark problems carry physical
  units and the path ignored them. If the target has the units of
  ``s = x1**a1 * x2**a2 * ...`` and ``p1, p2, ...`` are the independent
  dimensionless products of the inputs, every dimensionally consistent law is
  ``y = s * F(p1, p2, ...)``. The wrapped proposer searches for ``F`` on
  ``y / s`` against the ``p``, which are fewer than the inputs, and its
  formulas are rewritten in the original variables before they are returned.
  A law with a hidden dimensional constant is not of that form; the wrapper
  then returns formulas that do not fit, and the path's own checks turn them
  down, so wrong units cost a search and not a wrong answer.
* **A polynomial library** (``PolynomialLibraryProposer``). On ODEBench the
  path missed eleven polynomial laws that a sparse regression over monomials
  recovered at once. This is that regression as a proposer: it costs
  milliseconds and contributes one candidate.
* **Several at once** (``CompositeProposer``): the proposals of each member,
  in order, without duplicates.

None of these decides anything. Each returns ``CandidateProposal`` objects in
the original variable names, and confirmation is the path's.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed
from apedp_dyn.discovery_l2.guided_search import GuidedSearch
from apedp_dyn.discovery_l2.main_search import size_prior
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype

Units = Mapping[str, Mapping[str, Any]]


def formula_proposal(proposer_id: str, formula: str, features: Sequence[str], metadata: dict[str, Any] | None = None,
                     archetype: StrategyArchetype = StrategyArchetype.TACTICAL) -> CandidateProposal:
    """A proposal for a formula written in the feature names."""
    names = list(features)
    fn = sp.lambdify([sp.Symbol(n) for n in names], sp.sympify(formula, locals={n: sp.Symbol(n) for n in names}), "numpy")

    def compute_score(df: pd.DataFrame) -> np.ndarray:
        values = df[names].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            out: np.ndarray = np.asarray(fn(*values.T), dtype=float) * np.ones(len(df))
        return out

    return CandidateProposal(
        proposer_id=proposer_id, candidate_id=f"{proposer_id}:" + hashlib.sha256(formula.encode()).hexdigest()[:16],
        formula=formula, strategy_archetype=archetype, declared_scale_invariant=False,
        compute_score=compute_score, metadata=dict(metadata or {}))


def _fractions(units: Mapping[str, Any], bases: Sequence[str]) -> list[sp.Rational]:
    return [sp.nsimplify(units.get(b, 0), rational=True) for b in bases]


def dimensional_reduction(features: Sequence[str], units: Units, target_units: Mapping[str, Any]) -> dict[str, Any] | None:
    """The scale ``s`` with the target's units and the dimensionless products of the inputs, or ``None``.

    ``None`` means the units do not allow the rewriting: no product of the
    inputs has the target's units, or no units were given.
    """
    names = list(features)
    bases = sorted({b for n in names for b in units.get(n, {})} | set(target_units))
    if not bases:
        return None
    matrix = sp.Matrix([[_fractions(units.get(n, {}), bases)[k] for n in names] for k in range(len(bases))])
    target = sp.Matrix(_fractions(target_units, bases))
    if all(v == 0 for v in matrix) and all(v == 0 for v in target):
        return None  # nothing carries a unit: there is nothing to reduce
    symbols = sp.symbols(f"a0:{len(names)}")
    solutions = sp.linsolve((matrix, target), *symbols)
    if not solutions:
        return None
    particular = [sp.nsimplify(value.subs(dict.fromkeys(symbols, 0))) for value in next(iter(solutions))]
    groups = []
    for vector in matrix.nullspace():
        scale = sp.ilcm(*[sp.Rational(v).q for v in vector]) if any(sp.Rational(v).q != 1 for v in vector) else 1
        exponents = [sp.Rational(v) * scale for v in vector]
        if next(e for e in exponents if e != 0) < 0:
            exponents = [-e for e in exponents]
        groups.append(exponents)
    variables = [sp.Symbol(n, positive=True) for n in names]
    return {"scale": sp.Mul(*[v ** a for v, a in zip(variables, particular, strict=True)]),
            "scale_exponents": particular,
            "groups": [sp.Mul(*[v ** e for v, e in zip(variables, g, strict=True)]) for g in groups],
            "group_exponents": groups}


def _plain(expr: sp.Expr, names: Sequence[str]) -> sp.Expr:
    """The same expression in symbols without assumptions, as the rest of the package writes them."""
    return expr.xreplace({sp.Symbol(n, positive=True): sp.Symbol(n) for n in names})


class UnitsReducingProposer(ProposerBase):
    """Search for the dimensionless law, return it in the original variables."""

    def __init__(self, inner: ProposerBase | None, units: Units, target_units: Mapping[str, Any], *,
                 own_first_pass: bool = True, target_col: str = "target") -> None:
        self._inner = inner
        self._units, self._target_units = units, target_units
        self._own_first_pass = own_first_pass
        self._target_col = target_col
        #: what the last call did: the scale, the groups, or why no reduction was made
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "units_reduction"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        reduction = dimensional_reduction(features, self._units, self._target_units)
        if reduction is None:
            self.last_run = {"reduced": False, "reason": "the units do not determine a scale for the target"}
            return list(self._inner.fit_and_propose(frame, target_col=target, guesses=guesses, **kwargs)) if self._inner else []
        X = frame[features].to_numpy(dtype=float)
        symbols = [sp.Symbol(n, positive=True) for n in features]
        fractional = [n for n, a in zip(features, reduction["scale_exponents"], strict=True) if sp.Rational(a).q != 1]
        if any(np.any(X[:, features.index(n)] <= 0) for n in fractional) or np.any(X == 0):
            self.last_run = {"reduced": False, "reason": "a fractional power or a division would meet a non-positive value"}
            return list(self._inner.fit_and_propose(frame, target_col=target, guesses=guesses, **kwargs)) if self._inner else []
        with np.errstate(all="ignore"):
            scale = np.asarray(sp.lambdify(symbols, reduction["scale"], "numpy")(*X.T), dtype=float) * np.ones(len(X))
            columns = [np.asarray(sp.lambdify(symbols, g, "numpy")(*X.T), dtype=float) * np.ones(len(X)) for g in reduction["groups"]]
        if not np.all(np.isfinite(scale)) or np.any(scale == 0) or not all(np.all(np.isfinite(c)) for c in columns):
            self.last_run = {"reduced": False, "reason": "the scale or a dimensionless product is not finite on the data"}
            return list(self._inner.fit_and_propose(frame, target_col=target, guesses=guesses, **kwargs)) if self._inner else []
        scale_expr = _plain(reduction["scale"], features)
        group_names = [f"p{i}" for i in range(len(columns))]
        back = {sp.Symbol(p): _plain(g, features) for p, g in zip(group_names, reduction["groups"], strict=True)}
        self.last_run = {"reduced": True, "scale": str(scale_expr), "groups": {p: str(back[sp.Symbol(p)]) for p in group_names},
                         "variables_before": len(features), "variables_after": len(columns)}
        reduced_target = frame[target].to_numpy(dtype=float) / scale
        formulas: list[tuple[str, str]] = [("scale_alone", str(scale_expr))]  # with no product left, the law is a constant times the scale
        if columns:
            reduced_X = np.column_stack(columns)
            if self._own_first_pass:
                first = search_confirmed(GuidedSearch(size_prior), reduced_X, reduced_target, group_names,
                                         6 if len(group_names) <= 5 else 5)
                reduced = sp.sympify(es.model_string(first.best), locals={p: sp.Symbol(p) for p in group_names})
                formulas.append(("first_pass_on_reduced", str(scale_expr * reduced.xreplace(back))))
            if self._inner is not None:
                reduced_frame = pd.DataFrame({**{p: reduced_X[:, i] for i, p in enumerate(group_names)}, target: reduced_target})
                for proposal in self._inner.fit_and_propose(reduced_frame, target_col=target):
                    try:
                        inner_expr = sp.sympify(proposal.formula, locals={p: sp.Symbol(p) for p in group_names})
                    except (sp.SympifyError, TypeError, SyntaxError):
                        continue
                    formulas.append((proposal.proposer_id, str(scale_expr * inner_expr.xreplace(back))))
        out, seen = [], set()
        for source, formula in formulas:
            if formula not in seen:
                seen.add(formula)
                out.append(formula_proposal(self.proposer_id, formula, features, {"found_by": source, **self.last_run}))
        return out


class PolynomialLibraryProposer(ProposerBase):
    """Sparse regression over all monomials up to a degree; the threshold is chosen by BIC."""

    def __init__(self, degree: int = 3, max_features: int = 6, target_col: str = "target") -> None:
        self._degree, self._max_features, self._target_col = degree, max_features, target_col

    @property
    def proposer_id(self) -> str:
        return "polynomial_library"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        if not features or len(features) > self._max_features:
            return []
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        powers = [p for p in itertools.product(range(self._degree + 1), repeat=len(features)) if sum(p) <= self._degree]
        if len(powers) >= len(y):
            return []
        library = np.column_stack([np.prod(X ** np.array(p), axis=1) for p in powers])
        norms = np.sqrt(np.mean(library ** 2, axis=0))
        if not np.all(np.isfinite(library)) or np.any(norms == 0):
            return []
        scaled = library / norms  # thresholds act on contributions, not on coefficients in arbitrary units
        floor = (1e-6 * float(np.sqrt(np.mean(y ** 2)))) ** 2
        best: tuple[float, np.ndarray] | None = None
        for threshold in np.logspace(-6, 0, 19) * float(np.sqrt(np.mean(y ** 2))):
            coef = np.linalg.lstsq(scaled, y, rcond=None)[0]
            for _ in range(20):
                keep = np.abs(coef) >= threshold
                if not keep.any():
                    coef = np.zeros_like(coef)
                    break
                refit = np.zeros_like(coef)
                refit[keep] = np.linalg.lstsq(scaled[:, keep], y, rcond=None)[0]
                if np.array_equal(refit != 0, coef != 0):
                    coef = refit
                    break
                coef = refit
            rss = float(np.sum((scaled @ coef - y) ** 2))
            bic = len(y) * math.log(max(rss / len(y), floor, 1e-300)) + int(np.count_nonzero(coef)) * math.log(len(y))
            if best is None or bic < best[0]:
                best = (bic, coef / norms)
        assert best is not None
        symbols = [sp.Symbol(n) for n in features]
        law = sp.Add(*[sp.Float(float(c)) * sp.Mul(*[s ** k for s, k in zip(symbols, p, strict=True)])
                       for c, p in zip(best[1], powers, strict=True) if c != 0.0])
        if not law.free_symbols:
            return []
        return [formula_proposal(self.proposer_id, str(law), features, {"terms": int(np.count_nonzero(best[1]))})]


class CompositeProposer(ProposerBase):
    """The proposals of several proposers, in order, each formula once. A member that fails is recorded, not fatal."""

    def __init__(self, members: Sequence[ProposerBase], max_proposals: int = 11) -> None:
        self._members, self._max = list(members), max_proposals
        self.failures: list[str] = []

    @property
    def proposer_id(self) -> str:
        return "composite"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        self.failures = []
        per_member: list[list[CandidateProposal]] = []
        for member in self._members:
            try:
                per_member.append(list(member.fit_and_propose(train_data, target_col=target_col, guesses=guesses, **kwargs)))
            except Exception as exc:  # noqa: BLE001 - one search stage failing must not cost the others their candidates
                self.failures.append(f"{member.proposer_id}: {type(exc).__name__}: {exc}"[:200])
                per_member.append([])
        out: list[CandidateProposal] = []
        seen: set[str] = set()
        for rank in range(max((len(p) for p in per_member), default=0)):  # round-robin: every member's best comes first
            for proposals in per_member:
                if rank < len(proposals) and proposals[rank].formula not in seen and len(out) < self._max:
                    seen.add(proposals[rank].formula)
                    out.append(proposals[rank])
        return out
