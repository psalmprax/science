"""Sparse regression on a polynomial library, as a source of candidates.

Status: experimental

**Why.** In PREREG-ODE-BLIND-2026-11 the reference engine, sequentially
thresholded least squares over all monomials up to degree three with the
threshold chosen by BIC, recovered 255 of 310 components without noise where
the route recovered 199. In 50 of the 58 components it alone recovered, the
law's terms were not among the candidates the route's confirmation was given:
the polynomial library the refined path adds keeps at most six terms and
chooses differently. The route's gate, for its part, refused all eleven wrong
laws the reference reported without noise.

**What.** ``SparseLibraryProposer`` is that reference engine as a proposer.
It returns the support with the best BIC on the threshold path and, after it,
the next distinct supports, each as a fitted polynomial in the frame's own
column names. It decides nothing: its laws are candidates for the path's
confirmation like any search's.

**Limits.** A polynomial library can only propose polynomials. With noise the
reference engine's own laws were wrong in 23 of 310 components; whether the
gate keeps those out when they are handed to it is what has to be measured,
on systems this was not developed on.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_dyn.ode.weak_form import stlsq

#: the reference engine's grid (``scripts/run_odebench_main_search.py``)
THRESHOLDS = tuple(float(v) for v in np.logspace(-3, 0.5, 15))
EXACT_FLOOR = 1e-6          # exact data: fits below this share of the target's size are equally good


def supports(library: np.ndarray, y: np.ndarray, thresholds: Sequence[float] = THRESHOLDS) -> list[tuple[float, np.ndarray]]:
    """The distinct sparse fits on the threshold path as ``(BIC, coefficients)``, best BIC first."""
    floor = (EXACT_FLOOR * float(np.sqrt(np.mean(y ** 2)))) ** 2
    found: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for threshold in thresholds:
        coef: np.ndarray = stlsq(library, y, threshold)[:, 0]
        kept = tuple(int(i) for i in np.flatnonzero(coef))
        if not kept or not np.all(np.isfinite(coef)):
            continue
        rss = float(np.sum((library @ coef - y) ** 2))
        bic = len(y) * float(np.log(max(rss / len(y), floor, 1e-300))) + len(kept) * float(np.log(len(y)))
        if kept not in found or bic < found[kept][0]:
            found[kept] = (bic, coef)
    return sorted(found.values(), key=lambda item: item[0])


class SparseLibraryProposer(ProposerBase):
    """The sparse polynomial fits of the target on the frame's columns, best BIC first."""

    def __init__(self, degree: int = 3, max_proposals: int = 2, thresholds: Sequence[float] = THRESHOLDS,
                 target_col: str = "target",
                 strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL) -> None:
        self._degree, self._max = int(degree), int(max_proposals)
        self._thresholds = tuple(float(t) for t in thresholds)
        self._target_col = target_col
        self._strategy_archetype = strategy_archetype
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "sparse_library"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        powers = [p for p in itertools.product(range(self._degree + 1), repeat=len(features)) if sum(p) <= self._degree]
        library = np.column_stack([np.prod(X ** np.array(p), axis=1) for p in powers])
        symbols = [sp.Symbol(f) for f in features]
        fits = supports(library, y, self._thresholds)
        self.last_run = {"library_terms": len(powers), "distinct_supports": len(fits),
                         "terms_of_the_best": int(np.count_nonzero(fits[0][1])) if fits else 0}
        proposals = []
        for rank, (bic, coef) in enumerate(fits[: self._max]):
            law = sp.Add(*[sp.Float(float(c)) * sp.Mul(*[s ** k for s, k in zip(symbols, p, strict=True)])
                           for c, p in zip(coef, powers, strict=True) if c != 0.0])
            if not law.free_symbols:
                continue
            fn = sp.lambdify(symbols, law, "numpy")

            def compute_score(df: pd.DataFrame, fn: Any = fn, features: list[str] = features) -> np.ndarray:
                values = df[features].to_numpy(dtype=float)
                out: np.ndarray = np.asarray(fn(*values.T), dtype=float) * np.ones(len(df))
                return out

            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id, candidate_id=f"sparse:{rank}", formula=str(law),
                strategy_archetype=self._strategy_archetype, declared_scale_invariant=False,
                compute_score=compute_score, parameters={"bic": float(bic)},
                metadata={"rank_on_the_threshold_path": rank, "terms": int(np.count_nonzero(coef)),
                          "selection_pool_size": max(1, len(fits))}))
        return tuple(proposals)
