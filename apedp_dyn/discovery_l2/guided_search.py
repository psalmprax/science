"""Exhaustive search in order of a prior, stopping when nothing is left to explain.

Status: experimental

``exhaustive_search`` fits every candidate. With a prior over structures
(``apedp_mmlm.equation_language.search_prior``) the candidates can be tried
most-plausible first, and the search can stop once the best candidate so far
leaves residuals with no detectable dependence on the inputs: any later
candidate is less plausible and can only improve the fit by fitting noise.

The stop test is model-free. Each residual is predicted from the residuals of
its ``k`` nearest neighbours in input space (leaving itself out). If that
prediction explains none of the residual variance (leave-one-out R^2 at most
0), no smooth structure is left. The test can be wrong in one direction that
matters: a wrong law whose misfit is too small to see at this noise and sample
size is accepted. How often is a measured quantity, not an assumption
(PREREG-MMLM-S2-2026-11).

If the candidates run out without a stop, the full search (including sums of
terms) is returned, so the answer is never worse than the unguided search's
for lack of a stop.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from apedp_dyn.discovery_l2 import exhaustive_search as es


@dataclass(frozen=True)
class GuidedResult:
    best: es.Scored
    evaluated: int
    candidates: int
    stopped_early: bool
    residual_structure: float


def neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's ``k`` nearest other rows after standardising the columns."""
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    out: np.ndarray = np.argsort(d2, axis=1)[:, :k]
    return out


def residual_structure(resid: np.ndarray, nn: np.ndarray) -> float:
    """Leave-one-out R^2 of predicting each residual from its neighbours' residuals; about -1/k for pure noise."""
    centred = resid - resid.mean()
    total = float(np.sum(centred ** 2))
    if total <= 0.0:
        return -1.0  # an exact fit leaves nothing to explain
    return 1.0 - float(np.sum((resid - resid[nn].mean(axis=1)) ** 2)) / total


class GuidedSearch:
    """A prior and the candidate orders it implies, computed once and reused for every dataset.

    Pricing every candidate with the prior costs more than fitting the few the
    search then needs, so the order for each (number of variables, size limit)
    is cached here. Build a new instance when the prior changes (for example
    after the self-adjusting model has learned a law).
    """

    def __init__(self, structure_bits: Callable[[es.Node], float], k: int = 8) -> None:
        self.structure_bits = structure_bits
        self.k = k
        self._orders: dict[tuple[int, int], tuple[list[tuple[es.Node, int]], np.ndarray, np.ndarray]] = {}

    def order(self, n_vars: int, max_complexity: int) -> tuple[list[tuple[es.Node, int]], np.ndarray, np.ndarray]:
        key = (n_vars, max_complexity)
        if key not in self._orders:
            nodes = es.enumerate_expressions(n_vars, max_complexity)
            nats = np.array([self.structure_bits(node) for node, _c in nodes]) * np.log(2.0)
            self._orders[key] = (nodes, nats, np.argsort(nats, kind="stable"))
        return self._orders[key]

    def search(self, X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int,
               stop: bool = True) -> GuidedResult:
        """Fit candidates most-plausible first; stop when the best leaves no residual structure."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        nodes, nats, order = self.order(X.shape[1], max_complexity)
        alphabet = X.shape[1] + len(es.UNARY) + len(es.BINARY)
        nn = neighbours(X, min(self.k, len(y) - 1))
        best: es.Scored | None = None
        structure = 1.0
        for count, index in enumerate(order, start=1):
            node, c = nodes[int(index)]
            column = es.evaluate(node, X)
            fit = es._fit([column], y)
            if fit is None:
                continue
            coef, rss = fit
            length = es.description_length(rss, len(y), c, 2, alphabet, float(nats[index]))
            if best is not None and length >= best.description_length:
                continue
            expression = es.to_string(node, names)
            best = es.Scored(expression, (expression,), c, 2, rss, length, tuple(float(v) for v in coef))
            if stop:
                structure = residual_structure(y - (coef[0] * column + coef[1]), nn)
                if structure <= 0.0:
                    return GuidedResult(best, count, len(nodes), True, structure)
        full = es.search(X, y, names, max_complexity, structure_bits=self.structure_bits)[0]
        return GuidedResult(full, len(nodes), len(nodes), False, structure)


def search_guided(
    X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int,
    structure_bits: Callable[[es.Node], float], *, k: int = 8, stop: bool = True,
) -> GuidedResult:
    """One-off convenience for :class:`GuidedSearch` (prices every candidate again on each call)."""
    return GuidedSearch(structure_bits, k).search(X, y, names, max_complexity, stop)
