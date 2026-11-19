"""An early stop that has to be confirmed before the guided search returns.

Status: experimental

``GuidedSearch.search`` stops at the first candidate that leaves no structure in
its residuals. In its preregistered test (PREREG-MMLM-S2-2026-11) that rule
returned a near-miss law 7 times in 128 runs, one more than the limit: a simpler
wrong law can pass the residual check before the true one is reached.

``search_confirmed`` keeps the same order and the same residual check and changes
only when it returns. After the first candidate with no residual structure at
position ``c``, it keeps fitting until position ``max(patience * c, c + floor)``
and returns the candidate with the shortest description length among those that
left no structure. A better structure-free candidate found inside the window
extends the window from its own position. The cost is a bounded multiple of the
unconfirmed stop; the full search remains the fallback when nothing qualifies.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.guided_search import (
    GuidedResult,
    GuidedSearch,
    neighbours,
    residual_structure,
)


def search_confirmed(
    guided: GuidedSearch, X: np.ndarray, y: np.ndarray, names: Sequence[str], max_complexity: int,
    *, patience: float = 3.0, floor: int = 300,
) -> GuidedResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    nodes, nats, order = guided.order(X.shape[1], max_complexity)
    alphabet = X.shape[1] + len(es.UNARY) + len(es.BINARY)
    nn = neighbours(X, min(guided.k, len(y) - 1))
    best: es.Scored | None = None
    deadline: float | None = None
    for count, index in enumerate(order, start=1):
        if deadline is not None and count > deadline:
            assert best is not None
            return GuidedResult(best, count - 1, len(nodes), True, 0.0)
        node, c = nodes[int(index)]
        column = es.evaluate(node, X)
        fit = es._fit([column], y)
        if fit is None:
            continue
        coef, rss = fit
        length = es.description_length(rss, len(y), c, 2, alphabet, float(nats[index]))
        if best is not None and length >= best.description_length:
            continue
        if residual_structure(y - (coef[0] * column + coef[1]), nn) > 0.0:
            continue
        expression = es.to_string(node, names)
        best = es.Scored(expression, (expression,), c, 2, rss, length, tuple(float(v) for v in coef))
        deadline = max(deadline or 0.0, patience * count, count + floor)
    if best is not None:  # every candidate was fitted before the window closed: not an early stop
        return GuidedResult(best, len(nodes), len(nodes), False, 0.0)
    full = es.search(X, y, names, max_complexity, structure_bits=guided.structure_bits)[0]
    return GuidedResult(full, len(nodes), len(nodes), False, 1.0)
