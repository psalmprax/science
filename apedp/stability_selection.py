"""Stability selection (Meinshausen & Bühlmann style) for sparse selection.

Takes repeated fits with different random subsamples/seeds and keeps
features/terms with selection frequency >= pi. Coefficients averaged
over selections.
"""
from __future__ import annotations

from typing import Any, Callable


def stability_select(
    fit_fn: Callable[..., dict[str, float]],
    X=None,
    y=None,
    k: int = 20,
    pi: float = 0.9,
    seed_fn: Callable[[int], None] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    for i in range(k):
        if seed_fn is not None:
            seed_fn(i)
        try:
            if X is not None and y is not None:
                found = fit_fn(X, y, **kwargs)
            else:
                found = fit_fn(**kwargs)
        except Exception:
            continue
        if not isinstance(found, dict):
            continue
        for name, c in found.items():
            try:
                c_f = float(c)
            except Exception:
                continue
            counts[name] = counts.get(name, 0) + 1
            sums[name] = sums.get(name, 0.0) + c_f
    keep = {}
    if k == 0:
        return keep
    for n in counts:
        if counts[n] / k >= pi:
            keep[n] = sums[n] / counts[n]
    return keep
