"""Referee-side multivariate scope refusal (spec §16e wiring).

Prereg: `docs/RIME_PREREGISTRATION_SCOPE_REFUSAL_MV.md`
(PREREG-RIME-SCOPE-MV-2026-10-17). Probe CONSTRUCTION only: adjudication reuses
the frozen §16d contracts (`adjudicate_scope`, `scored_from_predict`) unchanged.
The referee sees only (dev, holdout) frames and the declared noise scale — no
world labels, planted truth or grammar declarations enter.

Three declared rival families, each fitted by OLS on dev with the full linear
nuisance span [x_1..x_F, 1] so additive nuisance terms can be absorbed while
the interaction structure remains non-monomial:

1. probe_mv_periodic_quadratic — [sin(c·q), cos(c·q), span] over pairwise
   products/squares q, c from C0 plus two deterministic refinement passes.
2. probe_mv_abs_linear2 — [|w·x|, span] over w = e_i ± e_j (i<j; ≥2 active
   components; exact rival for pairwise max via max(x_i,x_j) = (x_i+x_j+|x_i−x_j|)/2).
3. probe_mv_am_product — [sin(c·x_j)·x_k, cos(c·x_j)·x_k, span] over ordered
   pairs j≠k (amplitude-modulated family), same frequency search as family 1.

Families with no valid member on the given frame are skipped; if no probe can
be built at all, the builder raises (adjudication would be undefined).
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.scope_refusal import ScoredCandidate, scored_from_predict

C0: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
PROBE_MV_IDS = (
    "probe_mv_periodic_quadratic",
    "probe_mv_abs_linear2",
    "probe_mv_am_product",
)


def _lsq(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, float]:
    ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
    if int(ok.sum()) < Z.shape[1] + 1:
        return None, float("inf")
    w, *_ = np.linalg.lstsq(Z[ok], y[ok], rcond=None)
    mse = float(np.mean((Z[ok] @ w - y[ok]) ** 2))
    return (w if np.isfinite(mse) else None), (mse if np.isfinite(mse) else float("inf"))


def _search_frequency(fit_at: Callable[[float], tuple[np.ndarray | None, float]]
                      ) -> tuple[float, np.ndarray | None, float]:
    """Deterministic coarse-to-fine frequency search (prereg: C0, then two passes)."""
    def _eval(cs) -> tuple[float, np.ndarray | None, float]:
        best = (float("inf"), None, float("nan"))
        for c in cs:
            if c <= 0.0:
                continue
            w, mse = fit_at(float(c))
            if mse < best[0]:
                best = (mse, w, float(c))
        return best

    mse, w, c = _eval(C0)
    if w is None:
        return float("nan"), None, float("inf")
    for step, half in ((0.1, 3), (0.02, 5)):
        cs = [c + k * step for k in range(-half, half + 1)]
        m2, w2, c2 = _eval(cs)
        if m2 < mse:
            mse, w, c = m2, w2, c2
    return c, w, mse


def _select_periodic_quadratic(Xd: np.ndarray, y: np.ndarray):
    F = Xd.shape[1]
    best = None
    for i, j in combinations_with_replacement(range(F), 2):
        q = Xd[:, i] * Xd[:, j]
        if not np.isfinite(q).all() or float(np.var(q)) <= 1e-12:
            continue

        def fit_at(c, q=q):
            Z = np.column_stack([np.sin(c * q), np.cos(c * q), Xd, np.ones(len(q))])
            return _lsq(Z, y)

        c, w, mse = _search_frequency(fit_at)
        if w is not None and (best is None or mse < best[0]):
            best = (mse, ("pq", i, j, c), w)
    return best


def _select_abs_linear2(Xd: np.ndarray, y: np.ndarray):
    F = Xd.shape[1]
    best = None
    for i, j in combinations(range(F), 2):
        for s in (1.0, -1.0):
            z = Xd[:, i] + s * Xd[:, j]
            Z = np.column_stack([np.abs(z), Xd, np.ones(len(z))])
            w, mse = _lsq(Z, y)
            if w is not None and (best is None or mse < best[0]):
                best = (mse, ("abs", i, j, s), w)
    return best


def _select_am_product(Xd: np.ndarray, y: np.ndarray):
    F = Xd.shape[1]
    best = None
    for j, k in [(j, k) for j in range(F) for k in range(F) if j != k]:
        xj, xk = Xd[:, j], Xd[:, k]

        def fit_at(c, xj=xj, xk=xk):
            Z = np.column_stack([
                np.sin(c * xj) * xk, np.cos(c * xj) * xk, Xd, np.ones(len(xj)),
            ])
            return _lsq(Z, y)

        c, w, mse = _search_frequency(fit_at)
        if w is not None and (best is None or mse < best[0]):
            best = (mse, ("am", j, k, c), w)
    return best


def _predict_factory(kind: str, key: tuple, w: np.ndarray, feats: list[str],
                     target_col: str) -> Callable[[pd.DataFrame], np.ndarray]:
    def predict(fd: pd.DataFrame) -> np.ndarray:
        X = fd[feats].to_numpy(dtype=float)
        n = len(X)
        if kind == "pq":
            _, i, j, c = key
            q = X[:, i] * X[:, j]
            Z = np.column_stack([np.sin(c * q), np.cos(c * q), X, np.ones(n)])
        elif kind == "abs":
            _, i, j, s = key
            z = X[:, i] + s * X[:, j]
            Z = np.column_stack([np.abs(z), X, np.ones(n)])
        else:
            _, j, k, c = key
            xj, xk = X[:, j], X[:, k]
            Z = np.column_stack([np.sin(c * xj) * xk, np.cos(c * xj) * xk, X, np.ones(n)])
        return np.asarray(Z @ w, dtype=float)
    return predict


def build_family_probes_mv(dev: pd.DataFrame, hold: pd.DataFrame,
                           feature_cols: list[str] | None = None,
                           target_col: str = "target",
                           noise_sigma: float | None = None) -> list[ScoredCandidate]:
    """Fit the preregistered multivariate rival probes from the frames alone.

    Returns up to three ScoredCandidates (per prereg order: periodic-quadratic,
    abs-linear2, am-product), each fitted on dev and recorded on holdout.
    """
    feats = list(feature_cols) if feature_cols else [c for c in dev.columns if c != target_col]
    if not feats:
        raise ValueError("no feature columns: multivariate probes cannot be built")
    missing = [c for c in feats if c not in dev.columns or c not in hold.columns]
    if missing:
        raise ValueError(f"feature columns absent from frames: {missing}")
    y_dev = dev[target_col].to_numpy(dtype=float)
    Xd = dev[feats].to_numpy(dtype=float)
    if not np.isfinite(Xd).all() or not np.isfinite(y_dev).all():
        raise ValueError("non-finite entries in dev frame: probes cannot be fitted")

    probes: list[ScoredCandidate] = []
    builders = (
        (PROBE_MV_IDS[0], _select_periodic_quadratic),
        (PROBE_MV_IDS[1], _select_abs_linear2),
        (PROBE_MV_IDS[2], _select_am_product),
    )
    for pid, selector in builders:
        sel = selector(Xd, y_dev)
        if sel is None:
            continue
        _mse, key, w = sel
        predict = _predict_factory(key[0], key, w, feats, target_col)
        probes.append(scored_from_predict(
            pid, predict, dev, hold, target_col=target_col,
            noise_sigma=noise_sigma, term_name=pid))
    if not probes:
        raise ValueError(
            "no multivariate probe family could be built on this frame; "
            "adjudication is undefined")
    return probes


__all__ = ["C0", "PROBE_MV_IDS", "build_family_probes_mv"]
