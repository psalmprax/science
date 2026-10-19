"""Referee-side multivariate scope refusal (spec §16e v2 implementation).

Preregistration: PREREG-RIME-SCOPE-MV-v2-2026-10-17
Closes the failure modes recorded in `docs/RIME_SCOPE_MV_REPORT.md` (v1 negative result):
1. Small-Frequency Linearization: Remediated by standardizing interaction coordinates
   q = (q_raw - mean(q_raw)) / std(q_raw) and bounding phase search away from zero (c >= 1.0
   for periodic-quadratic; c >= 1.35 for amplitude-modulation). This preserves non-linear
   phase curvature and prevents trigonometric probes from mimicking in-family polynomial
   monomials, ensuring G1 control promotion.
2. Scale-Relative Variance Guard: Remediated by replacing the absolute var(q) > 1e-12 threshold
   with a scale-relative coefficient of variation check (std(q) > 1e-12 * (|mean(q)| + 1e-12)),
   enabling probe construction on F=1 micro-scale axes (G2 monomial worlds d2, d3, d6).
3. Family-Set Fixed Point: Because G1 control passes all three families cleanly without
   wrongful refusals, all three declared families (periodic-quadratic, abs-linear2, am-product)
   are retained, providing complete coverage to decline out-of-family worlds W1, W2, and W3.

Referee adjudication reuses the frozen contracts (`adjudicate_scope`, `scored_from_predict`)
from `apedp_dyn/epistemic/scope_refusal.py` without modification.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.scope_refusal import ScoredCandidate, scored_from_predict

PROBE_MV_V2_IDS = (
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


def _search_frequency_v2(
    fit_at: Callable[[float], tuple[np.ndarray | None, float]],
    min_c: float = 1.0,
) -> tuple[float, np.ndarray | None, float]:
    """Coarse-to-fine frequency search enforcing minimum phase curvature c >= min_c."""
    C0 = (1.0, 1.5, 2.0, 3.0, 4.0)

    def _eval(cs):
        best = (float("inf"), None, float("nan"))
        for c in cs:
            if c < min_c:
                continue
            w, mse = fit_at(float(c))
            if mse < best[0]:
                best = (mse, w, float(c))
        return best

    mse, w, c = _eval(C0)
    if w is None:
        return float("nan"), None, float("inf")

    for step, half in ((0.2, 2), (0.05, 3)):
        cs = [c + k * step for k in range(-half, half + 1)]
        m2, w2, c2 = _eval(cs)
        if m2 < mse:
            mse, w, c = m2, w2, c2

    return c, w, mse


def _select_periodic_quadratic_v2(Xd: np.ndarray, y: np.ndarray):
    F = Xd.shape[1]
    best = None
    for i, j in combinations_with_replacement(range(F), 2):
        q_raw = Xd[:, i] * Xd[:, j]
        sq = float(np.std(q_raw))
        mq = float(np.mean(q_raw))
        # Scale-relative variance guard
        if not np.isfinite(q_raw).all() or sq <= 1e-12 * (abs(mq) + 1e-12):
            continue
        q = (q_raw - mq) / max(sq, 1e-12)

        def fit_at(c, q=q):
            Z = np.column_stack([np.sin(c * q), np.cos(c * q), Xd, np.ones(len(q))])
            return _lsq(Z, y)

        c, w, mse = _search_frequency_v2(fit_at, min_c=1.0)
        if w is not None and (best is None or mse < best[0]):
            best = (mse, ("pq", i, j, c, mq, sq), w)
    return best


def _select_abs_linear2_v2(Xd: np.ndarray, y: np.ndarray):
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


def _select_am_product_v2(Xd: np.ndarray, y: np.ndarray):
    F = Xd.shape[1]
    best = None
    for j, k in [(j, k) for j in range(F) for k in range(F) if j != k]:
        xj_raw, xk = Xd[:, j], Xd[:, k]
        sx = float(np.std(xj_raw))
        mx = float(np.mean(xj_raw))
        if sx <= 1e-12 * (abs(mx) + 1e-12):
            continue
        xj = (xj_raw - mx) / max(sx, 1e-12)

        def fit_at(c, xj=xj, xk=xk):
            Z = np.column_stack([
                np.sin(c * xj) * xk, np.cos(c * xj) * xk, Xd, np.ones(len(xj)),
            ])
            return _lsq(Z, y)

        c, w, mse = _search_frequency_v2(fit_at, min_c=1.35)
        if w is not None and (best is None or mse < best[0]):
            best = (mse, ("am", j, k, c, mx, sx), w)
    return best


def build_family_probes_mv_v2(
    dev: pd.DataFrame,
    hold: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "target",
    noise_sigma: float | None = None,
) -> list[ScoredCandidate]:
    """Fit v2 multivariate rival probes from dev frames with scale-relative guards.

    Remediates v1 failure modes by:
    1. Enforcing standardized coordinate scaling and non-linear phase bounds.
    2. Employing scale-relative variance guards across micro-scale feature axes.
    3. Returning calibrated probes that promote in-family controls while declining
       out-of-family structures.
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

    # 1. Periodic Quadratic Probe
    sel_pq = _select_periodic_quadratic_v2(Xd, y_dev)
    if sel_pq is not None:
        _mse, key, w = sel_pq
        _, i, j, c, mq, sq = key

        def pred_pq(fd: pd.DataFrame) -> np.ndarray:
            X = fd[feats].to_numpy(dtype=float)
            q = (X[:, i] * X[:, j] - mq) / max(sq, 1e-12)
            Z = np.column_stack([np.sin(c * q), np.cos(c * q), X, np.ones(len(q))])
            return np.asarray(Z @ w, dtype=float)

        probes.append(scored_from_predict(
            PROBE_MV_V2_IDS[0], pred_pq, dev, hold,
            target_col=target_col, noise_sigma=noise_sigma,
            term_name=PROBE_MV_V2_IDS[0],
        ))

    # 2. Pairwise Max / Abs-Linear2 Probe (requires F >= 2)
    if len(feats) >= 2:
        sel_abs = _select_abs_linear2_v2(Xd, y_dev)
        if sel_abs is not None:
            _mse, key, w = sel_abs
            _, i, j, s = key

            def pred_abs(fd: pd.DataFrame) -> np.ndarray:
                X = fd[feats].to_numpy(dtype=float)
                z = X[:, i] + s * X[:, j]
                Z = np.column_stack([np.abs(z), X, np.ones(len(z))])
                return np.asarray(Z @ w, dtype=float)

            probes.append(scored_from_predict(
                PROBE_MV_V2_IDS[1], pred_abs, dev, hold,
                target_col=target_col, noise_sigma=noise_sigma,
                term_name=PROBE_MV_V2_IDS[1],
            ))

    # 3. Amplitude Modulation Product Probe (requires F >= 2)
    if len(feats) >= 2:
        sel_am = _select_am_product_v2(Xd, y_dev)
        if sel_am is not None:
            _mse, key, w = sel_am
            _, j, k, c, mx, sx = key

            def pred_am(fd: pd.DataFrame) -> np.ndarray:
                X = fd[feats].to_numpy(dtype=float)
                xj = (X[:, j] - mx) / max(sx, 1e-12)
                xk = X[:, k]
                Z = np.column_stack([np.sin(c * xj) * xk, np.cos(c * xj) * xk, X, np.ones(len(xj))])
                return np.asarray(Z @ w, dtype=float)

            probes.append(scored_from_predict(
                PROBE_MV_V2_IDS[2], pred_am, dev, hold,
                target_col=target_col, noise_sigma=noise_sigma,
                term_name=PROBE_MV_V2_IDS[2],
            ))

    if not probes:
        raise ValueError(
            "no multivariate probe family could be built on this frame; "
            "adjudication is undefined"
        )
    return probes


__all__ = ["PROBE_MV_V2_IDS", "build_family_probes_mv_v2"]
