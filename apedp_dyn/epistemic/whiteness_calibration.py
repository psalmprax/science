"""Null-calibrated residual-whiteness flag (separate module, adopter opt-in).

Why this is a separate module: ``adjudicate.py`` is a hash-pinned frozen
record (AGENTS.md §1.3 / frozen_spec). Editing it is blocked by the
project's own freeze-drift guard -- which is correct. So the calibration
lives here and callers opt in.

What it fixes
-------------
The frozen Guard 2 compares ``|rho1|`` of the holdout residuals against a
single magic constant (0.5). That is two constants pretending to be one:

  * it UNDER-refuses on strongly autocorrelated targets. A faithful model of
    an AR(2) target leaves AR(2)-shaped residuals whose |rho1| can exceed
    0.9 and so be read as "structure remains" when in fact the model is
    faithful and the target was simply persistent.
  * it OVER-refuses on nearly-iid targets. A good model of white noise
    leaves residuals whose |rho1| peaks near 1/sqrt(n); at n=200 that is
    0.07, and a flag of 0.5 then admits almost any AR(1)-ish misfit.

The calibrated flag asks the question the constant was standing in for:
"how much lag-1 autocorrelation would the residuals show if the model
explained none of the target's serial dependence?" It answers this by
demeaning the target in contiguous blocks. Block demeaning preserves the
target's own autocorrelation (so the null inherits the right dependence
shape) while removing level structure, so an intercept-only model scores
~0 against it and cannot launder AR residuals through an offset.

Usage
-----
    flag, diagnostics = calibrate_flag(holdout_target)
    ok, why = check_whiteness(residuals, flag, diagnostics)

``check_whiteness`` mirrors Guard 2's structure (raw |rho1| first, then a
standardized-innovations retry for heteroskedastic series) so a caller can
substitute it without re-implementing the check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _rho1(arr: np.ndarray) -> float:
    c = arr - arr.mean()
    denom = float(np.dot(c, c))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(c[:-1], c[1:]) / denom)


@dataclass(frozen=True)
class FlagCalibration:
    """Audit record for one resolved flag."""

    flag: float
    target_rho1: float
    null_median: float
    null_p95: float
    null_p99: float
    quantile: float
    block_size: int
    n_draws: int
    n_points: int
    method: str = "null_calibrated_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "target_rho1": self.target_rho1,
            "null_median": self.null_median,
            "null_p95": self.null_p95,
            "null_p99": self.null_p99,
            "quantile": self.quantile,
            "block_size": self.block_size,
            "n_draws": self.n_draws,
            "n_points": self.n_points,
            "method": self.method,
        }


def calibrate_flag(
    target: Any,
    *,
    quantile: float = 0.95,
    block_size: int = 20,
    n_draws: int = 400,
    seed: int = 0,
    min_points: int = 32,
) -> FlagCalibration | None:
    """Calibrate |rho1| from the target's own serial dependence.

    Returns None when the target is unusable (too short, non-finite, or the
    null distribution degenerates). A None result means "cannot judge
    structure", which a caller must treat as a refusal, never as a pass.
    """
    y = np.asarray(target, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < max(min_points, block_size + 4):
        return None

    rng = np.random.default_rng(seed)
    n = y.size
    block_size = int(min(block_size, max(4, n // 4)))
    n_blocks = int(np.ceil(n / block_size))
    rhos = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        starts = rng.integers(0, max(1, n - block_size), size=n_blocks)
        pieces: list[np.ndarray] = []
        total = 0
        for s in starts:
            blk = y[s:s + block_size]
            if blk.size == 0:
                continue
            pieces.append(blk - blk.mean())
            total += blk.size
            if total >= n:
                break
        if not pieces:
            rhos[d] = 0.0
            continue
        z = np.concatenate(pieces)[:n]
        if z.size < 8:
            rhos[d] = 0.0
            continue
        r = _rho1(z)
        rhos[d] = r if np.isfinite(r) else 0.0

    a = np.abs(rhos)
    flag = float(np.quantile(a, quantile))
    # Upper bound at the iid scale: on a white target the null concentrates
    # near 0, and a small n would otherwise admit AR(1)-shaped residuals.
    cap = 2.0 / np.sqrt(n)
    flag = min(flag, max(cap, flag))
    if not np.isfinite(flag):
        return None
    return FlagCalibration(
        flag=flag,
        target_rho1=float(_rho1(y)),
        null_median=float(np.median(a)),
        null_p95=float(np.quantile(a, 0.95)),
        null_p99=float(np.quantile(a, 0.99)),
        quantile=float(quantile),
        block_size=block_size,
        n_draws=int(n_draws),
        n_points=int(n),
    )


def check_whiteness(
    resid: Any,
    flag: float,
    calibration: FlagCalibration | None = None,
) -> tuple[bool, str]:
    """Guard-2-equivalent check against a resolved flag.

    Mirrors the frozen Guard 2 structure: raw |rho1| first, then a
    standardized-innovations retry for heteroskedastic series, so an adopter
    can substitute this for the constant-flag check without changing the
    surrounding logic. Refuses (does not pass) when the flag is NaN.
    """
    if resid is None:
        return False, "no residual series; whiteness unassessable"
    arr = np.asarray(resid, dtype=float).ravel()
    if arr.size < 4:
        return False, f"residual series too short ({arr.size} points) to test"
    if not np.all(np.isfinite(arr)):
        return False, "residual series contains non-finite values"
    if not np.isfinite(flag):
        return False, "whiteness flag unresolved; refusing rather than admitting"
    centred = arr - arr.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        return False, ("residual series is constant: |rho1| undefined, "
                       "whiteness unassessable")
    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
    ok = bool(abs(rho1) < flag)
    if calibration is not None:
        detail = (f"|rho1|={abs(rho1):.3f} vs calibrated flag {flag:.3f} "
                  f"(target rho1={calibration.target_rho1:.3f})")
    else:
        detail = f"|rho1|={abs(rho1):.3f} vs flag {flag:g}"
    if ok:
        return ok, detail

    if arr.size >= 8:
        alpha = 0.8
        curr_var = float(np.var(centred))
        if curr_var > 1e-12:
            var_est = np.empty_like(centred)
            for t in range(len(centred)):
                curr_var = alpha * curr_var + (1.0 - alpha) * (centred[t] ** 2)
                var_est[t] = max(curr_var, 1e-12)
            z = centred / np.sqrt(var_est)
            zc = z - z.mean()
            zd = float(np.dot(zc, zc))
            if zd > 0.0:
                rho1_z = float(np.dot(zc[:-1], zc[1:]) / zd)
                if abs(rho1_z) < flag:
                    return True, detail + " (standardized innovations)"

    return False, detail


def screen_structured_null_worlds(
    engine: Any,
    n_worlds: int = 40,
    n_points: int = 600,
    train_frac: float = 0.6,
    flag_quantile: float = 0.95,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Two-layer false-promotion screen on structured null worlds.

    Runs ``engine`` (any proposer exposing ``fit_and_propose(frame)``) on
    structured targets that have no dependence on the available feature, and
    reports BOTH layers separately:

        proposer-level: emissions (the search leaked)
        system-level: emissions surviving ``check_whiteness``

    The layered pair is the honest unit of report: a system whose safety
    rests entirely on the guard must say so.

    Null families (fix seed): AR(1), oscillation, polynomial trend,
    heteroskedastic bursts. The feature is iid and irrelevant in all of them.
    """
    proposer_fp = 0
    system_fp = 0
    rho_bad: list[float] = []
    for w in range(n_worlds):
        x, y = _structured_null(n_points, w)
        ntr = int(n_points * train_frac)
        frame = _as_frame(x[:ntr], y[:ntr])
        te = _as_frame(x[ntr:], y[ntr:])
        props = engine(frame) if callable(engine) else engine.fit_and_propose(frame)
        for q in props or []:
            proposer_fp += 1
            yp = np.asarray(q.compute_score(te), dtype=float)
            yt = te["target"].to_numpy()
            m = np.isfinite(yt) & np.isfinite(yp)
            resid = (yt - yp)[m]
            calib = calibrate_flag(y[:ntr], quantile=flag_quantile)
            flag = calib.flag if calib is not None else float("nan")
            ok, _why = check_whiteness(resid, flag, calib)
            if ok:
                system_fp += 1
            else:
                r = _rho1(resid)
                if np.isfinite(r):
                    rho_bad.append(abs(r))

    return {
        "proposer_promotions": proposer_fp,
        "system_promotions": system_fp,
        "null_worlds": n_worlds,
        "flag_quantile": flag_quantile,
        "median_blocked_rho1": float(np.median(rho_bad)) if rho_bad else float("nan"),
        "blocked_rho1_max": float(np.max(rho_bad)) if rho_bad else float("nan"),
    }


# --- structured null families (deterministic) -------------------------------
def _structured_null(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.RandomState(seed)
    x = r.randn(n)
    kind = seed % 4
    if kind == 0:
        e = np.zeros(n)
        for t_lag in range(1, n):
            e[t_lag] = 0.9 * e[t_lag - 1] + r.randn()
        y = e
    elif kind == 1:
        t = np.arange(n) / n
        y = np.sin(2 * np.pi * 7 * t) + 0.1 * r.randn(n)
    elif kind == 2:
        t = np.arange(n) / n
        y = 3 * t + 2 * t ** 2 - t ** 3 + 0.05 * r.randn(n)
    else:
        s = np.where(np.arange(n) > n // 2, 0.1, 5.0)
        y = s * r.randn(n)
    return x, y


def _as_frame(x: np.ndarray, y: np.ndarray):
    import pandas as pd

    return pd.DataFrame({"x": x, "target": y})
