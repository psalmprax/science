"""Data-driven shape inference for residual-guided search (RIME Repair step 1).

Why this exists. The Landauer staircase was only recovered after a human
declared a `floor` operator. That is a supplied primitive, not a discovery.
This module is the alternative: measure the data and *infer* which structural
family is worth proposing, with change-points estimated from the data rather
than declared.

Method
------
1. Sort by x.
2. Exact dynamic-programming segmentation of y into piecewise-constant
   segments (prefix-sum SSE, O(k·n²)).
3. Segment count k is chosen by a BIC-style criterion with an EXPLICIT
   penalty factor (caller argument, no module threshold literals per
   AGENTS.md §1.3) — k is data-estimated, never free.
4. Compare the piecewise-constant fit against a linear fit at matched
   complexity. A staircase beats the line by orders of magnitude; a straight
   line does not; a smooth saturating curve does not.

The output is a *prior over form families*, not an equation and not a
verdict. It tells the search what to propose; the search, the gates, and the
APEDP-X track still decide everything downstream. Nothing here promotes.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FAMILY_PIECEWISE_CONSTANT = "piecewise_constant"
FAMILY_SMOOTH = "smooth"
FAMILY_NO_STRUCTURE = "no_structure"

_VOCAB = frozenset({FAMILY_PIECEWISE_CONSTANT, FAMILY_SMOOTH, FAMILY_NO_STRUCTURE})


@dataclass(frozen=True)
class ShapePrior:
    """Inferred structural family + estimated change-points. A prior, not a claim."""

    family: str
    n_change_points: int
    change_points: tuple[float, ...]
    pc_r2: float                 # piecewise-constant fit R2 at BIC-selected k
    linear_r2: float             # straight-line fit R2 (matched-complexity control)
    improvement: float           # pc_r2 - linear_r2
    n_segments: int
    segments_to_reach_target: int
    parsimony_gap: float
    params_piecewise_constant: int
    params_piecewise_linear: int
    target_r2: float
    n_rows: int
    penalty_per_segment: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_change_points": self.n_change_points,
            "change_points": list(self.change_points),
            "pc_r2": self.pc_r2,
            "linear_r2": self.linear_r2,
            "improvement": self.improvement,
            "n_segments": self.n_segments,
            "segments_to_reach_target": self.segments_to_reach_target,
            "parsimony_gap": self.parsimony_gap,
            "params_piecewise_constant": self.params_piecewise_constant,
            "params_piecewise_linear": self.params_piecewise_linear,
            "target_r2": self.target_r2,
            "n_rows": self.n_rows,
            "penalty_per_segment": self.penalty_per_segment,
            "note": self.note,
        }


@dataclass(frozen=True)
class _FlatRuns:
    flat_fraction: float
    mean_run_length: float
    n_runs: int


def _flat_run_stats(y: np.ndarray, tol: float) -> _FlatRuns:
    """Fraction of steps that are flat, and mean maximal flat-run length.

    A "flat" step is |dy| <= tol * std(y). The MEAN RUN LENGTH is what
    separates structure from noise: a staircase has few very long runs,
    iid noise has many very short ones.
    """
    if y.size < 2:
        return _FlatRuns(0.0, 0.0, 0)
    dy = np.abs(np.diff(y))
    # Normalize by the JUMP SCALE, not std(y).
    #
    # Measured failure of std(y): for a steep line std(y) is large, so
    # tol*std exceeds the typical |dy| and every step reads "flat" (94% flat,
    # mean run 16.6 on a straight line). The 90th percentile of |dy| is the
    # natural jump scale: 1.0 for a staircase (most steps are plateau, a few
    # are jumps), ~0.07 for a line, ~1.4 for noise.
    scale = float(np.quantile(dy, 0.9))
    if scale <= 0.0:
        return _FlatRuns(1.0, float(y.size - 1), 1)
    is_flat = dy <= tol * scale
    flat_fraction = float(np.mean(is_flat))
    runs: list[int] = []
    cur = 0
    for f in is_flat:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    mean_len = float(np.mean([r + 1 for r in runs])) if runs else 0.0
    return _FlatRuns(flat_fraction, mean_len, len(runs))


def _prefix_sums(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([[0.0], np.cumsum(y)])
    s2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    return s, s2


def _seg_sse(s: np.ndarray, s2: np.ndarray, i: int, j: int) -> float:
    """SSE of y[i:j] about its own mean."""
    m = j - i
    if m <= 0:
        return 0.0
    tot = s[j] - s[i]
    tot2 = s2[j] - s2[i]
    return float(float(float(max(tot2 - tot * tot / m, 0.0))))


def _segments_to_reach(cost, n, tss, target, cap):
    """Smallest k with 1 - SSE(k)/tss >= target, or None if unreachable.

    This is the parsimony statistic: a staircase is cheap to describe in
    constant pieces, a line is not. It is computed from the DP cost curve, so
    it costs nothing extra.
    """
    best = None
    for k in range(1, cap + 1):
        sse = cost[k][n]
        if not np.isfinite(sse):
            continue
        if 1.0 - sse / tss >= target:
            best = k
            break
    return best


def _within_segment_sse(x: np.ndarray, y: np.ndarray, cuts: list[int]) -> tuple[float, float]:
    """(SSE for constant-per-segment, SSE for linear-per-segment) on the same cuts.

    This pair is the fundamental discriminator between a staircase and a curve.
    A staircase's within-segment residual is noise: adding a slope inside each
    segment buys nothing, so ss_const ~ ss_lin. A line or saturating curve's
    within-segment residual is DRAF: a slope buys nearly all of it, so
    ss_lin << ss_const.
    """
    bounds = [0, *list(cuts), len(x)]
    sc = sl = 0.0
    for a, b in itertools.pairwise(bounds):
        if b - a < 2:
            continue
        xs, ys = x[a:b], y[a:b]
        sc += float(np.sum((ys - ys.mean()) ** 2))
        A = np.column_stack([np.ones(b - a), xs])
        w, *_ = np.linalg.lstsq(A, ys, rcond=None)
        sl += float(np.sum((ys - A @ w) ** 2))
    return sc, sl


def _reach_k(cost, n, tss, target_r2, cap):
    """Smallest k whose fit reaches target R2, or None if unreachable within cap.

    This is the parsimony statistic the family decision rests on: how many
    SEGMENTS a family needs to reach a common fit quality. It is deliberately
    not a per-family BIC — separately selecting k per family was measured to
    degenerate, because a piecewise-constant model with enough segments drives
    SSE to exactly 0 and BIC's log-SSE term rewards that without bound (a clean
    straight line was assigned 32 constant segments).
    """
    for k in range(1, cap + 1):
        v = cost[k][n]
        if not np.isfinite(v):
            continue
        if 1.0 - v / tss >= target_r2:
            return k
    return None


def _dp_segment_linear(x, y, max_k, min_len):
    """Exact piecewise-LINEAR DP (prefix-sum normal equations, O(1) per span).

    Answers: how many segments does the data need if each may carry a slope,
    versus if each must be flat? A staircase needs the same number either way
    (its segments are already exactly flat, so the fitted slope is zero). A
    curve needs materially FEWER sloped segments than flat ones.
    """
    n = len(y)
    S1 = np.concatenate([[0.0], np.cumsum(np.ones(n))])
    Sx = np.concatenate([[0.0], np.cumsum(x)])
    Sy = np.concatenate([[0.0], np.cumsum(y)])
    Sxx = np.concatenate([[0.0], np.cumsum(x * x)])
    Sxy = np.concatenate([[0.0], np.cumsum(x * y)])
    Syy = np.concatenate([[0.0], np.cumsum(y * y)])

    def sse(j, i):
        m = S1[i] - S1[j]
        if m <= 0:
            return 0.0
        sx, sy = Sx[i] - Sx[j], Sy[i] - Sy[j]
        sxx, sxy = Sxx[i] - Sxx[j], Sxy[i] - Sxy[j]
        syy = Syy[i] - Syy[j]
        denom = sxx - sx * sx / m
        num = sxy - sx * sy / m
        if denom <= 1e-300:
            return max(syy - sy * sy / m, 0.0)
        return max(syy - sy * sy / m - num * num / denom, 0.0)

    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(max_k + 1)]
    cost[0][0] = 0.0
    for k in range(1, max_k + 1):
        for i in range(1, n + 1):
            for j in range(0, i - min_len + 1):
                if j < (k - 1) * min_len or cost[k - 1][j] == inf:
                    continue
                c = cost[k - 1][j] + sse(j, i)
                if c < cost[k][i]:
                    cost[k][i] = c
    return cost


def _dp_segment(y: np.ndarray, max_k: int, min_len: int) -> tuple[list[list[float]], np.ndarray, dict[int, list[int]]]:
    """Exact piecewise-constant DP. Returns (sse[k], boundaries per k)."""
    n = len(y)
    s, s2 = _prefix_sums(y)
    inf = float("inf")
    # cost[k][i] = min SSE for y[0:i] with k segments; bnd[k][i] = last cut
    cost = [[inf] * (n + 1) for _ in range(max_k + 1)]
    bnd = [[-1] * (n + 1) for _ in range(max_k + 1)]
    cost[0][0] = 0.0
    for k in range(1, max_k + 1):
        for i in range(1, n + 1):
            # last segment is y[j:i], must respect min_len
            lo = 0 if k == 1 else max(0, i - (n - (k - 1) * min_len))
            for j in range(lo, i - min_len + 1):
                if j < (k - 1) * min_len:
                    continue
                prev = cost[k - 1][j]
                if prev == inf:
                    continue
                c = prev + _seg_sse(s, s2, j, i)
                if c < cost[k][i]:
                    cost[k][i] = c
                    bnd[k][i] = j
    boundaries: dict[int, list[int]] = {}
    for k in range(1, max_k + 1):
        if cost[k][n] == inf:
            continue
        cuts: list[int] = []
        i, kk = n, k
        while kk > 0 and i > 0:
            j = bnd[kk][i]
            if j < 0:
                break
            cuts.append(j)
            i, kk = j, kk - 1
        boundaries[k] = sorted(cuts)
    return cost, s2, boundaries


def infer_shape_prior(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    *,
    penalty_per_segment: float = 1.0,
    min_segment_len: int = 5,
    max_segments: int = 32,
    improvement_margin: float = 0.05,
    min_rows: int = 40,
    flat_tol: float = 0.05,
    min_flat_fraction: float = 0.5,
    min_run_length: float = 5.0,
    min_pc_r2: float = 0.05,
    sse_target: float = 0.01,
    min_parsimony_gap: float = 0.0,
) -> ShapePrior:
    """Infer which structural family the data supports, with estimated steps.

    ``penalty_per_segment`` and ``improvement_margin`` are explicit arguments;
    the module defines no thresholds of its own.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if n < min_rows or len(x) < min_rows:
        return ShapePrior(FAMILY_NO_STRUCTURE, 0, (), float("nan"), float("nan"),
                          float("nan"), 0, 0, float("nan"), -1, -1,
                          float("nan"), len(x), penalty_per_segment,
                          f"too few rows ({len(x)}) for a shape prior")

    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    tss = float(np.sum((ys - ys.mean()) ** 2))
    if tss <= 0.0:
        return ShapePrior(FAMILY_NO_STRUCTURE, 0, (), float("nan"), float("nan"),
                          float("nan"), 0, 0, float("nan"), -1, -1,
                          float("nan"), len(x), penalty_per_segment,
                          "target is constant: no structure to infer")

    # Linear control fit (matched complexity: one slope + one intercept).
    A = np.column_stack([np.ones_like(xs), xs])
    w, *_ = np.linalg.lstsq(A, ys, rcond=None)
    lin_sse = float(np.sum((ys - A @ w) ** 2))
    linear_r2 = 1.0 - lin_sse / tss

    # Piecewise-constant DP.
    cap = int(min(max_segments, max(1, len(xs) // max(min_segment_len, 1))))
    _prefix_sums(ys)[1]
    cost = _dp_segment(ys, cap, min_segment_len)[0]
    best_k, best_bic = 1, float("inf")
    for k in range(1, cap + 1):
        sse = cost[k][len(ys)]
        if not np.isfinite(sse):
            continue
        bic = len(ys) * np.log(max(sse, 1e-300)) + penalty_per_segment * k * np.log(len(ys))
        if bic < best_bic:
            best_bic, best_k = bic, k
    pc_sse = float(cost[best_k][len(ys)])
    pc_r2 = 1.0 - pc_sse / tss

    improvement = pc_r2 - linear_r2

    # Discriminator: flat-run structure, not "beats the line".
    #
    # Measured failure of the obvious criterion: piecewise-constant gains over
    # a line on ANY data given enough segments (noise +0.14, saturating curve
    # +0.63) but only +0.027 on the actual staircase, because a line already
    # fits a ramp of steps at R2 0.97. So that statistic is close to useless.
    # What identifies a staircase is that y is locally constant except at
    # jumps: a high fraction of flat steps, in LONG runs.
    # Scale-free, noise-adaptive target: (1 - sse_target) of the improvement any
    # segmentation can achieve. A FIXED target does not separate line from
    # staircase (measured: both reach 99% of variance, the line with ~9 segments
    # of sawtooth approximation). Reported as a diagnostic; the family decision
    # below uses the drift ratio, which is the decisive statistic.
    achievable = 1.0 - float(min(cost[k][len(ys)] for k in range(1, cap + 1)
                                if np.isfinite(cost[k][len(ys)]))) / tss
    target = max(achievable * (1.0 - sse_target), 1e-12)
    k_target = _segments_to_reach(cost, len(ys), tss, target, cap)

    _flat_run_stats(ys, flat_tol)

    # Recorded, measured failures of four earlier discriminators (see
    # CHANGELOG): (a) "piecewise-constant beats a line" flags noise and a
    # saturating curve as stepped while granting only +0.027 on the real
    # staircase -- a line already fits a ramp of steps at R2 0.973.
    # (b) flat-run fraction on the p90 |dy| scale collapses under noise
    # (0.975 -> 0.065) because >90% flat steps put p90 inside the plateau.
    # (c) within-segment drift ratio mislabels a COARSE segmentation of a CURVE
    # as plateaus (measured 0.70 on a noisy Pade curve, 0.98 on iid noise)
    # because a strongly curved segment looks plateau-like.
    # (d) separately-BIC-selected k per family degenerates: a piecewise-constant
    # model with enough segments drives SSE to exactly 0, and BIC's log-SSE term
    # rewards that unboundedly (a clean line picked 32 constant segments).
    #
    # Decisive statistic: PARAMETERS NEEDED TO REACH A COMMON FIT TARGET.
    # Ask of each family how many parameters it needs to explain a fixed
    # fraction of what the better family can explain. A staircase's segments are
    # already flat, so slopes cost parameters without buying fit quality
    # (measured 12 flat vs 18 sloped). A curve is the opposite: slopes reach the
    # same quality with far fewer parameters (line: 3 vs 56).
    cost_lin = _dp_segment_linear(xs, ys, cap, min_segment_len)
    best_c = min(cost[k][len(ys)] for k in range(1, cap + 1)
                 if np.isfinite(cost[k][len(ys)]))
    best_l = min(cost_lin[k][len(ys)] for k in range(1, cap + 1)
                 if np.isfinite(cost_lin[k][len(ys)]))
    achievable = max(1.0 - float(best_c) / tss, 1.0 - float(best_l) / tss)
    target = max(achievable * (1.0 - sse_target), 0.0)
    k_c = _reach_k(cost, len(ys), tss, target, cap)
    k_l = _reach_k(cost_lin, len(ys), tss, target, cap)
    p_c = 2 * k_c + 1 if k_c is not None else None
    p_l = 3 * k_l + 1 if k_l is not None else None
    gap = (p_l - p_c) / max(p_c, 1) if (p_c and p_l) else 0.0

    if pc_r2 < min_pc_r2:
        family = FAMILY_NO_STRUCTURE
        note = f"piecewise-constant R2 {pc_r2:.3f} below floor {min_pc_r2}: no structure"
    elif k_c is None:
        family = FAMILY_SMOOTH
        note = (f"flat segments cannot reach the target R2 {target:.4f} at all "
                f"(best constant {pc_r2:.4f}) -> smooth family")
    elif k_l is None:
        family = FAMILY_PIECEWISE_CONSTANT
        note = (f"flat segments reach R2 {target:.4f} with {p_c} parameters while "
                f"sloped segments cannot reach it within {cap} segments -> plateaus")
    elif gap >= min_parsimony_gap:
        family = FAMILY_PIECEWISE_CONSTANT
        note = (f"slopes cost more than they earn: reaching R2 {target:.4f} costs "
                f"{p_c} parameters flat but {p_l} sloped (gap {gap:.3f} >= "
                f"{min_parsimony_gap}) -> the segments are ALREADY flat, so a "
                f"slope buys nothing and the leftover is noise; pc R2 {pc_r2:.6f}")
    else:
        family = FAMILY_SMOOTH
        note = (f"slopes buy parsimony: reaching R2 {target:.4f} costs {p_c} "
                f"parameters flat but only {p_l} sloped (gap {gap:.3f} < "
                f"{min_parsimony_gap}) -> smooth family, not plateaus")

    cps: list[float] = []
    if family == FAMILY_PIECEWISE_CONSTANT:
        boundaries = _dp_segment(ys, cap, min_segment_len)[2]
        for j in boundaries.get(best_k, []):
            if 0 < j < len(xs):
                cps.append(float((xs[j - 1] + xs[j]) / 2.0))
    return ShapePrior(family, len(cps), tuple(cps), pc_r2, linear_r2, improvement,
                      best_k, -1 if k_target is None else k_target, gap,
                      -1 if p_c is None else p_c, -1 if p_l is None else p_l,
                      target, len(x), penalty_per_segment, note)


def shape_vocabulary() -> Sequence[str]:
    return tuple(sorted(_VOCAB))


# ---------------------------------------------------------------------------
# Step 2: step-family basis at DATA-ESTIMATED change-points.
#
# Measured motivation. The staircase is exactly
#     y = n/2 + 0.5 * sum_i sign(x - e_i)
# so it needs NO new operator: Sign, Add, Const and Mul are all already in
# the grammar (max error 0.0, R2 1.0). What the seed catalogue lacks is a
# constant placed INSIDE Sign at a data-chosen location -- seeds contain
# sign(x), never sign(x - e). The change-points, not the step function, are
# the missing primitive.
#
# This module therefore supplies thresholds from the data. It EMITS A
# CANDIDATE FORM. It never promotes, never scores, never inspects the sealed
# holdout (AGENTS.md 1.1/1.2): fitting here is OLS on the dev partition only
# and the caller decides whether the result is admissible.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepFamilySeeds:
    """A candidate step-family form with fitted coefficients. A proposal, not a verdict."""

    thresholds: tuple[float, ...]
    basis_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    r2: float
    residual_reduction: float      # 1 - SSE_step / SSE_mean
    identifiable: bool
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": list(self.thresholds),
            "basis_names": list(self.basis_names),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "r2": self.r2,
            "residual_reduction": self.residual_reduction,
            "identifiable": self.identifiable,
            "n_rows": self.n_rows,
        }


def step_family_basis(
    x: np.ndarray | Sequence[float], thresholds: Sequence[float]
) -> tuple[list[str], np.ndarray]:
    """Sign basis at the given thresholds: [1, sign(x-t1), sign(x-t2), ...].

    This is the exact algebraic form of a rising staircase: a level ramp built
    from signs. The returned matrix is for OLS; the names say what each column
    is so a candidate form can be written symbolically.
    """
    xs = np.asarray(x, dtype=float).ravel()
    cols = [np.ones_like(xs)]
    names = ["1"]
    for t in thresholds:
        # Edge estimates are midpoints between adjacent rows, so an exact
        # comparison is correct and deterministic; no half-bin nudge.
        cols.append(np.where(xs >= float(t), 1.0, -1.0))
        names.append(f"sign(x - {float(t):.6g})")
    return names, np.column_stack(cols)


def fit_step_family(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    thresholds: Sequence[float],
    *,
    min_rows: int = 40,
    residual_reduction_floor: float = 0.05,
    condition_floor: float = 1e6,
) -> StepFamilySeeds:
    """Fit the sign-step basis at data-estimated thresholds by OLS.

    ``residual_reduction_floor`` and ``condition_floor`` are explicit caller
    arguments (AGENTS.md 1.3). Fitting is on the rows the caller hands over;
    holdout separation is the caller's responsibility, not this function's.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    n = min(xs.size, ys.size)
    xs, ys = xs[:n], ys[:n]
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]

    names, M = step_family_basis(xs, thresholds)
    sst = float(np.sum((ys - ys.mean()) ** 2))
    if xs.size < min_rows or sst <= 0.0 or M.shape[0] < M.shape[1]:
        return StepFamilySeeds(tuple(float(t) for t in thresholds), tuple(names),
                               (), float("nan"), float("nan"), float("nan"), False,
                               int(xs.size))

    beta, *_ = np.linalg.lstsq(M, ys, rcond=None)
    fit = M @ beta
    sse = float(np.sum((ys - fit) ** 2))
    r2 = 1.0 - sse / sst
    rr = 1.0 - sse / sst
    # Identifiability: the basis columns must be well conditioned and the
    # step coefficients must not vanish (a staircase with zero steps is a line).
    cond = float(np.linalg.cond(M))
    identifiable = bool(
        np.isfinite(cond) and cond < condition_floor and rr >= residual_reduction_floor
        and float(np.max(np.abs(beta[1:]))) > 0.0
    )
    return StepFamilySeeds(tuple(float(t) for t in thresholds), tuple(names),
                           tuple(float(b) for b in beta), float(beta[0]), r2, rr,
                           identifiable, int(xs.size))


def staircase_prior(
    x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float], **kwargs: Any
) -> tuple[ShapePrior, StepFamilySeeds | None]:
    """Convenience: infer the prior, and if stepped, fit the step family.

    Returns (prior, seeds_or_None). ``None`` when the prior does not support a
    step family -- the caller must not invent one.
    """
    prior = infer_shape_prior(x, y, **kwargs)
    if prior.family != FAMILY_PIECEWISE_CONSTANT or not prior.change_points:
        return prior, None
    seeds = fit_step_family(x, y, prior.change_points)
    return prior, seeds
