"""Six further published regularities of asset returns, audited as the first five were.

Status: experimental

``apedp_dyn.market.law_audit`` measures five published laws on a panel their
authors never saw, and is frozen with its preregistration. This module adds the
regularities of Cont's list that it left out and can be given a number on daily
data, and two published claims about what predicts returns. It is built around
the frozen module and changes nothing in it: the same cells, the same block
resampling, the same verdict rule.

==================  =============================================================  =============
Claim               Number measured here                                           Published set
==================  =============================================================  =============
``asymmetry``       share of an asset's largest moves that are losses, minus 1/2   above 0
``gaussianity``     share of large daily moves minus share of large weekly moves   above 0
``conditional``     share of large moves before, minus after, dividing each        above 0
                    return by a volatility estimate from earlier days
``still_heavy``     share of large moves after that division, minus what           above 0
                    independent normal returns in the same cells leave after it
``cascade``         how well a week's net move predicts the next week's daily      above 0
                    moves, minus how well its daily moves predict the next
                    week's net move
``momentum``        rank correlation, across assets, of the return of the last     above 0
                    one to four weeks with the return of the next week
``flow``            rank correlation of a day's order-flow imbalance with the      above 0
                    next day's return
==================  =============================================================  =============

The sources and the wording of each claim are in
``preregistration/PREREG-LAW-AUDIT-FINANCE-MORE-2026-11.md``.

*A large move* is one whose size exceeds ``TAIL_MULTIPLE`` times the median
size of the same asset's moves. For normal returns that is 0.7 % of the moves;
for returns with a tail index of three, 4.4 %. It is a share, so it has no
moment to diverge and resamples as a sum. The median is taken within the cells
that share a position in the week (or the month): a daily and a weekly share
are then both computed against medians of the same number of values, and the
error of those medians inflates both alike.

*The medians under resampling.* The medians are computed once, on the whole
panel. A resample that draws more turbulent blocks than the panel has would,
with medians of its own, have higher medians and so fewer large moves than the
fixed medians count; an interval from the fixed medians is several times too
wide for a number that compares a share with one that does not depend on the
level of volatility (``conditional``). So every share is corrected to first
order: for each resample, the share of cells below their median, minus one
half, says how far the resample's own medians would lie from the fixed ones,
and the count of large moves is adjusted by that times a slope measured once
on the panel (how fast the share of large moves changes with the threshold).
For the panel itself the correction is zero.

*The volatility estimate* of a day is the exponentially weighted geometric mean
of the asset's earlier absolute returns (half-life ``FILTER_HALFLIFE`` days).
Nothing of the day itself or of later days enters it.

*The order-flow imbalance* of a day is the volume bought by takers minus the
volume sold by takers, over the day's volume.

**Uncertainty and verdicts** are the first batch's: per-block sums, whole
blocks resampled, the percentile interval widened about the estimate by a
factor measured on simulated panels (``STRETCH``; the measurement is
``preregistration/POWER-LAW-AUDIT-FINANCE-MORE-2026-11/calibration.json``),
and ``fails`` / ``survives`` / ``not_rejected`` at the canonical level.

**Limits.** Medians, ranks and standardisations within an asset are computed
once on the whole panel and held fixed in the resamples (the shares with the
first-order correction above, which is not exact). Ranks across assets
(``momentum``) are computed within each day and are not affected. Windows of
several days are counted under their first day and overlap; the block
resampling carries that dependence inside a block and ignores it across block
edges, which the widening factor has to cover. ``flow`` is a correlation with
the next day's return and so contains whatever a day's return says about the
next; the partial correlation given the same day's return is reported beside
it, without a verdict.
"""
from __future__ import annotations

import functools
import gzip
import io
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.market import law_audit as la

# --- analysis settings (fixed here, frozen with the preregistration; none is a referee threshold) ---
TAIL_MULTIPLE = 4.0       # a large move exceeds this many times the asset's median move
WEEK = 7                  # days in the longer horizon of ``gaussianity`` and in every weekly window
MONTH = 28                # a second horizon, reported without a verdict
MIN_GROUP = 20            # a median needs at least this many values
FILTER_HALFLIFE = 20.0    # half-life, in days, of the volatility estimate
FILTER_MIN = 20           # ... which needs this many earlier returns
FILTER_FLOOR = 0.05       # absolute returns below this many standard deviations count as this much
MOMENTUM_WEEKS = (1, 2, 3, 4)   # how many past weeks the momentum number looks back, averaged
MIN_CROSS = 20            # a rank correlation across assets needs at least this many of them
SLOPE_WINDOW = 0.2        # half-width, in log size, of the two windows the correction's slope is counted in
REFERENCE_DRAWS = 3       # normal panels averaged for the reference of ``still_heavy``
REFERENCE_SEED = 20261121

#: How far each number's percentile interval is widened about the estimate. Measured on simulated
#: panels; ``preregistration/POWER-LAW-AUDIT-FINANCE-MORE-2026-11/calibration.json`` holds the measurement.
STRETCH: Mapping[str, float] = {"asymmetry": 1.2, "gaussianity": 1.5, "conditional": 1.8, "still_heavy": 1.6,
                                "cascade": 1.3, "momentum": 1.5, "flow": 1.3}

ABOVE_ZERO = (0.0, la.INF)
#: Every claim is that its number is above zero. Sources in the preregistration.
PUBLISHED: Mapping[str, tuple[float, float]] = {
    "asymmetry": ABOVE_ZERO,       # Cont (2001), fact 3: larger drawdowns than upward movements
    "gaussianity": ABOVE_ZERO,     # Cont (2001), fact 4: closer to normal over longer time scales
    "conditional": ABOVE_ZERO,     # Cont (2001), fact 7: tails less heavy once volatility is corrected for ...
    "still_heavy": ABOVE_ZERO,     # ... and still heavy
    "cascade": ABOVE_ZERO,         # Cont (2001), fact 11; Mueller et al. (1997): coarse volatility predicts fine
    "momentum": ABOVE_ZERO,        # Liu, Tsyvinski and Wu (2022): one- to four-week momentum across coins
    "flow": ABOVE_ZERO,            # Chordia and Subrahmanyam (2004): order imbalance and the next day's return
}
CLAIMS = tuple(PUBLISHED)
#: every number that is measured; those not in CLAIMS are reported without a verdict
NUMBERS = (*CLAIMS, "gaussianity_month", "share_day", "share_week", "share_month", "share_filtered",
           "coarse_to_fine", "fine_to_coarse", *(f"momentum_{k}" for k in MOMENTUM_WEEKS),
           "flow_same_day", "flow_partial", "return_next_day")


@dataclass(frozen=True)
class FlowPanel:
    """A panel and, for the same cells, the volume bought by takers (NaN where there is no bar)."""

    panel: la.Panel
    taker_buy: np.ndarray

    def between(self, start: str, end: str) -> FlowPanel:
        keep = (self.panel.dates >= np.datetime64(start)) & (self.panel.dates <= np.datetime64(end))
        return FlowPanel(self.panel.between(start, end), self.taker_buy[keep])

    def columns(self, keep: np.ndarray) -> FlowPanel:
        return FlowPanel(self.panel.columns(keep), self.taker_buy[:, keep])


def load_flow_panel(path: Path | str) -> FlowPanel:
    """Read ``panel_*.csv.gz`` as written by ``scripts/build_point_in_time_crypto_panel.py``, with taker-buy volume."""
    import pandas as pd

    panel = la.load_panel(path)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(Path(path).read_bytes())),
                        usecols=[f"{a}_TAKER_BUY" for a in panel.assets])
    return FlowPanel(panel, frame[[f"{a}_TAKER_BUY" for a in panel.assets]].to_numpy(dtype=float))


@dataclass(frozen=True)
class PreparedMore:
    """What the numbers are computed from. Arrays are (days, assets) with NaN for no value."""

    first: la.Prepared       # the first batch's preparation: the same assets and cells
    r: np.ndarray            # log returns, in the cells of ``first.g``
    imbalance: np.ndarray    # order-flow imbalance, where the cell has a return and a positive volume


def raw_returns(panel: la.Panel) -> tuple[np.ndarray, np.ndarray]:
    """Log returns in the cells the first batch uses, and which of the panel's assets it keeps.

    The same four steps as ``law_audit.prepare`` (a test holds the two together): returns between
    consecutive bars, the first ``SKIP_FIRST`` of an asset dropped, assets with ``MIN_DAYS`` returns.
    """
    close = np.where(np.isfinite(panel.close) & (panel.close > 0), panel.close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.full_like(close, np.nan)
        r[1:] = np.log(close[1:] / close[:-1])
    seen = np.cumsum(np.isfinite(r), axis=0)
    r[seen <= la.SKIP_FIRST] = np.nan
    keep: np.ndarray = np.isfinite(r).sum(axis=0) >= la.MIN_DAYS
    return r[:, keep], keep


def prepare(flows: FlowPanel) -> PreparedMore:
    first = la.prepare(flows.panel)
    r, keep = raw_returns(flows.panel)
    volume = np.where(np.isfinite(flows.panel.volume) & (flows.panel.volume > 0), flows.panel.volume, np.nan)[:, keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        imbalance = (2.0 * flows.taker_buy[:, keep] - volume) / volume
    imbalance = np.where(np.isfinite(first.g) & np.isfinite(imbalance), imbalance, np.nan)
    return PreparedMore(first, r, imbalance)


def window_sum(x: np.ndarray, length: int) -> np.ndarray:
    """Under day ``t``: the sum of ``x`` over days ``t .. t + length - 1``; NaN unless all are present."""
    zero = np.zeros((1, x.shape[1]))
    present = np.isfinite(x)
    total = np.vstack([zero, np.cumsum(np.where(present, x, 0.0), axis=0)])
    count = np.vstack([zero, np.cumsum(present, axis=0).astype(float)])
    out: np.ndarray = np.full_like(x, np.nan, dtype=float)
    if x.shape[0] >= length:
        whole = (count[length:] - count[:-length]) == length
        out[: x.shape[0] - length + 1] = np.where(whole, total[length:] - total[:-length], np.nan)
    return out


DaySums = tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LargeMoves:
    """Per day: large moves, cells below their median (one equal to it counts half), cells judged."""

    hits: np.ndarray
    below: np.ndarray
    count: np.ndarray
    slope: float     #: rise of the share of large moves per unit by which the share below the medians exceeds 1/2

    @property
    def sums(self) -> DaySums:
        return self.hits, self.below, self.count


def large_moves(x: np.ndarray, groups: int, multiple: float = TAIL_MULTIPLE) -> LargeMoves:
    """Which cells of ``x`` are large moves, against medians within (asset, day modulo ``groups``).

    A cell is large if its size exceeds ``multiple`` times the median size among the cells of the
    same asset whose day has the same remainder modulo ``groups``. A group with fewer than
    ``MIN_GROUP`` values, or a median of zero, is not judged. ``slope`` is the density of log size
    at the threshold over its density at the median, both counted over all judged cells in windows
    of half-width ``SLOPE_WINDOW``: lowering every median by a factor that puts a share ``d`` more
    of the cells above it adds ``slope * d`` to the share of large moves.
    """
    size = np.abs(x)
    days = x.shape[0]
    hits, below, count = np.zeros(days), np.zeros(days), np.zeros(days)
    near_median = near_threshold = 0.0
    for offset in range(groups):
        part = size[offset::groups]
        present = np.isfinite(part)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # an asset with no value in this group
            median = np.nanmedian(part, axis=0) if len(part) else np.full(x.shape[1], np.nan)
        judged = present & ((present.sum(axis=0) >= MIN_GROUP) & (median > 0))[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = np.where(judged, part / median[None, :], np.nan)
            log_size = np.log(relative)
            hits[offset::groups] = (relative > multiple).sum(axis=1)
            below[offset::groups] = (relative < 1.0).sum(axis=1) + 0.5 * (relative == 1.0).sum(axis=1)
            near_median += float((np.abs(log_size) <= SLOPE_WINDOW).sum())
            near_threshold += float((np.abs(log_size - np.log(multiple)) <= SLOPE_WINDOW).sum())
        count[offset::groups] = judged.sum(axis=1)
    return LargeMoves(hits, below, count, near_threshold / near_median if near_median else 0.0)


def volatility_estimate(g: np.ndarray, halflife: float = FILTER_HALFLIFE, minimum: int = FILTER_MIN,
                        floor: float = FILTER_FLOOR) -> np.ndarray:
    """Per cell: the exponentially weighted geometric mean of the asset's earlier absolute returns.

    NaN until the asset has ``minimum`` earlier returns. The day's own return is not used.
    """
    decay = 0.5 ** (1.0 / halflife)
    with np.errstate(invalid="ignore"):
        logs = np.log(np.maximum(np.abs(g), floor))
    total, weight, seen = (np.zeros(g.shape[1]) for _ in range(3))
    out: np.ndarray = np.full_like(g, np.nan, dtype=float)
    for t in range(g.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(seen >= minimum, np.exp(total / np.where(weight > 0, weight, 1.0)), np.nan)
        present = np.isfinite(logs[t])
        total = decay * total + np.where(present, logs[t], 0.0)
        weight = decay * weight + present
        seen = seen + present
    return out


def filtered(g: np.ndarray) -> np.ndarray:
    """Each return over the volatility estimate of its day."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out: np.ndarray = g / volatility_estimate(g)
    return out


@functools.lru_cache(maxsize=8)
def _normal_reference(packed: bytes, shape: tuple[int, int], draws: int, seed: int) -> float:
    present: np.ndarray = np.unpackbits(np.frombuffer(packed, dtype=np.uint8),
                                        count=shape[0] * shape[1]).reshape(shape).astype(bool)
    rng = np.random.default_rng(seed)
    hits = count = 0.0
    for _ in range(draws):
        moves = large_moves(filtered(np.where(present, rng.standard_normal(shape), np.nan)), 1)
        hits, count = hits + float(moves.hits.sum()), count + float(moves.count.sum())
    return hits / count if count else float("nan")


def normal_reference(present: np.ndarray, draws: int = REFERENCE_DRAWS, seed: int = REFERENCE_SEED) -> float:
    """The share of large moves that independent normal returns in the cells ``present`` leave after filtering.

    The volatility estimate is noisy and the medians are of finitely many values, so this is above
    the 0.7 % of a normal distribution, by an amount that depends on how long the assets are present.
    Seeded, and remembered for a pattern of cells already seen.
    """
    cells = np.asarray(present, dtype=bool)
    return _normal_reference(np.packbits(cells).tobytes(), (int(cells.shape[0]), int(cells.shape[1])), draws, seed)


def cross_ranks(x: np.ndarray, minimum: int = MIN_CROSS) -> np.ndarray:
    """Each day's finite cells ranked across assets, to mean zero and unit variance; NaN on thin days."""
    from scipy.stats import rankdata

    present = np.isfinite(x)
    if not x.size:
        empty: np.ndarray = np.full(x.shape, np.nan)
        return empty
    ranks = np.asarray(rankdata(np.where(present, x, np.nan), axis=1, nan_policy="omit"), dtype=float)
    count = present.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # a day with no asset
        mean = np.nanmean(ranks, axis=1, keepdims=True)
        scale = np.nanstd(ranks, axis=1, keepdims=True)
        out: np.ndarray = np.where(present & (count >= minimum) & (scale > 0), (ranks - mean) / scale, np.nan)
    return out


def _product_day_sums(x: np.ndarray, y: np.ndarray) -> DaySums:
    """Per day: the sum over assets of ``x * y`` where both exist, and the count."""
    product = x * y
    ok = np.isfinite(product)
    return np.where(ok, product, 0.0).sum(axis=1), ok.sum(axis=1).astype(float)


def _shifted(x: np.ndarray, days: int) -> np.ndarray:
    """``x`` moved so that day ``t`` holds the value of day ``t + days`` (``days`` may be negative)."""
    out: np.ndarray = np.full_like(x, np.nan, dtype=float)
    if days > 0:
        out[:-days] = x[days:]
    elif days < 0:
        out[-days:] = x[:days]
    else:
        out[:] = x
    return out


@dataclass(frozen=True)
class Sums:
    """Per-day sums by name, the slope of each share's correction, and the reference of ``still_heavy``."""

    day: dict[str, DaySums]
    slopes: dict[str, float]
    reference: float
    days: int


def day_sums(prepared: PreparedMore) -> Sums:
    """Every sum the numbers need, per day; a window of several days is counted under its first day."""
    first, r = prepared.first, prepared.r
    g = first.g
    day: dict[str, DaySums] = {}
    shares: dict[str, LargeMoves] = {}

    tail = la.tail_day_sums(first.excess, g)
    day["losses"] = (tail["n_neg"], tail["n_pos"] + tail["n_neg"])

    for name, length in (("week", WEEK), ("month", MONTH)):
        shares[f"day_for_{name}"] = large_moves(g, length)
        shares[name] = large_moves(window_sum(g, length), length)
    after = filtered(g)
    shares["filtered"] = large_moves(after, 1)
    shares["unfiltered"] = large_moves(np.where(np.isfinite(after), g, np.nan), 1)
    day.update({name: moves.sums for name, moves in shares.items()})

    coarse = la._standardise(la._ranks(np.abs(window_sum(g, WEEK))))
    fine = la._standardise(la._ranks(window_sum(np.abs(g), WEEK)))
    day["coarse_to_fine"] = _product_day_sums(coarse, _shifted(fine, WEEK))
    day["fine_to_coarse"] = _product_day_sums(fine, _shifted(coarse, WEEK))

    ahead = window_sum(r, WEEK)
    for weeks in MOMENTUM_WEEKS:
        past = _shifted(window_sum(r, WEEK * weeks), -WEEK * weeks)
        both = np.isfinite(past) & np.isfinite(ahead)
        day[f"momentum_{weeks}"] = _product_day_sums(cross_ranks(np.where(both, past, np.nan)),
                                                     cross_ranks(np.where(both, ahead, np.nan)))

    flow = la._standardise(la._ranks(prepared.imbalance))
    ret = la._standardise(la._ranks(g))
    day["flow_next"] = _product_day_sums(flow, _shifted(ret, 1))
    day["flow_same"] = _product_day_sums(flow, ret)
    day["return_next"] = _product_day_sums(ret, _shifted(ret, 1))
    return Sums(day, {name: moves.slope for name, moves in shares.items()}, normal_reference(np.isfinite(g)), first.days)


class BlockSums:
    """The sums per block of days; and the numbers for any weighting of the blocks."""

    def __init__(self, sums: Sums, block_days: int = la.BLOCK_DAYS) -> None:
        block, self.blocks = la.block_index(sums.days, block_days)
        self.sums = {key: tuple(np.bincount(block, weights=part, minlength=self.blocks) for part in parts)
                     for key, parts in sums.day.items()}
        self.slopes, self.reference = sums.slopes, sums.reference

    def numbers(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        """Every number for each row of ``weights`` (how many times each block is taken)."""
        w = np.atleast_2d(np.asarray(weights, dtype=float))

        def ratio(key: str) -> np.ndarray:
            with np.errstate(invalid="ignore", divide="ignore"):
                value: np.ndarray = (w @ self.sums[key][0]) / (w @ self.sums[key][-1])
            return value

        def share(key: str) -> np.ndarray:
            """Large moves over cells judged, corrected for where the resample's own medians would lie."""
            hits, below, count = self.sums[key]
            with np.errstate(invalid="ignore", divide="ignore"):
                value: np.ndarray = (w @ hits) / (w @ count) + self.slopes[key] * ((w @ below) / (w @ count) - 0.5)
            return value

        out: dict[str, np.ndarray] = {"asymmetry": ratio("losses") - 0.5}
        out["share_day"], out["share_week"], out["share_month"] = share("day_for_week"), share("week"), share("month")
        out["gaussianity"] = out["share_day"] - out["share_week"]
        out["gaussianity_month"] = share("day_for_month") - out["share_month"]
        out["share_filtered"] = share("filtered")
        out["conditional"] = share("unfiltered") - out["share_filtered"]
        out["still_heavy"] = out["share_filtered"] - self.reference
        out["coarse_to_fine"], out["fine_to_coarse"] = ratio("coarse_to_fine"), ratio("fine_to_coarse")
        out["cascade"] = out["coarse_to_fine"] - out["fine_to_coarse"]
        for weeks in MOMENTUM_WEEKS:
            out[f"momentum_{weeks}"] = ratio(f"momentum_{weeks}")
        out["momentum"] = np.mean([out[f"momentum_{weeks}"] for weeks in MOMENTUM_WEEKS], axis=0)
        out["flow"], out["flow_same_day"], out["return_next_day"] = ratio("flow_next"), ratio("flow_same"), ratio("return_next")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["flow_partial"] = ((out["flow"] - out["flow_same_day"] * out["return_next_day"])
                                   / np.sqrt((1.0 - out["flow_same_day"] ** 2) * (1.0 - out["return_next_day"] ** 2)))
        return out


def measure(prepared: PreparedMore, *, resamples: int = la.RESAMPLES, seed: int = la.SEED) -> dict[str, Any]:
    """Every number: ``{number: {estimate, percentile, yearly}}``, and the counts."""
    sums = day_sums(prepared)
    short, long = BlockSums(sums, la.BLOCK_DAYS), BlockSums(sums, la.YEAR_DAYS)
    point = short.numbers(np.ones((1, short.blocks)))
    draws = short.numbers(la.resample_weights(short.blocks, resamples, seed))
    each = long.numbers(np.eye(long.blocks))
    out: dict[str, Any] = {"blocks": short.blocks, "years": long.blocks,
                           "cells": {key: float(parts[-1].sum()) for key, parts in sums.day.items()},
                           "slopes": dict(sums.slopes), "normal_reference": sums.reference}
    for number in NUMBERS:
        out[number] = {"estimate": float(point[number][0]), "percentile": list(la.interval(draws[number])),
                       "yearly": each[number].tolist()}
    return out


def audit(flows: FlowPanel, *, resamples: int = la.RESAMPLES, seed: int = la.SEED,
          stretch: Mapping[str, float] = STRETCH) -> dict[str, Any]:
    """Measure the numbers on ``flows`` and compare each claim's number with its published set."""
    prepared = prepare(flows)
    measured = measure(prepared, resamples=resamples, seed=seed)
    readings: dict[str, Any] = {}
    for claim, published in PUBLISHED.items():
        number = measured[claim]
        low, high = la.stretched(number["estimate"], number["percentile"][0], number["percentile"][1], stretch[claim])
        readings[claim] = la.Reading(claim, number["estimate"], low, high, published,
                                     la.verdict(low, high, published), {"stretch": stretch[claim]}).as_dict()
    settings = {"level": la.LEVEL, "tail_multiple": TAIL_MULTIPLE, "week": WEEK, "month": MONTH,
                "min_group": MIN_GROUP, "filter_halflife": FILTER_HALFLIFE, "filter_min": FILTER_MIN,
                "filter_floor": FILTER_FLOOR, "slope_window": SLOPE_WINDOW,
                "reference_draws": REFERENCE_DRAWS, "reference_seed": REFERENCE_SEED,
                "momentum_weeks": list(MOMENTUM_WEEKS), "min_cross": MIN_CROSS, "block_days": la.BLOCK_DAYS,
                "year_days": la.YEAR_DAYS, "skip_first": la.SKIP_FIRST, "min_days": la.MIN_DAYS,
                "tail_quantile": la.TAIL_QUANTILE, "resamples": resamples, "seed": seed, "stretch": dict(stretch)}
    return {"readings": readings, "numbers": measured, "assets": len(prepared.first.assets),
            "asset_days": prepared.first.asset_days, "settings": settings}
