"""Audit published laws of asset returns on a panel they were not fitted on.

Status: experimental

**Why.** Every measured result of this repository is a benchmark whose answer is
known. The first step towards a finding on real data is to take laws that are
already published, state each as a number with the range its authors gave, and
measure that number on data the authors never saw, with an interval that is
honest about how much the data can tell. A law that fails there is a finding;
one that holds is a control for whatever is searched for next; and a test that
cannot tell is reported as that.

**What.** For a daily panel of closes and volumes (every asset, delisted ones
included), six numbers, each compared with a published set:

===============  ============================================================  ====================
Claim            Number measured here                                          Published set
===============  ============================================================  ====================
``tail``         Hill tail index over each asset's largest absolute returns    (2, 5)
``tail_cubic``   the same number                                               [2.72, 3.13]
``random``       variance ratio of 5-day to 1-day returns                      {1}
``memory``       decay exponent of the autocorrelation of absolute returns     [0.2, 0.4]
``leverage``     correlation of a return with later absolute returns           below 0
``volume``       correlation of absolute return with relative volume           above 0
``independent``  lag-1 autocorrelation of absolute returns (a control)         {0}
===============  ============================================================  ====================

The sources and the exact wording of each claim are in
``preregistration/PREREG-LAW-AUDIT-FINANCE-2026-11.md``.

*Absolute returns* enter the dependence numbers as their rank within the
asset. A rank is a monotone function of the absolute return, so independence,
the sign of a dependence and the exponent of a power-law decay are the same for
both, and a rank is bounded: the few days on which the whole market moves do
not carry the estimate. The memory exponent is also given for the absolute
returns themselves (``memory_abs``), without a verdict.

*The memory exponent* is read from the variogram, half the mean squared
difference of the series at two days ``lag`` apart, which is the variance times
one minus the autocorrelation and needs no estimate of the mean. Under long
memory the sample mean is itself uncertain and the usual autocorrelation
estimate is biased towards a faster decay; the variogram is not. The exponent
is the ``beta`` for which ``level - scale * lag ** -beta`` fits the variogram
best in least squares, with level and scale free.

**Uncertainty.** Every number is a function of sums over (asset, day) cells.
The sums are kept per block of ``BLOCK_DAYS`` consecutive days and whole blocks
are resampled, so that what many assets do on the same day counts once and
serial dependence inside a block is kept. Lagged pairs never straddle two
blocks, in the estimate and in every resample alike. On simulated panels with
heavy-tailed common shocks and long-memory volatility the percentile interval
of that resampling is too narrow for some of the numbers, so each number's
interval is widened about its estimate by a fixed factor, ``STRETCH``, measured
on those panels (``scripts/run_law_audit_finance.py calibrate``). The factors
are therefore as good as the simulated panels are like a market, and no better.

**Verdicts.** With the interval at the canonical level
(``DIEBOLD_MARIANO_P_VALUE_THRESHOLD``): ``fails`` if the interval lies wholly
outside the published set, ``survives`` if wholly inside (possible only for a
set with an interior), otherwise ``not_rejected``, in which case the interval is
itself the result: the law holds to within that much, and no closer.

**Limits.** Per-asset means, scales, ranks and tail thresholds are computed once
on the whole panel and held fixed in the resamples. A tail index measured over
the largest 5 % of returns is not the limiting exponent: in simulated worlds
whose shocks have exponent 3 it reads 2.95 when volatility is constant and 2.3
when volatility wanders. On five years of daily data the decay exponent's
interval is about twice as wide as the published range. Nothing here decides
whether a law is *the* law: a number inside a published range is agreement
with that range and no more.
"""
from __future__ import annotations

import gzip
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD

#: Two-sided level of every interval: the referee's own threshold for a forecast comparison.
LEVEL: float = float(DIEBOLD_MARIANO_P_VALUE_THRESHOLD)

# --- analysis settings (fixed here, frozen with the preregistration; none is a referee threshold) ---
SKIP_FIRST = 5          # returns dropped after an asset's first bar: listing-day price discovery
MIN_DAYS = 250          # an asset enters only with at least this many returns
WINSOR = 5.0            # the variance ratio and the leverage number use returns clipped at this many sd
TAIL_QUANTILE = 0.95    # the tail of an asset is its absolute returns above its own quantile at this level
BLOCK_DAYS = 90         # resampling block, in calendar days
YEAR_DAYS = 365         # the numbers are also given for each stretch of this many days, without a verdict
MEMORY_LAGS = 40        # the variogram is fitted over lags 1 .. this
LEVERAGE_LAGS = 10      # the leverage number averages lags 1 .. this
VARIANCE_RATIO_Q = 5    # horizon of the variance ratio, in days
VOLUME_WINDOW = 30      # relative volume is volume over its geometric mean in this many earlier days
VOLUME_MIN = 20         # ... with at least this many of them present
RESAMPLES = 4000
SEED = 20261120
#: exponents the decay fit chooses from; an estimate at either end means "beyond the grid"
BETA_GRID: np.ndarray = np.linspace(-0.5, 2.5, 301)
#: half-lives, in days, of the exponential decays the form test chooses from
HALFLIFE_GRID: np.ndarray = np.geomspace(0.5, 400.0, 400)
RHO_LAGS = tuple(range(1, VARIANCE_RATIO_Q))
MEMORY_RANGE_LAGS = tuple(range(1, MEMORY_LAGS + 1))
LEVERAGE_RANGE_LAGS = tuple(range(1, LEVERAGE_LAGS + 1))

#: How far each number's percentile interval is widened about the estimate. Measured on simulated
#: panels; ``preregistration/POWER-LAW-AUDIT-FINANCE-2026-11/calibration.json`` holds the measurement.
STRETCH: Mapping[str, float] = {"tail": 2.3, "random": 1.5, "memory": 1.3, "leverage": 1.3,
                                "volume": 1.1, "independent": 1.3}

# --- published sets (sources in the preregistration) ---
INF = float("inf")
#: Cont (2001), fact 2: a tail index "higher than two and less than five".
TAIL_RANGE_WEAK = (2.0, 5.0)
#: Gopikrishnan, Meyer, Amaral, Stanley (1998): 3.10 +- 0.03 and 2.84 +- 0.12 for the two tails.
TAIL_RANGE_CUBIC = (2.84 - 0.12, 3.10 + 0.03)
#: Cont (2001), fact 1: no linear autocorrelation, hence a variance ratio of one.
RANDOM_WALK = (1.0, 1.0)
#: Cont (2001), fact 8: the autocorrelation of absolute returns decays with exponent in [0.2, 0.4].
MEMORY_RANGE = (0.2, 0.4)
#: Cont (2001), fact 9; Bouchaud, Matacz, Potters (2001): past returns and later volatility, negative.
LEVERAGE_SIGN = (-INF, 0.0)
#: Cont (2001), fact 10; Karpoff (1987): volume and volatility, positively correlated.
VOLUME_SIGN = (0.0, INF)
#: The control: independent returns would have uncorrelated absolute values. Known to be false.
INDEPENDENT = (0.0, 0.0)

PUBLISHED: Mapping[str, tuple[float, float]] = {
    "tail": TAIL_RANGE_WEAK, "tail_cubic": TAIL_RANGE_CUBIC, "random": RANDOM_WALK,
    "memory": MEMORY_RANGE, "leverage": LEVERAGE_SIGN, "volume": VOLUME_SIGN,
    "independent": INDEPENDENT,
}
#: which number each claim reads
NUMBER_OF: Mapping[str, str] = {"tail": "tail", "tail_cubic": "tail", "random": "random",
                                "memory": "memory", "leverage": "leverage", "volume": "volume",
                                "independent": "independent"}
#: every number that is measured; those not in NUMBER_OF are reported without a verdict
NUMBERS = ("tail", "tail_pos", "tail_neg", "random", "memory", "memory_abs", "leverage", "volume",
           "independent")
CURVES = ("variogram", "variogram_abs", "leverage_curve", "autocorrelation")

FAILS, SURVIVES, NOT_REJECTED = "fails", "survives", "not_rejected"
PUBLISHED_FORM, RIVAL_FORM, NOT_DECIDED = "published_form", "rival_form", "not_decided"


@dataclass(frozen=True)
class Panel:
    """Daily closes and volumes, one column per asset, NaN where the asset has no bar."""

    dates: np.ndarray
    assets: tuple[str, ...]
    close: np.ndarray
    volume: np.ndarray

    def between(self, start: str, end: str) -> Panel:
        """The rows with ``start <= date <= end`` (ISO dates)."""
        keep = (self.dates >= np.datetime64(start)) & (self.dates <= np.datetime64(end))
        return Panel(self.dates[keep], self.assets, self.close[keep], self.volume[keep])

    def columns(self, keep: np.ndarray) -> Panel:
        """The assets selected by a boolean mask."""
        names = tuple(a for a, k in zip(self.assets, keep, strict=True) if k)
        return Panel(self.dates, names, self.close[:, keep], self.volume[:, keep])


def load_panel(path: Path | str) -> Panel:
    """Read ``panel_*.csv.gz`` as written by ``scripts/build_point_in_time_crypto_panel.py``."""
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(gzip.decompress(Path(path).read_bytes())))
    assets = tuple(sorted(c[: -len("_CLOSE")] for c in frame.columns if c.endswith("_CLOSE")))
    dates = frame["observation_date"].to_numpy(dtype="datetime64[D]")
    close = frame[[f"{a}_CLOSE" for a in assets]].to_numpy(dtype=float)
    volume = frame[[f"{a}_VOL" for a in assets]].to_numpy(dtype=float)
    return Panel(dates, assets, close, volume)


@dataclass(frozen=True)
class Prepared:
    """What every statistic is computed from. All arrays are (days, assets) with NaN for no value."""

    assets: tuple[str, ...]
    g: np.ndarray            # returns normalised by each asset's own mean and standard deviation
    gw: np.ndarray           # the same, clipped at WINSOR and standardised again
    a: np.ndarray            # |gw|, standardised within the asset
    vol: np.ndarray          # rank of |g| within the asset, standardised: the volatility proxy
    excess: np.ndarray       # |g| over the asset's tail threshold, where it exceeds it; NaN elsewhere
    a_rank: np.ndarray       # rank of |g| among the asset's days that have a relative volume
    v_rank: np.ndarray       # rank of relative volume among the same days
    days: int
    asset_days: int


def _standardise(x: np.ndarray) -> np.ndarray:
    """Each column to mean zero and unit variance over its finite cells."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
    scale = np.where(scale > 0, scale, np.nan)
    out: np.ndarray = (x - mean) / scale
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Rank of each finite cell within its column (ties share their mean rank), NaN elsewhere."""
    from scipy.stats import rankdata

    out: np.ndarray = np.full_like(x, np.nan, dtype=float)
    if x.size:
        out = np.asarray(rankdata(np.where(np.isfinite(x), x, np.nan), axis=0, nan_policy="omit"), dtype=float)
    return out


def _quantile(x: np.ndarray, level: float) -> np.ndarray:
    """The ``level`` quantile of each column's finite cells (linear between order statistics); NaN if none."""
    ordered = np.sort(np.where(np.isfinite(x), x, np.nan), axis=0)       # NaN sorts last
    count = np.isfinite(ordered).sum(axis=0)
    position = np.maximum(count - 1, 0) * level
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, np.maximum(count - 1, 0))
    columns = np.arange(x.shape[1])
    value: np.ndarray = np.full(x.shape[1], np.nan)
    if x.shape[0]:
        value = ordered[low, columns] + (position - low) * (ordered[high, columns] - ordered[low, columns])
        value = np.where(count > 0, value, np.nan)
    return value


def block_index(days: int, block_days: int = BLOCK_DAYS) -> tuple[np.ndarray, int]:
    """Consecutive blocks of ``block_days``; a last block under half that length joins the one before."""
    index: np.ndarray = np.arange(days, dtype=np.int64) // block_days
    blocks = int(index[-1]) + 1
    if blocks > 1 and int((index == blocks - 1).sum()) < block_days / 2:
        index[index == blocks - 1] = blocks - 2
        blocks -= 1
    return index, blocks


def relative_volume(log_volume: np.ndarray, window: int = VOLUME_WINDOW,
                    minimum: int = VOLUME_MIN) -> np.ndarray:
    """Log volume minus its mean over the ``window`` earlier days, where ``minimum`` of them exist."""
    present = np.isfinite(log_volume)
    zero = np.zeros((1, log_volume.shape[1]))
    total = np.vstack([zero, np.cumsum(np.where(present, log_volume, 0.0), axis=0)])
    count = np.vstack([zero, np.cumsum(present, axis=0).astype(float)])
    days = np.arange(log_volume.shape[0])
    start = np.maximum(days - window, 0)
    earlier_sum = total[days] - total[start]        # days start .. t-1
    earlier_count = count[days] - count[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(earlier_count >= minimum, earlier_sum / earlier_count, np.nan)
    relative: np.ndarray = log_volume - mean
    return relative


def prepare(panel: Panel) -> Prepared:
    """Returns, their normalised, clipped and ranked forms, tail excesses and relative volume."""
    close = np.where(np.isfinite(panel.close) & (panel.close > 0), panel.close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.full_like(close, np.nan)
        r[1:] = np.log(close[1:] / close[:-1])       # only between two consecutive daily bars
    seen = np.cumsum(np.isfinite(r), axis=0)
    r[seen <= SKIP_FIRST] = np.nan                   # the first returns after each asset's first bar
    keep = np.isfinite(r).sum(axis=0) >= MIN_DAYS
    r = r[:, keep]
    assets = tuple(a for a, k in zip(panel.assets, keep, strict=True) if k)
    g = _standardise(r)
    gw = _standardise(np.clip(g, -WINSOR, WINSOR))
    a = _standardise(np.abs(gw))
    size = np.abs(g)
    vol = _standardise(_ranks(size))

    threshold = _quantile(size, TAIL_QUANTILE)
    with np.errstate(invalid="ignore"):
        over = np.isfinite(size) & (size > threshold[None, :]) & (threshold[None, :] > 0)
        excess = np.where(over, size / np.where(threshold > 0, threshold, 1.0)[None, :], np.nan)

    volume = np.where(np.isfinite(panel.volume) & (panel.volume > 0), panel.volume, np.nan)[:, keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = relative_volume(np.log(volume))
    both = np.isfinite(g) & np.isfinite(relative)
    a_rank = _standardise(_ranks(np.where(both, size, np.nan)))
    v_rank = _standardise(_ranks(np.where(both, relative, np.nan)))

    return Prepared(assets, g, gw, a, vol, excess, a_rank, v_rank, len(panel.dates),
                    int(np.isfinite(g).sum()))


DayLagSums = tuple[np.ndarray, np.ndarray]


def lag_day_sums(x: np.ndarray, y: np.ndarray, lags: Sequence[int]) -> DayLagSums:
    """Per day ``t`` and lag: the sum over assets of ``x[t] * y[t + lag]`` where both exist, and the count."""
    days = x.shape[0]
    sums, counts = np.zeros((days, len(lags))), np.zeros((days, len(lags)))
    for j, lag in enumerate(lags):
        product = x * y if lag == 0 else x[:-lag] * y[lag:]
        ok = np.isfinite(product)
        sums[: days - lag, j] = np.where(ok, product, 0.0).sum(axis=1)
        counts[: days - lag, j] = ok.sum(axis=1)
    return sums, counts


def variogram_day_sums(x: np.ndarray, lags: Sequence[int]) -> DayLagSums:
    """Per day ``t`` and lag: the sum over assets of ``(x[t + lag] - x[t]) ** 2 / 2``, and the count.

    Half the mean squared difference is the variance times one minus the autocorrelation, and it
    needs no estimate of the mean: a series whose level wanders slowly does not bias it.
    """
    days = x.shape[0]
    sums, counts = np.zeros((days, len(lags))), np.zeros((days, len(lags)))
    for j, lag in enumerate(lags):
        half = 0.5 * (x[lag:] - x[:-lag]) ** 2
        ok = np.isfinite(half)
        sums[: days - lag, j] = np.where(ok, half, 0.0).sum(axis=1)
        counts[: days - lag, j] = ok.sum(axis=1)
    return sums, counts


def to_blocks(day: DayLagSums, lags: Sequence[int], block: np.ndarray, blocks: int) -> DayLagSums:
    """Per block and lag, from per-day sums. A pair counts only if both its days lie in the same block."""
    days = len(block)
    sums, counts = np.zeros((blocks, len(lags))), np.zeros((blocks, len(lags)))
    for j, lag in enumerate(lags):
        first = block[: days - lag]
        same = first == block[lag:]
        sums[:, j] = np.bincount(first[same], weights=day[0][: days - lag, j][same], minlength=blocks)
        counts[:, j] = np.bincount(first[same], weights=day[1][: days - lag, j][same], minlength=blocks)
    return sums, counts


def lag_sums(x: np.ndarray, y: np.ndarray, lags: Sequence[int], block: np.ndarray,
             blocks: int) -> DayLagSums:
    """Per block and lag: the sum of ``x[t] * y[t + lag]`` over pairs inside the block, and the count."""
    return to_blocks(lag_day_sums(x, y, lags), lags, block, blocks)


def variogram_sums(x: np.ndarray, lags: Sequence[int], block: np.ndarray, blocks: int) -> DayLagSums:
    """Per block and lag: the sum of ``(x[t + lag] - x[t]) ** 2 / 2`` over pairs inside the block."""
    return to_blocks(variogram_day_sums(x, lags), lags, block, blocks)


def tail_day_sums(excess: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray]:
    """Per day: how many cells lie in the tail, the sum of ``log(excess)`` and of ``excess - 1``.

    ``excess`` is the absolute return over the asset's tail threshold. Keys ``n``, ``log``,
    ``linear`` for both tails together, and ``n_pos``, ``log_pos``, ``n_neg``, ``log_neg`` by the
    sign of the return.
    """
    over = np.isfinite(excess)
    logs = np.log(np.where(over, excess, 1.0))
    out: dict[str, np.ndarray] = {"linear": np.where(over, excess - 1.0, 0.0).sum(axis=1)}
    with np.errstate(invalid="ignore"):
        sides = (("", over), ("_pos", over & (g > 0)), ("_neg", over & (g < 0)))
    for key, chosen in sides:
        out["n" + key] = chosen.sum(axis=1).astype(float)
        out["log" + key] = np.where(chosen, logs, 0.0).sum(axis=1)
    return out


def _unit(shapes: np.ndarray) -> np.ndarray:
    """Each row centred and scaled to unit length (a row with no variation becomes zero)."""
    centred = shapes - shapes.mean(axis=1, keepdims=True)
    length = np.sqrt((centred ** 2).sum(axis=1, keepdims=True))
    out: np.ndarray = centred / np.where(length > 0, length, 1.0)
    return out


def power_shapes(lags: int = MEMORY_LAGS, betas: np.ndarray = BETA_GRID) -> np.ndarray:
    """For each exponent, ``(1 - lag ** -beta) / beta`` at lags 1 .. ``lags``, centred, unit length.

    Every row rises with the lag; ``beta = 0`` is the logarithm, the limit of the family.
    """
    lag: np.ndarray = np.arange(1, lags + 1, dtype=float)[None, :]
    beta: np.ndarray = np.asarray(betas, dtype=float)[:, None]
    small = np.abs(beta) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        shapes = np.where(small, np.log(lag), (1.0 - lag ** (-beta)) / np.where(small, 1.0, beta))
    return _unit(shapes)


def exponential_shapes(lags: int = MEMORY_LAGS, halflives: np.ndarray = HALFLIFE_GRID) -> np.ndarray:
    """For each half-life, ``1 - 0.5 ** (lag / halflife)`` at lags 1 .. ``lags``, centred, unit length."""
    lag: np.ndarray = np.arange(1, lags + 1, dtype=float)[None, :]
    return _unit(1.0 - 0.5 ** (lag / np.asarray(halflives, dtype=float)[:, None]))


def best_shape(variogram: np.ndarray, shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``variogram``: which shape fits best with a free level and a positive scale.

    Returns the index of the shape and the share of the row's variation over lags that it explains;
    the index is -1 and the share NaN where no shape rises with the row.
    """
    rows = np.atleast_2d(variogram)
    centred = rows - rows.mean(axis=1, keepdims=True)
    score = centred @ shapes.T
    best = score.argmax(axis=1)
    top = score[np.arange(len(rows)), best]
    total = (centred ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (top > 0) & (total > 0) & np.isfinite(top)
        share = np.where(ok, top ** 2 / np.where(total > 0, total, 1.0), np.nan)
    index: np.ndarray = np.where(ok, best, -1)
    return index, share


def decay_exponent(variogram: np.ndarray) -> np.ndarray:
    """The ``beta`` of the power-law decay that fits each row of ``variogram`` best; NaN if none does."""
    rows = np.atleast_2d(variogram)
    index, _ = best_shape(rows, power_shapes(rows.shape[1]))
    beta: np.ndarray = np.where(index >= 0, BETA_GRID[np.maximum(index, 0)], np.nan)
    return beta


def verdict(low: float, high: float, published: tuple[float, float]) -> str:
    """``fails`` / ``survives`` / ``not_rejected`` for an interval against a published set."""
    p_low, p_high = published
    if not (np.isfinite(low) and np.isfinite(high)):
        return NOT_REJECTED
    if high < p_low or low > p_high:
        return FAILS
    if p_low < p_high and low >= p_low and high <= p_high:
        return SURVIVES
    return NOT_REJECTED


@dataclass(frozen=True)
class Reading:
    """One number, its interval, and what it says about a published set."""

    claim: str
    estimate: float
    low: float
    high: float
    published: tuple[float, float]
    verdict: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "estimate": self.estimate, "low": self.low, "high": self.high,
                "published": list(self.published), "verdict": self.verdict, **self.detail}


class DaySums:
    """Every sum the numbers need, per day; for a lagged pair, under the first of its two days."""

    def __init__(self, prepared: Prepared) -> None:
        p = prepared
        self.days = p.days
        self.tail = tail_day_sums(p.excess, p.g)
        self.rho = lag_day_sums(p.gw, p.gw, RHO_LAGS)
        self.vario = variogram_day_sums(p.vol, MEMORY_RANGE_LAGS)
        self.vario_abs = variogram_day_sums(p.a, MEMORY_RANGE_LAGS)
        self.lev = lag_day_sums(p.gw, p.vol, LEVERAGE_RANGE_LAGS)
        self.first = lag_day_sums(p.vol, p.vol, [1])
        self.volume = lag_day_sums(p.a_rank, p.v_rank, [0])


class BlockSums:
    """The same sums per block of days; and the numbers for any weighting of the blocks."""

    def __init__(self, day: DaySums, block_days: int = BLOCK_DAYS) -> None:
        block, blocks = block_index(day.days, block_days)
        self.blocks = blocks
        self.tail = {k: np.bincount(block, weights=v, minlength=blocks) for k, v in day.tail.items()}
        self.rho = to_blocks(day.rho, RHO_LAGS, block, blocks)
        self.vario = to_blocks(day.vario, MEMORY_RANGE_LAGS, block, blocks)
        self.vario_abs = to_blocks(day.vario_abs, MEMORY_RANGE_LAGS, block, blocks)
        self.lev = to_blocks(day.lev, LEVERAGE_RANGE_LAGS, block, blocks)
        self.first = to_blocks(day.first, [1], block, blocks)
        self.volume = to_blocks(day.volume, [0], block, blocks)

    @staticmethod
    def _ratio(weights: np.ndarray, sums_counts: DayLagSums) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio: np.ndarray = (weights @ sums_counts[0]) / (weights @ sums_counts[1])
        return ratio

    def numbers(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        """The numbers and curves for each row of ``weights`` (how many times each block is taken)."""
        w = np.atleast_2d(np.asarray(weights, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            tail = (w @ self.tail["n"]) / (w @ self.tail["log"])
            tail_pos = (w @ self.tail["n_pos"]) / (w @ self.tail["log_pos"])
            tail_neg = (w @ self.tail["n_neg"]) / (w @ self.tail["log_neg"])
        rho = self._ratio(w, self.rho)
        q = VARIANCE_RATIO_Q
        ratio = 1.0 + 2.0 * (rho * (1.0 - np.arange(1, q) / q)).sum(axis=1)
        variogram, variogram_abs = self._ratio(w, self.vario), self._ratio(w, self.vario_abs)
        leverage = self._ratio(w, self.lev)
        return {"tail": tail, "tail_pos": tail_pos, "tail_neg": tail_neg, "random": ratio,
                "memory": decay_exponent(variogram), "memory_abs": decay_exponent(variogram_abs),
                "leverage": leverage.mean(axis=1), "volume": self._ratio(w, self.volume)[:, 0],
                "independent": self._ratio(w, self.first)[:, 0],
                "variogram": variogram, "variogram_abs": variogram_abs, "leverage_curve": leverage,
                "autocorrelation": rho}


def resample_weights(blocks: int, resamples: int = RESAMPLES, seed: int = SEED) -> np.ndarray:
    """How many times each block is drawn in each resample of ``blocks`` blocks with replacement."""
    rng = np.random.default_rng(seed)
    weights: np.ndarray = rng.multinomial(blocks, np.full(blocks, 1.0 / blocks), size=resamples).astype(float)
    return weights


def interval(samples: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    """The central ``1 - level`` percentile interval of the samples; NaN if a tenth of them are not finite."""
    finite = samples[np.isfinite(samples)]
    if finite.size < 0.9 * samples.size or finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [level / 2.0, 1.0 - level / 2.0])
    return float(low), float(high)


def stretched(estimate: float, low: float, high: float, factor: float) -> tuple[float, float]:
    """The interval with each side ``factor`` times as far from ``estimate``.

    A percentile interval need not contain the estimate; a side that lies beyond it is left where it is.
    """
    if not (np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high)):
        return float("nan"), float("nan")
    return (estimate - factor * max(estimate - low, 0.0) if low <= estimate else low,
            estimate + factor * max(high - estimate, 0.0) if high >= estimate else high)


def measure(prepared: Prepared, *, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Every number: ``{number: {estimate, percentile, yearly}}``, the curves, and the counts."""
    day = DaySums(prepared)
    short, long = BlockSums(day, BLOCK_DAYS), BlockSums(day, YEAR_DAYS)
    point = short.numbers(np.ones((1, short.blocks)))
    draws = short.numbers(resample_weights(short.blocks, resamples, seed))
    each = long.numbers(np.eye(long.blocks))
    out: dict[str, Any] = {"blocks": short.blocks, "years": long.blocks,
                           "tail_cells": float(short.tail["n"].sum()),
                           "curves": {k: point[k][0].tolist() for k in CURVES}}
    for number in NUMBERS:
        out[number] = {"estimate": float(point[number][0]), "percentile": list(interval(draws[number])),
                       "yearly": each[number].tolist()}
    return out


def audit_settings() -> dict[str, Any]:
    """The analysis settings, as recorded with every result."""
    return {"level": LEVEL, "skip_first": SKIP_FIRST, "min_days": MIN_DAYS, "winsor": WINSOR,
            "tail_quantile": TAIL_QUANTILE, "block_days": BLOCK_DAYS, "year_days": YEAR_DAYS,
            "memory_lags": MEMORY_LAGS, "leverage_lags": LEVERAGE_LAGS,
            "variance_ratio_q": VARIANCE_RATIO_Q, "volume_window": VOLUME_WINDOW,
            "volume_min": VOLUME_MIN, "resamples": RESAMPLES, "seed": SEED, "stretch": dict(STRETCH)}


def audit(panel: Panel, *, resamples: int = RESAMPLES, seed: int = SEED,
          stretch: Mapping[str, float] = STRETCH) -> dict[str, Any]:
    """Measure the numbers on ``panel`` and compare each with its published set."""
    prepared = prepare(panel)
    measured = measure(prepared, resamples=resamples, seed=seed)
    readings: dict[str, Any] = {}
    for claim, published in PUBLISHED.items():
        name = NUMBER_OF[claim]
        number = measured[name]
        low, high = stretched(number["estimate"], number["percentile"][0], number["percentile"][1],
                              stretch[name])
        readings[claim] = Reading(claim, number["estimate"], low, high, published,
                                  verdict(low, high, published), {"stretch": stretch[name]}).as_dict()
    return {
        "readings": readings, "numbers": measured,
        "assets": len(prepared.assets), "asset_days": prepared.asset_days,
        "settings": {**audit_settings(), "resamples": resamples, "seed": seed, "stretch": dict(stretch)},
    }


def form_tests(fit: Panel, test: Panel, *, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Does the published *form* describe a later period better than its named rival?

    ``tail``: a power-law tail against an exponential one, each with its one parameter fitted on
    ``fit``; the number is the mean difference in log-likelihood per tail cell of ``test``.
    ``memory``: a power-law decay of the autocorrelation of absolute returns against an
    exponential decay, each with its shape fitted on ``fit`` and its level and scale on ``test``;
    the number is the difference in the share of the variogram's variation over lags explained.
    Positive favours the published form. The interval resamples the blocks of ``test`` and is not
    widened: it is a comparison of two fits on the same resample, not an estimate of a constant.
    """
    fitted, later = BlockSums(DaySums(prepare(fit))), BlockSums(DaySums(prepare(test)))
    weights = np.vstack([np.ones((1, later.blocks)), resample_weights(later.blocks, resamples, seed)])

    n, log, linear = (float(fitted.tail[k].sum()) for k in ("n", "log", "linear"))
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha, rate = np.float64(n) / log, np.float64(n) / linear
        count = weights @ later.tail["n"]
        power = np.log(alpha) - (alpha + 1.0) * (weights @ later.tail["log"]) / count
        exponential = np.log(rate) - rate * (weights @ later.tail["linear"]) / count
    tail = power - exponential

    whole = fitted.numbers(np.ones((1, fitted.blocks)))["variogram"]
    powers, exponentials = power_shapes(), exponential_shapes()
    (i_power,), _ = best_shape(whole, powers)
    (i_exponential,), _ = best_shape(whole, exponentials)
    beta = float(BETA_GRID[i_power]) if i_power >= 0 else float("nan")
    halflife = float(HALFLIFE_GRID[i_exponential]) if i_exponential >= 0 else float("nan")
    rows = later.numbers(weights)["variogram"]
    centred = rows - rows.mean(axis=1, keepdims=True)
    total = (centred ** 2).sum(axis=1)
    if i_power >= 0 and i_exponential >= 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            memory = ((centred @ powers[i_power]) ** 2 - (centred @ exponentials[i_exponential]) ** 2) / total
    else:
        memory = np.full(len(rows), np.nan)

    out: dict[str, Any] = {}
    for claim, values, extra in (("tail", tail, {"alpha_fit": float(alpha), "rate_fit": float(rate)}),
                                 ("memory", memory, {"beta_fit": beta, "halflife_fit": halflife})):
        low, high = interval(values[1:])
        decided = PUBLISHED_FORM if low > 0 else RIVAL_FORM if high < 0 else NOT_DECIDED
        out[claim] = {"estimate": float(values[0]), "low": low, "high": high, "favours": decided, **extra}
    return out
