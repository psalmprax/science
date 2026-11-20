"""Simulated return panels in which each audited law holds, or fails by a known amount.

Status: audit-only

**Why.** ``apedp_dyn.market.law_audit`` gives each published law a verdict from
a bootstrap interval. Before that is believed on real data it has to be shown,
on data whose truth is known, how often the verdict is wrong: how often a law
that holds is failed, and how often one that is broken is caught. A test that
cannot catch a violation of realistic size has nothing to say, and that has to
be known before the real panel is opened, not after.

**What.** ``simulate`` builds a panel of closes and volumes with the real
panel's shape (assets that start late and stop early), from a model with one
dial per law:

- ``tail``: the shocks are Student-t with this many degrees of freedom, which is
  the tail index of the returns;
- ``autocorrelation``: a moving average of the shocks, so returns are not a
  random walk;
- ``memory`` / ``beta`` / ``halflife``: log-volatility is a Gaussian series whose
  autocorrelation decays as a power law (``beta``), exponentially, or not at all;
- ``leverage``: how strongly a common shock is tied to the next day's surprise in
  the common volatility, negatively (positive value) or positively (negative);
- ``volume_link``: how much log-volume follows log-volatility.

Assets share one common shock and one common volatility component, as real
ones do; that dependence is what the block resampling has to cope with. The
leverage dial acts through the part of tomorrow's volatility that today's
volatility does not predict, so it leaves the autocorrelation of volatility
exactly as the memory dial sets it.

**Limits.** These worlds are simpler than a market: no jumps on listing days,
no regime changes, one common factor. A test calibrated here can still be
wrong on real data in ways these worlds do not contain.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from apedp_dyn.market.law_audit import Panel

DAILY_SCALE = 0.04        # typical daily standard deviation of a return
COMMON_VOLATILITY = 0.5   # share of the log-volatility variance that all assets share
VOLUME_NOISE = 0.7        # standard deviation of the part of log-volume unrelated to volatility
POWER_LEVEL = 0.7         # lag-1 autocorrelation of log-volatility in the power-law worlds


@dataclass(frozen=True)
class World:
    """One simulated market. The defaults are a world in which all five laws hold."""

    name: str
    tail: float = 3.0
    autocorrelation: float = 0.0
    memory: str = "power"
    beta: float = 0.3
    halflife: float = 5.0
    vol_scale: float = 0.7
    leverage: float = 0.5
    volume_link: float = 1.0
    common: float = 0.4


def volatility_autocorrelation(world: World, days: int) -> np.ndarray:
    """Autocorrelation of log-volatility at lags 0 .. days - 1."""
    lag: np.ndarray = np.arange(days, dtype=float)
    if world.memory == "power":
        # exactly POWER_LEVEL * lag ** -beta from lag 1 on. Convex and decreasing from lag 0 for
        # every beta used here (POWER_LEVEL <= 1 / (2 - 2 ** -beta)), hence a valid autocorrelation.
        rho: np.ndarray = np.ones(days)
        rho[1:] = POWER_LEVEL * lag[1:] ** (-world.beta)
        return rho
    if world.memory == "exponential":
        decay: np.ndarray = 0.5 ** (lag / world.halflife)
        return decay
    if world.memory == "none":
        white: np.ndarray = (lag == 0).astype(float)
        return white
    raise ValueError(f"unknown memory {world.memory!r}")


def gaussian_series(rho: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """``size`` independent stationary Gaussian series (columns) with autocorrelation ``rho``.

    Circulant embedding; exact when the embedded spectrum is non-negative, which holds for the
    three shapes above. A negative eigenvalue from rounding is set to zero.
    """
    row = np.concatenate([rho, rho[-2:0:-1]])
    spectrum = np.clip(np.fft.fft(row).real, 0.0, None)
    noise = rng.standard_normal((len(row), size)) + 1j * rng.standard_normal((len(row), size))
    series: np.ndarray = np.fft.fft(np.sqrt(spectrum / len(row))[:, None] * noise, axis=0).real[: len(rho)]
    return series


def surprises(rho: np.ndarray, series: np.ndarray) -> np.ndarray:
    """What each value of ``series`` adds to its best linear prediction from the earlier values.

    Durbin-Levinson recursion for a series of unit variance with autocorrelation ``rho``. The
    result is white noise of unit variance: ``series`` is a fixed linear function of it.
    """
    out: np.ndarray = np.empty(len(series))
    out[0] = series[0]
    weights: np.ndarray = np.zeros(0)
    variance = 1.0
    for t in range(1, len(series)):
        partial = (rho[t] - weights @ rho[t - 1:0:-1]) / variance
        weights = np.concatenate([weights - partial * weights[::-1], [partial]])
        variance *= max(1.0 - partial * partial, 1e-12)
        out[t] = (series[t] - weights @ series[t - 1::-1]) / np.sqrt(variance)
    return out


def shocks(tail: float, normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Student-t shocks with ``tail`` degrees of freedom from standard normal draws.

    A normal draw over the root of an independent chi-square, scaled to unit variance where the
    variance exists; the sign and rank order of ``normal`` along each day are kept.
    """
    if not np.isfinite(tail):
        return normal
    scale = np.sqrt((tail - 2.0 if tail > 2 else tail) / rng.chisquare(tail, normal.shape))
    draw: np.ndarray = normal * scale
    return draw


def presence(first: Mapping[str, str], last: Mapping[str, str], start: str,
             end: str) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Dates, assets and a (days, assets) mask: an asset is present from its first to its last bar."""
    dates = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"))
    assets = tuple(sorted(a for a in first if np.datetime64(first[a]) <= dates[-1]
                          and np.datetime64(last[a]) >= dates[0]))
    begin = np.array([np.datetime64(first[a]) for a in assets])
    finish = np.array([np.datetime64(last[a]) for a in assets])
    mask: np.ndarray = (dates[:, None] >= begin[None, :]) & (dates[:, None] <= finish[None, :])
    return dates, assets, mask


def simulate(world: World, dates: np.ndarray, assets: tuple[str, ...], mask: np.ndarray,
             rng: np.random.Generator) -> Panel:
    """A panel of closes and volumes from ``world``, with bars only where ``mask`` is true."""
    days, count = mask.shape
    rho = volatility_autocorrelation(world, days)
    shared = gaussian_series(rho, 1, rng)
    log_vol = world.vol_scale * (np.sqrt(COMMON_VOLATILITY) * shared
                                 + np.sqrt(1.0 - COMMON_VOLATILITY) * gaussian_series(rho, count, rng))

    # the common shock of day t is drawn[t + 1]; it is tied to the surprise in the common
    # volatility of day t + 1, which nothing known on day t predicts
    drawn = rng.standard_normal((days + 1, 1))
    if world.leverage and world.vol_scale:
        surprise = surprises(rho, shared[:, 0])
        drawn[1:days, 0] = (-world.leverage * surprise[1:]
                            + np.sqrt(1.0 - world.leverage ** 2) * drawn[1:days, 0])
    common = shocks(world.tail, drawn, rng)
    own = shocks(world.tail, rng.standard_normal((days + 1, count)), rng)
    theta = world.autocorrelation
    shock = (np.sqrt(world.common) * (common[1:] + theta * common[:-1])
             + np.sqrt(1.0 - world.common) * (own[1:] + theta * own[:-1])) / np.sqrt(1.0 + theta * theta)

    returns = DAILY_SCALE * np.exp(log_vol) * shock
    log_price = np.cumsum(np.where(mask, returns, 0.0), axis=0)
    close: np.ndarray = np.where(mask, 100.0 * np.exp(log_price), np.nan)
    log_volume = 10.0 + world.volume_link * log_vol + VOLUME_NOISE * rng.standard_normal((days, count))
    volume: np.ndarray = np.where(mask, np.exp(log_volume), np.nan)
    return Panel(dates, assets, close, volume)


#: The worlds of the power check. In ``all_hold`` every law holds at the middle of its published
#: range; ``tail_2.2``, ``tail_4``, ``memory_beta_0.2`` and ``memory_beta_0.4`` keep every law but sit
#: towards an edge; each of the others breaks one law (``constant_volatility`` breaks three).
WORLDS: tuple[World, ...] = (
    World("all_hold"),
    World("tail_2.2", tail=2.2),
    World("tail_1.7", tail=1.7),
    World("tail_4", tail=4.0),
    World("tail_6", tail=6.0),
    World("momentum_0.03", autocorrelation=0.03),
    World("reversal_0.03", autocorrelation=-0.03),
    World("momentum_0.06", autocorrelation=0.06),
    World("reversal_0.06", autocorrelation=-0.06),
    World("memory_beta_0.2", beta=0.2),
    World("memory_beta_0.4", beta=0.4),
    World("memory_beta_0.1", beta=0.1),
    World("memory_beta_0.7", beta=0.7),
    World("memory_exponential", memory="exponential"),
    World("no_leverage", leverage=0.0),
    World("inverse_leverage", leverage=-0.5),
    World("no_volume_link", volume_link=0.0),
    World("constant_volatility", memory="none", vol_scale=0.0, leverage=0.0, volume_link=0.0),
)
