"""Simulated panels for the second batch of audited laws: the first batch's worlds with five more dials.

Status: audit-only

``apedp_dyn.market.law_audit_more`` audits six further published regularities.
Before its verdicts are believed on real data, each has to be shown to be right
on data whose truth is known. The worlds of the first batch
(``law_audit_worlds``, frozen with its preregistration) have no dial for what
these six are about, so this module takes a panel from one of those worlds and
changes its returns, one mechanism per dial:

- ``skew``: every loss is scaled by ``1 + skew`` and every gain by ``1 - skew``,
  so large losses are more frequent than equally large gains;
- ``drift``: each asset's expected return wanders slowly (half-life
  ``DRIFT_HALFLIFE`` days), so an asset that rose tends to keep rising: momentum;
- ``transitory``: each price carries a component that fades (half-life
  ``TRANSITORY_HALFLIFE`` days), so an asset that rose tends to fall back: reversal;
- ``cascade``: the net move of the last ``CASCADE_DAYS`` days raises today's
  volatility (a negative value lowers it), which is what makes coarse volatility
  predict fine volatility better than the reverse;
- ``flow``: the part of today's order-flow imbalance that today's return does
  not explain is carried into tomorrow's return;
- ``stable``: the returns are replaced by independent draws from a symmetric
  stable law with this index (one common factor, constant volatility). This is
  Mandelbrot's 1963 hypothesis, the rival of aggregational Gaussianity: a sum of
  such returns has the same shape as one of them, at every horizon.

Every panel also gets a taker-buy volume, from which the audit computes the
order-flow imbalance: it follows the same day's return (``FLOW_SAME_DAY``) plus
noise. With every dial at zero the closes and volumes are the first batch's
panel, unchanged.

**Limits.** As for the first batch: these worlds are simpler than a market, and
a test calibrated here can still be wrong on real data in ways they do not
contain. The dials act on each asset separately; none changes how assets move
together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp_dyn.market import law_audit_worlds as lw
from apedp_dyn.market.law_audit import Panel
from apedp_dyn.market.law_audit_more import FlowPanel

DRIFT_HALFLIFE = 14.0       # days: how long an asset's expected return keeps half its value
TRANSITORY_HALFLIFE = 5.0   # days: how long the fading price component keeps half its value
CASCADE_DAYS = 7            # the net move of this many earlier days feeds today's volatility
CASCADE_CAP = 4.0           # ... measured in typical daily moves and capped here
FLOW_SAME_DAY = 0.5         # correlation of the order-flow imbalance with the same day's return
TYPICAL_MOVE = 1.3 * lw.DAILY_SCALE   # a typical daily return in the first batch's worlds
MEAN_ABS_NORMAL = float(np.sqrt(2.0 / np.pi))


@dataclass(frozen=True)
class MoreWorld:
    """A first-batch world and five further dials. The defaults change nothing."""

    name: str
    base: lw.World = field(default_factory=lambda: lw.World("all_hold"))
    skew: float = 0.0
    drift: float = 0.0
    transitory: float = 0.0
    cascade: float = 0.0
    flow: float = 0.0
    stable: float = 0.0

    @property
    def changes_returns(self) -> bool:
        return bool(self.skew or self.drift or self.transitory or self.cascade or self.flow or self.stable)


def persistent(halflife: float, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Independent stationary AR(1) series (columns) of unit variance with the given half-life in days."""
    phi = 0.5 ** (1.0 / halflife)
    noise = rng.standard_normal(shape)
    out: np.ndarray = np.empty(shape)
    out[0] = noise[0]
    step = np.sqrt(1.0 - phi * phi)
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + step * noise[t]
    return out


def returns_of(panel: Panel, mask: np.ndarray) -> np.ndarray:
    """The daily log returns a first-batch panel was built from (zero where there is no bar)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where(mask, np.log(panel.close / 100.0), 0.0)
    previous = np.vstack([np.zeros((1, level.shape[1])), level[:-1]])
    previous_present = np.vstack([np.zeros((1, mask.shape[1]), dtype=bool), mask[:-1]])
    out: np.ndarray = np.where(mask, level - np.where(previous_present, previous, 0.0), 0.0)
    return out


def stable_draws(index: float, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Independent draws from the symmetric stable law with this index and unit scale (Chambers, Mallows, Stuck)."""
    angle = rng.uniform(-np.pi / 2.0, np.pi / 2.0, shape)
    wait = rng.exponential(1.0, shape)
    draws: np.ndarray = (np.sin(index * angle) / np.cos(angle) ** (1.0 / index)
                         * (np.cos(angle - index * angle) / wait) ** ((1.0 - index) / index))
    return draws


def stable_returns(index: float, common: float, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Stable returns with one common factor carrying the share ``common`` of the scale; each asset's are stable too."""
    shared = stable_draws(index, (shape[0], 1), rng)
    own = stable_draws(index, shape, rng)
    out: np.ndarray = 0.5 * lw.DAILY_SCALE * (common ** (1.0 / index) * shared + (1.0 - common) ** (1.0 / index) * own)
    return out


def with_cascade(returns: np.ndarray, strength: float) -> np.ndarray:
    """Each day's return scaled by ``exp(strength * h)``, ``h`` the net move of the days before it.

    ``h`` is the absolute sum of the last ``CASCADE_DAYS`` returns in typical daily moves, over the
    root of that many days, capped at ``CASCADE_CAP``, minus what independent normal days would give.
    """
    out: np.ndarray = returns.copy()
    days = len(out)
    for t in range(1, days):
        recent = out[max(0, t - CASCADE_DAYS):t]
        net = np.abs(recent.sum(axis=0)) / (TYPICAL_MOVE * np.sqrt(CASCADE_DAYS))
        out[t] = out[t] * np.exp(strength * (np.minimum(net, CASCADE_CAP) - MEAN_ABS_NORMAL))
    return out


def simulate(world: MoreWorld, dates: np.ndarray, assets: tuple[str, ...], mask: np.ndarray,
             rng: np.random.Generator) -> FlowPanel:
    """A panel from ``world.base`` with the further dials applied, and its taker-buy volume."""
    base = lw.simulate(world.base, dates, assets, mask, rng)
    shape = (mask.shape[0], mask.shape[1])
    surprise = rng.standard_normal(shape)                 # the part of order flow the return does not explain
    returns = returns_of(base, mask)
    if world.stable:
        returns = np.where(mask, stable_returns(world.stable, world.base.common, shape, rng), 0.0)
    if world.drift:
        returns = returns + world.drift * lw.DAILY_SCALE * persistent(DRIFT_HALFLIFE, shape, rng)
    if world.transitory:
        fading = world.transitory * lw.DAILY_SCALE * persistent(TRANSITORY_HALFLIFE, shape, rng)
        returns = returns + np.diff(fading, axis=0, prepend=np.zeros((1, shape[1])))
    if world.flow:
        returns[1:] = returns[1:] + world.flow * lw.DAILY_SCALE * surprise[:-1]
    if world.skew:
        returns = returns * np.where(returns < 0, 1.0 + world.skew, 1.0 - world.skew)
    if world.cascade:
        returns = with_cascade(np.where(mask, returns, 0.0), world.cascade)
    panel = base
    if world.changes_returns:
        close: np.ndarray = np.where(mask, 100.0 * np.exp(np.cumsum(np.where(mask, returns, 0.0), axis=0)), np.nan)
        panel = Panel(base.dates, base.assets, close, base.volume)
    imbalance = np.clip((FLOW_SAME_DAY * returns / TYPICAL_MOVE
                         + np.sqrt(1.0 - FLOW_SAME_DAY ** 2) * surprise) / 3.0, -0.95, 0.95)
    taker: np.ndarray = np.where(mask, panel.volume * (1.0 + imbalance) / 2.0, np.nan)
    return FlowPanel(panel, taker)


def _base(name: str, **dials: Any) -> lw.World:
    return lw.World(name, **dials)


_NO_VOLATILITY: dict[str, Any] = {"memory": "none", "vol_scale": 0.0, "leverage": 0.0, "volume_link": 0.0}

#: Worlds in which every number keeps one long-run value from panel to panel: none of the dials
#: whose effect depends on the level of volatility is on. The widening factors are measured here.
CALIBRATION_WORLDS: tuple[MoreWorld, ...] = (
    MoreWorld("quiet"),
    MoreWorld("quiet_tail_2.2", base=_base("tail_2.2", tail=2.2)),
    MoreWorld("quiet_tail_4", base=_base("tail_4", tail=4.0)),
    MoreWorld("quiet_beta_0.2", base=_base("memory_beta_0.2", beta=0.2)),
    MoreWorld("quiet_beta_0.4", base=_base("memory_beta_0.4", beta=0.4)),
    MoreWorld("stable_1.7", stable=1.7),
    MoreWorld("normal", base=_base("normal", tail=float("inf"), **_NO_VOLATILITY)),
)

#: The worlds of the power check. ``all_hold`` has every claim true; the ``inverse`` worlds have
#: asymmetry, cascade, momentum and flow reversed; the ``quiet`` worlds have those four at zero;
#: the last five take away what ``gaussianity``, ``conditional`` or ``still_heavy`` rest on.
WORLDS: tuple[MoreWorld, ...] = (
    MoreWorld("all_hold", skew=0.1, drift=0.06, cascade=0.2, flow=0.03),
    MoreWorld("all_hold_weak", skew=0.05, drift=0.03, cascade=0.1, flow=0.015),
    MoreWorld("all_hold_tail_2.2", base=_base("tail_2.2", tail=2.2), skew=0.1, drift=0.06, cascade=0.2, flow=0.03),
    MoreWorld("all_hold_tail_4", base=_base("tail_4", tail=4.0), skew=0.1, drift=0.06, cascade=0.2, flow=0.03),
    MoreWorld("momentum_as_published", drift=0.1),
    MoreWorld("all_inverse", skew=-0.1, transitory=1.0, cascade=-0.2, flow=-0.03),
    MoreWorld("all_inverse_weak", skew=-0.05, transitory=0.5, cascade=-0.1, flow=-0.015),
    MoreWorld("quiet"),
    MoreWorld("quiet_no_leverage", base=_base("no_leverage", leverage=0.0)),
    MoreWorld("quiet_tail_1.7", base=_base("tail_1.7", tail=1.7)),
    MoreWorld("quiet_exponential_memory", base=_base("memory_exponential", memory="exponential")),
    MoreWorld("stable_1.7", stable=1.7),
    MoreWorld("stable_1.9", stable=1.9),
    MoreWorld("normal", base=_base("normal", tail=float("inf"), **_NO_VOLATILITY)),
    MoreWorld("constant_volatility", base=_base("constant_volatility", **_NO_VOLATILITY)),
    MoreWorld("normal_shocks", base=_base("normal_shocks", tail=float("inf"))),
)


def holds(world: MoreWorld, claim: str) -> bool:
    """Whether ``claim`` (a number above zero) is true in ``world``, by the dial that makes it."""
    moving = bool(world.base.vol_scale > 0) and not world.stable
    heavy_shocks = bool(world.stable) or bool(np.isfinite(world.base.tail))
    if claim == "asymmetry":
        return world.skew > 0
    if claim == "gaussianity":      # a stable law keeps its shape; a normal one has none left to gain
        return not world.stable and (moving or heavy_shocks)
    if claim == "conditional":
        return moving
    if claim == "still_heavy":
        return moving or heavy_shocks
    if claim == "cascade":
        return world.cascade > 0
    if claim == "momentum":
        return world.drift > 0 and not world.transitory
    if claim == "flow":
        return world.flow > 0
    raise ValueError(f"unknown claim {claim!r}")
