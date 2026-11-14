"""Universe integrity: who is in a multi-asset panel, decided when, on which bars.

Two defects make a backtest on a panel optimistic without any line of strategy
code being wrong, and both are properties of the *data file*:

* **Hindsight selection.** The asset list was drawn up after the sample period
  began -- "the top 45 tokens", chosen in 2026, for prices from 2021. Every name
  on such a list is one that was still worth listing years later; the ones that
  collapsed or were delisted in between are not there to lose money on.

* **Forward-filled delistings.** An asset stops trading and the panel builder
  fills the gap with its last bar. The file then shows a live instrument with a
  constant price, constant volume and constant order flow, for as long as the
  fill runs. Those bars never happened.

This module measures both (:func:`audit_universe`), removes the second
(:func:`mask_stale_bars`), and builds the alternative to the first: a universe
whose membership on each date depends only on what was observable before it
(:func:`point_in_time_membership`). It cannot conjure the assets a hindsight
list left out -- a point-in-time universe is only as honest as the candidate
pool it is drawn from, which must include the names that later died.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: A live, liquid instrument does not print the identical close *and* the
#: identical volume on consecutive bars. Five in a row is a fill, not a market.
STALE_RUN_MIN_BARS = 5

DEFAULT_DATE_COLUMN = "observation_date"
DEFAULT_CLOSE_SUFFIX = "_CLOSE"
DEFAULT_VOLUME_SUFFIX = "_VOL"


@dataclass(frozen=True)
class StaleRun:
    """A stretch of bars that repeat the bar before them."""

    asset: str
    last_real_bar: str
    first_stale_bar: str
    last_stale_bar: str
    bars: int
    reaches_end: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "last_real_bar": self.last_real_bar,
            "first_stale_bar": self.first_stale_bar,
            "last_stale_bar": self.last_stale_bar,
            "bars": self.bars,
            "reaches_end": self.reaches_end,
        }


@dataclass(frozen=True)
class UniverseAudit:
    """What is known about a panel's universe, and whether it supports a claim.

    ``hindsight_selected`` is ``None`` when no selection date is on record: an
    unknown selection basis is not evidence of a clean one, so it does not make
    the universe fit either.
    """

    n_assets: int
    data_start: str
    data_end: str
    selection_basis: str
    selected_as_of: str | None
    hindsight_selected: bool | None
    stale_runs: tuple[StaleRun, ...] = ()
    assets_that_stop_trading: tuple[str, ...] = ()
    applicable: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reasons(self) -> list[str]:
        """Why the universe does not support an economic claim (empty if it does)."""
        if not self.applicable:
            return []
        found: list[str] = []
        if self.hindsight_selected is True:
            found.append(
                f"universe selected as of {self.selected_as_of}, after the data begins "
                f"({self.data_start}): assets that failed in between are absent"
            )
        elif self.hindsight_selected is None:
            found.append("no selection date is on record for this universe")
        if self.stale_runs:
            assets = sorted({run.asset for run in self.stale_runs})
            found.append(
                f"{len(assets)} asset(s) carry forward-filled bars that never traded "
                f"({', '.join(assets)})"
            )
        return found

    @property
    def fit_for_economic_claims(self) -> bool:
        return not self.reasons

    @classmethod
    def single_instrument(cls, name: str, data_start: str = "", data_end: str = "") -> UniverseAudit:
        """A series of one named instrument: there is no universe to select."""
        return cls(
            n_assets=1,
            data_start=data_start,
            data_end=data_end,
            selection_basis="single_instrument",
            selected_as_of=None,
            hindsight_selected=False,
            applicable=False,
            notes=(f"single instrument: {name}",),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_assets": self.n_assets,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "selection_basis": self.selection_basis,
            "selected_as_of": self.selected_as_of,
            "hindsight_selected": self.hindsight_selected,
            "stale_runs": [run.to_dict() for run in self.stale_runs],
            "assets_that_stop_trading": list(self.assets_that_stop_trading),
            "applicable": self.applicable,
            "fit_for_economic_claims": self.fit_for_economic_claims,
            "reasons": self.reasons,
            "notes": list(self.notes),
        }


def panel_assets(panel: pd.DataFrame, close_suffix: str = DEFAULT_CLOSE_SUFFIX) -> list[str]:
    """Asset names in a wide panel, in column order (``BTC`` for ``BTC_CLOSE``)."""
    return [str(col)[: -len(close_suffix)] for col in panel.columns if str(col).endswith(close_suffix)]


def _bar_labels(panel: pd.DataFrame, date_column: str) -> list[str]:
    if date_column in panel.columns:
        return [str(value) for value in panel[date_column].tolist()]
    return [str(value) for value in panel.index.tolist()]


def find_stale_runs(
    panel: pd.DataFrame,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    close_suffix: str = DEFAULT_CLOSE_SUFFIX,
    volume_suffix: str = DEFAULT_VOLUME_SUFFIX,
    min_bars: int = STALE_RUN_MIN_BARS,
) -> list[StaleRun]:
    """Runs of at least ``min_bars`` bars that repeat the previous bar exactly.

    A bar is stale when its close equals the previous close and, if the panel
    has a volume column for the asset, its volume equals the previous volume
    too. Requiring both keeps a pegged or thinly quoted instrument, whose price
    can legitimately repeat, from being mistaken for a filled gap.
    """
    labels = _bar_labels(panel, date_column)
    runs: list[StaleRun] = []
    for asset in panel_assets(panel, close_suffix):
        close = panel[f"{asset}{close_suffix}"].to_numpy(dtype=float)
        repeats: np.ndarray = np.zeros(len(close), dtype=bool)
        repeats[1:] = close[1:] == close[:-1]
        volume_column = f"{asset}{volume_suffix}"
        if volume_column in panel.columns:
            volume = panel[volume_column].to_numpy(dtype=float)
            repeats[1:] &= volume[1:] == volume[:-1]
        start: int | None = None
        for i in range(len(close) + 1):
            is_stale = i < len(close) and bool(repeats[i])
            if is_stale and start is None:
                start = i
            elif not is_stale and start is not None:
                if i - start >= min_bars:
                    runs.append(
                        StaleRun(
                            asset=asset,
                            last_real_bar=labels[start - 1],
                            first_stale_bar=labels[start],
                            last_stale_bar=labels[i - 1],
                            bars=i - start,
                            reaches_end=i == len(close),
                        )
                    )
                start = None
    return runs


def mask_stale_bars(
    panel: pd.DataFrame,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    close_suffix: str = DEFAULT_CLOSE_SUFFIX,
    volume_suffix: str = DEFAULT_VOLUME_SUFFIX,
    min_bars: int = STALE_RUN_MIN_BARS,
) -> pd.DataFrame:
    """A copy of ``panel`` with every stale bar set to NaN in all of the asset's columns.

    After this an asset that stopped trading *ends*: its returns are missing,
    not zero, and anything computed across assets has to decide what to do
    about that instead of quietly averaging in a flat line.
    """
    labels = _bar_labels(panel, date_column)
    position = {label: i for i, label in enumerate(labels)}
    masked = panel.copy()
    for run in find_stale_runs(
        panel, date_column=date_column, close_suffix=close_suffix,
        volume_suffix=volume_suffix, min_bars=min_bars,
    ):
        rows = slice(position[run.first_stale_bar], position[run.last_stale_bar] + 1)
        columns = [col for col in panel.columns if str(col).startswith(f"{run.asset}_")]
        for column in columns:
            values = masked[column].to_numpy(dtype=float, copy=True)
            values[rows] = np.nan
            masked[column] = values
    return masked


def audit_universe(
    panel: pd.DataFrame,
    *,
    selected_as_of: str | None,
    selection_basis: str = "static_list",
    date_column: str = DEFAULT_DATE_COLUMN,
    close_suffix: str = DEFAULT_CLOSE_SUFFIX,
    volume_suffix: str = DEFAULT_VOLUME_SUFFIX,
) -> UniverseAudit:
    """Audit a wide panel's universe.

    ``selected_as_of`` is the date the asset list was fixed (ISO ``YYYY-MM-DD``),
    or ``None`` if it was never recorded. A list fixed on or before the first
    bar is not hindsight; one fixed later is. For a universe built by
    :func:`point_in_time_membership`, pass ``selection_basis="point_in_time"``
    and the first bar's date.
    """
    labels = _bar_labels(panel, date_column)
    assets = panel_assets(panel, close_suffix)
    data_start, data_end = (labels[0], labels[-1]) if labels else ("", "")
    hindsight: bool | None
    if selected_as_of is None:
        hindsight = None
    else:
        hindsight = str(selected_as_of)[:10] > str(data_start)[:10]
    stops: list[str] = []
    for asset in assets:
        close = panel[f"{asset}{close_suffix}"].to_numpy(dtype=float)
        if len(close) and np.isnan(close[-1]) and np.isfinite(close).any():
            stops.append(asset)
    stale = find_stale_runs(
        panel, date_column=date_column, close_suffix=close_suffix, volume_suffix=volume_suffix
    )
    notes: list[str] = []
    if len(assets) > 1 and not stops and not any(run.reaches_end for run in stale):
        notes.append(
            "no asset stops trading inside the window; over a multi-year sample that is what a "
            "list of survivors looks like"
        )
    return UniverseAudit(
        n_assets=len(assets),
        data_start=str(data_start),
        data_end=str(data_end),
        selection_basis=selection_basis,
        selected_as_of=selected_as_of,
        hindsight_selected=hindsight,
        stale_runs=tuple(stale),
        assets_that_stop_trading=tuple(stops),
        notes=tuple(notes),
    )


def point_in_time_membership(
    panel: pd.DataFrame,
    *,
    top_n: int,
    lookback_bars: int = 60,
    rebalance_every: int = 21,
    date_column: str = DEFAULT_DATE_COLUMN,
    close_suffix: str = DEFAULT_CLOSE_SUFFIX,
    volume_suffix: str = DEFAULT_VOLUME_SUFFIX,
) -> pd.DataFrame:
    """Universe membership that uses only what was observable before each bar.

    Every ``rebalance_every`` bars the assets are ranked by mean traded value
    (close x volume) over the ``lookback_bars`` bars *strictly before* the
    rebalance bar, and the ``top_n`` become the universe until the next
    rebalance. An asset with a missing or stale bar in the lookback window is
    not eligible, and a member whose bar is missing or stale drops out on that
    bar: a universe cannot hold what is not trading.

    Returns a boolean frame, one row per bar and one column per asset. Rows
    before the first full lookback window have no members.

    Give this the widest candidate pool available, with stale bars already
    masked and delisted assets present up to their last real bar. Ranking a
    list of survivors point-in-time removes selection *within* the list; it does
    not put back the assets the list never contained.
    """
    if top_n < 1 or lookback_bars < 1 or rebalance_every < 1:
        raise ValueError("top_n, lookback_bars and rebalance_every must all be at least 1")
    clean = mask_stale_bars(
        panel, date_column=date_column, close_suffix=close_suffix, volume_suffix=volume_suffix
    )
    assets = panel_assets(clean, close_suffix)
    missing_volume = [a for a in assets if f"{a}{volume_suffix}" not in clean.columns]
    if missing_volume:
        raise ValueError(f"no volume column for: {', '.join(missing_volume[:5])}")
    close = clean[[f"{a}{close_suffix}" for a in assets]].to_numpy(dtype=float)
    traded_value = close * clean[[f"{a}{volume_suffix}" for a in assets]].to_numpy(dtype=float)
    n_bars = len(clean)
    members: np.ndarray = np.zeros((n_bars, len(assets)), dtype=bool)
    current: np.ndarray = np.zeros(len(assets), dtype=bool)
    for t in range(lookback_bars, n_bars):
        if (t - lookback_bars) % rebalance_every == 0:
            window = traded_value[t - lookback_bars : t]
            eligible = np.isfinite(window).all(axis=0)
            score = np.where(eligible, window.mean(axis=0), -np.inf)
            order = np.argsort(-score, kind="stable")[:top_n]
            current = np.zeros(len(assets), dtype=bool)
            current[order[np.isfinite(score[order])]] = True
        members[t] = current & np.isfinite(close[t])
    index = _bar_labels(panel, date_column)
    return pd.DataFrame(members, index=pd.Index(index, name=date_column), columns=assets)


def load_universe_selection(provenance_path: Path | str, filename: str) -> dict[str, Any] | None:
    """The ``universe_selection`` block recorded for a data file, if any."""
    payload = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    entry = payload.get(filename)
    if not isinstance(entry, dict):
        return None
    selection = entry.get("universe_selection")
    return dict(selection) if isinstance(selection, dict) else None


__all__ = [
    "STALE_RUN_MIN_BARS",
    "StaleRun",
    "UniverseAudit",
    "audit_universe",
    "find_stale_runs",
    "load_universe_selection",
    "mask_stale_bars",
    "panel_assets",
    "point_in_time_membership",
]
