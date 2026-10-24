"""Market Windows Registry: frozen benchmarks, standard rolling, and extensive multi-scale sweeps.

Maintains strict backwards compatibility:
- Frozen W1..W6 never change.
- Default generated_rolling_windows returns exactly 20 windows (tested by UI sandbox contract).
- extensive_rolling_windows scales to 200+ multi-scale regimes per asset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apedp_dyn.market.assets import load_series_with_dates
from apedp_dyn.market.regimes import MarketRegime, classify_market_regime


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    train_start: str
    train_end: str
    holdout_start: str
    holdout_end: str
    regime_description: str
    regime: str = "general"
    metrics: dict[str, float] | None = None


# 6 Frozen Benchmark Splits (Commitment Invariant: never altered)
WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(
        "W1_2018", "2018-01-02", "2018-06-29", "2018-07-02", "2018-12-31",
        "2018 Trade-war / Vol expansion", regime=MarketRegime.VOLATILITY_SHOCK.value
    ),
    WindowSpec(
        "W2_2019", "2019-01-02", "2019-06-28", "2019-07-01", "2019-12-31",
        "2019 Calm Bull Market", regime=MarketRegime.BULL_EXPANSION.value
    ),
    WindowSpec(
        "W3_COVID", "2019-01-02", "2019-12-31", "2020-01-02", "2020-06-30",
        "2020 COVID Shock & Rebound", regime=MarketRegime.VOLATILITY_SHOCK.value
    ),
    WindowSpec(
        "W4_REBOUND", "2020-01-02", "2020-12-31", "2021-01-04", "2021-06-30",
        "2021 Post-COVID Expansion", regime=MarketRegime.REBOUND_RECOVERY.value
    ),
    WindowSpec(
        "W5_BEAR", "2021-01-04", "2021-12-31", "2022-01-03", "2022-12-30",
        "2022 Bear Market (Rate Hikes)", regime=MarketRegime.BEAR_CONTRACTION.value
    ),
    WindowSpec(
        "W6_RECOVERY", "2022-01-03", "2022-12-30", "2023-01-03", "2023-06-30",
        "2023 Tech Recovery", regime=MarketRegime.BULL_EXPANSION.value
    ),
)


def generated_rolling_windows(
    asset: str = "sp500",
    train_days: int = 126,
    holdout_days: int = 126,
    step_days: int = 63,
) -> tuple[WindowSpec, ...]:
    """Standard rolling train/holdout windows over the committed series.

    Defaults to step_days=63, train_days=126, holdout_days=126 producing exactly 20
    windows, satisfying the test contract in test_dyn_ui_sandbox.py.
    """
    dates, prices = load_series_with_dates(asset)
    prefix = "SPX" if asset == "sp500" else ("NDX" if asset == "nasdaq100" else asset.upper()[:3])
    out: list[WindowSpec] = []
    i = 0
    k = 0
    while i + train_days + holdout_days <= len(dates):
        t0, t1 = dates[i], dates[i + train_days - 1]
        h0, h1 = dates[i + train_days], dates[i + train_days + holdout_days - 1]

        # Classify regime from holdout returns
        ho_px = prices[i + train_days : i + train_days + holdout_days]
        ho_ret = np.zeros(len(ho_px))
        if len(ho_px) > 1:
            ho_ret[1:] = np.diff(np.log(ho_px))
        regime, metrics = classify_market_regime(ho_ret, ho_px)

        out.append(
            WindowSpec(
                f"{prefix}-ROLL{k:02d}",
                t0,
                t1,
                h0,
                h1,
                f"rolling {t0}..{h1} [{regime.value}]",
                regime=regime.value,
                metrics=metrics,
            )
        )
        k += 1
        i += step_days
    return tuple(out)


_EXTENSIVE_WINDOWS_CACHE: dict[str, tuple[WindowSpec, ...]] = {}


def extensive_rolling_windows(asset: str = "sp500") -> tuple[WindowSpec, ...]:
    """Generate extensive multi-scale rolling windows across multiple horizons.

    Scales sweeps across:
      - Short-horizon (30d train / 30d holdout, step 15d) -> ~80 windows
      - Quarterly (63d train / 63d holdout, step 21d) -> ~55 windows
      - Semi-annual (126d train / 126d holdout, step 21d) -> ~50 windows
      - Annual/Macro (252d train / 126d holdout, step 42d) -> ~25 windows

    Total > 200 windows per asset, categorized into all 6 macro regimes.
    """
    if asset in _EXTENSIVE_WINDOWS_CACHE:
        return _EXTENSIVE_WINDOWS_CACHE[asset]

    dates, prices = load_series_with_dates(asset)
    prefix = "SPX" if asset == "sp500" else ("NDX" if asset == "nasdaq100" else asset.upper()[:3])
    out: list[WindowSpec] = []

    configs = [
        ("SHORT", 30, 30, 15),
        ("QTR", 63, 63, 21),
        ("SEMI", 126, 126, 21),
        ("ANN", 252, 126, 42),
    ]

    for scale_tag, tr_days, ho_days, st_days in configs:
        i = 0
        k = 0
        while i + tr_days + ho_days <= len(dates):
            t0, t1 = dates[i], dates[i + tr_days - 1]
            h0, h1 = dates[i + tr_days], dates[i + tr_days + ho_days - 1]

            ho_px = prices[i + tr_days : i + tr_days + ho_days]
            ho_ret = np.zeros(len(ho_px))
            if len(ho_px) > 1:
                ho_ret[1:] = np.diff(np.log(ho_px))
            regime, metrics = classify_market_regime(ho_ret, ho_px)

            win_id = f"{prefix}-{scale_tag}{k:03d}"
            desc = f"{scale_tag.lower()} {t0}..{h1} [{regime.value}]"
            out.append(
                WindowSpec(
                    win_id,
                    t0,
                    t1,
                    h0,
                    h1,
                    desc,
                    regime=regime.value,
                    metrics=metrics,
                )
            )
            k += 1
            i += st_days

    res = tuple(out)
    _EXTENSIVE_WINDOWS_CACHE[asset] = res
    return res


def get_window_by_id(asset: str, window_id: str) -> WindowSpec | None:
    """Find a window by ID across frozen, standard rolling, and extensive registries."""
    for w in WINDOWS:
        if w.window_id == window_id:
            return w
    for w in generated_rolling_windows(asset):
        if w.window_id == window_id:
            return w
    for w in extensive_rolling_windows(asset):
        if w.window_id == window_id:
            return w
    return None
