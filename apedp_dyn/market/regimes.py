"""Real Market Regimes: scientific taxonomy and causal statistical classification.

Defines 6 canonical macro-financial regimes for evaluating open discovery:
    1. BULL_EXPANSION: Upward drift, low realized volatility, steady compounding.
    2. BEAR_CONTRACTION: Sustained downward drift, elevated volatility, negative drift.
    3. VOLATILITY_SHOCK: Extreme rapid drawdown and volatility explosion (e.g. COVID crash).
    4. REBOUND_RECOVERY: Sharp positive bounce from market troughs.
    5. SIDEWAYS_CONSOLIDATION: Range-bound oscillation, low directional signal.
    6. HIGH_VOLATILITY_CHOP: Elevated volatility without directional persistence.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class MarketRegime(str, Enum):
    """Canonical macro-financial market regimes."""

    BULL_EXPANSION = "bull_expansion"
    BEAR_CONTRACTION = "bear_contraction"
    VOLATILITY_SHOCK = "volatility_shock"
    REBOUND_RECOVERY = "rebound_recovery"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"
    HIGH_VOLATILITY_CHOP = "high_volatility_chop"


REGIME_METADATA: dict[MarketRegime, dict[str, Any]] = {
    MarketRegime.BULL_EXPANSION: {
        "name": "Bull Market Expansion",
        "description": "Persistent positive drift with low-to-moderate realized volatility.",
        "expected_volatility": "Low (< 16% annualized)",
    },
    MarketRegime.BEAR_CONTRACTION: {
        "name": "Bear Market Contraction",
        "description": "Persistent negative drift with elevated risk premia and rate shocks.",
        "expected_volatility": "Elevated (> 20% annualized)",
    },
    MarketRegime.VOLATILITY_SHOCK: {
        "name": "Extreme Volatility Shock",
        "description": "Sudden structural liquidity break and massive volatility spike.",
        "expected_volatility": "Extreme (> 35% annualized)",
    },
    MarketRegime.REBOUND_RECOVERY: {
        "name": "Post-Crash Rebound Recovery",
        "description": "Sharp momentum recovery following a crisis trough.",
        "expected_volatility": "Moderate-High (18-28% annualized)",
    },
    MarketRegime.SIDEWAYS_CONSOLIDATION: {
        "name": "Sideways Consolidation",
        "description": "Mean-reverting, range-bound price action with low directional trend.",
        "expected_volatility": "Low-Moderate (10-18% annualized)",
    },
    MarketRegime.HIGH_VOLATILITY_CHOP: {
        "name": "High-Volatility Chop",
        "description": "Turbulent whipsaw markets without coherent multi-month trend.",
        "expected_volatility": "High (22-35% annualized)",
    },
}


def classify_market_regime(
    returns: np.ndarray,
    prices: np.ndarray,
) -> tuple[MarketRegime, dict[str, float]]:
    """Classify a window into its dominant macro-financial regime using causal metrics."""
    if len(returns) == 0 or len(prices) == 0:
        return MarketRegime.SIDEWAYS_CONSOLIDATION, {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
        }

    # Annualized metrics assuming 252 trading days per year
    ann_ret = float(252.0 * np.mean(returns))
    ann_vol = float(np.sqrt(252.0) * np.std(returns))

    # Peak-to-trough max drawdown within the window
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (cum_max - prices) / np.maximum(cum_max, 1e-8)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    metrics = {
        "annualized_return": round(ann_ret, 4),
        "annualized_volatility": round(ann_vol, 4),
        "max_drawdown": round(max_dd, 4),
    }

    if max_dd > 0.22 or (ann_vol > 0.32 and ann_ret < -0.10):
        return MarketRegime.VOLATILITY_SHOCK, metrics
    if ann_ret > 0.18 and ann_vol < 0.22:
        return MarketRegime.BULL_EXPANSION, metrics
    if ann_ret < -0.12:
        return MarketRegime.BEAR_CONTRACTION, metrics
    if ann_ret > 0.14 and max_dd < 0.12:
        return MarketRegime.REBOUND_RECOVERY, metrics
    if ann_vol > 0.24:
        return MarketRegime.HIGH_VOLATILITY_CHOP, metrics
    return MarketRegime.SIDEWAYS_CONSOLIDATION, metrics
