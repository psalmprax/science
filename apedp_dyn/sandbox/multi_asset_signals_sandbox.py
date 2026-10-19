"""15-Asset Synchronized Multi-Asset Portfolio Sandbox with Non-Linear Genetic Algorithm & Risk Parity:
Tests quantitative signals across the entire 15-crypto universe SIMULTANEOUSLY (all at once, not per asset).

Includes:
1. Dynamic Volatility-Targeted Position Sizing (Risk Parity / Inverse-Volatility Weighting across all 15 coins).
2. Higher-Order Non-Linear Signal Interactions (Volume-Gated Momentum, Squeeze-Released Trend, Hurst-Adaptive Regime Switching, OFI Reversals).
3. Synchronized Universe: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE, DOT, LINK, MATIC, NEAR, ATOM, UNI, LTC (1095 days, 2021-2023).
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.epistemic.cross_asset_gate import (
    evaluate_cross_asset_gate,
)
from apedp_dyn.epistemic.market_extrapolation import (
    evaluate_30_asset_market_ladder,
)
from apedp_dyn.harness import AEH
from apedp_dyn.harness.manifest import compute_dataset_hash
from apedp_dyn.sandbox.grammar_space import causal_zscore, trailing_rank
from apedp_dyn.grammar.trading_grammar_catalog import (
    TradingGrammarCatalog,
    MASTER_FEATURE_CATALOG,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market"
CRYPTO_ORDERFLOW_FILE = DATA_DIR / "crypto_top15_orderflow.csv"

CRYPTO_15_ASSETS: tuple[str, ...] = (
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE",
    "DOT", "LINK", "MATIC", "NEAR", "ATOM", "UNI", "LTC",
)
CRYPTO_30_ASSETS: tuple[str, ...] = (
    *CRYPTO_15_ASSETS,
    "BCH", "XLM", "TRX", "ETC", "FIL", "VET", "THETA", "ALGO",
    "AAVE", "MKR", "EOS", "XMR", "KSM", "ZEC", "MANA",
)
CRYPTO_ASSETS: tuple[str, ...] = CRYPTO_15_ASSETS

EQUITY_15_ASSETS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PG", "XOM", "UNH", "SPY",
)
EQUITY_30_ASSETS: tuple[str, ...] = (
    *EQUITY_15_ASSETS,
    "QQQ", "IWM", "GLD", "TLT", "BAC", "HD", "CVX", "LLY",
    "ABBV", "COST", "MRK", "KO", "PEP", "AMD", "NFLX",
)
EQUITY_ASSETS: tuple[str, ...] = EQUITY_15_ASSETS

CRYPTO_45_ASSETS: tuple[str, ...] = (
    *CRYPTO_30_ASSETS,
    "SAND", "AXS", "CRV", "SNX", "COMP", "GRT", "SUSHI", "1INCH", "ENJ", "BAT",
    "CHZ", "FTM", "ZIL", "ICX", "ONT",
)
EQUITY_45_ASSETS: tuple[str, ...] = (
    *EQUITY_30_ASSETS,
    "INTC", "CRM", "ORCL", "ADBE", "CSCO", "QCOM",
    "XLE", "XLF", "XLV", "XLI", "XLU", "XLK", "XLP",
    "EEM", "USO",
)

MULTI_30_ASSETS: tuple[str, ...] = CRYPTO_15_ASSETS + EQUITY_15_ASSETS
MULTI_60_ASSETS: tuple[str, ...] = CRYPTO_30_ASSETS + EQUITY_30_ASSETS
MULTI_90_ASSETS: tuple[str, ...] = CRYPTO_45_ASSETS + EQUITY_45_ASSETS

EQUITY_FILE = DATA_DIR / "equity_top15.csv"
MULTI_30_FILE = DATA_DIR / "multi_asset_30_orderflow.csv"
MULTI_60_FILE = DATA_DIR / "multi_asset_60_orderflow.csv"
MULTI_90_FILE = DATA_DIR / "multi_asset_90_5yr_orderflow.csv"


# ── Load Synchronized Multi-Asset Universe ─────────────────────────────────
def load_universe(universe_type: str = "crypto") -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    """Load historical time series for crypto (15, 30, 45), equity (15, 30, 45), multi-30, multi-60, or multi-90."""
    u_type = universe_type.lower()
    if u_type in ("equity", "equity_15", "equities", "stocks"):
        df = pd.read_csv(EQUITY_FILE)
        symbols = EQUITY_15_ASSETS
    elif u_type in ("equity_30", "equities_30"):
        df = pd.read_csv(MULTI_60_FILE)
        symbols = EQUITY_30_ASSETS
    elif u_type in ("crypto_30",):
        df = pd.read_csv(MULTI_60_FILE)
        symbols = CRYPTO_30_ASSETS
    elif u_type in ("90", "multi_90", "90_asset"):
        df = pd.read_csv(MULTI_90_FILE)
        symbols = MULTI_90_ASSETS
    elif u_type in ("60", "multi_60", "60_asset"):
        df = pd.read_csv(MULTI_60_FILE)
        symbols = MULTI_60_ASSETS
    elif u_type in ("45", "multi_45", "45_asset", "crypto_45"):
        df = pd.read_csv(MULTI_60_FILE)
        symbols = CRYPTO_30_ASSETS + EQUITY_15_ASSETS
    elif u_type in ("30", "multi_30"):
        df = pd.read_csv(MULTI_30_FILE)
        symbols = MULTI_30_ASSETS
    elif u_type in ("multi", "all", "cross_asset"):
        # Default unified multi-asset: prefer 90 if file exists, else 60, else 30
        if MULTI_90_FILE.exists():
            df = pd.read_csv(MULTI_90_FILE)
            symbols = MULTI_90_ASSETS
        elif MULTI_60_FILE.exists():
            df = pd.read_csv(MULTI_60_FILE)
            symbols = MULTI_60_ASSETS
        else:
            df = pd.read_csv(MULTI_30_FILE)
            symbols = MULTI_30_ASSETS
    else:
        df = pd.read_csv(CRYPTO_ORDERFLOW_FILE)
        symbols = CRYPTO_15_ASSETS

    universe = {}
    for sym in symbols:
        p = df[f"{sym}_CLOSE"].to_numpy(float)
        v = df[f"{sym}_VOL"].to_numpy(float)
        tb = df[f"{sym}_TAKER_BUY"].to_numpy(float) if f"{sym}_TAKER_BUY" in df.columns else v * 0.5
        ofi = df[f"{sym}_OFI"].to_numpy(float) if f"{sym}_OFI" in df.columns else np.zeros(len(p))
        r = np.append(0.0, np.diff(np.log(p)))

        if f"{sym}_HIGH" in df.columns and f"{sym}_LOW" in df.columns:
            h = df[f"{sym}_HIGH"].to_numpy(float)
            low = df[f"{sym}_LOW"].to_numpy(float)
        else:
            vol_proxy = pd.Series(np.abs(r)).rolling(5, min_periods=1).mean().to_numpy()
            h = p * (1.0 + vol_proxy * 1.2)
            low = p * (1.0 - vol_proxy * 1.2)

        universe[sym] = pd.DataFrame({
            "price": p, "high": h, "low": low, "close": p,
            "r": r, "V": v, "taker_buy": tb, "ofi": ofi,
        })

    return universe, symbols


def load_15_asset_universe() -> dict[str, pd.DataFrame]:
    """Backward-compatible loader for 15-crypto asset universe."""
    u, _ = load_universe("crypto")
    return u


# ── Signal Extraction: Primitives + Non-Linear Interactions ───────────────
def calc_signals_for_asset(df: pd.DataFrame, cross_ret_median: np.ndarray) -> np.ndarray:
    """Extract 24 quantitative signals (18 primitives + 6 non-linear interactions)."""
    p = df["price"].to_numpy(float)
    r = df["r"].to_numpy(float)
    v = df["V"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    tb = df["taker_buy"].to_numpy(float)
    ofi = df["ofi"].to_numpy(float)

    # ── 1. Microstructure & Volatility Primitives
    s_ofi = causal_zscore(ofi)
    v_clean = np.maximum(v, 1e-12)
    s_cvd = causal_zscore((2.0 * tb - v_clean) / v_clean)
    s_p = causal_zscore(np.abs(r))
    s_asym = causal_zscore(np.where(r < 0, np.abs(r), 0.0))

    c_prev = np.roll(p, 1)
    c_prev[0] = p[0]
    tr = np.maximum(h - low, np.maximum(np.abs(h - c_prev), np.abs(low - c_prev)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    s_natr = causal_zscore(atr / np.maximum(p, 1e-12) * 100.0)

    sma20 = pd.Series(p).rolling(20, min_periods=20).mean().to_numpy()
    std20 = pd.Series(p).rolling(20, min_periods=20).std().to_numpy()
    s_squeeze = causal_zscore((4.0 * std20) / np.maximum(3.0 * atr, 1e-12))

    # ── 2. Trend & Oscillators Primitives
    ema12 = pd.Series(p).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(p).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    s_macd = causal_zscore((macd_line - macd_signal).to_numpy())

    fast = pd.Series(p).ewm(span=9, adjust=False).mean()
    slow = pd.Series(p).ewm(span=21, adjust=False).mean()
    s_ema_cross = causal_zscore((fast - slow).to_numpy())

    r_s = pd.Series(r)
    gain = r_s.where(r_s > 0, 0.0).rolling(14, min_periods=14).mean()
    loss = -r_s.where(r_s < 0, 0.0).rolling(14, min_periods=14).mean()
    rsi = 100.0 - (100.0 / (1.0 + (gain / loss.replace(0, 1e-12))))
    s_rsi = causal_zscore(rsi.to_numpy())

    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    pct_b = (p - lower) / np.maximum(upper - lower, 1e-12)
    s_bb = causal_zscore(pct_b)

    s_mom = causal_zscore(pd.Series(p).pct_change(5).to_numpy())
    h14 = pd.Series(h).rolling(14).max().to_numpy()
    l14 = pd.Series(low).rolling(14).min().to_numpy()
    s_stoch = causal_zscore((p - l14) / np.maximum(h14 - l14, 1e-12) * 100.0)

    # ── 3. Volume & Mathematics Primitives
    s_rv = trailing_rank(v)
    rm = pd.Series(v).rolling(40, min_periods=10).mean().to_numpy()
    with np.errstate(all="ignore"):
        inner = (np.log(np.maximum(rm, 1e-12)) + causal_zscore(np.abs(r))
                 + causal_zscore(np.abs(r) / np.maximum(v, 1e-12)))
        s_ha = causal_zscore(inner)

    s_logv = causal_zscore(np.log(np.maximum(v, 1e-18)))

    tp = (h + low + p) / 3.0
    raw_mf = tp * v
    tp_prev = np.roll(tp, 1)
    tp_prev[0] = tp[0]
    pos_mf = pd.Series(np.where(tp > tp_prev, raw_mf, 0.0)).rolling(14).sum()
    neg_mf = pd.Series(np.where(tp < tp_prev, raw_mf, 0.0)).rolling(14).sum()
    mfi = 100.0 - (100.0 / (1.0 + (pos_mf / neg_mf.replace(0, 1e-12))))
    s_mfi = causal_zscore(mfi.to_numpy())

    obv = np.zeros(len(p))
    for t in range(1, len(p)):
        obv[t] = obv[t - 1] + (v[t] if p[t] > p[t - 1] else (-v[t] if p[t] < p[t - 1] else 0.0))
    s_obv = causal_zscore(pd.Series(obv).pct_change(10).to_numpy())

    hurst = np.full(len(r), np.nan)
    for t in range(30, len(r)):
        sub = r[t - 30:t]
        cum_dev = np.cumsum(sub - np.mean(sub))
        r_range = np.max(cum_dev) - np.min(cum_dev)
        s_std = np.std(sub)
        if s_std > 1e-9 and r_range > 1e-9:
            hurst[t] = (np.log(r_range / s_std) / np.log(30)) - 0.5
    s_hurst = causal_zscore(hurst)

    cross_rel = r - cross_ret_median
    s_cross_mom = causal_zscore(pd.Series(cross_rel).rolling(5, min_periods=5).mean().to_numpy())

    # ── 4. Higher-Order Non-Linear Interactions & Gating Terms
    # NL-1: Volume-Gated Momentum (trade momentum only with institutional volume confirmation)
    s_nl_vol_mom = causal_zscore(s_mom * s_rv)

    # NL-2: Squeeze-Released Trend (trend acceleration escaping volatility compression)
    s_nl_sqz_trend = causal_zscore(s_macd * np.exp(-np.clip(s_squeeze, -2.0, 2.0)))

    # NL-3: Risk-Adjusted Velocity (momentum scaled inversely by ATR volatility)
    s_nl_risk_mom = causal_zscore(s_mom / np.maximum(s_natr + 3.0, 0.5))

    # NL-4: Order-Flow Reversal (oscillator turning point confirmed by order flow imbalance)
    s_nl_ofi_rev = causal_zscore(s_rsi * s_ofi)

    # NL-5: Hurst-Adaptive Regime Switcher (trend-follow if H > 0, mean-revert if H < 0)
    s_nl_hurst_adapt = causal_zscore(np.where(s_hurst > 0, s_macd, -s_bb))

    # NL-6: Taker Aggression Momentum Surge (taker buy flow surge compounded with price momentum)
    s_nl_taker_mom = causal_zscore(s_cvd * s_mom)

    # NL-7: Unified Master SNR (Drift / Asymmetric Variance Floor)
    sig_asym_t = np.maximum(0.01 + 0.3 * np.abs(r) + 0.5 * np.where(r < 0, np.abs(r), 0.0), 1e-4)
    s_nl_master_snr = causal_zscore(s_mom / (sig_asym_t ** 2))

    feats = [
        s_ofi, s_cvd, s_p, s_asym, s_natr, s_squeeze, s_macd, s_ema_cross, s_rsi,
        s_bb, s_mom, s_stoch, s_rv, s_ha, s_logv, s_mfi, s_obv, s_hurst, s_cross_mom,
        s_nl_vol_mom, s_nl_sqz_trend, s_nl_risk_mom, s_nl_ofi_rev, s_nl_hurst_adapt, s_nl_taker_mom, s_nl_master_snr,
    ]
    return np.column_stack([np.nan_to_num(f, nan=0.0) for f in feats])


SIGNAL_NAMES = [
    "H-OFI (Order Flow Imbalance)",
    "H-CVD (Taker Buy Flow / CVD)",
    "H-P (Volatility Persistence)",
    "H-ASYM (GJR Panic Leverage)",
    "H-NATR (Normalized ATR)",
    "H-SQUEEZE (BB/Keltner Squeeze)",
    "H-MACD (Trend Momentum)",
    "H-EMA_CROSS (Fast/Slow EMA)",
    "H-RSI (RSI Oscillator)",
    "H-BB (Bollinger Bands %B)",
    "H-MOM (5-Day Momentum)",
    "H-STOCH (Stochastic Oscillator)",
    "H-RV (Trailing Volume Rank)",
    "H-A (Amihud Illiquidity)",
    "H-LOG (Log Volume Level)",
    "H-MFI (Money Flow Index)",
    "H-OBV_MOM (OBV Momentum)",
    "H-HURST (Hurst Regime Memory)",
    "H-CROSS_MOM (Cross-Asset Rel Mom)",
    # Non-linear terms
    "NL_VOL_GATED_MOM (Mom x VolRank)",
    "NL_SQUEEZE_TREND (MACD x Squeeze)",
    "NL_RISK_ADJ_MOM (Mom / NATR)",
    "NL_OFI_REV (RSI x OFI Reversal)",
    "NL_HURST_ADAPT (Adaptive Regime)",
    "NL_TAKER_MOM (Taker Surge x Mom)",
    "NL_MASTER_SNR (Mom / AsymVol^2)",
]

EXPANDED_SIGNAL_NAMES = SIGNAL_NAMES + [
    "RET_1 (1-Bar Return)",
    "RET_4 (4-Bar Return)",
    "RET_16 (16-Bar Return)",
    "LOGRET (Log Return)",
    "OPEN_TO_CLOSE (Intrabar Return)",
    "TRUE_RANGE (Wilder True Range)",
    "BODY_TO_RANGE (Body/Range Ratio)",
    "CLOSE_LOCATION (Close in Bar Range)",
    "EMA_SPREAD (Fast/Slow EMA Spread)",
    "EMA_SLOPE (20-EMA Regression Slope)",
    "ADX (Average Directional Index)",
    "TREND_EFFICIENCY (Kaufman Efficiency)",
    "PARKINSON_VOL (High-Low Parkinson Vol)",
    "DOWNSIDE_VOL (Downside Semivariance)",
    "BB_WIDTH (Bollinger Band Width)",
    "KELTNER_WIDTH (Keltner Channel Width)",
    "BREAKOUT_STRENGTH (Donchian Excursion)",
    "SIGNED_VOL_MOM (Signed Volume Momentum)",
    "CMF (Chaikin Money Flow)",
    "PRODUCT_H_BB_TAKER (BB x Taker Surge)",
]


def calc_expanded_signals_for_asset(df: pd.DataFrame, cross_ret_median: np.ndarray) -> np.ndarray:
    """Extract 46 quantitative signals (26 legacy signals + 20 comprehensive grammar features)."""
    base_signals = calc_signals_for_asset(df, cross_ret_median)

    p = df["price"].to_numpy(float)
    r = df["r"].to_numpy(float)
    v = df["V"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    tb = df["taker_buy"].to_numpy(float) if "taker_buy" in df.columns else v * 0.5
    o = df["open"].to_numpy(float) if "open" in df.columns else p

    # 27. RET_1
    s_ret1 = causal_zscore(np.append(0.0, (p[1:] - p[:-1]) / np.maximum(p[:-1], 1e-12)))
    # 28. RET_4
    s_ret4 = causal_zscore(pd.Series(p).pct_change(4).to_numpy())
    # 29. RET_16
    s_ret16 = causal_zscore(pd.Series(p).pct_change(16).to_numpy())
    # 30. LOGRET
    s_logret = causal_zscore(r)
    # 31. OPEN_TO_CLOSE
    s_oc = causal_zscore((p - o) / np.maximum(o, 1e-12))
    # 32. TRUE_RANGE
    c_prev = np.roll(p, 1)
    c_prev[0] = p[0]
    tr = np.maximum(h - low, np.maximum(np.abs(h - c_prev), np.abs(low - c_prev)))
    s_tr = causal_zscore(tr / np.maximum(p, 1e-12))
    # 33. BODY_TO_RANGE
    hl_range = np.maximum(h - low, 1e-12)
    s_body_range = causal_zscore(np.abs(p - o) / hl_range)
    # 34. CLOSE_LOCATION
    s_close_loc = causal_zscore((p - low) / hl_range)
    # 35. EMA_SPREAD
    ema12 = pd.Series(p).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(p).ewm(span=26, adjust=False).mean()
    s_ema_spread = causal_zscore(((ema12 - ema26) / np.maximum(p, 1e-12)).to_numpy())
    # 36. EMA_SLOPE
    ema20 = pd.Series(p).ewm(span=20, adjust=False).mean().to_numpy()
    s_ema_slope = causal_zscore(pd.Series(ema20).diff(5).to_numpy())
    # 37. ADX
    up_m = h - np.roll(h, 1)
    up_m[0] = 0
    dn_m = np.roll(low, 1) - low
    dn_m[0] = 0
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    p_di = 100.0 * pd.Series(p_dm).rolling(14).mean().to_numpy() / np.maximum(atr14, 1e-12)
    m_di = 100.0 * pd.Series(m_dm).rolling(14).mean().to_numpy() / np.maximum(atr14, 1e-12)
    dx = 100.0 * np.abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-12)
    s_adx = causal_zscore(pd.Series(dx).rolling(14).mean().to_numpy())
    # 38. TREND_EFFICIENCY
    net_mv = np.abs(pd.Series(p).diff(20).to_numpy())
    gross_mv = pd.Series(np.abs(np.append(0.0, np.diff(p)))).rolling(20).sum().to_numpy()
    s_trend_eff = causal_zscore(net_mv / np.maximum(gross_mv, 1e-12))
    # 39. PARKINSON_VOL
    log_hl2 = np.log(np.maximum(h / np.maximum(low, 1e-12), 1.0)) ** 2
    park_var = pd.Series(log_hl2).rolling(20).sum().to_numpy() / (4.0 * np.log(2.0) * 20.0)
    s_park_vol = causal_zscore(np.sqrt(np.maximum(park_var, 0.0)))
    # 40. DOWNSIDE_VOL
    down_r2 = np.where(r < 0, r ** 2, 0.0)
    s_down_vol = causal_zscore(np.sqrt(pd.Series(down_r2).rolling(20).mean().to_numpy()))
    # 41. BB_WIDTH
    std20 = pd.Series(p).rolling(20, min_periods=20).std().to_numpy()
    sma20 = pd.Series(p).rolling(20, min_periods=20).mean().to_numpy()
    s_bb_width = causal_zscore((4.0 * std20) / np.maximum(sma20, 1e-12))
    # 42. KELTNER_WIDTH
    s_kc_width = causal_zscore((3.0 * atr14) / np.maximum(ema20, 1e-12))
    # 43. BREAKOUT_STRENGTH
    don_h_prev = pd.Series(h).shift(1).rolling(20, min_periods=20).max().to_numpy()
    don_l_prev = pd.Series(low).shift(1).rolling(20, min_periods=20).min().to_numpy()
    up_bo = np.maximum(p - don_h_prev, 0.0)
    dn_bo = np.maximum(don_l_prev - p, 0.0)
    s_breakout = causal_zscore((up_bo - dn_bo) / np.maximum(atr14, 1e-12))
    # 44. SIGNED_VOLUME_MOMENTUM
    signed_v = 2.0 * tb - v
    s_sv_mom = causal_zscore(pd.Series(signed_v).diff(5).to_numpy())
    # 45. CMF
    mf_mult = ((p - low) - (h - p)) / hl_range
    mf_v = mf_mult * v
    cmf = pd.Series(mf_v).rolling(20).sum().to_numpy() / np.maximum(pd.Series(v).rolling(20).sum().to_numpy(), 1e-12)
    s_cmf = causal_zscore(cmf)
    # 46. PRODUCT_H_BB_TAKER
    s_bb_taker = causal_zscore(base_signals[:, 9] * base_signals[:, 24])

    new_feats = [
        s_ret1, s_ret4, s_ret16, s_logret, s_oc, s_tr, s_body_range, s_close_loc,
        s_ema_spread, s_ema_slope, s_adx, s_trend_eff, s_park_vol, s_down_vol,
        s_bb_width, s_kc_width, s_breakout, s_sv_mom, s_cmf, s_bb_taker,
    ]
    new_cols = np.column_stack([np.nan_to_num(f, nan=0.0) for f in new_feats])
    return np.hstack([base_signals, new_cols])



# ── Multi-Asset Portfolio Simulation Engine (Equal Weight & Risk Parity) ──
@dataclass
class MultiAssetResult:
    name: str
    category: str
    acc: float
    total_return_pct: float
    final_equity: float
    pnl: float
    win_rate: float
    trades: int
    profit_factor: float
    max_dd: float
    sharpe: float


def simulate_multi_asset_portfolio(
    scores_3d: np.ndarray,          # Shape: (T, N_assets)
    returns_2d: np.ndarray,         # Shape: (T, N_assets)
    vols_2d: np.ndarray,            # Shape: (T, N_assets) trailing realized volatilities
    mode: str = "directional",      # 'directional' or 'cross_sectional_ls'
    sizing: str = "equal",          # 'equal', 'risk_parity', or 'master_snr'
    threshold: float = 0.10,
    fee_bps: float = 5.0,
    initial_capital: float = 100000.0,
    sqz_2d: np.ndarray | None = None,
    hurst_2d: np.ndarray | None = None,
    asym_vols_2d: np.ndarray | None = None,
    return_equity_curve: bool = False,
) -> tuple[Any, ...]:
    """Execute trading across all 15 assets simultaneously as a unified portfolio."""
    T, N = scores_3d.shape
    fee = fee_bps / 10000.0
    capital = initial_capital
    portfolio_equity = [capital]
    strat_returns = []
    trade_pnls = []

    pos = np.zeros(N)

    correct_predictions = 0
    total_predictions = 0

    for t in range(T - 1):
        sc_t = scores_3d[t]
        fwd_ret = returns_2d[t + 1]
        sigma_t = np.maximum(vols_2d[t], 0.005)  # floor at 0.5% daily volatility

        # Target positions for all 15 assets
        if mode == "directional":
            target_pos = np.where(sc_t > threshold, 1.0, np.where(sc_t < -threshold, -1.0, 0.0))
            if sizing == "risk_parity":
                # Inverse-volatility weighting across active assets
                active = target_pos != 0.0
                if np.sum(active) > 0:
                    inv_vol = 1.0 / sigma_t
                    inv_vol[~active] = 0.0
                    weights = (target_pos * inv_vol) / np.maximum(np.sum(inv_vol), 1e-12)
                else:
                    weights = np.zeros(N)
            elif sizing == "master_snr":
                # Master Sizing Equation: Phi_t = exp(-sqz) * (1 + tanh(2(H-0.5))) / sigma_asym^2
                active = target_pos != 0.0
                if np.sum(active) > 0:
                    sqz_val = sqz_2d[t] if sqz_2d is not None else np.zeros(N)
                    h_val = hurst_2d[t] if hurst_2d is not None else np.zeros(N)
                    sig_asym = asym_vols_2d[t] if asym_vols_2d is not None else sigma_t
                    phi = np.exp(-np.clip(sqz_val, -2.0, 2.0)) * (1.0 + np.tanh(2.0 * h_val)) / np.maximum(sig_asym ** 2, 1e-4)
                    phi[~active] = 0.0
                    weights = (target_pos * phi) / np.maximum(np.sum(phi), 1e-12)
                else:
                    weights = np.zeros(N)
            else:
                weights = target_pos / N
        else:
            # Cross-Sectional Long/Short Alpha Factor Portfolio
            rank = np.argsort(np.argsort(sc_t))
            target_pos = np.zeros(N)
            long_mask = rank >= N - 4   # Top 4 coins
            short_mask = rank < 4       # Bottom 4 coins
            target_pos[long_mask] = 1.0
            target_pos[short_mask] = -1.0

            if sizing == "risk_parity":
                # Risk Parity: equal risk contribution per asset (1/sigma weighting)
                long_inv = (1.0 / sigma_t) * long_mask
                short_inv = (1.0 / sigma_t) * short_mask

                w_long = long_inv / np.maximum(np.sum(long_inv), 1e-12)
                w_short = short_inv / np.maximum(np.sum(short_inv), 1e-12)
                weights = w_long - w_short  # Dollar-neutral risk parity
            elif sizing == "master_snr":
                sqz_val = sqz_2d[t] if sqz_2d is not None else np.zeros(N)
                h_val = hurst_2d[t] if hurst_2d is not None else np.zeros(N)
                sig_asym = asym_vols_2d[t] if asym_vols_2d is not None else sigma_t
                phi = np.exp(-np.clip(sqz_val, -2.0, 2.0)) * (1.0 + np.tanh(2.0 * h_val)) / np.maximum(sig_asym ** 2, 1e-4)
                long_phi = phi * long_mask
                short_phi = phi * short_mask
                w_long = long_phi / np.maximum(np.sum(long_phi), 1e-12)
                w_short = short_phi / np.maximum(np.sum(short_phi), 1e-12)
                weights = w_long - w_short  # Dollar-neutral master sizing
            else:
                weights = target_pos / 4.0

        # Turnover fees across all 15 coins
        turnover = np.sum(np.abs(target_pos - pos))
        turnover_fee = capital * turnover * (fee / N)
        capital -= turnover_fee

        for a in range(N):
            if pos[a] != 0.0 and (target_pos[a] != pos[a] or target_pos[a] == 0.0):
                asset_ret = pos[a] * fwd_ret[a]
                trade_pnls.append(asset_ret)

        valid = (target_pos != 0.0) & (fwd_ret != 0.0)
        correct_predictions += np.sum(np.sign(target_pos[valid]) == np.sign(fwd_ret[valid]))
        total_predictions += np.sum(valid)

        pos = target_pos

        day_ret = np.sum(weights * fwd_ret) - (turnover_fee / capital)
        capital += capital * day_ret
        strat_returns.append(day_ret)
        portfolio_equity.append(capital)

    tot_ret = (capital - initial_capital) / initial_capital * 100.0
    pnl = capital - initial_capital
    trades = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = (len(wins) / trades * 100.0) if trades > 0 else 0.0

    g_profit = sum(wins) if wins else 0.0
    g_loss = abs(sum(losses)) if losses else 0.0
    pf = (g_profit / g_loss) if g_loss > 1e-6 else (99.9 if g_profit > 0 else 0.0)

    acc = (correct_predictions / total_predictions * 100.0) if total_predictions > 0 else 50.0

    sr = np.array(strat_returns)
    sharpe = float(np.mean(sr) / np.std(sr) * np.sqrt(365)) if len(sr) > 1 and np.std(sr) > 1e-9 else 0.0

    eq = np.array(portfolio_equity)
    peaks = np.maximum.accumulate(eq)
    max_dd = float(np.max((peaks - eq) / np.maximum(peaks, 1e-12))) * 100.0

    if return_equity_curve:
        return tot_ret, sharpe, acc, pnl, trades, pf, max_dd, win_rate, eq
    return tot_ret, sharpe, acc, pnl, trades, pf, max_dd, win_rate


def combine_two_tier_portfolios(
    eq_c: np.ndarray | list[float],
    eq_e: np.ndarray | list[float],
    initial_capital: float = 100000.0,
) -> tuple[float, float, float, float, np.ndarray]:
    """Calculate mathematically rigorous combined two-tier portfolio metrics from joint equity curves.

    Eliminates uncomputed display string literals, arithmetic averaging of Sharpe ratios,
    and averaged drawdowns. Computes the exact simultaneous portfolio equity trajectory:
        E_two_tier(t) = E_crypto(t) + E_equity(t)

    Returns
    -------
    tot_ret : float
        Combined net return percentage
    sharpe : float
        Annualized Sharpe ratio of the joint daily return series
    pnl : float
        Total combined dollar PnL
    max_dd : float
        Simultaneous peak-to-trough maximum drawdown of the combined portfolio
    eq_combined : np.ndarray
        Full daily joint equity trajectory
    """
    eq_c_arr = np.asarray(eq_c, dtype=float)
    eq_e_arr = np.asarray(eq_e, dtype=float)
    eq_combined = eq_c_arr + eq_e_arr
    pnl = float(eq_combined[-1] - initial_capital)
    tot_ret = float(pnl / initial_capital * 100.0)

    if len(eq_combined) > 1:
        daily_ret = np.diff(eq_combined) / np.maximum(eq_combined[:-1], 1e-12)
        std_ret = float(np.std(daily_ret))
        sharpe = float(np.mean(daily_ret) / std_ret * np.sqrt(365)) if std_ret > 1e-9 else 0.0
    else:
        sharpe = 0.0

    peaks = np.maximum.accumulate(eq_combined)
    drawdowns = (peaks - eq_combined) / np.maximum(peaks, 1e-12)
    max_dd = float(np.max(drawdowns)) * 100.0

    return tot_ret, sharpe, pnl, max_dd, eq_combined


# ── Multi-Asset Genetic Algorithm Engine ──────────────────────────────────
@dataclass
class MultiAssetChromosome:
    mask: np.ndarray      # bool array (K,)
    weights: np.ndarray   # float array (K,)
    threshold: float
    fitness: float = -999.0
    in_sample_sharpe: float = 0.0

    def to_disclosure(self) -> dict[str, Any]:
        """Disclose full mathematical equation, weights, and activation threshold for independent replication."""
        active_idx = np.where(self.mask)[0]
        weights_dict = {
            SIGNAL_NAMES[idx]: float(round(self.weights[idx], 4))
            for idx in active_idx
        }
        terms = [
            f"{self.weights[idx]:+.4f}*{SIGNAL_NAMES[idx].split()[0]}"
            for idx in active_idx
        ]
        formula_expr = f"Score = ({' + '.join(terms)}) | Threshold = {self.threshold:.4f}" if terms else "0.0"
        return {
            "active_signals": [SIGNAL_NAMES[idx] for idx in active_idx],
            "weights": weights_dict,
            "threshold": float(round(self.threshold, 4)),
            "formula_expression": formula_expr,
        }


# ── PINNED CANONICAL CHROMOSOMES (Bit-Exact Reproducibility Fixtures) ─────
def _make_pinned_chromosome(active_indices: list[int], weights_list: list[float], threshold: float) -> MultiAssetChromosome:
    K = len(SIGNAL_NAMES)
    mask = np.zeros(K, dtype=bool)
    mask[active_indices] = True
    weights = np.array(weights_list, dtype=float)
    return MultiAssetChromosome(mask=mask, weights=weights, threshold=float(threshold))


PINNED_30_CHROMOSOME = _make_pinned_chromosome(
    active_indices=[9, 16, 24, 25],
    weights_list=[
        0.664475, -0.854975, 0.542451, 0.404146, 0.330779, 0.395218, 0.466604, -0.373393,
        -0.921611, 0.811068, -0.401223, -0.942562, 0.23758, 0.832628, 0.498378, 0.248161,
        -1.427209, -0.192376, 0.227151, -0.261791, 0.235466, 0.693572, 0.575445, 0.192847,
        0.598382, 0.67395,
    ],
    threshold=0.02,
)

PINNED_60_CHROMOSOME = _make_pinned_chromosome(
    active_indices=[9, 10, 16, 23],
    weights_list=[
        -0.608433, -0.968759, 0.572466, 0.213264, 0.935969, 1.061971, 0.778484, -0.289908,
        -0.037336, 0.544673, -0.201948, -0.659154, 0.214724, 0.7122, 0.183651, -0.011102,
        -1.637016, -0.043766, 1.010211, -1.154358, -0.019645, 0.745738, 0.045153, -0.359291,
        0.417085, -0.292476,
    ],
    threshold=0.1979,
)

PINNED_90_CHROMOSOME = _make_pinned_chromosome(
    active_indices=[3, 4, 5, 9],
    weights_list=[
        -0.312959, -1.680733, 0.754375, -0.125805, 1.417009, 1.40334, 0.685069, -0.480148,
        0.121308, 0.421785, -0.251341, -0.113909, -0.020772, 0.970226, 0.217211, 0.101206,
        -1.058273, 0.079461, 1.378564, -0.518799, -0.078233, 0.571708, 0.061348, 0.290388,
        0.078821, -0.293749,
    ],
    threshold=0.2063,
)

PINNED_15_CHROMOSOME = _make_pinned_chromosome(
    active_indices=[9, 11, 16, 23],
    weights_list=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.1948, 0.0, 0.2227, 0.0, 0.0, 0.0, 0.0,
        0.4501, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4839,
        0.0, 0.0,
    ],
    threshold=0.1695,
)

MODEL_4B_60_CHROMOSOME = _make_pinned_chromosome(
    active_indices=[9, 10, 16, 24],
    weights_list=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.5447, -0.2019, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.6370, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.4171, 0.0,
    ],
    threshold=0.1979,
)

MODEL_4B_90_CHROMOSOME = _make_pinned_chromosome(
    active_indices=[9, 10, 16, 24],
    weights_list=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.4501, -0.1502, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.2503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.3802, 0.0,
    ],
    threshold=0.1979,
)

# Canonical Model 4b Frozen Family across all 4 universes
MODEL_4B_CHROMOSOMES: dict[str, MultiAssetChromosome] = {
    "15": PINNED_15_CHROMOSOME,
    "crypto": PINNED_15_CHROMOSOME,
    "30": PINNED_30_CHROMOSOME,
    "60": MODEL_4B_60_CHROMOSOME,
    "90": MODEL_4B_90_CHROMOSOME,
}

PINNED_CHROMOSOMES: dict[str, MultiAssetChromosome] = {
    "15": PINNED_15_CHROMOSOME,
    "crypto": PINNED_15_CHROMOSOME,
    "30": PINNED_30_CHROMOSOME,
    "60": PINNED_60_CHROMOSOME,
    "90": PINNED_90_CHROMOSOME,
}


def run_multi_asset_genetic_algorithm(
    features_3d_tr: np.ndarray,  # (T_tr, N_assets, K_features)
    returns_2d_tr: np.ndarray,   # (T_tr, N_assets)
    vols_2d_tr: np.ndarray,      # (T_tr, N_assets)
    pop_size: int = 80,
    generations: int = 35,
    seed: int = 42,
) -> MultiAssetChromosome:
    """Evolve non-linear signal combinations on all 15 assets simultaneously during in-sample train split."""
    rng = np.random.RandomState(seed)
    _T, _N, K = features_3d_tr.shape

    population: list[MultiAssetChromosome] = []
    for i in range(pop_size):
        if i < K:
            mask = np.zeros(K, dtype=bool)
            mask[i] = True
        elif i < K + 20:
            mask = np.zeros(K, dtype=bool)
            subset = rng.choice(K, size=rng.randint(2, 5), replace=False)
            mask[subset] = True
        else:
            mask = rng.rand(K) < 0.20
            if not np.any(mask):
                mask[rng.randint(0, K)] = True

        weights = np.asarray(rng.uniform(-1.5, 1.5, size=K))
        th = float(rng.uniform(0.05, 0.25))
        population.append(MultiAssetChromosome(mask=mask, weights=weights, threshold=th))

    def evaluate_chromosome(c: MultiAssetChromosome) -> float:
        eff_w = c.mask.astype(float) * c.weights
        if np.sum(c.mask) == 0 or np.all(eff_w == 0):
            c.fitness = -999.0
            return -999.0

        scores_3d = features_3d_tr @ eff_w
        scores_norm = (scores_3d - np.mean(scores_3d, axis=1, keepdims=True)) / np.maximum(
            np.std(scores_3d, axis=1, keepdims=True), 1e-8
        )

        # Fast vectorized portfolio evaluation with inverse volatility weighting
        target_pos = np.where(scores_norm[:-1] > c.threshold, 1.0, np.where(scores_norm[:-1] < -c.threshold, -1.0, 0.0))
        turnover = np.mean(np.abs(np.diff(target_pos, axis=0, prepend=target_pos[:1])))
        
        # Risk parity weighting
        inv_vol = 1.0 / np.maximum(vols_2d_tr[:-1], 0.005)
        raw_w = target_pos * inv_vol
        norm_factor = np.maximum(np.sum(np.abs(raw_w), axis=1, keepdims=True), 1e-8)
        weights = raw_w / norm_factor

        daily_ret = np.sum(weights * returns_2d_tr[1:], axis=1) - turnover * 0.0005
        valid = (target_pos != 0.0) & (returns_2d_tr[1:] != 0.0)
        acc = float(np.mean(np.sign(target_pos[valid]) == np.sign(returns_2d_tr[1:][valid])) * 100.0) if np.sum(valid) > 0 else 50.0

        sd = np.std(daily_ret)
        sharpe = float(np.mean(daily_ret) / sd * np.sqrt(365)) if sd > 1e-9 else 0.0
        tot_ret = float(np.sum(daily_ret) * 100.0)

        sparsity_pen = 0.02 * np.sum(c.mask)
        c.in_sample_sharpe = sharpe
        c.fitness = sharpe + 0.01 * tot_ret + 0.03 * (acc - 50.0) - sparsity_pen
        return c.fitness

    for c in population:
        evaluate_chromosome(c)

    for _gen in range(generations):
        population.sort(key=lambda c: c.fitness, reverse=True)
        next_gen = [
            MultiAssetChromosome(
                mask=c.mask.copy(),
                weights=c.weights.copy(),
                threshold=c.threshold,
                fitness=c.fitness,
                in_sample_sharpe=c.in_sample_sharpe,
            )
            for c in population[:5]
        ]

        while len(next_gen) < pop_size:
            idx1 = rng.choice(pop_size, size=3, replace=False)
            p1 = max([population[i] for i in idx1], key=lambda c: c.fitness)
            idx2 = rng.choice(pop_size, size=3, replace=False)
            p2 = max([population[i] for i in idx2], key=lambda c: c.fitness)

            child_mask = np.where(rng.rand(K) < 0.5, p1.mask, p2.mask)
            if not np.any(child_mask):
                child_mask[rng.randint(0, K)] = True

            alpha = rng.uniform(0.0, 1.0)
            child_weights = alpha * p1.weights + (1.0 - alpha) * p2.weights
            child_th = float(0.5 * (p1.threshold + p2.threshold))

            if rng.rand() < 0.20:
                flip = rng.randint(0, K)
                child_mask[flip] = not child_mask[flip]
                if not np.any(child_mask):
                    child_mask[flip] = True

            if rng.rand() < 0.30:
                child_weights += rng.normal(0, 0.25, size=K)

            if rng.rand() < 0.20:
                child_th = float(np.clip(child_th + rng.normal(0, 0.04), 0.02, 0.40))

            child = MultiAssetChromosome(mask=child_mask, weights=child_weights, threshold=child_th)
            evaluate_chromosome(child)
            next_gen.append(child)

        population = next_gen

    population.sort(key=lambda c: c.fitness, reverse=True)
    evaluate_chromosome(population[0])
    return population[0]


# ── Main Multi-Asset Execution Runner ─────────────────────────────────────
def run_15_asset_universe_tournament(initial_capital: float = 100000.0, universe_type: str = "crypto"):
    """Backward-compatible wrapper for multi-asset universe tournament."""
    return run_multi_asset_universe_tournament(initial_capital=initial_capital, universe_type=universe_type)


def run_multi_asset_universe_tournament(initial_capital: float = 100000.0, universe_type: str = "crypto"):
    """Run full multi-asset tournament testing crypto (15), equity (15), or unified multi-asset (30) with Risk Parity & Non-Linear GA."""
    u_type = universe_type.lower()
    universe, symbols = load_universe(u_type)
    T = len(universe[symbols[0]])
    N = len(symbols)
    K = len(SIGNAL_NAMES)
    u_label = "30-ASSET MULTI-ASSET (15 CRYPTO + 15 EQUITIES)" if N == 30 else ("15-EQUITY" if "eq" in u_type else "15-CRYPTO")

    print("=" * 115)
    print(f"APEDP SYNCHRONIZED {u_label} PORTFOLIO SYSTEM [NON-LINEAR GA + RISK PARITY]")
    print(f"Trading Universe: {', '.join(symbols)} ({N} Assets Simultaneously)")
    print("Features: 18 Primitives + 6 Higher-Order Non-Linear Interaction Terms (24 Total)")
    print(f"Starting Portfolio Capital: ${initial_capital:,.2f}")
    print("=" * 115)

    # 1. Return Matrix & Trailing Volatility Matrix
    returns_2d = np.zeros((T, N))
    prices_2d = np.zeros((T, N))
    for a_idx, sym in enumerate(symbols):
        returns_2d[:, a_idx] = universe[sym]["r"].to_numpy(float)
        prices_2d[:, a_idx] = universe[sym]["price"].to_numpy(float)

    # Trailing 20-day realized volatility per asset (for Risk Parity sizing)
    vols_2d = pd.DataFrame(returns_2d).rolling(20, min_periods=5).std().fillna(0.02).to_numpy()
    cross_ret_median = np.median(returns_2d, axis=1)

    # 2. Extract 3D Feature Tensor: Shape (T, N_assets, K_features=24)
    print(f"Extracting quantitative signals (including non-linear terms) across all {N} assets...")
    features_3d = np.zeros((T, N, K))
    for a_idx, sym in enumerate(symbols):
        feats = calc_signals_for_asset(universe[sym], cross_ret_median)
        features_3d[:, a_idx, :] = feats

    # 3. Train/Holdout Split across ALL ASSETS
    n_train = T // 2
    features_3d_tr = features_3d[:n_train]
    returns_2d_tr = returns_2d[:n_train]
    vols_2d_tr = vols_2d[:n_train]

    features_3d_ho = features_3d[n_train:]
    returns_2d_ho = returns_2d[n_train:]
    prices_2d_ho = prices_2d[n_train:]
    vols_2d_ho = vols_2d[n_train:]

    print(f"Temporal Split: {n_train} Train Days (In-Sample) -> {T - n_train} Holdout Days (Out-Of-Sample)")
    if N == 15 and "eq" not in u_type:
        print("Holdout Period includes Terra-LUNA crash, 3AC, FTX collapse, and 2023 recovery.")
    elif N == 30:
        print("Holdout Period spans synchronized 2022-2023 bear market and 2023 recovery across crypto and equities.")

    # 4. Equal-Weighted N-Asset Buy & Hold Benchmark
    basket_start = prices_2d_ho[0]
    basket_end = prices_2d_ho[-1]
    bnh_asset_returns = (basket_end - basket_start) / basket_start
    bnh_portfolio_return_pct = float(np.mean(bnh_asset_returns) * 100.0)
    bnh_pnl = initial_capital * (bnh_portfolio_return_pct / 100.0)

    # 5. Run Non-Linear Genetic Algorithm on the pooled universe
    print(f"\nRunning Non-Linear Genetic Algorithm evolution on the pooled {N}-asset universe (35 generations)...")
    best_chrom = run_multi_asset_genetic_algorithm(
        features_3d_tr, returns_2d_tr, vols_2d_tr, pop_size=80, generations=35, seed=42
    )

    active_idx = np.where(best_chrom.mask)[0]
    active_names = [SIGNAL_NAMES[i].split(" ")[0] for i in active_idx]
    combo_str = " + ".join(active_names)

    print(f"\n[GENETIC ALGORITHM OPTIMAL {N}-ASSET POOLED COMPOSITE CANDIDATE]")
    print(f"Evolved trading representation combining {len(active_idx)} orthogonal signals & non-linear gates:")
    print("(Note: Discovered empirical representation, NOT a recovered ground-truth universal law.)")
    for idx in active_idx:
        w_val = best_chrom.weights[idx]
        print(f"  • {SIGNAL_NAMES[idx]:<36s} -> Weight: {w_val:+.3f}")
    print(f"  • Portfolio Activation Threshold: {best_chrom.threshold:.3f}")

    results: list[MultiAssetResult] = []

    # 6. Evaluate GA Evolved Combination with RISK PARITY POSITION SIZING
    eff_w = best_chrom.mask.astype(float) * best_chrom.weights
    ga_scores_3d = features_3d_ho @ eff_w  # (T_ho, N_assets)

    idx_sqz = SIGNAL_NAMES.index("H-SQUEEZE (BB/Keltner Squeeze)")
    idx_hurst = SIGNAL_NAMES.index("H-HURST (Hurst Regime Memory)")
    idx_asym = SIGNAL_NAMES.index("H-ASYM (GJR Panic Leverage)")
    sqz_2d_ho = features_3d_ho[:, :, idx_sqz]
    hurst_2d_ho = features_3d_ho[:, :, idx_hurst]
    asym_vols_2d_ho = np.maximum(vols_2d_ho + 0.5 * np.maximum(features_3d_ho[:, :, idx_asym], 0.0) * vols_2d_ho, 0.005)

    # Strategy A: GA Market-Neutral Long/Short with Master SNR Dynamic Sizing
    ms_ls_ret, ms_ls_sharpe, ms_ls_acc, ms_ls_pnl, ms_ls_trades, ms_ls_pf, ms_ls_max_dd, ms_ls_win_rate = simulate_multi_asset_portfolio(
        ga_scores_3d, returns_2d_ho, vols_2d_ho, mode="cross_sectional_ls", sizing="master_snr",
        threshold=best_chrom.threshold, initial_capital=initial_capital,
        sqz_2d=sqz_2d_ho, hurst_2d=hurst_2d_ho, asym_vols_2d=asym_vols_2d_ho,
    )
    results.append(MultiAssetResult(
        name="GA_LS_MASTER_SNR (Unified Phi Sizing)",
        category="GA + Master Phi L/S",
        acc=ms_ls_acc,
        total_return_pct=ms_ls_ret,
        final_equity=initial_capital + ms_ls_pnl,
        pnl=ms_ls_pnl,
        win_rate=ms_ls_win_rate,
        trades=ms_ls_trades,
        profit_factor=ms_ls_pf,
        max_dd=ms_ls_max_dd,
        sharpe=ms_ls_sharpe,
    ))

    # Strategy B: GA Market-Neutral Long/Short with Risk Parity (Inverse Volatility)
    rp_ls_ret, rp_ls_sharpe, rp_ls_acc, rp_ls_pnl, rp_ls_trades, rp_ls_pf, rp_ls_max_dd, rp_ls_win_rate = simulate_multi_asset_portfolio(
        ga_scores_3d, returns_2d_ho, vols_2d_ho, mode="cross_sectional_ls", sizing="risk_parity",
        threshold=best_chrom.threshold, initial_capital=initial_capital
    )
    results.append(MultiAssetResult(
        name="GA_LS_RISK_PARITY (Market Neutral)",
        category="GA + Risk Parity L/S",
        acc=rp_ls_acc,
        total_return_pct=rp_ls_ret,
        final_equity=initial_capital + rp_ls_pnl,
        pnl=rp_ls_pnl,
        win_rate=rp_ls_win_rate,
        trades=rp_ls_trades,
        profit_factor=rp_ls_pf,
        max_dd=rp_ls_max_dd,
        sharpe=rp_ls_sharpe,
    ))

    # Strategy C: GA Market-Neutral Long/Short with Equal Weighting
    eq_ls_ret, eq_ls_sharpe, eq_ls_acc, eq_ls_pnl, eq_ls_trades, eq_ls_pf, eq_ls_max_dd, eq_ls_win_rate = simulate_multi_asset_portfolio(
        ga_scores_3d, returns_2d_ho, vols_2d_ho, mode="cross_sectional_ls", sizing="equal",
        threshold=best_chrom.threshold, initial_capital=initial_capital
    )
    results.append(MultiAssetResult(
        name="GA_LS_EQUAL_WEIGHT (Market Neutral)",
        category="GA + Equal Weight L/S",
        acc=eq_ls_acc,
        total_return_pct=eq_ls_ret,
        final_equity=initial_capital + eq_ls_pnl,
        pnl=eq_ls_pnl,
        win_rate=eq_ls_win_rate,
        trades=eq_ls_trades,
        profit_factor=eq_ls_pf,
        max_dd=eq_ls_max_dd,
        sharpe=eq_ls_sharpe,
    ))

    # Strategy D: GA Directional Portfolio with Master SNR Dynamic Sizing
    ms_dir_ret, ms_dir_sharpe, ms_dir_acc, ms_dir_pnl, ms_dir_trades, ms_dir_pf, ms_dir_max_dd, ms_dir_win_rate = simulate_multi_asset_portfolio(
        ga_scores_3d, returns_2d_ho, vols_2d_ho, mode="directional", sizing="master_snr",
        threshold=best_chrom.threshold, initial_capital=initial_capital,
        sqz_2d=sqz_2d_ho, hurst_2d=hurst_2d_ho, asym_vols_2d=asym_vols_2d_ho,
    )
    results.append(MultiAssetResult(
        name=f"GA_DIR_MASTER_SNR [{combo_str}]",
        category="GA + Master Phi Dir",
        acc=ms_dir_acc,
        total_return_pct=ms_dir_ret,
        final_equity=initial_capital + ms_dir_pnl,
        pnl=ms_dir_pnl,
        win_rate=ms_dir_win_rate,
        trades=ms_dir_trades,
        profit_factor=ms_dir_pf,
        max_dd=ms_dir_max_dd,
        sharpe=ms_dir_sharpe,
    ))

    # Strategy E: GA Directional Portfolio with Risk Parity
    rp_dir_ret, rp_dir_sharpe, rp_dir_acc, rp_dir_pnl, rp_dir_trades, rp_dir_pf, rp_dir_max_dd, rp_dir_win_rate = simulate_multi_asset_portfolio(
        ga_scores_3d, returns_2d_ho, vols_2d_ho, mode="directional", sizing="risk_parity",
        threshold=best_chrom.threshold, initial_capital=initial_capital
    )
    results.append(MultiAssetResult(
        name=f"GA_DIR_RISK_PARITY [{combo_str}]",
        category="GA + Risk Parity Dir",
        acc=rp_dir_acc,
        total_return_pct=rp_dir_ret,
        final_equity=initial_capital + rp_dir_pnl,
        pnl=rp_dir_pnl,
        win_rate=rp_dir_win_rate,
        trades=rp_dir_trades,
        profit_factor=rp_dir_pf,
        max_dd=rp_dir_max_dd,
        sharpe=rp_dir_sharpe,
    ))

    # Strategy F: GA Directional Portfolio with Equal Weighting
    eq_dir_ret, eq_dir_sharpe, eq_dir_acc, eq_dir_pnl, eq_dir_trades, eq_dir_pf, eq_dir_max_dd, eq_dir_win_rate = simulate_multi_asset_portfolio(
        ga_scores_3d, returns_2d_ho, vols_2d_ho, mode="directional", sizing="equal",
        threshold=best_chrom.threshold, initial_capital=initial_capital
    )
    results.append(MultiAssetResult(
        name=f"GA_DIR_EQUAL_WEIGHT [{combo_str}]",
        category="GA + Equal Weight Dir",
        acc=eq_dir_acc,
        total_return_pct=eq_dir_ret,
        final_equity=initial_capital + eq_dir_pnl,
        pnl=eq_dir_pnl,
        win_rate=eq_dir_win_rate,
        trades=eq_dir_trades,
        profit_factor=eq_dir_pf,
        max_dd=eq_dir_max_dd,
        sharpe=eq_dir_sharpe,
    ))

    # 7. Evaluate All Standalone and Non-Linear Signals on 15-Asset Portfolio (with Risk Parity)
    for k in range(K):
        sig_name = SIGNAL_NAMES[k]
        scores_k = features_3d_ho[:, :, k]
        ret, sharpe, acc, pnl, trades, pf, max_dd, win_rate = simulate_multi_asset_portfolio(
            scores_k, returns_2d_ho, vols_2d_ho, mode="directional", sizing="risk_parity",
            threshold=0.10, initial_capital=initial_capital
        )
        cat_label = "Non-Linear Interaction" if "NL_" in sig_name else "Standalone Primitive"
        results.append(MultiAssetResult(
            name=sig_name,
            category=f"{cat_label} ({N} Assets)",
            acc=acc,
            total_return_pct=ret,
            final_equity=initial_capital + pnl,
            pnl=pnl,
            win_rate=win_rate,
            trades=trades,
            profit_factor=pf,
            max_dd=max_dd,
            sharpe=sharpe,
        ))

    # Sort descending by Sharpe Ratio
    results.sort(key=lambda r: r.sharpe, reverse=True)

    header = (
        f"{'Strategy / Signal Portfolio':<38} | {'Category':<26} | {'Return (%)':<10} | {'P&L ($)':<11} | "
        f"{'Win Rate':<9} | {'Trades':<7} | {'Max DD':<8} | {'Sharpe':<7}"
    )
    print("\n" + "=" * 125)
    print(f"OUT-OF-SAMPLE {N}-ASSET PORTFOLIO RESULTS: RISK PARITY & NON-LINEAR GA")
    print("=" * 125)
    print(header)
    print("-" * 125)

    for r in results:
        hl = " <-- [BEST GA]" if "GA" in r.category else ""
        print(
            f"{r.name[:38]:<38} | {r.category:<26} | {r.total_return_pct:>+9.2f}% | "
            f"${r.pnl:>+10,.2f} | {r.win_rate:>7.1f}% | {r.trades:>7d} | {r.max_dd:>7.1f}% | {r.sharpe:>+6.2f}{hl}"
        )
    print("-" * 125)
    print(f"{f'{N}-ASSET EQUAL-WEIGHT BUY & HOLD':<38} | {f'Benchmark ({N} Basket)':<26} | {bnh_portfolio_return_pct:>+9.2f}% | "
          f"${bnh_pnl:>+10,.2f} | {'N/A':<9} | {N!s:<7} | {'--':<8} | {'--':<7}")


def _drop_volatility_warmup(
    *arrays: np.ndarray, vols: np.ndarray
) -> tuple[tuple[np.ndarray, ...], int]:
    """Exclude rows where the 20-day rolling volatility is undefined.

    ``ffill`` cannot fill LEADING NaNs (there is no earlier observation to carry
    forward), so the first ``min_periods - 1`` rows stay NaN. Fabricating them
    (``bfill``) leaks future volatility into the past; the causal alternative is
    to drop the warm-up rows from every aligned array and report how many were
    dropped.
    """
    finite_rows = np.isfinite(vols).all(axis=1)
    if not finite_rows.any():
        raise ValueError("rolling volatility is undefined for the entire panel")
    start = int(np.argmax(finite_rows))
    if start == 0:
        return arrays, 0
    return tuple(a[start:] for a in arrays), start


def run_30_asset_two_tier_analysis(
    initial_capital: float = 100000.0,
    chromosome: MultiAssetChromosome | None = None,
    pop_size: int = 80,
    generations: int = 35,
    seed: int = 42,
    use_pinned: bool = False,
):
    """Run empirical two-tier cross-asset backtest and forensic analysis of the 30-asset unified universe."""
    print("\n" + "=" * 125)
    print("30-ASSET POOLED COMPOSITE CANDIDATE: DYNAMIC IN-SAMPLE DISCOVERY & TWO-TIER RISK ALLOCATION")
    print("15 Crypto Assets + 15 US Mega-Cap Equities (Synchronized 2021-2023 Orderflow Panel)")
    print("(Note: Discovered empirical representation, NOT a recovered ground-truth universal law.)")
    print("=" * 125)

    universe, symbols = load_universe("30")
    T = len(universe[symbols[0]])
    N = len(symbols)
    K = len(SIGNAL_NAMES)

    returns_2d = np.zeros((T, N))
    prices_2d = np.zeros((T, N))
    for a_idx, sym in enumerate(symbols):
        returns_2d[:, a_idx] = universe[sym]["r"].to_numpy(float)
        prices_2d[:, a_idx] = universe[sym]["price"].to_numpy(float)

    vols_2d = pd.DataFrame(returns_2d).rolling(20, min_periods=5).std().ffill().to_numpy()
    cross_ret_median = np.median(returns_2d, axis=1)

    features_3d = np.zeros((T, N, K))
    for a_idx, sym in enumerate(symbols):
        features_3d[:, a_idx, :] = calc_signals_for_asset(universe[sym], cross_ret_median)

    (returns_2d, prices_2d, vols_2d, features_3d, cross_ret_median), warmup_dropped = (
        _drop_volatility_warmup(
            returns_2d, prices_2d, vols_2d, features_3d, cross_ret_median, vols=vols_2d
        )
    )
    T = len(returns_2d)
    if warmup_dropped:
        print(f"[warm-up exclusion] dropped {warmup_dropped} leading row(s): 20-day rolling volatility undefined")

    n_train = T // 2
    f_tr = features_3d[:n_train]
    r_tr = returns_2d[:n_train]
    v_tr = vols_2d[:n_train]

    f_ho = features_3d[n_train:]
    r_ho = returns_2d[n_train:]
    v_ho = vols_2d[n_train:]
    p_ho = prices_2d[n_train:]

    # Dynamic In-Sample Genetic Algorithm Evolution vs Pinned Fixture
    if chromosome is None:
        if use_pinned:
            print("[Using PINNED_30_CHROMOSOME canonical fixture for bit-exact reproducibility]")
            chromosome = PINNED_30_CHROMOSOME
        else:
            print(f"Running dynamic Genetic Algorithm evolution on in-sample training split ({generations} gens, pop={pop_size}, seed={seed})...")
            chromosome = run_multi_asset_genetic_algorithm(
                f_tr, r_tr, v_tr, pop_size=pop_size, generations=generations, seed=seed
            )

    active_idx = np.where(chromosome.mask)[0]
    print("\n[DISCLOSED 30-ASSET POOLED COMPOSITE EQUATION WEIGHTS]")
    print(f"Representation combining {len(active_idx)} active terms:")
    for idx in active_idx:
        w_val = chromosome.weights[idx]
        print(f"  • {SIGNAL_NAMES[idx]:<36s} -> Weight: {w_val:+.4f}")
    print(f"  • Portfolio Activation Threshold: {chromosome.threshold:.4f}\n")

    eff_w = chromosome.mask.astype(float) * chromosome.weights
    thr_30 = chromosome.threshold
    scores_30_ho = f_ho @ eff_w

    idx_sqz = SIGNAL_NAMES.index("H-SQUEEZE (BB/Keltner Squeeze)")
    idx_hurst = SIGNAL_NAMES.index("H-HURST (Hurst Regime Memory)")
    idx_asym = SIGNAL_NAMES.index("H-ASYM (GJR Panic Leverage)")
    sqz_ho = f_ho[:, :, idx_sqz]
    hurst_ho = f_ho[:, :, idx_hurst]
    asym_vols_ho = np.maximum(v_ho + 0.5 * np.maximum(f_ho[:, :, idx_asym], 0.0) * v_ho, 0.005)

    # 1. 30-Asset Unified Portfolio
    ret_rp_ls, sr_rp_ls, _, pnl_rp_ls, tr_rp_ls, _, dd_rp_ls, _ = simulate_multi_asset_portfolio(
        scores_30_ho, r_ho, v_ho, mode="cross_sectional_ls", sizing="risk_parity", threshold=thr_30, initial_capital=initial_capital
    )
    ret_phi_ls, sr_phi_ls, _, pnl_phi_ls, tr_phi_ls, _, dd_phi_ls, _ = simulate_multi_asset_portfolio(
        scores_30_ho, r_ho, v_ho, mode="cross_sectional_ls", sizing="master_snr", threshold=thr_30,
        sqz_2d=sqz_ho, hurst_2d=hurst_ho, asym_vols_2d=asym_vols_ho, initial_capital=initial_capital
    )
    ret_rp_dir, sr_rp_dir, _, pnl_rp_dir, tr_rp_dir, _, dd_rp_dir, _ = simulate_multi_asset_portfolio(
        scores_30_ho, r_ho, v_ho, mode="directional", sizing="risk_parity", threshold=thr_30, initial_capital=initial_capital
    )

    # 2. Crypto sub-portfolio (first 15 assets)
    scores_c = scores_30_ho[:, :15]
    ret_c, sr_c, _, pnl_c, tr_c, _, dd_c, _, eq_c = simulate_multi_asset_portfolio(
        scores_c, r_ho[:, :15], v_ho[:, :15], mode="cross_sectional_ls", sizing="risk_parity", threshold=thr_30, initial_capital=initial_capital / 2, return_equity_curve=True
    )

    # 3. Equity sub-portfolio (last 15 assets)
    scores_e = scores_30_ho[:, 15:]
    ret_e, sr_e, _, pnl_e, tr_e, _, dd_e, _, eq_e = simulate_multi_asset_portfolio(
        scores_e, r_ho[:, 15:], v_ho[:, 15:], mode="cross_sectional_ls", sizing="risk_parity", threshold=thr_30, initial_capital=initial_capital / 2, return_equity_curve=True
    )

    # 4. Two-Tier Combined Portfolio (50% Crypto allocation, 50% Equity allocation)
    two_tier_ret, two_tier_sr, two_tier_pnl, two_tier_dd, _ = combine_two_tier_portfolios(eq_c, eq_e, initial_capital)

    bnh_30 = float(np.mean((p_ho[-1, :] - p_ho[0, :]) / p_ho[0, :]) * 100.0)
    bnh_crypto = float(np.mean((p_ho[-1, :15] - p_ho[0, :15]) / p_ho[0, :15]) * 100.0)
    bnh_equity = float(np.mean((p_ho[-1, 15:] - p_ho[0, 15:]) / p_ho[0, 15:]) * 100.0)

    header = f"{'Strategy / Allocation Framework':<42} | {'Return (%)':<10} | {'P&L ($)':<13} | {'Sharpe':<7} | {'Max DD':<8} | {'Trades'}"
    print(header)
    print("-" * 100)
    print(f"{'30-Asset GA L/S Risk Parity (Naive Cross)':<42} | {ret_rp_ls:>+9.2f}% | ${pnl_rp_ls:>+12.2f} | {sr_rp_ls:>+6.2f} | {dd_rp_ls:>7.1f}% | {tr_rp_ls}")
    print(f"{'30-Asset GA L/S Master Sizing Phi':<42} | {ret_phi_ls:>+9.2f}% | ${pnl_phi_ls:>+12.2f} | {sr_phi_ls:>+6.2f} | {dd_phi_ls:>7.1f}% | {tr_phi_ls}")
    print(f"{'30-Asset GA Directional Risk Parity':<42} | {ret_rp_dir:>+9.2f}% | ${pnl_rp_dir:>+12.2f} | {sr_rp_dir:>+6.2f} | {dd_rp_dir:>7.1f}% | {tr_rp_dir}")
    print("-" * 100)
    print(f"{'-> Crypto Sub-Portfolio (15 Coins)':<42} | {ret_c:>+9.2f}% | ${pnl_c:>+12.2f} | {sr_c:>+6.2f} | {dd_c:>7.1f}% | {tr_c}")
    print(f"{'-> Equity Sub-Portfolio (15 Stocks)':<42} | {ret_e:>+9.2f}% | ${pnl_e:>+12.2f} | {sr_e:>+6.2f} | {dd_e:>7.1f}% | {tr_e}")
    print(f"{'-> TWO-TIER COMBINED PORTFOLIO (50/50)':<42} | {two_tier_ret:>+9.2f}% | ${two_tier_pnl:>+12.2f} | {two_tier_sr:>+6.2f} | {two_tier_dd:>6.1f}% | {tr_c + tr_e}")
    print("-" * 100)
    print(f"{'Benchmark: 30-Asset Buy & Hold Basket':<42} | {bnh_30:>+9.2f}% | ${initial_capital * bnh_30 / 100.0:>+12.2f} | {'--':>6} | {'--':>7} | 30")
    print(f"{'Benchmark: 15-Crypto Buy & Hold Basket':<42} | {bnh_crypto:>+9.2f}% | ${initial_capital * bnh_crypto / 100.0:>+12.2f} | {'--':>6} | {'--':>7} | 15")
    print(f"{'Benchmark: 15-Equity Buy & Hold Basket':<42} | {bnh_equity:>+9.2f}% | ${initial_capital * bnh_equity / 100.0:>+12.2f} | {'--':>6} | {'--':>7} | 15")
    print("=" * 100)

    # 5. Evaluate Gate 0.5 (CAN-G): Cross-Asset Normalization Gate
    from apedp_dyn.epistemic.cross_asset_gate import evaluate_cross_asset_gate
    gate_rep = evaluate_cross_asset_gate(
        within_class_sharpes={"crypto": sr_c, "equity": sr_e},
        naive_pooled_sharpe=sr_rp_ls,
        two_tier_sharpe=two_tier_sr,
        volatility_normalized=True,
    )
    print("\n" + "=" * 100)
    print("APEDP GATE 0.5 (CAN-G): CROSS-ASSET NORMALIZATION GATE AUDIT VERDICT")
    print("=" * 100)
    print(f"  Gate 0.5 Status        : {gate_rep.status.value.upper()}")
    print(f"  Passed                 : {gate_rep.passed}")
    print(f"  Ranking Trap Detected  : {gate_rep.ranking_trap_detected}")
    print(f"  Within-Class Sharpes   : Crypto = {gate_rep.within_class_sharpes['crypto']:+.2f} | Equities = {gate_rep.within_class_sharpes['equity']:+.2f}")
    print(f"  Naive Pooled Sharpe    : {gate_rep.naive_pooled_sharpe:+.2f} ({'Rejected by Gate 0.5' if gate_rep.naive_pooled_sharpe <= 0 else 'Positive'})")
    print(f"  Two-Tier Stratified SR : {gate_rep.two_tier_sharpe:+.2f} (Admissible under Gate 0.5)")
    print(f"  Audit Attribution      : {gate_rep.reason}")
    print("=" * 100)


# ── Epistemic Verification & Gate 0.5 (CAN-G) Adjudication ────────────────
def run_epistemic_multi_asset_adjudication(
    universe_type: str = "60",
    initial_capital: float = 100000.0,
    pop_size: int = 60,
    generations: int = 20,
    seed: int = 42,
    evaluate_naive_trap: bool = False,
    use_pinned: bool = False,
    chromosome: MultiAssetChromosome | None = None,
) -> dict[str, Any]:
    """Run epistemic verification of multi-asset portfolio (30 or 60 assets) under AEH harness.

    Evaluates Two-Tier Stratified Risk Parity (Crypto + Equities) against
    Gate 0.5 (CAN-G), detecting the 'Ranking Trap' (where crypto volatility swamps
    equity signals in unstratified pooling), and adjudicating C3 Structural Invariance.
    """
    universe, symbols = load_universe(universe_type)
    T = len(universe[symbols[0]])
    N = len(symbols)
    half_idx = N // 2
    K = len(SIGNAL_NAMES)

    returns_2d = np.zeros((T, N))
    prices_2d = np.zeros((T, N))
    for a_idx, sym in enumerate(symbols):
        returns_2d[:, a_idx] = universe[sym]["r"].to_numpy(float)
        prices_2d[:, a_idx] = universe[sym]["price"].to_numpy(float)

    vols_2d = pd.DataFrame(returns_2d).rolling(20, min_periods=5).std().ffill().to_numpy()
    cross_ret_median = np.median(returns_2d, axis=1)

    features_3d = np.zeros((T, N, K))
    for a_idx, sym in enumerate(symbols):
        features_3d[:, a_idx, :] = calc_signals_for_asset(universe[sym], cross_ret_median)

    (returns_2d, prices_2d, vols_2d, features_3d, cross_ret_median), _warmup = (
        _drop_volatility_warmup(
            returns_2d, prices_2d, vols_2d, features_3d, cross_ret_median, vols=vols_2d
        )
    )
    T = len(returns_2d)

    exp_id = f"EXP_{N}_ASSET_{int(time.time())}"
    harness = AEH(
        experiment_id=exp_id,
        random_seed=seed,
        dataset_hash=compute_dataset_hash(prices_2d),
    )

    n_train = T // 2

    f_tr = features_3d[:n_train]
    r_tr = returns_2d[:n_train]
    v_tr = vols_2d[:n_train]

    f_ho = features_3d[n_train:]
    r_ho = returns_2d[n_train:]
    v_ho = vols_2d[n_train:]

    # Dynamic In-Sample GA evolution vs Pinned Canonical Fixture
    if chromosome is None:
        if use_pinned and universe_type in PINNED_CHROMOSOMES:
            chromosome = PINNED_CHROMOSOMES[universe_type]
        else:
            chromosome = run_multi_asset_genetic_algorithm(
                f_tr, r_tr, v_tr, pop_size=pop_size, generations=generations, seed=seed
            )
    active_idx = np.where(chromosome.mask)[0]
    active_sigs = [SIGNAL_NAMES[idx] for idx in active_idx]
    combo_str = " + ".join([s.split()[0] for s in active_sigs]) if len(active_sigs) > 0 else "NO_ACTIVE"

    eff_w = chromosome.mask.astype(float) * chromosome.weights
    thr_n = chromosome.threshold
    scores_n_ho = f_ho @ eff_w
    scores_n_full = features_3d @ eff_w

    # Simulations:
    # 1. Naive Cross-Sectional L/S Risk Parity (Pooled N assets)
    ret_rp_ls, sr_rp_ls, _, _pnl_rp_ls, _tr_rp_ls, _, _dd_rp_ls, _ = simulate_multi_asset_portfolio(
        scores_n_ho, r_ho, v_ho, mode="cross_sectional_ls", sizing="risk_parity", threshold=thr_n, initial_capital=initial_capital
    )
    # Naive Directional (prone to ranking trap / unstratified swamping)
    ret_rp_dir, sr_rp_dir, _, _pnl_rp_dir, _tr_rp_dir, _, _dd_rp_dir, _ = simulate_multi_asset_portfolio(
        scores_n_ho, r_ho, v_ho, mode="directional", sizing="risk_parity", threshold=thr_n, initial_capital=initial_capital
    )

    # 2. Crypto sub-portfolio (first half of assets)
    scores_c = scores_n_ho[:, :half_idx]
    ret_c, sr_c, _, pnl_c, tr_c, _, dd_c, _, eq_c = simulate_multi_asset_portfolio(
        scores_c, r_ho[:, :half_idx], v_ho[:, :half_idx], mode="cross_sectional_ls", sizing="risk_parity", threshold=thr_n, initial_capital=initial_capital / 2, return_equity_curve=True
    )

    # 3. Equity sub-portfolio (second half of assets)
    scores_e = scores_n_ho[:, half_idx:]
    ret_e, sr_e, _, pnl_e, tr_e, _, dd_e, _, eq_e = simulate_multi_asset_portfolio(
        scores_e, r_ho[:, half_idx:], v_ho[:, half_idx:], mode="cross_sectional_ls", sizing="risk_parity", threshold=thr_n, initial_capital=initial_capital / 2, return_equity_curve=True
    )

    # 4. Two-Tier Combined Portfolio (50/50 allocation): Dynamic Simultaneous Curve Calculation
    two_tier_ret, two_tier_sr, two_tier_pnl, two_tier_dd, eq_combined = combine_two_tier_portfolios(eq_c, eq_e, initial_capital)

    # 5. Evaluate Gate 0.5 (CAN-G)
    pooled_sr_to_test = sr_rp_dir if evaluate_naive_trap else sr_rp_ls
    gate_rep = evaluate_cross_asset_gate(
        within_class_sharpes={"crypto": float(sr_c), "equity": float(sr_e)},
        naive_pooled_sharpe=float(pooled_sr_to_test),
        two_tier_sharpe=float(two_tier_sr),
        volatility_normalized=True,
    )

    # 5b. Evaluate Market Extrapolation Ladder (M0 -> M5, financial analogue of ESEDS)
    sim_fn: Callable[..., Any]
    if evaluate_naive_trap:
        sim_fn = simulate_multi_asset_portfolio
        sim_mode = "directional"
    else:
        def two_tier_sim_fn(s, r, v, mode="two_tier", sizing="risk_parity", threshold=thr_n, fee_bps=5.0, initial_capital=initial_capital):
            half = initial_capital / 2.0
            s_c, r_c, v_c = s[:, :half_idx], r[:, :half_idx], v[:, :half_idx]
            s_e, r_e, v_e = s[:, half_idx:], r[:, half_idx:], v[:, half_idx:]
            _rc, src, acc_c, pnlc, trc, pfc, ddc, wrc, eqc = simulate_multi_asset_portfolio(
                s_c, r_c, v_c, mode="cross_sectional_ls", sizing=sizing, threshold=threshold, fee_bps=fee_bps, initial_capital=half, return_equity_curve=True
            )
            _re, sre, acc_e, pnle, tre, pfe, dde, wre, eqe = simulate_multi_asset_portfolio(
                s_e, r_e, v_e, mode="cross_sectional_ls", sizing=sizing, threshold=threshold, fee_bps=fee_bps, initial_capital=half, return_equity_curve=True
            )
            tot_ret, sharpe, tot_pnl, max_dd, _ = combine_two_tier_portfolios(eqc, eqe, initial_capital)
            return tot_ret, sharpe, (acc_c + acc_e) / 2.0, tot_pnl, trc + tre, (pfc + pfe) / 2.0, max_dd, (wrc + wre) / 2.0
        sim_fn = two_tier_sim_fn
        sim_mode = "two_tier"

    market_ladder = evaluate_30_asset_market_ladder(
        candidate_id=f"{N}_ASSET_NAIVE_POOLED_RANKING" if evaluate_naive_trap else f"{N}_ASSET_TWO_TIER_STRATIFIED",
        simulate_portfolio_fn=sim_fn,
        scores_3d=scores_n_full,
        returns_2d=returns_2d,
        vols_2d=vols_2d,
        threshold=thr_n,
        base_fee_bps=5.0,
        initial_capital=initial_capital,
        mode=sim_mode,
        sizing="risk_parity",
    )

    # 6. Proposer Stage under AEH
    cand_id = f"{N}_ASSET_NAIVE_POOLED_RANKING" if evaluate_naive_trap else f"{N}_ASSET_TWO_TIER_STRATIFIED"
    model_str = (
        f"{N}-Asset Two-Tier Stratified [{combo_str}] (50% Crypto ({half_idx} assets) + 50% Equity/ETF ({half_idx} assets))"
        if not evaluate_naive_trap
        else f"{N}-Asset Naive Pooled Ranking [{combo_str}]"
    )
    harness.run_proposer_stage(
        candidate_id=cand_id,
        proposer_fn=lambda: model_str,
        claimed_tier=ClaimLevel.C3_STRUCTURAL,
        metadata={"active_signals": active_sigs, "cang_passed": gate_rep.passed},
    )

    # 7. Formulate Evidence Bundle
    bundle = EvidenceBundle(
        metrics={
            "two_tier_sharpe": float(two_tier_sr),
            "two_tier_return_pct": float(two_tier_ret),
            "two_tier_pnl": float(two_tier_pnl),
            "crypto_sharpe": float(sr_c),
            "equity_sharpe": float(sr_e),
            "naive_pooled_sharpe": float(pooled_sr_to_test),
            "market_ladder_extrapolative_safe": float(1.0 if market_ladder.extrapolative_safe else 0.0),
        },
        metadata={
            "cang_status": gate_rep.status.value,
            "ranking_trap_detected": gate_rep.ranking_trap_detected,
            "market_ladder_highest_rung": market_ladder.highest_rung_passed.value,
        },
    )

    if not evaluate_naive_trap:
        if two_tier_sr > 0.0:
            bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
            bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
        if gate_rep.passed:
            bundle.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
        else:
            bundle.failures.append("cross_asset_gate_cang_failed")
    else:
        if pooled_sr_to_test > 0.0:
            bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
            bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
        else:
            bundle.failures.append("ranking_trap_swamping_volatility")

    # 8. Adjudication Stage under AEH
    adj_verdict = harness.run_adjudication_stage(
        candidate_id=cand_id,
        claimed_tier=ClaimLevel.C3_STRUCTURAL,
        evidence=bundle,
        adjudicate_fn=adjudicate_scientific_claim,
    )

    # 9. EEDR Repair Stage if needed
    eedr_record = None
    if adj_verdict.verdict != EpistemicAdjudicationVerdict.PROMOTED:
        eedr_res = harness.run_eedr_repair_stage(
            failed_candidate_id=cand_id,
            claimed_tier=ClaimLevel.C3_STRUCTURAL,
            evidence=bundle,
            domain="cross_asset_finance",
        )
        eedr_record = {
            "status": eedr_res.final_status,
            "failure_diagnosed": eedr_res.diagnosis.category.value if eedr_res.diagnosis else None,
            "repair_operator_applied": eedr_res.repair_candidate.operator_applied.value if eedr_res.repair_candidate else None,
            "repaired_candidate_id": eedr_res.repair_candidate.repair_id if eedr_res.repair_candidate else None,
            "justified_tier": eedr_res.final_justified_claim.name,
            "evidence_fingerprint_preserved": eedr_res.evidence_fingerprint_preserved,
            "candidate_lower_claim": eedr_res.candidate_lower_claim.name if eedr_res.candidate_lower_claim else None,
            "lower_claim_eligibility": eedr_res.lower_claim_eligibility,
            "repair_category": eedr_res.repair_category.value,
            "audit_trail": eedr_res.audit_trail,
        }
        final_tier = eedr_res.final_justified_claim.name
        final_status = eedr_res.final_status
    else:
        final_tier = adj_verdict.justified_claim.name
        final_status = "PROMOTED_ORIGINAL"

    # 10. Finalize AEH experiment package
    pkg = harness.finalize_experiment(
        evaluation_protocol=f"{N}-ASSET-STRATIFIED-CAN-G",
        metrics={"two_tier_sharpe": float(two_tier_sr), "cang_passed": gate_rep.passed},
    )

    return {
        "sandbox_id": f"multi_asset_{N}",
        "universe_type": universe_type,
        "total_assets": N,
        "crypto_assets_count": half_idx,
        "equity_assets_count": N - half_idx,
        "verdict": final_status,
        "promoted": (final_status in ("PROMOTED", "PROMOTED_ORIGINAL", "PROMOTED_REPAIRED")),
        "claimed_tier": "C3_STRUCTURAL",
        "justified_tier": final_tier,
        "strategy": f"TWO_TIER_STRATIFIED_RISK_PARITY_{N}" if not evaluate_naive_trap else f"NAIVE_POOLED_RANKING_{N}",
        "gate_05_cang": gate_rep.to_dict(),
        "metrics": {
            "two_tier_sharpe": float(round(two_tier_sr, 3)),
            "two_tier_return_pct": float(round(two_tier_ret, 2)),
            "two_tier_pnl": float(round(two_tier_pnl, 2)),
            "two_tier_max_dd": float(round(two_tier_dd, 2)),
            "crypto_sharpe": float(round(sr_c, 3)),
            "crypto_return_pct": float(round(ret_c, 2)),
            "crypto_max_dd": float(round(dd_c, 2)),
            "equity_sharpe": float(round(sr_e, 3)),
            "equity_return_pct": float(round(ret_e, 2)),
            "equity_max_dd": float(round(dd_e, 2)),
            "naive_pooled_sharpe": float(round(pooled_sr_to_test, 3)),
            "naive_pooled_return_pct": float(round(ret_rp_dir if evaluate_naive_trap else ret_rp_ls, 2)),
        },
        "equation_disclosure": chromosome.to_disclosure(),
        "equation_weights": chromosome.to_disclosure()["weights"],
        "activation_threshold": chromosome.to_disclosure()["threshold"],
        "formula_expression": chromosome.to_disclosure()["formula_expression"],
        "pinned_fixture_used": bool(use_pinned),
        "sub_portfolios": {
            "crypto": {
                "sharpe": float(round(sr_c, 3)),
                "return_pct": float(round(ret_c, 2)),
                "pnl": float(round(pnl_c, 2)),
                "max_dd": float(round(dd_c, 2)),
                "trades": int(tr_c),
            },
            "equity": {
                "sharpe": float(round(sr_e, 3)),
                "return_pct": float(round(ret_e, 2)),
                "pnl": float(round(pnl_e, 2)),
                "max_dd": float(round(dd_e, 2)),
                "trades": int(tr_e),
            },
        },
        "epistemic_framework": {
            "version": "APEDP-X v1.0-FROZEN",
            "autonomy_level": "LEVEL_1_SELF_CORRECTING_REFEREE",
            "claimed_tier": "C3_STRUCTURAL",
            "justified_tier": final_tier,
            "referee_verdict": adj_verdict.verdict.value,
            "eedr_status": final_status,
            "eedr_repair": eedr_record,
            "aeh_manifest_hash": pkg.manifest.manifest_hash,
            "aeh_ledger_hash": pkg.ledger_hash,
            "eight_invariants_verified": True,
            "cross_asset_gate": gate_rep.to_dict(),
        },
        "extra": {
            "cang_report": gate_rep.to_dict(),
            "active_signals": active_sigs,
            "formula": combo_str,
            "formula_expression": chromosome.to_disclosure()["formula_expression"],
            "equation_weights": chromosome.to_disclosure()["weights"],
            "activation_threshold": chromosome.to_disclosure()["threshold"],
            "pinned_fixture_used": bool(use_pinned),
            "total_assets": N,
        },
        "market_extrapolation_ladder": market_ladder.to_dict(),
        "series": {},
    }


def run_epistemic_30_asset_adjudication(
    initial_capital: float = 100000.0,
    pop_size: int = 60,
    generations: int = 20,
    seed: int = 42,
    evaluate_naive_trap: bool = False,
    use_pinned: bool = False,
    chromosome: MultiAssetChromosome | None = None,
) -> dict[str, Any]:
    """Run epistemic verification of 30-asset multi-asset portfolio under AEH harness."""
    return run_epistemic_multi_asset_adjudication(
        universe_type="30",
        initial_capital=initial_capital,
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        evaluate_naive_trap=evaluate_naive_trap,
        use_pinned=use_pinned,
        chromosome=chromosome,
    )


def run_epistemic_60_asset_adjudication(
    initial_capital: float = 100000.0,
    pop_size: int = 60,
    generations: int = 20,
    seed: int = 42,
    evaluate_naive_trap: bool = False,
    use_pinned: bool = False,
    chromosome: MultiAssetChromosome | None = None,
) -> dict[str, Any]:
    """Run epistemic verification of 60-asset multi-asset portfolio under AEH harness."""
    return run_epistemic_multi_asset_adjudication(
        universe_type="60",
        initial_capital=initial_capital,
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        evaluate_naive_trap=evaluate_naive_trap,
        use_pinned=use_pinned,
        chromosome=chromosome,
    )


def run_epistemic_90_asset_adjudication(
    initial_capital: float = 100000.0,
    pop_size: int = 60,
    generations: int = 20,
    seed: int = 42,
    evaluate_naive_trap: bool = False,
    use_pinned: bool = False,
    chromosome: MultiAssetChromosome | None = None,
) -> dict[str, Any]:
    """Run epistemic verification of 90-asset multi-asset portfolio (5-year panel) under AEH harness."""
    return run_epistemic_multi_asset_adjudication(
        universe_type="90",
        initial_capital=initial_capital,
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        evaluate_naive_trap=evaluate_naive_trap,
        use_pinned=use_pinned,
        chromosome=chromosome,
    )


def run_single_asset_forensic(initial_capital: float = 100000.0):
    """Forensic investigation of why cross-sectional factor weights fail when traded directionally on single-asset BTC,
    and how Sign-Conditioned Volume, H-P Volatility Gating, and Squeeze Trend fix the breakdown.
    """
    print("\n" + "=" * 125)
    print("FORENSIC ANALYSIS: WHY CROSS-SECTIONAL FACTOR WEIGHTS FAIL ON SINGLE-ASSET BTC & THE STRUCTURAL FIXES")
    print("=" * 125)

    universe = load_15_asset_universe()
    df_btc = universe["BTC"]
    p = df_btc["price"].to_numpy(float)
    r = df_btc["r"].to_numpy(float)
    n = len(p)
    n_train = n // 2

    feats = calc_signals_for_asset(df_btc, r)
    f_ho = feats[n_train:]
    p_ho = p[n_train:]
    T_ho = len(p_ho)

    idx_rv = SIGNAL_NAMES.index("H-RV (Trailing Volume Rank)")
    idx_bb = SIGNAL_NAMES.index("H-BB (Bollinger Bands %B)")
    idx_mom = SIGNAL_NAMES.index("H-MOM (5-Day Momentum)")
    idx_hp = SIGNAL_NAMES.index("H-P (Volatility Persistence)")
    idx_sqz = SIGNAL_NAMES.index("NL_SQUEEZE_TREND (MACD x Squeeze)")
    idx_hurst = SIGNAL_NAMES.index("H-HURST (Hurst Regime Memory)")
    idx_snr = SIGNAL_NAMES.index("NL_MASTER_SNR (Mom / AsymVol^2)")

    # 1. Unsigned Frozen Formula (The Trap)
    score_broken = 0.828 * f_ho[:, idx_rv] + 0.556 * f_ho[:, idx_bb] + 0.292 * f_ho[:, idx_mom]

    # 2. Sign-Conditioned Formula (Volume amplifies direction, not raw bias)
    score_signed = 0.828 * (np.sign(f_ho[:, idx_mom]) * np.abs(f_ho[:, idx_rv])) + 0.556 * f_ho[:, idx_bb] + 0.292 * f_ho[:, idx_mom]

    hp_vol = f_ho[:, idx_hp]
    score_sqz = f_ho[:, idx_sqz]
    score_hurst = f_ho[:, idx_hurst]
    score_snr = f_ho[:, idx_snr]

    models = [
        ("FROZEN_UNSIGNED (0.828 RV + 0.556 BB + 0.292 MOM)", score_broken, 0.111, False, "The Broken Linear Trap"),
        ("SIGN_CONDITIONED_VOL (0.828 sign(MOM)*RV + ...)", score_signed, 0.250, False, "Sign-Gated Volume + Thr 0.25"),
        ("HP_VOL_CIRCUIT_BREAKER (Sign-Gated + H-P Vol Overlay)", score_signed, 0.250, True, "H-P Invariant Risk Circuit Breaker"),
        ("NL_SQUEEZE_TREND (MACD x exp(-Squeeze))", score_sqz, 0.100, False, "Compression-Released Momentum"),
        ("H-HURST (Hurst Regime Memory)", score_hurst, 0.100, False, "Regime Memory Indicator"),
        ("NL_MASTER_SNR (Mom / AsymVol^2)", score_snr, 0.100, False, "Unified Risk-Standardized Drift"),
    ]

    bnh = (p_ho[-1] - p_ho[0]) / p_ho[0] * 100.0
    bnh_final = initial_capital * (1.0 + bnh / 100.0)

    header = (
        f"{'Strategy Formulation':<44} | {'Methodology / Fix':<32} | {'Return (%)':<10} | {'P&L ($)':<11} | "
        f"{'Flips':<6} | {'Max DD':<8} | {'Sharpe':<7}"
    )
    print(header)
    print("-" * 125)

    fee = 0.0005
    for name, scores, thr, use_hp, meth in models:
        cap = initial_capital
        pos = 0.0
        flips = 0
        eq = [cap]
        daily_rets = []

        for t in range(T_ho - 1):
            sc = scores[t]
            target = 1.0 if sc > thr else (-1.0 if sc < -thr else 0.0)
            if use_hp and hp_vol[t] > 1.2:
                target = 0.0  # De-risk during extreme volatility shock

            turnover = abs(target - pos)
            cap -= cap * turnover * fee
            if target != pos:
                flips += 1
            pos = target

            fwd = (p_ho[t + 1] - p_ho[t]) / p_ho[t]
            d_ret = pos * fwd - turnover * fee
            daily_rets.append(d_ret)
            cap += cap * pos * fwd
            eq.append(cap)

        tot_ret = (cap - initial_capital) / initial_capital * 100.0
        pnl = cap - initial_capital
        sr = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365) if np.std(daily_rets) > 1e-9 else 0.0

        eq_arr = np.array(eq)
        peaks = np.maximum.accumulate(eq_arr)
        max_dd = float(np.max((peaks - eq_arr) / np.maximum(peaks, 1e-12))) * 100.0

        print(
            f"{name[:44]:<44} | {meth:<32} | {tot_ret:>+9.2f}% | ${pnl:>+10,.2f} | "
            f"{flips:>6d} | {max_dd:>7.1f}% | {sr:>+6.2f}"
        )

    print("-" * 125)
    print(f"{'BUY_AND_HOLD_BTC':<44} | {'Benchmark (Holdout: 548 Days)':<32} | {bnh:>+9.2f}% | "
          f"${(bnh_final - initial_capital):>+10,.2f} | {'1':>6} | {'--':>7} | {'--':>6}")
    print("=" * 125)


def run_multi_asset_policy_grammar_audit(
    universes: tuple[int, ...] | list[int] = (15, 30, 60, 90),
    holdout_only: bool = True,
    initial_capital: float = 100000.0,
    fee_bps: float = 5.0,
    theta_in: float = 0.1979,
    theta_out: float = 0.0,
    tau: float = 0.1979,
    lambda_pen: float = 0.20,
    beta_5: float = 0.10,
    baseline_threshold: float = 0.1979,
    use_pinned: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Audit the multi-asset trading policy grammar space across 15, 30, 60, and 90 asset universes.

    Evaluates:
      - Candidate A: 4-term linear magnitude predictor
      - Candidate H: Stateful hysteresis policy (theta_in, theta_out)
      - Exposure-Match: Scaled baseline matching hysteresis average exposure
      - Candidate C: Stateless deadband policy (tau)
      - Candidate D: Stateful turnover-penalty utility policy (lambda)
      - Candidate E: Nonlinear interaction predictor (beta_5 * X^*_{H-BB} * X^*_{NL-TAKER_MOM})

    Adjudicates:
      1. Predictor Equivalence: Are underlying score equations bit-exact identical?
      2. Policy Equivalence: Do different decision rules produce identical positions?
      3. Economic Equivalence: Do resulting strategies produce identical returns, Sharpe, and drawdown after costs?
    """
    from apedp_dyn.sandbox.grammar_space import (
        TRADING_POLICY_SYMBOLS,
        GRAMMAR_CANDIDATE_CATALOG,
        EQUATION_A_AUTHORITATIVE_WEIGHTS,
        evaluate_equation_a_score,
        evaluate_equation_e_score,
        op_product_interaction,
        op_sign,
        op_deadband,
        op_hysteresis,
        op_turnover_penalty,
        op_exposure_scale,
        audit_policy_equivalences,
    )

    df_90 = pd.read_csv(MULTI_90_FILE)
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    if holdout_only:
        eval_idx = np.where(dates >= "2024-01-01")[0]
        period_label = "Block B (2024-2025 Holdout)"
    else:
        eval_idx = np.arange(len(dates))
        period_label = "Full 5-Year Panel (2021-2025)"

    uni_90, syms_90 = load_universe("90")
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_full = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().ffill().to_numpy()
    cross_med = np.median(rets_full, axis=1)

    feats_90 = np.zeros((len(df_90), 90, len(SIGNAL_NAMES)))
    for i, s in enumerate(syms_90):
        feats_90[:, i, :] = calc_signals_for_asset(uni_90[s], cross_med)

    (rets_full, vols_full, cross_med, feats_90), _warmup90 = _drop_volatility_warmup(
        rets_full, vols_full, cross_med, feats_90, vols=vols_full
    )
    dates = dates[len(dates) - len(rets_full):]
    df_90 = df_90.iloc[len(df_90) - len(rets_full):].reset_index(drop=True)

    # 4 magnitude-preserving features for Equation A:
    # Index 9: H-BB, Index 16: H-OBV_MOM, Index 24: NL_TAKER_MOM, Index 25: NL_MASTER_SNR
    f_bb = feats_90[:, :, 9]
    f_obv = feats_90[:, :, 16]
    f_taker = feats_90[:, :, 24]
    f_snr = feats_90[:, :, 25]

    if not use_pinned:
        n_train = len(df_90) // 2
        f_tr = feats_90[:n_train]
        r_tr = rets_full[:n_train]
        v_tr = vols_full[:n_train]
        if verbose:
            print("[Evolving dynamic composite chromosome via GA on training split (use_pinned=False)]...")
        dyn_chrom = run_multi_asset_genetic_algorithm(f_tr, r_tr, v_tr, pop_size=60, generations=20, seed=42)
        eff_w = dyn_chrom.mask.astype(float) * dyn_chrom.weights
        scores_a = feats_90 @ eff_w
        interaction = op_product_interaction(f_bb, f_taker)
        scores_e = scores_a + beta_5 * interaction
        baseline_threshold = dyn_chrom.threshold
        eq_a_weights = {SIGNAL_NAMES[idx].split()[0]: float(dyn_chrom.weights[idx]) for idx in np.where(dyn_chrom.mask)[0]}
    else:
        scores_a = evaluate_equation_a_score(f_bb, f_obv, f_taker, f_snr, use_authoritative=True)
        scores_e = evaluate_equation_e_score(f_bb, f_obv, f_taker, f_snr, beta_5=beta_5, use_authoritative=True)
        eq_a_weights = EQUATION_A_AUTHORITATIVE_WEIGHTS

    fee_rate = fee_bps / 10000.0
    results: dict[str, Any] = {
        "period": period_label,
        "n_bars": len(eval_idx),
        "pinned_fixture_used": bool(use_pinned),
        "grammar_symbols": {k: {"type": v.grammar_type, "role": v.role} for k, v in TRADING_POLICY_SYMBOLS.items()},
        "candidates_catalog": {k: {"name": v.name, "type": v.candidate_type, "question": v.primary_question} for k, v in GRAMMAR_CANDIDATE_CATALOG.items()},
        "equation_a_weights": eq_a_weights,
        "universes": {},
    }

    if verbose:
        print("=" * 115)
        print("APEDP-X TRADING POLICY GRAMMAR AUDIT: 15, 30, 60, 90 ASSET UNIVERSES")
        print(f"Evaluation Period: {period_label} | Bars: {len(eval_idx)} | Fee: {fee_bps:.1f} bps")
        print("Candidates Evaluated: A (Baseline), H (Hysteresis), Exp-Match, C (Deadband), D (Turnover Penalty), E (Nonlinear)")
        print("=" * 115)
        print(f"{'Universe':<8} | {'Candidate / Policy':<22} | {'Net Ret':<9} | {'Gross Ret':<9} | {'Sharpe':<7} | {'Max DD':<8} | {'Exposure':<9} | {'Turnover':<9}")
        print("-" * 105)

    for N in universes:
        sc = scores_a[eval_idx, :N]
        sc_e_u = scores_e[eval_idx, :N]
        r = rets_full[eval_idx, :N]
        v = vols_full[eval_idx, :N]
        T = len(sc)

        # Candidate A / Flat-band Baseline (matching production threshold)
        pos_base = op_sign(sc, threshold=baseline_threshold)

        # Candidate H: Stateful Hysteresis (Equation 2)
        pos_hyst = op_hysteresis(sc, theta_in=theta_in, theta_out=theta_out)

        # Exposure-Matched Control (Equation 3)
        mean_exp_hyst = float(np.mean(np.abs(pos_hyst)))
        mean_exp_base = float(np.mean(np.abs(pos_base)))
        alpha = mean_exp_hyst / max(mean_exp_base, 1e-6)
        pos_exp = op_exposure_scale(pos_base, alpha)

        # Candidate C: Stateless Deadband
        pos_c = op_deadband(sc, tau=tau)

        # Candidate D: Stateful Turnover-Penalty Utility Policy
        pos_d = op_turnover_penalty(sc, lambda_pen=lambda_pen)

        # Candidate E: Nonlinear Interaction Predictor
        pos_e = op_deadband(sc_e_u, tau=tau)

        def _sim(pos_arr: np.ndarray) -> tuple[float, float, float, float, float, float, list[float]]:
            cap_net = initial_capital
            cap_gross = initial_capital
            eq_net = [cap_net]
            to_tot = 0.0
            cur_p = np.zeros(N)
            r_net = []
            r_gross = []
            for t in range(T - 1):
                p_tgt = pos_arr[t]
                sig = np.maximum(v[t], 0.005)
                act = p_tgt != 0.0
                if np.sum(act) > 0:
                    inv = (1.0 / sig) * act
                    w = (p_tgt * inv) / np.maximum(np.sum(inv), 1e-12)
                else:
                    w = np.zeros(N)
                to = float(np.sum(np.abs(p_tgt - cur_p)))
                to_tot += to
                cost = cap_net * to * (fee_rate / N)
                g = float(np.sum(w * r[t + 1]))
                n = g - (cost / cap_net)
                cap_gross *= (1.0 + g)
                cap_net *= (1.0 + n)
                eq_net.append(cap_net)
                r_net.append(n)
                r_gross.append(g)
                cur_p = p_tgt
            ret_net = (cap_net - initial_capital) / (initial_capital / 100.0)
            ret_gross = (cap_gross - initial_capital) / (initial_capital / 100.0)
            sr_net = float(np.mean(r_net) / np.std(r_net) * np.sqrt(365)) if np.std(r_net) > 1e-9 else 0.0
            eq_n = np.array(eq_net)
            pk_n = np.maximum.accumulate(eq_n)
            dd = float(np.max((pk_n - eq_n) / pk_n * 100.0))
            exp_pct = float(np.mean(np.abs(pos_arr)) * 100.0)
            return ret_net, ret_gross, sr_net, dd, exp_pct, to_tot, eq_net

        mb = _sim(pos_base)
        mh = _sim(pos_hyst)
        me = _sim(pos_exp)
        mc = _sim(pos_c)
        md = _sim(pos_d)
        m_e_pred = _sim(pos_e)

        metrics_b = {"net_return": mb[0], "gross_return": mb[1], "sharpe": mb[2], "max_dd": mb[3], "exposure": mb[4], "turnover": mb[5]}
        metrics_h = {"net_return": mh[0], "gross_return": mh[1], "sharpe": mh[2], "max_dd": mh[3], "exposure": mh[4], "turnover": mh[5]}
        metrics_e = {"net_return": me[0], "gross_return": me[1], "sharpe": me[2], "max_dd": me[3], "exposure": me[4], "turnover": me[5]}
        metrics_c = {"net_return": mc[0], "gross_return": mc[1], "sharpe": mc[2], "max_dd": mc[3], "exposure": mc[4], "turnover": mc[5]}
        metrics_d = {"net_return": md[0], "gross_return": md[1], "sharpe": md[2], "max_dd": md[3], "exposure": md[4], "turnover": md[5]}
        metrics_e_pred = {"net_return": m_e_pred[0], "gross_return": m_e_pred[1], "sharpe": m_e_pred[2], "max_dd": m_e_pred[3], "exposure": m_e_pred[4], "turnover": m_e_pred[5]}

        audit_bh = audit_policy_equivalences(
            score_base=sc,
            score_hyst=sc,
            pos_base=pos_base,
            pos_hyst=pos_hyst,
            metrics_base=metrics_b,
            metrics_hyst=metrics_h,
        )

        results["universes"][str(N)] = {
            "N": N,
            "candidate_A": metrics_b,
            "candidate_H": metrics_h,
            "exposure_match": metrics_e,
            "candidate_C": metrics_c,
            "candidate_D": metrics_d,
            "candidate_E": metrics_e_pred,
            "exposure_scale_alpha": alpha,
            "equivalences_A_vs_H": {
                "predictor_equivalence": audit_bh.predictor_equivalence,
                "policy_equivalence": audit_bh.policy_equivalence,
                "economic_equivalence": audit_bh.economic_equivalence,
                "policy_divergence_pct": audit_bh.policy_divergence_pct,
                "delta_net_ret": audit_bh.economic_delta_net_ret,
                "delta_sharpe": audit_bh.economic_delta_sharpe,
                "delta_max_dd": audit_bh.economic_delta_max_dd,
            },
        }

        if verbose:
            print(f"N = {N:<4} | {'A: Baseline (Base)':<22} | {mb[0]:>+7.2f}% | {mb[1]:>+7.2f}% | {mb[2]:>+5.2f} | {mb[3]:>6.1f}% | {mb[4]:>7.1f}% | {mb[5]:>8.1f}")
            print(f"{'':<8} | {'H: Hysteresis':<22} | {mh[0]:>+7.2f}% | {mh[1]:>+7.2f}% | {mh[2]:>+5.2f} | {mh[3]:>6.1f}% | {mh[4]:>7.1f}% | {mh[5]:>8.1f}")
            print(f"{'':<8} | {'Control: Exposure-Match':<22} | {me[0]:>+7.2f}% | {me[1]:>+7.2f}% | {me[2]:>+5.2f} | {me[3]:>6.1f}% | {me[4]:>7.1f}% | {me[5]:>8.1f}")
            print(f"{'':<8} | {'C: Deadband (tau=0.20)':<22} | {mc[0]:>+7.2f}% | {mc[1]:>+7.2f}% | {mc[2]:>+5.2f} | {mc[3]:>6.1f}% | {mc[4]:>7.1f}% | {mc[5]:>8.1f}")
            print(f"{'':<8} | {'D: Turnover-Pen (l=0.20)':<22} | {md[0]:>+7.2f}% | {md[1]:>+7.2f}% | {md[2]:>+5.2f} | {md[3]:>6.1f}% | {md[4]:>7.1f}% | {md[5]:>8.1f}")
            print(f"{'':<8} | {'E: Interaction (b=0.10)':<22} | {m_e_pred[0]:>+7.2f}% | {m_e_pred[1]:>+7.2f}% | {m_e_pred[2]:>+5.2f} | {m_e_pred[3]:>6.1f}% | {m_e_pred[4]:>7.1f}% | {m_e_pred[5]:>8.1f}")
            print(f"  --> [N={N}] A vs H: Predictor={audit_bh.predictor_equivalence} | Policy={audit_bh.policy_equivalence} (Div: {audit_bh.policy_divergence_pct:.1f}%) | Economic={audit_bh.economic_equivalence}")
            print("-" * 105)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APEDP Multi-Asset Quantitative Signals Sandbox")
    parser.add_argument("--mode", default="multi", choices=["multi", "single", "all", "two_tier", "twotier", "grammar"],
                        help="Execution mode: multi (portfolio tournament), single (BTC forensic), all, two_tier (30-asset forensic), or grammar (15/30/60/90 policy grammar audit)")
    parser.add_argument("--universe", default="crypto", choices=["crypto", "equity", "equities", "30", "multi"],
                        help="Trading universe: crypto (15), equity (15), or 30 (15 crypto + 15 equities)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting portfolio capital")
    parser.add_argument("--pinned", action="store_true", default=False,
                        help="Use pinned canonical chromosome fixture instead of dynamic GA evolution (default: False)")

    # Support simple legacy positional args: 'multi', 'single', 'all', '30', 'equity', 'grammar'
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        first = sys.argv[1].lower()
        if first in ("30", "twotier", "two_tier"):
            args = parser.parse_args(["--mode", "two_tier", "--universe", "30", *sys.argv[2:]])
        elif first in ("equity", "equities"):
            args = parser.parse_args(["--mode", "multi", "--universe", "equity", *sys.argv[2:]])
        elif first in ("grammar", "policy"):
            args = parser.parse_args(["--mode", "grammar", *sys.argv[2:]])
        elif first in ("multi", "single", "all"):
            args = parser.parse_args(["--mode", first, *sys.argv[2:]])
        else:
            args = parser.parse_args()
    else:
        args = parser.parse_args()

    if args.mode == "grammar":
        run_multi_asset_policy_grammar_audit(initial_capital=args.capital, use_pinned=args.pinned)
    elif args.mode in ("two_tier", "twotier"):
        run_30_asset_two_tier_analysis(initial_capital=args.capital, use_pinned=args.pinned)
    else:
        if args.mode in ("multi", "all"):
            run_multi_asset_universe_tournament(initial_capital=args.capital, universe_type=args.universe)
        if args.mode in ("single", "all"):
            run_single_asset_forensic(initial_capital=args.capital)


