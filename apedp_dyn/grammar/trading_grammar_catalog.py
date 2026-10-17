"""APEDP Trading Grammar Space: Comprehensive Variable Catalogue & Typed Engine.

Implements the 5-layer trading grammar architecture:
  Layer 0: Raw Market Observations (OHLCV, Trades, Order Book, Derivatives, Options, Calendar)
  Layer 1: Primitive Transformations (Returns, Candle Shape, Reference Distance)
  Layer 2: Technical & Statistical Feature Families (Trend, Momentum, Volatility, Volume, VWAP, Cycle)
  Layer 0-2: Market Microstructure & Order Flow (Trade Flow, Spread, Depth Imbalance, Execution Cost)
  Layer 3: Candidate Equation Combinations (Interactions, Additive terms, Gated terms)
  Layer 4: Position, Risk & Trading Policy (Stateful execution, Sizing, Hysteresis, Cost penalty)

Additionally provides:
  - Cross-Asset & Cross-Sectional Feature Family (Relative rank, Beta to BTC, Correlation, Breadth)
  - Crypto-Specific Data Families (On-Chain, Stablecoin Liquidity, Sentiment)
  - Mathematical Operator Library (Causal Rolling, Temporal, Normalization, Logical Gating, Domain Safe Math)
  - Target & Outcome Grammar (Section 11: Return magnitude, Direction, Volatility, Triple Barrier)
  - Mandatory Grammar Rules & Machine-Readable Feature Registry Schema (Section 12, 14)
  - Executable First-Stage Feature Extraction Pipeline (Section 13)
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. Grammar Architecture Enums & Schema (Layers 0 - 4 + Metadata)
# ─────────────────────────────────────────────────────────────────────────────

class GrammarLayer(str, enum.Enum):
    LAYER_0_RAW = "Layer 0: Raw Market Observations"
    LAYER_1_PRIMITIVE = "Layer 1: Primitive Transformations"
    LAYER_2_FEATURE = "Layer 2: Technical & Statistical Features"
    LAYER_3_EQUATION = "Layer 3: Candidate Equations"
    LAYER_4_POLICY = "Layer 4: Trading Policy & Risk Control"


class FeatureFamily(str, enum.Enum):
    # Layer 0 & Microstructure
    RAW_OHLCV = "raw_ohlcv"
    RAW_TRADE = "raw_trade"
    RAW_ORDERBOOK = "raw_orderbook"
    RAW_DERIVATIVES = "raw_derivatives"
    RAW_OPTIONS = "raw_options"
    CALENDAR_SESSION = "calendar_session"
    MICROSTRUCTURE_TRADEFLOW = "microstructure_tradeflow"
    MICROSTRUCTURE_SPREAD = "microstructure_spread"
    MICROSTRUCTURE_DEPTH = "microstructure_depth"
    EXECUTION_COST = "execution_cost"

    # Layer 1
    PRICE_RETURNS = "price_returns"
    CANDLE_SHAPE = "candle_shape"
    REFERENCE_DISTANCE = "reference_distance"

    # Layer 2
    TREND_MOVING_AVERAGE = "trend_moving_average"
    MOMENTUM_OSCILLATORS = "momentum_oscillators"
    VOLATILITY_RANGE = "volatility_range"
    VOLUME_ACTIVITY = "volume_activity"
    VWAP_PRICE_VOLUME = "vwap_price_volume"
    CYCLE_STATISTICAL = "cycle_statistical"

    # Cross-Asset & Crypto-Specific
    CROSS_ASSET_RELATIVE = "cross_asset_relative"
    CROSS_ASSET_RELATIONSHIP = "cross_asset_relationship"
    MARKET_BREADTH = "market_breadth"
    CRYPTO_ONCHAIN = "crypto_onchain"
    CRYPTO_STABLECOIN = "crypto_stablecoin"
    CRYPTO_SENTIMENT = "crypto_sentiment"

    # Layer 4 Policy & Risk
    POLICY_STATE = "policy_state"
    RISK_CONTROL = "risk_control"


class DataType(str, enum.Enum):
    PRICE = "price"
    RETURN = "return"
    VOLUME = "volume"
    TIME = "time"
    PROBABILITY = "probability"
    RATIO = "dimensionless_ratio"
    ZSCORE = "zscore"
    RANK = "rank"
    STATE = "categorical_state"
    MONEY = "currency_notional"


class AvailabilityTiming(str, enum.Enum):
    BAR_CLOSE = "after_bar_close"
    TICK_BY_TICK = "immediate_tick"
    SESSION_CLOSE = "session_close"
    PERIODIC_SETTLEMENT = "periodic_settlement"
    PUBLICATION_DELAY = "publication_timestamp"


class FailurePolicy(str, enum.Enum):
    UNRESOLVED = "unresolved_error"
    FORWARD_FILL = "forward_fill"
    FILL_ZERO = "fill_zero"
    DROP_BAR = "drop_bar"
    ABSTAIN = "abstain_from_trade"


@dataclass(frozen=True)
class FeatureSpec:
    """Machine-readable feature specification matching Section 14."""
    feature_id: str
    layer: GrammarLayer
    family: FeatureFamily
    description: str
    inputs: tuple[str, ...]
    formula: str
    frequency: str = "bar"
    data_type: DataType = DataType.RATIO
    units: str = "dimensionless"
    lookback: int = 1
    min_history: int = 1
    availability: AvailabilityTiming = AvailabilityTiming.BAR_CLOSE
    normalization: str = "none"
    requires: tuple[str, ...] = ()
    failure_policy: FailurePolicy = FailurePolicy.UNRESOLVED
    causal_contract: bool = True
    domain_constraints: str = "none"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer"] = self.layer.value
        d["family"] = self.family.value
        d["data_type"] = self.data_type.value
        d["availability"] = self.availability.value
        d["failure_policy"] = self.failure_policy.value
        return d


@dataclass(frozen=True)
class TargetSpec:
    """Machine-readable target/label specification matching Section 11."""
    target_id: str
    description: str
    objective: str
    horizon_bars: int
    formula: str
    is_classification: bool = False
    requires_future: bool = True  # Strict distinction: targets MUST require future data, features NEVER can


# ─────────────────────────────────────────────────────────────────────────────
# 2. Comprehensive Mathematical Operators Library (Section 10)
# ─────────────────────────────────────────────────────────────────────────────

# --- A. Arithmetic and Scalar Transformations (Domain Safe) ---

def safe_div(x: np.ndarray | float, y: np.ndarray | float, eps: float = 1e-12) -> np.ndarray:
    """Safe division with explicit denominator flooring (Spec §10.A, §12)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    sign = np.where(y_arr >= 0.0, 1.0, -1.0)
    denom = np.where(np.abs(y_arr) < eps, sign * eps, y_arr)
    return x_arr / denom


def safe_log(x: np.ndarray | float, floor: float = 1e-12) -> np.ndarray:
    """Domain-safe natural logarithm enforcing x > 0 floor (Spec §10.A)."""
    x_arr = np.asarray(x, dtype=float)
    return np.log(np.maximum(x_arr, floor))


def safe_log1p(x: np.ndarray | float, floor: float = -0.999999) -> np.ndarray:
    """Domain-safe log(1+x) enforcing 1+x > 0 (Spec §10.A)."""
    x_arr = np.asarray(x, dtype=float)
    return np.log1p(np.maximum(x_arr, floor))


def safe_sqrt(x: np.ndarray | float) -> np.ndarray:
    """Domain-safe square root enforcing non-negative input (Spec §10.A)."""
    x_arr = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(x_arr, 0.0))


def safe_pow(x: np.ndarray | float, p: float) -> np.ndarray:
    """Domain-safe power with controlled exponent (Spec §10.A)."""
    x_arr = np.asarray(x, dtype=float)
    if p % 1 == 0:
        return np.power(x_arr, p)
    return np.power(np.maximum(x_arr, 0.0), p)


def exp_clamp(x: np.ndarray | float, min_val: float = -20.0, max_val: float = 20.0) -> np.ndarray:
    """Clamped exponential avoiding floating point overflow (Spec §10.A)."""
    x_arr = np.asarray(x, dtype=float)
    return np.exp(np.clip(x_arr, min_val, max_val))


def op_tanh(x: np.ndarray | float) -> np.ndarray:
    """Hyperbolic tangent bounded transformation in [-1, 1]."""
    return np.tanh(np.asarray(x, dtype=float))


def op_clip(x: np.ndarray | float, min_val: float, max_val: float) -> np.ndarray:
    """Explicit clipping operator."""
    return np.clip(np.asarray(x, dtype=float), min_val, max_val)


def positive_part(x: np.ndarray | float) -> np.ndarray:
    """max(x, 0) positive half-wave rectifier."""
    x_arr = np.asarray(x, dtype=float)
    return np.maximum(x_arr, 0.0)


def negative_part(x: np.ndarray | float) -> np.ndarray:
    """max(-x, 0) negative half-wave rectifier (downside magnitude)."""
    x_arr = np.asarray(x, dtype=float)
    return np.maximum(-x_arr, 0.0)


# --- B. Rolling-Window Causal Operators (Spec §10.B) ---

def roll_mean(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling mean; returns NaN until min_periods reached."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).mean().to_numpy()


def roll_std(x: np.ndarray, n: int, min_periods: int | None = None, ddof: int = 1) -> np.ndarray:
    """Causal rolling sample standard deviation."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).std(ddof=ddof).to_numpy()


def roll_var(x: np.ndarray, n: int, min_periods: int | None = None, ddof: int = 1) -> np.ndarray:
    """Causal rolling sample variance."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).var(ddof=ddof).to_numpy()


def roll_min(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling minimum."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).min().to_numpy()


def roll_max(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling maximum."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).max().to_numpy()


def roll_sum(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling sum."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).sum().to_numpy()


def roll_median(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling median."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).median().to_numpy()


def roll_quantile(x: np.ndarray, n: int, q: float, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling quantile in [0, 1]."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).quantile(q).to_numpy()


def roll_skew(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling sample skewness."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).skew().to_numpy()


def roll_kurtosis(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling sample kurtosis."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).kurt().to_numpy()


def ewma(x: np.ndarray, span: int, adjust: bool = False) -> np.ndarray:
    """Exponentially weighted moving average."""
    return pd.Series(x).ewm(span=span, adjust=adjust).mean().to_numpy()


def roll_slope(x: np.ndarray, n: int) -> np.ndarray:
    """Causal rolling ordinary least squares regression slope against time index."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    out = np.full(T, np.nan)
    if T < n:
        return out
    t_idx = np.arange(n, dtype=float)
    t_mean = (n - 1.0) / 2.0
    t_var = np.sum((t_idx - t_mean) ** 2)
    for i in range(n - 1, T):
        window = x[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            cov = np.sum((t_idx - t_mean) * (window - np.mean(window)))
            out[i] = cov / t_var
    return out


def roll_corr(x: np.ndarray, y: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling correlation between x and y."""
    mp = n if min_periods is None else min_periods
    return pd.Series(x).rolling(n, min_periods=mp).corr(pd.Series(y)).to_numpy()


def roll_beta(x: np.ndarray, benchmark: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Causal rolling OLS beta of asset return x against benchmark return."""
    mp = n if min_periods is None else min_periods
    sx = pd.Series(x)
    sb = pd.Series(benchmark)
    cov = sx.rolling(n, min_periods=mp).cov(sb)
    var_b = sb.rolling(n, min_periods=mp).var()
    with np.errstate(all="ignore"):
        return np.where(var_b > 1e-12, cov / var_b, np.nan)


# --- C. Temporal Causal Operators (Spec §10.C) ---

def op_lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Causal lag operator: x_{t-k}. Pads first k elements with NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def op_diff(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Causal difference operator: x_t - x_{t-k}."""
    return x - op_lag(x, k)


def op_pct_change(x: np.ndarray, k: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Causal percentage change: (x_t - x_{t-k}) / x_{t-k}."""
    lagged = op_lag(x, k)
    return safe_div(x - lagged, lagged, eps=eps)


def op_log_return(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Causal log return: ln(x_t) - ln(x_{t-k})."""
    x_arr = np.asarray(x, dtype=float)
    lagged = op_lag(x_arr, k)
    with np.errstate(all="ignore"):
        return safe_log(x_arr) - safe_log(lagged)


def op_acceleration(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Return acceleration: second temporal difference."""
    d1 = op_diff(x, k)
    return op_diff(d1, k)


def time_since_last_high(high: np.ndarray, n: int) -> np.ndarray:
    """Bars since the n-bar rolling high occurred."""
    high = np.asarray(high, dtype=float)
    T = len(high)
    out = np.full(T, np.nan)
    for i in range(n - 1, T):
        window = high[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            idx_max = int(np.argmax(window))
            out[i] = float(n - 1 - idx_max)
    return out


def time_since_last_low(low: np.ndarray, n: int) -> np.ndarray:
    """Bars since the n-bar rolling low occurred."""
    low = np.asarray(low, dtype=float)
    T = len(low)
    out = np.full(T, np.nan)
    for i in range(n - 1, T):
        window = low[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            idx_min = int(np.argmin(window))
            out[i] = float(n - 1 - idx_min)
    return out


def consecutive_up_bars(r: np.ndarray) -> np.ndarray:
    """Count of consecutive positive return bars."""
    r = np.asarray(r, dtype=float)
    out = np.zeros(len(r), dtype=float)
    c = 0.0
    for i in range(len(r)):
        if np.isnan(r[i]):
            c = 0.0
        elif r[i] > 0.0:
            c += 1.0
        else:
            c = 0.0
        out[i] = c
    return out


def consecutive_down_bars(r: np.ndarray) -> np.ndarray:
    """Count of consecutive negative return bars."""
    r = np.asarray(r, dtype=float)
    out = np.zeros(len(r), dtype=float)
    c = 0.0
    for i in range(len(r)):
        if np.isnan(r[i]):
            c = 0.0
        elif r[i] < 0.0:
            c += 1.0
        else:
            c = 0.0
        out[i] = c
    return out


def count_condition(condition_mask: np.ndarray, n: int) -> np.ndarray:
    """Causal count of true conditions in trailing window n."""
    numeric = np.asarray(condition_mask, dtype=float)
    return pd.Series(numeric).rolling(n, min_periods=n).sum().to_numpy()


# --- D. Ranking and Normalization Operators (Spec §10.D) ---

def causal_zscore(v: np.ndarray, min_history: int = 60) -> np.ndarray:
    """Expanding causal z-score; NaN until min_history finite observations."""
    v = np.asarray(v, dtype=float)
    s = pd.Series(v)
    mu = s.expanding(min_history).mean().to_numpy()
    sd = s.expanding(min_history).std().to_numpy()
    with np.errstate(all="ignore"):
        return np.where(sd > 0, (v - mu) / sd, np.nan)


def trailing_rank(v: np.ndarray, window: int = 40) -> np.ndarray:
    """Causal trailing-window percentile rank in [0,1]; NaN until window full."""
    v = np.asarray(v, dtype=float)
    out = np.full(len(v), np.nan)
    for t in range(window - 1, len(v)):
        w = v[t - window + 1 : t + 1]
        if np.isfinite(w).all():
            out[t] = float(np.sum(w <= v[t]) - 1) / (window - 1)
    return out


def cross_sectional_rank(matrix_2d: np.ndarray) -> np.ndarray:
    """Rank across assets per time bar; normalized to [0, 1].

    Args:
        matrix_2d: Shape (T, N_assets).
    """
    matrix_2d = np.asarray(matrix_2d, dtype=float)
    T, N = matrix_2d.shape
    if N <= 1:
        return np.full((T, N), 0.5)
    out = np.full((T, N), np.nan)
    for t in range(T):
        row = matrix_2d[t]
        valid = np.isfinite(row)
        n_valid = int(np.sum(valid))
        if n_valid > 1:
            ranks = pd.Series(row).rank(method="average", na_option="keep").to_numpy()
            out[t] = (ranks - 1.0) / (n_valid - 1.0)
    return out


def cross_sectional_zscore(matrix_2d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cross-sectional standard score per time bar across all active universe assets."""
    matrix_2d = np.asarray(matrix_2d, dtype=float)
    T, N = matrix_2d.shape
    out = np.full((T, N), np.nan)
    for t in range(T):
        row = matrix_2d[t]
        valid = np.isfinite(row)
        if np.sum(valid) > 2:
            mu = np.mean(row[valid])
            sd = np.std(row[valid], ddof=1)
            if sd > eps:
                out[t, valid] = (row[valid] - mu) / sd
            else:
                out[t, valid] = 0.0
    return out


def winsorize(x: np.ndarray, p_low: float = 0.01, p_high: float = 0.99) -> np.ndarray:
    """Winsorize outliers to designated percentiles."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return x.copy()
    q_low = float(np.percentile(x[valid], p_low * 100.0))
    q_high = float(np.percentile(x[valid], p_high * 100.0))
    return np.clip(x, q_low, q_high)


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust z-score using median and MAD (Median Absolute Deviation)."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if np.sum(valid) < 3:
        return np.full_like(x, np.nan)
    med = float(np.median(x[valid]))
    mad = float(np.median(np.abs(x[valid] - med)))
    # Normal consistency factor 1.4826
    norm_mad = 1.4826 * mad
    if norm_mad < eps:
        return np.zeros_like(x)
    return (x - med) / norm_mad


# --- E. Logical and Interaction Operators (Spec §10.E) ---

def op_deadband(score: np.ndarray, tau: float = 0.1979) -> np.ndarray:
    """Stateless deadband decision operator: filters noise when |score| < tau."""
    score = np.asarray(score, dtype=float)
    return np.where(score >= tau, 1.0, np.where(score <= -tau, -1.0, 0.0))


def signal_gate(signal: np.ndarray, gate_mask: np.ndarray) -> np.ndarray:
    """Binary signal gate: allows signal to pass only when gate_mask is True."""
    sig = np.asarray(signal, dtype=float)
    mask = np.asarray(gate_mask, dtype=bool)
    return np.where(mask, sig, 0.0)


def regime_gate(signal: np.ndarray, regime_var: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Continuous or banded regime gate: passes signal when min_val <= regime_var <= max_val."""
    sig = np.asarray(signal, dtype=float)
    reg = np.asarray(regime_var, dtype=float)
    active = (reg >= min_val) & (reg <= max_val)
    return np.where(active, sig, 0.0)


def op_cooldown(signal: np.ndarray, cooldown_bars: int) -> np.ndarray:
    """Cooldown filter: enforces zero position/trading for cooldown_bars after an exit."""
    sig = np.asarray(signal, dtype=float)
    T = len(sig)
    out = np.zeros(T, dtype=float)
    cooldown_left = 0
    prev_pos = 0.0
    for t in range(T):
        if cooldown_left > 0:
            out[t] = 0.0
            cooldown_left -= 1
            prev_pos = 0.0
            continue
        cur = sig[t]
        if prev_pos != 0.0 and cur == 0.0:
            # Trigger cooldown on position close
            cooldown_left = cooldown_bars
            out[t] = 0.0
        else:
            out[t] = cur
        prev_pos = out[t]
    return out


def op_hysteresis_path(
    scores: np.ndarray,
    theta_in: float = 0.1979,
    theta_out: float = 0.0,
    init_pos: float = 0.0,
) -> np.ndarray:
    """Path-dependent hysteresis operator: entry at theta_in, persistence until theta_out."""
    scores = np.asarray(scores, dtype=float)
    T = len(scores)
    pos = np.zeros(T, dtype=float)
    cur = init_pos
    for t in range(T):
        s = scores[t]
        if cur > 0.0:
            if s < -theta_in:
                cur = -1.0
            elif s < theta_out:
                cur = 0.0
            else:
                cur = 1.0
        elif cur < 0.0:
            if s > theta_in:
                cur = 1.0
            elif s > -theta_out:
                cur = 0.0
            else:
                cur = -1.0
        else:
            if s > theta_in:
                cur = 1.0
            elif s < -theta_in:
                cur = -1.0
            else:
                cur = 0.0
        pos[t] = cur
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# 3. Comprehensive Feature Catalogue Registry (Sections 2–9, 11)
# ─────────────────────────────────────────────────────────────────────────────

def _build_master_feature_catalogue() -> dict[str, FeatureSpec]:
    """Instantiate the comprehensive master catalogue of trading variables."""
    cat: dict[str, FeatureSpec] = {}

    def reg(spec: FeatureSpec) -> None:
        cat[spec.feature_id] = spec

    # ── Layer 0: Raw Market Variables ──
    # A. OHLCV Candle Data
    reg(FeatureSpec("OPEN", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Opening price of the bar", ("OPEN",), "P_open", data_type=DataType.PRICE, units="quote_currency"))
    reg(FeatureSpec("HIGH", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Highest traded price in the bar", ("HIGH",), "P_high", data_type=DataType.PRICE, units="quote_currency"))
    reg(FeatureSpec("LOW", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Lowest traded price in the bar", ("LOW",), "P_low", data_type=DataType.PRICE, units="quote_currency"))
    reg(FeatureSpec("CLOSE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Closing price of the bar", ("CLOSE",), "P_close", data_type=DataType.PRICE, units="quote_currency"))
    reg(FeatureSpec("VOLUME_BASE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Traded volume in base units", ("VOLUME",), "V_base", data_type=DataType.VOLUME, units="base_currency"))
    reg(FeatureSpec("VOLUME_QUOTE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Traded turnover in quote units", ("VOLUME_QUOTE",), "V_quote", data_type=DataType.MONEY, units="quote_currency"))
    reg(FeatureSpec("TRADE_COUNT", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Total number of executed transactions", ("COUNT",), "N_trades", data_type=DataType.RATIO, units="trades"))
    reg(FeatureSpec("TAKER_BUY_BASE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Aggressive buyer-initiated base volume", ("TAKER_BUY_BASE",), "V_taker_buy", data_type=DataType.VOLUME, units="base_currency"))
    reg(FeatureSpec("TAKER_BUY_QUOTE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Aggressive buyer-initiated quote turnover", ("TAKER_BUY_QUOTE",), "Turnover_taker_buy", data_type=DataType.MONEY, units="quote_currency"))
    reg(FeatureSpec("VWAP_BAR", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OHLCV, "Volume-weighted average price of the single bar", ("VOLUME_QUOTE", "VOLUME_BASE"), "V_quote / V_base", data_type=DataType.PRICE, units="quote_currency"))

    # B. Individual Trade Data
    reg(FeatureSpec("TRADE_PRICE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_TRADE, "Price of individual execution tick", ("TRADE_PX",), "P_trade", data_type=DataType.PRICE, units="quote_currency", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("TRADE_SIZE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_TRADE, "Size of individual execution tick", ("TRADE_SZ",), "S_trade", data_type=DataType.VOLUME, units="base_currency", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("AGGRESSOR_SIDE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_TRADE, "Direction of tick market order (+1 buy, -1 sell)", ("SIDE",), "Side_t", data_type=DataType.STATE, units="categorical", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("LARGE_TRADE_FLAG", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_TRADE, "Indicator for trade exceeding size threshold", ("TRADE_SZ",), "S_trade > Q99", data_type=DataType.STATE, units="binary", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("TRADE_INTENSITY", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_TRADE, "Arrival frequency of trades per second", ("TRADE_TIME",), "N_trades / dt", data_type=DataType.RATIO, units="Hz", availability=AvailabilityTiming.TICK_BY_TICK))

    # C. Order Book Data
    reg(FeatureSpec("BID_PRICE_L1", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_ORDERBOOK, "Best bid quote price", ("BID_1",), "P_bid1", data_type=DataType.PRICE, units="quote_currency", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("ASK_PRICE_L1", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_ORDERBOOK, "Best ask quote price", ("ASK_1",), "P_ask1", data_type=DataType.PRICE, units="quote_currency", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("BID_SIZE_L1", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_ORDERBOOK, "Depth available at best bid", ("BID_SZ_1",), "Q_bid1", data_type=DataType.VOLUME, units="base_currency", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("ASK_SIZE_L1", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_ORDERBOOK, "Depth available at best ask", ("ASK_SZ_1",), "Q_ask1", data_type=DataType.VOLUME, units="base_currency", availability=AvailabilityTiming.TICK_BY_TICK))
    reg(FeatureSpec("MICROPRICE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_ORDERBOOK, "Size-weighted instantaneous microprice", ("BID_1", "ASK_1", "BID_SZ_1", "ASK_SZ_1"), "(P_bid1*Q_ask1 + P_ask1*Q_bid1)/(Q_bid1 + Q_ask1)", data_type=DataType.PRICE, units="quote_currency", availability=AvailabilityTiming.TICK_BY_TICK))

    # D. Futures and Perpetual Derivatives
    reg(FeatureSpec("FUNDING_RATE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_DERIVATIVES, "8-hour periodic perpetual swap funding fee", ("FUNDING_RATE",), "F_t", data_type=DataType.RATIO, units="rate", availability=AvailabilityTiming.PERIODIC_SETTLEMENT))
    reg(FeatureSpec("OPEN_INTEREST", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_DERIVATIVES, "Total open perpetual/futures contracts", ("OI",), "OI_t", data_type=DataType.VOLUME, units="contracts", availability=AvailabilityTiming.BAR_CLOSE))
    reg(FeatureSpec("OPEN_INTEREST_CHANGE", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_DERIVATIVES, "Change in open interest over bar", ("OI",), "OI_t - OI_{t-1}", data_type=DataType.VOLUME, units="contracts", lookback=1))
    reg(FeatureSpec("PERPETUAL_BASIS", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_DERIVATIVES, "Spread between mark price and spot index", ("MARK_PRICE", "INDEX_PRICE"), "(P_mark - P_spot)/P_spot", data_type=DataType.RATIO, units="rate"))
    reg(FeatureSpec("LONG_SHORT_RATIO", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_DERIVATIVES, "Ratio of long to short trader accounts", ("LS_RATIO",), "N_long / N_short", data_type=DataType.RATIO, units="dimensionless"))
    reg(FeatureSpec("LIQUIDATION_VOLUME", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_DERIVATIVES, "Forced liquidation volume in bar", ("LIQ_VOL",), "V_liq", data_type=DataType.VOLUME, units="base_currency"))

    # E. Options Data
    reg(FeatureSpec("IMPLIED_VOLATILITY_ATM", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OPTIONS, "At-the-money 30-day implied volatility", ("IV_ATM",), "IV_atm", data_type=DataType.RATIO, units="annualized_vol"))
    reg(FeatureSpec("PUT_CALL_RATIO_VOLUME", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OPTIONS, "Volume ratio of traded put vs call options", ("PUT_VOL", "CALL_VOL"), "V_put / V_call", data_type=DataType.RATIO, units="dimensionless"))
    reg(FeatureSpec("DVOL", GrammarLayer.LAYER_0_RAW, FeatureFamily.RAW_OPTIONS, "Deribit 30-day constant-maturity volatility index", ("DVOL",), "DVOL_t", data_type=DataType.RATIO, units="annualized_vol"))

    # F. Calendar and Session Variables
    reg(FeatureSpec("HOUR_OF_DAY", GrammarLayer.LAYER_0_RAW, FeatureFamily.CALENDAR_SESSION, "UTC hour of bar timestamp [0..23]", ("TIMESTAMP",), "Hour(ts)", data_type=DataType.TIME, units="hours"))
    reg(FeatureSpec("DAY_OF_WEEK", GrammarLayer.LAYER_0_RAW, FeatureFamily.CALENDAR_SESSION, "Day of week [0=Monday..6=Sunday]", ("TIMESTAMP",), "Weekday(ts)", data_type=DataType.TIME, units="days"))
    reg(FeatureSpec("IS_WEEKEND", GrammarLayer.LAYER_0_RAW, FeatureFamily.CALENDAR_SESSION, "Weekend indicator (crypto liquidity shift)", ("TIMESTAMP",), "1 if day >= 5 else 0", data_type=DataType.STATE, units="binary"))
    reg(FeatureSpec("MINUTES_TO_FUNDING", GrammarLayer.LAYER_0_RAW, FeatureFamily.CALENDAR_SESSION, "Time in minutes to nearest funding timestamp", ("TIMESTAMP",), "funding_ts - ts", data_type=DataType.TIME, units="minutes"))

    # ── Layer 1: Primitive Transformations ──
    # A. Returns and Price Changes
    for k in (1, 2, 4, 8, 16):
        reg(FeatureSpec(f"RET_{k}", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.PRICE_RETURNS, f"Simple return over trailing {k} bars", ("CLOSE",), f"(P_t - P_{{t-{k}}}) / P_{{t-{k}}}", lookback=k, data_type=DataType.RETURN))
        reg(FeatureSpec(f"LOGRET_{k}", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.PRICE_RETURNS, f"Logarithmic return over trailing {k} bars", ("CLOSE",), f"ln(P_t) - ln(P_{{t-{k}}})", lookback=k, data_type=DataType.RETURN))
    reg(FeatureSpec("OPEN_TO_CLOSE", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.PRICE_RETURNS, "Intrabar return from open to close", ("OPEN", "CLOSE"), "(P_close - P_open) / P_open", lookback=1, data_type=DataType.RETURN))
    reg(FeatureSpec("RETURN_ACCELERATION", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.PRICE_RETURNS, "Second difference of price: RET_1 - RET_1(t-1)", ("CLOSE",), "RET_1(t) - RET_1(t-1)", lookback=2, data_type=DataType.RETURN))

    # B. Candle Shape and Range
    reg(FeatureSpec("TRUE_RANGE", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.CANDLE_SHAPE, "Welles Wilder true range", ("HIGH", "LOW", "CLOSE"), "max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)", lookback=1, data_type=DataType.PRICE))
    reg(FeatureSpec("BODY_TO_RANGE", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.CANDLE_SHAPE, "Ratio of candle body to full bar range", ("OPEN", "CLOSE", "HIGH", "LOW"), "|C - O| / max(H - L, eps)", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("UPPER_WICK_TO_RANGE", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.CANDLE_SHAPE, "Upper rejection shadow fraction", ("OPEN", "CLOSE", "HIGH", "LOW"), "(H - max(O, C)) / max(H - L, eps)", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("LOWER_WICK_TO_RANGE", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.CANDLE_SHAPE, "Lower rejection shadow fraction", ("OPEN", "CLOSE", "HIGH", "LOW"), "(min(O, C) - L) / max(H - L, eps)", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("CLOSE_LOCATION", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.CANDLE_SHAPE, "Close location within high-low range [0, 1]", ("HIGH", "LOW", "CLOSE"), "(C - L) / max(H - L, eps)", lookback=1, data_type=DataType.RATIO))

    # C. Distance from Reference Prices
    reg(FeatureSpec("DIST_TO_SMA", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.REFERENCE_DISTANCE, "Price distance from trailing 20-bar SMA", ("CLOSE",), "(P - SMA_20) / SMA_20", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("DIST_TO_EMA", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.REFERENCE_DISTANCE, "Price distance from trailing 20-bar EMA", ("CLOSE",), "(P - EMA_20) / EMA_20", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("DIST_TO_DONCHIAN_HIGH", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.REFERENCE_DISTANCE, "Price distance from 20-bar high", ("CLOSE", "HIGH"), "(P - MAX_20(H)) / P", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("DIST_TO_DONCHIAN_LOW", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.REFERENCE_DISTANCE, "Price distance from 20-bar low", ("CLOSE", "LOW"), "(P - MIN_20(L)) / P", lookback=20, data_type=DataType.RATIO))

    # ── Layer 2: Technical and Statistical Feature Families ──
    # A. Trend & Moving Average
    reg(FeatureSpec("EMA_SPREAD", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Fast EMA(12) minus Slow EMA(26) normalized", ("CLOSE",), "(EMA_12 - EMA_26) / P", lookback=26, data_type=DataType.RATIO))
    reg(FeatureSpec("EMA_SLOPE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Trailing regression slope of 20-bar EMA", ("CLOSE",), "Slope(EMA_20, 5)", lookback=25, data_type=DataType.RATIO))
    reg(FeatureSpec("MACD", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "MACD histogram (MACD line - Signal line)", ("CLOSE",), "(EMA_12 - EMA_26) - EMA_9(MACD)", lookback=35, data_type=DataType.RATIO))
    reg(FeatureSpec("ADX", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Wilder Average Directional Index (trend strength)", ("HIGH", "LOW", "CLOSE"), "ADX(14)", lookback=28, data_type=DataType.RATIO))
    reg(FeatureSpec("TREND_EFFICIENCY", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Kaufman efficiency ratio: net move / gross path", ("CLOSE",), "|P_t - P_{t-20}| / Sum(|P_i - P_{i-1}|, 20)", lookback=20, data_type=DataType.RATIO))

    # B. Momentum & Oscillators
    reg(FeatureSpec("RSI", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MOMENTUM_OSCILLATORS, "14-bar Relative Strength Index", ("CLOSE",), "100 - (100 / (1 + Gain / Loss))", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("ROC", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MOMENTUM_OSCILLATORS, "10-bar Rate of Change momentum", ("CLOSE",), "(P_t - P_{t-10}) / P_{t-10}", lookback=10, data_type=DataType.RETURN))
    reg(FeatureSpec("STOCH_K", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MOMENTUM_OSCILLATORS, "Stochastic %K oscillator", ("HIGH", "LOW", "CLOSE"), "(P - L_14) / (H_14 - L_14) * 100", lookback=14, data_type=DataType.RATIO))

    # C. Volatility & Range
    reg(FeatureSpec("ATR", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "14-bar Average True Range", ("HIGH", "LOW", "CLOSE"), "Wilder_SMA(TR, 14)", lookback=14, data_type=DataType.PRICE))
    reg(FeatureSpec("NATR", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Normalized ATR percentage", ("HIGH", "LOW", "CLOSE"), "ATR(14) / P * 100", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("REALIZED_VOL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Rolling 20-bar standard deviation of returns", ("CLOSE",), "Std(r, 20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("DOWNSIDE_VOL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Rolling 20-bar downside semivariance volatility", ("CLOSE",), "Sqrt(Mean(min(r, 0)^2, 20))", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("PARKINSON_VOL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Parkinson range-based volatility estimator", ("HIGH", "LOW"), "Sqrt(Sum(ln(H/L)^2, 20) / (4*ln(2)*20))", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("GARMAN_KLASS_VOL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Garman-Klass OHLC volatility estimator", ("OPEN", "HIGH", "LOW", "CLOSE"), "Sqrt(Sum(0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2, 20)/20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("BB_WIDTH", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Bollinger Band width (4 * std20 / sma20)", ("CLOSE",), "(Upper_BB - Lower_BB) / SMA_20", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("BB_PERCENT_B", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Bollinger %B oscillator", ("CLOSE",), "(P - Lower_BB) / (Upper_BB - Lower_BB)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("KELTNER_WIDTH", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Keltner Channel width (3 * ATR / EMA_20)", ("HIGH", "LOW", "CLOSE"), "(Upper_KC - Lower_KC) / EMA_20", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("SQUEEZE_STATE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "John Carter Squeeze indicator (BB inside Keltner)", ("HIGH", "LOW", "CLOSE"), "BB_WIDTH / KELTNER_WIDTH", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("BREAKOUT_STRENGTH", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Magnitude of excursion past Donchian channel / ATR", ("CLOSE", "HIGH", "LOW"), "max(0, P - Donchian_High) / ATR", lookback=20, data_type=DataType.RATIO))

    # D. Volume & Traded Activity
    reg(FeatureSpec("LOG_VOLUME", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Natural log of bar volume", ("VOLUME",), "ln(V)", lookback=1, data_type=DataType.VOLUME))
    reg(FeatureSpec("RELATIVE_VOLUME", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Volume relative to 20-bar trailing mean", ("VOLUME",), "V / Mean(V, 20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("VOLUME_ZSCORE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Causal z-score of volume", ("VOLUME",), "ZScore(V)", lookback=60, data_type=DataType.ZSCORE))
    reg(FeatureSpec("OBV_MOM", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "10-bar percentage change of On-Balance Volume", ("CLOSE", "VOLUME"), "PctChange(OBV, 10)", lookback=10, data_type=DataType.RATIO))
    reg(FeatureSpec("CMF", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "20-bar Chaikin Money Flow", ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"), "Sum(((C-L)-(H-C))/(H-L)*V, 20) / Sum(V, 20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("MFI", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "14-bar Money Flow Index", ("HIGH", "LOW", "CLOSE", "VOLUME"), "MFI(14)", lookback=14, data_type=DataType.RATIO))

    # ── Layer 0-2: Market Microstructure and Order Flow ──
    reg(FeatureSpec("TAKER_IMBALANCE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Normalized aggressive buy vs sell volume imbalance", ("TAKER_BUY_BASE", "VOLUME"), "(2*V_taker_buy - V) / max(V, eps)", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("ORDER_FLOW_IMBALANCE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Order book OFI delta z-score", ("OFI",), "ZScore(OFI)", lookback=60, data_type=DataType.ZSCORE))
    reg(FeatureSpec("SIGNED_VOLUME_MOMENTUM", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Momentum of signed trade volume", ("TAKER_BUY_BASE", "VOLUME"), "Slope(2*V_tb - V, 5)", lookback=5, data_type=DataType.RATIO))
    reg(FeatureSpec("SPREAD_BPS", GrammarLayer.LAYER_0_RAW, FeatureFamily.MICROSTRUCTURE_SPREAD, "Quoted bid-ask spread in basis points", ("BID_1", "ASK_1"), "(P_ask - P_bid) / P_mid * 10000", data_type=DataType.RATIO, units="bps"))
    reg(FeatureSpec("DEPTH_IMBALANCE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_DEPTH, "Order book Level 1 depth queue imbalance", ("BID_SZ_1", "ASK_SZ_1"), "(Q_bid - Q_ask) / (Q_bid + Q_ask)", data_type=DataType.RATIO))
    reg(FeatureSpec("MICROPRICE_DISPLACEMENT", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_DEPTH, "Displacement of microprice from mid-quote", ("MICROPRICE", "MID_PRICE"), "(P_micro - P_mid) / P_mid", data_type=DataType.RATIO))
    reg(FeatureSpec("ESTIMATED_SLIPPAGE", GrammarLayer.LAYER_4_POLICY, FeatureFamily.EXECUTION_COST, "Estimated market impact slippage under order size", ("VOLUME", "ATR"), "gamma * ATR * Sqrt(OrderSize / V)", data_type=DataType.RATIO, units="rate"))

    # ── Cross-Asset and Cross-Sectional Variables (Section 6) ──
    reg(FeatureSpec("BTC_RETURN", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIVE, "Return of benchmark BTC over same horizon", ("BTC_CLOSE",), "RET_1(BTC)", lookback=1, data_type=DataType.RETURN))
    reg(FeatureSpec("BTC_REALIZED_VOL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIVE, "Rolling 20-bar volatility of BTC", ("BTC_CLOSE",), "RealizedVol(BTC, 20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("ROLLING_BETA_TO_BTC", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIONSHIP, "Rolling 30-bar beta of asset return against BTC", ("CLOSE", "BTC_CLOSE"), "Cov(r_i, r_btc, 30) / Var(r_btc, 30)", lookback=30, data_type=DataType.RATIO))
    reg(FeatureSpec("CORRELATION_TO_BTC", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIONSHIP, "Rolling 30-bar Pearson correlation against BTC", ("CLOSE", "BTC_CLOSE"), "Corr(r_i, r_btc, 30)", lookback=30, data_type=DataType.RATIO))
    reg(FeatureSpec("CROSS_SECTIONAL_RETURN_RANK", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIVE, "Percentile rank of asset return across all universe assets", ("CLOSE_UNIVERSE",), "CSRank(RET_1)", lookback=1, data_type=DataType.RANK))
    reg(FeatureSpec("PERCENT_ABOVE_MA", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MARKET_BREADTH, "Fraction of universe assets trading above 20-bar SMA", ("CLOSE_UNIVERSE",), "Mean(P_i > SMA_20_i)", lookback=20, data_type=DataType.PROBABILITY))

    # ── Layer 4: Position, Risk, and Policy Variables (Section 9) ──
    reg(FeatureSpec("CURRENT_POSITION", GrammarLayer.LAYER_4_POLICY, FeatureFamily.POLICY_STATE, "Current signed portfolio exposure (-1, 0, +1)", ("POSITION",), "pos_t", data_type=DataType.STATE, units="weight"))
    reg(FeatureSpec("UNREALIZED_PNL", GrammarLayer.LAYER_4_POLICY, FeatureFamily.POLICY_STATE, "Mark-to-market unrealized profit/loss fraction", ("ENTRY_PX", "CLOSE"), "pos * (P - P_entry) / P_entry", data_type=DataType.RETURN))
    reg(FeatureSpec("PORTFOLIO_DRAWDOWN", GrammarLayer.LAYER_4_POLICY, FeatureFamily.RISK_CONTROL, "Current portfolio drawdown from peak equity", ("EQUITY",), "(Peak_Eq - Eq) / Peak_Eq", data_type=DataType.RATIO))
    reg(FeatureSpec("TURNOVER_COST", GrammarLayer.LAYER_4_POLICY, FeatureFamily.EXECUTION_COST, "Fee incurred from position adjustment", ("TARGET_POS", "PREV_POS"), "fee_rate * |pos_t - pos_{t-1}|", data_type=DataType.RATIO))

    # ── Additional Quantitative, Microstructure & Statistical Arbitrage Features ──
    reg(FeatureSpec("KYLE_LAMBDA", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Kyle's Lambda price impact per unit traded volume", ("CLOSE", "VOLUME", "TAKER_BUY_BASE"), "Cov(Delta P, Signed_Vol) / Var(Signed_Vol)", lookback=60, data_type=DataType.RATIO))
    reg(FeatureSpec("AMIHUD_ILLIQUIDITY", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Amihud price impact ratio: |ret| / turnover", ("CLOSE", "VOLUME_QUOTE"), "Mean(|RET_1| / Turnover, 20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("ROLL_SPREAD", GrammarLayer.LAYER_0_RAW, FeatureFamily.MICROSTRUCTURE_SPREAD, "Roll's effective bid-ask spread estimator", ("CLOSE",), "2 * Sqrt(max(0, -Cov(Delta P_t, Delta P_{t-1})))", lookback=20, data_type=DataType.RATIO, units="bps"))
    reg(FeatureSpec("OU_MEAN_REVERSION", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Ornstein-Uhlenbeck continuous mean-reversion speed parameter", ("CLOSE",), "kappa = -ln(1 + beta_AR1)", lookback=60, data_type=DataType.RATIO))
    reg(FeatureSpec("SPREAD_HALF_LIFE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Ornstein-Uhlenbeck mean-reversion half-life (bars)", ("CLOSE",), "ln(2) / max(kappa, 1e-6)", lookback=60, data_type=DataType.TIME, units="bars"))
    reg(FeatureSpec("COINTEGRATION_RESIDUAL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIONSHIP, "Standardized z-score of spread to benchmark cointegrating vector", ("CLOSE", "BTC_CLOSE"), "ZScore(P_t - beta * BTC_t)", lookback=60, data_type=DataType.ZSCORE))
    reg(FeatureSpec("KALMAN_HEDGE_RATIO", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIONSHIP, "Dynamic Kalman filter hedge ratio against benchmark", ("CLOSE", "BTC_CLOSE"), "KalmanState(P_asset, P_benchmark)", lookback=60, data_type=DataType.RATIO))
    reg(FeatureSpec("ROLLING_SKEWNESS", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Trailing 30-bar return skewness (3rd standardized moment)", ("CLOSE",), "Skew(RET_1, 30)", lookback=30, data_type=DataType.RATIO))
    reg(FeatureSpec("ROLLING_KURTOSIS", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Trailing 30-bar return excess kurtosis (4th standardized moment)", ("CLOSE",), "Kurt(RET_1, 30)", lookback=30, data_type=DataType.RATIO))
    reg(FeatureSpec("REALIZED_SEMIVARIANCE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Trailing 20-bar downside semivariance of negative returns", ("CLOSE",), "Sum(min(RET_1, 0)^2, 20) / 20", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("VALUE_AT_RISK", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Parametric 99% 1-bar Value-at-Risk under normal assumptions", ("CLOSE",), "2.326 * RealizedVol(RET_1, 20)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("EXPECTED_SHORTFALL", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Conditional Value at Risk (CVaR / Expected Shortfall 97.5%)", ("CLOSE",), "Mean(RET_1 | RET_1 <= VaR)", lookback=60, data_type=DataType.RATIO))
    reg(FeatureSpec("VWAP_DISPLACEMENT", GrammarLayer.LAYER_1_PRIMITIVE, FeatureFamily.REFERENCE_DISTANCE, "Normalized price displacement from session VWAP", ("CLOSE", "VWAP_BAR"), "(P_close - VWAP) / VWAP", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("TRADE_HOLDING_TIME", GrammarLayer.LAYER_4_POLICY, FeatureFamily.POLICY_STATE, "Number of elapsed bars since current position entry", ("POSITION",), "t - t_entry", data_type=DataType.TIME, units="bars"))
    reg(FeatureSpec("DYNAMIC_TRAILING_STOP", GrammarLayer.LAYER_4_POLICY, FeatureFamily.RISK_CONTROL, "Distance in ATR units to trailing stop-loss boundary", ("CLOSE", "ATR"), "|P_close - Stop_level| / ATR", data_type=DataType.RATIO))
    reg(FeatureSpec("OPTIMAL_KELLY_FRACTION", GrammarLayer.LAYER_4_POLICY, FeatureFamily.RISK_CONTROL, "Kelly criterion optimal capital allocation leverage fraction", ("RET_1",), "mu / sigma^2", lookback=60, data_type=DataType.RATIO))
    reg(FeatureSpec("LEVERAGE_RATIO", GrammarLayer.LAYER_4_POLICY, FeatureFamily.RISK_CONTROL, "Gross portfolio notional exposure divided by total equity", ("POSITION", "EQUITY"), "Sum(|Notional|) / Equity", data_type=DataType.RATIO))
    reg(FeatureSpec("MARGIN_UTILIZATION_PCT", GrammarLayer.LAYER_4_POLICY, FeatureFamily.RISK_CONTROL, "Percentage of available maintenance margin equity utilized", ("EQUITY",), "Margin_Used / Total_Margin", data_type=DataType.RATIO))

    # ── Canonical Sandbox Signals X_00 through X_25 and Interaction Term X_26 ──
    reg(FeatureSpec("H-OFI", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Order Flow Imbalance delta (Sandbox Signal X_00)", ("OFI",), "CausalZScore(OFI)", lookback=60, data_type=DataType.ZSCORE))
    reg(FeatureSpec("H-CVD", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Cumulative Volume Delta / Taker Buy Ratio (Sandbox Signal X_01)", ("TAKER_BUY_BASE", "VOLUME"), "(2*V_tb - V) / max(V, eps)", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("H-P", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Volatility Persistence Estimator (Sandbox Signal X_02)", ("HIGH", "LOW"), "ParkinsonVol(14)", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("H-ASYM", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "GJR Downside Panic Leverage Asymmetry (Sandbox Signal X_03)", ("CLOSE",), "DownsideVol(20) / max(RealizedVol(20), eps)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("H-NATR", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Normalized Average True Range Percentage (Sandbox Signal X_04)", ("HIGH", "LOW", "CLOSE"), "ATR(14) / P * 100", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("H-SQUEEZE", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Bollinger Band / Keltner Squeeze Ratio (Sandbox Signal X_05)", ("HIGH", "LOW", "CLOSE"), "BB_WIDTH / KELTNER_WIDTH", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("H-MACD", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Moving Average Convergence Divergence Trend Momentum (Sandbox Signal X_06)", ("CLOSE",), "MACD_Hist(12, 26, 9)", lookback=35, data_type=DataType.RATIO))
    reg(FeatureSpec("H-EMA_CROSS", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Fast vs Slow Exponential Moving Average Spread (Sandbox Signal X_07)", ("CLOSE",), "(EMA_12 - EMA_26) / P", lookback=26, data_type=DataType.RATIO))
    reg(FeatureSpec("H-RSI", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MOMENTUM_OSCILLATORS, "Relative Strength Index Oscillator (Sandbox Signal X_08)", ("CLOSE",), "RSI(14)", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("H-BB", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLATILITY_RANGE, "Bollinger Bands %B Price Location (Sandbox Signal X_09)", ("CLOSE",), "(P - Lower_BB) / (Upper_BB - Lower_BB)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("H-MOM", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MOMENTUM_OSCILLATORS, "5-Bar Relative Momentum (Sandbox Signal X_10)", ("CLOSE",), "(P_t - P_{t-5}) / P_{t-5}", lookback=5, data_type=DataType.RETURN))
    reg(FeatureSpec("H-STOCH", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.MOMENTUM_OSCILLATORS, "Stochastic Oscillator %K (Sandbox Signal X_11)", ("HIGH", "LOW", "CLOSE"), "Stoch_K(14)", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("H-RV", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Trailing Volume Percentile Rank (Sandbox Signal X_12)", ("VOLUME",), "TrailingRank(V, 40)", lookback=40, data_type=DataType.RANK))
    reg(FeatureSpec("H-A", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Amihud Illiquidity Ratio (Sandbox Signal X_13)", ("CLOSE", "VOLUME"), "|RET_1| / max(V, eps)", lookback=1, data_type=DataType.RATIO))
    reg(FeatureSpec("H-LOG", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Natural Log of Traded Volume (Sandbox Signal X_14)", ("VOLUME",), "ln(V)", lookback=1, data_type=DataType.VOLUME))
    reg(FeatureSpec("H-MFI", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "Money Flow Index (Sandbox Signal X_15)", ("HIGH", "LOW", "CLOSE", "VOLUME"), "MFI(14)", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("H-OBV_MOM", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.VOLUME_ACTIVITY, "On-Balance Volume 10-Bar Momentum (Sandbox Signal X_16)", ("CLOSE", "VOLUME"), "PctChange(OBV, 10)", lookback=10, data_type=DataType.RATIO))
    reg(FeatureSpec("H-HURST", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.TREND_MOVING_AVERAGE, "Hurst Exponent Long-Memory Persistence (Sandbox Signal X_17)", ("CLOSE",), "HurstExponent(64)", lookback=64, data_type=DataType.RATIO))
    reg(FeatureSpec("H-CROSS_MOM", GrammarLayer.LAYER_2_FEATURE, FeatureFamily.CROSS_ASSET_RELATIVE, "Cross-Asset Relative Momentum against Universe (Sandbox Signal X_18)", ("CLOSE", "CLOSE_UNIVERSE"), "RET_1 - CSRankMean(RET_1)", lookback=1, data_type=DataType.RETURN))

    # Layer 3 Nonlinear Interaction Terms (X_19 through X_25, plus X_26)
    reg(FeatureSpec("NL_VOL_GATED_MOM", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.MOMENTUM_OSCILLATORS, "Volume-Gated Momentum: H-MOM * H-RV (Sandbox Signal X_19)", ("H-MOM", "H-RV"), "X_10 * X_12", lookback=40, data_type=DataType.RATIO))
    reg(FeatureSpec("NL_SQUEEZE_TREND", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.TREND_MOVING_AVERAGE, "Squeeze-Amplified Trend: H-MACD * H-SQUEEZE (Sandbox Signal X_20)", ("H-MACD", "H-SQUEEZE"), "X_06 * X_05", lookback=35, data_type=DataType.RATIO))
    reg(FeatureSpec("NL_RISK_ADJ_MOM", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.MOMENTUM_OSCILLATORS, "Volatility-Adjusted Momentum: H-MOM / H-NATR (Sandbox Signal X_21)", ("H-MOM", "H-NATR"), "X_10 / max(X_04, eps)", lookback=14, data_type=DataType.RATIO))
    reg(FeatureSpec("NL_OFI_REV", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "OFI Reversal at RSI Extremes: H-OFI * (0.5 - H-RSI) (Sandbox Signal X_22)", ("H-OFI", "H-RSI"), "X_00 * (0.5 - X_08)", lookback=60, data_type=DataType.RATIO))
    reg(FeatureSpec("NL_HURST_ADAPT", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.TREND_MOVING_AVERAGE, "Hurst-Adaptive Regime Filter: H-MOM * (H-HURST - 0.5) (Sandbox Signal X_23)", ("H-MOM", "H-HURST"), "X_10 * (X_17 - 0.5)", lookback=64, data_type=DataType.RATIO))
    reg(FeatureSpec("NL_TAKER_MOM", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Taker Buy Surge x Price Momentum: H-CVD * H-MOM (Sandbox Signal X_24)", ("H-CVD", "H-MOM"), "X_01 * X_10", lookback=5, data_type=DataType.RATIO))
    reg(FeatureSpec("NL_MASTER_SNR", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.VOLATILITY_RANGE, "Master Signal-to-Noise Ratio: H-MOM / (H-ASYM^2 + eps) (Sandbox Signal X_25)", ("H-MOM", "H-ASYM"), "X_10 / (X_03^2 + eps)", lookback=20, data_type=DataType.RATIO))
    reg(FeatureSpec("PRODUCT_H_BB_TAKER", GrammarLayer.LAYER_3_EQUATION, FeatureFamily.MICROSTRUCTURE_TRADEFLOW, "Interaction Term: Bollinger Band Position x Taker Flow (Signal X_26)", ("H-BB", "H-CVD"), "X_09 * X_01", lookback=20, data_type=DataType.RATIO))

    return cat


MASTER_FEATURE_CATALOG: dict[str, FeatureSpec] = _build_master_feature_catalogue()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Target & Outcome Grammar Catalogue (Section 11)
# ─────────────────────────────────────────────────────────────────────────────

def _build_master_target_catalogue() -> dict[str, TargetSpec]:
    """Define formal target specifications separated from predictive features."""
    targets: dict[str, TargetSpec] = {}

    def reg_t(spec: TargetSpec) -> None:
        targets[spec.target_id] = spec

    reg_t(TargetSpec("FWD_RET_1", "Next 1-bar simple return", "Return magnitude", 1, "(P_{t+1} - P_t) / P_t"))
    reg_t(TargetSpec("FWD_RET_4", "Next 4-bar simple return", "Return magnitude", 4, "(P_{t+4} - P_t) / P_t"))
    reg_t(TargetSpec("FWD_RET_16", "Next 16-bar simple return", "Return magnitude", 16, "(P_{t+16} - P_t) / P_t"))
    reg_t(TargetSpec("FWD_DIRECTION_1", "Binary direction of next bar return (+1 or -1)", "Directional classification", 1, "sign(P_{t+1} - P_t)", is_classification=True))
    reg_t(TargetSpec("FWD_CS_RANK_1", "Cross-sectional rank of next 1-bar return in universe", "Cross-sectional selection", 1, "CSRank(r_{t+1})"))
    reg_t(TargetSpec("FWD_REALIZED_VOL_16", "Realized volatility over next 16 bars", "Volatility forecast", 16, "Std(r_{t+1:t+16})"))
    reg_t(TargetSpec("FWD_MAX_DRAWDOWN_16", "Maximum path drawdown over next 16 bars", "Downside tail risk", 16, "max_excursion_down(16)"))
    reg_t(TargetSpec("FWD_TRIPLE_BARRIER_16", "Triple barrier label (+1 take-profit, -1 stop-loss, 0 timeout)", "Path barrier outcome", 16, "barrier_hit(pt, sl, tp, 16)", is_classification=True))

    return targets


MASTER_TARGET_CATALOG: dict[str, TargetSpec] = _build_master_target_catalogue()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Executable First-Stage Feature Extraction Pipeline (Section 13)
# ─────────────────────────────────────────────────────────────────────────────

def compute_recommended_first_stage_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    cs_matrix_close: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the recommended first-stage trading grammar features (Section 13).

    Accepts an asset DataFrame with columns:
      'open', 'high', 'low', 'close', 'volume', and optional 'taker_buy', 'ofi'.
    Returns a dictionary of named, causal 1D numpy arrays.
    """
    p = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float) if "open" in df.columns else p
    h = df["high"].to_numpy(float) if "high" in df.columns else p
    lo = df["low"].to_numpy(float) if "low" in df.columns else p
    v = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(p))
    tb = df["taker_buy"].to_numpy(float) if "taker_buy" in df.columns else v * 0.5
    ofi = df["ofi"].to_numpy(float) if "ofi" in df.columns else np.zeros(len(p))

    T = len(p)
    out: dict[str, np.ndarray] = {}

    # --- 1. Price and returns ---
    out["RET_1"] = op_pct_change(p, 1)
    out["RET_2"] = op_pct_change(p, 2)
    out["RET_4"] = op_pct_change(p, 4)
    out["RET_8"] = op_pct_change(p, 8)
    out["RET_16"] = op_pct_change(p, 16)
    out["LOGRET"] = op_log_return(p, 1)
    out["OPEN_TO_CLOSE"] = safe_div(p - o, o)

    # True range & Candle shape
    p_prev = op_lag(p, 1)
    p_prev[0] = p[0]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - p_prev), np.abs(lo - p_prev)))
    out["TRUE_RANGE"] = tr
    hl_range = np.maximum(h - lo, 1e-12)
    out["BODY_TO_RANGE"] = np.abs(p - o) / hl_range
    out["UPPER_WICK_TO_RANGE"] = (h - np.maximum(o, p)) / hl_range
    out["CLOSE_LOCATION"] = (p - lo) / hl_range

    # --- 2. Trend and momentum ---
    ema12 = ewma(p, 12)
    ema26 = ewma(p, 26)
    out["EMA_SPREAD"] = safe_div(ema12 - ema26, p)
    ema20 = ewma(p, 20)
    out["EMA_SLOPE"] = roll_slope(ema20, 5)
    macd_line = ema12 - ema26
    macd_sig = ewma(macd_line, 9)
    out["MACD"] = macd_line - macd_sig

    # RSI
    r1 = out["RET_1"]
    gain = pd.Series(np.where(r1 > 0, r1, 0.0)).rolling(14, min_periods=14).mean().to_numpy()
    loss = pd.Series(np.where(r1 < 0, -r1, 0.0)).rolling(14, min_periods=14).mean().to_numpy()
    rs = safe_div(gain, loss)
    out["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    out["ROC"] = op_pct_change(p, 10)

    # Trend efficiency ratio (Kaufman)
    net_chg = np.abs(p - op_lag(p, 20))
    gross_path = roll_sum(np.abs(p - op_lag(p, 1)), 20)
    out["TREND_EFFICIENCY"] = safe_div(net_chg, gross_path)

    # ADX proxy
    up_move = h - op_lag(h, 1)
    down_move = op_lag(lo, 1) - lo
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = roll_mean(tr, 14)
    plus_di = 100.0 * safe_div(roll_mean(plus_dm, 14), atr14)
    minus_di = 100.0 * safe_div(roll_mean(minus_dm, 14), atr14)
    dx = 100.0 * safe_div(np.abs(plus_di - minus_di), plus_di + minus_di)
    out["ADX"] = roll_mean(dx, 14)

    # --- 3. Volatility and breakouts ---
    out["ATR"] = atr14
    out["NATR"] = safe_div(atr14, p) * 100.0
    out["REALIZED_VOL"] = roll_std(out["RET_1"], 20)
    out["DOWNSIDE_VOL"] = safe_sqrt(roll_mean(negative_part(out["RET_1"]) ** 2, 20))

    # Range based volatility: Parkinson
    log_hl = safe_log(safe_div(h, lo))
    out["PARKINSON_VOL"] = safe_sqrt(roll_sum(log_hl ** 2, 20) / (4.0 * np.log(2.0) * 20.0))

    # Bollinger Bands
    sma20 = roll_mean(p, 20)
    std20 = roll_std(p, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    out["BB_WIDTH"] = safe_div(bb_upper - bb_lower, sma20)
    out["BB_PERCENT_B"] = safe_div(p - bb_lower, bb_upper - bb_lower)
    out["KELTNER_WIDTH"] = safe_div(3.0 * atr14, ema20)
    out["SQUEEZE_STATE"] = safe_div(out["BB_WIDTH"], out["KELTNER_WIDTH"])
    donchian_high_prev = op_lag(roll_max(h, 20), 1)
    donchian_low_prev = op_lag(roll_min(lo, 20), 1)
    up_bo = positive_part(p - donchian_high_prev)
    dn_bo = positive_part(donchian_low_prev - p)
    out["BREAKOUT_STRENGTH"] = safe_div(up_bo - dn_bo, atr14)

    # --- 4. Volume and activity ---
    out["LOG_VOLUME"] = safe_log(v)
    vol_mean20 = roll_mean(v, 20)
    out["RELATIVE_VOLUME"] = safe_div(v, vol_mean20)
    out["VOLUME_ZSCORE"] = causal_zscore(v)

    # OBV Momentum
    obv = np.zeros(T)
    for t_i in range(1, T):
        obv[t_i] = obv[t_i - 1] + (v[t_i] if p[t_i] > p[t_i - 1] else (-v[t_i] if p[t_i] < p[t_i - 1] else 0.0))
    out["OBV_MOM"] = op_pct_change(obv, 10)

    # Taker Imbalance & Flow
    out["TAKER_IMBALANCE"] = safe_div(2.0 * tb - v, v)
    signed_vol = 2.0 * tb - v
    out["SIGNED_VOLUME_MOMENTUM"] = roll_slope(signed_vol, 5)

    # Chaikin Money Flow
    mf_multiplier = safe_div((p - lo) - (h - p), hl_range)
    mf_volume = mf_multiplier * v
    out["CMF"] = safe_div(roll_sum(mf_volume, 20), roll_sum(v, 20))

    # --- 5. Execution and liquidity proxies ---
    # Spread proxy: Corwin-Schultz high-low proxy or 5 bps floor
    out["SPREAD_BPS"] = np.full(T, 5.0)  # Standard 5.0 bps exchange baseline
    out["DEPTH_IMBALANCE"] = causal_zscore(ofi)
    out["MICROPRICE_DISPLACEMENT"] = out["TAKER_IMBALANCE"] * out["NATR"] * 0.01

    # --- 6. Cross-asset context ---
    if btc_df is not None and "close" in btc_df.columns:
        btc_p = btc_df["close"].to_numpy(float)
        btc_r = op_pct_change(btc_p, 1)
        out["BTC_RETURN"] = btc_r
        out["BTC_REALIZED_VOL"] = roll_std(btc_r, 20)
        out["ROLLING_BETA_TO_BTC"] = roll_beta(out["RET_1"], btc_r, 30)
        out["CORRELATION_TO_BTC"] = roll_corr(out["RET_1"], btc_r, 30)
    else:
        out["BTC_RETURN"] = np.zeros(T)
        out["BTC_REALIZED_VOL"] = np.zeros(T)
        out["ROLLING_BETA_TO_BTC"] = np.ones(T)
        out["CORRELATION_TO_BTC"] = np.ones(T)

    if cs_matrix_close is not None:
        # Cross sectional return rank
        cs_returns = np.zeros_like(cs_matrix_close)
        cs_returns[1:] = (cs_matrix_close[1:] - cs_matrix_close[:-1]) / np.maximum(cs_matrix_close[:-1], 1e-12)
        ranks = cross_sectional_rank(cs_returns)
        # Assuming asset is column 0 if unknown
        out["CROSS_SECTIONAL_RETURN_RANK"] = ranks[:, 0]
    else:
        out["CROSS_SECTIONAL_RETURN_RANK"] = np.full(T, 0.5)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Registry / Query Engine Class
# ─────────────────────────────────────────────────────────────────────────────

class TradingGrammarCatalog:
    """Master engine for querying, verifying, and serializing APEDP's trading grammar."""

    def __init__(self, catalog: dict[str, FeatureSpec] | None = None):
        self._catalog = catalog if catalog is not None else MASTER_FEATURE_CATALOG

    @property
    def total_features(self) -> int:
        return len(self._catalog)

    def get_feature(self, feature_id: str) -> FeatureSpec:
        if feature_id not in self._catalog:
            raise KeyError(f"Feature '{feature_id}' is not registered in TradingGrammarCatalog.")
        return self._catalog[feature_id]

    def list_features(self) -> list[str]:
        return sorted(self._catalog.keys())

    def filter_by_layer(self, layer: GrammarLayer) -> list[FeatureSpec]:
        return [f for f in self._catalog.values() if f.layer == layer]

    def filter_by_family(self, family: FeatureFamily) -> list[FeatureSpec]:
        return [f for f in self._catalog.values() if f.family == family]

    def filter_by_data_type(self, data_type: DataType) -> list[FeatureSpec]:
        return [f for f in self._catalog.values() if f.data_type == data_type]

    def get_first_stage_feature_ids(self) -> list[str]:
        """Return the pre-registered first-stage feature identifiers (Section 13)."""
        first_stage = [
            "RET_1", "RET_2", "RET_4", "RET_8", "RET_16", "LOGRET_1", "OPEN_TO_CLOSE",
            "TRUE_RANGE", "BODY_TO_RANGE", "CLOSE_LOCATION", "EMA_SPREAD", "EMA_SLOPE",
            "MACD", "RSI", "ROC", "ADX", "TREND_EFFICIENCY", "ATR", "NATR",
            "REALIZED_VOL", "DOWNSIDE_VOL", "PARKINSON_VOL", "BB_WIDTH", "BB_PERCENT_B",
            "KELTNER_WIDTH", "SQUEEZE_STATE", "BREAKOUT_STRENGTH", "LOG_VOLUME",
            "RELATIVE_VOLUME", "VOLUME_ZSCORE", "OBV_MOM", "TAKER_IMBALANCE",
            "SIGNED_VOLUME_MOMENTUM", "CMF", "SPREAD_BPS", "DEPTH_IMBALANCE",
            "MICROPRICE_DISPLACEMENT", "BTC_RETURN", "BTC_REALIZED_VOL",
            "ROLLING_BETA_TO_BTC", "CORRELATION_TO_BTC", "CROSS_SECTIONAL_RETURN_RANK",
        ]
        return [fid for fid in first_stage if fid in self._catalog]

    def to_json(self, indent: int = 2) -> str:
        """Serialize full catalogue to JSON string."""
        serialized = {k: v.to_dict() for k, v in self._catalog.items()}
        return json.dumps(serialized, indent=indent)

    def compute_catalog_fingerprint(self) -> str:
        """SHA-256 fingerprint guaranteeing catalogue integrity and provenance."""
        canonical = sorted([
            (k, v.layer.value, v.family.value, v.formula, v.lookback, v.units)
            for k, v in self._catalog.items()
        ])
        raw = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
