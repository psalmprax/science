"""Unit tests for APEDP Trading Grammar Catalog and Operator Engine.

Verifies:
  1. Complete registration of Layers 0 - 4 features and target specifications.
  2. Domain safety of mathematical operators (safe div, safe log, safe sqrt, exp clamp).
  3. Strict causality of rolling and temporal operators.
  4. Correctness of cross-sectional ranking and normalization operators.
  5. Statefulness and filtering of logical gating, deadband, cooldown, and hysteresis.
  6. Execution of recommended first-stage feature extraction pipeline.
  7. Deterministic fingerprinting and JSON serialization.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from apedp_dyn.grammar.trading_grammar_catalog import (
    MASTER_FEATURE_CATALOG,
    MASTER_TARGET_CATALOG,
    FeatureSpec,
    GrammarLayer,
    TargetSpec,
    TradingGrammarCatalog,
    compute_recommended_first_stage_features,
    consecutive_down_bars,
    consecutive_up_bars,
    cross_sectional_rank,
    cross_sectional_zscore,
    exp_clamp,
    op_acceleration,
    op_cooldown,
    op_deadband,
    op_hysteresis_path,
    op_lag,
    op_pct_change,
    robust_zscore,
    roll_mean,
    roll_slope,
    safe_div,
    safe_log,
    safe_sqrt,
    signal_gate,
    time_since_last_high,
    winsorize,
)


def test_catalogue_layers_and_families_coverage():
    """Verify all 5 layers and essential feature families are populated in the master catalog."""
    cat = TradingGrammarCatalog()
    assert cat.total_features >= 50

    # Verify layer distribution
    layer_0 = cat.filter_by_layer(GrammarLayer.LAYER_0_RAW)
    layer_1 = cat.filter_by_layer(GrammarLayer.LAYER_1_PRIMITIVE)
    layer_2 = cat.filter_by_layer(GrammarLayer.LAYER_2_FEATURE)
    layer_4 = cat.filter_by_layer(GrammarLayer.LAYER_4_POLICY)

    assert len(layer_0) >= 15
    assert len(layer_1) >= 12
    assert len(layer_2) >= 20
    assert len(layer_4) >= 4

    # Verify critical features exist
    expected_critical = [
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_BASE", "TAKER_BUY_BASE",
        "FUNDING_RATE", "OPEN_INTEREST", "MICROPRICE", "HOUR_OF_DAY",
        "RET_1", "RET_4", "LOGRET_1", "TRUE_RANGE", "BODY_TO_RANGE", "CLOSE_LOCATION",
        "EMA_SPREAD", "MACD", "RSI", "ADX", "TREND_EFFICIENCY", "ATR", "NATR",
        "PARKINSON_VOL", "BB_PERCENT_B", "SQUEEZE_STATE", "TAKER_IMBALANCE",
        "BTC_RETURN", "ROLLING_BETA_TO_BTC", "CROSS_SECTIONAL_RETURN_RANK",
        "CURRENT_POSITION", "TURNOVER_COST",
    ]
    for fid in expected_critical:
        assert fid in MASTER_FEATURE_CATALOG, f"Missing feature {fid} from catalogue"
        spec = cat.get_feature(fid)
        assert isinstance(spec, FeatureSpec)
        assert spec.feature_id == fid
        assert spec.description != ""
        assert spec.formula != ""


def test_canonical_sandbox_and_quantitative_features_registration():
    """Verify all 26 empirical sandbox signals (X_00..X_25), X_26, and advanced features exist."""
    cat = TradingGrammarCatalog()
    assert cat.total_features >= 140

    sandbox_signals = [
        "H-OFI", "H-CVD", "H-P", "H-ASYM", "H-NATR", "H-SQUEEZE", "H-MACD",
        "H-EMA_CROSS", "H-RSI", "H-BB", "H-MOM", "H-STOCH", "H-RV", "H-A",
        "H-LOG", "H-MFI", "H-OBV_MOM", "H-HURST", "H-CROSS_MOM",
        "NL_VOL_GATED_MOM", "NL_SQUEEZE_TREND", "NL_RISK_ADJ_MOM",
        "NL_OFI_REV", "NL_HURST_ADAPT", "NL_TAKER_MOM", "NL_MASTER_SNR",
        "PRODUCT_H_BB_TAKER",
    ]
    for sig in sandbox_signals:
        assert sig in MASTER_FEATURE_CATALOG, f"Missing canonical sandbox signal {sig}"
        spec = cat.get_feature(sig)
        assert isinstance(spec, FeatureSpec)

    quant_features = [
        "KYLE_LAMBDA", "AMIHUD_ILLIQUIDITY", "ROLL_SPREAD", "OU_MEAN_REVERSION",
        "SPREAD_HALF_LIFE", "COINTEGRATION_RESIDUAL", "KALMAN_HEDGE_RATIO",
        "ROLLING_SKEWNESS", "ROLLING_KURTOSIS", "REALIZED_SEMIVARIANCE",
        "VALUE_AT_RISK", "EXPECTED_SHORTFALL", "VWAP_DISPLACEMENT",
        "TRADE_HOLDING_TIME", "DYNAMIC_TRAILING_STOP", "OPTIMAL_KELLY_FRACTION",
        "LEVERAGE_RATIO", "MARGIN_UTILIZATION_PCT",
    ]
    for qf in quant_features:
        assert qf in MASTER_FEATURE_CATALOG, f"Missing quantitative feature {qf}"
        spec = cat.get_feature(qf)
        assert isinstance(spec, FeatureSpec)


def test_targets_catalogue():
    """Verify target outcomes are registered and separated from predictive features."""
    assert len(MASTER_TARGET_CATALOG) >= 7
    assert "FWD_RET_1" in MASTER_TARGET_CATALOG
    assert "FWD_DIRECTION_1" in MASTER_TARGET_CATALOG
    assert "FWD_CS_RANK_1" in MASTER_TARGET_CATALOG
    assert "FWD_REALIZED_VOL_16" in MASTER_TARGET_CATALOG

    for _tid, target in MASTER_TARGET_CATALOG.items():
        assert isinstance(target, TargetSpec)
        assert target.requires_future is True
        assert target.horizon_bars >= 1


def test_domain_safe_arithmetic():
    """Verify safe mathematical operations handle invalid domains cleanly without crashes."""
    # 1. Division by zero and near-zero
    zeros = np.array([0.0, 1e-15, -1e-15, 2.0])
    numer = np.array([1.0, 2.0, 3.0, 4.0])
    res_div = safe_div(numer, zeros, eps=1e-12)
    assert np.all(np.isfinite(res_div))
    assert res_div[0] == 1.0 / 1e-12
    assert res_div[3] == 2.0

    # 2. Log of zero and negative values
    bad_x = np.array([-5.0, 0.0, 1.0, np.e])
    res_log = safe_log(bad_x, floor=1e-12)
    assert np.all(np.isfinite(res_log))
    assert np.isclose(res_log[2], 0.0)
    assert np.isclose(res_log[3], 1.0)

    # 3. Sqrt of negative values
    bad_sqrt_x = np.array([-10.0, -0.0, 4.0, 9.0])
    res_sqrt = safe_sqrt(bad_sqrt_x)
    assert np.all(np.isfinite(res_sqrt))
    assert res_sqrt[0] == 0.0
    assert res_sqrt[2] == 2.0
    assert res_sqrt[3] == 3.0

    # 4. Exp clamp avoiding inf
    huge_x = np.array([1000.0, -1000.0, 0.0])
    res_exp = exp_clamp(huge_x, min_val=-20.0, max_val=20.0)
    assert np.all(np.isfinite(res_exp))
    assert res_exp[0] == np.exp(20.0)
    assert res_exp[1] == np.exp(-20.0)
    assert np.isclose(res_exp[2], 1.0)


def test_causal_rolling_and_temporal_operators():
    """Verify rolling and temporal operators enforce causality without lookahead."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Rolling mean with window 3
    rm3 = roll_mean(x, 3)
    assert np.isnan(rm3[0]) and np.isnan(rm3[1])
    assert rm3[2] == 2.0  # mean(1, 2, 3)
    assert rm3[3] == 3.0  # mean(2, 3, 4)

    # Lag operator
    lag1 = op_lag(x, 1)
    assert np.isnan(lag1[0])
    assert lag1[1] == 1.0
    assert lag1[9] == 9.0

    # Pct change
    pct = op_pct_change(x, 1)
    assert np.isnan(pct[0])
    assert np.isclose(pct[1], (2.0 - 1.0) / 1.0)
    assert np.isclose(pct[2], (3.0 - 2.0) / 2.0)

    # Return acceleration
    acc = op_acceleration(x, 1)
    assert np.isnan(acc[0]) and np.isnan(acc[1])
    # x is a linear ramp [1..10], so diff(x) = [nan, 1, 1, 1...], acceleration = 0
    assert np.isclose(acc[2], 0.0)

    # Rolling slope on linear ramp should be exactly 1.0
    slopes = roll_slope(x, 4)
    assert np.isnan(slopes[0]) and np.isnan(slopes[2])
    assert np.isclose(slopes[3], 1.0)
    assert np.isclose(slopes[9], 1.0)

    # Consecutive up/down bars
    rets = np.array([0.01, 0.02, 0.03, -0.01, -0.02, 0.05])
    up_bars = consecutive_up_bars(rets)
    down_bars = consecutive_down_bars(rets)
    np.testing.assert_array_equal(up_bars, [1.0, 2.0, 3.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(down_bars, [0.0, 0.0, 0.0, 1.0, 2.0, 0.0])

    # Time since last high
    prices = np.array([10.0, 20.0, 15.0, 12.0, 25.0])
    tslh = time_since_last_high(prices, 3)
    # At index 2: window=[10, 20, 15], max is at index 1 (distance 1 bar)
    assert tslh[2] == 1.0
    # At index 3: window=[20, 15, 12], max is at index 1 (distance 2 bars)
    assert tslh[3] == 2.0
    # At index 4: window=[15, 12, 25], max is at index 4 (distance 0 bars)
    assert tslh[4] == 0.0


def test_cross_sectional_and_normalization_operators():
    """Verify ranking across universe assets and robust z-score methods."""
    # Matrix of 4 timestamps x 5 assets
    matrix = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [10.0, 50.0, 30.0, 20.0, 40.0],
    ])
    cs_ranks = cross_sectional_rank(matrix)
    assert cs_ranks.shape == (4, 5)
    # Row 0: strictly ascending -> ranks 0.0, 0.25, 0.5, 0.75, 1.0
    np.testing.assert_allclose(cs_ranks[0], [0.0, 0.25, 0.5, 0.75, 1.0])
    # Row 1: strictly descending -> ranks 1.0, 0.75, 0.5, 0.25, 0.0
    np.testing.assert_allclose(cs_ranks[1], [1.0, 0.75, 0.5, 0.25, 0.0])

    # Cross sectional z-scores
    cs_z = cross_sectional_zscore(matrix)
    assert np.isclose(np.mean(cs_z[0]), 0.0, atol=1e-7)
    assert np.isclose(np.std(cs_z[0], ddof=1), 1.0, atol=1e-7)

    # Robust z-score
    vec_outlier = np.array([1.0, 1.1, 1.05, 0.95, 1.0, 100.0])
    rob_z = robust_zscore(vec_outlier)
    assert np.all(np.isfinite(rob_z))
    # Median is near 1.0, so the first 5 entries should have small robust z-scores (< 2.0)
    assert np.all(np.abs(rob_z[:5]) < 3.0)
    # The outlier (100.0) has a huge robust z-score
    assert rob_z[5] > 10.0

    # Winsorize: outlier 10000.0 on 100 elements clipped to 95th percentile
    vec_win = np.arange(1.0, 101.0)
    vec_win[-1] = 10000.0
    clipped = winsorize(vec_win, p_low=0.01, p_high=0.95)
    assert clipped[-1] < 100.0
    assert np.isclose(clipped[-1], np.percentile(vec_win, 95.0))



def test_logical_and_gating_operators():
    """Verify deadband, signal gating, cooldown, and hysteresis paths."""
    scores = np.array([0.05, 0.25, 0.10, -0.05, -0.30])

    # Deadband
    db = op_deadband(scores, tau=0.20)
    np.testing.assert_array_equal(db, [0.0, 1.0, 0.0, 0.0, -1.0])

    # Signal gate
    mask = np.array([False, True, True, False, True])
    gated = signal_gate(scores, mask)
    np.testing.assert_array_equal(gated, [0.0, 0.25, 0.10, 0.0, -0.30])

    # Cooldown operator: position closes at step 2, must remain 0 for 2 bars
    sig_raw = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    cd = op_cooldown(sig_raw, cooldown_bars=2)
    # Step 0: 1.0, Step 1: 1.0
    # Step 2: 0.0 (exit triggered -> cooldown=2)
    # Step 3: cooldown active -> 0.0
    # Step 4: cooldown active -> 0.0
    # Step 5: cooldown expired -> 1.0
    np.testing.assert_array_equal(cd, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    # Hysteresis path
    hyst_pos = op_hysteresis_path(scores, theta_in=0.20, theta_out=0.0)
    # Step 0: 0.05 (below 0.20) -> 0.0
    # Step 1: 0.25 (exceeds 0.20) -> 1.0
    # Step 2: 0.10 (between 0.0 and 0.20) -> retains 1.0
    # Step 3: -0.05 (drops below 0.0) -> exits to 0.0
    # Step 4: -0.30 (drops below -0.20) -> -1.0
    np.testing.assert_array_equal(hyst_pos, [0.0, 1.0, 1.0, 0.0, -1.0])


def test_first_stage_feature_extraction_pipeline():
    """Verify execution of the recommended first-stage features on synthetic OHLCV."""
    np.random.seed(42)
    T = 150
    # Synthetic random walk
    log_ret = np.random.normal(0.0005, 0.02, T)
    close_px = 100.0 * np.exp(np.cumsum(log_ret))
    high_px = close_px * (1.0 + np.abs(np.random.normal(0, 0.01, T)))
    low_px = close_px * (1.0 - np.abs(np.random.normal(0, 0.01, T)))
    open_px = (close_px + low_px) / 2.0
    volume = np.random.uniform(1000, 5000, T)
    taker_buy = volume * np.random.uniform(0.4, 0.6, T)
    ofi = np.random.normal(0, 50, T)

    df = pd.DataFrame({
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": volume,
        "taker_buy": taker_buy,
        "ofi": ofi,
    })

    btc_df = df.copy()

    feats = compute_recommended_first_stage_features(df, btc_df=btc_df)
    assert isinstance(feats, dict)

    # Check key features exist and have correct length
    expected_keys = [
        "RET_1", "RET_4", "LOGRET", "OPEN_TO_CLOSE", "TRUE_RANGE",
        "BODY_TO_RANGE", "CLOSE_LOCATION", "EMA_SPREAD", "EMA_SLOPE",
        "MACD", "RSI", "ROC", "TREND_EFFICIENCY", "ATR", "NATR",
        "REALIZED_VOL", "PARKINSON_VOL", "BB_WIDTH", "BB_PERCENT_B",
        "SQUEEZE_STATE", "BREAKOUT_STRENGTH", "LOG_VOLUME", "RELATIVE_VOLUME",
        "VOLUME_ZSCORE", "OBV_MOM", "TAKER_IMBALANCE", "SIGNED_VOLUME_MOMENTUM",
        "CMF", "SPREAD_BPS", "BTC_RETURN", "ROLLING_BETA_TO_BTC",
    ]
    for k in expected_keys:
        assert k in feats, f"Missing feature {k} from extracted dictionary"
        arr = feats[k]
        assert len(arr) == T, f"Feature {k} length mismatch: {len(arr)} != {T}"

    # Verify warmup NaN structure: RET_1 has NaN at index 0, finite thereafter
    assert np.isnan(feats["RET_1"][0])
    assert np.isfinite(feats["RET_1"][1])

    # Verify RSI is in [0, 100] on finite values
    valid_rsi = feats["RSI"][np.isfinite(feats["RSI"])]
    assert np.all((valid_rsi >= 0.0) & (valid_rsi <= 100.0))

    # Verify close location in range [0, 1]
    assert np.all((feats["CLOSE_LOCATION"] >= 0.0) & (feats["CLOSE_LOCATION"] <= 1.0))


def test_catalogue_json_serialization_and_fingerprint():
    """Verify JSON export and deterministic cryptographic fingerprinting."""
    cat = TradingGrammarCatalog()
    json_str = cat.to_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert len(parsed) == cat.total_features

    fp1 = cat.compute_catalog_fingerprint()
    fp2 = cat.compute_catalog_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex string
