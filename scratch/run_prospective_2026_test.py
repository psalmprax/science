"""Prospective Independent Test of Frozen Model 4b (N = 15) on 2026 Crypto Data.

Fetches daily candles directly from Binance public API with late-2025 warmup.
Evaluates strictly out-of-sample on 2026 calendar days [2026-01-01 to 2026-10-09].
Applies pre-registered APEDP evaluation criteria.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")

from scratch.standalone_model4b_test_suite import (
    CRYPTO_15,
    MODEL_4B_SPEC,
    FEE_RATE,
    calc_canonical_signals,
    get_weights_matrix,
    simulate_portfolio_wealth,
)

# Polygon migrated MATIC to POL in Sep 2024
BINANCE_SYMBOLS = {
    "BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "BNB": "BNB", "XRP": "XRP",
    "ADA": "ADA", "AVAX": "AVAX", "DOGE": "DOGE", "DOT": "DOT", "LINK": "LINK",
    "MATIC": "POL", "NEAR": "NEAR", "ATOM": "ATOM", "UNI": "UNI", "LTC": "LTC"
}


def fetch_symbol_klines(binance_sym: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch daily candles from Binance public API."""
    pair = f"{binance_sym}USDT"
    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&startTime={start_ms}&endTime={end_ms}&limit=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "APEDP-Auditor/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    rows = []
    for k in data:
        rows.append({
            "timestamp": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy": float(k[9]),
            "ofi": 2.0 * float(k[9]) - float(k[5]),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
    return df


def run_2026_test():
    print("=" * 100)
    print("APEDP PROSPECTIVE 2026 INDEPENDENT EVALUATION: FROZEN MODEL 4b (N = 15 CRYPTO)")
    print("Frozen Equation: S = +1.1948*H-BB + 0.2227*H-STOCH + 0.4501*H-OBV_MOM - 0.4839*NL_HURST_ADAPT")
    print(f"Frozen Threshold: theta = 0.1695 | Fee Rate: 5.0 bps flat | Weight Turnover Accounting")
    print("=" * 100)

    # Warmup start: 2025-08-01 (153 days warmup prior to 2026-01-01)
    warmup_start_ms = 1754006400000  # 2025-08-01 00:00:00 UTC
    end_ms = 1791590400000           # 2026-10-09 23:59:59 UTC
    test_start_date = "2026-01-01"

    print("\nFetching daily klines from Binance public API for 15 assets...")
    uni_data = {}
    date_series = None

    for sym in CRYPTO_15:
        b_sym = BINANCE_SYMBOLS[sym]
        df = fetch_symbol_klines(b_sym, warmup_start_ms, end_ms)
        df["r"] = np.append(0.0, np.diff(np.log(df["close"].to_numpy())))
        uni_data[sym] = df
        if date_series is None:
            date_series = df["date"].to_numpy()
        print(f"  • {sym:<6} (Binance: {b_sym+'USDT':<8}): {len(df)} total bars ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")

    # Align dates across assets
    min_len = min(len(df) for df in uni_data.values())
    for sym in CRYPTO_15:
        uni_data[sym] = uni_data[sym].iloc[:min_len]
    date_series = date_series[:min_len]

    test_idx = int(np.where(date_series >= test_start_date)[0][0])
    total_bars = len(date_series)
    eval_bars = total_bars - test_idx

    print(f"\nWarmup period: {date_series[0]} to {date_series[test_idx-1]} ({test_idx} bars)")
    print(f"Prospective 2026 Test Period: {date_series[test_idx]} to {date_series[-1]} ({eval_bars} bars)")

    # 1. Feature Extraction across 15 assets
    rets_15 = np.column_stack([uni_data[s]["r"].to_numpy() for s in CRYPTO_15])
    cross_med = np.median(rets_15, axis=1)

    vols_15 = np.zeros_like(rets_15)
    for i, s in enumerate(CRYPTO_15):
        r_s = pd.Series(uni_data[s]["r"].to_numpy())
        vols_15[:, i] = r_s.rolling(20, min_periods=20).std().fillna(0.01).to_numpy()

    feats_15 = np.zeros((total_bars, 15, 26))
    for i, s in enumerate(CRYPTO_15):
        feats_15[:, i, :] = calc_canonical_signals(uni_data[s], cross_med)

    # 2. Slice strictly for 2026 prospective evaluation
    f_test = feats_15[test_idx:]
    r_test = rets_15[test_idx:]
    v_test = vols_15[test_idx:]
    dates_test = date_series[test_idx:]

    # 3. Model 4b Evaluation (Frozen Block A parameters)
    spec15 = MODEL_4B_SPEC[15]
    w_vec = np.zeros(26)
    for k_idx, val in spec15["weights"].items():
        w_vec[k_idx] = val
    th = spec15["threshold"]

    scores_raw = f_test @ w_vec
    w_mat_m4b = get_weights_matrix(scores_raw, th, v_test)
    m_m4b = simulate_portfolio_wealth(w_mat_m4b, r_test)

    # 4. Equal-Weighted Buy & Hold Benchmark on 2026
    T_bh = len(r_test)
    W_bh = 1.0
    w_path_bh = [1.0]
    bh_rets = []
    for t in range(1, T_bh):
        r_bh_t = float(np.mean(np.expm1(r_test[t])))
        W_bh *= (1.0 + r_bh_t)
        w_path_bh.append(W_bh)
        bh_rets.append(r_bh_t)

    bh_arr = np.array(w_path_bh)
    bh_peaks = np.maximum.accumulate(bh_arr)
    bh_dds = (bh_arr - bh_peaks) / np.maximum(bh_peaks, 1e-12)
    bh_max_dd = float(np.abs(np.min(bh_dds)) * 100.0)
    bh_net_ret = float((W_bh - 1.0) * 100.0)
    bh_sd = np.std(bh_rets)
    bh_sr252 = float(np.mean(bh_rets) / bh_sd * np.sqrt(252)) if bh_sd > 1e-9 else 0.0
    bh_sr365 = float(np.mean(bh_rets) / bh_sd * np.sqrt(365)) if bh_sd > 1e-9 else 0.0

    print("\n" + "=" * 100)
    print("PROSPECTIVE 2026 EVALUATION RESULTS (2026-01-01 to 2026-10-09)")
    print("=" * 100)
    print(f"Evaluation Window: {eval_bars} calendar trading bars (unseen forward data)")
    print(f"\n[BENCHMARK: 15-Crypto Equal-Weighted Buy & Hold]")
    print(f"  • Cumulative Return: {bh_net_ret:>+6.2f}%")
    print(f"  • Annualized Sharpe (sqrt 252): {bh_sr252:>5.2f}")
    print(f"  • Annualized Sharpe (sqrt 365): {bh_sr365:>5.2f}")
    print(f"  • Maximum Drawdown: {bh_max_dd:>5.1f}%")
    print(f"  • Daily Turnover: 0.0000")

    print(f"\n[MODEL 4b: Frozen N = 15 Equation]")
    print(f"  • Cumulative Net Return:   {m_m4b['net_return']:>+6.2f}%")
    print(f"  • Cumulative Gross Return: {m_m4b['gross_return']:>+6.2f}%")
    print(f"  • Transaction Cost Drag:   {m_m4b['gross_return'] - m_m4b['net_return']:>6.2f} pp")
    print(f"  • Annualized Sharpe (sqrt 252): {m_m4b['sharpe_252']:>5.2f}")
    print(f"  • Annualized Sharpe (sqrt 365): {m_m4b['sharpe_365']:>5.2f}")
    print(f"  • Maximum Drawdown:        {m_m4b['max_drawdown']:>5.1f}%")
    print(f"  • Mean Daily Weight Turnover: {m_m4b['mean_daily_turnover']:.4f}")
    print(f"  • Total Weight Turnover:      {m_m4b['total_turnover']:.1f}")

    # Active trading positions breakdown
    pos_activity = np.sum(np.abs(w_mat_m4b), axis=1)
    active_days = int(np.sum(pos_activity > 0.01))
    print(f"  • Active Trading Days: {active_days} / {len(w_mat_m4b)} ({active_days/len(w_mat_m4b)*100:.1f}%)")

    # 5. Pre-Registered Epistemic Classification
    print("\n" + "=" * 100)
    print("PRE-REGISTERED EPISTEMIC VERDICT CLASSIFICATION")
    print("=" * 100)

    pass_conditions = [
        (m_m4b['net_return'] > bh_net_ret, f"Net Return ({m_m4b['net_return']:+.2f}%) > Buy & Hold ({bh_net_ret:+.2f}%)"),
        (m_m4b['sharpe_252'] >= 0.70, f"Sharpe(252) ({m_m4b['sharpe_252']:.2f}) >= +0.70"),
        (m_m4b['max_drawdown'] <= 25.0, f"Max Drawdown ({m_m4b['max_drawdown']:.1f}%) <= 25.0%"),
    ]

    fail_conditions = [
        (m_m4b['net_return'] < 0.0, f"Net Return is negative ({m_m4b['net_return']:+.2f}%)"),
        (m_m4b['sharpe_252'] < 0.30, f"Sharpe(252) is substandard ({m_m4b['sharpe_252']:.2f} < +0.30)"),
        (m_m4b['max_drawdown'] > 35.0, f"Max Drawdown exceeds threshold ({m_m4b['max_drawdown']:.1f}% > 35.0%)"),
    ]

    for cond, desc in pass_conditions:
        print(f"  PASS Check: [{'PASS' if cond else 'FAIL'}] {desc}")
    for cond, desc in fail_conditions:
        print(f"  FAIL Check: [{'TRIGGERED' if cond else 'CLEAR'}] {desc}")

    all_pass = all(c[0] for c in pass_conditions)
    any_fail = any(c[0] for c in fail_conditions)

    if all_pass and not any_fail:
        verdict = "PASS"
        verdict_detail = "Model 4b successfully outperformed Buy & Hold with controlled drawdown and robust risk-adjusted return."
    elif any_fail:
        verdict = "FAIL"
        verdict_detail = "Model 4b triggered one or more pre-registered failure criteria on prospective 2026 data."
    else:
        verdict = "INSUFFICIENT EVIDENCE"
        verdict_detail = "Model 4b failed to achieve pre-registered PASS thresholds, but did not breach hard failure bounds."

    print(f"\nFINAL OFFICIAL CLASSIFICATION: >>> {verdict} <<<")
    print(f"Detail: {verdict_detail}")
    print("=" * 100)

    # Save artifact
    res_payload = {
        "evaluation_period": f"{date_series[test_idx]} to {date_series[-1]}",
        "eval_bars": eval_bars,
        "model_4b_results": m_m4b,
        "buy_and_hold_results": {
            "net_return": round(bh_net_ret, 2),
            "sharpe_252": round(bh_sr252, 2),
            "sharpe_365": round(bh_sr365, 2),
            "max_drawdown": round(bh_max_dd, 1),
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }
    out_path = Path("scratch/prospective_2026_model4b_test_results.json")
    out_path.write_text(json.dumps(res_payload, indent=2))
    print(f"\nArtifact saved to: {out_path.resolve()}")
    return res_payload


if __name__ == "__main__":
    run_2026_test()
