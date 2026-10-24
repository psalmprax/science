"""Multitimeframe Genetic Search on Expanded 46-Signal Grammar Space.

Evolves non-linear equations across 1m, 5m, 15m, 30m, 45m, 60m intervals
using the comprehensive 46-feature grammar space (26 legacy + 20 newly added primitives & transforms).
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    EXPANDED_SIGNAL_NAMES,
    calc_expanded_signals_for_asset,
    run_multi_asset_genetic_algorithm,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
SHORT_SYMS = ["BTC", "ETH", "SOL", "BNB", "DOGE"]
INTERVALS = ["1m", "5m", "15m", "30m", "45m", "60m"]
BARS_LIMIT = 1000

short_names = [n.split()[0] for n in EXPANDED_SIGNAL_NAMES]


def fetch_klines(sym: str, interval: str, limit: int = BARS_LIMIT) -> pd.DataFrame:
    fetch_inv = "15m" if interval == "45m" else ("1h" if interval == "60m" else interval)
    fetch_lim = limit * 3 if interval == "45m" else limit
    url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={fetch_inv}&limit={fetch_lim}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "vol",
        "close_time", "q_vol", "trades", "tb_vol", "tb_q_vol", "ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["price"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["V"] = df["vol"].astype(float)
    df["taker_buy"] = df["tb_vol"].astype(float)

    if interval == "45m":
        n_chunks = len(df) // 3
        df = df.iloc[:n_chunks * 3]
        res = []
        for i in range(n_chunks):
            sub = df.iloc[i * 3:(i + 1) * 3]
            res.append({
                "open": sub["open"].iloc[0],
                "price": sub["price"].iloc[-1],
                "high": sub["high"].max(),
                "low": sub["low"].min(),
                "close": sub["close"].iloc[-1],
                "V": sub["V"].sum(),
                "taker_buy": sub["taker_buy"].sum(),
            })
        df = pd.DataFrame(res)

    df["ofi"] = (2.0 * df["taker_buy"] - df["V"]) / np.maximum(df["V"], 1e-12)
    p = df["price"].to_numpy()
    df["r"] = np.append(0.0, np.diff(np.log(p)))
    return df


def run_expanded_multitimeframe_experiment():
    print("=" * 110)
    print("APEDP-X EXPANDED 46-SIGNAL MULTI-TIMEFRAME GENETIC ALGORITHM DISCOVERY AUDIT")
    print(f"Target Assets: {SHORT_SYMS}")
    print(f"Intervals: {INTERVALS}")
    print(f"Feature Space Dimensionality: K = {len(EXPANDED_SIGNAL_NAMES)} features")
    print(f"Bars per interval: {BARS_LIMIT} bars")
    print("=" * 110)

    results = {}

    for inv in INTERVALS:
        print(f"\n--- Fetching & Evolving on Interval: {inv} ---")
        dfs = {}
        for sym, s_short in zip(SYMBOLS, SHORT_SYMS):
            df = fetch_klines(sym, inv, limit=BARS_LIMIT)
            dfs[s_short] = df

        N = len(SHORT_SYMS)
        T = min(len(dfs[s]) for s in SHORT_SYMS)
        for s in SHORT_SYMS:
            dfs[s] = dfs[s].iloc[-T:].reset_index(drop=True)

        rets_2d = np.column_stack([dfs[s]["r"].to_numpy(float) for s in SHORT_SYMS])
        vols_2d = pd.DataFrame(rets_2d).rolling(20, min_periods=5).std().bfill().to_numpy()
        cross_med = np.median(rets_2d, axis=1)

        K_tot = len(EXPANDED_SIGNAL_NAMES)
        feats_3d = np.zeros((T, N, K_tot))

        for i, s in enumerate(SHORT_SYMS):
            feats_3d[:, i, :] = calc_expanded_signals_for_asset(dfs[s], cross_med)

        feats_3d = np.nan_to_num(feats_3d, nan=0.0)

        t0 = time.time()
        chrom = run_multi_asset_genetic_algorithm(feats_3d, rets_2d, vols_2d, pop_size=60, generations=25, seed=42)
        elapsed = time.time() - t0

        w = chrom.mask.astype(float) * chrom.weights
        active_idx = list(np.where(w != 0.0)[0])
        eq_terms = [f"{w[k]:+.4f}*{short_names[k]}" for k in active_idx]
        eq_str = " ".join(eq_terms)

        # Explicit evaluation of winning chromosome
        scores_3d = feats_3d @ w
        scores_norm = (scores_3d - np.mean(scores_3d, axis=1, keepdims=True)) / np.maximum(
            np.std(scores_3d, axis=1, keepdims=True), 1e-8
        )
        target_pos = np.where(scores_norm[:-1] > chrom.threshold, 1.0, np.where(scores_norm[:-1] < -chrom.threshold, -1.0, 0.0))
        turnover = float(np.mean(np.abs(np.diff(target_pos, axis=0, prepend=target_pos[:1]))))
        inv_vol = 1.0 / np.maximum(vols_2d[:-1], 0.005)
        raw_w = target_pos * inv_vol
        norm_factor = np.maximum(np.sum(np.abs(raw_w), axis=1, keepdims=True), 1e-8)
        w_mat = raw_w / norm_factor
        daily_ret = np.sum(w_mat * rets_2d[1:], axis=1) - turnover * 0.0005
        tot_ret = float(np.sum(daily_ret) * 100.0)
        trades_count = int(np.sum(np.abs(np.diff(target_pos, axis=0)) > 0))

        results[inv] = {
            "interval": inv,
            "bars": T,
            "total_grammar_features": K_tot,
            "term_count": len(active_idx),
            "equation": f"S_{inv} = {eq_str}",
            "threshold": round(chrom.threshold, 4),
            "fitness_sr": round(chrom.fitness, 3),
            "in_sample_sharpe": round(chrom.in_sample_sharpe, 3),
            "trades_count": trades_count,
            "turnover": round(turnover, 4),
            "tot_return_pct": round(tot_ret, 2),
            "active_terms": [short_names[k] for k in active_idx],
            "weights_dict": {short_names[k]: round(w[k], 4) for k in active_idx},
            "elapsed_s": round(elapsed, 2),
        }

        print(f"[{inv:<4}] ({len(active_idx)} terms, elapsed {elapsed:.1f}s, Fit SR={chrom.fitness:.2f}, In-Sample Sharpe={chrom.in_sample_sharpe:.2f}, Trades={trades_count}, Ret={tot_ret:+.2f}%, Threshold={chrom.threshold:.4f}):")
        print(f"       S_{inv} = {eq_str}")

    out_file = Path("scratch/expanded_multitimeframe_evolution_results.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nAll 6 timeframes evolved under expanded 46-feature grammar. Saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_expanded_multitimeframe_experiment()
