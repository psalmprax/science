import sys
import time
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    calc_signals_for_asset,
    run_multi_asset_genetic_algorithm,
    SIGNAL_NAMES,
)
from apedp_dyn.sandbox.grammar_space import op_product_interaction

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
SHORT_SYMS = ["BTC", "ETH", "SOL", "BNB", "DOGE"]
INTERVALS = ["1m", "5m", "15m", "30m", "45m", "60m"]
BARS_LIMIT = 1000

expanded_names = SIGNAL_NAMES + ["PRODUCT_H_BB_TAKER (BB x Taker Surge)"]
short_names = [n.split()[0] for n in expanded_names]

def fetch_klines(sym: str, interval: str, limit: int = BARS_LIMIT) -> pd.DataFrame:
    # 45m is resampled from 15m (3x 15m bars)
    fetch_inv = "15m" if interval == "45m" else ("1h" if interval == "60m" else interval)
    fetch_lim = limit * 3 if interval == "45m" else limit
    url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={fetch_inv}&limit={fetch_lim}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "vol",
        "close_time", "q_vol", "trades", "tb_vol", "tb_q_vol", "ignore"
    ])
    df["price"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["V"] = df["vol"].astype(float)
    df["taker_buy"] = df["tb_vol"].astype(float)
    
    if interval == "45m":
        # Resample every 3 bars
        n_chunks = len(df) // 3
        df = df.iloc[:n_chunks * 3]
        res = []
        for i in range(n_chunks):
            sub = df.iloc[i * 3:(i + 1) * 3]
            res.append({
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

def run_experiment():
    print("=" * 105)
    print("APEDP-X MULTI-TIMEFRAME GENETIC ALGORITHM EVOLUTION AUDIT")
    print(f"Target Assets: {SHORT_SYMS}")
    print(f"Intervals: {INTERVALS}")
    print(f"Bars per interval: {BARS_LIMIT} bars")
    print("=" * 105)
    
    results = {}
    
    for inv in INTERVALS:
        print(f"\n--- Fetching and Evolving on Interval: {inv} ---")
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
        
        K_base = len(SIGNAL_NAMES)
        K_tot = len(expanded_names)
        feats_3d = np.zeros((T, N, K_tot))
        
        for i, s in enumerate(SHORT_SYMS):
            f = calc_signals_for_asset(dfs[s], cross_med)
            inter = op_product_interaction(f[:, 9], f[:, 24]) # BB x Taker
            feats_3d[:, i, :K_base] = f
            feats_3d[:, i, K_base] = inter
            
        # Clean NaNs
        feats_3d = np.nan_to_num(feats_3d, nan=0.0)
        
        # Run Genetic Algorithm: seed=42, pop=60, gen=25
        t0 = time.time()
        chrom = run_multi_asset_genetic_algorithm(feats_3d, rets_2d, vols_2d, pop_size=60, generations=25, seed=42)
        elapsed = time.time() - t0
        
        w = chrom.mask.astype(float) * chrom.weights
        active_idx = list(np.where(w != 0.0)[0])
        eq_terms = [f"{w[k]:+.4f}*{short_names[k]}" for k in active_idx]
        eq_str = " ".join(eq_terms)
        
        results[inv] = {
            "interval": inv,
            "bars": T,
            "term_count": len(active_idx),
            "equation": f"S_{inv} = {eq_str}",
            "threshold": round(chrom.threshold, 4),
            "fitness_sr": round(chrom.fitness, 3),
            "active_terms": [short_names[k] for k in active_idx],
            "weights_dict": {short_names[k]: round(w[k], 4) for k in active_idx},
            "elapsed_s": round(elapsed, 2),
        }
        
        print(f"[{inv:<4}] ({len(active_idx)} terms, elapsed {elapsed:.1f}s, Fit SR={chrom.fitness:.2f}, Threshold={chrom.threshold:.4f}):")
        print(f"       S = {eq_str}")
        
    out_file = Path("scratch/multitimeframe_evolution_results.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nAll 6 timeframes evolved. Results saved to: {out_file}")
    return results

if __name__ == "__main__":
    run_experiment()
