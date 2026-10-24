import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe,
    calc_signals_for_asset,
    MULTI_90_FILE,
)
from apedp_dyn.sandbox.grammar_space import op_product_interaction

df_90 = pd.read_csv(MULTI_90_FILE)
date_col = "observation_date" if "observation_date" in df_90.columns else "date"
dates = df_90[date_col].to_numpy()
idx_b = np.where(dates >= "2024-01-01")[0]

uni_90, syms_90 = load_universe("90")
syms_15 = syms_90[:15]
rets_15 = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_15])
vols_15 = pd.DataFrame(rets_15).rolling(20, min_periods=5).std().bfill().to_numpy()
cross_med = np.median(rets_15, axis=1)

feats_15 = np.zeros((len(df_90), 15, 27))
for i, s in enumerate(syms_15):
    f = calc_signals_for_asset(uni_90[s], cross_med)
    inter = op_product_interaction(f[:, 9], f[:, 24])
    K_b = f.shape[1]
    feats_15[:, i, :K_b] = f
    feats_15[:, i, K_b] = inter

weights = np.zeros(27)
weights[9] = 1.1948       # H-BB
weights[13] = 0.4501      # H-OBV_MOM
weights[10] = 0.2227      # H-STOCH
weights[19] = -0.4839     # NL-HURST_ADAPT
weights[24] = -0.0161     # PRODUCT_H_BB_TAKER
th = 0.1695
fee = 5.0 / 10000.0

scores_test = feats_15[idx_b, :, :] @ weights
rets_test = rets_15[idx_b, :]
vols_test = vols_15[idx_b, :]
T_test, N = scores_test.shape

# 1. Per-Asset Calculation
asset_rows = []
for i, s in enumerate(syms_15):
    p = uni_90[s]["price"].to_numpy()
    p0 = p[idx_b[0]]
    p1 = p[idx_b[-1]]
    bh_ret = (p1 / p0 - 1.0) * 100.0
    
    sc = scores_test[:, i]
    pos = np.where(sc > th, 1.0, np.where(sc < -th, -1.0, 0.0))
    r = rets_test[:, i]
    
    cap_net = 10000.0
    cap_gross = 10000.0
    cur_p = 0.0
    eq_net = [cap_net]
    r_net_list = []
    to_tot = 0.0
    trade_days = 0
    
    for t in range(T_test - 1):
        p_tgt = pos[t]
        to = abs(p_tgt - cur_p)
        to_tot += to
        cost = cap_net * to * fee
        g = p_tgt * r[t + 1]
        n = g - (cost / cap_net)
        cap_gross *= (1.0 + g)
        cap_net *= (1.0 + n)
        eq_net.append(cap_net)
        r_net_list.append(n)
        cur_p = p_tgt
        if p_tgt != 0.0:
            trade_days += 1
            
    strat_net = (cap_net - 10000.0) / 100.0
    strat_gross = (cap_gross - 10000.0) / 100.0
    sr = np.mean(r_net_list) / np.std(r_net_list) * np.sqrt(365) if np.std(r_net_list) > 1e-9 else 0.0
    eq_arr = np.array(eq_net)
    pk = np.maximum.accumulate(eq_arr)
    dd = np.max((pk - eq_arr) / np.maximum(pk, 1e-12) * 100.0)
    win_rate = np.mean(np.array(r_net_list)[pos[:-1] != 0] > 0) * 100.0 if trade_days > 0 else 0.0
    spread = strat_net - bh_ret
    
    asset_rows.append({
        "Asset": s,
        "StartPrice": p0,
        "EndPrice": p1,
        "BH_Return": bh_ret,
        "StratGross": strat_gross,
        "StratNet": strat_net,
        "Sharpe": sr,
        "MaxDD": dd,
        "Turnover": to_tot,
        "WinRate": win_rate,
        "TradeDays": trade_days,
        "AlphaSpread": spread,
    })

df_assets = pd.DataFrame(asset_rows)

print("=== TABLE 1: INDIVIDUAL ASSET AUDIT (BLOCK B: 2024-01-02 TO 2025-12-31) ===")
for r in asset_rows:
    print(f"{r['Asset']:<5} | ${r['StartPrice']:>10.2f} | ${r['EndPrice']:>10.2f} | B&H={r['BH_Return']:>+8.2f}% | Gross={r['StratGross']:>+8.2f}% | Net={r['StratNet']:>+8.2f}% | SR={r['Sharpe']:>5.2f} | DD={r['MaxDD']:>5.1f}% | TO={r['Turnover']:>5.1f} | Win={r['WinRate']:>5.1f}% | Alpha={r['AlphaSpread']:>+8.2f}pp")

print("\n--- SUMMARY ROW (MACHINE COMPUTED) ---")
print(f"B&H Mean:        {df_assets['BH_Return'].mean():>+8.2f}%")
print(f"B&H Median:      {df_assets['BH_Return'].median():>+8.2f}%")
print(f"Strat Net Mean:  {df_assets['StratNet'].mean():>+8.2f}%")
print(f"Strat Net Median:{df_assets['StratNet'].median():>+8.2f}%")
print(f"Sharpe Mean:     {df_assets['Sharpe'].mean():>8.2f}")
print(f"Max DD Mean:     {df_assets['MaxDD'].mean():>7.1f}%")
print(f"Alpha Spread Mean: {df_assets['AlphaSpread'].mean():>+8.2f}pp")
print(f"Alpha Spread Med:  {df_assets['AlphaSpread'].median():>+8.2f}pp")

# 2. Portfolio Simulation
pos_port = np.where(scores_test > th, 1.0, np.where(scores_test < -th, -1.0, 0.0))
cap_port = 100000.0
cur_p = np.zeros(N)
eq_port = [cap_port]
r_port = []
to_port = 0.0
for t in range(T_test - 1):
    p_tgt = pos_port[t]
    sig = np.maximum(vols_test[t], 0.005)
    act = p_tgt != 0.0
    if np.sum(act) > 0:
        inv = (1.0 / sig) * act
        w = (p_tgt * inv) / np.sum(inv)
    else:
        w = np.zeros(N)
    to = float(np.sum(np.abs(p_tgt - cur_p)))
    to_port += to
    cost = cap_port * to * (fee / N)
    g = float(np.sum(w * rets_test[t + 1]))
    n = g - (cost / cap_port)
    cap_port *= (1.0 + n)
    eq_port.append(cap_port)
    r_port.append(n)
    cur_p = p_tgt

port_net = (cap_port - 100000.0) / 1000.0
port_sr = np.mean(r_port) / np.std(r_port) * np.sqrt(365)
eq_p_arr = np.array(eq_port)
pk_p = np.maximum.accumulate(eq_p_arr)
port_dd = np.max((pk_p - eq_p_arr) / np.maximum(pk_p, 1e-12) * 100.0)

print(f"\nPortfolio Net: {port_net:.2f}%, Sharpe: {port_sr:.2f}, Max DD: {port_dd:.1f}%, Turnover: {to_port:.1f}")
