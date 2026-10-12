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

# Prices for 50-day SMA
prices_15 = np.column_stack([uni_90[s]["price"].to_numpy(float) for s in syms_15])
sma50 = pd.DataFrame(prices_15).rolling(50, min_periods=20).mean().to_numpy()
above_sma = (prices_15 > sma50)

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
above_sma_test = above_sma[idx_b, :]
T_test, N = scores_test.shape

# Regime-Filtered Positions:
# If above SMA50: only Longs allowed (no shorting into uptrend)
# If below SMA50: only Shorts allowed (no buying into downtrend)
pos_gated = np.zeros_like(scores_test)
for t in range(T_test):
    for i in range(N):
        sc = scores_test[t, i]
        bull = above_sma_test[t, i]
        if bull:
            if sc > th:
                pos_gated[t, i] = 1.0
        else:
            if sc < -th:
                pos_gated[t, i] = -1.0

# Simulate per asset with regime gate
print(f"{'Asset':<6} | {'Unfiltered Net':>14} | {'SMA50-Gated Net':>16} | {'Gated Sharpe':>12} | {'Gated Max DD':>12}")
print("-" * 75)

gated_nets = []
unfiltered_nets = []

for i, s in enumerate(syms_15):
    r = rets_test[:, i]
    
    # 1. Unfiltered
    sc = scores_test[:, i]
    p_unf = np.where(sc > th, 1.0, np.where(sc < -th, -1.0, 0.0))
    cap_u = 10000.0
    cur_p = 0.0
    for t in range(T_test - 1):
        to = abs(p_unf[t] - cur_p)
        cost = cap_u * to * fee
        g = p_unf[t] * r[t + 1]
        n = g - (cost / cap_u)
        cap_u *= (1.0 + n)
        cur_p = p_unf[t]
    net_u = (cap_u - 10000.0) / 100.0
    unfiltered_nets.append(net_u)
    
    # 2. Gated
    p_gat = pos_gated[:, i]
    cap_g = 10000.0
    cur_p = 0.0
    eq_g = [cap_g]
    r_g_list = []
    for t in range(T_test - 1):
        to = abs(p_gat[t] - cur_p)
        cost = cap_g * to * fee
        g = p_gat[t] * r[t + 1]
        n = g - (cost / cap_g)
        cap_g *= (1.0 + n)
        eq_g.append(cap_g)
        r_g_list.append(n)
        cur_p = p_gat[t]
    net_g = (cap_g - 10000.0) / 100.0
    sr_g = np.mean(r_g_list) / np.std(r_g_list) * np.sqrt(365) if np.std(r_g_list) > 1e-9 else 0.0
    eq_arr = np.array(eq_g)
    pk = np.maximum.accumulate(eq_arr)
    dd_g = np.max((pk - eq_arr) / np.maximum(pk, 1e-12) * 100.0)
    gated_nets.append(net_g)
    
    print(f"{s:<6} | {net_u:>13.2f}% | {net_g:>15.2f}% | {sr_g:>12.2f} | {dd_g:>11.1f}%")

print("-" * 75)
print(f"Mean across assets: Unfiltered = {np.mean(unfiltered_nets):.2f}% | Gated = {np.mean(gated_nets):.2f}%")

# Now simulate Portfolio with SMA50 Gating
w_gated_port = np.zeros_like(pos_gated)
for t in range(T_test):
    p = pos_gated[t]
    act = p != 0.0
    if np.sum(act) > 0:
        inv = (1.0 / np.maximum(vols_test[t], 0.005)) * act
        w_gated_port[t] = (p * inv) / np.sum(inv)

cap_port = 100000.0
cur_w = np.zeros(N)
eq_p = [cap_port]
r_p_list = []
to_p = 0.0
for t in range(T_test - 1):
    tgt = w_gated_port[t]
    to = np.sum(np.abs(tgt - cur_w))
    to_p += to
    cost = cap_port * to * fee
    g = float(np.sum(tgt * rets_test[t + 1]))
    n = g - (cost / cap_port)
    cap_port *= (1.0 + n)
    eq_p.append(cap_port)
    r_p_list.append(n)
    cur_w = tgt

port_net = (cap_port - 100000.0) / 1000.0
port_sr = np.mean(r_p_list) / np.std(r_p_list) * np.sqrt(365)
eq_arr = np.array(eq_p)
pk = np.maximum.accumulate(eq_arr)
port_dd = np.max((pk - eq_arr) / np.maximum(pk, 1e-12) * 100.0)

print(f"\nPORTFOLIO RESULT (SMA50 REGIME GATED): Net = {port_net:.2f}%, Sharpe = {port_sr:.2f}, Max DD = {port_dd:.1f}%, Turnover = {to_p:.1f}")
