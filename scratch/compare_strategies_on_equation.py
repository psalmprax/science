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

# Extract test block scores and returns
scores_test = feats_15[idx_b, :, :] @ weights # shape (502, 15)
rets_test = rets_15[idx_b, :]                  # shape (502, 15)
vols_test = vols_15[idx_b, :]                  # shape (502, 15)
T_test, N = scores_test.shape

def sim(weights_2d, rets_2d, fee_rate=fee):
    T, N = weights_2d.shape
    cap = 100000.0
    eq = [cap]
    cur_w = np.zeros(N)
    r_list = []
    to_tot = 0.0
    for t in range(T - 1):
        tgt_w = weights_2d[t]
        to = np.sum(np.abs(tgt_w - cur_w))
        to_tot += to
        cost = cap * to * fee_rate
        g = float(np.sum(tgt_w * rets_2d[t + 1]))
        n = g - (cost / cap)
        cap *= (1.0 + n)
        eq.append(cap)
        r_list.append(n)
        cur_w = tgt_w
    ret = (cap - 100000.0) / 1000.0
    sr = np.mean(r_list) / np.std(r_list) * np.sqrt(365) if np.std(r_list) > 1e-9 else 0.0
    eq_arr = np.array(eq)
    pk = np.maximum.accumulate(eq_arr)
    dd = np.max((pk - eq_arr) / np.maximum(pk, 1e-12) * 100.0)
    return {"net": ret, "sr": sr, "dd": dd, "to": to_tot}

# Baseline 1: Standard APEDP Binary Threshold + Inverse Vol
pos_base = np.where(scores_test > th, 1.0, np.where(scores_test < -th, -1.0, 0.0))
w_base = np.zeros_like(pos_base)
for t in range(T_test):
    p = pos_base[t]
    act = p != 0.0
    if np.sum(act) > 0:
        inv = (1.0 / np.maximum(vols_test[t], 0.005)) * act
        w_base[t] = (p * inv) / np.sum(inv)

res_base = sim(w_base, rets_test)

# Strategy Variant 1: Long-Only Binary (No Shorts)
pos_lo = np.where(scores_test > th, 1.0, 0.0)
w_lo = np.zeros_like(pos_lo)
for t in range(T_test):
    p = pos_lo[t]
    act = p > 0.0
    if np.sum(act) > 0:
        inv = (1.0 / np.maximum(vols_test[t], 0.005)) * act
        w_lo[t] = (p * inv) / np.sum(inv)
res_lo = sim(w_lo, rets_test)

# Strategy Variant 2: Cross-Sectional Top-K Long-Only (Top 3 highest score)
w_top3 = np.zeros_like(scores_test)
for t in range(T_test):
    sc_t = scores_test[t]
    top_idx = np.argsort(sc_t)[-3:] # top 3
    # Only buy if score > 0
    top_idx = [idx for idx in top_idx if sc_t[idx] > 0]
    if len(top_idx) > 0:
        inv = 1.0 / np.maximum(vols_test[t, top_idx], 0.005)
        w_top3[t, top_idx] = inv / np.sum(inv)
res_top3 = sim(w_top3, rets_test)

# Strategy Variant 3: Cross-Sectional Top-K Long-Only (Top 5 highest score)
w_top5 = np.zeros_like(scores_test)
for t in range(T_test):
    sc_t = scores_test[t]
    top_idx = np.argsort(sc_t)[-5:] # top 5
    top_idx = [idx for idx in top_idx if sc_t[idx] > 0]
    if len(top_idx) > 0:
        inv = 1.0 / np.maximum(vols_test[t, top_idx], 0.005)
        w_top5[t, top_idx] = inv / np.sum(inv)
res_top5 = sim(w_top5, rets_test)

# Strategy Variant 4: Cross-Sectional Long-Short Dollar-Neutral (Top 3 Long, Bottom 3 Short)
w_ls = np.zeros_like(scores_test)
for t in range(T_test):
    sc_t = scores_test[t]
    top3 = np.argsort(sc_t)[-3:]
    bot3 = np.argsort(sc_t)[:3]
    inv_top = 1.0 / np.maximum(vols_test[t, top3], 0.005)
    inv_bot = 1.0 / np.maximum(vols_test[t, bot3], 0.005)
    w_ls[t, top3] = 0.5 * (inv_top / np.sum(inv_top))
    w_ls[t, bot3] = -0.5 * (inv_bot / np.sum(inv_bot))
res_ls = sim(w_ls, rets_test)

# Strategy Variant 5: Continuous Score-Proportional Sizing (Tanh / Sigmoid scaling)
w_cont = np.zeros_like(scores_test)
for t in range(T_test):
    sc_t = scores_test[t]
    # Continuous sizing with soft threshold
    p = np.tanh(sc_t / 1.0) # smooth exposure
    act = np.abs(p) > 0.1
    if np.sum(act) > 0:
        inv = (1.0 / np.maximum(vols_test[t], 0.005)) * act
        raw_w = p * inv
        w_cont[t] = raw_w / np.maximum(np.sum(np.abs(raw_w)), 1e-12)
res_cont = sim(w_cont, rets_test)

print("=" * 85)
print("STRATEGY COMPARISON USING EVOLVED EQUATION S_15 ON BLOCK B (2024-2025)")
print("=" * 85)
print(f"{'Strategy Architecture':<35} | {'Net Return':>12} | {'Sharpe':>8} | {'Max DD':>8} | {'Turnover':>10}")
print("-" * 85)
print(f"{'1. Standard Baseline (Long-Short Bin)':<35} | {res_base['net']:>11.2f}% | {res_base['sr']:>8.2f} | {res_base['dd']:>7.1f}% | {res_base['to']:>10.1f}")
print(f"{'2. Long-Only Binary (No Shorts)':<35} | {res_lo['net']:>11.2f}% | {res_lo['sr']:>8.2f} | {res_lo['dd']:>7.1f}% | {res_lo['to']:>10.1f}")
print(f"{'3. Top-3 Cross-Sectional Long-Only':<35} | {res_top3['net']:>11.2f}% | {res_top3['sr']:>8.2f} | {res_top3['dd']:>7.1f}% | {res_top3['to']:>10.1f}")
print(f"{'4. Top-5 Cross-Sectional Long-Only':<35} | {res_top5['net']:>11.2f}% | {res_top5['sr']:>8.2f} | {res_top5['dd']:>7.1f}% | {res_top5['to']:>10.1f}")
print(f"{'5. Top-3 / Bot-3 Long-Short Neutral':<35} | {res_ls['net']:>11.2f}% | {res_ls['sr']:>8.2f} | {res_ls['dd']:>7.1f}% | {res_ls['to']:>10.1f}")
print(f"{'6. Continuous Tanh Sizing':<35} | {res_cont['net']:>11.2f}% | {res_cont['sr']:>8.2f} | {res_cont['dd']:>7.1f}% | {res_cont['to']:>10.1f}")
print("=" * 85)
