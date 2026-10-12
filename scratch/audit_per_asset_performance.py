import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe,
    calc_signals_for_asset,
    SIGNAL_NAMES,
    MULTI_90_FILE,
)
from apedp_dyn.sandbox.grammar_space import op_product_interaction

# Load data and setup signals
df_90 = pd.read_csv(MULTI_90_FILE)
date_col = "observation_date" if "observation_date" in df_90.columns else "date"
dates = df_90[date_col].to_numpy()
idx_b = np.where(dates >= "2024-01-01")[0]

uni_90, syms_90 = load_universe("90")
syms_15 = syms_90[:15]
rets_15 = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_15])
cross_med = np.median(rets_15, axis=1)

feats_15 = np.zeros((len(df_90), 15, 27))
for i, s in enumerate(syms_15):
    f = calc_signals_for_asset(uni_90[s], cross_med)
    inter = op_product_interaction(f[:, 9], f[:, 24]) # BB x Taker
    K_b = f.shape[1]
    feats_15[:, i, :K_b] = f
    feats_15[:, i, K_b] = inter

# Weights for S_15
weights = np.zeros(27)
weights[9] = 1.1948       # H-BB
weights[13] = 0.4501      # H-OBV_MOM
weights[10] = 0.2227      # H-STOCH
weights[19] = -0.4839     # NL-HURST_ADAPT
weights[24] = -0.0161     # PRODUCT_H_BB_TAKER
th = 0.1695

fee = 5.0 / 10000.0

print(f"{'Asset':<6} | {'Gross Ret':>10} | {'Net Ret':>10} | {'Sharpe':>8} | {'Max DD':>8} | {'Turnover':>10} | {'Win Rate':>8} | {'Trade Days':>10}")
print("-" * 85)

per_asset_results = []
for i, s in enumerate(syms_15):
    sc = feats_15[idx_b, i, :] @ weights
    pos = np.where(sc > th, 1.0, np.where(sc < -th, -1.0, 0.0))
    r = rets_15[idx_b, i]
    
    # Simulate single asset trading
    cap_net = 10000.0
    cap_gross = 10000.0
    cur_p = 0.0
    eq = [cap_net]
    r_net_list = []
    to_tot = 0.0
    trade_days = 0
    
    for t in range(len(pos) - 1):
        p_tgt = pos[t]
        to = abs(p_tgt - cur_p)
        to_tot += to
        cost = cap_net * to * fee
        g = p_tgt * r[t + 1]
        n = g - (cost / cap_net)
        cap_gross *= (1.0 + g)
        cap_net *= (1.0 + n)
        eq.append(cap_net)
        r_net_list.append(n)
        cur_p = p_tgt
        if p_tgt != 0.0:
            trade_days += 1
            
    net_ret = (cap_net - 10000.0) / 100.0
    gross_ret = (cap_gross - 10000.0) / 100.0
    sr = np.mean(r_net_list) / np.std(r_net_list) * np.sqrt(365) if np.std(r_net_list) > 1e-9 else 0.0
    eq_arr = np.array(eq)
    pk = np.maximum.accumulate(eq_arr)
    dd = np.max((pk - eq_arr) / np.maximum(pk, 1e-12) * 100.0)
    win_rate = np.mean(np.array(r_net_list)[pos[:-1] != 0] > 0) * 100 if trade_days > 0 else 0.0
    
    per_asset_results.append({
        "asset": s, "net": net_ret, "gross": gross_ret, "sr": sr, "dd": dd, "to": to_tot, "win_rate": win_rate, "days": trade_days
    })
    print(f"{s:<6} | {gross_ret:>9.2f}% | {net_ret:>9.2f}% | {sr:>8.2f} | {dd:>7.1f}% | {to_tot:>10.1f} | {win_rate:>7.1f}% | {trade_days:>10d}")

df_res = pd.DataFrame(per_asset_results)
print("-" * 85)
print(f"Mean across assets: Net={df_res['net'].mean():.2f}%, Gross={df_res['gross'].mean():.2f}%, Sharpe={df_res['sr'].mean():.2f}, Max DD={df_res['dd'].mean():.1f}%")
print(f"Median across assets: Net={df_res['net'].median():.2f}%, Gross={df_res['gross'].median():.2f}%, Sharpe={df_res['sr'].median():.2f}")
