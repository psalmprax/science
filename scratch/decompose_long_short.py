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

pos_base = np.where(scores_test > th, 1.0, np.where(scores_test < -th, -1.0, 0.0))

# Separate longs and shorts
long_gains = []
short_gains = []

for t in range(T_test - 1):
    p = pos_base[t]
    sig = np.maximum(vols_test[t], 0.005)
    
    # Longs
    act_l = p > 0
    if np.sum(act_l) > 0:
        inv_l = (1.0 / sig) * act_l
        w_l = inv_l / np.sum(inv_l)
        g_l = float(np.sum(w_l * rets_test[t + 1]))
        long_gains.append(g_l)
    else:
        long_gains.append(0.0)
        
    # Shorts
    act_s = p < 0
    if np.sum(act_s) > 0:
        inv_s = (1.0 / sig) * act_s
        w_s = inv_s / np.sum(inv_s)
        g_s = float(np.sum(-w_s * rets_test[t + 1]))
        short_gains.append(g_s)
    else:
        short_gains.append(0.0)

print(f"Total Cumulative Return Long Side Only:  {(np.prod(1.0 + np.array(long_gains)) - 1.0) * 100:.2f}%")
print(f"Total Cumulative Return Short Side Only: {(np.prod(1.0 + np.array(short_gains)) - 1.0) * 100:.2f}%")
print(f"Combined Gross Portfolio Return:         {(np.prod(1.0 + 0.5*np.array(long_gains) + 0.5*np.array(short_gains)) - 1.0) * 100:.2f}%")

# Now check buy-and-hold returns of each coin in Block B (2024-2025)
cum_asset_ret = (np.prod(1.0 + rets_test, axis=0) - 1.0) * 100
for i, s in enumerate(syms_15):
    print(f"Asset {s:<6} Buy-and-Hold 2024-2025: {cum_asset_ret[i]:>+8.2f}%")
