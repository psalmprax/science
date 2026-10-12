import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# 1. Inspect exact indices and dates
df_90 = pd.read_csv("data/market/multi_asset_90_5yr_orderflow.csv")
date_col = "observation_date" if "observation_date" in df_90.columns else "date"
dates = df_90[date_col].to_numpy()

idx_a = np.where(dates <= "2023-12-31")[0]
idx_b = np.where(dates >= "2024-01-01")[0]
idx_subtrain = idx_a[:500]
idx_val = idx_a[500:]

print("--- BOUNDARY AND INDEX VERIFICATION ---")
print(f"Total bars in CSV: {len(dates)}")
print(f"idx_a: len={len(idx_a)}, range=[{idx_a[0]}..{idx_a[-1]}], dates=[{dates[idx_a[0]]} .. {dates[idx_a[-1]]}]")
print(f"idx_b: len={len(idx_b)}, range=[{idx_b[0]}..{idx_b[-1]}], dates=[{dates[idx_b[0]]} .. {dates[idx_b[-1]]}]")
print(f"idx_subtrain: len={len(idx_subtrain)}, range=[{idx_subtrain[0]}..{idx_subtrain[-1]}], dates=[{dates[idx_subtrain[0]]} .. {dates[idx_subtrain[-1]]}]")
print(f"idx_val: len={len(idx_val)}, range=[{idx_val[0]}..{idx_val[-1]}], dates=[{dates[idx_val[0]]} .. {dates[idx_val[-1]]}]")

# Check overlap
overlap_sub_val = set(idx_subtrain).intersection(set(idx_val))
overlap_val_test = set(idx_val).intersection(set(idx_b))
print(f"Intersection(subtrain, val): {overlap_sub_val} (size {len(overlap_sub_val)})")
print(f"Intersection(val, test): {overlap_val_test} (size {len(overlap_val_test)})")

# Check Python slice representation
print(f"Python slice notation (exclusive upper bound):")
print(f"Subtrain: [0:500] -> {len(dates[0:500])} bars, end date: {dates[499]}")
print(f"Val:      [500:753] -> {len(dates[500:753])} bars, start date: {dates[500]}, end date: {dates[752]}")
print(f"Test:     [753:1255] -> {len(dates[753:1255])} bars, start date: {dates[753]}, end date: {dates[1254]}")

# 2. Statistical Analysis on the 20 paired runs (4 universes x 5 seeds)
results = json.loads(Path("provenance/controlled_ab_benchmark_20261009/results.json").read_text())

pairs_test_net = []
pairs_test_sr = []
pairs_test_dd = []
pairs_terms = []

for u, data in results["universes_results"].items():
    seeds_a = data["seeds_detail_arm_a"]
    seeds_b = data["seeds_detail_arm_b"]
    for sa, sb in zip(seeds_a, seeds_b):
        pairs_test_net.append((sa["test_perf"]["net_return"], sb["test_perf"]["net_return"]))
        pairs_test_sr.append((sa["test_perf"]["sharpe"], sb["test_perf"]["sharpe"]))
        pairs_test_dd.append((sa["test_perf"]["max_drawdown"], sb["test_perf"]["max_drawdown"]))
        pairs_terms.append((sa["term_count"], sb["term_count"]))

a_net = np.array([p[0] for p in pairs_test_net])
b_net = np.array([p[1] for p in pairs_test_net])
diff_net = b_net - a_net

a_sr = np.array([p[0] for p in pairs_test_sr])
b_sr = np.array([p[1] for p in pairs_test_sr])
diff_sr = b_sr - a_sr

a_dd = np.array([p[0] for p in pairs_test_dd])
b_dd = np.array([p[1] for p in pairs_test_dd])
diff_dd = b_dd - a_dd

a_terms = np.array([p[0] for p in pairs_terms])
b_terms = np.array([p[1] for p in pairs_terms])
diff_terms = b_terms - a_terms

print("\n--- STATISTICAL UNCERTAINTY ANALYSIS (N=20 runs across 4 universes) ---")
print(f"Mean Paired Diff in Terms (Arm B - Arm A): {np.mean(diff_terms):.2f} +/- {np.std(diff_terms)/np.sqrt(20):.2f}")
t_term, p_term = stats.ttest_rel(b_terms, a_terms)
print(f"  Paired t-test: t={t_term:.3f}, p={p_term:.4e}")

print(f"\nMean Paired Diff in Test Net Return (Arm B - Arm A): {np.mean(diff_net):.2f} pp")
print(f"  Std Err: {np.std(diff_net)/np.sqrt(20):.2f} pp, 95% CI: [{np.mean(diff_net) - 2.093*np.std(diff_net)/np.sqrt(20):.2f}, {np.mean(diff_net) + 2.093*np.std(diff_net)/np.sqrt(20):.2f}]")
t_net, p_net = stats.ttest_rel(b_net, a_net)
w_net = stats.wilcoxon(b_net, a_net)
print(f"  Paired t-test: t={t_net:.3f}, p={p_net:.4f}")
print(f"  Wilcoxon signed-rank: stat={w_net.statistic:.1f}, p={w_net.pvalue:.4f}")

print(f"\nMean Paired Diff in Test Sharpe (Arm B - Arm A): {np.mean(diff_sr):.2f}")
print(f"  Std Err: {np.std(diff_sr)/np.sqrt(20):.2f}, 95% CI: [{np.mean(diff_sr) - 2.093*np.std(diff_sr)/np.sqrt(20):.2f}, {np.mean(diff_sr) + 2.093*np.std(diff_sr)/np.sqrt(20):.2f}]")
t_sr, p_sr = stats.ttest_rel(b_sr, a_sr)
w_sr = stats.wilcoxon(b_sr, a_sr)
print(f"  Paired t-test: t={t_sr:.3f}, p={p_sr:.4f}")
print(f"  Wilcoxon signed-rank: stat={w_sr.statistic:.1f}, p={w_sr.pvalue:.4f}")

# Cross-seed position correlation uncertainty
# In each universe, 5 seeds define 10 pairs of positions.
print("\n--- POSITION CORRELATION ACROSS SEEDS (Outcome 1) ---")
for u, data in results["universes_results"].items():
    print(f"Universe {u}: Arm A r_bar = {data['outcome_1_stability']['decision_correlation_arm_a']:+.3f} vs Arm B r_bar = {data['outcome_1_stability']['decision_correlation_arm_b']:+.3f}")
