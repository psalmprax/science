import json
import numpy as np
from scipy import stats
from pathlib import Path

# Load positions from audit or recompute pairwise correlations
# Let's inspect pairwise correlations directly
import pandas as pd
from apedp_dyn.sandbox.multi_asset_signals_sandbox import load_universe, calc_signals_for_asset, SIGNAL_NAMES
from apedp_dyn.sandbox.grammar_space import op_product_interaction

df_90 = pd.read_csv("data/market/multi_asset_90_5yr_orderflow.csv")
date_col = "observation_date" if "observation_date" in df_90.columns else "date"
dates = df_90[date_col].to_numpy()
idx_b = np.where(dates >= "2024-01-01")[0]

results = json.loads(Path("provenance/controlled_ab_benchmark_20261009/results.json").read_text())

# In results.json, we have seeds_detail_arm_a and seeds_detail_arm_b
# Let's compute the exact 10 pairwise correlations for each universe
for u, data in results["universes_results"].items():
    N = int(u)
    # reconstruct positions
    pass
print("Script template ready")
