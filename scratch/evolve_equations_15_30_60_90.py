"""Evolve equations on expanded 27-signal grammar across all four universes: 15, 30, 60, 90 assets.

Training period: Block A (2021-2023, 753 bars).
Features: 26 base signals + 1 interaction signal (PRODUCT_H_BB_TAKER).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from apedp_dyn.sandbox.grammar_space import op_product_interaction
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe,
    calc_signals_for_asset,
    SIGNAL_NAMES,
    MULTI_90_FILE,
    run_multi_asset_genetic_algorithm,
)


def evolve_for_all_universes():
    df_90 = pd.read_csv(MULTI_90_FILE)
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    idx_train = np.where(dates <= "2023-12-31")[0]

    uni_90, syms_90 = load_universe("90")
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_full = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().bfill().to_numpy()
    cross_med = np.median(rets_full, axis=1)

    K_base = len(SIGNAL_NAMES)
    expanded_names = SIGNAL_NAMES + ["PRODUCT_H_BB_TAKER (BB x Taker Surge)"]

    print("=" * 115)
    print("APEDP-X AUTONOMOUS EQUATION EVOLUTION ACROSS ALL FOUR UNIVERSES (15, 30, 60, 90)")
    print("Expanded Feature Grammar Space: 26 Base Signals + PRODUCT_H_BB_TAKER (27 Terms)")
    print(f"Training Split: Block A (2021-2023, {len(idx_train)} bars)")
    print("=" * 115)

    results = {}

    for N in [15, 30, 60, 90]:
        print(f"\n>>> EVOLVING FOR UNIVERSE N = {N} ASSETS <<<")
        # Build features for N assets
        feats_n = np.zeros((len(df_90), N, K_base + 1))
        for i in range(N):
            f_asset = calc_signals_for_asset(uni_90[syms_90[i]], cross_med)
            inter = op_product_interaction(f_asset[:, 9], f_asset[:, 24])
            feats_n[:, i, :K_base] = f_asset
            feats_n[:, i, K_base] = inter

        f_tr = feats_n[idx_train]
        r_tr = rets_full[idx_train, :N]
        v_tr = vols_full[idx_train, :N]

        best_chrom = run_multi_asset_genetic_algorithm(f_tr, r_tr, v_tr, pop_size=60, generations=25, seed=42)
        active_idx = np.where(best_chrom.mask)[0]

        terms = [f"{best_chrom.weights[idx]:+.4f} * X^*_{expanded_names[idx].split()[0]}" for idx in active_idx]
        eq_str = "S_{i,t} = " + " ".join(terms)

        active_terms_dict = {expanded_names[idx]: float(best_chrom.weights[idx]) for idx in active_idx}
        has_inter = (K_base in active_idx)

        print(f"Universe N = {N}:")
        print(f"  Evolved Equation: {eq_str}")
        print(f"  Threshold: theta = {best_chrom.threshold:.4f}")
        print(f"  In-Sample Sharpe: {best_chrom.in_sample_sharpe:.3f} | Fitness: {best_chrom.fitness:.3f}")
        print(f"  Interaction Term Selected: {has_inter}")
        for k, w in active_terms_dict.items():
            print(f"    • {k:<40} -> Weight: {w:+.4f}")

        results[str(N)] = {
            "N": N,
            "equation": eq_str,
            "threshold": float(best_chrom.threshold),
            "in_sample_sharpe": float(best_chrom.in_sample_sharpe),
            "fitness": float(best_chrom.fitness),
            "interaction_selected": bool(has_inter),
            "active_terms": active_terms_dict,
        }

    print("\n" + "=" * 115)
    print("EVOLUTION COMPLETE ACROSS ALL UNIVERSES")
    print("=" * 115)
    return results


if __name__ == "__main__":
    evolve_for_all_universes()
