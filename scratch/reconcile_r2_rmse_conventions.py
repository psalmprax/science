"""Dual-Convention Reconciliation for R^2 and RMSE of APEDP Evolved Equations.

Reconciles:
  - Convention 1 (As-Is Raw Score): Score S_{i,t} is dimensionless O(1) and compared directly to return r_{i,t+1}.
  - Convention 2 (Train-Calibrated OLS Scale Factor): Score S_{i,t} is mapped to return units via beta_scale fitted on Block A Train.

Audits all four universes (15, 30, 60, 90) on Block A (Train) and Block B (Holdout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apedp_dyn.sandbox.grammar_space import op_product_interaction
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    MULTI_90_FILE,
    calc_signals_for_asset,
    load_universe,
)


def run_reconciliation():
    df_90 = pd.read_csv(MULTI_90_FILE)
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    idx_tr = np.where(dates <= "2023-12-31")[0]
    idx_ho = np.where(dates >= "2024-01-01")[0]

    uni_90, syms_90 = load_universe("90")
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    cross_med = np.median(rets_full, axis=1)

    feats_full = np.zeros((len(df_90), 90, 27))
    for i in range(90):
        f_asset = calc_signals_for_asset(uni_90[syms_90[i]], cross_med)
        inter = op_product_interaction(f_asset[:, 9], f_asset[:, 24])
        feats_full[:, i, :26] = f_asset
        feats_full[:, i, 26] = inter

    configs = {
        15: {9: 1.1948, 11: 0.2227, 16: 0.4501, 23: -0.4839, 26: -0.0161},
        30: {3: -0.1792, 9: 0.8179, 22: -0.2768},
        60: {3: -0.0545, 7: -0.0042, 9: 0.4323, 16: -0.8522, 17: 0.5357, 20: -1.3680, 23: -0.9019},
        90: {4: 0.7185, 5: -0.0925, 9: 0.7004, 11: -0.0685, 17: 0.7700},
    }

    print("=" * 115)
    print("APEDP-X R^2 AND RMSE METHODOLOGY RECONCILIATION: AS-IS RAW SCORE VS. TRAIN-SCALED RETURN CONVENTION")
    print(f"Data: Block A Train (753 bars, 2021-2023) | Block B Holdout (502 bars, 2024-2025)")
    print("=" * 115)

    reconciliation_results = {}

    for N, w_dict in configs.items():
        w = np.zeros(27)
        for k, val in w_dict.items():
            w[k] = val

        # Dimensionless composite score S_{i,t}
        S_tr = feats_full[idx_tr[:-1], :N, :] @ w
        Y_tr = rets_full[idx_tr[1:], :N]
        S_ho = feats_full[idx_ho[:-1], :N, :] @ w
        Y_ho = rets_full[idx_ho[1:], :N]

        s_tr = S_tr.flatten()
        y_tr = Y_tr.flatten()
        s_ho = S_ho.flatten()
        y_ho = Y_ho.flatten()

        ss_tot_tr = float(np.sum((y_tr - np.mean(y_tr)) ** 2))
        ss_tot_ho = float(np.sum((y_ho - np.mean(y_ho)) ** 2))
        null_rmse_tr = float(np.sqrt(np.mean((y_tr - np.mean(y_tr)) ** 2)))
        null_rmse_ho = float(np.sqrt(np.mean((y_ho - np.mean(y_ho)) ** 2)))

        # ---------------------------------------------------------------------
        # Convention 1: As-Is Raw Dimensionless Score
        # Direct prediction: \hat{r}_{i,t+1} = S_{i,t}
        # ---------------------------------------------------------------------
        res_tr_asis = y_tr - s_tr
        rmse_tr_asis = float(np.sqrt(np.mean(res_tr_asis ** 2)))
        r2_tr_asis = float(1.0 - np.sum(res_tr_asis ** 2) / ss_tot_tr)

        res_ho_asis = y_ho - s_ho
        rmse_ho_asis = float(np.sqrt(np.mean(res_ho_asis ** 2)))
        r2_ho_asis = float(1.0 - np.sum(res_ho_asis ** 2) / ss_tot_ho)

        # ---------------------------------------------------------------------
        # Convention 2: Train-Calibrated OLS Scaling Slope
        # \hat{\beta}_{scale} = \sum(S_{tr} * Y_{tr}) / \sum(S_{tr}^2)
        # Prediction: \hat{r}_{i,t+1} = \hat{\beta}_{scale} * S_{i,t}
        # ---------------------------------------------------------------------
        beta_scale = float(np.sum(s_tr * y_tr) / np.sum(s_tr ** 2))

        pred_tr_sc = s_tr * beta_scale
        res_tr_sc = y_tr - pred_tr_sc
        rmse_tr_sc = float(np.sqrt(np.mean(res_tr_sc ** 2)))
        r2_tr_sc = float(1.0 - np.sum(res_tr_sc ** 2) / ss_tot_tr)

        pred_ho_sc = s_ho * beta_scale
        res_ho_sc = y_ho - pred_ho_sc
        rmse_ho_sc = float(np.sqrt(np.mean(res_ho_sc ** 2)))
        r2_ho_sc = float(1.0 - np.sum(res_ho_sc ** 2) / ss_tot_ho)

        print(f"\nUniverse N = {N} Assets:")
        print(f"  Score Std Dev (O(1)):    Train = {np.std(s_tr):.4f} | Holdout = {np.std(s_ho):.4f}")
        print(f"  Return Std Dev (sigma):  Train = {null_rmse_tr:.4f} | Holdout = {null_rmse_ho:.4f}")
        print(f"  Fitted Train OLS Slope:  beta_scale = {beta_scale:.6f}")
        print(f"  ---------------------------------------------------------------------------------")
        print(f"  CONVENTION 1 [As-Is Raw Score (No Scaling)]:")
        print(f"    Train:   RMSE = {rmse_tr_asis:.4f} | R^2 = {r2_tr_asis * 100:>10,.1f}%")
        print(f"    Holdout: RMSE = {rmse_ho_asis:.4f} | R^2 = {r2_ho_asis * 100:>10,.1f}%")
        print(f"  ---------------------------------------------------------------------------------")
        print(f"  CONVENTION 2 [Train-Calibrated OLS Scale Factor (beta_scale={beta_scale:.6f})]:")
        print(f"    Train:   RMSE = {rmse_tr_sc:.5f} (Null: {null_rmse_tr:.5f}) | R^2 = {r2_tr_sc * 100:>+7.2f}%")
        print(f"    Holdout: RMSE = {rmse_ho_sc:.5f} (Null: {null_rmse_ho:.5f}) | R^2 = {r2_ho_sc * 100:>+7.2f}%")

        reconciliation_results[str(N)] = {
            "N": N,
            "score_std_train": round(float(np.std(s_tr)), 4),
            "score_std_holdout": round(float(np.std(s_ho)), 4),
            "return_std_train": round(null_rmse_tr, 5),
            "return_std_holdout": round(null_rmse_ho, 5),
            "beta_scale_train": round(beta_scale, 6),
            "convention_1_as_is": {
                "train_rmse": round(rmse_tr_asis, 4),
                "train_r2_pct": round(r2_tr_asis * 100.0, 1),
                "holdout_rmse": round(rmse_ho_asis, 4),
                "holdout_r2_pct": round(r2_ho_asis * 100.0, 1),
            },
            "convention_2_train_scaled": {
                "train_rmse": round(rmse_tr_sc, 5),
                "train_r2_pct": round(r2_tr_sc * 100.0, 2),
                "holdout_rmse": round(rmse_ho_sc, 5),
                "holdout_r2_pct": round(r2_ho_sc * 100.0, 2),
            },
        }

    out_file = Path("scratch/reconcile_r2_rmse_conventions.json")
    out_file.write_text(json.dumps(reconciliation_results, indent=2))
    print("\n" + "=" * 115)
    print(f"Reconciliation results saved to {out_file}")
    print("=" * 115)
    return reconciliation_results


if __name__ == "__main__":
    run_reconciliation()
