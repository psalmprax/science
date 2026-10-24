import numpy as np
import pandas as pd
import json
from pathlib import Path
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe, calc_signals_for_asset, PINNED_30_CHROMOSOME, PINNED_60_CHROMOSOME, SIGNAL_NAMES
)

def run_defense_vs_exposure_experiment():
    print("=" * 115)
    print("APEDP-X EMPIRICAL AUDIT: DEFENSIVE VALUE VS REDUCED EXPOSURE CONTROL")
    print("Evaluation: 3-Strategy Controlled Comparison on 5 Predeclared Assets & Multi-Asset Universes")
    print("  Strategy A: Baseline Predictor (unmodified sizing & timing)")
    print("  Strategy B: Hysteresis Defence (state-aware persistence rule)")
    print("  Strategy C: Exposure-Matched Control (baseline timing scaled to match hysteresis average exposure)")
    print("=" * 115)

    df_90 = pd.read_csv("data/market/multi_asset_90_5yr_orderflow.csv")
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    idx_train = np.where(dates <= "2023-12-31")[0]
    idx_holdout = np.where(dates >= "2024-01-01")[0]

    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    uni_90, syms_90 = load_universe("90")
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_full = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().bfill().to_numpy()
    cross_med = np.median(rets_full, axis=1)

    feats_90 = np.zeros((len(df_90), 90, len(SIGNAL_NAMES)))
    for i, s in enumerate(syms_90):
        feats_90[:, i, :] = calc_signals_for_asset(uni_90[s], cross_med)

    chrom_a = PINNED_30_CHROMOSOME
    eff_w_a = chrom_a.mask.astype(float) * chrom_a.weights

    chrom_b = PINNED_60_CHROMOSOME
    eff_w_b = chrom_b.mask.astype(float) * chrom_b.weights

    scores_a = feats_90 @ eff_w_a
    scores_b = feats_90 @ eff_w_b

    fee_bps = 5.0
    fee = fee_bps / 10000.0

    # =========================================================================
    # PART 1: MULTI-ASSET PORTFOLIO EXPOSURE-MATCHED CONTROL (4-TERM MODEL)
    # =========================================================================
    print("\n--- PART 1: 4-TERM MODEL MULTI-ASSET EXPOSURE-MATCHED CONTROL (BLOCK B 2024-2025) ---")
    th_in = 0.1979
    th_out = 0.0000

    print(f"{'Universe':<10} | {'Strategy':<16} | {'Net Ret':<9} | {'Gross Ret':<9} | {'Sharpe':<7} | {'Max DD':<8} | {'Exposure':<9} | {'Turnover':<9}")
    print("-" * 95)
    for N in [15, 30, 60, 90]:
        sc = scores_a[idx_holdout, :N]
        r = rets_full[idx_holdout, :N]
        v = vols_full[idx_holdout, :N]
        T = len(sc)

        pos_base = np.where(sc > th_in, 1.0, np.where(sc < -th_in, -1.0, 0.0))
        pos_hyst = np.zeros_like(sc)
        cur = np.zeros(N)
        for t in range(T):
            for a in range(N):
                s_val = sc[t, a]
                p_prev = cur[a]
                if p_prev > 0:
                    if s_val < -th_in: cur[a] = -1.0
                    elif s_val < th_out: cur[a] = 0.0
                elif p_prev < 0:
                    if s_val > th_in: cur[a] = 1.0
                    elif s_val > -th_out: cur[a] = 0.0
                else:
                    if s_val > th_in: cur[a] = 1.0
                    elif s_val < -th_in: cur[a] = -1.0
            pos_hyst[t] = cur.copy()

        mean_exp_hyst = float(np.mean(np.abs(pos_hyst)))
        mean_exp_base = float(np.mean(np.abs(pos_base)))
        alpha = mean_exp_hyst / max(mean_exp_base, 1e-6)
        pos_exp = alpha * pos_base

        def sim_portfolio(pos_arr):
            cap_net = 100000.0
            cap_gross = 100000.0
            eq_net = [cap_net]
            to_tot = 0.0
            cur_p = np.zeros(N)
            r_net = []
            r_gross = []
            for t in range(T - 1):
                p_tgt = pos_arr[t]
                sig = np.maximum(v[t], 0.005)
                act = p_tgt != 0.0
                if np.sum(act) > 0:
                    inv = (1.0 / sig) * act
                    w = (p_tgt * inv) / np.maximum(np.sum(inv), 1e-12)
                else:
                    w = np.zeros(N)
                to = np.sum(np.abs(p_tgt - cur_p))
                to_tot += to
                cost = cap_net * to * (fee / N)
                g = np.sum(w * r[t + 1])
                n = g - (cost / cap_net)
                cap_gross *= (1.0 + g)
                cap_net *= (1.0 + n)
                eq_net.append(cap_net)
                r_net.append(n)
                r_gross.append(g)
                cur_p = p_tgt
            ret_net = (cap_net - 100000.0) / 1000.0
            ret_gross = (cap_gross - 100000.0) / 1000.0
            sr_net = float(np.mean(r_net) / np.std(r_net) * np.sqrt(365)) if np.std(r_net) > 1e-9 else 0.0
            eq_n = np.array(eq_net)
            pk_n = np.maximum.accumulate(eq_n)
            dd = float(np.max((pk_n - eq_n) / pk_n * 100.0))
            exp_pct = float(np.mean(np.abs(pos_arr)) * 100.0)
            return ret_net, ret_gross, sr_net, dd, exp_pct, to_tot

        mb = sim_portfolio(pos_base)
        mh = sim_portfolio(pos_hyst)
        me = sim_portfolio(pos_exp)

        print(f"N = {N:<6} | {'Baseline':<16} | {mb[0]:>+7.2f}% | {mb[1]:>+7.2f}% | {mb[2]:>+5.2f} | {mb[3]:>6.1f}% | {mb[4]:>7.1f}% | {mb[5]:>8.1f}")
        print(f"{'':<10} | {'Hysteresis':<16} | {mh[0]:>+7.2f}% | {mh[1]:>+7.2f}% | {mh[2]:>+5.2f} | {mh[3]:>6.1f}% | {mh[4]:>7.1f}% | {mh[5]:>8.1f}")
        print(f"{'':<10} | {'Exposure-Match':<16} | {me[0]:>+7.2f}% | {me[1]:>+7.2f}% | {me[2]:>+5.2f} | {me[3]:>6.1f}% | {me[4]:>7.1f}% | {me[5]:>8.1f}")
        print("-" * 95)

    # =========================================================================
    # PART 2: GENERALIZED PENALTY RECONCILIATION (LAMBDA=0.20 VS DISCRETE)
    # =========================================================================
    print("\n--- PART 2: GENERALIZED PENALTY RECONCILIATION (LAMBDA=0.20 VS DISCRETE) ---")
    sc_30 = scores_a[idx_holdout, :30]
    r_30 = rets_full[idx_holdout, :30]
    v_30 = vols_full[idx_holdout, :30]
    T_30 = len(sc_30)

    # Discrete
    pos_dh = np.zeros_like(sc_30)
    cur_dh = np.zeros(30)
    for t in range(T_30):
        for a in range(30):
            val = sc_30[t, a]
            prev = cur_dh[a]
            if prev > 0:
                cur_dh[a] = -1.0 if val < -0.1979 else (0.0 if val < 0.0 else 1.0)
            elif prev < 0:
                cur_dh[a] = 1.0 if val > 0.1979 else (0.0 if val > 0.0 else -1.0)
            else:
                cur_dh[a] = 1.0 if val > 0.1979 else (-1.0 if val < -0.1979 else 0.0)
        pos_dh[t] = cur_dh.copy()

    # Penalty lambda = 0.20
    pos_pen = np.zeros_like(sc_30)
    cur_pen = np.zeros(30)
    for t in range(T_30):
        for a in range(30):
            val = sc_30[t, a]
            prev = cur_pen[a]
            u_m1 = -val - 0.20 * abs(-1.0 - prev)
            u_0  =  0.0 - 0.20 * abs(0.0 - prev)
            u_p1 =  val - 0.20 * abs(1.0 - prev)
            best_p = [-1.0, 0.0, 1.0][np.argmax([u_m1, u_0, u_p1])]
            cur_pen[a] = best_p
        pos_pen[t] = cur_pen.copy()

    pos_divergence = np.mean(pos_dh != pos_pen) * 100.0
    print(f"  • Position divergence between Discrete Hysteresis and Penalty (lambda=0.20): {pos_divergence:.1f}%")
    print(f"  • Mathematical cause: Discrete exits at S=0; Penalty stays long until S <= -lambda (-0.20).")
    print(f"  • Status: Sweeping lambda on Block B is exploratory; it does NOT constitute independent validation.")

    print("\n>>> EXECUTION COMPLETE & AUDITED <<<")

if __name__ == "__main__":
    run_defense_vs_exposure_experiment()
