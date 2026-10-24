import numpy as np
import pandas as pd
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe, calc_signals_for_asset, PINNED_30_CHROMOSOME, SIGNAL_NAMES
)

def run_defensive_hysteresis_investigation():
    chrom = PINNED_30_CHROMOSOME
    eff_w = chrom.mask.astype(float) * chrom.weights
    active_idx = np.where(chrom.mask)[0]
    
    print("=" * 115)
    print("APEDP-X EMPIRICAL INVESTIGATION: DEFENSIVE DECISION LAYER (HYSTERESIS)")
    print("Layer 1 (Frozen Predictor): S_{i,t} = sum_{k=1}^4 beta_k * X^*_{k,i,t}")
    print(f"Frozen Weights: {[round(float(chrom.weights[i]), 4) for i in active_idx]}")
    print("Layer 2 (Defensive Decision Rule): p_{i,t} = pi(S_{i,t}, p_{i,t-1}; theta_in, theta_out)")
    print("Parameters: theta_in = 0.1979, theta_out = 0.0000, fee = 5.0 bps (0.0005)")
    print("Universes: Breadth Expansion Panel (N=15, 30, 60, 90 assets)")
    print("=" * 115)

    df_90 = pd.read_csv("data/market/multi_asset_90_5yr_orderflow.csv")
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    
    uni_90, syms_90 = load_universe("90")
    rets_90 = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_90 = pd.DataFrame(rets_90).rolling(20, min_periods=5).std().bfill().to_numpy()
    cross_med = np.median(rets_90, axis=1)

    feats_90 = np.zeros((len(df_90), 90, len(SIGNAL_NAMES)))
    for i, s in enumerate(syms_90):
        feats_90[:, i, :] = calc_signals_for_asset(uni_90[s], cross_med)
    scores_90 = feats_90 @ eff_w

    idx_block_b = np.where(dates >= "2024-01-01")[0]
    idx_block_a = np.where(dates <= "2023-12-31")[0]
    idx_full = np.arange(len(df_90))

    th_in = 0.1979
    th_out = 0.0000
    fee_bps = 5.0
    fee = fee_bps / 10000.0

    def evaluate_universe(sc, r, v, u_size):
        T = len(sc)
        # Baseline (No Hysteresis: p_t = sign(S_t) if |S_t| > th_in else 0)
        pos_base = np.where(sc > th_in, 1.0, np.where(sc < -th_in, -1.0, 0.0))
        
        # Hysteresis (State-Aware Persistence: enter at th_in, clear/exit at th_out=0)
        pos_hyst = np.zeros_like(sc)
        cur = np.zeros(u_size)
        for t in range(T):
            for a in range(u_size):
                s_val = sc[t, a]
                p_prev = cur[a]
                if p_prev > 0:
                    if s_val < -th_in:
                        cur[a] = -1.0
                    elif s_val < th_out:
                        cur[a] = 0.0
                elif p_prev < 0:
                    if s_val > th_in:
                        cur[a] = 1.0
                    elif s_val > -th_out:
                        cur[a] = 0.0
                else:
                    if s_val > th_in:
                        cur[a] = 1.0
                    elif s_val < -th_in:
                        cur[a] = -1.0
            pos_hyst[t] = cur.copy()

        def sim(pos_arr):
            cap_net = 100000.0
            cap_gross = 100000.0
            eq_net = [cap_net]
            eq_gross = [cap_gross]
            to_tot = 0.0
            trades_count = 0
            cur_p = np.zeros(u_size)
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
                    w = np.zeros(u_size)
                
                to = np.sum(np.abs(p_tgt - cur_p))
                to_tot += to
                trades_count += np.sum(p_tgt != cur_p)
                cost = cap_net * to * (fee / u_size)
                
                g = np.sum(w * r[t + 1])
                n = g - (cost / cap_net)
                
                cap_gross *= (1.0 + g)
                cap_net *= (1.0 + n)
                
                eq_gross.append(cap_gross)
                eq_net.append(cap_net)
                
                r_gross.append(g)
                r_net.append(n)
                cur_p = p_tgt
                
            ret_net = (cap_net - 100000.0) / 1000.0
            ret_gross = (cap_gross - 100000.0) / 1000.0
            sr_net = float(np.mean(r_net) / np.std(r_net) * np.sqrt(365)) if np.std(r_net) > 1e-9 else 0.0
            sr_gross = float(np.mean(r_gross) / np.std(r_gross) * np.sqrt(365)) if np.std(r_gross) > 1e-9 else 0.0
            
            eq_n = np.array(eq_net)
            pk_n = np.maximum.accumulate(eq_n)
            dd_net = float(np.max((pk_n - eq_n) / pk_n * 100.0))
            
            eq_g = np.array(eq_gross)
            pk_g = np.maximum.accumulate(eq_g)
            dd_gross = float(np.max((pk_g - eq_g) / pk_g * 100.0))
            
            return {
                "net_ret": ret_net, "gross_ret": ret_gross,
                "net_sr": sr_net, "gross_sr": sr_gross,
                "net_dd": dd_net, "gross_dd": dd_gross,
                "to": to_tot, "trades": trades_count,
            }

        res_b = sim(pos_base)
        res_h = sim(pos_hyst)
        to_red = (1.0 - res_h["to"] / max(res_b["to"], 1.0)) * 100.0
        return res_b, res_h, to_red

    blocks = [
        ("Block B (2024-2025 Out-Of-Sample)", idx_block_b),
        ("Block A (2021-2023 Historical Baseline)", idx_block_a),
        ("Full 5-Year Sample (2021-2025)", idx_full),
    ]

    all_results = {}

    for label, blk_idx in blocks:
        print(f"\n--- {label.upper()} (T = {len(blk_idx)} days) ---")
        header = f"{'Universe':<10} | {'Base Net (SR, DD)':<26} | {'Hyst Net (SR, DD)':<26} | {'Base Gross':<12} | {'Hyst Gross':<12} | {'TO Red':<8}"
        print(header)
        print("-" * 105)
        
        block_dict = {}
        for N in [15, 30, 60, 90]:
            sc_sub = scores_90[blk_idx, :N]
            r_sub = rets_90[blk_idx, :N]
            v_sub = vols_90[blk_idx, :N]
            
            rb, rh, to_red = evaluate_universe(sc_sub, r_sub, v_sub, N)
            block_dict[N] = (rb, rh, to_red)
            
            b_net_str = f"{rb['net_ret']:>+6.2f}% ({rb['net_sr']:>+4.2f}, {rb['net_dd']:>4.1f}%)"
            h_net_str = f"{rh['net_ret']:>+6.2f}% ({rh['net_sr']:>+4.2f}, {rh['net_dd']:>4.1f}%)"
            b_gr_str = f"{rb['gross_ret']:>+6.2f}%"
            h_gr_str = f"{rh['gross_ret']:>+6.2f}%"
            to_str = f"{to_red:>5.1f}%"
            
            print(f"N = {N:<6} | {b_net_str:<26} | {h_net_str:<26} | {b_gr_str:<12} | {h_gr_str:<12} | {to_str}")
        print("=" * 105)
        all_results[label] = block_dict

    # AUDIT ASSERTIONS
    blk_b = all_results["Block B (2024-2025 Out-Of-Sample)"]
    # 1. Assert exact match with reported user observations for N=30, 60, 90
    assert abs(blk_b[30][0]["net_ret"] - (-28.58)) < 0.05, f"Mismatch on N=30 base net: {blk_b[30][0]['net_ret']}"
    assert abs(blk_b[30][1]["net_ret"] - (-5.79)) < 0.05, f"Mismatch on N=30 hyst net: {blk_b[30][1]['net_ret']}"
    assert abs(blk_b[60][0]["net_ret"] - (-22.75)) < 0.05, f"Mismatch on N=60 base net: {blk_b[60][0]['net_ret']}"
    assert abs(blk_b[60][1]["net_ret"] - (-6.89)) < 0.05, f"Mismatch on N=60 hyst net: {blk_b[60][1]['net_ret']}"
    assert abs(blk_b[90][0]["net_ret"] - (-11.40)) < 0.05, f"Mismatch on N=90 base net: {blk_b[90][0]['net_ret']}"
    assert abs(blk_b[90][1]["net_ret"] - (-9.89)) < 0.05, f"Mismatch on N=90 hyst net: {blk_b[90][1]['net_ret']}"

    # 2. Assert turnover falls by 63% to 72% across all 4 universes
    for N in [15, 30, 60, 90]:
        red = blk_b[N][2]
        assert 60.0 <= red <= 75.0, f"Turnover reduction outside expected band: {red}% for N={N}"

    # 3. Assert Max DD falls for all 4 universes
    for N in [15, 30, 60, 90]:
        dd_b = blk_b[N][0]["net_dd"]
        dd_h = blk_b[N][1]["net_dd"]
        assert dd_h < dd_b, f"Max DD did not fall for N={N}: base {dd_b}% vs hyst {dd_h}%"

    # 4. Assert N=90 Gross Return Deterioration (Counterexample verification)
    assert blk_b[90][1]["gross_ret"] < blk_b[90][0]["gross_ret"], "N=90 Gross return did not deteriorate!"

    print("\n>>> ALL EMPIRICAL AND AUDIT ASSERTIONS PASSED WITH ZERO TOLERANCE VIOLATIONS <<<")
    return all_results

if __name__ == "__main__":
    run_defensive_hysteresis_investigation()
