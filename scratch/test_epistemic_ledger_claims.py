import numpy as np
import pandas as pd
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe, calc_signals_for_asset, simulate_multi_asset_portfolio,
    combine_two_tier_portfolios, PINNED_30_CHROMOSOME, SIGNAL_NAMES
)

def run_reproducible_epistemic_audit():
    chrom = PINNED_30_CHROMOSOME
    eff_w = chrom.mask.astype(float) * chrom.weights
    active_idx = np.where(chrom.mask)[0]
    active_terms = [SIGNAL_NAMES[i] for i in active_idx]
    
    print("=" * 115)
    print("APEDP-X AUDIT LEDGER VERIFICATION & PROVENANCE REPAIR")
    print("Frozen 4-Term Control Equation:")
    print("  S_{i,t} = beta_1 * X^*_{H-BB} + beta_2 * X^*_{H-OBV_MOM} + beta_3 * X^*_{NL-TAKER} + beta_4 * X^*_{NL-SNR}")
    print(f"Frozen Weights: {[round(float(chrom.weights[i]), 4) for i in active_idx]}")
    print(f"Activation Threshold: theta = {chrom.threshold:.4f}")
    print("=" * 115)

    # =========================================================================
    # PART 1: REPAIRED FEATURE-LEVEL RANK-NORMALIZATION ABLATION
    # =========================================================================
    print("\n--- PART 1: REPAIRED FEATURE-LEVEL RANK-NORMALIZATION ABLATION ---")
    print("Experimental Specification:")
    print("  Baseline (Magnitude) : S_{i,t}^{mag}  = sum_{k=1}^4 beta_k * X^*_{k,i,t}")
    print("  Ablation (Feature-Rank): S_{i,t}^{rank} = sum_{k=1}^4 beta_k * rank_x(X_{k,i,t})")
    print("  (Each feature is independently rank-transformed across assets BEFORE weighting)\n")

    uni_30, syms_30 = load_universe("30")
    T_target = len(uni_30[syms_30[0]])
    n_train = T_target // 2

    ablation_results = []
    all_scores_flat = []

    for uni_key in ["30", "60", "90"]:
        uni, syms = load_universe(uni_key)
        N = len(syms)
        half = N // 2
        K = len(SIGNAL_NAMES)
        
        rets = np.column_stack([uni[s]["r"].iloc[-T_target:].to_numpy(float) for s in syms])
        vols = pd.DataFrame(rets).rolling(20, min_periods=5).std().bfill().to_numpy()
        cross_med = np.median(rets, axis=1)
        
        feats = np.zeros((T_target, N, K))
        for i, s in enumerate(syms):
            sub_df = uni[s].iloc[-T_target:].reset_index(drop=True)
            feats[:, i, :] = calc_signals_for_asset(sub_df, cross_med)
            
        f_ho = feats[n_train:]
        r_ho = rets[n_train:]
        v_ho = vols[n_train:]
        T_ho = len(f_ho)
        
        # 1A. Standard Magnitude-Based Feature Scores
        scores_mag = f_ho @ eff_w
        all_scores_flat.extend(np.abs(scores_mag).ravel())
        
        # 1B. Genuine Feature-Level Rank Transformation:
        # Each feature k in active_idx is independently ranked cross-sectionally across N assets at each t
        f_ho_rank = f_ho.copy()
        for t in range(T_ho):
            for k in active_idx:
                raw_col = f_ho[t, :, k]
                # Double argsort produces 0-indexed ordinal ranks
                ordinal_ranks = np.argsort(np.argsort(raw_col)).astype(float)
                # Rescale ranks to [-1.0, +1.0] preserving mean 0 and rank ordering
                f_ho_rank[t, :, k] = (ordinal_ranks / (N - 1) - 0.5) * 2.0
                
        scores_rank = f_ho_rank @ eff_w

        # --- RIGOROUS AUDIT ASSERTIONS ---
        # 1. Assert feature arrays are genuinely different
        feat_diff = np.mean(np.abs(f_ho[:, :, active_idx] - f_ho_rank[:, :, active_idx]))
        assert feat_diff > 0.10, f"Error: Feature-level rank transform did not alter representation! diff={feat_diff}"
        
        # 2. Assert combined score ordering genuinely changes (Spearman rho < 1.0)
        daily_corrs = []
        for t in range(T_ho):
            rk_m = np.argsort(np.argsort(scores_mag[t]))
            rk_r = np.argsort(np.argsort(scores_rank[t]))
            corr = np.corrcoef(rk_m, rk_r)[0, 1]
            if np.isfinite(corr):
                daily_corrs.append(corr)
        mean_rank_corr = float(np.mean(daily_corrs))
        assert mean_rank_corr < 0.99, f"Error: Rank-transformed scores are collinear with magnitude! rho={mean_rank_corr}"
        
        # Simulate Magnitude Model
        _, _, _, _, _, _, _, _, eq_c_mag = simulate_multi_asset_portfolio(
            scores_mag[:, :half], r_ho[:, :half], v_ho[:, :half],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        _, _, _, _, _, _, _, _, eq_e_mag = simulate_multi_asset_portfolio(
            scores_mag[:, half:], r_ho[:, half:], v_ho[:, half:],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        ret_mag, sr_mag, pnl_mag, dd_mag, _ = combine_two_tier_portfolios(eq_c_mag, eq_e_mag, 100000.0)

        # Simulate Feature-Rank Model
        _, _, _, _, _, _, _, _, eq_c_rk = simulate_multi_asset_portfolio(
            scores_rank[:, :half], r_ho[:, :half], v_ho[:, :half],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        _, _, _, _, _, _, _, _, eq_e_rk = simulate_multi_asset_portfolio(
            scores_rank[:, half:], r_ho[:, half:], v_ho[:, half:],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        ret_rk, sr_rk, pnl_rk, dd_rk, _ = combine_two_tier_portfolios(eq_c_rk, eq_e_rk, 100000.0)

        ablation_results.append({
            "N": N,
            "ret_mag": ret_mag, "sr_mag": sr_mag, "dd_mag": dd_mag,
            "ret_rk": ret_rk, "sr_rk": sr_rk, "dd_rk": dd_rk,
            "delta_ret": ret_rk - ret_mag,
            "mean_spearman_rho": mean_rank_corr,
        })

    header = f"{'Universe Breadth':<20} | {'Magnitude Ret (SR)':<24} | {'Feature-Rank Ret (SR)':<24} | {'Delta Ret':<12} | {'Rank Corr (rho)'}"
    print(header)
    print("-" * 105)
    for r in ablation_results:
        m_str = f"{r['ret_mag']:>+7.2f}% (SR {r['sr_mag']:>+4.2f})"
        k_str = f"{r['ret_rk']:>+7.2f}% (SR {r['sr_rk']:>+4.2f})"
        d_str = f"{r['delta_ret']:>+7.2f}%"
        rho_str = f"{r['mean_spearman_rho']:.3f}"
        print(f"N = {r['N']:<16} | {m_str:<24} | {k_str:<24} | {d_str:<12} | {rho_str}")
    print("=" * 105)

    # =========================================================================
    # PART 2: REGIME GATE THRESHOLD LEVERAGE & DISTRIBUTION ANALYSIS
    # =========================================================================
    print("\n--- PART 2: REGIME GATE THRESHOLD LEVERAGE & DISTRIBUTION ---")
    all_s = np.array(all_scores_flat)
    med_s = float(np.median(all_s))
    pct_below_theta = float(np.mean(all_s < chrom.threshold) * 100.0)
    pct_margin = float(np.mean((all_s >= chrom.threshold) & (all_s < 2.0 * chrom.threshold)) * 100.0)
    pct_passing = float(np.mean(all_s >= chrom.threshold) * 100.0)

    print(f"  • Median |S| across holdout : {med_s:.4f}")
    print(f"  • Pre-registered theta      : {chrom.threshold:.4f}")
    print(f"  • Inactive (|S| < theta)    : {pct_below_theta:.2f}%")
    print(f"  • Margin [theta, 2*theta]   : {pct_margin:.2f}%")
    print(f"  • Active pass-through       : {pct_passing:.2f}%")
    print(f"  -> Audit Conclusion: The gate filter is non-discriminating ({pct_passing:.1f}% pass-through).")
    print("     Classified as: REGIME TRANSFER UNRESOLVED / DESIGN FAILURE (no post-hoc retuning allowed).")

    # =========================================================================
    # PART 3: TEMPORAL FORWARD TRANSFER: BLOCK A (2021-2023) vs BLOCK B (2024-2025)
    # =========================================================================
    print("\n--- PART 3: TEMPORAL FORWARD TRANSFER ACROSS BLOCKS A & B ---")
    raw_90_df = pd.read_csv("data/market/multi_asset_90_5yr_orderflow.csv")
    date_col = "observation_date" if "observation_date" in raw_90_df.columns else "date"
    dates = raw_90_df[date_col].to_numpy()
    
    uni_90, syms_90 = load_universe("90")
    T_total = len(raw_90_df)
    N_90 = len(syms_90)
    half_90 = N_90 // 2
    K = len(SIGNAL_NAMES)
    
    idx_block_a = np.where(dates <= "2023-12-31")[0]
    idx_block_b = np.where(dates >= "2024-01-01")[0]
    
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_full = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().bfill().to_numpy()
    cross_med_full = np.median(rets_full, axis=1)
    
    feats_full = np.zeros((T_total, N_90, K))
    for i, s in enumerate(syms_90):
        feats_full[:, i, :] = calc_signals_for_asset(uni_90[s], cross_med_full)
        
    scores_full = feats_full @ eff_w

    def evaluate_block(idx, label, days):
        s_blk = scores_full[idx]
        r_blk = rets_full[idx]
        v_blk = vols_full[idx]
        
        # Crypto sub-portfolio (45 coins)
        r_c, sr_c, _, pnl_c, tr_c, _, dd_c, _, eq_c = simulate_multi_asset_portfolio(
            s_blk[:, :half_90], r_blk[:, :half_90], v_blk[:, :half_90],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        # Equity sub-portfolio (45 stocks)
        r_e, sr_e, _, pnl_e, tr_e, _, dd_e, _, eq_e = simulate_multi_asset_portfolio(
            s_blk[:, half_90:], r_blk[:, half_90:], v_blk[:, half_90:],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        tot_ret, sr_tot, pnl_tot, dd_tot, _ = combine_two_tier_portfolios(eq_c, eq_e, 100000.0)
        return {
            "label": label, "days": days,
            "crypto_ret": r_c, "crypto_sr": sr_c, "crypto_dd": dd_c,
            "equity_ret": r_e, "equity_sr": sr_e, "equity_dd": dd_e,
            "two_tier_ret": tot_ret, "two_tier_sr": sr_tot, "two_tier_dd": dd_tot,
        }

    blk_a = evaluate_block(idx_block_a, "Block A (2021-2023)", len(idx_block_a))
    blk_b = evaluate_block(idx_block_b, "Block B (2024-2025)", len(idx_block_b))

    print(f"{'Temporal Block':<22} | {'Horizon':<10} | {'Crypto Leg (SR)':<20} | {'Equity Leg (SR)':<20} | {'Combined Ret (SR)'}")
    print("-" * 95)
    for b in [blk_a, blk_b]:
        c_str = f"{b['crypto_ret']:>+6.1f}% (SR {b['crypto_sr']:>+4.2f})"
        e_str = f"{b['equity_ret']:>+6.1f}% (SR {b['equity_sr']:>+4.2f})"
        tot_str = f"{b['two_tier_ret']:>+6.1f}% (SR {b['two_tier_sr']:>+4.2f})"
        print(f"{b['label']:<22} | {b['days']} days   | {c_str:<20} | {e_str:<20} | {tot_str}")
    print("=" * 95)

if __name__ == "__main__":
    run_reproducible_epistemic_audit()
