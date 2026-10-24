import numpy as np
import pandas as pd
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    load_universe, calc_signals_for_asset, simulate_multi_asset_portfolio,
    combine_two_tier_portfolios, PINNED_30_CHROMOSOME, SIGNAL_NAMES
)

def run_controlled_breadth_transfer():
    chrom = PINNED_30_CHROMOSOME
    eff_w = chrom.mask.astype(float) * chrom.weights
    active_terms = [SIGNAL_NAMES[i].split()[0] for i in np.where(chrom.mask)[0]]
    
    print("=" * 105)
    print("APEDP-X CONTROLLED BREADTH-TRANSFER BENCHMARK")
    print(f"Frozen Discovered Equation : Score_30 [{' + '.join(active_terms)}] (Threshold = {chrom.threshold:.4f})")
    print("Control Condition           : Equation specification, cost model (5 bps), and 3-yr window held strictly CONSTANT.")
    print("Independent Variable        : Expanding Universe Breadth N in {30, 60, 90 assets}.")
    print("=" * 105)

    results = []
    uni_30, syms_30 = load_universe("30")
    T_target = len(uni_30[syms_30[0]])

    for uni_key in ["30", "60", "90"]:
        uni, syms = load_universe(uni_key)
        N = len(syms)
        half = N // 2
        K = len(SIGNAL_NAMES)
        
        # Take the synchronized 3-year trailing window matching universe 30
        rets = np.column_stack([uni[s]["r"].iloc[-T_target:].to_numpy(float) for s in syms])
        vols = pd.DataFrame(rets).rolling(20, min_periods=5).std().bfill().to_numpy()
        cross_med = np.median(rets, axis=1)
        
        feats = np.zeros((T_target, N, K))
        for i, s in enumerate(syms):
            sub_df = uni[s].iloc[-T_target:].reset_index(drop=True)
            feats[:, i, :] = calc_signals_for_asset(sub_df, cross_med)
            
        n_train = T_target // 2
        f_ho = feats[n_train:]
        r_ho = rets[n_train:]
        v_ho = vols[n_train:]
        
        scores_ho = f_ho @ eff_w
        
        # 1. Crypto sub-portfolio
        r_c, sr_c, _, pnl_c, tr_c, _, dd_c, _, eq_c = simulate_multi_asset_portfolio(
            scores_ho[:, :half], r_ho[:, :half], v_ho[:, :half],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        
        # 2. Equity sub-portfolio
        r_e, sr_e, _, pnl_e, tr_e, _, dd_e, _, eq_e = simulate_multi_asset_portfolio(
            scores_ho[:, half:], r_ho[:, half:], v_ho[:, half:],
            mode="cross_sectional_ls", sizing="risk_parity", threshold=chrom.threshold,
            initial_capital=50000.0, return_equity_curve=True
        )
        
        # 3. Combined two-tier
        tot_ret, sr_tot, pnl_tot, dd_tot, _ = combine_two_tier_portfolios(eq_c, eq_e, 100000.0)
        
        results.append({
            "universe": f"{N}-Asset (3-Yr Sync)",
            "N": N,
            "crypto_ret": r_c, "crypto_sr": sr_c, "crypto_dd": dd_c,
            "equity_ret": r_e, "equity_sr": sr_e, "equity_dd": dd_e,
            "two_tier_ret": tot_ret, "two_tier_sr": sr_tot, "two_tier_dd": dd_tot,
            "trades": tr_c + tr_e
        })

    header = f"{'Universe Breadth':<22} | {'Crypto Leg: Ret (SR)':<22} | {'Equity Leg: Ret (SR)':<22} | {'Combined Ret':<14} | {'Combined SR':<11} | {'Max DD':<8} | {'Trades'}"
    print(header)
    print("-" * 115)
    for res in results:
        c_str = f"{res['crypto_ret']:>+6.1f}% (SR {res['crypto_sr']:>+4.2f})"
        e_str = f"{res['equity_ret']:>+6.1f}% (SR {res['equity_sr']:>+4.2f})"
        print(f"{res['universe']:<22} | {c_str:<22} | {e_str:<22} | {res['two_tier_ret']:>+10.2f}% | {res['two_tier_sr']:>+9.2f} | {res['two_tier_dd']:>6.1f}% | {res['trades']}")
    print("=" * 115)

if __name__ == "__main__":
    run_controlled_breadth_transfer()
