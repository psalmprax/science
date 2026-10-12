"""Controlled A/B Benchmark: Baseline Engine (Arm A) vs. Validation-Pruned Engine (Arm B).

Tests the user's specific hypothesis:
  Does internal-validation backward ablation and parsimony pruning improve:
    1. Discovery stability (decision equivalence across seeds)?
    2. Incremental value on validation data?
    3. Trading validity on protected test data (Block B)?

Protocol:
  - Block A Development (753 bars: 2021-01-04 to 2023-12-29)
    * Sub-Train: bars 0 to 500 (501 bars, 2021-2022)
    * Internal Validation: bars 500 to 753 (252 bars, 2023)
  - Block B Protected Test (502 bars: 2024-01-02 to 2025-12-31)
  - Pre-registered seeds: [42, 101, 777, 1337, 2024]
  - Universes: N in [15, 30, 60, 90]
  - Friction: 5.0 bps transaction fee, inverse-volatility weighting, capital compounding.
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
    SIGNAL_NAMES,
    calc_signals_for_asset,
    load_universe,
    run_multi_asset_genetic_algorithm,
)

PRE_REGISTERED_SEEDS = [42, 101, 777, 1337, 2024]
UNIVERSES = [15, 30, 60, 90]
FEE_RATE = 5.0 / 10000.0
INITIAL_CAPITAL = 100000.0


def simulate_portfolio(
    pos_arr: np.ndarray,
    returns_2d: np.ndarray,
    vols_2d: np.ndarray,
    fee_rate: float = FEE_RATE,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, float]:
    T, N = pos_arr.shape
    cap_net = initial_capital
    cap_gross = initial_capital
    eq_net = [cap_net]
    to_tot = 0.0
    cur_p = np.zeros(N)
    r_net = []
    r_gross = []

    for t in range(T - 1):
        p_tgt = pos_arr[t]
        sig = np.maximum(vols_2d[t], 0.005)
        act = p_tgt != 0.0
        if np.sum(act) > 0:
            inv = (1.0 / sig) * act
            w = (p_tgt * inv) / np.maximum(np.sum(inv), 1e-12)
        else:
            w = np.zeros(N)

        to = float(np.sum(np.abs(p_tgt - cur_p)))
        to_tot += to
        cost = cap_net * to * (fee_rate / N)

        g = float(np.sum(w * returns_2d[t + 1]))
        n = g - (cost / cap_net)
        cap_gross *= (1.0 + g)
        cap_net *= (1.0 + n)
        eq_net.append(cap_net)
        r_net.append(n)
        r_gross.append(g)
        cur_p = p_tgt

    ret_net = float((cap_net - initial_capital) / (initial_capital / 100.0))
    ret_gross = float((cap_gross - initial_capital) / (initial_capital / 100.0))
    sr_net = float(np.mean(r_net) / np.std(r_net) * np.sqrt(365)) if np.std(r_net) > 1e-9 else 0.0
    eq_n = np.array(eq_net)
    pk_n = np.maximum.accumulate(eq_n)
    dd = float(np.max((pk_n - eq_n) / np.maximum(pk_n, 1e-12) * 100.0))
    exp_pct = float(np.mean(np.abs(pos_arr)) * 100.0)

    return {
        "net_return": round(ret_net, 2),
        "gross_return": round(ret_gross, 2),
        "sharpe": round(sr_net, 2),
        "max_drawdown": round(dd, 1),
        "exposure": round(exp_pct, 1),
        "turnover": round(to_tot, 1),
    }


def compute_positions(feats_split: np.ndarray, w_vec: np.ndarray, th: float) -> np.ndarray:
    sc = feats_split @ w_vec
    return np.where(sc > th, 1.0, np.where(sc < -th, -1.0, 0.0))


def run_controlled_ab_benchmark():
    print("=" * 115)
    print("APEDP-X CONTROLLED A/B EXPERIMENT: BASELINE (ARM A) VS. VALIDATION-PRUNED (ARM B)")
    print(f"Pre-Registered Seeds: {PRE_REGISTERED_SEEDS}")
    print(f"Target Universes: N in {UNIVERSES}")
    print("=" * 115)

    df_90 = pd.read_csv(MULTI_90_FILE)
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    idx_a = np.where(dates <= "2023-12-31")[0]
    idx_b = np.where(dates >= "2024-01-01")[0]

    # Strictly leakage-free temporal split inside Block A
    idx_subtrain = idx_a[:500]
    idx_val = idx_a[500:]

    uni_90, syms_90 = load_universe("90")
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_full = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().bfill().to_numpy()
    cross_med = np.median(rets_full, axis=1)

    K_base = len(SIGNAL_NAMES)
    expanded_names = SIGNAL_NAMES + ["PRODUCT_H_BB_TAKER (BB x Taker Surge)"]
    short_names = [n.split()[0] for n in expanded_names]
    K_total = len(expanded_names)

    feats_90 = np.zeros((len(df_90), 90, K_total))
    for i in range(90):
        f_asset = calc_signals_for_asset(uni_90[syms_90[i]], cross_med)
        inter = op_product_interaction(f_asset[:, 9], f_asset[:, 24])
        feats_90[:, i, :K_base] = f_asset
        feats_90[:, i, K_base] = inter

    print(f"Data Partitions: Sub-Train = {len(idx_subtrain)} bars | Validation = {len(idx_val)} bars | Protected Test = {len(idx_b)} bars")

    results_by_universe = {}

    for N in UNIVERSES:
        print(f"\n" + "=" * 115)
        print(f">>> BENCHMARKING UNIVERSE N = {N} ASSETS <<<")
        print("=" * 115)

        f_tr = feats_90[idx_subtrain, :N, :]
        r_tr = rets_full[idx_subtrain, :N]
        v_tr = vols_full[idx_subtrain, :N]

        f_val = feats_90[idx_val, :N, :]
        r_val = rets_full[idx_val, :N]
        v_val = vols_full[idx_val, :N]

        f_test = feats_90[idx_b, :N, :]
        r_test = rets_full[idx_b, :N]
        v_test = vols_full[idx_b, :N]

        arm_a_seeds = []
        arm_b_seeds = []

        arm_a_test_positions = []
        arm_b_test_positions = []

        for seed in PRE_REGISTERED_SEEDS:
            # 1. Evolve on Sub-Train
            chrom = run_multi_asset_genetic_algorithm(f_tr, r_tr, v_tr, pop_size=60, generations=25, seed=seed)
            w_orig = chrom.mask.astype(float) * chrom.weights
            th = chrom.threshold

            # Arm A: Unpruned baseline
            pos_val_a = compute_positions(f_val, w_orig, th)
            m_val_a = simulate_portfolio(pos_val_a, r_val, v_val)

            pos_test_a = compute_positions(f_test, w_orig, th)
            m_test_a = simulate_portfolio(pos_test_a, r_test, v_test)
            arm_a_test_positions.append(pos_test_a.flatten())

            active_a = list(np.where(w_orig != 0.0)[0])
            terms_a_str = " ".join([f"{w_orig[i]:+.4f}*{short_names[i]}" for i in active_a])

            # Arm B: Backward elimination using ONLY internal validation split (idx_val)
            w_pruned = w_orig.copy()
            active_b = active_a.copy()
            base_val_sr = m_val_a["sharpe"]

            # Backward elimination loop
            while len(active_b) > 1:
                drop_candidate = None
                best_sr = base_val_sr
                for k in active_b:
                    w_trial = w_pruned.copy()
                    w_trial[k] = 0.0
                    pos_trial = compute_positions(f_val, w_trial, th)
                    m_trial = simulate_portfolio(pos_trial, r_val, v_val)
                    # Prune if removing k maintains or improves validation Sharpe (tolerance epsilon = 0.02)
                    if m_trial["sharpe"] >= best_sr - 0.02:
                        best_sr = m_trial["sharpe"]
                        drop_candidate = k

                if drop_candidate is not None:
                    w_pruned[drop_candidate] = 0.0
                    active_b.remove(drop_candidate)
                    base_val_sr = best_sr
                else:
                    break

            pos_val_b = compute_positions(f_val, w_pruned, th)
            m_val_b = simulate_portfolio(pos_val_b, r_val, v_val)

            pos_test_b = compute_positions(f_test, w_pruned, th)
            m_test_b = simulate_portfolio(pos_test_b, r_test, v_test)
            arm_b_test_positions.append(pos_test_b.flatten())

            terms_b_str = " ".join([f"{w_pruned[i]:+.4f}*{short_names[i]}" for i in active_b])

            print(f"Seed {seed:<5d}:")
            print(f"  [Arm A Unpruned] ({len(active_a):2d} terms): Val Net={m_val_a['net_return']:>+6.2f}%, Val SR={m_val_a['sharpe']:>+5.2f} | Test Net={m_test_a['net_return']:>+6.2f}%, Test SR={m_test_a['sharpe']:>+5.2f}, TO={m_test_a['turnover']:6.1f}")
            print(f"                   Eq: S = {terms_a_str}")
            print(f"  [Arm B Pruned]   ({len(active_b):2d} terms): Val Net={m_val_b['net_return']:>+6.2f}%, Val SR={m_val_b['sharpe']:>+5.2f} | Test Net={m_test_b['net_return']:>+6.2f}%, Test SR={m_test_b['sharpe']:>+5.2f}, TO={m_test_b['turnover']:6.1f}")
            print(f"                   Eq: S = {terms_b_str}")

            arm_a_seeds.append({
                "seed": seed,
                "term_count": len(active_a),
                "active_indices": active_a,
                "val_perf": m_val_a,
                "test_perf": m_test_a,
            })
            arm_b_seeds.append({
                "seed": seed,
                "term_count": len(active_b),
                "active_indices": active_b,
                "val_perf": m_val_b,
                "test_perf": m_test_b,
            })

        # Calculate Decision Equivalence across seeds on Block B (Pearson & Spearman of positions)
        def calc_decision_correlations(pos_list):
            corrs = []
            for i in range(len(pos_list)):
                for j in range(i + 1, len(pos_list)):
                    c = np.corrcoef(pos_list[i], pos_list[j])[0, 1]
                    if np.isfinite(c):
                        corrs.append(float(c))
            return float(np.mean(corrs)) if corrs else 0.0

        dec_corr_a = calc_decision_correlations(arm_a_test_positions)
        dec_corr_b = calc_decision_correlations(arm_b_test_positions)

        # Average performance metrics across seeds
        avg_terms_a = float(np.mean([s["term_count"] for s in arm_a_seeds]))
        avg_terms_b = float(np.mean([s["term_count"] for s in arm_b_seeds]))

        avg_val_net_a = float(np.mean([s["val_perf"]["net_return"] for s in arm_a_seeds]))
        avg_val_net_b = float(np.mean([s["val_perf"]["net_return"] for s in arm_b_seeds]))
        avg_val_sr_a = float(np.mean([s["val_perf"]["sharpe"] for s in arm_a_seeds]))
        avg_val_sr_b = float(np.mean([s["val_perf"]["sharpe"] for s in arm_b_seeds]))

        avg_test_net_a = float(np.mean([s["test_perf"]["net_return"] for s in arm_a_seeds]))
        avg_test_net_b = float(np.mean([s["test_perf"]["net_return"] for s in arm_b_seeds]))
        avg_test_sr_a = float(np.mean([s["test_perf"]["sharpe"] for s in arm_a_seeds]))
        avg_test_sr_b = float(np.mean([s["test_perf"]["sharpe"] for s in arm_b_seeds]))
        avg_test_to_a = float(np.mean([s["test_perf"]["turnover"] for s in arm_a_seeds]))
        avg_test_to_b = float(np.mean([s["test_perf"]["turnover"] for s in arm_b_seeds]))
        avg_test_dd_a = float(np.mean([s["test_perf"]["max_drawdown"] for s in arm_a_seeds]))
        avg_test_dd_b = float(np.mean([s["test_perf"]["max_drawdown"] for s in arm_b_seeds]))

        print(f"\n--- Universe N = {N} A/B SUMMARY ACROSS 5 SEEDS ---")
        print(f"  Outcome 1 (Stability): Decision Correlation across seeds: Arm A = {dec_corr_a:+.3f} vs. Arm B = {dec_corr_b:+.3f}")
        print(f"                         Mean Term Count:                 Arm A = {avg_terms_a:.1f} terms vs. Arm B = {avg_terms_b:.1f} terms")
        print(f"  Outcome 2 (Val Incremental): Mean Val Net Return:       Arm A = {avg_val_net_a:>+6.2f}% vs. Arm B = {avg_val_net_b:>+6.2f}% (SR: {avg_val_sr_a:+.2f} vs {avg_val_sr_b:+.2f})")
        print(f"  Outcome 3 (Test Validity):   Mean Test Net Return:      Arm A = {avg_test_net_a:>+6.2f}% vs. Arm B = {avg_test_net_b:>+6.2f}%")
        print(f"                               Mean Test Sharpe:          Arm A = {avg_test_sr_a:>+5.2f} vs. Arm B = {avg_test_sr_b:>+5.2f}")
        print(f"                               Mean Test Turnover:        Arm A = {avg_test_to_a:6.1f} vs. Arm B = {avg_test_to_b:6.1f}")
        print(f"                               Mean Test Max Drawdown:    Arm A = {avg_test_dd_a:5.1f}% vs. Arm B = {avg_test_dd_b:5.1f}%")

        results_by_universe[str(N)] = {
            "N": N,
            "outcome_1_stability": {
                "decision_correlation_arm_a": round(dec_corr_a, 3),
                "decision_correlation_arm_b": round(dec_corr_b, 3),
                "mean_term_count_arm_a": round(avg_terms_a, 1),
                "mean_term_count_arm_b": round(avg_terms_b, 1),
            },
            "outcome_2_validation_incremental": {
                "mean_val_net_arm_a": round(avg_val_net_a, 2),
                "mean_val_net_arm_b": round(avg_val_net_b, 2),
                "mean_val_sharpe_arm_a": round(avg_val_sr_a, 2),
                "mean_val_sharpe_arm_b": round(avg_val_sr_b, 2),
            },
            "outcome_3_test_validity": {
                "mean_test_net_arm_a": round(avg_test_net_a, 2),
                "mean_test_net_arm_b": round(avg_test_net_b, 2),
                "mean_test_sharpe_arm_a": round(avg_test_sr_a, 2),
                "mean_test_sharpe_arm_b": round(avg_test_sr_b, 2),
                "mean_test_turnover_arm_a": round(avg_test_to_a, 1),
                "mean_test_turnover_arm_b": round(avg_test_to_b, 1),
                "mean_test_drawdown_arm_a": round(avg_test_dd_a, 1),
                "mean_test_drawdown_arm_b": round(avg_test_dd_b, 1),
            },
            "seeds_detail_arm_a": arm_a_seeds,
            "seeds_detail_arm_b": arm_b_seeds,
        }

    # Consolidated JSON artifact
    def _to_serializable(val):
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return str(val)

    audit_out = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "pre_registered_seeds": PRE_REGISTERED_SEEDS,
        "universes": UNIVERSES,
        "subtrain_bars": len(idx_subtrain),
        "validation_bars": len(idx_val),
        "test_bars": len(idx_b),
        "universes_results": results_by_universe,
    }

    out_json = Path("scratch/controlled_ab_benchmark_results.json")
    json_str = json.dumps(audit_out, indent=2, default=_to_serializable)
    out_json.write_text(json_str)

    prov_dir = Path("provenance/controlled_ab_benchmark_20261009")
    if prov_dir.exists():
        (prov_dir / "results.json").write_text(json_str)

    print(f"\nControlled A/B benchmark complete. Structured results written to: {out_json}")
    return audit_out


if __name__ == "__main__":
    run_controlled_ab_benchmark()
