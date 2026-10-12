"""Multi-Seed Search Stability Audit on Block A (2021-2023) and Controlled Two-Tier Ablation Test on Block B (2024-2025).

Investigates:
  1. Multi-Seed Search Stability on Development Data:
     - Pre-registered seeds: [42, 101, 777, 1337, 2024]
     - Universes: N in [15, 30, 60, 90]
     - Feature space: 26 base signals + 1 interaction (PRODUCT_H_BB_TAKER) = 27 terms
     - Records selection frequencies, coefficient distributions, sign stability,
       and pairwise Jaccard similarity across random seeds.
  2. Controlled Two-Tier Ablation Test on Diagnostic Holdout (Block B: 2024-2025):
     - Test 1: Does the recurring H-BB component alone retain predictive value out-of-sample?
     - Test 2: Do universe-specific terms add reproducible value beyond H-BB alone?
     - Models:
       * Policy 1: Standalone H-BB (w_bb * X^*_{H-BB})
       * Policy 2: Full Evolved Equation (H-BB + universe terms)
       * Policy 3: Universe-Specific Terms Alone (Ablation: Full minus H-BB)
       * Policy 4: Baseline Predictor (Candidate A, 4-term linear model)
     - Metrics: Net Return, Gross Return, Sharpe, Max Drawdown, Exposure, Turnover.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apedp_dyn.sandbox.grammar_space import (
    evaluate_equation_a_score,
    op_product_interaction,
    op_sign,
)
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    MULTI_90_FILE,
    SIGNAL_NAMES,
    calc_signals_for_asset,
    load_universe,
    run_multi_asset_genetic_algorithm,
)

PRE_REGISTERED_SEEDS = [42, 101, 777, 1337, 2024]
UNIVERSES = [15, 30, 60, 90]
FEE_RATE = 5.0 / 10000.0  # 5.0 bps
INITIAL_CAPITAL = 100000.0


def simulate_portfolio(
    pos_arr: np.ndarray,
    returns_2d: np.ndarray,
    vols_2d: np.ndarray,
    fee_rate: float = FEE_RATE,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, float]:
    """Execute canonical multi-asset portfolio simulation with inverse-volatility weighting."""
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


def compute_jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return float(intersection / union) if union > 0 else 1.0


def run_comprehensive_audit():
    print("=" * 115)
    print("APEDP-X EMPIRICAL AUDIT: MULTI-SEED SELECTION STABILITY & TWO-TIER ABLATION")
    print(f"Pre-Registered Seeds: {PRE_REGISTERED_SEEDS}")
    print(f"Target Universes: N in {UNIVERSES}")
    print("Feature Grammar: 26 Base Signals + 1 Interaction Term (PRODUCT_H_BB_TAKER) = 27 Terms")
    print("=" * 115)

    # 1. Load Data
    df_90 = pd.read_csv(MULTI_90_FILE)
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()
    idx_train = np.where(dates <= "2023-12-31")[0]
    idx_holdout = np.where(dates >= "2024-01-01")[0]

    uni_90, syms_90 = load_universe("90")
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_full = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().bfill().to_numpy()
    cross_med = np.median(rets_full, axis=1)

    K_base = len(SIGNAL_NAMES)
    expanded_names = SIGNAL_NAMES + ["PRODUCT_H_BB_TAKER (BB x Taker Surge)"]
    short_names = [n.split()[0] for n in expanded_names]
    K_total = len(expanded_names)

    # Precompute features for all 90 assets
    feats_90 = np.zeros((len(df_90), 90, K_total))
    for i in range(90):
        f_asset = calc_signals_for_asset(uni_90[syms_90[i]], cross_med)
        inter = op_product_interaction(f_asset[:, 9], f_asset[:, 24])
        feats_90[:, i, :K_base] = f_asset
        feats_90[:, i, K_base] = inter

    print(f"Dataset Loaded: Total Bars = {len(df_90)} | Train (Block A) = {len(idx_train)} | Holdout (Block B) = {len(idx_holdout)}")

    # =========================================================================
    # PART 1: MULTI-SEED SEARCH STABILITY AUDIT ON BLOCK A
    # =========================================================================
    print("\n" + "=" * 115)
    print("PART 1: MULTI-SEED SEARCH STABILITY AUDIT (BLOCK A: 2021-2023 DEVELOPMENT DATA)")
    print("=" * 115)

    search_results = {}
    global_feature_selection_counts = np.zeros(K_total, dtype=int)
    total_discovery_runs = len(UNIVERSES) * len(PRE_REGISTERED_SEEDS)

    for N in UNIVERSES:
        f_tr = feats_90[idx_train, :N, :]
        r_tr = rets_full[idx_train, :N]
        v_tr = vols_full[idx_train, :N]

        chroms = []
        masks = []
        weights_per_seed = []
        thresholds = []

        print(f"\n--- Universe N = {N} Assets (Running 5 Pre-Registered Seeds) ---")
        for s in PRE_REGISTERED_SEEDS:
            chrom = run_multi_asset_genetic_algorithm(
                f_tr, r_tr, v_tr, pop_size=60, generations=25, seed=s
            )
            chroms.append(chrom)
            masks.append(chrom.mask)
            weights_per_seed.append(chrom.weights)
            thresholds.append(chrom.threshold)
            global_feature_selection_counts += chrom.mask.astype(int)

            active_idx = np.where(chrom.mask)[0]
            terms_str = " ".join([f"{chrom.weights[i]:+.4f}*{short_names[i]}" for i in active_idx])
            print(f"  Seed {s:<5d} | Active Terms: {len(active_idx):2d} | Th: {chrom.threshold:.4f} | Fitness: {chrom.fitness:.3f} | S = {terms_str}")

        # Pairwise Jaccard similarities
        jaccards = []
        for i in range(len(PRE_REGISTERED_SEEDS)):
            for j in range(i + 1, len(PRE_REGISTERED_SEEDS)):
                jaccards.append(compute_jaccard_similarity(masks[i], masks[j]))
        mean_jaccard = float(np.mean(jaccards))

        # Per-feature statistics for this universe
        feat_stats = {}
        for k in range(K_total):
            selected_count = sum(masks[s_idx][k] for s_idx in range(len(PRE_REGISTERED_SEEDS)))
            if selected_count > 0:
                weights = [weights_per_seed[s_idx][k] for s_idx in range(len(PRE_REGISTERED_SEEDS)) if masks[s_idx][k]]
                pos_count = sum(1 for w in weights if w > 0)
                neg_count = sum(1 for w in weights if w < 0)
                feat_stats[short_names[k]] = {
                    "index": k,
                    "name": expanded_names[k],
                    "selection_count": selected_count,
                    "selection_freq": round(selected_count / len(PRE_REGISTERED_SEEDS) * 100.0, 1),
                    "mean_weight": round(float(np.mean(weights)), 4),
                    "std_weight": round(float(np.std(weights)), 4),
                    "min_weight": round(float(np.min(weights)), 4),
                    "max_weight": round(float(np.max(weights)), 4),
                    "sign_consistency": "100% Positive" if pos_count == selected_count else ("100% Negative" if neg_count == selected_count else f"{pos_count}+/{neg_count}-"),
                }

        term_counts = [int(np.sum(m)) for m in masks]
        search_results[str(N)] = {
            "N": N,
            "mean_active_terms": round(float(np.mean(term_counts)), 2),
            "std_active_terms": round(float(np.std(term_counts)), 2),
            "mean_threshold": round(float(np.mean(thresholds)), 4),
            "std_threshold": round(float(np.std(thresholds)), 4),
            "mean_pairwise_jaccard": round(mean_jaccard, 3),
            "feature_stats": feat_stats,
        }

    # Summary table of multi-seed selection for each universe
    print("\n" + "=" * 115)
    print("SELECTION STABILITY SUMMARY BY UNIVERSE (5 SEEDS PER UNIVERSE)")
    print("=" * 115)
    print(f"{'Universe':<8} | {'Mean Terms':<12} | {'Mean Th':<10} | {'Pairwise Jaccard':<18} | {'H-BB Frequency':<16} | {'H-BB Mean Weight (Sign)':<24}")
    print("-" * 115)
    for N in UNIVERSES:
        res = search_results[str(N)]
        bb_stat = res["feature_stats"].get("H-BB", None)
        if bb_stat:
            bb_str = f"{bb_stat['selection_count']}/5 ({bb_stat['selection_freq']}%)"
            bb_w_str = f"{bb_stat['mean_weight']:+.4f} ({bb_stat['sign_consistency']})"
        else:
            bb_str = "0/5 (0.0%)"
            bb_w_str = "N/A"
        print(f"N = {N:<4d} | {res['mean_active_terms']:4.1f} ± {res['std_active_terms']:3.1f} | {res['mean_threshold']:.4f}     | {res['mean_pairwise_jaccard']:<18.3f} | {bb_str:<16} | {bb_w_str:<24}")

    # Global cross-universe recurrence table across all 20 runs
    print("\n" + "=" * 115)
    print(f"GLOBAL FEATURE RECURRENCE RANKING ACROSS ALL {total_discovery_runs} SEARCH RUNS (4 UNIVERSES x 5 SEEDS)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Signal Identifier':<24} | {'Full Feature Name':<38} | {'Total Selected':<15} | {'Recurrence %':<12}")
    print("-" * 115)

    sorted_feat_indices = np.argsort(-global_feature_selection_counts)
    global_ranking = []
    for rank, k in enumerate(sorted_feat_indices, 1):
        cnt = int(global_feature_selection_counts[k])
        pct = round(cnt / total_discovery_runs * 100.0, 1)
        global_ranking.append({
            "rank": rank,
            "short_name": short_names[k],
            "full_name": expanded_names[k],
            "count": cnt,
            "total_runs": total_discovery_runs,
            "recurrence_pct": pct,
        })
        if cnt > 0:
            print(f"{rank:<5d} | {short_names[k]:<24} | {expanded_names[k]:<38} | {cnt:<2d} / {total_discovery_runs:<2d}         | {pct:>5.1f}%")

    # =========================================================================
    # PART 2: CONTROLLED TWO-TIER ABLATION TEST ON BLOCK B (HOLDOUT)
    # =========================================================================
    print("\n" + "=" * 115)
    print("PART 2: CONTROLLED TWO-TIER ABLATION TEST (BLOCK B: 2024-2025 DIAGNOSTIC HOLDOUT)")
    print("Evaluation Period: Block B (502 bars) | Fee: 5.0 bps | Inverse-Volatility Sizing")
    print("=" * 115)

    # Evolved canonical weights (from seed 42 development run)
    canonical_evolved = {
        15: {
            "weights": {9: 1.1948, 11: 0.2227, 16: 0.4501, 23: -0.4839, 26: -0.0161},
            "threshold": 0.1695,
            "bb_weight": 1.1948,
        },
        30: {
            "weights": {3: -0.1792, 9: 0.8179, 22: -0.2768},
            "threshold": 0.1572,
            "bb_weight": 0.8179,
        },
        60: {
            "weights": {3: -0.0545, 7: -0.0042, 9: 0.4323, 16: -0.8522, 17: 0.5357, 20: -1.3680, 23: -0.9019},
            "threshold": 0.0946,
            "bb_weight": 0.4323,
        },
        90: {
            "weights": {4: 0.7185, 5: -0.0925, 9: 0.7004, 11: -0.0685, 17: 0.7700},
            "threshold": 0.1525,
            "bb_weight": 0.7004,
        },
    }

    ablation_results = {}

    header = f"{'Universe':<8} | {'Strategy / Policy':<32} | {'Net Ret':<9} | {'Gross Ret':<9} | {'Sharpe':<7} | {'Max DD':<8} | {'Exposure':<9} | {'Turnover':<9}"
    print(header)
    print("-" * 115)

    for N in UNIVERSES:
        f_ho = feats_90[idx_holdout, :N, :]
        r_ho = rets_full[idx_holdout, :N]
        v_ho = vols_full[idx_holdout, :N]

        cfg = canonical_evolved[N]
        th = cfg["threshold"]
        w_dict = cfg["weights"]
        w_bb = cfg["bb_weight"]

        # Policy 1: Standalone H-BB
        sc_bb = f_ho[:, :, 9] * w_bb
        pos_bb = np.where(sc_bb > th, 1.0, np.where(sc_bb < -th, -1.0, 0.0))
        m_bb = simulate_portfolio(pos_bb, r_ho, v_ho)

        # Policy 2: Full Evolved Equation
        w_full = np.zeros(K_total)
        for k_idx, val in w_dict.items():
            w_full[k_idx] = val
        sc_full = f_ho @ w_full
        pos_full = np.where(sc_full > th, 1.0, np.where(sc_full < -th, -1.0, 0.0))
        m_full = simulate_portfolio(pos_full, r_ho, v_ho)

        # Policy 3: Universe-Specific Terms Alone (Ablation: Full minus H-BB)
        w_abl = w_full.copy()
        w_abl[9] = 0.0  # Zero out H-BB
        sc_abl = f_ho @ w_abl
        pos_abl = np.where(sc_abl > th, 1.0, np.where(sc_abl < -th, -1.0, 0.0))
        m_abl = simulate_portfolio(pos_abl, r_ho, v_ho)

        # Policy 4: Baseline Predictor (Candidate A, 4-term linear model with threshold 0.1979)
        sc_base = evaluate_equation_a_score(
            f_ho[:, :, 9], f_ho[:, :, 16], f_ho[:, :, 24], f_ho[:, :, 25], use_authoritative=True
        )
        pos_base = op_sign(sc_base, threshold=0.1979)
        m_base = simulate_portfolio(pos_base, r_ho, v_ho)

        # Incremental Deltas
        delta_net_universe_terms = round(m_full["net_return"] - m_bb["net_return"], 2)
        delta_sharpe_universe_terms = round(m_full["sharpe"] - m_bb["sharpe"], 2)
        delta_dd_universe_terms = round(m_full["max_drawdown"] - m_bb["max_drawdown"], 1)

        delta_hbb_contrib = round(m_full["net_return"] - m_abl["net_return"], 2)

        print(f"N = {N:<4d} | {'Policy 1: Standalone H-BB':<32} | {m_bb['net_return']:>+8.2f}% | {m_bb['gross_return']:>+8.2f}% | {m_bb['sharpe']:>+5.2f} | {m_bb['max_drawdown']:>6.1f}% | {m_bb['exposure']:>7.1f}% | {m_bb['turnover']:>8.1f}")
        print(f"         | {'Policy 2: Full Evolved Equation':<32} | {m_full['net_return']:>+8.2f}% | {m_full['gross_return']:>+8.2f}% | {m_full['sharpe']:>+5.2f} | {m_full['max_drawdown']:>6.1f}% | {m_full['exposure']:>7.1f}% | {m_full['turnover']:>8.1f}")
        print(f"         | {'Policy 3: Universe Terms (No H-BB)':<32} | {m_abl['net_return']:>+8.2f}% | {m_abl['gross_return']:>+8.2f}% | {m_abl['sharpe']:>+5.2f} | {m_abl['max_drawdown']:>6.1f}% | {m_abl['exposure']:>7.1f}% | {m_abl['turnover']:>8.1f}")
        print(f"         | {'Benchmark: Candidate A Baseline':<32} | {m_base['net_return']:>+8.2f}% | {m_base['gross_return']:>+8.2f}% | {m_base['sharpe']:>+5.2f} | {m_base['max_drawdown']:>6.1f}% | {m_base['exposure']:>7.1f}% | {m_base['turnover']:>8.1f}")
        print(f"  --> [N={N}] Universe Terms Δ over H-BB: Net={delta_net_universe_terms:+5.2f}pp | Sharpe={delta_sharpe_universe_terms:+4.2f} | DD={delta_dd_universe_terms:+4.1f}pp")
        print(f"  --> [N={N}] H-BB Marginal Value (Full vs Ablation): Net={delta_hbb_contrib:+5.2f}pp")
        print("-" * 115)

        ablation_results[str(N)] = {
            "N": N,
            "threshold": th,
            "standalone_hbb": m_bb,
            "full_evolved": m_full,
            "universe_terms_only": m_abl,
            "candidate_a_baseline": m_base,
            "delta_universe_over_hbb": {
                "net_return_pp": delta_net_universe_terms,
                "sharpe_delta": delta_sharpe_universe_terms,
                "max_drawdown_delta": delta_dd_universe_terms,
            },
            "hbb_marginal_value": {
                "net_return_pp": delta_hbb_contrib,
            },
        }

    def _to_serializable(val):
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return str(val)

    # Consolidated JSON artifact
    audit_data = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "pre_registered_seeds": [int(s) for s in PRE_REGISTERED_SEEDS],
        "universes": [int(u) for u in UNIVERSES],
        "train_period": "2021-2023 (Block A, 753 bars)",
        "holdout_period": "2024-2025 (Block B, 502 bars)",
        "part1_search_stability": search_results,
        "part1_global_feature_ranking": global_ranking,
        "part2_two_tier_ablation": ablation_results,
    }

    out_json = Path("scratch/audit_stability_ablation_results.json")
    json_str = json.dumps(audit_data, indent=2, default=_to_serializable)
    out_json.write_text(json_str)

    # Also persist to provenance record if available
    prov_dir = Path("provenance/selection_stability_and_ablation_20261009")
    if prov_dir.exists():
        (prov_dir / "results.json").write_text(json_str)

    print(f"\nAudit complete. Structured results written to: {out_json}")
    return audit_data


if __name__ == "__main__":
    run_comprehensive_audit()
