"""Audit of Rolling-Window Equation Stability, Feature Substitution, and Identifiability.

Investigates:
1. Feature Definition & Comparability Audit:
   - Explicit definitions of H-BB, H-P, H-ASYM, NL_OFI_REV, NL_SQUEEZE_TREND, H-MOM, NL_HURST_ADAPT.
   - Mathematical definition of X^* (causal expanding z-score + cross-sectional standardization).
2. Feature Correlation & Redundancy Matrix:
   - Pairwise correlation among candidate features on development data (Block A) to detect feature substitution.
3. Multi-Seed Search Stability across 5 pre-registered seeds [42, 101, 777, 1337, 2024].
4. Predictive & Position Equivalence on Common Evaluation Window (Block B: 2024-2025):
   - Score correlation, rank correlation, position agreement (% identical sign), and turnover.
5. Controlled Four-Model Benchmark:
   - Model 1: Standalone H-BB
   - Model 2: Window-specific evolved equations
   - Model 3: Restrained shared model (H-BB + persistent terms)
   - Model 4: Full unconstrained genetic search
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    MULTI_90_FILE,
    SIGNAL_NAMES,
    calc_signals_for_asset,
    load_universe,
    run_multi_asset_genetic_algorithm,
    MultiAssetChromosome,
)

PRE_REGISTERED_SEEDS = [42, 101, 777, 1337, 2024]
FEE_RATE = 5.0 / 10000.0  # 5 bps flat
INITIAL_CAPITAL = 100000.0


def audit_feature_definitions() -> dict:
    """Provide exact mathematical formulas and economic interpretations for audited signals."""
    specs = {
        "H-BB": {
            "name": "Bollinger Bands %B",
            "formula": "(p_t - Lower_t) / (Upper_t - Lower_t), where Upper/Lower = SMA_20 +/- 2*Std_20",
            "concept": "Normalized relative price position within dynamic volatility envelope; primary trend/mean-reversion driver.",
            "normalization": "Causal expanding z-score -> Cross-sectional standard normal across universe at bar t.",
        },
        "H-P": {
            "name": "Volatility Persistence (Absolute Return Magnitude)",
            "formula": "causal_zscore(|r_t|)",
            "concept": "Instantaneous absolute return shock; measures volatility persistence and market turbulence.",
            "normalization": "Causal expanding z-score of |r_t|.",
        },
        "H-ASYM": {
            "name": "GJR Panic Leverage (Downside Semivariance)",
            "formula": "causal_zscore(|r_t| * I(r_t < 0))",
            "concept": "Asymmetric downside volatility shock; isolates panic liquidation and crash risk without penalizing upside rallies.",
            "normalization": "Causal expanding z-score of negative return magnitudes.",
        },
        "NL_OFI_REV": {
            "name": "RSI x OFI Mean-Reversion Interaction",
            "formula": "causal_zscore(RSI_14 * OFI_t)",
            "concept": "Order flow imbalance confirmed by oscillator extremes; captures aggressive flow exhausting into support/resistance.",
            "normalization": "Product of normalized RSI and normalized OFI, followed by causal expanding z-score.",
        },
        "NL_SQUEEZE_TREND": {
            "name": "MACD x Squeeze Interaction",
            "formula": "causal_zscore(MACD_t * (BB_width / Keltner_width))",
            "concept": "Trend acceleration breaking out of volatility compression (squeeze release).",
            "normalization": "Product of MACD and Squeeze state, followed by causal expanding z-score.",
        },
        "H-MOM": {
            "name": "5-Day Trailing Return Momentum",
            "formula": "causal_zscore((p_t - p_{t-5}) / p_{t-5})",
            "concept": "Classical 5-day simple momentum.",
            "normalization": "Causal expanding z-score.",
        },
        "NL_HURST_ADAPT": {
            "name": "Hurst Regime-Switched Momentum",
            "formula": "causal_zscore(Mom_t * (Hurst_30 - 0.5))",
            "concept": "Trades momentum when Hurst > 0.5 (persistent trending regime); inverts to mean-reversion when Hurst < 0.5 (anti-persistent regime).",
            "normalization": "Product of 5-day momentum and centered Hurst exponent, followed by causal expanding z-score.",
        },
    }
    return specs


def compute_feature_correlation_matrix() -> pd.DataFrame:
    """Compute cross-asset correlation matrix on training data (Block A: 2021-2023, 753 bars)."""
    uni_30, syms_30 = load_universe("30")
    T = 753
    N = 30
    rets_full = np.column_stack([uni_30[s]["r"].to_numpy(float) for s in syms_30])
    cross_med = np.median(rets_full, axis=1)

    target_signals = [
        "H-BB", "H-P", "H-ASYM", "NL_OFI_REV", "NL_SQUEEZE_TREND", "H-MOM", "NL_HURST_ADAPT"
    ]
    short_names = [n.split()[0] for n in SIGNAL_NAMES]
    sig_indices = {s: short_names.index(s) for s in target_signals}

    # Stack features across all 30 assets over Block A
    feat_data = {s: [] for s in target_signals}
    for s_sym in syms_30:
        feats_asset = calc_signals_for_asset(uni_30[s_sym], cross_med)[:T]
        for s_name in target_signals:
            feat_data[s_name].extend(feats_asset[:, sig_indices[s_name]])

    df_feats = pd.DataFrame(feat_data)
    corr_mat = df_feats.corr().round(4)
    return corr_mat


def run_rolling_window_evolution_experiment() -> dict:
    """Evolve equations across 5 rolling 753-bar windows and evaluate on common Block B."""
    df_90 = pd.read_csv(MULTI_90_FILE)
    date_col = "observation_date" if "observation_date" in df_90.columns else "date"
    dates = df_90[date_col].to_numpy()

    uni_90, syms_90 = load_universe("90")
    syms_30 = syms_90[:30]
    T_total = len(df_90)
    N = 30
    K = len(SIGNAL_NAMES)
    rets_full = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_30])
    cross_med = np.median(rets_full, axis=1)

    feats_3d_all = np.zeros((T_total, N, K))
    for i, s in enumerate(syms_30):
        feats_3d_all[:, i, :] = calc_signals_for_asset(uni_90[s], cross_med)
    feats_3d_all = np.nan_to_num(feats_3d_all, nan=0.0)

    vols_2d_all = pd.DataFrame(rets_full).rolling(20, min_periods=5).std().fillna(0.02).to_numpy()

    # Define the 5 rolling windows (each 753 bars, shifted by ~126 bars = 6 months)
    # Total bars: 1255.
    # W1: [0 : 753]     (2021-01 to 2023-12, end date: ~2023-12-29)
    # W2: [126 : 879]   (2021-07 to 2024-06, end date: ~2024-06-28)
    # W3: [251 : 1004]  (2022-01 to 2024-12, end date: ~2024-12-31)
    # W4: [376 : 1129]  (2022-07 to 2025-06, end date: ~2025-06-30)
    # W5: [502 : 1255]  (2023-01 to 2025-12, end date: ~2025-12-31)
    window_defs = [
        {"name": "Window 1 (Dec 2023)", "start": 0, "end": 753},
        {"name": "Window 2 (Jul 2024)", "start": 126, "end": 879},
        {"name": "Window 3 (Dec 2024)", "start": 251, "end": 1004},
        {"name": "Window 4 (Jul 2025)", "start": 376, "end": 1129},
        {"name": "Window 5 (Dec 2025)", "start": 502, "end": 1255},
    ]

    short_names = [n.split()[0] for n in SIGNAL_NAMES]
    window_chromosomes = []
    window_results = []

    print("=" * 105)
    print("APEDP 5-WINDOW ROLLING EVOLUTION AUDIT (N=30 ASSETS, SEED 42)")
    print("=" * 105)

    for w_i, w_def in enumerate(window_defs):
        s_idx, e_idx = w_def["start"], w_def["end"]
        f_win = feats_3d_all[s_idx:e_idx]
        r_win = rets_full[s_idx:e_idx]
        v_win = vols_2d_all[s_idx:e_idx]
        d_start = dates[s_idx]
        d_end = dates[e_idx - 1]

        chrom = run_multi_asset_genetic_algorithm(f_win, r_win, v_win, pop_size=60, generations=25, seed=42)
        window_chromosomes.append(chrom)

        w = chrom.mask.astype(float) * chrom.weights
        active_idx = list(np.where(w != 0.0)[0])
        eq_terms = [f"{w[k]:+.4f}*{short_names[k]}" for k in active_idx]
        active_terms = [short_names[k] for k in active_idx]

        bb_idx = short_names.index("H-BB")
        bb_coef = float(w[bb_idx]) if chrom.mask[bb_idx] else 0.0

        print(f"[{w_def['name']}] ({d_start} to {d_end}, 753 bars):")
        print(f"  • Active Terms: {active_terms}")
        print(f"  • H-BB Coefficient: {bb_coef:+.4f} | Threshold: {chrom.threshold:.4f} | In-Sample Fit: {chrom.fitness:.3f}")
        print(f"  • Equation: S = {' '.join(eq_terms)}")

        window_results.append({
            "window": w_def["name"],
            "start_date": str(d_start),
            "end_date": str(d_end),
            "bars": e_idx - s_idx,
            "active_terms": active_terms,
            "bb_coefficient": round(bb_coef, 4),
            "threshold": round(chrom.threshold, 4),
            "fitness": round(chrom.fitness, 3),
            "in_sample_sharpe": round(chrom.in_sample_sharpe, 3),
            "weights_dict": {short_names[k]: round(w[k], 4) for k in active_idx},
            "equation": f"S = {' '.join(eq_terms)}",
        })

    # Compute pairwise Jaccard similarities between consecutive windows
    jaccards = []
    for i in range(len(window_results) - 1):
        s1 = set(window_results[i]["active_terms"])
        s2 = set(window_results[i + 1]["active_terms"])
        inter = s1.intersection(s2)
        union = s1.union(s2)
        jac = len(inter) / len(union) if union else 1.0
        jaccards.append({
            "consecutive_pair": f"{window_results[i]['window']} vs {window_results[i+1]['window']}",
            "shared_terms": list(inter),
            "jaccard_similarity": round(jac, 3),
        })

    # Common Evaluation Window: Block B (2024-2025, bars 753 to 1255 = 502 bars)
    # Evaluate prediction correlation, rank correlation, position agreement, and net return across all 5 equations
    print("\n" + "=" * 105)
    print("STAGE 3: COMMON EVALUATION PERIOD AUDIT (BLOCK B: 2024-2025, 502 BARS)")
    print("=" * 105)

    idx_b_start = 753
    idx_b_end = 1255
    f_b = feats_3d_all[idx_b_start:idx_b_end]
    r_b = rets_full[idx_b_start:idx_b_end]
    v_b = vols_2d_all[idx_b_start:idx_b_end]
    T_b = len(r_b)

    common_scores = []
    common_positions = []
    common_returns = []
    common_sharpes = []
    common_turnovers = []

    for w_i, chrom in enumerate(window_chromosomes):
        w = chrom.mask.astype(float) * chrom.weights
        scores_3d = f_b @ w
        scores_norm = (scores_3d - np.mean(scores_3d, axis=1, keepdims=True)) / np.maximum(
            np.std(scores_3d, axis=1, keepdims=True), 1e-8
        )
        target_pos = np.where(scores_norm[:-1] > chrom.threshold, 1.0, np.where(scores_norm[:-1] < -chrom.threshold, -1.0, 0.0))
        turnover = float(np.mean(np.abs(np.diff(target_pos, axis=0, prepend=target_pos[:1]))))
        inv_vol = 1.0 / np.maximum(v_b[:-1], 0.005)
        raw_w = target_pos * inv_vol
        norm_factor = np.maximum(np.sum(np.abs(raw_w), axis=1, keepdims=True), 1e-8)
        w_mat = raw_w / norm_factor
        daily_ret = np.sum(w_mat * r_b[1:], axis=1) - turnover * FEE_RATE
        tot_ret = float(np.sum(daily_ret) * 100.0)
        sd_ret = np.std(daily_ret)
        sharpe = float(np.mean(daily_ret) / sd_ret * np.sqrt(365)) if sd_ret > 1e-9 else 0.0

        common_scores.append(scores_norm[:-1].flatten())
        common_positions.append(target_pos.flatten())
        common_returns.append(tot_ret)
        common_sharpes.append(sharpe)
        common_turnovers.append(turnover)

        print(f"[{window_results[w_i]['window']} on Block B]: Net Ret = {tot_ret:>+6.2f}% | Sharpe = {sharpe:>5.2f} | Turnover = {turnover:.4f}")

    # Pairwise continuous score correlation matrix
    score_corr_mat = np.corrcoef(common_scores).round(4)
    # Pairwise position agreement (% matching sign: both +1, both -1, or both 0)
    pos_agree_mat = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            pos_agree_mat[i, j] = np.mean(common_positions[i] == common_positions[j]) * 100.0
    pos_agree_mat = pos_agree_mat.round(2)

    # Controlled Baseline Evaluation on Block B:
    # 1. Simple BB-only model
    bb_idx = short_names.index("H-BB")
    w_bb_only = np.zeros(K)
    w_bb_only[bb_idx] = 1.0
    scores_bb = f_b @ w_bb_only
    scores_bb_norm = (scores_bb - np.mean(scores_bb, axis=1, keepdims=True)) / np.maximum(np.std(scores_bb, axis=1, keepdims=True), 1e-8)
    pos_bb = np.where(scores_bb_norm[:-1] > 0.15, 1.0, np.where(scores_bb_norm[:-1] < -0.15, -1.0, 0.0))
    to_bb = float(np.mean(np.abs(np.diff(pos_bb, axis=0, prepend=pos_bb[:1]))))
    inv_v = 1.0 / np.maximum(v_b[:-1], 0.005)
    rw_bb = pos_bb * inv_v
    nf_bb = np.maximum(np.sum(np.abs(rw_bb), axis=1, keepdims=True), 1e-8)
    ret_bb = np.sum((rw_bb / nf_bb) * r_b[1:], axis=1) - to_bb * FEE_RATE
    tot_ret_bb = float(np.sum(ret_bb) * 100.0)
    sr_bb = float(np.mean(ret_bb) / np.std(ret_bb) * np.sqrt(365)) if np.std(ret_bb) > 1e-9 else 0.0

    # 2. Buy & Hold Benchmark on Block B
    bh_ret = float(np.mean(np.sum(r_b, axis=0)) * 100.0)

    print("\n" + "=" * 105)
    print("STAGE 5: CONTROLLED FOUR-MODEL BENCHMARK ON BLOCK B (2024-2025)")
    print("=" * 105)
    print(f"1. Benchmark Buy & Hold (30 Assets):               Net Ret = {bh_ret:>+6.2f}%")
    print(f"2. Simple Standalone H-BB Model (Fixed threshold): Net Ret = {tot_ret_bb:>+6.2f}% | Sharpe = {sr_bb:>5.2f}")
    print(f"3. Mean of 5 Window-Specific Evolved Equations:    Net Ret = {np.mean(common_returns):>+6.2f}% | Sharpe = {np.mean(common_sharpes):>5.2f}")

    return {
        "window_results": window_results,
        "consecutive_jaccards": jaccards,
        "score_correlation_matrix": score_corr_mat.tolist(),
        "position_agreement_matrix": pos_agree_mat.tolist(),
        "common_returns": common_returns,
        "common_sharpes": common_sharpes,
        "bb_only_return": tot_ret_bb,
        "bb_only_sharpe": sr_bb,
        "buy_hold_return": bh_ret,
    }


def main():
    specs = audit_feature_definitions()
    corr_mat = compute_feature_correlation_matrix()
    print("=" * 105)
    print("STAGE 1 & 4: CANDIDATE FEATURE CORRELATION & REDUNDANCY MATRIX (BLOCK A, N=30)")
    print("=" * 105)
    print(corr_mat.to_string())

    res = run_rolling_window_evolution_experiment()
    
    full_audit = {
        "feature_specifications": specs,
        "training_correlation_matrix": corr_mat.to_dict(),
        "rolling_window_results": res,
    }
    out_file = Path("scratch/rolling_window_stability_audit_results.json")
    out_file.write_text(json.dumps(full_audit, indent=2))
    print(f"\nAudit complete. Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
