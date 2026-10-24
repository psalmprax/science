"""Stage 5: Full 4-Model x 4-Universe Controlled Scoreboard on Block B (2024-2025).

Evaluates:
- Universes: N in [15, 30, 60, 90]
- Evaluation Period: Block B (bars 753 to 1255 = 502 bars, 2024-01-01 to 2025-12-31)
- Fee Rate: 5.0 bps flat turnover fee
- Sizing: Risk parity / Inverse-volatility weighting

Models:
1. Benchmark: Equal-Weighted Buy & Hold
2. Model 1: Standalone H-BB (S = +1.0 * X^*_{H-BB}, theta = 0.15)
3. Model 2: Window-Specific Evolved Equations (Rolling 5-window GA mean)
4. Model 3: Restrained Shared-Feature Model (Persistent core: S = +0.84 * X^*_{H-BB} - 0.25 * X^*_{H-ASYM}, theta = 0.15)
5. Model 4: Full Unconstrained Genetic Algorithm (Window 5 evolved equation and Block A canonical evolved equation)

Outputs artifact:
scratch/stage5_four_model_four_universe_scoreboard.json
"""
from __future__ import annotations

import json
import sys
import time
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
)

UNIVERSES = [15, 30, 60, 90]
FEE_RATE = 5.0 / 10000.0  # 5.0 bps flat
POP_SIZE = 60
GENERATIONS = 25
SEED = 42

WINDOW_DEFS = [
    {"name": "Window 1 (Dec 2023)", "start": 0, "end": 753},
    {"name": "Window 2 (Jul 2024)", "start": 126, "end": 879},
    {"name": "Window 3 (Dec 2024)", "start": 251, "end": 1004},
    {"name": "Window 4 (Jul 2025)", "start": 376, "end": 1129},
    {"name": "Window 5 (Dec 2025)", "start": 502, "end": 1255},
]

IDX_B_START = 753
IDX_B_END = 1255


def simulate_portfolio_metrics(
    scores_raw: np.ndarray,
    threshold: float,
    returns_2d: np.ndarray,
    vols_2d: np.ndarray,
    fee_rate: float = FEE_RATE,
) -> dict[str, float]:
    """Execute standardized portfolio simulation matching audit_rolling_window_stability.py."""
    T_eval, N = returns_2d.shape
    # Cross-sectional standardization per bar
    sc_mean = np.mean(scores_raw, axis=1, keepdims=True)
    sc_std = np.maximum(np.std(scores_raw, axis=1, keepdims=True), 1e-8)
    scores_norm = (scores_raw - sc_mean) / sc_std

    target_pos = np.where(
        scores_norm[:-1] > threshold,
        1.0,
        np.where(scores_norm[:-1] < -threshold, -1.0, 0.0),
    )
    # Daily turnover per asset
    pos_diff = np.diff(target_pos, axis=0, prepend=target_pos[:1])
    daily_to = float(np.mean(np.abs(pos_diff)))
    total_to = float(np.sum(np.abs(pos_diff)))

    inv_vol = 1.0 / np.maximum(vols_2d[:-1], 0.005)
    raw_w = target_pos * inv_vol
    norm_factor = np.maximum(np.sum(np.abs(raw_w), axis=1, keepdims=True), 1e-8)
    w_mat = raw_w / norm_factor

    gross_daily = np.sum(w_mat * returns_2d[1:], axis=1)
    fee_daily = daily_to * fee_rate
    net_daily = gross_daily - fee_daily

    net_ret = float(np.sum(net_daily) * 100.0)
    gross_ret = float(np.sum(gross_daily) * 100.0)

    sd_net = np.std(net_daily)
    sharpe = float(np.mean(net_daily) / sd_net * np.sqrt(365)) if sd_net > 1e-9 else 0.0

    cum_net = np.cumsum(net_daily)
    peak = np.maximum.accumulate(cum_net)
    max_dd = float(np.max(peak - cum_net) * 100.0)
    exposure = float(np.mean(np.abs(target_pos)) * 100.0)

    return {
        "net_return": round(net_ret, 2),
        "gross_return": round(gross_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "daily_turnover": round(daily_to, 4),
        "total_turnover": round(total_to, 1),
        "exposure": round(exposure, 1),
    }


def compute_buy_hold_metrics(returns_2d: np.ndarray) -> dict[str, float]:
    """Compute equal-weighted buy-and-hold metrics across assets in universe."""
    bh_daily = np.mean(returns_2d[1:], axis=1)
    tot_ret = float(np.mean(np.sum(returns_2d, axis=0)) * 100.0)
    sd = np.std(bh_daily)
    sharpe = float(np.mean(bh_daily) / sd * np.sqrt(365)) if sd > 1e-9 else 0.0
    cum = np.cumsum(bh_daily)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum) * 100.0)
    return {
        "net_return": round(tot_ret, 2),
        "gross_return": round(tot_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "daily_turnover": 0.0,
        "total_turnover": 0.0,
        "exposure": 100.0,
    }


def run_stage5_scoreboard():
    start_time = time.time()
    print("=" * 115)
    print("APEDP STAGE 5: FULL 4-MODEL x 4-UNIVERSE CONTROLLED SCOREBOARD ON BLOCK B")
    print("Period: 2024-2025 (502 bars) | Fee: 5.0 bps | Risk Parity Inverse-Vol Sizing")
    print("Universes: N in [15, 30, 60, 90]")
    print("=" * 115)

    # 1. Load Data
    df_90 = pd.read_csv(MULTI_90_FILE)
    T_total = len(df_90)
    uni_90, syms_90 = load_universe("90")
    rets_90 = np.column_stack([uni_90[s]["r"].to_numpy(float) for s in syms_90])
    vols_90 = pd.DataFrame(rets_90).rolling(20, min_periods=5).std().fillna(0.02).to_numpy()
    cross_med = np.median(rets_90, axis=1)

    K = len(SIGNAL_NAMES)
    short_names = [n.split()[0] for n in SIGNAL_NAMES]
    bb_idx = short_names.index("H-BB")
    asym_idx = short_names.index("H-ASYM")

    print(f"Precomputing features for 90 assets across {T_total} bars...")
    feats_90 = np.zeros((T_total, 90, K))
    for i in range(90):
        feats_90[:, i, :] = calc_signals_for_asset(uni_90[syms_90[i]], cross_med)
    feats_90 = np.nan_to_num(feats_90, nan=0.0)

    # Slice evaluation window Block B
    f_b_all = feats_90[IDX_B_START:IDX_B_END]
    r_b_all = rets_90[IDX_B_START:IDX_B_END]
    v_b_all = vols_90[IDX_B_START:IDX_B_END]
    T_b = len(r_b_all)
    print(f"Block B evaluation slice: {T_b} bars ({IDX_B_START} to {IDX_B_END})")

    # Canonical Block A evolved weights (from development run seed 42)
    canonical_block_a_models = {
        15: {
            "weights": {9: 1.1948, 11: 0.2227, 16: 0.4501, 23: -0.4839},
            "threshold": 0.1695,
        },
        30: {
            "weights": {3: -0.1792, 9: 0.8179, 22: -0.2768},
            "threshold": 0.1572,
        },
        60: {
            "weights": {3: -0.0545, 7: -0.0042, 9: 0.4323, 16: -0.8522, 17: 0.5357, 20: -1.3680, 23: -0.9019},
            "threshold": 0.0946,
        },
        90: {
            "weights": {4: 0.7185, 5: -0.0925, 9: 0.7004, 11: -0.0685, 17: 0.7700},
            "threshold": 0.1525,
        },
    }

    scoreboard = {}
    detailed_results = {}

    for N in UNIVERSES:
        print("\n" + "-" * 90)
        print(f"EVALUATING UNIVERSE N = {N} ASSETS")
        print("-" * 90)

        f_b = f_b_all[:, :N, :]
        r_b = r_b_all[:, :N]
        v_b = v_b_all[:, :N]

        # ── Benchmark: Equal-Weighted Buy & Hold ────────────────────────────
        bh_metrics = compute_buy_hold_metrics(r_b)
        print(f"  [Benchmark: Buy & Hold] Net Ret: {bh_metrics['net_return']:>+6.2f}% | Sharpe: {bh_metrics['sharpe']:>5.2f} | Max DD: {bh_metrics['max_drawdown']:>5.1f}%")

        # ── Model 1: Standalone H-BB ───────────────────────────────────────
        w_m1 = np.zeros(K)
        w_m1[bb_idx] = 1.0
        scores_m1 = f_b @ w_m1
        m1_metrics = simulate_portfolio_metrics(scores_m1, threshold=0.15, returns_2d=r_b, vols_2d=v_b)
        print(f"  [Model 1: Standalone H-BB] Net Ret: {m1_metrics['net_return']:>+6.2f}% | Sharpe: {m1_metrics['sharpe']:>5.2f} | Max DD: {m1_metrics['max_drawdown']:>5.1f}% | Turnover: {m1_metrics['daily_turnover']:.4f}")

        # ── Model 2: Window-Specific Evolved Equations (Rolling GA) ─────────
        print(f"  Running 5-window rolling genetic evolution for N={N}...")
        rolling_runs = []
        w_returns, w_sharpes, w_dds, w_tos = [], [], [], []

        for w_def in WINDOW_DEFS:
            s_idx, e_idx = w_def["start"], w_def["end"]
            f_win = feats_90[s_idx:e_idx, :N, :]
            r_win = rets_90[s_idx:e_idx, :N]
            v_win = vols_90[s_idx:e_idx, :N]

            chrom = run_multi_asset_genetic_algorithm(
                f_win, r_win, v_win, pop_size=POP_SIZE, generations=GENERATIONS, seed=SEED
            )
            w_vec = chrom.mask.astype(float) * chrom.weights
            scores_win = f_b @ w_vec
            metrics_win = simulate_portfolio_metrics(
                scores_win, threshold=chrom.threshold, returns_2d=r_b, vols_2d=v_b
            )

            active_idx = list(np.where(w_vec != 0.0)[0])
            eq_str = " ".join([f"{w_vec[k]:+.4f}*{short_names[k]}" for k in active_idx])

            rolling_runs.append({
                "window": w_def["name"],
                "equation": f"S = {eq_str}",
                "threshold": round(chrom.threshold, 4),
                "in_sample_fitness": round(chrom.fitness, 3),
                "metrics_on_block_b": metrics_win,
            })
            w_returns.append(metrics_win["net_return"])
            w_sharpes.append(metrics_win["sharpe"])
            w_dds.append(metrics_win["max_drawdown"])
            w_tos.append(metrics_win["daily_turnover"])

        m2_mean_metrics = {
            "net_return": round(float(np.mean(w_returns)), 2),
            "gross_return": round(float(np.mean([r["metrics_on_block_b"]["gross_return"] for r in rolling_runs])), 2),
            "sharpe": round(float(np.mean(w_sharpes)), 2),
            "max_drawdown": round(float(np.mean(w_dds)), 1),
            "daily_turnover": round(float(np.mean(w_tos)), 4),
            "total_turnover": round(float(np.mean([r["metrics_on_block_b"]["total_turnover"] for r in rolling_runs])), 1),
            "exposure": round(float(np.mean([r["metrics_on_block_b"]["exposure"] for r in rolling_runs])), 1),
        }
        print(f"  [Model 2: Window-Specific Mean] Net Ret: {m2_mean_metrics['net_return']:>+6.2f}% | Sharpe: {m2_mean_metrics['sharpe']:>5.2f} | Max DD: {m2_mean_metrics['max_drawdown']:>5.1f}% | Turnover: {m2_mean_metrics['daily_turnover']:.4f}")

        # ── Model 2b: Strict Walk-Forward Out-of-Sample (Zero Temporal Overlap) ──
        # Block B [753:1255] is divided into 4 non-overlapping forward slices:
        # Slice 1: [753:879] (126 bars) traded by Window 1 equation (trained [0:753])
        # Slice 2: [879:1004] (125 bars) traded by Window 2 equation (trained [126:879])
        # Slice 3: [1004:1129] (125 bars) traded by Window 3 equation (trained [251:1004])
        # Slice 4: [1129:1255] (126 bars) traded by Window 4 equation (trained [376:1129])
        wf_slices = [
            (753, 879, 0),
            (879, 1004, 1),
            (1004, 1129, 2),
            (1129, 1255, 3),
        ]
        wf_positions = []
        for s_start, s_end, w_idx in wf_slices:
            w_eq = rolling_runs[w_idx]
            th = w_eq["threshold"]
            w_vec = np.zeros(K)
            eq_terms = w_eq["equation"].replace("S = ", "").split()
            for term in eq_terms:
                c, name = term.split("*")
                w_vec[short_names.index(name)] = float(c)
            f_slice = feats_90[s_start:s_end, :N, :]
            sc_raw = f_slice @ w_vec
            sc_mean = np.mean(sc_raw, axis=1, keepdims=True)
            sc_std = np.maximum(np.std(sc_raw, axis=1, keepdims=True), 1e-8)
            sc_norm = (sc_raw - sc_mean) / sc_std
            pos = np.where(sc_norm > th, 1.0, np.where(sc_norm < -th, -1.0, 0.0))
            wf_positions.append(pos)

        pos_wf_all = np.vstack(wf_positions)
        target_pos_wf = pos_wf_all[:-1]
        pos_diff_wf = np.diff(target_pos_wf, axis=0, prepend=target_pos_wf[:1])
        daily_to_wf = float(np.mean(np.abs(pos_diff_wf)))
        total_to_wf = float(np.sum(np.abs(pos_diff_wf)))

        inv_vol_wf = 1.0 / np.maximum(v_b[:-1], 0.005)
        raw_w_wf = target_pos_wf * inv_vol_wf
        norm_factor_wf = np.maximum(np.sum(np.abs(raw_w_wf), axis=1, keepdims=True), 1e-8)
        w_mat_wf = raw_w_wf / norm_factor_wf

        gross_daily_wf = np.sum(w_mat_wf * r_b[1:], axis=1)
        net_daily_wf = gross_daily_wf - daily_to_wf * FEE_RATE

        net_ret_wf = float(np.sum(net_daily_wf) * 100.0)
        gross_ret_wf = float(np.sum(gross_daily_wf) * 100.0)
        sd_net_wf = np.std(net_daily_wf)
        sharpe_wf = float(np.mean(net_daily_wf) / sd_net_wf * np.sqrt(365)) if sd_net_wf > 1e-9 else 0.0
        cum_net_wf = np.cumsum(net_daily_wf)
        max_dd_wf = float(np.max(np.maximum.accumulate(cum_net_wf) - cum_net_wf) * 100.0)
        exposure_wf = float(np.mean(np.abs(target_pos_wf)) * 100.0)

        m2_wf_metrics = {
            "net_return": round(net_ret_wf, 2),
            "gross_return": round(gross_ret_wf, 2),
            "sharpe": round(sharpe_wf, 2),
            "max_drawdown": round(max_dd_wf, 1),
            "daily_turnover": round(daily_to_wf, 4),
            "total_turnover": round(total_to_wf, 1),
            "exposure": round(exposure_wf, 1),
        }
        print(f"  [Model 2b: Strict Walk-Forward OOS] Net Ret: {m2_wf_metrics['net_return']:>+6.2f}% | Sharpe: {m2_wf_metrics['sharpe']:>5.2f} | Max DD: {m2_wf_metrics['max_drawdown']:>5.1f}% | Turnover: {m2_wf_metrics['daily_turnover']:.4f}")

        # ── Model 3: Restrained Shared-Feature Model ───────────────────────
        w_m3 = np.zeros(K)
        w_m3[bb_idx] = 0.84
        w_m3[asym_idx] = -0.25
        scores_m3 = f_b @ w_m3
        m3_metrics = simulate_portfolio_metrics(scores_m3, threshold=0.15, returns_2d=r_b, vols_2d=v_b)
        print(f"  [Model 3: Restrained Shared] Net Ret: {m3_metrics['net_return']:>+6.2f}% | Sharpe: {m3_metrics['sharpe']:>5.2f} | Max DD: {m3_metrics['max_drawdown']:>5.1f}% | Turnover: {m3_metrics['daily_turnover']:.4f}")

        # ── Model 4: Full Unconstrained GA ────────────────────────────────
        # 4a: Window 5 Evolved Model (Most recent rolling window: [502:1255] - NOTE: In-sample overlap with Block B [753:1255])
        m4_win5_metrics = rolling_runs[4]["metrics_on_block_b"]
        m4_win5_eq = rolling_runs[4]["equation"]
        # 4b: Canonical Block A Development Model (Seed 42: [0:753] - NOTE: Strict 0% overlap with Block B [753:1255])
        w_m4_canon = np.zeros(K)
        for k_idx, val in canonical_block_a_models[N]["weights"].items():
            w_m4_canon[k_idx] = val
        th_canon = canonical_block_a_models[N]["threshold"]
        scores_m4_canon = f_b @ w_m4_canon
        m4_canon_metrics = simulate_portfolio_metrics(scores_m4_canon, threshold=th_canon, returns_2d=r_b, vols_2d=v_b)

        print(f"  [Model 4a: In-Sample Overlap Win-5] Net Ret: {m4_win5_metrics['net_return']:>+6.2f}% | Sharpe: {m4_win5_metrics['sharpe']:>5.2f} | Max DD: {m4_win5_metrics['max_drawdown']:>5.1f}% | Turnover: {m4_win5_metrics['daily_turnover']:.4f}")
        print(f"  [Model 4b: Strict Frozen OOS Block-A] Net Ret: {m4_canon_metrics['net_return']:>+6.2f}% | Sharpe: {m4_canon_metrics['sharpe']:>5.2f} | Max DD: {m4_canon_metrics['max_drawdown']:>5.1f}% | Turnover: {m4_canon_metrics['daily_turnover']:.4f}")

        scoreboard[str(N)] = {
            "universe": f"N={N}",
            "buy_and_hold": bh_metrics,
            "model_1_standalone_hbb": m1_metrics,
            "model_2_window_specific_mean": m2_mean_metrics,
            "model_2b_strict_walk_forward_oos": m2_wf_metrics,
            "model_3_restrained_shared": m3_metrics,
            "model_4a_unconstrained_win5_insample_overlap": m4_win5_metrics,
            "model_4b_canonical_block_a_frozen_oos": m4_canon_metrics,
        }

        detailed_results[str(N)] = {
            "rolling_windows": rolling_runs,
            "strict_walk_forward_oos": m2_wf_metrics,
            "canonical_block_a_equation": {
                "weights": {short_names[k]: v for k, v in canonical_block_a_models[N]["weights"].items()},
                "threshold": th_canon,
            },
        }

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds.")

    # Build final artifact payload
    artifact_payload = {
        "metadata": {
            "description": "APEDP Stage 5: Full 4-Model x 4-Universe Controlled Scoreboard on Block B (2024-2025)",
            "evaluation_period": "2024-01-01 to 2025-12-31 (502 bars)",
            "fee_rate_bps": 5.0,
            "universes": UNIVERSES,
            "models_tested": [
                "Benchmark: Equal-Weighted Buy & Hold",
                "Model 1: Standalone H-BB (S = +1.0 * X^*_{H-BB}, theta = 0.15)",
                "Model 2a: Window-Specific Evolved Mean (Descriptive average across 5 windows on Block B)",
                "Model 2b: Strict Walk-Forward OOS (Adaptive forward trading with zero temporal overlap)",
                "Model 3: Restrained Shared-Feature Model (S = +0.84 * X^*_{H-BB} - 0.25 * X^*_{H-ASYM}, theta = 0.15)",
                "Model 4a: Unconstrained Win-5 (In-Sample Overlap: trained on [502:1255] containing Block B)",
                "Model 4b: Canonical Block-A Frozen (Strict OOS: trained on [0:753], zero Block B overlap)",
            ],
            "chronological_audit_notes": {
                "model_4a_contamination_warning": "Model 4a was evolved on Window 5 (2023-01 to 2025-12). Its training set fully overlaps Block B (2024-01 to 2025-12). Its reported return reflects in-sample fit, NOT independent forward alpha.",
                "model_4b_provenance": "Model 4b was frozen on Block A (2021-01 to 2023-12). It has 0 bars of temporal intersection with Block B, serving as a genuine frozen forward out-of-sample benchmark.",
                "model_2b_provenance": "Model 2b implements true walk-forward execution where each 6-month slice of Block B is traded exclusively by the model trained prior to that slice, ensuring 0% lookahead or overlap.",
            },
            "accounting_conventions": {
                "calendar": "502 daily bars across 2024-01-02 to 2025-12-31 (US equity calendar, NYSE holiday exclusion, weekdays only).",
                "turnover_accounting": "daily_to = mean daily absolute target position change per asset (mean(|diff(target_pos)|)). Subtracted daily as daily_to * fee_rate.",
                "fee_rate": "Flat 5.0 bps (0.0005) one-way per unit of turnover.",
                "sharpe_annualization": "mean(daily_net) / std(daily_net) * sqrt(365).",
                "max_drawdown": "Peak-to-trough decline of cumulative net arithmetic equity curve: max(peak - cum_net) * 100%.",
            },
            "execution_time_seconds": round(elapsed, 2),
        },
        "scoreboard": scoreboard,
        "detailed_universe_breakdowns": detailed_results,
    }

    out_path = Path("scratch/stage5_four_model_four_universe_scoreboard.json")
    out_path.write_text(json.dumps(artifact_payload, indent=2))
    print(f"\nArtifact successfully saved to: {out_path.resolve()}")
    return artifact_payload


if __name__ == "__main__":
    run_stage5_scoreboard()

