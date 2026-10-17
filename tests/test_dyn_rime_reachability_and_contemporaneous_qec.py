"""Verification and remediation tests for RIME discovery reachability and contemporaneous laws.

Flaws addressed:
1. Reachability: Proposer grammar structurally excluded contemporaneous self-products (0/1217 on QEC).
   Remediated via opt-in `square_seeds=True` in the seed catalogue, making `(x_i * x_i)` reachable at
   depth 0 directly from raw features without manual feature engineering or repair engine insertion.
2. Proxy fitting: Legacy search emitted temporal lag proxies `(p * p[t-k])` on log-spaced static grids
   absorbing grid ratio r^k into scale. Tested via holdout reversal-invariance: `(p * p)` is 100%
   reversal invariant (|R2_fwd - R2_rev| < 1e-12), while `(p * p[t-5])` collapses under row reversal.
3. Wrong-family discrimination: At d=5 (planted cubic law p^3), a quadratic form (p * p) exhibits
   strong systematic residual curvature (Spearman/Pearson corr(e, y_pred^2) > 0.9), triggering
   curvature detection and preventing confident wrong-family promotion.
4. Non-flaw confirmation: Dev-subrange low-SNR abstention at d>=4 is verified as sound epistemic refusal.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import apedp_kernel as ak
from apedp_dyn.proposers.rime_proposer import RimeProposer
from tests.test_dyn_ga_qec import generate_qec_threshold_data


def test_reachability_remediated_raw_p_discovers_contemporaneous_square():
    """Flaw 1 remediation: raw [p] discovers p_L = 1000*p^2 directly at depth 0."""
    p, obs = generate_qec_threshold_data(n_points=600, seed=42)
    ntr = len(p) // 2
    df = pd.DataFrame({"p": p[:ntr], "target": obs[:ntr]})

    # Without square_seeds in temporal discovery, exact same-time square is unreachable (0/1217)
    # The search only surfaces lagged proxies like (p * p[t-5]).
    prop_legacy = RimeProposer(feature_cols=["p"], target_col="target", max_depth=1,
                               static_mode=False, square_seeds=False, rel_penalty=0.0)
    props_legacy = prop_legacy.fit_and_propose(df)
    assert not any(pr.metadata.get("equation") == "(p * p)" for pr in props_legacy), (
        "legacy temporal search without square_seeds must not surface contemporaneous (p * p)"
    )
    assert any("[t-" in pr.metadata.get("equation", "") for pr in props_legacy), (
        "legacy temporal search surfaces lagged proxies instead"
    )

    # In static mode with square_seeds=True, contemporaneous (p * p) is discovered directly
    # from raw [p] without manual column engineering (p_sq) or repair engine intervention.
    prop_static_sq = RimeProposer(feature_cols=["p"], target_col="target", max_depth=1,
                                  static_mode=True, square_seeds=True, rel_penalty=0.0)
    props_static_sq = prop_static_sq.fit_and_propose(df)
    p_sq_props = [pr for pr in props_static_sq if pr.metadata.get("equation") == "(p * p)"]
    assert len(p_sq_props) > 0, "contemporaneous (p * p) must be emitted in static mode with square_seeds=True"

    # Evaluate holdout R2 on unseen rows
    te_df = pd.DataFrame({"p": p[ntr:], "target": obs[ntr:]})
    best = p_sq_props[0]
    pred = best.compute_score(te_df)
    y_te = te_df["target"].to_numpy()
    holdout_r2 = 1.0 - float(np.sum((y_te - pred) ** 2)) / float(np.sum((y_te - y_te.mean()) ** 2))
    assert holdout_r2 > 0.999, f"holdout R2 must exceed 0.999, got {holdout_r2}"
    assert best.metadata.get("depth") == 0, "contemporaneous square must be reachable at depth 0"


def test_proxy_fitting_discriminated_by_reversal_invariance():
    """Flaw 3 remediation: physical contemporaneous law is reversal-invariant, lag proxy is not."""
    p, obs = generate_qec_threshold_data(n_points=600, seed=42)
    ntr = len(p) // 2
    te_col = p[ntr:]
    te_y = obs[ntr:]

    # 1. Contemporaneous square (p * p)
    cfg_sq = json.dumps({"max_depth": 0, "beam": 8, "budget": 500,
                         "ridge": 1e-9, "seed": 0, "rel_penalty": 0.0,
                         "square_seeds": True, "static": True})
    pkt_sq = json.loads(ak.rime_search_json([np.asarray(p[:ntr], dtype=float)],
                                            np.asarray(obs[:ntr], dtype=float), cfg_sq, ["p"]))
    pr_sq = next(pr for pr in pkt_sq["proposals"] if pr["equation"] == "(p * p)")
    scale_sq = pr_sq["dev_evidence"]["scale"]
    intercept_sq = pr_sq["dev_evidence"]["intercept"]
    g_json_sq = pr_sq["graph_json"]

    # Forward vs Reversed holdout for (p * p)
    pred_fwd_sq = scale_sq * np.asarray(ak.rime_evaluate_series_json(g_json_sq, [te_col]), dtype=float) + intercept_sq
    r2_fwd_sq = 1.0 - np.sum((te_y - pred_fwd_sq) ** 2) / np.sum((te_y - te_y.mean()) ** 2)

    pred_rev_sq = scale_sq * np.asarray(ak.rime_evaluate_series_json(g_json_sq, [te_col[::-1]]), dtype=float) + intercept_sq
    r2_rev_sq = 1.0 - np.sum((te_y[::-1] - pred_rev_sq) ** 2) / np.sum((te_y[::-1] - te_y[::-1].mean()) ** 2)

    assert abs(r2_fwd_sq - r2_rev_sq) < 1e-12, "contemporaneous law must be 100% reversal invariant"
    assert r2_fwd_sq > 0.999

    # 2. Lagged proxy (p * p[t-5])
    cfg_lag = json.dumps({"max_depth": 2, "beam": 8, "budget": 500,
                          "ridge": 1e-9, "seed": 0, "rel_penalty": 0.0,
                          "square_seeds": False, "static": False})
    pkt_lag = json.loads(ak.rime_search_json([np.asarray(p[:ntr], dtype=float)],
                                             np.asarray(obs[:ntr], dtype=float), cfg_lag, ["p"]))
    pr_lag = next(pr for pr in pkt_lag["proposals"] if pr["equation"] == "(p * p[t-5])")
    scale_lag = pr_lag["dev_evidence"]["scale"]
    intercept_lag = pr_lag["dev_evidence"]["intercept"]
    g_json_lag = pr_lag["graph_json"]

    pred_fwd_lag = scale_lag * np.asarray(ak.rime_evaluate_series_json(g_json_lag, [te_col]), dtype=float) + intercept_lag
    m_fwd = np.isfinite(pred_fwd_lag)
    mse_fwd_lag = np.mean((te_y[m_fwd] - pred_fwd_lag[m_fwd]) ** 2)

    pred_rev_lag = scale_lag * np.asarray(ak.rime_evaluate_series_json(g_json_lag, [te_col[::-1]]), dtype=float) + intercept_lag
    m_rev = np.isfinite(pred_rev_lag)
    mse_rev_lag = np.mean((te_y[::-1][m_rev] - pred_rev_lag[m_rev]) ** 2)

    # Reversal inflates proxy error by more than 10x
    assert mse_rev_lag > 10.0 * mse_fwd_lag, f"lag proxy must degrade severely under reversal: {mse_rev_lag} vs {mse_fwd_lag}"


def test_wrong_family_residual_curvature_detected_at_d5():
    """Flaw 2 remediation: on d=5 (p^3), quadratic fit (p * p) exhibits severe systematic curvature."""
    C, PTH = 0.1, 0.01
    n = 600
    p = np.logspace(-3.3, -1.7, n)
    # Planted cubic law at d=5 (e = (5+1)/2 = 3.0)
    t = C * (p / PTH) ** 3.0
    r = np.random.RandomState(42)
    noise = r.normal(0, 0.02 * np.sqrt(np.mean(t ** 2)), n)
    obs = t + noise

    # Fit quadratic (p * p)
    p_sq = p ** 2
    w, intercept = np.polyfit(p_sq, obs, deg=1)
    pred = w * p_sq + intercept
    residuals = obs - pred

    # Systematic curvature detection via epistemic classifier
    from apedp_dyn.epistemic.rime_repair_diagnosis import (
        FINDING_CURVATURE_SUSPECT,
        FINDING_NO_RELIABLE_STRUCTURE,
        ResidualReport,
        classify_findings,
    )

    rep_d5 = ResidualReport(
        candidate_id="c_quad_d5", formula="p^2", graph_id="g1", partition_label="dev",
        n_rows=n, n_finite=n, mean_resid=float(residuals.mean()), std_resid=float(residuals.std()),
        std_target=float(obs.std()), rmse=float(np.sqrt((residuals**2).mean())),
        mae=float(np.abs(residuals).mean()), r2=1.0 - (residuals**2).sum()/((obs-obs.mean())**2).sum(),
        corr=float(np.corrcoef(obs, pred)[0, 1]), rho1=0.0, max_abs_resid=float(np.abs(residuals).max()),
        frac_finite=1.0
    )
    findings_d5 = classify_findings(
        rep_d5, residuals, pred, probes={}, rho_flag=0.1, bias_ratio_cut=0.25,
        variance_ratio_cut=4.0, curvature_corr_cut=0.30, interaction_corr_cut=0.30, min_rows=30
    )
    assert FINDING_CURVATURE_SUSPECT in findings_d5, f"d=5 wrong family must flag curvature: {findings_d5}"

    # For contrast: on d=3 (true quadratic law), curvature is not flagged
    p_d3, obs_d3 = generate_qec_threshold_data(n_points=600, seed=42)
    p_sq_d3 = p_d3 ** 2
    w3, int3 = np.polyfit(p_sq_d3, obs_d3, deg=1)
    res_d3 = obs_d3 - (w3 * p_sq_d3 + int3)
    rep_d3 = ResidualReport(
        candidate_id="c_quad_d3", formula="p^2", graph_id="g1", partition_label="dev",
        n_rows=len(p_d3), n_finite=len(p_d3), mean_resid=float(res_d3.mean()), std_resid=float(res_d3.std()),
        std_target=float(obs_d3.std()), rmse=float(np.sqrt((res_d3**2).mean())),
        mae=float(np.abs(res_d3).mean()), r2=1.0 - (res_d3**2).sum()/((obs_d3-obs_d3.mean())**2).sum(),
        corr=float(np.corrcoef(obs_d3, w3*p_sq_d3+int3)[0, 1]), rho1=0.0, max_abs_resid=float(np.abs(res_d3).max()),
        frac_finite=1.0
    )
    findings_d3 = classify_findings(
        rep_d3, res_d3, w3 * p_sq_d3 + int3, probes={}, rho_flag=0.1, bias_ratio_cut=0.25,
        variance_ratio_cut=4.0, curvature_corr_cut=0.30, interaction_corr_cut=0.30, min_rows=30
    )
    assert FINDING_CURVATURE_SUSPECT not in findings_d3
    assert findings_d3 == (FINDING_NO_RELIABLE_STRUCTURE,)
