"""Real Market Sweeps Engine: scalable open discovery across extensive regimes and windows.

Evaluates:
  What structures does the open grammar invent, and what does APEDP permit it to claim?
  Decoupled completely from portfolio sizing / execution rules.
  Guarantees zero future lookahead and strict temporal causality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.epistemic.adjudicate import DataOrigin
from apedp_dyn.market.assets import compute_strictly_causal_features, load_series_with_dates
from apedp_dyn.market.windows import (
    WindowSpec,
    extensive_rolling_windows,
)


@dataclass(frozen=True)
class WindowDiscoveryRecord:
    window_id: str
    asset: str
    train_period: tuple[str, str]
    holdout_period: tuple[str, str]
    regime_name: str
    n_train_samples: int
    n_holdout_samples: int
    discovered_expression: str
    expression_complexity: int
    train_error: float
    holdout_error: float
    fidelity_margin: float | None
    whiteness_rho1: float | None
    whiteness_passed: bool
    bootstrap_margin: float | None
    epistemic_verdict: str
    epistemic_attribution: str
    boundary_artifact: bool
    out_of_grammar: bool
    full_reason: str


@dataclass(frozen=True)
class MarketSweepSummary:
    asset: str
    total_windows: int
    unique_promotions: int
    multi_hypotheses: int
    insufficient_evidence: int
    no_valid_candidates: int
    false_promotion_rate: float
    regime_breakdown: dict[str, dict[str, int]]
    records: list[WindowDiscoveryRecord]


def execute_window_discovery(
    asset: str,
    window: WindowSpec,
    bootstrap: bool = False,
    seed: int = 42,
    multiterm: bool = False,
) -> WindowDiscoveryRecord:
    """Run L2 open discovery on a specific rolling window under strict causal protocol."""
    dates, prices = load_series_with_dates(asset)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    if window.train_start not in date_to_idx or window.holdout_end not in date_to_idx:
        raise KeyError(f"Window {window.window_id} dates not found in {asset} series.")

    i_tr0 = date_to_idx[window.train_start]
    i_tr1 = date_to_idx[window.train_end]
    i_ho0 = date_to_idx[window.holdout_start]
    i_ho1 = date_to_idx[window.holdout_end]

    causal_data = compute_strictly_causal_features(prices)
    y_all = causal_data["returns"]
    feat_rlag = causal_data["r_lag"]
    feat_vol = causal_data["vol"]
    feat_madev = causal_data["ma_dev"]

    # Slice train and holdout segments
    tr_slice = slice(i_tr0, i_tr1 + 1)
    ho_slice = slice(i_ho0, i_ho1 + 1)

    y_tr = y_all[tr_slice]
    y_ho = y_all[ho_slice]

    X_tr = {
        "r_lag": feat_rlag[tr_slice],
        "vol": feat_vol[tr_slice],
        "ma_dev": feat_madev[tr_slice],
    }
    X_ho = {
        "r_lag": feat_rlag[ho_slice],
        "vol": feat_vol[ho_slice],
        "ma_dev": feat_madev[ho_slice],
    }

    n_tr = len(y_tr)
    n_ho = len(y_ho)

    y_combined = np.concatenate([y_tr, y_ho])
    X_combined = {k: np.concatenate([X_tr[k], X_ho[k]]) for k in X_tr}
    train_ratio = float(n_tr) / float(n_tr + n_ho) if (n_tr + n_ho) > 0 else 0.5

    dim_map = {k: Dimension() for k in X_tr}

    if multiterm:
        m_engine = MultiTermL2Engine(
            max_depth=2,
            max_term_complexity=3,
            max_terms=3,
            operators=("+", "-", "*", "/"),
        )
        m_res = m_engine.discover_multiterm(
            X=X_combined,
            Y=y_combined,
            dimensions=dim_map,
            target_dimension=Dimension(),
            train_ratio=train_ratio,
            data_origin=DataOrigin.REAL_OBSERVATIONAL,
            bootstrap=bootstrap,
            seed=seed,
        )
        adj = m_res.adjudication
        top_cand = (
            adj.promoted
            if adj.promoted is not None
            else (m_res.pareto_candidates[0] if m_res.pareto_candidates else None)
        )
    else:
        engine = L2DiscoveryEngine(
            # rollmean/rollstd: causal rolling aggregation (frozen W=40) enables
            # rolling volume/state features; validated via volume-surprise
            # rediscovery (§22). FIN_TERMS-style mechanical slate intentionally
            # untouched: rolling terms are delay-differential, unintegrable as
            # ODE RHS, and repo daily bars carry no volume channel.
            operators=("+", "-", "*", "/", "pow", "rollmean", "rollstd"),
            max_depth=2,
            max_complexity=5,
        )
        s_res = engine.discover(
            X=X_combined,
            Y=y_combined,
            dimensions=dim_map,
            target_dimension=Dimension(),
            train_ratio=train_ratio,
            data_origin=DataOrigin.REAL_OBSERVATIONAL,
            bootstrap=bootstrap,
            seed=seed,
        )
        adj = s_res.adjudication
        top_cand = (
            adj.promoted
            if adj.promoted is not None
            else (s_res.pareto_candidates[0] if s_res.pareto_candidates else None)
        )

    verdict_str = adj.status.value.lower()
    attr_val = getattr(adj, "epistemic_attribution", getattr(adj, "attribution", None))
    attr_str = (attr_val.value.lower() if attr_val is not None and hasattr(attr_val, "value")
                else str(attr_val or "unknown").lower())

    if top_cand is not None:
        expr_str = str(
            top_cand.provenance.get("canonical_expr")
            or getattr(top_cand.candidate, "canonical_equation", None)
            or top_cand.candidate.metadata.get("canonical_expr")
            or top_cand.candidate.candidate_id
        )
        complexity = int(
            getattr(top_cand.candidate, "complexity", top_cand.candidate.metadata.get("complexity", 1))
        )
        tr_err = float(
            getattr(top_cand, "train_error", getattr(top_cand, "train_loss", float("nan")))
        )
        ho_err = float(
            getattr(top_cand, "holdout_error", getattr(top_cand, "holdout_rmse", float("nan")))
        )
        g4 = top_cand.provenance.get("gate4_status", {})
        fid_margin = g4.get("fidelity_margin")
        w_rho1 = g4.get("whiteness_rho1", g4.get("rho1"))
        w_passed = bool(g4.get("whiteness_passed", g4.get("is_white", False)))
        boot_margin = top_cand.provenance.get("bootstrap_margin")
        boundary = bool(top_cand.provenance.get("boundary_artifact", False))
        oog = bool(top_cand.provenance.get("out_of_grammar", False))
    else:
        expr_str = ""
        complexity = 0
        tr_err = float("nan")
        ho_err = float("nan")
        fid_margin = None
        w_rho1 = None
        w_passed = False
        boot_margin = None
        boundary = False
        oog = True

    return WindowDiscoveryRecord(
        window_id=window.window_id,
        asset=asset,
        train_period=(window.train_start, window.train_end),
        holdout_period=(window.holdout_start, window.holdout_end),
        regime_name=window.regime_description,
        n_train_samples=n_tr,
        n_holdout_samples=n_ho,
        discovered_expression=expr_str,
        expression_complexity=complexity,
        train_error=tr_err,
        holdout_error=ho_err,
        fidelity_margin=fid_margin,
        whiteness_rho1=w_rho1,
        whiteness_passed=w_passed,
        bootstrap_margin=boot_margin,
        epistemic_verdict=verdict_str,
        epistemic_attribution=attr_str,
        boundary_artifact=boundary,
        out_of_grammar=oog,
        full_reason=adj.reason,
    )


def run_market_sweep(
    asset: str = "sp500",
    windows: tuple[WindowSpec, ...] | list[WindowSpec] | None = None,
    max_windows: int | None = None,
    multiterm: bool = False,
    bootstrap: bool = False,
) -> MarketSweepSummary:
    """Execute a batch open discovery sweep across specified or extensive market windows."""
    if windows is None:
        windows = extensive_rolling_windows(asset)

    if max_windows is not None:
        windows = tuple(windows[:max_windows])

    records: list[WindowDiscoveryRecord] = []
    unique_cnt = 0
    multi_cnt = 0
    insuff_cnt = 0
    novalid_cnt = 0

    regime_breakdown: dict[str, dict[str, int]] = {}

    for w in windows:
        rec = execute_window_discovery(
            asset=asset,
            window=w,
            bootstrap=bootstrap,
            multiterm=multiterm,
        )
        records.append(rec)

        verd = rec.epistemic_verdict
        if verd == "unique_support":
            unique_cnt += 1
        elif verd == "multi_hypothesis":
            multi_cnt += 1
        elif verd == "insufficient_evidence":
            insuff_cnt += 1
        else:
            novalid_cnt += 1

        reg = getattr(w, "regime", "general")
        if reg not in regime_breakdown:
            regime_breakdown[reg] = {
                "total": 0,
                "unique_support": 0,
                "multi_hypothesis": 0,
                "insufficient_evidence": 0,
                "no_valid_candidate": 0,
            }
        regime_breakdown[reg]["total"] += 1
        regime_breakdown[reg][verd] = regime_breakdown[reg].get(verd, 0) + 1

    total = len(records)
    fpr = float(unique_cnt) / float(total) if total > 0 else 0.0

    return MarketSweepSummary(
        asset=asset,
        total_windows=total,
        unique_promotions=unique_cnt,
        multi_hypotheses=multi_cnt,
        insufficient_evidence=insuff_cnt,
        no_valid_candidates=novalid_cnt,
        false_promotion_rate=round(fpr, 4),
        regime_breakdown=regime_breakdown,
        records=records,
    )
