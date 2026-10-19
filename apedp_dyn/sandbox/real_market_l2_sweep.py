"""Frozen Real-Data Rolling-Window L2 Open Discovery Sweep.

Scientific Benchmark Protocol:
  - Input Data: Genuine historical records for S&P 500 and Nasdaq 100 (2018-2023).
  - Open Operator Grammar: {+, -, *, /, pow} over physically typed ASTs.
  - No Financial Templates: The engine receives only raw causal features and dimensions.
  - Strict Temporal Causality: Any feature for predicting return r_t depends exclusively
    on price/return history up to t-1. Zero future lookahead.
  - Multi-Regime Rolling Windows:
      1. 2018H1 -> 2018H2 (Pre-trade-war / volatility shock)
      2. 2019H1 -> 2019H2 (Calm bull market)
      3. 2019 Full -> 2020H1 (COVID crash regime break)
      4. 2020 Full -> 2021H1 (Monetary rebound)
      5. 2021 Full -> 2022 Full (Inflation & rate-hike bear market)
      6. 2022 Full -> 2023H1 (Tech recovery)

Evaluates:
  What structures does the open grammar invent, and what does APEDP permit it to claim?
  Decoupled completely from portfolio sizing / execution rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.epistemic.adjudicate import (
    DataOrigin,
)
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market"


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    train_start: str
    train_end: str
    holdout_start: str
    holdout_end: str
    regime_description: str


def generated_rolling_windows(asset: Literal["sp500", "nasdaq100"] = "sp500",
                              train_days: int = 126,
                              holdout_days: int = 126,
                              step_days: int = 63) -> tuple[WindowSpec, ...]:
    """Rolling train/holdout windows over the committed series.

    The 6 curated WINDOWS above are the frozen benchmark splits and are
    never altered by this generator. Generated ids carry an ROLL prefix
    so frozen and rolling experiments cannot be confused downstream.
    """
    dates, _ = load_series_with_dates(asset)
    prefix = "SPX" if asset == "sp500" else "NDX"
    out: list[WindowSpec] = []
    i = 0
    k = 0
    while i + train_days + holdout_days <= len(dates):
        t0, t1 = dates[i], dates[i + train_days - 1]
        h0, h1 = dates[i + train_days], dates[i + train_days + holdout_days - 1]
        out.append(WindowSpec(f"{prefix}-ROLL{k:02d}", t0, t1, h0, h1,
                              f"rolling {t0}..{h1}"))
        k += 1
        i += step_days
    return tuple(out)


WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("W1_2018", "2018-01-02", "2018-06-29", "2018-07-02", "2018-12-31", "2018 Trade-war / Vol expansion"),
    WindowSpec("W2_2019", "2019-01-02", "2019-06-28", "2019-07-01", "2019-12-31", "2019 Calm Bull Market"),
    WindowSpec("W3_COVID", "2019-01-02", "2019-12-31", "2020-01-02", "2020-06-30", "2020 COVID Shock & Rebound"),
    WindowSpec("W4_REBOUND", "2020-01-02", "2020-12-31", "2021-01-04", "2021-06-30", "2021 Post-COVID Expansion"),
    WindowSpec("W5_BEAR", "2021-01-04", "2021-12-31", "2022-01-03", "2022-12-30", "2022 Bear Market (Rate Hikes)"),
    WindowSpec("W6_RECOVERY", "2022-01-03", "2022-12-30", "2023-01-03", "2023-06-30", "2023 Tech Recovery"),
)

PARTITION_COUNT = 2 * len(WINDOWS)


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
    # Search-size accounting: every candidate the engine scored before one was
    # promoted. A 1-of-N promotion is optimistically biased without this.
    trial_multiplicity: dict | None = None


def load_series_with_dates(asset: Literal["sp500", "nasdaq100"]) -> tuple[list[str], np.ndarray]:
    """Load historical dates and close prices."""
    fname = "sp500.csv" if asset == "sp500" else "nasdaq100.csv"
    col = "SP500" if asset == "sp500" else "NASDAQ100"
    fpath = DATA_DIR / fname

    dates, prices = [], []
    with open(fpath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[col])
                if v > 0:
                    dates.append(row["observation_date"])
                    prices.append(v)
            except (ValueError, KeyError):
                continue
    return dates, np.array(prices, dtype=float)


def compute_strictly_causal_features(prices: np.ndarray) -> dict[str, np.ndarray]:
    """Compute features with strict backward causality.

    For prediction of return r_t = ln(P_t / P_{t-1}):
      - r_lag[t]: r_{t-1} = ln(P_{t-1} / P_{t-2})
      - vol[t]: std(r_{t-10}, ..., r_{t-1})
      - ma_dev[t]: (P_{t-1} - mean(P_{t-20}, ..., P_{t-1})) / mean(...)

    Zero lookahead: at timestamp t, only prices up to t-1 are used.
    """
    n = len(prices)
    returns = np.zeros(n)
    returns[1:] = np.diff(np.log(prices))

    # Features aligned with index t:
    r_lag = np.zeros(n)
    vol = np.zeros(n)
    ma_dev = np.zeros(n)

    for t in range(25, n):
        r_lag[t] = returns[t - 1]
        # 10-day causal volatility strictly before t
        past_rets = returns[t - 10 : t]
        v = float(np.std(past_rets))
        vol[t] = max(v, 0.002)

        # 20-day causal moving average strictly before t
        past_px = prices[t - 20 : t]
        m = float(np.mean(past_px))
        ma_dev[t] = (prices[t - 1] - m) / m if m > 0 else 0.0

    return {
        "returns": returns,
        "r_lag": r_lag,
        "vol": vol,
        "ma_dev": ma_dev,
    }


def execute_window_discovery(
    asset: Literal["sp500", "nasdaq100"],
    window: WindowSpec,
    bootstrap: bool = True,
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

    # Concatenate for L2 engine's internal train/holdout partition
    y_combined = np.concatenate([y_tr, y_ho])
    X_combined = {k: np.concatenate([X_tr[k], X_ho[k]]) for k in X_tr}
    train_ratio = float(n_tr) / float(n_tr + n_ho)

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
            max_depth=2,
            max_complexity=5,
            # rollmean/rollstd: causal rolling aggregation (frozen W=40); see
            # market/sweep.py note. Mechanical FIN_TERMS slate untouched:
            # rolling terms are delay-differential (unintegrable ODE RHS).
            operators=("+", "-", "*", "/", "pow", "rollmean", "rollstd"),
        )
        l2_res = engine.discover(
            X=X_combined,
            Y=y_combined,
            dimensions=dim_map,
            target_dimension=Dimension(),
            train_ratio=train_ratio,
            data_origin=DataOrigin.REAL_OBSERVATIONAL,
            bootstrap=bootstrap,
            seed=seed,
        )
        adj = l2_res.adjudication
        top_cand = (
            adj.promoted
            if adj.promoted is not None
            else (l2_res.pareto_candidates[0] if l2_res.pareto_candidates else None)
        )

    expr_str = ""
    complexity = 0
    tr_err = 0.0
    ho_err = 0.0
    fid_margin = None
    rho1 = None
    white_passed = False
    boot_margin = None
    out_of_grammar = True

    if top_cand is not None:
        expr_str = (
            top_cand.provenance.get("canonical_expr")
            or top_cand.candidate.metadata.get("canonical_expr")
            or top_cand.candidate.candidate_id
        )
        complexity = top_cand.candidate.metadata.get("complexity", 1)
        tr_err = float(top_cand.train_loss) if top_cand.train_loss is not None else 0.0
        ho_err = float(top_cand.holdout_rmse) if top_cand.holdout_rmse is not None else 0.0
        g4 = top_cand.provenance.get("gate4_status", {})
        rho1 = g4.get("rho1")
        white_passed = bool(g4.get("is_white", False))
        out_of_grammar = not bool(g4.get("passed", False))
        noise_sigma = g4.get("holdout_noise_sigma", 0.01)
        if noise_sigma and noise_sigma > 0:
            fid_margin = float(ho_err / (3.0 * (noise_sigma / 1.8)))

    if adj.bootstrap_lo is not None and adj.bootstrap_hi is not None:
        boot_margin = float(adj.bootstrap_lo)

    # Multiple-testing accounting for the DISCOVERY search: every candidate the
    # engine scored is one attempt, and the promoted candidate is an
    # order statistic over that pool. DSR does not apply (the selection
    # statistic here is in-sample loss, not a Sharpe ratio), so what is recorded
    # is the pool itself plus the promoted candidate's rank within it.
    trial_multiplicity: dict | None = None
    # MultiTermDiscoveryResult does not retain every scored candidate; it
    # reports the evaluated count and the surviving Pareto set, so the pool
    # is reconstructed from the Pareto candidates with the true evaluated
    # count carried in the metadata.
    if multiterm:
        scored_pool = list(m_res.pareto_candidates)
        evaluated_count = int(m_res.all_evaluated_count)
    else:
        scored_pool = list(l2_res.candidates)
        evaluated_count = len(scored_pool)
    if scored_pool:
        ledger = CandidateTrialLedger()
        search_id = f"l2:{asset}:{window.window_id}:{'multiterm' if multiterm else 'single'}:{seed}"
        ranked_pool = sorted(
            scored_pool,
            key=lambda s: (s.train_loss if s.train_loss is not None and np.isfinite(s.train_loss) else float("inf")),
        )
        for rank, sc in enumerate(ranked_pool):
            ledger.record_trial(
                candidate_id=sc.candidate.candidate_id,
                proposer_id="l2_open_discovery",
                formula=str(sc.candidate.metadata.get("canonical_expr") or sc.candidate.candidate_id),
                strategy_archetype="continuous_factor",
                in_sample_sharpe=0.0,
                n_bars=n_tr,
                status="EVALUATED",
                selection_statistic=(
                    float(sc.train_loss)
                    if sc.train_loss is not None and np.isfinite(sc.train_loss)
                    else None
                ),
                metadata={
                    "search_id": search_id,
                    "selection_pool_size": evaluated_count,
                    "selection_rank": rank,
                    "complexity": int(sc.candidate.metadata.get("complexity", 1)),
                },
            )
        promoted_id = (
            adj.promoted.candidate.candidate_id if adj.promoted is not None else None
        )
        promoted_rank = next(
            (r for r, sc in enumerate(ranked_pool) if sc.candidate.candidate_id == promoted_id),
            None,
        )
        trial_multiplicity = {
            "recorded_trials": ledger.trial_count,
            "effective_trials": ledger.effective_trial_count,
            "evaluated_count": evaluated_count,
            "search_id": search_id,
            "promoted_rank": promoted_rank,
            "selection_statistic": "in_sample_train_loss (lower is better; DSR not applicable)",
        }

    # Boundary artifact check: inspect if error is concentrated on first/last 3 points of holdout
    boundary_artifact = False
    if top_cand is not None:
        residuals = top_cand.provenance.get("gate4_status", {}).get("holdout_residuals", [])
        if len(residuals) >= 6:
            edge_err = np.mean(np.abs([residuals[0], residuals[1], residuals[-2], residuals[-1]]))
            core_err = np.mean(np.abs(residuals[2:-2]))
            if core_err > 0 and (edge_err / core_err) > 2.0:
                boundary_artifact = True

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
        whiteness_rho1=rho1,
        whiteness_passed=white_passed,
        bootstrap_margin=boot_margin,
        epistemic_verdict=adj.status.value,
        epistemic_attribution=adj.epistemic_attribution,
        boundary_artifact=boundary_artifact,
        out_of_grammar=out_of_grammar,
        full_reason=adj.reason,
        trial_multiplicity=trial_multiplicity,
    )


def run_full_real_market_l2_sweep() -> list[WindowDiscoveryRecord]:
    """Execute complete 12-window real market L2 open discovery sweep across S&P 500 and Nasdaq 100."""
    records: list[WindowDiscoveryRecord] = []
    for asset in ("sp500", "nasdaq100"):
        for win in WINDOWS:
            rec = execute_window_discovery(asset, win, bootstrap=True)
            records.append(rec)
    return records


def run_partition(
    partition_index: int,
    total_partitions: int,
    output_dir: str | Path | None = None,
) -> list[WindowDiscoveryRecord]:
    """Execute and persist this partition's assigned asset/window discoveries."""
    if total_partitions < 1 or total_partitions > PARTITION_COUNT:
        raise ValueError(f"total_partitions must be between 1 and {PARTITION_COUNT}")
    if not 0 <= partition_index < total_partitions:
        raise ValueError(
            f"partition_index must be in [0, {total_partitions}), got {partition_index}"
        )

    jobs = [(asset, window) for asset in ("sp500", "nasdaq100") for window in WINDOWS]
    assigned_jobs = [
        (asset, window)
        for job_index, (asset, window) in enumerate(jobs)
        if job_index % total_partitions == partition_index
    ]
    records = [
        execute_window_discovery(asset, window, bootstrap=True)
        for asset, window in assigned_jobs
    ]

    results_dir = Path(
        output_dir or Path(os.environ.get("APEDP_STORAGE_DIR", "artifacts")) / "sweep_results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"partition_{partition_index:03d}.json"
    result_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a partition of the real-market L2 sweep.")
    parser.add_argument("--partition-index", type=int)
    parser.add_argument("--total-partitions", type=int)
    args = parser.parse_args(argv)
    if args.partition_index is None and args.total_partitions is None:
        records = run_full_real_market_l2_sweep()
        print(f"Completed {len(records)} rolling window discoveries.")
        for record in records:
            print(
                f"[{record.asset.upper()} {record.window_id}] "
                f"Verdict: {record.epistemic_verdict} ({record.epistemic_attribution}) | "
                f"Discovered: {record.discovered_expression} | Gate4 Fail: {record.out_of_grammar}"
            )
        return 0
    if args.partition_index is None or args.total_partitions is None:
        parser.error("--partition-index and --total-partitions must be provided together")
    records = run_partition(args.partition_index, args.total_partitions)
    print(
        f"Partition {args.partition_index}/{args.total_partitions} completed "
        f"{len(records)} discoveries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
