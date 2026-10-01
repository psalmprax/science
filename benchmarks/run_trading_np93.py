"""TRADING-NP93: 93-Window No-Promotion / Persistent-Degeneracy Benchmark.

Evaluates APEDP's epistemic resistance to noise-fitting and observational non-identifiability
on 93 walk-forward rolling windows of genuine historical S&P 500 price data (2018-2023).

Scientific Hypothesis:
  Financial price time series in passive observational windows exhibit structural collinearity
  between autoregressive lag-1 (AR1), short-term moving average (MA3), and medium-term moving
  average (MA5). An honest discovery system must abstain from declaring UNIQUE_SUPPORT,
  returning MULTI_HYPOTHESIS with diagnostic attribution to OBSERVATIONAL_NON_IDENTIFIABILITY,
  yielding a False Promotion Rate of exactly 0.0%.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.epistemic.diagnosis import diagnose
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term
from apedp_dyn.sandbox.market_real_data_benchmark import load_market_series

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class TradingNP93WindowResult:
    """Evaluation record of a single rolling walk-forward window."""

    window_index: int
    start_date: str
    end_date: str
    train_size: int
    val_size: int
    status: str
    promoted: str | None
    tau_epi: float | None
    delta_rmse: float | None
    diagnosis: dict[str, Any]


@dataclass(frozen=True)
class TradingNP93Summary:
    """Summary record of the TRADING-NP93 benchmark."""

    benchmark: str
    full_name: str
    windows: int
    candidate_space: str
    validation: int
    unique: int
    multi: int
    insufficient: int
    promotion: int
    false_promotion_rate: float
    passed_contract: bool
    question: str


def run_trading_np93(
    asset: str = "sp500",
    n_windows: int = 93,
    n_train: int = 30,
    n_val: int = 30,
    bootstrap_n_boot: int = 100,
    output_path: Path | None = None,
) -> tuple[TradingNP93Summary, list[TradingNP93WindowResult]]:
    """Execute the 93 walk-forward rolling windows on historical market prices."""
    dates, prices = load_market_series(asset)  # type: ignore[arg-type]

    win_size = n_train + n_val
    total_len = len(prices)
    if total_len < win_size:
        raise ValueError(f"Insufficient price history: {total_len} < {win_size}")

    step = (total_len - win_size) // (n_windows - 1)

    window_results: list[TradingNP93WindowResult] = []
    unique_count = 0
    multi_count = 0
    insufficient_count = 0
    promotion_count = 0

    for i in range(n_windows):
        start = i * step
        w = prices[start : start + win_size]
        w_dates = dates[start : start + win_size]

        # Target: P_t for t in 5..win_size
        target = w[5:]
        lag1 = w[4:-1]
        ma3 = np.array([np.mean(w[j + 2 : j + 5]) for j in range(len(target))])
        ma5 = np.array([np.mean(w[j : j + 5]) for j in range(len(target))])

        y_tr, y_ho = target[:n_train], target[n_train:]
        obs_rms = float(np.sqrt(np.mean(y_ho**2))) if len(y_ho) > 0 else 1.0

        cands: list[ScoredCandidate] = []
        for cid, feat in [("ar1", lag1), ("ma3", ma3), ("ma5", ma5)]:
            x_tr, x_ho = feat[:n_train], feat[n_train:]
            X_tr = np.column_stack([np.ones_like(x_tr), x_tr])
            X_ho = np.column_stack([np.ones_like(x_ho), x_ho])
            w_fit, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
            pred_ho = X_ho @ w_fit
            resid = y_ho - pred_ho
            rmse = float(np.sqrt(np.mean(resid**2)))
            loss_tr = float(np.mean((y_tr - X_tr @ w_fit) ** 2))

            sc = ScoredCandidate(
                CandidateDynamics("P_t", (Term(cid, "b"),), candidate_id=cid),
                provenance={
                    "data_origin": DataOrigin.REAL_OBSERVATIONAL.value,
                    "benchmark_level": BenchmarkLevel.L0_SELECTION.value,
                    "gate4_status": {
                        "passed": True,
                        "holdout_obs_rms": obs_rms,
                        "holdout_noise_sigma": float(np.std(resid)),
                        "holdout_n_samples": len(y_ho),
                        "rmse": rmse,
                        "holdout_residuals": resid.tolist(),
                        "holdout_sq_error": (resid**2).tolist(),
                    },
                },
                train_loss=loss_tr,
                holdout_rmse=rmse,
            )
            cands.append(sc)

        adj = adjudicate(
            cands,
            bootstrap=True,
            bootstrap_n_boot=bootstrap_n_boot,
            bootstrap_seed=42,
            data_origin=DataOrigin.REAL_OBSERVATIONAL.value,
            benchmark_level=BenchmarkLevel.L0_SELECTION.value,
        )

        diag = diagnose(adj)

        status_val = adj.status.value
        promoted_id = adj.promoted.candidate.candidate_id if adj.promoted else None

        if adj.status == EpistemicStatus.UNIQUE_SUPPORT:
            unique_count += 1
            promotion_count += 1
        elif adj.status == EpistemicStatus.MULTI_HYPOTHESIS:
            multi_count += 1
        else:
            insufficient_count += 1

        window_results.append(
            TradingNP93WindowResult(
                window_index=i,
                start_date=w_dates[0],
                end_date=w_dates[-1],
                train_size=n_train,
                val_size=n_val,
                status=status_val,
                promoted=promoted_id,
                tau_epi=adj.tau_epi,
                delta_rmse=adj.delta_rmse,
                diagnosis=diag,
            )
        )

    false_promotion_rate = promotion_count / n_windows
    passed_contract = (promotion_count == 0) and (unique_count == 0)

    summary = TradingNP93Summary(
        benchmark="TRADING-NP93",
        full_name="No-Promotion / Persistent-Degeneracy Benchmark",
        windows=n_windows,
        candidate_space="3-form",
        validation=n_val,
        unique=unique_count,
        multi=multi_count,
        insufficient=insufficient_count,
        promotion=promotion_count,
        false_promotion_rate=false_promotion_rate,
        passed_contract=passed_contract,
        question="can a constrained discovery system resist promoting structurally ambiguous financial relationships",
    )

    target_json = output_path or (OUTPUTS_DIR / "trading_np93_freeze.json")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_dict = {
        "benchmark": summary.benchmark,
        "full_name": summary.full_name,
        "windows": summary.windows,
        "candidate_space": summary.candidate_space,
        "validation": summary.validation,
        "unique": summary.unique,
        "multi": summary.multi,
        "promotion": summary.promotion,
        "question": summary.question,
    }
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=1)

    return summary, window_results


if __name__ == "__main__":
    print("Executing TRADING-NP93 Benchmark...")
    summ, details = run_trading_np93()
    print(f"Benchmark: {summ.benchmark} ({summ.full_name})")
    print(f"Windows: {summ.windows}, Candidate Space: {summ.candidate_space}")
    print(f"Verdicts: UNIQUE={summ.unique}, MULTI={summ.multi}, INSUFFICIENT={summ.insufficient}")
    print(f"Promotion Count: {summ.promotion}, False Promotion Rate: {summ.false_promotion_rate:.1%}")
    print(f"Passed Honest Contract: {summ.passed_contract}")
