#!/usr/bin/env python3
"""Walk the sizing rule forward over the repository's real market panels.

This is a mechanical check, not a strategy validation.  It asks whether
``size_positions`` keeps its own contract when the forecast and covariance are
estimated from real prices instead of the synthetic audit grid:

* every weight stays long-only and inside the gross cap,
* the reported KKT residual meets the solver tolerance on every rebalance,
* turnover and fees are charged as the rule says they are,
* the book is held out of sample (the forecast window ends before the return
  window begins).

The panels are ``data/market/equity_top15.csv`` (2021-2023 US equities) and
``data/market/crypto_top15.csv`` (2021-2023 Binance spot).  Both files carry
provenance in ``data/market/PROVENANCE.json``.  The benchmark column (SPY or
BTC) is excluded from the investable universe and used only as a reference.

The forecast is the trailing mean daily return over ``--lookback`` days and the
covariance is the sample covariance of the same window.  Rebalancing happens
every ``--rebalance`` days.  Each rebalance pays ``fee_bps`` per unit of L1
turnover, so the net return is ``w @ held_returns - cost``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.sandbox.finance_l2_demo import size_positions  # noqa: E402

MARKET_DIR = ROOT / "data" / "market"
DEFAULT_OUT = ROOT / "benchmarks" / "sizing_real_market.json"

PANELS = {
    "equity_top15": {
        "file": "equity_top15.csv",
        "assets": [
            "AAPL_CLOSE", "MSFT_CLOSE", "NVDA_CLOSE", "AMZN_CLOSE",
            "GOOGL_CLOSE", "META_CLOSE", "TSLA_CLOSE", "JPM_CLOSE",
            "V_CLOSE", "JNJ_CLOSE", "WMT_CLOSE", "PG_CLOSE",
            "XOM_CLOSE", "UNH_CLOSE",
        ],
        "benchmark": "SPY_CLOSE",
        "periods_per_year": 252,
    },
    "crypto_top15": {
        "file": "crypto_top15.csv",
        "assets": [
            "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "DOT",
            "LINK", "MATIC", "NEAR", "ATOM", "UNI", "LTC",
        ],
        "benchmark": "BTC",
        "periods_per_year": 365,
    },
}


def _load_panel(
    path: Path, assets: list[str], benchmark: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    dates: list[str] = []
    prices: list[list[float]] = []
    benchmarks: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values = [float(row[name]) for name in assets]
            reference = float(row[benchmark])
            if min(values) <= 0.0 or not all(
                math.isfinite(value) for value in [*values, reference]
            ):
                continue
            dates.append(row["observation_date"])
            prices.append(values)
            benchmarks.append(reference)
    return (
        dates,
        np.asarray(prices, dtype=float),
        np.asarray(benchmarks, dtype=float),
    )


def _walk_forward(
    prices: np.ndarray,
    dates: list[str],
    *,
    lookback: int,
    rebalance: int,
    fee_bps: float,
    risk_aversion: float,
    limit: int,
) -> list[dict[str, object]]:
    returns = prices[1:] / prices[:-1] - 1.0
    previous = np.zeros(prices.shape[1])
    records: list[dict[str, object]] = []
    index = lookback
    while index + rebalance <= returns.shape[0]:
        window = returns[index - lookback:index]
        forecast = window.mean(axis=0)
        covariance = np.cov(window, rowvar=False)
        decision = size_positions(
            forecast,
            covariance,
            fee_bps=fee_bps,
            risk_aversion=risk_aversion,
            previous_weights=previous,
        )
        weights = np.asarray(decision.weights, dtype=float)
        held = float(weights @ returns[index:index + rebalance].sum(axis=0))
        records.append(
            {
                "date": dates[index + 1],
                "gross": float(weights.sum()),
                "turnover": float(decision.turnover),
                "cost": float(decision.cost),
                "realized": held,
                "net": held - float(decision.cost),
                "iterations": int(decision.iterations),
                "converged": bool(decision.converged),
                "kkt_residual": float(decision.kkt_residual or 0.0),
                "cap_binding": bool(decision.cap_binding),
                "condition_number": min(float(np.linalg.cond(covariance)), 1e300),
                "non_zero": int(np.count_nonzero(weights > 1e-12)),
                "weights": [float(weight) for weight in weights],
            }
        )
        previous = weights
        index += rebalance
        if limit and len(records) >= limit:
            break
    return records


def _equal_weight(
    prices: np.ndarray, dates: list[str], *, start: int, rebalance: int,
    fee_bps: float, limit: int,
) -> list[dict[str, object]]:
    returns = prices[1:] / prices[:-1] - 1.0
    count = prices.shape[1]
    weights = np.full(count, 1.0 / count)
    previous = np.zeros(count)
    cost_rate = fee_bps / 10_000.0
    records: list[dict[str, object]] = []
    index = start
    while index + rebalance <= returns.shape[0]:
        turnover = float(np.abs(weights - previous).sum())
        held = float(weights @ returns[index:index + rebalance].sum(axis=0))
        records.append(
            {
                "date": dates[index + 1],
                "gross": 1.0,
                "turnover": turnover,
                "cost": cost_rate * turnover,
                "realized": held,
                "net": held - cost_rate * turnover,
            }
        )
        previous = weights
        index += rebalance
        if limit and len(records) >= limit:
            break
    return records


def _metrics(
    records: list[dict[str, object]], periods_per_year: float, rebalance: int,
) -> dict[str, object]:
    net = np.asarray([record["net"] for record in records], dtype=float)
    count = net.size
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    periods = periods_per_year / rebalance
    annual_return = (
        float(equity[-1] ** (periods / count) - 1.0)
        if count and equity[-1] > 0.0 else -1.0
    )
    volatility = float(net.std(ddof=1) * math.sqrt(periods)) if count > 1 else 0.0
    sharpe = (
        float(net.mean() / net.std(ddof=1) * math.sqrt(periods))
        if count > 1 and net.std(ddof=1) > 0.0 else 0.0
    )
    summary: dict[str, object] = {
        "periods": count,
        "start_date": records[0]["date"] if records else None,
        "end_date": records[-1]["date"] if records else None,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "total_net_return": float(equity[-1] - 1.0) if count else 0.0,
        "max_drawdown": float((equity / peak - 1.0).min()) if count else 0.0,
        "total_fees": float(sum(float(record["cost"]) for record in records)),
    }
    if "kkt_residual" in records[0]:
        iterations = np.asarray([record["iterations"] for record in records])
        kkt = np.asarray([record["kkt_residual"] for record in records])
        summary.update(
            {
                "avg_gross": float(np.mean([record["gross"] for record in records])),
                "avg_turnover": float(np.mean([record["turnover"] for record in records])),
                "cap_binding_pct": float(
                    100.0 * np.mean([record["cap_binding"] for record in records])
                ),
                "no_trade_pct": float(
                    100.0 * np.mean([record["turnover"] <= 1e-12 for record in records])
                ),
                "unconverged": int(sum(not record["converged"] for record in records)),
                "max_kkt_residual": float(kkt.max()),
                "max_iterations": int(iterations.max()),
                "median_iterations": float(np.median(iterations)),
                "median_condition_number": float(
                    np.median([record["condition_number"] for record in records])
                ),
                "max_condition_number": float(
                    max(record["condition_number"] for record in records)
                ),
            }
        )
    return summary


def _benchmark_metrics(
    prices: np.ndarray, *, periods_per_year: float, start: int, periods: int,
    rebalance: int,
) -> dict[str, object]:
    returns = prices[1:] / prices[:-1] - 1.0
    stop = min(start + periods * rebalance, returns.shape[0])
    net = returns[start:stop]
    equity = np.cumprod(1.0 + net)
    count = net.size
    peak = np.maximum.accumulate(equity)
    annual_return = float(equity[-1] ** (periods_per_year / count) - 1.0)
    volatility = float(net.std(ddof=1) * math.sqrt(periods_per_year))
    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": float(net.mean() / net.std(ddof=1) * math.sqrt(periods_per_year)),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "total_net_return": float(equity[-1] - 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback", type=int, default=126)
    parser.add_argument("--rebalance", type=int, default=5)
    parser.add_argument("--fees", type=float, nargs="+", default=[0.0, 5.0, 20.0])
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="0 runs the full panel")
    parser.add_argument("--panels", nargs="+", default=list(PANELS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    provenance = json.loads((MARKET_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    output: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "lookback": args.lookback,
            "rebalance": args.rebalance,
            "fee_bps": args.fees,
            "risk_aversion": args.risk_aversion,
            "limit": args.limit,
        },
        "panels": {},
    }
    print(
        f"{'panel':<13} {'fee':>5} {'per':>4} {'ann_ret':>8} {'ann_vol':>8} "
        f"{'sharpe':>7} {'maxdd':>7} {'gross':>6} {'turn':>6} {'fees':>6} "
        f"{'no_trade':>8} {'cap%':>5} {'unconv':>6} {'max_kkt':>9} {'max_iter':>8}"
    )
    for panel_name in args.panels:
        config = PANELS[panel_name]
        path = MARKET_DIR / config["file"]
        dates, prices, benchmark = _load_panel(
            path, config["assets"], config["benchmark"],
        )
        panel_output: dict[str, object] = {
            "file": config["file"],
            "universe": config["assets"],
            "benchmark": config["benchmark"],
            "provenance": provenance.get(config["file"], {}),
            "runs": {},
        }
        first_record = args.lookback
        for fee in args.fees:
            records = _walk_forward(
                prices,
                dates,
                lookback=args.lookback,
                rebalance=args.rebalance,
                fee_bps=fee,
                risk_aversion=args.risk_aversion,
                limit=args.limit,
            )
            summary = _metrics(records, config["periods_per_year"], args.rebalance)
            panel_output["runs"][f"{fee:g}"] = {
                "summary": summary,
                "records": records,
            }
            print(
                f"{panel_name:<13} {fee:>5g} {summary['periods']:>4} "
                f"{100 * summary['annual_return']:>7.2f}% "
                f"{100 * summary['annual_volatility']:>7.2f}% "
                f"{summary['sharpe']:>7.2f} "
                f"{100 * summary['max_drawdown']:>6.2f}% "
                f"{summary['avg_gross']:>6.3f} {summary['avg_turnover']:>6.3f} "
                f"{100 * summary['total_fees']:>5.2f}% "
                f"{summary['no_trade_pct']:>7.1f}% "
                f"{summary['cap_binding_pct']:>4.0f}% "
                f"{summary['unconverged']:>6} "
                f"{summary['max_kkt_residual']:>9.2e} "
                f"{summary['max_iterations']:>8}"
            )
        equal = _equal_weight(
            prices,
            dates,
            start=args.lookback,
            rebalance=args.rebalance,
            fee_bps=args.fees[-1],
            limit=args.limit,
        )
        equal_summary = _metrics(equal, config["periods_per_year"], args.rebalance)
        print(
            f"{panel_name:<13} {'hold':>5} {equal_summary['periods']:>4} "
            f"{100 * equal_summary['annual_return']:>7.2f}% "
            f"{100 * equal_summary['annual_volatility']:>7.2f}% "
            f"{equal_summary['sharpe']:>7.2f} "
            f"{100 * equal_summary['max_drawdown']:>6.2f}% "
            f"{1.0:>6.3f} {equal_summary['total_fees']:>6.3f} "
            f"{100 * equal_summary['total_fees']:>5.2f}% "
            f"{'--':>8} {'--':>5} {'--':>6} {'--':>9} {'--':>8}"
        )
        panel_output["equal_weight_hold"] = {
            "fee_bps": args.fees[-1],
            "summary": equal_summary,
        }
        bench = _benchmark_metrics(
            benchmark,
            periods_per_year=config["periods_per_year"],
            start=first_record,
            periods=len(records),
            rebalance=args.rebalance,
        )
        panel_output["benchmark_buy_hold"] = {
            "asset": config["benchmark"],
            **bench,
        }
        print(
            f"{panel_name:<13} {'bench':>5} {'':>4} "
            f"{100 * bench['annual_return']:>7.2f}% "
            f"{100 * bench['annual_volatility']:>7.2f}% "
            f"{bench['sharpe']:>7.2f} "
            f"{100 * bench['max_drawdown']:>6.2f}%"
        )
        output["panels"][panel_name] = panel_output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")


if __name__ == "__main__":
    main()
