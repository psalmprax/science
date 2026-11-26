#!/usr/bin/env python3
"""Counting independent trials on real prices: the referee's count against the cluster count (development).

Status: audit-only

**Development, not a result, and nothing in force changed.** The owner asked on 26 November to try the
independent-trials count (``apedp_dyn.epistemic.independent_trials``) on real data. A real search: four
families of rule-based strategies with many settings each (momentum over 5 to 250 days, moving-average
crossovers, short-term reversal, channel breakouts; 57 strategies), positions from prices up to day t,
applied to the return of day t + 1. Real daily prices from the repository's ``data/market`` (FRED's S&P 500
and NASDAQ-100, 2018 to 2023; Binance BTC and ETH and Yahoo's SPY and AAPL, 2021 to 2023). The held-out
panel of the finance audits is not read, and no finance audit is run.

On real prices the truth is unknown, so three conditions:

* **no edge**: the returns are shifted in time by a random offset of at least 60 days against the
  positions. The strategies keep their real behaviour and their real correlations with one another, fat
  tails and volatility clustering included, but nothing can predict anything. A judge that calls a
  strategy significant here is wrong; the line is 5 %.
* **planted edge**: the shifted returns plus a small component that one family's average position predicts,
  sized so that the family's strategies earn an annualised Sharpe ratio of about 2.5 or 3.5. Found means
  significant with the selected strategy in that family. Strategies of other families that move with it
  earn part of the edge too; a verdict counts as wrong only if the selected strategy's true share of the
  edge is below an annual Sharpe ratio of 0.5.
* **as is**: the real search on the real alignment, judged on its first 70 % of days; the selected
  strategy's Sharpe ratio on the last 30 % is reported beside each verdict. Descriptive only.

    python scripts/study_independent_trials_real.py --out DIR [--reps 200]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.epistemic import independent_trials as it  # noqa: E402

TAG = "DEV-INDEPENDENT-TRIALS-REAL-2026-11"
SEED = 2026112612
DATA = ROOT / "data" / "market"
SERIES = {"S&P 500 (FRED)": ("sp500.csv", "SP500"), "NASDAQ-100 (FRED)": ("nasdaq100.csv", "NASDAQ100"),
          "BTC (Binance)": ("crypto_top15.csv", "BTC"), "ETH (Binance)": ("crypto_top15.csv", "ETH"),
          "SPY (Yahoo)": ("equity_top15.csv", "SPY_CLOSE"), "AAPL (Yahoo)": ("equity_top15.csv", "AAPL_CLOSE")}
BARS_PER_YEAR = {"BTC (Binance)": 365, "ETH (Binance)": 365}
EDGES = (2.5, 3.5)
METHODS = ("as_referee", "by_cluster", "by_eigenvalues")


def prices(name: str) -> np.ndarray:
    path, column = SERIES[name]
    out = []
    with open(DATA / path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = row.get(column, "")
            try:
                x = float(v)
            except ValueError:
                continue
            if x > 0 and math.isfinite(x):
                out.append(x)
    return np.array(out)


def _ma(p: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(p, 0, 0.0))
    m = np.full(len(p), np.nan)
    m[k - 1:] = (c[k:] - c[:-k]) / k
    return m


def positions(p: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(days x strategies) positions in {-1, 0, 1} known at the close of each day, their names, their families."""
    cols, names, fam = [], [], []
    logp = np.log(p)
    for k in np.unique(np.geomspace(5, 250, 21).astype(int)):          # momentum
        x = np.full(len(p), np.nan)
        x[k:] = np.sign(logp[k:] - logp[:-k])
        cols.append(x)
        names.append(f"momentum {k}")
        fam.append(0)
    for f in (5, 10, 20):                                               # moving-average crossovers
        for s in (30, 50, 100, 150, 200):
            cols.append(np.sign(_ma(p, f) - _ma(p, s)))
            names.append(f"crossover {f}/{s}")
            fam.append(1)
    r1 = np.diff(logp, prepend=np.nan)
    for k in (2, 3, 4, 5, 7, 10, 15, 20, 25, 30):                      # short-term reversal
        c = np.cumsum(np.nan_to_num(r1))
        x = np.full(len(p), np.nan)
        x[k:] = -np.sign(c[k:] - c[:-k])
        cols.append(x)
        names.append(f"reversal {k}")
        fam.append(2)
    for k in (10, 15, 20, 30, 40, 55, 70, 90, 120, 150):                # channel breakouts
        x = np.full(len(p), np.nan)
        state = 0.0
        for t in range(k, len(p)):
            window = p[t - k:t]
            if p[t] >= window.max():
                state = 1.0
            elif p[t] <= window.min():
                state = -1.0
            x[t] = state
        cols.append(x)
        names.append(f"breakout {k}")
        fam.append(3)
    return np.column_stack(cols), names, np.array(fam)


def strategy_returns(pos: np.ndarray, r_next: np.ndarray) -> np.ndarray:
    return np.nan_to_num(pos) * r_next[:, None]


def run(reps: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {}
    for name in SERIES:
        p = prices(name)
        pos, names, fam = positions(p)
        start = 250                                                     # every rule has its look-back
        r = np.diff(np.log(p))                                          # r[t] = return from t to t + 1
        pos, r = pos[start:-1], r[start:]
        n_days = len(r)
        per_year = BARS_PER_YEAR.get(name, 252)
        block: dict[str, Any] = {"days": n_days, "strategies": pos.shape[1], "families": 4}
        # no edge: returns shifted against positions
        counts = dict.fromkeys(METHODS, 0)
        clusters = []
        for _ in range(reps):
            u = int(rng.integers(60, n_days - 60))
            d = it.deflated(strategy_returns(pos, np.roll(r, u)))
            clusters.append(d["by_cluster"]["n_trials"])
            for m in METHODS:
                counts[m] += int(d[m]["significant"])
        block["no edge"] = {m: {"rate_significant": round(counts[m] / reps, 4), "upper_95": round(_upper(counts[m], reps), 4)}
                            for m in METHODS}
        block["no edge"]["clusters_median"] = float(np.median(clusters))
        # planted edge in one family, chosen at random each time
        for edge in EDGES:
            found = dict.fromkeys(METHODS, 0)
            other = dict.fromkeys(METHODS, 0)
            unearned = dict.fromkeys(METHODS, 0)
            for _ in range(reps // 2):
                u = int(rng.integers(60, n_days - 60))
                g = int(rng.integers(0, 4))
                base = np.roll(r, u)
                s = np.nan_to_num(pos[:, fam == g]).mean(axis=1)
                members = np.nan_to_num(pos[:, fam == g])
                exposure = float(np.mean(members * s[:, None]))
                if exposure <= 0:
                    continue
                mu = edge / math.sqrt(per_year) * float(np.std(base)) / exposure
                planted = base + mu * s
                d = it.deflated(strategy_returns(pos, planted))
                pick = d["selected"]
                hit = int(fam[pick]) == g
                # what the selected strategy really earns from the planted component, as an annual Sharpe ratio
                true_sr = mu * float(np.mean(np.nan_to_num(pos[:, pick]) * s)) / float(np.std(base)) * math.sqrt(per_year)
                for m in METHODS:
                    if d[m]["significant"]:
                        (found if hit else other)[m] += 1
                        if true_sr < 0.5:
                            unearned[m] += 1
            n = reps // 2
            block[f"edge {edge}"] = {m: {"rate_edge_found": round(found[m] / n, 4), "rate_other_called": round(other[m] / n, 4),
                                         "rate_called_without_real_edge": round(unearned[m] / n, 4)}
                                     for m in METHODS}
        # as is: judged on the first 70 %, the selected strategy's Sharpe ratio on the last 30 %
        cut = int(0.7 * n_days)
        sr = strategy_returns(pos, r)
        d = it.deflated(sr[:cut])
        pick = d["selected"]
        later = sr[cut:, pick]
        block["as is"] = {"selected": names[pick], "family": int(fam[pick]),
                          "in_sample_sharpe_annual": round(float(it.sharpe(sr[:cut, pick]) * math.sqrt(per_year)), 3),
                          "later_sharpe_annual": round(float(later.mean() / later.std(ddof=1) * math.sqrt(per_year)), 3),
                          **{m: {"dsr": round(d[m]["dsr"], 4), "significant": d[m]["significant"], "n_trials": d[m]["n_trials"]}
                             for m in METHODS}}
        out[name] = block
        print(name, json.dumps({k: v for k, v in block.items() if k != "as is"}), json.dumps(block["as is"]), flush=True)
    return out


def _upper(k: int, n: int) -> float:
    from scipy import stats
    return 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    per_series = run(args.reps)

    def pooled(cond: str, field: str) -> dict[str, float]:
        return {m: round(float(np.mean([b[cond][m][field] for b in per_series.values()])), 4) for m in METHODS}
    record = {"tag": TAG, "development_not_a_result": True, "seed": SEED, "repetitions_no_edge": args.reps,
              "repetitions_each_edge": args.reps // 2, "threshold": it.DSR_THRESHOLD,
              "pooled": {"no edge, rate significant": pooled("no edge", "rate_significant"),
                         **{f"edge {e}, rate found": pooled(f"edge {e}", "rate_edge_found") for e in EDGES},
                         **{f"edge {e}, rate something else called": pooled(f"edge {e}", "rate_other_called") for e in EDGES},
                         **{f"edge {e}, rate called without a real edge": pooled(f"edge {e}", "rate_called_without_real_edge") for e in EDGES}},
              "series": per_series, "seconds": round(time.time() - began, 1)}
    (args.out / "independent_trials_real.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["pooled"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
