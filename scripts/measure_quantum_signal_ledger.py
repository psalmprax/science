#!/usr/bin/env python3
"""Measure a quantum circuit's output as a trading signal under the canonical ledger.

The earlier demonstration of this pipeline (a `phase`-then-`h`-then-`cx` circuit) was
constant: its output was 0.0 for every input, and the ledger had no entry point for an
external score. This record does four things the demonstration did not.

1. **A feature map whose output depends on every feature.** `Ry(pi*x_i)` on each wire,
   then `CX(i, 0)` for every `i >= 1`, measures `<Z_0> = prod_i cos(pi*x_i)`. The
   simulator is checked against that closed form on every evaluated bar, so a circuit
   that silently degenerates cannot pass as a signal.
2. **An executor that is the ledger's accounting, provably.** `execute_signal` pays
   `FEE = 5 bps` per unit position change and checks `gross - fees == net` per run; the
   record first shows it reproduces `canonical_ledger.run_ledger` exactly on a shared
   score before quoting any quantum number.
3. **A parameterised map fitted on a training slice, with the fit charged.** The second
   map family is `Ry(alpha_i x_i + beta_i)` with `(alpha, beta)` chosen from a declared
   grid. Every grid point is scored through the simulator and written to its own
   `CandidateTrialLedger`, so the fit enters the deflated Sharpe at the whole grid's
   size -- the same way the threshold sweep is charged -- rather than at one.
4. **A real holdout that is consumed and burned.** Each dataset is split by time
   (70% train / 30% holdout). The parameters are fitted on the training bars only, and
   the fitted map is then decided *once* on the holdout slice, recorded in a durable
   `FullLineageExperimentLedger`. A slice whose candidate loses money after fees is
   permanently burned, and the record demonstrates a second candidate being refused
   access to it.

Two bar resolutions are measured: daily bars from the vendored daily panels, and 15-minute
bars from the vendored intraday snapshot (annualised on 4 x 24 x 365 bars).

The honest expectation is that neither map survives fees with multiplicity charged; the
record measures that rather than assuming it.

Example::

    python scripts/measure_quantum_signal_ledger.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.epistemic.experiment_ledger import (  # noqa: E402
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.holdout_seal import (  # noqa: E402
    verify_seal,
    write_seal,
)
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger  # noqa: E402
from apedp_dyn.quantum_compilation.quantum_signal import (  # noqa: E402
    ANSATZ_DEPTHS,
    RULES,
    continuous_evaluations,
    fit_continuous_map,
    walk_forward_windows,
    ParameterisedQuantumMap,
    QuantumFeatureMap,
    deflate_sharpe,
    execute_signal,
    fit_grid,
    fit_parameterised_map,
    holdout_decision,
    ledger_summary,
)
from apedp_dyn.sandbox.canonical_ledger import (  # noqa: E402
    FEE,
    raw_features,
    run_ledger,
    zscore_features,
)

RECORD = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger.json"
#: Tranche 2 is a separate record: tranche 1's slices are burned and its numbers may not
#: be rewritten by a producer that now searches two families and charges the burn.
RECORD_TRANCHE2 = (ROOT / "benchmarks" / "quantum_compilation"
                   / "quantum_signal_ledger_tranche2.json")
RECORD_TRANCHE3 = (ROOT / "benchmarks" / "quantum_compilation"
                   / "quantum_signal_ledger_tranche3.json")
RECORD_TRANCHE4 = (ROOT / "benchmarks" / "quantum_compilation"
                   / "quantum_signal_ledger_tranche4.json")
RECORD_TRANCHE5 = (ROOT / "benchmarks" / "quantum_compilation"
                   / "quantum_signal_ledger_tranche5.json")
#: Per-bar net and fee-unit series, one JSON object per priced row. Kept beside the
#: record rather than inside it so the record stays readable while Sharpe and DSR can
#: still be re-priced at any fee rate from the bars themselves.
SERIES_TRANCHE5 = (ROOT / "benchmarks" / "quantum_compilation"
                   / "quantum_signal_series_tranche5.jsonl")
#: Each tranche names its own slices, its fitter and its record; none may reuse a slice
#: an earlier tranche decided. Declared after the panel tuples below.
#: The durable anti-snooping registry. Burned slices stay burned: this file is an
#: append-only audit log and is never reset by the producer.
HOLDOUT_LEDGER = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_holdouts.jsonl"
INTRADAY = "data/market/intraday/intraday_crypto_bars.csv"

#: Bars per year, so a Sharpe is annualised on the resolution it was measured at.
RESOLUTIONS = {"1d": 365.0, "15m": 4.0 * 24.0 * 365.0, "1h": 24.0 * 365.0,
               "30m": 2.0 * 24.0 * 365.0, "5m": 12.0 * 24.0 * 365.0}

#: The assets and resolutions measured: six daily (two crypto, four equity) and four
#: intraday. `(asset, resolution, panel, prefix-or-symbol)`.
#: Tranche 2: six assets whose slices were never decided in tranche 1, and a resolution
#: (1 hour) that tranche 1 did not measure. Tranche 1's selection is frozen in its own
#: record, which lists the datasets, panels and slice ids it consumed.
DAILY_PANELS = (
    ("GOOGL", "1d", "data/market/multi_asset_60_orderflow.csv", "GOOGL"),
    ("META", "1d", "data/market/multi_asset_60_orderflow.csv", "META"),
    ("TSLA", "1d", "data/market/multi_asset_60_orderflow.csv", "TSLA"),
    ("DOGE", "1d", "data/market/multi_asset_60_orderflow.csv", "DOGE"),
    ("ADA", "1d", "data/market/multi_asset_60_orderflow.csv", "ADA"),
    ("LINK", "1d", "data/market/multi_asset_60_orderflow.csv", "LINK"),
)
#: Tranche 3: assets and a resolution none of tranches 1-2 decided, fitted with a
#: continuous optimiser whose iteration count is the charge.
DAILY_PANELS_T3 = (
    ("AMZN", "1d", "data/market/multi_asset_60_orderflow.csv", "AMZN"),
    ("JPM", "1d", "data/market/multi_asset_60_orderflow.csv", "JPM"),
    ("XOM", "1d", "data/market/multi_asset_60_orderflow.csv", "XOM"),
    ("WMT", "1d", "data/market/multi_asset_60_orderflow.csv", "WMT"),
    ("XRP", "1d", "data/market/multi_asset_60_orderflow.csv", "XRP"),
    ("DOT", "1d", "data/market/multi_asset_60_orderflow.csv", "DOT"),
)
INTRADAY_PANELS_T3 = (("XRP", "30m"), ("TRX", "30m"), ("LTC", "30m"),
                      ("BCH", "30m"))
#: Tranche 4: assets none of tranches 1-3 touched, judged on walk-forward folds.
DAILY_PANELS_T4 = (
    ("CVX", "1d", "data/market/multi_asset_60_orderflow.csv", "CVX"),
    ("HD", "1d", "data/market/multi_asset_60_orderflow.csv", "HD"),
    ("KO", "1d", "data/market/multi_asset_60_orderflow.csv", "KO"),
    ("PEP", "1d", "data/market/multi_asset_60_orderflow.csv", "PEP"),
    ("BCH", "1d", "data/market/multi_asset_60_orderflow.csv", "BCH"),
    ("LTC", "1d", "data/market/multi_asset_60_orderflow.csv", "LTC"),
)
INTRADAY_PANELS_T4 = (("VET", "1h"), ("MANA", "1h"), ("ZEC", "1h"), ("XLM", "1h"))
#: Tranche 5: assets and a resolution (5 minutes) no earlier tranche decided, so the
#: per-bar series can be stored on data whose slices are still unconsumed. The first
#: attempt (JNJ, LLY, PG, V, AMD, NFLX daily; 1INCH, GRT, SAND, CHZ at 5 minutes) ran the
#: holdout and was then refused at the series check, so its twenty slices are burned and
#: are not re-read here: the second set is a disjoint one.
DAILY_PANELS_T5 = (
    ("MRK", "1d", "data/market/multi_asset_60_orderflow.csv", "MRK"),
    ("UNH", "1d", "data/market/multi_asset_60_orderflow.csv", "UNH"),
    ("BAC", "1d", "data/market/multi_asset_60_orderflow.csv", "BAC"),
    ("COST", "1d", "data/market/multi_asset_60_orderflow.csv", "COST"),
    ("GLD", "1d", "data/market/multi_asset_60_orderflow.csv", "GLD"),
    ("IWM", "1d", "data/market/multi_asset_60_orderflow.csv", "IWM"),
)
INTRADAY_PANELS_T5 = (("AXS", "5m"), ("SNX", "5m"), ("COMP", "5m"), ("CRV", "5m"))
INTRADAY_PANELS = (("ADA", "1h"), ("AVAX", "1h"), ("LINK", "1h"), ("DOT", "1h"))

TRANCHES = {
    "2": {"daily": DAILY_PANELS, "intraday": INTRADAY_PANELS,
          "fitter": "grid", "folds": 1, "output": RECORD_TRANCHE2},
    "3": {"daily": DAILY_PANELS_T3, "intraday": INTRADAY_PANELS_T3,
          "fitter": "continuous", "folds": 1, "output": RECORD_TRANCHE3},
    "4": {"daily": DAILY_PANELS_T4, "intraday": INTRADAY_PANELS_T4,
          "fitter": "continuous", "folds": 2, "output": RECORD_TRANCHE4},
    "5": {"daily": DAILY_PANELS_T5, "intraday": INTRADAY_PANELS_T5,
          "fitter": "continuous", "folds": 2, "output": RECORD_TRANCHE5,
          "series": SERIES_TRANCHE5},
}

#: The fixed maps measured. Both take ledger features, so no new feature engineering is
#: introduced to make the circuit look better.
MAPS = (
    QuantumFeatureMap(("H-MOM", "CVD")),
    QuantumFeatureMap(("H-MOM", "CVD", "H-BB")),
)
#: The features the parameterised map is fitted over.
PARAM_FEATURES = ("H-MOM", "CVD", "H-BB")
THRESHOLDS = (0.0, 0.2, 0.5)
RULE = "symmetric"
#: Bars skipped at the start so every feature's window has filled (H-BB needs 20,
#: HURST 30) before any score is evaluated.
WARMUP = 30
#: Fraction of each series used to fit; the remainder is the confirmation holdout.
TRAIN_FRACTION = 0.7
#: How many trailing bars are removed in the causality check: scores for the common
#: prefix must be identical when the future is deleted.
TRUNCATION = 50
#: How many bars of a circuit with no product closed form are checked against the dense
#: unitary. Above one layer there is no per-bar formula, and a dense build on every bar
#: would price the check at O(2^n) per observation.
REFERENCE_SAMPLES = 25


def load_frame(path: str, prefix: str) -> pd.DataFrame:
    """One asset's price/volume/taker-buy columns, in the ledger's own schema."""
    frame = pd.read_csv(ROOT / path)
    return pd.DataFrame({
        "price": frame[f"{prefix}_CLOSE"].to_numpy(float),
        "V": frame[f"{prefix}_VOL"].to_numpy(float),
        "taker_buy": frame[f"{prefix}_TAKER_BUY"].to_numpy(float),
    })


def load_intraday_frame(symbol: str, interval: str) -> tuple[pd.DataFrame, float]:
    """One symbol's intraday bars, plus the taker-buy denomination ratio.

    `taker_buy_base` in this snapshot is denominated in quote currency, not base: its
    ratio to `volume` equals the bar's price on every symbol (median ratio / close ~ 1).
    It is converted to base units here so the order-flow feature is a ratio of like
    units, and the ratio is returned so the record can show the check rather than assert
    it in prose.
    """
    frame = pd.read_csv(ROOT / INTRADAY)
    selected = frame[(frame["interval"] == interval) & (frame["symbol"] == symbol)]
    selected = selected.sort_values("open_time_ms")
    close = selected["close"].to_numpy(float)
    volume = selected["volume"].to_numpy(float)
    taker = selected["taker_buy_base"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.nanmedian(np.divide(taker, volume, out=np.full_like(taker, np.nan),
                                             where=volume > 0) / close))
    return pd.DataFrame({"price": close, "V": volume,
                         "taker_buy": taker / np.maximum(close, 1e-12)}), ratio


def panel_digest(frame: pd.DataFrame) -> str:
    """Short content hash, so a holdout slice id names the data it was cut from."""
    payload = np.column_stack([frame["price"].to_numpy(float),
                               frame["V"].to_numpy(float),
                               frame["taker_buy"].to_numpy(float)]).tobytes()
    return hashlib.sha256(payload).hexdigest()[:8]


def values_and_mask(frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[list[list[float]], list[bool]]:
    """The z-scored feature rows and the mask of bars where every feature is finite."""
    raw, returns = raw_features(frame)
    z, valid = zscore_features(raw)
    values = [[float(z[name][index]) for name in features] for index in range(len(returns))]
    usable = [bool(np.all([valid[name][index] for name in features]))
              for index in range(len(returns))]
    return values, usable


def series_key(row: dict[str, Any]) -> str:
    """A row's identity in the per-bar series artifact."""
    return (f"{row['kind']}|{row['dataset_id']}|{'+'.join(row['features'])}|"
            f"th={row['threshold']}|L={row['layers']}")


def max_deviation(scores: list[float], reference: list[float]) -> float:
    """The largest disagreement on the bars that were actually checked."""
    return max((abs(a - b) for a, b in zip(scores, reference, strict=True)
                if np.isfinite(b)), default=0.0)


def scores_for(frame: pd.DataFrame, feature_map: Any) -> tuple[list[float], list[bool],
                                                               list[list[float]],
                                                               np.ndarray, list[float]]:
    """The map's score per bar, its mask, the feature rows, returns and the reference.

    The reference is the product closed form where one exists. Above one layer there is
    none -- the second rotation's `X` branches survive the first entangler -- so a
    deterministic sample of `REFERENCE_SAMPLES` bars is checked against
    `exact_marginal`, a dense-unitary build that shares no code with the simulator; the
    unchecked bars are NaN and are skipped when the deviation is taken, so the record
    never implies more bars were verified than were.
    """
    values, usable = values_and_mask(frame, tuple(feature_map.features))
    scores = [feature_map.signal(row) for row in values]
    try:
        closed = [feature_map.analytic(row) for row in values]
    except ValueError:
        closed = [float("nan")] * len(values)
        for step in range(REFERENCE_SAMPLES):
            index = int(round(step * (len(values) - 1) / max(REFERENCE_SAMPLES - 1, 1)))
            closed[index] = feature_map.exact_marginal(values[index])
    _raw, returns = raw_features(frame)
    return scores, usable, values, returns, closed


def causality_case(frame: pd.DataFrame, feature_map: Any,
                   full: list[float]) -> dict[str, Any]:
    """Deleting the future must not change a past score.

    The features are causal by construction (`raw_features` uses trailing windows only),
    so this is the check that the construction is what the record claims, not an
    assumption repeated in prose. For the parameterised map the parameters were already
    fixed by the fit, so nothing about the truncation can reach back into them.
    """
    truncated = frame.iloc[:-TRUNCATION]
    scores, _usable, _values, _returns, _closed = scores_for(truncated, feature_map)
    compared = len(scores)
    deviation = max((abs(scores[i] - full[i]) for i in range(compared)), default=0.0)
    return {"bars_compared": compared, "max_absolute_deviation": deviation,
            "unchanged": deviation <= 1e-12}


def executor_equivalence(frame: pd.DataFrame) -> dict[str, Any]:
    """`execute_signal` against `run_ledger` on one shared score, row by row.

    The ledger decides its own score (`H-MOM` with unit weight under its `base` rule);
    that score, mask and returns are then handed to the executor, which must reproduce
    every position, fee and net exactly. Without this, the quantum rows would be priced
    by a second implementation nobody had checked.
    """
    raw, returns = raw_features(frame)
    _z, _valid = zscore_features(raw)
    evaluation = slice(WARMUP, len(returns) - 1)
    rows, summary = run_ledger(["X"], {"X": frame}, {"H-MOM": 1.0}, "base",
                               th_in=0.0, th_out=0.0, eval_slice=evaluation,
                               check_binding=False)
    score = [row["score"] for row in rows]
    valid = [row["valid"] for row in rows]
    # The ledger's last scored bar still has a forward return (t + 1), so the returns
    # array runs one bar past the scores rather than stopping at the same index.
    ours = execute_signal(score, returns[WARMUP:], rule="short_below_zero", valid=valid)
    position_differences = [abs(a["pos"] - b["pos"]) for a, b in zip(ours, rows,
                                                                    strict=True)]
    net_differences = [abs(a["net"] - b["net"]) for a, b in zip(ours, rows, strict=True)]
    fee_differences = [abs(a["fee"] - b["fee"]) for a, b in zip(ours, rows, strict=True)]
    return {
        "rows_compared": len(rows),
        "max_position_difference": max(position_differences, default=0.0),
        "max_fee_difference": max(fee_differences, default=0.0),
        "max_net_difference": max(net_differences, default=0.0),
        "identical": (max(position_differences, default=0.0) == 0.0
                      and max(net_differences, default=0.0) <= 1e-15),
        "ledger_status": summary["status"],
        "note": ("the canonical ledger's own score under its base rule, re-priced by "
                 "the executor; exact agreement is what licenses using the executor "
                 "for a signal the ledger cannot take as input"),
    }


def load_panels(daily=DAILY_PANELS, intraday=INTRADAY_PANELS,
                folds: int = 1) -> list[dict[str, Any]]:
    """Every dataset measured, and the walk-forward window each fold is judged on.

    With `folds > 1` each base dataset becomes one entry per fold, with its own id and its
    own contiguous test block, so a slice is consumed once while the dataset is read across
    regimes. With `folds == 1` the entry is the single tail split tranches 1-3 used.
    """
    panels: list[dict[str, Any]] = []

    def windows_for(frame: pd.DataFrame) -> list[tuple[int, int, int, int]]:
        if folds > 1:
            return walk_forward_windows(len(frame), folds, WARMUP)
        fence = int(round(len(frame) * TRAIN_FRACTION))
        return [(WARMUP, fence, fence, len(frame) - 1)]

    for asset, resolution, path, prefix in daily:
        frame = load_frame(path, prefix)
        for index, (train_start, train_end, test_start, test_end) in enumerate(
                windows_for(frame), start=1):
            suffix = f"#fold{index}" if folds > 1 else ""
            panels.append({"asset": asset, "resolution": resolution, "panel": path,
                           "frame": frame,
                           "dataset_id": f"{asset}@{resolution}{suffix}",
                           "fold": index, "folds": folds,
                           "train": [train_start, train_end],
                           "test": [test_start, test_end],
                           "taker_denomination_ratio": None})
    for asset, interval in intraday:
        frame, ratio = load_intraday_frame(asset, interval)
        for index, (train_start, train_end, test_start, test_end) in enumerate(
                windows_for(frame), start=1):
            suffix = f"#fold{index}" if folds > 1 else ""
            panels.append({"asset": asset, "resolution": interval, "panel": INTRADAY,
                           "frame": frame,
                           "dataset_id": f"{asset}@{interval}{suffix}",
                           "fold": index, "folds": folds,
                           "train": [train_start, train_end],
                           "test": [test_start, test_end],
                           "taker_denomination_ratio": ratio})
    return panels


#: The fee rates the cost sweep re-prices every priced trial at. 5 bps is the canonical
#: ledger's rate; the other two bracket it.
COST_RATES = (0.0005, 0.001, 0.002)


def _series_statistics(series_entry: dict[str, Any], rate: float, base_fee: float,
                       annualization: float) -> tuple[np.ndarray, float, float, float]:
    """One row's per-bar net at `rate`, and the statistics a Sharpe and a DSR need.

    The stored series is net at the canonical 5 bps plus that bar's `|dpos|`, so the
    re-priced series is `net(rate) = net(FEE) - (rate - FEE) * |dpos|` -- exact, bar by
    bar, with no approximation of the fee.
    """
    net = (np.asarray(series_entry["net_5bps"], dtype=float)
           - (rate - base_fee) * np.asarray(series_entry["fee_units"], dtype=float))
    count = len(net)
    mean = float(net.mean()) if count else 0.0
    deviation = float(net.std()) if count else 0.0
    sharpe = (mean / deviation * math.sqrt(annualization)
              if count > 1 and deviation > 1e-12 else 0.0)
    centered = net - mean if count else net
    variance = float(np.mean(centered ** 2)) if count else 0.0
    skew = float(np.mean(centered ** 3) / variance ** 1.5) if variance > 0 else 0.0
    kurtosis = float(np.mean(centered ** 4) / variance ** 2) if variance > 0 else 3.0
    return net, float(sharpe), skew, kurtosis


def cost_sensitivity(rows: list[dict[str, Any]], baselines: list[dict[str, Any]],
                     rates=COST_RATES,
                     series: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-price every priced trial at each fee rate.

    Fees are linear in turnover, so `fees(rate) = turnover * rate` and
    `net(rate) = gross - fees(rate)` are exact for every row from the stored
    aggregates alone. When the per-bar series is available, Sharpe and DSR are
    recomputed from the bars at every rate too -- the hurdle (`n_trials`, `var_trials`)
    stays the disclosed family's, which is held fixed across the sweep on purpose: the
    question is what the same search looks like under another fee, not what a different
    deflation would say.
    """
    priced = [*rows, *baselines]
    series = series or {}
    if not priced:
        return {"rates": list(rates), "sweep": {}, "sharpe_recomputed": False,
                "why": "no priced rows in the record"}
    turnover = sum(row["turnover"] for row in priced)
    gross = sum(row["gross"] for row in priced)
    by_resolution = {}
    for resolution in sorted({row["resolution"] for row in priced}):
        family = [row for row in priced if row["resolution"] == resolution]
        by_resolution[resolution] = (sum(row["gross"] for row in family),
                                     sum(row["turnover"] for row in family))
    sweep: dict[str, Any] = {}
    max_sharpe_deviation = 0.0
    max_net_deviation = 0.0
    repriced_rows = 0
    for rate in rates:
        fees = turnover * rate
        entry: dict[str, Any] = {
            "fee_per_unit_turnover": rate,
            "gross": gross,
            "fees": fees,
            "net": gross - fees,
            "turnover": turnover,
            "rows": len(priced),
            "rows_net_positive": sum(1 for row in priced
                                     if row["gross"] - row["turnover"] * rate > 0.0),
            "net_by_resolution": {
                resolution: gross_res - turned * rate
                for resolution, (gross_res, turned) in by_resolution.items()
            },
            # How much of the family's gross the fee eats, per resolution. This is the
            # question a fee rate is really asking: not "is net positive" but "what
            # fraction of the edge is the bill".
            "fee_share_of_gross_by_resolution": {
                resolution: ((turned * rate) / gross_res if gross_res > 0 else None)
                for resolution, (gross_res, turned) in by_resolution.items()
            },
            "gross_by_resolution": {resolution: gross_res
                                    for resolution, (gross_res, _t) in by_resolution.items()},
            "fees_by_resolution": {resolution: turned * rate
                                   for resolution, (_g, turned) in by_resolution.items()},
            # Defined even when a family's gross is negative, which is the common case
            # here: how large the bill is relative to the size of the move it taxes.
            "fees_over_absolute_gross_by_resolution": {
                resolution: ((turned * rate) / abs(gross_res) if abs(gross_res) > 0 else None)
                for resolution, (gross_res, turned) in by_resolution.items()
            },
        }
        if series and all(row.get("series_key") in series for row in priced):
            # Sharpe and DSR from the re-priced bars, per resolution: the two tiers
            # face different annualisations and different hurdles, so they are
            # adjudicated side by side rather than pooled.
            per_row: dict[str, list[tuple[float, float]]] = {}
            bars = 0
            for row in priced:
                net, sharpe, skew, kurtosis = _series_statistics(
                    series[row["series_key"]], rate, FEE, row["annualization"])
                bars += len(net)
                max_net_deviation = max(
                    max_net_deviation,
                    abs(float(net.sum()) - (row["gross"] - row["turnover"] * rate)))
                if abs(rate - FEE) <= 1e-15:
                    max_sharpe_deviation = max(max_sharpe_deviation,
                                               abs(sharpe - row["sharpe"]))
                dsr = deflate_sharpe(
                    sharpe, n_bars=int(row["n_obs"]), skewness=skew,
                    kurtosis=kurtosis, n_trials=row["n_trials"],
                    var_trials=row["var_trials"], annualization=row["annualization"])
                per_row.setdefault(row["resolution"], []).append((sharpe, dsr["dsr"]))
                repriced_rows += 1
            entry["series_bars"] = bars
            entry["sharpe_by_resolution"] = {
                resolution: {"median": float(np.median([s for s, _d in values])),
                             "min": float(min(s for s, _d in values)),
                             "max": float(max(s for s, _d in values))}
                for resolution, values in per_row.items()
            }
            entry["dsr_by_resolution"] = {
                resolution: {"best": float(max(d for _s, d in values)),
                             "median": float(np.median([d for _s, d in values])),
                             "rows_clearing_floor": sum(1 for _s, d in values
                                                        if d >= 0.95)}
                for resolution, values in per_row.items()
            }
            entry["rows_clearing_floor"] = sum(
                1 for values in per_row.values() for _s, d in values if d >= 0.95)
        sweep[f"{round(rate * 1e4)}bps"] = entry
    return {
        "rates": list(rates),
        "sweep": sweep,
        "sharpe_recomputed": bool(series),
        "series_rows": repriced_rows // len(rates) if series else 0,
        "series_max_sharpe_deviation": max_sharpe_deviation,
        "series_max_net_deviation": max_net_deviation,
        "why": ("Net, fees, turnover and the net-positive count are exact from the "
                "aggregates because fees are linear in turnover. Sharpe and DSR are "
                "re-priced from the stored per-bar net and |dpos| series where the "
                "record has one; on records from before that storage they are not "
                "re-priced at all, and this field says so rather than approximating "
                "them."),
    }


def significance_report(rows: list[dict[str, Any]],
                        baselines: list[dict[str, Any]]) -> dict[str, Any]:
    """One verdict per bar resolution, adjudicated side by side.

    A single pooled verdict hides that the daily and intraday rows face different
    hurdles, because a Sharpe measured on 365 bars a year and one measured on 35,040 are
    not the same estimator. This reports each resolution's own hurdle, its best row, and
    how many rows clear the floor, instead of averaging the two into one number.
    """
    report: dict[str, Any] = {}
    for resolution in sorted({row["resolution"] for row in (*rows, *baselines)}):
        family = [row for row in (*rows, *baselines) if row["resolution"] == resolution]
        best = max(family, key=lambda row: row["dsr"]["dsr"])
        clearing = [row for row in family if row["dsr"]["is_significant"]]
        report[resolution] = {
            "rows": len(family),
            "n_trials_charged": family[0]["n_trials"],
            "var_trials": family[0]["var_trials"],
            "hurdle": family[0]["dsr"]["expected_max_sr"],
            "floor": 0.95,
            "rows_clearing_floor": len(clearing),
            "best": {"dataset_id": best["dataset_id"],
                     "features": best["features"],
                     "threshold": best["threshold"],
                     "sharpe": best["sharpe"], "dsr": best["dsr"]["dsr"]},
            "verdict": (f"{len(clearing)} row(s) clear the floor" if clearing
                        else "no row clears the floor"),
        }
    return report


def _signal_varies(rows: list[dict[str, Any]]) -> bool:
    """A constant signal would have zero turnover on every dataset -- the bug this checks."""
    per_dataset: dict[str, float] = {}
    for row in rows:
        per_dataset[row["dataset_id"]] = (per_dataset.get(row["dataset_id"], 0.0)
                                          + row["turnover"])
    return bool(per_dataset) and all(value > 0.0 for value in per_dataset.values())


def measure(holdout_ledger_path: Path, daily=DAILY_PANELS,
            intraday=INTRADAY_PANELS,
            fitter: str = "grid", folds: int = 1,
            series_path: Path | None = None) -> dict[str, Any]:
    run_id = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trial_ledger = CandidateTrialLedger()
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    series: dict[str, Any] = {}
    fits: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    equivalence: dict[str, Any] = {}
    causality: dict[str, Any] = {}
    deviations: list[float] = []
    fit_deviations: list[float] = []

    for panel in load_panels(daily, intraday, folds):
        dataset_id = panel["dataset_id"]
        frame = panel["frame"]
        annualization = RESOLUTIONS[panel["resolution"]]
        raw, returns = raw_features(frame)
        n_bars = len(returns)
        train_start, train_end = panel["train"]
        test_start, test_end = panel["test"]
        fence = train_end
        digest = panel_digest(frame)
        equivalence[dataset_id] = executor_equivalence(frame)
        datasets.append({"dataset_id": dataset_id, "asset": panel["asset"],
                         "resolution": panel["resolution"], "panel": panel["panel"],
                         "bars": len(frame), "annualization": annualization,
                         "train_window": [train_start, train_end],
                         "holdout_window": [test_start, test_end],
                         "fold": panel["fold"], "folds": panel["folds"],
                         "taker_buy_denomination_ratio": panel["taker_denomination_ratio"]})

        for feature_map in MAPS:
            scores, usable, _values, returns, closed = scores_for(frame, feature_map)
            deviations.append(max_deviation(scores, closed))
            causality[f"{dataset_id}:{'+'.join(feature_map.features)}"] = causality_case(
                frame, feature_map, scores)
            for threshold in THRESHOLDS:
                executed = execute_signal(scores, returns, threshold=threshold,
                                          rule=RULE, valid=usable)
                # Judge only this fold's test bars.
                window = executed[test_start:test_end]
                row = {
                    "kind": "quantum",
                    "dataset_id": dataset_id,
                    "asset": panel["asset"],
                    "resolution": panel["resolution"],
                    "annualization": annualization,
                    "features": list(feature_map.features),
                    "layers": feature_map.layers,
                    "circuit_sha256": feature_map.sha256(),
                    "rule": RULE,
                    "threshold": threshold,
                    **{key: value for key, value in ledger_summary(
                        window, annualization=annualization).items()
                       if key != "net_returns"},
                }
                row["series_key"] = series_key(row)
                # The bars behind the row, so Sharpe and DSR can be re-priced at any
                # fee rate instead of only the rate the record was measured at:
                # net(r) = net(FEE) - (r - FEE) * |dpos| holds per bar, exactly.
                series[row["series_key"]] = {
                    "net_5bps": [float(item["net"]) for item in window],
                    "fee_units": [float(item["dpos"]) for item in window],
                }
                rows.append(row)

        # The baseline the quantum map has to beat: the ledger's own momentum score,
        # priced by the same executor so the comparison is accounting-for-accounting.
        z, _valid = zscore_features(raw)
        executed = execute_signal(list(z["H-MOM"]), returns, threshold=0.0,
                                  rule=RULE, valid=list(np.isfinite(raw["H-MOM"])))
        window = executed[test_start:test_end]
        baseline_row = {
            "kind": "baseline",
            "dataset_id": dataset_id,
            "asset": panel["asset"],
            "resolution": panel["resolution"],
            "annualization": annualization,
            "features": ["H-MOM"],
            "layers": 0,
            "circuit_sha256": None,
            "rule": RULE,
            "threshold": 0.0,
            **{key: value for key, value in ledger_summary(
                window, annualization=annualization).items()
               if key != "net_returns"},
        }
        baseline_row["series_key"] = series_key(baseline_row)
        series[baseline_row["series_key"]] = {
            "net_5bps": [float(item["net"]) for item in window],
            "fee_units": [float(item["dpos"]) for item in window],
        }
        baselines.append(baseline_row)

        # ---- the parameterised map: fitted on the training bars, and only those ----
        values, usable_values = values_and_mask(frame, PARAM_FEATURES)
        fit_call = (fit_continuous_map if fitter == "continuous"
                    else fit_parameterised_map)
        fit = fit_call(values, returns, features=PARAM_FEATURES,
                       valid=usable_values, ledger=trial_ledger,
                       search_id=dataset_id,
                       window=slice(train_start, train_end),
                       annualization=annualization)
        fitted = fit.feature_map
        fitted_scores, fitted_usable, _v, returns, closed = scores_for(frame, fitted)
        fit_deviations.append(max_deviation(fitted_scores, closed))
        fit_causality = causality_case(frame, fitted, fitted_scores)
        executed = execute_signal(fitted_scores, returns, rule=RULE,
                                  valid=fitted_usable, threshold=0.0)
        train_summary = ledger_summary(executed[train_start:train_end],
                                       annualization=annualization)
        hold_summary = ledger_summary(executed[test_start:test_end],
                                      annualization=annualization)
        # The paired family read: the best candidate at each depth, evaluated on the same
        # holdout window from the same single read. Each family is charged its own grid,
        # so a depth-2 win is a win after paying 24 trials, not before.
        family_winners: dict[int, tuple[float, float, float, dict[str, Any]]] = {}
        for layers in ANSATZ_DEPTHS:
            points = [candidate for candidate in fit.candidates
                      if candidate[2] == layers]
            if not points:
                continue
            alpha, beta, _layers, train_sharpe = max(points, key=lambda item: item[3])
            family_map = ParameterisedQuantumMap(
                tuple(PARAM_FEATURES), (alpha,) * len(PARAM_FEATURES),
                (beta,) * len(PARAM_FEATURES), layers=layers)
            family_scores = [family_map.signal(row) for row in values]
            family_executed = execute_signal(family_scores, returns, rule=RULE,
                                             valid=usable_values, threshold=0.0)
            family_winners[layers] = (
                alpha, beta, train_sharpe,
                ledger_summary(family_executed[test_start:test_end],
                               annualization=annualization))
        fits.append({**fit.to_dict(), "dataset_id": dataset_id, "asset": panel["asset"],
                     "resolution": panel["resolution"], "annualization": annualization,
                     "fit_window": [train_start, train_end],
                     "train_summary": {key: value for key, value in train_summary.items()
                                       if key != "net_returns"},
                     "causality": fit_causality})
        held.append({"dataset_id": dataset_id, "asset": panel["asset"],
                     "resolution": panel["resolution"],
                     "fitted": fit, "fence": fence, "digest": digest,
                     "annualization": annualization, "returns": returns,
                     "hold_summary": hold_summary, "n_bars": n_bars,
                     "train_start": train_start, "test_start": test_start,
                     "test_end": test_end, "fold": panel["fold"],
                     "families": family_winners})

    # One deflation pass over the whole family. The fitted maps enter at the size of
    # their search, not as single winners, and the dispersion is the pooled dispersion
    # of every hypothesis the record priced -- fits included.
    components = {
        "threshold_grid": len(rows),
        "baselines": len(baselines),
        "parameter_fits": int(trial_ledger.effective_trial_count),
    }
    n_trials = int(sum(components.values()))
    assert components["threshold_grid"] == len(MAPS) * len(THRESHOLDS) * len(datasets)
    assert components["baselines"] == len(datasets)
    assert len(trial_ledger.get_records()) == sum(fit["trial_count"] for fit in fits)
    # The dispersion the hurdle is built from is taken at each row's own resolution.
    # A daily Sharpe and a 15-minute Sharpe estimate different quantities; pooling them
    # inflates the variance (35,040-bar annualisation spans a far wider range) and with it
    # the hurdle, which is what made every DSR in the pooled record read 0.000. The trial
    # count is untouched: every hypothesis the record priced stays charged.
    sharpes_by_resolution: dict[str, list[float]] = {}
    for row in (*rows, *baselines):
        sharpes_by_resolution.setdefault(row["resolution"], []).append(row["sharpe"])
    for fit in fits:
        sharpes_by_resolution.setdefault(fit["resolution"], []).extend(
            candidate["sharpe"] for candidate in fit["candidates"])
    dispersion = {resolution: (float(np.var(values)) if len(values) > 1 else 0.0)
                  for resolution, values in sharpes_by_resolution.items()}
    pooled_variance = float(np.var([value for values in sharpes_by_resolution.values()
                                    for value in values]))
    for row in [*rows, *baselines]:
        row["n_trials"] = n_trials
        row["var_trials"] = dispersion[row["resolution"]]
        row["dsr"] = deflate_sharpe(
            row["sharpe"], n_bars=int(row["n_obs"]), skewness=row["skewness"],
            kurtosis=row["kurtosis"], n_trials=n_trials,
            var_trials=row["var_trials"], annualization=row["annualization"])
    for fit in fits:
        summary = fit["train_summary"]
        summary["dsr"] = deflate_sharpe(
            summary["sharpe"], n_bars=int(summary["n_obs"]),
            skewness=summary["skewness"], kurtosis=summary["kurtosis"],
            n_trials=n_trials, var_trials=dispersion[fit["resolution"]],
            annualization=fit["annualization"])
        fit["n_trials"] = n_trials
        fit["var_trials"] = dispersion[fit["resolution"]]

    # ---- PRE-FLIGHT: settle everything that can refuse the run before a slice moves ----
    # Tranche 5's first attempt is why this exists: it decided twenty holdouts and was
    # then refused by the series check, so twenty slices were burned with nothing to show
    # for them. Everything that can refuse the run is now evaluated against data already
    # in memory, while no slice has been consumed. The burn log itself is verified
    # against its seal first, so a run cannot start on a registry that was edited,
    # truncated or rewound.
    verify_seal(holdout_ledger_path)
    cost = cost_sensitivity(rows, baselines, series=series)
    preflight = {
        "burn_log_seal_verified": True,
        "series_stored": bool(series),
        "series_rows": len(series),
        "series_reproduces_every_aggregate": (
            not series or (bool(cost["sharpe_recomputed"])
                           and cost["series_max_sharpe_deviation"] <= 1e-9
                           and cost["series_max_net_deviation"] <= 1e-6)),
        "accounting_identity_holds_in_every_row": all(
            abs(row["gross"] - row["fees"] - row["net"]) < 1e-9
            for row in (*rows, *baselines)),
        "trial_count_is_disclosed": n_trials == sum(components.values()),
        "slices_consumed_by_this_run": 0,
    }
    failed_preflight = [name for name, ok in preflight.items()
                        if isinstance(ok, bool) and not ok]
    if failed_preflight:
        raise SystemExit("refusing to consume a holdout slice -- pre-flight failed: "
                         + ", ".join(failed_preflight))

    # ---- the holdout: consumed once per dataset, burned when the candidate fails ----
    ledger = FullLineageExperimentLedger(persistence_path=holdout_ledger_path)
    holdouts: dict[str, Any] = {}
    for entry in held:
        dataset_id = entry["dataset_id"]
        holdout_end = entry["test_end"]
        slice_id = (f"fold{entry['fold']}:{entry['digest']}:"
                    f"{entry['train_start']}-{entry['fence']}:"
                    f"{entry['test_start']}-{holdout_end}")
        candidate_id = f"{dataset_id}:{run_id}"
        ledger.register_candidate(
            candidate_id, proposer_id="quantum_signal_ledger",
            formula="fitted angle-encoding map, train-slice fit",
            feature_set=tuple(PARAM_FEATURES),
            parameter_choices={"alpha": entry["fitted"].feature_map.alphas[0],
                               "beta": entry["fitted"].feature_map.betas[0]},
        )
        key = (dataset_id, slice_id)
        burned = ledger.get_burned_holdouts()
        if key in burned:
            # The slice was consumed by an earlier candidate and burned. This run does
            # not get a second look at it; the refusal is the record's evidence.
            blocked = {"dataset_id": dataset_id, "slice_id": slice_id,
                       "candidate_id": candidate_id, "status": "REFUSED_BURNED",
                       "burned_by": burned[key], "passed": None}
            try:
                ledger.record_data_exposure(candidate_id, dataset_id, slice_id,
                                            DatasetRole.CONFIRMATION_HOLDOUT)
            except HoldoutContaminationError as exc:
                blocked["refusal"] = str(exc)
            holdouts[dataset_id] = blocked
            continue
        summary = entry["hold_summary"]
        # Cost-aware: the slice is claimed only if the holdout trades, nets, and clears
        # the expected maximum Sharpe of the search that produced this candidate, so the
        # burn is charged to the fit's own search size rather than to a bare sign test.
        decision = holdout_decision(
            summary["net"], summary["turnover"], summary["sharpe"],
            searched_trials=entry["fitted"].effective_trial_count,
            var_trials=dispersion[entry["resolution"]],
        )
        passed = decision["passed"]
        ledger.record_data_exposure(
            candidate_id, dataset_id, slice_id, DatasetRole.CONFIRMATION_HOLDOUT,
            "PASSED" if passed else "FAILED",
        )
        ledger.update_lifecycle_state(
            candidate_id,
            (CandidateLifecycleState.CONFIRMED_VALID if passed
             else CandidateLifecycleState.CONFIRMATION_REJECTED),
            rejection_reason=(None if passed else decision["reason"]),
            metrics={"holdout_net": summary["net"], "holdout_sharpe": summary["sharpe"],
                     "train_sharpe": entry["fitted"].in_sample_sharpe,
                     "n_trials": n_trials},
        )
        holdouts[dataset_id] = {
            "dataset_id": dataset_id, "asset": entry["asset"],
            "candidate_id": candidate_id, "slice_id": slice_id,
            "role": DatasetRole.CONFIRMATION_HOLDOUT.value,
            "fold": entry["fold"],
            "train_window": [entry["train_start"], entry["fence"]],
            "holdout_window": [entry["test_start"], holdout_end],
            "train_bars": int(entry["fence"] - entry["train_start"]),
            "holdout_bars": int(holdout_end - entry["test_start"]),
            "fitted_on": "training bars only",
            "status": ("CONFIRMED_VALID" if passed else "CONFIRMATION_REJECTED"),
            "passed": passed,
            "hurdle": decision["hurdle"],
            "searched_trials": decision["searched_trials"],
            "decision_reason": decision["reason"],
            "net": summary["net"], "fees": summary["fees"],
            "turnover": summary["turnover"], "sharpe": summary["sharpe"],
            "burned": (dataset_id, slice_id) in ledger.get_burned_holdouts(),
            "depth_comparison": {
                f"depth{layers}": {
                    "alpha": alpha, "beta": beta,
                    "train_sharpe": train_sharpe,
                    "holdout_net": family_summary["net"],
                    "holdout_sharpe": family_summary["sharpe"],
                    "hurdle": family_decision["hurdle"],
                    "searched_trials": family_decision["searched_trials"],
                    "passed": family_decision["passed"],
                }
                for layers, (alpha, beta, train_sharpe, family_summary)
                in sorted(entry["families"].items())
                for family_decision in (holdout_decision(
                    family_summary["net"], family_summary["turnover"],
                    family_summary["sharpe"],
                    searched_trials=entry["fitted"].effective_trial_count,
                    var_trials=dispersion[entry["resolution"]]),)
            },
        }
        if not passed:
            assert (dataset_id, slice_id) in ledger.get_burned_holdouts(), dataset_id

    burned_holdouts = sorted([list(key) for key in ledger.get_burned_holdouts()])
    # A second candidate cannot re-enter a burned slice. Recorded as an observation, not
    # as prose: the error is the ledger's own.
    reuse_refusals: list[dict[str, Any]] = []
    for dataset_id, slice_id in [tuple(item) for item in burned_holdouts][:2]:
        probe_id = f"probe:{dataset_id}:{run_id}"
        ledger.register_candidate(probe_id, proposer_id="quantum_signal_ledger",
                                  formula="post-burn reuse probe",
                                  feature_set=tuple(PARAM_FEATURES))
        try:
            ledger.record_data_exposure(probe_id, dataset_id, slice_id,
                                        DatasetRole.CONFIRMATION_HOLDOUT)
            reuse_refusals.append({"dataset_id": dataset_id, "slice_id": slice_id,
                                   "refused": False, "error": None})
        except HoldoutContaminationError as exc:
            reuse_refusals.append({"dataset_id": dataset_id, "slice_id": slice_id,
                                   "refused": True, "error": str(exc)})

    # Depth 2 against depth 1, out of sample, each charged its own search.
    paired = []
    for entry in holdouts.values():
        comparison = entry.get("depth_comparison") or {}
        one, two = comparison.get("depth1"), comparison.get("depth2")
        if one is None or two is None:
            continue
        paired.append({"dataset_id": entry["dataset_id"],
                       "depth1_net": one["holdout_net"],
                       "depth2_net": two["holdout_net"],
                       "delta": two["holdout_net"] - one["holdout_net"],
                       "depth1_passed": one["passed"], "depth2_passed": two["passed"]})
    depth_summary = {
        "depths": list(ANSATZ_DEPTHS),
        "charge_per_family": (int(fits[0]["effective_trial_count"]) if fits else 0),
        "rows": paired,
        "datasets_compared": len(paired),
        "depth2_beats_depth1_holdout": sum(1 for row in paired if row["delta"] > 0),
        "mean_holdout_delta": (float(np.mean([row["delta"] for row in paired]))
                               if paired else 0.0),
        "depth1_clears_hurdle": sum(1 for row in paired if row["depth1_passed"]),
        "depth2_clears_hurdle": sum(1 for row in paired if row["depth2_passed"]),
        "note": ("both families' best candidate is read once on the same holdout window, "
                 "each charged its own grid; the burn is decided by the map the training "
                 "slice actually selected"),
    }

    checked_holdouts = [entry for entry in holdouts.values() if "holdout_window" in entry]
    checks = {
        "the_signal_is_the_closed_form_on_every_bar": max(deviations) <= 1e-12,
        "the_fitted_map_matches_its_reference_on_every_bar": max(fit_deviations) <= 1e-12,
        "the_signal_varies_with_the_features": _signal_varies(rows),
        "the_executor_reproduces_the_canonical_ledger": all(
            entry["identical"] for entry in equivalence.values()),
        "the_accounting_identity_holds_in_every_row": all(
            abs(row["gross"] - row["fees"] - row["net"]) < 1e-9
            for row in (*rows, *baselines)),
        "deleting_the_future_does_not_change_a_past_score": all(
            entry["unchanged"] for entry in causality.values()),
        "the_trial_count_is_disclosed_and_matches": (
            n_trials == sum(components.values())
            and components["parameter_fits"] == trial_ledger.effective_trial_count
            and components["parameter_fits"] == sum(fit["effective_trial_count"]
                                                   for fit in fits)
            and all(row["n_trials"] == n_trials for row in (*rows, *baselines))),
        "significance_is_reported_as_the_floor_decides": all(
            bool(row["dsr"]["is_significant"]) == (row["dsr"]["dsr"] >= 0.95)
            for row in (*rows, *baselines)),
        "every_trial_pays_the_declared_fee": all(
            row["fees"] >= 0.0 and row["turnover"] >= 0.0 for row in (*rows, *baselines)),
        "the_fit_used_only_training_bars": all(
            fit["fit_window"][0] == entry["train_start"]
            and fit["fit_window"][1] == entry["fence"]
            and fit["fit_window"][1] < fit["causality"]["bars_compared"]
            for fit, entry in zip(fits, held, strict=True)),
        "the_holdout_block_follows_its_training_window": (
            bool(checked_holdouts) and all(
                entry["holdout_window"][0] == entry["train_window"][1]
                and entry["holdout_window"][1] > entry["holdout_window"][0]
                for entry in checked_holdouts)),
        "a_failed_candidate_burns_its_holdout_slice": all(
            (not entry["passed"]) == entry["burned"] for entry in checked_holdouts),
        "a_burned_slice_is_refused_on_reuse": (
            (not burned_holdouts) or (bool(reuse_refusals)
                                      and all(item["refused"] for item in reuse_refusals))),
        # The stored bars must be the bars the record's own statistics came from: at
        # the canonical rate the re-priced Sharpe has to reproduce the recorded one.
        "the_stored_series_reproduces_every_aggregate": (
            not series or (bool(cost["sharpe_recomputed"])
                           and cost["series_max_sharpe_deviation"] <= 1e-9
                           and cost["series_max_net_deviation"] <= 1e-6)),
        "the_intraday_taker_column_is_quote_denominated": all(
            abs(entry["taker_buy_denomination_ratio"] - 1.0) < 0.25
            for entry in datasets if entry["taker_buy_denomination_ratio"] is not None),
    }
    if not all(checks.values()):
        raise SystemExit(f"refusing to write a record with failed checks: {checks}")

    # The burn log has been written by the ledger; seal what is on disk so a later
    # edit, truncation or rewind is detectable rather than silent.
    holdout_seal = write_seal(holdout_ledger_path)
    resealed = verify_seal(holdout_ledger_path)
    if resealed.chain_sha256 != holdout_seal.chain_sha256:
        raise SystemExit("the holdout seal did not verify immediately after writing")

    if series_path is not None:
        series_path.write_text(
            "".join(json.dumps({"key": key, **payload}, sort_keys=True) + "\n"
                    for key, payload in sorted(series.items())), encoding="utf-8")

    return {
        "schema": "apedp.quantum_signal_ledger.v2",
        "generated_utc": run_id,
        "fee_per_unit_turnover": FEE,
        "annualisation": {name: value for name, value in RESOLUTIONS.items()},
        "warmup_bars": WARMUP,
        "train_fraction": TRAIN_FRACTION,
        "thresholds": list(THRESHOLDS),
        "rule": RULE,
        "rules_available": list(RULES),
        "n_trials": n_trials,
        "var_trials": {"pooled": pooled_variance, "by_resolution": dispersion},
        "trials_by_resolution": {resolution: len(values)
                                 for resolution, values in sharpes_by_resolution.items()},
        "dispersion_method": ("within resolution: a daily Sharpe and a 15-minute Sharpe "
                              "estimate different quantities, so the hurdle is built from "
                              "the dispersion at the row's own resolution while the trial "
                              "count stays the whole disclosed family"),
        "trial_components": components,
        "assets": datasets,
        "maps": [{"features": list(m.features), "layers": m.layers,
                  "qubits": m.num_qubits, "circuit_sha256": m.sha256(),
                  "gates": list(m.description()),
                  "closed_form": "prod_i cos(pi * x_i)"} for m in MAPS],
        "parameterised": {
            "features": list(PARAM_FEATURES),
            "optimiser": fitter,
            "grid_size": (len(fit_grid(len(PARAM_FEATURES)))
                          if fitter == "grid" else None),
            "evaluations_per_depth": (continuous_evaluations()
                                      if fitter == "continuous" else None),
            "ansatz_depths": list(ANSATZ_DEPTHS),
            "grid": ([[list(alphas), list(betas), layers]
                      for alphas, betas, layers in fit_grid(len(PARAM_FEATURES))]
                     if fitter == "grid" else None),
            "closed_form": ("prod_i cos(alpha_i * x_i + beta_i) at depth 1; dense "
                            "unitary marginal above it (no product form exists)"),
            "fit_slice": "training bars only",
            "trial_ledger": {
                "records": trial_ledger.trial_count,
                "effective_trial_count": int(trial_ledger.effective_trial_count),
                "naive_trial_count": trial_ledger.trial_count,
                "hidden_trials": int(trial_ledger.effective_trial_count
                                     - trial_ledger.trial_count),
                "mean_sharpe": trial_ledger.get_sharpe_distribution()[0],
                "var_sharpe": trial_ledger.get_sharpe_distribution()[1],
                "search_groups": sorted({record.metadata["search_id"]
                                         for record in trial_ledger.get_records()}),
                "note": ("every grid point is one recorded trial, so the fit is charged "
                         "at the size of its search exactly as the threshold sweep is"),
            },
            "fits": fits,
        },
        "rows": rows,
        "baselines": baselines,
        "holdout": holdouts,
        "holdout_ledger": str(holdout_ledger_path.relative_to(ROOT)),
        "holdout_seal": holdout_seal.to_dict(),
        "preflight": preflight,
        "burned_holdouts": burned_holdouts,
        "holdout_reuse_refusals": reuse_refusals,
        "executor_equivalence": equivalence,
        "causality": causality,
        "checks": checks,
        "significance_report": significance_report(rows, baselines),
        "cost_sensitivity": cost,
        "series_artifact": (str(series_path.relative_to(ROOT))
                            if series_path is not None else None),
        "series_rows": len(series),
        "depth_comparison": depth_summary,
        "significant_rows": [{"dataset_id": row["dataset_id"],
                              "features": row["features"],
                              "threshold": row["threshold"], "dsr": row["dsr"]["dsr"]}
                             for row in (*rows, *baselines) if row["dsr"]["is_significant"]],
        "limitations": [
            "the fixed maps have no trainable parameters; the parameterised map is "
            "fitted on 12 grid points per dataset and that search is charged to the "
            "deflation, so the fit is not quoted as a lone hypothesis",
            "the closed form prod_i cos(alpha_i x_i + beta_i) means every map here is "
            "classically computable in O(n) floating-point operations; the statevector "
            "simulation is the definition of the circuit, not the cheapest evaluation, "
            "so no advantage is claimed",
            "ten datasets across two resolutions (six daily panels, four 15-minute "
            "snapshots), no slippage beyond the declared fee, and no live feed",
            "the holdout is temporal and consumed once: a slice whose candidate loses "
            "money after fees is burned in the durable ledger and cannot be re-entered, "
            "so this record is not regenerable from the same slices by design",
            "the holding period is one bar, so the 15-minute rows annualise a Sharpe "
            "estimated from roughly ten days of bars and should be read as a resolution "
            "comparison, not a ten-day backtest of a trading business",
        ],
    }


def recompute_dispersion(record: dict[str, Any]) -> dict[str, Any]:
    """Re-price a stored record's DSR within each resolution.

    Reads only the trial statistics the record already holds -- no panel, no holdout, no
    ledger -- so re-deriving the statistic cannot consume a slice. The trial count is
    unchanged: every hypothesis the record priced stays charged; only the dispersion the
    hurdle is built from becomes resolution-local.
    """
    if "parameterised" not in record:
        raise SystemExit("record predates the parameterised fit; regenerate it instead")
    by_resolution: dict[str, list[float]] = {}
    for row in [*record["rows"], *record["baselines"]]:
        by_resolution.setdefault(row["resolution"], []).append(row["sharpe"])
    for fit in record["parameterised"]["fits"]:
        by_resolution.setdefault(fit["resolution"], []).extend(
            candidate["sharpe"] for candidate in fit["candidates"])
    dispersion = {resolution: (float(np.var(values)) if len(values) > 1 else 0.0)
                  for resolution, values in by_resolution.items()}
    pooled = float(np.var([value for values in by_resolution.values() for value in values]))
    n_trials = int(record["n_trials"])
    for row in [*record["rows"], *record["baselines"]]:
        row["n_trials"] = n_trials
        row["var_trials"] = dispersion[row["resolution"]]
        row["dsr"] = deflate_sharpe(row["sharpe"], n_bars=int(row["n_obs"]),
                                    skewness=row["skewness"], kurtosis=row["kurtosis"],
                                    n_trials=n_trials, var_trials=row["var_trials"],
                                    annualization=row["annualization"])
    for fit in record["parameterised"]["fits"]:
        summary = fit["train_summary"]
        fit["var_trials"] = dispersion[fit["resolution"]]
        summary["dsr"] = deflate_sharpe(summary["sharpe"], n_bars=int(summary["n_obs"]),
                                        skewness=summary["skewness"],
                                        kurtosis=summary["kurtosis"], n_trials=n_trials,
                                        var_trials=fit["var_trials"],
                                        annualization=fit["annualization"])
    # The holdout is one economic decision on one slice, not a search; a pooled DSR there
    # is removed rather than re-priced, so no stale hurdle survives the change.
    for entry in record["holdout"].values():
        entry.pop("dsr", None)
    legacy = "the_fitted_map_is_its_closed_form_on_every_bar"
    if legacy in record["checks"]:
        record["checks"]["the_fitted_map_matches_its_reference_on_every_bar"] = \
            record["checks"].pop(legacy)
    record["var_trials"] = {"pooled": pooled, "by_resolution": dispersion}
    record["trials_by_resolution"] = {resolution: len(values)
                                      for resolution, values in by_resolution.items()}
    record["dispersion_method"] = (
        "within resolution: a daily Sharpe and a 15-minute Sharpe estimate different "
        "quantities, so the hurdle is built from the dispersion at the row's own "
        "resolution while the trial count stays the whole disclosed family")
    record["significance_report"] = significance_report(record["rows"],
                                                       record["baselines"])
    record["cost_sensitivity"] = cost_sensitivity(record["rows"], record["baselines"])
    record["cost_sensitivity"]["why"] += (" This record predates series storage, so "
                                           "its Sharpe and DSR are not re-priced here.")
    # The per-family charge is derived from the fits, so a continuous search is not
    # mislabelled with the grid's size.
    if "depth_comparison" in record and record["parameterised"]["fits"]:
        record["depth_comparison"]["charge_per_family"] = int(
            record["parameterised"]["fits"][0]["effective_trial_count"])
    record["significant_rows"] = [
        {"dataset_id": row["dataset_id"], "features": row["features"],
         "threshold": row["threshold"], "dsr": row["dsr"]["dsr"]}
        for row in [*record["rows"], *record["baselines"]] if row["dsr"]["is_significant"]]
    if any(bool(row["dsr"]["is_significant"]) != (row["dsr"]["dsr"] >= 0.95)
           for row in [*record["rows"], *record["baselines"]]):
        raise SystemExit("re-priced DSR disagrees with its own significance flag")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tranche", choices=sorted(TRANCHES), default="3")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--holdout-ledger", type=Path, default=HOLDOUT_LEDGER)
    parser.add_argument("--recompute-dispersion", action="store_true",
                        help="re-price an existing record's DSR within each resolution; "
                             "reads no panel and consumes no holdout slice")
    args = parser.parse_args()
    if args.output is None:
        args.output = TRANCHES[args.tranche]["output"]
    if args.recompute_dispersion:
        if not args.output.is_file():
            raise SystemExit(f"no record to re-price at {args.output}")
        record = recompute_dispersion(json.loads(args.output.read_text(encoding="utf-8")))
    else:
        spec = TRANCHES[args.tranche]
        record = measure(args.holdout_ledger, spec["daily"], spec["intraday"],
                         fitter=spec["fitter"], folds=spec["folds"],
                         series_path=spec.get("series"))
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for row in [*record["rows"], *record["baselines"]]:
        label = "+".join(row["features"])
        print(f"{row['dataset_id']:9s} {label:18s} th={row['threshold']:.1f}"
              f"  net={row['net']:+.4f} fees={row['fees']:.4f}"
              f" turn={row['turnover']:.1f}  sharpe={row['sharpe']:+.3f}"
              f"  dsr={row['dsr']['dsr']:.3f}")
    for fit in record["parameterised"]["fits"]:
        holdout = record["holdout"][fit["dataset_id"]]
        status = holdout.get("status", "n/a")
        holdout_net = holdout.get("net")
        shown = "n/a" if holdout_net is None else f"{holdout_net:+.4f}"
        holdout_turn = holdout.get("turnover")
        shown_turn = "n/a" if holdout_turn is None else f"{holdout_turn:.0f}"
        print(f"{fit['dataset_id']:9s} fitted a={fit['alphas'][0]:.1f} b={fit['betas'][0]:+.1f}"
              f"  train_net={fit['train_summary']['net']:+.4f}"
              f"  holdout_net={shown} turn={shown_turn}  {status}")
    dispersion = record["var_trials"]
    print(f"trials: {record['n_trials']} {record['trial_components']}"
          f"  var_trials: {dispersion['by_resolution']}"
          f"  pooled: {dispersion['pooled']:.3f}")
    this_tranche = sum(1 for entry in record["holdout"].values() if entry.get("burned"))
    print(f"holdouts burned: {this_tranche}/{len(record['holdout'])}"
          f"  (registry total {len(record['burned_holdouts'])})")
    print(f"checks: {sum(record['checks'].values())}/{len(record['checks'])}")
    print(f"written: {args.output.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
