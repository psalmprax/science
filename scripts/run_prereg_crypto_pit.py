"""The analysis frozen by preregistration/PREREG-CRYPTO-PIT-2026-11.md.

Run exactly once, on the panel built by scripts/build_point_in_time_crypto_panel.py:

    python scripts/run_prereg_crypto_pit.py --panel-dir <dir with the two panels> --out <run dir>

Everything this does is specified in the preregistration, and the digest of
this file is recorded in preregistration/DIGEST.json before any data exists.
A change to it after that is a deviation and is handled by the deviation rule
there, not by editing.

Candidates (all predict a within-day cross-sectional rank among the universe):
  H1-OFI     sign * OFI_t        -> r_{t+1}    the test claim; sign fixed on the
                                               development period
  CPLUS-VOL  mean |r| over 20d   -> |r_{t+1}|  positive control (volatility
                                               persistence); stages 2-4
  CZERO-PLAC hash(asset, date)   -> r_{t+1}    null control; must be refused

Each runs through apedp_dyn.epistemic.promotion_lifecycle.PromotionTracker:
stage 1 on reported metrics (see the preregistration), stages 2-3 on the
development period, stage 4 on a holdout window through a persistent
experiment ledger, and for H1-OFI stage 5 on its holdout window's long-short
returns with a persistent trial ledger and the universe audit.

The experiment ledger lets one candidate confirm on a given window of data
(``HoldoutContaminationError`` otherwise, which is never bypassed). So the
holdout is split once, by date, before anything is scored: the first
CONTROL_HOLDOUT_DAYS days are the positive control's, the rest are H1-OFI's.
The placebo has no holdout window; if it ever reached stage 4 that would be a
false pass at stages 2-3, recorded as such.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.experiment_ledger import FullLineageExperimentLedger
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionPrerequisiteError,
    PromotionStage,
    PromotionTracker,
    information_coefficient,
)
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.market.universe import audit_universe, point_in_time_membership

PREREGISTRATION_ID = "PREREG-CRYPTO-PIT-2026-11"
DATASET_ID = "binance-usdt-every-pair-daily"
#: Universe: the 30 largest by traded value over the 60 bars before each
#: rebalance, rebalanced every 21 bars (point_in_time_membership).
TOP_N, LOOKBACK, REBALANCE = 30, 60, 21
#: A day enters the pooled sample only with at least this many valid pairs.
MIN_ASSETS_PER_DAY = 10
VOL_WINDOW = 20
MOMENTUM_WINDOW = 20
#: Long the top fifth by prediction, short the bottom fifth, equal weights.
QUANTILE = 0.2
#: The positive control's holdout window: the holdout's first 31 days.
CONTROL_HOLDOUT_DAYS = 31


def load_panel(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())))


def assets_of(panel: pd.DataFrame) -> list[str]:
    return sorted(c[: -len("_CLOSE")] for c in panel.columns if c.endswith("_CLOSE"))


def wide(panel: pd.DataFrame, suffix: str, assets: list[str]) -> np.ndarray:
    return panel[[f"{a}{suffix}" for a in assets]].to_numpy(dtype=float)


def simple_returns(close: np.ndarray) -> np.ndarray:
    """r_t = close_t / close_{t-1} - 1; NaN unless both bars exist and are positive."""
    out = np.full_like(close, np.nan)
    prev, cur = close[:-1], close[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0) & (cur > 0)
    out[1:][ok] = cur[ok] / prev[ok] - 1.0
    return out


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean over the ``window`` bars ending at t; NaN unless all are finite."""
    out = np.full_like(values, np.nan)
    for t in range(window - 1, len(values)):
        block = values[t - window + 1: t + 1]
        ok = np.isfinite(block).all(axis=0)
        out[t, ok] = block[:, ok].mean(axis=0)
    return out


def placebo(dates: list[str], assets: list[str]) -> np.ndarray:
    """A deterministic score per (asset, date) that knows nothing about markets."""
    out = np.empty((len(dates), len(assets)))
    for i, d in enumerate(dates):
        for j, a in enumerate(assets):
            digest = hashlib.sha256(f"{PREREGISTRATION_ID}|{a}|{d}".encode()).digest()
            out[i, j] = int.from_bytes(digest[:8], "big") / 2**64
    return out


def features(panel: pd.DataFrame) -> dict[str, Any]:
    """Every predictor and target, aligned so row t predicts from bars <= t."""
    assets = assets_of(panel)
    dates = panel["observation_date"].astype(str).tolist()
    close = wide(panel, "_CLOSE", assets)
    ret = simple_returns(close)
    nxt = np.full_like(ret, np.nan)
    nxt[:-1] = ret[1:]
    momentum = np.full_like(close, np.nan)
    momentum[MOMENTUM_WINDOW:] = close[MOMENTUM_WINDOW:] / close[:-MOMENTUM_WINDOW] - 1.0
    momentum[~np.isfinite(momentum)] = np.nan
    return {
        "assets": assets,
        "dates": dates,
        "ofi": wide(panel, "_OFI", assets),
        "vol20": trailing_mean(np.abs(ret), VOL_WINDOW),
        "placebo": placebo(dates, assets),
        "reversal": -ret,
        "momentum": momentum,
        "next_return": nxt,
        "next_abs_return": np.abs(nxt),
    }


def _centred_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks / (len(values) - 1) - 0.5


def pairs(feat: dict[str, Any], members: np.ndarray, predictor: str, target: str,
          start: str, end: str, sign: float = 1.0, extra: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Pooled within-day ranks for days in [start, end] whose target is also <= end.

    A pair needs the asset to be a member on day t and both the prediction and
    the target to be finite; a day needs MIN_ASSETS_PER_DAY pairs. Extra
    columns (literature features) are ranked over the same assets.
    """
    pred_all, targ_all = feat[predictor] * sign, feat[target]
    out: dict[str, list] = {"date": [], "asset": [], "pred": [], "target": [], "raw_target": []}
    for name in extra:
        out[name] = []
    dates = feat["dates"]
    for t in range(len(dates) - 1):
        if not (start <= dates[t] and dates[t + 1] <= end):
            continue
        ok = members[t] & np.isfinite(pred_all[t]) & np.isfinite(targ_all[t])
        for name in extra:
            ok &= np.isfinite(feat[name][t])
        if ok.sum() < MIN_ASSETS_PER_DAY:
            continue
        idx = np.flatnonzero(ok)
        out["date"] += [dates[t]] * len(idx)
        out["asset"] += [feat["assets"][j] for j in idx]
        out["pred"] += list(_centred_rank(pred_all[t, idx]))
        out["target"] += list(_centred_rank(targ_all[t, idx]))
        out["raw_target"] += list(targ_all[t, idx])
        for name in extra:
            out[name] += list(_centred_rank(feat[name][t, idx]))
    return {k: np.asarray(v) for k, v in out.items()}


def long_short(feat: dict[str, Any], members: np.ndarray, sign: float, start: str, end: str) -> dict[str, np.ndarray]:
    """Daily long-short returns and turnover of the frozen H1-OFI rule."""
    dates, assets = feat["dates"], feat["assets"]
    pred, nxt = feat["ofi"] * sign, feat["next_return"]
    prev_w = np.zeros(len(assets))
    out: dict[str, list] = {"date": [], "gross": [], "turnover": []}
    for t in range(len(dates) - 1):
        if not (start <= dates[t] and dates[t + 1] <= end):
            continue
        ok = members[t] & np.isfinite(pred[t]) & np.isfinite(nxt[t])
        if ok.sum() < MIN_ASSETS_PER_DAY:
            continue
        idx = np.flatnonzero(ok)
        k = max(1, math.floor(QUANTILE * len(idx)))
        order = idx[np.argsort(pred[t, idx], kind="mergesort")]
        w = np.zeros(len(assets))
        w[order[-k:]] = 1.0 / k
        w[order[:k]] = -1.0 / k
        out["date"].append(dates[t + 1])
        out["gross"].append(float(np.dot(w[idx], nxt[t, idx])))
        out["turnover"].append(0.5 * float(np.abs(w - prev_w).sum()))
        prev_w = w
    return {k: np.asarray(v) for k, v in out.items()}


def _sharpe(returns: np.ndarray) -> float:
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return float(np.mean(returns) / sd * math.sqrt(365.0)) if sd > 0 else 0.0


def _record(rec: Any) -> dict[str, Any]:
    out = asdict(rec)
    out["stage"] = PromotionStage(rec.stage).name
    return out


def _stage1(tracker: PromotionTracker) -> Any:
    """Reported, as the preregistration states: a centred cross-sectional rank is
    unit-free and bounded in [-0.5, 0.5] by construction; there is no Lean model."""
    return tracker.verify_stage_1("MATHEMATICALLY_CONSISTENT", dimensions_consistent=True, bounded=True)


def run_candidate(cid: str, formula: str, feat: dict[str, Any], members: np.ndarray, predictor: str,
                  target: str, sign: float, periods: dict[str, str], window: tuple[str, str] | None,
                  run_dir: Path, ledger: FullLineageExperimentLedger,
                  literature: tuple[str, ...]) -> dict[str, Any]:
    tracker = PromotionTracker(cid, state_path=run_dir / "promotion_journal.jsonl", allow_reported_metrics=True)
    records = [_record(_stage1(tracker))]
    dev = pairs(feat, members, predictor, target, periods["dev_start"], periods["dev_end"], sign, literature)
    hold = (pairs(feat, members, predictor, target, window[0], window[1], sign) if window
            else {"pred": np.empty(0), "target": np.empty(0), "date": np.empty(0), "asset": np.empty(0)})
    summary: dict[str, Any] = {
        "candidate": cid, "formula": formula, "sign": sign, "holdout_window": window,
        "development_pairs": len(dev["pred"]), "development_days": len(set(dev["date"])),
        "holdout_pairs": len(hold["pred"]), "holdout_days": len(set(hold["date"])),
    }
    try:
        rec = tracker.verify_stage_2_from_predictions(dev["pred"], dev["target"])
        records.append(_record(rec))
        if rec.passed:
            beta = float(np.dot(dev["pred"], dev["target"]) / np.dot(dev["pred"], dev["pred"]))
            rec = tracker.verify_stage_3_from_forecasts(
                dev["target"], beta * dev["pred"], {"zero": np.zeros_like(dev["target"])},
                dev["pred"], {name: dev[name] for name in literature},
            )
            records.append(_record(rec))
        if rec.passed and window is None:
            summary["control_failure"] = "reached stage 4 with no holdout window: a false pass at stages 2-3"
        elif rec.passed:
            ledger.register_candidate(cid, "prereg", formula)
            rec = tracker.verify_stage_4_from_holdout(
                hold["pred"], hold["target"], experiment_ledger=ledger, dataset_id=DATASET_ID,
                slice_identifier=f"{cid}@{window[0]}..{window[1]}",
                index=np.asarray([f"{d}|{a}" for d, a in zip(hold["date"], hold["asset"], strict=True)]),
            )
            records.append(_record(rec))
    except PromotionPrerequisiteError as exc:
        summary["stopped"] = str(exc)
    summary["stages"] = records
    summary["highest_stage_passed"] = PromotionStage(tracker.highest_stage_passed).name
    summary["halted"] = tracker.is_halted
    summary["halt_reason"] = tracker.halt_reason
    summary["tracker"] = tracker
    return summary


def run(panel_dir: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    development = load_panel(panel_dir / "panel_development.csv.gz")
    holdout = load_panel(panel_dir / "panel_holdout.csv.gz")
    panel = pd.concat([development, holdout], ignore_index=True)
    panel = panel.sort_values("observation_date").reset_index(drop=True)
    dates = panel["observation_date"].astype(str).tolist()
    holdout_start = str(holdout["observation_date"].iloc[0])
    split = (pd.Timestamp(holdout_start) + pd.Timedelta(days=CONTROL_HOLDOUT_DAYS)).date().isoformat()
    periods = {
        "dev_start": dates[0], "dev_end": str(development["observation_date"].iloc[-1]),
        "holdout_start": holdout_start, "holdout_end": dates[-1],
        "control_window_end": (pd.Timestamp(split) - pd.Timedelta(days=1)).date().isoformat(),
        "test_window_start": split,
    }
    windows = {
        "CPLUS-VOL": (holdout_start, periods["control_window_end"]),
        "CZERO-PLAC": None,
        "H1-OFI": (split, dates[-1]),
    }
    members = point_in_time_membership(panel, top_n=TOP_N, lookback_bars=LOOKBACK,
                                       rebalance_every=REBALANCE).to_numpy(dtype=bool)
    feat = features(panel)
    ledger = FullLineageExperimentLedger(persistence_path=out / "experiment_ledger.jsonl")

    # H1-OFI's sign: the sign of its pooled development IC. Both signs are
    # journaled as trials, because both were looked at.
    ofi_dev = pairs(feat, members, "ofi", "next_return", periods["dev_start"], periods["dev_end"])
    dev_ic = information_coefficient(ofi_dev["pred"], ofi_dev["target"])[0]
    sign = 1.0 if dev_ic >= 0 else -1.0

    results = {
        "preregistration": PREREGISTRATION_ID,
        "periods": periods,
        "universe": {"top_n": TOP_N, "lookback_bars": LOOKBACK, "rebalance_every": REBALANCE,
                     "assets_in_panel": len(feat["assets"]),
                     "assets_ever_members": int(members.any(axis=0).sum())},
        "h1_sign_rule": {"development_ic_of_raw_ofi": dev_ic, "sign": sign},
        "candidates": {},
    }
    lit_returns = ("reversal", "momentum")
    specs = [
        ("CPLUS-VOL", "rank(mean |r| over 20 bars) -> rank(|r_{t+1}|)", "vol20", "next_abs_return", 1.0),
        ("CZERO-PLAC", "rank(sha256(asset, date)) -> rank(r_{t+1})", "placebo", "next_return", 1.0),
        ("H1-OFI", f"rank({'+' if sign > 0 else '-'}OFI_t) -> rank(r_{{t+1}})", "ofi", "next_return", sign),
    ]
    trackers = {}
    for cid, formula, predictor, target, s in specs:
        res = run_candidate(cid, formula, feat, members, predictor, target, s, periods, windows[cid], out,
                            ledger, lit_returns)
        trackers[cid] = res.pop("tracker")
        results["candidates"][cid] = res

    # Stage 5, H1-OFI only, on the holdout's long-short returns.
    h1 = results["candidates"]["H1-OFI"]
    tracker = trackers["H1-OFI"]
    if tracker.highest_stage_passed >= PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER and not tracker.is_halted:
        trials = CandidateTrialLedger(persistence_path=out / "trial_ledger.jsonl")
        for variant in (1.0, -1.0):
            ls = long_short(feat, members, variant, periods["dev_start"], periods["dev_end"])
            trials.record_trial(f"H1-OFI{'+' if variant > 0 else '-'}", "prereg", "rank(OFI) long-short",
                                "TACTICAL", _sharpe(ls["gross"]), returns=ls["gross"], n_bars=len(ls["gross"]),
                                metadata={"search_id": PREREGISTRATION_ID})
        ls = long_short(feat, members, sign, periods["test_window_start"], periods["holdout_end"])
        traded = members[[i for i, d in enumerate(dates) if d >= periods["test_window_start"]]].any(axis=0)
        held_panel = panel[panel["observation_date"] >= periods["test_window_start"]]
        cols = ["observation_date"] + [c for a, keep in zip(feat["assets"], traded, strict=True) if keep
                                        for c in (f"{a}_CLOSE", f"{a}_VOL")]
        universe = audit_universe(held_panel[cols].reset_index(drop=True), selected_as_of=periods["test_window_start"],
                                  selection_basis="point_in_time")
        rec = tracker.verify_stage_5_from_returns(
            ls["gross"], ls["turnover"], np.asarray(ls["date"], dtype="datetime64[D]"),
            trial_ledger=trials, universe=universe,
        )
        h1["stages"].append(_record(rec))
        h1["highest_stage_passed"] = PromotionStage(tracker.highest_stage_passed).name
        h1["halted"], h1["halt_reason"] = tracker.is_halted, tracker.halt_reason
        h1["holdout_long_short"] = {"days": len(ls["gross"]), "gross_annual_sharpe": _sharpe(ls["gross"]),
                                    "mean_daily_turnover": float(np.mean(ls["turnover"])) if len(ls["turnover"]) else None,
                                    "universe_reasons": universe.reasons}
    for cid, tr in trackers.items():
        results["candidates"][cid]["evidence_backed_through_highest"] = tr.is_evidence_backed()
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str) + "\n",
                                      encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = run(args.panel_dir, args.out)
    for cid, res in results["candidates"].items():
        print(f"{cid}: highest stage passed {res['highest_stage_passed']}, halted={res['halted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
