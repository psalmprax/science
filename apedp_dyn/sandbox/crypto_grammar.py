"""Grammar-driven discovery on combined 15-asset crypto universe (CRYPTO-PRICE).

Discovers dynamic rate laws and second-order price mechanics on a synchronized,
combined 15-asset crypto dataset containing BTC, ETH, SOL, BNB, XRP, ADA, AVAX,
DOGE, DOT, LINK, MATIC, NEAR, ATOM, UNI, LTC, plus an equal-weighted 15-token
composite basket index (COMPOSITE_TOP15).

Like FIN-PRICE (S&P 500), this wires the genuine mechanical grammar search:
  enumerate (22 structures) -> prune_admissible (gates 1-2) ->
  fit_candidate (gates 3-4) -> sovereign adjudicate (bootstrap tiebreak +
  residual whiteness + epistemic floor tau_epi).

Regimes:
  - Train: 2021-01-01 to 2021-12-31 (the 2021 bull cycle and ATH expansion)
  - Holdout: 2022-05-05 to 2022-05-25 (the Terra-LUNA liquidation cascade /
    crypto systemic shock window)
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apedp_dyn.epistemic.adjudicate import Adjudication, adjudicate  # noqa: E402
from apedp_dyn.epistemic.bootstrap import holdout_residual_series  # noqa: E402
from apedp_dyn.fitting.parameter_fit import fit_candidate  # noqa: E402
from apedp_dyn.grammar.mechanical_grammar import (  # noqa: E402
    enumerate_candidates,
    prune_admissible,
)
from apedp_dyn.problems.base import DynProblem  # noqa: E402
from apedp_dyn.problems.finance import spec_from_series  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market"
CRYPTO_DATA_FILE = DATA_DIR / "crypto_top15.csv"

CRYPTO_ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "AVAX",
    "DOGE",
    "DOT",
    "LINK",
    "MATIC",
    "NEAR",
    "ATOM",
    "UNI",
    "LTC",
)

FIN_TERMS = (
    "one",
    "x",
    "x3",
    "v",
    "abs_v_v",
    "sin_x",
    "u",
    "sat_x",
    "sat_v",
    "sat_u",
    "inv_x2",
    "u_relax",
)

# Shock regimes usable as holdout: key -> (label, first date, last date).
HOLDOUT_WINDOWS: dict[str, tuple[str, str, str]] = {
    "luna": ("Terra-LUNA shock", "2022-05-05", "2022-05-25"),
    "ftx": ("FTX collapse", "2022-11-06", "2022-11-20"),
}


def load_crypto_dataset(path: str | Path | None = None) -> dict[str, Any]:
    """Load synchronized historical daily closes and order flow for 15 assets + composite basket."""
    fpath = Path(path) if path is not None else CRYPTO_DATA_FILE
    if not fpath.exists():
        raise FileNotFoundError(f"Crypto multi-asset data file not found at {fpath}")

    dates: list[str] = []
    assets: dict[str, list[float]] = {sym: [] for sym in CRYPTO_ASSETS}
    assets["COMPOSITE_TOP15"] = []
    ofi: dict[str, list[float]] = {sym: [] for sym in CRYPTO_ASSETS}
    ofi["COMPOSITE_TOP15"] = []

    with open(fpath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row["observation_date"]
            dates.append(dt)
            for k in assets:
                try:
                    assets[k].append(float(row[k]))
                except (KeyError, ValueError):
                    assets[k].append(np.nan)
            for k in ofi:
                col = f"{k}_OFI"
                if col in row:
                    try:
                        ofi[k].append(float(row[col]))
                    except (KeyError, ValueError):
                        ofi[k].append(0.0)
                else:
                    ofi[k].append(0.0)

    asset_arrays = {k: np.array(v, dtype=float) for k, v in assets.items()}
    ofi_arrays = {k: np.array(v, dtype=float) for k, v in ofi.items()}
    return {
        "dates": dates,
        "assets": asset_arrays,
        "ofi": ofi_arrays,
        "composite": asset_arrays["COMPOSITE_TOP15"],
        "symbols": list(CRYPTO_ASSETS),
        "path": str(fpath),
    }


def run_crypto_experiment(
    target_asset: str = "COMPOSITE_TOP15",
    experiment_name: str = "Crypto Top-15 Basket Bull->Crash",
    path: str | Path | None = None,
    holdout_window: str = "luna",
) -> tuple[Adjudication, DynProblem, dict[str, Any]]:
    """Run grammar discovery on the designated crypto series across bull vs shock.

    ``holdout_window`` picks the shock regime ("luna" default, "ftx"). One verdict
    on one 21-day event is thin evidence; running both windows and comparing
    verdicts is the cheap robustness check.
    """
    ds = load_crypto_dataset(path)
    dates = ds["dates"]
    assets = ds["assets"]
    ofi = ds.get("ofi", {})

    target_key = target_asset.upper() if target_asset.upper() in assets else "COMPOSITE_TOP15"
    px = assets[target_key]
    of = ofi.get(target_key)

    if holdout_window not in HOLDOUT_WINDOWS:
        raise ValueError(
            f"unknown holdout_window {holdout_window!r}; choose from {sorted(HOLDOUT_WINDOWS)}")
    ho_label, ho_start_date, ho_end_date = HOLDOUT_WINDOWS[holdout_window]

    idx = {d: i for i, d in enumerate(dates)}
    # Bull market train: 2021-01-01 to 2021-12-31
    i_tr_start = idx.get("2021-01-01", 0)
    i_tr_end = idx.get("2021-12-31", 364) + 1

    # Shock holdout (default: 2022-05-05 to 2022-05-25, Terra-LUNA liquidation cascade)
    if ho_start_date not in idx or ho_end_date not in idx:
        raise ValueError(
            f"holdout window {holdout_window!r} ({ho_start_date}..{ho_end_date}) "
            "is outside the loaded dataset")
    i_ho_start = idx[ho_start_date]
    i_ho_end = idx[ho_end_date] + 1

    prob = spec_from_series(
        px,
        (i_tr_start, i_tr_end),
        (i_ho_start, i_ho_end),
        problem_id=f"CRYP-{target_key}",
        orderflow=of,
    )

    print(f"--- Experiment {experiment_name} [{target_key}] ---")
    cands = enumerate_candidates(
        required=(), optional=FIN_TERMS, max_terms=2, id_prefix="cryp"
    )
    print(f"  enumerated: {len(cands)} grammar structures")
    kept = prune_admissible(prob, cands)
    rej = len(cands) - len(kept)
    print(f"  gates 1-2 survivors: {len(kept)} (rejected: {rej})")

    fitted = []
    n_fit_failed = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        for s in kept:
            try:
                f = fit_candidate(prob, s.candidate)
            except Exception as e:  # noqa: BLE001
                n_fit_failed += 1
                print(f"    fit failed {s.candidate.candidate_id}: {type(e).__name__}")
                continue
            fitted.append(f)
    n_numeric_warnings = sum(
        1 for w in caught
        if issubclass(w.category, RuntimeWarning)
        and any(k in str(w.message) for k in ("overflow", "invalid value", "divide by zero"))
    )
    if n_numeric_warnings:
        print(f"  numeric warnings during fitting: {n_numeric_warnings} "
              "(overflow/invalid; unstable candidate simulations)")

    g4 = [s for s in fitted if s.provenance.get("gate4_status", {}).get("passed", False)]
    print(f"  gate-4 survivors: {len(g4)} / {len(fitted)} fitted")

    def series(sc):
        try:
            if sc.fit is None:
                return None
            return holdout_residual_series(prob, sc.candidate, sc.fit.param_values)
        except (RuntimeError, ValueError, TypeError):
            return None

    ranked = sorted(
        [s for s in g4 if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: float(s.train_loss) if s.train_loss is not None else float("inf"),
    )
    se = {id(s): series(s) for s in ranked}
    best_se = se.get(id(ranked[0])) if ranked else None
    rival_se = se.get(id(ranked[1])) if len(ranked) > 1 else None

    try:
        adj = adjudicate(
            ranked,
            resid_best=best_se,
            resid_rival=rival_se,
            resid_sole=best_se,
            bootstrap=True,
            bootstrap_n_boot=1000,
            bootstrap_seed=7000,
            data_origin="experimental",
        )
    except TypeError:
        adj = adjudicate(ranked)

    print(f"  verdict: {adj.status.value}")
    if adj.promoted is not None:
        terms = sorted(t.name for t in adj.promoted.candidate.terms)
        print(
            f"  DISCOVERED: {adj.promoted.candidate.candidate_id} "
            f"terms={terms} train_loss={adj.promoted.train_loss:.4g} "
            f"holdout_rmse={adj.promoted.holdout_rmse:.4g}"
        )
    print(f"  reason: {adj.reason[:110]}")

    seg_dates = dates[i_ho_start:i_ho_end]
    seg_px = px[i_ho_start:i_ho_end]
    cand_pool = ranked if ranked else sorted(
        [s for s in fitted if getattr(s, "holdout_rmse", None) is not None],
        key=lambda s: float(s.holdout_rmse) if s.holdout_rmse is not None else float("inf"),
    )
    series_info = {
        "x": list(range(len(seg_px))),
        "dates": seg_dates,
        "actual": seg_px.tolist(),
        "x_label": f"{target_key} {ho_label} window (USD)",
        "target": target_key,
        "candidates": cand_pool,
        "holdout_window": holdout_window,
        "holdout_window_label": ho_label,
        "holdout_n": len(seg_px),
        # Run metadata. The server used to hard-code "29 structures"; the real
        # enumeration was 46, then 79 after the grammar grew.
        "n_structures": len(cands),
        "n_gate12_survivors": len(kept),
        "n_fit_attempted": len(kept),
        "n_fit_failed": n_fit_failed,
        "n_numeric_warnings": n_numeric_warnings,
        "n_gate4_survivors": len(g4),
    }

    return adj, prob, series_info


def main():
    """Run CLI benchmark across composite basket and top assets."""
    print(f"Loaded crypto 15-asset dataset from {CRYPTO_DATA_FILE}")
    adj, _prob, _info = run_crypto_experiment("COMPOSITE_TOP15")
    print(f"\nComposite verdict: {adj.status.value}, best RMSE: {adj.best_rmse}")

    for sym in ("BTC", "ETH", "SOL"):
        print(f"\nRunning {sym} individual series:")
        a, _, _ = run_crypto_experiment(sym)
        print(f"{sym} status: {a.status.value}, best RMSE: {a.best_rmse}")


if __name__ == "__main__":
    main()
