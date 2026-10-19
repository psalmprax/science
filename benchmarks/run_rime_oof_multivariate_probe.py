#!/usr/bin/env python3
"""Registered replication: RIME-OOF-MV multivariate out-of-family probe.

Protocol frozen in docs/RIME_PREREGISTRATION_OOF_MV.md (PREREG-RIME-OOF-MV-2026-10-17).
Fresh seeds 600/601/603 (exploratory sandbox used 500/501/503, disclosed in the
prereg). One run; no tuning; classification rules exactly as preregistered.

Run: python3 -u benchmarks/run_rime_oof_multivariate_probe.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

JSON_OUT = ROOT / "benchmarks" / "rime_oof_multivariate_results.json"


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 4:
        return float("nan")
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def law_control(x1, x2, x3):
    y = np.zeros_like(x1)
    y[2:] = 2.0 * x1[1:-1] * x2[1:-1] + 3.0 * x3[:-2] + 10.0
    return y


def law_fn_of_interaction(x1, x2, x3):
    y = np.zeros_like(x1)
    y[2:] = 4.0 * np.sin(x1[1:-1] * x2[1:-1]) + 1.5 * x3[1:-1] + 2.0
    return y


def law_max_switch(x1, x2, x3):
    y = np.zeros_like(x1)
    y[2:] = 2.0 * np.maximum(x1[1:-1], x2[1:-1]) + 1.5 * x3[1:-1]
    return y


def law_amplitude_mod(x1, x2, x3):
    y = np.zeros_like(x1)
    y[2:] = x3[1:-1] * np.sin(1.5 * x1[1:-1]) + 2.0
    return y


LAWS = {
    "control_in_family": law_control,
    "oof_fn_of_interaction": law_fn_of_interaction,
    "oof_max_switch": law_max_switch,
    "oof_amplitude_mod": law_amplitude_mod,
}


def make_pop(seed: int, law, scale: float = 1.0, n: int = 600) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    x1, x2, x3 = scale * r.randn(n), scale * r.randn(n), scale * r.randn(n)
    y = law(x1, x2, x3)
    y[2:] += 0.05 * r.randn(n - 2)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def run_rime(df):
    p = RimeProposer(
        target_col="target", budget=200, max_depth=2, additive_terms=2,
        max_seeds=120, lagged_products=True, gamma=1.0,
    )
    t0 = time.perf_counter()
    props = list(p.fit_and_propose(df))
    return props, (time.perf_counter() - t0) * 1000.0


def run_rime_ad(df):
    p = RimeAdProposer(target_col="target")
    t0 = time.perf_counter()
    props = list(p.fit_and_propose(df))
    return props, (time.perf_counter() - t0) * 1000.0


ENGINES = {"RimeProposer": run_rime, "RimeAdProposer": run_rime_ad}


def classify(n_props: int, test_r2: float, wide_r2: float) -> str:
    if n_props == 0:
        return "ABSTAINED"
    if np.isfinite(test_r2) and test_r2 >= 0.95 and np.isfinite(wide_r2) and wide_r2 >= 0.85:
        return "RECOVERED"
    if np.isfinite(test_r2) and test_r2 >= 0.5:
        return "CONFIDENT_SURROGATE"
    return "NO_SIGNAL"


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    results = []
    for dname, law in LAWS.items():
        df_tr = make_pop(600, law).iloc[:400].reset_index(drop=True)
        df_te = make_pop(600, law).iloc[400:].reset_index(drop=True)
        df_fr = make_pop(601, law)
        df_wide = make_pop(603, law, scale=2.5)

        for ename, runner in ENGINES.items():
            rec = {"law": dname, "engine": ename}
            try:
                props, ms = runner(df_tr)
                rec["wall_ms"] = round(ms, 1)
                rec["n_props"] = len(props)
                if props:
                    scored = []
                    for i, cand in enumerate(props):
                        try:
                            te = _r2(df_te["target"].to_numpy(float), np.asarray(cand.compute_score(df_te), float).ravel())
                            fr = _r2(df_fr["target"].to_numpy(float), np.asarray(cand.compute_score(df_fr), float).ravel())
                            wd = _r2(df_wide["target"].to_numpy(float), np.asarray(cand.compute_score(df_wide), float).ravel())
                        except Exception:
                            continue
                        scored.append({
                            "list_rank": i, "test_r2": te, "fresh_r2": fr, "wide_r2": wd,
                            "formula": (cand.formula or "")[:120],
                            "inputs_used": cand.metadata.get("inputs_used_names"),
                            "identifiability_status": cand.parameters.get("identifiability_status"),
                        })
                    if scored:
                        best = max(scored, key=lambda r: (r["test_r2"] if np.isfinite(r["test_r2"]) else -9))
                        rec["best"] = best
                        rec["classification"] = classify(len(props), best["test_r2"], best["wide_r2"])
                if "classification" not in rec:
                    rec["classification"] = classify(rec["n_props"], float("nan"), float("nan"))
            except Exception:
                rec["classification"] = "ERROR"
                rec["error"] = traceback.format_exc().splitlines()[-1][:160]
            results.append(rec)

    harness_ok = any(
        r["law"] == "control_in_family" and r["engine"] == "RimeProposer"
        and r.get("classification") == "RECOVERED"
        for r in results
    )

    payload = {
        "prereg_id": "PREREG-RIME-OOF-MV-2026-10-17",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {"git_head": git_head(), "python": sys.version.split()[0],
                       "numpy": np.__version__, "pandas": pd.__version__},
        "protocol": {
            "seeds": {"train_test": 600, "fresh": 601, "wide": 603},
            "train_rows": "0:400", "test_rows": "400:600", "n": 600,
            "noise": 0.05, "wide_scale": 2.5,
            "rime_proposer_config": {
                "budget": 200, "max_depth": 2, "additive_terms": 2,
                "max_seeds": 120, "lagged_products": True, "gamma": 1.0,
            },
        },
        "harness_sanity_control_recovered": harness_ok,
        "results": results,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 110)
    print("RIME-OOF-MV registered run — PREREG-RIME-OOF-MV-2026-10-17")
    print(f"harness sanity (control RECOVERED under RimeProposer): {harness_ok}")
    print("=" * 110)
    for r in results:
        b = r.get("best")
        print(f"\n[{r['law']}] engine={r['engine']}  wall={r.get('wall_ms')}ms  "
              f"n_props={r.get('n_props')}  class={r['classification']}")
        if b:
            print(f"  test={b['test_r2']:.4f}  fresh={b['fresh_r2']:.4f}  wide={b['wide_r2']:.4f}  "
                  f"ident={b['identifiability_status']}  inputs={b['inputs_used']}")
            print(f"  formula: {b['formula']}")
        if r["classification"] == "ERROR":
            print(f"  error: {r.get('error')}")
    print(f"\nJSON written to {JSON_OUT}")


if __name__ == "__main__":
    main()
