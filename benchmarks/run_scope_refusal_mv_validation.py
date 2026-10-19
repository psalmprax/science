#!/usr/bin/env python3
"""Spec §16e validation run (prereg PREREG-RIME-SCOPE-MV-2026-10-17).

Frozen protocol: fresh noise seeds {700, 701, 702}, one repetition of the full
two-sided suite per seed. Multivariate worlds are aligned frames (600 rows:
x1_l1 = x1[t-1], x2_l1, x3_l1, x3_l2 = x3[t-2]; train 0:400, holdout 400:600,
noise 0.05). Emitted = RimeProposer house-config list-top on aligned dev
(dev-BIC ranking, never holdout-picked). Probes = build_family_probes_mv.
Adjudication = §16d `adjudicate_scope` defaults (bootstrap n=1000, seed=7).

Pass criteria (frozen): ALL W declined AND ALL G promoted on ALL seeds, and
determinism (same seed -> identical verdicts and probe fits across re-runs).
The family-set rule excludes guard-failing families mechanically; pass
criteria are then evaluated on the final set. ANY failure is reported as a
negative result; modifying the prereg means v2, not a patch.

Run: python3 -u benchmarks/run_scope_refusal_mv_validation.py
"""
from __future__ import annotations

import hashlib
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

from apedp_dyn.epistemic.scope_refusal import adjudicate_scope, scored_from_predict
from apedp_dyn.epistemic.scope_refusal_mv import PROBE_MV_IDS, build_family_probes_mv
from apedp_dyn.proposers.rime_proposer import RimeProposer
from benchmarks.known_truth_challenge import build_worlds

JSON_OUT = ROOT / "benchmarks" / "scope_refusal_mv_validation_results.json"

SEEDS = (700, 701, 702)
FEATS = ["x1_l1", "x2_l1", "x3_l1", "x3_l2"]
NOISE = 0.05
HOUSE = dict(target_col="target", budget=200, max_depth=2, additive_terms=2,
             max_seeds=120, lagged_products=True, gamma=1.0)
G2_CELLS = (("d2_C1000_uniform_1x", 2), ("d3_C100_uniform_1x", 3),
            ("d6_C1e12_log_1x", 6), ("d2_C1000_log_1x", 2))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def law_control(a, b, c, d):
    return 2.0 * a * b + 3.0 * d + 10.0


def law_w1_sin_interaction(a, b, c, d):
    return 4.0 * np.sin(a * b) + 1.5 * c + 2.0


def law_w2_max_switch(a, b, c, d):
    return 2.0 * np.maximum(a, b) + 1.5 * c


def law_w3_am_product(a, b, c, d):
    return c * np.sin(1.5 * a) + 2.0


W_LAWS = {
    "w1_sin_interaction": law_w1_sin_interaction,
    "w2_max_switch": law_w2_max_switch,
    "w3_am_product": law_w3_am_product,
}


def aligned_frame(seed: int, law, n: int = 602) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = np.random.RandomState(seed)
    x1, x2, x3 = r.randn(n), r.randn(n), r.randn(n)
    noise = NOISE * r.randn(n - 2)
    a, b, c, d = x1[1:-1], x2[1:-1], x3[1:-1], x3[:-2]
    y = law(a, b, c, d) + noise
    df = pd.DataFrame({"x1_l1": a, "x2_l1": b, "x3_l1": c, "x3_l2": d, "target": y})
    # Prereg split is explicit: train 0:400, holdout 400:600 (n=600). The imported
    # known_truth DEV_ROWS/HOLDOUT_ROWS constants (600/300) are for the 16c axes,
    # not for these aligned frames.
    return (df.iloc[:400].reset_index(drop=True),
            df.iloc[400:600].reset_index(drop=True))


def null_frame(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev, hold = aligned_frame(seed, lambda a, b, c, d: np.zeros_like(a))
    rt = np.random.RandomState(seed + 777)
    for df in (dev, hold):
        df["target"] = NOISE * rt.randn(len(df))
    return dev, hold


def emit_list_top(dev: pd.DataFrame, hold: pd.DataFrame):
    p = RimeProposer(**HOUSE)
    t0 = time.perf_counter()
    props = list(p.fit_and_propose(dev))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if not props:
        return None, wall_ms, {"n_props": 0, "classification": "ABSTAINED"}
    top = props[0]

    def predict(fd: pd.DataFrame) -> np.ndarray:
        return np.asarray(top.compute_score(fd), dtype=float).ravel()

    emitted = scored_from_predict("emitted_list_top", predict, dev, hold,
                                  noise_sigma=NOISE)
    info = {
        "n_props": len(props),
        "formula": (top.formula or "")[:160],
        "inputs_used": top.metadata.get("inputs_used_names"),
        "identifiability_status": top.parameters.get("identifiability_status"),
        "dev_r2": _r2(dev["target"].to_numpy(float), predict(dev)),
        "hold_r2": _r2(hold["target"].to_numpy(float), predict(hold)),
    }
    return emitted, wall_ms, info


def referee(dev: pd.DataFrame, hold: pd.DataFrame, emitted, probes, sigma: float):
    hold_y = hold["target"].to_numpy(float)
    res = adjudicate_scope(emitted, probes, holdout_scale={
        "obs_rms": float(np.sqrt(np.mean(hold_y ** 2))),
        "noise_sigma": sigma, "n_samples": len(hold_y)})
    return res.to_dict()


def probe_metrics(probes) -> list[dict]:
    return [{"probe": pr.candidate.candidate_id,
             "train_loss": float(getattr(pr, "train_loss", float("nan"))),
             "holdout_rmse": float(getattr(pr, "holdout_rmse", float("nan")))}
            for pr in probes]


def mv_cell(seed: int, name: str) -> dict:
    law = law_control if name == "control" else W_LAWS[name]
    dev, hold = aligned_frame(seed, law)
    emitted, wall_ms, info = emit_list_top(dev, hold)
    rec = {"cell": name, "seed": seed, "kind": "mv", "wall_ms": round(wall_ms, 1), **info}
    if emitted is None:
        return rec
    probes = build_family_probes_mv(dev, hold, feature_cols=FEATS, noise_sigma=NOISE)
    rec["probe_families"] = [pr.candidate.candidate_id for pr in probes]
    rec["probe_fits"] = probe_metrics(probes)
    rec["adjudication"] = referee(dev, hold, emitted, probes, NOISE)
    return rec


def null_cell(seed: int) -> dict:
    dev, hold = null_frame(seed)
    emitted, wall_ms, info = emit_list_top(dev, hold)
    rec = {"cell": "null_noise", "seed": seed, "kind": "g3",
           "wall_ms": round(wall_ms, 1), **info}
    if emitted is None:
        rec["adjudication"] = None
        return rec
    probes = build_family_probes_mv(dev, hold, feature_cols=FEATS, noise_sigma=NOISE)
    rec["probe_families"] = [pr.candidate.candidate_id for pr in probes]
    rec["probe_fits"] = probe_metrics(probes)
    rec["adjudication"] = referee(dev, hold, emitted, probes, NOISE)
    return rec


def g2_cell(seed: int, world_name: str, k: int) -> dict:
    worlds = {bw.world.name: bw for bw in build_worlds(noise_seed=seed)}
    bw = worlds[world_name]
    dev, hold = bw.dev, bw.hold
    x = dev["p"].to_numpy(float)
    y = dev["target"].to_numpy(float)
    w, *_ = np.linalg.lstsq(np.column_stack([x ** k]), y, rcond=None)
    alpha = float(w[0])
    emitted = scored_from_predict(
        f"emitted_p{k}",
        lambda fd: np.asarray(fd["p"].to_numpy(float) ** k * alpha),
        dev, hold, noise_sigma=bw.sigma)
    rec = {"cell": f"g2_{world_name}_k{k}", "seed": seed, "kind": "g2_mv_p1",
           "formula": f"{alpha:.6g} * p**{k}"}
    try:
        probes = build_family_probes_mv(dev, hold, feature_cols=["p"], noise_sigma=bw.sigma)
    except ValueError as e:
        if "no multivariate probe family" not in str(e):
            raise
        # prereg: if NO probe can be built, adjudication is undefined and the
        # harness raises. Record the raise as the cell outcome for the record.
        rec["status"] = "undefined_no_probe_buildable"
        rec["undefined_reason"] = str(e)
        rec["probe_families"] = []
        rec["adjudication"] = None
        return rec
    rec["probe_families"] = [pr.candidate.candidate_id for pr in probes]
    rec["probe_fits"] = probe_metrics(probes)
    rec["adjudication"] = referee(dev, hold, emitted, probes, bw.sigma)
    return rec


def digest(rec: dict) -> str:
    parts = [rec.get("cell"), rec.get("seed"), rec.get("formula", "")]
    adj = rec.get("adjudication") or {}
    parts += [adj.get("scope_verdict", "none"), adj.get("emitted_id", "none")]
    for p in adj.get("probes", []):
        parts.append(f"{p.get('probe')}|{p.get('status')}|{p.get('verdict')}")
    for pf in rec.get("probe_fits", []):
        parts.append(f"{pf['probe']}|{pf['train_loss']:.12e}|{pf['holdout_rmse']:.12e}")
    h = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


def declined_families(rec: dict) -> list[str]:
    adj = rec.get("adjudication") or {}
    return sorted({p.get("probe") for p in adj.get("probes", [])
                   if p.get("verdict") == "declined"})


def main() -> None:
    t_start = time.perf_counter()
    cells: list[dict] = []
    errors: list[dict] = []

    for seed in SEEDS:
        for name in ("control", *W_LAWS):
            try:
                cells.append(mv_cell(seed, name))
            except Exception:
                errors.append({"cell": name, "seed": seed,
                               "error": traceback.format_exc().splitlines()[-1][:200]})
        try:
            cells.append(null_cell(seed))
        except Exception:
            errors.append({"cell": "null_noise", "seed": seed,
                           "error": traceback.format_exc().splitlines()[-1][:200]})
        for world_name, k in G2_CELLS:
            try:
                cells.append(g2_cell(seed, world_name, k))
            except Exception:
                errors.append({"cell": f"g2_{world_name}_k{k}", "seed": seed,
                               "error": traceback.format_exc().splitlines()[-1][:200]})

    # determinism: re-run two cells (same seed, independent construction)
    det_pairs = [(700, "control"), (700, "w1_sin_interaction")]
    determinism = []
    for seed, name in det_pairs:
        a, b = mv_cell(seed, name), mv_cell(seed, name)
        determinism.append({"cell": name, "seed": seed,
                            "digest_a": digest(a), "digest_b": digest(b),
                            "identical": digest(a) == digest(b)})
    determinism_ok = all(d["identical"] for d in determinism)

    # family-set rule (mechanical): family fails (a) iff it declined any G cell
    guard_failures: dict[str, list[str]] = {f: [] for f in PROBE_MV_IDS}
    w_contributions: dict[str, list[str]] = {f: [] for f in PROBE_MV_IDS}
    for rec in cells:
        fams = declined_families(rec)
        if rec["cell"] == "control" or rec["kind"] == "g2_mv_p1":
            for f in fams:
                guard_failures.setdefault(f, []).append(rec["cell"])
        elif rec["cell"].startswith("w"):
            for f in fams:
                w_contributions.setdefault(f, []).append(rec["cell"])
    final_set = [f for f in PROBE_MV_IDS
                 if not guard_failures.get(f) and w_contributions.get(f)]
    excluded = {f: {"guard_failures": guard_failures.get(f, []),
                    "w_declines": w_contributions.get(f, [])}
                for f in PROBE_MV_IDS if f not in final_set}

    # pass criteria evaluated on the final family set (prereg rule); the raw
    # readings (declared probe set, before the family-set rule) are recorded too
    raw_w = {f"{r['cell']}|{r['seed']}": (r.get("adjudication") or {}).get("scope_verdict")
             for r in cells if r["cell"].startswith("w")}
    final_w = {f"{r['cell']}|{r['seed']}": any(f in final_set for f in declined_families(r))
               for r in cells if r["cell"].startswith("w")}
    g1 = [r for r in cells if r["cell"] == "control"]
    g2 = [r for r in cells if r["kind"] == "g2_mv_p1"]
    g2_defined = [r for r in g2 if r.get("adjudication")]
    g3 = [r for r in cells if r["kind"] == "g3"]

    w_ok = len(final_w) == 9 and all(final_w.values())
    w_raw_ok = len(raw_w) == 9 and all(v == "declined" for v in raw_w.values())
    # final-set reading: recompute the verdict from decline membership in the
    # retained families only (prereg: the rule is mechanical, not re-tuned)
    g1_final_ok = (len(g1) == 3 and all(
        not any(f in final_set for f in declined_families(r)) for r in g1))
    g1_raw_ok = (len(g1) == 3 and all(
        (r.get("adjudication") or {}).get("scope_verdict") == "promoted" for r in g1))
    g2_final_ok = (len(g2) == 12 and len(g2_defined) == 12 and all(
        not any(f in final_set for f in declined_families(r)) for r in g2_defined))
    g2_raw_ok = (len(g2) == 12 and len(g2_defined) == 12 and all(
        (r.get("adjudication") or {}).get("scope_verdict") == "promoted"
        for r in g2_defined))
    g3_ok = all(r.get("classification") == "ABSTAINED"
                or (r.get("adjudication") or {}).get("scope_verdict") == "declined"
                for r in g3)

    pass_all = bool(w_ok and g1_final_ok and g2_final_ok and g3_ok
                    and determinism_ok and not errors and final_set)

    payload = {
        "prereg_id": "PREREG-RIME-SCOPE-MV-2026-10-17",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "protocol": {
            "seeds": list(SEEDS), "rows": {"dev": "0:400", "hold": "400:600"},
            "n": 600, "noise": NOISE, "feature_cols": FEATS,
            "house_config": HOUSE,
            "frame_rule": ("RandomState(seed): x1,x2,x3 ~ randn(602), noise ~ "
                           "0.05*randn(600); aligned columns a=x1[1:-1], b=x2[1:-1], "
                           "c=x3[1:-1], d=x3[:-2]; y = law(a,b,c,d) + noise; "
                           "null target: RandomState(seed+777) * 0.05, independent"),
            "g2_cells": [f"{w}_k{k}" for w, k in G2_CELLS],
            "adjudication": {"bootstrap": True, "n_boot": 1000, "bootstrap_seed": 7},
        },
        "cells": cells,
        "determinism": determinism,
        "determinism_ok": determinism_ok,
        "family_set": {"declared": list(PROBE_MV_IDS), "final": final_set,
                       "excluded": excluded},
        "criteria": {"w_all_declined_final_set": w_ok,
                     "w_all_declined_raw": w_raw_ok,
                     "g1_promoted_final_set": g1_final_ok,
                     "g1_promoted_raw": g1_raw_ok,
                     "g2_monomials_promoted_final_set": g2_final_ok,
                     "g2_monomials_promoted_raw": g2_raw_ok,
                     "g2_cells_defined": f"{len(g2_defined)}/12",
                     "g3_no_promotion": g3_ok},
        "raw_w_verdicts": raw_w,
        "final_w_declined": final_w,
        "errors": errors,
        "pass": pass_all,
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 104)
    print("Spec §16e validation — PREREG-RIME-SCOPE-MV-2026-10-17")
    print(f"seeds {SEEDS}  |  pass={pass_all}  |  determinism={determinism_ok}  |  "
          f"wall={payload['wall_s']}s")
    print("=" * 104)
    for r in cells:
        adj = r.get("adjudication") or {}
        v = adj.get("scope_verdict") or r.get("status") or r.get("classification", "-")
        dec = ",".join(declined_families(r)) or "-"
        print(f"[{r['cell']:<34} s{r['seed']}] {v:<9} "
              f"props={r.get('n_props', 0):<3} declines={dec}")
        if r["cell"] in ("control", "w1_sin_interaction", "w2_max_switch", "w3_am_product"):
            print(f"   formula: {r.get('formula', '-')[:96]}")
    print(f"\nfinal family set: {final_set}")
    for f, info in excluded.items():
        print(f"  excluded {f}: guard_failures={info['guard_failures']} "
              f"w_declines={info['w_declines']}")
    print(f"criteria: {payload['criteria']}")
    print(f"JSON written to {JSON_OUT}")


if __name__ == "__main__":
    main()
