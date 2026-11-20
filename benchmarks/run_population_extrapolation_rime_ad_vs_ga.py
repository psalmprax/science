#!/usr/bin/env python3
"""Population Extrapolation Benchmark: RIME and RIME-AD vs a genetic term search.

Until 2026-11-19 this benchmark ran ``RimeProposer`` (RIME) and reported it as
RIME-AD; RIME-AD itself was never run (SYSTEM_FLAWS_AUDIT M-05). Each engine now
runs under its own name, and the "GA" arms are named for what they are: a
genetic search over which columns of a fixed term library enter a least-squares
fit, not genetic programming over expressions.

Tests multi-population generalization across:
  - Pop A-Train: Discovery partition (seed 500, rows 0:400)
  - Pop A-Test: In-distribution holdout (seed 500, rows 400:600)
  - Pop B: Fresh random realization, same mechanism (seed 501, 600 rows)
  - Pop C: Parameter shift (seed 502, c1=2.5, c2=2.0, 600 rows)
  - Pop D: Out-of-Distribution heavy-tailed inputs (x ~ Laplace(0, 1.5), 600 rows)
  - Distractor Challenge: 7 irrelevant noise channels added to input

Generating Ground Truth:
  y[t] = 2.0 * x1[t-1]*x2[t-1] + 3.0 * x3[t-2] + 10.0 + noise

Outputs:
  - benchmarks/population_extrapolation_results.json
  - docs/population_extrapolation_rime_ad_vs_ga.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

import apedp_kernel as ak
from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer
from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.epistemic.report_fidelity import render_float, render_verdict_cell, verdict_for_row

ROOT = Path(__file__).resolve().parent.parent
JSON_OUT = ROOT / "benchmarks" / "population_extrapolation_results.json"
MD_OUT = ROOT / "docs" / "population_extrapolation_rime_ad_vs_ga.md"


def _make_population(seed: int, c1: float = 2.0, c2: float = 3.0, n: int = 600, laplace: bool = False) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    if laplace:
        x1 = r.laplace(0.0, 1.5, n)
        x2 = r.laplace(0.0, 1.5, n)
        x3 = r.laplace(0.0, 1.5, n)
    else:
        x1, x2, x3 = r.randn(n), r.randn(n), r.randn(n)
    y = np.zeros(n)
    y[2:] = c1 * (x1[1:-1] * x2[1:-1]) + c2 * x3[:-2] + 10.0 + 0.05 * r.randn(n - 2)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _lagcol(col: np.ndarray, k: int) -> np.ndarray:
    col = np.asarray(col, dtype=float)
    v = np.full(len(col), np.nan)
    if k > 0:
        v[k:] = col[:-k]
    else:
        v[:] = col
    return v


def _ga_basis(df: pd.DataFrame, level: int) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {"1": np.ones(len(df))}
    for c in ("x1", "x2", "x3"):
        raw = np.asarray(df[c], dtype=float)
        d[c] = raw
        for k in (1, 2):
            d[f"{c}[t-{k}]"] = _lagcol(raw, k)
    if level >= 1:
        for c in ("x1", "x2", "x3"):
            for k in (0, 1, 2):
                key = c if k == 0 else f"{c}[t-{k}]"
                d[f"{key}^2"] = d[key] ** 2
    if level >= 2:
        d["x1[t-1]*x2[t-1]"] = d["x1[t-1]"] * d["x2[t-1]"]
        d["x1[t-2]*x2[t-2]"] = d["x1[t-2]"] * d["x2[t-2]"]
        d["x1*x2"] = d["x1"] * d["x2"]
        d["x1*x3"] = d["x1"] * d["x3"]
        d["x2*x3"] = d["x2"] * d["x3"]
    return d


def _calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 4:
        return float("nan")
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 0.0


def _run_ga(df_tr: pd.DataFrame, y_tr: np.ndarray, level: int, generations: int = 30) -> tuple[dict[str, float], float, np.ndarray, list[str]]:
    rng = np.random.RandomState(42)
    d = _ga_basis(df_tr, level)
    names = list(d.keys())
    Z = np.column_stack([d[k] for k in names])
    lam = 1e-6

    def fit(mask: np.ndarray):
        M = Z[:, mask]
        ok = np.isfinite(M).all(axis=1) & np.isfinite(y_tr)
        if int(ok.sum()) < int(mask.sum()) + 1:
            return None, np.inf
        w, *_ = np.linalg.lstsq(M[ok], y_tr[ok], rcond=None)
        pred = M[ok] @ w
        mse = float(np.mean((y_tr[ok] - pred) ** 2))
        return w, mse + lam * int(mask.sum())

    t0 = time.perf_counter()
    pop: list[np.ndarray] = []
    while len(pop) < 40:
        m = rng.rand(len(names)) < 0.3
        if not m.any():
            continue
        pop.append(m)

    for _ in range(generations):
        pop = sorted(pop, key=lambda m: fit(m)[1])
        nxt = [m.copy() for m in pop[:6]]
        while len(nxt) < 40:
            a = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            b = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            cut = rng.randint(1, len(names))
            child = np.concatenate([a[:cut], b[cut:]])
            if rng.rand() < 0.15:
                pos = rng.randint(0, len(names))
                child[pos] = not child[pos]
            if child.any():
                nxt.append(child)
        pop = nxt

    pop = sorted(pop, key=lambda m: fit(m)[1])
    best_mask = pop[0]
    weights, best_loss = fit(best_mask)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    active_names = [names[i] for i in range(len(names)) if best_mask[i]]
    return {"weights": weights, "mask": best_mask}, wall_ms, weights, active_names


def eval_ga_on_df(df: pd.DataFrame, level: int, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    d = _ga_basis(df, level)
    names = list(d.keys())
    Z = np.column_stack([d[k] for k in names])
    M = Z[:, mask]
    return M @ weights


def run_benchmark():
    np.random.seed(42)
    # 1. Generate Populations
    df_A_full = _make_population(500, c1=2.0, c2=3.0, n=600)
    df_A_tr = df_A_full.iloc[:400].reset_index(drop=True)
    df_A_te = df_A_full.iloc[400:].reset_index(drop=True)
    df_B = _make_population(501, c1=2.0, c2=3.0, n=600)
    df_C = _make_population(502, c1=2.5, c2=2.0, n=600) # Parameter shift
    df_D = _make_population(503, c1=2.0, c2=3.0, n=600, laplace=True) # Heavy-tailed distribution shift

    y_A_tr = df_A_tr["target"].to_numpy()

    results = {
        "metadata": {
            "title": "Population Extrapolation: RIME-AD vs GA/GP Benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_workspace": str(ROOT),
            "generating_truth": "y[t] = 2.0*x1[t-1]*x2[t-1] + 3.0*x3[t-2] + 10.0",
        },
        "methods": {},
    }

    print("Running GA-Small (Level 0: linear + lags)...")
    _, ga_small_ms, w_small, names_small = _run_ga(df_A_tr, y_A_tr, level=0)
    ga_small_mask = _ga_basis(df_A_tr, 0)
    mask_small = np.ones(len(names_small), dtype=bool)

    pred_A_tr = eval_ga_on_df(df_A_tr, 0, mask_small, w_small)
    pred_A_te = eval_ga_on_df(df_A_te, 0, mask_small, w_small)
    pred_B = eval_ga_on_df(df_B, 0, mask_small, w_small)
    pred_C = eval_ga_on_df(df_C, 0, mask_small, w_small)
    pred_D = eval_ga_on_df(df_D, 0, mask_small, w_small)

    results["methods"]["ga_small"] = {
        "name": "GA term selection, small library (linear + lags)",
        "wall_ms": ga_small_ms,
        "active_terms": names_small,
        "r2_A_tr": _calc_r2(df_A_tr["target"].to_numpy()[2:], pred_A_tr[2:]),
        "r2_A_te": _calc_r2(df_A_te["target"].to_numpy()[2:], pred_A_te[2:]),
        "r2_B": _calc_r2(df_B["target"].to_numpy()[2:], pred_B[2:]),
        "r2_C_shift": _calc_r2(df_C["target"].to_numpy()[2:], pred_C[2:]),
        "r2_D_heavy": _calc_r2(df_D["target"].to_numpy()[2:], pred_D[2:]),
    }

    print("Running GA-Large (Level 2: linear + lags + squares + interactions)...")
    ga_large_res, ga_large_ms, w_large, names_large = _run_ga(df_A_tr, y_A_tr, level=2)
    mask_large = ga_large_res["mask"]

    pred_A_tr_l = eval_ga_on_df(df_A_tr, 2, mask_large, w_large)
    pred_A_te_l = eval_ga_on_df(df_A_te, 2, mask_large, w_large)
    pred_B_l = eval_ga_on_df(df_B, 2, mask_large, w_large)
    pred_C_l = eval_ga_on_df(df_C, 2, mask_large, w_large)
    pred_D_l = eval_ga_on_df(df_D, 2, mask_large, w_large)

    results["methods"]["ga_large"] = {
        "name": "GA term selection, large library (squares + planted interactions)",
        "wall_ms": ga_large_ms,
        "active_terms": names_large,
        "r2_A_tr": _calc_r2(df_A_tr["target"].to_numpy()[2:], pred_A_tr_l[2:]),
        "r2_A_te": _calc_r2(df_A_te["target"].to_numpy()[2:], pred_A_te_l[2:]),
        "r2_B": _calc_r2(df_B["target"].to_numpy()[2:], pred_B_l[2:]),
        "r2_C_shift": _calc_r2(df_C["target"].to_numpy()[2:], pred_C_l[2:]),
        "r2_D_heavy": _calc_r2(df_D["target"].to_numpy()[2:], pred_D_l[2:]),
    }

    print("Running RIME (RimeProposer, native Rust additive and seed engine)...")
    t0 = time.perf_counter()
    proposer = RimeProposer(
        target_col="target",
        budget=200,
        max_depth=2,
        additive_terms=2,
        max_seeds=120,
        lagged_products=True,
        gamma=1.0,
    )
    proposals = proposer.fit_and_propose(df_A_tr)
    rime_ms = (time.perf_counter() - t0) * 1000.0

    assert len(proposals) > 0, "RIME must return proposals"
    top_p = proposals[0]

    def _pred_rime(df: pd.DataFrame) -> np.ndarray:
        return top_p.compute_score(df)

    pred_rime_A_tr = _pred_rime(df_A_tr)
    pred_rime_A_te = _pred_rime(df_A_te)
    pred_rime_B = _pred_rime(df_B)
    pred_rime_C = _pred_rime(df_C)
    pred_rime_D = _pred_rime(df_D)

    results["methods"]["rime"] = {
        "name": "RIME (RimeProposer)",
        "engine_class": "apedp_dyn.proposers.rime_proposer.RimeProposer",
        "wall_ms": rime_ms,
        "formula": top_p.formula,
        "active_terms": top_p.metadata.get("inputs_used_names", []),
        "r2_A_tr": _calc_r2(df_A_tr["target"].to_numpy()[2:], pred_rime_A_tr[2:]),
        "r2_A_te": _calc_r2(df_A_te["target"].to_numpy()[2:], pred_rime_A_te[2:]),
        "r2_B": _calc_r2(df_B["target"].to_numpy()[2:], pred_rime_B[2:]),
        "r2_C_shift": _calc_r2(df_C["target"].to_numpy()[2:], pred_rime_C[2:]),
        "r2_D_heavy": _calc_r2(df_D["target"].to_numpy()[2:], pred_rime_D[2:]),
    }

    print("Running RIME-AD (RimeAdProposer, default configuration)...")
    t0 = time.perf_counter()
    ad_proposals = list(RimeAdProposer(target_col="target").fit_and_propose(df_A_tr))
    ad_ms = (time.perf_counter() - t0) * 1000.0
    if ad_proposals:
        ad_top = ad_proposals[0]
        ad_pred = {name: ad_top.compute_score(frame) for name, frame in
                   (("A_tr", df_A_tr), ("A_te", df_A_te), ("B", df_B), ("C", df_C), ("D", df_D))}
        results["methods"]["rime_ad"] = {
            "name": "RIME-AD (RimeAdProposer)",
            "engine_class": "apedp_dyn.proposers.rime_ad_proposer.RimeAdProposer",
            "wall_ms": ad_ms,
            "formula": ad_top.formula,
            "n_proposals": len(ad_proposals),
            "r2_A_tr": _calc_r2(df_A_tr["target"].to_numpy()[2:], np.asarray(ad_pred["A_tr"], dtype=float)[2:]),
            "r2_A_te": _calc_r2(df_A_te["target"].to_numpy()[2:], np.asarray(ad_pred["A_te"], dtype=float)[2:]),
            "r2_B": _calc_r2(df_B["target"].to_numpy()[2:], np.asarray(ad_pred["B"], dtype=float)[2:]),
            "r2_C_shift": _calc_r2(df_C["target"].to_numpy()[2:], np.asarray(ad_pred["C"], dtype=float)[2:]),
            "r2_D_heavy": _calc_r2(df_D["target"].to_numpy()[2:], np.asarray(ad_pred["D"], dtype=float)[2:]),
        }
    else:
        results["methods"]["rime_ad"] = {"name": "RIME-AD (RimeAdProposer)", "wall_ms": ad_ms,
                                         "formula": None, "n_proposals": 0}

    # 4. Distractor Noise Robustness Challenge (7 Distractor Features)
    print("Running Distractor Challenge (7 pure noise channels)...")
    rng_dist = np.random.RandomState(42)
    n_dist = 600
    df_dist = pd.DataFrame({
        "x1": rng_dist.randn(n_dist),
        "x2": rng_dist.randn(n_dist),
        "x3": rng_dist.randn(n_dist),
    })
    for k in range(4, 11):
        df_dist[f"x{k}"] = rng_dist.randn(n_dist) # Pure noise distractors
    x1_arr = df_dist["x1"].to_numpy()
    x2_arr = df_dist["x2"].to_numpy()
    x3_arr = df_dist["x3"].to_numpy()
    y_dist = np.zeros(n_dist)
    y_dist[2:] = 2.0 * (x1_arr[1:-1] * x2_arr[1:-1]) + 3.0 * x3_arr[:-2] + 10.0 + 0.02 * rng_dist.randn(n_dist - 2)
    df_dist["target"] = y_dist

    # RIME on distractors
    t0 = time.perf_counter()
    proposer_dist = RimeProposer(
        target_col="target",
        budget=200,
        max_depth=2,
        additive_terms=2,
        max_seeds=800,
        lagged_products=True,
        gamma=1.0,
    )
    proposals_dist = proposer_dist.fit_and_propose(df_dist)
    t_rime_dist = (time.perf_counter() - t0) * 1000.0
    top_dist = proposals_dist[0]
    inputs_used = top_dist.metadata.get("inputs_used_names", [])

    results["distractor_challenge"] = {
        "engine": "RIME (RimeProposer)",
        "wall_ms": t_rime_dist,
        "dev_r2": top_dist.parameters["dev_r2"],
        "inputs_used": inputs_used,
        "distractors_admitted": [f"x{k}" for k in range(4, 11) if f"x{k}" in inputs_used],
        "distractor_resilience_passed": len([f"x{k}" for k in range(4, 11) if f"x{k}" in inputs_used]) == 0,
    }

    # Save JSON
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {JSON_OUT}")

    generate_markdown_report(results)
    return results


def _row(label: str, m: dict) -> str:
    if m.get("r2_A_tr") is None:
        return f"| {label} | no proposal | | | | | `{m['wall_ms']:.1f} ms` |"
    return (f"| {label} | `{m['r2_A_tr']:.4f}` | `{m['r2_A_te']:.4f}` | `{m['r2_B']:.4f}` | "
            f"`{m['r2_C_shift']:.4f}` | `{m['r2_D_heavy']:.4f}` | `{m['wall_ms']:.1f} ms` |")


def generate_markdown_report(results: dict[str, Any]) -> None:
    """Write the report from the measured numbers only; no verdict is asserted that the table does not show."""
    m = results["methods"]
    dist = results["distractor_challenge"]
    lines = [
        "# Population extrapolation: RIME and RIME-AD against a genetic term search",
        "",
        "> **Correction, 2026-11-19 (SYSTEM_FLAWS_AUDIT M-05).** Earlier versions of this page reported",
        "> a result as RIME-AD's that the script had produced with `RimeProposer` (RIME); RIME-AD was",
        "> never run. Each engine now has its own row. The \"GA\" rows are a genetic search over which",
        "> columns of a fixed, hand-built term library enter a least-squares fit; the large library",
        "> contains the true interaction term, so those rows measure term selection, not discovery.",
        "",
        "Ground truth: $y[t] = 2 x_1[t-1] x_2[t-1] + 3 x_3[t-2] + 10 + \\varepsilon$.",
        "Populations: A-train (400 rows, seed 500), A-test (200 rows), B (fresh seed 501), C (parameter",
        "shift, $c_1 = 2.5$, $c_2 = 2.0$), D (Laplace(0, 1.5) inputs). Machine-readable results:",
        "[`benchmarks/population_extrapolation_results.json`](../benchmarks/population_extrapolation_results.json).",
        "",
        "| Engine | A-train $R^2$ | A-test $R^2$ | B $R^2$ | C (shift) $R^2$ | D (heavy tail) $R^2$ | Wall time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        _row("GA term selection, small library", m["ga_small"]),
        _row("GA term selection, large library", m["ga_large"]),
        _row("RIME (`RimeProposer`)", m["rime"]),
        _row("RIME-AD (`RimeAdProposer`)", m["rime_ad"]),
        "",
        f"- RIME's top formula: `{m['rime']['formula']}`",
        f"- RIME-AD's top formula: `{m['rime_ad'].get('formula')}`",
        f"- GA large-library terms selected ({len(m['ga_large']['active_terms'])}): `{m['ga_large']['active_terms']}`",
        "",
        "## Distractor challenge (RIME, seven irrelevant inputs $x_4 \\dots x_{10}$)",
        "",
        f"Inputs used: `{dist['inputs_used']}`; distractors admitted: `{dist['distractors_admitted']}`;",
        f"development $R^2$ `{dist['dev_r2']:.6f}`; wall time `{dist['wall_ms']:.1f} ms`.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {MD_OUT}")


if __name__ == "__main__":
    run_benchmark()
