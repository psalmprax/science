"""Does fitting a power's exponent at birth help or cost at benchmark scale?

Protocol, fixed BEFORE either arm was built (do not tune per result):

  **Arms.** ``on`` = the kernel as shipped (a Pow-bearing initial candidate
  has its exponent fitted by the polish stage where it is born). ``off`` = the
  same tree compiled with ``has_pow_const`` returning false, so birth fitting
  is skipped and that candidate is scored with the exponent the generator drew.
  Everything else -- continuous exponent grid, polish sweeps, seeds, data,
  budgets -- is identical. The two kernel builds are identified by the .so
  sha256 recorded in each arm.

  **Cells.**
    1. matched-compute Gate 4 (``run_five_gate_validation``): Rabi and Brus at
       that harness's frozen config (2 islands x 30 x 15), seeds 0..9, scored on
       the same holdout grids the harness scores. This is the one *named*
       benchmark path that runs the native GA.
    2. External-60 native adapter at its frozen config (300 x 50 x 4), all 60
       problems, seed 42, scored on the PUBLIC validation grid only (no hidden
       truth is read). This is the large-budget cell: ~55k-90k candidate
       evaluations per problem.
    3. Call-graph audit: the synthesis engine (``AutonomousModelSynthesizer``)
       and the Landauer harness (``run_ga_polynomial`` + ``run_assisted_rime``)
       are run with ``RustKernelBridge.evolve_native`` and
       ``evolutionary_search.evolve`` wrapped by counters, and report how many
       native-GA calls each makes. The claim under test is that these two
       benchmarks do not route through the changed code at all.

  **Cost** is candidate evaluations and wall seconds. **Benefit** is holdout /
  validation R2 or MSE. Wall time on a shared machine is noisy; per-cell
  medians over seeds/problems are reported, and the paired per-problem
  External-60 comparison is the primary wall-cost instrument.

  **Status.** Audit-only measurement. No claim about any other engine.

Usage:
    python benchmarks/pow_birth_fit_benchmark.py --label on --out /tmp/on.json
    python benchmarks/pow_birth_fit_benchmark.py --label off --out /tmp/off.json
    python benchmarks/pow_birth_fit_benchmark.py --merge /tmp/on.json /tmp/off.json \
        --out benchmarks/pow_birth_fit_benchmark_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE4_SEEDS = list(range(10))
EXTERNAL60_SEED = 42
EXTERNAL60_CONFIG = {"population_size": 300, "generations": 50, "islands": 4}
LANDAUER_ROW = 60


# ── helpers ──────────────────────────────────────────────────────────────────

def _kernel_sha256() -> str | None:
    try:
        import apedp_kernel

        path = getattr(apedp_kernel, "__file__", None)
        if path is None:
            return None
        # ``__file__`` is the package __init__; the compiled arm identity is
        # the .so beside it (the repo-root symlink resolves to the same tree).
        so = next(Path(os.path.realpath(path)).parent.glob("*.so"), None)
        if so is None:
            return None
        with open(so, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:  # noqa: BLE001 - arm identity is best-effort metadata
        return None


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination: 1 - SSE/SST.

    Not ``1 - var(residual)/var(y)``: subtracting the residual mean would
    credit an offset formula with the bias it fails to reproduce. Fixed
    2026-10-23 after that bug inflated Planck candidates by up to 0.4 R2.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred)
    if finite.sum() < 3:
        return float("nan")
    var = float(np.var(y[finite]))
    if var <= 0.0:
        return float("nan")
    sse = float(np.mean((y[finite] - pred[finite]) ** 2))
    return float(1.0 - sse / var)


def _mse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred)
    if finite.sum() < 3:
        return float("nan")
    return float(np.mean((y[finite] - pred[finite]) ** 2))


# ── cell 1: matched-compute Gate 4 ───────────────────────────────────────────

def _gate4_cell() -> dict[str, Any]:
    from apedp.evolutionary_search import SearchConfig, evolve
    from apedp.rust_bridge import RustKernelBridge

    t_tr = np.arange(0.0, 4.0, 0.05)
    t_ho = np.arange(4.0, 8.0, 0.05)
    y_rabi_tr = 0.5 - 0.5 * np.cos(2.0 * t_tr)
    y_rabi_ho = 0.5 - 0.5 * np.cos(2.0 * t_ho)

    r_tr = np.linspace(1.5, 8.0, 50)
    r_ho = np.linspace(8.1, 14.0, 40)
    y_brus_tr = 1.74 + 2.45 * r_tr**-2 - 0.28 * r_tr**-1
    y_brus_ho = 1.74 + 2.45 * r_ho**-2 - 0.28 * r_ho**-1

    bridge = RustKernelBridge()
    cases = {
        "rabi": ({"t": t_tr}, y_rabi_tr, ["t"], t_ho, y_rabi_ho),
        "brus": ({"R": r_tr}, y_brus_tr, ["R"], r_ho, y_brus_ho),
    }
    out: dict[str, Any] = {}
    for name, (x_tr, y_tr, names, x_ho, y_ho) in cases.items():
        rows = []
        for seed in GATE4_SEEDS:
            t0 = time.perf_counter()
            ga = evolve(x_tr, y_tr, config=SearchConfig(
                use_native_ga=True, strict_native_ga=True,
                islands=2, island_size=30, generations=15, seed=seed))
            wall_ms = (time.perf_counter() - t0) * 1000.0
            pred = bridge.evaluate_expression(ga.best_expression, names, [x_ho])
            rows.append({
                "seed": seed,
                "formula": ga.best_expression,
                "holdout_r2": _r2(y_ho, pred),
                "wall_ms": wall_ms,
                "n_evaluations": int(ga.n_evaluations),
            })
        r2s = np.array([r["holdout_r2"] for r in rows], dtype=float)
        evals = np.array([r["n_evaluations"] for r in rows], dtype=float)
        walls = np.array([r["wall_ms"] for r in rows], dtype=float)
        out[name] = {
            "rows": rows,
            "summary": {
                "seeds": len(rows),
                "r2_median": float(np.nanmedian(r2s)),
                "r2_min": float(np.nanmin(r2s)),
                "r2_at_or_above_0_99": int(np.sum(r2s >= 0.99)),
                "evals_median": float(np.median(evals)),
                "evals_total": int(evals.sum()),
                "wall_ms_median": float(np.median(walls)),
                "wall_ms_total": float(walls.sum()),
            },
        }
    return out


# ── cell 2: External-60 native adapter ───────────────────────────────────────

def _external60_cell() -> dict[str, Any]:
    from benchmarks.external_60_adapters import RimeADNativeAdapter, load_public_grids

    corpus = ROOT / "benchmark_corpus_external_60"
    manifest = json.loads((corpus / "manifest.json").read_text())
    adapter = RimeADNativeAdapter(seed=EXTERNAL60_SEED, **EXTERNAL60_CONFIG)
    rows = []
    for problem in manifest["problems"]:
        pid = problem["problem_id"]
        grids = load_public_grids(corpus, pid)
        res = adapter.run(corpus, pid)
        y_val = grids["validation.csv"][:, 1]
        pred = np.asarray(res.predictions.get("validation.csv", []), dtype=float)
        rows.append({
            "problem_id": pid,
            "block": problem.get("block"),
            "expression": res.expression,
            "abstained": bool(res.abstained),
            "train_mse": res.metadata.get("train_mse"),
            "val_mse": _mse(y_val, pred) if pred.size else float("nan"),
            "val_r2": _r2(y_val, pred) if pred.size else float("nan"),
            "n_evaluations": res.metadata.get("n_evaluations"),
            "wall_seconds": float(res.wall_seconds),
        })

    def _summary(key: str) -> dict[str, float]:
        vals = np.array(
            [r[key] for r in rows if r[key] is not None], dtype=float)
        vals = vals[np.isfinite(vals)]
        return {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "config": dict(EXTERNAL60_CONFIG, seed=EXTERNAL60_SEED),
        "problems": len(rows),
        "rows": rows,
        "summary": {
            "evals_median": _summary("n_evaluations")["median"],
            "evals_total": int(sum(r["n_evaluations"] or 0 for r in rows)),
            "wall_seconds_median": _summary("wall_seconds")["median"],
            "wall_seconds_total": float(sum(r["wall_seconds"] for r in rows)),
            "train_mse_median": _summary("train_mse")["median"],
            "val_mse_median": _summary("val_mse")["median"],
            "val_r2_median": _summary("val_r2")["median"],
            "abstentions": int(sum(r["abstained"] for r in rows)),
        },
    }


# ── cell 3: call-graph audit ─────────────────────────────────────────────────

class _Counter:
    def __init__(self) -> None:
        self.evolve_native = 0
        self.evolve = 0

    def install(self) -> None:
        import apedp.evolutionary_search as es
        import apedp.rust_bridge as rb

        counter = self
        original_native = rb.RustKernelBridge.evolve_native
        original_evolve = es.evolve

        def counted_native(self, *args, **kwargs):
            counter.evolve_native += 1
            return original_native(self, *args, **kwargs)

        def counted_evolve(*args, **kwargs):
            counter.evolve += 1
            return original_evolve(*args, **kwargs)

        rb.RustKernelBridge.evolve_native = counted_native
        es.evolve = counted_evolve


def _read_corpus_grids(problem_dir: Path) -> dict[str, np.ndarray]:
    import csv

    out = {}
    for name in ("train.csv", "validation.csv", "near_ood.csv"):
        with open(problem_dir / name, encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        out[name] = np.array([[float(v) for v in r] for r in rows[1:]])
    return out


def _call_graph_cell() -> dict[str, Any]:
    counter = _Counter()
    counter.install()
    t_synth0 = time.perf_counter()
    from apedp.model_synthesis import AutonomousModelSynthesizer

    synth_dir = ROOT / "benchmark_corpus_synthesis" / "public" / "SYNTH_01"
    grids = _read_corpus_grids(synth_dir)
    tr, va, near = grids["train.csv"], grids["validation.csv"], grids["near_ood.csv"]
    models = AutonomousModelSynthesizer().synthesize(
        tr[:, 0], tr[:, 1], va[:, 0], va[:, 1], near[:, 0], near[:, 1],
        allow_near_ood_promote=False)
    synth_wall = time.perf_counter() - t_synth0
    top = models[0]
    synthesis = {
        "problem": "SYNTH_01",
        "wall_seconds": synth_wall,
        "top_form": top.expression_str,
        "top_epistemic_class": top.epistemic_class,
        "n_models": len(models),
        "evolve_native_calls": counter.evolve_native,
        "evolve_calls": counter.evolve,
    }

    before_native, before_evolve = counter.evolve_native, counter.evolve
    from benchmarks.run_landauer_rime_vs_ga_benchmark import (
        run_assisted_rime,
        run_ga_polynomial,
        run_unassisted_rime,
    )

    x = np.arange(LANDAUER_ROW, dtype=float)
    y = np.floor(0.5 * x + 0.1) * 2.0
    t0 = time.perf_counter()
    landauer_ga = run_ga_polynomial(x, y, x, y, seed=42)
    landauer_ga_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    landauer_rime = run_unassisted_rime(x, y, x, y)
    run_assisted_rime(x, y, x, y)
    landauer_rime_wall = time.perf_counter() - t0
    landauer = {
        "rows": LANDAUER_ROW,
        "ga_wall_seconds": landauer_ga_wall,
        "rime_wall_seconds": landauer_rime_wall,
        "ga_formula_powers": landauer_ga.get("selected_powers"),
        "rime_formula": landauer_rime.get("formula"),
        "evolve_native_calls": counter.evolve_native - before_native,
        "evolve_calls": counter.evolve - before_evolve,
    }
    return {"synthesis": synthesis, "landauer": landauer}


# ── recompute (metric fix on an existing arm) ────────────────────────────────

def _recompute(artifact: dict[str, Any]) -> dict[str, Any]:
    """Recompute R2 fields from the recorded formulas with the corrected metric.

    Formulas, data and seeds are deterministic; only the R2 helper was wrong,
    so stored rows can be corrected without re-running either search.
    """
    from apedp.rust_bridge import RustKernelBridge
    from benchmarks.external_60_adapters import load_public_grids

    bridge = RustKernelBridge()
    t_ho = np.arange(4.0, 8.0, 0.05)
    r_ho = np.linspace(8.1, 14.0, 40)
    holdouts = {
        "rabi": (["t"], [t_ho], 0.5 - 0.5 * np.cos(2.0 * t_ho)),
        "brus": (["R"], [r_ho], 1.74 + 2.45 * r_ho**-2 - 0.28 * r_ho**-1),
    }

    out = dict(artifact)
    gate4: dict[str, Any] = {}
    for case, data in artifact["gate4"].items():
        names, xs, y_ho = holdouts[case]
        rows = []
        for row in data["rows"]:
            pred = bridge.evaluate_expression(row["formula"], names, xs)
            rows.append({**row, "holdout_r2": _r2(y_ho, pred)})
        r2s = np.array([r["holdout_r2"] for r in rows], dtype=float)
        summary = dict(data["summary"])
        summary.update({
            "r2_median": float(np.nanmedian(r2s)),
            "r2_min": float(np.nanmin(r2s)),
            "r2_at_or_above_0_99": int(np.sum(r2s >= 0.99)),
        })
        gate4[case] = {**data, "rows": rows, "summary": summary}
    out["gate4"] = gate4

    corpus = ROOT / "benchmark_corpus_external_60"
    ext = artifact["external60"]
    rows = []
    for row in ext["rows"]:
        y_val = load_public_grids(corpus, row["problem_id"])["validation.csv"][:, 1]
        pred = bridge.evaluate_expression(row["expression"], ["x"],
                                          [load_public_grids(corpus, row["problem_id"])["validation.csv"][:, 0]])
        rows.append({**row, "val_mse": _mse(y_val, pred), "val_r2": _r2(y_val, pred)})
    val_mse = np.array([r["val_mse"] for r in rows], dtype=float)
    val_r2 = np.array([r["val_r2"] for r in rows], dtype=float)
    summary = dict(ext["summary"])
    summary["val_mse_median"] = float(np.nanmedian(val_mse))
    summary["val_r2_median"] = float(np.nanmedian(val_r2))
    out["external60"] = {**ext, "rows": rows, "summary": summary}
    out["r2_metric_recomputed"] = True
    return out


# ── merge ────────────────────────────────────────────────────────────────────

def _merge(on: dict[str, Any], off: dict[str, Any]) -> dict[str, Any]:
    def delta(a: float, b: float) -> dict[str, float]:
        return {
            "on_minus_off": float(a - b),
            "relative": float((a - b) / b) if b not in (0.0,) else float("nan"),
        }

    merged: dict[str, Any] = {
        "arms": {
            "on": {
                "label": on["label"],
                "kernel_sha256": on["kernel_sha256"],
                "ga_cell": on["gate4"],
                "external60": on["external60"],
                "call_graph": on["call_graph"],
            },
            "off": {
                "label": off["label"],
                "kernel_sha256": off["kernel_sha256"],
                "ga_cell": off["gate4"],
                "external60": off["external60"],
                "call_graph": off["call_graph"],
            },
        },
        "matched_compute_gate4": {},
        "external60": {},
        "call_graph": {},
    }

    for case in ("rabi", "brus"):
        a = on["gate4"][case]["summary"]
        b = off["gate4"][case]["summary"]
        merged["matched_compute_gate4"][case] = {
            "r2_median": {"on": a["r2_median"], "off": b["r2_median"]},
            "r2_min": {"on": a["r2_min"], "off": b["r2_min"]},
            "recoveries_at_0_99": {
                "on": a["r2_at_or_above_0_99"], "off": b["r2_at_or_above_0_99"],
                "of_seeds": a["seeds"],
            },
            "evals_median": delta(a["evals_median"], b["evals_median"]),
            "evals_total": delta(a["evals_total"], b["evals_total"]),
            "wall_ms_median": delta(a["wall_ms_median"], b["wall_ms_median"]),
            "wall_ms_total": delta(a["wall_ms_total"], b["wall_ms_total"]),
        }

    ea, eb = on["external60"]["summary"], off["external60"]["summary"]
    # Paired per-problem comparison (same problem, same config, one seed).
    by_off = {r["problem_id"]: r for r in off["external60"]["rows"]}
    paired = []
    for r in on["external60"]["rows"]:
        o = by_off.get(r["problem_id"])
        if o is None:
            continue
        paired.append({
            "problem_id": r["problem_id"],
            "val_mse_on": r["val_mse"],
            "val_mse_off": o["val_mse"],
            "evals_on": r["n_evaluations"],
            "evals_off": o["n_evaluations"],
            "wall_on": r["wall_seconds"],
            "wall_off": o["wall_seconds"],
        })
    better = sum(
        1 for p in paired
        if np.isfinite(p["val_mse_on"]) and np.isfinite(p["val_mse_off"])
        and p["val_mse_on"] < p["val_mse_off"])
    worse = sum(
        1 for p in paired
        if np.isfinite(p["val_mse_on"]) and np.isfinite(p["val_mse_off"])
        and p["val_mse_on"] > p["val_mse_off"])
    merged["external60"] = {
        "problems": len(paired),
        "evals_median": delta(ea["evals_median"], eb["evals_median"]),
        "evals_total": delta(float(ea["evals_total"]), float(eb["evals_total"])),
        "wall_seconds_median": delta(
            ea["wall_seconds_median"], eb["wall_seconds_median"]),
        "wall_seconds_total": delta(
            ea["wall_seconds_total"], eb["wall_seconds_total"]),
        "val_mse_median": delta(ea["val_mse_median"], eb["val_mse_median"]),
        "val_r2_median": delta(ea["val_r2_median"], eb["val_r2_median"]),
        "train_mse_median": delta(ea["train_mse_median"], eb["train_mse_median"]),
        "per_problem_better_on": better,
        "per_problem_better_off": worse,
        "per_problem_tied_or_nonfinite": len(paired) - better - worse,
        "paired": paired,
    }

    for name in ("synthesis", "landauer"):
        merged["call_graph"][name] = {
            "on": on["call_graph"][name],
            "off": off["call_graph"][name],
        }
    return merged


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs=2, metavar=("ON", "OFF"))
    ap.add_argument("--recompute", metavar="ARM_JSON",
                    help="recompute R2 fields from the recorded formulas")
    ap.add_argument("--label", choices=["on", "off"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-external60", action="store_true")
    args = ap.parse_args()

    if args.recompute:
        arm = json.loads(Path(args.recompute).read_text())
        Path(args.out).write_text(json.dumps(_recompute(arm), indent=2))
        print(f"recomputed {args.out}")
        return

    if args.merge:
        on = json.loads(Path(args.merge[0]).read_text())
        off = json.loads(Path(args.merge[1]).read_text())
        merged = _merge(on, off)
        Path(args.out).write_text(json.dumps(merged, indent=2))
        print(f"wrote {args.out}")
        return

    if not args.label:
        raise SystemExit("--label is required unless --merge is used")

    print(f"[{args.label}] kernel sha256: {_kernel_sha256()}")
    t0 = time.perf_counter()
    gate4 = _gate4_cell()
    print(f"[{args.label}] gate4 done in {time.perf_counter() - t0:.1f}s")
    for case, data in gate4.items():
        s = data["summary"]
        print(f"  {case}: r2 median {s['r2_median']:.6f} min {s['r2_min']:.6f} "
              f"({s['r2_at_or_above_0_99']}/{s['seeds']} >= 0.99) "
              f"evals median {s['evals_median']:.0f} wall median {s['wall_ms_median']:.1f}ms")

    external60 = ({"rows": [], "summary": {}, "skipped": True}
                  if args.skip_external60 else _external60_cell())
    if not args.skip_external60:
        print(f"[{args.label}] external60 median evals {external60['summary']['evals_median']:.0f} "
              f"median val_mse {external60['summary']['val_mse_median']:.6g} "
              f"total wall {external60['summary']['wall_seconds_total']:.1f}s")

    call_graph = _call_graph_cell()
    print(f"[{args.label}] synthesis native-GA calls: "
          f"{call_graph['synthesis']['evolve_native_calls']} "
          f"(evolve: {call_graph['synthesis']['evolve_calls']}); "
          f"landauer native-GA calls: {call_graph['landauer']['evolve_native_calls']} "
          f"(evolve: {call_graph['landauer']['evolve_calls']})")

    artifact = {
        "label": args.label,
        "kernel_sha256": _kernel_sha256(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate4": gate4,
        "external60": external60,
        "call_graph": call_graph,
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
