"""Does birth-fitting trig frequency/phase templates help at the Planck pin?

Protocol fixed before either arm ran:

  **Arms.** ``before`` = kernel at f1e4ade (birth fit applies to Pow-bearing
  candidates only). ``after`` = the same tree with the trig template predicate
  added to the birth-fit condition, so `sin|cos(f*x)` and `sin|cos(f*x + p)`
  candidates are fitted where they are born. Everything else identical.

  **Cells.**
    1. Planck radiance (the must-fail pin, `tests/test_dyn_planck_limitation.py`):
       v in [0.1, 6] (60 points) -> u = v^3/(e^v - 1); GA at the pin's
       2x30x15 budget, seeds 0..9. Scored on the pin's holdout ([6.1, 10]) and
       the extended domain ([0.05, 12]); record train MSE, holdout R2,
       extended R2, evaluations and wall. The pin requires holdout R2 < 0.5 and
       extended R2 < 0.9: a *higher* number is a better fit and a *weaker*
       must-fail pin, which is the change being measured.
    2. Trig capability pins (`tests/test_dyn_native_ga_trig_templates.py`):
       sin(2x), 2cos(1.3x) and 3sin(1.7x + 0.4) at the pin config
       (population 100, 25 generations, 2 islands), seeds 0..9, plus the small
       2x30x15 budget. Reported as exact-recovery count (<1e-12), the
       phase-shift threshold (<1e-5) and median best MSE.

  **Status.** Audit-only measurement for one code change; makes no claim about
  any other engine or equation.

Usage:
    python benchmarks/trig_birth_fit_planck_benchmark.py --label before --out /tmp/before.json
    python benchmarks/trig_birth_fit_planck_benchmark.py --label after  --out /tmp/after.json
    python benchmarks/trig_birth_fit_planck_benchmark.py --merge /tmp/before.json /tmp/after.json \
        --out benchmarks/trig_birth_fit_planck_results.json
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

PLANCK_SEEDS = [*range(10), 42]
TRIG_SEEDS = [*range(10), 42]
PLANCK_CONFIG = {"islands": 2, "island_size": 30, "generations": 15}
TRIG_PIN_ARGS = [100, 25, 2, 5, 4, 4, 0.35, 0.70,
                 True, True, True, True, True, 3, 42]
TRIG_SMALL_CONFIG = {"islands": 2, "island_size": 30, "generations": 15}


def _kernel_sha256() -> str | None:
    try:
        import apedp_kernel

        path = getattr(apedp_kernel, "__file__", None)
        if path is None:
            return None
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
    credit an offset formula with the bias it fails to reproduce. That bug
    inflated a Planck candidate from R2 = 0.55 to 0.96 before this fix.
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


# ── cell 1: Planck must-fail pin ────────────────────────────────────────────

def _planck_cell() -> dict[str, Any]:
    from apedp.evolutionary_search import SearchConfig, evolve
    from apedp.rust_bridge import RustKernelBridge

    v_tr = np.linspace(0.1, 6.0, 60)
    v_ho = np.linspace(6.1, 10.0, 40)
    v_ext = np.linspace(0.05, 12.0, 120)
    u_tr = (v_tr**3) / (np.exp(v_tr) - 1.0)
    u_ho = (v_ho**3) / (np.exp(v_ho) - 1.0)
    u_ext = (v_ext**3) / (np.exp(v_ext) - 1.0)
    bridge = RustKernelBridge()

    rows = []
    for seed in PLANCK_SEEDS:
        t0 = time.perf_counter()
        res = evolve({"v": v_tr}, u_tr, config=SearchConfig(
            use_native_ga=True, strict_native_ga=True, seed=seed,
            **PLANCK_CONFIG))
        wall = time.perf_counter() - t0
        pred_ho = bridge.evaluate_expression(res.best_expression, ["v"], [v_ho])
        pred_ext = bridge.evaluate_expression(res.best_expression, ["v"], [v_ext])
        rows.append({
            "seed": seed,
            "formula": res.best_expression,
            "train_mse": float(min(p["train_mse"] for p in res.pareto_front)),
            "holdout_r2": _r2(u_ho, pred_ho),
            "extended_r2": _r2(u_ext, pred_ext),
            "n_evaluations": int(res.n_evaluations),
            "wall_seconds": wall,
        })

    def med(key: str) -> float:
        return float(np.nanmedian([r[key] for r in rows]))

    return {
        "config": PLANCK_CONFIG,
        "seeds": len(rows),
        "rows": rows,
        "summary": {
            "train_mse_median": med("train_mse"),
            "holdout_r2_median": med("holdout_r2"),
            "holdout_r2_max": float(np.nanmax([r["holdout_r2"] for r in rows])),
            "extended_r2_median": med("extended_r2"),
            "extended_r2_max": float(np.nanmax([r["extended_r2"] for r in rows])),
            "holdout_pass_must_fail": int(sum(
                1 for r in rows if np.isfinite(r["holdout_r2"]) and r["holdout_r2"] < 0.5)),
            "extended_pass_must_fail": int(sum(
                1 for r in rows if np.isfinite(r["extended_r2"]) and r["extended_r2"] < 0.9)),
            "evals_median": med("n_evaluations"),
            "wall_seconds_median": med("wall_seconds"),
        },
    }


# ── cell 2: trig capability pins ────────────────────────────────────────────

def _trig_cell() -> dict[str, Any]:
    import apedp_kernel as ak
    from apedp.evolutionary_search import SearchConfig, evolve

    x = np.linspace(0.05, 3.0, 60)
    targets = {
        "sin_2x": np.sin(2.0 * x),
        "cos_2_1p3x": 2.0 * np.cos(1.3 * x),
        "phase_3sin_1p7x_0p4": 3.0 * np.sin(1.7 * x + 0.4),
    }
    out: dict[str, Any] = {}
    for name, y in targets.items():
        pin_rows = []
        for seed in TRIG_SEEDS:
            args = list(TRIG_PIN_ARGS)
            args[-1] = seed
            t0 = time.perf_counter()
            raw, evals = ak.evolve_native(
                [np.ascontiguousarray(x)], ["x"], np.ascontiguousarray(y), *args)
            wall = time.perf_counter() - t0
            best = min(raw, key=lambda m: m[1])
            pin_rows.append({"seed": seed, "mse": float(best[1]),
                             "formula": str(best[0]), "n_evaluations": int(evals),
                             "wall_seconds": wall})
        small_rows = []
        for seed in TRIG_SEEDS[:5]:
            t0 = time.perf_counter()
            res = evolve({"x": x}, y, config=SearchConfig(
                use_native_ga=True, strict_native_ga=True, seed=seed,
                **TRIG_SMALL_CONFIG))
            wall = time.perf_counter() - t0
            small_rows.append({
                "seed": seed,
                "mse": float(min(p["train_mse"] for p in res.pareto_front)),
                "formula": res.best_expression,
                "n_evaluations": int(res.n_evaluations),
                "wall_seconds": wall,
            })
        pin_mse = np.array([r["mse"] for r in pin_rows], dtype=float)
        small_mse = np.array([r["mse"] for r in small_rows], dtype=float)
        out[name] = {
            "pin": {"rows": pin_rows, "summary": {
                "seeds": len(pin_rows),
                "exact_counts": int((pin_mse < 1e-12).sum()),
                "below_1e_5": int((pin_mse < 1e-5).sum()),
                "mse_median": float(np.median(pin_mse)),
                "evals_median": float(np.median([r["n_evaluations"] for r in pin_rows])),
                "wall_seconds_median": float(np.median([r["wall_seconds"] for r in pin_rows])),
            }},
            "small": {"rows": small_rows, "summary": {
                "seeds": len(small_rows),
                "exact_counts": int((small_mse < 1e-12).sum()),
                "below_1e_5": int((small_mse < 1e-5).sum()),
                "mse_median": float(np.median(small_mse)),
                "evals_median": float(np.median([r["n_evaluations"] for r in small_rows])),
                "wall_seconds_median": float(np.median([r["wall_seconds"] for r in small_rows])),
            }},
        }
    return out


# ── recompute (metric fix on an existing arm) ───────────────────────────────

def _recompute(artifact: dict[str, Any]) -> dict[str, Any]:
    """Recompute Planck R2 from the recorded formulas with the corrected metric.

    The formulas and grids are deterministic; only the R2 helper was wrong, so
    the stored rows can be corrected without re-running the search.
    """
    from apedp.rust_bridge import RustKernelBridge

    v_ho = np.linspace(6.1, 10.0, 40)
    v_ext = np.linspace(0.05, 12.0, 120)
    u_ho = (v_ho**3) / (np.exp(v_ho) - 1.0)
    u_ext = (v_ext**3) / (np.exp(v_ext) - 1.0)
    bridge = RustKernelBridge()
    rows = []
    for row in artifact["planck"]["rows"]:
        pred_ho = bridge.evaluate_expression(row["formula"], ["v"], [v_ho])
        pred_ext = bridge.evaluate_expression(row["formula"], ["v"], [v_ext])
        updated = dict(row)
        updated["holdout_r2"] = _r2(u_ho, pred_ho)
        updated["extended_r2"] = _r2(u_ext, pred_ext)
        rows.append(updated)
    summary = dict(artifact["planck"]["summary"])
    ho = [r["holdout_r2"] for r in rows]
    ext = [r["extended_r2"] for r in rows]
    summary.update({
        "holdout_r2_median": float(np.nanmedian(ho)),
        "holdout_r2_max": float(np.nanmax(ho)),
        "extended_r2_median": float(np.nanmedian(ext)),
        "extended_r2_max": float(np.nanmax(ext)),
        "holdout_pass_must_fail": int(sum(
            1 for v in ho if np.isfinite(v) and v < 0.5)),
        "extended_pass_must_fail": int(sum(
            1 for v in ext if np.isfinite(v) and v < 0.9)),
    })
    artifact = dict(artifact)
    artifact["planck"] = dict(artifact["planck"], rows=rows, summary=summary)
    artifact["r2_metric_recomputed"] = True
    return artifact


# ── merge ───────────────────────────────────────────────────────────────────

def _merge(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def delta(after: float, before: float) -> dict[str, float]:
        return {
            "before": float(before),
            "after": float(after),
            "change": float(after - before),
            "relative": float((after - before) / before) if before != 0.0 else float("nan"),
        }

    merged: dict[str, Any] = {
        "arms": {
            "before": {"label": before["label"], "kernel_sha256": before["kernel_sha256"]},
            "after": {"label": after["label"], "kernel_sha256": after["kernel_sha256"]},
        },
        "planck": {},
        "trig": {},
    }
    b, a = before["planck"]["summary"], after["planck"]["summary"]
    for key in ("train_mse_median", "holdout_r2_median", "holdout_r2_max",
                "extended_r2_median", "extended_r2_max", "evals_median",
                "wall_seconds_median"):
        merged["planck"][key] = delta(a[key], b[key])
    merged["planck"]["must_fail_counts"] = {
        "holdout_lt_0_5": {"before": b["holdout_pass_must_fail"],
                           "after": a["holdout_pass_must_fail"],
                           "of_seeds": after["planck"]["seeds"]},
        "extended_lt_0_9": {"before": b["extended_pass_must_fail"],
                            "after": a["extended_pass_must_fail"],
                            "of_seeds": after["planck"]["seeds"]},
    }
    for name in before["trig"]:
        entry = {}
        for budget in ("pin", "small"):
            sb = before["trig"][name][budget]["summary"]
            sa = after["trig"][name][budget]["summary"]
            entry[budget] = {
                "exact_counts": {"before": sb["exact_counts"], "after": sa["exact_counts"],
                                 "of_seeds": sb["seeds"]},
                "below_1e_5": {"before": sb["below_1e_5"], "after": sa["below_1e_5"]},
                "mse_median": delta(sa["mse_median"], sb["mse_median"]),
                "evals_median": delta(sa["evals_median"], sb["evals_median"]),
                "wall_seconds_median": delta(sa["wall_seconds_median"], sb["wall_seconds_median"]),
            }
        merged["trig"][name] = entry
    return merged


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--recompute", metavar="ARM_JSON",
                    help="recompute Planck R2 from the recorded formulas")
    ap.add_argument("--label", choices=["before", "after"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.recompute:
        arm = json.loads(Path(args.recompute).read_text())
        Path(args.out).write_text(json.dumps(_recompute(arm), indent=2))
        print(f"recomputed {args.out}")
        return

    if args.merge:
        before = json.loads(Path(args.merge[0]).read_text())
        after = json.loads(Path(args.merge[1]).read_text())
        Path(args.out).write_text(json.dumps(_merge(before, after), indent=2))
        print(f"wrote {args.out}")
        return

    if not args.label:
        raise SystemExit("--label is required unless --merge is used")

    print(f"[{args.label}] kernel sha256: {_kernel_sha256()}")
    t0 = time.perf_counter()
    planck = _planck_cell()
    print(f"[{args.label}] planck done in {time.perf_counter() - t0:.1f}s")
    s = planck["summary"]
    print(f"  train_mse_median={s['train_mse_median']:.4g} "
          f"holdout_r2_median={s['holdout_r2_median']:.4f} max={s['holdout_r2_max']:.4f} "
          f"extended_r2_median={s['extended_r2_median']:.4f} max={s['extended_r2_max']:.4f} "
          f"evals_median={s['evals_median']:.0f}")

    t0 = time.perf_counter()
    trig = _trig_cell()
    print(f"[{args.label}] trig done in {time.perf_counter() - t0:.1f}s")
    for name, data in trig.items():
        p, sm = data["pin"]["summary"], data["small"]["summary"]
        print(f"  {name}: pin exact {p['exact_counts']}/{p['seeds']} mse_med "
              f"{p['mse_median']:.3g} | small exact {sm['exact_counts']}/{sm['seeds']} "
              f"mse_med {sm['mse_median']:.3g}")

    artifact = {
        "label": args.label,
        "kernel_sha256": _kernel_sha256(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planck": planck,
        "trig": trig,
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
