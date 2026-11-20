#!/usr/bin/env python3
"""The template engine with the referee's Guard 3 on, for PREREG-REFEREE-STRUCTURE-2026-11.

The same problems, noise levels, rows, seed and time limit as the L2 arm of
PREREG-SRBENCH-BASELINE-2026-11, whose recorded runs are the comparison: there the referee promoted
268 laws and 160 of them were not accurate. The one difference is ``structure_rule=True``.

    python scripts/run_srbench_referee_structure.py smoke   --out DIR
    python scripts/run_srbench_referee_structure.py develop --out DIR [--noise 0.1] [--shard k/N]   (development problems only)
    python scripts/run_srbench_referee_structure.py run     --out DIR --shard k/N
    python scripts/run_srbench_referee_structure.py collect --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

PREREGISTRATION_ID = "PREREG-REFEREE-STRUCTURE-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-REFEREE-STRUCTURE.json"
FROZEN = ("preregistration/PREREG-REFEREE-STRUCTURE-2026-11.md", "scripts/run_srbench_referee_structure.py",
          "scripts/run_srbench_main_search.py", "scripts/run_srbench_baseline.py", "benchmarks/srbench/manifest.json",
          "apedp_dyn/epistemic/structure_guard.py", "apedp_dyn/epistemic/adjudicate.py",
          "apedp_dyn/discovery_l2/l2_engine.py")

ENGINE = "l2_structure"
NOISE_LEVELS = (0.0, 0.001, 0.01, 0.1)   # the baseline's
DEVELOPMENT_TAG = "dev-refined|"          # the same 30 development problems as PREREG-REFINED-SEARCH


def used_in_development(name: str) -> bool:
    return int(hashlib.sha256((DEVELOPMENT_TAG + name).encode()).hexdigest(), 16) % 5 == 0


def run_l2_structure(split: dict[str, Any], entry: dict[str, Any], budget: float = 0.0) -> dict[str, Any]:
    from apedp_dyn.discovery_l2.dimensional_types import Dimension
    from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
    from apedp_dyn.epistemic.adjudicate import EpistemicStatus

    names = split["names"]
    X = {n: split["X_train"][:, i] for i, n in enumerate(names)}
    dims = {n: base.srb_dimension(entry["units"][f], Dimension) for n, f in zip(names, entry["features"], strict=True)}
    result = L2DiscoveryEngine().discover(
        X=X, Y=split["y_train"], dimensions=dims, target_dimension=base.srb_dimension(entry["target_units"], Dimension),
        target_name="y", seed=base.SEED, structure_rule=True)
    adj = result.adjudication

    def structure(scored: Any) -> sp.Expr | None:
        node = scored.candidate.metadata.get("ast") if scored is not None else None
        return node.to_sympy() if node is not None else None

    top = structure(adj.ranking[0]) if adj.ranking else None
    promoted = structure(adj.promoted) if adj.status is EpistemicStatus.UNIQUE_SUPPORT else None
    model, pred = base._refit_terms(top, names, split) if top is not None else (None, None)
    promoted_ok = bool(base.symbolic_recovery(promoted, base.truth_expr(entry), names, split["X_test"])["recovered"]) \
        if promoted is not None else False
    extra = {"l2_status": adj.status.name, "l2_promoted": str(promoted) if promoted is not None else None,
             "l2_outcome": ("promoted_correct" if promoted_ok else "false_promotion") if promoted is not None else "no_promotion",
             "l2_reason": str(adj.reason)[:300], "l2_generated": len(result.generated_nodes),
             "refused_by_structure_guard": "STRUCTURED_RESIDUALS_IN_INPUT_SPACE" in tuple(adj.diagnostic_flags),
             "used_in_development": used_in_development(entry["name"])}
    return {"model": model, "pred": pred, "formula": str(model) if model is not None else None, "extra": extra}


path_run.RUNNERS[ENGINE] = run_l2_structure


def all_jobs() -> list[dict[str, Any]]:
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": ENGINE}
            for p in base.load_manifest() for noise in NOISE_LEVELS]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in ("all", "held_out", "development"):
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if noise in (r["noise"], "all") and (
                group == "all" or (group == "development") == used_in_development(r["problem"]))]
            ok = [r for r in sel if r["status"] == "ok"]
            promoted = [r for r in ok if r.get("l2_outcome") in ("promoted_correct", "false_promotion")]
            out[f"{group}|{noise}"] = {
                "n": len(sel), "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "promotions": len(promoted),
                "promoted_correct": sum(r["l2_outcome"] == "promoted_correct" for r in promoted),
                "false_promotions": sum(r["l2_outcome"] == "false_promotion" for r in promoted),
                "promotions_not_accurate": sum(not r["accuracy_solution"] for r in promoted),
                "refused_by_structure_guard": sum(bool(r.get("refused_by_structure_guard")) for r in ok),
                "statuses": {s: sum(r.get("l2_status") == s for r in ok) for s in sorted({str(r["l2_status"]) for r in ok if r.get("l2_status")})},
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
            }
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect", "smoke", "develop"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: this shard's problems")
    parser.add_argument("--noise", type=float, action="append", help="develop only: these noise levels (default: all four)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {p["name"]: p for p in base.load_manifest()}
    if args.command in ("smoke", "develop"):  # development problems only; never the held-out ones
        names = [name for name in sorted(entries) if used_in_development(name)]
        names = names[:3] if args.command == "smoke" else names
        levels = (0.01,) if args.command == "smoke" else tuple(args.noise or NOISE_LEVELS)
        k, n = (int(v) for v in args.shard.split("/"))
        failed = 0
        for index, (noise, problem) in enumerate((noise, problem) for noise in levels for problem in names):
            if index % n != k:
                continue
            row = path_run.run_one({"problem": problem, "noise": noise, "seed": base.SEED, "engine": ENGINE}, entries, 0.0, base.TIME_LIMIT_S)
            print("ROW " + json.dumps({key: v for key, v in row.items() if key not in ("candidates", "trace")}, default=str), flush=True)
            failed += row["status"] == "error"
        return 1 if failed else 0
    check_frozen()
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for job in base.shard(all_jobs(), k, n):
                handle.write(json.dumps(path_run.run_one(job, entries, 0.0, base.TIME_LIMIT_S), sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    expected = len(all_jobs())
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} runs, found {len(rows)}")
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(
        {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summarise(rows)},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
