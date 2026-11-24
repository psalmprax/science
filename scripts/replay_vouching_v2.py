#!/usr/bin/env python3
"""What the second version of the last checks would have done on every recorded run that showed the first version's faults.

    python scripts/replay_vouching_v2.py run --source srbench|counterfeit|campaign1|campaign2 [--per-cell N] --out FILE

**Development, not a result.** Every source here was read before the second version was
written, its faults (audit M-33, M-34, M-35) were found on them, and the second version was
changed while being replayed on them: rival shapes were taken out of its tests, its polynomial
shapes raised to the fourth degree, and a largest-residual test dropped. This says what was built.
Nothing is searched again: each record is drawn again from its own description, the law the
route reported and the candidates it could not tell from it are read from the record, and both
versions are given the same development rows and pool inputs the route had. A law each version
reports is scored by the source's own scorer.

* ``srbench``: the 266 runs of PREREG-VOUCHED-SRBENCH (the physics benchmark behind four searches);
* ``counterfeit``: the 360 cases of PREREG-COUNTERFEIT;
* ``campaign1``, ``campaign2``: the red-team campaigns' worlds in which the route reported a law,
  up to ``--per-cell`` of each arena and stage.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_red_team as campaign  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import vouching, vouching_v2  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

SOURCES = {
    "srbench": ROOT / "preregistration" / "RESULT-VOUCHED-SRBENCH-2026-11" / "runs.jsonl.gz",
    "counterfeit": ROOT / "preregistration" / "RESULT-COUNTERFEIT-2026-11" / "runs.jsonl.gz",
    "campaign1": ROOT / "benchmarks" / "red_team_worlds.jsonl.gz",
    "campaign2": ROOT / "benchmarks" / "red_team_checks_worlds.jsonl.gz",
}
SEED = 20261202
VERSIONS = ("first", "second")


def kind(reported: bool, symbolic: bool, accurate: bool) -> str:
    if not reported:
        return "nothing"
    if symbolic:
        return "the law"
    return "accurate, not the law" if accurate else "wrong"


def _read(source: str) -> list[dict[str, Any]]:
    with gzip.open(SOURCES[source], "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def records(source: str, per_cell: int, seed: int = SEED) -> list[dict[str, Any]]:
    """The records whose route reported a law (all of them for the two benchmarks; a sample per cell for the campaigns)."""
    rows = _read(source)
    if source == "srbench":
        return [row for row in rows if row["status"] == "ok" and row["route"].get("formula")]
    if source == "counterfeit":
        return [row for row in rows if row.get("status") == "ok" and row["route"]["reported"]]
    rows = [row for row in rows if row.get("status") == "ok" and row["route"]["reported"]]
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row.get("arena", ""), row.get("stage", "")), []).append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for key in sorted(cells):
        sel = cells[key]
        out.extend(rng.sample(sel, min(per_cell, len(sel))))
    return out


_ENTRIES: dict[str, Any] = {}


def _srbench(row: dict[str, Any]) -> dict[str, Any]:
    if not _ENTRIES:
        _ENTRIES.update({p["name"]: p for p in base.load_manifest()})
    entry = _ENTRIES[row["problem"]]
    split = base.make_split(entry, path_run.srb.load_dataset(entry), float(row["noise"]))
    names, X, y = split["names"], split["X_train"], split["y_train"]
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    law, others = row["route"]["formula"], list(row.get("equivalent") or [])
    out: dict[str, Any] = {"source": "srbench", "case": f"{row['problem']}|{row['noise']}", "noise": float(row["noise"]),
                           "inputs": len(names), "before": kind(True, bool(row["route"]["symbolic"]),
                                                                bool(row["route"]["accuracy_solution"]))}
    truth = base.truth_expr(entry)
    for version, module in (("first", vouching), ("second", vouching_v2)):
        started = time.perf_counter()
        done = module.vouch(law, others, names, X[dev], y[dev], X[pool])
        if done.law is None:
            out[version] = "nothing"
        elif done.law == law:
            out[version] = out["before"]
        else:
            model = sp.sympify(done.law, locals={n: sp.Symbol(n) for n in names})
            sym = bool(base.symbolic_recovery(model, truth, names, split["X_test"])["recovered"])
            r2 = base._r2(split["y_test"], path_run._predict(model, names, split["X_test"]))
            out[version] = kind(True, sym, bool(r2 is not None and r2 >= base.R2_ACCURACY))
        out[f"{version}_reason"] = done.reason[:90]
        out[f"{version}_seconds"] = round(time.perf_counter() - started, 3)
    return out


def _world(row: dict[str, Any], source: str) -> dict[str, Any]:
    world = rt.World.from_dict(row["world"])
    names = [f"x{i}" for i in range(world.inputs)]
    truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
    data = rt.rows_of(world)
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    law, others = row["route"]["law"], list(row["route"].get("equivalent") or [])
    route = row["route"]
    out: dict[str, Any] = {"source": source, "case": row.get("case") or f"{row.get('arena')}|{row.get('stage')}",
                           "arena": row.get("arena", "counterfeit"), "stage": row.get("stage", "run"),
                           "noise": float(world.noise), "rows": world.rows, "inputs": world.inputs,
                           "noise_shape": world.noise_shape, "sampling": world.sampling,
                           "before": kind(True, bool(route["symbolic"]), not bool(route["wrong"]))}
    for version, module in (("first", vouching), ("second", vouching_v2)):
        started = time.perf_counter()
        done = module.vouch(law, others, names, data.X[dev], data.y[dev], data.X[pool])
        scored = campaign._scored(done.law, truth, names, data)
        out[version] = kind(scored["reported"], scored["symbolic"], not scored["wrong"])
        out[f"{version}_reason"] = done.reason[:90]
        out[f"{version}_seconds"] = round(time.perf_counter() - started, 3)
    return out


def replay(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    source, row = job
    try:
        return _srbench(row) if source == "srbench" else _world(row, source)
    except Exception as exc:  # noqa: BLE001 - a record that cannot be replayed is counted, with its error
        return {"source": source, "case": str(row.get("case") or row.get("problem")), "error": f"{type(exc).__name__}: {exc}"[:200]}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Before and after each version, overall and by noise level (and by arena and stage for the campaigns)."""
    ok = [row for row in rows if "error" not in row]
    out: dict[str, Any] = {"records": len(rows), "errors": len(rows) - len(ok), "cells": {}}

    def cell(sel: list[dict[str, Any]]) -> dict[str, Any]:
        entry: dict[str, Any] = {"n": len(sel), "before": dict(Counter(row["before"] for row in sel))}
        right = [row for row in sel if row["before"] == "the law"]
        wrong = [row for row in sel if row["before"] == "wrong"]
        for version in VERSIONS:
            entry[version] = dict(Counter(row[version] for row in sel))
            entry[f"{version}_kept"] = [sum(row[version] == "the law" for row in right), len(right)]
            entry[f"{version}_wrong_stopped"] = [sum(row[version] != "wrong" for row in wrong), len(wrong)]
            reported = sum(row[version] != "nothing" for row in sel)
            entry[f"{version}_wrong_share"] = (sum(row[version] == "wrong" for row in sel) / reported) if reported else None
        return entry

    out["cells"]["all"] = cell(ok)
    for noise in sorted({row["noise"] for row in ok}):
        out["cells"][f"noise {noise:g}"] = cell([row for row in ok if row["noise"] == noise])
    for key in sorted({(row.get("arena"), row.get("stage")) for row in ok if row.get("arena")}):
        out["cells"][f"{key[0]}|{key[1]}"] = cell([row for row in ok if (row.get("arena"), row.get("stage")) == key])
    out["second_withheld_right_laws_by_reason"] = dict(Counter(row["second_reason"][:60] for row in ok
                                                               if row["before"] == "the law" and row["second"] == "nothing"))
    out["second_passed_wrong"] = [{k: row.get(k) for k in ("case", "arena", "stage", "noise", "rows", "inputs")}
                                  for row in ok if row["second"] == "wrong"][:40]
    out["seconds"] = {version: round(float(np.mean([row[f"{version}_seconds"] for row in ok])), 3) if ok else None
                      for version in VERSIONS}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--per-cell", type=int, default=300)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs = [(args.source, row) for row in records(args.source, args.per_cell)]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with mp.get_context("fork").Pool(args.cores) as pool:
        for k, row in enumerate(pool.imap(replay, jobs, chunksize=4), 1):
            rows.append(row)
            if k % 200 == 0:
                print(f"{k} of {len(jobs)} in {time.perf_counter() - started:.0f} s", flush=True)
    summary = summarise(rows)
    args.out.write_text(json.dumps({"what": "Development replay of both versions of the last checks.", "source": args.source,
                                    "summary": summary}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with gzip.open(args.out.with_suffix(".rows.jsonl.gz"), "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary["cells"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
