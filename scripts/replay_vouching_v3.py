#!/usr/bin/env python3
"""Development replay: the second version of the last checks, the third, and a switch by noise.

**Development, not a result.** The same records as ``replay_vouching_v2.py``
(the physics benchmark's run behind the four searches, the counterfeit
benchmark, a sample of each red-team campaign), drawn again with the same
rows. On each record, three arms:

* ``second``: ``vouching_v2.vouch``, the version in force;
* ``third``: the second, then the pole rule of ``vouching_v3``;
* ``switch``: the second where the noise estimated from the route's law is at
  most 5 % of the target's size (``measured_range.noise_share``), the first
  version above it.

The third version was designed after reading the second campaign's records
(the wrong laws the second version passed there have poles), so its numbers on
that campaign are not a test of it.

    python scripts/replay_vouching_v3.py run --source campaign2 --per-cell 150 --out benchmarks/vouching_v3_dev_campaign2.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
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

import replay_vouching_v2 as replay  # noqa: E402
import run_red_team as campaign  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import measured_range, vouching, vouching_v2, vouching_v3  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

ARMS = ("second", "third", "switch")


def _arms(law: str, others: list[str], names: list[str], X: np.ndarray, y: np.ndarray, X_more: np.ndarray,
          judge: Any) -> dict[str, Any]:
    """The three arms on one record; ``judge(text)`` says what a reported text is: the law, accurate, or wrong."""
    out: dict[str, Any] = {}
    started = time.perf_counter()
    second = vouching_v2.vouch(law, others, names, X, y, X_more)
    third = vouching_v3.after_second(second, names, X, X_more)
    read = measured_range.noise_share(law, names, X, y)
    noisy = read is not None and read[0] > measured_range.MEASURED_NOISE
    switch = vouching.vouch(law, others, names, X, y, X_more) if noisy else second
    for arm, done in (("second", second), ("third", third), ("switch", switch)):
        out[arm] = judge(done.law) if done.law is not None else "nothing"
        out[f"{arm}_reason"] = done.reason[:120]
    out["noise_estimate"] = None if read is None else round(read[0], 5)
    out["pole"] = third.record.get("pole_rule") if second.law is not None else None
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def _srbench(row: dict[str, Any]) -> dict[str, Any]:
    if not replay._ENTRIES:
        replay._ENTRIES.update({p["name"]: p for p in base.load_manifest()})
    entry = replay._ENTRIES[row["problem"]]
    split = base.make_split(entry, path_run.srb.load_dataset(entry), float(row["noise"]))
    names, X, y = split["names"], split["X_train"], split["y_train"]
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    truth = base.truth_expr(entry)
    law = row["route"]["formula"]

    def judge(text: str) -> str:
        model = sp.sympify(text, locals={n: sp.Symbol(n) for n in names})
        sym = bool(base.symbolic_recovery(model, truth, names, split["X_test"])["recovered"])
        r2 = base._r2(split["y_test"], path_run._predict(model, names, split["X_test"]))
        return str(replay.kind(True, sym, bool(r2 is not None and r2 >= base.R2_ACCURACY)))

    out = {"source": "srbench", "case": f"{row['problem']}|{row['noise']}", "noise": float(row["noise"]),
           "inputs": len(names), "before": replay.kind(True, bool(row["route"]["symbolic"]), bool(row["route"]["accuracy_solution"]))}
    return {**out, **_arms(law, list(row.get("equivalent") or []), names, X[dev], y[dev], X[pool], judge)}


def _world(row: dict[str, Any], source: str) -> dict[str, Any]:
    world = rt.World.from_dict(row["world"])
    names = [f"x{i}" for i in range(world.inputs)]
    truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
    data = rt.rows_of(world)
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    route = row["route"]

    def judge(text: str) -> str:
        scored = campaign._scored(text, truth, names, data)
        return str(replay.kind(scored["reported"], scored["symbolic"], not scored["wrong"]))

    out = {"source": source, "case": row.get("case") or f"{row.get('arena')}|{row.get('stage')}",
           "arena": row.get("arena", "counterfeit"), "stage": row.get("stage", "run"), "noise": float(world.noise),
           "rows": world.rows, "inputs": world.inputs, "before": replay.kind(True, bool(route["symbolic"]), not bool(route["wrong"]))}
    return {**out, **_arms(route["law"], list(route.get("equivalent") or []), names, data.X[dev], data.y[dev],
                           data.X[pool], judge)}


def one(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    source, row = job
    try:
        return _srbench(row) if source == "srbench" else _world(row, source)
    except Exception as exc:  # noqa: BLE001 - a record that cannot be replayed is counted, with its error
        return {"source": source, "case": str(row.get("case") or row.get("problem")), "error": f"{type(exc).__name__}: {exc}"[:200]}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    out: dict[str, Any] = {"records": len(rows), "errors": len(rows) - len(ok), "cells": {}}
    for band, sel in (("all", ok), ("noise up to 5 %", [r for r in ok if r["noise"] <= 0.05 + 1e-12]),
                      ("noise above 5 %", [r for r in ok if r["noise"] > 0.05 + 1e-12])):
        if not sel:
            continue
        right = [r for r in sel if r["before"] == "the law"]
        cell: dict[str, Any] = {"n": len(sel), "route_wrong": sum(r["before"] == "wrong" for r in sel), "right": len(right)}
        for arm in ARMS:
            cell[arm] = {"wrong": sum(r[arm] == "wrong" for r in sel), "kept": sum(r[arm] == "the law" for r in right),
                         "the_law": sum(r[arm] == "the law" for r in sel), "reported": sum(r[arm] != "nothing" for r in sel)}
        out["cells"][band] = cell
    out["pole_withheld"] = dict(Counter(r["second"] for r in ok if r["second"] != "nothing" and r["third"] == "nothing"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--source", choices=tuple(replay.SOURCES), required=True)
    parser.add_argument("--per-cell", type=int, default=150)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=2)
    args = parser.parse_args()
    jobs = [(args.source, row) for row in replay.records(args.source, args.per_cell)]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with mp.get_context("fork").Pool(args.cores) as pool:
        for k, result in enumerate(pool.imap(one, jobs, chunksize=4), 1):
            rows.append(result)
            if k % 100 == 0:
                print(f"{k} of {len(jobs)} in {time.perf_counter() - started:.0f} s", flush=True)
    record = {"what": "development replay of the last checks' second and third versions and a switch by noise; not a result",
              "source": args.source, "per_cell": args.per_cell, "summary": summarise(rows)}
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with gzip.open(args.out.with_suffix(".rows.jsonl.gz"), "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
