#!/usr/bin/env python3
"""Development check of ``discovery_l2/measured_range.py``: does its mark say what it should on recorded worlds?

**Development, not a result.** The mark tells a user when a table lies
outside the conditions the route in force was measured on. This script
rebuilds recorded worlds whose noise, inputs and rows are known, takes the
law the route reported on each, and compares the mark with the truth:

* the blind run's handed cases (``RESULT-VOUCHING-V2-2026-11/handed.jsonl.gz``):
  1,000 rows, up to three inputs, Gaussian noise 0, 1 % and 5 %, all inside
  the measured conditions by construction, so a mark of "outside" there is a
  false alarm unless the law is wrong;
* a sample of the second red-team campaign's worlds (``--per-cell``), with
  noise up to 10 %, 60 to 1,000 rows, noise of other shapes and rows not
  spread evenly.

    python scripts/check_measured_range.py --per-cell 100 --out benchmarks/measured_range_dev.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_vouching_v2 as replay  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import measured_range as mr  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

HANDED = ROOT / "preregistration" / "RESULT-VOUCHING-V2-2026-11" / "handed.jsonl.gz"


def marked(world_dict: dict[str, Any], route: dict[str, Any], source: str) -> dict[str, Any]:
    """The mark on the rows the route fitted its law on, beside what is known about the world."""
    fields = set(rt.World.__dataclass_fields__)
    world = rt.World.from_dict({k: v for k, v in world_dict.items() if k in fields})
    names = [f"x{i}" for i in range(world.inputs)]
    data = rt.rows_of(world)
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)
    dev = order[:round(path_run.DEV_SHARE * world.rows)]
    mark = mr.assess(route["law"], names, data.X[dev], data.y[dev])
    inside = (world.noise <= mr.MEASURED_NOISE + 1e-12 and world.inputs <= mr.MEASURED_INPUTS
              and len(dev) >= mr.MEASURED_ROWS)
    return {"source": source, "noise": world.noise, "inputs": world.inputs, "rows": len(dev),
            "noise_shape": world.noise_shape, "sampling": world.sampling, "route_wrong": bool(route["wrong"]),
            "truly_inside": inside, "state": mark["state"], "estimate": mark["noise"], "error": mark["noise_error"]}


def collect(per_cell: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(HANDED, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok" and row["route"].get("reported") and row["route"].get("law"):
                rows.append(marked(row["world"], row["route"], "blind handed"))
    for row in replay.records("campaign2", per_cell):
        rows.append(marked(row["world"], row["route"], "campaign2"))
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def band(noise: float) -> str:
        return "noise 0" if noise == 0 else "up to 1 %" if noise <= 0.01 + 1e-12 else \
            "1 to 5 %" if noise <= 0.05 + 1e-12 else "above 5 %"

    out: dict[str, Any] = {"records": len(rows), "cells": {}}
    for source in sorted({r["source"] for r in rows}):
        for b in ("noise 0", "up to 1 %", "1 to 5 %", "above 5 %"):
            for wrong in (False, True):
                sel = [r for r in rows if r["source"] == source and band(r["noise"]) == b and r["route_wrong"] == wrong]
                if not sel:
                    continue
                right_gauss = [r for r in sel if r["noise"] > 0 and r["estimate"] is not None and r["noise_shape"] == "gaussian"]
                ratio = [r["estimate"] / r["noise"] for r in right_gauss]
                out["cells"][f"{source} | {b} | {'wrong law' if wrong else 'not wrong'}"] = {
                    "n": len(sel), "states": dict(Counter(r["state"] for r in sel)),
                    "truly_inside": sum(r["truly_inside"] for r in sel),
                    "estimate_over_truth_median": float(np.median(ratio)) if ratio else None,
                }
    inside = [r for r in rows if r["truly_inside"] and not r["route_wrong"]]
    outside = [r for r in rows if not r["truly_inside"]]
    out["false_alarms_inside"] = [sum(r["state"] == mr.OUTSIDE for r in inside), len(inside)]
    out["missed_outside"] = [sum(r["state"] == mr.INSIDE for r in outside), len(outside)]
    out["missed_outside_by_reason"] = dict(Counter(
        ("noise" if r["noise"] > mr.MEASURED_NOISE + 1e-12 else "") + ("rows" if r["rows"] < mr.MEASURED_ROWS else "")
        + ("inputs" if r["inputs"] > mr.MEASURED_INPUTS else "") for r in outside if r["state"] == mr.INSIDE))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-cell", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.per_cell)
    record = {"what": "development check of discovery_l2/measured_range.py on recorded worlds; not a result",
              "summary": summarise(rows)}
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
