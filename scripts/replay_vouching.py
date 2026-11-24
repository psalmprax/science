#!/usr/bin/env python3
"""What ``discovery_l2/vouching.py`` would have done to the laws the route reported in the red-team campaign.

    python scripts/replay_vouching.py run --per-cell 600 --out benchmarks/vouching_dev_replay.json

**Development, not a result.** The campaign's worlds were chosen to defeat the route, their
answers are known, and the checks replayed here were written after looking at them. This says
what was built; a run fixed in advance on other laws says whether it works.

Nothing is searched again. Every recorded world is drawn again from its own description (a world
is a law, candidates, a way of drawing rows and a seed), the law the route reported and the
candidates it could not tell from it are read from the record, and ``vouching.vouch`` is given
the same development rows and pool inputs the route had. A law that is passed on unchanged keeps
its recorded score; one that is replaced by another member of its class is scored by the
campaign's own scorer.
"""
from __future__ import annotations

import argparse
import gzip
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
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import vouching  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

WORLDS = ROOT / "benchmarks" / "red_team_worlds.jsonl.gz"
SEED = 20261129
STAGES = ("survey", "climb")
KINDS = ("the law", "accurate, not the law", "wrong")


def kind_of(scored: dict[str, Any]) -> str:
    if not scored.get("reported"):
        return "nothing"
    if scored.get("symbolic"):
        return "the law"
    return "wrong" if scored.get("wrong") else "accurate, not the law"


def recorded(per_cell: int, seed: int = SEED) -> list[dict[str, Any]]:
    """Worlds of the campaign in which the route reported a law: up to ``per_cell`` of each arena and stage."""
    with gzip.open(WORLDS, "rt", encoding="utf-8") as handle:
        rows = [row for row in map(json.loads, handle) if row.get("status") == "ok" and row["route"]["reported"]]
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for arena in sorted({row["arena"] for row in rows}):
        for stage in STAGES:
            cell = [row for row in rows if row["arena"] == arena and row["stage"] == stage]
            chosen += cell if len(cell) <= per_cell else rng.sample(cell, per_cell)
    return chosen


def replay(row: dict[str, Any]) -> dict[str, Any]:
    """One recorded world: what was reported, what the checks make of it, and how that scores."""
    world = rt.World(**{**row["world"], "candidates": tuple(row["world"]["candidates"])})
    rows = rt.rows_of(world)
    names = [f"x{i}" for i in range(world.inputs)]
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    started = time.perf_counter()
    done = vouching.vouch(row["route"]["law"], list(row["route"]["equivalent"]), names, rows.X[dev], rows.y[dev], rows.X[pool])
    out = {"arena": row["arena"], "stage": row["stage"], "noise": world.noise, "rows": world.rows, "inputs": world.inputs,
           "truth_listed": world.truth_listed, "best_fit_wrong": bool(row["best_fit"]["wrong"]),
           "before": kind_of(row["route"]), "reason": done.reason, "switched": done.switched,
           "power": done.power, "members": len(done.members), "refused": sum(m.refused for m in done.members),
           "p_value": done.members[0].p_value if done.members else None, "seconds": round(time.perf_counter() - started, 2)}
    if done.law is None:
        out["after"] = "nothing"
    elif not done.switched:
        out["after"] = out["before"]
    else:
        truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
        out["after"] = kind_of(campaign._scored(done.law, truth, names, rows))
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arena in sorted({row["arena"] for row in rows}):
        for stage in STAGES:
            cell = [row for row in rows if row["arena"] == arena and row["stage"] == stage]
            if not cell:
                continue
            before, after = Counter(row["before"] for row in cell), Counter(row["after"] for row in cell)
            reported = sum(after[k] for k in KINDS)
            out[f"{arena}|{stage}"] = {
                "worlds_with_a_reported_law": len(cell),
                "before": {k: before[k] for k in KINDS},
                "after": {**{k: after[k] for k in KINDS}, "nothing": after["nothing"]},
                "wrong_share_before": round(before["wrong"] / len(cell), 4),
                "wrong_share_after": round(after["wrong"] / reported, 4) if reported else None,
                "laws_kept": [after["the law"], before["the law"]],
                "wrong_laws_stopped": [before["wrong"] - sum(1 for r in cell if r["before"] == "wrong" and r["after"] == "wrong"),
                                       before["wrong"]],
                "laws_lost_by_reason": dict(Counter(r["reason"][:60] for r in cell
                                                    if r["before"] == "the law" and r["after"] != "the law").most_common(6)),
                "switched": sum(r["switched"] for r in cell),
                "switched_into_the_law": sum(r["switched"] and r["after"] == "the law" and r["before"] != "the law" for r in cell)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--per-cell", type=int, default=600)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    chosen = recorded(args.per_cell)
    started = time.perf_counter()
    with mp.get_context("fork").Pool(args.cores) as pool:
        rows = []
        for done in pool.imap_unordered(replay, chosen, chunksize=8):
            rows.append(done)
            if len(rows) % 500 == 0:
                print(f"{len(rows)} of {len(chosen)} in {time.perf_counter() - started:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["arena"], r["stage"], r["noise"], r["before"], r["after"], r["reason"]))
    record = {"what": "Development. The red-team campaign's recorded worlds, the route's reported law and unseparated "
                      "candidates read from the record, and vouching.vouch applied to them. Chosen to defeat the route; "
                      "answers known; the checks were written after looking at them.",
              "source": "benchmarks/red_team_worlds.jsonl.gz", "per_cell": args.per_cell, "sample_seed": SEED,
              "settings": {"accuracy": vouching.ACCURACY_R2, "level": vouching.LEVEL, "power": vouching.POWER,
                           "scales": list(vouching.SCALES), "flips": vouching.FLIPS, "trials": vouching.TRIALS},
              "worlds": len(rows), "summary": summarise(rows)}
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with gzip.open(args.out.with_name(args.out.stem + "_rows.jsonl.gz"), "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    for cell, s in record["summary"].items():
        print(f"{cell:20s} before {s['before']} -> after {s['after']} | laws kept {s['laws_kept']} | wrong stopped {s['wrong_laws_stopped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
