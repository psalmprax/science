#!/usr/bin/env python3
"""Development: where the true law lands in the memory's proposals, as the memory is taught more laws.

Status: audit-only

**Development, not a result.** Nothing was fixed beforehand. The measurement is
``apedp_dyn/recursion/order_score.py``: on fresh problems of the curriculum, the place of the first
candidate the judge would count as solved in a proposer's ordered list. The proposer is the memory
of confirmed laws with its work counted in fits (``proposers/structure_memory_counted.py``), so the
same table and memory give the same list on any machine.

Arms: the memory taught 0, 5, 20, 80 and 320 laws of a teaching set drawn apart from the test
problems (no test problem's structure is in it), and the memory in force for tables
(``memory_stage.recorded_memory``). A memory "taught" a law here is given the generator's own law:
that is an upper bound on what the same number of laws confirmed by the referee would teach, and it
says whether the learner has room to learn at all.

For every problem the whole list is kept with what each candidate was (accurate, solved, same form)
and a few cheap features, so that other orders of the same candidates can be scored afterwards
without running anything again.

``--set fresh`` reads a second set of 200 problems drawn afterwards, none with the structure of a
problem of the first set or of a law taught; the memories are the same as for the first set.

    python scripts/measure_order_score.py run --out DIR --arm taught-80 --shard 0 --shards 8
    python scripts/measure_order_score.py collect --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.proposers import structure_memory as timed  # noqa: E402
from apedp_dyn.proposers.structure_memory import Memory  # noqa: E402
from apedp_dyn.proposers.structure_memory_counted import (  # noqa: E402
    GROW_FITS,
    WHOLE_FITS,
    grow_counted,
    remembered_whole_counted,
)
from apedp_dyn.recursion import judge, mdl, order_score  # noqa: E402
from apedp_dyn.recursion.curriculum import (  # noqa: E402
    MEASURED_LEVELS,
    Curriculum,
    Problem,
    structure,
)

TAG = "DEV-ORDER-SCORE-2026-11"
TEST_SEED, TEACH_SEED, FRESH_SEED = 2026112501, 2026112502, 2026112503
PROBLEMS = 300
FRESH = 200                     # a second set, drawn after the first run was read; level 1 has no more than 40 unused laws
TAUGHT = (0, 5, 20, 80, 320)
ARMS = (*(f"taught-{n}" for n in TAUGHT), "in-force")
POOL = 40                       # candidates kept for a problem, best BIC first; the product hands the referee the first 3
HANDED = 3


def test_problems(count: int = PROBLEMS) -> list[Problem]:
    return Curriculum(TEST_SEED).measurement(count, "order score, test")


def teaching_problems(count: int, tests: list[Problem]) -> list[Problem]:
    """Problems to teach from, none with the structure of a test problem; from the measured levels, moving on where one runs out."""
    if count <= 0:
        return []
    teacher = Curriculum(TEACH_SEED)
    teacher.seen |= {structure(problem.truth) for problem in tests}
    return teacher._draw(count, "order score, teaching", list(MEASURED_LEVELS))


def teaching_laws(count: int, tests: list[Problem]) -> list[str]:
    return [problem.text for problem in teaching_problems(count, tests)]


def fresh_problems(count: int = FRESH) -> list[Problem]:
    """A second set of problems: none with the structure of a test problem of the first set or of any law taught."""
    tests = test_problems(PROBLEMS)
    drawer = Curriculum(FRESH_SEED)
    drawer.seen |= {structure(problem.truth) for problem in [*tests, *teaching_problems(max(TAUGHT), tests)]}
    return drawer.measurement(count, "order score, fresh")


def memory_of(arm: str, tests: list[Problem]) -> Memory:
    if arm == "in-force":
        from apedp_dyn.discovery_l2 import memory_stage

        return memory_stage.recorded_memory()
    memory = Memory()
    for law in teaching_laws(int(arm.split("-")[1]), tests):
        memory.learn(law)
    return memory


def candidates(memory: Memory, problem: Problem, pool: int = POOL) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The memory's candidates for a problem, best BIC first, as ``CountedMemoryProposer`` orders them, with features."""
    rows = problem.rows()
    names = problem.names
    X, y = rows["X"], rows["y"]
    best, motifs, report = grow_counted(memory, X, y, max_fits=GROW_FITS, keep=pool)
    whole, whole_report = remembered_whole_counted(memory, names, X, y, max_fits=WHOLE_FITS)
    pairs = [*[(g.bic, timed.written(g, names, motifs), len(g.steps)) for g in best], *[(bic, law, 0) for bic, law in whole]]
    symbols = [sp.Symbol(name) for name in names]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bic, law, steps in sorted(pairs, key=lambda item: item[0]):
        if law.free_symbols != set(symbols):
            continue
        key = str(law.xreplace(dict.fromkeys(law.atoms(sp.Float), timed.C)))
        if key in seen:
            continue
        seen.add(key)
        constants = [abs(float(value)) for value in law.atoms(sp.Float)]
        with np.errstate(all="ignore"):
            function = sp.lambdify(symbols, law, "numpy")
            train = np.asarray(function(*X.T), dtype=float) * np.ones(len(X))
            held = np.asarray(function(*rows["pool_X"].T), dtype=float) * np.ones(len(rows["pool_X"]))
        out.append({"text": str(law), "bic": round(float(bic), 3), "steps": int(steps), "constants": len(constants),
                    "small_constants": sum(value < 0.02 for value in constants),
                    "large_constants": sum(value > 50.0 for value in constants),
                    "operations": int(sp.count_ops(law)),
                    "train_r2": judge.r2(y, train), "held_r2": judge.r2(rows["pool_y"], held),
                    "law_bits": round(float(mdl.law_bits(law, len(names))), 1)})
        if len(out) == pool:
            break
    return out, {"fits": report.get("fits"), "motifs": report.get("motifs"), "whole_fits": whole_report.get("fits")}


def one(job: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    arm, memory_record, problem_record = job
    problem = Problem(**problem_record)
    started = time.perf_counter()
    pool, report = candidates(Memory.from_dict(memory_record), problem)
    row = order_score.measure(problem, [item["text"] for item in pool])
    for item, judged in zip(pool, row["candidates"], strict=True):
        judged.update({key: value for key, value in item.items() if key != "text"})
    row.update({"arm": arm, "text": problem.text, "seconds": round(time.perf_counter() - started, 1), **report})
    return row


FIELDS = ("bic", "law_bits", "constants", "small_constants", "large_constants", "operations", "steps", "train_r2", "held_r2",
          "accurate", "solved", "same_form")


def pools(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every problem's candidates as numbers only (``FIELDS``, in the proposer's order): small enough to print in a job's log."""
    def number(value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return None if value is None else int(value)
        return float(f"{float(value):.7g}")
    return {"fields": list(FIELDS),
            "rows": [[row["arm"], row["key"], row["level"], row["inputs"], row["noise"], row["truth_bits"],
                      [[number(item.get(field)) for field in FIELDS] for item in row["candidates"]]]
                     for row in sorted(rows, key=lambda r: (r["arm"], r["key"]))]}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A reading of the collected rows: every arm's places, each arm against the one before, and what can be seen."""
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    by_arm = {arm: sorted(found, key=lambda row: row["key"]) for arm, found in by_arm.items() if found}
    arms = {arm: {"problems": len(found), "motifs": found[0].get("motifs"),
                  "seconds_a_problem_median": float(np.median([row["seconds"] for row in found])),
                  **{what: order_score.summarise(found, what) for what in order_score.PLACES},
                  "solved_among_the_first_3": sum(row["solved_place"] is not None and row["solved_place"] <= HANDED for row in found),
                  "an_accurate_candidate_but_none_counted_solved": sum(row["accurate_place"] is not None and row["solved_place"] is None
                                                                       for row in found)}
            for arm, found in by_arm.items()}
    order = [arm for arm in ARMS if arm in by_arm]
    steps = {f"{second} against {first}": order_score.compare(by_arm[first], by_arm[second])
             for first, second in itertools.pairwise(order) if second != "in-force"}
    reference = by_arm.get("taught-80") or next(iter(by_arm.values()), [])
    seen = {str(n): {f"churn {churn:g}": order_score.smallest_visible_gain(reference, n, churn=churn)
                     for churn in (0.0, 0.1, 0.25)} for n in (10, 48, 100, 300)} if reference else {}
    return {"arms": arms, "steps": steps, "what_can_be_seen": {"measured_on": "taught-80", "by_problems": seen}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "collect", "plan"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--problems", type=int, default=PROBLEMS)
    parser.add_argument("--set", choices=("test", "fresh"), default="test",
                        help="fresh: the second set, drawn apart from the first set and from every law taught")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--processes", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        rows = [json.loads(line) for path in sorted(args.out.glob("rows_*.jsonl")) for line in path.read_text().splitlines()]
        record = {"tag": TAG, "development": True,
                  "what": "development: the place of the true law in the memory's proposals as the memory is taught more laws; "
                          "not a result, nothing was fixed beforehand",
                  "problems": len({row["key"] for row in rows}), "pool": POOL, "handed_to_the_referee": HANDED,
                  "set": args.set, "seeds": {"test": TEST_SEED, "teaching": TEACH_SEED, "fresh": FRESH_SEED}, **summary(rows)}
        (args.out / "order_score.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        with gzip.open(args.out / "order_score_rows.jsonl.gz", "wt", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda r: (r["arm"], r["key"])):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        with gzip.open(args.out / "order_score_pools.json.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(pools(rows), separators=(",", ":")))
        print(json.dumps({arm: {k: v for k, v in block["solved_place"].items() if k != "by_level"}
                          for arm, block in record["arms"].items()}, indent=1))
        return 0
    tests = fresh_problems(min(args.problems, FRESH)) if args.set == "fresh" else test_problems(args.problems)
    if args.command == "plan":
        print(json.dumps({"problems": len(tests), "levels": sorted({p.level for p in tests}), "arms": list(ARMS)}))
        return 0
    if args.arm is None:
        parser.error("run needs --arm")
    memory = memory_of(args.arm, test_problems(PROBLEMS) if args.set == "fresh" else tests).as_dict()
    jobs = [(args.arm, memory, problem.as_dict()) for problem in tests[args.shard::args.shards]]
    processes = args.processes or min(multiprocessing.cpu_count(), 4)
    path = args.out / f"rows_{args.arm}_{args.shard}.jsonl"
    with path.open("w", encoding="utf-8") as handle, multiprocessing.get_context("spawn").Pool(processes) as pool:
        for row in pool.imap(one, jobs):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
