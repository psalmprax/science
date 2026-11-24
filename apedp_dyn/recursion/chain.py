"""Level 1 cut into steps that separate processes can run, the state handed from one to the next.

Status: experimental

With the route's search stage switched on, one problem costs minutes of
processor time and a whole Level 1 run does not fit in one job of a hosted
runner. This module is ``levels.run_level1`` as steps:

* ``plan``: seals the judge, draws **every generation's measurement problems
  before anything runs**, and writes the starting state. Drawing them first
  does two things: no later study or selection batch can repeat one of them
  (they are in the curriculum's record of what was drawn), and what a
  generation is measured on cannot depend on anything that generation learned;
* ``improve_step``: generation ``k``'s study, settings copies and code step
  (``levels.improve``), from the state of generation ``k - 1``;
* ``measure_step``: one arm on generation ``k``'s problems, or a shard of them.
  The arm ``current`` is the searcher as it stands after generation ``k``.
  The arm ``start`` is the searcher the run began with, on the same problems;
* ``collect``: the measurements joined, ``progress.verdict`` on the arm
  ``current``, and for each generation the problems the current searcher
  solved that the starting one did not, and the reverse.

**Why the second arm.** Each generation is measured on different problems, so
the rate moves with how hard a batch happens to be (in the runs without
searches it went 11, 33, 0, 18, 8 with nothing changing underneath). The same
problems given to the starting searcher take that out: what the current
searcher gained or lost is counted problem by problem.

**The judge across processes.** The seal taken by ``plan`` travels in the
state, and every step checks the files on its own disk against it before
running, so a step run from a tree with a different referee, threshold or
curriculum stops instead of adding to the run.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from apedp_dyn.recursion import judge, levels
from apedp_dyn.recursion.curriculum import Curriculum, Problem
from apedp_dyn.recursion.progress import verdict

ARMS = ("current", "start")


def plan(start: levels.Searcher, *, generations: int, measure: int, meta: levels.Meta, seed: int, tag: str = "chain"
         ) -> dict[str, Any]:
    """The starting state: the seal, the curriculum, and every generation's measurement problems, drawn now."""
    curriculum = Curriculum(int.from_bytes(hashlib.sha256(f"{seed}|{tag}".encode()).digest()[:6], "big"))
    measurement = {str(k): [p.as_dict() for p in curriculum.measurement(measure, f"measurement {k}")]
                   for k in range(generations + 1)}
    return {"v": 1, "seed": seed, "tag": tag, "generations": generations, "generation": 0, "meta": meta.as_dict(),
            "start": start.as_dict(), "searcher": start.as_dict(), "curriculum": curriculum.as_dict(), "carried": [],
            "measurement": measurement, "sealed": judge.seal()}


def improve_step(state: dict[str, Any], k: int, *, workers: int = 1, client: Callable[[str], str] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generation ``k`` from the state of generation ``k - 1``: the new state and the generation's record."""
    judge.check(state["sealed"])
    if k != int(state["generation"]) + 1:
        raise ValueError(f"the state is of generation {state['generation']}; generation {k} cannot follow it")
    curriculum = Curriculum.from_dict(state["curriculum"])
    rng = np.random.default_rng([curriculum.seed % 2 ** 32, k])
    chosen, record, unsolved = levels.improve(
        levels.Searcher.from_dict(state["searcher"]), curriculum, levels.Meta(**state["meta"]), k, workers=workers,
        client=client, carried=[Problem(**p) for p in state["carried"]], sealed=state["sealed"], rng=rng)
    after = {**state, "generation": k, "searcher": chosen.as_dict(), "curriculum": curriculum.as_dict(),
             "carried": [p.as_dict() for p in unsolved]}
    return after, record


def measure_step(state: dict[str, Any], k: int, *, arm: str = "current", shard: int = 0, shards: int = 1,
                 workers: int = 1) -> dict[str, Any]:
    """One arm on generation ``k``'s problems (or every ``shards``-th of them from ``shard``), learning nothing."""
    judge.check(state["sealed"])
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if arm == "current" and int(state["generation"]) != k:
        raise ValueError(f"the state is of generation {state['generation']}, not {k}: its searcher is not generation {k}'s")
    searcher = levels.Searcher.from_dict(state["searcher" if arm == "current" else "start"])
    problems = [Problem(**p) for p in state["measurement"][str(k)]][shard::shards]
    rows = levels.run_batch(searcher, problems, workers)
    return {"generation": k, "arm": arm, "shard": shard, "shards": shards, "settings": dict(searcher.settings),
            "laws_known": len(searcher.laws), "rows": rows, **levels.tally(rows)}


def collect(state: dict[str, Any], improved: Sequence[dict[str, Any]], measured: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The run's record from its pieces. A generation whose measurement is incomplete is listed and left out of the verdict."""
    expected = {int(k): {p["key"] for p in problems} for k, problems in state["measurement"].items()}
    rows: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for piece in measured:
        bucket = rows.setdefault((int(piece["generation"]), str(piece["arm"])), {})
        for row in piece["rows"]:
            bucket[row["key"]] = row
    for k in expected:                                   # at generation 0 the two arms are the same searcher
        if k == 0 and (0, "current") in rows:
            rows.setdefault((0, "start"), rows[(0, "current")])
    arms: dict[str, list[Any]] = {arm: [] for arm in ARMS}
    incomplete: list[dict[str, Any]] = []
    for arm in ARMS:
        for k in sorted(expected):
            got = rows.get((k, arm), {})
            if set(got) != expected[k]:
                incomplete.append({"generation": k, "arm": arm, "rows": len(got), "expected": len(expected[k])})
                continue
            arms[arm].append(levels.measurement_of(k, list(got.values())))
    paired = []
    for k in sorted(expected):
        current, start = rows.get((k, "current"), {}), rows.get((k, "start"), {})
        if set(current) != expected[k] or set(start) != expected[k]:
            continue
        gained = sorted(key for key in expected[k] if current[key]["solved"] and not start[key]["solved"])
        lost = sorted(key for key in expected[k] if start[key]["solved"] and not current[key]["solved"])
        paired.append({"generation": k, "problems": len(expected[k]), "gained": len(gained), "lost": len(lost),
                       "solved_current": sum(r["solved"] for r in current.values()),
                       "solved_start": sum(r["solved"] for r in start.values()),
                       "wrong_current": sum(r["wrong"] for r in current.values()),
                       "wrong_start": sum(r["wrong"] for r in start.values())})
    improving = float(sum(record.get("cpu_seconds_improving", 0.0) for record in improved))
    measuring = float(sum(m.cpu_seconds for m in arms["current"]))
    return {"seed": state["seed"], "tag": state["tag"], "meta": state["meta"], "start": state["start"]["settings"],
            "generations": sorted(improved, key=lambda record: int(record["generation"])),
            "measurements": [m.as_dict() for m in arms["current"]],
            "measurements_start": [m.as_dict() for m in arms["start"]],
            "progress": verdict(arms["current"]), "paired": paired, "incomplete": incomplete,
            "searcher": state["searcher"], "curriculum": state["curriculum"]["history"],
            "sealed_files": len(state["sealed"]), "cpu_seconds": round(improving + measuring, 1)}
