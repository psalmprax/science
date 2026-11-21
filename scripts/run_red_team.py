#!/usr/bin/env python3
"""A campaign against the default law-discovery route: a search for data on which it reports a wrong law.

What the route is known to do comes from benchmarks nobody built to defeat it. This script builds
worlds to defeat it (``apedp_dyn.redteam``): a law from the invented-law grammar, counterfeits of
it handed over as what a search returned (with or without the law itself), rows sampled evenly or
not, noise of the kind the route assumes or another. The route under attack is the one in force
for tables, ``refined_search.discover_refined`` with a search stage; "wrong" is the benchmarks'
own standard (a reported law that is neither the law nor accurate to R^2 0.999 on fresh rows drawn
like the others, without noise).

    python scripts/run_red_team.py laws                    # write benchmarks/red_team_laws.json
    python scripts/run_red_team.py laws --check
    python scripts/run_red_team.py smoke                   # a few worlds, seconds each
    python scripts/run_red_team.py survey --shard 0/24 --minutes 20 --out out
    python scripts/run_red_team.py climb --shard 0/24 --minutes 12 --out out
    python scripts/run_red_team.py collect out --out benchmarks/red_team_campaign.json
    python scripts/run_red_team.py world '{"law": "...", ...}'     # one world again, from its record

``survey`` draws worlds at random in every arena: how often the route is wrong where nothing was
tuned against it yet. ``climb`` takes a shard's worst worlds in one arena and changes them step by
step towards a worse outcome. Two plain ways of ending a search are run on the same rows and the
same candidates for comparison (``redteam.plain_selection``).

**This is development by its nature.** Its worlds are chosen adversarially, so its rates are not
rates on data nobody designed, and a repair found with it has to be tested on worlds drawn
afterwards. The laws are drawn with their own seed and leave out every structure of the
invented-laws benchmark by fingerprint, without reading a held-back law.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.redteam import attack, counterfeit, plain_selection  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

LAWS_FILE = ROOT / "benchmarks" / "red_team_laws.json"
LAW_SEED = 20261125
LAW_COUNT = 150
SEED = 0
TIME_LIMIT_S = 240.0            # one world: the route and the two plain methods
CHECK_ROWS = 30                 # fresh rows the validated plain method reads: the route's own budget
#: the share of a survey's worlds drawn in each arena
SURVEY_SHARES = {"measured": 0.3, "few_rows": 0.15, "sampling": 0.2, "noise_shape": 0.15, "noise_level": 0.2}
METHODS = ("route", "best_fit", "validated")
KEEP, CHILDREN = 8, 3


# ---------------------------------------------------------------- laws
def build_laws(count: int = LAW_COUNT, seed: int = LAW_SEED) -> dict[str, Any]:
    """``count`` laws from the invented-law grammar with their counterfeits, none of them a benchmark structure."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(laws) == count:
            break
        inputs = int(rng.integers(1, 4))
        recipe = counterfeit.draw_recipe(rng, inputs, grammar.CONSTANTS)
        expr = counterfeit.law_of(recipe, inputs)
        key = grammar.structure(expr)
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark:
            continue
        fakes = counterfeit.counterfeits(recipe, inputs, grammar.LOW, grammar.HIGH, seed=seed + len(laws))
        if not fakes:
            continue
        laws.append({"id": 3000 + len(laws), "inputs": inputs, "law": str(expr), "recipe": [asdict(step) for step in recipe],
                     "counterfeits": [asdict(fake) for fake in fakes]})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    return {"what": "Laws for the red team: the invented-law grammar, each with its counterfeits (one block exchanged, "
                    "constants fitted to the law itself). Development material.",
            "seed": seed, "domain": [grammar.LOW, grammar.HIGH], "laws": laws}


def laws() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(LAWS_FILE.read_text(encoding="utf-8"))["laws"]
    return out


# ---------------------------------------------------------------- one world
_GUIDED: Any = None


def _guided() -> Any:
    """The first pass's candidate order, prepared once per process tree (it is the same for every dataset)."""
    global _GUIDED
    if _GUIDED is None:
        from apedp_dyn.discovery_l2 import main_search
        from apedp_dyn.discovery_l2.guided_search import GuidedSearch

        _GUIDED = GuidedSearch(main_search.size_prior)
        for inputs in (1, 2, 3):
            _GUIDED.order(inputs, 6)
    return _GUIDED


def _scored(text: str | None, truth: sp.Expr, names: list[str], rows: rt.Rows) -> dict[str, Any]:
    """A reported law against the truth, by the benchmarks' scoring; ``badness`` is what a climb pushes up."""
    model = sp.sympify(text, locals={n: sp.Symbol(n) for n in names}) if text else None
    if model is None:
        return {"reported": False, "law": None, "symbolic": False, "test_r2": None, "accurate": False, "wrong": False,
                "badness": 0.0}
    recovered = bool(base.symbolic_recovery(model, truth, names, rows.X_test)["recovered"])
    r2 = base._r2(rows.y_test, path_run._predict(model, names, rows.X_test))
    accurate = bool(r2 is not None and r2 >= base.R2_ACCURACY)
    badness = 0.0 if recovered else 1.0 if r2 is None else float(np.clip(1.0 - r2, 0.0, 1.0))
    return {"reported": True, "law": str(text), "symbolic": recovered, "test_r2": r2, "accurate": accurate,
            "wrong": not recovered and not accurate, "badness": badness}


def decide(world: rt.World) -> dict[str, Any]:
    """The route in force and the two plain methods on one world: what each reported and whether it was wrong."""
    from apedp_dyn.discovery_l2 import refined_search
    from apedp_dyn.proposers.fit_guard import FitGuardProposer
    from apedp_dyn.proposers.portfolio import FixedFormulasProposer

    rows = rt.rows_of(world)
    names = [f"x{i}" for i in range(world.inputs)]
    truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    X, y = rows.X, rows.y
    pool_y = y[pool]
    stage = FitGuardProposer(FixedFormulasProposer(list(world.candidates)), target_col="__target__")
    started = time.perf_counter()
    result = refined_search.discover_refined({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                             lambda picked: pool_y[picked], proposer=stage, noise="absolute", seed=SEED,
                                             guided=_guided())
    path = result.path
    assert path is not None
    contrast = path.contrast
    route = {**_scored(result.law, truth, names, rows), "action": result.action, "tentative_law": result.tentative_law,
             "confirmed_as": path.law, "found_by": path.found_by, "rejected_law": path.rejected_law,
             "adequacy": path.adequacy, "rows_read": path.rows_read, "verdict": contrast.verdict if contrast else None,
             "equivalent": list(contrast.equivalent) if contrast else [], "eliminated": len(contrast.eliminated) if contrast else 0,
             "offered": len(path.candidates), "dropped_terms": list(result.dropped_terms),
             "seconds": round(time.perf_counter() - started, 2)}
    plain = plain_selection.best_fit(world.candidates, names, X[dev], y[dev])
    fresh = np.random.default_rng([world.seed, 5]).choice(len(pool), size=min(CHECK_ROWS, len(pool)), replace=False)
    checked = plain_selection.validated(world.candidates, names, X[dev], y[dev], X[pool][fresh], pool_y[fresh])
    return {"route": route, "best_fit": _scored(plain.law, truth, names, rows),
            "validated": {**_scored(checked.law, truth, names, rows), "p_value": checked.p_value}}


def _child(conn: Any, world: rt.World) -> None:
    try:
        conn.send(("ok", decide(world)))
    except Exception as exc:  # noqa: BLE001 - a crash on one world is an outcome of the campaign, recorded with its class
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def evaluate(batch: list[rt.World], cores: int, time_limit: float = TIME_LIMIT_S) -> list[dict[str, Any]]:
    """Every world of ``batch`` in its own process, ``cores`` side by side; one that runs over is killed and recorded."""
    _guided()
    context = mp.get_context("fork")
    waiting = list(enumerate(batch))
    live: dict[int, tuple[Any, Any, float]] = {}
    out: list[dict[str, Any]] = [{} for _ in batch]
    while waiting or live:
        while waiting and len(live) < cores:
            index, world = waiting.pop(0)
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_child, args=(child, world))
            process.start()
            child.close()
            live[index] = (process, parent, time.perf_counter())
        for index, (process, conn, started) in list(live.items()):
            status: tuple[str, Any] | None = None
            if conn.poll(0.02):
                try:
                    status = conn.recv()
                except EOFError:
                    status = ("error", {"error": "the world's process died without a result"})
            elif time.perf_counter() - started > time_limit:
                status = ("timeout", {})
            if status is None:
                continue
            process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join()
            del live[index]
            out[index] = {"world": batch[index].as_dict(), "status": status[0], **status[1]}
    return out


def badness(row: dict[str, Any]) -> float:
    return float(row["route"]["badness"]) if row.get("status") == "ok" and row["route"]["reported"] else 0.0


# ---------------------------------------------------------------- campaign
def _cores() -> int:
    return max(1, int(os.environ.get("RED_TEAM_CORES", os.cpu_count() or 1)))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _read(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def survey(shard: int, minutes: float, out: Path, batch_size: int | None = None) -> int:
    """Random worlds in every arena until the time is up; returns how many were run."""
    pool = laws()
    rng = np.random.default_rng([LAW_SEED, 1, shard])
    names, shares = list(SURVEY_SHARES), np.array(list(SURVEY_SHARES.values()))
    cores = _cores()
    target = out / f"survey-{shard:03d}.jsonl.gz"
    deadline, done = time.monotonic() + 60.0 * minutes, 0
    while time.monotonic() < deadline:
        chosen = [str(rng.choice(names, p=shares / shares.sum())) for _ in range(batch_size or 2 * cores)]
        batch = [attack.draw(attack.ARENAS[name], pool, rng) for name in chosen]
        rows = evaluate(batch, cores)
        _write(target, [{**row, "arena": name, "stage": "survey", "shard": shard} for row, name in zip(rows, chosen, strict=True)])
        done += len(rows)
    return done


def climb(shard: int, minutes: float, out: Path) -> int:
    """This shard's worst surveyed worlds in one arena, changed towards worse until the time is up."""
    arenas = list(attack.ARENAS)
    arena = attack.ARENAS[arenas[shard % len(arenas)]]
    by_text = {str(entry["law"]): entry for entry in laws()}
    surveyed = [row for row in _read(sorted(out.glob(f"survey-{shard:03d}.jsonl.gz")))
                if row.get("arena") == arena.name and row.get("status") == "ok"]
    held = attack.keep_worst([(badness(row), rt.World.from_dict(row["world"])) for row in surveyed], KEEP)
    if not held:
        return 0
    rng = np.random.default_rng([LAW_SEED, 2, shard])
    cores = _cores()
    target = out / f"climb-{shard:03d}.jsonl.gz"
    deadline, done, generation = time.monotonic() + 60.0 * minutes, 0, 0
    while time.monotonic() < deadline:
        generation += 1
        known = {world for _score, world in held}
        tried: list[rt.World] = []
        for _score, world in held:
            for _ in range(CHILDREN):
                changed = attack.change(world, arena, by_text, rng)
                if changed not in known:
                    known.add(changed)
                    tried.append(changed)
        rows = evaluate(tried, cores)
        _write(target, [{**row, "arena": arena.name, "stage": "climb", "shard": shard, "generation": generation} for row in rows])
        held = attack.keep_worst([*held, *((badness(row), world) for row, world in zip(rows, tried, strict=True))], KEEP)
        done += len(rows)
    return done


def _tally(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    reported = [row for row in ok if row[method]["reported"]]
    wrong = [row for row in reported if row[method]["wrong"]]
    with_law = [row for row in ok if row["world"]["truth_listed"]]
    return {"worlds": len(rows), "errors": sum(row.get("status") == "error" for row in rows),
            "timeouts": sum(row.get("status") == "timeout" for row in rows),
            "reported": len(reported), "wrong": len(wrong),
            "wrong_share_of_reported": len(wrong) / len(reported) if reported else None,
            "reported_the_law": sum(bool(row[method]["symbolic"]) for row in reported),
            "law_offered": len(with_law), "law_offered_and_reported": sum(bool(row[method]["symbolic"]) for row in with_law)}


def _band(noise: float) -> str:
    return "none" if noise == 0 else "up to 0.3 %" if noise <= 0.003 else "0.3 to 1 %" if noise <= 0.01 else \
        "1 to 3 %" if noise <= 0.03 else "3 to 10 %"


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arena and method: worlds, reported laws, wrong ones; the worst worlds found; wrong laws by what they were."""
    out: dict[str, Any] = {"survey": {}, "climb": {}, "survey_by_noise": {}, "worst": {}, "wrong_laws_of_the_route": {}}
    for stage in ("survey", "climb"):
        for arena in attack.ARENAS:
            sel = [row for row in rows if row.get("stage") == stage and row.get("arena") == arena]
            out[stage][arena] = {method: _tally(sel, method) for method in METHODS}
    surveyed = [row for row in rows if row.get("stage") == "survey"]
    for band in ("none", "up to 0.3 %", "0.3 to 1 %", "1 to 3 %", "3 to 10 %"):
        sel = [row for row in surveyed if _band(float(row["world"]["noise"])) == band
               and row["world"]["sampling"] == "uniform" and row["world"]["noise_shape"] == "gaussian"
               and int(row["world"]["rows"]) == 1000]
        out["survey_by_noise"][band] = {method: _tally(sel, method) for method in METHODS}
    for arena in attack.ARENAS:
        sel = sorted((row for row in rows if row.get("arena") == arena and row.get("status") == "ok"), key=lambda r: -badness(r))
        out["worst"][arena] = [{"badness": badness(row), "stage": row.get("stage"), "world": row["world"],
                                "reported": row["route"]["law"], "action": row["route"]["action"],
                                "test_r2": row["route"]["test_r2"], "adequacy": row["route"]["adequacy"],
                                "equivalent": len(row["route"]["equivalent"])} for row in sel[:5] if badness(row) > 0]
        wrong = [row for row in rows if row.get("arena") == arena and row.get("status") == "ok" and row["route"]["wrong"]]
        out["wrong_laws_of_the_route"][arena] = {
            "count": len(wrong),
            "law_was_offered": sum(bool(row["world"]["truth_listed"]) for row in wrong),
            "by_action": {action: sum(row["route"]["action"] == action for row in wrong)
                          for action in sorted({row["route"]["action"] for row in wrong})},
            "by_source": {source: sum(row["route"]["found_by"] == source for row in wrong)
                          for source in sorted({str(row["route"]["found_by"]) for row in wrong})}}
    return out


# ---------------------------------------------------------------- command line
def _shard(text: str) -> tuple[int, int]:
    k, n = text.split("/")
    return int(k), int(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_laws = sub.add_parser("laws")
    p_laws.add_argument("--check", action="store_true")
    p_laws.add_argument("--count", type=int, default=LAW_COUNT)
    sub.add_parser("smoke")
    for name in ("survey", "climb"):
        p = sub.add_parser(name)
        p.add_argument("--shard", type=_shard, required=True)
        p.add_argument("--minutes", type=float, required=True)
        p.add_argument("--out", type=Path, required=True)
    p_collect = sub.add_parser("collect")
    p_collect.add_argument("directory", type=Path)
    p_collect.add_argument("--out", type=Path, required=True)
    p_world = sub.add_parser("world")
    p_world.add_argument("record")
    args = parser.parse_args(argv)

    if args.command == "laws":
        record = build_laws(args.count)
        text = json.dumps(record, indent=1, sort_keys=True) + "\n"
        if args.check:
            if LAWS_FILE.read_text(encoding="utf-8") != text:
                print(f"{LAWS_FILE.relative_to(ROOT)} is not what the generator gives")
                return 1
            print(f"{LAWS_FILE.relative_to(ROOT)}: {len(record['laws'])} laws, as generated")
            return 0
        LAWS_FILE.write_text(text, encoding="utf-8")
        fakes = [fake for law in record["laws"] for fake in law["counterfeits"]]
        print(f"wrote {LAWS_FILE.relative_to(ROOT)}: {len(record['laws'])} laws, {len(fakes)} counterfeits")
        return 0
    if args.command == "smoke":
        pool = laws()
        rng = np.random.default_rng(1)
        batch = [attack.draw(attack.ARENAS[name], pool, rng) for name in attack.ARENAS]
        rows = evaluate(batch, _cores())
        for row, name in zip(rows, attack.ARENAS, strict=True):
            if row["status"] != "ok":
                print(name, row["status"], row.get("error"), row.get("trace", ""))
                return 1
            print(f"{name:12s} route: {row['route']['action']:17s} wrong={row['route']['wrong']!s:5s} "
                  f"best_fit wrong={row['best_fit']['wrong']!s:5s} {row['route']['seconds']} s")
        return 0
    if args.command in ("survey", "climb"):
        args.out.mkdir(parents=True, exist_ok=True)
        done = (survey if args.command == "survey" else climb)(args.shard[0], args.minutes, args.out)
        print(f"{args.command} shard {args.shard[0]}: {done} worlds")
        return 0
    if args.command == "collect":
        rows = _read(sorted(args.directory.rglob("*.jsonl.gz")))
        record = {"what": "A campaign against the default law-discovery route. Development: the worlds were chosen to "
                          "defeat it, so these are not rates on data nobody designed.",
                  "worlds": len(rows), "arenas": {name: arena.what for name, arena in attack.ARENAS.items()},
                  "summary": summarise(rows)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        for stage in ("survey", "climb"):
            for arena, methods in record["summary"][stage].items():
                r = methods["route"]
                print(f"{stage:6s} {arena:12s} worlds {r['worlds']:5d} reported {r['reported']:5d} wrong {r['wrong']:4d} | "
                      f"best fit wrong {methods['best_fit']['wrong']:5d} | validated wrong {methods['validated']['wrong']:5d}")
        return 0
    world = rt.World.from_dict(json.loads(args.record))
    print(json.dumps(decide(world), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
