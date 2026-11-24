#!/usr/bin/env python3
"""Development: the three levels of ``apedp_dyn.recursion`` on a self-made curriculum of invented laws.

**Development, not a result.** Nothing here is preregistered, and nothing it finds is put in force.

In one process:

    python scripts/run_recursion.py level1 --out DIR --generations 4 --workers 4
    python scripts/run_recursion.py level2 --out DIR --rounds 2 --metas 3 --generations 3 --workers 4
    python scripts/run_recursion.py level2 --out DIR2 --rounds 1 --first-round 2 --meta-from DIR/level2.json

``level1`` measures the starting searcher (generation 0), then runs generations of study, settings
variation and measurement, and reports ``g_k`` (fresh laws solved per CPU-hour) with the verdict of
``progress.verdict``. ``level2`` compares ways of running Level 1; a later round can be run by itself from
an earlier round's record. Each generation (``level1``) and each Level 1 run (``level2``) is appended to
``<command>_partial.jsonl`` as it ends, so a run stopped by a time limit leaves what it finished.

**With the route's searches** (``--searches SECONDS``: the search stage in force, four searches side by side
where four cores are free) a problem costs minutes, so the levels are cut into steps that separate jobs run
(``apedp_dyn.recursion.chain``; ``.github/workflows/recursion-searches.yml``):

    python scripts/run_recursion.py plan --out S0 --searches 90 --generations 4 --measure 48
    python scripts/run_recursion.py improve --state S0/state_0.json --generation 1 --out S1
    python scripts/run_recursion.py measure --state S1/state_1.json --generation 1 --arm current --shard 0 --shards 2 --out M
    python scripts/run_recursion.py measure --state S1/state_1.json --generation 1 --arm start --shard 0 --shards 2 --out M
    python scripts/run_recursion.py chain-collect --inputs ALL --out OUT
    python scripts/run_recursion.py level2-one --round 1 --index 0 --searches 90 --out R
    python scripts/run_recursion.py level2-choose --round 1 --inputs R --out OUT

The searches need PySR, which installs Julia on first use; with ``--searches`` the script stops at once if PySR
cannot be imported, and runs one problem at a time because the four searches take the cores.

The generator's code is varied only when a coding model is given in the environment (``CODING_MODEL``
and ``CODING_MODEL_API_KEY``, as for ``scripts/run_search_evolution.py``); otherwise that step records
that it did not run. No law of ``benchmarks/invented_laws.json`` is read: the curriculum leaves the
benchmark's structures out by fingerprint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from apedp_dyn.recursion import chain, levels  # noqa: E402

SEED = 20261124
COMMANDS = ("level1", "level2", "collect", "plan", "improve", "measure", "chain-collect", "level2-one", "level2-choose")


def client_from_environment() -> tuple[Callable[[str], str] | None, str | None]:
    if not os.environ.get("CODING_MODEL", "").strip() or not os.environ.get("CODING_MODEL_API_KEY", "").strip():
        return None, None
    import run_search_evolution

    return run_search_evolution.coding_model()


def brief(run: dict[str, Any]) -> dict[str, Any]:
    """A Level 1 run without its per-problem rows."""
    out = {k: v for k, v in run.items() if k != "generations"}
    out["generations"] = [{k: v for k, v in g.items() if k != "rows"} for g in run.get("generations", [])]
    return out


def searches_ready(seconds: float) -> dict[str, Any]:
    """What the search stage will be on this machine; stops the run if it was asked for and cannot be built."""
    if seconds <= 0:
        return {"search_seconds": 0.0, "stage": "none"}
    try:
        import pysr  # noqa: F401 - the search stage cannot run without it; better to stop here than count every problem unsolved
    except Exception as exc:  # noqa: BLE001 - whatever stops PySR loading (no package, no Julia, no network) ends the run with its reason
        raise SystemExit(f"--searches {seconds:g} was asked for and PySR cannot be imported ({type(exc).__name__}: {exc}). "
                         "Nothing was run.") from exc
    from apedp_dyn.discovery_l2 import search_stage

    cores = search_stage.available_cores()
    return {"search_seconds": float(seconds), "cores": cores,
            "stage": "four searches side by side" if cores >= search_stage.CORES_FOR_PORTFOLIO else "one PySR search"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, record: Any) -> None:
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--metas", type=int, default=3)
    parser.add_argument("--measure", type=int, default=24)
    parser.add_argument("--study", type=int, default=12)
    parser.add_argument("--selection", type=int, default=10)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--searches", type=float, default=0.0,
                        help="seconds for the route's search stage in Level 0 (0: the first pass, memory and library only)")
    parser.add_argument("--first-round", type=int, default=1, help="level2: the number of the first round run here")
    parser.add_argument("--meta-from", type=Path, help="level2: a level2.json of an earlier round; its chosen way starts this one")
    parser.add_argument("--state", type=Path, help="improve, measure: the state file to start from")
    parser.add_argument("--generation", type=int, help="improve, measure: the generation")
    parser.add_argument("--arm", choices=chain.ARMS, default="current", help="measure: the searcher as it stands, or the starting one")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--round", type=int, default=1, help="level2-one, level2-choose: the round")
    parser.add_argument("--index", type=int, default=0, help="level2-one: which of the round's ways (0 is the incumbent)")
    parser.add_argument("--inputs", type=Path, help="chain-collect, level2-choose: the directory holding the pieces")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "collect":
        record: dict[str, Any] = {"what": "development runs of apedp_dyn.recursion; not a result"}
        for name in ("level1", "level2"):
            path = args.out / f"{name}.json"
            if path.exists():
                record[name] = read(path)
        print(json.dumps(record, sort_keys=True))
        return 0

    if args.command == "chain-collect":
        if args.inputs is None:
            raise SystemExit("--inputs is needed")
        states = sorted(args.inputs.glob("state_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        if not states:
            raise SystemExit(f"no state file in {args.inputs}")
        record = {"what": "development; not a result",
                  **chain.collect(read(states[-1]), [read(p) for p in sorted(args.inputs.glob("improve_*.json"))],
                                  [read(p) for p in sorted(args.inputs.glob("measure_*.json"))])}
        write(args.out / "chain.json", record)
        print(json.dumps({k: record[k] for k in ("measurements", "measurements_start", "paired", "progress", "incomplete")},
                         indent=1)[:8000])
        return 0

    if args.command == "level2-choose":
        if args.inputs is None:
            raise SystemExit("--inputs is needed")
        incumbent = levels.Meta(**read(args.meta_from)["meta"]) if args.meta_from else levels.Meta(
            variants=args.variants, study=args.study, selection=args.selection)
        tried = levels.metas_for_round(incumbent, args.metas, args.seed, args.round)
        found = {int(read(p)["index"]): read(p) for p in sorted(args.inputs.glob(f"level2_r{args.round}_m*.json"))}
        present = sorted(found)
        if not present:
            raise SystemExit(f"no run of round {args.round} in {args.inputs}")
        entry, chosen = levels.round_record(args.round, [tried[i] for i in present], [found[i] for i in present])
        entry["missing"] = [i for i in range(len(tried)) if i not in found]
        entry["changed"] = chosen != tried[0]
        record = {"what": "development; not a result", "rounds": [entry], "meta": chosen.as_dict(),
                  "note": "the chosen way won on the runs that chose it; a round in which the incumbent wins again on new "
                          "problems is the only confirmation here"}
        write(args.out / "level2.json", record)
        print(json.dumps(record, indent=1)[:8000])
        return 0

    stage = searches_ready(args.searches if args.command not in ("improve", "measure") else
                           float(read(args.state)["start"]["settings"].get("search_seconds", 0.0)) if args.state else 0.0)
    workers = 1 if stage["search_seconds"] > 0 else args.workers
    client, model = client_from_environment()
    meta = levels.Meta(variants=args.variants, study=args.study, selection=args.selection)
    if args.meta_from is not None:
        meta = levels.Meta(**read(args.meta_from)["meta"])
    start = levels.Searcher({"search_seconds": float(args.searches)})
    header = {"what": "development; not a result", "coding_model": model, "search_stage": stage}

    if args.command == "plan":
        state = chain.plan(start, generations=args.generations, measure=args.measure, meta=meta, seed=args.seed)
        write(args.out / "state_0.json", state)
        print(json.dumps({**header, "generations": args.generations, "measure": args.measure, "meta": meta.as_dict(),
                          "start": start.settings, "sealed_files": len(state["sealed"])}, indent=1))
        return 0

    if args.command in ("improve", "measure"):
        if args.state is None or args.generation is None:
            raise SystemExit("--state and --generation are needed")
        state = read(args.state)
        if args.command == "improve":
            after, record = chain.improve_step(state, args.generation, workers=workers, client=client)
            write(args.out / f"state_{args.generation}.json", after)
            write(args.out / f"improve_{args.generation}.json", {**header, **record})
            print(json.dumps({k: v for k, v in record.items() if k != "rows"}, indent=1)[:6000])
        else:
            piece = chain.measure_step(state, args.generation, arm=args.arm, shard=args.shard, shards=args.shards,
                                       workers=workers)
            write(args.out / f"measure_{args.generation}_{args.arm}_{args.shard}.json", {**header, **piece})
            print(json.dumps({k: v for k, v in piece.items() if k != "rows"}, indent=1))
        return 0

    partial = args.out / f"{args.command}_partial.jsonl"
    partial.write_text("", encoding="utf-8")

    def note(item: dict[str, Any]) -> None:
        """What has finished so far, written as it finishes: a run stopped by a time limit leaves this."""
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(json.dumps({k: item[k] for k in ("generation", "round", "index", "measurement", "gain_per_cpu_hour") if k in item}),
              flush=True)

    if args.command == "level2-one":
        tried = levels.metas_for_round(meta, args.metas, args.seed, args.round)
        if not 0 <= args.index < len(tried):
            raise SystemExit(f"round {args.round} has {len(tried)} ways; --index {args.index} is not one of them")
        run = levels.run_level1(start, generations=args.generations, meta=tried[args.index], seed=args.seed,
                                measure=args.measure, workers=workers, client=client,
                                tag=levels.level2_tag(args.round, args.index), on_generation=note)
        write(args.out / f"level2_r{args.round}_m{args.index}.json", {**header, "round": args.round, "index": args.index,
                                                                      **brief(run)})
        print(json.dumps({k: run[k] for k in ("meta", "measurements", "progress", "gain_per_cpu_hour")}, indent=1)[:6000])
        return 0

    if args.command == "level1":
        run = levels.run_level1(start, generations=args.generations, meta=meta, seed=args.seed, measure=args.measure,
                                workers=workers, client=client, tag="level1", on_generation=note)
        (args.out / "level1_rows.json").write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
        record = {**header, **brief(run)}
    else:
        record = {**header,
                  **levels.run_level2(start, rounds=args.rounds, metas=args.metas, generations=args.generations, seed=args.seed,
                                      meta=meta, measure=args.measure, workers=workers, client=client,
                                      first_round=args.first_round, on_run=note)}
    write(args.out / f"{args.command}.json", record)
    print(json.dumps({k: record[k] for k in ("measurements", "progress", "gain_per_cpu_hour", "meta", "rounds") if k in record},
                     indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
