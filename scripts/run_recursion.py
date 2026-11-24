#!/usr/bin/env python3
"""Development: the three levels of ``apedp_dyn.recursion`` on a self-made curriculum of invented laws.

**Development, not a result.** Nothing here is preregistered, and nothing it finds is put in force.

    python scripts/run_recursion.py level1 --out DIR --generations 4 --workers 4
    python scripts/run_recursion.py level2 --out DIR --rounds 2 --metas 3 --generations 3 --workers 4
    python scripts/run_recursion.py collect --out DIR

``level1`` measures the starting searcher (generation 0), then runs generations of study, settings
variation and measurement, and reports ``g_k`` (fresh laws solved per CPU-hour) with the verdict of
``progress.verdict``. ``level2`` compares ways of running Level 1.

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

from apedp_dyn.recursion import levels  # noqa: E402

SEED = 20261124


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("level1", "level2", "collect"))
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
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        record = {"what": "development runs of apedp_dyn.recursion; not a result"}
        for name in ("level1", "level2"):
            path = args.out / f"{name}.json"
            if path.exists():
                record[name] = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(record, sort_keys=True))
        return 0
    client, model = client_from_environment()
    meta = levels.Meta(variants=args.variants, study=args.study, selection=args.selection)
    start = levels.Searcher()
    if args.command == "level1":
        run = levels.run_level1(start, generations=args.generations, meta=meta, seed=args.seed, measure=args.measure,
                                workers=args.workers, client=client, tag="level1")
        (args.out / "level1_rows.json").write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
        record = {"what": "development; not a result", "coding_model": model, **brief(run)}
    else:
        record = {"what": "development; not a result", "coding_model": model,
                  **levels.run_level2(start, rounds=args.rounds, metas=args.metas, generations=args.generations, seed=args.seed,
                                      meta=meta, measure=args.measure, workers=args.workers, client=client)}
    (args.out / f"{args.command}.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("measurements", "progress", "gain_per_cpu_hour", "meta", "rounds") if k in record},
                     indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
