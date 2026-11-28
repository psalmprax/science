#!/usr/bin/env python3
"""Development: what trying the next law of a class adds, and what it lets through.

Status: audit-only

**Development, not a result.** ``apedp_dyn/discovery_l2/next_member_route.py`` tries the other laws of a class when
the one the route took fails the residual check. This reads the 122 tables the second family run left without a law
three ways, with no search, by the route in force and then with that step:

* ``last_of_ten``: nine ordinary laws and then the true form (where 13 readings were lost this way);
* ``alone``: the true form alone;
* ``none``: ten ordinary laws and no true form. This is the trap: whatever the step reports here is not the law.

Counted for each way: laws reported by the route, and of those the law itself, accurate (R squared 0.999 on rows
without noise) and not accurate; readings where the step applied; laws it added, counted the same way; and laws it
found but the last checks did not vouch for.

    python scripts/study_next_member.py --out benchmarks/family_stream/dev_next_member.json --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import study_family_stream as study  # noqa: E402 - after the path is set
import study_handed_truth as handed  # noqa: E402
import study_list_sensitivity as lists  # noqa: E402

from apedp_dyn.discovery_l2 import family_route, next_member_route  # noqa: E402
from apedp_dyn.ode.extended_route import replayed  # noqa: E402

TAG = "DEV-NEXT-MEMBER-2026-11"
WAYS = ("last_of_ten", "alone", "none")


def one(job: tuple[int, float, tuple[int, ...]]) -> list[dict[str, Any]]:
    index, noise, wanted = job
    spec = study.draw_families(handed.FAMILIES_DRAWN)[index]
    family, scoring = study.tables(spec, noise)
    rows = []
    for table, how in zip(family.tables, scoring, strict=True):
        if how["index"] not in wanted:
            continue
        own, truth = list(lists.OWN[spec["inputs"]]), str(how["truth"])
        row: dict[str, Any] = {"family": spec["family"], "member": how["index"], "noise": noise, "inputs": spec["inputs"]}
        for way, laws in (("last_of_ten", [*own[:9], truth]), ("alone", [truth]), ("none", own)):
            reading = family_route.read(table, replayed(laws), noise="absolute", seed=spec["family"])
            scored = study.score(reading["law"] if reading["reported"] else None, how["truth"], table.names, how["test_X"],
                                 how["test_clean"])
            cell: dict[str, Any] = {"route_reported": bool(reading["reported"]), "route_action": reading["action"],
                                    "route_accurate": scored["accurate"], "route_the_law_itself": scored["recovered"]}
            if reading["action"] != "not_run":
                after = next_member_route.next_member(reading["record"], table.columns(), table.y, table.pool_X, searched=True,
                                                      noise="absolute", seed=spec["family"])
                state = after["next_member"]
                law = after.get("law") if after.get("reported") else None
                held = after.get("tentative_law") if state["applied"] and not after.get("reported") else None
                cell.update({"applied": bool(state["applied"]), "tried": len(state["tried"]), "class": state.get("class"),
                             "added": bool(state["applied"] and law),
                             **{f"added_{k}": bool(v and state["applied"]) for k, v in
                                study.score(law, how["truth"], table.names, how["test_X"], how["test_clean"]).items()},
                             "found_and_not_vouched": bool(held),
                             "not_vouched_is_the_law": bool(held and study.score(held, how["truth"], table.names, how["test_X"],
                                                                                 how["test_clean"])["recovered"])})
            row[way] = cell
        rows.append(row)
    return rows


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"tables": len(rows)}
    for way in WAYS:
        got = [r[way] for r in rows]
        added = [g for g in got if g.get("added")]
        out[way] = {"route_reported": sum(g["route_reported"] for g in got),
                    "route_reported_the_law_itself": sum(g["route_the_law_itself"] for g in got),
                    "route_reported_accurate": sum(g["route_accurate"] for g in got),
                    "route_reported_and_not_accurate": sum(g["route_reported"] and not g["route_accurate"] for g in got),
                    "step_applied": sum(bool(g.get("applied")) for g in got),
                    "added": len(added), "added_the_law_itself": sum(g["added_recovered"] for g in added),
                    "added_accurate": sum(g["added_accurate"] for g in added),
                    "added_and_not_accurate": sum(not g["added_accurate"] for g in added),
                    "found_and_not_vouched": sum(bool(g.get("found_and_not_vouched")) for g in got),
                    "not_vouched_was_the_law": sum(bool(g.get("not_vouched_is_the_law")) for g in got),
                    "added_by_noise": {str(noise): sum(bool(r[way].get("added")) for r in rows if r["noise"] == noise)
                                       for noise in sorted({r["noise"] for r in rows})}}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    wanted: dict[tuple[int, float], list[int]] = {}
    for (family, member, noise), row in handed.second_run().items():
        if not row["reported"]:
            wanted.setdefault((family, noise), []).append(member)
    jobs = [(family, noise, tuple(sorted(members))) for (family, noise), members in sorted(wanted.items())]
    started, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for got in pool.map(one, jobs):
            rows.extend(got)
    record = {"tag": TAG, "development_not_a_result": True,
              "what": "the 122 tables the second family run left without a law, read with no search by the route in force and "
                      "then with the next law of a class tried; no law is written here",
              "settings": {"seconds": round(time.perf_counter() - started), "max_members": next_member_route.MAX_MEMBERS},
              "summary": summarise(rows), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
