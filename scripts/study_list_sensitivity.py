#!/usr/bin/env python3
"""Development: does what the route reports on a table depend on the laws that stand beside its own?

Status: audit-only

**Development, not a result.** It was written to find the cause of what the first run of
``scripts/study_family_stream.py`` showed: a table read a second time with laws from a family it does not
belong to gained a reported law almost as often as one read with its own family's.

**What it does.** The 16 families of that run, at 1 % and 5 % noise: 160 tables. No search is run. Each
table gets a list of ten ordinary laws for its number of inputs (``OWN``: a polynomial, an exponential, a
rational law and so on, every constant fitted by the route), and is read three times by the route in force:

* the list as it is;
* six unrelated laws at the head of the list (``UNRELATED``: fast oscillations, which fit none of these
  tables), which is what the family route as first built did with the laws it handed in;
* the same six taking turns with the table's own (``across_tables.beside``).

The route gives a search ten of the twelve candidates its referee contrasts, in the order given, so six
laws at the head leave room for the table's first four, and six in turn for its first five.

**What is counted.** For each way: laws reported, reported and not accurate (R squared 0.999 on rows
without noise), and whether the reported law is one of the unrelated ones. Between the list as it is and
each other way: tables that gained a report and tables that lost one.

    python scripts/study_list_sensitivity.py --out benchmarks/family_stream/dev_list_sensitivity.json --jobs 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import study_family_stream as study  # noqa: E402 - after the path is set

from apedp_dyn.discovery_l2 import family_route  # noqa: E402
from apedp_dyn.discovery_l2.across_tables import beside  # noqa: E402
from apedp_dyn.ode.extended_route import replayed  # noqa: E402

TAG = "DEV-LIST-SENSITIVITY-2026-11"
NOISES = (0.01, 0.05)
FAMILIES = study.FAMILIES                                  #: the first run's sixteen
#: Ten ordinary laws for a table of one input and of two; the route fits every constant.
OWN = {1: ("0.5 + 0.5*x0 + 0.5*x0**2 + 0.5*x0**3", "0.5*exp(-0.5*x0) + 0.5", "0.5/(x0 + 0.5) + 0.5", "0.5*x0**0.5 + 0.5",
           "0.5*exp(-0.5*x0)*x0**0.5", "(0.5 + 0.5*x0)/(1 + 0.5*x0)", "0.5*log(x0 + 0.5) + 0.5", "0.5*sqrt(x0 + 0.5) + 0.5*x0",
           "0.5*x0 + 0.5/x0 + 0.5", "0.5*tanh(0.5*x0) + 0.5*x0"),
       2: ("0.5 + 0.5*x0 + 0.5*x1 + 0.5*x0*x1", "0.5*x0**0.5*x1**0.5", "0.5*exp(-0.5*x0) + 0.5*x1", "0.5*x0/(x1 + 0.5) + 0.5",
           "0.5*log(x0 + 0.5) + 0.5*log(x1 + 0.5)", "0.5*x0**2 + 0.5*x1**2 + 0.5", "(0.5 + 0.5*x0)/(1 + 0.5*x1)",
           "0.5*sqrt(x0 + 0.5)*sqrt(x1 + 0.5)", "0.5*x1/(x0 + 0.5) + 0.5*x0", "0.5*exp(-0.5*x1)*x0 + 0.5")}
#: Six laws that belong to none of these tables.
UNRELATED = {1: ("0.5*sin(2.9*x0)", "0.5*cos(3.7*x0) + 0.5", "0.5*sin(1.3*x0)**2", "0.5*sin(5.1*x0) + 0.5*x0", "0.5*cos(2.3*x0)*x0",
                 "0.5*sin(4.1*x0)/x0"),
             2: ("0.5*sin(2.9*x0)*sin(3.1*x1)", "0.5*cos(3.7*x0) + 0.5*cos(2.9*x1)", "0.5*sin(1.3*x0*x1)", "0.5*sin(5.1*x0) + 0.5*x1",
                 "0.5*cos(2.3*x1)*x0", "0.5*sin(4.1*x1)/x0")}
WAYS = ("as_it_is", "unrelated_at_the_head", "unrelated_in_turn")


def lists(inputs: int) -> dict[str, list[str]]:
    """The three lists a table of ``inputs`` inputs is read with."""
    own, other = list(OWN[inputs]), list(UNRELATED[inputs])
    return {"as_it_is": own, "unrelated_at_the_head": [*other, *own], "unrelated_in_turn": beside(own, other)}


def one(job: tuple[int, float]) -> list[dict[str, Any]]:
    """The tables of one family at one noise level, each read the three ways."""
    index, noise = job
    spec = study.draw_families(FAMILIES)[index]
    family, scoring = study.tables(spec, noise)
    out = []
    for table, how in zip(family.tables, scoring, strict=True):
        row: dict[str, Any] = {"family": index, "member": how["index"], "noise": noise, "inputs": spec["inputs"]}
        for way, laws in lists(spec["inputs"]).items():
            started = time.perf_counter()
            reading = family_route.read(table, replayed(laws), noise="absolute", seed=index)
            row[way] = {"reported": reading["reported"], "action": reading["action"],
                        **study.score(reading["law"], how["truth"], table.names, how["test_X"], how["test_clean"]),
                        "an_unrelated_law": family_route.brought_in(reading, UNRELATED[spec["inputs"]], table.names),
                        "seconds": round(time.perf_counter() - started, 1)}
        out.append(row)
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"tables": len(rows)}
    for way in WAYS:
        reported = [r for r in rows if r[way]["reported"]]
        out[way] = {"reported": len(reported), "reported_and_not_accurate": sum(not r[way]["accurate"] for r in reported),
                    "reported_an_unrelated_law": sum(r[way]["an_unrelated_law"] for r in reported)}
        if way != "as_it_is":
            gained = [r for r in rows if r[way]["reported"] and not r["as_it_is"]["reported"]]
            lost = [r for r in rows if r["as_it_is"]["reported"] and not r[way]["reported"]]
            out[way].update({"gained_a_report": len(gained), "of_those_accurate": sum(r[way]["accurate"] for r in gained),
                             "lost_a_report": len(lost),
                             "verdicts_changed": len(gained) + len(lost),
                             "of_tables_without_a_report_gained": f"{len(gained)} of {sum(not r['as_it_is']['reported'] for r in rows)}",
                             "of_tables_with_a_report_lost": f"{len(lost)} of {sum(bool(r['as_it_is']['reported']) for r in rows)}"})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--families", type=int, default=FAMILIES)
    args = parser.parse_args(argv)
    jobs = [(index, noise) for noise in NOISES for index in range(args.families)]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = [row for part in pool.map(one, jobs) for row in part]
    else:
        rows = [row for job in jobs for row in one(job)]
    record = {"tag": TAG, "development_not_a_result": True,
              "what": "the route in force on tables with a fixed list of ordinary laws, read as it is, with six unrelated laws at "
                      "the head of the list, and with the same six taking turns with the table's own; no search is run",
              "settings": {"families": args.families, "noises": list(NOISES), "seed": study.SEED},
              "summary": {"all": summarise(rows), **{f"noise {noise:g}": summarise([r for r in rows if r["noise"] == noise])
                                                     for noise in NOISES}},
              "tables": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
