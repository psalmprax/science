#!/usr/bin/env python3
"""Development: when the route in force is handed the law itself, does it report it, and if not, which step declines?

Status: audit-only

**Development, not a result.** The second run of ``scripts/study_family_stream.py`` left 122 of its 240 tables without
a law, and on 91 of them an accurate law had been in front of the referee (the law itself on 30). Two readings of that
are possible: the referee declines laws it should report, or the accurate laws it had were approximations and the law
itself was missing. This study tells them apart on those 122 tables. **No search is run.**

Each table is read twice by the route in force (``family_route.read``):

* ``alone``: the stage proposes the true form and nothing else. The route adds its polynomial and its first pass's law,
  and fits every constant itself;
* ``last_of_ten``: nine ordinary laws (``study_list_sensitivity.OWN``) and then the true form, last, so that it has to
  win its place.

**What is written down** for each reading: whether a law was reported and whether it is the law itself; the step at
which a reading without a law ended (``last_checks``: the last checks did not vouch; ``adequacy``: the law taken from
the contrast failed the residual check; ``refuted_on_rows_read``: no survivor; ``not_run``); the contrast's verdict and
whether the true form survived, was eliminated, or sat among the laws that could not be told apart; and the last
checks' own reason. No law is written to the record.

    python scripts/study_handed_truth.py --out benchmarks/family_stream/dev_handed_truth.json --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import study_family_stream as study  # noqa: E402 - after the path is set
import study_list_sensitivity as lists  # noqa: E402

from apedp_dyn.discovery_l2 import family_route  # noqa: E402
from apedp_dyn.ode.extended_route import replayed  # noqa: E402

TAG = "DEV-HANDED-TRUTH-2026-11"
RECORDS = ("dev_run2_as_run.json", "dev_run3_two_units.json")     #: the second run as it was read, and its two re-read units
FAMILIES_DRAWN, FIRST_FAMILY = 32, 16                              #: the second run's sixteen families
WAYS = ("alone", "last_of_ten")
STEPS = {"no_survivor": "refuted_on_rows_read", "unchallenged": "unchallenged",
         "equivalent_class": "equivalent_class_not_reported", "unique": "unique_not_reported"}


def second_run(directory: Path | None = None) -> dict[tuple[int, int, float], dict[str, Any]]:
    """What the second run recorded for each of its tables, by (family, member, noise)."""
    base = ROOT / "benchmarks" / "family_stream" if directory is None else directory
    out: dict[tuple[int, int, float], dict[str, Any]] = {}
    for name in RECORDS:
        for row in json.loads((base / name).read_text(encoding="utf-8"))["members"]:
            had = row["where"]["among_its_referee_s_candidates"]
            out[(int(row["family"]), int(row["member"]), float(row["noise"]))] = {
                "reported": bool(row["A"]["reported"]), "first_action": str(row["first_action"]),
                "had_accurate": bool(had["accurate"]), "had_law": bool(had["law_itself"])}
    return out


def kind_of_reason(reason: str) -> str:
    """The last checks' reason, by kind."""
    if "candidates passed every test and differ" in reason:
        return "several pass and disagree"
    if "constants cannot be fitted" in reason:
        return "constants cannot be fitted to the accuracy claimed"
    if "would have been seen" in reason:
        return "too little power to see an error"
    if "failed a test of its residuals" in reason:
        return "every candidate failed a residual test"
    return "other"


def decision(reading: dict[str, Any], scored: Sequence[dict[str, bool]], final: dict[str, bool], held: dict[str, bool]) -> dict[str, Any]:
    """What one reading decided and where it ended. ``scored`` is the listed candidates against the truth, in order;
    ``final`` the reported law; ``held`` the law reported, held back or rejected."""
    record = reading["record"]
    path = record.get("main_search")
    contrast = getattr(path, "contrast", None)
    vouching = record.get("vouching") or {}
    out: dict[str, Any] = {"reported": bool(reading["reported"]), "action": reading["action"], "listed": len(reading["listed"]),
                           "the_law_itself": bool(final["recovered"]), "accurate": bool(final["accurate"]),
                           "held_the_law_itself": bool(held["recovered"]), "held_accurate": bool(held["accurate"])}
    true_ids = {reading["listed"][i] for i, s in enumerate(scored) if s["recovered"]}
    out["true_form_listed"] = bool(true_ids)
    if reading["action"] == "not_run":
        out["step"] = "not_run"
    elif reading["reported"]:
        out["step"] = "reported"
    elif vouching.get("applied") and record.get("tentative_law"):
        out["step"] = "last_checks"
        out["reason"] = kind_of_reason(str(vouching.get("reason")))
    elif getattr(path, "rejected_law", None):
        out["step"] = "adequacy"
    elif contrast is not None:
        out["step"] = STEPS.get(contrast.verdict, str(contrast.verdict))
    else:
        out["step"] = "no_contrast"
    if contrast is not None:
        out["contrast"] = {"verdict": contrast.verdict, "rows_read": int(contrast.rows_read),
                           "indistinguishable": len(contrast.equivalent), "eliminated": len(contrast.eliminated),
                           "true_form_survived": bool(contrast.survivor in true_ids) if contrast.survivor else False,
                           "true_form_eliminated": any(e["candidate"] in true_ids for e in contrast.eliminated),
                           "true_form_indistinguishable": any(e in true_ids for e in contrast.equivalent)}
    return out


def one(job: tuple[int, float, tuple[int, ...]]) -> list[dict[str, Any]]:
    """The wanted tables of one family at one noise level, each read the two ways."""
    index, noise, wanted = job
    spec = study.draw_families(FAMILIES_DRAWN)[index]
    family, scoring = study.tables(spec, noise)
    rows = []
    for table, how in zip(family.tables, scoring, strict=True):
        if how["index"] not in wanted:
            continue
        def score(law: str | None, how: dict[str, Any] = how, names: list[str] = table.names) -> dict[str, bool]:
            return study.score(law, how["truth"], names, how["test_X"], how["test_clean"])
        own = list(lists.OWN[spec["inputs"]])
        row: dict[str, Any] = {"family": spec["family"], "member": how["index"], "noise": noise, "inputs": spec["inputs"]}
        for way, laws in (("alone", [str(how["truth"])]), ("last_of_ten", [*own[:9], str(how["truth"])])):
            reading = family_route.read(table, replayed(laws), noise="absolute", seed=spec["family"])
            path = reading["record"].get("main_search")
            held = reading["law"] or reading["tentative_law"] or getattr(path, "rejected_law", None)
            row[way] = decision(reading, [score(text) for text in reading["listed"]], score(reading["law"]), score(held))
        rows.append(row)
    return rows


def summarise(rows: Sequence[dict[str, Any]], before: dict[tuple[int, int, float], dict[str, Any]]) -> dict[str, Any]:
    """The counts the study is read by."""
    def key(row: dict[str, Any]) -> tuple[int, int, float]:
        return (int(row["family"]), int(row["member"]), float(row["noise"]))

    out: dict[str, Any] = {"tables": len(rows)}
    for way in WAYS:
        got = [r[way] for r in rows]
        cell: dict[str, Any] = {
            "reported": sum(g["reported"] for g in got), "reported_the_law_itself": sum(g["reported"] and g["the_law_itself"] for g in got),
            "reported_and_not_accurate": sum(g["reported"] and not g["accurate"] for g in got),
            "ended_at": dict(sorted(Counter(g["step"] for g in got).items())),
            "true_form_listed": sum(g["true_form_listed"] for g in got),
            "contrast": {"verdicts": dict(sorted(Counter(g["contrast"]["verdict"] for g in got if "contrast" in g).items())),
                         "true_form_survived": sum(g["contrast"]["true_form_survived"] for g in got if "contrast" in g),
                         "true_form_eliminated": sum(g["contrast"]["true_form_eliminated"] for g in got if "contrast" in g),
                         "true_form_indistinguishable": sum(g["contrast"]["true_form_indistinguishable"] for g in got if "contrast" in g)},
            "last_checks_withheld": {"count": sum(g["step"] == "last_checks" for g in got),
                                     "reasons": dict(sorted(Counter(g["reason"] for g in got if g["step"] == "last_checks").items())),
                                     "by_noise": dict(sorted(Counter(str(r["noise"]) for r in rows if r[way]["step"] == "last_checks").items()))},
            "adequacy_with_the_true_form_in_the_class": sum(
                g["step"] == "adequacy" and (g["contrast"]["true_form_survived"] or g["contrast"]["true_form_indistinguishable"])
                for g in got if "contrast" in g),
            "by_noise": {str(noise): {"tables": sum(r["noise"] == noise for r in rows),
                                      "reported": sum(r[way]["reported"] for r in rows if r["noise"] == noise),
                                      "ended_at": dict(sorted(Counter(r[way]["step"] for r in rows if r["noise"] == noise).items()))}
                         for noise in sorted({r["noise"] for r in rows})},
            "by_the_second_run_s_first_action": {
                action: {"tables": sum(before[key(r)]["first_action"] == action for r in rows),
                         "reported": sum(r[way]["reported"] for r in rows if before[key(r)]["first_action"] == action)}
                for action in sorted({before[key(r)]["first_action"] for r in rows})},
            "where_the_second_run_had": {
                name: {"tables": sum(before[key(r)][field] for r in rows), "reported": sum(r[way]["reported"] for r in rows if before[key(r)][field])}
                for name, field in (("an_accurate_candidate", "had_accurate"), ("the_law_itself", "had_law"))},
        }
        out[way] = cell
    out["reported_both_ways"] = sum(r["alone"]["reported"] and r["last_of_ten"]["reported"] for r in rows)
    out["reported_neither_way"] = sum(not r["alone"]["reported"] and not r["last_of_ten"]["reported"] for r in rows)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    before = second_run()
    wanted: dict[tuple[int, float], list[int]] = {}
    for (family, member, noise), row in before.items():
        if not row["reported"]:
            wanted.setdefault((family, noise), []).append(member)
    jobs = [(family, noise, tuple(sorted(members))) for (family, noise), members in sorted(wanted.items())]
    started, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for got in pool.map(one, jobs):
            rows.extend(got)
    record = {"tag": TAG, "development_not_a_result": True,
              "what": ("the 122 tables the second family run left without a law, read by the route in force with the true form "
                       "handed in (alone, and last of ten); no search is run; no law is written here"),
              "settings": {"families": [FIRST_FAMILY, FAMILIES_DRAWN - 1], "noises": list(study.NOISES), "records_read": list(RECORDS),
                           "seconds": round(time.perf_counter() - started)},
              "summary": summarise(rows, before), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
