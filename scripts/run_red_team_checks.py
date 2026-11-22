#!/usr/bin/env python3
"""A second campaign: a search for data on which the table route, with its last checks, reports a wrong law.

The first campaign (``scripts/run_red_team.py``) attacked the route as it then was. The checks of
``apedp_dyn/discovery_l2/vouching.py`` were built from what it found, held on a preregistered run
(PREREG-COUNTERFEIT-2026-11) and are the route's last step for tables since 2026-11-22. Nothing
had searched for data that defeats *them*. This does, with the first campaign's machinery: the
same arenas, a survey of worlds drawn at random, then a climb from each shard's worst worlds.

    python scripts/run_red_team_checks.py laws [--check]       # benchmarks/red_team_checks_laws.json
    python scripts/run_red_team_checks.py smoke
    python scripts/run_red_team_checks.py survey --shard 0/24 --minutes 30 --out out
    python scripts/run_red_team_checks.py climb  --shard 0/24 --minutes 20 --out out
    python scripts/run_red_team_checks.py collect DIR --out benchmarks/red_team_checks_campaign.json

**What differs from the first campaign.**

* *New laws.* 150 laws drawn here with their own seed, leaving out every structure of the first
  campaign, of the counterfeit benchmark, of the memory's laws and of the invented-laws benchmark
  (by fingerprint; no held-back law is read). The checks' settings were chosen on the first
  campaign's worlds; these are worlds they have not met.
* *The decision under attack* is the route's report after the checks, made as
  ``discovery_l2/vouched_route`` makes it: ``vouching.vouch`` on the route's law, the candidates the
  path could not tell from it, the development rows and the pool's inputs.
* *What the climb pushes up* is a wrong law that the checks pass. So that it has a slope before
  the first such world is found, a wrong law of the route that the checks stopped still scores,
  the more the nearer it came to passing (``nearness``).

**This is development by its nature**, like the first: the worlds are chosen to defeat the
decision, so its rates are not rates on data nobody designed, and a repair found with it has to
be run on laws drawn afterwards. ``run_red_team.py`` and ``run_counterfeit_benchmark.py`` are
frozen by PREREG-COUNTERFEIT's digest and are used here as they are.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_counterfeit_benchmark as bench  # noqa: E402
import run_red_team as campaign  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import vouching  # noqa: E402
from apedp_dyn.redteam import attack  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

LAWS_FILE = ROOT / "benchmarks" / "red_team_checks_laws.json"
FIRST_LAWS_FILE = campaign.LAWS_FILE
LAW_SEED = 20261201
LAW_COUNT = 150
SPARE = 60                      # drawn beyond the count, for those left out as used elsewhere
FIRST_ID = 7000
METHODS = ("route", "vouched", "best_fit", "validated")
RULES = ("structure", "class", "power", "fit", "other")


# ---------------------------------------------------------------- laws
def taken() -> set[str]:
    """Structures used elsewhere: the first campaign's laws, the memory's laws, the counterfeit benchmark's."""
    if campaign.LAWS_FILE != FIRST_LAWS_FILE:
        raise RuntimeError("the first campaign's laws are read here; not while this campaign's are in their place")
    used: set[str] = set(bench._taken())
    used |= {grammar.structure(sp.sympify(law["law"])) for law in json.loads(bench.OUT.read_text(encoding="utf-8"))["laws"]}
    return used


def build_laws(count: int = LAW_COUNT, seed: int = LAW_SEED) -> dict[str, Any]:
    """``count`` laws with their counterfeits, drawn as the first campaign drew its own, none of them used anywhere else."""
    drawn = campaign.build_laws(count + SPARE, seed)
    used = taken()
    kept = [law for law in drawn["laws"] if grammar.structure(sp.sympify(law["law"])) not in used][:count]
    if len(kept) != count:
        raise SystemExit(f"only {len(kept)} laws that are used nowhere else")
    for k, law in enumerate(kept):
        law["id"] = FIRST_ID + k
    return {"what": "Laws for the second red-team campaign, against the route with its last checks: the invented-law "
                    "grammar, each with its counterfeits, none with a structure of the first campaign, of the "
                    "counterfeit benchmark or of the memory's laws. Development material.",
            "seed": seed, "domain": drawn["domain"], "laws": kept}


def laws() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(LAWS_FILE.read_text(encoding="utf-8"))["laws"]
    return out


# ---------------------------------------------------------------- one world
def rule(reason: str) -> str:
    """Which check a report was stopped by, from the reason ``vouching.vouch`` gives; ``passed`` if none."""
    if not reason or reason.startswith("another member"):
        return "passed"
    if "reported nothing" in reason:
        return "silent"
    if "structure" in reason:
        return "structure"
    if "differ by more" in reason:
        return "class"
    if "would have been seen" in reason:
        return "power"
    if "cannot be fitted" in reason:
        return "fit"
    return "other"


def checked(world: rt.World, route: dict[str, Any]) -> dict[str, Any]:
    """The route's report after the checks: ``run_counterfeit_benchmark.vouched_row``, with what a climb needs added."""
    if not route.get("reported") or not route.get("law"):
        return {**bench.silent("the route reported nothing"), "rule": "silent"}
    names = [f"x{i}" for i in range(world.inputs)]
    truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
    rows = rt.rows_of(world)
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)          # the split the route was given
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    done = vouching.vouch(route["law"], list(route.get("equivalent") or []), names, rows.X[dev], rows.y[dev], rows.X[pool])
    first = done.members[0] if done.members else None
    return {**campaign._scored(done.law, truth, names, rows), "reason": done.reason, "switched": done.switched,
            "power": done.power, "members": len(done.members), "refused": sum(m.refused for m in done.members),
            "p_value": first.p_value if first is not None else None, "rule": rule(done.reason)}


_ROUTE_DECIDES = campaign.decide


def decide(world: rt.World) -> dict[str, Any]:
    """The first campaign's decision on a world (the route, the two plain methods), and the route's report after the checks."""
    out: dict[str, Any] = _ROUTE_DECIDES(world)
    try:
        out["vouched"] = checked(world, out["route"])
    except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing; recorded with the error's class
        out["vouched"] = {**bench.silent(f"the checks failed: {type(exc).__name__}: {exc}"[:200]), "rule": "other", "failed": True}
    return out


def nearness(row: dict[str, Any]) -> float:
    """What the climb pushes up. Above 2: a wrong law the checks passed, by how wrong. Below: how near a stopped one came.

    A wrong law of the route stopped only by the power rule scores between 1 and 2, by how near
    its estimated chance came to the minimum; stopped because members of its class disagree, or
    for its constants, 1; stopped for structure, below 1, by how near its p-value came to the
    level. Everything else scores nothing.
    """
    if row.get("status") != "ok":
        return 0.0
    route, after = row["route"], row["vouched"]
    if after["reported"] and after["wrong"]:
        return 2.0 + float(after["badness"])
    if not (route["reported"] and route["wrong"]):
        return 0.0
    kind = after.get("rule")
    power, level = float(vouching.POWER), float(vouching.LEVEL)
    if kind == "power":
        return 1.0 + min(1.0, float(after.get("power") or 0.0) / power)
    if kind in ("class", "fit"):
        return 1.0
    if kind == "structure":
        return min(1.0, float(after.get("p_value") or 0.0) / level)
    return 0.0


@contextlib.contextmanager
def aimed_at_the_checks() -> Iterator[None]:
    """The first campaign's survey and climb, with this campaign's laws, seed, decision and objective in their place."""
    before = (campaign.LAWS_FILE, campaign.LAW_SEED, campaign.decide, campaign.badness)
    campaign.LAWS_FILE, campaign.LAW_SEED, campaign.decide, campaign.badness = LAWS_FILE, LAW_SEED, decide, nearness
    try:
        yield
    finally:
        campaign.LAWS_FILE, campaign.LAW_SEED, campaign.decide, campaign.badness = before


def evaluate(batch: list[rt.World], cores: int) -> list[dict[str, Any]]:
    with aimed_at_the_checks():
        rows: list[dict[str, Any]] = campaign.evaluate(batch, cores)
    return rows


# ---------------------------------------------------------------- what the campaign found
def _outcome(scored: dict[str, Any]) -> str:
    if not scored["reported"]:
        return "nothing"
    if scored["symbolic"]:
        return "the law"
    return "wrong" if scored["wrong"] else "accurate, not the law"


def _checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the checks made of the route's reports on these worlds."""
    ok = [row for row in rows if row.get("status") == "ok"]
    wrong = [row for row in ok if row["route"]["reported"] and row["route"]["wrong"]]
    right = [row for row in ok if row["route"]["reported"] and row["route"]["symbolic"]]
    passed_wrong = [row for row in ok if row["vouched"]["reported"] and row["vouched"]["wrong"]]
    stopped = {name: sum(row["vouched"].get("rule") == name for row in wrong if not row["vouched"]["reported"]) for name in RULES}
    lost = {name: sum(row["vouched"].get("rule") == name for row in right if not row["vouched"]["reported"]) for name in RULES}
    return {"wrong_laws_of_the_route": len(wrong),
            "of_them_passed_by_the_checks": sum(bool(row["vouched"]["reported"] and row["vouched"]["wrong"]) for row in wrong),
            "of_them_turned_into_the_law_or_an_accurate_one": sum(bool(row["vouched"]["reported"] and not row["vouched"]["wrong"])
                                                                   for row in wrong),
            "of_them_stopped_by": {name: count for name, count in stopped.items() if count},
            "wrong_laws_after_the_checks": len(passed_wrong),
            "of_them_not_wrong_before": sum(not row["route"]["wrong"] for row in passed_wrong),
            "right_laws_of_the_route": len(right),
            "of_them_kept": sum(bool(row["vouched"]["reported"] and row["vouched"]["symbolic"]) for row in right),
            "right_laws_lost_to": {name: count for name, count in lost.items() if count},
            "reports_turned_into_the_law": sum(bool(row["vouched"]["symbolic"] and row["vouched"]["reported"]
                                                    and not row["route"]["symbolic"]) for row in ok),
            "checks_failed": sum(bool(row["vouched"].get("failed")) for row in ok)}


def _passed(row: dict[str, Any]) -> dict[str, Any]:
    """One wrong law the checks passed, with what is needed to run its world again."""
    after, route = row["vouched"], row["route"]
    return {"nearness": nearness(row), "arena": row.get("arena"), "stage": row.get("stage"), "world": row["world"],
            "reported": after["law"], "test_r2": after["test_r2"], "power": after.get("power"), "p_value": after.get("p_value"),
            "members": after.get("members"), "switched": after.get("switched"), "found_by": route.get("found_by"),
            "action": route.get("action"), "route_reported": _outcome(route),
            "handed_to_the_route": bool(after["law"] in row["world"]["candidates"])}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per stage and arena: every method's reports and wrong laws, and what the checks did; then the wrong laws they passed."""
    out: dict[str, Any] = {"survey": {}, "climb": {}, "survey_by_noise": {}, "checks": {"survey": {}, "climb": {}}}
    for stage in ("survey", "climb"):
        for arena in attack.ARENAS:
            sel = [row for row in rows if row.get("stage") == stage and row.get("arena") == arena]
            out[stage][arena] = {method: campaign._tally(sel, method) for method in METHODS}
            out["checks"][stage][arena] = _checks(sel)
        out["checks"][stage]["all"] = _checks([row for row in rows if row.get("stage") == stage])
    surveyed = [row for row in rows if row.get("stage") == "survey"]
    for band in ("none", "up to 0.3 %", "0.3 to 1 %", "1 to 3 %", "3 to 10 %"):
        sel = [row for row in surveyed if campaign._band(float(row["world"]["noise"])) == band
               and row["world"]["sampling"] == "uniform" and row["world"]["noise_shape"] == "gaussian"
               and int(row["world"]["rows"]) == 1000]
        out["survey_by_noise"][band] = {**{method: campaign._tally(sel, method) for method in METHODS}, "checks": _checks(sel)}
    ok = [row for row in rows if row.get("status") == "ok"]
    passed = [row for row in ok if row["vouched"]["reported"] and row["vouched"]["wrong"]]
    out["wrong_laws_the_checks_passed"] = {
        "count": len(passed),
        "by_stage_and_arena": {f"{stage}|{arena}": sum(row.get("stage") == stage and row.get("arena") == arena for row in passed)
                               for stage in ("survey", "climb") for arena in attack.ARENAS},
        "by_source": {str(source): sum(str(row["route"].get("found_by")) == str(source) for row in passed)
                      for source in sorted({str(row["route"].get("found_by")) for row in passed})},
        "one_of_the_candidates_handed_over": sum(bool(row["vouched"]["law"] in row["world"]["candidates"]) for row in passed),
        "came_out_of_an_unseparated_class": sum(row["route"].get("action") == "equivalent_class" for row in passed),
        "test_r2_at_least_0.99": sum(float(row["vouched"]["test_r2"] or 0.0) >= 0.99 for row in passed),
        "worst": {arena: [_passed(row) for row in sorted((r for r in passed if r.get("arena") == arena), key=lambda r: -nearness(r))[:5]]
                  for arena in attack.ARENAS}}
    return out


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_laws = sub.add_parser("laws")
    p_laws.add_argument("--check", action="store_true")
    sub.add_parser("smoke")
    for name in ("survey", "climb"):
        p = sub.add_parser(name)
        p.add_argument("--shard", type=campaign._shard, required=True)
        p.add_argument("--minutes", type=float, required=True)
        p.add_argument("--out", type=Path, required=True)
    p_collect = sub.add_parser("collect")
    p_collect.add_argument("directory", type=Path)
    p_collect.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "laws":
        text = json.dumps(build_laws(), indent=1, sort_keys=True) + "\n"
        if args.check:
            if LAWS_FILE.read_text(encoding="utf-8") != text:
                print(f"{LAWS_FILE.relative_to(ROOT)} is not what the generator gives")
                return 1
            print(f"{LAWS_FILE.relative_to(ROOT)}: as generated")
            return 0
        LAWS_FILE.write_text(text, encoding="utf-8")
        record = json.loads(text)
        print(f"wrote {LAWS_FILE.relative_to(ROOT)}: {len(record['laws'])} laws, "
              f"{sum(len(law['counterfeits']) for law in record['laws'])} counterfeits")
        return 0
    if args.command == "smoke":
        rng = np.random.default_rng(1)
        batch = [attack.draw(attack.ARENAS[name], laws(), rng) for name in attack.ARENAS]
        rows = evaluate(batch, campaign._cores())
        for row, name in zip(rows, attack.ARENAS, strict=True):
            if row["status"] != "ok":
                print(name, row["status"], row.get("error"), row.get("trace", ""))
                return 1
            print(f"{name:12s} route: {_outcome(row['route']):22s} after the checks: {_outcome(row['vouched']):22s} "
                  f"({row['vouched'].get('rule')}) nearness {nearness(row):.2f}")
        return 0
    if args.command in ("survey", "climb"):
        args.out.mkdir(parents=True, exist_ok=True)
        with aimed_at_the_checks():
            done = (campaign.survey if args.command == "survey" else campaign.climb)(args.shard[0], args.minutes, args.out)
        print(f"{args.command} shard {args.shard[0]}: {done} worlds")
        return 0
    rows = campaign._read(sorted(args.directory.rglob("*.jsonl.gz")))
    record = {"what": "A second campaign, against the table route with its last checks. Development: the worlds were chosen "
                      "to defeat the decision, so these are not rates on data nobody designed.",
              "worlds": len(rows), "laws": str(LAWS_FILE.relative_to(ROOT)),
              "arenas": {name: arena.what for name, arena in attack.ARENAS.items()}, "summary": summarise(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    for stage in ("survey", "climb"):
        for arena, methods in record["summary"][stage].items():
            before, after = methods["route"], methods["vouched"]
            print(f"{stage:6s} {arena:12s} worlds {before['worlds']:5d} | route: reported {before['reported']:5d} wrong "
                  f"{before['wrong']:4d} | after the checks: reported {after['reported']:5d} wrong {after['wrong']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
