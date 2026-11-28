"""The study of the next law of a class (development; ``scripts/study_next_member.py``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_next_member as study

from apedp_dyn.discovery_l2 import report_share

RECORD = ROOT / "benchmarks" / "family_stream" / "dev_next_member.json"


def _cell(*, reported: bool = False, the_law: bool = False, accurate: bool = False, applied: bool = False, added: bool = False,
          added_law: bool = False, added_accurate: bool = False, held: bool = False, held_law: bool = False) -> dict[str, Any]:
    return {"route_reported": reported, "route_the_law_itself": the_law, "route_accurate": accurate, "route_action": "x",
            "applied": applied, "added": added, "added_recovered": added_law, "added_accurate": added_accurate,
            "found_and_not_vouched": held, "not_vouched_is_the_law": held_law}


def test_the_summary_counts_what_its_rows_say() -> None:
    quiet = _cell()
    rows = [
        {"noise": 0.01, "last_of_ten": _cell(reported=True, the_law=True, accurate=True), "alone": quiet, "none": quiet},
        {"noise": 0.01, "last_of_ten": _cell(applied=True, added=True, added_law=True, added_accurate=True), "alone": quiet,
         "none": _cell(applied=True, added=True, added_accurate=True)},
        {"noise": 0.1, "last_of_ten": _cell(applied=True, held=True, held_law=True), "alone": quiet,
         "none": _cell(applied=True, added=True)},                                     # an added law that is not accurate
        {"noise": 0.1, "last_of_ten": {"route_reported": False, "route_the_law_itself": False, "route_accurate": False,
                                       "route_action": "not_run"}, "alone": quiet, "none": _cell(reported=True, accurate=True)},
    ]
    told = study.summarise(rows)
    assert told["tables"] == 4
    assert told["last_of_ten"] == {
        "route_reported": 1, "route_reported_the_law_itself": 1, "route_reported_accurate": 1, "route_reported_and_not_accurate": 0,
        "step_applied": 2, "added": 1, "added_the_law_itself": 1, "added_accurate": 1, "added_and_not_accurate": 0,
        "found_and_not_vouched": 1, "not_vouched_was_the_law": 1, "added_by_noise": {"0.01": 1, "0.1": 0}}
    assert told["none"]["added"] == 2 and told["none"]["added_accurate"] == 1 and told["none"]["added_and_not_accurate"] == 1
    assert told["none"]["route_reported"] == 1 and told["none"]["route_reported_the_law_itself"] == 0
    assert told["alone"]["step_applied"] == 0 and told["alone"]["added"] == 0


def test_the_record_says_what_its_rows_say_and_is_marked_development() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["tag"] == study.TAG and record["development_not_a_result"] is True and len(record["rows"]) == 122
    assert record["summary"] == study.summarise(record["rows"])
    assert "truth" not in RECORD.read_text(encoding="utf-8") and "x0" not in RECORD.read_text(encoding="utf-8")   # no law is written


def test_what_the_record_is_cited_for() -> None:
    told = json.loads(RECORD.read_text(encoding="utf-8"))["summary"]
    applied = {way: told[way]["step_applied"] for way in study.WAYS}
    added = {way: told[way]["added"] for way in study.WAYS}
    assert applied == {"last_of_ten": 13, "alone": 2, "none": 24} and added == {"last_of_ten": 7, "alone": 2, "none": 8}
    assert [told[way]["added_the_law_itself"] for way in study.WAYS] == [4, 2, 0]
    assert [told[way]["found_and_not_vouched"] for way in study.WAYS] == [6, 0, 10]
    assert all(told[way]["added_and_not_accurate"] == 0 and told[way]["route_reported_and_not_accurate"] == 0 for way in study.WAYS)
    # on lists with no true form the route itself reports accurate laws that are not the law, and the step adds more of them
    assert told["none"]["route_reported"] == 10 and told["none"]["route_reported_the_law_itself"] == 1 and told["none"]["added_accurate"] == 8
    # seventeen added with none wrong does not show a share within the limit: the project's own check says so
    shown = report_share.verdict(0, sum(added.values()))
    assert sum(added.values()) == 17 and shown["verdict"] == "not shown" and shown["upper_bound"] > shown["upper_bound_allowed"]
    assert report_share.reports_needed(0) == 29


def test_the_step_never_acted_on_a_reading_the_route_had_reported() -> None:
    for row in json.loads(RECORD.read_text(encoding="utf-8"))["rows"]:
        for way in study.WAYS:
            cell = row[way]
            assert not (cell.get("applied") and cell["route_reported"])
            assert not cell.get("added") or cell["applied"]
