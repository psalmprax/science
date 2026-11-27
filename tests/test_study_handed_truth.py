"""The study that hands the route the law itself (development; ``scripts/study_handed_truth.py``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_handed_truth as study

RECORD = ROOT / "benchmarks" / "family_stream" / "dev_handed_truth.json"
NO, YES = {"recovered": False, "accurate": False}, {"recovered": True, "accurate": True}


def _reading(*, reported: bool = False, action: str = "widen_search", listed: tuple[str, ...] = ("truth", "other"), tentative: str | None = None,
             rejected: str | None = None, verdict: str | None = "equivalent_class", survivor: str | None = "truth",
             equivalent: tuple[str, ...] = ("other",), eliminated: tuple[str, ...] = (), reason: str = "") -> dict[str, Any]:
    contrast = None if verdict is None else SimpleNamespace(
        verdict=verdict, survivor=survivor, equivalent=list(equivalent), rows_read=30,
        eliminated=[{"candidate": name, "by": "truth"} for name in eliminated])
    path = SimpleNamespace(contrast=contrast, rejected_law=rejected)
    record: dict[str, Any] = {"main_search": path, "tentative_law": tentative}
    if tentative:
        record["vouching"] = {"applied": True, "reason": reason}
    return {"record": record, "reported": reported, "action": action, "listed": list(listed), "law": "truth" if reported else None,
            "tentative_law": tentative}


def test_where_a_reading_ended_is_read_from_the_route_s_own_record() -> None:
    scored = [YES, NO]
    told = study.decision(_reading(reported=True, action="confirmed", verdict="unique", equivalent=(), eliminated=("other",)), scored, YES, YES)
    assert told["step"] == "reported" and told["the_law_itself"] and told["contrast"]["true_form_survived"]
    withheld = study.decision(_reading(tentative="truth", reason="an error large enough to make it wrong would have been seen 74% of the time, not 80%"),
                              scored, NO, YES)
    assert withheld["step"] == "last_checks" and withheld["reason"] == "too little power to see an error" and withheld["held_the_law_itself"]
    class_lost = study.decision(_reading(rejected="other", survivor="other", equivalent=("truth",)), scored, NO, NO)
    assert class_lost["step"] == "adequacy" and class_lost["contrast"]["true_form_indistinguishable"] and not class_lost["contrast"]["true_form_survived"]
    refuted = study.decision(_reading(verdict="no_survivor", survivor=None, equivalent=(), eliminated=("truth",)), scored, NO, NO)
    assert refuted["step"] == "refuted_on_rows_read" and refuted["contrast"]["true_form_eliminated"]
    assert study.decision(_reading(action="not_run", verdict=None, listed=()), [], NO, NO)["step"] == "not_run"
    assert study.decision(_reading(listed=("a", "b")), [NO, NO], NO, NO)["true_form_listed"] is False


@pytest.mark.parametrize("reason, kind", [
    ("2 candidates passed every test and differ by more than the accuracy claimed", "several pass and disagree"),
    ("at this noise and with these rows its constants cannot be fitted to the accuracy claimed", "constants cannot be fitted to the accuracy claimed"),
    ("an error large enough to make it wrong would have been seen 6% of the time, not 80%", "too little power to see an error"),
    ("something new", "other"),
])
def test_the_last_checks_reasons_are_sorted_by_kind(reason: str, kind: str) -> None:
    assert study.kind_of_reason(reason) == kind


def test_the_summary_counts_what_its_rows_say() -> None:
    def row(member: int, noise: float, alone: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
        return {"family": 16, "member": member, "noise": noise, "inputs": 1, "alone": alone, "last_of_ten": last}

    reported = study.decision(_reading(reported=True, action="confirmed", verdict="unique", equivalent=(), eliminated=("other",)), [YES, NO], YES, YES)
    withheld = study.decision(_reading(tentative="truth", reason="2 candidates passed every test and differ by more than the accuracy claimed"),
                              [YES, NO], NO, YES)
    class_lost = study.decision(_reading(rejected="other", survivor="other", equivalent=("truth",)), [YES, NO], NO, NO)
    rows = [row(0, 0.01, reported, reported), row(1, 0.05, reported, class_lost), row(2, 0.1, withheld, withheld)]
    before = {(16, 0, 0.01): {"reported": False, "first_action": "widen_search", "had_accurate": True, "had_law": False},
              (16, 1, 0.05): {"reported": False, "first_action": "widen_search", "had_accurate": True, "had_law": True},
              (16, 2, 0.1): {"reported": False, "first_action": "equivalent_class", "had_accurate": False, "had_law": False}}
    got = study.summarise(rows, before)
    assert got["tables"] == 3 and got["reported_both_ways"] == 1 and got["reported_neither_way"] == 1
    assert got["alone"]["reported"] == 2 and got["alone"]["ended_at"] == {"last_checks": 1, "reported": 2}
    assert got["alone"]["last_checks_withheld"] == {"count": 1, "reasons": {"several pass and disagree": 1}, "by_noise": {"0.1": 1}}
    assert got["last_of_ten"]["adequacy_with_the_true_form_in_the_class"] == 1
    assert got["alone"]["by_the_second_run_s_first_action"] == {"equivalent_class": {"tables": 1, "reported": 0}, "widen_search": {"tables": 2, "reported": 2}}
    assert got["alone"]["where_the_second_run_had"]["an_accurate_candidate"] == {"tables": 2, "reported": 2}
    assert got["alone"]["by_noise"]["0.05"] == {"tables": 1, "reported": 1, "ended_at": {"reported": 1}}


def test_the_second_run_s_records_give_the_122_tables_it_left_without_a_law() -> None:
    before = study.second_run()
    assert len(before) == 240
    without = [row for row in before.values() if not row["reported"]]
    assert len(without) == 122 and sum(row["had_accurate"] for row in without) == 91 and sum(row["had_law"] for row in without) == 30


def test_the_record_s_summary_is_what_its_rows_give_and_holds_no_law() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["development_not_a_result"] is True and record["tag"] == study.TAG
    assert record["summary"] == json.loads(json.dumps(study.summarise(record["rows"], study.second_run())))
    assert record["summary"]["tables"] == 122
    text = RECORD.read_text(encoding="utf-8")
    assert "x0" not in text and "exp(" not in text and "**" not in text
