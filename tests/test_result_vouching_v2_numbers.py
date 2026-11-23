"""Every count in RESULT-VOUCHING-V2-2026-11's tables, recomputed from the raw rows and found in the record.

Status: audit-only

The record was written by the agent that built and ran the test. This is not an
independent review (AGENTS.md 1.6); it is the mechanical part of one: no number in
the tables may differ from what the committed rows give.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "preregistration" / "RESULT-VOUCHING-V2-2026-11"
RECORD = (ROOT / "preregistration" / "RESULT-VOUCHING-V2-2026-11.md").read_text(encoding="utf-8")
PROSE = " ".join(RECORD.split())               # the record's words, line breaks and indents taken out


def _rows(name: str) -> list[dict]:
    with gzip.open(RUN / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cell(text: str) -> str:
    return text.replace("-", "−")


def test_the_handed_table_and_the_third_expectation() -> None:
    rows = [r for r in _rows("handed.jsonl.gz") if r.get("status", "ok") == "ok"]
    for arm in ("route", "first", "second", "best_fit", "validated"):
        reported = [r for r in rows if r[arm]["reported"]]
        right = sum(bool(r[arm]["symbolic"]) for r in reported)
        wrong = sum(bool(r[arm]["wrong"]) for r in rows)
        at5 = [r for r in reported if r["world"]["noise"] == 0.05]
        score = (right - 19 * wrong) / 180
        line = next(line for line in RECORD.splitlines() if (line.startswith("|") and f"`{arm}`" in line)
                    or (arm == "best_fit" and line.startswith("| best fit |"))
                    or (arm == "validated" and line.startswith("| best fit confirmed")))
        cells = [c.strip().strip("*") for c in line.split("|")[2:-1]]
        assert cells[:3] == [str(len(reported)), str(right), str(wrong)], (arm, cells)
        assert cells[3] == f"{sum(bool(r[arm]['wrong']) for r in at5)} of {len(at5)}", (arm, cells)
        assert cells[4] == _cell(f"{score:.2f}"), (arm, cells, score)
    for arm, text in (("second", "109 of 110 at noise up to 1 % (limit 95 %) and in 29 of 30"), ("first", "109 of 110 and 27 of 30")):
        low = [r for r in rows if r["route"]["reported"] and r["route"]["symbolic"] and r["world"]["noise"] <= 0.01]
        high = [r for r in rows if r["route"]["reported"] and r["route"]["symbolic"] and r["world"]["noise"] == 0.05]
        kept_low = sum(bool(r[arm]["reported"] and r[arm]["symbolic"]) for r in low)
        kept_high = sum(bool(r[arm]["reported"] and r[arm]["symbolic"]) for r in high)
        assert f"{kept_low} of {len(low)}" in text and f"{kept_high} of {len(high)}" in text and text in PROSE


def test_the_searched_table_and_the_fourth_expectation() -> None:
    rows = [r for r in _rows("searched.jsonl.gz") if r["status"] == "ok"]
    for arm in ("route", "first", "second"):
        reported = [r for r in rows if r[arm]["reported"]]
        recovered = sum(bool(r[arm]["symbolic"]) for r in reported)
        inaccurate = sum(not r[arm]["accuracy_solution"] for r in reported)
        line = next(line for line in RECORD.splitlines() if line.startswith("|") and f"`{arm}`" in line
                    and "at noise 0" not in line and line.count("|") == 8)
        cells = [c.strip().strip("*") for c in line.split("|")[2:-1]]
        assert cells[:3] == [str(len(reported)), str(recovered), str(inaccurate)], (arm, cells)
        for column, noise in ((3, 0.0), (4, 0.01), (5, 0.05)):
            sel = [r for r in reported if r["noise"] == noise]
            assert cells[column].startswith(f"{len(sel)}") and f"{sum(bool(r[arm]['symbolic']) for r in sel)}" in cells[column]
    recovered_by_route = [r for r in rows if r["route"]["reported"] and r["route"]["symbolic"]]
    low = [r for r in recovered_by_route if r["noise"] <= 0.01]
    high = [r for r in recovered_by_route if r["noise"] == 0.05]
    kept = lambda sel: sum(bool(r["second"]["reported"] and r["second"]["symbolic"]) for r in sel)
    assert f"recovered it in {kept(low)} of {len(low)} at noise up to 1 %" in PROSE
    assert f"**{kept(high)} of {len(high)}**" in PROSE
