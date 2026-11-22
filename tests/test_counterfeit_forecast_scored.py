"""The forecast filed with PREREG-COUNTERFEIT-2026-11, scored against the run: the file is what the records give."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_scored_file_is_what_the_forecast_and_the_result_give() -> None:
    scorer = _load("score_counterfeit_forecast")
    assert json.loads(scorer.OUT.read_text(encoding="utf-8")) == scorer.score()
    assert scorer.main(["--check"]) == 0


def test_what_the_score_says() -> None:
    scorer = _load("score_counterfeit_forecast")
    scored = scorer.score()
    assert scored["totals_inside_their_intervals"] == [9, 12]
    outside = {(method, name) for method, totals in scored["totals"].items() for name, entry in totals.items() if not entry["inside"]}
    assert outside == {("route", "right"), ("route", "wrong"), ("route", "score")}          # the route did better than forecast
    assert scored["totals"]["route"]["wrong"]["seen"] == 16 < scored["totals"]["route"]["wrong"]["interval"][0]
    assert all(entry["held"] for entry in scored["expectations"].values())
    assert scored["expectations"]["3"]["chance"] < 0.9 < scored["expectations"]["1"]["chance"]
    assert scored["brier"] < scored["brier_of_even_odds"]
    # every cell is sixty cases, and each method's outcomes in it add up to them
    for cell in scored["cells"].values():
        assert cell["cases"] == 60
        for method in scorer.METHODS:
            assert sum(cell[method]["seen"]) == 60 and abs(sum(cell[method]["forecast_share"]) - 1.0) < 0.01


def test_a_case_that_did_not_finish_counts_as_nothing_reported() -> None:
    scorer = _load("score_counterfeit_forecast")
    silent = {"reported": False, "symbolic": False, "wrong": False}
    right = {"reported": True, "symbolic": True, "wrong": False}
    wrong = {"reported": True, "symbolic": False, "wrong": True}
    close = {"reported": True, "symbolic": False, "wrong": False}
    assert scorer.outcome({"status": "timeout"}, "route") == "nothing"
    row = {"status": "ok", "route": right, "vouched": silent, "best_fit": wrong, "validated": close}
    assert [scorer.outcome(row, method) for method in scorer.METHODS] == ["the law", "nothing", "wrong", "accurate, not the law"]
    # the checked route's forecast share: the route's reports, passed on as the replay passed reports of that kind
    cell = {"route": [50, 30, 20, 0], "after": {"the law": [8, 0, 0, 2], "accurate, not the law": [1, 5, 0, 4], "wrong": [0, 0, 1, 9]}}
    assert scorer.expected_shares(cell, "vouched") == [0.43, 0.15, 0.02, 0.4]
    assert scorer.expected_shares({"route": [1, 0, 0, 1], "after": {name: [0, 0, 0, 0] for name in scorer.OUTCOMES[:-1]}}, "vouched") == [0.5, 0.0, 0.0, 0.5]
