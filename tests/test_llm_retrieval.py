"""Pins for the study of a language model shown solved cases: the prompt, the fitted lists, the ceiling on calls."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from apedp_dyn.proposers import experience
from apedp_dyn.recursion.curriculum import Problem

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = [Problem("llm-retrieval|0", "1.3*x0/(x0 + 0.7)", 1, 1, 0.0), Problem("llm-retrieval|1", "2.9*x0*x1", 2, 2, 0.0)]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_llm_retrieval", ROOT / "scripts" / "study_llm_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _store() -> experience.Experience:
    store = experience.Experience()
    for key, law, inputs in (("case-a", "0.7*x0/(x0 + 2.3)", 1), ("case-b", "1.7*exp(-0.45*x0)", 1), ("case-c", "0.3*x0/x1", 2)):
        rows = Problem(key, law, inputs, 1, 0.0).rows()
        store.add(key, law, rows["X"], rows["y"])
    return store


def test_the_prompt_shows_numbers_and_solved_cases_without_their_constants(study) -> None:
    rows = PROBLEMS[0].rows()
    alone = study.prompt(rows["X"], rows["y"])
    assert "x0" in alone and "c0, c1" in alone and "Laws that held" not in alone
    assert len([line for line in alone.splitlines() if line.count(",") == 1 and line[0].isdigit()]) == study.ROWS_SHOWN
    shown = study.prompt(rows["X"], rows["y"], _store().nearest(rows["X"], rows["y"], k=2))
    assert "Laws that held on earlier tables" in shown and "c0*x0/(c1 + x0)" in shown and "0.7" not in shown.split("Rows")[0]


def test_skeletons_are_fitted_in_the_order_given_each_structure_once(study) -> None:
    rows = PROBLEMS[0].rows()
    laws, unread = study.fitted(["c0*x0", "c0*x0/(x0 + c1)", "c3*x0/(c4 + x0)", "import os", "c0"], rows["X"], rows["y"])
    assert len(laws) == 2 and unread == 2                         # the third is the second again; two cannot be read as laws
    assert laws[1].replace(" ", "") in ("1.3*x0/(x0+0.7)", "1.3*x0/(0.7+x0)")


def test_a_model_that_names_the_law_puts_it_first_and_cases_alone_need_no_model(study) -> None:
    calls = []

    def client(prompt: str) -> str:
        calls.append(prompt)
        return "c0*x0\nc0*x0/(x0 + c1)"

    row = study.one(PROBLEMS[0], _store(), client)
    assert len(calls) == 2 and "Laws that held" not in calls[0] and "Laws that held" in calls[1]
    assert row["arms"]["model alone"]["solved_place"] == 2 and row["arms"]["model with cases"]["form_place"] == 2
    assert row["arms"]["cases alone"]["solved_place"] == 1      # the nearest case has the law's own shape here
    kept = row["arms"]["model alone"]["candidates"]              # the fitted laws are kept, for joining lists afterwards
    assert len(kept) == 2 and kept[1]["solved"] and not kept[0]["solved"] and "x0" in kept[1]["text"]
    without = study.one(PROBLEMS[0], _store(), None)
    assert set(without["arms"]) == {"cases alone"}


def test_the_run_stops_at_its_ceiling_on_calls_and_keeps_what_it_measured(study, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study, "problems_and_store", lambda count=0, teaching=0: (PROBLEMS, _store()))

    class CallLimitReachedError(RuntimeError):
        pass

    made = []

    def client(prompt: str) -> str:
        if len(made) == 2:
            raise CallLimitReachedError("the ceiling of 2 calls to the model was reached")
        made.append(prompt)
        return "c0*x0/(x0 + c1)"

    assert study.main(["--out", str(tmp_path), "--max-calls", "2"], client=client, model="scripted") == 0
    record = json.loads((tmp_path / "llm_retrieval.json").read_text())
    assert record["problems"] == 1 and "ceiling of 2 calls" in record["stopped"] and record["model"] == "scripted"
    assert record["arms"]["model alone"]["solved_place"]["first"] == 1
    assert record["against"]["model with cases against model alone"]["solved_place"]["problems"] == 1
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 1
