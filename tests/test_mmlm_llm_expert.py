"""A language model as one of MMLM's experts: what it is shown, what is read from it, and how its hint orders the search."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from apedp_mmlm import equation_language as el
from apedp_mmlm import llm_expert

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> object:
    for folder in (str(ROOT), str(ROOT / "scripts")):
        if folder not in sys.path:
            sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = _script("study_mmlm_llm_expert")
track_s = study.track_s


def test_the_model_is_shown_numbers_and_anonymous_names_only() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (300, 2))
    y = 1.7 * X[:, 0] / X[:, 1] ** 2
    prompt = llm_expert.table_prompt(X, y)
    rows = prompt.split("Rows (x0, x1, y):\n")[1].splitlines()
    assert len(rows) == llm_expert.ROWS_SHOWN and all(len(row.split(", ")) == 3 for row in rows)
    assert float(rows[0].split(", ")[2]) == pytest.approx(y.min(), rel=1e-3)                          # spread over the range
    assert float(rows[-1].split(", ")[2]) == pytest.approx(y.max(), rel=1e-3)
    assert "x0, x1" in prompt and "Do not write fitted constants" in prompt
    assert not any(word in prompt.lower() for word in ("kepler", "pressure", "mass", "seed"))


def test_only_allowed_expressions_are_read_from_a_reply() -> None:
    reply = "\n".join(["1. x0/x1**2", "2) c0*x0/x1^2 + c1", "- sqrt(x0)*x1", "__import__('os').system('true')", "x0 + y", "",
                       "`exp(-x0)*x1`", "Here is why: the data fall with x1"])
    shapes, unread = llm_expert.shapes_in(reply, 2)
    assert [str(s) for s in shapes] == ["x0/x1**2", "x0/x1**2 + 1", "sqrt(x0)*x1", "x1*exp(-x0)"]
    assert unread == 3                                                                                # the unsafe line, the unknown name, the prose
    assert llm_expert.shapes_in("x0\nx0\n1.0*x0\nx0*x1", 2, wanted=2)[0] == [sp.Symbol("x0"), sp.Symbol("x0") * sp.Symbol("x1")]
    assert llm_expert.shapes_in("3.7\nc0", 1) == ([], 0)                                              # a constant is not a shape


def test_a_named_shape_is_found_in_the_search_whatever_its_scale_and_offset() -> None:
    _nodes, index, _limit = track_s.candidates(2)
    plain = llm_expert.place_in_search(sp.sympify("x0/x1**2"), 2, index)
    assert plain is not None and plain == llm_expert.place_in_search(sp.sympify("1.73*x0/x1**2 + 0.2"), 2, index)
    assert llm_expert.place_in_search(sp.sympify("x0/(x0 + 1.3)"), 2, index) is None                  # a constant inside: not enumerated


def test_teaching_the_equation_model_changes_a_copy_and_makes_the_shape_cheaper() -> None:
    model = el.SelfAdjustingEquationModel().fit(track_s.srbench_trees())
    shape = sp.sympify("sin(x0)*sin(x1)/x0")
    before = model.bits(el.to_tree(shape))
    informed = llm_expert.taught(model, [shape])
    assert informed.bits(el.to_tree(shape)) < before and model.bits(el.to_tree(shape)) == before


def test_the_named_shapes_are_tried_first_and_the_rest_by_the_taught_prior() -> None:
    scores = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    assert [llm_expert.rank_by(scores, i) for i in range(5)] == [5, 1, 3, 2, 4]
    assert llm_expert.rank_with_expert(0, [4, 0, 4], scores) == 2                                     # named second
    assert llm_expert.rank_with_expert(2, [4, 0], scores) == 5                                        # after the two named and two cheaper
    assert llm_expert.rank_with_expert(1, [], scores) == 1


def test_an_expert_that_cannot_be_reached_names_nothing_and_token_counts_are_the_services() -> None:
    X = np.random.default_rng(1).uniform(1.0, 5.0, (40, 1))

    def down(prompt: str) -> str:
        raise RuntimeError("HTTP 503")

    shapes, call = llm_expert.LlmExpert(down).shapes(X, X[:, 0] ** 2)
    assert shapes == [] and call["error"] == "RuntimeError: HTTP 503"

    def answers(prompt: str) -> str:
        answers.last = {"usage": {"prompt_tokens": 410, "completion_tokens": 12, "total_tokens": 422}}  # type: ignore[attr-defined]
        return "x0**2\nx0**3"

    shapes, call = llm_expert.LlmExpert(answers).shapes(X, X[:, 0] ** 2)
    assert [str(s) for s in shapes] == ["x0**2", "x0**3"] and call["usage"]["total_tokens"] == 422 and call["lines_not_read"] == 0


def test_the_study_scores_the_four_orders_on_the_same_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = [law for law in study.laws(3) if law["family"] == "invented"]
    assert len(chosen) == 3 and all(track_s.position(law) is not None for law in chosen)
    monkeypatch.setattr(study, "laws", lambda count: chosen)
    truth = {str(law["expr"]): law for law in chosen}

    def scripted(prompt: str) -> str:                                                                 # right on the first law only
        scripted.calls += 1                                                                           # type: ignore[attr-defined]
        return "x0*x1*x1\n" + (str(chosen[0]["expr"]) if scripted.calls == 1 else "sin(x0)")          # type: ignore[attr-defined]

    scripted.calls = 0                                                                                # type: ignore[attr-defined]
    scripted.model = "scripted"                                                                       # type: ignore[attr-defined]
    record = study.run(llm_expert.LlmExpert(scripted), 3)
    rows = record["rows"]
    assert record["model"] == "scripted" and [r["law"] for r in rows] == list(truth)
    assert rows[0]["rank"]["llm_alone"] in (1, 2) and rows[0]["rank"]["mmlm_with_llm"] == rows[0]["rank"]["llm_alone"]
    assert rows[1]["rank"]["llm_alone"] is None or str(chosen[1]["expr"]) in ("x0*x1**2",)
    assert all(r["rank"]["mmlm_with_llm"] >= 1 and r["rank"]["mmlm"] >= 1 and r["rank"]["size"] >= 1 for r in rows)
    summary = record["summary"]
    assert summary["all"]["laws"] == 3 and summary["calls"] == 3 and summary["calls_failed"] == 0
    assert summary["all"]["mmlm_with_llm"]["within_10"] >= 1 and summary["all"]["law_among_the_named_shapes"] >= 1
    assert set(summary["all"]["with_llm_against_mmlm"]) == {"earlier", "later", "same", "sign_test_p_one_sided"}
