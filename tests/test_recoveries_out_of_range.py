"""The recorded recoveries, judged again on a range twice as wide (audit M-23)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check():
    name = "check_recoveries_out_of_range"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _entry(formula: str) -> dict:
    return {"name": "made_up", "features": ["a", "b"], "ranges": {"a": [1.0, 5.0], "b": [1.0, 5.0]}, "formula": formula}


def test_the_wide_range_is_twice_as_wide_and_stays_on_its_side_of_zero(check):
    assert check.wide_range(1.0, 5.0) == (0.5, 7.0)          # -1 would cross zero: half the lower bound instead
    assert check.wide_range(2.0, 4.0) == (1.0, 5.0)
    assert check.wide_range(-5.0, -1.0) == (-7.0, -0.5)
    assert check.wide_range(-1.0, 1.0) == (-2.0, 2.0)


def test_the_law_itself_holds_on_the_wide_range(check):
    verdict = check.judge("1.0000001*x0*x1 + 0.0", _entry("a*b"))
    assert verdict["testable"] and verdict["holds"] and verdict["rel_rmse"] < 1e-12


def test_a_form_that_is_the_law_only_on_the_sampled_range_does_not_hold(check):
    # max(b, 1/b) is b wherever b was sampled (1 to 5), so the benchmark's criterion passes it there exactly;
    # below 1 it is 1/b, and it has no constant a refit could move
    entry = _entry("a*b")
    imitation = "1.0*x0*Max(x1, 1/x1)"
    base = sys.modules["run_srbench_baseline"]
    rng = np.random.default_rng(0)
    sampled = rng.uniform(1.0, 5.0, (500, 2))
    inside = base.symbolic_recovery(base.parse_formula(imitation), base.truth_expr(entry), ["x0", "x1"], sampled)
    assert inside["recovered"] and inside["rel_rmse"] < check.BY_FORM
    verdict = check.judge(imitation, entry)
    assert verdict["testable"] and not verdict["holds"] and verdict["rel_rmse"] > 1e-4


def test_a_pole_inside_the_wide_range_does_not_fail_the_true_law(check):
    entry = _entry("a/(b - 6)")                       # sampled up to 5; the wide range reaches 7 and crosses the pole
    rows = check.wide_rows(entry, sys.modules["run_srbench_baseline"].truth_expr(entry))
    truth = rows[:, 0] / (rows[:, 1] - 6.0)
    assert check.MIN_ROWS < len(rows) < check.DRAWS and np.abs(truth).max() <= check.CAP * 5.0
    assert check.judge("1.0*x0/(x1 - 6.0)", entry)["holds"]


def test_a_law_that_is_finite_on_too_few_wide_rows_is_called_not_testable(check, monkeypatch):
    monkeypatch.setattr(check, "MIN_ROWS", check.DRAWS + 1)
    assert check.judge("1.0*x0*x1", _entry("a*b")) == {"testable": False, "rows": check.DRAWS}


def test_what_the_committed_file_says_about_the_recorded_runs(check):
    record = json.loads(check.OUT.read_text(encoding="utf-8"))
    sets = record["result_sets"]
    assert set(sets) == {f"RESULT-{name}-2026-11" for name in ("MAIN-SEARCH", "REFINED-SEARCH", "LLM-CONTROLLER",
                                                               "PORTFOLIO-SEARCH")}
    for cell in sets.values():
        assert cell["testable"] == cell["recovered"] and cell["hold"] + len(cell["do_not_hold"]) == cell["recovered"]
    # every recovery of the refined path and of the four searches holds on the wide range
    assert sets["RESULT-REFINED-SEARCH-2026-11"]["hold"] == 174 and not sets["RESULT-REFINED-SEARCH-2026-11"]["do_not_hold"]
    assert sets["RESULT-PORTFOLIO-SEARCH-2026-11"]["hold"] == 189 and not sets["RESULT-PORTFOLIO-SEARCH-2026-11"]["do_not_hold"]
    # the two narrow passes found by reading formulas are the two that fail here
    controller = sets["RESULT-LLM-CONTROLLER-2026-11"]
    assert sorted(x["problem"] for x in controller["do_not_hold"]) == ["feynman_III_4_33", "feynman_I_10_7"]
    assert all(not x["by_form"] for x in controller["do_not_hold"])
    # one more, in a comparison engine: exact on the sampled range, so "by form" does not catch it
    main = sets["RESULT-MAIN-SEARCH-2026-11"]["do_not_hold"]
    assert [(x["problem"], x["engine"], x["by_form"]) for x in main] == [("feynman_I_25_13", "gagp", True)]


def test_the_committed_file_is_what_the_script_gives(check):
    if not (ROOT / "data" / "srbench").is_dir():
        pytest.skip("the Strogatz problems' ranges come from their data, which is not cached here")
    assert check.main(["--check"]) == 0
