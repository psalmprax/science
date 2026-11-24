"""The recursion package: its tools, its fixed judge, the test of "recursive", and the levels' rules."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.recursion import curriculum, egraph, judge, levels, mdl, progress, smt
from apedp_dyn.recursion.library import Library, candidates, fragments_of

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "apedp_dyn" / "recursion"


def _p(text: str) -> sp.Expr:
    return sp.parse_expr(text, evaluate=False)


# ------------------------------------------------------------------ e-graph
@pytest.mark.parametrize(("left", "right"), [
    ("(x0+x1)**2", "x0**2+2*x0*x1+x1**2"),
    ("x0*x0*x0", "x0**3"),
    ("2*x0+3*x0", "5*x0"),
    ("(x0+1)*(x0-1)", "x0**2-1"),
    ("exp(x0)*exp(-x0)", "1"),
    ("x0*(x1+x2)-x0*x1", "x0*x2"),
    ("sin(x0+x1)", "sin(x1+x0)"),
])
def test_rewrites_show_equal_expressions_equal(left: str, right: str) -> None:
    assert egraph.equal(_p(left), _p(right))["equal"] is True


@pytest.mark.parametrize(("left", "right"), [("x0**2", "x0**3"), ("x0+x1", "x0*x1"), ("sin(x0)", "cos(x0)"),
                                             ("x0/(x0+1.3)", "x0/(x0+1.31)")])
def test_rewrites_never_call_different_expressions_equal_and_never_answer_false(left: str, right: str) -> None:
    assert egraph.equal(_p(left), _p(right))["equal"] is None


def test_the_log_of_a_product_is_split_only_when_positivity_is_granted() -> None:
    a, b = _p("log(x0*x1)"), _p("log(x0)+log(x1)")
    assert egraph.equal(a, b)["equal"] is None
    assert egraph.equal(a, b, positive=True)["equal"] is True


def test_saturation_stays_inside_its_node_limit() -> None:
    found = egraph.equal(_p("(x0+x1+x2+1)**3*(x0-x1)**3"), _p("x0"), node_limit=500)
    assert found["equal"] is None and found["nodes"] < 5000


def test_the_simplest_form_is_smaller_and_numerically_the_same() -> None:
    expr = _p("x0*1+0*x1+(x0+x0)")
    small = egraph.simplest(expr)
    assert sp.count_ops(small) < sp.count_ops(expr)
    assert abs(float(small.subs({"x0": 1.7, "x1": 2.0})) - 3 * 1.7) < 1e-12


# ---------------------------------------------------------------------- SMT
BOX = {"x0": (1.0, 5.0), "x1": (1.0, 5.0)}


def test_a_solver_finds_a_pole_inside_the_box_and_proves_there_is_none_outside() -> None:
    pytest.importorskip("z3")
    x0, x1 = sp.symbols("x0 x1")
    found = smt.pole_in_box(x0 / (x0 - 2.5) + x1, BOX)
    assert found["pole"] is True and abs(found["witness"]["x0"] - 2.5) < 1e-9
    assert smt.pole_in_box(x0 / (x0 + 1.3) + sp.log(1 + 0.7 * x1), BOX) == {"pole": False}
    assert smt.pole_in_box(1 / (x0 * x1 - 30), BOX) == {"pole": False}       # 30 is outside what x0*x1 reaches
    assert smt.pole_in_box(sp.log(x0 - x1 + 0.5), BOX)["pole"] is True


def test_what_the_solver_cannot_reason_about_is_not_decided() -> None:
    pytest.importorskip("z3")
    x0 = sp.Symbol("x0")
    assert smt.pole_in_box(1 / (sp.sin(x0) - 0.2), BOX)["pole"] is None


def test_two_rational_laws_are_proved_equal_or_told_apart_with_a_point() -> None:
    pytest.importorskip("z3")
    x0 = sp.Symbol("x0")
    assert smt.equal_in_box((x0 ** 2 - 1) / (x0 - 1), x0 + 1, BOX) == {"equal": True}
    apart = smt.equal_in_box(x0 / (x0 + 1.3), x0 / (x0 + 1.31), BOX)
    assert apart["equal"] is False and 1.0 <= apart["witness"]["x0"] <= 5.0


# ---------------------------------------------------------------------- MDL
def test_a_longer_law_costs_more_bits_and_a_better_fit_costs_fewer() -> None:
    x0, x1 = sp.symbols("x0 x1")
    assert mdl.law_bits(x0 + x1, 2) < mdl.law_bits(1.3 * x0 / (0.7 + x1) + sp.sin(2.3 * x0), 2)
    y = np.linspace(1, 5, 200)
    assert mdl.data_bits(0.01 * np.ones(200), y) < mdl.data_bits(0.5 * np.ones(200), y)
    assert np.isfinite(mdl.data_bits(np.zeros(200), y))                      # an exact fit is floored, not unbounded


def test_naming_a_block_is_cheaper_than_writing_it_out() -> None:
    written = mdl.law_bits(sp.sympify("h0/(1.5 + h1)"), 3, 8)
    assert mdl.block_use_bits(2, 1, 3, 8) < written


# ------------------------------------------------------------------ library
LAWS = ["1.3*x0/(0.7+x1) + 2.3*x2", "0.45*x1/(2.9+x0)", "1.7*exp(-0.3*x0)*x1/(1.3+x2)", "2.3*x0/(1.7+x1)-0.7*sin(1.3*x2)",
        "0.7*sin(2.3*x0)+x1**2", "3.7*exp(-0.45*x1)+0.3*x0", "1.3*log(1+0.7*x0)*x1", "2.9*log(1+0.3*x1)/x0"]


def test_the_library_keeps_a_block_in_two_inputs_that_several_laws_share() -> None:
    library = Library.learn(LAWS)
    assert "h0/(C + h1)" in library.blocks
    block = library.blocks["h0/(C + h1)"]
    assert block["holes"] == 2 and block["uses"] >= 3 and block["worth_bits"] > 0
    assert all(info["uses"] >= 2 for info in library.blocks.values())        # a part met once is never worth naming
    sizes = library.compression(LAWS)
    assert sizes["bits_with"] < sizes["bits_without"]


def test_a_library_is_the_same_after_a_round_trip_and_no_larger_than_asked() -> None:
    library = Library.learn(LAWS, max_blocks=1)
    assert len(library.blocks) == 1
    assert Library.from_dict(library.as_dict()).as_dict() == library.as_dict()


def test_fragments_number_their_inputs_as_holes_and_hide_constants() -> None:
    found = fragments_of("1.3*x2/(0.7+x0)")
    assert "h0/(C + h1)" in found or "h0/(h1 + C)" in found


def test_candidates_built_from_the_blocks_fit_a_law_made_of_them() -> None:
    library = Library.learn(LAWS)
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (400, 2))
    y = 2.9 * X[:, 1] / (0.45 + X[:, 0])
    laws, report = candidates(library, ["x0", "x1"], X, y, seconds=30)
    assert laws and report["placed"] > 0
    best = sp.lambdify(sp.symbols("x0 x1"), laws[0], "numpy")(*X.T)
    assert float(np.mean((best - y) ** 2)) < 1e-8 * float(np.var(y))
    assert all(law.free_symbols == set(sp.symbols("x0 x1")) for law in laws)
    shapes = [str(law.xreplace(dict.fromkeys(law.atoms(sp.Float), sp.Symbol("C")))) for law in laws]
    assert len(set(shapes)) == len(shapes)


# --------------------------------------------------------------- curriculum
def test_the_three_kinds_of_problem_never_share_a_law_and_none_is_in_the_benchmark() -> None:
    course = curriculum.Curriculum(seed=5)
    study = course.batch(6, "study 1")
    deciding = course.measurement(5, "selection 1")
    measured = course.measurement(5, "measurement 1")
    shapes = [curriculum.structure(p.truth) for p in [*study, *deciding, *measured]]
    assert len(set(shapes)) == len(shapes)
    excluded = curriculum.benchmark_fingerprints()
    assert excluded and not any(curriculum.fingerprint(sp.sympify(p.text)) in excluded for p in [*study, *deciding, *measured])


def test_measurement_problems_come_from_fixed_levels_whatever_the_curriculum_level() -> None:
    low, high = curriculum.Curriculum(seed=7, level=0.0), curriculum.Curriculum(seed=7, level=6.0)
    assert [p.level for p in low.measurement(5, "m")] == [p.level for p in high.measurement(5, "m")] == list(
        curriculum.MEASURED_LEVELS)
    assert {p.level for p in low.batch(4, "s")} <= {0, 1}


def test_the_level_follows_what_was_solved() -> None:
    course = curriculum.Curriculum(seed=1, level=3.0)
    assert course.update(9, 10) == 4.0
    assert course.update(5, 10) == 4.0
    assert course.update(1, 10) == 3.0


def test_a_level_that_runs_out_hands_study_to_the_next_level_up_but_never_a_measurement() -> None:
    """Level 0 has under ten laws. The first GitHub run of the levels stopped when a study batch asked it for more."""
    course = curriculum.Curriculum(seed=9, level=0.0)
    drawn = [p for k in range(4) for p in course.batch(8, f"study {k}")]
    assert len(drawn) == 32 and len({curriculum.structure(p.truth) for p in drawn}) == 32
    assert 0 in course.used_up and any(entry.get("level_used_up") == 0 for entry in course.history)
    assert max(p.level for p in drawn) >= 1
    course.used_up.clear()
    with pytest.raises(RuntimeError, match="does not change level"):
        course._draw(5, "measurement", [0], fixed=True)


def test_a_problems_rows_depend_on_its_key_alone() -> None:
    (problem,) = curriculum.Curriculum(seed=3).measurement(1, "m")
    a, b = problem.rows(), curriculum.Problem(**problem.as_dict()).rows()
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["X"].shape == (curriculum.ROWS, problem.inputs) and len(a["test_clean"]) == curriculum.TEST_ROWS


def test_the_package_never_opens_the_benchmark_only_its_fingerprints() -> None:
    """The forty held-back laws stay unread: no source here names the benchmark file as something to read."""
    for path in [*PACKAGE.glob("*.py"), ROOT / "scripts" / "run_recursion.py"]:
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"invented_laws[\w.]*", source):
            assert match.group(0) in ("invented_laws_fingerprints.json", "invented_laws.json``", "invented_laws.json"), path
        code = "\n".join(line for line in source.splitlines() if "``" not in line and not line.lstrip().startswith(("#", "*")))
        assert '"invented_laws.json"' not in code and "'invented_laws.json'" not in code, path
    assert curriculum.FINGERPRINTS.name == "invented_laws_fingerprints.json"


# -------------------------------------------------------------------- judge
def test_the_seal_covers_the_referee_the_thresholds_and_the_last_checks() -> None:
    sealed = judge.seal()
    for name in ("specifications/canonical_preregistered_spec_v1.json", "specifications/discovery_checks_v1.json",
                 "apedp_dyn/discovery_l2/vouching_v2.py", "apedp_dyn/discovery_l2/measured_range.py",
                 "apedp_dyn/recursion/judge.py", "apedp_dyn/recursion/mdl.py", "apedp_dyn/recursion/progress.py",
                 "apedp_referee/canonical_spec.py",
                 # what decides adoption and carries a measurement across processes (outside audit, 2026-11-24)
                 "apedp_dyn/recursion/levels.py", "apedp_dyn/recursion/chain.py", "apedp_dyn/recursion/library.py",
                 "scripts/run_recursion.py"):
        assert name in sealed
    judge.check(sealed)


def test_a_changed_judge_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    moved = tmp_path / "judge_copy.py"
    moved.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(judge, "sealed_paths", lambda: [moved])
    monkeypatch.setattr(judge, "ROOT", tmp_path)
    sealed = judge.seal()
    judge.check(sealed)
    moved.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(judge.JudgeChanged):
        judge.check(sealed)


@pytest.mark.parametrize("settings", [{"LAST_CHECKS_TEST_LEVEL": 0.5}, {"DISCOVERY_ACCURACY_R2": 0.9}, {"memory_seconds": 1e6},
                                      {"library_blocks": -1}, {"library_top": True}, {"noise": 0.2}])
def test_a_searcher_cannot_set_a_threshold_or_leave_its_bounds(settings: dict[str, float]) -> None:
    with pytest.raises(judge.SettingRefused):
        judge.check_settings(settings)
    with pytest.raises(judge.SettingRefused):
        levels.Searcher(settings)


def test_no_setting_shares_a_name_with_a_threshold() -> None:
    from apedp_dyn.conf import discovery_checks

    assert not set(judge.SETTINGS) & set(discovery_checks.THRESHOLDS)
    assert not any("level" in name or "power" in name or "r2" in name.lower() for name in judge.SETTINGS)


def test_the_verdict_separates_solved_wrong_and_not_reported() -> None:
    x0 = sp.Symbol("x0")
    X = np.linspace(1, 5, 200).reshape(-1, 1)
    clean = 1.3 * X[:, 0] / (0.7 + X[:, 0])
    right = judge.verdict({"reported": True, "law": "1.3*x0/(x0 + 0.7)"}, ["x0"], 1.3 * x0 / (0.7 + x0), X, clean)
    wrong = judge.verdict({"reported": True, "law": "0.2*x0"}, ["x0"], 1.3 * x0 / (0.7 + x0), X, clean)
    silent = judge.verdict({"reported": False, "law": None, "tentative_law": "1.3*x0/(x0 + 0.7)"}, ["x0"], x0, X, clean)
    assert (right["accurate"], right["solved"], right["wrong"], right["same_form"]) == (True, True, False, True)
    assert (wrong["accurate"], wrong["solved"], wrong["wrong"]) == (False, False, True)
    assert (silent["reported"], silent["accurate"], silent["solved"], silent["wrong"]) == (False, False, False, False)


def test_an_accurate_approximation_longer_than_the_law_is_not_the_law() -> None:
    """A ratio of cubics reaches the accuracy line on a saturating law without being it. The first run counted such laws solved."""
    x0 = sp.Symbol("x0")
    X = np.linspace(1, 5, 400).reshape(-1, 1)
    truth = 1.3 * sp.exp(-0.45 * x0)
    clean = 1.3 * np.exp(-0.45 * X[:, 0])
    top, bottom = np.polyfit(X[:, 0], clean, 3), [1.0]
    for _ in range(30):                                        # a rational fit by repeated linear solves
        design = np.column_stack([X[:, 0] ** 3, X[:, 0] ** 2, X[:, 0], np.ones(len(X)), -clean * X[:, 0], -clean * X[:, 0] ** 2])
        solved, *_ = np.linalg.lstsq(design, clean, rcond=None)
        top, bottom = solved[:4], [1.0, solved[4], solved[5]]
    a, b = [float(v) for v in top], [float(v) for v in bottom]
    approx = f"({a[0]!r}*x0**3 + {a[1]!r}*x0**2 + {a[2]!r}*x0 + {a[3]!r})/(1.0 + {b[1]!r}*x0 + {b[2]!r}*x0**2)"
    found = judge.verdict({"reported": True, "law": approx}, ["x0"], truth, X, clean)
    assert found["accurate"] is True and found["wrong"] is False
    assert found["solved"] is False and found["bits"] > found["truth_bits"]
    exact = judge.verdict({"reported": True, "law": "1.3*exp(-0.45*x0)"}, ["x0"], truth, X, clean)
    assert exact["solved"] is True and exact["bits"] == exact["truth_bits"]


def test_the_judge_uses_the_named_accuracy_line() -> None:
    from apedp_dyn.conf.discovery_checks import DISCOVERY_ACCURACY_R2

    assert judge.DISCOVERY_ACCURACY_R2 == DISCOVERY_ACCURACY_R2
    source = (PACKAGE / "judge.py").read_text(encoding="utf-8")
    assert "0.999" not in source


# ----------------------------------------------------------------- progress
def _m(k: int, solved: int, wrong: int = 0, cpu: float = 1000.0) -> progress.Measurement:
    return progress.Measurement(k, 40, solved, wrong, solved + wrong, cpu)


def test_recursive_means_the_rate_rose_at_every_step() -> None:
    assert progress.verdict([_m(0, 5), _m(1, 14), _m(2, 28)])["verdict"] == "recursive"
    once = progress.verdict([_m(0, 5), _m(1, 14), _m(2, 15)])
    assert once["verdict"] == "improved, not shown recursive" and once["steps_rose"] == [True, False]
    assert progress.verdict([_m(0, 10), _m(1, 11), _m(2, 10)])["verdict"] == "not shown"


def test_two_generations_cannot_show_recursion_and_wrong_laws_cannot_buy_it() -> None:
    assert progress.verdict([_m(0, 5), _m(1, 30)])["verdict"] == "not shown"
    bought = progress.verdict([_m(0, 5), _m(1, 14, 2), _m(2, 28, 6)])
    assert bought["verdict"] == "not shown" and "wrong" in bought["reason"]


def test_the_rate_counts_cpu_time_so_a_slower_searcher_that_solves_the_same_is_worse() -> None:
    assert _m(0, 10, cpu=2000.0).g == pytest.approx(_m(0, 10, cpu=1000.0).g / 2)
    assert _m(0, 10).g == pytest.approx(36.0)


# ------------------------------------------------------------------- levels
def test_varied_settings_stay_the_searchers_own_and_inside_bounds() -> None:
    rng = np.random.default_rng(0)
    settings = dict(levels.DEFAULT_SETTINGS)
    seen_change = False
    for _ in range(200):
        moved = levels.vary_settings(settings, 0.9, rng)
        assert set(moved) == set(judge.SETTINGS)
        judge.check_settings(moved)
        seen_change |= moved != settings
        settings = moved
    assert seen_change


def test_a_variant_is_adopted_only_if_faster_per_cpu_and_no_more_wrong() -> None:
    incumbent = {"solved": 5, "per_cpu_hour": 10.0, "cpu_seconds": 1800.0, "wrong": 1}
    assert levels.better({"solved": 6, "per_cpu_hour": 12.0, "cpu_seconds": 1800.0, "wrong": 1}, incumbent)
    assert not levels.better({"solved": 6, "per_cpu_hour": 12.0, "cpu_seconds": 1800.0, "wrong": 2}, incumbent)
    assert not levels.better({"solved": 6, "per_cpu_hour": 9.0, "cpu_seconds": 2400.0, "wrong": 1}, incumbent)   # more, but slower per law
    assert not levels.better({"solved": 4, "per_cpu_hour": 20.0, "cpu_seconds": 720.0, "wrong": 0}, incumbent)


def test_a_copy_that_solves_the_same_is_not_taken_for_a_few_seconds_of_timing_noise() -> None:
    """In the first run a copy with the same problems solved and 2 % less CPU replaced the incumbent, seven times of eight."""
    incumbent = {"solved": 3, "per_cpu_hour": 24.637, "cpu_seconds": 438.36, "wrong": 0}
    assert not levels.better({"solved": 3, "per_cpu_hour": 25.092, "cpu_seconds": 430.42, "wrong": 0}, incumbent)
    assert levels.better({"solved": 3, "per_cpu_hour": 36.0, "cpu_seconds": 300.0, "wrong": 0}, incumbent)


def test_without_a_coding_model_the_code_is_not_varied_and_the_record_says_so() -> None:
    code, report = levels.code_step(levels.Searcher(), None, [], [])
    assert code is None and report == {"ran": False, "reason": levels.NO_MODEL}


def test_a_searcher_with_nothing_learned_adds_no_proposer_and_round_trips() -> None:
    searcher = levels.Searcher({"memory_seconds": 5})
    assert searcher.proposer() is None and searcher.settings["library_blocks"] == levels.DEFAULT_SETTINGS["library_blocks"]
    again = levels.Searcher.from_dict(searcher.as_dict())
    assert again.as_dict() == searcher.as_dict()
    assert levels.Searcher(laws=LAWS).proposer() is not None


def _fake_batch(searcher: levels.Searcher, problems: list[curriculum.Problem], workers: int = 1) -> list[dict[str, object]]:
    """Stands in for the route: solves a problem if the searcher has learned enough for its level; costs its seconds."""
    cost = float(searcher.settings["memory_seconds"]) + float(searcher.settings["library_seconds"]) + 1.0
    return [{"key": p.key, "level": p.level, "noise": p.noise, "reported": p.level <= 1 + len(searcher.laws),
             "solved": p.level <= 1 + len(searcher.laws), "wrong": False, "learnable": True,
             "law": p.text if p.level <= 1 + len(searcher.laws) else None, "cpu_seconds": cost} for p in problems]


def test_level_one_learns_from_study_measures_on_other_problems_and_reports_the_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    run = levels.run_level1(levels.Searcher(), generations=3, meta=levels.Meta(variants=1, study=4, selection=5), seed=11,
                            measure=5, tag="test")
    assert [m["generation"] for m in run["measurements"]] == [0, 1, 2, 3]
    assert len(run["searcher"]["laws"]) > 0 and run["generations"][0]["study"]["learned"] > 0
    assert run["measurements"][-1]["solved"] >= run["measurements"][0]["solved"]
    assert run["progress"]["verdict"] in ("recursive", "improved, not shown recursive", "not shown")
    assert all(g["code"] == {"ran": False, "reason": levels.NO_MODEL} for g in run["generations"])
    assert run["sealed_files"] == len(judge.seal())
    keys = [r["key"] for g in run["generations"] for r in g["rows"]["measurement"]]
    assert len(set(keys)) == len(keys) and all("measurement" in key for key in keys)


def test_level_one_stops_if_the_judge_changes_mid_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    calls = {"n": 0}
    real = judge.seal

    def moving() -> dict[str, str]:
        calls["n"] += 1
        sealed = dict(real())
        if calls["n"] > 2:
            sealed["apedp_dyn/discovery_l2/vouching_v2.py"] = "changed"
        return sealed

    monkeypatch.setattr(judge, "seal", moving)
    with pytest.raises(judge.JudgeChanged):
        levels.run_level1(levels.Searcher(), generations=2, meta=levels.Meta(variants=1, study=4, selection=4), seed=2, measure=4)


def test_level_two_compares_ways_of_improving_from_the_same_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    out = levels.run_level2(levels.Searcher(), rounds=2, metas=3, generations=3, seed=4, measure=5,
                            meta=levels.Meta(variants=1, study=4, selection=4))
    assert len(out["rounds"]) == 2
    for round_ in out["rounds"]:
        assert len(round_["runs"]) == 3 and round_["runs"][0]["meta"] == round_["incumbent"]
        best = max(round_["runs"], key=lambda run: run["gain_per_cpu_hour"])
        assert round_["chosen"] == best["meta"]
    assert out["rounds"][1]["incumbent"] == out["rounds"][0]["chosen"]
    assert out["meta"] == out["rounds"][-1]["chosen"]


def test_a_later_round_run_alone_draws_other_problems_and_reports_each_run_as_it_ends(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    seen: list[dict[str, object]] = []
    start = levels.Meta(variants=1, study=4, selection=4)
    first = levels.run_level2(levels.Searcher(), rounds=1, metas=2, generations=2, seed=4, measure=5, meta=start, on_run=seen.append)
    second = levels.run_level2(levels.Searcher(), rounds=1, metas=2, generations=2, seed=4, measure=5,
                               meta=levels.Meta(**first["meta"]), first_round=2, on_run=seen.append)
    assert [(item["round"], item["index"]) for item in seen] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert first["rounds"][0]["round"] == 1 and second["rounds"][0]["round"] == 2
    assert second["rounds"][0]["incumbent"] == first["meta"]
    generations: list[dict[str, object]] = []
    levels.run_level1(levels.Searcher(), generations=2, meta=start, seed=4, measure=5, on_generation=generations.append)
    assert [g["generation"] for g in generations] == [0, 1, 2] and all("rows" not in g for g in generations)


def test_a_varied_meta_differs_in_exactly_one_thing() -> None:
    rng = np.random.default_rng(1)
    meta = levels.Meta()
    for _ in range(50):
        copy = levels.vary_meta(meta, rng)
        changed = [k for k in meta.as_dict() if meta.as_dict()[k] != copy.as_dict()[k]]
        assert len(changed) <= 1


# ------------------------------------------------- with the route's searches
def test_the_search_stage_is_a_bounded_setting_and_stands_first_beside_the_library() -> None:
    assert levels.DEFAULT_SETTINGS["search_seconds"] == 0.0            # off unless asked for
    with pytest.raises(judge.SettingRefused):
        levels.Searcher({"search_seconds": 500})
    alone = levels.Searcher({"search_seconds": 30}).proposer(3)
    assert alone is not None and type(alone).__name__ == "FitGuardProposer"
    both = levels.Searcher({"search_seconds": 30}, laws=LAWS).proposer(3)
    assert type(both).__name__ == "CompositeProposer" and len(both._members) == 2 and both._max == levels.MAX_PROPOSALS
    rng = np.random.default_rng(0)                                     # a searcher without searches cannot switch them on
    assert all(levels.vary_settings(dict(levels.DEFAULT_SETTINGS), 0.9, rng)["search_seconds"] == 0.0 for _ in range(50))


# ------------------------------------------------------------------- chain
def _through_json(record: dict[str, object]) -> dict[str, object]:
    import json

    return json.loads(json.dumps(record))


def test_level_one_in_steps_hands_its_state_on_and_measures_both_arms_on_the_same_problems(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from apedp_dyn.recursion import chain

    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    state = _through_json(chain.plan(levels.Searcher(), generations=2, measure=6,
                                     meta=levels.Meta(variants=1, study=4, selection=4), seed=3))
    drawn_first = {p["key"] for problems in state["measurement"].values() for p in problems}
    assert len(drawn_first) == 18 and set(state["measurement"]) == {"0", "1", "2"}
    measured = [chain.measure_step(state, 0)]
    improved = []
    for k in (1, 2):
        state, record = chain.improve_step(state, k)
        state = _through_json(state)
        improved.append(record)
        for arm in chain.ARMS:
            measured += [chain.measure_step(state, k, arm=arm, shard=shard, shards=2) for shard in (0, 1)]
    assert state["generation"] == 2 and len(state["searcher"]["laws"]) > 0 and state["start"]["laws"] == []
    measured_shapes = {curriculum.structure(curriculum.Problem(**p).truth) for ps in state["measurement"].values() for p in ps}
    assert measured_shapes <= set(state["curriculum"]["seen"])                 # so no later batch could draw one of them
    record = chain.collect(state, improved, measured)
    assert [m["generation"] for m in record["measurements"]] == [0, 1, 2] == [m["generation"] for m in record["measurements_start"]]
    assert record["incomplete"] == [] and record["measurements_start"][0] == record["measurements"][0]
    assert all(m["problems"] == 6 for m in record["measurements"])
    for pair in record["paired"]:
        assert pair["solved_current"] - pair["solved_start"] == pair["gained"] - pair["lost"]
    assert sum(pair["gained"] for pair in record["paired"]) > 0               # the fake searcher solves more once it has learned
    assert record["progress"]["verdict"] in ("recursive", "improved, not shown recursive", "not shown")


def test_a_step_refuses_the_wrong_generation_a_missing_shard_is_listed_and_a_changed_judge_stops_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from apedp_dyn.recursion import chain

    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    state = chain.plan(levels.Searcher(), generations=1, measure=4, meta=levels.Meta(variants=1, study=4, selection=4), seed=8)
    with pytest.raises(ValueError, match="cannot follow"):
        chain.improve_step(state, 2)
    with pytest.raises(ValueError, match="not 1"):
        chain.measure_step(state, 1, arm="current")
    after, record = chain.improve_step(state, 1)
    pieces = [chain.measure_step(state, 0), chain.measure_step(after, 1, shard=0, shards=2),
              chain.measure_step(after, 1, arm="start")]
    collected = chain.collect(after, [record], pieces)
    assert collected["incomplete"] == [{"generation": 1, "arm": "current", "rows": 2, "expected": 4}]
    assert [m["generation"] for m in collected["measurements"]] == [0] and collected["paired"][-1]["generation"] == 0
    tampered = {**after, "sealed": {**after["sealed"], "apedp_dyn/recursion/judge.py": "changed"}}
    with pytest.raises(judge.JudgeChanged):
        chain.measure_step(tampered, 1)
    with pytest.raises(judge.JudgeChanged):
        chain.improve_step({**state, "sealed": tampered["sealed"]}, 1)


def _script() -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_recursion", ROOT / "scripts" / "run_recursion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_runs_the_steps_as_separate_calls_and_collects_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    import json

    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    script = _script()
    sizes = ["--measure", "4", "--study", "4", "--selection", "4", "--variants", "1"]
    everything = tmp_path / "all"
    assert script.main(["plan", "--out", str(everything), "--generations", "2", *sizes]) == 0
    script.main(["measure", "--state", str(everything / "state_0.json"), "--generation", "0", "--out", str(everything)])
    for k in (1, 2):
        script.main(["improve", "--state", str(everything / f"state_{k - 1}.json"), "--generation", str(k), "--out", str(everything)])
        for arm in ("current", "start"):
            for shard in ("0", "1"):
                script.main(["measure", "--state", str(everything / f"state_{k}.json"), "--generation", str(k), "--arm", arm,
                             "--shard", shard, "--shards", "2", "--out", str(everything)])
    assert script.main(["chain-collect", "--inputs", str(everything), "--out", str(tmp_path / "out")]) == 0
    record = json.loads((tmp_path / "out" / "chain.json").read_text(encoding="utf-8"))
    assert [m["generation"] for m in record["measurements"]] == [0, 1, 2] and record["incomplete"] == []
    assert len(record["paired"]) == 3 and len(record["generations"]) == 2
    capsys.readouterr()


def test_the_script_runs_a_round_of_level_two_as_separate_runs_and_chooses_as_one_process_would(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    monkeypatch.setattr(levels, "run_batch", _fake_batch)
    script = _script()
    sizes = ["--generations", "2", "--measure", "5", "--study", "4", "--selection", "4", "--variants", "1", "--metas", "3",
             "--seed", "4"]
    runs = tmp_path / "runs"
    for index in ("0", "1", "2"):
        script.main(["level2-one", "--round", "1", "--index", index, "--out", str(runs), *sizes])
    script.main(["level2-choose", "--round", "1", "--inputs", str(runs), "--out", str(tmp_path / "r1"), *sizes])
    apart = json.loads((tmp_path / "r1" / "level2.json").read_text(encoding="utf-8"))
    together = levels.run_level2(levels.Searcher(), rounds=1, metas=3, generations=2, seed=4, measure=5,
                                 meta=levels.Meta(variants=1, study=4, selection=4))
    assert apart["meta"] == together["meta"] and apart["rounds"][0]["missing"] == []
    assert [run["meta"] for run in apart["rounds"][0]["runs"]] == [run["meta"] for run in together["rounds"][0]["runs"]]
    assert [run["g"] for run in apart["rounds"][0]["runs"]] == [run["g"] for run in together["rounds"][0]["runs"]]
    second = tmp_path / "runs2"                                             # round 2 starts from what round 1 chose
    script.main(["level2-one", "--round", "2", "--index", "0", "--meta-from", str(tmp_path / "r1" / "level2.json"),
                 "--out", str(second), *sizes])
    assert json.loads((second / "level2_r2_m0.json").read_text(encoding="utf-8"))["meta"] == apart["meta"]
    with pytest.raises(SystemExit):
        script.main(["level2-one", "--round", "1", "--index", "7", "--out", str(runs), *sizes])
    capsys.readouterr()


def test_asking_for_searches_without_pysr_stops_before_anything_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real = builtins.__import__

    def no_pysr(name: str, *args: object, **kwargs: object) -> object:
        if name == "pysr":
            raise ImportError("no pysr here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pysr)
    monkeypatch.delitem(__import__("sys").modules, "pysr", raising=False)
    with pytest.raises(SystemExit, match="PySR cannot be imported"):
        _script().main(["plan", "--out", str(tmp_path), "--searches", "90"])
    assert not (tmp_path / "state_0.json").exists()
