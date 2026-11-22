"""The counterfeit-law benchmark: what a case is, how a method is scored, and that the file is its generator's."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import sympy as sp

from apedp_dyn.redteam import worlds

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load("run_counterfeit_benchmark")


def _row(offered: bool, noise: float, **methods: str) -> dict:
    """A scored case: each method's outcome is ``right``, ``wrong``, ``close`` (accurate, not the law) or ``silent``."""
    outcome = {"right": {"reported": True, "symbolic": True, "wrong": False},
               "wrong": {"reported": True, "symbolic": False, "wrong": True},
               "close": {"reported": True, "symbolic": False, "wrong": False},
               "silent": {"reported": False, "symbolic": False, "wrong": False}}
    methods.setdefault("vouched", methods.get("route", "silent"))       # unless a test says otherwise, the checks change nothing
    return {"status": "ok", "world": {"truth_listed": offered, "noise": noise},
            **{method: dict(outcome[what]) for method, what in methods.items()}}


def test_a_wrong_law_costs_nineteen_right_ones(bench) -> None:
    assert bench.PENALTY == pytest.approx(19.0)
    rows = [_row(True, 0.0, route="right", best_fit="right", validated="right") for _ in range(19)]
    rows.append(_row(False, 0.0, route="silent", best_fit="wrong", validated="wrong"))
    route, plain = bench.tally(rows, "route"), bench.tally(rows, "best_fit")
    assert route["score"] == pytest.approx(1.0) and route["wrong"] == 0 and route["reported_when_withheld"] == 0
    assert plain["score"] == pytest.approx(0.0) and plain["wrong_share_of_reported"] == pytest.approx(1 / 20)


def test_saying_nothing_and_an_accurate_approximation_count_nothing(bench) -> None:
    rows = [_row(True, 0.01, route="silent", best_fit="close", validated="close"),
            _row(False, 0.01, route="close", best_fit="close", validated="silent")]
    for method in bench.METHODS:
        scored = bench.tally(rows, method)
        assert scored["score"] == 0.0 and scored["right"] == 0 and scored["wrong"] == 0
    assert bench.tally(rows, "route")["accurate_not_the_law"] == 1


def test_the_expectations_are_judged_as_written(bench) -> None:
    good = [_row(True, noise, route="right", best_fit="right", validated="right") for noise in bench.NOISE_LEVELS for _ in range(10)]
    good += [_row(False, noise, route="silent", best_fit="wrong", validated="wrong" if k == 0 else "silent")
             for noise in bench.NOISE_LEVELS for k in range(10)]
    e = bench.summarise(good)["expectations"]
    assert e["1_holds"] and e["2_holds"] and e["3_holds"] and e["5_holds"] and e["2_scores"]["route"] == pytest.approx(1.0)
    assert not e["4_holds"]                                                    # checks that change nothing score the same, not higher
    assert e["2_scores"]["best_fit"] < e["2_scores"]["validated"] < 0          # one wrong law in ten cases already costs it all
    # one wrong law at 5 % noise does not touch the first expectation, which is about noise up to 1 %; two at 1 % do
    high = [*good, _row(False, 0.05, route="wrong", best_fit="wrong", validated="wrong")]
    assert bench.summarise(high)["expectations"]["1_holds"]
    low = [*good, *[_row(False, 0.01, route="wrong", best_fit="wrong", validated="wrong") for _ in range(2)]]
    assert not bench.summarise(low)["expectations"]["1_holds"]
    # a route that says nothing at all scores zero: above methods that report wrong laws
    mute = [dict(row, route={"reported": False, "symbolic": False, "wrong": False}) for row in good]
    e = bench.summarise(mute)["expectations"]
    assert e["2_holds"] and e["2_scores"]["route"] == 0.0 and not e["5_holds"]         # and it had no right law to keep


def test_the_checks_are_judged_on_every_noise_level_on_the_score_and_on_what_they_keep(bench) -> None:
    def cases(vouched_when_wrong: str, kept: int) -> list[dict]:
        rows = [_row(True, noise, route="right", vouched="right" if k < kept else "silent", best_fit="right", validated="right")
                for noise in bench.NOISE_LEVELS for k in range(10)]
        rows += [_row(False, 0.05, route="wrong", vouched=vouched_when_wrong, best_fit="wrong", validated="wrong")]
        return rows

    e = bench.summarise(cases("silent", 10))["expectations"]
    assert e["3_holds"] and e["4_holds"] and e["5_holds"] and e["5_right_laws_of_the_route_kept"] == [30, 30]
    assert e["2_scores"]["vouched"] == pytest.approx(1.0) and e["2_scores"]["route"] == pytest.approx((30 - 19) / 30)
    # a wrong law let through at one noise level breaks the third expectation, whatever the other levels show
    e = bench.summarise(cases("wrong", 10))["expectations"]
    assert not e["3_holds"] and e["3_wrong_share_of_vouched_reports_by_noise"]["0.05"] == pytest.approx(1 / 11)
    # keeping seven right laws in ten is below the canonical share of valid laws a referee must retain
    assert bench.KEPT_SHARE == 0.75
    assert not bench.summarise(cases("silent", 7))["expectations"]["5_holds"]
    assert bench.summarise(cases("silent", 8))["expectations"]["5_holds"]
    summary = bench.summarise(cases("silent", 8))
    assert summary["vouched_switched"] == {"runs": 0, "into_the_law": 0} and summary["why_vouched_withheld"] == {"": 7}


def test_the_vouched_method_is_the_routes_report_after_the_checks(bench) -> None:
    """On a law of the smoke job, which is in no benchmark: no case of the benchmark is read here."""
    offered = worlds.World.from_dict(bench.smoke_cases()[2]["world"])                  # 1 % noise, the law offered
    assert offered.law == bench.SMOKE_LAW and offered.truth_listed and offered.noise == 0.01
    passed = bench.vouched_row(offered, {"reported": True, "law": offered.law, "equivalent": []})
    assert passed["reported"] and passed["symbolic"] and passed["reason"] == "" and not passed["switched"]
    silent = bench.vouched_row(offered, {"reported": False, "law": None, "equivalent": []})
    assert not silent["reported"] and silent["reason"] == "the route reported nothing"
    far = "0.52*x0 + 2.2"                                                              # a line: far from the law
    swapped = bench.vouched_row(offered, {"reported": True, "law": far, "equivalent": [offered.law]})
    assert swapped["reported"] and swapped["symbolic"] and swapped["switched"]         # it gives way to the law
    alone = bench.vouched_row(offered, {"reported": True, "law": far, "equivalent": []})
    assert not alone["reported"] and "structure" in alone["reason"]
    assert {case["case"] for case in bench.smoke_cases()}.isdisjoint(case["case"] for case in bench.cases())


def test_checks_that_crash_vouch_for_nothing_and_the_case_is_kept(bench, monkeypatch) -> None:
    red_team = _load("run_red_team")
    picked = bench.smoke_cases()[:2]
    made = [worlds.World.from_dict(case["world"]) for case in picked]
    route = {"reported": True, "symbolic": True, "wrong": False, "law": bench.SMOKE_LAW, "equivalent": []}
    plain = {"reported": True, "symbolic": False, "wrong": True}
    monkeypatch.setattr(red_team, "evaluate", lambda batch, cores, limit: [
        {"world": world.as_dict(), "status": "ok", "route": dict(route), "best_fit": dict(plain), "validated": dict(plain)}
        for world in batch])

    def broken(*args, **kwargs):
        raise FloatingPointError("no neighbours")

    monkeypatch.setattr(bench.vouching, "vouch", broken)
    rows = bench.evaluate(made, 1)
    assert [row["vouched"]["reported"] for row in rows] == [False, False] and all(row["vouched"]["failed"] for row in rows)
    assert "FloatingPointError" in rows[0]["vouched"]["reason"]
    summary = bench.summarise(rows)
    assert summary["checks_failed"] == 2 and summary["vouched|all"]["reported"] == 0 and summary["route|all"]["right"] == 2
    assert summary["expectations"]["5_right_laws_of_the_route_kept"] == [0, 2] and not summary["expectations"]["5_holds"]
    # what silence looks like is what the red team's scoring records for a method that reported nothing
    names = ["x0"]
    nothing = red_team._scored(None, sp.sympify(bench.SMOKE_LAW), names, worlds.rows_of(made[0]))
    assert {key: bench.silent("")[key] for key in nothing} == nothing


def test_the_forecast_on_file_is_what_the_campaign_and_the_replay_give(bench) -> None:
    filed = json.loads(bench.FORECAST.read_text(encoding="utf-8"))
    again = bench.forecast()
    # what it counted is exact; what it drew is compared within what another build of the random generator could move
    drawn = ("totals", "chance_that_an_expectation_holds")
    assert {key: value for key, value in filed.items() if key not in drawn} == \
        {key: value for key, value in again.items() if key not in drawn}
    one_law = bench.PENALTY / (bench.LAW_COUNT * len(bench.NOISE_LEVELS))
    for method in bench.METHODS:
        for name, slack in (("right", 1.0), ("wrong", 1.0), ("score", one_law)):
            was, now = filed["totals"][method][name], again["totals"][method][name]
            assert was["expected"] == pytest.approx(now["expected"], abs=0.3 * slack)
            assert was["interval"] == pytest.approx(now["interval"], abs=1.01 * slack)
    for key in "12345":
        assert filed[drawn[1]][key] == pytest.approx(again[drawn[1]][key], abs=0.02)
    totals = filed["totals"]
    assert set(totals) == set(bench.METHODS) and all(set(v) == {"right", "wrong", "score"} for v in totals.values())
    assert totals["route"]["score"]["expected"] < 0 < totals["vouched"]["score"]["expected"]
    chances = filed["chance_that_an_expectation_holds"]
    assert set(chances) == {"1", "2", "3", "4", "5", "3_by_noise"} and set(chances["3_by_noise"]) == {"0", "0.01", "0.05"}
    assert all(0.0 <= chances[key] <= 1.0 for key in "12345") and chances["3"] <= min(chances["3_by_noise"].values())
    # a withheld case offers wrong candidates only, so the campaign's worlds standing in for it are those whose best fit was wrong
    for key, cell in filed["counts"].items():
        if key.endswith("withheld"):
            assert cell["best_fit"] == [0, 0, cell["survey_worlds"], 0]


def test_the_forecast_judges_each_drawn_run_as_the_run_will_be_judged(bench) -> None:
    import numpy as np

    many = 40_000                       # so many worlds behind each share that the half count added to each is nothing
    silent, right, wrong = [0, 0, 0, many], [many, 0, 0, 0], [0, 0, many, 0]

    def cells(route_when_withheld_at_5: list[int], after_wrong: list[int], after_law: list[int]) -> dict:
        out = {}
        for noise in bench.NOISE_LEVELS:
            out[(noise, True)] = {"route": right, "best_fit": right, "validated": right,
                                  "after": {"the law": after_law, "accurate, not the law": [0, 0, 0, 0], "wrong": [0, 0, 0, 0]}}
            out[(noise, False)] = {"route": route_when_withheld_at_5 if noise == 0.05 else silent, "best_fit": wrong,
                                   "validated": wrong,
                                   "after": {"the law": [0, 0, 0, 0], "accurate, not the law": [0, 0, 0, 0], "wrong": after_wrong}}
        return out

    def chances(*args: list[int]) -> dict:
        after_wrong, after_law = args[1], args[2]
        pooled = {"the law": after_law, "accurate, not the law": [0, many, 0, 0], "wrong": after_wrong}
        return bench.simulate(cells(*args), pooled, np.random.default_rng(3), draws=400)["chance_that_an_expectation_holds"]

    passes, stops = [many, 0, 0, 0], [0, 0, 0, many]
    # a route that is wrong at 5 % noise only, and checks that stop every wrong law and pass every law
    c = chances(wrong, stops, passes)
    assert c["1"] > 0.99 and c["2"] > 0.99 and c["3"] > 0.95 and c["4"] > 0.99 and c["5"] > 0.99
    # checks that let the wrong laws through: the third fails at that noise level and nowhere else, and nothing is gained
    c = chances(wrong, [0, 0, many, 0], passes)
    assert c["3"] < 0.01 and c["3_by_noise"]["0.05"] < 0.01 and c["3_by_noise"]["0"] > 0.99 and c["4"] < 0.05
    # checks that stop everything, laws included: no wrong law, a higher score than the route, and nothing kept
    c = chances(wrong, stops, stops)
    assert c["3"] > 0.95 and c["4"] > 0.99 and c["5"] < 0.01
    # a route that is never wrong: the checks cannot score above it
    c = chances(silent, stops, passes)
    assert c["1"] > 0.99 and c["4"] < 0.05


def test_the_benchmark_file_is_cases_of_laws_with_their_counterfeits(bench) -> None:
    record = json.loads(bench.OUT.read_text(encoding="utf-8"))
    assert len(record["laws"]) == bench.LAW_COUNT and record["seed"] == bench.SEED and record["rows"] == bench.ROWS
    assert len(record["cases"]) == bench.LAW_COUNT * len(bench.NOISE_LEVELS) * 2
    assert len({case["case"] for case in record["cases"]}) == len(record["cases"])
    by_id = {law["id"]: law for law in record["laws"]}
    accurate_up_to, wrong_from = bench.BANDS[1][1], bench.BANDS[2][0]
    for case in record["cases"]:
        world = worlds.World.from_dict(case["world"])
        law = by_id[case["law_id"]]
        assert world.law == law["law"] and world.sampling == "uniform" and world.noise_shape == "gaussian"
        assert world.truth_listed == (world.law in world.candidates)
        if world.truth_listed:                          # the law beside one counterfeit of each band it has
            assert set(world.candidates) - {world.law} == {fake["text"] for fake in law["counterfeits"]}
        else:                                           # no law, and nothing that is nearly right
            assert set(world.candidates) == {fake["text"] for fake in law["wrong_counterfeits"]}
    for law in record["laws"]:
        beside = [fake["distance"] for fake in law["counterfeits"]]
        alone = [fake["distance"] for fake in law["wrong_counterfeits"]]
        assert beside == sorted(beside) and 2 <= len(beside) <= len(bench.BANDS)
        assert beside[0] < accurate_up_to and beside[-1] >= wrong_from          # one that is nearly right, one that is wrong
        assert not any(accurate_up_to <= d < wrong_from for d in beside)        # and none on the line between
        assert 1 <= len(alone) <= bench.WITHHELD and all(wrong_from <= d < bench.BANDS[2][1] for d in alone)
        assert law["counterfeits"][-1] in law["wrong_counterfeits"]
        assert sp.sympify(law["law"]).free_symbols == {sp.Symbol(f"x{i}") for i in range(law["inputs"])}
    # a distance of the accurate bands' upper end is the benchmarks' accuracy line
    base = _load("run_srbench_baseline")
    assert 1.0 - accurate_up_to ** 2 == pytest.approx(base.R2_ACCURACY, abs=2e-6)


def test_no_law_of_the_benchmark_is_used_anywhere_else(bench) -> None:
    grammar = _load("build_invented_laws")
    record = json.loads(bench.OUT.read_text(encoding="utf-8"))
    structures = {grammar.structure(sp.sympify(law["law"])) for law in record["laws"]}
    assert len(structures) == bench.LAW_COUNT and not structures & bench._taken()
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    assert not {grammar.fingerprint(law["law"]) for law in record["laws"]} & in_benchmark
