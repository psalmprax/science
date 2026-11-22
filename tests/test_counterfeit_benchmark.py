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
    return {"status": "ok", "world": {"truth_listed": offered, "noise": noise},
            **{method: outcome[what] for method, what in methods.items()}}


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
    assert e["1_holds"] and e["2_holds"] and e["3_holds"] and e["2_scores"]["route"] == pytest.approx(1.0)
    assert e["2_scores"]["best_fit"] < e["2_scores"]["validated"] < 0          # one wrong law in ten cases already costs it all
    # one wrong law at 5 % noise does not touch the first expectation, which is about noise up to 1 %; two at 1 % do
    high = [*good, _row(False, 0.05, route="wrong", best_fit="wrong", validated="wrong")]
    assert bench.summarise(high)["expectations"]["1_holds"]
    low = [*good, *[_row(False, 0.01, route="wrong", best_fit="wrong", validated="wrong") for _ in range(2)]]
    assert not bench.summarise(low)["expectations"]["1_holds"]
    # a route that says nothing at all scores zero: above methods that report wrong laws, and still worth nothing
    mute = [dict(row, route={"reported": False, "symbolic": False, "wrong": False}) for row in good]
    e = bench.summarise(mute)["expectations"]
    assert not e["3_holds"] and e["2_holds"] and e["3_route_score"] == 0.0


def test_the_benchmark_file_is_cases_of_laws_with_their_counterfeits(bench) -> None:
    record = json.loads(bench.OUT.read_text(encoding="utf-8"))
    assert len(record["laws"]) == bench.LAW_COUNT and record["seed"] == bench.SEED and record["rows"] == bench.ROWS
    assert len(record["cases"]) == bench.LAW_COUNT * len(bench.NOISE_LEVELS) * 2
    assert len({case["case"] for case in record["cases"]}) == len(record["cases"])
    by_id = {law["id"]: law for law in record["laws"]}
    for case in record["cases"]:
        world = worlds.World.from_dict(case["world"])
        law = by_id[case["law_id"]]
        fakes = {fake["text"] for fake in law["counterfeits"]}
        assert world.law == law["law"] and world.sampling == "uniform" and world.noise_shape == "gaussian"
        assert world.truth_listed == (world.law in world.candidates)
        assert set(world.candidates) - {world.law} == fakes and 2 <= len(fakes) <= len(bench.BANDS)
    for law in record["laws"]:
        distances = [fake["distance"] for fake in law["counterfeits"]]
        assert distances == sorted(distances) and all(bench.BANDS[0][0] <= d < bench.BANDS[-1][1] for d in distances)
        assert sp.sympify(law["law"]).free_symbols == {sp.Symbol(f"x{i}") for i in range(law["inputs"])}


def test_no_law_of_the_benchmark_is_used_anywhere_else(bench) -> None:
    grammar = _load("build_invented_laws")
    record = json.loads(bench.OUT.read_text(encoding="utf-8"))
    structures = {grammar.structure(sp.sympify(law["law"])) for law in record["laws"]}
    assert len(structures) == bench.LAW_COUNT and not structures & bench._taken()
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    assert not {grammar.fingerprint(law["law"]) for law in record["laws"]} & in_benchmark
