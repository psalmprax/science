"""PREREG-VOUCHING-V2: how the run is judged, and that its laws are new."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import sympy as sp

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
def run():
    return _load("run_vouching_v2_blind")


OUT = {"right": {"reported": True, "symbolic": True, "wrong": False, "accuracy_solution": True},
       "wrong": {"reported": True, "symbolic": False, "wrong": True, "accuracy_solution": False},
       "close": {"reported": True, "symbolic": False, "wrong": False, "accuracy_solution": True},
       "silent": {"reported": False, "symbolic": False, "wrong": False, "accuracy_solution": False}}


def _handed(noise: float, offered: bool, **arms: str) -> dict:
    row = {"status": "ok", "world": {"noise": noise, "truth_listed": offered}}
    for method in ("route", "first", "second", "best_fit", "validated"):
        row[method] = dict(OUT[arms.get(method, arms.get("route", "silent"))])
    return row


def _searched(noise: float, **arms: str) -> dict:
    return {"status": "ok", "noise": noise, **{arm: dict(OUT[arms.get(arm, arms.get("route", "silent"))])
                                               for arm in ("route", "first", "second")}}


def test_the_limits_are_the_canonical_ones(run) -> None:
    assert run.WRONG_LIMIT == 0.05 and run.KEPT_SHARE == pytest.approx(0.95) and run.KEPT_SHARE_NOISY == 0.75
    assert run.NOISE_LEVELS == (0.0, 0.01, 0.05) and run.QUIET == 0.01


def test_the_expectations_are_judged_as_written(run) -> None:
    handed = [_handed(n, True, route="right", best_fit="right", validated="right") for n in run.NOISE_LEVELS for _ in range(20)]
    handed += [_handed(n, False, route="silent", best_fit="wrong", validated="wrong") for n in run.NOISE_LEVELS for _ in range(20)]
    searched = [_searched(n, route="right") for n in run.NOISE_LEVELS for _ in range(20)]
    e = run.summarise(handed, searched)["expectations"]
    assert e["1_holds"] and e["3_holds"] and e["4_holds"] and e["5_holds"]
    assert not e["2_holds"]                                            # checks that change nothing do not score above the route
    # at 5 % noise the checks may withhold up to a quarter of the right laws; up to 1 % no more than one in twenty
    noisy = [dict(r, second=dict(OUT["silent"])) if r["world"]["noise"] == 0.05 and k % 4 == 0 else r for k, r in enumerate(handed)]
    e = run.summarise(noisy, searched)["expectations"]
    assert e["3_right_laws_of_the_route_kept_handed"]["at_5_percent"] == [15, 20] and e["3_holds"]
    quiet = [dict(r, second=dict(OUT["silent"])) if r["world"]["noise"] == 0.01 and k % 10 == 0 else r for k, r in enumerate(handed)]
    e = run.summarise(quiet, searched)["expectations"]
    assert e["3_right_laws_of_the_route_kept_handed"]["up_to_1_percent"] == [38, 40] and e["3_holds"]
    quiet = [dict(r, second=dict(OUT["silent"])) if r["world"]["noise"] == 0.01 and k % 5 == 0 else r for k, r in enumerate(handed)]
    assert not run.summarise(quiet, searched)["expectations"]["3_holds"]
    # wrong laws the second version lets through at one noise level: the first fails whatever the others show
    bad = [*handed, *[_handed(0.05, False, route="wrong", second="wrong") for _ in range(2)]]      # 2 of 22 reports
    assert not run.summarise(bad, searched)["expectations"]["1_holds"]
    # the checks that stop a wrong law of the route score above it
    good = [*handed, _handed(0.05, False, route="wrong", first="silent", second="silent")]
    e = run.summarise(good, searched)["expectations"]
    assert e["2_holds"] and e["2_scores"]["second"] > e["2_scores"]["route"]
    # behind the searches: losing laws at 1 % noise breaks the fourth, inaccurate reports the fifth
    lost = [dict(r, second=dict(OUT["silent"])) if r["noise"] == 0.0 and k % 5 == 0 else r for k, r in enumerate(searched)]
    assert not run.summarise(handed, lost)["expectations"]["4_holds"]
    inaccurate = [*searched, *[_searched(0.05, route="wrong") for _ in range(4)]]
    assert not run.summarise(handed, inaccurate)["expectations"]["5_holds"]


def test_the_laws_are_new(run) -> None:
    grammar = _load("build_invented_laws")
    made = run.record()
    handed, searched = made["handed"]["laws"], made["searched"]
    assert len(handed) == run.HANDED_COUNT and len(searched) == run.SEARCHED_COUNT
    assert len(made["handed"]["cases"]) == run.HANDED_COUNT * len(run.NOISE_LEVELS) * 2
    structures = [grammar.structure(sp.sympify(law["law"])) for law in handed] + \
                 [grammar.structure(sp.sympify(law["text"])) for law in searched]
    assert len(set(structures)) == len(structures) and not set(structures) & run.used()
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    assert not {grammar.fingerprint(law["law"]) for law in handed} & in_benchmark
    assert not {grammar.fingerprint(law["text"]) for law in searched} & in_benchmark
    ids = {law["id"] for law in handed}
    assert all(case["law_id"] in ids and case["case"].startswith(f"{case['law_id']}-") for case in made["handed"]["cases"])
