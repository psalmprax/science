"""PREREG-RANKED-MEMORY: how the run is judged, what its arms share, and that its laws are new."""
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
    return _load("run_ranked_memory_blind")


OUT = {"the law": {"reported": True, "model": "x0", "symbolic": True, "accuracy_solution": True},
       "accurate": {"reported": True, "model": "x0", "symbolic": False, "accuracy_solution": True},
       "wrong": {"reported": True, "model": "x0", "symbolic": False, "accuracy_solution": False},
       "tentative": {"reported": False, "model": None, "symbolic": False, "accuracy_solution": False, "tentative": True},
       "silent": {"reported": False, "model": None, "symbolic": False, "accuracy_solution": False}}


def _unit(number: int, noise: float, in_force: str, ranked: str, **done) -> list[dict]:
    head = {"problem": f"law_{number}", "noise": noise, "status": "ok", "seconds": 1.0}
    return [{**head, "arm": "in force", **OUT[in_force]}, {**head, "arm": "ranked", **OUT[ranked], "ranked": dict(done)}]


def test_the_run_is_the_table_route_at_the_budget_the_memory_was_adopted_at(run) -> None:
    assert run.ARMS == ("in force", "ranked") and run.NOISE_LEVELS == (0.0, 0.01)
    assert run.SEARCH_BUDGET_S == 240.0 and run.LAW_COUNT == 100
    for rel in run.FROZEN:                                # every frozen file is a file; the laws are drawn at freezing
        if rel != "benchmarks/ranked_memory_blind_laws.json":
            assert (ROOT / rel).is_file(), rel
    assert "benchmarks/order_score/learned_order_model.json" in run.FROZEN
    for module in ("ranked_stage.py", "ranked_memory.py", "learned_order.py", "vouched_route_v2.py", "hard_exit.py"):
        assert any(rel.endswith("/" + module) for rel in run.FROZEN)


def test_what_counts_as_reported_right_and_wrong(run) -> None:
    rows = _unit(0, 0.0, "the law", "accurate") + _unit(1, 0.0, "wrong", "tentative") + _unit(2, 0.0, "silent", "silent")
    by = {(r["problem"], r["arm"]): r for r in rows}
    assert run._right(by["law_0", "in force"]) and run._recovered(by["law_0", "in force"])
    assert run._right(by["law_0", "ranked"]) and not run._recovered(by["law_0", "ranked"])      # accurate, not the law
    assert run._wrong(by["law_1", "in force"]) and not run._reported(by["law_1", "ranked"])     # a tentative law is not a report
    assert not run._reported(by["law_2", "in force"])
    crashed = {"problem": "law_3", "noise": 0.0, "arm": "ranked", "status": "timeout", "seconds": 1800.0}
    assert not run._reported(crashed) and not run._right(crashed) and not run._wrong(crashed)


def test_the_expectations_are_judged_as_written(run) -> None:
    rows: list[dict] = []
    for noise in run.NOISE_LEVELS:
        for number in range(30):                          # both arms report the law on thirty laws at each noise level
            rows += _unit(number, noise, "the law", "the law")
        for number in range(30, 34):                      # the ranked arm reports four more, two of them the law itself
            rows += _unit(number, noise, "silent", "the law" if number % 2 else "accurate", added=["c*x0"],
                          the_law_among_the_added=bool(number % 2))
        for number in range(34, 50):
            rows += _unit(number, noise, "silent", "silent")
    summary = run.summarise(rows)
    e = summary["expectations"]
    assert e["1_holds"] and e["2_holds"] and e["3_holds"]
    assert e["2_right_ranked"] == {"0.0": 34, "0.01": 34, "all": 68} and e["2_right_in_force"]["all"] == 60
    assert e["3_recovered_ranked"] == 64 and e["3_recovered_in_force"] == 60
    assert summary["run_for_run"]["right|all"] == {"gained": 8, "lost": 0, "sign_test_p": pytest.approx(0.0078125)}
    assert summary["ranked|all"]["ranked_added_the_law"] == 4 and summary["ranked|all"]["ranked_added_a_candidate"] == 8

    # as many right laws in total is not more: the second expectation misses
    level = [r for r in rows if not (r["arm"] == "ranked" and r["problem"] in {f"law_{n}" for n in range(30, 34)})]
    level += [dict(r, **OUT["silent"]) for r in rows if r["arm"] == "ranked" and r["problem"] in {f"law_{n}" for n in range(30, 34)}]
    assert not run.summarise(level)["expectations"]["2_holds"]

    # fewer at one noise level misses it even with more in total
    uneven = [dict(r, **OUT["silent"]) if r["arm"] == "ranked" and r["noise"] == 0.01 and r["problem"] in ("law_0", "law_1", "law_2",
              "law_3", "law_4") else r for r in rows]
    e = run.summarise(uneven)["expectations"]
    assert e["2_right_ranked"]["all"] == 63 and not e["2_holds"]

    # the law itself traded for an accurate law: more right laws, fewer recovered, and the third expectation misses
    traded = [dict(r, **OUT["accurate"]) if r["arm"] == "ranked" and r["problem"] in {f"law_{n}" for n in range(0, 6)} else r
              for r in rows]
    e = run.summarise(traded)["expectations"]
    assert e["2_holds"] and not e["3_holds"] and e["3_recovered_ranked"] == 52

    # wrong reports: four of sixty-eight is above one in twenty
    spoiled = [dict(r, **OUT["wrong"]) if r["arm"] == "ranked" and r["noise"] == 0.0 and r["problem"] in ("law_30", "law_31", "law_32",
               "law_33") else r for r in rows]
    e = run.summarise(spoiled)["expectations"]
    assert e["1_share_of_reported_laws_wrong"]["ranked"]["verdict"] == "missed" and not e["1_holds"]
    assert e["1_share_of_reported_laws_wrong"]["in force"]["verdict"] == "held"

    # too few reports to show the share is small: not shown is not held
    few = [r for r in rows if r["problem"] in {f"law_{n}" for n in range(0, 10)} and r["noise"] == 0.0]
    e = run.summarise(few)["expectations"]
    assert e["1_share_of_reported_laws_wrong"]["ranked"]["verdict"] == "not shown" and not e["1_holds"]


def test_a_unit_that_crashes_counts_for_nothing_in_either_arm(run) -> None:
    rows = _unit(0, 0.0, "the law", "the law")
    rows += [{"problem": "law_1", "noise": 0.0, "arm": arm, "status": "error", "seconds": 5.0, "error": "x"} for arm in run.ARMS]
    summary = run.summarise(rows)
    assert summary["ranked|all"]["errors"] == 1 and summary["ranked|all"]["reported"] == 1 and summary["in force|all"]["runs"] == 2


def test_the_laws_of_the_run_are_new_to_every_earlier_set(run) -> None:
    grammar = _load("build_invented_laws")
    record = run.build_laws(count=8, seed=run.LAW_SEED)
    met = run._structures_seen_before()
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    assert len(record["laws"]) == 8 and len({law["text"] for law in record["laws"]}) == 8
    for law in record["laws"]:
        assert grammar.structure(sp.sympify(law["text"])) not in met and grammar.fingerprint(law["text"]) not in in_benchmark
        assert 1 <= law["inputs"] <= 3
    memory_laws = json.loads((ROOT / "benchmarks" / "memory_blind_laws.json").read_text(encoding="utf-8"))["laws"]
    assert all(grammar.structure(sp.sympify(law["text"])) in met for law in memory_laws)           # the memory's own test laws
    rehearsal = run.rehearsal_entries(4)
    assert all(grammar.structure(sp.sympify(entry["formula"])) in met for entry in rehearsal.values())  # development laws only
    if run.LAWS_FILE.exists():                            # once the plan is frozen: the file is what the generator gives
        assert json.loads(run.LAWS_FILE.read_text(encoding="utf-8")) == run.build_laws()
