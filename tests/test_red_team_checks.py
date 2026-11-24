"""The second red-team campaign: what is attacked, what the climb pushes up, and that its laws are used nowhere else."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import vouching
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
def second():
    return _load("run_red_team_checks")


def _row(route: str, after: str, **extra) -> dict:
    """A decided world: ``route`` and ``after`` are ``right``, ``wrong``, ``close`` or ``silent``."""
    outcome = {"right": {"reported": True, "symbolic": True, "wrong": False, "badness": 0.0, "law": "x0", "test_r2": 1.0},
               "wrong": {"reported": True, "symbolic": False, "wrong": True, "badness": 0.02, "law": "x0**2", "test_r2": 0.98},
               "close": {"reported": True, "symbolic": False, "wrong": False, "badness": 0.0005, "law": "x0 + 0.01", "test_r2": 0.9995},
               "silent": {"reported": False, "symbolic": False, "wrong": False, "badness": 0.0, "law": None, "test_r2": None}}
    vouched = {**outcome[after], "rule": "passed" if after != "silent" else "structure", **extra}
    return {"status": "ok", "arena": "measured", "stage": "survey",
            "world": {"noise": 0.01, "sampling": "uniform", "noise_shape": "gaussian", "rows": 1000, "truth_listed": True,
                      "candidates": ["x0**2", "x0"]},
            "route": {**outcome[route], "action": "equivalent_class", "found_by": "first_pass"}, "vouched": vouched,
            "best_fit": dict(outcome["right"]), "validated": dict(outcome["right"])}


def test_every_reason_the_checks_give_has_a_rule(second) -> None:
    assert second.rule("") == "passed"
    assert second.rule("another member of the class is reported: the first left structure") == "passed"
    assert second.rule("every candidate left structure in its residuals") == "structure"
    assert second.rule("3 candidates passed every check and differ by more than the accuracy claimed") == "class"
    assert second.rule("an error large enough to make it wrong would have been seen 34% of the time, not 80%") == "power"
    assert second.rule("at this noise and with these rows its constants cannot be fitted to the accuracy claimed") == "fit"
    assert second.rule("the route reported nothing") == "silent"
    assert second.rule("the law is not finite on the inputs at hand") == "other"


def test_the_climb_is_pushed_towards_a_wrong_law_that_the_checks_pass(second) -> None:
    passed = second.nearness(_row("wrong", "wrong"))
    by_power = second.nearness(_row("wrong", "silent", rule="power", power=0.6))
    nearly = second.nearness(_row("wrong", "silent", rule="power", power=0.79))
    by_class = second.nearness(_row("wrong", "silent", rule="class"))
    by_structure = second.nearness(_row("wrong", "silent", rule="structure", p_value=0.004))
    far = second.nearness(_row("wrong", "silent", rule="structure", p_value=0.0005))
    assert passed > 2.0 > nearly > by_power > by_class == 1.0 > by_structure > far > 0.0
    assert passed == pytest.approx(2.02) and by_structure == pytest.approx(0.004 / vouching.LEVEL)
    # nothing is gained by worlds where the route was not wrong, or where a wrong law became a right one
    for route, after in (("right", "right"), ("close", "close"), ("silent", "silent"), ("right", "silent"), ("wrong", "right")):
        assert second.nearness(_row(route, after)) == 0.0
    assert second.nearness({"status": "timeout"}) == 0.0
    # a law that was not wrong before the checks and is wrong after them counts in full
    assert second.nearness(_row("close", "wrong")) > 2.0


def test_the_decision_attacked_is_the_one_the_benchmark_measured(second) -> None:
    """``checked`` is ``run_counterfeit_benchmark.vouched_row`` with two fields added, on a law that is in no benchmark."""
    bench = _load("run_counterfeit_benchmark")
    world = worlds.World.from_dict(bench.smoke_cases()[2]["world"])                    # 1 % noise, the law offered
    far = "0.52*x0 + 2.2"
    for route in ({"reported": True, "law": world.law, "equivalent": []},
                  {"reported": True, "law": far, "equivalent": [world.law]},
                  {"reported": True, "law": far, "equivalent": []},
                  {"reported": False, "law": None, "equivalent": []}):
        mine, theirs = second.checked(world, route), bench.vouched_row(world, route)
        assert {key: mine[key] for key in theirs} == theirs
        assert mine["rule"] == second.rule(mine["reason"])
    assert second.checked(world, {"reported": True, "law": far, "equivalent": []})["p_value"] < vouching.LEVEL


def test_the_first_campaigns_machinery_runs_with_this_campaigns_decision_and_is_put_back(second, monkeypatch) -> None:
    campaign = _load("run_red_team")
    before = (campaign.LAWS_FILE, campaign.LAW_SEED, campaign.decide, campaign.badness)
    with second.aimed_at_the_checks():
        assert campaign.LAWS_FILE == second.LAWS_FILE and campaign.LAW_SEED == second.LAW_SEED
        assert campaign.decide is second.decide and campaign.badness is second.nearness
        assert len(campaign.laws()) == second.LAW_COUNT
        with pytest.raises(RuntimeError):
            second.taken()                                    # the first campaign's laws cannot be read from here
    assert (campaign.LAWS_FILE, campaign.LAW_SEED, campaign.decide, campaign.badness) == before
    # the decision: the route's own, then the checks; checks that crash vouch for nothing
    world = worlds.World(law="0.8*x0 + 1.9/(x0 + 0.6)", inputs=1, candidates=("0.52*x0 + 2.2",), rows=300, noise=0.01, seed=5)
    monkeypatch.setattr(second, "_ROUTE_DECIDES", lambda w: {"route": {"reported": True, "law": "0.52*x0 + 2.2", "equivalent": [],
                                                                       "wrong": True, "symbolic": False}})
    out = second.decide(world)
    assert out["vouched"]["reported"] is False and out["vouched"]["rule"] == "structure"

    def broken(*args, **kwargs):
        raise ZeroDivisionError("no rows")

    monkeypatch.setattr(vouching, "vouch", broken)
    out = second.decide(world)
    assert out["vouched"]["failed"] and out["vouched"]["reported"] is False and "ZeroDivisionError" in out["vouched"]["reason"]


def test_the_laws_are_new_and_used_nowhere_else(second) -> None:
    grammar = _load("build_invented_laws")
    campaign = _load("run_red_team")
    record = json.loads(second.LAWS_FILE.read_text(encoding="utf-8"))
    assert record["seed"] == second.LAW_SEED and len(record["laws"]) == second.LAW_COUNT
    assert [law["id"] for law in record["laws"]] == list(range(second.FIRST_ID, second.FIRST_ID + second.LAW_COUNT))
    structures = {grammar.structure(sp.sympify(law["law"])) for law in record["laws"]}
    assert len(structures) == second.LAW_COUNT and not structures & second.taken()
    first = {grammar.structure(sp.sympify(law["law"])) for law in json.loads(campaign.LAWS_FILE.read_text(encoding="utf-8"))["laws"]}
    assert not structures & first
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    assert not {grammar.fingerprint(law["law"]) for law in record["laws"]} & in_benchmark
    assert all(law["counterfeits"] for law in record["laws"])


def test_the_summary_counts_what_the_checks_did(second) -> None:
    rows = [_row("right", "right"), _row("right", "silent", rule="power", power=0.5), _row("close", "right"),
            _row("wrong", "silent", rule="structure", p_value=0.001), _row("wrong", "silent", rule="class"),
            _row("wrong", "wrong"), _row("wrong", "right"), _row("silent", "silent", rule="silent"),
            {"status": "timeout", "arena": "measured", "stage": "survey", "world": _row("right", "right")["world"]}]
    climbed = [dict(_row("wrong", "wrong"), stage="climb", arena="noise_level")]
    summary = second.summarise([*rows, *climbed])
    checks = summary["checks"]["survey"]["measured"]
    assert checks["wrong_laws_of_the_route"] == 4 and checks["of_them_passed_by_the_checks"] == 1
    assert checks["of_them_stopped_by"] == {"structure": 1, "class": 1} and checks["of_them_turned_into_the_law_or_an_accurate_one"] == 1
    assert checks["right_laws_of_the_route"] == 2 and checks["of_them_kept"] == 1 and checks["right_laws_lost_to"] == {"power": 1}
    assert checks["reports_turned_into_the_law"] == 2 and checks["wrong_laws_after_the_checks"] == 1
    assert summary["survey"]["measured"]["vouched"]["wrong"] == 1 and summary["survey"]["measured"]["route"]["wrong"] == 4
    assert summary["survey"]["measured"]["route"]["timeouts"] == 1
    passed = summary["wrong_laws_the_checks_passed"]
    assert passed["count"] == 2 and passed["by_stage_and_arena"]["climb|noise_level"] == 1
    assert passed["by_source"] == {"first_pass": 2} and passed["one_of_the_candidates_handed_over"] == 2
    assert passed["worst"]["noise_level"][0]["handed_to_the_route"] is True
    assert summary["survey_by_noise"]["0.3 to 1 %"]["checks"]["wrong_laws_of_the_route"] == 4
