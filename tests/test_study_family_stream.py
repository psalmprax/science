"""The development study of the family route: what its families are, how its arms are scored and counted."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import study_family_stream as study

from apedp_dyn.discovery_l2 import across_tables as at
from apedp_dyn.discovery_l2 import family_route
from apedp_dyn.ode.extended_route import replayed
from apedp_dyn.recursion import curriculum


@pytest.fixture(scope="module")
def families() -> list[dict[str, Any]]:
    return study.draw_families()


def test_the_families_are_the_same_every_time_and_come_in_pairs_with_the_same_inputs(families: list[dict[str, Any]]) -> None:
    assert families == study.draw_families() and len(families) == study.FAMILIES
    held_back = curriculum.benchmark_fingerprints()
    for first, second in zip(families[::2], families[1::2], strict=True):
        assert first["inputs"] == second["inputs"] and first["level"] == second["level"]
        assert first["writable"] is second["writable"]
    assert [f["writable"] for f in families[:4]] == [True, True, False, False]
    assert sum(f["writable"] for f in families) == study.FAMILIES // 2
    assert len({curriculum.structure(sp.sympify(f["shape"])) for f in families}) == study.FAMILIES
    assert not {curriculum.fingerprint(sp.sympify(f["shape"])) for f in families} & held_back
    assert len(study.units(families)) == (study.FAMILIES // 2) * len(study.NOISES)


def test_a_member_s_law_is_the_wide_law_at_its_value_and_the_members_differ(families: list[dict[str, Any]]) -> None:
    c = sp.Symbol(study.CONDITION)
    for spec in families:
        wide = sp.sympify(spec["wide"])
        assert c in wide.free_symbols and len(spec["values"]) == study.MEMBERS == len(set(spec["values"]))
        forms = study.WRITABLE if spec["writable"] else study.NOT_WRITABLE
        assert 1 <= len(spec["moving"]) <= 2 and set(spec["moving"].values()) <= set(forms)
        X = np.random.default_rng(0).uniform(1.0, 5.0, (50, spec["inputs"]))
        symbols = [sp.Symbol(f"x{i}") for i in range(spec["inputs"])]
        seen = []
        for value, law in zip(spec["values"], spec["laws"], strict=True):
            member = np.asarray(sp.lambdify(symbols, sp.sympify(law), "numpy")(*X.T), dtype=float)
            at_value = np.asarray(sp.lambdify(symbols, wide.subs(c, value), "numpy")(*X.T), dtype=float)
            assert np.allclose(member, at_value)
            seen.append(member)
        assert float(np.sqrt(np.mean((seen[0] - seen[-1]) ** 2))) >= study.MOVES_ENOUGH * float(np.sqrt(np.mean(seen[0] ** 2)))


def test_a_constant_is_made_to_move_at_its_own_place() -> None:
    x = sp.Symbol("x0")
    c = sp.Symbol("c")
    moved = study._with_moving(sp.sympify("0.7*x0 + 0.7/x0"), {1: 2 * c})
    assert {sp.simplify(moved - (0.7 * x + 1.4 * c / x)), sp.simplify(moved - (1.4 * c * x + 0.7 / x))} & {0}
    assert len(study._floats(sp.sympify("0.7*x0 + 0.7/x0"))) == 2


def test_a_family_s_tables_share_one_noise_and_keep_rows_without_noise_to_score_on(families: list[dict[str, Any]]) -> None:
    spec = families[0]
    family, scoring = study.tables(spec, 0.05)
    again, _ = study.tables(spec, 0.05)
    assert [t.conditions[study.CONDITION] for t in family.tables] == spec["values"]
    sigmas = []
    for table, how in zip(family.tables, scoring, strict=True):
        assert table.X.shape == (study.ROWS, spec["inputs"]) and table.pool_X.shape == (study.POOL, spec["inputs"])
        assert how["test_X"].shape == (study.TEST_ROWS, spec["inputs"])
        law = sp.lambdify([sp.Symbol(n) for n in table.names], how["truth"], "numpy")
        assert np.allclose(how["test_clean"], law(*how["test_X"].T))
        sigmas.append(float(np.std(table.y - law(*table.X.T))))
        measured = table.measure(np.array([0, 5]))
        assert not np.allclose(measured, law(*table.pool_X[[0, 5]].T))            # the pool is measured with its noise
    assert max(sigmas) < 1.25 * min(sigmas)                                          # one noise for the whole family
    assert np.array_equal(family.tables[2].y, again.tables[2].y)
    quiet, _ = study.tables(spec, 0.01)
    assert float(np.std(quiet.tables[0].y - family.tables[0].y)) > 0


def test_a_law_is_scored_for_accuracy_and_for_form() -> None:
    names = ["x0", "x1"]
    truth = sp.sympify("1.3*x0**2/(x1 + 0.7)")
    X = np.random.default_rng(0).uniform(1.0, 5.0, (200, 2))
    clean = 1.3 * X[:, 0] ** 2 / (X[:, 1] + 0.7)
    assert study.score("1.3001*x0**2/(x1 + 0.7002)", truth, names, X, clean) == {"accurate": True, "recovered": True}
    assert study.score("0.9*x0**2 - 0.3*x1", truth, names, X, clean) == {"accurate": False, "recovered": False}
    assert study.score(None, truth, names, X, clean) == {"accurate": False, "recovered": False}
    assert study.score("not a law (", truth, names, X, clean) == {"accurate": False, "recovered": False}


def test_the_sign_test_is_two_sided_and_exact() -> None:
    assert study.sign_test(0, 0) == 1.0 and study.sign_test(3, 3) == 1.0
    assert study.sign_test(5, 0) == pytest.approx(2 / 32) and study.sign_test(0, 6) == pytest.approx(2 / 64)
    assert study.sign_test(8, 1) == pytest.approx(2 * (1 + 9) / 512)


def test_the_store_holds_other_laws_of_one_and_two_inputs() -> None:
    store = study.store(12)
    assert len(store.cases) == 12 and {case.inputs for case in store.cases} <= {1, 2}
    assert [case.law for case in store.cases] == [case.law for case in study.store(12).cases]


def test_every_arm_of_a_unit_is_counted_from_the_same_readings(families: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in for the route that reports a law exactly when one of the true shape (or the true wide law) is handed over."""
    pair = families[2:4]                                           # two inputs, laws of the condition it cannot write
    keys = {}
    for spec in pair:
        names = [f"x{i}" for i in range(spec["inputs"])]
        keys[spec["family"]] = {at.shape_of(law, names).key for law in spec["laws"]}

    def read(table: at.Table, proposer: Any, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
        formulas: list[str] = []
        if proposer is not None:
            frame = pd.DataFrame({**table.columns(), "__target__": table.y})
            formulas = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col="__target__")]
        family = int(table.key.split(",")[0].split("|")[0].removeprefix("family "))
        right = [f for f in formulas if study.CONDITION not in table.names and at.shape_of(f, table.names).key in keys[family]]
        law = right[0] if right else None
        return {"record": {"vouching": {}, "measured_range": {}}, "reported": law is not None, "law": law, "tentative_law": None,
                "action": "confirmed" if law else "widen_search", "formulas": formulas, "listed": list(formulas),
                "rows_read": 30, "seconds": 0.0}

    monkeypatch.setattr(family_route, "read", read)

    def stage(table: at.Table) -> Any:
        # each family's search finds the shape on its first member only; on the pooled table it finds nothing
        if study.CONDITION in table.names:
            return replayed(["0.5*x0 - 0.2*x1 + 0.1*c"])
        family = int(table.key.split(",")[0].removeprefix("family "))
        first = min(pair[[s["family"] for s in pair].index(family)]["values"])
        return replayed([pair[[s["family"] for s in pair].index(family)]["laws"][0] if table.conditions[study.CONDITION] == first
                         else "0.5*x0 - 0.2*x1"])

    got = study.run_unit(pair, 0.01, 0.0, seed=1, stage=stage, cases=study.store(6))
    members, fams = got["members"], got["families"]
    assert len(members) == 2 * study.MEMBERS and len(fams) == 2
    assert sum(m["A"]["reported"] for m in members) == 2            # the route alone: the one member a family
    assert sum(m["B"]["reported"] for m in members) == 10           # with its siblings: every member
    assert all(m["B"]["accurate"] and m["B"]["recovered"] for m in members)
    assert {m["source"] for m in members} == {"route", "siblings"} and all(m["fallbacks"] == [] for m in members)
    assert [m["F"] for m in members] == [m["B"] for m in members]   # nothing was left for a fallback
    assert sum(m["R"]["reported"] for m in members) == 2            # the wrong family hands shapes, and none is the law
    assert all(m["R"]["handed"] >= 1 and m["R"]["readings"] == 1 for m in members if not m["A"]["reported"])
    assert all(not f["W"]["reported"] and f["readings"] >= 9 for f in fams)
    summary = study.summarise(members, fams)["noise 0.01"]
    assert summary["members"] == 10 and summary["families"] == 2
    assert summary["B"]["against_A"]["reported"] == {"gained": 8, "lost": 0, "families_with_more": 2, "families_with_fewer": 0,
                                                     "sign_test_p_over_families": 0.5}
    assert summary["R"]["against_A"]["reported"]["gained"] == 0 and summary["B_against_R"]["reported"]["gained"] == 8
    assert summary["sources"] == {"route": 2, "siblings": 8}
    assert summary["fallbacks"]["solved cases"] == {"tried": 0, "led_to_a_reported_law": 0, "of_those_accurate": 0}
    assert summary["wider_W"]["reported"] == 0 and summary["wider_B_not_writable"]["families"] == 2


def test_a_unit_that_fails_is_written_down_and_the_rest_still_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run_unit(pair: Any, noise: float, searches: float, **_: Any) -> dict[str, Any]:
        calls.append(noise)
        if noise == study.NOISES[0]:
            raise RuntimeError("the search could not start")
        return {"members": [], "families": []}

    monkeypatch.setattr(study, "run_unit", run_unit)
    monkeypatch.setattr(study, "store", lambda *a, **k: None)
    assert study.main(["run", "--out", str(tmp_path), "--limit", "3", "--searches", "0"]) == 1          # the first pair at each noise
    assert calls == list(study.NOISES)
    assert len(list(tmp_path.glob("unit_*.json"))) == 2 and len(list(tmp_path.glob("failed_*.json"))) == 1
    assert study.main(["collect", "--out", str(tmp_path)]) == 0
    record = __import__("json").loads((tmp_path / "family_stream.json").read_text())
    assert record["complete"] is False and record["units_found"] == 2 and record["units_expected"] == 24
    assert record["units_failed"][0]["noise"] == study.NOISES[0] and "could not start" in record["units_failed"][0]["error"]
    assert record["development_not_a_result"] is True
