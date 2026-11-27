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
    more = study.draw_families(2 * study.FAMILIES)                  # the second run's families follow the first run's
    assert more[: study.FAMILIES] == families and [f["family"] for f in more] == list(range(2 * study.FAMILIES))
    assert len({curriculum.structure(sp.sympify(f["shape"])) for f in more}) == 2 * study.FAMILIES
    assert not {curriculum.fingerprint(sp.sympify(f["shape"])) for f in more} & held_back


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


def _stand_in(pair: list[dict[str, Any]]) -> Any:
    """In place of the route. It reports a law of the family's true shape whenever one is among the laws it is given.
    Given none, it does what the first run showed the route doing: if the laws at the head of the list are not the
    table's own (the first build's way), it reports the table's own first law, which it had refused before."""
    shapes, wide = {}, {}
    for spec in pair:
        names = [f"x{i}" for i in range(spec["inputs"])]
        shapes[spec["family"]] = {at.shape_of(law, names).key for law in spec["laws"]}
        wide[spec["family"]] = at.shape_of(spec["wide"], [*names, study.CONDITION], 60).key
    own: dict[str, list[str]] = {}

    def read(table: at.Table, proposer: Any, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
        formulas: list[str] = []
        if proposer is not None:
            frame = pd.DataFrame({**table.columns(), "__target__": table.y})
            formulas = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col="__target__")]
        family = int(table.key.split(",")[0].split("|")[0].removeprefix("family "))
        if study.CONDITION in table.names:                         # the pooled table: the wide law or nothing
            right = [f for f in formulas if (shape := at.shape_of(f, table.names, 60)) is not None and shape.key == wide[family]]
            law = right[0] if right else None
        else:
            mine = own.setdefault(table.key, list(formulas))
            right = [f for f in formulas if (shape := at.shape_of(f, table.names)) is not None and shape.key in shapes[family]]
            law = right[0] if right else mine[0] if formulas and formulas[0] not in mine else None
        return {"record": {"vouching": {}, "measured_range": {}}, "reported": law is not None, "law": law, "confirmed_as": law,
                "tentative_law": None, "action": "confirmed" if law else "widen_search", "formulas": formulas,
                "listed": list(formulas), "rows_read": 30, "seconds": 0.0}

    return read


def test_every_arm_of_a_unit_is_counted_from_the_same_readings(families: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    pair = families[2:4]                                           # two inputs, laws of the condition it cannot write
    monkeypatch.setattr(family_route, "read", _stand_in(pair))
    by_family = {spec["family"]: spec for spec in pair}

    def stage(table: at.Table) -> Any:
        # each family's search finds the shape on its first member only
        spec = by_family[int(table.key.split(",")[0].removeprefix("family "))]
        return replayed([spec["laws"][0] if table.conditions[study.CONDITION] == min(spec["values"]) else "0.5*x0 - 0.2*x1"])

    def pooled_stage(table: at.Table) -> Any:
        # the pooled table's search finds the wide law of the first family and misses the second's
        spec = by_family[int(table.key.split("|")[0].removeprefix("family "))]
        return replayed([spec["wide"] if spec is pair[0] else "0.5*x0 - 0.2*x1 + 0.1*c"])

    got = study.run_unit(pair, 0.01, 0.0, seed=1, stage=stage, pooled_stage=pooled_stage, cases=study.store(6))
    members, fams = got["members"], got["families"]
    assert len(members) == 2 * study.MEMBERS and len(fams) == 2
    count = lambda arm, key="reported": sum(bool(m[arm][key]) for m in members)
    assert count("A") == 2                                          # the route alone: the one member a family
    assert count("B") == 10 and count("B", "recovered") == 10       # with its siblings, beside its own law: every member
    assert {m["source"] for m in members} == {"route", "siblings"} and all(m["fallbacks"] == [] for m in members)
    assert [m["F"] for m in members] == [m["B"] for m in members]   # nothing was left for a fallback
    assert count("R") == 2 and count("O") == 2                      # the wrong family brings nothing, nor does reading again
    missed = [m for m in members if not m["A"]["reported"]]
    assert all(m["R"]["handed"] >= 1 and m["R"]["passed_over"] == 0 for m in missed)
    # the control is also handed the wrong family's wider law where that family has one: a second reading
    assert {m["family"]: m["R"]["readings"] for m in missed} == {pair[0]["family"]: 1, pair[1]["family"]: 2}
    # the first build's way: with its own family the handed law is reported; with the wrong family at the head of the
    # list the referee reports the member's own law, which is not the law
    assert count("B1") == 10 and all(m["B1"]["origin"] == "handed" for m in missed)
    assert count("R1") == 10 and count("R1", "accurate") == 2 and all(m["R1"]["origin"] == "own" for m in missed)
    assert all(m["where"]["on_a_sibling_s_lists"]["law_itself"] and m["where"]["among_the_first_three_handed"]["law_itself"]
               and not m["where"]["among_its_referee_s_candidates"]["law_itself"] for m in missed)
    wider = [f["wider"] for f in fams]
    assert [w["searched"]["reported"] for w in wider] == [True, False] and [w["final"]["reported"] for w in wider] == [True, False]
    assert wider[0]["source"] == "pooled search" and wider[0]["final"]["recovered"] and wider[0]["final"]["accurate_between"]
    assert not any(w["alone"]["reported"] for w in wider)           # a sine of the condition is not a law it can write
    summary = study.summarise(members, fams)
    assert set(summary) == {"all noise levels", "noise 0.01"} and summary["all noise levels"] == summary["noise 0.01"]
    cell = summary["noise 0.01"]
    assert cell["members"] == 10 and cell["families"] == 2
    assert cell["B"]["against_A"]["reported"] == {"gained": 8, "lost": 0, "families_with_more": 2, "families_with_fewer": 0,
                                                  "sign_test_p_over_families": 0.5}
    assert cell["R"]["against_A"]["reported"]["gained"] == 0 and cell["B_against_R"]["reported"]["gained"] == 8
    assert cell["R1"]["reported_and_not_accurate"] == 8 and cell["B1_against_R1"]["reported"]["gained"] == 0
    assert cell["first_build_origin"] == {"B1": {"handed": 8, "own": 0}, "R1": {"handed": 0, "own": 8}}
    assert cell["sources"] == {"route": 2, "siblings": 8} and cell["passed_over"] == {"siblings": 0, "wider law": 0, "control": 0}
    assert cell["fallbacks"]["solved cases"] == {"tried": 0, "its_own_law_reported": 0, "of_those_accurate": 0,
                                                 "of_those_where_the_shape_was_a_solved_case": 0, "of_those_the_law_itself": 0,
                                                 "another_law_reported_and_passed_over": 0}
    assert not any(m["shape_among_the_solved_cases"] for m in members)
    known = study.store()                                          # the full store holds the shape of two later families
    solved = {curriculum.structure(sp.sympify(case.law)) for case in known.cases}
    assert [f["family"] for f in study.draw_families(32) if curriculum.structure(sp.sympify(f["shape"])) in solved] == [16, 24]
    lost = cell["members_without_a_first_report"]
    assert lost["count"] == 8 and lost["first_action"] == {"widen_search": 8} and lost["handed_at_least_one_law"] == 8
    assert lost["the_law_itself_was"] == {"among_its_referee_s_candidates": 0, "on_its_own_search_s_list": 0,
                                          "on_a_sibling_s_lists": 8, "among_the_laws_handed": 8, "among_the_first_three_handed": 8}
    assert cell["wider"]["searched"]["reported"] == 1 and cell["wider"]["final"]["reported"] == 1
    assert cell["wider"]["final_source"] == {"None": 1, "pooled search": 1}
    assert cell["wider"]["final_and_searched"] == {"both": 1, "only_final": 0, "only_searched": 0}
    assert cell["wider"]["not_writable"] == {"families": 2, "alone": 0, "searched": 1, "final": 1}
    nothing = [{**m, "source": None} for m in members[:3]]
    assert study.summarise(nothing, fams)["noise 0.01"]["sources"] == {"None": 3}      # a member with no law is counted too


def test_a_unit_that_fails_is_written_down_and_the_rest_still_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run_unit(pair: Any, noise: float, searches: float, **_: Any) -> dict[str, Any]:
        calls.append((pair[0]["family"], noise))
        if noise == study.NOISES[0]:
            raise RuntimeError("the search could not start")
        return {"members": [], "families": [{"family": pair[0]["family"]}, {"family": pair[1]["family"]}]}

    monkeypatch.setattr(study, "run_unit", run_unit)
    monkeypatch.setattr(study, "store", lambda *a, **k: None)
    args = ["--out", str(tmp_path), "--families", "32", "--skip", "16"]      # the second run's families
    assert study.main(["run", *args, "--limit", "3", "--searches", "0"]) == 1          # the first pair at each noise
    assert calls == [(16, noise) for noise in study.NOISES]
    assert sorted(path.name for path in tmp_path.glob("*.json")) == ["failed_16_0.01.json", "unit_16_0.05.json", "unit_16_0.1.json"]
    assert study.main(["collect", *args]) == 0
    record = __import__("json").loads((tmp_path / "family_stream.json").read_text())
    assert record["complete"] is False and record["units_found"] == 2 and record["units_expected"] == 24
    assert record["units_failed"][0]["pair"] == [16, 17] and "could not start" in record["units_failed"][0]["error"]
    assert record["families"] == [{"family": 16}, {"family": 17}] * 2          # each unit's family rows, kept as rows
    assert record["development_not_a_result"] is True and record["settings"]["families_read"] == list(range(16, 32))
    with pytest.raises(SystemExit):
        study.main(["run", "--out", str(tmp_path), "--skip", "3"])           # the families come in pairs


def test_units_read_again_in_another_run_are_merged_and_a_unit_read_twice_is_refused(tmp_path: Path) -> None:
    row = lambda family, noise: {"family": family, "noise": noise}
    one = {"tag": study.TAG, "what": "w", "settings": {"seed": 1}, "units_found": 1, "members": [],
           "families": [row(16, 0.01), row(17, 0.01)], "units_failed": [{"pair": [18, 19], "noise": 0.1, "error": "x"}]}
    two = {"tag": study.TAG, "what": "w", "settings": {"seed": 1}, "units_found": 1, "members": [],
           "families": [row(18, 0.1), row(19, 0.1)], "units_failed": []}
    got = study.merged([one, two], ["a.json", "b.json"], 24)
    assert got["units_found"] == 2 and got["complete"] is False and got["development_not_a_result"] is True
    assert got["merged_from"] == [{"record": "a.json", "units_found": 1, "units_failed": [{"pair": [18, 19], "noise": 0.1}]},
                                  {"record": "b.json", "units_found": 1, "units_failed": []}]
    assert [f["family"] for f in got["families"]] == [16, 17, 18, 19]
    with pytest.raises(ValueError, match="more than one record"):
        study.merged([one, one], ["a.json", "a.json"], 24)
    for name, record in (("a.json", one), ("b.json", two)):
        (tmp_path / name).write_text(__import__("json").dumps(record))
    assert study.main(["merge", "--out", str(tmp_path), "--families", "32", "--skip", "16",
                       "--records", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    assert __import__("json").loads((tmp_path / "family_stream_merged.json").read_text())["units_found"] == 2


def test_the_second_run_s_summary_is_what_its_two_records_give() -> None:
    """Two of the second run's units failed in the route's own fitting step and were read again in a third run. The
    summary kept beside the two records is the merge of their rows, and its headline counts are pinned here."""
    folder = Path(__file__).resolve().parents[1] / "benchmarks" / "family_stream"
    names = ["dev_run2_as_run.json", "dev_run3_two_units.json"]
    records = [__import__("json").loads((folder / name).read_text()) for name in names]
    assert [r["units_found"] for r in records] == [22, 2] and [len(r["units_failed"]) for r in records] == [2, 0]
    got = study.merged(records, names, 24)
    kept = __import__("json").loads((folder / "dev_run2_summary.json").read_text())
    assert got["complete"] and got["units_found"] == 24 and kept["summary"] == got["summary"] and kept["merged_from"] == got["merged_from"]
    cell = got["summary"]["all noise levels"]
    assert cell["members"] == 240 and cell["families"] == 48
    assert {arm: cell[arm]["reported"] for arm in study.ARMS} == {"A": 118, "B": 124, "F": 129, "R": 120, "O": 118, "B1": 138, "R1": 133}
    assert {arm: cell[arm]["reported_and_not_accurate"] for arm in study.ARMS} == {"A": 0, "B": 0, "F": 0, "R": 0, "O": 0, "B1": 0, "R1": 2}
    assert cell["first_build_origin"] == {"B1": {"handed": 10, "own": 10}, "R1": {"handed": 4, "own": 11}}
    assert {way: cell["wider"][way]["reported"] for way in study.WAYS} == {"alone": 12, "searched": 22, "final": 24}
    assert cell["members_without_a_first_report"]["an_accurate_law_was"]["among_its_referee_s_candidates"] == 91
