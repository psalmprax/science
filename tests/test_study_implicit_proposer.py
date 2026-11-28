"""The study of the proposer of polynomial relations as a search stage (development; ``scripts/study_implicit_proposer.py``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_implicit_proposer as study

from apedp_dyn.recursion import curriculum

RECORD = ROOT / "benchmarks" / "outside_proposals" / "dev_implicit_proposer.json"


def _way(*, reported: bool = False, law: bool = False, accurate: bool = False, wider: bool = False, by: str | None = None,
         held: bool = False, held_law: bool = False, action: str = "confirmed") -> dict[str, Any]:
    return {"action": action, "reported": reported, "the_law_itself": law, "accurate": accurate, "accurate_on_wider_rows": wider,
            "found_by": by, "proposed": 0, "held_back": held, "held_back_was_the_law": held_law, "seconds": 1.0}


def test_the_counts_are_of_reported_laws_and_say_who_found_them() -> None:
    rows = [
        {"within_reach": True, "noise": 0.01, "first_pass_alone": _way(held=True, held_law=True),
         "implicit_stage": _way(reported=True, law=True, accurate=True, wider=True, by="search")},
        {"within_reach": False, "noise": 0.01, "first_pass_alone": _way(held=True),
         "implicit_stage": _way(reported=True, accurate=True, by="first_pass")},            # accurate inside the range only
        {"within_reach": False, "noise": 0.1, "first_pass_alone": _way(action="not_run"),
         "implicit_stage": _way(reported=True, by="search")},                               # a wrong law
        {"within_reach": False, "noise": 0.1, "first_pass_alone": _way(), "implicit_stage": _way(held=True)},
    ]
    told = study.summarise(rows, ("within_reach", "noise"))
    assert told["all"]["tables"] == 4
    assert told["all"]["implicit_stage"] == {"reported": 3, "the_law_itself": 1, "accurate": 2, "reported_and_not_accurate": 1,
                                             "accurate_on_wider_rows": 1, "found_by_the_stage": 2, "held_back": 1,
                                             "held_back_was_the_law": 0, "not_run": 0}
    assert told["all"]["first_pass_alone"]["reported"] == 0 and told["all"]["first_pass_alone"]["held_back_was_the_law"] == 1
    assert told["all"]["first_pass_alone"]["not_run"] == 1
    assert told["by_within_reach"]["True"]["tables"] == 1 and told["by_within_reach"]["False"]["implicit_stage"]["reported"] == 2
    assert told["by_noise"]["0.1"]["implicit_stage"]["reported_and_not_accurate"] == 1


def test_a_law_s_table_has_the_stream_s_sizes_and_reach_is_read_from_rows_without_noise() -> None:
    table, _truth, test_X, test_clean, wide_X, wide_clean = study.law_table("saturation (Langmuir)", 0.05, 0)
    assert table.X.shape == (curriculum.ROWS, 1) and table.pool_X.shape == (curriculum.POOL, 1) and len(test_X) == curriculum.TEST_ROWS
    np.testing.assert_allclose(table.measure(np.array([0, 3])), table.measure(np.array([0, 3])))       # the pool answers the same twice
    assert float(wide_X.max()) > float(test_X.max()) and len(wide_clean) == curriculum.TEST_ROWS
    assert study.within_reach(table.names, test_X, test_clean) is True
    trap = study.law_table("trap: exponential", 0.05, 0)
    assert study.within_reach(trap[0].names, trap[2], trap[3]) is False
    again = study.law_table("saturation (Langmuir)", 0.05, 0)[0]
    np.testing.assert_array_equal(again.y, table.y)                                                    # the same table every time


def test_the_record_says_what_its_rows_say_and_is_marked_development() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["tag"] == study.TAG and record["development_not_a_result"] is True
    assert len(record["stream"]["rows"]) == 122 and len(record["laws"]["rows"]) == len(study.LAWS) * len(study.NOISES) * len(study.SEEDS)
    assert record["stream"]["summary"] == study.summarise(record["stream"]["rows"], ("within_reach", "noise"))
    assert record["laws"]["summary"] == study.summarise(record["laws"]["rows"], ("law", "noise"))
    assert "x0" not in json.dumps(record["stream"])                               # no law of the stream is written
    assert all(isinstance(record["settings"]["seconds"][arm], int) for arm in ("stream", "laws", "real"))


def test_what_the_record_is_cited_for() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    stream, laws = record["stream"]["summary"], record["laws"]["summary"]
    # with no search stage the route reports nothing, by its own rule
    assert stream["all"]["first_pass_alone"]["reported"] == 0 and laws["all"]["first_pass_alone"]["reported"] == 0
    told = stream["all"]["implicit_stage"]
    assert (told["reported"], told["the_law_itself"], told["reported_and_not_accurate"], told["found_by_the_stage"]) == (16, 2, 0, 7)
    assert told["accurate_on_wider_rows"] == 14
    reach = stream["by_within_reach"]
    assert reach["True"]["tables"] == 17 and reach["False"]["tables"] == 105
    assert reach["False"]["implicit_stage"]["reported"] == 11 and reach["False"]["implicit_stage"]["the_law_itself"] == 0
    whole = laws["all"]["implicit_stage"]
    assert (whole["reported"], whole["the_law_itself"], whole["found_by_the_stage"], whole["reported_and_not_accurate"]) == (28, 24, 18, 0)
    by_law = laws["by_law"]
    assert [by_law[name]["implicit_stage"]["the_law_itself"] for name in ("Kepler with two masses", "saturation (Langmuir)", "time dilation")] == [4, 2, 3]
    assert [by_law[name]["first_pass_alone"]["held_back_was_the_law"] for name in ("Kepler with two masses", "saturation (Langmuir)", "time dilation")] == [0, 0, 0]
    # the trap: an exponential read as an algebraic law once in six tables, accurate inside the range and not beyond it
    trap = by_law["trap: exponential"]["implicit_stage"]
    assert (trap["reported"], trap["the_law_itself"], trap["accurate"], trap["accurate_on_wider_rows"]) == (1, 0, 1, 0)
    assert by_law["trap: sine"]["implicit_stage"]["reported"] == 0


def test_the_real_tables_as_recorded() -> None:
    real = {t["key"]: t for t in json.loads(RECORD.read_text(encoding="utf-8"))["real"]}
    assert set(real) == set(study.REAL_RELATIONS)
    assert real["kepler_solar_system"]["own_relation_place_among_the_proposals"] == 1
    # Langmuir's first proposal is his form with a constant added, which the benchmark's test of form passes;
    # his own three terms are not among the ten
    assert real["langmuir_table_ix"]["known_form_place_among_the_proposals"] == 1
    assert real["langmuir_table_ix"]["own_relation_place_among_the_proposals"] is None
    assert (real["langmuir_table_ix"]["own_relation_rank"], real["langmuir_table_ix"]["own_relation_rank_among_its_size"]) == (33, 3)
    assert real["leavitt"]["known_form_place_among_the_proposals"] is None and real["leavitt"]["own_relation_rank_among_its_size"] == 6
    for key, table in real.items():
        with_stage, without = table["readings"]["implicit_stage"], table["readings"]["first_pass_alone"]
        assert with_stage["reported"] == without["reported"] == (key == "kepler_solar_system")       # the stage changed no reading
