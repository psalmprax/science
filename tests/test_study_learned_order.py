"""Pins for the study of other orders of the memory's lists: it reads the printed pools and scores orders, nothing else."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("bic", "law_bits", "constants", "small_constants", "large_constants", "operations", "steps", "train_r2", "held_r2",
          "accurate", "solved", "same_form")


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_learned_order", ROOT / "scripts" / "study_learned_order.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pools(path: Path, problems: int = 60, seed: int = 0, prefix: str = "p") -> Path:
    """Lists as the memory hands them: over-fitted long laws first, the short law that would count solved further down."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ("taught-20", "taught-80"):
        for number in range(problems):
            where = int(rng.integers(3, 8))                       # the place of the law, from 0: never among the first 3
            pool = []
            for place in range(8):
                if place == where:
                    pool.append([-900.0 + place, 30.0, 2, 0, 0, 3, 1, 0.99995, 0.99995, 1, 1, int(number % 2 == 0)])
                else:
                    long = 80.0 + float(rng.integers(0, 40))
                    pool.append([-1000.0 + place, long, 6, int(rng.integers(0, 3)), int(rng.integers(0, 2)), 9, 3,
                                 0.999999, 0.9990 + 0.0005 * float(rng.random()), 1, 0, 0])
            if number % 10 == 9:
                pool = [item for index, item in enumerate(pool) if index != where]      # some lists do not hold the law at all
            rows.append([arm, f"{prefix}{number:03d}", 1 + number % 5, 1 + number % 3, 0.01, 40.0, pool])
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"fields": list(FIELDS), "rows": rows}, handle)
    return path


def test_the_pools_are_read_back_as_lists_with_their_places_as_handed(study, tmp_path) -> None:
    arms = study.read(_pools(tmp_path / "pools.json.gz"))
    assert set(arms) == {"taught-20", "taught-80"} and all(len(rows) == 60 for rows in arms.values())
    row = arms["taught-80"][0]
    assert row["solved_place"] is not None and row["solved_place"] > 3 and row["accurate_place"] == 1
    assert isinstance(row["candidates"][0]["solved"], bool) and set(row["candidates"][0]) == set(FIELDS)
    assert arms["taught-80"][9]["solved_place"] is None                                  # a list without the law


def test_an_order_moves_candidates_and_adds_none(study, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study, "HELD_APART", 20)
    monkeypatch.setattr(study, "CURVE", (5, 20, 500))
    monkeypatch.setattr(study, "REPEATS", 3)
    out = tmp_path / "learned.json"
    assert study.main(["--pools", str(_pools(tmp_path / "pools.json.gz")), "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["development"] is True and record["tag"] == study.TAG and record["learning_curve"]["arm"] == "taught-80"
    for arm in ("taught-20", "taught-80"):
        block = record["arms"][arm]
        assert block["as_handed"]["found"] == block["learned"]["found"] == 54            # found cannot change, only the place
        assert block["as_handed"]["within_3"] == 0 and block["learned"]["within_3"] == 54
        assert block["rule"]["tolerance 1"]["within_3"] == 54                            # here the rule is as good: the law is the shortest
        assert block["learned_against_as_handed"]["sign_test_p"] < 0.001
    curve = record["learning_curve"]
    assert curve["held_apart"] == 20 and set(curve["trained_on"]) == {"5", "20"}         # a size larger than the rest is left out
    carried = record["carried_to_other_arms"]
    assert carried["learned_on"] == "taught-80" and carried["problems_trained_on"] == 40
    assert carried["on"]["taught-20"]["learned"]["within_3"] == carried["on"]["taught-20"]["as_handed"]["found"] == 18
    assert carried["model"]["features"] == record["features"]


def test_an_order_learned_on_the_first_set_is_scored_on_a_second_set_that_shares_no_problem(study, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study, "HELD_APART", 20)
    monkeypatch.setattr(study, "CURVE", (5,))
    monkeypatch.setattr(study, "REPEATS", 2)
    first = _pools(tmp_path / "first.json.gz")
    second = _pools(tmp_path / "second.json.gz", problems=30, seed=1, prefix="q")
    out = tmp_path / "learned.json"
    assert study.main(["--pools", str(first), "--fresh", str(second), "--out", str(out)]) == 0
    block = json.loads(out.read_text())["on_the_second_set"]
    for arm in ("taught-20", "taught-80"):
        scored = block[arm]["learned_on"][arm]                    # the order learned on the same arm of the first set
        assert scored["trained_on"]["problems"] == 60 and block[arm]["as_handed"]["problems"] == 30
        assert block[arm]["as_handed"]["within_3"] == 0 and scored["learned"]["within_3"] == block[arm]["as_handed"]["found"] == 27
        assert scored["model"]["features"] == list(study.learned_order.FEATURES)
    with pytest.raises(ValueError, match="not a second set"):
        study.main(["--pools", str(first), "--fresh", str(first), "--out", str(out)])


def test_a_list_built_as_a_proposer_builds_it_is_scored_with_both_orders_that_could_serve_it(study, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study, "HELD_APART", 20)
    monkeypatch.setattr(study, "CURVE", (5,))
    monkeypatch.setattr(study, "REPEATS", 2)

    def renamed(source: Path, target: Path, old: str, new: str, keep: str) -> Path:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            packed = json.load(handle)
        packed["rows"] = [[new if row[0] == old else row[0], *row[1:]] for row in packed["rows"] if row[0] in (old, keep)]
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(packed, handle)
        return target

    first = _pools(tmp_path / "first.json.gz")
    first_split = renamed(first, tmp_path / "first_split.json.gz", "taught-20", "split-80", "none")
    second = renamed(_pools(tmp_path / "second_all.json.gz", problems=30, seed=1, prefix="q"), tmp_path / "second.json.gz",
                     "taught-20", "split-80", "none")
    out = tmp_path / "learned.json"
    assert study.main(["--pools", str(first), str(first_split), "--fresh", str(second), "--out", str(out),
                       "--model-out", str(tmp_path / "model.json"), "--model-arm", "split-80"]) == 0
    record = json.loads(out.read_text())
    assert set(record["arms"]) == {"taught-20", "taught-80", "split-80"} and len(record["pools"]) == 2
    block = record["on_the_second_set"]["split-80"]
    assert set(block["learned_on"]) == {"split-80", "taught-80"}                       # its own first-set lists, and its memory's
    assert all(scored["learned"]["within_3"] == 27 for scored in block["learned_on"].values())
    assert json.loads((tmp_path / "model.json").read_text())["learned_on"]["arm"] == "split-80"


def test_the_order_of_the_reference_arm_is_written_where_a_proposer_can_load_it(study, tmp_path, monkeypatch) -> None:
    from apedp_dyn.proposers import ranked_memory

    monkeypatch.setattr(study, "HELD_APART", 20)
    monkeypatch.setattr(study, "CURVE", (5,))
    monkeypatch.setattr(study, "REPEATS", 2)
    first = _pools(tmp_path / "first.json.gz")
    model = tmp_path / "model.json"
    assert study.main(["--pools", str(first), "--model-out", str(model), "--out", str(tmp_path / "learned.json")]) == 0
    record = json.loads(model.read_text())
    assert record["learned_on"]["arm"] == "taught-80" and record["development"] is True
    order = ranked_memory.load_order(model)
    assert order.trained_on["problems"] == 60 and list(order.names) == list(study.learned_order.FEATURES)
    arms = study.read(first)
    assert study.learned_order.place_under(arms["taught-80"][0]["candidates"], order.order(arms["taught-80"][0]["candidates"])) == 1
