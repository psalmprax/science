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


def _pools(path: Path, problems: int = 60, seed: int = 0) -> Path:
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
            rows.append([arm, f"p{number:03d}", 1 + number % 5, 1 + number % 3, 0.01, 40.0, pool])
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
