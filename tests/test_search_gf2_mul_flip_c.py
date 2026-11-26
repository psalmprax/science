"""The C flip walk: what it returns is exact, and collect checks it again."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf2_mul_flip_c as script

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


@needs_gcc
def test_a_short_walk_from_the_schoolbook_of_gf2_7_is_exact_and_shortens(tmp_path: Path) -> None:
    binary = script.build(tmp_path / "walk")
    row = script.walk(binary, 7, "schoolbook", seed=1, seconds=3.0, slack=2, plateau=100, reset=1_000_000)
    assert row["exact"] and row["starting_products"] == 49 and row["best"] <= 30


@needs_gcc
def test_a_walk_from_the_gf16_tower_of_gf2_8_reaches_24(tmp_path: Path) -> None:
    binary = script.build(tmp_path / "walk")
    row = script.walk(binary, 8, "tower:4:0", seed=2, seconds=5.0, slack=2, plateau=100, reset=1_000_000)
    assert row["starting_products"] == 27 and row["exact"] and row["best"] == 24


def test_collect_checks_kept_decompositions_again(tmp_path: Path) -> None:
    good = script.start(9, "tower:3:0")
    bad = [list(t) for t in good[1:]]
    for name, triples in (("good", [list(t) for t in good]), ("bad", bad)):
        row = {"field": 9, "start": "tower:3:0", "starting_products": 30, "seed": 0, "slack": 2, "plateau": 100,
               "reset": 1, "seconds": 1.0, "best": len(triples), "flips": 0, "plus_transitions": 0, "resets": 0,
               "exact": True, "progress": [], "triples": triples}
        (tmp_path / f"walk_{name}.json").write_text(json.dumps(row))
    assert script.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "flip_walk_c.json").read_text())
    assert record["by_field"]["9"]["walks_not_exact"] == 1
    assert record["by_field"]["9"]["best_reached_exactly"] == 30 and not record["by_field"]["9"]["below_known"]
