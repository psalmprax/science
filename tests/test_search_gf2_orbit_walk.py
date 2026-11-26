"""The symmetric flip walk: its starts are exact and invariant, every move keeps the tensor, and collect checks again."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers
import search_gf2_orbit_walk as script

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


def test_the_normal_basis_schoolbook_is_exact_and_invariant_under_every_step() -> None:
    t = script.normal_schoolbook(7)
    assert len(t) == 49 and towers.exact(t, 7)
    assert all(script.invariant(t, 7, step) for step in range(1, 7))


def test_the_gf4_starts_are_exact_42s_invariant_under_x_to_the_4() -> None:
    t = script.start(12, 2, "gf4:3")
    assert len(t) == 42 and script.verified(t, 12, 2) and script.verified(t, 12, 6)
    assert not script.invariant(t, 12, 1)


@needs_gcc
def test_every_move_of_a_symmetric_walk_keeps_the_tensor(tmp_path: Path) -> None:
    binary = script.build(tmp_path / "walk")
    row = script.walk(binary, 7, 1, "normal", seed=3, seconds=2.0, slack=2, plateau=100, reset=1_000_000, target=0, check=True)
    assert row["exact"] and row["starting_products"] == 49 and row["best"] <= 35 and row["flips"] > 0


@needs_gcc
def test_a_start_that_is_not_invariant_is_refused(tmp_path: Path) -> None:
    binary = script.build(tmp_path / "walk")
    with pytest.raises(ValueError):
        script.walk(binary, 12, 1, "gf4:0", seed=1, seconds=1.0, slack=2, plateau=100, reset=1, target=0)


def test_collect_checks_kept_decompositions_again(tmp_path: Path) -> None:
    good = [list(t) for t in script.start(12, 2, "gf4:1")]
    for name, triples in (("good", good), ("bad", good[1:])):
        row = {"field": 12, "step": 2, "start": "gf4:1", "starting_products": 42, "seed": 0, "slack": 2, "plateau": 100,
               "reset": 1, "seconds": 1.0, "best": len(triples), "flips": 0, "plus_transitions": 0, "resets": 0,
               "exact": True, "progress": [], "triples": triples}
        (tmp_path / f"orbit_{name}.json").write_text(json.dumps(row))
    assert script.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "orbit_walk.json").read_text())
    cls = record["by_class"]["GF(2^12) x -> x^(2^2)"]
    assert cls["walks_not_exact"] == 1 and cls["best_reached_exactly"] == 42 and not record["target_met"]
