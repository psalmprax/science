"""The flip walk over GF(4): its starts and what it returns are exact over GF(4) and over GF(2), and collect checks again."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf4_mul_flip as script

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


def test_the_starts_are_exact_and_give_three_products_over_gf2_each() -> None:
    school = script.start(4, "schoolbook")
    assert len(school) == 16 and script.verified(school, 4)
    fourteen = script.start(6, "f14:0")
    assert len(fourteen) == 14 and script.verified(fourteen, 6)
    assert len(script.over_gf2(fourteen, 6)) == 42


def test_a_changed_entry_is_refused() -> None:
    terms = script.start(6, "f14:1")
    broken = [(terms[0][0] ^ 1, terms[0][1], terms[0][2]), *terms[1:]]
    assert not script.exact4(broken, 6) and not script.verified(broken, 6)


@needs_gcc
def test_a_walk_from_the_schoolbook_of_gf4_4_reaches_the_published_8(tmp_path: Path) -> None:
    binary = script.build(tmp_path / "walk")
    row = script.walk(binary, 4, "schoolbook", seed=1, seconds=10.0, slack=2, plateau=100, reset=1_000_000, target=8)
    assert row["starting_products"] == 16 and row["exact"] and row["best"] == 8 and len(row["terms"]) == 8


def test_collect_checks_kept_decompositions_again(tmp_path: Path) -> None:
    good = [list(t) for t in script.start(6, "f14:2")]
    for name, terms in (("good", good), ("bad", good[1:])):
        row = {"degree": 6, "start": "f14:2", "starting_products": 14, "seed": 0, "slack": 2, "plateau": 100,
               "reset": 1, "seconds": 1.0, "best": len(terms), "flips": 0, "plus_transitions": 0, "resets": 0,
               "exact": True, "progress": [], "terms": terms}
        (tmp_path / f"walk4_{name}.json").write_text(json.dumps(row))
    assert script.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "gf4_flip_walk.json").read_text())
    six = record["by_degree"]["6"]
    assert six["walks_not_exact"] == 1 and six["best_reached_exactly"] == 14 and not record["target_met"]
