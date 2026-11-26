"""The invariant search: its decompositions are exact, and it finds what is known where it should."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_invariant_search as inv
import gf2_towers as towers

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


def test_orbits_cover_the_field_and_the_kernel_has_the_orbit_s_dimension() -> None:
    field = inv.Field(6, 2)
    orbs = inv.orbits(field, 1)
    assert sum(d for _, d in orbs) == 63
    assert all(len(inv.kernel(field, lam, d, 1)) == d for lam, d in orbs)
    four = inv.Field(12, 4)
    assert sum(d for _, d in inv.orbits(four, 2)) == 4095 // 3


def test_profiles_and_leaves_count_the_choices() -> None:
    counts = {6: 9, 3: 2, 2: 1, 1: 1}
    profs = inv.profiles(counts, 15)
    assert [6, 6, 3] in profs and all(sum(p) == 15 for p in profs)
    assert inv.leaves(counts, [6, 6, 3]) == 36 * 2


@needs_gcc
def test_gf2_6_has_an_invariant_decomposition_of_15_and_it_is_exact(tmp_path: Path) -> None:
    field = inv.Field(6, 2)
    path = tmp_path / "orb.bin"
    inv.prepare(field, 1, path)
    out = tmp_path / "out"
    assert inv.main(["run", "--file", str(path), "--over", "2", "--step", "1", "--rank", "15", "--out", str(out)]) == 0
    record = json.loads(next(out.glob("shard_*.json")).read_text())
    found = [reps for row in record["profiles"] for reps in row["reps"]]
    assert found
    triples, products = inv.extract(field, 1, found[0])
    assert products == len(triples) == 15 and towers.exact(triples, 6)


@needs_gcc
def test_gf2_7_has_none_of_22_in_the_class(tmp_path: Path) -> None:
    field = inv.Field(7, 2)
    path = tmp_path / "orb.bin"
    inv.prepare(field, 1, path)
    out = tmp_path / "out"
    inv.main(["run", "--file", str(path), "--over", "2", "--step", "1", "--rank", "22", "--out", str(out)])
    record = json.loads(next(out.glob("shard_*.json")).read_text())
    assert sum(row["found"] for row in record["profiles"]) == 0


def test_a_gf4_decomposition_of_14_gives_42_exact_products() -> None:
    field = inv.Field(12, 4)
    triples, products = inv.extract(field, 2, [103, 131, 138, 195, 1])
    from apedp_dyn.quantum import flip_search
    assert products == 14 and len(triples) == 42 and flip_search.verified(triples)


def test_extract_refuses_orbits_that_do_not_reach_the_multiplication() -> None:
    with pytest.raises(ValueError):
        inv.extract(inv.Field(12, 4), 2, [103, 131])
