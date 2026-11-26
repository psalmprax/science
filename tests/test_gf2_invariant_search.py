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


@pytest.mark.parametrize("q", [4, 8, 16])
def test_the_subfield_decomposition_multiplies_gfq_inside_gf2_12(q: int) -> None:
    field = inv.Field(12, q)
    coord, base = inv._subfield_base(field)
    assert len(base) == {4: 3, 8: 6, 16: 9}[q]
    elems = dict(coord)
    assert len(elems) == q
    for u in elems:
        for v in elems:
            acc = 0
            for form, _, delta in base:
                pu = bin(form & elems[u]).count("1") & 1
                pv = bin(form & elems[v]).count("1") & 1
                if pu and pv:
                    acc ^= delta
            assert acc == field.mul(u, v), (u, v)


def test_mixed_extraction_of_a_recorded_gf4_choice_gives_42_exact() -> None:
    from apedp_dyn.quantum import flip_search
    triples, cost = inv.extract_mixed(2, [406, 403, 402, 402, 401], [103, 131, 138, 195, 1])
    assert cost == 42 and len(triples) == 42 and flip_search.verified(triples)
    with pytest.raises(ValueError):
        inv.extract_mixed(2, [406, 403, 402, 402, 401], [103, 131, 138, 195, 459])   # not among the six the enumeration finds


@needs_gcc
def test_a_mixed_run_finds_the_known_class_and_collect_checks_it(tmp_path: Path) -> None:
    orbit_file = tmp_path / "m4.bin"
    inv.main(["prepare-mixed", "--families", "4", "--step", "2", "--out", str(orbit_file)])
    out = tmp_path / "out"
    assert inv.main(["run-mixed", "--file", str(orbit_file), "--step", "2", "--budget", "42", "--min-leaves", "50000",
                     "--max-leaves", "60000", "--out", str(out)]) == 0
    shard = json.loads(next(out.glob("shard_*.json")).read_text())
    assert [r["profile"] for r in shard["profiles"]] == [[406, 403, 402, 402, 401]]
    assert shard["profiles"][0]["found"] == 6
    assert inv.main(["collect", "--out", str(out)]) == 0
    record = json.loads((out / "invariant_search.json").read_text())
    assert all(c["exact"] and c["products_over_gf2"] == 42 for c in record["checked"]) and not record["below_42"]
