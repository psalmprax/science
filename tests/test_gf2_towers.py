"""Tower decompositions of GF(2^n) multiplication: exact, of the sizes the construction says, and distinct."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers


@pytest.mark.parametrize(("n", "k", "size"), [(6, 2, 15), (9, 3, 30), (12, 3, 42), (12, 4, 45), (8, 4, 27)])
def test_a_tower_is_exact_and_has_2m_minus_1_times_the_subfield_s_products(n: int, k: int, size: int) -> None:
    triples = towers.tower(n, k)
    assert len(triples) == size
    assert towers.exact(triples, n)


def test_variants_of_the_42_differ_by_construction() -> None:
    first = towers.tower(12, 3, *towers.variants(12, 3)[0])
    other = towers.tower(12, 3, *towers.variants(12, 3)[17])
    assert len(first) == len(other) == 42
    assert set(first) != set(other)


def test_exact_refuses_a_tower_with_one_product_changed() -> None:
    triples = towers.tower(9, 3)
    a, b, c = triples[0]
    assert not towers.exact([(a, b, c ^ 1), *triples[1:]], 9)


def test_too_few_points_and_a_wrong_subfield_are_refused() -> None:
    with pytest.raises(ValueError):
        towers.tower(12, 2)           # 11 points needed, the line over GF(4) has 5
    with pytest.raises(ValueError):
        towers.tower(12, 5)
