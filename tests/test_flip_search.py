"""The flip-graph search: every move leaves the tensor unchanged, and the walk shortens what it can."""
from __future__ import annotations

import random

import numpy as np
import pytest

from apedp_dyn.quantum import flip_search as fs
from apedp_dyn.quantum import tower_reference
from apedp_dyn.quantum.field_tensor_search import build_multiplication_tensor, gf_mul


def _field(n: int, modulus: int) -> np.ndarray:
    T = np.zeros((n, n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            p = gf_mul(1 << i, 1 << j, modulus)
            for k in range(n):
                T[i, j, k] = (p >> k) & 1
    return T


def _invariant_holds(scheme: fs.Scheme) -> bool:
    rows = scheme.triples()
    if any(0 in row for row in rows):
        return False
    for i in range(len(rows)):
        for j in range(i):
            if sum(rows[i][m] == rows[j][m] for m in range(3)) >= 2:
                return False
    for m in range(3):
        held = sorted(v for v, bucket in scheme.index[m].items() if len(bucket) >= 2)
        if sorted(scheme.shared[m]) != held:
            return False
        if any(t not in scheme.index[m][scheme.f[m][t]] for t in range(len(scheme))):
            return False
    return True


def test_flips_and_plus_transitions_never_change_the_tensor() -> None:
    T = build_multiplication_tensor()
    scheme = fs.from_decomposition(tower_reference.build_tower_42_reference())
    rng = random.Random(7)
    for step in range(4000):
        if scheme.flip(rng) is None or step % 400 == 0:
            assert scheme.plus(rng)
    assert np.array_equal(scheme.tensor(), T)
    assert _invariant_holds(scheme)


def test_the_42_products_share_no_factor_so_no_flip_leaves_them() -> None:
    scheme = fs.from_decomposition(tower_reference.build_tower_42_reference())
    assert len(scheme) == 42
    assert scheme.flip(random.Random(0)) is None
    assert not any(scheme.shared)


def test_two_products_sharing_two_factors_are_merged_into_one() -> None:
    scheme = fs.Scheme([(1, 2, 4), (1, 2, 8), (3, 5, 6)], degree=4)
    assert sorted(scheme.triples()) == sorted([(1, 2, 12), (3, 5, 6)])
    assert fs.Scheme([(1, 2, 4), (1, 2, 4)], degree=4).triples() == []


def test_a_dependent_group_loses_one_product_and_keeps_the_tensor() -> None:
    rows = [(1, 1, 1), (1, 2, 2), (1, 3, 4)]                  # y: 1 + 2 = 3, all sharing x = 1
    before = fs.reconstruct(rows, 4)
    scheme = fs.Scheme(rows, degree=4)
    assert len(scheme) == 2
    assert np.array_equal(scheme.tensor(), before)


def test_the_walk_reaches_the_least_counts_of_small_fields_from_the_schoolbook() -> None:
    # GF(2^3) needs 6 products and GF(2^4) needs 9 (arXiv:2602.15285, Table 2)
    for n, modulus, least in ((3, 0b1011, 6), (4, 0b10011, 9)):
        T = _field(n, modulus)
        result = fs.walk(fs.schoolbook(T), seconds=20.0, seed=0, target=least, check=T)
        assert result.best == least
        assert fs.verified(result.best_triples, T)


def test_a_walk_from_the_42_stays_exact() -> None:
    T = build_multiplication_tensor()
    result = fs.walk(fs.from_decomposition(tower_reference.build_tower_42_reference()), seconds=1.5, seed=3, check=T)
    assert result.best <= 42
    assert fs.verified(result.best_triples, T)
    assert result.plus_transitions > 0


def test_verified_refuses_a_decomposition_with_one_bit_wrong() -> None:
    T = build_multiplication_tensor()
    rows = fs.from_decomposition(tower_reference.build_tower_42_reference()).triples()
    assert fs.verified(rows, T)
    a, b, c = rows[0]
    assert not fs.verified([(a, b, c ^ 1), *rows[1:]], T)
    assert not fs.verified(rows[1:], T)


def test_to_and_from_a_decomposition_round_trip() -> None:
    scheme = fs.schoolbook(build_multiplication_tensor())
    back = fs.from_decomposition(fs.to_decomposition(scheme))
    assert sorted(back.triples()) == sorted(scheme.triples())


@pytest.mark.parametrize("seed", [0, 1])
def test_random_moves_keep_the_bookkeeping_right(seed: int) -> None:
    T = _field(5, 0b100101)
    scheme = fs.schoolbook(T)
    rng = random.Random(seed)
    for step in range(3000):
        if scheme.flip(rng) is None or step % 97 == 0:
            scheme.plus(rng)
        if step % 500 == 0:
            assert _invariant_holds(scheme)
    assert np.array_equal(scheme.tensor(), T)
