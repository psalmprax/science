"""Pins for the exact GF(2^12) rank search that replaced the ``search_sat_rank`` stub.

Status: audit-only

The point of these tests is that the search cannot lie: a rank is reported only for a
decomposition whose 1,728 coefficients all reconstruct, and the two negatives it produces are
complete over their declared family (every two-atom and every three-atom rewrite of the
shipped 54).
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.quantum.field_tensor_search import (
    TensorDecomposition,
    build_54_reference,
    build_karatsuba_reference,
    build_multiplication_tensor,
    build_schoolbook_reference,
    gf_mul,
    search_sat_rank,
    verify_tensor_decomposition,
)
from apedp_dyn.quantum.gf2_rank_search import (
    atom_masks,
    build_slices,
    deflate,
    is_rank_one,
    search_rank,
    solve_weights,
)

BASE54_TWO_THREE_MOVES = 1431 + 24804  # C(54, 2) + C(54, 3) = 26235
BASE54_FULL_MOVES = 1431 + 24804 + 316251  # C(54, 2) + C(54, 3) + C(54, 4) = 342486

@pytest.fixture(scope="module")
def tensor():
    return build_multiplication_tensor()


def _small_field_tensor(degree: int, modulus: int) -> np.ndarray:
    order = 1 << degree
    T = np.zeros((degree, degree, degree), dtype=np.uint8)
    for i in range(degree):
        for j in range(degree):
            product = gf_mul(1 << i, 1 << j, modulus)
            for k in range(degree):
                T[i, j, k] = (product >> k) & 1
    assert any(T.reshape(-1)), order
    return T


def _schoolbook_start(degree: int) -> tuple[np.ndarray, np.ndarray]:
    rank = degree * degree
    U = np.zeros((rank, degree), dtype=np.uint8)
    V = np.zeros((rank, degree), dtype=np.uint8)
    index = 0
    for i in range(degree):
        for j in range(degree):
            U[index, i] = 1
            V[index, j] = 1
            index += 1
    return U, V


def test_the_solver_reproduces_every_shipped_reference(tensor):
    """``solve_weights`` must agree with the engine's own verifier on all three references."""
    for reference in (build_schoolbook_reference(), build_karatsuba_reference(), build_54_reference()):
        consistent, W = solve_weights(tensor, reference.U, reference.V)
        assert consistent, reference.name
        assert W is not None
        solved = TensorDecomposition(rank=reference.rank, U=reference.U, V=reference.V, W=W, name="solved")
        exact, mismatches = verify_tensor_decomposition(tensor, solved)
        assert exact, (reference.name, mismatches[:5])


def test_karatsuba_63_deflates_to_the_base_composed_54(tensor):
    """The search rediscovers rank 54 from Karatsuba's atoms by span alone."""
    reference = build_karatsuba_reference()
    result = search_rank(tensor, target_rank=50, time_limit_sec=30.0, start=(reference.U, reference.V))

    assert result.start_rank == 63
    assert result.rank == 54, result.summary()
    assert result.deflations == 9
    assert result.residual_dimension == 0
    assert result.U is not None and result.V is not None and result.W is not None
    exact, _ = verify_tensor_decomposition(
        tensor, TensorDecomposition(rank=result.rank, U=result.U, V=result.V, W=result.W, name="deflated")
    )
    assert exact


def test_the_base_54_is_closed_under_every_two_and_three_atom_rewrite(tensor):
    """A negative result: the whole ``k -> k-1`` family for k in {2, 3} is exhausted."""
    reference = build_54_reference()
    result = search_rank(tensor, target_rank=50, time_limit_sec=120.0, start=(reference.U, reference.V), max_k=3)

    assert result.rank == 54, result.summary()
    assert result.rewrites == 0
    assert result.moves_tried == BASE54_TWO_THREE_MOVES, "the k in {2,3} family was not exhausted"
    assert result.found is False
    assert result.residual_dimension == 0


def test_a_missing_start_is_not_a_search(tensor):
    """No atoms means no moves; the result must say so instead of looking like a failed search."""
    result = search_rank(tensor, target_rank=50, time_limit_sec=1.0)
    assert result.found is False
    assert result.moves_tried == 0
    assert result.rank is None


def test_a_small_field_is_reconstructed_and_verified():
    """On GF(2^3) the search must return an exact decomposition for a field it was not tuned on."""
    degree = 3
    small = _small_field_tensor(degree, 0b1011)  # x^3 + x + 1
    U, V = _schoolbook_start(degree)
    result = search_rank(small, target_rank=5, time_limit_sec=20.0, start=(U, V), max_k=3)

    assert result.U is not None, result.summary()
    assert result.rank is not None and result.rank <= degree * degree, result.summary()
    from apedp_dyn.quantum.gf2_rank_search import reconstruct

    rebuilt = reconstruct(result.U, result.V, result.W)
    assert np.array_equal(rebuilt, small.astype(np.uint8))
    assert build_slices(small)[1] == degree


def test_a_duplicated_atom_is_deflated_away(tensor):
    """A redundant atom must be removed -- and the 54 recovered -- not merely tolerated."""
    reference = build_54_reference()
    U = np.vstack([reference.U, reference.U[:1]])
    V = np.vstack([reference.V, reference.V[:1]])
    result = search_rank(tensor, target_rank=50, time_limit_sec=30.0, start=(U, V), max_k=2)

    assert result.start_rank == 55
    assert result.deflations == 1, result.summary()
    assert result.rank == 54, result.summary()
    exact, _ = verify_tensor_decomposition(
        tensor, TensorDecomposition(rank=result.rank, U=result.U, V=result.V, W=result.W, name="deduped")
    )
    assert exact


def test_base54_closed_under_k2_k3_k4_rewrites(tensor):
    """Verify the base 54 is exhausted under k in {2,3,4} rewrite families."""
    reference = build_54_reference()
    result = search_rank(tensor, target_rank=50, time_limit_sec=120.0, start=(reference.U, reference.V), max_k=4)

    assert result.start_rank == 54
    assert result.deflations == 0
    assert result.rank == 54
    assert result.rewrites == 0
    assert result.moves_tried == BASE54_FULL_MOVES, "expected complete enumeration of k in {2,3,4}"
    assert result.found is False
    assert result.residual_dimension == 0
    assert result.max_rewrite_k == 4


def test_atoms_are_rank_one_and_slices_cover_the_tensor(tensor):
    reference = build_54_reference()
    slices, degree = build_slices(tensor)
    atoms = atom_masks(reference.U, reference.V)
    assert len(atoms) == 54
    assert all(is_rank_one(atom, degree) for atom in atoms)
    reduced, dropped = deflate(atoms, slices)
    assert dropped == 0, "the base 54 is not supposed to have redundant atoms"
    assert len(reduced) == 54


def test_search_sat_rank_runs_the_real_search(tensor, monkeypatch):
    """The engine entry point must call the search module, not return a constant ``None``."""
    import apedp_dyn.quantum.gf2_rank_search as rank_search

    calls: list[dict[str, object]] = []
    original = rank_search.search_rank

    def recording(*args, **kwargs):
        calls.append({"target_rank": kwargs.get("target_rank"), "start": kwargs.get("start") is not None})
        return original(*args, **kwargs)

    monkeypatch.setattr(rank_search, "search_rank", recording)
    outcome = search_sat_rank(50, time_limit_sec=10.0, seed=42)

    assert calls, "search_sat_rank did not call the rank search"
    assert calls[0]["target_rank"] == 50
    assert calls[0]["start"] is True
    assert outcome is None, "the shipped 54 must not be reported as reaching rank 50"
