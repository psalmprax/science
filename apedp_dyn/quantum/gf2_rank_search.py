"""Exact local rank search for GF(2^n) multiplication tensors.

Status: experimental

Why this module exists
----------------------
``field_tensor_search.search_sat_rank`` was a stub that returned ``None``, so the
specification's "< 51 Toffoli" objective had no search behind it — only three hand-written
references and a flip graph.

The object being searched
-------------------------
Write ``T[i, j, k]`` for the multiplication tensor.  If ``B_k[(i, j)] = T[i, j, k]`` are the
``n`` output *slices*, a bilinear algorithm of rank ``r`` is a list of rank-one matrices
``P_l = u_l v_l^T`` together with coefficients ``W[l, k]`` such that

    B_k = sum_l W[l, k] * P_l        for every k.

So the rank is the smallest set of rank-one matrices whose **span contains every slice**, and
the question "can this decomposition be shortened?" is a question about spans, not about
entries.  Once the atoms ``P_l`` are chosen, ``W`` is *determined* by GF(2) elimination; that
is precisely the step ``search_flip_graph`` skipped, which is why each of its variants changed
``U`` and ``V`` and then failed the exact coefficient check (850 / 728 / 813 mismatches).

The moves, and what a negative means
-----------------------------------
Two moves are implemented, each exact and exhaustive over its declared family:

* **deflation** — drop an atom whose removal still leaves every slice inside the span.  A
  success is a strictly smaller decomposition; a failure says every atom of the input is
  load-bearing.
* **k -> k-1 rewrite** — replace ``k`` atoms with ``k-1`` rank-one atoms.  Against the
  remaining atoms the slices leave a residual space of dimension at most ``k``, so the whole
  question collapses to: is that residual space spanned by ``k-1`` rank-one matrices?  For
  ``k = 2`` the space has three nonzero elements and **all three** must be rank-one; for
  ``k = 3``, all seven.  Because a small space has so few nonzero elements, the test is
  complete over *all* rank-one matrices, not only those in some candidate pool.

A negative result from these moves is a statement about this family only.  It is not a proof
of minimality, and this module never reports one: nothing is called a rank unless the
reconstruction matches all ``n**3`` coefficients and ``W`` solves.

Deliberately not implemented: greedy cover.  Adding one atom to a basis can only cover a
residual that is *equal* to that atom (reduced), so on a 12-dimensional slice space a greedy
one-atom lookahead cannot make progress; it would be a stub with extra steps.
"""
from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Sequence

import numpy as np

__all__ = [
    "RankSearchResult",
    "atom_mask",
    "atom_masks",
    "build_slices",
    "deflate",
    "is_rank_one",
    "reconstruct",
    "reduce_once",
    "search_rank",
    "solve_weights",
]


# --- GF(2) bit-mask linear algebra -------------------------------------------------------
#
# A d x d matrix is a Python int with bit (i * d + j) for entry (i, j).  A subspace is a dict
# {pivot_bit: vector} in row-echelon form, where each stored vector's highest set bit is its
# pivot.  Integer XOR is the field addition, so reductions are a handful of big-int XORs.


def _reduce_in(vector: int, pivots: dict[int, int]) -> int:
    """Reduce ``vector`` to its normal form modulo the span of ``pivots``."""
    while vector:
        pivot = vector.bit_length() - 1
        basis_vector = pivots.get(pivot)
        if basis_vector is None:
            break
        vector ^= basis_vector
    return vector


def _insert(pivots: dict[int, int], vector: int) -> bool:
    """Insert ``vector`` into ``pivots``; False when it was already in the span."""
    reduced = _reduce_in(vector, pivots)
    if not reduced:
        return False
    pivots[reduced.bit_length() - 1] = reduced
    return True


def _pivots_from(atoms: Sequence[int]) -> dict[int, int]:
    pivots: dict[int, int] = {}
    for atom in atoms:
        _insert(pivots, atom)
    return pivots


def _row_mask(row: np.ndarray) -> int:
    mask = 0
    for index in np.flatnonzero(row):
        mask |= 1 << int(index)
    return mask


def build_slices(tensor: np.ndarray) -> tuple[list[int], int]:
    """Split the tensor into one bit-mask per output coefficient, with the field degree."""
    degree = int(tensor.shape[0])
    if tensor.shape != (degree, degree, degree):
        raise ValueError(f"expected a cubic tensor, got shape {tensor.shape}")
    slices = np.ascontiguousarray(tensor.transpose(2, 0, 1)).reshape(degree, degree * degree) & 1
    return [_row_mask(row) for row in slices], degree


def atom_mask(u: int, v: int, degree: int) -> int:
    """Bit-mask of the rank-one matrix ``u v^T``."""
    mask = 0
    u_bits, v_bits = u, v
    while u_bits:
        low_u = u_bits & -u_bits
        i = low_u.bit_length() - 1
        u_bits ^= low_u
        row_base = i * degree
        rest = v_bits
        while rest:
            low_v = rest & -rest
            mask |= 1 << (row_base + low_v.bit_length() - 1)
            rest ^= low_v
    return mask


def atom_masks(U: np.ndarray, V: np.ndarray) -> list[int]:
    """Rank-one matrices of a decomposition, row by row, skipping zero atoms."""
    degree = int(U.shape[1])
    atoms: list[int] = []
    for row_u, row_v in zip(U, V, strict=True):
        u = sum(1 << int(index) for index in np.flatnonzero(row_u))
        v = sum(1 << int(index) for index in np.flatnonzero(row_v))
        mask = atom_mask(u, v, degree) if u and v else 0
        atoms.append(mask)
    return atoms


def _rows_of(mask: int, degree: int) -> list[int]:
    low_mask = (1 << degree) - 1
    return [(mask >> (index * degree)) & low_mask for index in range(degree)]


def is_rank_one(mask: int, degree: int) -> bool:
    """True when the mask is one nonzero rank-one matrix over GF(2)."""
    rows = _rows_of(mask, degree)
    nonzero = [row for row in rows if row]
    if not nonzero:
        return False
    first = nonzero[0]
    return all(row in (0, first) for row in rows)


def _rank_one_factors(mask: int, degree: int) -> tuple[int, int]:
    """``(u, v)`` with ``mask == u v^T``; the mask must be rank one."""
    rows = _rows_of(mask, degree)
    v, u = 0, 0
    for _index, row in enumerate(rows):
        if row:
            v = row
            break
    if not v:
        raise ValueError("zero matrix has no rank-one factors")
    for index, row in enumerate(rows):
        if row == v:
            u |= 1 << index
        elif row:
            raise ValueError("matrix is not rank one")
    return u, v


def _nonzero_elements(pivots: dict[int, int]) -> list[int]:
    """Every nonzero element of the space — at most 2**dim - 1 of them, dim is tiny here."""
    basis = list(pivots.values())
    elements: list[int] = []
    for size in range(1, len(basis) + 1):
        for combination in itertools.combinations(basis, size):
            value = 0
            for vector in combination:
                value ^= vector
            if value not in elements:
                elements.append(value)
    return elements


def _covers(slice_masks: Sequence[int], pivots: dict[int, int]) -> bool:
    return all(_reduce_in(slice_mask, pivots) == 0 for slice_mask in slice_masks)


def solve_weights(
    tensor: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """Solve for ``W`` over GF(2); returns ``(consistent, W)``.

    Rows are the ``n**2`` pairs ``(i, j)``, columns the ``r`` atoms, and the right-hand side
    carries all ``n`` output coefficients of that pair at once, so one elimination decides the
    whole system.
    """
    degree = int(tensor.shape[0])
    rank = int(U.shape[0])
    output_bits = (1 << degree) - 1
    pivots: dict[int, int] = {}

    for i in range(degree):
        for j in range(degree):
            variables = 0
            for index in range(rank):
                if U[index, i] and V[index, j]:
                    variables |= 1 << index
            rhs = 0
            for k in range(degree):
                if int(tensor[i, j, k]) & 1:
                    rhs |= 1 << k
            row = (variables << degree) | rhs
            while True:
                variable_part = row >> degree
                if not variable_part:
                    if row & output_bits:
                        return False, None
                    break
                pivot = variable_part.bit_length() - 1 + degree
                if pivot in pivots:
                    row ^= pivots[pivot]
                else:
                    pivots[pivot] = row
                    break

    # The rows are in echelon form, so a pivot's value is its own right-hand side plus the
    # already-solved lower-index variables of its row. Ascending pivot order makes that one pass.
    values: dict[int, int] = {}
    for pivot in sorted(pivots):
        row = pivots[pivot]
        index = pivot - degree
        value = row & output_bits
        lower = (row >> degree) & ((1 << index) - 1)
        while lower:
            low = lower & -lower
            # A free variable is not a pivot; it takes the value 0 throughout.
            value ^= values.get(low.bit_length() - 1, 0)
            lower ^= low
        values[index] = value

    W: np.ndarray = np.zeros((rank, degree), dtype=np.uint8)
    for index, value in values.items():
        for k in range(degree):
            if (value >> k) & 1:
                W[index, k] = 1

    # Unpivoted (free) variables stay 0; recompute to prove the whole system is satisfied.
    if not np.array_equal((tensor & 1).astype(np.uint8), reconstruct(U, V, W)):
        return False, None
    return True, W


def reconstruct(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Rebuild the tensor from a decomposition, exactly as the verifier does."""
    degree = int(U.shape[1])
    rank = int(U.shape[0])
    T: np.ndarray = np.zeros((degree, degree, degree), dtype=np.uint8)
    for index in range(rank):
        T ^= np.einsum("i,j,k->ijk", U[index], V[index], W[index]) & 1
    return T


@dataclasses.dataclass(frozen=True)
class RankSearchResult:
    """Outcome of a bounded search. ``rank`` is set only for an exactly verified result."""

    found: bool
    rank: int | None
    start_rank: int
    U: np.ndarray | None
    V: np.ndarray | None
    W: np.ndarray | None
    atom_count: int
    deflations: int
    rewrites: int
    moves_tried: int
    max_rewrite_k: int
    residual_dimension: int
    elapsed_sec: float
    seed: int
    time_limit_sec: float

    def summary(self) -> dict[str, object]:
        return {
            "found": self.found,
            "rank": self.rank,
            "start_rank": self.start_rank,
            "atom_count": self.atom_count,
            "deflations": self.deflations,
            "rewrites": self.rewrites,
            "moves_tried": self.moves_tried,
            "max_rewrite_k": self.max_rewrite_k,
            "residual_dimension": self.residual_dimension,
            "elapsed_sec": self.elapsed_sec,
            "seed": self.seed,
            "time_limit_sec": self.time_limit_sec,
        }


def deflate(atoms: Sequence[int], slice_masks: Sequence[int]) -> tuple[list[int], int]:
    """Drop every atom whose removal still leaves all slices spanned. Returns (atoms, dropped)."""
    current = list(atoms)
    dropped = 0
    changed = True
    while changed:
        changed = False
        for index in range(len(current)):
            trial = current[:index] + current[index + 1 :]
            if _covers(slice_masks, _pivots_from(trial)):
                current = trial
                dropped += 1
                changed = True
                break
    return current, dropped


def _residual_space(atoms: Sequence[int], slice_masks: Sequence[int]) -> dict[int, int]:
    """Span of the slices not covered by ``atoms``."""
    pivots = _pivots_from(atoms)
    residual: dict[int, int] = {}
    for slice_mask in slice_masks:
        _insert(residual, _reduce_in(slice_mask, pivots))
    return residual


def reduce_once(
    atoms: Sequence[int],
    slice_masks: Sequence[int],
    degree: int,
    k: int,
    deadline: float | None = None,
) -> tuple[list[int] | None, int]:
    """Try every ``k -> k-1`` rewrite. Returns ``(new_atoms, moves_tried)``.

    Complete over all rank-one matrices: the residual space has at most ``k`` dimensions, so
    all of its nonzero elements are enumerated and each must itself be rank one.
    """
    tried = 0
    count = len(atoms)
    if k < 2 or count <= k:
        return None, 0
    for combination in itertools.combinations(range(count), k):
        if deadline is not None and (tried & 0x3F) == 0 and time.perf_counter() > deadline:
            return None, tried
        tried += 1
        dropped = set(combination)
        rest = [atom for index, atom in enumerate(atoms) if index not in dropped]
        residual = _residual_space(rest, slice_masks)
        if not residual:
            continue  # deflation owns this case
        if len(residual) > k - 1:
            continue
        elements = _nonzero_elements(residual)
        if not all(is_rank_one(element, degree) for element in elements):
            continue
        added: list[int] = []
        added_pivots: dict[int, int] = {}
        for element in elements:
            if _insert(added_pivots, element):
                added.append(element)
        if len(added) >= k:
            continue
        candidate = rest + added
        if _covers(slice_masks, _pivots_from(candidate)):
            return candidate, tried
    return None, tried


def _verified(
    tensor: np.ndarray,
    atoms: Sequence[int],
    degree: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Build ``U``, ``V``, ``W`` for the atoms, or Nones when the reconstruction is not exact."""
    if not atoms:
        return None, None, None
    U: np.ndarray = np.zeros((len(atoms), degree), dtype=np.uint8)
    V: np.ndarray = np.zeros((len(atoms), degree), dtype=np.uint8)
    for index, atom in enumerate(atoms):
        if not is_rank_one(atom, degree):
            return None, None, None
        u, v = _rank_one_factors(atom, degree)
        U[index] = [(u >> bit) & 1 for bit in range(degree)]
        V[index] = [(v >> bit) & 1 for bit in range(degree)]
    consistent, W = solve_weights(tensor, U, V)
    if not consistent or W is None:
        return None, None, None
    if not np.array_equal((tensor & 1).astype(np.uint8), reconstruct(U, V, W)):
        return None, None, None
    return U, V, W


def search_rank(
    tensor: np.ndarray,
    target_rank: int,
    time_limit_sec: float = 2.0,
    seed: int = 42,
    start: tuple[np.ndarray, np.ndarray] | None = None,
    max_k: int = 3,
) -> RankSearchResult:
    """Shorten a decomposition by exact rewrites until ``target_rank`` or the budget runs out.

    ``start`` supplies the atoms to shorten; without it the search has nothing to rewrite and
    returns ``found=False`` with ``moves_tried == 0`` (an empty input is not a search).
    """
    begin = time.perf_counter()
    deadline = begin + max(0.0, time_limit_sec)
    slice_masks, degree = build_slices(tensor)
    if start is None:
        return RankSearchResult(
            found=False, rank=None, start_rank=0, U=None, V=None, W=None, atom_count=0,
            deflations=0, rewrites=0, moves_tried=0, max_rewrite_k=max_k,
            residual_dimension=len(_residual_space([], slice_masks)),
            elapsed_sec=time.perf_counter() - begin, seed=seed, time_limit_sec=time_limit_sec,
        )

    atoms = [atom for atom in atom_masks(*start) if atom]
    start_rank = len(atoms)
    atoms, deflations = deflate(atoms, slice_masks)
    rewrites = 0
    moves_tried = 0
    U = V = W = None

    while time.perf_counter() <= deadline:
        U, V, W = _verified(tensor, atoms, degree)
        if U is None:
            atoms, extra = deflate(atoms, slice_masks)
            deflations += extra
            U, V, W = _verified(tensor, atoms, degree)
            if U is None:
                break
        if len(atoms) <= target_rank:
            break
        progressed = False
        for k in range(2, max_k + 1):
            shorter, tried = reduce_once(atoms, slice_masks, degree, k, deadline=deadline)
            moves_tried += tried
            if shorter is None:
                continue
            candidate_U, candidate_V, candidate_W = _verified(tensor, shorter, degree)
            if candidate_U is None:
                continue
            atoms, U, V, W = shorter, candidate_U, candidate_V, candidate_W
            rewrites += 1
            progressed = True
            break
        if not progressed:
            break

    residual = len(_residual_space(atoms, slice_masks))
    found = U is not None and len(atoms) <= target_rank
    return RankSearchResult(
        found=found,
        rank=len(atoms) if U is not None else None,
        start_rank=start_rank,
        U=U,
        V=V,
        W=W,
        atom_count=len(atoms),
        deflations=deflations,
        rewrites=rewrites,
        moves_tried=moves_tried,
        max_rewrite_k=max_k,
        residual_dimension=residual,
        elapsed_sec=time.perf_counter() - begin,
        seed=seed,
        time_limit_sec=time_limit_sec,
    )
