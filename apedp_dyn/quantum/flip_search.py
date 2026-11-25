"""A search that moves: random walks in the flip graph of a bilinear decomposition over GF(2).

Status: experimental

**What it is.** A decomposition of a multiplication tensor is a list of products ``a ⊗ b ⊗ c``
(``a`` a form in the bits of ``x``, ``b`` a form in the bits of ``y``, ``c`` the outputs the product
is added to). Kauers and Moosbauer ("Flip graphs for matrix multiplication", arXiv:2212.01175)
walk between decompositions of the same length by *flips*: two products that share a factor,

    x ⊗ y1 ⊗ z1 + x ⊗ y2 ⊗ z2  =  x ⊗ (y1 + y2) ⊗ z1 + x ⊗ y2 ⊗ (z1 + z2),

are replaced by the right-hand side, which is the same tensor. Whenever two products come to share
two factors they are merged into one (a *reduction*), and a product with a zero factor is dropped:
that is how the walk gets shorter. Where the walk stays at one length too long, a *plus transition*
(Arai, Ichikawa and Hukushima, "Adaptive flip graph algorithm", arXiv:2312.17293) makes it one
product longer,

    a ⊗ b ⊗ c + a' ⊗ b' ⊗ c'  =  (a + a') ⊗ b ⊗ c + a' ⊗ (b + b') ⊗ c + a' ⊗ b' ⊗ (c + c'),

so that it can leave a neighbourhood with no reduction in it. Every move leaves the tensor
unchanged exactly; nothing is approximated.

**Why it is here.** The exact search the repository already had (``gf2_rank_search.py``) walks a
narrow family around its starting point exhaustively and cannot leave it; from the 42 products of
GF(2^12) it found nothing (``benchmarks/gf2_multiplication/rank_search_from_42.json``). This search
leaves the starting point, and it holds no model. It is the baseline with no language model that a
language-model search has to beat (owner's decision 4 of 25 November).

**What it is not.** It decides nothing about itself: what it returns is accepted only through
``field_tensor_search.verify_tensor_decomposition`` (all ``n**3`` entries). ``search_flip_graph`` in
``field_tensor_search.py`` is a different, older function: its moves change the tensor, which is why
its variants failed the exact check. It is left as it is.
"""
from __future__ import annotations

import dataclasses
import random
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from apedp_dyn.quantum.field_tensor_search import (
    TensorDecomposition,
    build_multiplication_tensor,
    verify_tensor_decomposition,
)

__all__ = ["Scheme", "WalkResult", "from_decomposition", "reconstruct", "schoolbook", "to_decomposition", "verified", "walk"]

PLATEAU = 100            #: flips at one length without a reduction before a plus transition is tried
SLACK = 2                #: how many products above the shortest found a walk may climb by plus transitions
RESET = 300_000          #: flips without a new shortest after which the walk goes back to a shortest one found
KEPT = 16                #: shortest decompositions kept to go back to


def _bits(row: np.ndarray) -> int:
    return int(sum(int(v & 1) << i for i, v in enumerate(row)))


def _row(value: int, degree: int) -> np.ndarray:
    row: np.ndarray = np.array([(value >> i) & 1 for i in range(degree)], dtype=np.uint8)
    return row


class Scheme:
    """A decomposition as three lists of forms, indexed so that shared factors are found at once.

    ``f[0][t], f[1][t], f[2][t]`` are the three factors of product ``t``, each an integer whose bit
    ``i`` is coefficient ``i``. ``index[m][v]`` is the set of products whose factor ``m`` is ``v``.
    Invariant (kept by every move): no factor is zero and no two products share two factors.
    """

    def __init__(self, triples: Sequence[tuple[int, int, int]], degree: int) -> None:
        self.degree = int(degree)
        self.f: list[list[int]] = [[], [], []]
        self.index: list[dict[int, set[int]]] = [{}, {}, {}]
        self.shared: list[list[int]] = [[], [], []]           # factors held by two products or more, for each mode
        self._where: list[dict[int, int]] = [{}, {}, {}]      # position of such a factor in ``shared``
        for a, b, c in triples:
            self._append(a, b, c)
        while self._settle(list(range(len(self.f[0])))):
            pass

    # -- bookkeeping -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.f[0])

    def triples(self) -> list[tuple[int, int, int]]:
        return list(zip(self.f[0], self.f[1], self.f[2], strict=True))

    def copy(self) -> Scheme:
        return Scheme(self.triples(), self.degree)

    def _add(self, m: int, v: int, t: int) -> None:
        bucket = self.index[m].setdefault(v, set())
        bucket.add(t)
        if len(bucket) == 2:
            self._where[m][v] = len(self.shared[m])
            self.shared[m].append(v)

    def _drop(self, m: int, v: int, t: int) -> None:
        bucket = self.index[m][v]
        bucket.discard(t)
        if len(bucket) == 1:
            at = self._where[m].pop(v)
            moved = self.shared[m].pop()
            if moved != v:
                self.shared[m][at] = moved
                self._where[m][moved] = at
        elif not bucket:
            del self.index[m][v]

    def _append(self, a: int, b: int, c: int) -> int:
        t = len(self.f[0])
        for m, v in enumerate((a, b, c)):
            self.f[m].append(v)
            self._add(m, v, t)
        return t

    def _set(self, m: int, t: int, value: int) -> None:
        self._drop(m, self.f[m][t], t)
        self.f[m][t] = value
        self._add(m, value, t)

    def _remove(self, t: int) -> None:
        """Remove product ``t``; the last product takes its place."""
        last = len(self.f[0]) - 1
        for m in range(3):
            self._drop(m, self.f[m][t], t)
        if t != last:
            for m in range(3):
                v = self.f[m][last]
                bucket = self.index[m][v]
                bucket.discard(last)
                bucket.add(t)
                self.f[m][t] = v
        for m in range(3):
            self.f[m].pop()

    def _partner(self, t: int) -> tuple[int, int] | None:
        """Another product sharing two factors with ``t``, and the mode they do not share."""
        for m3 in (2, 0, 1):
            m1, m2 = [m for m in range(3) if m != m3]
            one, two = self.index[m1][self.f[m1][t]], self.index[m2][self.f[m2][t]]
            small, large = (one, two) if len(one) <= len(two) else (two, one)
            for s in small:
                if s != t and s in large:
                    return s, m3
        return None

    def _dependent(self, m: int, v: int, p: int) -> list[int] | None:
        """Products sharing factor ``v`` in mode ``m`` whose factors in mode ``p`` sum to zero, if any do."""
        basis: dict[int, tuple[int, list[int]]] = {}         # pivot bit -> (vector, the products it sums)
        for k in self.index[m][v]:
            vector, used = self.f[p][k], [k]
            while vector:
                top = vector.bit_length() - 1
                if top not in basis:
                    basis[top] = (vector, used)
                    break
                other, with_ = basis[top]
                vector ^= other
                used = used + with_
            else:
                counted: dict[int, int] = {}
                for u in used:
                    counted[u] = counted.get(u, 0) ^ 1
                return [u for u, odd in counted.items() if odd]
        return None

    def _settle(self, pending: list[int]) -> int:
        """Restore the invariant around the products in ``pending`` after they changed; return how many went.

        Three reductions, each an identity: a product with a zero factor is dropped; two products
        sharing two factors are merged; and where products share one factor and their factors in a
        second mode sum to zero over a subset, one of the subset is written as the sum of the others
        and dropped (``x⊗y_r⊗z_r = Σ x⊗y_k⊗z_r`` when ``y_r = Σ y_k``), its third factor added to theirs.
        """
        gone = 0
        pending = list(pending)
        while pending:
            t = pending.pop()
            if t >= len(self.f[0]):
                continue
            changed: list[int] = []
            if any(self.f[m][t] == 0 for m in range(3)):
                drop = t
            else:
                found = self._partner(t)
                if found is not None:
                    s, m3 = found
                    self._set(m3, t, self.f[m3][t] ^ self.f[m3][s])   # x⊗y⊗z1 + x⊗y⊗z2 = x⊗y⊗(z1+z2)
                    drop, changed = s, [t]
                else:
                    subset = None
                    for m in range(3):
                        if len(self.index[m][self.f[m][t]]) < 3:
                            continue
                        for p in range(3):
                            if p != m:
                                subset = self._dependent(m, self.f[m][t], p)
                                if subset:
                                    q = 3 - m - p
                                    break
                        if subset:
                            break
                    if not subset:
                        continue
                    drop = subset[0]
                    for k in subset[1:]:
                        self._set(q, k, self.f[q][k] ^ self.f[q][drop])
                    changed = subset[1:]
            last = len(self.f[0]) - 1
            self._remove(drop)                                # the last product now sits at ``drop``
            gone += 1
            pending = [drop if x == last else x for x in pending if x != drop or last != drop]
            pending += [drop if k == last else k for k in changed]
        return gone

    # -- moves -------------------------------------------------------------------------------
    def flip(self, rng: random.Random) -> int | None:
        """One random flip; ``None`` if no two products share a factor. Returns how many products went.

        A shared factor is drawn uniformly among all (mode, factor) pairs held by two products or more,
        then two of the products holding it.
        """
        sizes = [len(x) for x in self.shared]
        total = sizes[0] + sizes[1] + sizes[2]
        if not total:
            return None
        k = rng.randrange(total)
        m = 0 if k < sizes[0] else (1 if k < sizes[0] + sizes[1] else 2)
        v = self.shared[m][k - sum(sizes[:m])]
        t, s = rng.sample(tuple(self.index[m][v]), 2)
        p, q = [k for k in range(3) if k != m]
        if rng.random() < 0.5:
            p, q = q, p
        self._set(p, t, self.f[p][t] ^ self.f[p][s])          # t: x ⊗ (y1+y2) ⊗ z1
        self._set(q, s, self.f[q][s] ^ self.f[q][t])          # s: x ⊗ y2 ⊗ (z1+z2)
        return self._settle([t, s])

    def plus(self, rng: random.Random, tries: int = 64) -> bool:
        """One plus transition: one product more. ``False`` if no two products differ in all factors."""
        n = len(self.f[0])
        for _ in range(tries):
            i, j = rng.sample(range(n), 2)
            a, b, c = self.f[0][i], self.f[1][i], self.f[2][i]
            a2, b2, c2 = self.f[0][j], self.f[1][j], self.f[2][j]
            if a == a2 or b == b2 or c == c2:
                continue
            modes = [0, 1, 2]
            rng.shuffle(modes)
            # written for modes (0, 1, 2); a permutation of the modes is the same identity
            first = [0, 0, 0]
            first[modes[0]], first[modes[1]], first[modes[2]] = (self.f[modes[0]][i] ^ self.f[modes[0]][j],
                                                                 self.f[modes[1]][i], self.f[modes[2]][i])
            second = [0, 0, 0]
            second[modes[0]], second[modes[1]], second[modes[2]] = (self.f[modes[0]][j],
                                                                   self.f[modes[1]][i] ^ self.f[modes[1]][j],
                                                                   self.f[modes[2]][i])
            third = [0, 0, 0]
            third[modes[0]], third[modes[1]], third[modes[2]] = (self.f[modes[0]][j], self.f[modes[1]][j],
                                                                self.f[modes[2]][i] ^ self.f[modes[2]][j])
            for m in range(3):
                self._set(m, i, first[m])
                self._set(m, j, second[m])
            k = self._append(*third)
            self._settle([i, j, k])
            return True
        return False

    def tensor(self) -> np.ndarray:
        return reconstruct(self.triples(), self.degree)


def from_decomposition(decomposition: TensorDecomposition) -> Scheme:
    degree = decomposition.U.shape[1]
    triples = [(_bits(u), _bits(v), _bits(w)) for u, v, w in zip(decomposition.U, decomposition.V, decomposition.W, strict=True)]
    return Scheme(triples, degree)


def to_decomposition(scheme: Scheme, name: str = "FlipWalk", metadata: dict[str, Any] | None = None) -> TensorDecomposition:
    d = scheme.degree
    rows = scheme.triples()
    return TensorDecomposition(rank=len(rows), U=np.stack([_row(a, d) for a, _, _ in rows]),
                               V=np.stack([_row(b, d) for _, b, _ in rows]), W=np.stack([_row(c, d) for _, _, c in rows]),
                               name=name, metadata=dict(metadata or {}))


def schoolbook(tensor: np.ndarray | None = None) -> Scheme:
    """One product for each pair of input bits: ``x_i y_j`` added to the outputs of ``u^i u^j``."""
    tensor = build_multiplication_tensor() if tensor is None else tensor
    d = tensor.shape[0]
    triples = []
    for i in range(d):
        for j in range(d):
            c = _bits(tensor[i, j, :])
            if c:
                triples.append((1 << i, 1 << j, c))
    return Scheme(triples, d)


@dataclasses.dataclass
class WalkResult:
    start: int
    best: int
    best_triples: list[tuple[int, int, int]]
    flips: int
    flips_tried: int
    plus_transitions: int
    resets: int
    seconds: float
    reached: list[dict[str, Any]]        # when each new shortest length was first reached

    def summary(self) -> dict[str, Any]:
        return {"start": self.start, "best": self.best, "flips": self.flips, "flips_tried": self.flips_tried,
                "plus_transitions": self.plus_transitions, "resets": self.resets, "seconds": round(self.seconds, 1),
                "reached": self.reached}


def walk(scheme: Scheme, seconds: float, seed: int = 0, target: int | None = None, plateau: int = PLATEAU,
         slack: int = SLACK, reset: int = RESET, kept: int = KEPT, check: np.ndarray | None = None) -> WalkResult:
    """Walk the flip graph from ``scheme`` for ``seconds``; stop early at ``target`` products or fewer.

    ``check`` (a tensor) makes every new shortest decomposition be verified against it before it is
    kept; a failure raises, since it would mean a move that changes the tensor.
    """
    rng = random.Random(seed)
    current = scheme.copy()
    start = len(current)
    best = len(current)
    pool = [current.triples()]
    reached = [{"products": best, "flips": 0, "seconds": 0.0}]
    flips = tried = pluses = resets = 0
    since_change = since_best = 0
    began = time.perf_counter()
    deadline = began + seconds
    while True:
        if (tried & 0x3FF) == 0 and time.perf_counter() >= deadline:
            break
        if target is not None and best <= target:
            break
        tried += 1
        gone = current.flip(rng)
        if gone is None:                                      # no product shares a factor: no flip leaves this point
            if len(current) < best + slack and current.plus(rng):
                pluses += 1
            else:
                current = Scheme(rng.choice(pool), scheme.degree)
                resets += 1
            since_change = 0
            continue
        flips += 1
        since_change += 1
        since_best += 1
        if gone:
            since_change = 0
        n = len(current)
        if n < best:
            if check is not None and not np.array_equal(current.tensor(), check):
                raise AssertionError("a move changed the tensor")
            best = n
            pool = [current.triples()]
            since_best = 0
            reached.append({"products": n, "flips": flips, "seconds": round(time.perf_counter() - began, 1)})
        elif n == best and gone and len(pool) < kept:
            pool.append(current.triples())
        if since_best >= reset:
            current = Scheme(rng.choice(pool), scheme.degree)
            resets += 1
            since_best = since_change = 0
        elif since_change >= plateau and len(current) < best + slack:
            if current.plus(rng):
                pluses += 1
            since_change = 0
    return WalkResult(start=start, best=best, best_triples=pool[0], flips=flips, flips_tried=tried,
                      plus_transitions=pluses, resets=resets, seconds=time.perf_counter() - began, reached=reached)


def reconstruct(triples: Sequence[tuple[int, int, int]], degree: int) -> np.ndarray:
    """The tensor a list of products adds up to, entry by entry, without simplifying the list first."""
    out: np.ndarray = np.zeros((degree, degree, degree), dtype=np.uint8)
    for a, b, c in triples:
        out ^= np.einsum("i,j,k->ijk", _row(a, degree), _row(b, degree), _row(c, degree)).astype(np.uint8)
    return out


def verified(triples: Sequence[tuple[int, int, int]], tensor: np.ndarray | None = None) -> bool:
    """The products reproduce the tensor in every entry; for GF(2^12), also by the repository's own check."""
    tensor = build_multiplication_tensor() if tensor is None else tensor
    d = tensor.shape[0]
    rows = list(triples)
    if not rows or not np.array_equal(reconstruct(rows, d), tensor):
        return False
    if d != build_multiplication_tensor().shape[0]:
        return True
    decomposition = TensorDecomposition(rank=len(rows), U=np.stack([_row(a, d) for a, _, _ in rows]),
                                        V=np.stack([_row(b, d) for _, b, _ in rows]),
                                        W=np.stack([_row(c, d) for _, _, c in rows]), name="checked")
    ok, _ = verify_tensor_decomposition(tensor, decomposition)
    return bool(ok)
