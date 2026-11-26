#!/usr/bin/env python3
"""Tower decompositions of multiplication in GF(2^n) over GF(2), in many variants, as starting points.

Status: audit-only

**Development.** The best decompositions known for GF(2^8), GF(2^9), GF(2^10) and GF(2^12) (24, 30, 33
and 42 products) are towers: multiplication over a subfield GF(2^k) by interpolation at 2m - 1 points
(n = k m), each product in GF(2^k) done by an optimal decomposition of GF(2^k) over GF(2). A walk of
the flip graph cannot leave the 42 of ``tower_reference.py`` by flips (no two of its products share a
factor), so starting points that differ, not by flips but by construction, are worth more than more
time from one: this builds towers over GF(2^k) for any choice of the 2m - 1 points among the
subfield and infinity, of the element ``theta`` that generates GF(2^n) over the subfield, and of the
subfield's own decomposition.

Every tower is checked against every entry of its tensor before it is returned. Bit ``i`` of a factor is
the coefficient of ``u^i`` in the polynomial basis of ``MODULI[n]`` (for n = 12 the repository's own
modulus), the convention of ``scripts/search_gf2_mul_flip.py``.

    python scripts/gf2_towers.py --field 12 --sub 3 --variant 5 > start.txt
"""
from __future__ import annotations

import argparse
import itertools
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum.field_tensor_search import MODULUS_INT, gf_mul  # noqa: E402

MODULI = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 7: 0b10000011, 8: 0b100011011,
          9: 0b1000010001, 10: 0b10000001001, 11: 0b100000000101, 12: MODULUS_INT}
OPTIMAL = {2: 3, 3: 6, 4: 9}   #: least products for GF(4), GF(8), GF(16) over GF(2) (Barbulescu et al. 2012)


def gpow(a: int, e: int, mod: int) -> int:
    r = 1
    while e:
        if e & 1:
            r = gf_mul(r, a, mod)
        a = gf_mul(a, a, mod)
        e >>= 1
    return r


def _primes(x: int) -> list[int]:
    out, p = [], 2
    while p * p <= x:
        if x % p == 0:
            out.append(p)
            while x % p == 0:
                x //= p
        p += 1
    if x > 1:
        out.append(x)
    return out


@cache
def primitive_elements(n: int) -> tuple[int, ...]:
    mod, size = MODULI[n], (1 << n) - 1
    return tuple(g for g in range(2, 1 << n) if all(gpow(g, size // p, mod) != 1 for p in _primes(size)))


def exact(triples: list[tuple[int, int, int]], n: int) -> bool:
    """Every product u^i u^j is what the products give: all n**3 entries of the tensor."""
    mod = MODULI[n]
    for i in range(n):
        for j in range(n):
            z = 0
            for a, b, c in triples:
                if a >> i & 1 and b >> j & 1:
                    z ^= c
            if z != gf_mul(1 << i, 1 << j, mod):
                return False
    return True


def _solve_gf2(rows: list[list[int]], rhs: list[int]) -> list[int] | None:
    """``sum_s rows[r][s] x_s = rhs[r]`` with each ``x_s`` and ``rhs[r]`` a vector of bits (an int)."""
    nvar = len(rows[0])
    aug = [[*row, v] for row, v in zip(rows, rhs, strict=True)]
    pivots, r = [], 0
    for col in range(nvar):
        p = next((i for i in range(r, len(aug)) if aug[i][col]), None)
        if p is None:
            continue
        aug[r], aug[p] = aug[p], aug[r]
        for i in range(len(aug)):
            if i != r and aug[i][col]:
                aug[i] = [x ^ y for x, y in zip(aug[i], aug[r], strict=True)]
        pivots.append(col)
        r += 1
    if any(aug[i][-1] for i in range(r, len(aug))):
        return None
    x = [0] * nvar
    for i, col in enumerate(pivots):
        x[col] = aug[i][-1]
    return x


def subfield_decompositions(k: int, basis: list[int], coord: dict[int, int], mod: int, limit: int = 8) -> list[list[tuple[int, int, int]]]:
    """Symmetric optimal decompositions of GF(2^k) in ``basis`` (``(form, form, output)``), up to ``limit`` of them."""
    target = OPTIMAL[k]
    prods = {(i, j): coord[gf_mul(basis[i], basis[j], mod)] for i in range(k) for j in range(k)}
    found = []
    for chosen in itertools.combinations(range(1, 1 << k), target):
        rows = [[(form >> i) & (form >> j) & 1 for form in chosen] for i in range(k) for j in range(i, k)]
        rhs = [prods[i, j] for i in range(k) for j in range(i, k)]
        sol = _solve_gf2(rows, rhs)
        if sol is None or not all(sol):
            continue
        out = []
        for form, d in zip(chosen, sol, strict=True):
            delta = 0
            for bit in range(k):
                if d >> bit & 1:
                    delta ^= basis[bit]
            out.append((form, form, delta))
        found.append(out)
        if len(found) >= limit:
            break
    return found


def _invert(matrix: list[list[int]], mod: int, n: int) -> list[list[int]]:
    size = len(matrix)
    a = [row[:] + [int(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        p = next(i for i in range(col, size) if a[i][col])
        a[col], a[p] = a[p], a[col]
        iv = gpow(a[col][col], (1 << n) - 2, mod)
        a[col] = [gf_mul(v, iv, mod) for v in a[col]]
        for i in range(size):
            if i != col and a[i][col]:
                f = a[i][col]
                a[i] = [v ^ gf_mul(f, w, mod) for v, w in zip(a[i], a[col], strict=True)]
    return [row[size:] for row in a]


def tower(n: int, k: int, points: tuple[int, ...] | None = None, theta_index: int = 0, base_index: int = 0) -> list[tuple[int, int, int]]:
    """A tower decomposition with 2m - 1 interpolation points over GF(2^k) (n = k m).

    ``points`` index the 2^k + 1 places of the subfield's projective line (0 .. 2^k - 1: the subfield's
    elements in the order zero, then powers of its generator; 2^k: infinity); by default the first
    2m - 1. ``theta_index`` chooses the generator of GF(2^n) over the subfield among the primitive
    elements; ``base_index`` the subfield's decomposition.
    """
    if n % k:
        raise ValueError("k must divide n")
    mod, m = MODULI[n], n // k
    count = 2 * m - 1
    if count > (1 << k) + 1:
        raise ValueError("not enough points on the subfield's line")
    prim = primitive_elements(n)
    g = gpow(prim[0], ((1 << n) - 1) // ((1 << k) - 1), mod)
    field = [0] + [gpow(g, e, mod) for e in range((1 << k) - 1)]
    points = tuple(range(count)) if points is None else tuple(points)
    if len(points) != count or len(set(points)) != count:
        raise ValueError("2m - 1 distinct points")
    theta = prim[theta_index % len(prim)]
    powers = [gpow(theta, i, mod) for i in range(2 * m - 1)]
    coord: dict[int, tuple[int, ...]] = {}
    for cs in itertools.product(field, repeat=m):
        v = 0
        for c, p in zip(cs, powers[:m], strict=True):
            v ^= gf_mul(c, p, mod)
        coord[v] = cs
    if len(coord) != 1 << n:
        raise ValueError("theta does not generate GF(2^n) over the subfield")
    infinity = 1 << k

    def row(place: int) -> list[int]:
        if place == infinity:
            return [0] * (count - 1) + [1]
        t = field[place]
        return [1 if i == 0 else (gpow(t, i, mod) if t else 0) for i in range(count)]

    vinv = _invert([row(p) for p in points], mod, n)
    gamma = []
    for e in range(count):
        s = 0
        for kk in range(count):
            s ^= gf_mul(vinv[kk][e], powers[kk], mod)
        gamma.append(s)

    def evaluate(x: int, place: int) -> int:
        cs = coord[x]
        if place == infinity:
            return cs[m - 1]
        t = field[place]
        acc = 0
        for i, c in enumerate(cs):
            acc ^= gf_mul(c, 1 if i == 0 else (gpow(t, i, mod) if t else 0), mod)
        return acc

    sub_basis = [gpow(g, i, mod) for i in range(k)]
    sub_coord = {}
    for bits in range(1 << k):
        v = 0
        for i in range(k):
            if bits >> i & 1:
                v ^= sub_basis[i]
        sub_coord[v] = bits
    bases = subfield_decompositions(k, sub_basis, sub_coord, mod)
    base = bases[base_index % len(bases)]
    triples = []
    for e, place in enumerate(points):
        values = [sub_coord[evaluate(1 << bit, place)] for bit in range(n)]
        for form_a, form_b, delta in base:
            a = sum(1 << bit for bit in range(n) if bin(form_a & values[bit]).count("1") & 1)
            b = sum(1 << bit for bit in range(n) if bin(form_b & values[bit]).count("1") & 1)
            triples.append((a, b, gf_mul(delta, gamma[e], mod)))
    if not exact(triples, n):
        raise AssertionError("a tower that is not exact")
    return triples


def variants(n: int, k: int) -> list[tuple[tuple[int, ...], int, int]]:
    """(points, theta_index, base_index) for the variant numbers 0, 1, 2, ...: every choice of points first."""
    m = n // k
    places = range((1 << k) + 1)
    return [(pts, t, b) for t in range(3) for b in range(2) for pts in itertools.combinations(places, 2 * m - 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--sub", type=int, required=True, help="k: the subfield GF(2^k)")
    parser.add_argument("--variant", type=int, default=0)
    args = parser.parse_args(argv)
    pts, t, b = variants(args.field, args.sub)[args.variant % len(variants(args.field, args.sub))]
    triples = tower(args.field, args.sub, pts, t, b)
    print(args.field, len(triples))
    for a, b2, c in triples:
        print(a, b2, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
