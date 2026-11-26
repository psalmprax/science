#!/usr/bin/env python3
"""Exhaustive search for symmetric decompositions of GF(2^n) multiplication invariant under a Frobenius power.

Status: audit-only

**Development, not a result** (the GF(2^12) target of 41 products or fewer, decision 4; owner's word of
26 November to pursue it by any means). A symmetric decomposition writes ``xy = sum_t Tr(lam_t x) Tr(lam_t y) c_t``.
Over GF(2) the trace is to GF(2) and each term is one product; over GF(4) (``--over 4``, GF(2^12) as GF(4^6)) the
trace is to GF(4), each term is one GF(4) product, three over GF(2), and forms are taken up to GF(4)* scaling.
If the decomposition is invariant under ``phi = x -> x^(2^step)``, its forms fall into phi-orbits and each orbit's
outputs are fixed by one output ``c`` in a small subspace; given the orbits, the outputs follow by linear algebra.
So the decompositions of such a class can be enumerated completely: choose the orbits (``gf2_invariant_search.c``),
ask whether the multiplication lies in the span of what they can produce, and, where it does, solve for the
outputs, build the GF(2) decomposition and check every entry of the tensor (``extract``).

Measured (26 November, locally): over GF(2) with the full Frobenius, decompositions exist for GF(2^6) at 15 and
GF(2^8) at 24 (the best known counts) and none for GF(2^7) at 22, GF(2^9) at 30, GF(2^10) at 33, GF(2^12) at 41
or 42. Over GF(4) with x -> x^4, GF(4^6) has decompositions of 14 GF(4) products (42 over GF(2), each checked
exactly) and none of 13 or 12. The class invariant under x -> x^16 is larger; its search at 13 is the run.

    python scripts/gf2_invariant_search.py prepare --field 12 --over 4 --step 4 --out orb.bin
    python scripts/gf2_invariant_search.py run --file orb.bin --over 4 --step 4 --rank 13 --shard 0 --shards 60 --out DIR
    python scripts/gf2_invariant_search.py extract --field 12 --over 4 --step 2 --reps 103 131 138 195 1
    python scripts/gf2_invariant_search.py collect --out DIR
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers  # noqa: E402

TAG = "DEV-GF2-12-INVARIANT-SEARCH-2026-11"
SOURCE = ROOT / "scripts" / "gf2_invariant_search.c"
gf_mul = towers.gf_mul


class Field:
    """GF(2^n) with the modulus of ``gf2_towers.MODULI``, seen over its subfield GF(q), q = 2^k with k dividing n."""

    def __init__(self, n: int, q: int) -> None:
        k = q.bit_length() - 1
        if q != 1 << k or k < 1 or n % k:
            raise ValueError("q must be 2^k with k dividing n")
        self.n, self.q, self.mod = n, q, towers.MODULI[n]
        size = (1 << n) - 1
        prim = towers.primitive_elements(n)[0]
        g = self.pow(prim, size // (q - 1))
        self.scalars = [self.pow(g, e) for e in range(q - 1)]          # GF(q)*
        self.omega = self.pow(prim, size // 3) if q == 4 else 1
        self.degree = n // k                                             # degree over GF(q)
        self.cost = {2: 1, 4: 3, 8: 6, 16: 9, 64: 15}[q]                 # GF(2) products of one GF(q) product

    def pow(self, a: int, e: int) -> int:
        return towers.gpow(a, e, self.mod)

    def mul(self, a: int, b: int) -> int:
        return gf_mul(a, b, self.mod)

    def frob(self, a: int, k: int) -> int:                # x -> x^(2^k), a power of the absolute Frobenius
        for _ in range(k % self.n):
            a = self.mul(a, a)
        return a

    def trace(self, a: int) -> int:                       # to GF(q)
        s, t = 0, a
        for _ in range(self.degree):
            s ^= t
            t = self.pow(t, self.q)
        return s

    def canon(self, lam: int) -> int:
        return min(self.mul(s, lam) for s in self.scalars)


def orbits(field: Field, step: int) -> list[tuple[int, int]]:
    seen: set[int] = set()
    out = []
    for lam in range(1, 1 << field.n):
        cl = field.canon(lam)
        if cl in seen:
            continue
        orb = [cl]
        t = field.canon(field.frob(cl, step))
        while t != cl:
            orb.append(t)
            t = field.canon(field.frob(t, step))
        seen.update(orb)
        out.append((min(orb), len(orb)))
    out.sort(key=lambda o: (-o[1], o[0]))
    return out


def kernel(field: Field, lam: int, d: int, step: int) -> list[int]:
    """Outputs c with which the orbit of (lam, c) closes: c = alpha^2 phi^d(c), where phi^d(lam) = alpha lam."""
    lam_d = field.frob(lam, d * step)
    alpha = next(s for s in field.scalars if field.mul(s, lam) == lam_d)
    a2 = field.mul(alpha, alpha)
    imgs = [field.mul(a2, field.frob(1 << b, d * step)) ^ (1 << b) for b in range(field.n)]
    piv: dict[int, tuple[int, int]] = {}
    basis = []
    for b in range(field.n):
        img, tag = imgs[b], 1 << b
        for bit in sorted(piv, reverse=True):
            if img >> bit & 1:
                img ^= piv[bit][0]
                tag ^= piv[bit][1]
        if img:
            piv[img.bit_length() - 1] = (img, tag)
        else:
            basis.append(tag)
    return basis


def pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i, n)]


def contribution(field: Field, lam: int, d: int, step: int, c: int) -> int:
    """The bilinear map of the orbit of (lam, c) on the pairs (u^i, u^j), i <= j, as one integer of bits."""
    n = field.n
    lams = [field.frob(lam, j * step) for j in range(d)]
    cs = [field.frob(c, j * step) for j in range(d)]
    tr = [[field.trace(field.mul(lj, 1 << i)) for i in range(n)] for lj in lams]
    out = 0
    for p, (i, j) in enumerate(pairs(n)):
        val = 0
        for s in range(d):
            val ^= field.mul(field.mul(tr[s][i], tr[s][j]), cs[s])
        out |= val << (p * n)
    return out


def target(field: Field) -> int:
    n = field.n
    out = 0
    for p, (i, j) in enumerate(pairs(n)):
        out |= field.mul(1 << i, 1 << j) << (p * n)
    return out


def prepare(field: Field, step: int, out: Path) -> dict[int, int]:
    n = field.n
    words = (len(pairs(n)) * n + 63) // 64

    def pack(value: int) -> bytes:
        return struct.pack(f"<{words}Q", *[(value >> (64 * w)) & ((1 << 64) - 1) for w in range(words)])

    counts: dict[int, int] = {}
    orbs = orbits(field, step)
    with out.open("wb") as f:
        f.write(struct.pack("<iiii", n, len(orbs), words, len(pairs(n))))
        f.write(pack(target(field)))
        for lam, d in orbs:
            basis = kernel(field, lam, d, step)
            f.write(struct.pack("<iii", lam, d, len(basis)))
            for e in basis:
                f.write(pack(contribution(field, lam, d, step, e)))
            counts[d] = counts.get(d, 0) + 1
    return counts


def prepare_mixed(n: int, families: list[int], step: int, out: Path) -> dict[int, int]:
    """One orbit file for several subfields at once: an orbit's label is q * 100 + its size (q the subfield)."""
    words = (len(pairs(n)) * n + 63) // 64

    def pack(value: int) -> bytes:
        return struct.pack(f"<{words}Q", *[(value >> (64 * w)) & ((1 << 64) - 1) for w in range(words)])

    counts: dict[int, int] = {}
    entries = []
    for q in families:
        field = Field(n, q)
        for lam, d in orbits(field, step):
            basis = kernel(field, lam, d, step)
            entries.append((q * 100 + d, lam, [contribution(field, lam, d, step, e) for e in basis]))
            counts[q * 100 + d] = counts.get(q * 100 + d, 0) + 1
    entries.sort(key=lambda e: (-e[0], e[1]))
    with out.open("wb") as f:
        f.write(struct.pack("<iiii", n, len(entries), words, len(pairs(n))))
        f.write(pack(target(Field(n, 2))))
        for label, lam, vecs in entries:
            f.write(struct.pack("<iii", lam, label, len(vecs)))
            for v in vecs:
                f.write(pack(v))
    return counts


def mixed_profiles(counts: dict[int, int], budget: int, max_leaves: int) -> list[tuple[list[int], int]]:
    """Multisets of labels whose GF(2) cost (cost of the subfield times the orbit size) is exactly ``budget``."""
    costs = {2: 1, 4: 3, 8: 6, 16: 9, 64: 15}
    labels = sorted(counts, key=lambda lab: -costs[lab // 100] * (lab % 100))
    out: list[tuple[list[int], int]] = []

    def rec(left: int, i: int, acc: list[int]) -> None:
        if left == 0:
            n_leaves = 1
            for lab in set(acc):
                n_leaves *= comb(counts[lab], acc.count(lab))
            if n_leaves <= max_leaves:
                out.append((acc, n_leaves))
            return
        if i == len(labels):
            return
        lab = labels[i]
        unit = costs[lab // 100] * (lab % 100)
        for k in range(min(left // unit, counts[lab]), -1, -1):
            rec(left - k * unit, i + 1, acc + [lab] * k)

    rec(budget, 0, [])
    return out


def read_counts(path: Path) -> dict[int, int]:
    data = path.read_bytes()
    _, norb, words, _ = struct.unpack_from("<iiii", data, 0)
    off = 16 + 8 * words
    counts: dict[int, int] = {}
    for _ in range(norb):
        _, d, nv = struct.unpack_from("<iii", data, off)
        off += 12 + 8 * words * nv
        counts[d] = counts.get(d, 0) + 1
    return counts


def profiles(counts: dict[int, int], rank: int) -> list[list[int]]:
    sizes = sorted(counts, reverse=True)
    out: list[list[int]] = []

    def rec(left: int, i: int, acc: list[int]) -> None:
        if left == 0:
            out.append(acc)
            return
        if i == len(sizes):
            return
        d = sizes[i]
        for k in range(min(left // d, counts[d]), -1, -1):
            rec(left - k * d, i + 1, acc + [d] * k)

    rec(rank, 0, [])
    return out


def leaves(counts: dict[int, int], profile: list[int]) -> int:
    total = 1
    for d in set(profile):
        total *= comb(counts[d], profile.count(d))
    return total


def build(binary: Path) -> Path:
    subprocess.run(["gcc", "-O3", "-march=native", "-o", str(binary), str(SOURCE)], check=True)  # noqa: S603,S607 - the repository's own source
    return binary


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    counts = read_counts(args.file)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "search")
        for profile in profiles(counts, args.rank):
            done = subprocess.run([str(binary), str(args.file), "--shard", str(args.shard), str(args.shards), *map(str, profile)],  # noqa: S603 - built here
                                  capture_output=True, text=True, check=True)
            lines = done.stdout.strip().split("\n")
            found = [[int(v) for v in line.split(":")[1].split()] for line in lines if line.startswith("FOUND")]
            last = lines[-1]
            rows.append({"profile": profile, "leaves": int(last.split("leaves ")[1].split()[0]),
                         "found": int(last.split("found ")[1]), "reps": found})
            print(json.dumps(rows[-1]), flush=True)
    record = {"file": args.file.name, "over": args.over, "step": args.step, "rank": args.rank, "shard": args.shard,
              "shards": args.shards, "profiles": rows}
    (args.out / f"shard_{args.file.stem}_r{args.rank}_{args.shard}.json").write_text(json.dumps(record) + "\n")
    return 0


def gf_terms(field: Field, step: int, reps: list[int]) -> list[tuple[int, int]]:
    """The GF(q) products (lam, c), Tr_q(lam x) Tr_q(lam y) c, of the orbits of reps, the outputs solved."""
    orbit_size = {}
    for lam in reps:
        d, t = 1, field.canon(field.frob(lam, step))
        while t != lam:
            d += 1
            t = field.canon(field.frob(t, step))
        orbit_size[lam] = d
    columns, meta = [], []
    for lam in reps:
        for e in kernel(field, lam, orbit_size[lam], step):
            columns.append(contribution(field, lam, orbit_size[lam], step, e))
            meta.append((lam, e))
    piv: dict[int, tuple[int, int]] = {}
    for k, col in enumerate(columns):
        v, tag = col, 1 << k
        for bit in sorted(piv, reverse=True):
            if v >> bit & 1:
                v ^= piv[bit][0]
                tag ^= piv[bit][1]
        if v:
            piv[v.bit_length() - 1] = (v, tag)
    v, tag = target(field), 0
    for bit in sorted(piv, reverse=True):
        if v >> bit & 1:
            v ^= piv[bit][0]
            tag ^= piv[bit][1]
    if v:
        raise ValueError("the multiplication is not in the span of these orbits")
    c_of: dict[int, int] = {}
    for k, (lam, e) in enumerate(meta):
        if tag >> k & 1:
            c_of[lam] = c_of.get(lam, 0) ^ e
    return [(field.frob(lam, j * step), field.frob(c, j * step)) for lam, c in c_of.items() if c for j in range(orbit_size[lam])]


def extract(field: Field, step: int, reps: list[int]) -> tuple[list[tuple[int, int, int]], int]:
    """The GF(2) decomposition from orbit representatives, the outputs solved; and the number of GF(q) products."""
    n = field.n
    terms = gf_terms(field, step, reps)
    triples = []
    for lam, c in terms:
        if field.q == 2:
            form = sum(field.trace(field.mul(lam, 1 << i)) << i for i in range(n))
            triples.append((form, form, c))
            continue
        # Tr4(lam x) = u0 + u1 w; uv = (p0 + p1) + (p2 + p0) w with p0 = u0 v0, p1 = u1 v1, p2 = (u0 + u1)(v0 + v1)
        f0 = f1 = 0
        for i in range(n):
            t = field.trace(field.mul(lam, 1 << i))
            u1 = int(t not in (0, 1))
            u0 = int((t ^ (field.omega if u1 else 0)) == 1)
            f0 |= u0 << i
            f1 |= u1 << i
        omega = field.omega
        for form, out in ((f0, field.mul(1 ^ omega, c)), (f1, c), (f0 ^ f1, field.mul(omega, c))):
            if form and out:
                triples.append((form, form, out))
    return triples, len(terms)


def extract_mixed(step: int, labels: list[int], reps: list[int]) -> tuple[list[tuple[int, int, int]], int]:
    """As ``extract`` for a mixed choice: reps paired with the labels (q * 100 + size) in decreasing order, as the
    enumerator prints them; each GF(q) product becomes an optimal symmetric decomposition of GF(q) over GF(2)."""
    n = 12
    fields: dict[int, Field] = {}
    columns, meta = [], []
    for label, lam in zip(sorted(labels, reverse=True), reps, strict=True):
        q, d = label // 100, label % 100
        field = fields.setdefault(q, Field(n, q))
        for e in kernel(field, lam, d, step):
            columns.append(contribution(field, lam, d, step, e))
            meta.append((q, lam, d, e))
    piv: dict[int, tuple[int, int]] = {}
    for k, col in enumerate(columns):
        v, tag = col, 1 << k
        for bit in sorted(piv, reverse=True):
            if v >> bit & 1:
                v ^= piv[bit][0]
                tag ^= piv[bit][1]
        if v:
            piv[v.bit_length() - 1] = (v, tag)
    v, tag = target(Field(n, 2)), 0
    for bit in sorted(piv, reverse=True):
        if v >> bit & 1:
            v ^= piv[bit][0]
            tag ^= piv[bit][1]
    if v:
        raise ValueError("the multiplication is not in the span of these orbits")
    c_of: dict[tuple[int, int, int], int] = {}
    for k, (q, lam, d, e) in enumerate(meta):
        if tag >> k & 1:
            c_of[(q, lam, d)] = c_of.get((q, lam, d), 0) ^ e
    triples: list[tuple[int, int, int]] = []
    cost = 0
    for (q, lam, d), c in c_of.items():
        if not c:
            continue
        field = fields[q]
        cost += field.cost * d
        base = _subfield_base(field)
        for j in range(d):
            lj, cj = field.frob(lam, j * step), field.frob(c, j * step)
            if q == 2:
                form = sum(field.trace(field.mul(lj, 1 << i)) << i for i in range(n))
                triples.append((form, form, cj))
                continue
            sub_coord, base_terms = base
            k = q.bit_length() - 1
            masks = [0] * k
            for i in range(n):
                bits = sub_coord[field.trace(field.mul(lj, 1 << i))]
                for s in range(k):
                    masks[s] |= (bits >> s & 1) << i
            for form_a, _, delta in base_terms:
                a = 0
                for s in range(k):
                    if form_a >> s & 1:
                        a ^= masks[s]
                out = field.mul(delta, cj)
                if a and out:
                    triples.append((a, a, out))
    return triples, cost


def _subfield_base(field: Field) -> tuple[dict[int, int], list[tuple[int, int, int]]]:
    """Coordinates of GF(q) inside GF(2^n) in the basis of powers of its generator, and an optimal symmetric
    decomposition of GF(q) multiplication in that basis."""
    if field.q == 2:
        return {0: 0, 1: 1}, []
    k = field.q.bit_length() - 1
    g = field.pow(towers.primitive_elements(field.n)[0], ((1 << field.n) - 1) // (field.q - 1))
    sub_basis = [field.pow(g, i) for i in range(k)]
    sub_coord = {}
    for bits in range(1 << k):
        v = 0
        for i in range(k):
            if bits >> i & 1:
                v ^= sub_basis[i]
        sub_coord[v] = bits
    return sub_coord, towers.subfield_decompositions(k, sub_basis, sub_coord, field.mod, limit=1)[0]


def run_mixed(args: argparse.Namespace) -> int:
    """Every mixed profile of a GF(2) cost with between --min-leaves and --max-leaves choices, one shard."""
    args.out.mkdir(parents=True, exist_ok=True)
    counts = read_counts(args.file)
    chosen = [(p, n_leaves) for p, n_leaves in mixed_profiles(counts, args.budget, int(args.max_leaves)) if n_leaves > args.min_leaves]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "search")
        for profile, _ in sorted(chosen, key=lambda pl: pl[1]):
            done = subprocess.run([str(binary), str(args.file), "--shard", str(args.shard), str(args.shards), *map(str, profile)],  # noqa: S603 - built here
                                  capture_output=True, text=True, check=True)
            lines = done.stdout.strip().split("\n")
            found = [[int(v) for v in line.split(":")[1].split()] for line in lines if line.startswith("FOUND")]
            last = lines[-1]
            rows.append({"profile": sorted(profile, reverse=True), "leaves": int(last.split("leaves ")[1].split()[0]),
                         "found": int(last.split("found ")[1]), "reps": found})
            print(json.dumps(rows[-1]), flush=True)
    record = {"kind": "mixed", "file": args.file.name, "step": args.step, "budget": args.budget,
              "min_leaves": args.min_leaves, "max_leaves": int(args.max_leaves), "shard": args.shard,
              "shards": args.shards, "profiles": rows}
    (args.out / f"shard_{args.file.stem}_b{args.budget}_{args.shard}.json").write_text(json.dumps(record) + "\n")
    return 0

def collect(args: argparse.Namespace) -> int:
    shards = [json.loads(p.read_text()) for p in sorted(args.out.glob("shard_*.json"))]
    by_class: dict[str, Any] = {}
    for s in shards:
        if s.get("kind") == "mixed":
            key = f"{s['file']} budget {s['budget']} leaves {s['min_leaves']}-{s['max_leaves']}"
            entry = by_class.setdefault(key, {"kind": "mixed", "step": s["step"], "budget": s["budget"], "shards_read": 0,
                                              "shards": s["shards"], "profiles": 0, "leaves": 0, "found": 0, "reps": []})
        else:
            key = f"{s['file']} rank {s['rank']}"
            entry = by_class.setdefault(key, {"over": s["over"], "step": s["step"], "rank": s["rank"], "shards_read": 0,
                                              "shards": s["shards"], "leaves": 0, "found": 0, "reps": []})
        entry["shards_read"] += 1
        for row in s["profiles"]:
            entry["leaves"] += row["leaves"]
            entry["found"] += row["found"]
            if s.get("kind") == "mixed":
                entry["profiles"] = max(entry["profiles"], len(s["profiles"]))
                entry["reps"] += [{"labels": row["profile"], "reps": r} for r in row["reps"][:3]]
            else:
                entry["reps"] += row["reps"][:3]
    checked = []
    from apedp_dyn.quantum import flip_search
    for entry in by_class.values():
        for reps in entry["reps"][:5]:
            if entry.get("kind") == "mixed":
                triples, products = extract_mixed(entry["step"], reps["labels"], reps["reps"])
            else:
                triples, products = extract(Field(12, entry["over"]), entry["step"], reps)
            checked.append({"reps": reps, "products_over_gf2": len(triples), "terms_over_gfq": products,
                            "exact": flip_search.verified(triples), "triples": [list(t) for t in triples]})
    record = {"tag": TAG, "development_not_a_result": True, "classes": by_class, "checked": checked,
              "below_42": any(c["exact"] and c["products_over_gf2"] < 42 for c in checked)}
    (args.out / "invariant_search.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "checked"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("prepare", "prepare-mixed", "profiles", "run", "run-mixed", "extract", "collect"))
    parser.add_argument("--families", type=int, nargs="*", default=[4, 2])
    parser.add_argument("--budget", type=int, default=41)
    parser.add_argument("--min-leaves", type=float, default=0)
    parser.add_argument("--max-leaves", type=float, default=1e7)
    parser.add_argument("--field", type=int, default=12)
    parser.add_argument("--over", type=int, choices=(2, 4), default=4)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--reps", type=int, nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(Field(args.field, args.over), args.step, args.out)))
        return 0
    if args.command == "prepare-mixed":
        print(json.dumps(prepare_mixed(args.field, args.families, args.step, args.out)))
        return 0
    if args.command == "run-mixed":
        return run_mixed(args)
    if args.command == "profiles":
        counts = read_counts(args.file)
        for p in profiles(counts, args.rank):
            print(" ".join(map(str, p)), leaves(counts, p))
        return 0
    if args.command == "run":
        return run(args)
    if args.command == "extract":
        triples, products = extract(Field(args.field, args.over), args.step, args.reps)
        from apedp_dyn.quantum import flip_search
        ok = flip_search.verified(triples) if args.field == 12 else towers.exact(triples, args.field)
        print(json.dumps({"terms_over_gfq": products, "products_over_gf2": len(triples), "exact": ok}))
        return 0
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
