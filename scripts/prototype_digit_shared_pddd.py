#!/usr/bin/env python3
"""Digit-by-digit accumulator: per-digit plans versus one shared PDDD.

Status: experimental

`scripts/prototype_bitserial_sqrt.py` phases each output digit as it appears, and
its width grows because every digit's function is compiled over that digit's full
input prefix. `boolean_plan` is already a reduced, node-shared decision diagram —
not a lookup table — so the interesting question is not "PDDD instead of a table"
but **per-digit plans versus one multi-output plan shared across all digits**.

This script answers that by building both shapes from the same truth data and
measuring both with `kickmix`:

* ``per_digit``   — one reusable digit qubit, `k` plans, each over its own prefix
  (the shipped prototype). Width = `2k + transients`.
* ``shared``      — all `k` digits computed at once from a single multi-output
  plan over the whole `2k`-bit address, phased together, then uncomputed.
  Width = `2k + nodes + 1 + k`.

If sharing helped, ``shared`` would be the smaller circuit at full width. It is
reported here rather than assumed.

Usage:
    .venv-kickmix/bin/python scripts/prototype_digit_shared_pddd.py [--max-k 9] [--json]
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(
    0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix")
)

import numpy as np  # noqa: E402
import kickmix as km  # noqa: E402
from prototype_bitserial_sqrt import build_bitserial  # noqa: E402
from test_shared_pddd_circuit import build_shared_plan, restore_shared_boolean  # noqa: E402


def sqrt_bits(k: int) -> tuple[np.ndarray, np.ndarray]:
    """`w` and `floor(sqrt(w))` over the whole `2k`-bit domain, integer-exact."""
    n = 2 * k
    w = np.arange(1 << n, dtype=np.int64)
    r = np.sqrt(w.astype(np.float64)).astype(np.int64)
    for delta in (-1, 0, 1):
        cand = r + delta
        good = (cand >= 0) & (cand * cand <= w) & ((cand + 1) * (cand + 1) > w)
        r = np.where(good, cand, r)
    return w, r


def build_shared(k: int) -> tuple[km.Circuit, dict]:
    """All digits from one multi-output plan over the full address, phased together."""
    n = 2 * k
    _, r = sqrt_bits(k)
    plan = build_shared_plan(
        {f"d{i}": ((r >> i) & 1).astype(np.uint8) for i in range(k)},
        order=tuple(range(n)),
        polarity=0,
    )
    b = km.CircuitBuilder()
    x = b.create_quantum_register(n, name="x")
    targets = b.alloc_qubits(k)
    restore_shared_boolean(b, plan, list(x), {f"d{i}": targets[i] for i in range(k)})
    for i in range(k):
        b.z_pow(targets[i], Fraction(2) ** (i - k + 1))
    restore_shared_boolean(b, plan, list(x), {f"d{i}": targets[i] for i in range(k)})
    circuit = b.finish_circuit()
    stats = circuit.operation_statistics()
    return circuit, {
        "nodes": len(plan["nodes"]),
        "qubits": circuit.num_qubits,
        "toffoli": int(stats.num_toffoli_like),
        "rotations": int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
    }


def main() -> int:
    max_k = 9
    if "--max-k" in sys.argv:
        max_k = int(sys.argv[sys.argv.index("--max-k") + 1])

    rows = []
    print(f"{'k':>2} {'w bits':>6} | {'per-digit Q':>11} {'T':>7} {'nodes':>6} "
          f"| {'shared Q':>8} {'T':>7} {'nodes':>6} {'build s':>7}")
    for k in range(3, max_k + 1):
        t0 = time.perf_counter()
        _, per = build_bitserial(k)
        per_nodes = sum(1 for _ in range(0))  # per-digit plans are not exposed; see note
        if k <= max_k:
            try:
                _, shared = build_shared(k)
            except MemoryError:
                shared = {"nodes": None, "qubits": None, "toffoli": None, "rotations": None}
        seconds = time.perf_counter() - t0
        rows.append({"k": k, "bits": 2 * k, "per_digit": per, "shared": shared})
        print(f"{k:2d} {2*k:6d} | {per['qubits']:11d} {per['toffoli']:7d} {'-':>6} "
              f"| {str(shared['qubits']):>8} {str(shared['toffoli']):>7} "
              f"{str(shared['nodes']):>6} {seconds:7.1f}")

    print("\nper-digit plans are compiled per digit inside build_bitserial; the shared plan is")
    print("one multi-output PDDD over the same address, so its node count is comparable to the")
    print("sum the per-digit shape would spend if its prefixes were widened to the whole address.")
    if "--json" in sys.argv:
        print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
