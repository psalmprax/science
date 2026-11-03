#!/usr/bin/env python3
"""Isolate the derived `_update_flag` predicate from the decoder around it.

Status: audit-only

`dqi_recycled_equivalence.json` records that the substituted `_update_flag` decodes wrongly on
52 of 64 shots while the arithmetic identity `n_a == n_b  <=>  n_A == n + 1 - n_b` holds over
every pair. Those two facts can both be true: the identity can hold and the *emitted cascade*
still fail to compute it. This script separates the two.

It builds one small circuit holding the three registers the predicate reads — `A`, `B`, and a
register loaded with `n + 1 - A` — and flips two flags: one from the reference comparison
`a == B`, one from the derived loop over `B` with `A == n + 1 - B`. It then simulates every
`(A, B)` basis pair and reports where the two disagree. If the derived block is exact here, the
decoder's failure is not in this predicate's formulation; if it is not, the first disagreeing
pair says which address the cascade gets wrong.

Usage:
    PYTHONPATH=.../google-verifiable-quantum-circuit-compilation/kickmix \
        /dev/shm/qvcc-venv313/bin/python scripts/diagnose_derived_flag_predicate.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))

import dqi_reference_circuit as DR  # noqa: E402

import kickmix as km  # noqa: E402

N = 70
WIDTH = 8
LIMIT = N + 1


def bits(values: np.ndarray, width: int) -> np.ndarray:
    """Little-endian bits, shape `(width, shots)`."""
    return np.stack([((values >> i) & 1).astype(bool) for i in range(width)])


def build_pair(*, qubit_control: bool):
    builder = km.CircuitBuilder()
    A = builder.create_quantum_register(WIDTH, name="A")
    B = builder.create_quantum_register(WIDTH, name="B")
    a = builder.create_quantum_register(WIDTH, name="a")
    flag_ref = builder.create_quantum_register(1, name="fref")
    flag_der = builder.create_quantum_register(1, name="fder")

    builder.iadd(LIMIT, target=a)
    builder.isub(A, target=a)  # a = n + 1 - A, the invariant the substitution relies on

    # The decoder never opens the pass with `control=True`: the flag predicate is gated on the
    # counter decode, so the iteration is opened on a qubit that holds 1 exactly when the
    # sub-step is active. A pass whose semantics differ under a qubit control would be exact in
    # the test and wrong in the decoder, which is the difference this variant isolates.
    control = True
    if qubit_control:
        gate = builder.create_quantum_register(1, name="gate")
        builder.x(gate[0])
        control = gate[0]

    builder.flip_if_equal(a, B, target=flag_ref[0], control=control)

    with builder.unary_iteration(B, control=control) as cur_b:
        for k in range(LIMIT + 1):
            hit = DR._active(cur_b, k, LIMIT)
            if hit is False:
                continue
            builder.flip_if_equal(A, LIMIT - k, target=flag_der[0], control=hit)

    return builder.finish_circuit(), A, B, a, flag_ref, flag_der


def check(qubit_control: bool) -> int:
    values = np.arange(LIMIT + 1, dtype=np.int64)
    shots = [(A, B) for A in values for B in values]
    A_vals = np.array([s[0] for s in shots], dtype=np.int64)
    B_vals = np.array([s[1] for s in shots], dtype=np.int64)

    circuit, A, B, a, flag_ref, flag_der = build_pair(qubit_control=qubit_control)
    sim = km.Simulator(batch_size=len(shots), debug_print=False, counting_operations=False, seed=7)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array(A), bits(A_vals, WIDTH))
    sim.write_across_shots(km.array(B), bits(B_vals, WIDTH))
    sim.do(circuit)

    got_ref = sim.read_across_shots(km.array(flag_ref))[0]
    got_der = sim.read_across_shots(km.array(flag_der))[0]
    weights = (1 << np.arange(WIDTH)).astype(np.int64)
    loaded_a = sim.read_across_shots(km.array(a)).astype(np.int64).T @ weights

    expected = A_vals == LIMIT - B_vals
    mismatches = np.flatnonzero(got_ref != got_der)
    ref_wrong = np.flatnonzero(got_ref != expected)
    der_wrong = np.flatnonzero(got_der != expected)
    load_wrong = np.flatnonzero(loaded_a != (LIMIT - A_vals) % (1 << WIDTH))

    mode = "qubit control" if qubit_control else "control=True"
    print(f"== {mode}")
    print(f"pairs checked            : {len(shots)}")
    print(f"a == n+1-A                : {len(loaded_a) - len(load_wrong)} good, {len(load_wrong)} bad")
    print(f"reference flag wrong      : {len(ref_wrong)}")
    print(f"derived flag wrong        : {len(der_wrong)}")
    print(f"reference vs derived diff : {len(mismatches)}")
    for idx in list(mismatches[:8]) + list(der_wrong[:8]):
        print(
            f"  A={A_vals[idx]:3d} B={B_vals[idx]:3d} a={loaded_a[idx]:3d} "
            f"ref={int(got_ref[idx])} der={int(got_der[idx])} expected={int(expected[idx])}"
        )
    return len(mismatches)


def main() -> int:
    bad = check(False) + check(True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
