#!/usr/bin/env python3
"""Build and check the Benchmark III angle-circuit family at a chosen ``(brot, J)``.

This is the tool the notebook runs. It builds the same XOR-oracle family the
competition's reference builder writes (a segment-tree QROM over the selection
register), but it takes ``brot`` and ``J`` as arguments instead of fixing them,
so the qubit/Toffoli trade-off at the 128-qubit cap can be measured rather than
asserted.

Why the trade-off is worth a script at all: the Toffoli count of a QROM family is
set by the number of tree nodes, not by the word width, so precision is nearly
free in ``T``. What costs qubits is ``J * brot``. The cap therefore binds the
*product*, and since ``T`` scales with the circuit count ``ceil(75 / J)``, the
cheapest family is the largest ``J`` that still fits:

    selected + J*brot + (selected - 1) <= 128,      selected = 12 for 3,976 rows

At ``brot = 12`` that gives ``J = 8`` (119 qubits, ``K = 10`` circuits); at ``brot = 10``
it gives ``J = 10`` (123 qubits, five spare) and ``K = 8`` circuits -- two fewer trees,
so the family costs two trees less in ``T`` for one bit less of precision.

``--compressed`` stores 9 of the 10 delta bits and rebuilds the tenth from the
Givens sign invariant (``bit 9 == bit 8`` for any angle with ``cos(theta) > 0``),
which costs one extra ``CX`` per set bit and nothing else. It is checked, not
assumed: the invariant is verified against the real table before the family is
built, and the build refuses if it does not hold.

Checked against the competition's own ``scoreinfra.circuits`` -- register
contract, word recovery on every address, phase, workspace and the reverse
pass -- see ``--verify``.

    python3.13 scripts/build_dfthcsos_sub128.py --npz path/to/submission.npz \
        --brot 10 --batch-size 10 --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
VERIFIERS = BUNDLE / "verifiers"
DEFAULT_NPZ = BUNDLE / "sample_submission" / "dfthcsos" / "submission.npz"
QUBIT_CAP = 128


def _scoreinfra() -> Any:
    """Import the competition's circuit module from the bundle."""
    if str(VERIFIERS) not in sys.path:
        sys.path.insert(0, str(VERIFIERS))
    from dfthcsos.scoreinfra import circuits as C  # noqa: PLC0415

    return C


def requantise(words: np.ndarray, src_bits: int, dst_bits: int) -> np.ndarray:
    """Round-trip a word table to ``dst_bits``, preserving ``theta/pi = word/2**(b-1)``."""
    words = np.asarray(words, dtype=np.int64)
    mask = (1 << dst_bits) - 1
    if dst_bits >= src_bits:
        return (words << (dst_bits - src_bits)) & mask
    shift = src_bits - dst_bits
    rounded = (words + (1 << (shift - 1))) >> shift
    # A word is a turn count modulo one turn: 4095 rounds to 1024, which is 0 at
    # brot=10. Without the mask the caller gets a word one past the register width.
    return rounded & ((1 << dst_bits) - 1)


def sign_invariant_fraction(words: np.ndarray, brot: int) -> float:
    """Fraction of words whose top two bits agree (the Givens sign invariant)."""
    top = (words >> (brot - 1)) & 1
    nxt = (words >> (brot - 2)) & 1
    return float((top == nxt).mean())


def batch_deltas(words: np.ndarray, batch_size: int) -> np.ndarray:
    """The ``(K, A, J)`` tables: ``delta[0] = batch[0]``, ``delta[k] = batch[k] ^ batch[k-1]``."""
    num_vectors, num_angles = words.shape
    num_batches = max(1, ceil(num_angles / batch_size))
    padded = np.zeros((num_vectors, num_batches * batch_size), dtype=np.int64)
    padded[:, :num_angles] = words
    batches = padded.reshape(num_vectors, num_batches, batch_size).transpose(1, 0, 2)
    deltas = batches.copy()
    deltas[1:] ^= batches[:-1]
    return deltas


def count_breaks(delta: np.ndarray) -> int:
    """How many multi-row ranges the vector-equality early-break collapses.

    The builder's break is `np.all(delta[lo:hi] == delta[lo])`: a range loads once when
    every row in it is the same word vector. This counts those ranges per circuit, which is
    the only way to say whether a read *could* be cheaper on this table -- on the shipped
    table the answer is zero, and that is why the family costs `rows - 2` per circuit.
    """
    delta = np.asarray(delta)
    num_rows = delta.shape[0]
    num_bits = (num_rows - 1).bit_length()
    found = 0

    def node(lo: int, hi: int) -> None:
        nonlocal found
        if lo >= num_rows:
            return
        if hi <= num_rows and np.all(delta[lo:hi] == delta[lo]):
            if hi - lo > 1:
                found += 1
            return
        mid = (lo + hi) // 2
        if num_rows <= mid:
            node(lo, mid)
            return
        node(lo, mid)
        node(mid, hi)

    half = 1 << (num_bits - 1)
    node(0, half)
    node(half, 2 * half)
    return found


def build_delta_circuit(delta: np.ndarray, brot: int, *, compressed: bool,
                        ancilla_mode: str = "per-level") -> Any:
    """One XOR oracle as a segment-tree QROM, with optional 9-bit delta payload."""
    import kickmix as km  # noqa: PLC0415 - only importable with the bundle on the path

    C = _scoreinfra()
    delta = np.asarray(delta, dtype=np.int64)
    num_rows, batch_size = delta.shape
    num_bits = (num_rows - 1).bit_length()
    stored_bits = brot - 1 if compressed else brot  # noqa: F841 - see load()

    builder = km.CircuitBuilder()
    selection = builder.create_quantum_register(num_bits, C.SELECTION_REGISTER)
    words = [
        builder.create_quantum_register(brot, C.word_register(j)) for j in range(batch_size)
    ]
    if ancilla_mode == "shared":
        # One qubit reused at every level, to test the claim that the tree's ancillas
        # are shareable. A parent's control has to survive its entire left subtree, so
        # a child writing the same qubit destroys it; the checker is what decides.
        ancillas = builder.alloc_qubits(1) if num_bits > 1 else []
    else:
        ancillas = builder.alloc_qubits(num_bits - 1) if num_bits > 1 else []
    bits = [selection[i] for i in reversed(range(num_bits))]

    def load(row: int, control: Any) -> None:
        for j in range(batch_size):
            word = int(delta[row, j])
            for bit in reversed(range(stored_bits)):
                if (word >> bit) & 1:
                    builder.cx(control, words[j][bit])
                    if compressed and bit == brot - 2:
                        # delta bit 8 also lands on bit 9: the invariant bit9 == bit8
                        # holds before and after every load, so this restores it.
                        builder.cx(control, words[j][brot - 1])

    def node(control: Any, level: int, lo: int, hi: int) -> None:
        if lo >= num_rows:
            return
        # Leaf-level fusion (replacing the deepest AND-ancillas with multi-controlled X
        # loads) was implemented and measured, then removed. It frees exactly one ancilla
        # per fused level and costs about 46x the family's Toffolis, because every set
        # target bit becomes its own Toffoli instead of a free CX; and the construction
        # tried here did not reproduce the table. See docs/DFTHCSOS_SUB128_SCOPE.md.
        # Early-break on *vector* equality rather than on a scalar. Comparing every row
        # to `delta[lo]` is a strict generalisation of comparing every cell to
        # `delta[lo, 0]`: the scalar form only fires when the whole block is one repeated
        # 0/1 pattern, while the vector form also catches blocks that are uniform with a
        # non-constant word vector. A block that satisfies the old condition satisfies
        # this one, so the change can only remove nodes, never add them.
        if hi <= num_rows and (hi - lo == 1 or np.all(delta[lo:hi] == delta[lo])):
            load(lo, control)
            return
        mid = (lo + hi) // 2
        if num_rows <= mid:
            node(control, level + 1, lo, mid)
            return
        bit = bits[level]
        ancilla = ancillas[0] if ancilla_mode == "shared" else ancillas[level - 1]
        builder.x(bit)
        builder.init_and(control, bit, ancilla)
        builder.x(bit)
        node(ancilla, level + 1, lo, mid)
        builder.cx(control, ancilla)
        node(ancilla, level + 1, mid, hi)
        builder.del_and(control, bit, ancilla)

    half = 1 << (num_bits - 1)
    builder.x(bits[0])
    node(bits[0], 1, 0, half)
    builder.x(bits[0])
    node(bits[0], 1, half, 2 * half)
    return builder.finish_circuit()


def build_family(words: np.ndarray, brot: int, batch_size: int, *, compressed: bool,
                 ancilla_mode: str = "per-level"):
    return tuple(
        build_delta_circuit(delta, brot, compressed=compressed, ancilla_mode=ancilla_mode)
        for delta in batch_deltas(words, batch_size)
    )


def report(words12: np.ndarray, brot: int, batch_size: int, *, compressed: bool,
           ancilla_mode: str = "per-level") -> dict:
    C = _scoreinfra()
    words = requantise(words12, 12, brot)
    invariant = sign_invariant_fraction(words, brot)
    if compressed and invariant < 1.0:
        raise SystemExit(
            f"--compressed needs bit{brot-1} == bit{brot-2} on every word; "
            f"it holds on {invariant:.4%} of this table, so the payload cannot drop a bit"
        )
    circuits = build_family(words, brot, batch_size, compressed=compressed,
                            ancilla_mode=ancilla_mode)
    costs = C.circuit_costs(circuits)
    peak = max(c["num_qubits"] for c in costs)
    return {
        "brot": brot,
        "batch_size": batch_size,
        "circuits_K": len(circuits),
        "compressed_delta": compressed,
        "ancilla_mode": ancilla_mode,
        "sign_invariant_fraction": invariant,
        "peak_qubits": peak,
        "qubit_cap": QUBIT_CAP,
        "headroom": QUBIT_CAP - peak,
        "toffoli_per_circuit": [c["max_magic"] for c in costs],
        "toffoli_total_two_passes": 2 * sum(c["max_magic"] for c in costs),
        "reaction_depth_max": max(c["reaction_depth"] for c in costs),
        "selection_qubits": (words.shape[0] - 1).bit_length(),
        "circuits": circuits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help="submission.npz holding angle_words (default: the bundle's sample)")
    parser.add_argument("--brot", type=int, default=10, help="bits per angle word (default: 10)")
    parser.add_argument("--batch-size", type=int, default=10, help="words per circuit, J (default: 10)")
    parser.add_argument("--compressed", action="store_true",
                        help="store brot-1 delta bits and rebuild the top bit from the sign invariant")
    parser.add_argument("--ancilla-mode", choices=("per-level", "shared"), default="per-level",
                        help="tree ancillas: one per level (correct), or one shared by every "
                             "level (tests whether the tree's controls are shareable)")
    parser.add_argument("--out", type=Path, help="directory to write angle_k_<k>.kmb into")
    parser.add_argument("--verify", action="store_true",
                        help="run the competition's word-recovery check on every address")
    parser.add_argument("--shots-per-address", type=int, default=8,
                        help="shots per address for --verify (default: 8)")
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument("--break-audit", action="store_true",
                        help="count the ranges the early-break collapses in each circuit")
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as archive:
        words12 = np.array(archive["angle_words"], dtype=np.int64)

    result = report(words12, args.brot, args.batch_size, compressed=args.compressed,
                    ancilla_mode=args.ancilla_mode)
    circuits = result.pop("circuits")
    print(json.dumps(result, indent=2))

    if args.break_audit:
        words = requantise(words12, 12, args.brot)
        counts = [count_breaks(delta) for delta in batch_deltas(words, args.batch_size)]
        print(f"collapsible ranges per circuit: {counts}")
        print(f"total: {sum(counts)} over {len(counts)} circuits "
              f"({sum(counts)} nodes would be removed if any existed)")

    if args.out:
        C = _scoreinfra()
        args.out.mkdir(parents=True, exist_ok=True)
        for k, circuit in enumerate(circuits):
            circuit.to_file(C.kmb_path(args.out, k), format="kmb")
        print(f"wrote {len(circuits)} circuits to {args.out}")

    if args.verify:
        C = _scoreinfra()
        words = requantise(words12, 12, args.brot)
        recovered, layout, toffoli = C.recover_words_from_circuits(
            circuits,
            num_vectors=words.shape[0],
            num_angles=words.shape[1],
            shots_per_address=args.shots_per_address,
            seed=20261102,
        )
        exact = bool(np.array_equal(recovered, words))
        print(json.dumps({
            "verify": "PASS" if exact else "FAIL",
            "words_exact": exact,
            "layout_brot": layout.brot,
            "layout_J": layout.batch_size,
            "toffoli_forward_reverse": toffoli,
        }, indent=2))
        if not exact:
            return 1

    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
