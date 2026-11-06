#!/usr/bin/env python3
"""Prototype a chunked, dirty-ancilla QROM read and measure it against the tree.

The tree read costs one Toffoli per internal node and amortises it over every set payload
bit of the addresses it covers: at `J = 4`, `brot = 10` that is up to 40 CX loads per
Toffoli. A scheme that computes one match per *address* cannot amortise that way, so this
script exists to measure the alternatives rather than assume either way.

Variants built here, all over the real angle table:

``tree``
    The shipped read: a segment tree, one ``init_and`` per node, loads by ``cx``.
``chunked``
    Split the address space into blocks of ``2**low`` addresses. A tree over the block index
    gives one control per block; inside a block, each address's match on the low bits is
    computed into a shared ancilla with one ``init_and`` (toggling the bits that must be 0),
    and the payload is loaded with ``ccx`` per set bit. This is the structure that a
    dirty-ancilla construction would have to beat.
``chunked_fused``
    The same, but the payload load is a single multi-controlled X on the block control plus
    the low address bits, with no per-address ancilla -- the cheapest chunked form that
    needs no extra scratch.

The measurement is on a 64-address slice of the table so every variant is a real circuit, and
the full-table cost is then counted from the same construction rule.
"""
from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_dfthcsos_sub128 as B  # noqa: E402


def build_chunked(delta: np.ndarray, brot: int, *, low: int, fused: bool):
    """Block tree plus per-address match; ``fused`` loads with multi-controlled X."""
    import kickmix as km  # noqa: PLC0415

    C = B._scoreinfra()
    rows, J = delta.shape
    nbits = (rows - 1).bit_length()
    block = 1 << low
    nblocks = ceil(rows / block)

    b = km.CircuitBuilder()
    selection = b.create_quantum_register(nbits, C.SELECTION_REGISTER)
    words = [b.create_quantum_register(brot, C.word_register(j)) for j in range(J)]
    bits = [selection[i] for i in reversed(range(nbits))]  # MSB first
    block_bits = bits[: nbits - low]
    low_bits = bits[nbits - low :]
    block_ancillas = b.alloc_qubits(max(0, len(block_bits) - 1))
    match = b.alloc_qubits(1)[0]

    def block_controls() -> dict[int, object]:
        """Control qubit carrying 'block index == i', computed by a tree."""
        controls: dict[int, object] = {}

        def node(control, level: int, lo: int, hi: int) -> None:
            if lo >= nblocks:
                return
            if hi - lo == 1:
                controls[lo] = control
                return
            mid = (lo + hi) // 2
            if nblocks <= mid:
                node(control, level + 1, lo, mid)
                return
            bit, ancilla = block_bits[level], block_ancillas[level - 1]
            b.x(bit)
            b.init_and(control, bit, ancilla)
            b.x(bit)
            node(ancilla, level + 1, lo, mid)
            b.cx(control, ancilla)
            node(ancilla, level + 1, mid, hi)
            b.del_and(control, bit, ancilla)

        if not block_bits:
            controls[0] = bits[0] if bits else True
            return controls
        half = 1 << (len(block_bits) - 1)
        b.x(block_bits[0])
        node(block_bits[0], 1, 0, half)
        b.x(block_bits[0])
        node(block_bits[0], 1, half, 2 * half)
        return controls

    controls = block_controls()
    for row in range(rows):
        block_index, offset = divmod(row, block)
        control = controls.get(block_index)
        if control is None:
            continue
        zeros = [lb for (i, lb) in enumerate(reversed(low_bits)) if not (offset >> i) & 1]
        for edge in zeros:
            b.x(edge)
        load = [(j, bit) for j in range(J) for bit in reversed(range(brot))
                if (int(delta[row, j]) >> bit) & 1]
        if fused:
            for j, bit in load:
                b.multi_controlled_x([control, *low_bits], words[j][bit])
        else:
            b.init_and(control, low_bits[-1] if low_bits else True, match)
            for j, bit in load:
                b.ccx(match, control, words[j][bit])
            b.del_and(control, low_bits[-1] if low_bits else True, match)
        for edge in zeros:
            b.x(edge)
    return b.finish_circuit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=64, help="address slice to build (default 64)")
    parser.add_argument("--low", type=int, default=3, help="log2 addresses per block (default 3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--brot", type=int, default=10)
    args = parser.parse_args()

    C = B._scoreinfra()
    with np.load(B.DEFAULT_NPZ, allow_pickle=False) as archive:
        words12 = np.array(archive["angle_words"], dtype=np.int64)

    slice_words = B.requantise(words12, 12, args.brot)[: args.rows]
    delta = B.batch_deltas(slice_words, args.batch_size)[0]

    def cost(family):
        c = C.circuit_costs(family)
        return c[0]["num_qubits"], c[0]["max_magic"]

    print(f"=== {args.rows}-address slice, J={args.batch_size}, brot={args.brot} ===")
    print(f"{'variant':16} {'qubits':>7} {'Toffolis':>9} {'vs tree':>9}")

    tree = B.build_delta_circuit(delta, args.brot, compressed=False)
    tq, tt = cost((tree,))
    print(f"{'tree':16} {tq:7d} {tt:9d} {1.0:8.2f}x")

    for fused in (False, True):
        name = "chunked_fused" if fused else "chunked"
        try:
            circuit = build_chunked(delta, args.brot, low=args.low, fused=fused)
            cq, ct = cost((circuit,))
            print(f"{name:16} {cq:7d} {ct:9d} {ct/tt:8.2f}x")
        except Exception as exc:  # a structure that cannot be built is a result too
            print(f"{name:16} {'-':>7} {'-':>9}  {type(exc).__name__}: {str(exc)[:90]}")

    # full-table count from the same rule: block tree + one init_and per address + one ccx
    # per set payload bit
    # batch_deltas returns (K, A, J); one circuit's table is A rows of J words.
    one_circuit = B.batch_deltas(B.requantise(words12, 12, args.brot), args.batch_size)[0]
    rows = one_circuit.shape[0]
    blocks = ceil(rows / (1 << args.low))
    load_bits = sum(int(np.unpackbits(np.ascontiguousarray(row).astype(np.uint16).view(np.uint8)).sum())
                    for row in one_circuit)
    tree_full = rows - 1
    chunked_full = (blocks - 1) + rows + load_bits
    print(f"\n=== counted on the full table ({rows} rows, {args.low} low bits) ===")
    print(f"  tree          {tree_full}")
    print(f"  chunked       {chunked_full}  ({chunked_full/tree_full:.1f}x) "
          f"= {blocks-1} block tree + {rows} matches + {load_bits} ccx loads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
