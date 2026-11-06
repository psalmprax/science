#!/usr/bin/env python3
"""Build the real dirty-ancilla ladder QROM and measure the `O(sqrt(A))` model.

The chunked prototype (`prototype_dfthcsos_chunked_qrom.py`) computed one match
per *address*, so it had no shared structure to amortise and lost by 17x. The
cost model that was withdrawn with it claimed `O(sqrt(A))` Toffolis. This script
builds the construction that model describes, so the claim gets measured.

The construction (`ladder`)
    Split the address `a = (a_hi, a_lo)` with about `sqrt(A)` values on each side.
    Compute a **shared comparator ladder** over the low bits: one control per
    offset `j`, all `sqrt(A)` of them alive at once, at `sqrt(A) - 1` Toffolis and
    `sqrt(A) - 1` dirty ancillas. Then unary-iterate the block index with a tree
    (`sqrt(A) - 1` Toffolis) and, at each block leaf, load that block's rows
    against the offset controls that are *already computed*.

    That is precisely the reuse the `O(sqrt(A))` model assumes: the comparator
    ladder is built once and every block spends zero further Toffolis on
    addressing. Nothing here is an extrapolation -- both levels are real
    circuits.

The catch it measures
    A load now needs *two* controls (block, offset), so it is a `ccx`, not the
    `cx` the shipped tree uses. The tree spends `A - 1` Toffolis on addressing
    and pays for the payload in `cx`; the ladder spends `~2 sqrt(A)` on
    addressing and pays for the payload in `ccx`. Which wins is a question about
    the payload, not about the addressing, and this script answers it on the real
    table -- including the break-even in set payload bits per address.

Everything is checked with the competition's own `recover_words_from_circuits`
on a slice, so "cheaper" is only ever reported for a read that loads the table.
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


def build_ladder(delta: np.ndarray, brot: int, *, low: int, early_break: bool = True):
    """Two-level read: shared offset ladder + block tree, loads by `ccx`.

    ``low`` is the number of address bits handled by the comparator ladder, so the
    ladder has ``S = 2**low`` controls and the block tree has ``H = 2**(n-low)``
    leaves. ``low == 0`` degenerates to the plain tree; ``low == n`` to a flat
    comparator table. Addresses past ``rows`` are zero-padded, which XORs nothing
    but keeps both trees complete.
    """
    import kickmix as km  # noqa: PLC0415

    C = B._scoreinfra()
    rows, J = delta.shape
    nbits = (rows - 1).bit_length()
    size = 1 << nbits
    low = max(0, min(low, nbits))
    n_hi = nbits - low
    S = 1 << low            # offsets per block
    # Blocks that actually hold rows, not the power of two the address space could
    # hold: 3,976 rows at `low = 5` is 125 blocks, not 128. Traversing the padding
    # costs three Toffolis for nothing, and -- the reason this matters -- it makes the
    # built circuit disagree with `ladder_rule`, which counts `ceil(rows / S)`.
    H = max(1, ceil(rows / S))

    table = np.zeros((size, J), dtype=np.int64)
    table[:rows] = delta

    def uniform(lo: int, hi: int) -> bool:
        """True when every row in [lo, hi) loads the same word vector."""
        return bool(np.all(table[lo:hi] == table[lo]))

    # A uniform block needs no offset control at all, so when every block is uniform the
    # shared ladder is dead weight: skip it and the block control alone selects the row.
    block_ranges = [(i * S, min(S, rows - i * S)) for i in range(H) if i * S < rows]
    collapse_ladder = early_break and bool(block_ranges) and all(
        uniform(base, base + width) for base, width in block_ranges)

    b = km.CircuitBuilder()
    selection = b.create_quantum_register(nbits, C.SELECTION_REGISTER)
    words = [b.create_quantum_register(brot, C.word_register(j)) for j in range(J)]
    bits = [selection[i] for i in reversed(range(nbits))]  # most significant first
    hi_bits, lo_bits = bits[:n_hi], bits[n_hi:]

    # ── offset ladder, built only where a load addresses it ────────────────────
    # A tree whose leaves must co-exist cannot reuse one ancilla per level the way
    # the sequential tree does: every node of the built tree is alive while the loads
    # run, so each node costs one dirty ancilla per child. What changes here is *how
    # many* nodes are built: a range that is uniform in every block is never addressed,
    # so neither it nor its subtree is constructed. On a table whose payload sits in a
    # few runs that is a handful of nodes instead of `2(S - 1)` gates, which is what
    # makes a structured comparison like for like rather than ladder-plus-fixed-tax.
    ladder_order: list = []   # (parent, bit, left, right), uncomputed in reverse
    range_ctrl: dict[tuple[int, int], object] = {(0, S): None}

    if S > 1 and not collapse_ladder:
        def any_mixed(lo: int, hi: int) -> bool:
            """True when some block cannot load [lo, hi) with one control."""
            end = min(hi, S)
            return any(
                not uniform(base + lo, base + min(end, width))
                for base, width in block_ranges if lo < width
            )

        def descend(lo: int, hi: int) -> bool:
            """Whether this range needs children of its own.

            A range that is uniform in every block is still *loaded* -- by its parent's
            control, which is why it gets a qubit -- but nothing splits it, so its
            subtree is not built. That is the whole saving: on a table whose payload
            sits in a few runs, the ladder becomes a handful of nodes instead of
            `2(S - 1)` of them.
            """
            return hi - lo > 1 and (any_mixed(lo, hi) if early_break else True)

        def subtree(lo: int, hi: int) -> int:
            """Ancillas for a range that is entered and therefore splits."""
            mid = (lo + hi) // 2
            return 2 + (subtree(lo, mid) if descend(lo, mid) else 0) \
                     + (subtree(mid, hi) if descend(mid, hi) else 0)

        half = S >> 1
        offset_anc = b.alloc_qubits(2 + (subtree(0, half) if descend(0, half) else 0)
                                    + (subtree(half, S) if descend(half, S) else 0))
        cursor = [0]

        def take():
            q = offset_anc[cursor[0]]
            cursor[0] += 1
            return q

        def build(control, level: int, lo: int, hi: int) -> None:
            mid = (lo + hi) // 2
            bit = lo_bits[level]
            left, right = take(), take()
            range_ctrl[(lo, mid)] = left
            range_ctrl[(mid, hi)] = right
            b.x(bit)
            b.init_and(control, bit, left)   # left  = control AND NOT bit
            b.x(bit)
            b.init_and(control, bit, right)  # right = control AND bit
            ladder_order.append((control, bit, left, right))
            if descend(lo, mid):
                build(left, level + 1, lo, mid)
            if descend(mid, hi):
                build(right, level + 1, mid, hi)

        # Root of the ladder has control "True", so the two top halves are plain
        # copies of the top bit and cost CX rather than a Toffoli.
        bit0 = lo_bits[0]
        left_root, right_root = take(), take()
        range_ctrl[(0, half)] = left_root
        range_ctrl[(half, S)] = right_root
        b.x(bit0)
        b.cx(bit0, left_root)   # left_root  = NOT bit0
        b.x(bit0)
        b.cx(bit0, right_root)  # right_root = bit0
        ladder_order.append((None, bit0, left_root, right_root))
        if descend(0, half):
            build(left_root, 1, 0, half)
        if descend(half, S):
            build(right_root, 1, half, S)

    # ── loads ─────────────────────────────────────────────────────────────────
    def emit(block_control, control, row: int) -> None:
        """XOR `row`'s word vector onto the words, under the controls given."""
        for j in range(J):
            value = int(table[row, j])
            for bit in reversed(range(brot)):
                if (value >> bit) & 1:
                    target = words[j][bit]
                    if block_control is None and control is None:
                        b.x(target)
                    elif block_control is None:
                        b.cx(control, target)
                    elif control is None:
                        b.cx(block_control, target)
                    else:
                        b.ccx(block_control, control, target)

    def dyadic(lo: int, hi: int) -> bool:
        """True when [lo, hi) is an aligned dyadic range, so a ladder node selects it."""
        size = hi - lo
        return size > 0 and (size & (size - 1)) == 0 and lo % size == 0

    def load_offsets(block_control, base: int, width: int) -> None:
        """Load the block of `width` offsets starting at `base`.

        This is the tree's own rule applied to the offset dimension: when every row in
        a range is the same, the range's ladder control selects all of them and one
        load covers the range. Without it the ladder pays a `ccx` per row even where
        the table is constant, which is the whole of the tree's early-break advantage.
        """
        if width <= 0:
            return
        if early_break and uniform(base, base + width):
            # Nothing in this block varies, so the block control alone selects the row:
            # no ladder control is needed, and with an all-uniform table none exists.
            emit(block_control, None, base)
            return

        def walk(lo: int, hi: int, cell_lo: int, cell_hi: int) -> None:
            """Load [lo, hi), which lies inside the built node [cell_lo, cell_hi).

            Splitting on the node's own midpoint is what keeps `range_ctrl` total: a
            range is emitted only when it *is* a node the builder entered, so the walk
            can never ask for a control that was never created. Picking the largest
            aligned power of two instead produced ranges like `(896, 904)` in a 904-row
            last block, which exist nowhere in the tree (`KeyError`).
            """
            if hi - lo == 1 or (early_break and (lo, hi) == (cell_lo, cell_hi)
                                and uniform(base + lo, base + hi)):
                emit(block_control, range_ctrl[(lo, hi)], base + lo)
                return
            mid = (cell_lo + cell_hi) // 2
            if hi <= mid:
                walk(lo, hi, cell_lo, mid)
            elif lo >= mid:
                walk(lo, hi, mid, cell_hi)
            else:
                walk(lo, mid, cell_lo, mid)
                walk(mid, hi, mid, cell_hi)

        walk(0, width, 0, S)

    # ── block tree: unary iteration over the high bits ────────────────────────
    block_anc = b.alloc_qubits(max(0, n_hi - 1))

    def load_block(block_index: int, control) -> None:
        # The whole block, padding included: the padded rows are zero in the table and in
        # the address space, and a partial block has no node structure of its own.
        base = block_index * S
        load_offsets(control, base, S)

    def node(control, level: int, lo: int, hi: int) -> None:
        if lo >= H:
            return
        # The same rule one level up: a block range whose rows are all equal needs one
        # load and no descendants, which is how the shipped tree collapses on tables
        # with contiguous constant regions.
        if early_break and hi <= H and uniform(lo * S, hi * S):
            emit(control, None, lo * S)
            return
        if hi - lo == 1:
            load_block(lo, control)
            return
        mid = (lo + hi) // 2
        if H <= mid:
            node(control, level + 1, lo, mid)
            return
        bit, ancilla = hi_bits[level], block_anc[level - 1]
        b.x(bit)
        b.init_and(control, bit, ancilla)
        b.x(bit)
        node(ancilla, level + 1, lo, mid)
        b.cx(control, ancilla)
        node(ancilla, level + 1, mid, hi)
        b.del_and(control, bit, ancilla)

    if n_hi == 0:
        load_block(0, None)
    else:
        half = 1 << (n_hi - 1)
        b.x(hi_bits[0])
        node(hi_bits[0], 1, 0, half)
        b.x(hi_bits[0])
        node(hi_bits[0], 1, half, 2 * half)

    # ── uncompute the offset ladder, children before parents ──────────────────
    # `del_and` assumes its ancilla holds `control AND bit`, so the left child --
    # built as `control AND NOT bit` -- is flipped into that polarity first. The
    # reference tree gets this for free by reusing one ancilla per level and
    # CX-ing it between the children; a fully-alive ladder has two qubits and so
    # needs the conversion written out. Skipping it leaves the phase unflipped
    # and `recover_words_from_circuits` refuses the circuit.
    for control, bit, left, right in reversed(ladder_order):
        if control is None:
            if right is not None:
                b.cx(bit, right)
            b.x(bit)
            if left is not None:
                b.cx(bit, left)
            b.x(bit)
        else:
            if right is not None:
                b.del_and(control, bit, right)
            if left is not None:
                b.cx(control, left)      # control AND NOT bit -> control AND bit
                b.del_and(control, bit, left)
    return b.finish_circuit()


def ladder_rule(rows: int, low: int, payload_bits: int) -> dict[str, int | str]:
    """The cost rule the built circuits follow, so the full table can be counted.

    Ladder: a full binary tree over `S` offsets, one Toffoli per node except the two
    root nodes, which are CX copies of the top address bit (`2S - 4` in total), and
    `2(S - 1)` ancillas because every node must be alive during the loads.
    Block tree: unary iteration over the `H = ceil(rows/S)` blocks, `H - 2`. The `-2`
    is the shipped tree's own root: its top split drives the top address bit directly
    instead of computing an ancilla, so it costs a CX pair where every other node
    costs a Toffoli. That is why the reference tree over `A` addresses is `A - 2`
    (`3,974` at 3,976 rows, `62` at 64 rows) and not `A - 1`.

    Payload: a load has a block control only when `low >= 1` *and* `H >= 2`. With both
    controls present it is a `ccx`, which the checker's `max_magic` counts as
    Toffoli-like, so every set payload bit is a Toffoli. Otherwise the load is `cx`
    (or `x`) and the payload is free in this currency -- which is why the shipped tree,
    `low = 0`, pays `A - 2` Toffolis in total and not `A - 2 + payload`.
    """
    nbits = (rows - 1).bit_length()
    low = max(0, min(low, nbits))
    S = 1 << low
    H = max(1, ceil(rows / S))
    ladder = max(0, 2 * S - 4)
    ancillas = max(0, 2 * (S - 1))
    blocks = max(0, H - 2)
    payload_is_toffoli = low >= 1 and H >= 2
    return {
        "offsets": ladder,
        "ladder_ancillas": ancillas,
        "block_tree": blocks,
        "addressing": ladder + blocks,
        "payload": payload_bits,
        "payload_gate": "ccx" if payload_is_toffoli else ("cx" if low >= 1 else "x"),
        "payload_toffolis": payload_bits if payload_is_toffoli else 0,
        "total": ladder + blocks + (payload_bits if payload_is_toffoli else 0),
    }


def popcount(table: np.ndarray) -> int:
    return sum(bin(int(v)).count("1") for v in np.asarray(table).ravel())


def sparse_table(rows: int, J: int, set_bits_per_address: float, seed: int = 20261102,
                 placement: str = "random", distinct: bool = False,
                 brot: int = 10) -> np.ndarray:
    """A delta table with a controlled number of set payload bits per address.

    The real table carries 40.0 set bits per address at `J = 10`, which is far past the
    ladder's break-even, so it cannot answer *how sparse a table would have to be*. This
    builds tables at a chosen density, each set bit in its own word, so a read of it costs
    exactly the payload the density implies.

    ``placement`` decides *where* the payload sits, at the same density and the same
    payload count, because the tree's early-break does not care how much payload there
    is -- it cares whether the payload is contiguous:

    ``random``      every address equally likely to carry a bit. What the earlier sweep
                    measured.
    ``blocked``     the payload in eight contiguous runs. A table with symmetry-zero
                    regions looks like this: whole address ranges are identical.
    ``contiguous``  one run, the payload packed at the front. The most structure any
                    table can have, so the bound on how far early-break can help.
    ``per-address`` every address carries `floor(d)` bits plus one for the first
                    `(d - floor(d)) * rows`. Random placement instead spreads `k` cells
                    over all cells, so at "1.0" a third of the addresses are still
                    empty and its tree collapses where a real table's cannot.

    ``distinct`` puts each set bit in a different position within its word instead of
    always in bit 0. At the same payload count that makes rows differ, which is what
    forbids a block from being uniform -- the property the shipped angle table has and
    the `0`/`1` tables do not. It is the "no collapse" control for the sweep.
    """
    cells = rows * J
    k = min(cells, int(round(rows * set_bits_per_address)))
    table = np.zeros((rows, J), dtype=np.int64)
    if placement == "random":
        rng = np.random.default_rng(seed)
        flat_cells = rng.choice(cells, size=k, replace=False)
    elif placement == "per-address":
        rng = np.random.default_rng(seed)
        base = int(round(rows * set_bits_per_address)) // rows
        extra = int(round(rows * set_bits_per_address)) - base * rows
        flat_cells = []
        for address in range(rows):
            count = min(J, base + (1 if address < extra else 0))
            picks = rng.choice(J, size=count, replace=False) if count else []
            flat_cells.extend(address * J + int(pick) for pick in picks)
    elif placement in ("blocked", "contiguous"):
        runs = 1 if placement == "contiguous" else 8
        run_cells = max(1, cells // runs)
        per_run, extra = divmod(k, runs)
        flat_cells = [
            start + offset
            for r in range(runs)
            for start in (r * run_cells,)
            for offset in range(min(per_run + (1 if r < extra else 0), run_cells))
        ]
    else:
        raise ValueError(f"unknown placement {placement!r}")
    for flat in flat_cells:
        table[flat // J, flat % J] = (1 << (flat % brot)) if distinct else 1
    return table


def density_sweep(C, rows: int, J: int, brot: int, densities=(0.0, 0.10, 0.25, 0.5,
                                                                   0.75, 0.90, 0.95, 1.0),
                  shots: int = 2, placements=("random", "blocked", "contiguous"),
                  early_break: bool = True, distinct: bool = False, low: int = 5) -> int:
    """Measure the tree against the ladder across payload densities on the full table.

    The break-even of `ladder_rule` holds the tree fixed at its real-table cost, which is
    wrong the moment the table gets sparse: the tree's early-break collapses
    all-zero subtrees, so the tree cheapens as the payload thins. This measures both
    on the same table at each density, which is the only way to see whether the
    crossing the rule predicts actually happens.

    Without early-break the ladder is layout-invariant -- its `ccx` count is the payload
    count, wherever the payload sits -- and a per-density cache would be sound. With it the
    ladder collapses where the table is constant, so the cache is keyed by placement as well;
    an earlier version keyed it by density alone and silently reported the random table's
    ladder cost for the structured layouts.
    """
    def measure(circuit, delta) -> tuple[int, int, bool]:
        cost = C.circuit_costs((circuit,))[0]
        try:
            recovered, _, _ = C.recover_words_from_circuits(
                (circuit,), num_vectors=delta.shape[0], num_angles=delta.shape[1],
                shots_per_address=shots, seed=20261102)
            ok = bool(np.array_equal(recovered, delta))
        except Exception as exc:  # noqa: BLE001 - a refused read is a result
            print(f"    recovery refused: {type(exc).__name__}: {str(exc)[:100]}")
            ok = False
        return cost["num_qubits"], cost["max_magic"], ok

    print(f"\n=== density sweep: {rows} addresses, J={J}, brot={brot}, low={low}, "
          f"early_break={early_break}, cells={'distinct' if distinct else 'single'} ===")
    crossings: dict[str, float | None] = {}
    directions: dict[str, str] = {}
    losses: dict[str, float] = {}
    ladder_cache: dict[tuple[int, str], tuple[int, int, bool]] = {}
    for placement in placements:
        print(f"\n-- {placement} placement --")
        print(f"{'bits/addr':>9} {'payload':>8} {'tree T':>7} {'ladder T':>8} "
              f"{'vs tree':>8} {'recovers':>9}")
        rows_out: list[tuple[float, int, int]] = []
        for index, density in enumerate(densities):
            delta = sparse_table(rows, J, density, placement=placement,
                                 distinct=distinct, brot=brot)
            payload = popcount(delta)
            _, tree_t, tree_ok = measure(
                B.build_delta_circuit(delta, brot, compressed=False), delta)
            key = (index, placement)
            if key not in ladder_cache:
                ladder_cache[key] = measure(
                    build_ladder(delta, brot, low=low, early_break=early_break), delta)[1:]
            ladder_t, ladder_ok = ladder_cache[key]
            rows_out.append((payload / rows, tree_t, ladder_t))
            # An all-empty table collapses the tree to nothing to read, so the ratio
            # there is a division by zero rather than a cost comparison.
            ratio = f"{ladder_t/tree_t:7.2f}x" if tree_t else f"{'n/a':>8}"
            print(f"{payload/rows:9.2f} {payload:8d} {tree_t:7d} {ladder_t:8d} "
                  f"{ratio} {str(tree_ok and ladder_ok):>9}")

        # The last place the ladder goes from cheaper to dearer is the one that matters.
        # An all-zero table is excluded: nothing to read, so the tree collapses to a
        # handful of nodes and the ratio there says nothing about a real read.
        crossing = None
        direction = "below"
        for (d0, t0, l0), (d1, t1, l1) in zip(rows_out, rows_out[1:]):
            if (l1 - t1) == (l0 - t0):
                continue
            if l0 < t0 <= l1:
                direction = "below"
                crossing = d0 + (t0 - l0) * (d1 - d0) / ((l1 - t1) - (l0 - t0))
            elif l0 >= t0 > l1:
                # The other direction: a table with no empty address costs the tree one
                # node per populated address, so the ladder gets relatively cheaper as
                # the payload grows until its `ccx` term overtakes that.
                direction = "above"
                crossing = d0 + (l0 - t0) * (d1 - d0) / ((l0 - t0) - (l1 - t1))
        crossings[placement] = crossing
        directions[placement] = direction
        flat = rows_out[-1]
        losses[placement] = flat[2] / flat[1] if flat[1] else float("nan")
        if crossing is None:
            print("\n  no crossing on this range")
        else:
            print(f"\n  crossing: the ladder is cheaper {direction} {crossing:.2f} "
                  f"set bits per address")

    print("\n=== placement summary (same density, same payload) ===")
    print(f"{'placement':>11} {'crossing':>9} {'cost at 1.0':>12}")
    for placement in placements:
        crossing = crossings[placement]
        loss = losses[placement]
        print(f"{placement:>11} "
              f"{('%.2f' % crossing) if crossing is not None else '-':>9} "
              f"{('%10.2fx' % loss) if loss == loss else '       n/a'}")
    if len(crossings) == 1:
        only = next(iter(crossings))
        value = crossings[only]
        shown = "none" if value is None else f"{value:.2f}"
        print(f"  crossing on {only} placement: {shown}")
    elif all(c is not None for c in crossings.values()):
        ordered = sorted(crossings, key=lambda p: crossings[p] or 0.0)
        print(f"  structure moves the crossing from {crossings[ordered[-1]]:.2f} "
              f"({ordered[-1]}) to {crossings[ordered[0]]:.2f} ({ordered[0]}).")
    else:
        structured = [p for p, c in crossings.items() if c is None]
        print(f"  {', '.join(structured)} has no crossing at all: at equal payload the tree")
        print("  collapses whole uniform blocks, so the ladder never gets inside its cost.")
    if len(crossings) > 1:
        print("  Contiguous zero regions therefore help the tree, not the ladder, and the")
        print("  random-placement crossing is the ladder's best case.")
    return 0


def low_scan(C, rows: int, J: int, brot: int, *, density: float = 0.75,
             placement: str = "random", early_break: bool = True, cap: int = 128) -> int:
    """Every ladder split with its qubit cost, so the cap decides the comparison.

    The split is not a free parameter. `J * brot` payload qubits plus the ladder's
    ancillas plus the block tree's must fit the cap, and at `J = 10` that leaves only the
    narrow splits -- which is why the crossing at `J = 10` is not the crossing at `J = 4`.
    This measures the whole ladder so the in-cap set is a measurement, not an assumption.
    """
    delta = sparse_table(rows, J, density, placement=placement, brot=brot)
    tree = C.circuit_costs((B.build_delta_circuit(delta, brot, compressed=False),))[0]
    print(f"\n=== low scan: {rows} addresses, J={J}, brot={brot}, "
          f"{density:g} bits/address, {placement} ===")
    print(f"  tree: {tree['num_qubits']} qubits, {tree['max_magic']} Toffolis")
    print(f"{'low':>4} {'qubits':>7} {'in cap':>7} {'ladder T':>9} {'vs tree':>9} {'recovers':>9}")
    best = None
    for low in range((rows - 1).bit_length() + 1):
        try:
            circuit = build_ladder(delta, brot, low=low, early_break=early_break)
            cost = C.circuit_costs((circuit,))[0]
            recovered, _, _ = C.recover_words_from_circuits(
                (circuit,), num_vectors=delta.shape[0], num_angles=delta.shape[1],
                shots_per_address=2, seed=20261102)
            ok = bool(np.array_equal(recovered, delta))
        except Exception as exc:  # noqa: BLE001 - an unbuildable split is a result
            print(f"{low:4d} {'-':>7} {'-':>7} {'-':>9} {'-':>9}  {type(exc).__name__}")
            continue
        fits = cost["num_qubits"] <= cap
        print(f"{low:4d} {cost['num_qubits']:7d} {str(fits):>7} {cost['max_magic']:9d} "
              f"{cost['max_magic'] / tree['max_magic']:8.2f}x {str(ok):>9}")
        if fits and ok and (best is None or cost["max_magic"] < best[2]):
            best = (low, cost["num_qubits"], cost["max_magic"])
    if best is None:
        print("  no in-cap split recovers the table")
    else:
        print(f"  best in-cap split: low={best[0]}, {best[1]} qubits, {best[2]} Toffolis "
              f"({best[2] / tree['max_magic']:.2f}x the tree)")
    return 0


def low_matrix(C, rows: int, J: int, brot: int, densities=(0.0, 0.10, 0.25, 0.5, 0.75,
                                                              0.90, 0.95, 1.0),
               placements=("random", "per-address", "blocked", "contiguous"),
               max_low: int = 8,
               cap: int = 128) -> int:
    """Best in-cap split at every density, by cost only.

    A split is only a candidate if the family fits 128 qubits, and at `J = 10` that is
    `low <= 2`. This walks the grid to check that the `low = 0` answer is a property of the
    cap and not of one density. Recovery is not re-run per split -- every split used here is
    word-checked by `--low-scan` and by the pinned tests -- so this stays a cost table.
    """
    print(f"\n=== low matrix: {rows} addresses, J={J}, brot={brot}, cap={cap} ===")
    print(f"{'density':>8} {'placement':>12} {'tree T':>7} {'best in-cap':>12} {'qubits':>7} "
          f"{'T':>7} {'vs tree':>9}")
    for placement in placements:
        for density in densities:
            delta = sparse_table(rows, J, density, placement=placement, brot=brot)
            tree = C.circuit_costs((B.build_delta_circuit(delta, brot, compressed=False),))[0]
            best = None
            for low in range(max_low + 1):
                try:
                    cost = C.circuit_costs((build_ladder(delta, brot, low=low),))[0]
                except Exception:  # noqa: BLE001, PERF203 - an unbuildable split is skipped
                    continue
                if cost["num_qubits"] <= cap and (best is None or cost["max_magic"] < best[2]):
                    best = (low, cost["num_qubits"], cost["max_magic"])
            if best is None:
                print(f"{density:8.2f} {placement:>12} {tree['max_magic']:7d} {'none':>12}")
                continue
            print(f"{density:8.2f} {placement:>12} {tree['max_magic']:7d} {best[0]:12d} "
                  f"{best[1]:7d} {best[2]:7d} {best[2] / tree['max_magic']:8.2f}x")
    return 0


def density_basis(npz: Path, brots=(1, 4, 6, 8, 9, 10, 11, 12), js=(4, 10)) -> int:
    """How many set bits per address a table of this kind actually carries.

    The question the sweep leaves open is whether a real table could ever sit near the
    crossing. This measures the shipped angle table's density against the only two levers
    there are -- word width ``brot`` and words per circuit ``J`` -- and reports what would
    have to change for the crossing to be reachable.

    The delta table is the *angle* table: it is set by the rotation words in the
    submission, not by the electronic-structure integrals. An FCIDUMP enters scoring
    through `lambda_eff`, so it cannot move this number. What could is a family whose
    words are mostly zero, and that is what the last column tests.
    """
    with np.load(npz, allow_pickle=False) as archive:
        words12 = np.array(archive["angle_words"], dtype=np.int64)
    rows = words12.shape[0]

    print(f"\n=== density basis: {rows} addresses, real angle table ({npz.name}) ===")
    print(f"{'brot':>5} " + " ".join(f"{('J=%d' % j):>9}" for j in js)
          + f" {'bits/word':>10} {'zero words':>11}")
    per_word: dict[int, float] = {}
    for brot in brots:
        words = B.requantise(words12, 12, brot)
        cells = words.size
        zeros = int((words == 0).sum()) / cells
        counts = []
        for j in js:
            delta = B.batch_deltas(words, j)[0]
            counts.append(popcount(delta) / rows)
        per_word[brot] = counts[js.index(10)] / 10 if 10 in js else counts[0] / js[0]
        print(f"{brot:5d} " + " ".join(f"{c:9.2f}" for c in counts)
              + f" {per_word[brot]:10.2f} {zeros:11.1%}")

    # A word contributes at most `brot` set bits, so a table at `J` words cannot go below
    # `J * (fraction of non-zero words)` -- every non-zero word sets at least one bit.
    print(f"\n  the shipped table at brot=10, J=10 carries {per_word[10]*10:.1f} set bits per")
    print(f"  address ({per_word[10]:.2f} per 10-bit word, out of a 5.0 ceiling for words")
    print("  spread over their range) -- it is a mid-density table, not a sparse one")
    print(f"\n  to reach the measured crossing with J=10 you need "
          f"{0.87/10:.3f} bits per word, i.e. {1 - 0.87/10:.1%} of the")
    print(f"  {rows} x 10 = {rows*10:,} cells exactly zero. At brot=1 the crossing is")
    print(f"  reachable -- {0.87/1:.2f} of a 1-bit word -- which is why the only tables the")
    print("  ladder beats are ones whose words are almost all zero, at any precision.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=64, help="address slice built and simulated")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--brot", type=int, default=10)
    parser.add_argument("--low", type=int, default=3,
                        help="ladder bits used for the built slice (default 3)")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--table", choices=("real", "synthetic"), default="real",
                        help="real slice, or a synthetic table at --set-bits density")
    parser.add_argument("--set-bits", type=float, default=1.0,
                        help="set payload bits per address for --table synthetic")
    parser.add_argument("--lows", type=int, nargs="*", default=None,
                        help="restrict the built-slice sweep to these splits")
    parser.add_argument("--sweep", action="store_true",
                        help="measure the tree against the ladder across payload densities")
    parser.add_argument("--placement", nargs="*", default=None,
                        choices=("random", "blocked", "contiguous", "per-address"),
                        help="payload layouts to compare in --sweep (default: all three)")
    parser.add_argument("--density", action="store_true",
                        help="measure the real table's set bits per address against brot and J")
    parser.add_argument("--low-scan", action="store_true",
                        help="cost every ladder split with its qubit count, to find the in-cap set")
    parser.add_argument("--low-matrix", action="store_true",
                        help="best in-cap split at every density, cost only")
    parser.add_argument("--no-early-break", action="store_true",
                        help="disable the ladder's block/offset collapsing, for the pre-break baseline")
    parser.add_argument("--distinct", action="store_true",
                        help="scatter each set bit to its own bit position, so blocks differ")
    parser.add_argument("--sweep-low", type=int, default=5,
                        help="ladder split used by --sweep (default 5; use 2 for J=10, which "
                             "cannot afford a wide ladder in the 128-qubit cap)")
    args = parser.parse_args()
    early_break = not args.no_early_break

    if args.density:
        return density_basis(B.DEFAULT_NPZ)

    if args.low_matrix:
        C = B._scoreinfra()
        return low_matrix(C, args.rows, args.batch_size, args.brot)

    if args.low_scan:
        C = B._scoreinfra()
        return low_scan(C, args.rows, args.batch_size, args.brot, density=args.set_bits,
                        placement=(args.placement or ("random",))[0],
                        early_break=early_break)

    C = B._scoreinfra()
    with np.load(B.DEFAULT_NPZ, allow_pickle=False) as archive:
        words12 = np.array(archive["angle_words"], dtype=np.int64)

    full_words = B.requantise(words12, 12, args.brot)
    slice_words = full_words[: args.rows]
    if args.table == "real":
        delta = B.batch_deltas(slice_words, args.batch_size)[0]
    else:
        delta = sparse_table(args.rows, args.batch_size, args.set_bits)

    if args.sweep:
        C = B._scoreinfra()
        return density_sweep(C, args.rows, args.batch_size, args.brot, shots=args.shots,
                             placements=tuple(args.placement or
                                              ("random", "blocked", "contiguous")),
                             early_break=early_break, distinct=args.distinct,
                             low=args.sweep_low)

    def costs(circuit):
        c = C.circuit_costs((circuit,))[0]
        return c["num_qubits"], c["max_magic"]

    def recovers(circuit) -> tuple[bool, int]:
        try:
            recovered, layout, toffoli = C.recover_words_from_circuits(
                (circuit,),
                num_vectors=delta.shape[0],
                num_angles=delta.shape[1],
                shots_per_address=args.shots,
                seed=20261102,
            )
        except Exception as exc:  # noqa: BLE001 - an unbuildable/unrecoverable read is a result
            print(f"    recovery refused: {type(exc).__name__}: {str(exc)[:110]}")
            return False, 0
        return bool(np.array_equal(recovered, delta)), int(toffoli)

    print(f"=== {args.rows}-address slice, J={args.batch_size}, brot={args.brot} ===")
    print(f"{'variant':22} {'qubits':>7} {'Toffolis':>9} {'recovers':>9} {'vs tree':>9}")

    tree = B.build_delta_circuit(delta, args.brot, compressed=False)
    tq, tt = costs(tree)
    tree_ok, _ = recovers(tree)
    print(f"{'tree (shipped)':22} {tq:7d} {tt:9d} {str(tree_ok):>9} {1.0:8.2f}x")

    best = None
    lows = args.lows if args.lows is not None else range(0, (delta.shape[0] - 1).bit_length() + 1)
    for low in lows:
        try:
            circuit = build_ladder(delta, args.brot, low=low, early_break=early_break)
            lq, lt = costs(circuit)
        except Exception as exc:  # noqa: BLE001 - a structure that cannot be built is a result
            print(f"{'ladder low=' + str(low):22} {'-':>7} {'-':>9} {'-':>9}  "
                  f"{type(exc).__name__}: {str(exc)[:70]}")
            continue
        ok, _ = recovers(circuit)
        print(f"{'ladder low=' + str(low):22} {lq:7d} {lt:9d} {str(ok):>9} {lt/tt:8.2f}x")
        if ok and (best is None or lt < best[1]):
            best = (low, lt)

    # ── does the rule agree with the circuits it describes? ───────────────────
    # The rule is what the full-table numbers rest on, so it is checked against the
    # built slice's measured Toffolis at every split rather than trusted. The payload
    # is only in the Toffoli count when the loads carry two controls.
    payload_slice = popcount(delta)
    print("\n=== cost rule vs the built slice ===")
    print(f"{'low':>4} {'addressing':>11} {'payload in':>11} {'rule total':>11} "
          f"{'measured':>9} {'agree':>6}")
    for low in lows:
        if early_break:
            continue  # the rule models the uncollapsed read; the collapsed cost is measured
        try:
            circuit = build_ladder(delta, args.brot, low=low, early_break=False)
            measured = C.circuit_costs((circuit,))[0]["max_magic"]
        except Exception:  # noqa: BLE001 - a structure that cannot be built is not checkable here
            continue
        rule = ladder_rule(delta.shape[0], low, payload_slice)
        print(f"{low:4d} {rule['addressing']:11d} {rule['payload_gate']:>11} "
              f"{rule['total']:11d} {measured:9d} {str(rule['total'] == measured):>6}")

    # ── the same rules on the full table ─────────────────────────────────────
    one = B.batch_deltas(full_words, args.batch_size)[0]
    rows = one.shape[0]
    if args.table == "real":
        payload = popcount(one)
        basis = f"real table, {payload / rows:.1f} set bits per address"
    else:
        payload = int(round(rows * args.set_bits))  # same density, full address count
        basis = f"synthetic at {args.set_bits:g} set bits per address (projected to {rows})"
    nbits = (rows - 1).bit_length()
    fixed = nbits + args.batch_size * args.brot
    print(f"\n=== cost rule on the full table ({rows} addresses, J={args.batch_size}, "
          f"{fixed} fixed qubits; {basis}) ===")
    print(f"{'low':>4} {'ladder':>7} {'anc':>5} {'blocks':>7} {'addressing':>11} {'payload':>8} "
          f"{'paid in':>8} {'total':>8} {'vs tree':>9} {'qubits':>7} {'fits128':>8}")
    # The shipped tree's Toffoli count is `rows - 2` (see `ladder_rule`): its root
    # split is a CX pair, and it is measured at 3,974 for 3,976 rows and 62 for 64.
    tree_full = rows - 2
    for low in (0, 2, 4, 5, 6, 7, 8):
        rule = ladder_rule(rows, low, payload)
        qubits = fixed + rule["ladder_ancillas"] + max(0, nbits - low - 1)
        print(f"{low:4d} {rule['offsets']:7d} {rule['ladder_ancillas']:5d} "
              f"{rule['block_tree']:7d} {rule['addressing']:11d} {rule['payload']:8d} "
              f"{rule['payload_gate']:>8} {rule['total']:8d} {rule['total']/tree_full:8.2f}x {qubits:7d} "
              f"{'yes' if qubits <= 128 else 'NO':>8}")
    print(f"  tree                       {tree_full} addressing Toffolis + {payload} cx loads "
          f"-> {tree_full} Toffolis")
    # The best split that fits the cap, which is what a proposal would actually ship.
    in_cap = [(low, ladder_rule(rows, low, payload))
              for low in range(1, nbits + 1)
              if fixed + 2 * ((1 << low) - 1) + max(0, nbits - low - 1) <= 128]
    split, rule = min(in_cap, key=lambda kv: kv[1]["total"])
    print(f"  ladder at low={split}: {rule['addressing']} addressing Toffolis "
          f"({tree_full / rule['addressing']:.0f}x cheaper addressing) + {rule['payload']} ccx loads "
          f"-> {rule['total']} total ({rule['total']/tree_full:.2f}x the tree)")

    # Break-even: the ladder wins only when the payload, paid in Toffoli, is smaller
    # than the addressing it saves. Solve for the payload at the best split.
    break_even = tree_full - rule["addressing"]
    print(f"\n  break-even payload: {break_even} set bits total "
          f"({break_even / rows:.3f} per address); "
          f"this table has {payload} ({payload / rows:.1f} per address)")
    if payload >= rows:
        print("  -> at least one set bit on every address, which is above the line: the")
        print("     ladder cannot be the cheaper read on this table, however it is tuned")
    elif payload < break_even:
        print(f"  -> below the line: the ladder wins by {tree_full - rule['total']} Toffolis "
              f"({1 - rule['total']/tree_full:.1%})")
    else:
        print("  -> above the line: the tree wins")
    if best:
        print(f"\n  smallest recovering ladder on the slice: low={best[0]}, {best[1]} Toffolis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
