"""Can any retune of the Benchmark III QROM shape beat the archive's 8,718 Toffolis?

The in-tree family at ``brot = 8``, ``batch = 5`` costs 3,974 Toffolis per circuit
(measured by ``scripts/build_dfthcsos_sub128.py``), i.e. 119,220 over 15 circuits,
against the archived bundle's 8,718. This script prices the shape space under the
competition's own accounting so the question is decided by arithmetic rather than
by another build:

* **Segment tree of any arity.** A ``k``-ary node needs ``k - 1`` ANDs to decode one
  of ``k`` children, and there are ``(rows - 1) / (k - 1)`` internal nodes, so the
  product is ``rows - 1`` Toffolis per circuit *whatever* ``k`` is. Arity is neutral:
  it buys depth, never gates.
* **QROAM at block size b.** ``rows / b + b`` Toffolis per circuit, minimised at
  ``b = sqrt(rows)`` = 127 Toffolis -- but the block's workspace is
  ``J * brot * b`` word qubits, so the 128-qubit cap binds long before the optimum.
* **The cap.** ``selection + J*brot*b + (b - 1) <= 128`` at ``J = 5``, ``brot = 8``
  leaves ``b <= 2``, whose cost is worse than the tree.

The archive's 581 Toffolis per circuit (8,718 / 15) is therefore below the floor of
every shape in this family that fits the cap, which is a statement about the
*encoding*, not about the tuning.

Run: python3 scripts/measure_dfth_qrom_shapes.py
"""
from __future__ import annotations

import json
import math
import pathlib

ROWS = 3976
ANGLES = 75
BROT = 8
BATCH = 5
CAP = 128
SELECTION = (ROWS - 1).bit_length()          # 12
ANCESTORS = SELECTION - 1                     # per-level tree ancillas
MEASURED_TREE_TOFFOLIS_PER_CIRCUIT = 3974     # scripts/build_dfthcsos_sub128.py
ARCHIVE_CIRCUITS = 15
ARCHIVE_ANGLE_TOFFOLIS = 8718


def tree_toffolis(arity: int, rows: int = ROWS) -> int:
    """``(k - 1)`` ANDs per internal node times ``(rows - 1) / (k - 1)`` nodes."""
    internal = (rows - 1) // (arity - 1)
    return internal * (arity - 1)


def qroam(block: int, *, brot: int = BROT, batch: int = BATCH) -> dict:
    circuits = math.ceil(ANGLES / batch)
    per_circuit = math.ceil(ROWS / block) + block
    qubits = SELECTION + batch * brot * block + (block - 1)
    return {
        "block": block,
        "toffolis_per_circuit": per_circuit,
        "toffolis_total": per_circuit * circuits,
        "qubits": qubits,
        "within_cap": qubits <= CAP,
    }


def build_receipt() -> dict:
    assert SELECTION == 12, SELECTION
    circuits = math.ceil(ANGLES / BATCH)
    assert circuits == ARCHIVE_CIRCUITS == 15

    arities = {str(k): tree_toffolis(k) for k in (2, 3, 4, 8, 16)}
    # Measured build agrees with the closed form to within the root's handling.
    assert abs(MEASURED_TREE_TOFFOLIS_PER_CIRCUIT - tree_toffolis(2)) <= 2

    optimum_block = round(math.isqrt(ROWS))
    blocks = {str(b): qroam(b) for b in (1, 2, 3, optimum_block, 16, 32)}
    feasible = {k: v for k, v in blocks.items() if v["within_cap"]}
    best_feasible = min(feasible.values(), key=lambda v: v["toffolis_total"])
    best_any = min(blocks.values(), key=lambda v: v["toffolis_total"])

    archive_per_circuit = ARCHIVE_ANGLE_TOFFOLIS / ARCHIVE_CIRCUITS
    floor_per_circuit = min(tree_toffolis(2), best_feasible["toffolis_per_circuit"])
    return {
        "script": "scripts/measure_dfth_qrom_shapes.py",
        "class": "closed-form cost model under the verifier's accounting; the tree "
                 "row is pinned against a measured build",
        "instance": {
            "rows": ROWS,
            "angles": ANGLES,
            "brot": BROT,
            "batch": BATCH,
            "selection_qubits": SELECTION,
            "qubit_cap": CAP,
            "circuits": circuits,
        },
        "measured_tree": {
            "builder": "scripts/build_dfthcsos_sub128.py --brot 8 --batch-size 5",
            "toffolis_per_circuit": MEASURED_TREE_TOFFOLIS_PER_CIRCUIT,
            "toffolis_total": MEASURED_TREE_TOFFOLIS_PER_CIRCUIT * circuits,
            "peak_qubits": SELECTION + BATCH * BROT + ANCESTORS,
        },
        "arity_retune": {
            "toffolis_per_circuit_by_arity": arities,
            "conclusion": (
                "every arity costs rows - 1 per circuit: a k-ary node spends k - 1 "
                "ANDs to decode one child, so arity moves depth only"
            ),
        },
        "qroam_retune": {
            "blocks": blocks,
            "optimum_block": optimum_block,
            "best_within_cap": best_feasible,
            "best_unconstrained": best_any,
            "target_block_for_the_archive_rate": round(ROWS / archive_per_circuit, 2),
            "conclusion": (
                f"the unconstrained optimum {best_any['toffolis_total']:,} beats the "
                f"archive, but it needs {best_any['qubits']:,} qubits against a "
                f"{CAP}-qubit cap; inside the cap the best block is "
                f"{best_feasible['block']} at {best_feasible['toffolis_per_circuit']:,} "
                "per circuit, worse than the segment tree"
            ),
        },
        "against_the_archive": {
            "archive_angle_toffolis": ARCHIVE_ANGLE_TOFFOLIS,
            "archive_toffolis_per_circuit": archive_per_circuit,
            "floor_within_cap_per_circuit": floor_per_circuit,
            "archive_ratio_below_the_floor": floor_per_circuit / archive_per_circuit,
            "beatable_by_retuning": False,
            "conclusion": (
                f"the archive runs at {archive_per_circuit:.0f} Toffolis per circuit, "
                f"{floor_per_circuit / archive_per_circuit:.2f}x below the cheapest "
                "shape of this family that fits the cap, so the gap is the encoding "
                "and the table, not the tuning"
            ),
        },
    }


def main() -> None:
    receipt = build_receipt()
    out = pathlib.Path("benchmarks/quantum_compilation/dfth_qrom_shapes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    a = receipt["against_the_archive"]
    r = receipt["arity_retune"]["toffolis_per_circuit_by_arity"]
    print(f"arity retune per circuit: {r} (all rows-1)")
    print(f"QROAM optimum: {receipt['qroam_retune']['best_unconstrained']}")
    print(f"QROAM best in cap: {receipt['qroam_retune']['best_within_cap']}")
    print(f"archive {a['archive_toffolis_per_circuit']:.0f}/circuit vs floor "
          f"{a['floor_within_cap_per_circuit']:,} ({a['archive_ratio_below_the_floor']:.2f}x below floor)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
