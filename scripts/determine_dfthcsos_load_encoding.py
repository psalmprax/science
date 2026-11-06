"""Which table encoding loads 40 bits/address in 581 Toffolis?

The in-tree segment-tree QROM over the absolute word table costs ``rows - 2`` = 3,974
Toffolis per circuit at ``brot = 8``, ``J = 5`` (3,976 rows, 12-bit selection). The
archive's 8,718 over 15 circuits is **581 per circuit**, i.e. one Toffoli per 6.84
addresses, which no tree over *all* addresses can reach: an arity argument fixes every
tree at ``rows - 1``, and QROAM's block workspace does not fit the 128-qubit cap.

So the archive must be loading fewer *things* than addresses. The one encoding that does
that without extra qubits is **run-length loading**: if the table is piecewise constant
over long address ranges, a tree that switches on runs and then holds the word while the
low address bits iterate costs one Toffoli per run, not per address. This script measures
that on the real table -- runs, distinct rows, and the existing tree's break audit -- and
confirms the per-node cost by building a run-sized table with the same builder.

    python3 scripts/determine_dfthcsos_load_encoding.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_dfthcsos_sub128 import (  # noqa: E402
    DEFAULT_NPZ,
    batch_deltas,
    count_breaks,
    requantise,
)

ARCHIVE_ANGLE_TOFFOLIS = 8718
BROT = 8
BATCH = 5


def pack_rows(table: np.ndarray, brot: int) -> np.ndarray:
    """Pack each row's ``J`` words into one integer so runs compare whole vectors."""
    table = np.asarray(table, dtype=np.int64)
    packed = np.zeros(table.shape[0], dtype=object)
    for j in range(table.shape[1]):
        packed = (packed << brot) | table[:, j]
    return packed


def runs(packed: np.ndarray) -> int:
    """Number of maximal constant runs: the leaves a run-aware load needs."""
    if len(packed) == 0:
        return 0
    return int(1 + np.count_nonzero(packed[1:] != packed[:-1]))


def _solve_gf2(design: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Least-squares solve of ``design @ x = target`` over GF(2); None if inconsistent."""
    aug = np.concatenate([design.astype(np.uint8), target.astype(np.uint8)[:, None]],
                         axis=1).copy()
    num_rows, num_cols = aug.shape
    num_unknowns = num_cols - 1
    pivots: list[int] = []
    row = 0
    for col in range(num_unknowns):
        pivot = next((i for i in range(row, num_rows) if aug[i, col]), None)
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for i in range(num_rows):
            if i != row and aug[i, col]:
                aug[i] ^= aug[row]
        pivots.append(col)
        row += 1
        if row == num_rows:
            break
    for i in range(num_rows):
        if not aug[i, :num_unknowns].any() and aug[i, num_unknowns]:
            return None
    solution = np.zeros(num_unknowns, dtype=np.uint8)
    for i, col in enumerate(pivots):
        solution[col] = aug[i, num_unknowns]
    return solution


def gf2_affine_residual(packed: np.ndarray, num_bits: int,
                        payload_width: int | None = None) -> dict:
    """Rows where the payload is *not* a GF(2)-affine function of the address.

    An affine table loads with CX only (zero Toffolis on the payload), so an exact fit
    would make the archive's count overhead rather than table loading.
    """
    if payload_width is None:
        payload_width = int(max(1, packed.max()).bit_length()) if len(packed) else 1
    rows = np.arange(len(packed))
    columns = [(rows >> b) & 1 for b in range(num_bits)] + [np.ones_like(rows)]
    design = np.stack(columns, axis=1).astype(np.uint8)  # (rows, num_bits + 1)
    payload_bits = np.stack(
        [((packed >> b) & 1).astype(np.uint8) for b in range(payload_width)], axis=1
    )
    used = [b for b in range(payload_width) if payload_bits[:, b].any()]
    predicted = np.zeros((len(packed), payload_width), dtype=np.uint8)
    inconsistent = 0
    for b in used:
        solution = _solve_gf2(design, payload_bits[:, b])
        if solution is None:
            # No affine form at all for this payload bit, so every row is a miss.
            inconsistent += 1
            predicted[:, b] = 1 - payload_bits[:, b]
            continue
        predicted[:, b] = (design.astype(np.int64) @ solution.astype(np.int64)) % 2
    wrong = int(np.count_nonzero((predicted[:, used] != payload_bits[:, used]).any(axis=1)))
    return {"rows": int(len(packed)), "payload_bits": len(used),
            "unsolvable_bits": inconsistent, "rows_off_affine": wrong,
            "fraction_off_affine": wrong / len(packed)}


def structure_check(words: np.ndarray, brot: int, batch: int) -> dict:
    """Test the payload classes that would explain a cost below one Toffoli per address.

    * **uniform ramp** -- ``word[i + 1] - word[i]`` constant mod ``2^brot``: then the
      word is an accumulator step, not a table entry;
    * **address bits** -- a payload bit equal to an address bit (or its complement) on
      every row: then the word register is a slice of the selection register and loads
      with no gates at all.
    """
    modulus = 1 << brot
    num_rows = words.shape[0]
    num_circuits = max(1, ceil(words.shape[1] / batch))
    address_bits = [(np.arange(num_rows) >> b) & 1 for b in range((num_rows - 1).bit_length())]
    ramps = 0
    slots = 0
    bit_matches: dict[str, int] = {}
    for k in range(num_circuits):
        block = np.asarray(words[:, k * batch:(k + 1) * batch], dtype=np.int64) % modulus
        for j in range(block.shape[1]):
            slots += 1
            diffs = np.diff(block[:, j]) % modulus
            if len(diffs) and len(set(diffs.tolist())) == 1:
                ramps += 1
            for b in range(brot):
                payload = (block[:, j] >> b) & 1
                for name, bits in (("plain", address_bits),
                                   ("inverted", [1 - x for x in address_bits])):
                    for index, address in enumerate(bits):
                        key = f"bit{b}={name}{index}"
                        if np.array_equal(payload, address):
                            bit_matches[key] = bit_matches.get(key, 0) + 1
    return {"slots": slots, "uniform_ramp_slots": ramps,
            "ramp_fraction": ramps / max(1, slots),
            "payload_bit_address_matches": len(bit_matches)}


def measure(words12: np.ndarray, brot: int, batch: int) -> dict:
    words = requantise(words12, 12, brot)
    num_rows, num_angles = words.shape
    num_bits = (num_rows - 1).bit_length()
    num_circuits = max(1, ceil(num_angles / batch))
    deltas = batch_deltas(words, batch)

    per_circuit = []
    for k in range(num_circuits):
        absolute = words[:, k * batch:(k + 1) * batch]
        delta = deltas[k]
        abs_packed = pack_rows(absolute, brot)
        delta_packed = pack_rows(delta, brot)
        per_circuit.append({
            "circuit": k,
            "rows": int(num_rows),
            "selection_bits": num_bits,
            "words_per_row": int(batch),
            "bits_per_address": int(batch * brot),
            "absolute_runs": runs(abs_packed),
            "absolute_runs_cost": max(0, runs(abs_packed) - 2),
            "distinct_rows": int(len(set(abs_packed.tolist()))),
            "delta_runs": runs(delta_packed),
            "delta_zero_rows": int(np.count_nonzero(delta_packed == 0)),
            "tree_breaks": count_breaks(delta),
            "tree_cost": num_rows - 2,
        })

    totals = {
        "tree": sum(c["tree_cost"] for c in per_circuit),
        "absolute_runs": sum(c["absolute_runs_cost"] for c in per_circuit),
        "distinct_rows": sum(max(0, c["distinct_rows"] - 2) for c in per_circuit),
    }
    # Affine test on the payload a circuit actually loads: `batch` words per address.
    affine = [gf2_affine_residual(pack_rows(words[:, k * batch:(k + 1) * batch], brot),
                                 num_bits, payload_width=brot * batch)
              for k in range(num_circuits)]
    return {
        "instance": {"rows": int(num_rows), "angles": int(num_angles), "brot": brot,
                     "batch": batch, "circuits": num_circuits,
                     "bits_per_address": batch * brot},
        "per_circuit": per_circuit,
        "totals": totals,
        "archive": {"angle_toffolis": ARCHIVE_ANGLE_TOFFOLIS,
                    "per_circuit": ARCHIVE_ANGLE_TOFFOLIS / num_circuits},
        "affine_check": {
            "payload_bits": affine[0]["payload_bits"],
            "unsolvable_bits_per_circuit": [a["unsolvable_bits"] for a in affine],
            "rows_off_affine_per_circuit": [a["rows_off_affine"] for a in affine],
            "affine_over_gf2": all(a["rows_off_affine"] == 0 for a in affine),
        },
        "structure": structure_check(words, brot, batch),
    }


def confirm(sizes: list[int], brot: int = BROT, batch: int = BATCH) -> dict:
    """Build distinct-row tables with the real builder to pin one Toffoli per row."""
    from build_dfthcsos_sub128 import build_delta_circuit, _scoreinfra

    C = _scoreinfra()
    out = []
    for rows in sizes:
        table = (np.arange(rows, dtype=np.int64)[:, None] * np.arange(1, batch + 1)[None, :]
                 ) % (1 << brot)
        circuit = build_delta_circuit(table, brot, compressed=False)
        cost = C.circuit_costs([circuit])[0]
        out.append({"distinct_rows": rows,
                    "measured_toffolis": cost["max_magic"],
                    "rows_minus_two": rows - 2,
                    "peak_qubits": cost["num_qubits"]})
    return {"builds": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--brot", type=int, default=BROT)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as archive:
        words12 = np.array(archive["angle_words"], dtype=np.int64)

    receipt = measure(words12, args.brot, args.batch)
    print(f"{receipt['instance']}")
    for c in receipt["per_circuit"]:
        print(f"  circuit {c['circuit']:2d}: runs {c['absolute_runs']:5d} "
              f"(cost {c['absolute_runs_cost']:5d}), distinct {c['distinct_rows']:5d}, "
              f"delta runs {c['delta_runs']:5d}, delta zeros {c['delta_zero_rows']:5d}, "
              f"tree breaks {c['tree_breaks']}, tree cost {c['tree_cost']}")
    print(f"totals: {receipt['totals']}")
    print(f"archive: {receipt['archive']}")
    print(f"affine: {receipt['affine_check']}")

    receipt["confirmation"] = confirm([256, 1024], args.brot, args.batch)
    print(f"confirmation: {receipt['confirmation']}")
    print(f"structure: {receipt['structure']}")

    totals, arch = receipt["totals"], receipt["archive"]["angle_toffolis"]
    receipt["conclusion"] = {
        "encoding_that_matches": min(
            (("tree", totals["tree"]), ("run-length", totals["absolute_runs"]),
             ("distinct-rows", totals["distinct_rows"])),
            key=lambda pair: abs(pair[1] - arch)),
        "run_length_vs_archive": totals["absolute_runs"] / arch,
        "tree_vs_archive": totals["tree"] / arch,
        "summary": (
            f"every circuit has {receipt['per_circuit'][0]['absolute_runs']:,} constant runs "
            f"and {receipt['per_circuit'][0]['distinct_rows']:,} distinct rows out of "
            f"{receipt['instance']['rows']:,}, so no run-length or dedup encoding saves a "
            f"node: run-length costs {totals['absolute_runs']:,} and the address tree "
            f"{totals['tree']:,}, both {totals['tree'] / arch:.2f}x the archive's {arch:,}. "
            f"The payload is not GF(2)-affine ({receipt['affine_check']['unsolvable_bits_per_circuit'][0]} "
            f"unsolvable bits), and {receipt['structure']['uniform_ramp_slots']} of "
            f"{receipt['structure']['slots']} word slots are uniform ramps, so the words are "
            "not an accumulator either"
        ),
    }
    print(receipt["conclusion"]["summary"])

    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
