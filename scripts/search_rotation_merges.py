#!/usr/bin/env python3
"""Attack the 315.3 Toffoli-equivalent rotation overhead in the invsqrt champion.

Status: audit-only

The verifier prices every non-Clifford `Z_POW` (hierarchy level >= 4) at

    rotation_t_count(eps) = 0.56 * log2(1 / eps) + 5.3      T gates
    eps = pi * min_margin / 2**64 / n_rot                   (shared budget)

so the 44 rotations in the 89Q champion cost `(1 + 44 * r(eps)) / 4 ~= 315.3`
Toffoli-equivalents, i.e. 17.8 % of the score. This script answers the only
question that matters: can any of those rotations be merged or dropped while the
verified `worst_error_over_tolerance` stays below 1.0?

Two levers, both checked exactly:

  * merge - two `Z_POW`s on the same qubit can be replaced by one `Z_POW` whose
    angle is their exact dyadic sum, provided every intervening op on that qubit
    commutes with a Z rotation. Exact sum => identical ideal unitary => the
    verified margin cannot move, and `n_rot` falls.
  * truncate - drop the lowest-order rotation. That shifts the phase by
    2**(j-f) turns for half of the domain, so it must be paid out of the
    0.0377 % slack the champion has left.

    /dev/shm/qvcc-venv313/bin/python scripts/search_rotation_merges.py
"""
from __future__ import annotations

import io
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"))

import kickmix as km
from verify_submission_invsqrt import rotation_t_count, T_PER_TOFFOLI

CHAMPION = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd.kmb"
RECEIPT = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd_full_receipt.json"

# Ops that leave a Z rotation on qubit q untouched, whatever their other operands.
COMMUTES_ALWAYS = {"Z", "CZ", "Z_POW", "PUSH_CONDITION", "POP_CONDITION", "INFO_TAG", "REGISTER"}
# Ops where q is a control: Z(q) commutes with the controlled-X/SWAP.
CONTROL_POS = {"CX": 0, "CCX": 0, "CSWAP": 0}
CONTROL_POS_CCX_1 = 1  # for CCX either control works; CSWAP's control is index 0


def load_lines(circuit: km.Circuit) -> list[str]:
    buf = io.StringIO()
    circuit.to_file(buf, format="kmx")
    return [line.rsplit("  #", 1)[0].strip() for line in buf.getvalue().splitlines() if line.strip()]


def parse(line: str) -> tuple[str, list[int], list[int], str | None]:
    parts = line.split()
    op = parts[0]
    qubits = [int(p[1:]) for p in parts[1:] if p.startswith("q") and p[1:].isdigit()]
    bits = [int(p[1:]) for p in parts[1:] if p.startswith("b") and p[1:].isdigit()]
    cond = next((p for p in parts if p.startswith("if")), None)
    return op, qubits, bits, cond


def z_pow_level(turns: Fraction) -> int:
    """Hierarchy level of a Z rotation: 1 = identity, 2 = Z, 3 = T, >=4 = priced."""
    for level, denominator in enumerate((1, 2, 4, 8), start=1):
        if turns * denominator % 1 == 0:
            return level
    return 4


def main() -> None:
    circuit = km.Circuit.from_file(str(CHAMPION), format="kmb")
    lines = load_lines(circuit)
    receipt = json.loads(RECEIPT.read_text())
    q = receipt["num_qubits"]
    base_equiv = receipt["toffoli_equivalent"]
    # The verifier scores the shot-mean Toffoli count, not the static maximum.
    scored_toffoli = float(receipt["toffoli_count"])
    t_base = float(receipt["t_count"])
    n_rot = int(receipt["rotation_count"])
    min_margin = int(receipt["min_margin_units"])
    eps = math.pi * min_margin / 2.0**64 / n_rot

    rotation_cost = (t_base + n_rot * rotation_t_count(eps)) / T_PER_TOFFOLI
    print(f"Champion: Q={q}, scored Toffoli={scored_toffoli:.6f}, rotations={n_rot}")
    print(f"  eps                = {eps:.6e}")
    print(f"  r(eps)             = {rotation_t_count(eps):.6f} T per rotation")
    print(f"  rotation equivalent= {rotation_cost:.6f}  ({100 * rotation_cost / base_equiv:.2f}% of {base_equiv:.6f})")
    print(f"  min_margin         = {min_margin} units = {min_margin / 2**64:.6e} turns (slack)")

    # ---- Lever 1: exact merges under Z-commutation -------------------------
    # For each qubit keep the position and angle of the pending, unmerged rotation.
    pending: dict[int, tuple[int, Fraction, int]] = {}  # qubit -> (index in lines, turns, level)
    merges: list[dict] = []
    merge_of: dict[int, int] = {}  # line index -> index of the rotation it merges into
    for index, line in enumerate(lines):
        op, qubits, _bits, cond = parse(line)
        if op == "Z_POW":
            turns = Fraction(line.split()[2])
            level = z_pow_level(turns)
            target = qubits[0]
            if cond is not None:
                # A conditional rotation is not part of the global phase; leave it alone.
                pending.pop(target, None)
                continue
            if level < 4:
                # Clifford/T: commutes with a Z rotation, keep pending untouched.
                continue
            previous = pending.get(target)
            if previous is not None:
                merges.append(
                    {
                        "qubit": target,
                        "first_line": previous[0],
                        "first_turns": str(previous[1]),
                        "second_line": index,
                        "second_turns": str(turns),
                        "merged_turns": str(previous[1] + turns),
                        "merged_level": z_pow_level(previous[1] + turns),
                        "gap_ops": index - previous[0],
                    }
                )
                merge_of[index] = previous[0]
                pending[target] = (previous[0], previous[1] + turns, z_pow_level(previous[1] + turns))
            else:
                pending[target] = (index, turns, level)
            continue
        if op in COMMUTES_ALWAYS:
            continue
        for position, qubit in enumerate(qubits):
            control = (
                (op == "CX" and position == 0)
                or (op == "CCX" and position in (0, 1))
                or (op == "CSWAP" and position == 0)
            )
            if not control:
                pending.pop(qubit, None)

    surviving = n_rot - len(merges)
    free_merges = [m for m in merges if m["merged_level"] < 4]
    print(f"\nLever 1 - exact merges under Z-commutation:")
    print(f"  mergeable pairs    = {len(merges)} (of {n_rot} rotations)")
    print(f"  rotations after    = {surviving}")
    print(f"  merges that become Clifford/T (free): {len(free_merges)}")
    for merge in merges:
        print(
            f"    q{merge['qubit']}: {float(Fraction(merge['first_turns'])):.3e}"
            f" + {float(Fraction(merge['second_turns'])):.3e}"
            f" -> {float(Fraction(merge['merged_turns'])):.3e} turns"
            f" (gap {merge['gap_ops']} ops)"
        )

    # ---- Price whatever survived ------------------------------------------
    def price(n: int, margin: int) -> dict:
        if n == 0:
            return {"n_rot": 0, "equiv": scored_toffoli + t_base / T_PER_TOFFOLI, "eps": None, "score": q * (scored_toffoli + t_base / T_PER_TOFFOLI)}
        e = math.pi * margin / 2.0**64 / n
        equiv = scored_toffoli + (t_base + n * rotation_t_count(e)) / T_PER_TOFFOLI
        return {"n_rot": n, "equiv": equiv, "eps": e, "score": q * equiv}

    baseline = price(n_rot, min_margin)
    merged = price(surviving, min_margin)
    print(f"\nPricing at the unchanged margin (exact merge cannot move the margin):")
    print(f"  baseline      n={n_rot:2d}  equiv={baseline['equiv']:.6f}  score={baseline['score']:,.3f}")
    print(f"  after merges  n={surviving:2d}  equiv={merged['equiv']:.6f}  score={merged['score']:,.3f}")
    delta = baseline["score"] - merged["score"]
    print(f"  saving        {delta:,.3f} units ({100 * delta / baseline['score']:.3f}%)")

    # ---- Lever 2: truncation versus the remaining slack --------------------
    rotations = [(Fraction(line.split()[2]), parse(line)[1][0]) for line in lines
                 if line.split()[0] == "Z_POW" and z_pow_level(Fraction(line.split()[2])) >= 4]
    rotations.sort()
    slack_turns = Fraction(min_margin, 2**64)
    print(f"\nLever 2 - truncate the lowest-order rotations (slack {float(slack_turns):.6e} turns):")
    cumulative = Fraction(0)
    dropped = 0
    for turns, qubit in rotations:
        cumulative += turns
        if cumulative >= slack_turns:
            break
        dropped += 1
        print(f"  drop q{qubit} 2^{math.log2(float(turns)):.0f} turns -> cumulative {float(cumulative):.6e}")
    print(f"  affordable drops   = {dropped} (before the accumulated shift exceeds the slack)")
    if dropped:
        truncated = price(n_rot - dropped, min_margin - int(cumulative * 2**64))
        print(f"  n={truncated['n_rot']}  equiv={truncated['equiv']:.6f}  score={truncated['score']:,.3f}")
    else:
        print("  every rotation is larger than the entire remaining slack -> no truncation is free")

    payload = {
        "champion": {
            "num_qubits": q,
            "rotations": n_rot,
            "rotation_equivalent": rotation_cost,
            "rotation_share": rotation_cost / base_equiv,
            "min_margin_units": min_margin,
            "score": receipt["score"],
        },
        "merges": merges,
        "surviving_rotations": surviving,
        "merged_score": merged["score"],
        "saving_units": delta,
        "truncation": {"dropped": dropped, "cumulative_turns": str(cumulative)},
    }
    (Path(__file__).resolve().parent.parent / "artifacts/quantum_compilation/invsqrt/rotation_merge_scan.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
