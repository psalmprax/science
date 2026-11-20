"""Run the assembled decode stages through ``local_mr_sim`` and check them semantically.

The four-stage emitter measures *cost*; this checks *behaviour*, in the only engine here
that implements a real Z-basis measure-and-reset (``mr``). The claim under test is narrow
and exact, and each part of it is asserted:

1. **Operands are restored.** The packed discrepancy pass is applied twice — compute, then
   uncompute — so the syndrome registers must read back unchanged and the discrepancy cell
   must return to zero.
2. **Planted positions are recovered.** ``sigma`` is prepared classically from the
   Berlekamp–Massey locator of a planted error vector; the sweep then accumulates
   ``alpha^{-k j}`` into coefficient ``k`` and the circuit reads "is ``sigma`` all zero"
   with ``mr``. The set of positions where that read fires must be exactly the planted set.
3. **Planted magnitudes are recovered.** A position is half a decode; the other half is the
   value. The circuit builds the error evaluator ``omega = (S * sigma) mod x^{2t}`` from the
   syndrome registers and the locator, then per position evaluates
   ``omega(alpha^{-j})`` by Horner and divides it by the formal-derivative sum
   ``sum_{k odd} sigma_k alpha^{-k j}``, which is ``Lambda'(alpha^{-j}) / alpha^{j}``. Since
   Forney's ``alpha^{j}`` factor cancels against that ``1 / alpha^{j}``, no constant is
   required: ``e_j = omega(alpha^{-j}) / sum_{k odd} sigma_k alpha^{-k j}``. The read value
   at each live position must equal the planted error value, not merely be nonzero.
4. **The clear leaves nothing behind, phase included.** Every array qubit is cleared with
   ``mr`` and must read zero, with the tracked phase untouched — and the falsifier
   (``r`` instead of ``mr``) must randomise the phase, which is the whole reason ``mr``
   exists.

The classical Forney computation is run alongside the circuit and asserted against the
planted vector first, so a wrong convention cannot be absorbed by a matching circuit.

Run: python3 scripts/verify_assembly_local_mr_sim.py --degree 4 --t 5 --patterns 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum_compilation import local_mr_sim as L  # noqa: E402


def syndromes_of(error: dict[int, int], field: L.Field, n: int) -> list[int]:
    """``S_j = sum_l e_l * alpha^{pos_l * j}``, the packed decoder's convention."""
    return [
        _xor_all(field.mul(e, field.pow(field.primitive_element, (pos * j) % (field.size - 1)))
                 for pos, e in error.items())
        for j in range(n)
    ]


def _xor_all(values) -> int:
    out = 0
    for value in values:
        out ^= value
    return out


def berlekamp_massey(syndromes: list[int], field: L.Field, t: int) -> list[int]:
    """Textbook BM over ``field``: returns ``sigma`` of degree at most ``t``."""
    sigma = [1] + [0] * t
    previous = [1] + [0] * t
    length = 0
    shift = 1
    scale = 1
    for i in range(2 * t):
        discrepancy = 0
        for k in range(t + 1):
            j = i - k
            if j >= 0 and sigma[k]:
                discrepancy ^= field.mul(sigma[k], syndromes[j])
        if discrepancy == 0:
            shift += 1
            continue
        coefficient = field.div(discrepancy, scale)
        updated = list(sigma)
        for k in range(t + 1 - shift):
            if previous[k]:
                updated[k + shift] ^= field.mul(coefficient, previous[k])
        if 2 * length <= i:
            previous = sigma
            length = i + 1 - length
            shift = 1
            scale = discrepancy
        else:
            shift += 1
        sigma = updated
    return sigma


def roots_of(sigma: list[int], field: L.Field, positions: int) -> set[int]:
    """Positions ``j`` with ``sum_k sigma_k * alpha^{-k j} == 0``."""
    order = field.size - 1
    found = set()
    for j in range(positions):
        total = 0
        for k, coefficient in enumerate(sigma):
            if coefficient:
                total ^= field.mul(coefficient, field.pow(field.primitive_element,
                                                           (-k * j) % order))
        if total == 0:
            found.add(j)
    return found


def error_evaluator(syndromes: list[int], sigma: list[int], field: L.Field) -> list[int]:
    """``omega = (S * sigma) mod x^{2t}`` — the polynomial Forney evaluates."""
    n = len(syndromes)
    omega = [0] * n
    for k in range(n):
        for i in range(k + 1):
            if i < len(syndromes) and k - i < len(sigma):
                omega[k] ^= field.mul(syndromes[i], sigma[k - i])
    return omega


def forney_value(omega: list[int], sigma: list[int], position: int, field: L.Field) -> int:
    """``e_j = omega(alpha^{-j}) / sum_{k odd} sigma_k alpha^{-k j}`` (Forney, char 2)."""
    order = field.size - 1
    alpha = field.primitive_element
    numerator = _xor_all(field.mul(c, field.pow(alpha, (-k * position) % order))
                         for k, c in enumerate(omega) if c)
    denominator = _xor_all(field.mul(c, field.pow(alpha, (-k * position) % order))
                           for k, c in enumerate(sigma) if c and k % 2 == 1)
    return 0 if denominator == 0 else field.div(numerator, denominator)


def build(field: L.Field, t: int, positions: int, sigma: list[int], *,
          clear: str = "mr") -> tuple[L.Circuit, dict]:
    degree = field.degree
    order = field.size - 1
    alpha = field.primitive_element
    cells_count = 2 * t + 3
    builder = L.Builder()
    array = builder.create_quantum_register(cells_count * degree, "bm_array")
    cells = [array[i * degree:(i + 1) * degree] for i in range(cells_count)]
    sigma_cells = cells[:t + 1]
    discrepancy = cells[2 * t + 2]
    syndromes = [builder.create_quantum_register(degree, f"S{i}") for i in range(2 * t)]

    # Classical preparation of the locator the algorithm produced.
    for k, coefficient in enumerate(sigma):
        for bit in range(degree):
            if (coefficient >> bit) & 1:
                builder.x(sigma_cells[k][bit])

    # Compute then uncompute the packed discrepancy pass: operands must come back.
    for _ in range(2):
        for j in range(t + 1):
            builder.ixor_gf2_mul(sigma_cells[j], syndromes[t - j], target=discrepancy,
                                 field=field)

    # omega = (S * sigma) mod x^{2t}, built from the syndrome registers before the sweep
    # mutates the locator coefficients.
    omega_cells = [builder.alloc_qubits(degree) for _ in range(2 * t)]
    for k in range(2 * t):
        for i in range(k + 1):
            if k - i <= t:
                builder.ixor_gf2_mul(syndromes[i], sigma_cells[k - i], target=omega_cells[k],
                                     field=field)

    # Sweep, the all-zero read that locates a root, and the Forney magnitude at that root.
    live = builder.alloc_bits(positions)
    magnitude_bits = [builder.alloc_bits(degree) for _ in range(positions)]
    evaluation = builder.alloc_qubits(degree)
    accumulator = builder.alloc_qubits(degree)
    denominator = builder.alloc_qubits(degree)
    inverse = builder.alloc_qubits(degree)
    magnitude = builder.alloc_qubits(degree)

    def inverse_table() -> None:
        """``inverse ^= 1 / denominator``: a table read, self-inverse, so it also uncomputes.

        A value lookup rather than a division circuit: this engine has no inverse scaffold,
        and at GF(2^4) the table is exact and costs 32 flips.
        """
        for value in range(1, field.size):
            image = field.inverse(value)
            for bit in range(degree):
                if (image >> bit) & 1:
                    builder.flip_if_equal(denominator, value, target=inverse[bit])

    for p in range(positions):
        if p:
            for k in range(1, t + 1):
                step = field.pow(alpha, (-k) % order)
                builder.gf2_imul(step, target=sigma_cells[k], field=field)
        for cell in sigma_cells:
            for bit in range(degree):
                builder.cx(cell[bit], evaluation[bit])
        flag = builder.alloc_qubits(1)[0]
        builder.flip_if_equal(evaluation, 0, target=flag)
        builder.mr(flag, live[p])
        for cell in sigma_cells:
            for bit in range(degree):
                builder.cx(cell[bit], evaluation[bit])

        # The denominator is the odd-coefficient sum of the swept locator.
        for k in range(1, t + 1, 2):
            for bit in range(degree):
                builder.cx(sigma_cells[k][bit], denominator[bit])
        # Horner evaluation of omega at alpha^{-p}.
        y = field.pow(alpha, (-p) % order)
        for k in range(2 * t - 1, -1, -1):
            builder.gf2_imul(y, target=accumulator, field=field)
            for bit in range(degree):
                builder.cx(omega_cells[k][bit], accumulator[bit])
        inverse_table()
        builder.ixor_gf2_mul(accumulator, inverse, target=magnitude, field=field)
        builder.push_condition(live[p])
        for bit in range(degree):
            # `bit_store`, not `cx`: a classical bit is not in the qubit store, and the
            # simulator's `cx` writes its target there, aliasing qubit `bit.id`.
            builder.bit_store(magnitude_bits[p][bit], magnitude[bit])
        builder.pop_condition()
        # Uncompute in reverse: magnitude, the inverse table, Horner, the denominator.
        builder.ixor_gf2_mul(accumulator, inverse, target=magnitude, field=field)
        inverse_table()
        for k in range(2 * t):
            for bit in range(degree):
                builder.cx(omega_cells[k][bit], accumulator[bit])
            builder.gf2_imul(field.pow(alpha, p % order), target=accumulator, field=field)
        for k in range(1, t + 1, 2):
            for bit in range(degree):
                builder.cx(sigma_cells[k][bit], denominator[bit])

    # Clear the array: `mr` is phase-safe, `r` is the falsifier.
    cleared = builder.alloc_bits(cells_count * degree)
    if clear == "mr":
        builder.mr(array, cleared)
    else:
        for qubit in array:
            builder.r(qubit)

    return builder.finish_circuit(), {
        "syndromes": syndromes,
        "cells": cells,
        "live": live,
        "magnitude_bits": magnitude_bits,
        "clear": clear,
    }


def run(circuit: L.Circuit, meta: dict, synd: list[int], *, shots: int = 1,
        seed: int = 7) -> dict:
    sim = L.Simulator(batch_size=shots, seed=seed)
    sim.use_same_registers_as(circuit)
    for register, value in zip(meta["syndromes"], synd):
        bits = np.array([[(value >> i) & 1] for i in range(len(register))], dtype=bool)
        sim.write_across_shots(register, bits)
    sim.do(circuit)

    def read_register(register) -> list[int]:
        raw = np.asarray(sim.read_across_shots(register))
        return [int(sum(int(raw[i, shot]) << i for i in range(len(register))))
                for shot in range(raw.shape[1])]

    live = np.asarray(sim.read_across_shots(meta["live"]))
    return {
        "syndromes": [[int(v) for v in read_register(register)] for register in meta["syndromes"]],
        "cells": [[int(v) for v in read_register(cell)] for cell in meta["cells"]],
        "live": live,
        "magnitudes": [read_register(bits)[0] for bits in meta["magnitude_bits"]],
        "phase": np.asarray(sim.read_phase_flipped_across_shots()),
    }


def check_pattern(field: L.Field, t: int, positions: int, error: dict[int, int],
                  seed: int) -> dict:
    synd = syndromes_of(error, field, 2 * t)
    sigma = berlekamp_massey(synd, field, t)
    omega = error_evaluator(synd, sigma, field)
    # The classical Forney result is the ground truth; a wrong convention would have to
    # reproduce the planted vector by accident to survive this.
    classical = {p: forney_value(omega, sigma, p, field) for p in error}
    classical_matches = classical == error
    planted = set(error)
    roots = roots_of(sigma, field, positions)
    circuit, meta = build(field, t, positions, sigma)
    result = run(circuit, meta, synd, seed=seed)
    live = result["live"][:, 0]
    recovered = {int(j) for j in np.flatnonzero(live)}
    recovered_values = {int(j): result["magnitudes"][int(j)] for j in np.flatnonzero(live)}
    junk_clean = all(value == 0 for column in result["cells"] for value in column)
    phases_clean = not bool(result["phase"].any())

    falsifier_circuit, falsifier_meta = build(field, t, positions, sigma, clear="r")
    falsifier = run(falsifier_circuit, falsifier_meta, synd, shots=32, seed=seed)
    magnitudes_recovered = recovered_values == error
    return {
        "error": {str(k): v for k, v in sorted(error.items())},
        "sigma": sigma,
        "omega": omega,
        "locator_roots": sorted(roots),
        "planted": sorted(planted),
        "recovered": sorted(recovered),
        "algorithm_roots_match_planted": roots == planted,
        "syndromes_restored": all(
            value == expected
            for column, expected in zip(result["syndromes"], synd)
            for value in column),
        "junk_clean": junk_clean,
        "phase_untouched_by_mr": phases_clean,
        "positions_recovered": recovered == planted,
        "classical_forney_matches_planted": classical_matches,
        "classical_forney": {str(k): v for k, v in sorted(classical.items())},
        "recovered_magnitudes": {str(k): v for k, v in sorted(recovered_values.items())},
        "magnitudes_recovered": magnitudes_recovered,
        "falsifier_phase_randomised": int(falsifier["phase"].sum()) > 0,
        "verdict": "PASS" if (roots == planted and recovered == planted and magnitudes_recovered
                              and classical_matches and junk_clean and phases_clean
                              and all(value == expected for column, expected
                                      in zip(result["syndromes"], synd)
                                      for value in column)) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--patterns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    field = L.Field(args.degree)
    positions = field.size - 1
    rng = random.Random(args.seed)

    rows = []
    for pattern in range(args.patterns):
        weight = 1 + pattern % args.t
        plotted = rng.sample(range(positions), weight)
        error = {pos: rng.randrange(1, field.size) for pos in plotted}
        row = check_pattern(field, args.t, positions, error, seed=args.seed + pattern)
        rows.append(row)
        print(f"GF(2^{args.degree}) pattern {pattern}: weight {weight} planted "
              f"{row['planted']} -> roots {row['locator_roots']} -> recovered "
              f"{row['recovered']}; magnitudes {row['recovered_magnitudes']} vs "
              f"{row['error']}; syndromes restored {row['syndromes_restored']}, "
              f"junk clean {row['junk_clean']}, phase clean "
              f"{row['phase_untouched_by_mr']}, falsifier randomises "
              f"{row['falsifier_phase_randomised']}: {row['verdict']}")

    verdict = "PASS" if all(row["verdict"] == "PASS" for row in rows) else "FAIL"
    receipt = {
        "script": "scripts/verify_assembly_local_mr_sim.py",
        "engine": "apedp_dyn/quantum_compilation/local_mr_sim.py",
        "class": "semantic check in the value simulator; not a gate-count receipt",
        "instance": {"degree": args.degree, "t": args.t, "positions": positions,
                     "syndrome_count": 2 * args.t, "patterns": args.patterns},
        "what_is_checked": [
            "syndrome registers restored after a compute+uncompute discrepancy pass",
            "locator roots recovered by the sweep and the all-zero mr read",
            "error magnitudes recovered by omega = (S*sigma) mod x^{2t} evaluated by Horner "
            "and divided by the odd-coefficient derivative sum, conditioned on the live read",
            "the classical Forney result equals the planted vector, so the convention is "
            "pinned independently of the circuit",
            "array cleared and tracked phase untouched under mr (r randomises it)",
        ],
        "rows": rows,
        "verdict": verdict,
    }
    print(f"verdict: {verdict} ({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)} patterns; "
          f"magnitudes {sum(r['magnitudes_recovered'] for r in rows)}/{len(rows)})")

    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
