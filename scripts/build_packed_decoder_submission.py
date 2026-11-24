#!/usr/bin/env python3
"""Build the packed decoder at the official instance and hand it to the DQI verifier.

The four stages ``scripts/assemble_packed_decoder.py`` assembles are built here at the
official parameters (degree 12, ``t`` 35, 4,095 positions) and shaped to the verifier's
**register contract**, which the scaled assembly is not:

* ``S0 .. S69``  70 quantum registers of 12 bits — the syndromes, restored in place;
* ``E0 .. E4094`` 4,095 classical registers of 12 bits — the MBUC values, which the
  Forney step's phase kickback is masked by (``gf2_phase_by_product(..., mask=E[j])``,
  exactly the call ``dqi_reference_circuit._chien_search_and_forney_mbuc`` makes);
* no other *named* register, because the verifier rejects any name it did not ask for.

What this circuit is, and is not
--------------------------------
It is the repository's own decoder stages in one gate stream: the packed single-array key
equation (70 iterations over ``sigma | B | d``), the constant-multiply coefficient sweep,
the gated Forney block (inversion + phase product + uncompute) at all 4,095 positions, and
a measured clear. Every one of those stages is separately measured in
``benchmarks/quantum_compilation/``.

It is not a working decoder. The key-equation loop is the one the assembly emits, which is
not the update ``inplace_bm.py`` verifies (that module checks an algorithm, not a gate
stream), and the Forney numerator holds no ``omega`` — the packed array has no room for
one, and ``omega`` is what makes a located value the error value. So the expected verdict
is FAIL, and the FAIL is the measurement: it is the difference between "this repository
has a packed-decoder design" and "this repository has a submission". The width it lands at
is measured here too, and it is the binding constraint:

    Q = 840 syndrome + 876 array + 12 omega + 12 inverse = 1,740

against the archived artifact's 891. 891 − 840 = 51 qubits, and the array alone is 876.

    /dev/shm/qvcc-venv313/bin/python scripts/build_packed_decoder_submission.py \
        --out /dev/shm/dqi_packed_4095.kmb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
if str(BUNDLE / "kickmix") not in sys.path:
    sys.path.insert(0, str(BUNDLE / "kickmix"))

import kickmix as km  # noqa: E402

ARCHIVED_QUBITS = 891
REFERENCE_QUBITS = 1893


def build_submission(degree: int, t: int, m: int) -> tuple[km.Circuit, dict]:
    """The four stages at ``(degree, t, m)``, shaped to the verifier's register contract."""
    field = km.GF2Field(degree)
    builder = km.CircuitBuilder()

    # The contract's registers, and nothing else that is named.
    syndromes = [builder.create_quantum_register(degree, f"S{j}") for j in range(2 * t)]
    error_mbuc = [builder.create_classical_register(degree, f"E{j}") for j in range(m)]

    n_cells = 2 * t + 3
    array = builder.alloc_qubits(n_cells * degree)
    cells = [array[i * degree:(i + 1) * degree] for i in range(n_cells)]
    sigma = cells[:t + 1]
    b_poly = cells[t + 1:2 * t + 2]
    disc = cells[2 * t + 2]
    numerator = builder.alloc_qubits(degree)
    inverse = builder.alloc_qubits(degree)
    steps = [field.pow(field.primitive_element, k) for k in range(1, 2 * t + 2)]

    # Stage 1: the key equation, 2t packed iterations over the one array.
    for _ in range(2 * t):
        for j in range(t + 1):
            builder.ixor_gf2_mul(sigma[j], syndromes[t - j], target=disc, field=field)
        for j in range(1, t + 1):
            builder.ixor_gf2_mul(disc, b_poly[j - 1], target=sigma[j], field=field)

    # Stages 2 and 3: the sweep, then the Forney block, phase-masked by E[j] -- the
    # verifier's own phase condition, (-1) ** sum_j popcount(e_j & M_j).
    for p in range(m):
        for cell, step in zip(sigma + b_poly, steps):
            builder.gf2_imul(step, target=cell, field=field)
        _, chain = builder.init_gf2_inverse_with_scaffold(sigma[0], target=inverse, field=field)
        builder.gf2_phase_by_product(numerator, inverse, mask=error_mbuc[p], field=field)
        builder.del_gf2_inverse_with_scaffold(sigma[0], target=inverse, scaffold=chain,
                                              field=field)
        builder.free(chain)

    # Stage 4: measured clear of everything outside the contract's registers.
    clear = "mr" if hasattr(km.CircuitBuilder, "mr") else "reset"
    clear_bits = builder.alloc_bits(degree) if clear == "mr" else None
    for cell in cells:
        if clear == "mr":
            builder.mr(cell, clear_bits)
        else:
            for qubit in list(cell):
                builder.reset(qubit)
    for register in (numerator, inverse):
        if clear == "mr":
            builder.mr(register, clear_bits)
        else:
            for qubit in list(register):
                builder.reset(qubit)
    if clear_bits is not None:
        builder.free(clear_bits)

    circuit = builder.finish_circuit()
    stats = circuit.operation_statistics()
    syndrome_qubits = 2 * t * degree
    array_qubits = n_cells * degree
    return circuit, {
        "degree": degree,
        "t": t,
        "positions": m,
        "iterations": 2 * t,
        "qubits": int(circuit.num_qubits),
        "num_ops": len(circuit),
        "num_bits": int(circuit.num_bits),
        "static_toffolis": int(stats.num_toffoli_like),
        "clear": clear,
        "width_accounting": {
            "syndrome_qubits": syndrome_qubits,
            "packed_array_qubits": array_qubits,
            "omega_qubits": degree,
            "inverse_qubits": degree,
            "total": syndrome_qubits + array_qubits + 2 * degree,
            "verified_layout_floor": syndrome_qubits + array_qubits + degree + 3,
        },
        "ops": {k: int(v) for k, v in stats.items() if int(v)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--out", type=pathlib.Path, help="write the circuit as .kmb here")
    parser.add_argument("--json", type=pathlib.Path, help="write the build record here")
    args = parser.parse_args()

    started = time.time()
    circuit, meta = build_submission(args.degree, args.t, args.m)
    meta["build_seconds"] = round(time.time() - started, 2)
    print(f"GF(2^{args.degree}) t={args.t}: {meta['iterations']} iterations, "
          f"{meta['positions']:,} positions, {meta['qubits']} qubits, "
          f"{meta['num_ops']:,} ops, static {meta['static_toffolis']:,} T "
          f"({meta['build_seconds']} s)")

    # The contract check the verifier will run first.
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V  # noqa: PLC0415

    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=2 * args.t, degree=args.degree)
    try:
        V.validate_registers(circuit, spec)
        meta["register_contract"] = "ACCEPTED"
    except Exception as ex:  # noqa: BLE001 - the message is the measurement
        meta["register_contract"] = f"REJECTED: {type(ex).__name__}: {ex}"
    print(f"register contract: {meta['register_contract']}")
    print(f"width: {meta['qubits']} vs archived {ARCHIVED_QUBITS} "
          f"(+{meta['qubits'] - ARCHIVED_QUBITS}) and reference {REFERENCE_QUBITS} "
          f"({REFERENCE_QUBITS - meta['qubits']} narrower)")

    if args.out:
        circuit.to_file(args.out, format="kmb")
        meta["circuit_file"] = str(args.out)
        meta["circuit_bytes"] = args.out.stat().st_size
        meta["circuit_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
        print(f"wrote {args.out} ({meta['circuit_bytes']:,} bytes, "
              f"sha256 {meta['circuit_sha256'][:16]}...)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
