#!/usr/bin/env python3
"""Run the halting-counter recycling idea up the DQI size ladder, piece by piece.

Status: audit-only

``scripts/test_recycled_flags_small.py`` replaces two comparisons inside
``ProosAndZalkaDecoder`` with arithmetic on ``n_B`` and reports a Toffoli saving
on the toy instance ``m=15, n=6, GF(2^4)``. A toy percentage is not a benchmark
result, and the two replacements are independent, so this script measures each
one separately at several instances:

  * ``flag``    - ``_update_flag`` rewritten without reading ``n_a``/``n_b``,
  * ``counter`` - ``_update_halting_counter`` rewritten without reading ``n_b``,
  * ``both``    - the two together (the shipped prototype).

Toffoli counts are exact properties of the emitted circuit, so they need no
shots. Correctness is verified with the benchmark verifier only where the shot
budget is affordable; nothing is claimed about the full instance beyond its
Toffoli count.

    /dev/shm/qvcc-venv313/bin/python scripts/run_halt_recycling_scale.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import kickmix as km
import dqi_reference_circuit as DR
from test_recycled_flags_small import RecycledFlagDecoder
from verify_submission_dqi import verify, ProblemSpecDQI, default_num_accepted_values

N_ITERS = 30
# (m, n, degree); the decoder requires n <= m <= 2**degree - 1.
LADDER = [(15, 6, 4), (63, 10, 6), (255, 14, 8), (1023, 30, 10)]


class CounterOnly(DR.ProosAndZalkaDecoder):
    """Only the halting counter is recycled."""

    _update_halting_counter = RecycledFlagDecoder._update_halting_counter


class FlagOnly(DR.ProosAndZalkaDecoder):
    """Only the flag is recycled."""

    _update_flag = RecycledFlagDecoder._update_flag


VARIANTS = [
    ("reference", DR.ProosAndZalkaDecoder),
    ("flag", FlagOnly),
    ("counter", CounterOnly),
    ("both", RecycledFlagDecoder),
]


def build(decoder_class: type, m: int, n: int, degree: int) -> km.Circuit:
    gf = km.GF2Field(degree)
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    errors = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    decoder = decoder_class(n, gf)
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=errors, n_iters=N_ITERS
    )
    return builder.finish_circuit()


def main() -> None:
    rows = []
    for m, n, degree in LADDER:
        row = {"m": m, "n": n, "degree": degree, "variants": {}}
        for label, cls in VARIANTS:
            start = time.perf_counter()
            circuit = build(cls, m, n, degree)
            stats = circuit.operation_statistics()
            row["variants"][label] = {
                "toffoli": float(stats.num_toffoli_like),
                "qubits": int(circuit.num_qubits),
                "seconds": round(time.perf_counter() - start, 2),
            }
        reference = row["variants"]["reference"]["toffoli"]
        for label in ("flag", "counter", "both"):
            delta = row["variants"][label]["toffoli"] - reference
            row["variants"][label]["delta"] = delta
            row["variants"][label]["pct"] = 100.0 * delta / reference if reference else 0.0
        rows.append(row)
        print(f"m={m:5d} n={n:3d} b={degree:2d}  reference {reference:14,.0f} Toffoli")
        for label in ("flag", "counter", "both"):
            entry = row["variants"][label]
            print(f"    {label:8s} {entry['toffoli']:14,.0f}  delta {entry['delta']:+12,.0f} "
                  f"({entry['pct']:+.3f}%)  Q={entry['qubits']}  {entry['seconds']:.1f}s")
        sys.stdout.flush()

    # Correctness at the toy instance, where shots are affordable.
    m, n, degree = LADDER[0]
    spec = ProblemSpecDQI(m=m, n=n, degree=degree,
                          r=default_num_accepted_values(degree), num_shots=256)
    print("\nToy-instance verification with the benchmark verifier:")
    for label, cls in (("reference", DR.ProosAndZalkaDecoder), ("counter", CounterOnly)):
        circuit = build(cls, m, n, degree)
        result = verify(circuit, spec, max_workers=1, batch_size=256, log2_delta=-10.0)
        print(f"  {label:9s} status={result['status']} score={result['score']:.3f} "
              f"toffoli={result['toffoli_count']:.0f} Q={result['num_qubits']}")
        rows[0]["variants"][label]["verified"] = {
            "status": result["status"], "score": result["score"],
        }

    out = REPO_ROOT / "artifacts/quantum_compilation/dqi/halt_recycling_scale.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_iters": N_ITERS, "ladder": rows}, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
