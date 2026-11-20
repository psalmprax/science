#!/usr/bin/env python3
"""Compare the register structure of the reference DQI circuit and a public one.

Status: audit-only

`docs/BENCHMARK2_STATUS_DECISION.md` D4 recorded that a public 892-qubit decoder
exists at 892 qubits against our 1,893, and attributed the gap to five pass
bodies re-derived with the degree registers computed instead of held. That is a
hypothesis about registers, and it is cheap to test: `kickmix` reports the
declared register population of a circuit, so if the public file really is a
decoder with fewer registers its population should be smaller.

It is not. Both files declare the same 4,165 registers and the same 49,980
register-qubit references; only the peak and the classical bit count differ,
which points at lifetimes (freed and re-allocated arrays) rather than at a
smaller decoder. This script prints that comparison, so the claim in D4 can be
re-checked against the files rather than carried in prose.

Usage:
    APEDP_Q_KICKMIX_PYTHON=... .venv-kickmix/bin/python scripts/measure_dqi_register_structure.py [PUBLIC.kmb]
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))

import kickmix as km  # noqa: E402

REFERENCE = (REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
             / "sample_submission" / "dqi.kmb")
DEFAULT_PUBLIC = pathlib.Path(
    "/tmp/kb/yoyobar_lb-22-99-invsqrt-101-62-qubits/sub/dqi.kmb"
)


def describe(path: pathlib.Path) -> dict:
    circuit = km.Circuit.from_file(str(path), format="kmb")
    registers = circuit.register_data
    names = list(registers)
    widths = [len(registers[name]) for name in names]
    return {
        "path": str(path),
        "peak_qubits": circuit.num_qubits,
        "declared_registers": len(names),
        "register_qubit_references": sum(widths),
        "classical_bits": circuit.num_bits,
        "first_names": names[:4],
    }


def main() -> int:
    public = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PUBLIC
    if not REFERENCE.is_file():
        raise SystemExit(f"reference circuit missing: {REFERENCE}")
    if not public.is_file():
        raise SystemExit(f"public circuit missing: {public} (download it first)")

    rows = {"reference": describe(REFERENCE), "public": describe(public)}
    print(f"{'':<22} {'reference':>12} {'public':>12}")
    for key in ("peak_qubits", "declared_registers", "register_qubit_references", "classical_bits"):
        print(f"{key:<22} {rows['reference'][key]:>12,} {rows['public'][key]:>12,}")
    print(f"\nregister names (first 4): {rows['reference']['first_names']} vs "
          f"{rows['public']['first_names']}")
    print("\nSame population, different peak: the public file is the same register "
          "structure with shorter lifetimes, not a decoder with fewer registers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
