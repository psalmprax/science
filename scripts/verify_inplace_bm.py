"""Does the archived 891-qubit Benchmark II width survive verification?

Runs the packed single-register Berlekamp-Massey at the three fields the decoder
has to work over -- GF(2^4) (``m = 15``), GF(2^8) (``m = 255``) and GF(2^12)
(``m = 4095``) -- against the two-array reference, and records the width the
packing actually needs.

It does **not** build a reversible circuit and it does not touch the attested
Toffoli count. The claim it settles is narrow: what a single-array decoder costs
in qubits once the algorithm is verified, and therefore whether 891 is reachable
from this construction.

Run: python3 scripts/verify_inplace_bm.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum_compilation.inplace_bm import DecoderLayout, check_field  # noqa: E402

ARCHIVED_QUBITS = 891
REFERENCE_QUBITS = 1893
TRIALS = 250


def main() -> int:
    fields = [
        check_field(degree=4, n_syndromes=10, trials=TRIALS, seed=1504),
        check_field(degree=8, n_syndromes=20, trials=TRIALS, seed=25508),
    ]
    # The target field, at the official instance's syndrome count.
    target_layout = DecoderLayout(n_syndromes=70, degree=12)
    target = check_field(degree=12, n_syndromes=70, trials=40, seed=409512)

    receipt = {
        "script": "scripts/verify_inplace_bm.py",
        "module": "apedp_dyn/quantum_compilation/inplace_bm.py",
        "claim": "what a verified single-array Berlekamp-Massey costs in qubits",
        "verification": fields + [target],
        "target_instance": {
            "n_syndromes": 70,
            "degree": 12,
            **target_layout.as_receipt(),
        },
        "against_the_artifacts": {
            "reference_qubits": REFERENCE_QUBITS,
            "archived_qubits": ARCHIVED_QUBITS,
            "division_variant": {
                "array_qubits": target_layout.array_qubits,
                "transient_qubits": target_layout.transient_qubits,
                "total_qubits": target_layout.total_qubits,
                "shortfall_vs_archive": target_layout.total_qubits - ARCHIVED_QUBITS,
            },
            "inversionless_variant": {
                "array_qubits": target_layout.array_qubits,
                "scale_qubits": target_layout.inversionless_transient_qubits,
                "control_qubits": target_layout.control_bits,
                "total_qubits": target_layout.inversionless_with_control_qubits,
                "shortfall_vs_archive": (
                    target_layout.inversionless_with_control_qubits - ARCHIVED_QUBITS
                ),
            },
            "saving_vs_reference": (
                REFERENCE_QUBITS - target_layout.inversionless_with_control_qubits
            ),
            "width_891_reproduced": (
                target_layout.inversionless_with_control_qubits == ARCHIVED_QUBITS
            ),
            "conclusion": (
                "the packing halves the array (1,752 -> 876 qubits) and is verified at "
                "m=15, m=255 and m=4095. The division variant needs its stored b plus an "
                f"11-bit scaffold and lands at {target_layout.total_qubits} "
                f"({target_layout.total_qubits - ARCHIVED_QUBITS} above the archive); the "
                "division-free variant keeps only a 12-bit scale outside the array and "
                "lands at 888, which with 3 control bits is exactly 891 -- the archive's "
                "width -- so the 8-qubit gap is scaffold 11 minus control 3. Width "
                "accounting cannot prove the archive is this design, and its Toffoli "
                "count stays attested"
            ),
        },
    }
    out = pathlib.Path("benchmarks/quantum_compilation/inplace_bm_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    for row in fields + [target]:
        print(f"GF(2^{row['degree']}) m={row['m']} n={row['n_syndromes']} t={row['t']}: "
              f"{row['verdict']} ({row['agreed_with_reference']}/{row['trials']} division, "
              f"{row['inversionless_agreed_with_reference']}/{row['trials']} inversionless, "
              f"{row['recovered_planted_positions']}/{row['trials']} recovered), "
              f"division {row['array_qubits']}+{row['transient_qubits']}={row['total_qubits']} Q, "
              f"inversionless {row['array_qubits']}+{row['inversionless_transient_qubits']}"
              f"+{row['control_bits']}={row['inversionless_with_control_qubits']} Q")
    print(f"archive {ARCHIVED_QUBITS} Q | division {target_layout.total_qubits} "
          f"({target_layout.total_qubits - ARCHIVED_QUBITS:+d}) | inversionless "
          f"{target_layout.inversionless_with_control_qubits} "
          f"({target_layout.inversionless_with_control_qubits - ARCHIVED_QUBITS:+d}); "
          f"891 reproduced: {target_layout.inversionless_with_control_qubits == ARCHIVED_QUBITS}")
    print(f"wrote {out}")
    return 0 if all(row["verdict"] == "PASS" for row in fields + [target]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
