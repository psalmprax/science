"""Record the MMLM graph-state result as numbers rather than inequalities.

`tests/test_apedp_mmlm_architecture.py` used to assert `wins >= 47` and
`total_saved >= 300`, which pins a floor but never states the result. This writes
the per-target table and the totals to
`benchmarks/quantum_compilation/vqc_mmlm_graph_state.json`, with every circuit
checked by the corpus's own referee on the way through, and the test then compares
the corpus against that record.

    .venv/bin/python scripts/measure_vqc_mmlm_graph_states.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apedp_dyn.vqc.corpus import CORPUS  # noqa: E402
from apedp_dyn.vqc.synthesis import adjudicate_specification, graph_of_state  # noqa: E402
from apedp_mmlm import synthesize_graph_state_mmlm  # noqa: E402

OUTPUT = REPO_ROOT / "benchmarks" / "quantum_compilation" / "vqc_mmlm_graph_state.json"


def main() -> int:
    rows = []
    for entry in CORPUS:
        if getattr(entry, "kind", "") == "random_stabilizer":
            continue
        baseline = len(graph_of_state(entry.target))
        construction = synthesize_graph_state_mmlm(entry.target)
        mmlm = int(construction.circuit.resources()["two_qubit_gates"])
        report = adjudicate_specification(construction.circuit, entry.target)
        rows.append({
            "label": entry.label,
            "family": getattr(entry, "family", getattr(entry, "kind", "unknown")),
            "width": int(entry.target.num_qubits),
            "baseline_two_qubit_gates": baseline,
            "mmlm_two_qubit_gates": mmlm,
            "saved_two_qubit_gates": baseline - mmlm,
            "verdict": report.verdict,
        })

    wins = sum(1 for row in rows if row["saved_two_qubit_gates"] > 0)
    ties = sum(1 for row in rows if row["saved_two_qubit_gates"] == 0)
    losses = len(rows) - wins - ties
    payload = {
        "benchmark": "vqc_mmlm_graph_state",
        "what": "MMLM local-complementation + symplectic CX graph-state synthesis against one CZ per edge.",
        "generated_by": "scripts/measure_vqc_mmlm_graph_states.py",
        "targets": len(rows),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "saved_two_qubit_gates": sum(row["saved_two_qubit_gates"] for row in rows),
        "baseline_two_qubit_gates": sum(row["baseline_two_qubit_gates"] for row in rows),
        "mmlm_two_qubit_gates": sum(row["mmlm_two_qubit_gates"] for row in rows),
        "certified_valid": sum(1 for row in rows if row["verdict"] == "VALID"),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{wins} wins / {ties} ties / {losses} losses on {len(rows)} graph states; "
          f"{payload['saved_two_qubit_gates']} two-qubit gates saved "
          f"({payload['baseline_two_qubit_gates']} -> {payload['mmlm_two_qubit_gates']}); "
          f"{payload['certified_valid']}/{len(rows)} certified VALID")
    print("wrote", OUTPUT.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
