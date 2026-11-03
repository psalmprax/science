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

    # The total is what gets quoted and it hides where it came from: the three families are
    # not comparable, and on the regular ones the saving thins with width. Both cuts are
    # written here so a reader (and `tests/test_documentation_and_spec_integrity.py`) can
    # check the headline against the family and width it is made of.
    families: dict[str, dict[str, int]] = {}
    by_width: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        fam = families.setdefault(
            row["family"],
            {
                "targets": 0,
                "wins": 0,
                "ties": 0,
                "baseline_two_qubit_gates": 0,
                "saved_two_qubit_gates": 0,
            },
        )
        band = by_width.setdefault(row["family"], {}).setdefault(
            str(row["width"]),
            {
                "targets": 0,
                "wins": 0,
                "baseline_two_qubit_gates": 0,
                "saved_two_qubit_gates": 0,
            },
        )
        for block in (fam, band):
            block["targets"] += 1
            block["baseline_two_qubit_gates"] += row["baseline_two_qubit_gates"]
            block["saved_two_qubit_gates"] += row["saved_two_qubit_gates"]
            if row["saved_two_qubit_gates"] > 0:
                block["wins"] += 1
        if row["saved_two_qubit_gates"] == 0:
            fam["ties"] += 1
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
        "families": families,
        "by_width": by_width,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{wins} wins / {ties} ties / {losses} losses on {len(rows)} graph states; "
          f"{payload['saved_two_qubit_gates']} two-qubit gates saved "
          f"({payload['baseline_two_qubit_gates']} -> {payload['mmlm_two_qubit_gates']}); "
          f"{payload['certified_valid']}/{len(rows)} certified VALID")
    for name, block in sorted(families.items()):
        print(
            f"  {name:<13} {block['wins']:>2}/{block['targets']} wins, "
            f"{block['saved_two_qubit_gates']:>4} of {block['baseline_two_qubit_gates']:>5} gates"
        )
    for name, bands in sorted(by_width.items()):
        cells = ", ".join(
            f"{width}q={band['saved_two_qubit_gates']}({band['wins']}/{band['targets']})"
            for width, band in sorted(bands.items(), key=lambda item: int(item[0]))
        )
        print(f"  {name:<13} {cells}")
    print("wrote", OUTPUT.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
