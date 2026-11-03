"""Write the DQI design projection to its committed artifact.

Status: audit-only

The projection is a function in `dqi_gluzdov_v6`; this is the file a reader can diff and a
test can compare against it, so the committed numbers cannot drift from the code that
computed them.

    .venv/bin/python scripts/write_dqi_design_projection.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apedp_dyn.quantum_compilation.dqi_gluzdov_v6 import (  # noqa: E402
    design_projection_dqi_gluzdov_v6,
)

OUTPUT = REPO_ROOT / "benchmarks" / "quantum_compilation" / "dqi_design_projection.json"


def main() -> int:
    projection = design_projection_dqi_gluzdov_v6()
    OUTPUT.write_text(json.dumps(projection, indent=2, sort_keys=True) + "\n")
    qubits = projection["qubits"]
    operative = projection["projection"]
    target = projection["toffoli_target"]
    print(
        f"Q {operative['peak_qubits']} (cap {qubits['qubit_cap']}), "
        f"withdrawn {qubits['withdrawn_peak_qubits']}, next {qubits['aspirational_peak_qubits']}",
        f"(gap {qubits['withdrawn_gap_to_aspirational_qubits']}), floor {qubits['peak_floor_under_recycling']}",
    )
    print(
        f"cost {operative['cost']:.6g}, gain {operative['gain_factor']:.3f}x, "
        f"T at aim {target['algorithm_floor_mean_toffolis']:,.1f} "
        f"({target['measured_frontier_mean_toffolis']:,.1f} measured, "
        f"{target['frontier_to_floor_ratio']:.3f}x it)"
    )
    print("wrote", OUTPUT.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
