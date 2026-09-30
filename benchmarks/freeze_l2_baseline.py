"""Freezes the APEDP L2 Control Group Baseline.

Calculates SHA-256 digests for all L2 discovery and epistemic adjudication modules,
recording git commit hash and timestamp to ensure complete cryptographic reproducibility
before executing the strategic roadmap.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORE_FILES = (
    "apedp_dyn/discovery_l2/ast_generator.py",
    "apedp_dyn/discovery_l2/dimensional_types.py",
    "apedp_dyn/discovery_l2/pruning.py",
    "apedp_dyn/discovery_l2/l2_engine.py",
    "apedp_dyn/discovery_l2/ode_engine.py",
    "apedp_dyn/discovery_l2/active_intervention.py",
    "apedp_dyn/epistemic/adjudicate.py",
    "apedp_dyn/epistemic/calibration.py",
    "apedp_dyn/epistemic/bootstrap.py",
    "apedp_dyn/estimation/weak_form.py",
)


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def freeze_control_baseline() -> dict:
    git_head = get_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()

    file_digests = {}
    for rel_path in CORE_FILES:
        full_path = ROOT / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Core module missing: {full_path}")
        content = full_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        file_digests[rel_path] = {
            "sha256": digest,
            "byte_count": len(content),
            "line_count": len(content.splitlines()),
        }

    record = {
        "status": "FROZEN_L2_CONTROL_GROUP",
        "timestamp_utc": timestamp,
        "git_head": git_head,
        "core_modules": file_digests,
        "frozen_grammar_scope": {
            "operators": ["+", "-", "*", "/", "pow"],
            "max_depth": 2,
            "max_complexity": 5,
            "physical_typing": "SI_dimension_calculus",
            "differentiation": ["spline", "central_diff", "weak_form"],
            "adjudication_gates": ["gate_a_algebraic", "gate_b_dimensional", "gate_c_evidential", "guard1_ceiling", "guard2_whiteness", "tau_epi_margin", "block_bootstrap"],
        },
    }

    out_file = ROOT / "benchmarks" / "l2_control_freeze.json"
    out_file.write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    rec = freeze_control_baseline()
    print(f"Successfully frozen L2 Control Group at {rec['timestamp_utc']}")
    print(f"Git HEAD: {rec['git_head']}")
    print(f"Recorded digests for {len(rec['core_modules'])} core modules in benchmarks/l2_control_freeze.json")
