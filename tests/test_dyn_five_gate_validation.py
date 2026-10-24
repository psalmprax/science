"""Five-gate validation programme regression.

Runs benchmarks/run_five_gate_validation.py end to end and asserts every
gate passes: exact recovery (with the Planck limitation held), honest noise
degradation, adversarial abstention, matched-compute ablation, and sealed
confirmation under a fresh seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from run_five_gate_validation import run_all_gates


def test_five_gate_programme_passes():
    results = run_all_gates()
    for name, gate in results["gates"].items():
        assert gate["passed"], f"{name} failed: {gate}"
    assert results["all_passed"]
