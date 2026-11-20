"""Pins the measured DQI degree-register offsets and the projection's basis.

Status: audit-only

`apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py` used to assert

    deg(A) + deg(a) = n
    deg(B) + deg(b) = n + 1

and `design_projection_dqi_gluzdov_v6()` derived a 1,789-qubit row from them.
`scripts/measure_dqi_recycling_offsets.py` reads the decoder's own `n_A`, `n_B`,
`n_a`, `n_b` registers after every EEA sub-step: the a-pair is constant but at
`n + 1`, and the b-pair varies, so only `n_a` can be derived. These tests pin
that measurement and the projection's declared basis, so the target cannot drift
back to an unsourced number without failing here.

The measurement needs the ``kickmix`` extension, so only the record is checked
unconditionally; the run is opt-in, like the other kickmix tests:

    APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 pytest tests/test_dyn_dqi_recycling_offsets.py
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from apedp_dyn.quantum_compilation.dqi_gluzdov_v6 import (
    PEAK_QUBITS_REFERENCE,
    PEAK_QUBITS_V6,
    RECYCLING_MEASUREMENT,
    design_projection_dqi_gluzdov_v6,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KICKMIX_PYTHON = os.environ.get("APEDP_Q_KICKMIX_PYTHON")
SCRIPT = REPO_ROOT / "scripts" / "measure_dqi_recycling_offsets.py"


def test_the_record_states_what_the_circuit_registers_do() -> None:
    assert RECYCLING_MEASUREMENT["n_A_plus_n_a"]["behaviour"] == "constant"
    assert RECYCLING_MEASUREMENT["n_A_plus_n_a"]["value_offset"] == "n + 1"
    assert RECYCLING_MEASUREMENT["n_B_plus_n_b"]["behaviour"] == "varies"
    assert RECYCLING_MEASUREMENT["recyclable"] == ["n_a"]


def test_only_one_register_is_claimed_recyclable() -> None:
    """One logm-bit register at n=70 is 8 qubits, not the 104 the target implies."""
    removable = RECYCLING_MEASUREMENT["qubits_removable_at_benchmark"]
    assert removable == 8
    assert RECYCLING_MEASUREMENT["supported_peak_qubits"] == PEAK_QUBITS_REFERENCE - removable
    assert PEAK_QUBITS_REFERENCE - PEAK_QUBITS_V6 == 104  # the old, unsourced gap
    assert RECYCLING_MEASUREMENT["supported_peak_qubits"] > PEAK_QUBITS_V6


def test_projection_carries_its_basis_and_does_not_claim_pass() -> None:
    projection = design_projection_dqi_gluzdov_v6()
    basis = projection["recycling_basis"]
    assert basis["target_status"].startswith("UNSOURCED")
    assert projection["status"] == "ANALYTICAL_PROJECTION_ONLY"
    assert projection["official_verifier_run"] is False
    assert projection["acceptance_gate"]["status"] == "HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT"


def test_measurement_script_reproduces_the_record() -> None:
    if not KICKMIX_PYTHON:
        pytest.skip("set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix")
    if not Path(KICKMIX_PYTHON).exists():
        pytest.skip(f"APEDP_Q_KICKMIX_PYTHON does not exist: {KICKMIX_PYTHON}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [KICKMIX_PYTHON, str(SCRIPT), "--valid"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    n = int(re.search(r"n=(\d+)", result.stdout).group(1))
    sums = re.findall(r"(n_A \+ n_a|n_B \+ n_b): (CONSTANT at \[([\d, ]+)\]|varies at \[([\d, ]+)\])",
                      result.stdout)
    assert len(sums) == 2, result.stdout[-2000:]
    a_name, a_state, a_values, _ = sums[0]
    b_name, b_state, _, b_values = sums[1]

    assert a_state.startswith("CONSTANT"), f"{a_name} {a_state}"
    assert [int(v) for v in a_values.split(",")] == [n + 1]  # the docs claimed n
    assert b_state.startswith("varies"), f"{b_name} {b_state}"
    assert sorted(int(v) for v in b_values.split(",")) == RECYCLING_MEASUREMENT["n_B_plus_n_b"]["observed"]
