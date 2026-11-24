"""Pins the bit-serial schedule prototype: its widths and its exact-phase result.

Status: audit-only

``scripts/prototype_bitserial_sqrt.py`` builds the same digit arithmetic two
ways -- one reusable digit qubit phased as each digit appears, versus a full
digit register phased at the end -- and checks both against the exact phase with
the competition's own simulator. This test pins what that script measured, so a
schedule change cannot move the widths or the exactness quietly.

The build needs the ``kickmix`` extension (CPython 3.13), so it is opt-in and
follows the same convention as ``tests/test_dyn_invsqrt_champion_rebuild.py``:

    APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 pytest tests/test_dyn_bitserial_prototype.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KICKMIX_PYTHON = os.environ.get("APEDP_Q_KICKMIX_PYTHON")
SCRIPT = REPO_ROOT / "scripts/prototype_bitserial_sqrt.py"

#: Measured on the widths this prototype covers. `inputs` is every `w < 2^(2k)`,
#: so the phase check is exhaustive for those k rather than sampled.
PINNED = {
    3: {"bitserial_q": 15, "materialised_q": 17, "toffoli": 22, "inputs": 64},
    4: {"bitserial_q": 29, "materialised_q": 32, "toffoli": 60, "inputs": 256},
    5: {"bitserial_q": 58, "materialised_q": 62, "toffoli": 152, "inputs": 1024},
}

#: Digit `i` is a function of the top `2(k-i)` bits of `w`; measured, not assumed.
PINNED_PREFIX_BITS = {
    3: [2, 4, 6],
    4: [2, 4, 6, 8],
    5: [2, 4, 6, 8, 10],
    6: [2, 4, 6, 8, 10, 12],
}


@pytest.fixture(scope="module")
def rows() -> dict[int, dict]:
    if not KICKMIX_PYTHON:
        pytest.skip("set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix")
    if not Path(KICKMIX_PYTHON).exists():
        pytest.skip(f"APEDP_Q_KICKMIX_PYTHON does not exist: {KICKMIX_PYTHON}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [KICKMIX_PYTHON, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    payloads = [line for line in result.stdout.strip().splitlines() if line.startswith("[")]
    assert payloads, f"no JSON payload in output:\n{result.stdout[-4000:]}"
    parsed = json.loads(payloads[-1])
    return {row["k"]: row for row in parsed}


@pytest.mark.parametrize("k", sorted(PINNED))
def test_widths_and_toffolis_are_pinned(rows: dict[int, dict], k: int) -> None:
    row = rows[k]
    pin = PINNED[k]
    assert row["bitserial"]["qubits"] == pin["bitserial_q"], row["bitserial"]
    assert row["materialised"]["qubits"] == pin["materialised_q"], row["materialised"]
    # The schedule moves the digit register and nothing else: the Boolean work --
    # compute and uncompute, twice -- is identical between the two builds.
    assert row["bitserial"]["toffoli"] == pin["toffoli"]
    assert row["materialised"]["toffoli"] == pin["toffoli"]


@pytest.mark.parametrize("k", sorted(PINNED_PREFIX_BITS))
def test_each_digit_needs_exactly_its_prefix(rows: dict[int, dict], k: int) -> None:
    assert rows[k]["prefix_bits"] == PINNED_PREFIX_BITS[k]


@pytest.mark.parametrize("k", sorted(PINNED))
def test_phase_is_exact_for_every_input(rows: dict[int, dict], k: int) -> None:
    row = rows[k]
    pin = PINNED[k]
    for variant in ("verification", "control"):
        check = row[variant]
        assert check is not None, f"k={k} {variant} was not simulated"
        assert check["inputs_checked"] == pin["inputs"]
        assert check["worst_phase_error_turns"] == 0.0, (
            f"k={k} {variant} phase error {check['worst_phase_error_turns']:.3e} turns"
        )


def test_streaming_needs_the_fewer_register(rows: dict[int, dict]) -> None:
    """The saving is the digit register: k bits held versus one reusable qubit."""
    for k in sorted(PINNED):
        saving = rows[k]["materialised"]["qubits"] - rows[k]["bitserial"]["qubits"]
        assert saving == k - 1, f"k={k}: saved {saving} qubits, expected {k - 1}"
