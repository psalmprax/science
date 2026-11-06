"""Pins for the phase-safe 54-AND multiply (`scripts/gf54_mul_reduced.py`).

Experiment D2 showed that clearing a product window with `hmr` dephases it, so the
54-AND multiply could not be emitted by the pinned competition binary at all. This
emitter accumulates the field product **already reduced** into a 12-qubit window, so
the one thing that needs clearing really is `lhs * rhs` — which `del_gf2_mul`
measures and phase-corrects from the factors, for 0 Toffolis and no workspace.

These pins are behavioural, on the pinned extension: the function, the junk, the
tracked phase, the Toffoli cost, and the absence of `mr`/`hmr` from the stream. They
skip, rather than fail, if the pinned build is not importable for this interpreter.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED = (REPO_ROOT / "apedp_dyn" / "sandbox"
          / "google-verifiable-quantum-circuit-compilation" / "kickmix")
EMITTER = REPO_ROOT / "scripts" / "gf54_mul_reduced.py"

#: 54 ANDs for the product, 12 for the controlled add. No clear contributes: both are
#: measurements with classical fixups.
UNCONTROLLED_TOFFOLI = 54
CONTROLLED_TOFFOLI = 66
#: The built-in `ixor_gf2_mul` pays 63 ANDs plus the same 12-Toffoli add.
BUILTIN_CONTROLLED_TOFFOLI = 75


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def km():
    if not (PINNED / "kickmix.so").is_file() and not (PINNED / "kickmix").is_dir():
        pytest.skip("pinned kickmix extension not available")
    sys.path.insert(0, str(PINNED))
    try:
        import kickmix as module
    except ImportError:  # pragma: no cover - build-specific
        pytest.skip("pinned kickmix extension not importable")
    if not hasattr(module.CircuitBuilder, "del_gf2_mul"):
        pytest.skip("this build has no del_gf2_mul")
    return module


@pytest.fixture(scope="module")
def emitter():
    return _load("gf54_mul_reduced", EMITTER)


def _bits(values, width: int = 12) -> np.ndarray:
    return np.array([[(int(v) >> i) & 1 for i in range(width)] for v in values],
                    dtype=bool).T


def _run(km, emitter, *, controlled: bool, shots: int = 6, field_degree: int = 12):
    builder = km.CircuitBuilder()
    field = km.GF2Field(field_degree)
    lhs = builder.alloc_qubits(field_degree)
    rhs = builder.alloc_qubits(field_degree)
    target = builder.alloc_qubits(field_degree)
    ctrl = builder.alloc_qubits(1)
    if controlled:
        builder.x(ctrl[0])
        emitter.emit_gf54_mul_reduced(builder, lhs, rhs, target=target, field=field,
                                      control=ctrl[0])
    else:
        emitter.emit_gf54_mul_reduced(builder, lhs, rhs, target=target, field=field)
    circuit = builder.finish_circuit()

    rng = np.random.default_rng(11)
    a = rng.integers(0, 1 << field_degree, size=shots)
    b = rng.integers(0, 1 << field_degree, size=shots)
    sim = km.Simulator(batch_size=shots, seed=5, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array(list(lhs)), _bits(a, field_degree))
    sim.write_across_shots(km.array(list(rhs)), _bits(b, field_degree))
    sim.do(circuit)

    out = np.asarray(sim.read_across_shots(list(target)))
    got = np.array([sum(int(out[k, s]) << k for k in range(field_degree))
                    for s in range(shots)])
    want = np.array([int(field.mul(int(a[s]), int(b[s]))) for s in range(shots)])
    kept = {q.id for q in list(lhs) + list(rhs) + list(target)} | {ctrl[0].id}
    junk = [i for i in range(circuit.num_qubits) if i not in kept]
    return {
        "matches": bool(np.array_equal(got, want)),
        "junk_clean": bool(not np.asarray(sim.read_across_shots(
            [km.q(i) for i in junk])).any()),
        "phase_flips": int(np.asarray(sim.read_phase_flipped_across_shots()).sum()),
        "toffoli": int(sim.operation_statistics(0).num_toffoli_like),
        "qubits": circuit.num_qubits,
        "circuit": circuit,
    }


def test_the_uncontrolled_product_is_correct_and_clean(km, emitter):
    row = _run(km, emitter, controlled=False)
    assert row["matches"] is True
    assert row["junk_clean"] is True
    assert row["phase_flips"] == 0
    assert row["toffoli"] == UNCONTROLLED_TOFFOLI


def test_the_controlled_product_is_correct_and_phase_safe(km, emitter):
    row = _run(km, emitter, controlled=True)
    assert row["matches"] is True
    # The window is cleared by `del_gf2_mul`: a measurement plus classical CZ fixups.
    # An `hmr` clear of the same window is what Experiment D2 measured dephasing.
    assert row["junk_clean"] is True
    assert row["phase_flips"] == 0
    assert row["toffoli"] == CONTROLLED_TOFFOLI
    assert row["toffoli"] < BUILTIN_CONTROLLED_TOFFOLI


def test_every_measured_clear_in_the_stream_carries_a_phase_fixup(km, emitter):
    """No Z-basis `MR` and no standalone `H`; every `HMR` is paired with a fixup.

    `del_and` and `del_gf2_mul` both measure with `HMR` — that is the point, and it is
    what Experiment D2 showed is *not* safe on its own. What makes them safe is the
    classically conditioned `CZ` that follows, computed from the operands: the tracked
    phase is restored rather than left to the measurement outcome.
    """
    row = _run(km, emitter, controlled=True)
    lines = [line.strip() for line in str(row["circuit"]).splitlines()]
    assert not any(line.startswith("MR ") for line in lines), "a Z-basis read"
    assert not any(line.startswith("H ") for line in lines), "a standalone Hadamard"
    clears = sum(1 for line in lines if "HMR" in line)
    fixups = sum(1 for line in lines if "CZ" in line and " if " in line)
    assert clears > 0
    assert fixups >= clears


def test_the_window_is_twelve_qubits_not_nineteen(km, emitter):
    """The reduced-window design is smaller in scratch, not larger."""
    uncontrolled = _run(km, emitter, controlled=False)
    controlled = _run(km, emitter, controlled=True)
    # lhs + rhs + target (+ control), plus one AND target and the operand XOR scratch.
    assert uncontrolled["qubits"] <= 3 * 12 + 8
    assert controlled["qubits"] <= 3 * 12 + 1 + 12 + 8
