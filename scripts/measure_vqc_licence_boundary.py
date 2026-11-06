#!/usr/bin/env python3
"""Measure the licence boundary of the read-then-gate arms, at amplitude level.

`docs/EEA_SCHEDULE_CAMPAIGN.md` licenses every gate in the DQI arm on P1: *the
register being read holds a computational-basis value*. That is a statement about
the inputs, and it is the whole licence. This script measures what it means by
comparing the two maps directly:

* **the reference map** -- the controlled operation the decoder emits, e.g.
  ``cswap(control, x, y)``;
* **the gated map** -- what the arms replace it with: copy the control into a
  shared ancilla, measure the ancilla, and apply the operation only under the
  classical outcome.

On every computational-basis control value the two are the *same state*, exactly
and including the classical bit. On a control in superposition they are not the
same map at all: the gated one is a channel (a mixture of trajectories) and the
coherent branch structure is gone. The recorded number is the fidelity of the
mixture against the reference state, so the record says how much of the arm's
licence survives outside its own preconditions rather than asserting that it
does.

The script also settles the one substitution that would have made the arm
constructible on the pinned competition binary: ``MR`` is ``HMR`` with one extra
``H`` in front *if the builder has an ``H`` gate*. It does not
(Experiment D records the surface), so the identity is measured here as algebra
that is correct and unavailable -- and ``HMR`` alone is measured to be a random
outcome on a basis state, which is why it cannot stand in for ``MR``.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_vqc_licence_boundary.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_licence_boundary.json"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"

sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum_compilation.statevector_sim import (  # noqa: E402
    StatevectorSimulator,
    dimension_report,
    fidelity,
    global_phase_free_equality,
)

#: The four qubits of the gadget: control, readout ancilla, and the swapped pair.
CONTROL, ANCILLA, LEFT, RIGHT = 0, 1, 2, 3


def reference_ops() -> list:
    """The map the decoder emits: a controlled exchange, no measurement."""
    return [("gate", "cswap", (CONTROL, LEFT, RIGHT))]


def gated_ops() -> list:
    """The map the arms emit: copy, measure, and act under the classical outcome."""
    return [
        ("gate", "cx", (CONTROL, ANCILLA)),
        ("measure", ANCILLA, 0),
        ("conditioned", 0, [("gate", "swap", (LEFT, RIGHT))]),
    ]


def basis_inputs() -> list[tuple[int, int, int, int, int]]:
    """``(control, left, right, ancilla, index)`` for every basis input of the gadget."""
    out = []
    for control in (0, 1):
        for left in (0, 1):
            for right in (0, 1):
                index = control | (left << LEFT) | (right << RIGHT)
                out.append((control, left, right, 0, index))
    return out


def ancilla_probability_one(state) -> float:
    """The probability that the readout ancilla is 1, i.e. that it was not reset."""
    amplitudes = np.asarray(state, dtype=np.complex128).reshape(-1)
    index = np.arange(amplitudes.size)
    return float(np.sum(np.abs(amplitudes[((index >> ANCILLA) & 1).astype(bool)]) ** 2))


def run_ops(num_qubits: int, initial: int, ops) -> list:
    sim = StatevectorSimulator(num_qubits)
    sim.set_basis_state(initial)
    return sim.branches(ops)


def basis_comparison() -> dict:
    """The two maps on every computational-basis input."""
    rows = []
    for control, left, right, ancilla, index in basis_inputs():
        reference = run_ops(4, index, reference_ops())[0]
        gated = run_ops(4, index, gated_ops())
        row = {
            "input": {"control": control, "left": left, "right": right,
                      "ancilla": ancilla},
            "reference_branches": len(run_ops(4, index, reference_ops())),
            "gated_branches": len(gated),
            "classical_bit_equals_control": bool(gated[0].bits[0] == bool(control)),
            "states_equal_up_to_phase": bool(
                len(gated) == 1
                and global_phase_free_equality(reference.state, gated[0].state)),
            "ancilla_returns_to_zero": bool(
                len(gated) == 1 and ancilla_probability_one(gated[0].state) < 1e-12),
        }
        rows.append(row)
    return {
        "rows": rows,
        "all_equal": all(row["states_equal_up_to_phase"]
                         and row["classical_bit_equals_control"]
                         for row in rows),
        "gadget": {"qubits": 4, "control": CONTROL, "ancilla": ANCILLA,
                   "swapped": [LEFT, RIGHT]},
    }


def superposition_comparison() -> dict:
    """What the two maps do to ``(|0> + |1>)/sqrt(2)`` on the control."""
    # |control> = |+>, left = |1>, right = |0>: the swapped pair differs, so the
    # reference entangles the control with the pair instead of doing nothing.
    initial = (1 << LEFT)
    reference_sim = StatevectorSimulator(4)
    reference_sim.set_basis_state(initial)
    reference_sim.h(CONTROL)
    reference = reference_sim.branches(reference_ops())[0]

    gated_sim = StatevectorSimulator(4)
    gated_sim.set_basis_state(initial)
    gated_sim.h(CONTROL)
    gated = gated_sim.branches(gated_ops())

    control_p0, control_p1 = reference_sim.probabilities(CONTROL)
    # The reference leaves the control coherent: the reduced state of the control
    # is still |+>, so its off-diagonal is 1/2.
    coherence = reference_sim.state.reshape(-1)
    zero_half = coherence[[index for index in range(16)
                           if not (index >> CONTROL) & 1]]
    one_half = coherence[[index for index in range(16)
                          if (index >> CONTROL) & 1]]
    reference_coherence = float(abs(np.vdot(zero_half, one_half)))

    gated_coherence = []
    for branch in gated:
        state = branch.state
        z = state[[index for index in range(16) if not (index >> CONTROL) & 1]]
        o = state[[index for index in range(16) if (index >> CONTROL) & 1]]
        gated_coherence.append(branch.probability * float(abs(np.vdot(z, o))))

    return {
        "input": "control = |+>, left = |1>, right = |0>",
        "reference_branches": 1,
        "gated_branches": len(gated),
        "gated_branch_weights": [branch.probability for branch in gated],
        "gated_branch_bits": [list(branch.bits) for branch in gated],
        "fidelity_of_gated_mixture_against_reference": fidelity(reference.state, gated),
        "reference_control_offdiagonal": reference_coherence,
        "gated_control_offdiagonal": float(sum(gated_coherence)),
        "reference_control_p": [control_p0, control_p1],
        "states_equal_up_to_phase": any(
            len(gated) == 1 and global_phase_free_equality(reference.state, b.state)
            for b in gated),
    }


def readout_identity() -> dict:
    """``HMR`` is one ``H`` short of ``MR``; with the ``H`` it is ``MR`` exactly."""
    # `HMR` as the builder defines it: X-basis measurement and Z-basis reset, i.e.
    # `h` then a Z-basis measure-and-reset.
    hmr = [("gate", "h", (CONTROL,)), ("measure", CONTROL, 0)]
    # The claimed substitution for `MR`: an `H` in front of `HMR`.
    substituted = [("gate", "h", (CONTROL,))] + hmr
    mr = [("measure", CONTROL, 0)]

    def branches(ops, prepare_h: bool):
        sim = StatevectorSimulator(4)
        sim.set_basis_state(0)
        if prepare_h:
            sim.h(CONTROL)
        return sim.branches(ops)

    hmr_basis = branches(hmr, prepare_h=False)
    hmr_plus = branches([("gate", "h", (CONTROL,)), ("gate", "h", (CONTROL,)),
                         ("measure", CONTROL, 0)], prepare_h=False)
    substituted_basis = branches(substituted, prepare_h=False)
    mr_basis = branches(mr, prepare_h=False)

    # On a superposed input too, where the substitution has to agree as a channel.
    hmr_super = branches(hmr, prepare_h=True)
    substituted_super = branches(substituted, prepare_h=True)
    mr_super = branches(mr, prepare_h=True)

    def same(left, right) -> bool:
        if len(left) != len(right):
            return False
        return all(
            abs(a.probability - b.probability) < 1e-12
            and a.bits == b.bits
            and global_phase_free_equality(a.state, b.state)
            for a, b in zip(sorted(left, key=lambda x: x.bits),
                            sorted(right, key=lambda x: x.bits), strict=True)
        )

    return {
        "question": "can the pinned binary's HMR stand in for the MR instruction the "
                    "arms use?",
        "hmr_on_zero": {
            "branches": [branch.as_dict() for branch in hmr_basis],
            "outcome_is_random": len(hmr_basis) == 2
            and all(abs(branch.probability - 0.5) < 1e-12 for branch in hmr_basis),
            "matches_mr": same(hmr_basis, mr_basis),
        },
        "h_then_hmr": {
            "branches": [branch.as_dict() for branch in substituted_basis],
            "matches_mr_on_basis": same(substituted_basis, mr_basis),
            "matches_mr_on_superposition": same(substituted_super, mr_super),
        },
        "double_h_hmr": {
            "note": "the substitution applied twice is the same map, since H*H = I",
            "matches_mr_on_basis": same(hmr_plus, mr_basis),
        },
        "conclusion": "the substitution is exact algebra and needs a Hadamard gate, "
                      "which the pinned competition binary's builder does not expose "
                      "(Experiment D records the surface); HMR on its own measures in "
                      "the X basis, so it randomises a basis-state qubit instead of "
                      "reporting it",
    }


def kickmix_cross_check() -> dict:
    """Cross-check the tier against `kickmix` on the gated gadget's basis inputs."""
    sys.path.insert(0, str(DEFAULT_BUILD))
    try:
        import kickmix as km
    except ImportError as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"kickmix not importable: {exc}"}
    if not hasattr(km.CircuitBuilder, "mr"):
        return {"available": False, "reason": f"{km.__file__} has no mr method"}

    builder = km.CircuitBuilder()
    control = builder.alloc_qubits(1)[0]
    ancilla = builder.alloc_qubits(1)[0]
    left = builder.alloc_qubits(1)[0]
    right = builder.alloc_qubits(1)[0]
    bit = builder.alloc_bits(1)
    builder.cx(control, ancilla)
    builder.mr(ancilla, bit[0])
    builder.push_condition(bit[0])
    builder.swap(left, right)
    builder.pop_condition()
    circuit = builder.finish_circuit()

    rows = []
    for value in range(8):
        sim = km.Simulator(batch_size=1, seed=7)
        sim.use_same_registers_as(circuit)
        sim.write_across_shots(km.array([control, ancilla, left, right]),
                               np.array([[bool((value >> i) & 1) for i in range(4)]]).T)
        sim.do(circuit)
        observed = np.asarray(sim.read_across_shots(
            km.array([control, ancilla, left, right]))).reshape(-1)
        kickmix_value = int(sum(int(bool(observed[i])) << i for i in range(4)))
        mine = run_ops(4, value, gated_ops())[0]
        mine_value = int(np.argmax(np.abs(mine.state)))
        rows.append({"input": value, "kickmix": kickmix_value, "statevector": mine_value})
    return {
        "available": True,
        "extension": str(Path(km.__file__).resolve()),
        "rows": rows,
        "agree": all(row["kickmix"] == row["statevector"] for row in rows),
        "note": "the local MR rebuild is the only engine that can execute this gadget; "
                "the pinned binary cannot (no `mr`)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--skip-kickmix", action="store_true")
    args = parser.parse_args()

    basis = basis_comparison()
    superposition = superposition_comparison()
    readout = readout_identity()
    cross = ({"available": False, "reason": "skipped"} if args.skip_kickmix
             else kickmix_cross_check())

    checks = [{
        "check": "on every computational-basis input the two maps are the same state",
        "requirement": "all 8 inputs equal up to global phase, the bit equals the control, "
                       "and the ancilla returns to |0>",
        "measured": {"inputs": len(basis["rows"]), "all_equal": basis["all_equal"],
                     "ancilla_clean": all(row["ancilla_returns_to_zero"]
                                          for row in basis["rows"])},
        "passed": basis["all_equal"],
    }, {
        "check": "outside the basis the two maps differ, and by how much",
        "requirement": "the gated map is a mixture (more than one branch) and the "
                       "fidelity against the coherent reference is below 1",
        "measured": {"gated_branches": superposition["gated_branches"],
                     "fidelity": superposition[
                         "fidelity_of_gated_mixture_against_reference"]},
        "passed": (superposition["gated_branches"] > 1
                   and superposition["fidelity_of_gated_mixture_against_reference"] < 1.0),
    }, {
        "check": "HMR alone is not MR",
        "requirement": "a random outcome on |0>, i.e. two branches at 1/2 each",
        "measured": readout["hmr_on_zero"]["branches"],
        "passed": bool(readout["hmr_on_zero"]["outcome_is_random"]),
    }, {
        "check": "H then HMR is MR exactly, which is why the blocker is the missing H",
        "requirement": "identical branch structure on basis and superposed inputs",
        "measured": {"basis": readout["h_then_hmr"]["matches_mr_on_basis"],
                     "superposition": readout["h_then_hmr"][
                         "matches_mr_on_superposition"]},
        "passed": (readout["h_then_hmr"]["matches_mr_on_basis"]
                   and readout["h_then_hmr"]["matches_mr_on_superposition"]),
    }]

    if cross.get("available"):
        checks.append({
            "check": "the tier agrees with kickmix on the gated gadget's basis inputs",
            "requirement": "identical output register values on all 8 inputs",
            "measured": cross["rows"],
            "passed": bool(cross["agree"]),
        })

    failed = [entry for entry in checks if not entry["passed"]]
    payload = {
        "experiment": "VQC -- the amplitude tier and the licence boundary",
        "declared_in": "docs/EEA_SCHEDULE_CAMPAIGN.md §2.1 (P1) and "
                       "docs/DQI_OFFICIAL_COMPATIBILITY.md",
        "scope": "the read-then-gate gadget the whole arm's saving rests on, at sizes "
                 "where amplitudes are representable; not a competition engine and not "
                 "a receipt path",
        "basis_states": basis,
        "superposition": superposition,
        "readout_identity": readout,
        "kickmix_cross_check": cross,
        "dimension": {
            "arm": dimension_report(1913),
            "reference": dimension_report(1893),
            "note": "an amplitude-level verification of the arm would need one complex "
                    "number per basis state of a 1,913-qubit register; the counts in "
                    "this repository are therefore basis-state counts, and this tier's "
                    "reach is the gadget, not the decoder",
        },
        "checks": checks,
        "failed_checks": [entry["check"] for entry in failed],
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "module_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "quantum_compilation"
                 / "statevector_sim.py").read_bytes()).hexdigest(),
        },
        "verdict": ("the licence is exactly the basis: identical on every "
                    "computational-basis input, a dephasing outside it"
                    if basis["all_equal"] and not failed else "inconsistent"),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"basis: {len(basis['rows'])} inputs, all equal = {basis['all_equal']}")
    print(f"superposition: {superposition['gated_branches']} branches, fidelity "
          f"{superposition['fidelity_of_gated_mixture_against_reference']:.4f}")
    print(f"HMR random on |0>: {readout['hmr_on_zero']['outcome_is_random']}; "
          f"H+HMR == MR: {readout['h_then_hmr']['matches_mr_on_basis']}")
    if cross.get("available"):
        print(f"kickmix cross-check agrees: {cross['agree']}")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
