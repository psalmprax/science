#!/usr/bin/env python3
"""Run the MR-gated DQI arms on a real ``kickmix``, not on the local model.

``scripts/run_mr_gated_arms.py`` runs the arms on
``apedp_dyn/quantum_compilation/local_mr_sim.py`` because the pinned competition
extension has no Z-basis measurement. This script closes that last gap: it builds
the same decoder with a real ``kickmix.CircuitBuilder`` and simulates it with a
real ``kickmix.Simulator``, against an MR-capable ``kickmix`` compiled from the
vendored sources (``scripts/build_kickmix_mr.py``, patch in
``scripts/kickmix_mr_opcode.patch``).

What this does and does not establish:

* It establishes that the circuits execute on ``kickmix`` with the measured
  instruction, and that the arm's function and tracked phase match the reference
  shot for shot.
* It does **not** produce a score. The pinned build in
  ``apedp_dyn/quantum_compilation/contracts.py`` is the only build that may be
  hashed into a receipt, and this one is not it. T counts here come from the same
  builder, but a competition receipt still needs the upstream rebuild.

The falsifiers from the local run carry over: ``--clear-with r`` must fail once any
gating is applied (``R`` randomizes the tracked phase) and ``--shift-under`` must
fail (it drops shifts the decoder needs). A harness that passes those too is not
measuring anything.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py
    /dev/shm/qvcc-venv313/bin/python scripts/run_mr_gated_arms_kickmix.py --arm all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"


def load_kickmix(build_dir: Path):
    """Import the locally built extension, and refuse the pinned one."""
    sys.path.insert(0, str(build_dir))
    try:
        import kickmix as km
    except ImportError as exc:
        raise SystemExit(
            f"could not import kickmix from {build_dir}: {exc}\n"
            f"build it first: python scripts/build_kickmix_mr.py"
        ) from exc
    if not hasattr(km.CircuitBuilder, "mr"):
        raise SystemExit(
            f"{km.__file__} has no mr method; that is the pinned competition build. "
            f"Build the MR-capable extension with scripts/build_kickmix_mr.py and point "
            f"--kickmix-build at its lib directory."
        )
    return km


def builder_class(km):
    """The real builder, plus the conventions ``run_mr_gated_arms`` expects.

    The arms script was written against the local shim: it calls
    ``init_and``/``del_and`` with one control where the real builder takes two, and
    spells ``reset`` as ``r``. Each shim is function-preserving — a one-control AND
    is a ``CX``.
    """

    class Builder(km.CircuitBuilder):
        @staticmethod
        def _controls(obj):
            return list(obj) if isinstance(obj, km.array) else [obj]

        def init_and(self, *args):
            if len(args) == 3:
                return super().init_and(*args)
            controls, target = args
            ctrls = self._controls(controls)
            if len(ctrls) == 1:
                self.cx(ctrls[0], target)
            elif len(ctrls) == 2:
                super().init_and(ctrls[0], ctrls[1], target)
            else:
                raise NotImplementedError(f"init_and over {len(ctrls)} controls")

        def del_and(self, *args):
            if len(args) == 3:
                return super().del_and(*args)
            controls, target = args
            ctrls = self._controls(controls)
            if len(ctrls) == 1:
                self.cx(ctrls[0], target)
            elif len(ctrls) == 2:
                super().del_and(ctrls[0], ctrls[1], target)
            else:
                raise NotImplementedError(f"del_and over {len(ctrls)} controls")

        def r(self, target, control=True):
            # The arms only reset unconditionally; the real builder has no reset_if.
            self.reset(target)

    return Builder


def shim_for(km, builder):
    """The module surface ``dqi_reference_circuit`` reads off ``km``."""

    class RealKmShim:
        array = km.array
        GF2Field = km.GF2Field
        UnaryIterationCursor = km.UnaryIterationCursor
        CircuitBuilder = builder

    return RealKmShim


def all_qubits(km, registers):
    """Flatten registers of registers, recognising ``kickmix`` arrays."""
    sequences = (list, km.array)
    out = []
    for register in registers:
        items = list(register) if isinstance(register, sequences) else [register]
        for item in items:
            out.extend(list(item) if isinstance(item, sequences) else [item])
    return out


def build(km, arms, shim, arm: str, clear_with: str, n: int, degree: int, m: int,
          shift_under: bool = False):
    """Build the requested arm with the real ``kickmix`` builder."""
    decoder = arms.MRGatedDecoder(
        n, km.GF2Field(degree),
        gate_swap=arm in ("swap", "all", "full"),
        # The quotient gate refines the sweep gate: the divisions sit inside the
        # gated step, so the arm that reads them implies it.
        gate_sweep=arm in ("sweep", "all", "full", "quotient"),
        clear_eea=arm in ("clear", "all", "full"),
        gate_forney=arm in ("forney", "all", "full"),
        gate_shift=arm in ("shift", "all", "full"),
        gate_quotient=arm in ("quotient", "full"),
        shift_under=shift_under,
        clear_with=clear_with,
    )
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=arms.DR.default_n_iters(n))
    return builder.finish_circuit(), syndromes, error_mbuc


def simulate(km, circuit, syndromes, error_mbuc, syndromes_in, mbuc_in, degree, shots):
    sim = km.Simulator(batch_size=shots, seed=99)
    sim.use_same_registers_as(circuit)
    flat_syndromes = km.array([q for register in syndromes for q in register])
    flat_errors = km.array([b for register in error_mbuc for b in register])
    sim.write_across_shots(flat_syndromes, _bits(syndromes_in, degree))
    sim.write_across_shots(flat_errors, _bits(mbuc_in, degree))
    sim.do(circuit)
    syndrome_ids = {q.id for register in syndromes for q in register}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "syndromes": sim.read_across_shots(flat_syndromes),
        "junk": sim.read_across_shots([km.q(i) for i in junk_ids]),
        "phase": sim.read_phase_flipped_across_shots(),
    }


def _bits(values, width):
    from verify_submission_dqi import to_bits
    return to_bits(values, width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm",
                        choices=("none", "swap", "sweep", "clear", "forney", "shift",
                                 "quotient", "all", "full"),
                        default="all")
    parser.add_argument("--clear-with", choices=("mr", "r"), default="mr")
    parser.add_argument("--shift-under", action="store_true",
                        help="falsifier: drop the ¬flag∧(c0∨c3) branch from the gate")
    parser.add_argument("--m", type=int, default=15, help="number of MBUC positions")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD,
                        help="lib directory of the MR-capable extension")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    km = load_kickmix(args.kickmix_build)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    # Imported after kickmix so its module-level import resolves to this build.
    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms", ROOT / "scripts" / "run_mr_gated_arms.py")
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    arms._all_qubits = lambda registers: all_qubits(km, registers)

    shim = shim_for(km, builder_class(km))
    original = arms.DR.km
    arms.DR.km = shim
    try:
        spec_dqi = V.ProblemSpecDQI.benchmark_instance(
            m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
        weights = V.optimal_dqi_weights(
            spec_dqi.m, spec_dqi.max_errors, spec_dqi.degree, spec_dqi.r) ** 2
        tables = V.FieldTables(km.GF2Field(spec_dqi.degree))
        rng = np.random.default_rng(20261028)
        syndromes_in, mbuc_in, _phase = V.sample_shots(
            spec_dqi, tables, rng, args.shots, weights)

        ref_circuit, ref_syn, ref_mbuc = build(km, arms, shim, "none", "mr", args.n,
                                              args.degree, args.m)
        reference = simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in, mbuc_in,
                             args.degree, args.shots)

        if args.arm == "none":
            arm_circuit, arm = ref_circuit, reference
        else:
            arm_circuit, arm_syn, arm_mbuc = build(
                km, arms, shim, args.arm, args.clear_with, args.n, args.degree, args.m,
                shift_under=args.shift_under)
            arm = simulate(km, arm_circuit, arm_syn, arm_mbuc, syndromes_in, mbuc_in,
                           args.degree, args.shots)
    finally:
        arms.DR.km = original

    result = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "arm": args.arm,
        "clear_with": args.clear_with,
        "shift_under": args.shift_under,
        "instance": {"m": args.m, "n": args.n, "degree": args.degree, "shots": args.shots},
        "syndromes_match_reference": bool(np.array_equal(reference["syndromes"], arm["syndromes"])),
        "phase_matches_reference": bool(np.array_equal(reference["phase"], arm["phase"])),
        "syndromes_restored": bool(np.array_equal(
            arm["syndromes"], V.to_bits(syndromes_in, args.degree))),
        "junk_clean": bool(not arm["junk"].any()),
        "reference_junk_clean": bool(not reference["junk"].any()),
        "reference_qubits": ref_circuit.num_qubits,
        "arm_qubits": arm_circuit.num_qubits,
        "reference_phase_ones": int(reference["phase"].sum()),
        "arm_phase_ones": int(arm["phase"].sum()),
    }
    good = (result["syndromes_match_reference"] and result["phase_matches_reference"]
            and result["syndromes_restored"] and result["junk_clean"])
    result["verdict"] = "PASS" if good else "FAIL"
    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if good else 1

    print(f"arm={args.arm} read with {args.clear_with.upper()}  "
          f"m={args.m} n={args.n} GF(2^{args.degree}) shots={args.shots}")
    print(f"  kickmix  {result['kickmix']}")
    print(f"  qubits   reference={ref_circuit.num_qubits} arm={arm_circuit.num_qubits}")
    print(f"  syndromes match reference : {result['syndromes_match_reference']}")
    print(f"  phase matches reference   : {result['phase_matches_reference']}"
          f"  (ones: ref={result['reference_phase_ones']} arm={result['arm_phase_ones']})")
    print(f"  syndromes restored        : {result['syndromes_restored']}")
    print(f"  all other qubits |0>      : {result['junk_clean']}")
    print(f"VERDICT: {result['verdict']}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
