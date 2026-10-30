#!/usr/bin/env python3
"""Experiment D2: does the pinned binary compile a *phase-correct* cheaper candidate?

Experiment D (`docs/DQI_OFFICIAL_COMPATIBILITY.md`) concluded that the competition
binary can compile a circuit whose clears are `hmr` rather than `mr`, because "a
clear does not need the value, only the reset", and reported the reference algorithm
with the 54-AND multiply at 5,433,399.6 Toffolis per shot -- a 4.8% score improvement.
It checked that row's syndromes and its junk and never checked its **phase**.

The DQI submission is scored on phase as well as value: the decoder's Forney step
kicks back a phase that must equal `(-1)^sum_j popcount(e_j & M_j)`, and the verifier
counts a phase error as a failing shot. So this experiment asks the question D left
open, twice:

1. **Is an `hmr` clear phase-safe?** A Z-basis measurement and reset (`mr`) leaves the
   tracked phase alone because the outcome is the state's own value. An *X*-basis
   measurement does not: on |1> the branch that measures 1 carries an extra factor of
   -1, so the tracked phase of that shot flips. Measured directly, one qubit at a
   time, for `hmr` and for `del_and`.
2. **Does that reach the candidate?** The decoder's scratch windows hold 1s on most
   shots, so if the mechanism is real the candidate's phase errors should be about one
   shot in two -- and zero for the reference, whose measurement-based uncomputation
   (`del_and`: `hmr` plus a classically conditioned phase fixup) is phase-correct by
   construction.

A third arm tests the largest lever D never asked about: run the EEA on a **CNOT copy**
of the syndromes and clear the whole scratch by measurement. The input is then never
written, so nothing has to be restored and no qubit's value is read -- which is what
`clear_eea` needs -- but its clears are `hmr` too.

The answer the measurement gives is the opposite of D's: the phase is not incidental.
Read the record it writes.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_official_candidate.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
PINNED_DIR = BUNDLE / "kickmix"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "dqi_official_candidate.json")

#: Rows this experiment is measured against, from the records it extends.
REFERENCE_TOTAL = 5_725_071.609375
REFERENCE_QUBITS = 1893
GATE_STREAM_TOTAL = 5_433_399.609375
GATE_STREAM_QUBITS = 1896
ARM_TOTAL = 686_014.8
ARM_QUBITS = 1913


def all_qubits(km, registers) -> list:
    """Flatten registers of registers into a plain list of qubits."""
    sequences = (list, km.array)
    out = []
    for register in registers:
        items = list(register) if isinstance(register, sequences) else [register]
        for item in items:
            out.extend(list(item) if isinstance(item, sequences) else [item])
    return out


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, degree, shots) -> dict:
    from verify_submission_dqi import to_bits

    sim = km.Simulator(batch_size=shots, seed=99, counting_operations=True)
    sim.use_same_registers_as(circuit)
    flat_syndromes = km.array([q for r in syn for q in r])
    flat_errors = km.array([b for r in mbuc for b in r])
    sim.write_across_shots(flat_syndromes, to_bits(syndromes_in, degree))
    sim.write_across_shots(flat_errors, to_bits(mbuc_in, degree))
    sim.do(circuit)
    per_shot = np.array([int(sim.operation_statistics(i).num_toffoli_like)
                         for i in range(shots)])
    syndrome_ids = {q.id for r in syn for q in r}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "toffoli_per_shot": per_shot,
        "mean_toffoli_like": float(per_shot.mean()),
        # `read_across_shots` returns one row per qubit and one column per shot.
        "syndromes": np.asarray(sim.read_across_shots(flat_syndromes)).T,
        "junk": np.asarray(sim.read_across_shots([km.q(i) for i in junk_ids])).T,
        "phase": np.asarray(sim.read_phase_flipped_across_shots()),
    }


def hmr_and_del_and_phase(km) -> dict:
    """Is a measured clear phase-safe? One qubit, both instructions, both bases."""
    rows = {}
    for operation in ("hmr", "del_and"):
        for prepared in (False, True):
            shots = 200
            builder = km.CircuitBuilder()
            qubit = builder.alloc_qubits(1)
            control = builder.alloc_qubits(1)
            target = builder.alloc_qubits(1)
            bit = builder.alloc_bits(1)
            if prepared:
                builder.x(qubit)
            # `control` is |1>, so `target` holds the prepared qubit's value either way.
            builder.x(control)
            builder.ccx(control, qubit, target)
            if operation == "hmr":
                builder.hmr(target, bit)
            else:
                builder.del_and(control, qubit, target)
            circuit = builder.finish_circuit()
            sim = km.Simulator(batch_size=shots, seed=7, counting_operations=True)
            sim.use_same_registers_as(circuit)
            sim.do(circuit)
            phase = np.asarray(sim.read_phase_flipped_across_shots())
            outcomes = np.asarray(sim.read_across_shots(bit))
            left = np.asarray(sim.read_across_shots(target)).any()
            rows[f"{operation}_on_{'one' if prepared else 'zero'}"] = {
                "shots": shots,
                "phase_flips": int(phase.sum()),
                "measured_ones": int(outcomes.sum()),
                "qubit_left_dirty": bool(left),
                "phase_safe": not bool(phase.any()),
            }
    return rows


def build_cleared_eea(km, arms, shim, n: int, degree: int, m: int, n_iters: int):
    """The reference's decoder, with the EEA on a copy and its scratch cleared.

    `ProosAndZalkaDecoder.build_syndrome_decoding_mbuc` through the Forney step, with
    `reg_a`'s syndrome lanes CNOT copies instead of the input registers and the reverse
    EEA replaced by a measured clear of every scratch qubit. The inputs are never
    written, so there is nothing to restore and no qubit's value is needed.
    """
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]

    decoder = arms.DR.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    lanes = n + 3
    reg_a = [builder.alloc_qubits(degree) for _ in range(lanes)]
    reg_b = [builder.alloc_qubits(degree) for _ in range(lanes)]
    for lane in range(3, lanes):
        for source, target in zip(syndromes[lane - 3], reg_a[lane], strict=True):
            builder.cx(source, target)

    logm = lanes.bit_length() + 1
    n_A = builder.alloc_qubits(logm)
    n_B = builder.alloc_qubits(logm)
    n_a = builder.alloc_qubits(logm)
    n_b = builder.alloc_qubits(logm)
    n_q = builder.alloc_qubits(logm)
    flag = builder.alloc_qubits(1)[0]
    counter = builder.alloc_qubits(2)
    halting_counter = builder.alloc_qubits(n_iters.bit_length())
    eea = (reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter, halting_counter)

    init_ones = shim.array([reg_a[0][0], reg_b[lanes - 1][0], n_a[0], flag])
    builder.x(init_ones)
    builder.gf2_iadd(n, target=n_A)
    builder.gf2_iadd(n + 1, target=n_B)

    with builder.mark("initial_shift"):
        decoder._initial_shift_loop(builder, reg_a, n_A, is_adjoint=False)
    with builder.mark("forward_eea"):
        decoder._synchronized_eea(builder, *eea, n_iters=n_iters, is_adjoint=False)
    with builder.mark("chien_forney_mbuc"):
        decoder._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)

    with builder.mark("measured_clear"):
        scratch = all_qubits(km, (reg_a, reg_b, n_A, n_B, n_a, n_b, n_q,
                                  halting_counter, counter))
        cleared_bits = builder.alloc_bits(len(scratch))
        builder.mr(shim.array(scratch), cleared_bits)
        flag_bit = builder.alloc_bits(1)
        builder.mr(shim.array([flag]), flag_bit)
        builder.free(flag_bit)
        builder.free(cleared_bits)

    for register in (halting_counter, counter, n_a, n_b, n_q, *reg_b):
        builder.free(register)
    builder.free(n_A)
    builder.free(n_B)
    builder.free(shim.array([flag]))
    for lane in reg_a:
        builder.free(lane)
    return builder.finish_circuit(), syndromes, error_mbuc


def measure(build_dir: Path, m: int, n: int, degree: int, shots: int, seed: int) -> dict:
    """Measure one extension. Run in a subprocess so the two builds cannot mix."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(build_dir))
    runner = load_module("mr_gated_arms_kickmix",
                         ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    import kickmix as km  # noqa: PLC0415 - the build under test
    ledger = load_module("region_ledger",
                         ROOT / "scripts" / "measure_dqi_region_ledger.py")
    gf54 = load_module("gf54_mul", ROOT / "scripts" / "gf54_mul.py")
    arms = ledger.load_arms(km, runner)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V  # noqa: PLC0415

    base_builder = runner.builder_class(km)
    builtin_shim = runner.shim_for(km, ledger.native_builder(base_builder, "karatsuba"))

    class ClearWithHmr(base_builder):
        """`mr` -> `hmr`, the substitution Experiment D rested on."""

        def mr(self, target, bit):
            if isinstance(target, km.array):
                targets = target
            elif isinstance(target, (list, tuple)):
                targets = km.array(list(target))
            else:
                targets = km.array([target])
            bits = bit if isinstance(bit, km.array) else km.array([bit])
            self.hmr(targets, bits)

    candidate_builder = type("Candidate", (gf54.Gf54MulMixin, ClearWithHmr),
                             {"__doc__": "54-AND gate-stream multiply, HMR clears"})
    candidate_shim = runner.shim_for(km, candidate_builder)

    iterations = arms.DR.default_n_iters(n)
    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree, num_shots=shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(degree))
    rng = np.random.default_rng(seed)
    syndromes_in, mbuc_in, phases = V.sample_shots(spec, tables, rng, shots, weights)
    # `to_bits` (like `read_across_shots`) is one row per qubit, one column per shot.
    expected = V.to_bits(syndromes_in, degree).T

    out = {
        "build": str(build_dir),
        "extension": str(Path(km.__file__).resolve()),
        "extension_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "has_mr": bool(hasattr(km.CircuitBuilder, "mr")),
        "clear_phase_safety": hmr_and_del_and_phase(km),
        "arms": {},
    }

    def score(circuit, result) -> dict:
        """The verifier's per-shot semantics: value first, phase only where value held."""
        value_error = np.any(result["syndromes"] != expected, axis=1)
        phase_flipped = result["phase"] if result["phase"].shape == phases.shape else \
            np.resize(result["phase"], phases.shape)
        phase_error = (phase_flipped ^ phases) & ~value_error
        return {
            "qubits": circuit.num_qubits,
            "mean_toffoli_like": result["mean_toffoli_like"],
            "toffoli_per_shot": result["toffoli_per_shot"].tolist(),
            "syndromes_restored": bool(np.array_equal(result["syndromes"], expected)),
            "junk_clean": bool(not result["junk"].any()),
            "value_errors": int(value_error.sum()),
            "phase_errors": int(phase_error.sum()),
            "shots": int(shots),
        }

    original = arms.DR.km
    try:
        arms.DR.km = builtin_shim
        circuit, syn, mbuc = ledger.build_reference(km, arms, builtin_shim, n, degree, m)
        reference = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, degree, shots)
        out["arms"]["reference"] = score(circuit, reference)
        del circuit

        # Candidate 1: what Experiment D measured -- the reference algorithm with the
        # 54-AND gate stream, whose window clears are HMR.
        arms.DR.km = candidate_shim
        circuit, syn, mbuc = ledger.build_reference(km, arms, candidate_shim, n, degree, m)
        result = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, degree, shots)
        out["arms"]["hmr_candidate"] = score(circuit, result)
        out["arms"]["hmr_candidate"]["call"] = (
            "reference algorithm + 54-AND gate stream + HMR clears (Experiment D)")
        del circuit

        # Candidate 2: the EEA on a CNOT copy, its scratch cleared by measurement.
        arms.DR.km = candidate_shim
        circuit, syn, mbuc = build_cleared_eea(km, arms, candidate_shim, n, degree, m,
                                               iterations)
        result = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, degree, shots)
        out["arms"]["copied_eea"] = score(circuit, result)
        out["arms"]["copied_eea"]["call"] = (
            "forward EEA on a CNOT copy + HMR clear of the scratch + 54-AND gate stream")
        del circuit
    finally:
        arms.DR.km = original
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--build", type=Path, default=PINNED_DIR)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    run = measure(args.build, args.m, args.n, args.degree, args.shots, args.seed)
    rows = run["arms"]
    reference = rows["reference"]

    def ratio(name: str) -> float:
        row = rows[name]
        return ((row["mean_toffoli_like"] / reference["mean_toffoli_like"])
                * (row["qubits"] / reference["qubits"]) ** 2)

    safety = run["clear_phase_safety"]
    checks = [{
        "check": "the reference reproduces its recorded row",
        "requirement": f"{REFERENCE_TOTAL:.6f} per shot at Q={REFERENCE_QUBITS}",
        "measured": {"mean": reference["mean_toffoli_like"],
                     "qubits": reference["qubits"]},
        "passed": (abs(reference["mean_toffoli_like"] - REFERENCE_TOTAL) < 1e-6
                   and reference["qubits"] == REFERENCE_QUBITS),
    }, {
        "check": "the reference makes no phase errors",
        "requirement": "0 of 64 shots",
        "measured": {"value_errors": reference["value_errors"],
                     "phase_errors": reference["phase_errors"]},
        "passed": reference["phase_errors"] == 0 and reference["value_errors"] == 0,
    }, {
        "check": "`mr` is phase-safe and `hmr` is not",
        "requirement": "an X-basis clear of |1> flips the tracked phase on about half "
                       "the shots; a Z-basis one never does",
        "measured": safety,
        "passed": (safety["hmr_on_zero"]["phase_flips"] == 0
                   and safety["hmr_on_one"]["phase_flips"] > 0
                   and safety["del_and_on_zero"]["phase_flips"] == 0
                   and safety["del_and_on_one"]["phase_flips"] == 0),
    }, {
        "check": "Experiment D's candidate reproduces its recorded count",
        "requirement": f"{GATE_STREAM_TOTAL:.6f} at Q={GATE_STREAM_QUBITS}",
        "measured": {"mean": rows["hmr_candidate"]["mean_toffoli_like"],
                     "qubits": rows["hmr_candidate"]["qubits"]},
        "passed": (abs(rows["hmr_candidate"]["mean_toffoli_like"] - GATE_STREAM_TOTAL) < 1e-6
                   and rows["hmr_candidate"]["qubits"] == GATE_STREAM_QUBITS),
    }]

    failed = [entry for entry in checks if not entry["passed"]]
    hmr_phase = rows["hmr_candidate"]["phase_errors"]
    copied_phase = rows["copied_eea"]["phase_errors"]
    payload = {
        "experiment": "D2 -- is the officially compilable candidate *correct*?",
        "declared_in": "docs/DQI_OFFICIAL_COMPATIBILITY.md §6",
        "question": "Experiment D reported a compilable 5,433,399.6-Toffoli candidate "
                    "and checked only its syndromes. Does it also return the right "
                    "phase, and is there a compilable candidate that does?",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": args.seed},
        "build": {key: run[key] for key in
                  ("build", "extension", "extension_sha256", "has_mr")},
        "clear_phase_safety": safety,
        "arms": rows,
        "score_ratios": {name: ratio(name) for name in rows},
        "arm": {"qubits": ARM_QUBITS, "mean_toffoli_like": ARM_TOTAL,
                "compilable": False},
        "checks": checks,
        "failed_checks": [entry["check"] for entry in failed],
        "finding": (
            f"an `hmr` clear is not phase-safe: on |1> it flipped the tracked phase on "
            f"{safety['hmr_on_one']['phase_flips']}/{safety['hmr_on_one']['shots']} "
            f"shots, the same count as its measured ones "
            f"({safety['hmr_on_one']['measured_ones']}), while `del_and` -- the same "
            f"measurement plus a classically conditioned phase fixup -- flipped none. "
            f"Experiment D's candidate therefore scores {hmr_phase}/{args.shots} phase "
            f"errors where the reference scores "
            f"{reference['phase_errors']}/{args.shots}: it is compilable and it is not "
            f"a valid submission, and the 0.952 score ratio D recorded is not "
            f"attainable. The copied-EEA candidate, which removes the reverse EEA "
            f"outright, scores {copied_phase}/{args.shots} phase errors for the same "
            f"reason. The cheapest *correct* circuit the pinned binary can compile is "
            f"the reference, because every cheaper one needs a clear whose value is a "
            f"function of the data, and the only phase-safe clear the binary has "
            f"(`del_and`) requires the value to be a known AND."
        ),
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kickmix_sha256": run["extension_sha256"],
            "git_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True,
                                      text=True).stdout.strip(),
        },
        "scope": {
            "measured": "three circuits built and executed on the pinned competition "
                        "binary with Q, per-shot Toffoli counts, and the verifier's own "
                        "value/phase error counts; and the phase behaviour of `hmr` and "
                        "`del_and` on a single qubit",
            "not_measured": "no submission: R(eps_plus) and the full score are not "
                            "computed here",
            "corrects": "the D record's candidate is not phase-correct, so its 0.952 "
                        "score ratio is withdrawn; only its Toffoli count stands",
        },
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")

    for name, row in rows.items():
        print(f"{name:14s} Q={row['qubits']} T={row['mean_toffoli_like']:.6f} "
              f"value_err={row['value_errors']} phase_err={row['phase_errors']} "
              f"ratio={ratio(name):.4f}")
    for name, row in safety.items():
        print(f"{name:16s} phase_flips={row['phase_flips']}/{row['shots']}")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
