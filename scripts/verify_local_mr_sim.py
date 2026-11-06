#!/usr/bin/env python3
"""Compare the local MR-capable simulator against ``kickmix`` shot for shot.

The local engine in ``apedp_dyn/quantum_compilation/local_mr_sim.py`` exists so the
gated DQI arms can be run without waiting for ``MR`` upstream. That is only worth
something if it reproduces ``kickmix`` on the circuit ``kickmix`` *can* run, so this
script builds the reference decoder twice — once with each builder — feeds both the
same syndromes and MBUC outcomes, and requires identical results:

* the syndrome registers come back to their input values,
* every other qubit is ``|0>``,
* the tracked phase matches bit for bit.

Run it under the interpreter ``kickmix`` was built for (CPython 3.13):

    python3.13 scripts/verify_local_mr_sim.py --m 15 --n 6 --degree 4 --shots 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
sys.path.insert(0, str(BUNDLE / "kickmix"))
sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
sys.path.insert(0, str(ROOT))

# Loaded by path: importing the package would pull in the whole apedp_dyn tree,
# which needs packages the kickmix interpreter does not have.
import importlib.util  # noqa: E402

import dqi_reference_circuit as DR  # noqa: E402
import kickmix as km  # noqa: E402
import verify_submission_dqi as V  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "local_mr_sim", ROOT / "apedp_dyn" / "quantum_compilation" / "local_mr_sim.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


def to_bits(values: np.ndarray, width: int) -> np.ndarray:
    return V.to_bits(values, width)


class _LocalKmShim:
    """What `dqi_reference_circuit` needs from ``kickmix`` when built locally.

    The module calls ``km.array(...)`` on its own registers, and ``kickmix.array``
    only accepts ``kickmix.q``; pointing the module at the local ``array`` is what
    lets the same code build against either engine.
    """

    array = L.array
    GF2Field = km.GF2Field
    UnaryIterationCursor = L.Cursor
    CircuitBuilder = L.Builder


def build_local(m: int, n: int, degree: int):
    original_km = DR.km
    DR.km = _LocalKmShim
    try:
        return _build_local(m, n, degree)
    finally:
        DR.km = original_km


def _build_local(m: int, n: int, degree: int):
    builder = L.Builder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    DR.ProosAndZalkaDecoder(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        builder,
        syndromes=syndromes,
        error_mbuc=error_mbuc,
        n_iters=DR.default_n_iters(n),
    )
    return builder.finish_circuit(), syndromes, error_mbuc


def run_kickmix(m, n, degree, shots, syndromes, mbuc):
    builder = km.CircuitBuilder()
    syn = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    err = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    DR.ProosAndZalkaDecoder(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        builder, syndromes=syn, error_mbuc=err, n_iters=DR.default_n_iters(n))
    circuit = builder.finish_circuit()
    sim = km.Simulator(batch_size=shots, debug_print=False, seed=1234)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array([q for reg in syn for q in reg]),
                           to_bits(syndromes, degree))
    sim.write_across_shots(km.array([b for reg in err for b in reg]),
                           to_bits(mbuc, degree))
    sim.do(circuit)
    syndrome_values = sim.read_across_shots(km.array([q for reg in syn for q in reg]))
    junk = [q for q in range(circuit.num_qubits) if q not in {x.id for reg in syn for x in reg}]
    return {
        "num_qubits": circuit.num_qubits,
        "syndromes": syndrome_values,
        "junk": sim.read_across_shots(km.array([km.q(i) for i in junk])),
        "phase": sim.read_phase_flipped_across_shots(),
    }


def run_local(m, n, degree, shots, syndromes, mbuc):
    circuit, syn, err = build_local(m, n, degree)
    sim = L.Simulator(batch_size=shots, seed=1234)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(syn, to_bits(syndromes, degree))
    sim.write_across_shots(err, to_bits(mbuc, degree))
    sim.do(circuit)
    syndrome_ids = {qubit.id for reg in syn for qubit in reg}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "num_qubits": circuit.num_qubits,
        "syndromes": sim.read_across_shots(syn),
        "junk": sim.read_across_shots([L.Q(i) for i in junk_ids]),
        "phase": sim.read_phase_flipped_across_shots(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--shots", type=int, default=32)
    args = parser.parse_args()

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes, mbuc, expected_phase = V.sample_shots(spec, tables, rng, args.shots, weights)

    reference = run_kickmix(args.m, args.n, args.degree, args.shots, syndromes, mbuc)
    local = run_local(args.m, args.n, args.degree, args.shots, syndromes, mbuc)
    input_bits = to_bits(syndromes, args.degree)

    print(f"instance m={args.m} n={args.n} GF(2^{args.degree}) shots={args.shots}")
    print(f"  qubits        kickmix={reference['num_qubits']}  local={local['num_qubits']}")
    # Qubit ids are not reused locally (free is bookkeeping), so the id count is
    # larger by construction; values are compared by role, never by id.
    problems = []
    for name in ("syndromes", "phase"):
        same = np.array_equal(reference[name], local[name])
        print(f"  {name:<12} identical={same}")
        if not same:
            problems.append(name)
    both_clean = (not reference["junk"].any()) and (not local["junk"].any())
    print(f"  junk        clean in both={both_clean}")
    if not both_clean:
        problems.append("junk")

    restored = np.array_equal(local["syndromes"], input_bits)
    junk_clean = not local["junk"].any()
    contract = np.array_equal(local["phase"], expected_phase.astype(bool))
    print(f"  syndromes restored      : {restored}")
    print(f"  all other qubits |0>    : {junk_clean}")
    print(f"  phase matches contract  : {contract}")
    print(f"  phase errors            : {int((local['phase'] != expected_phase.astype(bool)).sum())}"
          f"/{args.shots}")

    if problems or not (restored and junk_clean and contract):
        print("VERDICT: MISMATCH", problems)
        return 1
    print("VERDICT: local engine reproduces kickmix on the reference decoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
