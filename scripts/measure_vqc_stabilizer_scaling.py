"""Stabilizer scaling, 100 to 1,000 qubits, in three circuit families.

Status: experimental

The 100-qubit GHZ record (`measure_vqc_stabilizer_ghz.py`) shows the tier reaches
past the dense bound. This one asks how the reach scales, and refuses to answer with
a single family:

* **GHZ chain** -- `h(0)` then `cx(q-1, q)`. Generators have a closed form, so the
  prepared state is checked against the textbook set rather than against itself.
* **GHZ star** -- `h(0)` then `cx(0, q)`: the *same prepared state* by a different
  circuit. The two are compared with the state-scope referee, which is an exact
  stabilizer-group comparison, so the row checks the path as well as the size.
* **Ring graph state** -- `h` on all, `cz(q, q+1 mod n)` on every edge. Its
  generators are `X_q Z_(q-1) Z_(q+1)` by definition, so each one is required to
  stabilise the state it is supposed to stabilise.
* **Random Clifford layers** -- `h`, `s` and `cx` chosen by a seeded RNG: no closed
  form, so the checks are structural (`validate()` and every generator being an
  exact eigenstate) plus **equivalence at scale**: the same circuit rewritten with
  cancelling gate pairs must be VALID, and a single mutated gate must be INVALID.
  That last row is the useful one: it decides two 1,000-qubit circuits without ever
  building a matrix.

What is measured per row: wall clock to build the tableau, to validate it, and to
sample a shot; the tableau's label count (`n**2` bytes, the lower bound the
representation pays) against the dense tier's `2**n` amplitudes; and peak RSS.

What is **not** claimed: this is not a general 1,000-qubit simulator (Clifford only,
pure states only, no noise), the dense comparison stops where the dense tier stops,
and a scaling row is a measurement on one host, not a complexity proof. The record
states those limitations itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_stabilizer_scaling.json"
SIZES = (100, 200, 500, 1000)

sys.path.insert(0, str(ROOT))

from apedp_dyn.vqc import Circuit, Gate  # noqa: E402
from apedp_dyn.vqc.equivalence import (  # noqa: E402
    adjudicate_clifford_equivalence,
)
from apedp_dyn.vqc.stabilizer import (  # noqa: E402
    StabilizerState,
    ghz_generators,
    is_clifford_circuit,
)


# ------------------------------------------------------------------ the families
def ghz_chain(num_qubits: int) -> Circuit:
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits,
                   metadata={"generator": "vqc.scaling.ghz_chain"})


def ghz_star(num_qubits: int) -> Circuit:
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (0, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits,
                   metadata={"generator": "vqc.scaling.ghz_star"})


def ring_graph_state(num_qubits: int) -> Circuit:
    ops = [Gate("h", (qubit,)) for qubit in range(num_qubits)]
    ops.extend(Gate("cz", (qubit, (qubit + 1) % num_qubits)) for qubit in range(num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits,
                   metadata={"generator": "vqc.scaling.ring_graph_state"})


def random_clifford_layers(num_qubits: int, *, seed: int, layers: int = 3) -> Circuit:
    """Layered random Clifford: `h` everywhere, `s` on alternate qubits, `cx` ring.

    The layout is seeded and layer-rotated rather than drawn gate by gate, because a
    gate-by-gate draw at 1,000 qubits would cost more in RNG calls than in tableau
    updates; the point of the family is an unstructured Clifford circuit, and the
    three layer types in a rotated order are that.
    """
    rng = random.Random(seed)
    ops: list = []
    for layer in range(layers):
        ops.extend(Gate("h", (qubit,)) for qubit in range(num_qubits))
        offset = rng.randrange(2)
        ops.extend(Gate("s", (qubit,)) for qubit in range(offset, num_qubits, 2))
        # A ring of CXs, each layer rotated by one so the layers are not repeated.
        ops.extend(Gate("cx", (qubit, (qubit + 1 + layer) % num_qubits))
                   for qubit in range(num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits,
                   metadata={"generator": "vqc.scaling.random_clifford_layers",
                             "layers": str(layers), "seed": str(seed)})


def rewritten(circuit: Circuit) -> Circuit:
    """The same unitary with cancelling pairs inserted: a genuinely different circuit."""
    ops: list = []
    for index, op in enumerate(circuit.ops):
        ops.append(op)
        if index % 37 == 0:
            qubit = index % circuit.num_qubits
            ops.extend([Gate("h", (qubit,)), Gate("h", (qubit,))])
        if index % 53 == 0:
            first, second = index % circuit.num_qubits, (index + 1) % circuit.num_qubits
            if first != second:
                ops.extend([Gate("cx", (first, second)), Gate("cx", (first, second))])
    return Circuit(ops=tuple(ops), num_qubits=circuit.num_qubits,
                   metadata={"generator": "vqc.scaling.rewritten"})


def mutated(circuit: Circuit) -> Circuit:
    """One `s` replaced by `sdg`: the same shape, a different unitary."""
    ops = list(circuit.ops)
    for index, op in enumerate(ops):
        if op.name == "s":
            ops[index] = Gate("sdg", op.targets)
            break
    else:  # pragma: no cover - the families above always contain an `s`
        raise AssertionError("no `s` gate to mutate")
    return Circuit(ops=tuple(ops), num_qubits=circuit.num_qubits,
                   metadata={"generator": "vqc.scaling.mutated"})


# ------------------------------------------------------------------ measurements
def build_state(circuit: Circuit) -> tuple[StabilizerState, float]:
    start = time.perf_counter()
    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        state.apply_gate(op.name, op.targets, op.params)
    return state, time.perf_counter() - start


def timed_sample(state: StabilizerState, *, shots: int, seed: int) -> tuple[list, float]:
    """Independent shots: each one measures a fresh copy of the prepared state.

    Sampling destroys the state (that is what a measurement is), so reusing one state
    for every shot would report the same collapsed outcome `shots` times.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    samples = [state.clone().sample(rng=rng) for _ in range(shots)]
    return samples, (time.perf_counter() - start) / shots


def shots_for(num_qubits: int, requested: int) -> int:
    """Fewer shots as `n` grows: one Z-basis sample at 1,000 qubits costs seconds.

    The cost is real and is recorded per row: a sample measures every qubit, and each
    measurement runs the GF(2) elimination. Spending four minutes of a record on
    repeated samples would buy a narrower error bar on a number that is not the point
    of the row.
    """
    if num_qubits <= 200:
        return requested
    return 1


def peak_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def ghz_row(num_qubits: int, *, family: str, shots: int, seed: int) -> dict:
    circuit = ghz_chain(num_qubits) if family == "ghz_chain" else ghz_star(num_qubits)
    state, build = build_state(circuit)
    validate_start = time.perf_counter()
    state.validate()
    validate = time.perf_counter() - validate_start
    expected = ghz_generators(num_qubits)
    if family == "ghz_chain":
        # The chain's own generator list must be the textbook list, entry for entry.
        generators_match = tuple(state.generators) == tuple(expected)
    else:
        # The star prepares the same state by a different route, so its generator
        # basis differs; the state is pinned by membership of the same group.
        generators_match = all(state.stabilizes(labels) for labels in expected)
    samples, per_shot = timed_sample(state.clone(), shots=shots_for(num_qubits, shots),
                                     seed=seed)
    shots = len(samples)
    # Sampling collapses the state, so the structural checks run first and the shots
    # come from copies -- otherwise every later sample would return the same
    # collapsed outcome and the generator check would read the measured state.
    only_ghz = all(sum(bits) in (0, num_qubits) for bits in samples)
    even = all(sum(bits) % 2 == 0 for bits in samples)
    return {
        "family": family,
        "qubits": num_qubits,
        "gates": len(circuit.ops),
        "generators": len(state.generators),
        "tableau_label_bytes": num_qubits * num_qubits,
        "dense_amplitudes_log10": round(num_qubits * math.log10(2.0), 3),
        "build_seconds": build,
        "validate_seconds": validate,
        "sample_seconds_per_shot": per_shot,
        "shots": shots,
        "checks": {
            "generators match the closed form": generators_match,
            "every sample is 0...0 or 1...1": only_ghz,
            "every sample has even weight": even,
            "tableau validates": True,
        },
        "peak_rss_kib": peak_rss_kib(),
    }


def graph_row(num_qubits: int, *, shots: int, seed: int) -> dict:
    circuit = ring_graph_state(num_qubits)
    state, build = build_state(circuit)
    validate_start = time.perf_counter()
    state.validate()
    validate = time.perf_counter() - validate_start
    stabilised = 0
    for qubit in range(num_qubits):
        labels = [0] * num_qubits
        labels[qubit] = 1                                  # X_q
        labels[(qubit - 1) % num_qubits] = 3               # Z_(q-1)
        labels[(qubit + 1) % num_qubits] = 3               # Z_(q+1)
        if state.stabilizes(tuple(labels)):
            stabilised += 1
    samples, per_shot = timed_sample(state.clone(), shots=shots_for(num_qubits, shots),
                                     seed=seed)
    shots = len(samples)
    return {
        "family": "ring_graph_state",
        "qubits": num_qubits,
        "gates": len(circuit.ops),
        "generators": len(state.generators),
        "tableau_label_bytes": num_qubits * num_qubits,
        "dense_amplitudes_log10": round(num_qubits * math.log10(2.0), 3),
        "build_seconds": build,
        "validate_seconds": validate,
        "sample_seconds_per_shot": per_shot,
        "shots": shots,
        "checks": {
            "every X_q Z_(q-1) Z_(q+1) generator stabilises the state":
                stabilised == num_qubits,
            "tableau validates": True,
            "samples are distinct states of the support": len(set(samples)) > 1 or shots == 1,
        },
        "stabilised_generators": stabilised,
        "peak_rss_kib": peak_rss_kib(),
    }


def random_row(num_qubits: int, *, shots: int, seed: int, layers: int = 3) -> dict:
    circuit = random_clifford_layers(num_qubits, seed=seed, layers=layers)
    assert is_clifford_circuit(circuit)
    state, build = build_state(circuit)
    validate_start = time.perf_counter()
    state.validate()
    validate = time.perf_counter() - validate_start
    eigenstates = all(abs(state.expectation(labels)) == 1.0
                      for labels in state.generators)
    samples, per_shot = timed_sample(state.clone(), shots=shots_for(num_qubits, shots),
                                     seed=seed)
    shots = len(samples)
    return {
        "family": "random_clifford_layers",
        "qubits": num_qubits,
        "gates": len(circuit.ops),
        "generators": len(state.generators),
        "tableau_label_bytes": num_qubits * num_qubits,
        "dense_amplitudes_log10": round(num_qubits * math.log10(2.0), 3),
        "build_seconds": build,
        "validate_seconds": validate,
        "sample_seconds_per_shot": per_shot,
        "shots": shots,
        "checks": {
            "tableau validates": True,
            "every generator is an exact eigenstate": eigenstates,
            "samples were produced": all(len(bits) == num_qubits for bits in samples),
        },
        "peak_rss_kib": peak_rss_kib(),
    }


def equivalence_at_scale(num_qubits: int, *, seed: int) -> dict:
    """Two different circuits, one unitary: decided with 2n Pauli images, no matrix."""
    base = random_clifford_layers(num_qubits, seed=seed)
    same = rewritten(base)
    changed = mutated(base)

    start = time.perf_counter()
    agreed = adjudicate_clifford_equivalence(base, same)
    agree_seconds = time.perf_counter() - start
    start = time.perf_counter()
    differed = adjudicate_clifford_equivalence(base, changed)
    differ_seconds = time.perf_counter() - start

    return {
        "qubits": num_qubits,
        "compared_images": agreed.compared,
        "rewritten": {
            "gates": len(same.ops),
            "verdict": agreed.verdict,
            "mismatched": agreed.mismatched,
            "seconds": agree_seconds,
        },
        "mutated": {
            "gates": len(changed.ops),
            "verdict": differed.verdict,
            "mismatched": differed.mismatched,
            "first_mismatch": (None if not differed.mismatches
                               else dict(differed.mismatches[0])),
            "seconds": differ_seconds,
        },
    }


def state_scope_at_scale(num_qubits: int) -> dict:
    """Chain versus star: the same prepared state by two different circuits."""
    chain, star = ghz_chain(num_qubits), ghz_star(num_qubits)
    start = time.perf_counter()
    report = adjudicate_clifford_equivalence(chain, star, on="state")
    seconds = time.perf_counter() - start
    unitary = adjudicate_clifford_equivalence(chain, star)
    return {
        "qubits": num_qubits,
        "state_scope": {"verdict": report.verdict, "mismatched": report.mismatched},
        "unitary_scope": {"verdict": unitary.verdict, "mismatched": unitary.mismatched},
        "seconds": seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    rows: list[dict] = []
    for num_qubits in args.sizes:
        rows.append(ghz_row(num_qubits, family="ghz_chain", shots=args.shots,
                            seed=args.seed))
        rows.append(ghz_row(num_qubits, family="ghz_star", shots=args.shots,
                            seed=args.seed))
        rows.append(graph_row(num_qubits, shots=args.shots, seed=args.seed))
        rows.append(random_row(num_qubits, shots=args.shots, seed=args.seed))

    # The equivalence row is the expensive one (2n images through every gate), so it
    # is measured at the smallest and the largest size rather than at every size.
    equivalence_sizes = [args.sizes[0]] if len(args.sizes) == 1 else [args.sizes[0],
                                                                    args.sizes[-1]]
    equivalence = [equivalence_at_scale(num_qubits, seed=args.seed)
                   for num_qubits in equivalence_sizes]
    scopes = [state_scope_at_scale(num_qubits) for num_qubits in equivalence_sizes]

    failing = [f"{row['family']} at {row['qubits']} qubits"
               for row in rows if not all(row["checks"].values())]
    failing += [f"equivalence at {item['qubits']} qubits"
                for item in equivalence
                if item["rewritten"]["verdict"] != "VALID"
                or item["mutated"]["verdict"] != "INVALID"]
    failing += [f"state scope at {item['qubits']} qubits"
                for item in scopes
                if item["state_scope"]["verdict"] != "VALID"
                or item["unitary_scope"]["verdict"] != "INVALID"]

    checks = [
        {
            "check": "every circuit family passes its own correctness checks at every size",
            "requirement": "closed-form GHZ generators, the defined graph-state "
                           "generators, a valid pure tableau, and exact eigenstates",
            "measured": {"rows": len(rows), "failed": [row["family"] + "@" + str(row["qubits"])
                                                       for row in rows
                                                       if not all(row["checks"].values())]},
            "passed": not any(not all(row["checks"].values()) for row in rows),
        },
        {
            "check": "an equivalent rewriting is VALID and a mutated gate is INVALID",
            "requirement": "decided from 2n Pauli images, with no dense matrix at any size",
            "measured": {"sizes": [item["qubits"] for item in equivalence],
                         "rewritten": [item["rewritten"]["verdict"] for item in equivalence],
                         "mutated": [item["mutated"]["verdict"] for item in equivalence]},
            "passed": all(item["rewritten"]["verdict"] == "VALID"
                          and item["mutated"]["verdict"] == "INVALID" for item in equivalence),
        },
        {
            "check": "chain versus star is the same prepared state and a different unitary",
            "requirement": "state scope VALID, unitary scope INVALID, at every size",
            "measured": {"sizes": [item["qubits"] for item in scopes],
                         "state_scope": [item["state_scope"]["verdict"] for item in scopes],
                         "unitary_scope": [item["unitary_scope"]["verdict"] for item in scopes]},
            "passed": all(item["state_scope"]["verdict"] == "VALID"
                          and item["unitary_scope"]["verdict"] == "INVALID" for item in scopes),
        },
    ]
    failed = bool(failing)

    payload = {
        "schema": "apedp.vqc.stabilizer_scaling.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": {"sizes": args.sizes, "shots": args.shots, "seed": args.seed},
        "sizes": args.sizes,
        "rows": rows,
        "equivalence_at_scale": equivalence,
        "state_scope_at_scale": scopes,
        "checks": checks,
        "limitations": [
            "Clifford gates only (h, x, y, z, s, sdg, cx, cz, swap); a non-Clifford gate "
            "or any noise operation is refused rather than approximated",
            "pure stabilizer states only: every Noise operation is refused, so sampled "
            "Pauli trajectories are not measured here",
            "the dense comparison stops where the dense tier stops (about ten qubits "
            "for a matrix); at 1,000 qubits the only checks available are the closed "
            "form, the stabilizer definitions, and the equivalence referee",
            "wall clock is one host and one process: the record measures this "
            "implementation, not an asymptotic scaling law",
            "the GHZ chain is a Clifford circuit with a closed form; reaching 1,000 "
            "qubits with it is a reach-and-agreement result, not a claim about "
            "arbitrary 1,000-qubit circuits",
            f"sampling is Z-basis only, at most {args.shots} shots per row and one shot "
            f"above 200 qubits where a shot costs seconds; the samples establish "
            f"support and parity, not a distribution over 2**999 strings",
        ],
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "module_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "vqc" / "stabilizer.py").read_bytes()).hexdigest(),
            "equivalence_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "vqc" / "equivalence.py").read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": ("python scripts/measure_vqc_stabilizer_scaling.py --sizes "
                        + " ".join(str(size) for size in args.sizes)
                        + f" --shots {args.shots} --seed {args.seed}"),
        },
        "verdict": ("the stabilizer tier builds, validates, samples and adjudicates "
                    "Clifford circuits at every size measured, and the two circuits of "
                    "each pair are decided from 2n images with no matrix"
                    if not failed else "inconsistent"),
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")

    for row in rows:
        print(f"{row['family']:<24} n={row['qubits']:>4} gates={row['gates']:>5} "
              f"build={row['build_seconds']:.3f}s validate={row['validate_seconds']:.3f}s "
              f"sample={row['sample_seconds_per_shot'] * 1000:.1f}ms "
              f"checks={'ok' if all(row['checks'].values()) else 'FAILED'}")
    for item in equivalence:
        print(f"equivalence n={item['qubits']:>4} rewritten={item['rewritten']['verdict']} "
              f"({item['rewritten']['seconds']:.2f}s) mutated={item['mutated']['verdict']} "
              f"({item['mutated']['seconds']:.2f}s)")
    print("failed checks:", failing or "none")
    print(f"wrote {args.record}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
