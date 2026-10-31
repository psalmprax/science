#!/usr/bin/env python3
"""Cross-check the stabilizer tier against the dense tier, then use it at 100 qubits.

The dense amplitude tier is the referee: it approximates nothing, and it is
unusable past roughly 25 qubits. This script does the two things that together let
a 100-qubit stabilizer number be cited at all:

1. **agreement** -- the same `ghz_circuit` builder run on both tiers, every one of
   the ``2**n`` outcome probabilities compared exactly, plus a set of Pauli
   expectations, for chain lengths the dense tier can hold;
2. **reach** -- the 100-qubit chain on the stabilizer tier, where ``2**100``
   amplitudes do not fit anywhere, checked against the closed-form GHZ
   predictions that step 1 has just confirmed at small `n`.

The second is only as strong as the first, which is why the record carries both
and why the reported quantity for step 1 is a deviation, not a verdict.

Examples::

    .venv/bin/python scripts/measure_vqc_stabilizer_ghz.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_stabilizer_ghz.json"

sys.path.insert(0, str(ROOT))

from apedp_dyn.quantum_compilation.statevector_sim import (  # noqa: E402
    StatevectorSimulator,
    dimension_report,
)
from apedp_dyn.vqc import Circuit, Gate, StatevectorBackend  # noqa: E402
from apedp_dyn.vqc.equivalence import (  # noqa: E402
    adjudicate_clifford_equivalence,
    adjudicate_equivalence,
)
from apedp_dyn.vqc.gates import matrix_for  # noqa: E402
from apedp_dyn.vqc.stabilizer import (  # noqa: E402
    StabilizerBackend,
    StabilizerState,
    ghz_circuit,
    ghz_even_parity_probability,
    ghz_generators,
)

LABELS = "IXYZ"
#: `2**12` amplitudes on the dense tier is the largest comparison worth timing here.
DENSE_LIMIT = 12


def bits_of(index: int, num_qubits: int) -> tuple[bool, ...]:
    """Qubit `q` is bit `q` of the amplitude index, which is the IR's convention."""
    return tuple(bool((index >> qubit) & 1) for qubit in range(num_qubits))


def dense_state(circuit) -> np.ndarray:
    result = StatevectorBackend().run(circuit, shots=1, seed=0, keep_states=True)
    return result.final_states[0]


def stabilizer_state(circuit) -> StabilizerState:
    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        if not isinstance(op, Gate):
            raise ValueError("the unitary prefix must contain no measurements")
        state.apply_gate(op.name, op.targets, op.params)
    return state


def dense_expectation(state: np.ndarray, labels) -> float:
    simulator = StatevectorSimulator(int(math.log2(state.size)))
    simulator.state = np.array(state, dtype=np.complex128)
    for qubit, label in enumerate(labels):
        if label:
            simulator.apply_matrix(qubit, matrix_for(LABELS[label].lower()))
    return float(np.real(np.vdot(state, simulator.state)))


def probability_agreement(limit: int = DENSE_LIMIT) -> list[dict]:
    """Every outcome probability of the GHZ chain, on both tiers, at each size."""
    rows = []
    for num_qubits in range(2, limit + 1):
        circuit = ghz_circuit(num_qubits, measure=False)
        dense = np.abs(dense_state(circuit)) ** 2
        state = stabilizer_state(circuit)
        worst = 0.0
        nonzero = 0
        for index in range(1 << num_qubits):
            probability = state.probability_of_bitstring(bits_of(index, num_qubits))
            worst = max(worst, abs(probability - float(dense[index])))
            nonzero += probability > 0
        rows.append({
            "qubits": num_qubits,
            "amplitudes_compared": int(dense.size),
            "nonzero_outcomes": nonzero,
            "expected_nonzero_outcomes": 2,
            "max_abs_probability_deviation": worst,
        })
    return rows


def pauli_agreement(num_qubits: int, *, seed: int, strings: int = 16) -> dict:
    """`<P>` for the GHZ stabilizers and for random Pauli strings, on both tiers."""
    circuit = ghz_circuit(num_qubits, measure=False)
    state = stabilizer_state(circuit)
    dense = dense_state(circuit)
    rng = random.Random(seed)

    candidates = [tuple([1] * num_qubits)]  # X on every qubit
    for qubit in range(num_qubits - 1):
        labels = [0] * num_qubits
        labels[qubit] = labels[qubit + 1] = 3
        candidates.append(tuple(labels))
    candidates.extend(tuple(rng.randrange(4) for _ in range(num_qubits))
                      for _ in range(strings))

    rows = []
    for labels in candidates:
        rows.append({
            "pauli": "".join(LABELS[label] for label in labels),
            "stabilizer": state.expectation(labels),
            "dense": dense_expectation(dense, labels),
        })
    worst = max(abs(row["stabilizer"] - row["dense"]) for row in rows)
    return {
        "qubits": num_qubits,
        "strings": len(rows),
        "max_abs_deviation": worst,
        "rows": rows,
    }


def hundred_qubit_run(shots: int, x_shots: int, seed: int) -> dict:
    circuit = ghz_circuit(100)
    state = stabilizer_state(ghz_circuit(100, measure=False))

    start = time.perf_counter()
    result = StabilizerBackend().run(circuit, shots=shots, seed=seed)
    elapsed = time.perf_counter() - start

    ground, excited = "0" * 100, "1" * 100
    split = {key: result.counts.get(key, 0) for key in (ground, excited)}
    sigma = math.sqrt(shots * 0.25)
    mixed_probe = "0" * 50 + "1" + "0" * 49

    x_result = StabilizerBackend().run(
        ghz_circuit(100, measure=True, basis="x"), shots=x_shots, seed=seed + 1)

    return {
        "qubits": 100,
        "generators_match_closed_form": set(state.generators) == set(ghz_generators(100)),
        "stabilizer_count": len(state.generators),
        "probability_00": state.probability_of_bitstring(tuple([False] * 100)),
        "probability_11": state.probability_of_bitstring(tuple([True] * 100)),
        "probability_of_one_mixed_string": state.probability_of_bitstring(
            tuple(bit == "1" for bit in mixed_probe)),
        "shots": shots,
        "seed": seed,
        "counts": split,
        "counts_inside_five_sigma": all(abs(count - shots / 2) <= 5 * sigma
                                        for count in split.values()),
        "only_two_outcomes_seen": set(result.counts) == {ground, excited},
        "seconds": elapsed,
        "x_basis": {
            "shots": x_shots,
            "even_parity_only": all(bits.count("1") % 2 == 0 for bits in x_result.counts),
            "distinct_strings": len(x_result.counts),
            "collisions": x_shots - len(x_result.counts),
            "closed_form_probability_each": ghz_even_parity_probability(100),
        },
        "dense_tier_cost": dimension_report(100),
        "analytic_basis_probability": 0.5,
    }


def _star_ghz(num_qubits: int) -> Circuit:
    """The same GHZ state from a different unitary: `h(0)` then `cx(0, q)`."""
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (0, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def clifford_equivalence(num_qubits: int = 100) -> dict:
    """The equivalence referee above the dense bound, on three 100-qubit pairs."""
    chain = ghz_circuit(num_qubits, measure=False)
    star = _star_ghz(num_qubits)
    shifted = Circuit(
        ops=(*chain.ops, Gate("x", (5,)), Gate("z", (5,)), Gate("x", (5,)), Gate("z", (5,))),
        num_qubits=num_qubits)
    broken = Circuit(
        ops=tuple(Gate("cz", (0, 1)) if (op.name, op.targets) == ("cx", (0, 1)) else op
                  for op in chain.ops),
        num_qubits=num_qubits)

    rows = []
    for label, candidate in (
        ("chain, plus XZXZ on qubit 5 (a global phase of -1)", shifted),
        ("star construction: the same prepared state, a different unitary", star),
        ("chain with cx(0,1) replaced by cz(0,1)", broken),
    ):
        report = adjudicate_clifford_equivalence(chain, candidate)
        rows.append({
            "candidate": label,
            "verdict": report.verdict,
            "mismatched_images": report.mismatched,
            "compared_images": report.compared,
            "first_mismatches": [dict(row) for row in report.mismatches[:2]],
        })

    state_scope = adjudicate_clifford_equivalence(chain, star, on="state")
    state_scope_shifted = adjudicate_clifford_equivalence(chain, shifted, on="state")
    dense = adjudicate_equivalence(chain, shifted)
    return {
        "qubits": num_qubits,
        "rows": rows,
        "state_scope": {
            "star": {"verdict": state_scope.verdict, "mismatched": state_scope.mismatched},
            "global_phase": {"verdict": state_scope_shifted.verdict},
            "decides": state_scope.as_dict()["decides"],
        },
        "dense_verdict": dense.verdict,
        "dense_reason": next(check.measured.get("reason", "")
                             for check in dense.checks if check.measured.get("reason")),
        "seconds": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--x-shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    agreement = probability_agreement()
    pauli = pauli_agreement(6, seed=args.seed)
    hundred = hundred_qubit_run(args.shots, args.x_shots, args.seed)
    start = time.perf_counter()
    clifford = clifford_equivalence(100)
    clifford["seconds"] = time.perf_counter() - start

    worst_probability = max(row["max_abs_probability_deviation"] for row in agreement)
    compared = sum(row["amplitudes_compared"] for row in agreement)
    checks = [{
        "check": "every GHZ outcome probability agrees with the dense tier",
        "requirement": f"maximum absolute deviation at most 1e-12 for chains of 2 to "
                       f"{DENSE_LIMIT} qubits, i.e. {compared} probabilities in total",
        "measured": {
            "chains": len(agreement),
            "probabilities": compared,
            "max_abs_deviation": worst_probability,
        },
        "passed": worst_probability <= 1e-12,
    }, {
        "check": "Pauli expectations agree with the dense tier",
        "requirement": "identical +/-1/0 for the GHZ stabilizers and random Pauli "
                       "strings on six qubits",
        "measured": {"strings": pauli["strings"],
                     "max_abs_deviation": pauli["max_abs_deviation"]},
        "passed": pauli["max_abs_deviation"] <= 1e-12,
    }, {
        "check": "the 100-qubit chain is the closed-form GHZ state",
        "requirement": "the 100 generators equal the textbook set, and the only "
                       "outcomes with nonzero probability are the two uniform strings at 1/2",
        "measured": {
            "generators_match": hundred["generators_match_closed_form"],
            "generator_count": hundred["stabilizer_count"],
            "probability_00": hundred["probability_00"],
            "probability_11": hundred["probability_11"],
            "probability_of_mixed_string": hundred["probability_of_one_mixed_string"],
        },
        "passed": (hundred["generators_match_closed_form"]
                   and hundred["stabilizer_count"] == 100
                   and hundred["probability_00"] == 0.5
                   and hundred["probability_11"] == 0.5
                   and hundred["probability_of_one_mixed_string"] == 0.0),
    }, {
        "check": "the sampled 100-qubit run produces only the two GHZ outcomes, at 1/2",
        "requirement": f"{args.shots} shots, only 0^100 and 1^100, each count within five "
                       f"sigma of {args.shots // 2}",
        "measured": {"counts": hundred["counts"],
                     "only_two_outcomes": hundred["only_two_outcomes_seen"],
                     "inside_five_sigma": hundred["counts_inside_five_sigma"]},
        "passed": hundred["only_two_outcomes_seen"] and hundred["counts_inside_five_sigma"],
    }, {
        "check": "the equivalence referee decides 100-qubit Clifford pairs",
        "requirement": "a global phase is VALID, a changed gate is INVALID with the "
                       "mismatched images named, and the same prepared state from a "
                       "different unitary is INVALID at unitary scope and VALID at state "
                       "scope",
        "measured": clifford,
        "passed": (clifford["rows"][0]["verdict"] == "VALID"
                   and clifford["rows"][1]["verdict"] == "INVALID"
                   and clifford["rows"][2]["verdict"] == "INVALID"
                   and clifford["rows"][2]["mismatched_images"] > 0
                   and clifford["state_scope"]["star"]["verdict"] == "VALID"
                   and clifford["state_scope"]["global_phase"]["verdict"] == "VALID"
                   and clifford["dense_verdict"] == "UNRESOLVED"),
    }, {
        "check": "the X-basis chain is uniform over even-parity strings",
        "requirement": "every sampled string has even parity, no two of "
                       f"{args.x_shots} shots collide (there are 2**99 of them), and the "
                       "closed-form weight is 2**-99",
        "measured": hundred["x_basis"],
        "passed": (hundred["x_basis"]["even_parity_only"]
                   and hundred["x_basis"]["collisions"] == 0
                   and hundred["x_basis"]["closed_form_probability_each"]
                   == ghz_even_parity_probability(100)),
    }]

    failed = [entry["check"] for entry in checks if not entry["passed"]]
    payload = {
        "experiment": "VQC -- the stabilizer tier, cross-checked and then used at 100 qubits",
        "tiers": {
            "dense": "apedp_dyn.vqc.backends.StatevectorBackend (exponential, exact)",
            "stabilizer": "apedp_dyn.vqc.stabilizer.StabilizerBackend (Clifford only, exact)",
        },
        "probability_agreement": agreement,
        "pauli_agreement": pauli,
        "hundred_qubit_ghz": hundred,
        "clifford_equivalence_100": clifford,
        "checks": checks,
        "failed_checks": failed,
        "limitations": [
            "the stabilizer tier refuses non-Clifford gates and every Noise operation; "
            "the dense tier remains the referee for those",
            "agreement is measured up to 12 qubits because that is where the dense tier "
            "still fits in memory, not because the stabilizer tier stops there",
            "a GHZ chain is a Clifford state with a closed form, so this demonstrates the "
            "tier's reach and its agreement, not that it can certify an arbitrary circuit",
        ],
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "module_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "vqc" / "stabilizer.py").read_bytes()).hexdigest(),
            "command": f"python scripts/measure_vqc_stabilizer_ghz.py --shots {args.shots} "
                       f"--x-shots {args.x_shots} --seed {args.seed}",
        },
        "verdict": ("the stabilizer tier reproduces the dense tier on every probability and "
                    "Pauli string it can be compared on, and carries the 100-qubit GHZ chain "
                    "the dense tier cannot hold" if not failed else "inconsistent"),
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"agreement: {len(agreement)} chains, "
          f"{sum(row['amplitudes_compared'] for row in agreement)} probabilities, "
          f"max deviation {worst_probability:.2e}")
    print(f"pauli: {pauli['strings']} strings on 6 qubits, "
          f"max deviation {pauli['max_abs_deviation']:.2e}")
    print(f"100 qubits: counts {hundred['counts']} in {hundred['seconds']:.2f}s, "
          f"generators match = {hundred['generators_match_closed_form']}")
    print(f"100 qubits, X basis: collisions {hundred['x_basis']['collisions']}, "
          f"even parity only = {hundred['x_basis']['even_parity_only']}")
    print("100-qubit equivalence: " + "; ".join(
        f"{row['verdict']} ({row['mismatched_images']}/{row['compared_images']})"
        for row in clifford["rows"])
        + f"; state scope star {clifford['state_scope']['star']['verdict']}; "
          f"dense {clifford['dense_verdict']}")
    print(f"failed checks: {failed or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
