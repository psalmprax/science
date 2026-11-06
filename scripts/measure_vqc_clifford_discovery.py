"""Discovery to adjudication at 100 qubits, without a dense matrix.

Status: experimental

One workload, run end to end: a **reference** Clifford circuit that is deliberately
inefficient (a GHZ chain with cancelling junk spliced into it), a **search** that
proposes local rewrites, the **referee** that certifies every accepted step as the
same unitary, the **simulator** that re-checks the two circuits by sampling, the
**resource counter** that says what was saved, and **admission** that promotes the
result only if the evidence supports it.

Three things this record is built to demonstrate, because each one is a way the loop
could be wrong:

* **A cheaper circuit that is only state-equivalent is refused.** The star GHZ
  prepares the same 100-qubit state as the chain in half the gates of the junk
  reference; it is not the same unitary, so unitary scope says INVALID and admission
  refuses it. A resource-only search would have promoted it.
* **Mutated circuits are refused, with the image named.** The search's mutations
  (delete a gate, flip a phase, swap `cx` for `cz`) are certified too, so the loop
  demonstrates rejection and not only acceptance.
* **The accepted steps are certified against the original reference**, not against
  the previous step, so the search cannot drift by accumulating small errors.

Run::

    python scripts/measure_vqc_clifford_discovery.py

The record is `benchmarks/quantum_compilation/vqc_clifford_discovery.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "vqc_clifford_discovery.json")

sys.path.insert(0, str(ROOT))

from apedp_dyn.vqc import Circuit, Gate, Measure  # noqa: E402
from apedp_dyn.vqc.clifford_search import (  # noqa: E402
    propose_rewrites,
    search,
)
from apedp_dyn.vqc.discovery import (  # noqa: E402
    CircuitCandidate,
    PromotionRefused,
    adjudicate_candidate,
    promote,
)
from apedp_dyn.vqc.equivalence import (  # noqa: E402
    adjudicate_clifford_equivalence,
    adjudicate_equivalence,
)
from apedp_dyn.vqc.stabilizer import StabilizerBackend  # noqa: E402


def ghz_chain(num_qubits: int) -> Circuit:
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def ghz_star(num_qubits: int) -> Circuit:
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (0, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def junked_chain(num_qubits: int) -> Circuit:
    """The GHZ chain plus work that cancels: the search's problem instance.

    Three kinds of junk, because they exercise three different rewrite rules:

    * `z z` pairs on the same qubit -- a self-inverse pair;
    * `s s s s` blocks -- the `S` algebra (`s s = z`, then `z z = I`);
    * `h q` far apart with only gates on other qubits between -- cancelling a pair
      that only commutes past everything in between.
    """
    ops: list = [Gate("h", (0,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    for qubit in range(0, num_qubits, 5):
        ops.extend([Gate("z", (qubit,)), Gate("z", (qubit,))])
    for qubit in range(1, num_qubits, 7):
        ops.extend([Gate("s", (qubit,)) for _ in range(4)])
    # A pair on qubit 99 after the chain's last gate touching it, separated by work on
    # other qubits: cancelling it requires commuting the pair past that work.
    ops.append(Gate("h", (num_qubits - 1,)))
    for qubit in range(0, num_qubits, 11):
        ops.extend([Gate("z", (qubit,)), Gate("z", (qubit,))])
    ops.append(Gate("h", (num_qubits - 1,)))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def sampling_agreement(reference: Circuit, candidate: Circuit, *, shots: int,
                       seed: int) -> dict:
    """The simulator's cross-check: same noise-free distribution, same support."""
    ops = tuple(reference.ops) + tuple(Measure(qubit, qubit)
                                       for qubit in range(reference.num_qubits))
    ops_candidate = tuple(candidate.ops) + tuple(Measure(qubit, qubit)
                                                 for qubit in range(candidate.num_qubits))
    left = Circuit(ops=ops, num_qubits=reference.num_qubits,
                   num_bits=reference.num_qubits)
    right = Circuit(ops=ops_candidate, num_qubits=candidate.num_qubits,
                    num_bits=candidate.num_qubits)
    start = time.perf_counter()
    a = StabilizerBackend().run(left, shots=shots, seed=seed)
    b = StabilizerBackend().run(right, shots=shots, seed=seed)
    seconds = time.perf_counter() - start
    support_a = {key for key, count in a.counts.items() if count}
    support_b = {key for key, count in b.counts.items() if count}
    sigma = 4.0 * np.sqrt(max(0.5 * (1 - 0.5), 1e-12) * shots) + 10
    worst = 0
    for key in support_a | support_b:
        worst = max(worst, abs(a.counts.get(key, 0) - b.counts.get(key, 0)))
    return {
        "shots": shots,
        "seed": seed,
        "reference_support": sorted(support_a),
        "candidate_support": sorted(support_b),
        "support_identical": support_a == support_b,
        "max_count_difference": int(worst),
        "within_four_sigma": bool(worst <= sigma),
        "seconds": seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--qubits", type=int, default=100)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    reference = junked_chain(args.qubits)
    # Generous cap: the first version of this record stopped at the default 64 steps
    # and reported 116 operations, which was the cap and not a local optimum.
    step_limit = 1000
    start = time.perf_counter()
    outcome = search(reference, seed=args.seed, mutations=6, max_steps=step_limit,
                     candidate_id="ghz-chain-rewritten")
    search_seconds = time.perf_counter() - start

    proposals = len(list(propose_rewrites(reference)))
    reference_resources = reference.resources()
    best_resources = outcome.best.resources()

    sampling = sampling_agreement(reference, outcome.best, shots=args.shots,
                                  seed=args.seed)

    # The trap: a cheaper circuit that only prepares the same *state*.
    star = ghz_star(args.qubits)
    star_unitary = adjudicate_clifford_equivalence(reference, star)
    star_state = adjudicate_clifford_equivalence(reference, star, on="state")
    star_candidate = CircuitCandidate("ghz-star", star, reference, source="script",
                                      claims={"operations": star.resources()["operations"]})
    star_verdict = adjudicate_candidate(star_candidate)
    star_promoted = True
    star_refusal = None
    try:
        promote(star_verdict)
    except PromotionRefused as error:
        star_promoted = False
        star_refusal = str(error)

    # A dense check where the dense tier can still hold the circuit, on a small copy.
    small_reference = junked_chain(8)
    small = search(small_reference, seed=args.seed, mutations=2)
    dense = adjudicate_equivalence(small_reference, small.best)

    refusals = [step for step in outcome.mutations if not step.accepted]
    passing = [step for step in outcome.mutations if step.accepted]

    checks = [
        {
            "check": "every accepted rewrite is certified as the same unitary",
            "requirement": "each step compared against the original reference at "
                           "100 qubits, from 2n Pauli images, with no matrix",
            "measured": {"steps": len(outcome.steps),
                         "accepted": len(outcome.accepted),
                         "rejected": len(outcome.steps) - len(outcome.accepted)},
            "passed": (len(outcome.steps) > 0
                       and len(outcome.accepted) == len(outcome.steps)
                       and all(step.verdict == "VALID" for step in outcome.steps)),
        },
        {
            "check": "the best circuit is strictly cheaper than the reference",
            "requirement": "fewer operations, and fewer two-qubit gates",
            "measured": {"reference": reference_resources["operations"],
                         "best": best_resources["operations"],
                         "reference_two_qubit": reference_resources["two_qubit_gates"],
                         "best_two_qubit": best_resources["two_qubit_gates"]},
            "passed": (best_resources["operations"] < reference_resources["operations"]
                       and best_resources["two_qubit_gates"]
                       <= reference_resources["two_qubit_gates"]),
        },
        {
            "check": "the search converged before its step cap",
            "requirement": "a local optimum, not a truncated loop",
            "measured": {"steps": len(outcome.steps), "cap": step_limit},
            "passed": len(outcome.steps) < step_limit,
        },
        {
            "check": "the promoted candidate carries sufficient evidence",
            "requirement": "valid, claim met, evidence SUFFICIENT, tier stabilizer",
            "measured": {"promoted": outcome.promoted,
                         "tier": None if outcome.verdict is None else outcome.verdict.tier,
                         "semantic": None if outcome.verdict is None else
                         outcome.verdict.verdict,
                         "resource": None if outcome.verdict is None else
                         outcome.verdict.claim_verdict,
                         "evidence": None if outcome.verdict is None else
                         outcome.verdict.evidence},
            "passed": bool(outcome.promoted and outcome.verdict is not None
                           and outcome.verdict.evidence == "SUFFICIENT"
                           and outcome.verdict.tier == "stabilizer"
                           and outcome.verdict.claim_verdict == "MET"),
        },
        {
            "check": "mutated candidates are refused, not promoted",
            "requirement": "every semantics-breaking edit comes back INVALID with a "
                           "named image",
            "measured": {"mutations": len(outcome.mutations),
                         "refused": len(refusals),
                         "passed_by_accident": len(passing)},
            "passed": (len(refusals) > 0
                       and all(step.mismatched > 0 for step in refusals)
                       and all(step.first_mismatch is not None for step in refusals)),
        },
        {
            "check": "a cheaper candidate that only prepares the same state is refused",
            "requirement": "unitary scope INVALID, state scope VALID, admission refuses",
            "measured": {"star_operations": star.resources()["operations"],
                         "reference_operations": reference_resources["operations"],
                         "unitary_scope": star_unitary.verdict,
                         "unitary_mismatched": star_unitary.mismatched,
                         "state_scope": star_state.verdict,
                         "admission": None if star_verdict is None
                         else star_verdict.verdict,
                         "promoted": star_promoted},
            "passed": (star_unitary.verdict == "INVALID"
                       and star_state.verdict == "VALID"
                       and star_promoted is False
                       and star.resources()["operations"]
                       < reference_resources["operations"]),
        },
        {
            "check": "the two circuits sample identically on the stabilizer backend",
            "requirement": "same support, counts within four sigma at 64 shots",
            "measured": sampling,
            "passed": bool(sampling["support_identical"] and sampling["within_four_sigma"]),
        },
        {
            "check": "the dense tier agrees on a small copy of the same workload",
            "requirement": "8-qubit version checked with the dense referee as well",
            "measured": {"dense_verdict": dense.verdict, "dense_distance": dense.distance,
                         "small_reference_operations": small_reference.resources()["operations"],
                         "small_best_operations": small.best.resources()["operations"],
                         "small_step_cap_hit": len(small.steps) >= 64},
            "passed": dense.verdict == "VALID",
        },
    ]

    failed = [check["check"] for check in checks if not check["passed"]]

    payload = {
        "schema": "apedp.vqc.clifford_discovery.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": {"qubits": args.qubits, "shots": args.shots, "seed": args.seed},
        "qubits": args.qubits,
        "workload": {
            "reference_operations": reference_resources["operations"],
            "reference_two_qubit_gates": reference_resources["two_qubit_gates"],
            "reference_depth": reference_resources["depth"],
            "best_operations": best_resources["operations"],
            "best_two_qubit_gates": best_resources["two_qubit_gates"],
            "best_depth": best_resources["depth"],
            "gates_saved": reference_resources["operations"]
            - best_resources["operations"],
            "rewrites_available_at_the_reference": proposals,
            "step_limit": step_limit,
            "search_seconds": search_seconds,
            # Stated in the record itself, not only in the prose around it: the
            # reference is *deliberately* inefficient. The headline reduction is a
            # measurement of the rewrite engine against that workload, and it is not
            # an advantage over an already-optimised GHZ preparation, which is n
            # operations and depth n for n qubits -- exactly what the search reaches.
            "reference_provenance": {
                "described_as": "a GHZ chain with cancelling pairs, s/sdg algebra and "
                                "Pauli-conjugation identities spliced into it",
                "deliberately_inefficient": True,
                "optimised_ghz_reference_operations": args.qubits,
                "optimised_ghz_reference_depth": args.qubits,
                "what_the_reduction_measures": (
                    "how much of the inserted work the rewrite rules remove; it does "
                    "not establish a 54.95% advantage over an optimised GHZ "
                    "implementation, and the discovered circuit is that optimised "
                    "preparation, not something shorter"),
                "what_it_does_not_measure": (
                    "cost against a circuit built by a synthesiser from the "
                    "specification, which is the experiment in "
                    "measure_vqc_synthesis.py"),
            },
        },
        "steps": [step.as_dict() for step in outcome.steps],
        "mutations": [step.as_dict() for step in outcome.mutations],
        "admission": None if outcome.verdict is None else outcome.verdict.as_dict(),
        "sampling": sampling,
        "state_scope_only": {
            "unitary_scope": star_unitary.verdict,
            "unitary_mismatched": star_unitary.mismatched,
            "state_scope": star_state.verdict,
            "admission": None if star_verdict is None else {
                "verdict": star_verdict.verdict,
                "evidence": star_verdict.evidence,
                "promotable": star_verdict.promotable},
            "refusal": star_refusal,
        },
        "dense_cross_check": {"qubits": 8, "verdict": dense.verdict,
                              "distance": dense.distance},
        "checks": checks,
        "limitations": [
            "the search space is local rewrites over Clifford gates, not a general "
            "synthesiser: it cannot propose a circuit that is not reachable by "
            "cancellations, S/Z algebra, Pauli conjugation or commuting a pair past "
            "disjoint or otherwise commuting gates",
            "the reference circuit is deliberately inefficient (see "
            "workload.reference_provenance): the measured reduction is against an "
            "unoptimised circuit, and the discovered 100-operation chain is the "
            "optimised GHZ preparation itself, so the number to quote is 'it reached "
            "the n-operation, depth-n circuit', not 'it is 55% faster than GHZ'",
            "synthesis from a specification, with no reference circuit, is a separate "
            "experiment (scripts/measure_vqc_synthesis.py); this loop still needs a "
            "circuit to start from",
            "the ranking is (operations, depth); the search is greedy and stops at a "
            "local optimum, so 'best' means best reachable, not optimal",
            "the referee is exact but the workload is a Clifford circuit with a closed "
            "form; a general non-Clifford circuit at this size is still UNRESOLVED",
            "sampling is a spot check at a stated shot count, not a distribution test, "
            "and neither backends nor the referee execute noise here",
            "wall clock is one host and one process",
        ],
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "search_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "vqc" / "clifford_search.py").read_bytes()).hexdigest(),
            "equivalence_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "vqc" / "equivalence.py").read_bytes()).hexdigest(),
            "discovery_sha256": hashlib.sha256(
                (ROOT / "apedp_dyn" / "vqc" / "discovery.py").read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": (f"python scripts/measure_vqc_clifford_discovery.py "
                        f"--qubits {args.qubits} --shots {args.shots} --seed {args.seed}"),
        },
        "verdict": ("a 100-qubit Clifford circuit was rewritten, every step certified "
                    "as the same unitary without a matrix, the result promoted on "
                    "sufficient evidence, and the state-only and mutated alternatives "
                    "refused" if not failed else "inconsistent"),
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"reference: {reference_resources['operations']} ops, "
          f"{reference_resources['depth']} deep -> best "
          f"{best_resources['operations']} ops, {best_resources['depth']} deep "
          f"in {search_seconds:.2f}s")
    print(f"steps: {len(outcome.steps)} accepted, "
          f"{len(outcome.steps) - len(outcome.accepted)} rejected; "
          f"mutations refused {len(refusals)}/{len(outcome.mutations)}")
    print(f"promoted: {outcome.promoted} "
          f"(tier={None if outcome.verdict is None else outcome.verdict.tier}, "
          f"evidence={None if outcome.verdict is None else outcome.verdict.evidence})")
    print(f"star: unitary={star_unitary.verdict} ({star_unitary.mismatched}/"
          f"{star_unitary.compared}) state={star_state.verdict} "
          f"promoted={star_promoted}")
    print(f"sampling: identical support={sampling['support_identical']}, "
          f"max count diff={sampling['max_count_difference']}")
    print(f"dense 8-qubit cross-check: {dense.verdict} at distance {dense.distance}")
    print("failed checks:", failed or "none")
    print(f"wrote {args.record}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
