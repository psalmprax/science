"""Discovery over Clifford circuits: propose, certify, rank, promote.

Status: experimental

This is the smallest complete discovery loop the platform can run today, and it is
deliberately not a general circuit synthesiser:

* **propose** -- local rewrites that are semantics-preserving *by construction*:
  cancelling self-inverse pairs (`h h`, `x x`, `cx cx`, `cz cz`, `swap swap`),
  commuting such a pair past gates that commute with it, the `S`/`Sdg`
  algebra (`s sdg` vanishes, `s s = z`), and the two Pauli identities `h z h = x`
  and `h x h = z`;
* **certify** -- every accepted step is decided by
  :func:`apedp_dyn.vqc.equivalence.adjudicate_clifford_equivalence` against the
  *original* reference, so a rewrite rule that is wrong is caught rather than
  trusted, and a candidate cannot drift by accumulating small errors;
* **rank** -- the resource counter (`operations`, then `depth`), not a heuristic;
* **promote** -- through :mod:`apedp_dyn.vqc.discovery`, so the promoted candidate
  has to be `VALID`, meet its claim and carry sufficient evidence.

The search also runs **mutations** that are not semantics-preserving (delete a gate,
swap `s` for `sdg`, flip `cx` to `cz`). They exist to be refused: a discovery loop
whose wrong candidates are never tested is a loop that cannot be told apart from one
that accepts everything. Each refusal is recorded with the image that differed.

Nothing here needs a dense matrix: a 100-qubit pair is decided from `2n` signed Pauli
strings, which is why the loop can run at that size at all.
"""
from __future__ import annotations

import dataclasses
import itertools
import random
from collections.abc import Iterator, Mapping, Sequence

from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.discovery import (
    CandidateVerdict,
    CircuitCandidate,
    PromotionRefused,
    adjudicate_candidate,
    promote,
)
from apedp_dyn.vqc.equivalence import CliffordEquivalenceReport, adjudicate_clifford_equivalence

__all__ = [
    "RewriteStep",
    "SearchOutcome",
    "propose_mutations",
    "propose_rewrites",
    "resource_score",
    "search",
]

#: Gates that are their own inverse: two adjacent ones on the same targets are `I`.
SELF_INVERSE = frozenset({"h", "x", "y", "z", "cx", "cz", "swap"})


def resource_score(circuit: Circuit) -> tuple[int, int]:
    """The ordering the search minimises: `(operations, depth)`."""
    resources = circuit.resources()
    return (int(resources["operations"]), int(resources["depth"]))


def _gate(op: object) -> Gate | None:
    """The gate, if this operation is an unconditioned one; `None` otherwise."""
    if isinstance(op, Gate) and not op.condition and not op.params:
        return op
    return None


def _commute(left: Gate, right: Gate) -> bool:
    """Conservative commutation rules for the supported unparameterized gates."""
    left_targets, right_targets = set(left.targets), set(right.targets)
    if not left_targets & right_targets:
        return True
    diagonal = {"z", "s", "sdg", "cz", "t", "tdg"}
    if left.name in diagonal and right.name in diagonal:
        return True
    if left.name == right.name == "cx":
        left_control, left_target = left.targets
        right_control, right_target = right.targets
        # CNOTs commute when their controls or their targets are shared.
        return left_control == right_control or left_target == right_target
    return False


def _transparent_between(ops: Sequence[object], start: int, stop: int,
                         mover: Gate) -> bool:
    """Whether every operation strictly between the pair commutes with it."""
    for index in range(start + 1, stop):
        op = _gate(ops[index])
        if op is None or not _commute(mover, op):
            return False
    return True


def _with_pair_removed(circuit: Circuit, first: int, second: int,
                       replacement: tuple[Gate, ...] = ()) -> Circuit:
    ops = [op for index, op in enumerate(circuit.ops) if index not in (first, second)]
    ops[first:first] = list(replacement)
    return Circuit(ops=tuple(ops), num_qubits=circuit.num_qubits,
                   num_bits=circuit.num_bits, ancillas=circuit.ancillas,
                   metadata={"generator": "vqc.clifford_search.rewrite"})


def _pairs_by_key(circuit: Circuit) -> dict[tuple[str, tuple[int, ...]], list[int]]:
    """Positions grouped by `(name, targets)`, so only real candidates pair up."""
    groups: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for index, op in enumerate(circuit.ops):
        gate = _gate(op)
        if gate is None:
            continue
        groups.setdefault((gate.name, tuple(gate.targets)), []).append(index)
    return groups


def propose_rewrites(circuit: Circuit) -> Iterator[tuple[str, Circuit]]:
    """Every semantics-preserving rewrite available in one step of the search."""
    ops = circuit.ops
    groups = _pairs_by_key(circuit)
    for (name, targets), positions in sorted(groups.items()):
        if len(positions) < 2:
            continue
        mover = Gate(name, targets)
        for first_index, second_index in itertools.pairwise(positions):
            if not _transparent_between(ops, first_index, second_index, mover):
                continue
            if name in SELF_INVERSE:
                yield (f"cancel {name}{tuple(targets)}", _with_pair_removed(
                    circuit, first_index, second_index))
            elif name == "s":
                # `s s = z`, which is one gate fewer and the same unitary.
                qubit = targets[0]
                yield ("s s = z", _with_pair_removed(circuit, first_index, second_index,
                                                     (Gate("z", (qubit,)),)))

    # `s sdg = I` pairs the two gate names, so they are not in one group above.
    for (name, targets), positions in sorted(groups.items()):
        if name not in ("s", "sdg"):
            continue
        others = groups.get((("sdg" if name == "s" else "s"), targets), [])
        for first_index in positions:
            for second_index in others:
                if second_index < first_index:
                    continue
                if not _transparent_between(ops, first_index, second_index,
                                            Gate(name, targets)):
                    continue
                yield ("cancel s*sdg", _with_pair_removed(circuit, first_index,
                                                          second_index))

    # `h z h = x` and `h x h = z`: three gates become one, exactly.
    # H Y H = -Y; the minus sign is one free global phase for this standalone
    # unitary, just as in the equivalence referee's other conjugation checks.
    conjugation = {("h", "z"): "x", ("h", "x"): "z", ("h", "y"): "y"}
    for index in range(len(ops) - 2):
        first, middle, last = (_gate(ops[index]), _gate(ops[index + 1]),
                               _gate(ops[index + 2]))
        if first is None or middle is None or last is None:
            continue
        if first.name != "h" or last.name != "h" or first.targets != last.targets:
            continue
        if middle.targets != first.targets:
            continue
        replacement = conjugation.get(("h", middle.name))
        if replacement is None:
            continue
        ops_out = [*ops[:index], Gate(replacement, first.targets), *ops[index + 3:]]
        yield (f"h {middle.name} h = {replacement}", Circuit(
            ops=tuple(ops_out), num_qubits=circuit.num_qubits, num_bits=circuit.num_bits,
            ancillas=circuit.ancillas,
            metadata={"generator": "vqc.clifford_search.rewrite"}))


def propose_mutations(circuit: Circuit, *, seed: int, count: int = 4
                      ) -> Iterator[tuple[str, Circuit]]:
    """Semantics-*breaking* edits, offered so the referee can refuse them."""
    rng = random.Random(seed)
    gate_positions = [index for index, op in enumerate(circuit.ops)
                      if _gate(op) is not None]
    if not gate_positions:
        return
    for _ in range(count):
        choice = rng.choice(gate_positions)
        ops = list(circuit.ops)
        gate = _gate(ops[choice])
        assert gate is not None
        flavour = rng.randrange(3)
        if flavour == 0 and len(ops) > 1:
            ops.pop(choice)
            label = f"delete {gate.name}{tuple(gate.targets)}"
        elif gate.name in ("s", "sdg"):
            # Phase flipped: same arity, one phase sign the other way.
            ops[choice] = Gate("sdg" if gate.name == "s" else "s", gate.targets)
            label = f"{gate.name} -> phase"
        elif gate.name in ("cx", "cz"):
            ops[choice] = Gate("cz" if gate.name == "cx" else "cx", gate.targets)
            label = f"{gate.name} -> the other two-qubit gate"
        else:
            # `h`, `x`, `y`, `z`: inserting a different Pauli is a different unitary.
            ops[choice] = Gate("x" if gate.name != "x" else "z", gate.targets)
            label = f"{gate.name} -> Pauli"
        yield (label, Circuit(ops=tuple(ops), num_qubits=circuit.num_qubits,
                              num_bits=circuit.num_bits, ancillas=circuit.ancillas,
                              metadata={"generator": "vqc.clifford_search.mutation"}))


@dataclasses.dataclass(frozen=True)
class RewriteStep:
    """One proposed circuit, and what the referee said about it."""

    rule: str
    circuit: Circuit
    score: tuple[int, int]
    verdict: str
    mismatched: int
    first_mismatch: Mapping | None
    accepted: bool

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "score": list(self.score),
            "verdict": self.verdict,
            "mismatched": self.mismatched,
            "first_mismatch": (None if self.first_mismatch is None
                               else dict(self.first_mismatch)),
            "accepted": self.accepted,
        }


@dataclasses.dataclass(frozen=True)
class SearchOutcome:
    """The whole loop: what was proposed, what was certified, what was promoted."""

    reference: Circuit
    best: Circuit
    steps: tuple[RewriteStep, ...]
    mutations: tuple[RewriteStep, ...]
    verdict: CandidateVerdict | None
    promoted: bool
    promotion_refusal: str | None

    @property
    def accepted(self) -> tuple[RewriteStep, ...]:
        return tuple(step for step in self.steps if step.accepted)

    def as_dict(self) -> dict:
        return {
            "reference": self.reference.resources(),
            "best": self.best.resources(),
            "steps": len(self.steps),
            "accepted": len(self.accepted),
            "rejected": len(self.steps) - len(self.accepted),
            "mutations": len(self.mutations),
            "mutations_refused": sum(1 for step in self.mutations if not step.accepted),
            "promoted": self.promoted,
            "promotion_refusal": self.promotion_refusal,
            "verdict": (None if self.verdict is None
                        else {"semantic": self.verdict.verdict,
                              "resource": self.verdict.claim_verdict,
                              "evidence": self.verdict.evidence,
                              "promotable": self.verdict.promotable,
                              "tier": self.verdict.tier}),
            "step_log": [step.as_dict() for step in self.steps],
            "mutation_log": [step.as_dict() for step in self.mutations],
        }


def _certify(reference: Circuit, candidate: Circuit, rule: str,
             ) -> tuple[CliffordEquivalenceReport, RewriteStep]:
    report = adjudicate_clifford_equivalence(reference, candidate)
    step = RewriteStep(
        rule=rule,
        circuit=candidate,
        score=resource_score(candidate),
        verdict=report.verdict,
        mismatched=report.mismatched,
        first_mismatch=(report.mismatches[0] if report.mismatches else None),
        accepted=report.verdict == "VALID",
    )
    return report, step


def search(reference: Circuit, *, max_steps: int = 64, mutations: int = 4,
           seed: int = 0, candidate_id: str = "clifford-rewrite",
           claims: Mapping[str, int] | None = None) -> SearchOutcome:
    """Greedy best-first rewrite search, certified at every step, then admitted.

    Each step takes the rewrite with the smallest `(operations, depth)` and asks the
    Clifford referee whether it is the *same unitary* as the reference. Only a VALID
    candidate becomes the new current circuit, so the loop cannot walk away from the
    reference; a rejected step is recorded and the search stops, because a
    semantics-preserving rule that fails certification is a bug in this module and
    silently trying the next rule would hide it.
    """
    reference.validate()
    current = reference
    steps: list[RewriteStep] = []
    for _ in range(max_steps):
        best: tuple[tuple[int, int], str, Circuit] | None = None
        for rule, candidate in propose_rewrites(current):
            score = resource_score(candidate)
            if best is None or score < best[0]:
                best = (score, rule, candidate)
        if best is None:
            break
        _report, step = _certify(reference, best[2], best[1])
        steps.append(step)
        if not step.accepted:
            break
        # Keep the certified candidate, not the proposal: they are the same circuit,
        # and this makes the invariant explicit.
        current = step.circuit

    mutation_steps: list[RewriteStep] = []
    for rule, candidate in propose_mutations(current, seed=seed, count=mutations):
        _report, step = _certify(reference, candidate, rule)
        # A mutation that happens to be semantics-preserving is a rewrite in disguise;
        # it is recorded as such and not counted as a refusal.
        mutation_steps.append(step)

    admitted_claims = dict(claims) if claims else {
        "operations": int(current.resources()["operations"]),
    }
    candidate = CircuitCandidate(candidate_id=candidate_id, circuit=current,
                                 reference=reference, source="vqc.clifford_search",
                                 claims=admitted_claims,
                                 metadata={"search": "greedy-rewrite"})
    verdict = adjudicate_candidate(candidate)
    promoted = False
    refusal: str | None = None
    try:
        promote(verdict)
        promoted = True
    except PromotionRefused as error:
        refusal = str(error)

    return SearchOutcome(reference=reference, best=current, steps=tuple(steps),
                         mutations=tuple(mutation_steps), verdict=verdict,
                         promoted=promoted, promotion_refusal=refusal)
