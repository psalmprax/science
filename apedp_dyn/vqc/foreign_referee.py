"""A foreign referee for preparation circuits: adjudication that is not self-referential.

Status: experimental

Every comparison row in :mod:`apedp_dyn.vqc.external` carries a verdict, and until this
module that verdict always came from this package's own specification referee. For the
GHZ rows that was a minor asymmetry; for the shared corpus it is the serious one the
record's own limitations name: both engines' circuits were decided by the same code that
built one of them, so a shared bug in that referee would look like two correct engines,
and a disagreement would be resolved by our referee rather than by anyone else's.

This module is the second opinion. It hands the circuit to **Stim** -- an independent
implementation, not a re-export of this package -- simulates it as a tableau, and asks
whether every generator of the frozen specification is a stabilizer of the state it
prepares. The specification is the same signed Pauli strings both engines were given, so
the third party decides the *same question*; it does not decide a smaller one.

What it checks, and what it does not
------------------------------------
* **The state, not the schedule.** Stim is asked whether the prepared state is the
  requested one, generator by generator, at any width. It says nothing about gate
  counts, depth or wall clock, and its verdict is therefore never quoted as a resource
  claim.
* **Clifford only.** A tableau simulator has no `t` gate, and this adapter refuses a
  parameterised rotation rather than rounding it: a `REFUSED` verdict is an abstention
  with a reason, never a pass. Every circuit in the corpus is Clifford, so the refusal
  path is exercised by tests rather than by the record.
* **The bridge is ours.** The circuit reaches Stim through this package's IR and the
  gate table below. A bug in the IR or in that table would be invisible to both
  referees; what the second opinion rules out is a bug *in the referee* -- the
  simulation, the phase bookkeeping, and the generator comparison. The record says so in
  its limitations instead of implying a neutrality it does not have.
* **Not a proof of optimality, or of anything about the target being the right target.**
  Both engines are handed the same frozen specification; whether that specification is
  the one a user wanted is not a question a referee can answer.

The verdict vocabulary matches the rest of the comparison: `VALID`, `INVALID`,
`UNAVAILABLE` (the tool is absent) and `REFUSED` (the circuit is outside what a tableau
simulator can decide). Only `VALID`/`INVALID` are verdicts about the circuit; the other
two carry a reason and are never counted as a loss.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import time
from typing import Any

from apedp_dyn.vqc.circuit import Circuit
from apedp_dyn.vqc.synthesis import TargetSpecification

__all__ = [
    "FOREIGN_REFEREES",
    "STIM_GATES",
    "ForeignUnavailable",
    "ForeignVerdict",
    "stim_verdict",
]

#: The IR's gate names, mapped to Stim's. The IR's control convention (first target is
#: the control) is Stim's for `CX`/`CCX`, and the table is the whole bridge: a name
#: that is not here is refused with the name in the reason, so an unmapped gate cannot
#: be silently dropped.
STIM_GATES: dict[str, str] = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "S_DAG",
    "cx": "CX",
    "cz": "CZ",
    "swap": "SWAP",
    "ccx": "CCX",
    "cswap": "CSWAP",
}

#: The third-party referees this package knows about, and what each can decide.
FOREIGN_REFEREES: dict[str, dict[str, Any]] = {
    "stim": {
        "package": "stim",
        "decides": "whether a Clifford preparation circuit puts the state in the requested stabilizer group",
        "does_not_decide": ("gate counts, depth, wall clock, or whether a non-Clifford "
                            "circuit prepares the state"),
        "note": ("tableau simulation of the circuit, then every generator of the frozen "
                 "specification evaluated as a Pauli observable. Independent of both "
                 "engines; the bridge from this package's IR to Stim's circuit is ours, "
                 "which the record states rather than hides"),
    },
}


class ForeignUnavailable(RuntimeError):
    """The third-party referee could not be run, with the reason a record can print."""


@dataclasses.dataclass(frozen=True)
class ForeignVerdict:
    """One third-party decision about one circuit, with what it was decided on."""

    referee: str
    version: str | None
    verdict: str
    reason: str
    qubits: int
    generators: int
    satisfied: int
    failing: tuple[dict[str, Any], ...]
    seconds: float

    @property
    def comparative(self) -> bool:
        """True only for a verdict that is about the circuit rather than about the run."""
        return self.verdict in ("VALID", "INVALID")

    def as_dict(self) -> dict[str, Any]:
        return {
            "referee": self.referee,
            "version": self.version,
            "verdict": self.verdict,
            "reason": self.reason,
            "qubits": self.qubits,
            "generators": self.generators,
            "satisfied": self.satisfied,
            "failing": list(self.failing),
            "seconds": self.seconds,
            "decides_the_circuit": self.comparative,
        }


def _stim_module() -> Any:
    try:
        return importlib.import_module("stim")
    except ImportError as error:
        raise ForeignUnavailable(
            f"stim is not importable ({type(error).__name__}: {error}); install it with "
            f"`pip install 'apedp[compare]'` to run the third-party referee") from error


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - import worked
        return None


def _stim_circuit(stim: Any, circuit: Circuit) -> Any:
    """Translate the IR into a Stim circuit, or raise with the operation that stopped us.

    Every refusal names the operation and the gate, because "this circuit cannot be
    decided by a tableau simulator" is only useful information if it says which gate.
    """
    text = stim.Circuit()
    for op in circuit.ops:
        if op.kind != "gate":
            raise ForeignUnavailable(
                f"a tableau simulator decides unitary preparation only; this circuit "
                f"contains a {op.kind} operation")
        if op.condition:
            raise ForeignUnavailable(
                f"'{op.name}' is classically conditioned; a tableau simulator is only "
                f"handed the unconditional preparation")
        name = STIM_GATES.get(op.name)
        if name is None:
            raise ForeignUnavailable(
                f"gate '{op.name}' is not a Clifford gate this referee can apply"
                + (f" (parameters {tuple(op.params)})" if op.params else ""))
        text.append(name, list(op.targets))
    return text


def stim_verdict(circuit: Circuit, target: TargetSpecification, *,
                 max_failing: int = 8) -> ForeignVerdict:
    """Stim's decision on whether `circuit` prepares `target`.

    The circuit is simulated as a stabilizer tableau and each generator of the
    specification is evaluated as a Pauli observable. A pure stabilizer state is the
    unique +1 eigenstate of `n` independent stabilizers, so `n` correct generators with
    the correct signs decide the state -- no amplitude is ever computed, which is what
    lets the same referee decide a 127-qubit routed circuit in a hundredth of a second.

    A missing Stim is `UNAVAILABLE`, an undecidable circuit is `REFUSED`, and only a
    simulated but wrong state is `INVALID`. The three are never conflated: an abstention
    counted as a failure would be exactly the contamination this module exists to avoid.
    """
    start = time.perf_counter()
    try:
        stim = _stim_module()
    except ForeignUnavailable as error:
        return ForeignVerdict("stim", None, "UNAVAILABLE", str(error), target.num_qubits,
                              len(target.stabilizers), 0, (), 0.0)
    version = _version("stim")
    if circuit.num_qubits != target.num_qubits:
        return ForeignVerdict(
            "stim", version, "REFUSED",
            (f"the circuit is on {circuit.num_qubits} qubits, the specification on "
             f"{target.num_qubits}"), target.num_qubits, len(target.stabilizers), 0, (),
            time.perf_counter() - start)
    try:
        text = _stim_circuit(stim, circuit)
    except ForeignUnavailable as error:
        return ForeignVerdict("stim", version, "REFUSED", str(error), target.num_qubits,
                             len(target.stabilizers), 0, (),
                             time.perf_counter() - start)

    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(circuit.num_qubits)
    simulator.do(text)
    satisfied = 0
    failing: list[dict[str, Any]] = []
    for index, (labels, phase) in enumerate(target.stabilizers):
        if int(phase) % 2:
            # A stabilizer with an imaginary phase is not a real-eigenvalue stabilizer
            # of a pure state; the specification validator refuses one, so reaching this
            # is a bug in the caller, not a verdict about the circuit.
            return ForeignVerdict(
                "stim", version, "REFUSED",
                f"generator {index} has phase {phase}, which is not ±1",
                target.num_qubits, len(target.stabilizers), satisfied, (),
                time.perf_counter() - start)
        sign = 1 if int(phase) % 4 == 0 else -1
        pauli = "".join("IXYZ"[int(label)] for label in labels)
        # The generator is `sign * P`; the state stabilizes it iff the expectation of
        # the *unsigned* Pauli `P` equals the sign. Measuring the signed string and
        # comparing it to the sign would compare it with itself.
        expectation = simulator.peek_observable_expectation(stim.PauliString("+" + pauli))
        if expectation == sign:
            satisfied += 1
        elif len(failing) < max_failing:
            failing.append({
                "generator": index,
                "pauli": pauli,
                "expected_expectation": sign,
                "observed_expectation": expectation,
            })
    valid = satisfied == len(target.stabilizers)
    return ForeignVerdict(
        referee="stim",
        version=version,
        verdict="VALID" if valid else "INVALID",
        reason=("every generator of the specification stabilizes the prepared state"
                if valid else
                f"{len(target.stabilizers) - satisfied} of {len(target.stabilizers)} "
                f"generators do not stabilize the prepared state"),
        qubits=target.num_qubits,
        generators=len(target.stabilizers),
        satisfied=satisfied,
        failing=tuple(failing),
        seconds=time.perf_counter() - start,
    )
