"""Backends: one interface, so a circuit is the same circuit everywhere.

Status: experimental

A backend takes a validated :class:`~apedp_dyn.vqc.circuit.Circuit` and returns a
:class:`RunResult`: per-shot classical bits, the counts derived from them, and the
residual excitation left on each declared ancilla. Backends are where the
platform's claims have to be earned, so this module contains one real
implementation and no stubs:

* :class:`StatevectorBackend` -- dense exact amplitudes through
  :class:`apedp_dyn.quantum_compilation.statevector_sim.StatevectorSimulator`,
  which already implements the `MR` semantics (measure, collapse, reset) and
  classical bits. Noise operations are executed as **quantum trajectories**: a
  Pauli error is sampled per shot with the stated probability and applied to that
  shot's state. That is exact for Pauli noise in the ensemble sense (the average
  over shots is the channel) and it is *sampled*, not exact, per shot -- which is
  why every result carries its shot count and why statistical claims in this
  package are accompanied by a standard error.

A backend that cannot execute something must refuse it. Nothing here silently
ignores an operation: an unknown gate name is a validation error, and the
statevector backend raises on an unphysical shot rather than returning a
zero-probability branch.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from apedp_dyn.quantum_compilation.statevector_sim import StatevectorSimulator
from apedp_dyn.vqc.circuit import Circuit, CircuitValidationError, Gate, Measure, Noise, Reset
from apedp_dyn.vqc.gates import apply_gate

__all__ = [
    "Backend",
    "RunResult",
    "StatevectorBackend",
    "bitstring",
    "distribution",
    "sample_bits",
]


def bitstring(bits: Sequence[bool]) -> str:
    """Classical bits as a string, bit 0 first."""
    return "".join("1" if bit else "0" for bit in bits)


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What one backend run produced, with the evidence needed to check it."""

    backend: str
    shots: int
    seed: int | None
    counts: dict[str, int]
    samples: tuple[tuple[bool, ...], ...]
    #: Mean probability that each declared ancilla is in |1> at the end of a shot.
    ancilla_excitation: dict[int, float]
    final_states: tuple[np.ndarray, ...] | None = None
    #: ``"python"`` or ``"rust"``: which implementation executed the shots.
    #: A result can therefore prove its own provenance instead of being trusted
    #: about it when a report labels it native.
    engine: str = "python"

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "shots": self.shots,
            "seed": self.seed,
            "counts": dict(sorted(self.counts.items())),
            "ancilla_excitation": {str(k): v for k, v in sorted(self.ancilla_excitation.items())},
        }


@runtime_checkable
class Backend(Protocol):
    """The interface every execution backend implements."""

    name: str

    def run(self, circuit: Circuit, *, shots: int, seed: int | None = None,
            keep_states: bool = False) -> RunResult:  # pragma: no cover - protocol
        ...


def distribution(result: RunResult) -> dict[str, float]:
    """Empirical frequencies over the classical bit strings of a run."""
    total = sum(result.counts.values())
    if not total:
        raise ValueError("run has no shots")
    return {key: count / total for key, count in sorted(result.counts.items())}


def sample_bits(circuit: Circuit, *, shots: int, seed: int | None = None,
                backend: Backend | None = None) -> RunResult:
    """Run `circuit` on `backend` (default: statevector) and return the result."""
    return (backend or StatevectorBackend()).run(circuit, shots=shots, seed=seed)


def _condition_holds(condition: Any, bits: dict[int, bool]) -> bool:
    return all(bits.get(bit, False) is value for bit, value in condition)


def _apply_noise(sim: StatevectorSimulator, op: Noise, rng: np.random.Generator) -> None:
    """One sampled Pauli trajectory of a single-qubit stochastic channel."""
    if rng.random() >= op.probability:
        return
    if op.kind == "bit_flip":
        sim.x(op.qubit)
    elif op.kind == "phase_flip":
        sim.z(op.qubit)
    elif op.kind == "depolarizing":
        choice = int(rng.integers(0, 3))
        if choice == 0:
            sim.x(op.qubit)
        elif choice == 1:
            sim.z(op.qubit)
        else:
            # `Y` up to the global phase that `X` then `Z` introduces: `XZ = -iY`,
            # and a global phase is not observable in a trajectory.
            sim.x(op.qubit)
            sim.z(op.qubit)
    else:  # pragma: no cover - validation rejects other kinds
        raise CircuitValidationError(f"unknown noise kind {op.kind!r}")


class StatevectorBackend:
    """Exact single-trajectory simulation, one dense state per shot."""

    name = "statevector"

    def run(self, circuit: Circuit, *, shots: int, seed: int | None = None,
            keep_states: bool = False) -> RunResult:
        circuit.validate()
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}")
        rng = np.random.default_rng(seed)
        counts: Counter[str] = Counter()
        samples: list[tuple[bool, ...]] = []
        excitation = dict.fromkeys(circuit.ancillas, 0.0)
        states: list[np.ndarray] | None = [] if keep_states else None

        for _shot in range(shots):
            sim = StatevectorSimulator(circuit.num_qubits)
            for op in circuit.ops:
                if op.condition and not _condition_holds(op.condition, sim.bits):
                    continue
                if isinstance(op, Gate):
                    apply_gate(sim, op.name, op.targets, op.params)
                elif isinstance(op, Measure):
                    outcome = sim.measure_and_reset(op.qubit, rng=rng)
                    sim.bits[op.bit] = outcome
                elif isinstance(op, Reset):
                    # `measure_and_reset` already collapses and returns the qubit to
                    # |0>; reset differs from `Measure` only in discarding the
                    # outcome, so nothing else is applied.
                    sim.measure_and_reset(op.qubit, rng=rng)
                else:
                    _apply_noise(sim, op, rng)
            bits = tuple(bool(sim.bits.get(index, False)) for index in range(circuit.num_bits))
            counts[bitstring(bits)] += 1
            samples.append(bits)
            for ancilla in circuit.ancillas:
                excitation[ancilla] += sim.probabilities(ancilla)[1]
            if states is not None:
                states.append(sim.state.copy())

        return RunResult(
            backend=self.name,
            shots=shots,
            seed=seed,
            counts=dict(counts),
            samples=tuple(samples),
            ancilla_excitation={key: value / shots for key, value in excitation.items()},
            final_states=tuple(states) if states is not None else None,
        )
