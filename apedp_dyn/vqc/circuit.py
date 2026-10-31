"""The VQC circuit IR: one versioned, canonical, hashable circuit language.

Status: experimental

Everything in APEDP-VQC -- discovery, optimisation, simulation, logical-qubit
experiments, adjudication -- has to agree on what a circuit *is*. This module is
that agreement, and it is deliberately small:

* qubits, classical bits, and declared ancillas with an expected cleanup;
* gates from one named table (:mod:`apedp_dyn.vqc.gates`), with the control
  convention ``cx(control, target)`` / ``ccx(c0, c1, target)`` / ``cswap(control,
  first, second)``;
* measurement with the reset semantics the competition's ``MR`` has (the qubit
  is cleared and the outcome is written to a classical bit);
* classical feedforward as a conjunction of bit tests on any operation;
* stochastic noise as a first-class operation, so a noisy run is the same
  circuit with noise ops in it rather than a different code path;
* metadata, a schema version, and a canonical serialisation whose SHA-256 is the
  circuit's identity.

The hash is a *reproducibility* device, as in
:mod:`apedp_dyn.quantum_compilation.contracts`: it fixes which circuit a receipt
is about. It is not a proof of correctness, and nothing here claims one.

Round-tripping is exact: ``Circuit.from_json(circuit.to_json()).sha256() ==
circuit.sha256()``, and executing the reconstructed circuit gives the same
statistics as executing the original, because both go through the same backend.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from apedp_dyn.vqc.gates import GATE_SPECS

__all__ = [
    "NOISE_KINDS",
    "SCHEMA_VERSION",
    "Circuit",
    "CircuitValidationError",
    "Condition",
    "Gate",
    "Measure",
    "Noise",
    "Op",
    "Reset",
]

SCHEMA_VERSION = 1

#: The noise kinds the IR may name. These are single-qubit stochastic Pauli (or
#: Pauli-mixture) errors -- a *trajectory* model: the backend samples the error
#: for each shot instead of carrying a density matrix.
NOISE_KINDS = ("bit_flip", "phase_flip", "depolarizing")

#: A conjunction of classical bit tests: every pair must hold for the op to run.
Condition = tuple[tuple[int, bool], ...]


class CircuitValidationError(ValueError):
    """An operation that cannot be executed as written."""


@dataclasses.dataclass(frozen=True)
class Gate:
    """A named unitary on `targets`, optionally classically conditioned."""

    name: str
    targets: tuple[int, ...]
    params: tuple[float, ...] = ()
    condition: Condition = ()

    kind = "gate"


@dataclasses.dataclass(frozen=True)
class Measure:
    """Measure `qubit` in the Z basis, reset it, and write the outcome to `bit`."""

    qubit: int
    bit: int
    condition: Condition = ()

    kind = "measure"


@dataclasses.dataclass(frozen=True)
class Reset:
    """Return `qubit` to |0> without reporting an outcome.

    Not the same operation as `Measure`: a measurement branch *reports* what it
    found, a reset discards it. The backend implements it as a projection onto
    |0> followed by renormalisation, which is the reset trace-preserving map.
    """

    qubit: int
    condition: Condition = ()

    kind = "reset"


@dataclasses.dataclass(frozen=True)
class Noise:
    """A stochastic single-qubit error, sampled per shot by the backend."""

    qubit: int
    kind: str
    probability: float
    condition: Condition = ()

    #: The dataclass field `kind` shadows the op tag, so the tag is spelled out.
    op = "noise"


Op = Gate | Measure | Reset | Noise


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _condition_to_json(condition: Condition) -> list[dict[str, Any]]:
    return [{"bit": int(bit), "value": bool(value)} for bit, value in condition]


def _condition_from_json(raw: Any) -> Condition:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise CircuitValidationError(f"condition must be a list, got {type(raw).__name__}")
    pairs = []
    for entry in raw:
        if not isinstance(entry, dict) or "bit" not in entry or "value" not in entry:
            raise CircuitValidationError(f"bad condition entry {entry!r}")
        pairs.append((int(entry["bit"]), bool(entry["value"])))
    return tuple(pairs)


@dataclasses.dataclass(frozen=True)
class Circuit:
    """A versioned sequence of operations over numbered qubits and bits."""

    ops: tuple[Op, ...]
    num_qubits: int
    num_bits: int = 0
    ancillas: tuple[int, ...] = ()
    metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        """Refuse anything a backend would have to guess about.

        Raises :class:`CircuitValidationError`. Validating is idempotent and is
        called by every constructor path (`from_json`) and by the backends, so a
        circuit that executes has been checked.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise CircuitValidationError(
                f"schema version {self.schema_version} is not {SCHEMA_VERSION}")
        if self.num_qubits < 1:
            raise CircuitValidationError(f"num_qubits must be positive, got {self.num_qubits}")
        if self.num_bits < 0:
            raise CircuitValidationError(f"num_bits must be non-negative, got {self.num_bits}")
        if len(set(self.ancillas)) != len(self.ancillas):
            raise CircuitValidationError(f"duplicate ancillas in {self.ancillas}")
        for qubit in self.ancillas:
            if not 0 <= qubit < self.num_qubits:
                raise CircuitValidationError(f"ancilla {qubit} out of range")

        written: set[int] = set()
        for index, op in enumerate(self.ops):
            for bit, _value in op.condition:
                if not 0 <= bit < self.num_bits:
                    raise CircuitValidationError(
                        f"op {index}: condition bit {bit} out of range")
                if bit not in written:
                    raise CircuitValidationError(
                        f"op {index}: condition bit {bit} is read before it is written")
            if isinstance(op, Measure):
                if not 0 <= op.bit < self.num_bits:
                    raise CircuitValidationError(
                        f"op {index}: measurement bit {op.bit} out of range")
                if not 0 <= op.qubit < self.num_qubits:
                    raise CircuitValidationError(f"op {index}: qubit {op.qubit} out of range")
                written.add(op.bit)
            elif isinstance(op, Reset):
                if not 0 <= op.qubit < self.num_qubits:
                    raise CircuitValidationError(f"op {index}: qubit {op.qubit} out of range")
            elif isinstance(op, Noise):
                if op.kind not in NOISE_KINDS:
                    raise CircuitValidationError(
                        f"op {index}: noise kind {op.kind!r} is not one of {NOISE_KINDS}")
                if not 0.0 <= op.probability <= 1.0:
                    raise CircuitValidationError(
                        f"op {index}: probability {op.probability} is not in [0, 1]")
                if not 0 <= op.qubit < self.num_qubits:
                    raise CircuitValidationError(f"op {index}: qubit {op.qubit} out of range")
            elif isinstance(op, Gate):
                spec = GATE_SPECS.get(op.name)
                if spec is None:
                    raise CircuitValidationError(
                        f"op {index}: unknown gate {op.name!r}")
                if len(op.targets) != spec.arity:
                    raise CircuitValidationError(
                        f"op {index}: {op.name!r} takes {spec.arity} qubits, got "
                        f"{len(op.targets)}")
                if len(set(op.targets)) != len(op.targets):
                    raise CircuitValidationError(
                        f"op {index}: {op.name!r} repeats a qubit in {op.targets}")
                for target in op.targets:
                    if not 0 <= target < self.num_qubits:
                        raise CircuitValidationError(
                            f"op {index}: qubit {target} out of range")
                if len(op.params) != spec.num_params:
                    raise CircuitValidationError(
                        f"op {index}: {op.name!r} takes {spec.num_params} parameters, got "
                        f"{len(op.params)}")
                for value in op.params:
                    if not math.isfinite(float(value)):
                        raise CircuitValidationError(
                            f"op {index}: parameter {value!r} is not finite")
            else:  # pragma: no cover - the union is closed
                raise CircuitValidationError(f"op {index}: unknown operation {op!r}")

    # ------------------------------------------------------------ resources
    def qubits_touched(self, op: Op) -> tuple[int, ...]:
        if isinstance(op, (Measure, Reset, Noise)):
            return (op.qubit,)
        return tuple(op.targets)

    def depth(self) -> int:
        """Circuit depth: qubits accumulate one layer per operation they carry."""
        last: dict[int, int] = {}
        depth = 0
        for op in self.ops:
            layer = 0
            for qubit in self.qubits_touched(op):
                layer = max(layer, last.get(qubit, 0))
            layer += 1
            for qubit in self.qubits_touched(op):
                last[qubit] = layer
            depth = max(depth, layer)
        return depth

    def resources(self) -> dict[str, Any]:
        """What the circuit costs, counted rather than estimated."""
        gates = Counter(op.name for op in self.ops if isinstance(op, Gate))
        # Tolerate an unknown name: `resources()` is also called on a circuit that
        # failed validation, and a receipt for an invalid circuit must still be
        # printable.
        two_qubit = sum(count for name, count in gates.items()
                        if name in GATE_SPECS and GATE_SPECS[name].arity == 2)
        return {
            "qubits": self.num_qubits,
            "bits": self.num_bits,
            "ancillas": len(self.ancillas),
            "operations": len(self.ops),
            "gates": dict(sorted(gates.items())),
            "two_qubit_gates": two_qubit,
            "measurements": sum(1 for op in self.ops if isinstance(op, Measure)),
            "resets": sum(1 for op in self.ops if isinstance(op, Reset)),
            "noise_operations": sum(1 for op in self.ops if isinstance(op, Noise)),
            "depth": self.depth(),
        }

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        ops: list[dict[str, Any]] = []
        for op in self.ops:
            if isinstance(op, Gate):
                ops.append({"op": "gate", "name": op.name,
                            "targets": [int(t) for t in op.targets],
                            "params": [float(p) for p in op.params],
                            "condition": _condition_to_json(op.condition)})
            elif isinstance(op, Measure):
                ops.append({"op": "measure", "qubit": int(op.qubit), "bit": int(op.bit),
                            "condition": _condition_to_json(op.condition)})
            elif isinstance(op, Reset):
                ops.append({"op": "reset", "qubit": int(op.qubit),
                            "condition": _condition_to_json(op.condition)})
            else:
                ops.append({"op": "noise", "kind": op.kind, "qubit": int(op.qubit),
                            "probability": float(op.probability),
                            "condition": _condition_to_json(op.condition)})
        return {
            "schema_version": self.schema_version,
            "num_qubits": self.num_qubits,
            "num_bits": self.num_bits,
            "ancillas": [int(a) for a in self.ancillas],
            "metadata": {str(key): str(value)
                         for key, value in sorted(self.metadata.items())},
            "ops": ops,
        }

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, no whitespace, stable for a given circuit."""
        return _canonical_json(self.to_dict())

    def sha256(self) -> str:
        """The circuit's identity, over its canonical serialisation."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Circuit:
        if not isinstance(payload, Mapping):
            raise CircuitValidationError(f"expected an object, got {type(payload).__name__}")
        ops: list[Op] = []
        for entry in payload.get("ops", ()):
            if not isinstance(entry, Mapping):
                raise CircuitValidationError(f"operation {entry!r} is not an object")
            tag = entry.get("op")
            condition = _condition_from_json(entry.get("condition"))
            if tag == "gate":
                ops.append(Gate(name=str(entry["name"]),
                                targets=tuple(int(t) for t in entry["targets"]),
                                params=tuple(float(p) for p in entry.get("params", ())),
                                condition=condition))
            elif tag == "measure":
                ops.append(Measure(qubit=int(entry["qubit"]), bit=int(entry["bit"]),
                                   condition=condition))
            elif tag == "reset":
                ops.append(Reset(qubit=int(entry["qubit"]), condition=condition))
            elif tag == "noise":
                ops.append(Noise(qubit=int(entry["qubit"]), kind=str(entry["kind"]),
                                 probability=float(entry["probability"]),
                                 condition=condition))
            else:
                raise CircuitValidationError(f"unknown operation tag {tag!r}")
        circuit = cls(ops=tuple(ops),
                      num_qubits=int(payload["num_qubits"]),
                      num_bits=int(payload.get("num_bits", 0)),
                      ancillas=tuple(int(a) for a in payload.get("ancillas", ())),
                      metadata={str(k): str(v) for k, v in
                                dict(payload.get("metadata", {})).items()},
                      schema_version=int(payload.get("schema_version", SCHEMA_VERSION)))
        circuit.validate()
        return circuit

    @classmethod
    def from_json(cls, text: str) -> Circuit:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise CircuitValidationError(f"not JSON: {error}") from error
        return cls.from_dict(payload)

    # ------------------------------------------------------------- builders
    def with_metadata(self, **values: str) -> Circuit:
        merged = {**self.metadata, **{key: str(value) for key, value in values.items()}}
        return dataclasses.replace(self, metadata=merged)

    def with_ops(self, *ops: Op) -> Circuit:
        candidate = dataclasses.replace(self, ops=self.ops + tuple(ops))
        candidate.validate()
        return candidate

    def execution_order(self) -> Iterable[tuple[int, Op]]:
        """`(index, op)` pairs, in order, for anything that reports per-op evidence."""
        return enumerate(self.ops)
