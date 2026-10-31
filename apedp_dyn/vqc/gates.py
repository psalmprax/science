"""The gate table: one definition of each unitary, shared by every backend.

Status: experimental

The VQC circuit IR (`apedp_dyn/vqc/circuit.py`) names gates; this module decides
what a name means. Both the statevector backend and the density-matrix channel
simulator apply gates through here, so a circuit cannot mean two things on two
backends.

Gate arguments follow the IR's convention: the *first* targets of a controlled
gate are its controls.

    cx     (control, target)
    cz     (control, target)
    swap   (first, second)
    ccx    (control, control, target)
    cswap  (control, first, second)
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = [
    "GATE_SPECS",
    "GateSpec",
    "UnknownGateError",
    "apply_gate",
    "matrix_for",
]

def _dagger(matrix: np.ndarray) -> np.ndarray:
    return np.conjugate(np.transpose(matrix))


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0],
                     [0.0, np.exp(0.5j * theta)]], dtype=np.complex128)


_FIXED = {
    "h": np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    "s": np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128),
    "t": np.array([[1.0, 0.0], [0.0, np.exp(0.25j * math.pi)]], dtype=np.complex128),
}
_FIXED["sdg"] = _dagger(_FIXED["s"])
_FIXED["tdg"] = _dagger(_FIXED["t"])


def _rotation(axis: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    if axis == "x":
        return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    return _rz(theta)


@dataclasses.dataclass(frozen=True)
class GateSpec:
    """What one gate name requires: how many qubits, and how many parameters."""

    arity: int
    num_params: int = 0
    #: For parameterised rotations, the axis `_rotation` builds; empty otherwise.
    symbols: tuple[str, ...] = ()


def _spec(arity: int, num_params: int = 0, symbols: tuple[str, ...] = ()) -> GateSpec:
    return GateSpec(arity=arity, num_params=num_params, symbols=symbols)


GATE_SPECS: dict[str, GateSpec] = {
    **{name: _spec(1) for name in _FIXED},
    "rx": _spec(1, 1, ("x",)),
    "ry": _spec(1, 1, ("y",)),
    "rz": _spec(1, 1, ("z",)),
    # A phase gate is `rz` up to a global phase the statevector cannot see, and for a
    # controlled-`phase` it would matter -- so it is defined by its own matrix.
    "phase": GateSpec(arity=1, num_params=1, symbols=("phase",)),
    "cx": _spec(2),
    "cz": _spec(2),
    "swap": _spec(2),
    "ccx": _spec(3),
    "cswap": _spec(3),
}


class UnknownGateError(KeyError):
    """A gate name that is not in `GATE_SPECS`."""


def matrix_for(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    """The 2x2 unitary of a one-qubit gate."""
    spec = GATE_SPECS.get(name)
    if spec is None:
        raise UnknownGateError(name)
    if spec.arity != 1:
        raise ValueError(f"{name!r} is a {spec.arity}-qubit gate; it has no 2x2 matrix")
    if spec.num_params:
        if len(params) != spec.num_params:
            raise ValueError(f"expected {spec.num_params} parameters, got {len(params)}")
        if spec.symbols == ("phase",):
            return np.array([[1.0, 0.0], [0.0, np.exp(1j * float(params[0]))]],
                            dtype=np.complex128)
        return _rotation(spec.symbols[0], float(params[0]))
    return _FIXED[name]


def apply_gate(sim, name: str, targets: tuple[int, ...],
               params: tuple[float, ...] = ()) -> None:
    """Apply `name` to a `StatevectorSimulator`, by the IR's control convention."""
    spec = GATE_SPECS.get(name)
    if spec is None:
        raise UnknownGateError(name)
    if len(targets) != spec.arity:
        raise ValueError(f"{name!r} takes {spec.arity} qubits, got {len(targets)}")
    if spec.arity == 1:
        sim.apply_matrix(targets[0], matrix_for(name, params))
    elif spec.arity == 2:
        if name == "cx":
            sim.cx(*targets)
        elif name == "cz":
            sim.cz(*targets)
        elif name == "swap":
            sim.swap(*targets)
        else:  # pragma: no cover - GATE_SPECS has no other two-qubit gate
            raise UnknownGateError(name)
    else:
        if name == "ccx":
            sim.ccx(*targets)
        elif name == "cswap":
            sim.cswap(*targets)
        else:  # pragma: no cover
            raise UnknownGateError(name)
