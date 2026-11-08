"""OpenQASM 2 import and export, with the parts it cannot carry refused loudly.

Status: experimental

Interchange is where a circuit representation either earns its keep or quietly
lies. OpenQASM 2 is the one format every other tool speaks, so this module
supports the subset that maps exactly onto the VQC IR and raises
:class:`QasmUnsupportedError` for everything else -- never an approximation:

* gates: `h x y z s sdg t tdg rx ry rz u1 cx cz swap ccx cswap`, plus `id` and
  `barrier` on import (`id` is dropped, `barrier` is a scheduling hint and has no
  IR meaning);
* `measure q[i] -> c[j]` and `reset q[i]`;
* `if(c==k) <gate|measure|reset>`, but **only** when the condition tests every bit
  of the register. OpenQASM 2 compares the whole classical register to an integer,
  while the IR's condition is a conjunction of per-bit tests; a condition on a
  subset of bits has no faithful QASM 2 spelling (`if (c[1] == 1)` is QASM 3), so
  exporting one is an error rather than a changed circuit;
* noise operations have no QASM 2 construct and are refused.

Metadata and the declared ancillas have no QASM syntax either, so they travel as
`// vqc-...` header comments. That is a convention of this module, not a claim
about the standard -- and it is what lets a QASM round-trip preserve the circuit's
hash instead of silently dropping the ancilla contract.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any

from apedp_dyn.vqc.circuit import (
    SCHEMA_VERSION,
    Circuit,
    CircuitValidationError,
    Gate,
    Measure,
    Noise,
    Reset,
)

__all__ = ["QasmUnsupportedError", "from_qasm", "to_qasm"]

_SINGLE_QUBIT = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz")
_MULTI_QUBIT = {"cx": 2, "cz": 2, "swap": 2, "ccx": 3, "cswap": 3}
#: `phase` is `u1` in QASM terms; the export name differs from the IR name.
_QASM_NAMES = {"phase": "u1"}
_IR_NAMES = {"u1": "phase"}
_QELIB_ONLY = {"id", "barrier"}


class QasmUnsupportedError(ValueError):
    """A circuit (or source) that OpenQASM 2 cannot express faithfully."""


# --------------------------------------------------------------------- export
def _condition_constant(condition: Any) -> int:
    bits = {bit for bit, _value in condition}
    if not bits:
        return 0
    value = 0
    for bit, expected in condition:
        if expected:
            value |= 1 << bit
    return value


def _format_condition(condition: Any, num_bits: int) -> str:
    tested = {bit for bit, _value in condition}
    if tested != set(range(num_bits)):
        raise QasmUnsupportedError(
            f"condition on bits {sorted(tested)} of {num_bits} cannot be written in "
            f"OpenQASM 2: `if(c==k)` compares the whole register, and a per-bit test "
            f"(`if (c[{sorted(tested)[0]}] == ..)`) is QASM 3 syntax")
    return f"if(c=={_condition_constant(condition)}) "


def to_qasm(circuit: Circuit) -> str:
    """Serialise `circuit` as OpenQASM 2, refusing what the format cannot carry."""
    circuit.validate()
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', ""]
    lines.append(f"// vqc-schema-version: {SCHEMA_VERSION}")
    if circuit.ancillas:
        lines.append("// vqc-ancillas: " + ",".join(str(a) for a in circuit.ancillas))
    for key, value in sorted(circuit.metadata.items()):
        lines.append(f"// vqc-metadata: {key}={value}")
    lines.append("")
    lines.append(f"qreg q[{circuit.num_qubits}];")
    if circuit.num_bits:
        lines.append(f"creg c[{circuit.num_bits}];")
    lines.append("")

    for op in circuit.ops:
        prefix = _format_condition(op.condition, circuit.num_bits) if op.condition else ""
        if isinstance(op, Noise):
            raise QasmUnsupportedError(
                f"noise operation ({op.kind} on q[{op.qubit}]) has no OpenQASM 2 "
                f"construct; export the ideal circuit and describe the noise in the "
                f"receipt instead")
        if isinstance(op, Measure):
            lines.append(f"{prefix}measure q[{op.qubit}] -> c[{op.bit}];")
        elif isinstance(op, Reset):
            lines.append(f"{prefix}reset q[{op.qubit}];")
        else:
            name = _QASM_NAMES.get(op.name, op.name)
            arguments = ""
            if op.params:
                arguments = "(" + ", ".join(repr(float(p)) for p in op.params) + ")"
            targets = ", ".join(f"q[{target}]" for target in op.targets)
            lines.append(f"{prefix}{name}{arguments} {targets};")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- import
_REGISTER = re.compile(r"^(qreg|creg)\s+(\w+)\[(\d+)\];$")
_OPERATION = re.compile(r"^(?:if\s*\(\s*(c)\s*==\s*(\d+)\s*\)\s*)?([a-z][a-z0-9_]*)(.*)$")
_CALL = re.compile(r"^\s*(?:\(([^)]*)\))?\s*(.+)$")
_TARGET = re.compile(r"\w+\[(\d+)\]")
_PARAMETER_SAFE = re.compile(r"^[0-9eE+\-*/().\s pi]+$")


def _parse_parameter(text: str) -> float:
    """A QASM parameter: an arithmetic expression over literals and `pi`."""
    expression = text.strip()
    if not expression or not _PARAMETER_SAFE.fullmatch(expression):
        raise QasmUnsupportedError(f"unsupported parameter expression {text!r}")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
                             ast.Name, ast.Load)):
            continue
        raise QasmUnsupportedError(f"unsupported parameter expression {text!r}")
    try:
        return float(eval(compile(tree, "<qasm>", "eval"),  # noqa: S307 - whitelisted above
                          {"__builtins__": {}}, {"pi": math.pi}))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise QasmUnsupportedError(f"bad parameter expression {text!r}: {error}") from error


def _parse_arguments(text: str) -> tuple[tuple[float, ...], str]:
    """Split `(params) targets` into numeric parameters and the remaining text."""
    match = _CALL.match(text)
    if match is None:  # pragma: no cover - regex always matches
        raise QasmUnsupportedError(f"cannot parse {text!r}")
    parameters, targets = match.groups()
    if parameters is None:
        return (), targets.strip()
    return tuple(_parse_parameter(part) for part in parameters.split(",")), targets.strip()


def from_qasm(text: str) -> Circuit:
    """Parse an OpenQASM 2 program into the IR, refusing anything not exactly mappable."""
    registers: dict[str, tuple[str, int]] = {}
    offsets = {"q": 0, "c": 0}
    ops: list[Any] = []
    ancillas: tuple[int, ...] = ()
    metadata: dict[str, str] = {}
    seen_header = False
    num_bits = 0

    body = []
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        comment = raw.split("//", 1)[1].strip() if "//" in raw else ""
        if comment:
            if comment.startswith("vqc-ancillas:"):
                payload = comment.split(":", 1)[1].strip()
                ancillas = tuple(int(part) for part in payload.split(",") if part.strip())
            elif comment.startswith("vqc-metadata:"):
                key, _, value = comment.split(":", 1)[1].strip().partition("=")
                metadata[key.strip()] = value.strip()
        if not line:
            continue
        body.append(line)

    for line in body:
        if line == "OPENQASM 2.0;":
            seen_header = True
            continue
        if line.startswith("include "):
            continue
        register = _REGISTER.match(line)
        if register is not None:
            kind, name, size = register.group(1), register.group(2), int(register.group(3))
            if kind == "qreg":
                registers[name] = ("q", offsets["q"])
                offsets["q"] += size
            else:
                if name != "c":
                    raise QasmUnsupportedError(
                        f"classical register {name!r}: the IR has one bit namespace, "
                        f"declared `c`")
                registers[name] = ("c", 0)
                offsets["c"] = size
                num_bits = size
            continue
        if line.startswith("gate "):
            raise QasmUnsupportedError("user-defined `gate` declarations are not supported")

        statement = line[:-1] if line.endswith(";") else line
        operation = _OPERATION.match(statement)
        if operation is None:
            raise QasmUnsupportedError(f"cannot parse line {line!r}")
        _register_name, condition_value, name, rest = operation.groups()
        parameters, targets = _parse_arguments(rest)

        target_indices = [int(value) for value in _TARGET.findall(targets)]
        if name in ("barrier",):
            continue
        if name == "id":
            continue

        condition: tuple[tuple[int, bool], ...] = ()
        if condition_value is not None:
            value = int(condition_value)
            if num_bits == 0:
                raise QasmUnsupportedError("conditional operation without a classical register")
            if value >= 1 << num_bits:
                raise QasmUnsupportedError(
                    f"condition c=={value} does not fit in {num_bits} bits")
            condition = tuple((bit, bool((value >> bit) & 1)) for bit in range(num_bits))

        if name == "measure":
            if "->" not in targets:
                raise QasmUnsupportedError(f"malformed measure {line!r}")
            source, _, destination = targets.partition("->")
            qubits = [int(v) for v in _TARGET.findall(source)]
            bits = [int(v) for v in _TARGET.findall(destination)]
            if len(qubits) != 1 or len(bits) != 1:
                raise QasmUnsupportedError("measure takes one qubit and one bit")
            ops.append(Measure(qubit=qubits[0], bit=bits[0], condition=condition))
        elif name == "reset":
            if len(target_indices) != 1:
                raise QasmUnsupportedError("reset takes one qubit")
            ops.append(Reset(qubit=target_indices[0], condition=condition))
        else:
            ir_name = _IR_NAMES.get(name, name)
            if name in _QELIB_ONLY:
                raise QasmUnsupportedError(f"{name!r} has no IR meaning")
            if ir_name in _MULTI_QUBIT:
                expected = _MULTI_QUBIT[ir_name]
            elif ir_name in _SINGLE_QUBIT or ir_name == "phase":
                expected = 1
            else:
                raise QasmUnsupportedError(
                    f"gate {name!r} is not in the supported QASM 2 subset "
                    f"({', '.join(sorted(_SINGLE_QUBIT))}, u1, "
                    f"{', '.join(sorted(_MULTI_QUBIT))})")
            if len(target_indices) != expected:
                raise QasmUnsupportedError(
                    f"{name!r} takes {expected} qubits, found {len(target_indices)}")
            required_params = 1 if (ir_name in ("rx", "ry", "rz", "phase")) else 0
            if len(parameters) != required_params:
                raise QasmUnsupportedError(
                    f"{name!r} takes {required_params} parameter(s), found {len(parameters)}")
            ops.append(Gate(name=ir_name, targets=tuple(target_indices),
                            params=parameters, condition=condition))

    if not seen_header:
        raise QasmUnsupportedError("missing `OPENQASM 2.0;` header")
    if offsets["q"] == 0:
        raise QasmUnsupportedError("no quantum register declared")

    circuit = Circuit(ops=tuple(ops), num_qubits=offsets["q"], num_bits=num_bits,
                      ancillas=ancillas, metadata=metadata)
    try:
        circuit.validate()
    except CircuitValidationError as error:
        # The IR is stricter than QASM in one place -- it requires a classical bit
        # to be written before a condition reads it, while QASM leaves undeclared
        # bits at zero -- so a program that leans on that difference is refused
        # here rather than executed with different semantics.
        raise QasmUnsupportedError(f"imported program is not a valid circuit: {error}") from error
    return circuit
