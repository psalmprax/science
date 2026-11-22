"""The competition's circuit text, read strictly.

Status: experimental

A circuit of the Kaggle competition "Google Verifiable Quantum Circuit
Compilation" is a list of instructions for a simulator of reversible circuits
that tracks the phase. Its text form (``.kmx``) has one instruction a line, and
the binary form (``.kmb``) is the same list: converting one into the other and
back gives the same bytes. A model that is to improve a circuit can therefore
be handed its lines and asked for lines.

Lines written by a model are data from an untrusted source on their way to a
parser written in C++. This module is what stands between the two. It reads a
line only if it is one of the instructions below, written exactly as the
simulator writes it, and writes it out again itself; anything else is refused
with the line number and the reason. It imports nothing but the standard
library and never loads the simulator.

    X q7            Z q7            R q7             one qubit
    CX q1 q2        CZ q1 q2        SWAP q1 q2       two
    CCX q1 q2 q3    CCZ q1 q2 q3    CSWAP q1 q2 q3   three: the gates that cost
    Z_POW q7 0.125                                   a rotation, in half turns
    HMR q7 b3                                        measure across, reset, keep the outcome in a bit
    NEG                                              flip the global phase
    BIT_INVERT b3   BIT_STORE0 b3   BIT_STORE1 b3    classical bits
    PUSH_CONDITION if b3 ... POP_CONDITION           what lies between happens only if the bit is 1
    INFO_TAG "text"                                  a note; no effect
    ... if b3                                        any gate or bit instruction, done only if the bit is 1

**What is left out on purpose.** ``REGISTER`` and ``APPEND_TO_REGISTER`` say
which qubits are the input: they are written by the caller, never read from a
model. ``DEBUG_PRINT`` prints. ``MR`` exists only in a patched build of the
simulator, not in the one the competition ships.

**What this does not check.** Whether the circuit computes anything. That is
the verifier's job, and only the verifier's.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

MAX_LINES = 400_000
MAX_CHARS = 16_000_000
MAX_QUBIT = 4_096                 # the best circuits use under a hundred
MAX_BIT = 1_000_000
MAX_TAG = 80
MAX_ANGLE_DIGITS = 200            # the simulator keeps 128 bits of an angle

ONE_QUBIT = frozenset({"X", "Z", "R"})
TWO_QUBITS = frozenset({"CX", "CZ", "SWAP"})
THREE_QUBITS = frozenset({"CCX", "CCZ", "CSWAP"})
TOFFOLI_LIKE = THREE_QUBITS       # what the competition's cost counts, with the rotations
BIT_WRITES = frozenset({"BIT_INVERT", "BIT_STORE0", "BIT_STORE1"})
_ARITY = {**dict.fromkeys(ONE_QUBIT, 1), **dict.fromkeys(TWO_QUBITS, 2), **dict.fromkeys(THREE_QUBITS, 3),
          "Z_POW": 1, "HMR": 1}
INSTRUCTIONS = frozenset({*_ARITY, *BIT_WRITES, "NEG", "PUSH_CONDITION", "POP_CONDITION", "INFO_TAG"})
_TAG_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _@:.,=+-/()")
_DIGITS = frozenset("0123456789")
REGISTER_LINES = ("APPEND_TO_REGISTER", "REGISTER")


class CircuitTextError(ValueError):
    """The text is not a circuit this module reads; the message says where and why."""


@dataclass(frozen=True)
class Op:
    """One instruction."""

    name: str
    qubits: tuple[int, ...] = ()
    bit: int | None = None            #: the bit ``HMR`` or a ``BIT_`` instruction writes
    angle: str = ""                   #: ``Z_POW``: half turns, as written
    condition: int | None = None      #: done only if this bit is 1
    text: str = ""                    #: ``INFO_TAG``

    def line(self) -> str:
        if self.name == "INFO_TAG":
            return f'INFO_TAG "{self.text}"'
        parts = [self.name, *(f"q{q}" for q in self.qubits)]
        if self.angle:
            parts.append(self.angle)
        if self.bit is not None:
            parts.append(f"b{self.bit}")
        if self.condition is not None:
            parts += ["if", f"b{self.condition}"]
        return " ".join(parts)


def _index(word: str, letter: str, limit: int, what: str) -> int:
    digits = word[1:]
    if word[:1] != letter or not digits or len(digits) > 7 or not _DIGITS.issuperset(digits):
        raise CircuitTextError(f"expected {what} such as {letter}7, found {word[:20]!r}")
    value = int(digits)
    if value >= limit:
        raise CircuitTextError(f"{what} {word} is beyond the limit of {limit}")
    return value


def _angle(word: str) -> str:
    body = word[1:] if word[:1] == "-" else word
    whole, dot, fraction = body.partition(".")
    if (not whole or len(whole) > 3 or not _DIGITS.issuperset(whole) or (dot and not fraction)
            or len(fraction) > MAX_ANGLE_DIGITS or not _DIGITS.issuperset(fraction)):
        raise CircuitTextError(f"expected an angle in half turns such as 0.125, found {word[:30]!r}")
    return word


def parse_line(line: str) -> Op:
    """One line as an instruction, or ``CircuitTextError``."""
    if not isinstance(line, str):
        raise CircuitTextError(f"a line must be text, found {type(line).__name__}")
    if len(line) > 400:
        raise CircuitTextError("a line of more than 400 characters")
    if line.startswith('INFO_TAG "') and line.endswith('"') and len(line) >= 11:
        text = line[10:-1]
        if len(text) > MAX_TAG or not _TAG_CHARACTERS.issuperset(text):
            raise CircuitTextError(f"a note may hold up to {MAX_TAG} letters, digits, spaces and _@:.,=+-/()")
        return Op("INFO_TAG", text=text)
    words = line.split(" ")
    name = words[0]
    if name not in INSTRUCTIONS or name == "INFO_TAG":
        raise CircuitTextError(f"{name[:30]!r} is not an instruction that may be written here")
    rest = words[1:]
    condition = None
    if len(rest) >= 2 and rest[-2] == "if":
        condition = _index(rest[-1], "b", MAX_BIT, "a bit")
        rest = rest[:-2]
    if name == "PUSH_CONDITION":
        if condition is None or rest:
            raise CircuitTextError("PUSH_CONDITION is written PUSH_CONDITION if b7")
        return Op(name, condition=condition)
    if name == "POP_CONDITION":
        if condition is not None or rest:
            raise CircuitTextError("POP_CONDITION takes nothing")
        return Op(name)
    if name == "NEG":
        if rest:
            raise CircuitTextError("NEG takes no qubit")
        return Op(name, condition=condition)
    if name in BIT_WRITES:
        if len(rest) != 1:
            raise CircuitTextError(f"{name} takes one bit")
        return Op(name, bit=_index(rest[0], "b", MAX_BIT, "a bit"), condition=condition)
    arity = _ARITY[name]
    extra = 1 if name in ("Z_POW", "HMR") else 0
    if len(rest) != arity + extra:
        raise CircuitTextError(f"{name} takes {arity} qubit{'s' if arity > 1 else ''}"
                               + (" and an angle" if name == "Z_POW" else " and a bit" if name == "HMR" else ""))
    qubits = tuple(_index(word, "q", MAX_QUBIT, "a qubit") for word in rest[:arity])
    if len(set(qubits)) != arity:
        raise CircuitTextError(f"{name} names the same qubit twice")
    if name == "Z_POW":
        return Op(name, qubits, angle=_angle(rest[arity]), condition=condition)
    if name == "HMR":
        return Op(name, qubits, bit=_index(rest[arity], "b", MAX_BIT, "a bit"), condition=condition)
    return Op(name, qubits, condition=condition)


def read(lines: Any, *, max_lines: int = MAX_LINES) -> list[Op]:
    """The instructions of a circuit's body, or ``CircuitTextError`` naming the first line that is not one.

    ``lines`` is a list of lines or one text with a line each. Conditions opened with
    ``PUSH_CONDITION`` have to be closed again, in order.
    """
    if isinstance(lines, str):
        if len(lines) > MAX_CHARS:
            raise CircuitTextError(f"more than {MAX_CHARS} characters")
        lines = lines.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    if not isinstance(lines, list | tuple):
        raise CircuitTextError(f"a circuit is a list of lines, found {type(lines).__name__}")
    if len(lines) > max_lines:
        raise CircuitTextError(f"{len(lines)} lines; at most {max_lines} are read")
    ops: list[Op] = []
    depth = 0
    for number, line in enumerate(lines, 1):
        try:
            op = parse_line(line)
        except CircuitTextError as exc:
            raise CircuitTextError(f"line {number}: {exc}") from None
        if op.name == "PUSH_CONDITION":
            depth += 1
        elif op.name == "POP_CONDITION":
            depth -= 1
            if depth < 0:
                raise CircuitTextError(f"line {number}: POP_CONDITION without a PUSH_CONDITION before it")
        ops.append(op)
    if depth:
        raise CircuitTextError(f"{depth} PUSH_CONDITION were never closed by a POP_CONDITION")
    return ops


def written(ops: Iterable[Op]) -> list[str]:
    return [op.line() for op in ops]


def header(inputs: int, name: str = "x") -> list[str]:
    """The lines that declare qubits ``0 .. inputs-1`` as the register ``name``, as the simulator writes them."""
    return [*(f"APPEND_TO_REGISTER q{q} r0" for q in range(inputs)), f'REGISTER r0 "{name}"']


def split(text: str) -> tuple[list[str], list[str]]:
    """A circuit file's text as its register lines and its body. For a file the caller trusts."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    cut = 0
    while cut < len(lines) and lines[cut].startswith(REGISTER_LINES):
        cut += 1
    if any(line.startswith(REGISTER_LINES) for line in lines[cut:]):
        raise CircuitTextError("register lines after the first instruction")
    return lines[:cut], lines[cut:]


def sha256(lines: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


# ---------------------------------------------------------------- what a circuit's text says about its cost
def rotation_kind(angle: str) -> str:
    """``free`` for a multiple of a quarter turn, ``t`` for an odd eighth, ``rotation`` for anything finer."""
    half_turns = Fraction(angle)
    if (half_turns * 2).denominator == 1:
        return "free"
    return "t" if (half_turns * 4).denominator == 1 else "rotation"


def lifetimes(ops: Sequence[Op], inputs: int) -> list[tuple[int, int, int]]:
    """``(first line, last line, qubit)`` for every stretch in which a working qubit is in use.

    A working qubit starts at zero. ``HMR`` measures it and leaves it at zero again, so its use ends
    there for certain and a later instruction on the same qubit starts a new stretch. A qubit that
    is never measured is taken to be in use from its first instruction to its last, which can only
    overstate. Two stretches that do not overlap could share one qubit.
    """
    spans: list[tuple[int, int, int]] = []
    opened: dict[int, int] = {}
    last: dict[int, int] = {}
    for number, op in enumerate(ops):
        for qubit in op.qubits:
            if qubit < inputs:
                continue
            opened.setdefault(qubit, number)
            last[qubit] = number
            if op.name == "HMR" and op.condition is None:
                spans.append((opened.pop(qubit), number, qubit))
    spans += [(start, last[qubit], qubit) for qubit, start in opened.items()]
    return spans


def _most(spans: Sequence[tuple[int, int, int]]) -> tuple[int, int]:
    events = sorted([(start, 1) for start, _, _ in spans] + [(end + 1, -1) for _, end, _ in spans])
    live = most = at = 0
    for position, change in events:
        live += change
        if live > most:
            most, at = live, position
    return most, at


def held(ops: Sequence[Op], inputs: int) -> tuple[int, int]:
    """The most working qubits in use at once, and the line where that happens."""
    return _most(lifetimes(ops, inputs))


def under_a_condition(ops: Sequence[Op]) -> int:
    """How many of the costly gates a shot may skip: those ending in ``if`` or lying inside a condition."""
    depth = skipped = 0
    for op in ops:
        if op.name == "PUSH_CONDITION":
            depth += 1
        elif op.name == "POP_CONDITION":
            depth -= 1
        elif op.name in TOFFOLI_LIKE and (depth > 0 or op.condition is not None):
            skipped += 1
    return skipped


def profile(ops: Sequence[Op], inputs: int, *, parts: int = 12) -> dict[str, Any]:
    """What can be counted without running anything: the width, the gates that cost, where the width peaks.

    ``width`` is what the competition multiplies by: the highest qubit named, plus one. ``held`` is
    the input plus the most working qubits in use at once; where it is below ``width``, renaming
    qubits alone makes the circuit cheaper. ``toffoli_like`` counts the lines, which is the most a
    shot can execute; the verifier counts what a shot does execute. ``along`` cuts the circuit into
    ``parts`` equal pieces and gives, for each, its first line, the qubits in use there and its
    costly gates. ``stages`` does the same for the pieces the circuit's own notes mark, if it has any.
    ``toffoli_like_under_a_condition`` are the costly gates a shot may skip: how many it does skip
    depends on what its measurements give, which is why two circuits that differ in nothing but a
    name do not cost exactly the same.
    """
    counts = Counter(op.name for op in ops)
    rotations = Counter(rotation_kind(op.angle) for op in ops if op.name == "Z_POW")
    highest = max((q for op in ops for q in op.qubits), default=-1)
    spans = lifetimes(ops, inputs)
    most, at = _most(spans)
    starts = sorted(start for start, _, _ in spans)
    ends = sorted(end + 1 for _, end, _ in spans)

    def in_use(line: int) -> int:
        return inputs + bisect_right(starts, line) - bisect_right(ends, line)

    size = max(1, -(-len(ops) // parts))
    along = [{"line": first, "in_use": in_use(first),
              "toffoli_like": sum(op.name in TOFFOLI_LIKE for op in ops[first:first + size])}
             for first in range(0, len(ops), size)]
    stages: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    for number, op in enumerate(ops):
        if op.name != "INFO_TAG":
            continue
        if op.text.startswith("@enter:"):
            stack.append((op.text[7:], number))
        elif op.text.startswith("@exit:") and stack and stack[-1][0] == op.text[6:]:
            name, first = stack.pop()
            if len(stack) <= 1:                          # the outermost two levels: enough to find one's way
                stages.append({"stage": name, "depth": len(stack), "line": first, "lines": number - first,
                               "in_use": in_use(first),
                               "toffoli_like": sum(o.name in TOFFOLI_LIKE for o in ops[first:number])})
    stages.sort(key=lambda row: row["line"])
    return {"lines": len(ops), "width": max(highest + 1, inputs), "held": inputs + most, "held_at_line": at,
            "working_qubits_named": len({q for _, _, q in spans}),
            "toffoli_like": sum(counts[name] for name in TOFFOLI_LIKE),
            "toffoli_like_under_a_condition": under_a_condition(ops),
            "rotations": rotations.get("rotation", 0), "t_gates": rotations.get("t", 0),
            "measurements": counts["HMR"], "bits": len({b for op in ops for b in (op.bit, op.condition) if b is not None}),
            "instructions": dict(sorted(counts.items())), "along": along, "stages": stages[:40]}
