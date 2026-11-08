"""A local, MR-capable Z-basis simulator for the DQI decoder.

Why this exists: the four gated DQI optimisations in
``docs/PHOTONIC_ACCELERATOR_BOUNDARY.md`` §5.5 are measured with the predicate
supplied as an **oracle** classical register, because ``kickmix`` has no Z-basis
measurement. That leaves their correctness untested: an oracle bit is not a
readout, so nothing checks that reading the predicate for real leaves the phase
alone. ``kickmix`` ships as a binary with no sources, so the instruction cannot be
added there; this module provides the instruction locally so the arms can be run.

What it is not: a competition engine. It computes values and phases, not gate
counts, and it cannot produce a receipt. `apedp_dyn/quantum_compilation/contracts.py`
pins the official ``kickmix`` build and verifiers by SHA-256, and only those may
score. T counts must keep coming from ``kickmix``.

Model
-----
Every gate in the DQI circuit is Clifford and every state it reaches is a
computational-basis state times a global sign. ``Z_POW`` is forbidden by the
Benchmark II contract, so no other amplitude is reachable. A shot is therefore a
bit per qubit, a bit per classical register bit, and one phase bit — and the
semantics of every builder primitive in this subset are *exact* on that model,
not approximations. The primitives were pinned against ``kickmix`` itself
(`tests/test_local_mr_sim.py` records the expectations, and the reference decoder
is compared shot for shot in ``scripts/verify_local_mr_sim.py``).

The one deliberate divergence from ``kickmix`` is the addition of ``mr``:

* ``r`` (reset) clears the qubit **and randomizes the tracked phase** — the
  defect that makes the clear arms wrong in the probe (``x(q); reset(q)`` reports
  a random phase while the same circuit without the reset reports none).
* ``hmr`` measures in X, so on ``|1>`` it returns a random bit.
* ``mr`` measures in Z (deterministic on a basis state), writes the outcome, and
  clears the qubit **without touching the phase**. That is the instruction
  ``docs/MR_INSTRUCTION_SPEC.md`` specifies.

Field arithmetic uses the same primitive polynomials ``kickmix`` uses, recovered
from it directly: GF(2^4) = x^4+x+1, GF(2^8) = x^8+x^4+x^3+x^2+1, GF(2^12) =
x^12+x^6+x^4+x+1, all with primitive element ``alpha = 2``.
"""
from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Generator, Iterable, Sequence
from typing import Any, ClassVar

import numpy as np

#: Reduction polynomials, recovered from ``kickmix.GF2Field(degree)``.
_PRIMITIVE_POLYNOMIALS = {
    4: 0b10011,
    8: 0b100011101,
    12: 0b1000001010011,
}


class Field:
    """GF(2^degree) with ``kickmix``'s primitive polynomial and generator ``2``."""

    _cache: ClassVar[dict[int, Field]] = {}

    def __new__(cls, degree: int) -> Any:
        cached = cls._cache.get(degree)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        cls._cache[degree] = instance
        return instance

    def __init__(self, degree: int):
        if getattr(self, "_ready", False):
            return
        if degree not in _PRIMITIVE_POLYNOMIALS:
            raise ValueError(
                f"no pinned polynomial for GF(2^{degree}); pinned degrees are "
                f"{sorted(_PRIMITIVE_POLYNOMIALS)}"
            )
        self.degree = degree
        self.size = 1 << degree
        self.primitive_element = 2
        poly = _PRIMITIVE_POLYNOMIALS[degree]
        order = self.size - 1
        self._exp: np.ndarray = np.zeros(2 * order, dtype=np.int64)
        self._log: np.ndarray = np.zeros(self.size, dtype=np.int64)
        value = 1
        for exponent in range(order):
            self._exp[exponent] = value
            self._log[value] = exponent
            value <<= 1
            if value & self.size:
                value ^= poly
        self._exp[order:] = self._exp[:order]
        self._popcount = np.array([bin(v).count("1") for v in range(self.size)], dtype=np.int64)
        self._ready = True

    # Scalar helpers (used when building circuits and by tests).
    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self._exp[int(self._log[a]) + int(self._log[b])])

    def div(self, a: int, b: int) -> int:
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^degree)")
        return int(self._exp[int(self._log[a]) - int(self._log[b]) % (self.size - 1)])

    def inverse(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("0 has no inverse")
        return int(self._exp[(self.size - 1 - int(self._log[a])) % (self.size - 1)])

    def pow(self, a: int, k: int) -> int:
        if k == 0:
            return 1
        if a == 0:
            return 0
        return int(self._exp[(int(self._log[a]) * k) % (self.size - 1)])

    # Vectorised helpers (used by the simulator).
    def mul_values(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        out = np.zeros(np.broadcast(a, b).shape, dtype=np.int64)
        nz = (a != 0) & (b != 0)
        la = np.where(nz, self._log[np.where(nz, a, 0)], 0)
        lb = np.where(nz, self._log[np.where(nz, b, 0)], 0)
        out[nz] = self._exp[(la[nz] + lb[nz]) % (self.size - 1)]
        _out: np.ndarray = out
        return _out

    def mul_constant(self, constant: int, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if constant == 0:
            _out: np.ndarray = np.zeros_like(values)
            return _out
        out = np.zeros_like(values)
        nz = values != 0
        out[nz] = self._exp[(int(self._log[constant]) + self._log[values[nz]]) % (self.size - 1)]
        _out = out
        return _out

    def div_constant(self, constant: int, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if constant == 0:
            raise ZeroDivisionError("division by zero in GF(2^degree)")
        out = np.zeros_like(values)
        nz = values != 0
        shift = int(self._log[constant])
        out[nz] = self._exp[(self._log[values[nz]] - shift) % (self.size - 1)]
        _out: np.ndarray = out
        return _out

    def inverse_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        out = np.zeros_like(values)
        nz = values != 0
        out[nz] = self._exp[(self.size - 1 - self._log[values[nz]]) % (self.size - 1)]
        _out: np.ndarray = out
        return _out

    def parity(self, values: np.ndarray) -> np.ndarray:
        _out: np.ndarray = self._popcount[np.asarray(values, dtype=np.int64)] & 1
        return _out


class Q:
    """A qubit id."""

    __slots__ = ("id", "name")

    def __init__(self, ident: int, name: str | None = None):
        self.id = ident
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"q{self.id}"


class B:
    """A classical bit id."""

    __slots__ = ("id", "name")

    def __init__(self, ident: int, name: str | None = None):
        self.id = ident
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"b{self.id}"


class array(list[Any]):
    """List of qubits or bits, matching ``kickmix.array``'s role."""

    def __init__(self, values: Iterable[Any] = ()):
        super().__init__(values)


def _flatten(obj: Any) -> list[Any]:
    if isinstance(obj, (Q, B, Match)):
        return [obj]
    if isinstance(obj, (list, tuple, np.ndarray)):
        out: list[Any] = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    if isinstance(obj, (bool, np.bool_)):
        return [bool(obj)]
    raise TypeError(f"cannot use {type(obj).__name__} as a qubit or bit")


class Match:
    """The virtual match qubit of a unary-iteration cursor (``x == address``).

    The address is snapshotted when the match is taken, because the circuit can
    call ``move_to`` again later while the recorded operation keeps the address it
    was emitted at.
    """

    __slots__ = ("address", "cursor")

    def __init__(self, cursor: Cursor, address: int):
        self.cursor = cursor
        self.address = address

    def __bool__(self) -> bool:  # pragma: no cover - guards misuse
        raise TypeError("a unary-iteration match is not a Python bool")


class Cursor:
    """Unary-iteration cursor: holds one address at a time."""

    def __init__(self, register: Sequence[Q], control: Any):
        self.register = list(register)
        self.control = control
        self.address: int | None = None
        self.active = False

    def move_to(self, address: int) -> None:
        self.address = int(address)

    @property
    def match_qubit(self) -> Match:
        if not self.active:
            raise ValueError("match_qubit used outside its unary_iteration block")
        if self.address is None:
            raise ValueError("move_to must be called before match_qubit")
        return Match(self, self.address)


class _Mark:
    """Timing marker. It records nothing: this engine does not account for gates."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self) -> _Mark:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


class Op:
    __slots__ = ("args", "cond", "name")

    def __init__(self, name: str, args: tuple[Any, ...], cond: tuple[Any, ...]):
        self.name = name
        self.args = args
        self.cond = cond


class Circuit:
    """A recorded circuit: instruction list plus qubit/bit counts."""

    def __init__(self, ops: list[Op], num_qubits: int, num_bits: int):
        self.ops = ops
        self.num_qubits = num_qubits
        self.num_bits = num_bits

    def operation_counts(self) -> Counter[str]:
        """Instructions by name.

        This is **not** the competition's Toffoli accounting: a high-level op here
        is one instruction whatever gates it expands to. Score only with
        ``kickmix``.
        """
        return Counter(op.name for op in self.ops)

    def __len__(self) -> int:
        return len(self.ops)


class Builder:
    """Records the subset of ``kickmix.CircuitBuilder`` the DQI decoder uses, plus ``mr``."""

    def __init__(self) -> None:
        self._ops: list[Op] = []
        self._next_qubit = 0
        self._next_bit = 0
        self._cond: list[Any] = []
        self._cursors: list[Cursor] = []

    # -- registers -------------------------------------------------------
    def alloc_qubits(self, count: int) -> array:
        qubits = array(Q(self._next_qubit + i) for i in range(count))
        self._next_qubit += count
        return qubits

    def alloc_bits(self, count: int) -> array:
        bits = array(B(self._next_bit + i) for i in range(count))
        self._next_bit += count
        return bits

    def create_quantum_register(self, count: int, name: str | None = None) -> array:
        qubits = self.alloc_qubits(count)
        if name is not None:
            for qubit in qubits:
                qubit.name = name
        return qubits

    def create_classical_register(self, count: int, name: str | None = None) -> array:
        bits = self.alloc_bits(count)
        if name is not None:
            for bit in bits:
                bit.name = name
        return bits

    def free(self, *_objects: Any) -> None:
        """Bookkeeping only: the simulation never reuses ids."""
        return None

    def mark(self, name: str) -> _Mark:
        """A marker, reusable across ``with`` blocks as ``kickmix``'s marks are."""
        return _Mark(name)

    @contextlib.contextmanager
    def unary_iteration(self, register: Sequence[Q], control: Any = True) -> Generator[Any, None, None]:
        cursor = Cursor(register, control)
        cursor.active = True
        self._cursors.append(cursor)
        try:
            yield cursor
        finally:
            cursor.active = False
            self._cursors.pop()

    # -- recording -------------------------------------------------------
    def _emit(self, name: str, *args: Any) -> None:
        self._ops.append(Op(name, args, tuple(self._cond)))

    # -- single-qubit / multi-qubit gates --------------------------------
    def x(self, target: Any, control: Any = True) -> None:
        self._emit("x", target, control)

    def z(self, target: Any, control: Any = True) -> None:
        self._emit("z", target, control)

    def cz(self, a: Any, b: Any, control: Any = True) -> None:
        self._emit("cz", a, b, control)

    def ccz(self, a: Any, b: Any, c: Any, control: Any = True) -> None:
        self._emit("ccz", a, b, c, control)

    def z_pow(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("Z_POW gates are not allowed in Benchmark II.")

    def cx(self, src: Any, dst: Any, control: Any = True) -> None:
        self._emit("cx", src, dst, control)

    def ccx(self, a: Any, b: Any, dst: Any, control: Any = True) -> None:
        self._emit("ccx", a, b, dst, control)

    def swap(self, a: Any, b: Any, control: Any = True) -> None:
        self._emit("swap", a, b, control)

    def cswap(self, control_qubit: Any, a: Any, b: Any) -> None:
        self._emit("cswap", control_qubit, a, b)

    def swap_registers(self, a: Any, b: Any) -> None:
        self.swap(a, b)

    # -- measurement / reset --------------------------------------------
    def r(self, target: Any, control: Any = True) -> None:
        """Reset. Clears the qubit and **randomizes the tracked phase**, as ``kickmix`` does."""
        self._emit("r", target, control)

    def hmr(self, target: Any, output: Any) -> None:
        """X-basis measure-and-reset: random on ``|1>``."""
        self._emit("hmr", target, output)

    def mr(self, target: Any, output: Any) -> None:
        """Z-basis measure-and-reset: deterministic, and the phase is untouched.

        This is the instruction `docs/MR_INSTRUCTION_SPEC.md` asks ``kickmix`` for.
        """
        self._emit("mr", target, output)

    # -- classical bits --------------------------------------------------
    def bit_store(self, target: Any, value: Any, control: Any = True) -> None:
        self._emit("bit_store", target, value, control)

    def push_condition(self, bit: Any) -> None:
        self._cond.append(bit)

    def pop_condition(self) -> None:
        if not self._cond:
            raise IndexError("pop_condition without a matching push_condition")
        self._cond.pop()

    # -- reversible integer arithmetic -----------------------------------
    def iadd(self, value: Any, target: Sequence[Q], control: Any = True) -> None:
        """``target += value`` where ``value`` is an int or another register."""
        self._emit("iadd", value, 1, target, control)

    def isub(self, value: Any, target: Sequence[Q], control: Any = True) -> None:
        self._emit("iadd", value, -1, target, control)

    def flip_if_equal(self, lhs: Any, rhs: Any, *, target: Any, control: Any = True) -> None:
        self._emit("flip_if_equal", lhs, rhs, target, control)

    def flip_if_less_than(self, lhs: Any, rhs: Any, *, target: Any, control: Any = True) -> None:
        self._emit("flip_if_less_than", lhs, rhs, target, control)

    def left_rotate(self, lane: Sequence[Q], count: Any, control: Any = True) -> None:
        self._emit("left_rotate", lane, count, control)

    def right_rotate(self, lane: Sequence[Q], count: Any, control: Any = True) -> None:
        self._emit("right_rotate", lane, count, control)

    def init_and(self, *args: Any) -> None:
        """``init_and(*controls, targets)``: ``target ^= AND(controls)``, elementwise."""
        self._emit("and", args)

    def del_and(self, *args: Any) -> None:
        """Self-inverse of :meth:`init_and`."""
        self._emit("and", args)

    # -- GF(2^degree) arithmetic -----------------------------------------
    def gf2_iadd(self, source: Any, target: Any, field: Field | None = None,
                 control: Any = True) -> None:
        self._emit("gf2_iadd", source, target, control)

    def gf2_imul(self, constant: int, target: Sequence[Q], field: Field | None = None,
                 control: Any = True) -> None:
        self._emit("gf2_imul", int(constant), target, control)

    def gf2_idiv(self, constant: int, target: Sequence[Q], field: Field | None = None,
                 control: Any = True) -> None:
        self._emit("gf2_idiv", int(constant), target, control)

    def ixor_gf2_mul(self, a: Any, b: Any, target: Sequence[Q], field: Field | None = None,
                     control: Any = True) -> None:
        self._emit("ixor_gf2_mul", a, b, target, control)

    def gf2_phase_by_product(self, a: Any, b: Any, mask: Any,
                             field: Field | None = None) -> None:
        self._emit("phase_by_product", a, b, mask, field)

    def init_gf2_inverse_with_scaffold(self, source: Sequence[Q], target: Sequence[Q],
                                       field: Field) -> tuple[array, array]:
        scaffold = self.alloc_qubits(max(1, field.degree - 1))
        self._emit("gf2_inverse", source, target)
        return array(), scaffold

    def del_gf2_inverse_with_scaffold(self, source: Sequence[Q], target: Sequence[Q],
                                      scaffold: Sequence[Q], field: Field) -> None:
        self._emit("gf2_clear", target)

    def init_gf2_div_with_scaffold(self, numerator: Sequence[Q], denominator: Sequence[Q],
                                   field: Field) -> tuple[array, array]:
        ratio = self.alloc_qubits(field.degree)
        scaffold = self.alloc_qubits(max(1, field.degree - 1))
        self._emit("gf2_divide", numerator, denominator, ratio)
        return ratio, scaffold

    def del_gf2_div_with_scaffold(self, numerator: Sequence[Q], denominator: Sequence[Q],
                                  target: Sequence[Q], scaffold: Sequence[Q],
                                  field: Field) -> None:
        self._emit("gf2_clear", target)

    # -- finish ----------------------------------------------------------
    def finish_circuit(self) -> Circuit:
        if self._cond:
            raise ValueError("finish_circuit with unbalanced push_condition")
        return Circuit(self._ops, self._next_qubit, self._next_bit)


class Simulator:
    """Executes a :class:`Circuit` per shot on the Z-basis + sign model."""

    def __init__(self, batch_size: int = 1, debug_print: bool = False, seed: int | None = None,
                 counting_operations: bool = False):
        self.batch_size = batch_size
        self.debug_print = debug_print
        self.counting_operations = counting_operations
        self._rng = np.random.default_rng(seed)
        self._circuits: list[Circuit] = []
        self._num_qubits = 0
        self._num_bits = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self.q: np.ndarray = np.zeros((self._num_qubits, self.batch_size), dtype=np.uint8)
        self.b: np.ndarray = np.zeros((self._num_bits, self.batch_size), dtype=np.uint8)
        self.phase: np.ndarray = np.zeros(self.batch_size, dtype=np.uint8)

    def use_same_registers_as(self, circuit: Circuit) -> None:
        self._circuits.append(circuit)
        self._num_qubits = max(self._num_qubits, circuit.num_qubits)
        self._num_bits = max(self._num_bits, circuit.num_bits)
        self._reset_state()

    def clear_for_shot(self) -> None:
        self._reset_state()

    def write_across_shots(self, register: Sequence[Any], bits: Any) -> None:
        values = np.asarray(bits, dtype=np.uint8)
        if values.ndim == 1:
            values = values[:, None]
        targets = _flatten(register)
        if values.shape[0] != len(targets):
            raise ValueError(f"expected {len(targets)} rows of bits, got {values.shape[0]}")
        if values.shape[1] not in (1, self.batch_size):
            raise ValueError(f"expected 1 or {self.batch_size} shots, got {values.shape[1]}")
        if values.shape[1] == 1 and self.batch_size > 1:
            values = np.repeat(values, self.batch_size, axis=1)
        for row, obj in enumerate(targets):
            store = self.q if isinstance(obj, Q) else self.b
            store[obj.id, :] = values[row]

    def read_across_shots(self, register: Sequence[Any]) -> np.ndarray:
        targets = _flatten(register)
        if not targets:
            return np.zeros((0, self.batch_size), dtype=bool)
        store = self.q if isinstance(targets[0], Q) else self.b
        return store[[obj.id for obj in targets], :].astype(bool)

    def read_phase_flipped_across_shots(self) -> np.ndarray:
        return self.phase.astype(bool)

    # -- condition resolution -------------------------------------------
    def _mask(self, obj: Any) -> np.ndarray:
        """Truth of a control: a bool, a qubit/bit id, or a cursor match."""
        if isinstance(obj, Match):
            cursor = obj.cursor
            matched = self._reg_value(cursor.register) == obj.address
            if cursor.control is not True:
                matched = matched & self._mask(cursor.control)
            _out: np.ndarray = matched
            return _out
        if isinstance(obj, Q):
            return self.q[obj.id, :].astype(bool)
        if isinstance(obj, B):
            return self.b[obj.id, :].astype(bool)
        if isinstance(obj, (bool, np.bool_)):
            return np.full(self.batch_size, bool(obj))
        if isinstance(obj, (list, tuple, array)):
            if not obj:
                return np.ones(self.batch_size, dtype=bool)
            mask: np.ndarray = np.ones(self.batch_size, dtype=bool)
            for item in obj:
                mask &= self._mask(item)
            return mask
        raise TypeError(f"cannot use {type(obj).__name__} as a control")

    def _cond_mask(self, op: Op, control: Any = True) -> np.ndarray:
        mask = self._mask(control)
        for entry in op.cond:
            mask &= self._mask(entry)
        return mask

    def _any_mask(self, obj: Any) -> np.ndarray:
        """Truth of a register: nonzero (any bit set), as ``bit_store`` control uses."""
        targets = _flatten(obj)
        if targets and isinstance(targets[0], (Q, B)):
            mask: np.ndarray = np.zeros(self.batch_size, dtype=bool)
            for target in targets:
                mask |= self._mask(target)
            return mask
        return self._mask(obj)

    # -- register values -------------------------------------------------
    def _reg_value(self, register: Sequence[Any]) -> np.ndarray:
        items = _flatten(register)
        value: np.ndarray = np.zeros(self.batch_size, dtype=np.int64)
        if not items:
            return value
        store = self.q if isinstance(items[0], Q) else self.b
        for index, item in enumerate(items):
            value |= store[item.id, :].astype(np.int64) << index
        return value

    def _write_reg_value(self, register: Sequence[Q], value: np.ndarray,
                         mask: np.ndarray | None = None) -> None:
        for index, qubit in enumerate(_flatten(register)):
            bit = ((value >> index) & 1).astype(np.uint8)
            if mask is None:
                self.q[qubit.id, :] = bit
            else:
                self.q[qubit.id, :] = np.where(mask, bit, self.q[qubit.id, :])

    def _write_bit(self, bit: B, value: np.ndarray, mask: np.ndarray | None = None) -> None:
        value = value.astype(np.uint8)
        if mask is None:
            self.b[bit.id, :] = value
        else:
            self.b[bit.id, :] = np.where(mask, value, self.b[bit.id, :])

    # -- execution -------------------------------------------------------
    def do(self, circuit: Circuit) -> None:
        for op in circuit.ops:
            self._execute(op)

    def _execute(self, op: Op) -> None:
        handler = getattr(self, f"_op_{op.name}")
        handler(op)

    def _op_x(self, op: Op) -> None:
        target, control = op.args
        mask = self._cond_mask(op, control)
        for qubit in _flatten(target):
            self.q[qubit.id, :] ^= mask.astype(np.uint8)

    def _op_z(self, op: Op) -> None:
        target, control = op.args
        mask = self._cond_mask(op, control)
        qubit = _flatten(target)[0]
        self.phase ^= (self.q[qubit.id, :].astype(bool) & mask).astype(np.uint8)

    def _op_cz(self, op: Op) -> None:
        a, b, control = op.args
        mask = self._cond_mask(op, control)
        self.phase ^= (self._mask(a) & self._mask(b) & mask).astype(np.uint8)

    def _op_ccz(self, op: Op) -> None:
        a, b, c, control = op.args
        mask = self._cond_mask(op, control)
        self.phase ^= (self._mask(a) & self._mask(b) & self._mask(c) & mask).astype(np.uint8)

    def _op_cx(self, op: Op) -> None:
        src, dst, control = op.args
        mask = self._cond_mask(op, control)
        sources, targets = _flatten(src), _flatten(dst)
        # ``kickmix`` folds constant controls into the operation and XORs every
        # remaining control into a single target when only one target is given.
        sources = [s for s in sources if not isinstance(s, bool)]
        if not targets:
            return
        if len(targets) == 1 and len(sources) >= 1:
            parity: np.ndarray = np.zeros(self.batch_size, dtype=bool)
            for source in sources:
                parity ^= self._mask(source)
            self.q[targets[0].id, :] ^= (parity & mask).astype(np.uint8)
            return
        if len(sources) == 1 and len(targets) > 1:
            sources = sources * len(targets)
        if len(sources) != len(targets):
            raise ValueError(f"cx: {len(sources)} controls against {len(targets)} targets")
        for source, target in zip(sources, targets, strict=False):
            self.q[target.id, :] ^= (self._mask(source) & mask).astype(np.uint8)

    def _op_ccx(self, op: Op) -> None:
        a, b, dst, control = op.args
        mask = self._cond_mask(op, control)
        targets = _flatten(dst)
        for target in targets:
            self.q[target.id, :] ^= (self._mask(a) & self._mask(b) & mask).astype(np.uint8)

    def _op_swap(self, op: Op) -> None:
        a, b, control = op.args
        mask = self._cond_mask(op, control)
        for left, right in zip(_flatten(a), _flatten(b), strict=False):
            lhs, rhs = self.q[left.id, :].copy(), self.q[right.id, :].copy()
            self.q[left.id, :] = np.where(mask, rhs, lhs)
            self.q[right.id, :] = np.where(mask, lhs, rhs)

    def _op_cswap(self, op: Op) -> None:
        control_qubit, a, b = op.args
        mask = self._cond_mask(op, control_qubit)
        for left, right in zip(_flatten(a), _flatten(b), strict=False):
            lhs, rhs = self.q[left.id, :].copy(), self.q[right.id, :].copy()
            self.q[left.id, :] = np.where(mask, rhs, lhs)
            self.q[right.id, :] = np.where(mask, lhs, rhs)

    def _op_r(self, op: Op) -> None:
        target, control = op.args
        mask = self._cond_mask(op, control)
        for qubit in _flatten(target):
            self.q[qubit.id, :] = np.where(mask, 0, self.q[qubit.id, :]).astype(np.uint8)
        # kickmix's defect, reproduced: the reset randomizes the tracked phase.
        flip = (self._rng.random(self.batch_size) < 0.5) & mask
        self.phase ^= flip.astype(np.uint8)

    def _op_hmr(self, op: Op) -> None:
        target, output = op.args
        qubits, bits = _flatten(target), _flatten(output)
        mask = self._cond_mask(op)
        for qubit, bit in zip(qubits, bits, strict=False):
            value: np.ndarray = self.q[qubit.id, :].astype(bool)
            outcome = value & (self._rng.random(self.batch_size) < 0.5)
            self._write_bit(bit, outcome.astype(np.uint8), mask)
            self.q[qubit.id, :] = np.where(mask, 0, self.q[qubit.id, :]).astype(np.uint8)

    def _op_mr(self, op: Op) -> None:
        target, output = op.args
        qubits, bits = _flatten(target), _flatten(output)
        mask = self._cond_mask(op)
        for qubit, bit in zip(qubits, bits, strict=False):
            outcome = self.q[qubit.id, :].copy()
            self._write_bit(bit, outcome, mask)
            self.q[qubit.id, :] = np.where(mask, 0, self.q[qubit.id, :]).astype(np.uint8)
        # No phase bookkeeping: that is the whole difference from ``r``.

    def _op_bit_store(self, op: Op) -> None:
        target, value, control = op.args
        mask = self._cond_mask(op) & self._any_mask(control)
        targets = _flatten(target)
        if isinstance(value, (bool, np.bool_)):
            stored = np.full(self.batch_size, bool(value))
        else:
            stored = self._mask(value)
        for bit in targets:
            self._write_bit(bit, stored.astype(np.uint8), mask)

    def _op_iadd(self, op: Op) -> None:
        value, sign, target, control = op.args
        mask = self._cond_mask(op, control)
        width = len(_flatten(target))
        delta = self._comparison_operand(value) * sign
        current = self._reg_value(target)
        updated = (current + delta) & ((1 << width) - 1)
        self._write_reg_value(target, updated, mask)

    def _comparison_operand(self, operand: Any) -> Any:
        if isinstance(operand, (int, np.integer, bool, np.bool_)):
            return int(operand)
        items = _flatten(operand)
        return self._reg_value(items)

    def _op_flip_if_equal(self, op: Op) -> None:
        lhs, rhs, target, control = op.args
        mask = self._cond_mask(op, control)
        equal = (self._comparison_operand(lhs) == self._comparison_operand(rhs)) & mask
        for qubit in _flatten(target):
            self.q[qubit.id, :] ^= equal.astype(np.uint8)

    def _op_flip_if_less_than(self, op: Op) -> None:
        lhs, rhs, target, control = op.args
        mask = self._cond_mask(op, control)
        less = (self._comparison_operand(lhs) < self._comparison_operand(rhs)) & mask
        for qubit in _flatten(target):
            self.q[qubit.id, :] ^= less.astype(np.uint8)

    def _rotate(self, op: Op, direction: int) -> None:
        lane, count, control = op.args
        mask = self._cond_mask(op, control)
        qubits = _flatten(lane)
        width = len(qubits)
        if isinstance(count, (int, np.integer)):
            amounts = np.full(self.batch_size, int(count) * direction)
        else:
            amounts = self._reg_value(_flatten(count)) * direction
        amounts = np.mod(amounts, width)
        values = self._reg_value(qubits)
        rotated = np.zeros_like(values)
        indices = np.arange(width)
        for amount in np.unique(amounts):
            subset = amounts == amount
            # left by k: new[i] = old[(i + k) % width]; right by k is the inverse.
            gather = np.mod(indices + amount, width)
            bits = np.stack([((values[subset] >> int(g)) & 1) for g in gather], axis=0)
            rotated[subset] = (bits.astype(np.int64) << np.arange(width)[:, None]).sum(axis=0)
        self._write_reg_value(qubits, rotated, mask)

    # Direction pinned against kickmix: left by k makes new[i] = old[i - k].
    def _op_left_rotate(self, op: Op) -> None:
        self._rotate(op, -1)

    def _op_right_rotate(self, op: Op) -> None:
        self._rotate(op, 1)

    def _op_and(self, op: Op) -> None:
        args, = op.args
        if len(args) < 2:
            raise ValueError("and needs at least one control and a target")
        mask = self._cond_mask(op)
        targets = _flatten(args[-1])
        control_lists = [_flatten(control) for control in args[:-1]]
        for controls in control_lists:
            if len(controls) not in (1, len(targets)):
                raise ValueError(f"and: {len(controls)} controls against {len(targets)} targets")
            if len(controls) == 1 and len(targets) > 1:
                controls.extend(controls * (len(targets) - 1))
        for index, target in enumerate(targets):
            conjunction: np.ndarray = np.ones(self.batch_size, dtype=bool)
            for controls in control_lists:
                conjunction &= self._mask(controls[index])
            self.q[target.id, :] ^= (conjunction & mask).astype(np.uint8)

    def _op_gf2_iadd(self, op: Op) -> None:
        source, target, control = op.args
        mask = self._cond_mask(op, control)
        current = self._reg_value(target)
        if isinstance(source, (int, np.integer)):
            updated = current ^ int(source)
        else:
            updated = current ^ self._reg_value(_flatten(source))
        self._write_reg_value(target, updated, mask)

    def _op_gf2_imul(self, op: Op) -> None:
        constant, target, control = op.args
        mask = self._cond_mask(op, control)
        field = self._field_for(target)
        updated = field.mul_constant(constant, self._reg_value(target))
        self._write_reg_value(target, updated, mask)

    def _op_gf2_idiv(self, op: Op) -> None:
        constant, target, control = op.args
        mask = self._cond_mask(op, control)
        field = self._field_for(target)
        updated = field.div_constant(constant, self._reg_value(target))
        self._write_reg_value(target, updated, mask)

    def _op_ixor_gf2_mul(self, op: Op) -> None:
        a, b, target, control = op.args
        mask = self._cond_mask(op, control)
        field = self._field_for(target)
        product = field.mul_values(self._reg_value(_flatten(a)), self._reg_value(_flatten(b)))
        self._write_reg_value(target, self._reg_value(target) ^ product, mask)

    def _op_phase_by_product(self, op: Op) -> None:
        a, b, mask_bits, _field = op.args
        # The passed field is whatever object the caller uses; arithmetic is local.
        field = self._field_for(a)
        product = field.mul_values(self._reg_value(_flatten(a)), self._reg_value(_flatten(b)))
        mask_value = self._reg_value(_flatten(mask_bits)) if not isinstance(mask_bits, (int, np.integer)) else np.full(self.batch_size, int(mask_bits))
        self.phase ^= field.parity(product & mask_value).astype(np.uint8)

    def _op_gf2_inverse(self, op: Op) -> None:
        source, target = op.args
        field = self._field_for(target)
        self._write_reg_value(target, field.inverse_values(self._reg_value(source)))

    def _op_gf2_divide(self, op: Op) -> None:
        numerator, denominator, target = op.args
        field = self._field_for(target)
        num = self._reg_value(numerator)
        den = self._reg_value(denominator)
        quotient = np.zeros_like(num)
        nonzero = den != 0
        if np.any(nonzero):
            quotient[nonzero] = field.mul_values(num[nonzero], field.inverse_values(den[nonzero]))
        self._write_reg_value(target, quotient)

    def _op_gf2_clear(self, op: Op) -> None:
        target, = op.args
        self._write_reg_value(target, np.zeros(self.batch_size, dtype=np.int64))

    def _field_for(self, register: Any) -> Field:
        width = len(_flatten(register))
        return Field(width)
