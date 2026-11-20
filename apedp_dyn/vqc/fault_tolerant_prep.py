"""Fault-tolerant preparation of a CSS code's logical zero: a referee that certifies it, and a search that uses it.

Status: experimental

The discovery engines in this package minimise operation count. A circuit that
is short and exact without noise can still be useless on hardware: one faulty
gate in a seven-qubit encoder spreads to an error the code cannot correct. This
module makes resistance to faults the thing that is searched for and the thing
that is certified.

**Circuits considered.** Data qubits start in ``|0>`` or ``|+>``, CNOTs between
data qubits build the code state, and each verification ancilla (``|0>``,
CNOTs from some data qubits onto it, measured in Z) checks one Z-type operator
that the ideal state satisfies. The preparation is accepted when every ancilla
reads 0. No flags, no X-basis checks, no mid-circuit measurement: a code that
needs those gets "none found" here, not a wrong circuit.

**Noise model.** After every CNOT one of the 15 two-qubit Paulis occurs with
total probability ``p``; a prepared qubit is flipped to the orthogonal state
with probability ``p``; a measurement reports the wrong outcome with
probability ``p``. Idle qubits are noiseless.

**The referee (``certify``)** does not sample.

1. Without faults the circuit must leave the data in the logical zero and the
   ancillas in ``|0>``, decided by linear algebra over GF(2).
2. Every single fault is propagated to the end. If the run is accepted, the
   error left on the data must have X part and Z part each of weight at most
   one after reduction by the state's own stabilisers. That is fault tolerance
   for a distance-three code, proved by enumeration.
3. Every pair of faults at different places is propagated too, and the pairs
   that are accepted and decode to the wrong logical state are counted with
   their probabilities. The sum ``c2`` is exact: the logical error rate of an
   accepted preparation is ``c2 * p**2`` to leading order, and ``c1``, the
   first-order term, is zero exactly when step 2 passes for X errors.

**The search (``discover``)** enumerates every shortest CNOT encoder for the
code on the given coupling graph (breadth-first over the subspace the circuit
has built so far), finds for each the cheapest set of checks that catches every
dangerous single-fault error, tries every order of the check's CNOTs, certifies
each candidate, and returns them ranked by CNOT count, then ancillas, then
``c2``. Gate-optimal circuits are not equally good: the last key is what makes
the choice noise-aware.

``stim_logical_error`` samples the same circuit under the same noise in Stim,
an outside simulator, so that the exact ``c2`` can be checked against a rate
this module did not compute.

What is not here: codes of distance above three (the single-fault criterion is
the distance-three one), non-CSS codes, non-Clifford circuits, idle noise, and
any claim about hardware beyond a coupling graph.
"""
from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from apedp_dyn.vqc.circuit import Circuit, Gate, Measure


def _popcount_table(bits: int) -> np.ndarray:
    values: np.ndarray = np.arange(1 << bits, dtype=np.uint32)
    count: np.ndarray = np.zeros(1 << bits, dtype=np.uint8)
    for k in range(bits):
        count += ((values >> k) & 1).astype(np.uint8)
    return count


def _span(generators: Sequence[int]) -> tuple[int, ...]:
    out = {0}
    for g in generators:
        out |= {v ^ g for v in out}
    return tuple(sorted(out))


@dataclass(frozen=True)
class CssCode:
    """A CSS code by its X-type and Z-type parity checks, each a bit mask over the data qubits."""

    name: str
    n: int
    hx: tuple[int, ...]
    hz: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n > 16:
            raise ValueError("tables are built over all 2**n errors; n is limited to 16")
        if any(bin(x & z).count("1") % 2 for x in self.hx for z in self.hz):
            raise ValueError("X and Z checks must commute")

    @cached_property
    def x_space(self) -> tuple[int, ...]:
        """The logical zero is the uniform superposition over these bit strings."""
        return _span(self.hx)

    @cached_property
    def weight(self) -> np.ndarray:
        return _popcount_table(self.n)

    @cached_property
    def zero_z_group(self) -> tuple[int, ...]:
        """Z-type operators that leave the logical zero unchanged: Z checks and logical Z operators."""
        parity = self.weight
        return tuple(v for v in range(1 << self.n) if all(parity[v & x] % 2 == 0 for x in self.hx))

    def _reduced(self, group: Sequence[int]) -> np.ndarray:
        values: np.ndarray = np.arange(1 << self.n, dtype=np.uint32)
        best: np.ndarray = np.full(1 << self.n, 255, dtype=np.uint8)
        for g in group:
            best = np.minimum(best, self.weight[values ^ np.uint32(g)])
        return best

    @cached_property
    def x_reduced_weight(self) -> np.ndarray:
        """Weight of an X error after multiplying by the X stabiliser that shrinks it most."""
        return self._reduced(self.x_space)

    @cached_property
    def z_reduced_weight(self) -> np.ndarray:
        return self._reduced(self.zero_z_group)

    @cached_property
    def decodes_wrongly(self) -> np.ndarray:
        """For every X error: whether minimum-weight correction from its Z-check syndrome leaves a logical error."""
        values: np.ndarray = np.arange(1 << self.n, dtype=np.uint32)
        syndrome: np.ndarray = np.zeros(1 << self.n, dtype=np.uint32)
        for k, z in enumerate(self.hz):
            syndrome |= (self.weight[values & np.uint32(z)] % 2).astype(np.uint32) << k
        order = np.lexsort((values, self.weight))  # lightest error first, smallest index on ties
        correction: np.ndarray = np.zeros(1 << len(self.hz), dtype=np.uint32)
        seen: np.ndarray = np.zeros(1 << len(self.hz), dtype=bool)
        for v in order:
            s = int(syndrome[v])
            if not seen[s]:
                seen[s], correction[s] = True, v
        in_space: np.ndarray = np.zeros(1 << self.n, dtype=bool)
        in_space[list(self.x_space)] = True
        wrong: np.ndarray = ~in_space[values ^ correction[syndrome]]
        return wrong


def steane() -> CssCode:
    """The [[7,1,3]] Steane code: both check matrices are the Hamming parity checks."""
    checks = (0b1010101, 0b1100110, 0b1111000)
    return CssCode("steane", 7, checks, checks)


@dataclass(frozen=True)
class PrepCircuit:
    """Data qubits ``0..n_data-1`` (those in ``plus`` start in ``|+>``), ancillas after them, all measured in Z."""

    n_data: int
    n_anc: int
    plus: tuple[int, ...]
    cnots: tuple[tuple[int, int], ...]

    @property
    def qubits(self) -> int:
        return self.n_data + self.n_anc

    def to_circuit(self) -> Circuit:
        """The same circuit in the package's IR: H on the ``|+>`` qubits, the CNOTs, the ancilla measurements."""
        ops: list[Any] = [Gate("h", (q,)) for q in self.plus]
        ops += [Gate("cx", (c, t)) for c, t in self.cnots]
        ops += [Measure(self.n_data + a, a) for a in range(self.n_anc)]
        circuit = Circuit(ops=tuple(ops), num_qubits=self.qubits, num_bits=self.n_anc,
                          ancillas=tuple(range(self.n_data, self.qubits)))
        circuit.validate()
        return circuit


def prepares_logical_zero(circuit: PrepCircuit, code: CssCode) -> bool:
    """Without faults: the data end in the logical zero and every ancilla in ``|0>``."""
    generators = [1 << q for q in circuit.plus]
    for c, t in circuit.cnots:
        generators = [g ^ (((g >> c) & 1) << t) for g in generators]
    return set(_span(generators)) == set(code.x_space)  # no support on ancillas, and exactly the code's X space


@dataclass(frozen=True)
class Certificate:
    """What ``certify`` established about one circuit, by enumeration."""

    prepares_state: bool
    fault_tolerant: bool            #: no accepted single fault leaves an X or Z error of reduced weight above one
    single_faults: int
    worst_x: int                    #: largest reduced X weight an accepted single fault leaves
    worst_z: int
    c1: float                       #: first-order coefficient of the accepted logical error rate
    c2: float                       #: second-order coefficient; the rate is c2 * p**2 when c1 is zero
    rejection_c1: float             #: first-order coefficient of the rejection rate
    fault_pairs: int
    dangerous: tuple[int, ...] = field(default=(), repr=False)  #: X errors of reduced weight above one from single faults

    def logical_error_rate(self, p: float) -> float:
        """The leading-order prediction; valid where ``p`` times the number of fault locations is small."""
        return self.c1 * p + self.c2 * p * p


_TWO_QUBIT_PAULIS = tuple((xc, zc, xt, zt) for xc in (0, 1) for zc in (0, 1) for xt in (0, 1) for zt in (0, 1)
                          if (xc, zc, xt, zt) != (0, 0, 0, 0))


def single_faults(circuit: PrepCircuit) -> dict[str, np.ndarray]:
    """Every single fault, propagated to the end: X and Z masks on the data, ancilla outcomes, weight, place.

    One backward pass gives, for every moment, what an X or a Z on each qubit
    becomes at the end of the circuit; a fault is then a product of those.
    """
    count = circuit.qubits
    image_x = [1 << q for q in range(count)]   # at the end of the circuit an X on q is an X on q
    image_z = [1 << q for q in range(count)]
    after: list[tuple[list[int], list[int]]] = []
    for c, t in reversed(circuit.cnots):
        after.append((image_x[:], image_z[:]))  # images of a Pauli placed just after this gate
        image_x[c] ^= image_x[t]                 # before the gate, X on the control is X on both after it
        image_z[t] ^= image_z[c]
    after.reverse()
    xs: list[int] = []
    zs: list[int] = []
    flips: list[int] = []
    weights: list[float] = []
    places: list[int] = []
    place = 0
    for q in range(count):  # a prepared qubit flipped to the orthogonal state
        xs.append(0 if q in circuit.plus else image_x[q])
        zs.append(image_z[q] if q in circuit.plus else 0)
        flips.append(0)
        weights.append(1.0)
        places.append(place)
        place += 1
    for (c, t), (ix, iz) in zip(circuit.cnots, after, strict=True):
        for xc, zc, xt, zt in _TWO_QUBIT_PAULIS:
            xs.append((ix[c] if xc else 0) ^ (ix[t] if xt else 0))
            zs.append((iz[c] if zc else 0) ^ (iz[t] if zt else 0))
            flips.append(0)
            weights.append(1.0 / 15.0)
            places.append(place)
        place += 1
    for a in range(circuit.n_anc):  # a measurement that reports the wrong outcome
        xs.append(0)
        zs.append(0)
        flips.append(1 << a)
        weights.append(1.0)
        places.append(place)
        place += 1
    data_mask = np.uint32((1 << circuit.n_data) - 1)
    x_all = np.array(xs, dtype=np.uint32)
    return {"x": x_all & data_mask, "z": np.array(zs, dtype=np.uint32) & data_mask,
            "outcome": (x_all >> np.uint32(circuit.n_data)) ^ np.array(flips, dtype=np.uint32),
            "weight": np.array(weights), "place": np.array(places)}


def certify(circuit: PrepCircuit, code: CssCode) -> Certificate:
    """Decide fault tolerance by enumerating every single fault, and count the fault pairs that cause a logical error."""
    if circuit.n_data != code.n:
        raise ValueError("the circuit and the code disagree on the number of data qubits")
    table = single_faults(circuit)
    x, z, outcome, weight, place = table["x"], table["z"], table["outcome"], table["weight"], table["place"]
    accepted = outcome == 0
    xw, zw = code.x_reduced_weight[x], code.z_reduced_weight[z]
    worst_x = int(xw[accepted].max()) if accepted.any() else 0
    worst_z = int(zw[accepted].max()) if accepted.any() else 0
    wrong = code.decodes_wrongly
    c1 = float(np.sum(weight[accepted & wrong[x]]))
    pair_ok = ((outcome[:, None] ^ outcome[None, :]) == 0) & wrong[x[:, None] ^ x[None, :]] & (place[:, None] != place[None, :])
    c2 = 0.5 * float(np.sum(weight[:, None] * weight[None, :] * pair_ok))
    different = place[:, None] != place[None, :]
    return Certificate(
        prepares_state=prepares_logical_zero(circuit, code), fault_tolerant=bool(worst_x <= 1 and worst_z <= 1),
        single_faults=len(x), worst_x=worst_x, worst_z=worst_z, c1=c1, c2=c2,
        rejection_c1=float(np.sum(weight[~accepted])), fault_pairs=int(different.sum() // 2),
        dangerous=tuple(sorted({int(v) for v in x[accepted & (xw > 1)]})))


def _apply(vectors: tuple[int, ...], c: int, t: int) -> tuple[int, ...]:
    return tuple(sorted(v ^ (((v >> c) & 1) << t) for v in vectors))


def _permuted(mask: int, order: Sequence[int]) -> int:
    return sum(((mask >> q) & 1) << order[q] for q in range(len(order)))


def symmetries(code: CssCode, pairs: Sequence[tuple[int, int]]) -> list[tuple[int, ...]]:
    """Relabellings of the data qubits that keep both the code's X space and the allowed CNOTs."""
    space, allowed = set(code.x_space), set(pairs)
    return [order for order in itertools.permutations(range(code.n))
            if {_permuted(v, order) for v in space} == space and {(order[a], order[b]) for a, b in allowed} == allowed]


def normal_form(cnots: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """One representative of a gate sequence up to swapping neighbouring gates on disjoint qubits.

    Gates are placed in layers, each as early as the gates before it on the same
    qubits allow, and each layer is sorted (the Foata normal form). Two sequences
    have the same normal form exactly when one can be turned into the other by
    swapping neighbouring disjoint gates, which changes neither the circuit nor
    where a fault can land.
    """
    busy: dict[int, int] = {}
    layers: list[list[tuple[int, int]]] = []
    for c, t in cnots:
        layer = max(busy.get(c, 0), busy.get(t, 0))
        if layer == len(layers):
            layers.append([])
        layers[layer].append((c, t))
        busy[c] = busy[t] = layer + 1
    return tuple(gate for layer in layers for gate in sorted(layer))


def shortest_encoders(code: CssCode, edges: Iterable[tuple[int, int]] | None = None,
                      limit: int | None = None, reduce_symmetry: bool = True) -> list[PrepCircuit]:
    """Every shortest CNOT circuit that builds the code's X space from ``|0>`` and ``|+>`` qubits, up to symmetry.

    ``edges`` are the pairs of data qubits a CNOT may act on, in either
    direction; ``None`` allows every pair. The search is breadth-first from the
    code's X space over the subspaces a CNOT can reach, which is exact, and
    then walks the shortest paths back from the product states.

    Two reductions keep the list finite in practice (the Steane code has 94.8
    million shortest gate sequences). Sequences that differ only in the order
    of neighbouring gates on disjoint qubits are one circuit (``normal_form``).
    And with ``reduce_symmetry`` (codes of up to eight qubits) circuits that a
    symmetry of the code and the coupling graph maps onto each other are listed
    once: such a relabelling carries faults to faults of the same probability
    and errors to errors of the same weight, so every number ``certify``
    produces is the same for both. ``limit`` stops the raw enumeration early,
    in a fixed order, before duplicates are removed.
    """
    pairs = [(a, b) for a in range(code.n) for b in range(code.n) if a != b] if edges is None else \
        sorted({(a, b) for a, b in edges} | {(b, a) for a, b in edges})
    goal = tuple(sorted(code.x_space))
    distance = {goal: 0}
    queue = deque([goal])
    while queue:  # a CNOT is its own inverse, so distances from the goal are distances to it
        state = queue.popleft()
        for c, t in pairs:
            nxt = _apply(state, c, t)
            if nxt not in distance:
                distance[nxt] = distance[state] + 1
                queue.append(nxt)
    dimension = len(code.hx)
    choices = list(itertools.combinations(range(code.n), dimension))
    group: list[tuple[int, ...]] = []
    if reduce_symmetry and code.n <= 8:
        group = symmetries(code, pairs)
        choices = sorted({min(tuple(sorted(order[q] for q in plus)) for order in group) for plus in choices})
    starts = [(plus, tuple(sorted(_span([1 << q for q in plus])))) for plus in choices]
    reachable = [(plus, state) for plus, state in starts if state in distance]
    if not reachable:
        return []
    shortest = min(distance[state] for _, state in reachable)
    out: list[PrepCircuit] = []

    def walk(state: tuple[int, ...], path: list[tuple[int, int]], plus: tuple[int, ...]) -> bool:
        if distance[state] == 0:
            out.append(PrepCircuit(code.n, 0, plus, tuple(path)))
            return limit is not None and len(out) >= limit
        for gate in pairs:
            if path and gate < path[-1] and not set(gate) & set(path[-1]):
                continue  # the same circuit with two disjoint neighbours swapped is listed once
            nxt = _apply(state, *gate)
            if distance.get(nxt, -1) == distance[state] - 1:
                path.append(gate)
                if walk(nxt, path, plus):
                    return True
                path.pop()
        return False

    for plus, state in reachable:
        if distance[state] == shortest and walk(state, [], plus):
            break
    kept: dict[tuple[tuple[int, ...], tuple[tuple[int, int], ...]], PrepCircuit] = {}
    for circuit in out:
        same_start = [order for order in group if tuple(sorted(order[q] for q in circuit.plus)) == circuit.plus] or [tuple(range(code.n))]
        form = min(normal_form([(order[c], order[t]) for c, t in circuit.cnots]) for order in same_start)
        kept.setdefault((circuit.plus, form), PrepCircuit(code.n, 0, circuit.plus, form))
    return list(kept.values())


def _covers(dangerous: Sequence[int], checks: Sequence[int], parity: np.ndarray) -> bool:
    return all(any(parity[d & m] % 2 for m in checks) for d in dangerous)


def verified(encoder: PrepCircuit, code: CssCode, *, max_ancillas: int = 2,
             allowed_support: Sequence[int] | None = None) -> list[tuple[PrepCircuit, Certificate]]:
    """The cheapest fault-tolerant completions of one encoder: checks that catch its dangerous errors, every CNOT order.

    ``allowed_support`` is a bit mask per ancilla of the data qubits it may
    couple to; ``None`` lets every ancilla reach every data qubit.
    """
    bare = certify(encoder, code)
    if not bare.prepares_state:
        return []
    if bare.fault_tolerant:
        return [(encoder, bare)]
    parity = code.weight
    operators = sorted((m for m in code.zero_z_group if m), key=lambda m: (int(parity[m]), m))
    best: list[tuple[PrepCircuit, Certificate]] = []
    best_cost: int | None = None
    for count in range(1, max_ancillas + 1):
        for checks in itertools.combinations(operators, count):
            cost = sum(int(parity[m]) for m in checks)
            if best_cost is not None and cost > best_cost:
                continue
            if allowed_support is not None and any(m & ~allowed_support[a] for a, m in enumerate(checks)):
                continue
            if not _covers(bare.dangerous, checks, parity):
                continue
            supports = [[q for q in range(code.n) if (m >> q) & 1] for m in checks]
            for orders in itertools.product(*(itertools.permutations(s) for s in supports)):
                cnots = encoder.cnots + tuple((q, code.n + a) for a, order in enumerate(orders) for q in order)
                candidate = PrepCircuit(code.n, count, encoder.plus, cnots)
                certificate = certify(candidate, code)
                if certificate.prepares_state and certificate.fault_tolerant:
                    if best_cost is None or cost < best_cost:
                        best, best_cost = [], cost
                    best.append((candidate, certificate))
        if best:
            break  # more ancillas cannot be cheaper in ancillas, and the CNOT cost is already minimal for this count
    return best


def discover(code: CssCode, edges: Iterable[tuple[int, int]] | None = None, *, max_ancillas: int = 2,
             allowed_support: Sequence[int] | None = None, encoder_limit: int | None = None) -> list[tuple[PrepCircuit, Certificate]]:
    """Fault-tolerant preparations of the logical zero, best first: fewest CNOTs, then fewest ancillas, then lowest ``c2``."""
    found: list[tuple[PrepCircuit, Certificate]] = []
    for encoder in shortest_encoders(code, edges, encoder_limit):
        found.extend(verified(encoder, code, max_ancillas=max_ancillas, allowed_support=allowed_support))
    found.sort(key=lambda item: (len(item[0].cnots), item[0].n_anc, item[1].c2, item[0].plus, item[0].cnots))
    return found


def stim_logical_error(circuit: PrepCircuit, code: CssCode, p: float, shots: int, seed: int = 0,
                       batch: int = 2_000_000) -> dict[str, Any]:
    """Sample the circuit under the module's noise model in Stim; decode the data; count accepted logical errors."""
    import stim

    text = stim.Circuit()
    everything = list(range(circuit.qubits))
    text.append("R", everything)
    zeros = [q for q in everything if q not in circuit.plus]
    if zeros:
        text.append("X_ERROR", zeros, p)
    if circuit.plus:
        text.append("H", list(circuit.plus))
        text.append("Z_ERROR", list(circuit.plus), p)
    for c, t in circuit.cnots:
        text.append("CX", [c, t])
        text.append("DEPOLARIZE2", [c, t], p)
    ancillas = list(range(circuit.n_data, circuit.qubits))
    if ancillas:
        text.append("M", ancillas, p)
    text.append("M", list(range(circuit.n_data)))
    sampler = text.compile_sampler(seed=seed)
    ancilla_mask = np.uint32((1 << circuit.n_anc) - 1)
    accepted = wrong = done = 0
    while done < shots:
        take = min(batch, shots - done)
        packed = sampler.sample(take, bit_packed=True)  # one row per shot, measurement k in bit k (little-endian)
        value: np.ndarray = np.zeros(take, dtype=np.uint32)
        for k in range(packed.shape[1]):
            value |= packed[:, k].astype(np.uint32) << np.uint32(8 * k)
        keep = (value & ancilla_mask) == 0
        accepted += int(keep.sum())
        wrong += int(code.decodes_wrongly[value[keep] >> np.uint32(circuit.n_anc)].sum())
        done += take
    rate = wrong / accepted if accepted else float("nan")
    return {"shots": shots, "accepted": accepted, "logical_errors": wrong, "rate": rate,
            "standard_error": float(np.sqrt(max(wrong, 1)) / accepted) if accepted else float("nan"),
            "acceptance": accepted / shots}
