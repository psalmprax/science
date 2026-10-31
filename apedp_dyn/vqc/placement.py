"""Architecture-aware depth: the same construction placed on a real connectivity graph.

Status: experimental

The synthesis record reports *logical* depth -- the depth of the abstract circuit,
where any qubit may interact with any other. A depth-12 balanced tree is not depth 12
on hardware that only lets neighbours interact: the non-local edges have to be paid
for with routing, and this module is where that price is measured instead of assumed.

    logical depth  -- `circuit.depth()`, the number the synthesis record quotes
    mapped depth   -- the depth after routing onto a fixed connectivity graph

The two are different numbers, and neither dominates: a chain is depth ``n`` but needs
no routing at all on a line, while a balanced tree is depth ``log2(n)`` and needs
movement on every topology that is not all-to-all. That trade-off is why the synthesis
report keeps a Pareto frontier over depth and edge span rather than a single winner,
and it is why the review asked for the mapped number as a second measurement.

What routing does
-----------------
Qubits start at the node of the same index and a two-qubit gate can only be emitted
between adjacent nodes. When a gate's operands are not adjacent, one of them is
*SWAPped along a shortest path* until they are, which changes where every logical
qubit lives; the permutation is tracked exactly, so the mapped circuit is a relabelled
version of the original rather than an approximation of it. Three-or-more-qubit gates
are routed by bringing every operand into the neighbourhood of the first control and
are refused if the topology has no such neighbourhood -- a refusal, not a silent
three-qubit gate on a device that does not have one.

Restoring the permutation
-------------------------
A routed circuit implements the original *up to where its qubits ended up*. Left
alone, that final permutation is not the identity, so comparing the mapped circuit
with a state specification would compare two different states. With `restore=True`
(the default) the emitted swaps are undone in reverse order at the end -- the swap
product then telescopes to the identity -- so the mapped circuit is the same unitary
on the same wires and the specification referee decides it directly.The restore ladder is part of the reported cost, and it is reported *separately*
(:attr:`Placement.depth_before_restore`, :attr:`Placement.restore_swap_gates`), so a
compiler that can relabel its outputs is free to quote the smaller number while the
record still shows what the verifiable circuit costs. The reported :attr:`mapped_depth`
is the **total** depth including that ladder -- the depth a device executes -- and
:attr:`restore_depth_cost` is how many layers the ladder adds above the pre-restore
circuit, so neither number can be mistaken for the other.

Two-qubit accounting is reported in three bases, because they answer different
questions and the record previously showed only the first:

* **abstract operations** (:attr:`Placement.two_qubit_gates`) -- every `cx`, `cz` and
  `swap` instruction in the mapped circuit, with a SWAP counted as one operation;
* **native entanglers** (:attr:`Placement.native_entanglers`) -- the same circuit in a
  declared basis that charges :attr:`Placement.swap_entanglers` per SWAP: 1 for a device
  whose basis has a native SWAP, 3 for a CNOT-only one;
* **CNOT-equivalent** (:attr:`Placement.cx_equivalent_two_qubit_gates`) -- always at
  three CNOTs per SWAP, which is the count a CNOT-only backend must execute.

What this module does not do
----------------------------
Two routers are built and measured. `strategy="greedy"` is the original: move the
second operand along a shortest path to the first, with no lookahead. `strategy=
"lookahead"` scores every swap that makes progress by the total wire distance of the
next `window` two-qubit gates that involve either qubit the swap moves, and takes the
best one, breaking ties by how soon those qubits are needed again. The distance is
still the only cost model, and neither router schedules gates: a fixed program order is
routed as written. The record reports both, so the lookahead is a measured improvement
over the greedy numbers rather than an assertion that a better heuristic exists. It
also assumes every node can hold a qubit and every edge is a two-qubit interaction of
equal cost, which is the standard abstract topology model and not a device calibration.
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from collections.abc import Sequence
from typing import Any

from apedp_dyn.vqc.circuit import Circuit, Gate, Measure, Noise, Reset

#: The router strategies `place` accepts. `greedy` is the original; `lookahead` scores
#: the swaps that make progress by what they do to the next few gates.
ROUTERS = ("greedy", "lookahead")

#: A SWAP is one abstract two-qubit operation, one native instruction on a device with
#: a SWAP in its basis, and three CNOT-equivalent gates on one that has only CNOTs. The
#: record reports all three counts rather than choosing one and leaving the reader to
#: guess the basis (review, 2026-10-30, §6).
SWAP_AS_CNOT = 3

__all__ = [
    "CONNECTIVITIES",
    "ROUTERS",
    "Connectivity",
    "Placement",
    "PlacementError",
    "all_to_all",
    "grid",
    "line",
    "place",
    "ring",
]


class PlacementError(ValueError):
    """A circuit or topology this router cannot place honestly."""


class Connectivity:
    """An undirected interaction graph on ``num_nodes`` nodes, with BFS distances."""

    def __init__(self, name: str, num_nodes: int,
                 edges: Sequence[tuple[int, int]] = (), *,
                 complete: bool = False) -> None:
        if num_nodes < 2:
            raise PlacementError(f"a connectivity graph needs at least two nodes, "
                                 f"got {num_nodes}")
        self.name = name
        self.num_nodes = int(num_nodes)
        self.complete = bool(complete)
        self.edges: tuple[tuple[int, int], ...] = tuple(
            (int(first), int(second)) for first, second in edges)
        adjacency: list[set[int]] = [set() for _ in range(self.num_nodes)]
        for first, second in self.edges:
            if not (0 <= first < self.num_nodes and 0 <= second < self.num_nodes):
                raise PlacementError(f"edge {(first, second)} is outside the graph")
            if first == second:
                raise PlacementError(f"edge {(first, second)} is a self-loop")
            adjacency[first].add(second)
            adjacency[second].add(first)
        self._adjacency: tuple[frozenset[int], ...] = tuple(
            frozenset(row) for row in adjacency)
        self._distances: tuple[tuple[int, ...], ...] | None = None
        if not self.complete:
            reachable = self._reachable(0)
            if len(reachable) != self.num_nodes:
                raise PlacementError(
                    f"the {name!r} graph is not connected: {self.num_nodes - len(reachable)} "
                    f"of {self.num_nodes} nodes are unreachable from node 0, so a circuit "
                    f"cannot be routed on it")

    # ------------------------------------------------------------------ structure
    def _reachable(self, start: int) -> set[int]:
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in self._adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return seen

    def adjacent(self, first: int, second: int) -> bool:
        """Whether a two-qubit gate can be emitted between these two nodes."""
        if first == second:
            return False
        return self.complete or second in self._adjacency[first]

    def neighbours(self, node: int) -> frozenset[int]:
        if self.complete:
            return frozenset(other for other in range(self.num_nodes) if other != node)
        return self._adjacency[node]

    def shortest_path(self, start: int, end: int) -> tuple[int, ...]:
        """A shortest node path from `start` to `end`, inclusive of both ends."""
        if start == end:
            return (start,)
        if self.complete or end in self._adjacency[start]:
            return (start, end)
        previous: dict[int, int] = {start: start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in self._adjacency[node]:
                if neighbour in previous:
                    continue
                previous[neighbour] = node
                if neighbour == end:
                    path = [end]
                    while path[-1] != start:
                        path.append(previous[path[-1]])
                    return tuple(reversed(path))
                queue.append(neighbour)
        raise PlacementError(f"no path from node {start} to node {end}")  # pragma: no cover

    def distance(self, first: int, second: int) -> int:
        return len(self.shortest_path(first, second)) - 1

    @property
    def distances(self) -> tuple[tuple[int, ...], ...]:
        """All-pairs BFS distances, computed once: the lookahead asks for them a lot.

        The router scores several candidate swaps per step and each score needs the
        distance of several upcoming gates, so a BFS per question would dominate the
        run. The matrix is a cached property, not part of the topology's identity.
        """
        if self._distances is not None:
            return self._distances
        width = self.num_nodes
        matrix: list[tuple[int, ...]] = []
        for start in range(width):
            row = [-1] * width
            row[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbour in self._adjacency[node]:
                    if row[neighbour] < 0:
                        row[neighbour] = row[node] + 1
                        queue.append(neighbour)
            matrix.append(tuple(row))
        self._distances = tuple(matrix)
        return self._distances

    def degree(self, node: int) -> int:
        return self.num_nodes - 1 if self.complete else len(self._adjacency[node])

    @property
    def max_degree(self) -> int:
        return max(self.degree(node) for node in range(self.num_nodes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": self.num_nodes,
            "edges": ("complete" if self.complete else len(self.edges)),
            "max_degree": self.max_degree,
            "fully_connected": self.complete,
        }


# ----------------------------------------------------------------------- topologies
def line(num_nodes: int) -> Connectivity:
    """Nearest-neighbour in one dimension: node ``i`` touches ``i-1`` and ``i+1``."""
    return Connectivity("line", num_nodes,
                        [(node, node + 1) for node in range(num_nodes - 1)])


def ring(num_nodes: int) -> Connectivity:
    """A line with the closing edge, so the end nodes are neighbours too."""
    if num_nodes < 3:
        raise PlacementError(f"a ring needs at least three nodes, got {num_nodes}")
    return Connectivity("ring", num_nodes,
                        [(node, (node + 1) % num_nodes) for node in range(num_nodes)])


def grid(num_nodes: int) -> Connectivity:
    """A rectangular lattice, filled row-major, with a possibly short last row."""
    rows = math.isqrt(num_nodes)
    while rows > 1 and num_nodes % rows:
        rows -= 1
    columns = num_nodes // rows if rows > 1 else 1
    if rows * columns != num_nodes:  # pragma: no cover - `rows` is a divisor by loop
        raise PlacementError(f"cannot factor {num_nodes} into a rectangle")
    edges: list[tuple[int, int]] = []
    for node in range(num_nodes):
        row, column = divmod(node, columns)
        if column + 1 < columns:
            edges.append((node, node + 1))
        if row + 1 < rows:
            edges.append((node, node + columns))
    return Connectivity(f"grid_{rows}x{columns}", num_nodes, edges)


def all_to_all(num_nodes: int) -> Connectivity:
    """Every pair interacts: the logical circuit's own assumption, as a topology."""
    return Connectivity("all_to_all", num_nodes, complete=True)


#: name -> builder, in the order the record reports them.
CONNECTIVITIES: dict[str, Any] = {
    "line": line,
    "ring": ring,
    "grid": grid,
    "all_to_all": all_to_all,
}


# ------------------------------------------------------------------------ placement
@dataclasses.dataclass(frozen=True)
class Placement:
    """One circuit routed onto one topology, with both depths and the bill for routing."""

    connectivity: str
    num_qubits: int
    circuit: Circuit
    logical_depth: int
    depth_before_restore: int
    mapped_depth: int
    logical_operations: int
    mapped_operations: int
    logical_two_qubit_gates: int
    two_qubit_gates: int
    swap_gates: int
    restore_swap_gates: int
    longest_hop: int
    restored: bool
    permutation: tuple[int, ...]
    connectivity_details: dict[str, Any] = dataclasses.field(default_factory=dict)
    router: str = "greedy"
    lookahead_window: int | None = None
    #: Entanglers the declared basis charges per SWAP: 1 for a native SWAP, 3 for a
    #: CNOT-only basis. The CX-equivalent column is always computed at 3.
    swap_entanglers: int = 1

    @property
    def native_entanglers(self) -> int:
        """Two-qubit entanglers in the declared basis, swaps included."""
        return self.logical_two_qubit_gates + self.swap_entanglers * self.total_swaps

    @property
    def cx_equivalent_two_qubit_gates(self) -> int:
        """The same circuit if every SWAP is decomposed into three CNOTs.

        The logical two-qubit gates are one CNOT-equivalent each (they are `cx`, `cz`
        and a `swap` is the only multi-gate instruction), so this is the count a device
        with no native SWAP must execute.
        """
        return self.logical_two_qubit_gates + SWAP_AS_CNOT * self.total_swaps

    @property
    def restore_depth_cost(self) -> int:
        """Layers the restore ladder adds above the pre-restoration circuit."""
        return self.mapped_depth - self.depth_before_restore

    @property
    def total_swaps(self) -> int:
        return self.swap_gates + self.restore_swap_gates

    @property
    def routing_overhead(self) -> float:
        """Extra two-qubit gates the topology charges, per logical two-qubit gate."""
        if not self.logical_two_qubit_gates:
            return 0.0
        return self.total_swaps / float(self.logical_two_qubit_gates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "connectivity": self.connectivity,
            "router": self.router,
            "lookahead_window": self.lookahead_window,
            **self.connectivity_details,
            "qubits": self.num_qubits,
            "logical_depth": self.logical_depth,
            "depth_before_restore": self.depth_before_restore,
            "mapped_depth": self.mapped_depth,
            "logical_operations": self.logical_operations,
            "mapped_operations": self.mapped_operations,
            "logical_two_qubit_gates": self.logical_two_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "abstract_two_qubit_gates": self.two_qubit_gates,
            "swap_gates": self.swap_gates,
            "restore_swap_gates": self.restore_swap_gates,
            "total_swaps": self.total_swaps,
            "swap_entanglers": self.swap_entanglers,
            "swap_as_cnot": SWAP_AS_CNOT,
            "native_entanglers": self.native_entanglers,
            "cx_equivalent_two_qubit_gates": self.cx_equivalent_two_qubit_gates,
            "restore_depth_cost": self.restore_depth_cost,
            "routing_overhead_per_two_qubit_gate": self.routing_overhead,
            "longest_hop": self.longest_hop,
            "restored": self.restored,
            "permutation_is_identity": all(node == index for index, node in
                                           enumerate(self.permutation)),
            "circuit_sha256": self.circuit.sha256(),
        }


def place(circuit: Circuit, connectivity: Connectivity, *,
          restore: bool = True, strategy: str = "greedy",
          lookahead: int = 8, swap_entanglers: int = 1) -> Placement:
    """Route `circuit` onto `connectivity` and report what it cost.

    `strategy="greedy"` moves the second operand along a shortest path to the first,
    one swap at a time, with no view of what comes next. `strategy="lookahead"` also
    makes progress one swap at a time, but it chooses *which* swap: either operand may
    move, along any shortest path, and each candidate is scored by the total wire
    distance of the next `lookahead` two-qubit gates that involve a qubit it moves.
    Ties go to the swap that moves the qubit needed later, and then to the greedy
    choice, so the two routers agree wherever looking ahead sees nothing.

    With `restore=True` the routing swaps are undone at the end, so the returned
    circuit is the *same* unitary on the same wires and a state specification can be
    checked against it directly. With `restore=False` the circuit is only correct up
to :attr:`Placement.permutation`, which is exactly the relabelling a compiler that
tracks its final layout would absorb for free.
    """
    if strategy not in ROUTERS:
        raise PlacementError(f"strategy must be one of {ROUTERS}, got {strategy!r}")
    if lookahead < 0:
        raise PlacementError(f"lookahead must be non-negative, got {lookahead}")
    if swap_entanglers < 1:
        raise PlacementError(
            f"a SWAP costs at least one entangler, got {swap_entanglers}")
    circuit.validate()
    if circuit.num_qubits > connectivity.num_nodes:
        raise PlacementError(
            f"the circuit acts on {circuit.num_qubits} qubits but the "
            f"{connectivity.name!r} graph has {connectivity.num_nodes} nodes")

    num_nodes = connectivity.num_nodes
    position = list(range(circuit.num_qubits))          # logical qubit -> node
    occupant = list(range(num_nodes))                   # node -> logical qubit
    for node in range(circuit.num_qubits, num_nodes):
        occupant[node] = -1

    emitted: list[Any] = []
    routing_swaps: list[tuple[int, int]] = []
    restore_swaps = 0
    longest_hop = 0
    #: The layer each node was last used in. Every emitted operation updates it, so
    #: `max(available[a], available[b]) + 1` is the layer a swap would occupy -- the
    #: schedule the circuit's own depth is computed from. The lookahead router uses it
    #: to fill idle wires instead of extending the critical path.
    available = [0] * num_nodes

    def emit(op: Any) -> None:
        emitted.append(op)
        layer = max(available[node] for node in op.targets) + 1
        for node in op.targets:
            available[node] = layer

    def emit_swap(first: int, second: int, *, record: bool) -> None:
        """Swap two nodes' contents: one `swap` gate, and the permutation follows it."""
        emit(Gate("swap", (first, second)))
        if record:
            routing_swaps.append((first, second))
        left, right = occupant[first], occupant[second]
        occupant[first], occupant[second] = right, left
        if left >= 0:
            position[left] = second
        if right >= 0:
            position[right] = first

    budget = 4 * (circuit.num_qubits + connectivity.num_nodes) + 16
    # The lookahead's corpus: every two-qubit gate in program order, so "the next
    # `window` gates" means the next ones that need routing at all.
    program = tuple(tuple(op.targets) for op in circuit.ops
                    if isinstance(op, Gate) and len(op.targets) == 2)
    program_rank = 0
    matrix = connectivity.distances if strategy == "lookahead" else ()

    def choose_swap(first_logical: int, second_logical: int) -> tuple[int, int]:
        """The swap that makes progress, scheduled earliest, with the cheapest future.

        Every candidate is one step along a shortest path for one of the two operands,
        so each reduces the distance to the partner by exactly one. What differs is
        *when* the swap can run and *what* it does to the next few gates.

        The primary key is the layer the swap would occupy -- one past the later of the
        two nodes' last uses -- so routing fills idle wires instead of lengthening the
        critical path. That is the scheduler. The secondary key is the lookahead: the
        sum of the wire distances of the next `window` two-qubit gates that involve a
        moved qubit (a gate involving neither is unaffected, so it cannot distinguish
        the candidates). The tertiary key is how soon those qubits are needed again,
        and the last keys make the choice deterministic and prefer the greedy direction
        when looking ahead sees nothing.
        """
        first_node, second_node = position[first_logical], position[second_logical]
        gap = matrix[first_node][second_node]
        candidates: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for side, node, goal in ((1, second_node, first_node), (0, first_node, second_node)):
            for neighbour in sorted(connectivity.neighbours(node)):
                if matrix[neighbour][goal] != gap - 1:
                    continue
                edge = (min(node, neighbour), max(node, neighbour))
                if edge in seen:
                    continue
                seen.add(edge)
                candidates.append((edge[0], edge[1], side))
        upcoming = program[program_rank + 1:program_rank + 1 + lookahead]
        best: tuple[tuple[Any, ...], tuple[int, int]] | None = None
        for left, right, side in candidates:
            left_logical, right_logical = occupant[left], occupant[right]
            occupant[left], occupant[right] = right_logical, left_logical
            if left_logical >= 0:
                position[left_logical] = right
            if right_logical >= 0:
                position[right_logical] = left
            moved = [logical for logical in (left_logical, right_logical) if logical >= 0]
            future = 0
            for targets in upcoming:
                if not any(logical in targets for logical in moved):
                    continue
                future += matrix[position[targets[0]]][position[targets[1]]]
            urgency = 0
            for logical in moved:
                for rank, targets in enumerate(upcoming):
                    if logical in targets:
                        urgency += lookahead - rank
                        break
            occupant[left], occupant[right] = left_logical, right_logical
            if left_logical >= 0:
                position[left_logical] = left
            if right_logical >= 0:
                position[right_logical] = right
            layer = max(available[left], available[right]) + 1
            key = (layer, future, urgency, side, left, right)
            if best is None or key < best[0]:
                best = (key, (left, right))
        if best is None:  # pragma: no cover - a shortest path always has a next step
            raise PlacementError(
                f"no swap reduces the distance between logical qubits "
                f"{first_logical} and {second_logical}")
        return best[1]

    for op in circuit.ops:
        if isinstance(op, Gate) and len(op.targets) >= 2:
            targets = list(op.targets)
            hops = 0
            pair = len(targets) == 2
            for logical in targets[1:]:
                # Bring each operand into the anchor's neighbourhood. The distance
                # before routing is the wire length the topology would have needed.
                longest_hop = max(longest_hop, connectivity.distance(
                    position[targets[0]], position[logical]))
                while not connectivity.adjacent(position[targets[0]], position[logical]):
                    if strategy == "lookahead" and pair:
                        first, second = choose_swap(targets[0], logical)
                    else:
                        path = connectivity.shortest_path(position[logical],
                                                          position[targets[0]])
                        first, second = path[0], path[1]
                    emit_swap(first, second, record=True)
                    hops += 1
                    if hops > budget:  # pragma: no cover - routing cannot cycle
                        raise PlacementError(
                            f"routing {op!r} did not converge in {budget} swaps")
            if pair:
                program_rank += 1
            placed = tuple(position[logical] for logical in targets)
            # Everything adjacent to the anchor is not necessarily adjacent to each
            # other: on a line the two neighbours of `i` are two hops apart, so a
            # three-qubit gate cannot be executed there. Refuse rather than emit it.
            if len(placed) > 2:
                for first in range(len(placed)):
                    for second in range(first + 1, len(placed)):
                        if not connectivity.adjacent(placed[first], placed[second]):
                            raise PlacementError(
                                f"{op.name!r} needs {len(placed)} mutually adjacent "
                                f"qubits; nodes {placed} are not, on the "
                                f"{connectivity.name!r} graph")
            emit(Gate(op.name, placed, op.params, op.condition))
        elif isinstance(op, Gate):
            emit(Gate(op.name, (position[op.targets[0]],), op.params, op.condition))
        elif isinstance(op, Measure):
            emit(Measure(position[op.qubit], op.bit, op.condition))
        elif isinstance(op, Reset):
            emit(Reset(position[op.qubit], op.condition))
        elif isinstance(op, Noise):
            emit(Noise(position[op.qubit], op.kind, op.probability, op.condition))
        else:  # pragma: no cover - Circuit.validate rejects other ops
            raise PlacementError(f"cannot place unknown operation {op!r}")

    before = Circuit(ops=tuple(emitted), num_qubits=num_nodes, num_bits=circuit.num_bits,
                     ancillas=tuple(position[ancilla] for ancilla in circuit.ancillas),
                     metadata={"generator": "vqc.placement", "connectivity": connectivity.name,
                               "restored": "False"})
    depth_before_restore = before.depth()

    if restore:
        # Reversing the emitted swaps makes the whole swap product telescope to the
        # identity, so the mapped circuit is the original unitary on the same wires.
        for first, second in reversed(routing_swaps):
            emit_swap(first, second, record=False)
            restore_swaps += 1

    mapped = Circuit(ops=tuple(emitted), num_qubits=num_nodes, num_bits=circuit.num_bits,
                     ancillas=tuple(position[ancilla] for ancilla in circuit.ancillas),
                     metadata={"generator": "vqc.placement", "connectivity": connectivity.name,
                               "restored": "True" if restore else "False"})
    mapped.validate()

    permutation = tuple(position[logical] for logical in range(circuit.num_qubits))
    if restore and any(node != index for index, node in enumerate(permutation)):
        raise PlacementError(  # pragma: no cover - the reverse product is the identity
            f"restoring the routing permutation left {permutation}")

    resources = mapped.resources()
    return Placement(
        connectivity=connectivity.name,
        num_qubits=circuit.num_qubits,
        circuit=mapped,
        logical_depth=circuit.depth(),
        depth_before_restore=depth_before_restore,
        mapped_depth=mapped.depth(),
        logical_operations=len(circuit.ops),
        mapped_operations=resources["operations"],
        logical_two_qubit_gates=int(circuit.resources()["two_qubit_gates"]),
        two_qubit_gates=int(resources["two_qubit_gates"]),
        swap_gates=len(routing_swaps),
        restore_swap_gates=restore_swaps,
        longest_hop=longest_hop,
        restored=restore,
        permutation=permutation,
        connectivity_details=connectivity.as_dict(),
        router=strategy,
        lookahead_window=lookahead if strategy == "lookahead" else None,
        swap_entanglers=swap_entanglers,
    )
