"""Circuit synthesis from a mathematical specification, with no reference circuit.

Status: experimental

The discovery loop in :mod:`apedp_dyn.vqc.clifford_search` *optimises a supplied
circuit*: it starts from a reference, proposes local rewrites and certifies each one
against that reference. This module answers the next question, and it is the one the
review put first:

    given only a description of the state to prepare, can the platform *construct*
    a circuit, and decide whether the construction is correct?

There is no reference circuit anywhere in this module, and that is a structural
property rather than a promise: a :class:`TargetSpecification` is a name, a qubit
count, a set of signed Pauli *stabilizers* and (optionally) the outcome support -- the
mathematics of the state, not a circuit that prepares it. A target is therefore also
checkable in itself: :meth:`TargetSpecification.validate` feeds its stabilizers to the
stabilizer tableau's structural check, so a specification that does not describe a
pure state (anticommuting or dependent generators) is refused before any search runs.

Two referees decide a construction, and neither needs a reference:

* **stabilizer tier, any size** -- build the state the circuit prepares and require
  ``phase_of(stabilizer) == +1`` for every specified stabilizer. This is exact
  integer arithmetic on the same tableau :mod:`apedp_dyn.vqc.stabilizer` uses, so a
  1,000-qubit construction is decided as cheaply as a 10-qubit one.
* **dense tier, at most :data:`DENSE_SPEC_LIMIT` qubits** -- solve the specification
  for its state *vector* by projecting ``|0...0>`` onto ``+1`` eigenspaces of the
  Pauli matrices, and compare it with the prepared amplitudes up to one global phase.
  This tier decides non-Clifford constructions, which the images cannot, and it is
  the only part of the platform that can check a circuit containing ``t``/``rx``/``ccx``
  against a specification instead of against another circuit.

The constructions this module can build are a *grammar*, not a general synthesiser:
for a GHZ target, a spanning tree of ``cx`` gates rooted at the Hadamard, in chain,
star, balanced-binary and seeded-random wireings; for a ring graph state, the ring
itself; for any graph state -- including a two-dimensional cluster and an irregular
graph -- ``h`` on every qubit followed by the ``cz`` edges of the graph, emitted in
five different schedules. That is enough to make the point the review asked for --
the engine reaches the target without being shown a circuit, and it can say which
construction is cheapest in depth, in operations and in connectivity -- and it is
not enough to claim general synthesis. The record says so, and so does §7.

For a graph-state target the wiring is not a choice the grammar makes: the
specification names every edge, so all its families carry the same edge set and the
same connectivity, and the only free variable is the **schedule**. That makes
Experiment E a scheduling measurement, and it is why the module reports a lower
bound on the entangling depth (the maximum degree of the graph: a qubit with ``d``
neighbours needs ``d`` distinct ``cz`` layers) beside the depth it achieved.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.statevector_sim import StatevectorSimulator
from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.colouring import greedy_peel, minimum_edge_colouring
from apedp_dyn.vqc.discovery import (
    CandidateVerdict,
    PromotionRefused,
    promote,
)
from apedp_dyn.vqc.equivalence import MAX_UNITARY_QUBITS
from apedp_dyn.vqc.gates import apply_gate, matrix_for
from apedp_dyn.vqc.receipts import Check
from apedp_dyn.vqc.stabilizer import (
    I,
    StabilizerState,
    X,
    Y,
    Z,
    is_clifford_circuit,
    pauli_string,
)

__all__ = [
    "DENSE_SPEC_LIMIT",
    "FAMILIES",
    "GRAPH_TARGETS",
    "REFUSAL_FAMILIES",
    "SYNTHESIS_GRAPH_TARGETS",
    "Construction",
    "DenseTierCost",
    "I",
    "RefereeAgreement",
    "SpecificationCandidate",
    "SpecificationReport",
    "SynthesisError",
    "SynthesisOutcome",
    "TargetSpecification",
    "X",
    "Y",
    "Z",
    "adjudicate_specification",
    "adjudicate_specification_candidate",
    "build",
    "cluster_state_target",
    "default_families",
    "dense_tier_cost",
    "entangling_depth_lower_bound",
    "family_targets",
    "ghz_target",
    "graph_of_state",
    "graph_state_target",
    "irregular_graph_target",
    "lattice_edges",
    "referee_agreement",
    "ring_graph_target",
    "seeded_graph_target",
    "small_world_shape",
    "specification_state_vector",
    "synthesize",
    "watts_strogatz_target",
]

#: Widest circuit the dense specification referee will materialise. The state is
#: ``2**n`` amplitudes and the projection is ``n`` matrix applications of it.
DENSE_SPEC_LIMIT = min(10, MAX_UNITARY_QUBITS)

#: The tolerance for a dense state comparison. It is not a tunable: the amplitudes are
#: computed by exact matrix arithmetic on both sides, and anything above rounding is a
#: different state.
DENSE_TOLERANCE = 1e-9


class SynthesisError(ValueError):
    """A specification that is not a state, or a family asked to build the wrong target."""


# --------------------------------------------------------------------------- target
@dataclasses.dataclass(frozen=True)
class TargetSpecification:
    """What to prepare, mathematically: signed stabilizers, plus optional support.

    ``stabilizers`` is ``(((labels...), phase), ...)`` in the tableau's own encoding
    (``I, X, Y, Z`` are ``0..3`` and ``phase`` is the power of ``i``), so the
    specification and the referee speak one language. Exactly ``num_qubits``
    independent commuting generators are required -- that is a pure state, and
    anything else is refused rather than approximated.
    """

    name: str
    num_qubits: int
    stabilizers: tuple[tuple[tuple[int, ...], int], ...]
    support: Mapping[tuple[bool, ...], float] | None = None
    #: What the target decides, in words, for the report.
    decides: str = "the prepared state, by its stabilizer group"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise SynthesisError(f"num_qubits must be positive, got {self.num_qubits}")
        if len(self.stabilizers) != self.num_qubits:
            raise SynthesisError(
                f"a pure state on {self.num_qubits} qubits needs {self.num_qubits} "
                f"stabilizers, got {len(self.stabilizers)}")
        for labels, phase in self.stabilizers:
            if len(labels) != self.num_qubits:
                raise SynthesisError(
                    f"a stabilizer has {len(labels)} labels, expected {self.num_qubits}")
            if any(label not in (I, X, Y, Z) for label in labels):
                raise SynthesisError(f"stabilizer labels are not I/X/Y/Z: {labels}")
            if int(phase) % 4 not in (0, 2):
                raise SynthesisError(
                    f"a specification is a projector onto a Hermitian +-1 eigenspace; "
                    f"phase {phase} is imaginary")
        if self.support is not None:
            total = sum(self.support.values())
            if abs(total - 1.0) > 1e-12:
                raise SynthesisError(f"the declared support sums to {total}, not 1")
            for bits in self.support:
                if len(bits) != self.num_qubits:
                    raise SynthesisError(
                        f"a declared outcome has {len(bits)} bits, expected {self.num_qubits}")
        # The specification is checked as a state, not as a wish.
        self.stabilizer_state().validate()

    # ------------------------------------------------------------------ accessors
    def stabilizer_state(self) -> StabilizerState:
        """The state described by this specification, as a tableau."""
        state = StabilizerState(self.num_qubits)
        state.generators = [tuple(int(label) for label in labels)
                            for labels, _phase in self.stabilizers]
        state.phases = [int(phase) % 4 for _labels, phase in self.stabilizers]
        return state

    def stabilizer_strings(self) -> tuple[str, ...]:
        return tuple(pauli_string(labels, phase) for labels, phase in self.stabilizers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "decides": self.decides,
            "stabilizers": [pauli_string(labels, phase)
                            for labels, phase in self.stabilizers],
            "support": (None if self.support is None
                        else {_bits_text(bits): float(probability)
                              for bits, probability in sorted(self.support.items())}),
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()

    def support_matches(self, distribution: Mapping[tuple[bool, ...], float],
                        *, tolerance: float = 1e-15) -> tuple[bool, dict[str, Any]]:
        """Whether an exact distribution is the declared support, and the evidence."""
        if self.support is None:
            raise SynthesisError(f"target {self.name!r} declares no support")
        missing = [bits for bits in self.support if distribution.get(bits, 0.0) == 0.0]
        extra = [bits for bits, probability in distribution.items()
                 if probability > 0.0 and bits not in self.support]
        wrong = {_bits_text(bits): {"declared": float(self.support[bits]),
                                    "measured": float(distribution.get(bits, 0.0))}
                 for bits in self.support
                 if abs(float(self.support[bits]) - float(distribution.get(bits, 0.0)))
                 > tolerance}
        evidence = {"declared_outcomes": len(self.support),
                    "measured_outcomes": len(distribution),
                    "missing": [_bits_text(bits) for bits in missing[:4]],
                    "unexpected": [_bits_text(bits) for bits in extra[:4]],
                    "probabilities_off": wrong}
        return (not missing and not extra and not wrong), evidence


def _bits_text(bits: Sequence[bool]) -> str:
    return "".join("1" if bit else "0" for bit in bits)


def ghz_target(num_qubits: int) -> TargetSpecification:
    """``(|0...0> + |1...1>)/sqrt(2)``, described without a circuit.

    The stabilizers are the textbook ones -- ``X`` on every qubit and ``Z_q Z_{q+1}``
    on every neighbouring pair -- which the platform already asserts against sampled
    runs, so the specification and the closed-form check are the same mathematics.
    """
    if num_qubits < 2:
        raise SynthesisError(f"a GHZ state needs at least two qubits, got {num_qubits}")
    x_all = tuple([X] * num_qubits)
    stabilizers: list[tuple[tuple[int, ...], int]] = [(x_all, 0)]
    for qubit in range(num_qubits - 1):
        labels = [I] * num_qubits
        labels[qubit] = Z
        labels[qubit + 1] = Z
        stabilizers.append((tuple(labels), 0))
    support = None
    if num_qubits <= 12:
        support = {tuple([False] * num_qubits): 0.5, tuple([True] * num_qubits): 0.5}
    return TargetSpecification(
        name="ghz",
        num_qubits=num_qubits,
        stabilizers=tuple(stabilizers),
        support=support,
        decides="the GHZ state, up to one global phase",
    )


def lattice_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """The nearest-neighbour edges of a ``rows`` x ``cols`` square lattice, open boundary.

    Qubit ``(r, c)`` is ``r * cols + c``, so the wiring of a row-major register is also
    the labelling: a vertical edge spans exactly ``cols`` wires, which is the
    connectivity cost the placement experiment has to pay for.
    """
    if rows < 1 or cols < 1:
        raise SynthesisError(f"a lattice needs positive dimensions, got {rows}x{cols}")
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(cols):
            here = row * cols + column
            if column + 1 < cols:
                edges.append((here, here + 1))
            if row + 1 < rows:
                edges.append((here, here + cols))
    return tuple(edges)


def graph_state_target(
    name: str,
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    decides: str | None = None,
) -> TargetSpecification:
    """A graph state: ``X_v Z_{N(v)}`` on every vertex, the graph given by its edges.

    This is the general form behind the ring and the two-dimensional cluster, and it
    is written as mathematics rather than as a circuit: the edges are not a build
    plan, they are part of what the state *is*. A self-loop, a repeated edge, an
    endpoint out of range and a disconnected graph are refused -- a graph state on a
    disconnected graph is a product state, which is a legitimate state but is never
    what a caller asking for one graph means.
    """
    if num_qubits < 2:
        raise SynthesisError(f"a graph state needs at least two qubits, got {num_qubits}")
    neighbours: list[set[int]] = [set() for _ in range(num_qubits)]
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        pair = tuple(int(end) for end in edge)
        if len(pair) != 2:
            raise SynthesisError(f"an edge needs two endpoints, got {edge}")
        parent, child = pair
        if parent == child:
            raise SynthesisError(f"a self-loop is not an edge of a graph state: {pair}")
        if not (0 <= parent < num_qubits and 0 <= child < num_qubits):
            raise SynthesisError(f"edge {pair} is out of range for {num_qubits} qubits")
        key = (min(pair), max(pair))
        if key in seen:
            raise SynthesisError(f"edge {key} is listed twice")
        seen.add(key)
        neighbours[parent].add(child)
        neighbours[child].add(parent)
    if num_qubits > 1:
        reached = {0}
        frontier = [0]
        while frontier:
            vertex = frontier.pop()
            for neighbour in neighbours[vertex]:
                if neighbour not in reached:
                    reached.add(neighbour)
                    frontier.append(neighbour)
        if len(reached) != num_qubits:
            raise SynthesisError(
                f"the graph has {len(reached)} of {num_qubits} vertices connected; a "
                "disconnected graph state is a product state, not one preparation")
    stabilizers = []
    for vertex in range(num_qubits):
        labels = [I] * num_qubits
        labels[vertex] = X
        for neighbour in neighbours[vertex]:
            labels[neighbour] = Z
        stabilizers.append((tuple(labels), 0))
    return TargetSpecification(
        name=name,
        num_qubits=num_qubits,
        stabilizers=tuple(stabilizers),
        support=None,
        decides=decides or f"the {name} graph state, up to one global phase",
    )


def cluster_state_target(rows: int, cols: int) -> TargetSpecification:
    """A two-dimensional cluster state on a ``rows`` x ``cols`` lattice, open boundary.

    Qubits sit on the **vertices** of the lattice and ``cz`` acts on nearest
    neighbours; this is the graph state of the grid graph. It is *not* the
    surface-code cluster whose qubits sit on the edges of a lattice, and the name is
    spelled out here so the record cannot be read as claiming that one.
    """
    if rows * cols < 3:
        raise SynthesisError(f"a 2-D cluster needs at least three qubits, got {rows}x{cols}")
    return graph_state_target(
        "cluster_2d",
        rows * cols,
        lattice_edges(rows, cols),
        decides=f"the {rows}x{cols} grid-graph cluster state, up to one global phase",
    )


def irregular_graph_target(
    num_qubits: int, edges: Sequence[tuple[int, int]], *, decides: str | None = None
) -> TargetSpecification:
    """A graph state on a graph with no regular structure, given edge by edge.

    The target's ``name`` stays ``irregular_graph`` because that is what the
    construction families dispatch on; which irregular graph it is, is carried by its
    stabilizers and therefore by its SHA-256, and described in ``decides``.
    """
    return graph_state_target("irregular_graph", num_qubits, edges, decides=decides)


def seeded_graph_target(
    num_qubits: int,
    *,
    seed: int = 0,
    extra_edge_probability: float = 0.15,
    decides: str | None = None,
) -> TargetSpecification:
    """An irregular graph state, reproducible from its seed.

    A random spanning tree (each qubit attaches to an earlier one) guarantees
    connectivity; extra edges are then added independently with the given
    probability, so the graph has no regular structure, several vertices of differing
    degree, and a maximum degree that is a property of the draw rather than of the
    construction. The seed is the whole specification: two runs with it agree edge for
    edge, and the record cites it.
    """
    if num_qubits < 3:
        raise SynthesisError(f"an irregular graph needs at least three qubits, got {num_qubits}")
    if not 0.0 <= extra_edge_probability <= 1.0:
        raise SynthesisError(
            f"extra_edge_probability must be in [0, 1], got {extra_edge_probability}")
    rng = random.Random(seed)
    edges: list[tuple[int, int]] = []
    for child in range(1, num_qubits):
        edges.append((rng.randrange(child), child))
    present = {tuple(sorted(edge)) for edge in edges}
    for parent in range(num_qubits):
        for child in range(parent + 1, num_qubits):
            if (parent, child) in present:
                continue
            if rng.random() < extra_edge_probability:
                edges.append((parent, child))
                present.add((parent, child))
    return graph_state_target("irregular_graph", num_qubits, edges, decides=decides)


def watts_strogatz_target(
    num_qubits: int,
    *,
    k: int = 4,
    beta: float = 0.1,
    seed: int = 0,
    decides: str | None = None,
) -> TargetSpecification:
    """A Watts-Strogatz small-world graph state, reproducible from ``(k, beta, seed)``.

    The second graph source in Experiment E, and a *different* graph family rather than
    another draw from the seeded-tree generator: every vertex starts on a ring joined to
    its ``k / 2`` nearest neighbours on each side, then each ring edge is rewired with
    probability ``beta`` to a uniformly chosen non-neighbour. ``beta = 0`` is the ring
    lattice, ``beta = 1`` is a random graph, and ``0 < beta < 1`` is the small-world
    regime the family exists to cover -- high clustering with short paths.

    The draw is the specification: ``(k, beta, seed)`` reproduces the edge set exactly,
    and the edge set is what the stabilizers encode, so the target's SHA-256 pins it. A
    rewire that would strand a vertex is skipped (a graph state on a graph with an
    isolated vertex is disconnected and would be refused by
    :func:`graph_state_target` anyway, but the draw should not produce one silently).
    """
    if num_qubits < 4:
        raise SynthesisError(f"a small-world graph needs at least four qubits, got {num_qubits}")
    if k < 2 or k % 2:
        raise SynthesisError(f"the ring degree k must be even and at least 2, got {k}")
    if k >= num_qubits:
        raise SynthesisError(f"k must be smaller than the qubit count, got k={k} on {num_qubits}")
    if not 0.0 <= beta <= 1.0:
        raise SynthesisError(f"the rewiring probability must be in [0, 1], got {beta}")
    rng = random.Random(seed)
    neighbours: list[set[int]] = [set() for _ in range(num_qubits)]
    ring: list[tuple[int, int]] = []
    for offset in range(1, k // 2 + 1):
        for vertex in range(num_qubits):
            other = (vertex + offset) % num_qubits
            neighbours[vertex].add(other)
            neighbours[other].add(vertex)
            ring.append((vertex, other))
    for vertex, other in ring:
        if rng.random() >= beta:
            continue
        if other not in neighbours[vertex]:
            continue  # already rewired away by the other endpoint's turn
        candidates = [w for w in range(num_qubits)
                      if w != vertex and w not in neighbours[vertex]]
        if not candidates:
            continue
        target = rng.choice(candidates)
        if len(neighbours[other]) < 2:
            continue  # never strand a vertex the draw has already thinned
        neighbours[vertex].discard(other)
        neighbours[other].discard(vertex)
        neighbours[vertex].add(target)
        neighbours[target].add(vertex)
    edges = sorted({(min(a, b), max(a, b))
                    for a in range(num_qubits) for b in neighbours[a]})
    return graph_state_target(
        "watts_strogatz", num_qubits, edges,
        decides=(decides or f"the Watts-Strogatz graph state on {num_qubits} qubits, "
                             f"k={k}, beta={beta}, seed={seed}, up to one global phase"),
    )


def small_world_shape(
    num_qubits: int, *, k: int = 4, beta: float = 0.1, seed: int = 0
) -> dict[str, float]:
    """The two statistics that make a Watts-Strogatz draw a *small-world* draw.

    Mean local clustering (the average over vertices of the fraction of neighbour pairs
    that are themselves joined) and mean degree. A ring lattice (`beta = 0`) has high
    clustering; a random graph (`beta = 1`) has low clustering; the small-world regime
    sits between them at the same degree, which is the contrast the record measures
    rather than asserts.
    """
    target = watts_strogatz_target(num_qubits, k=k, beta=beta, seed=seed)
    edges = graph_of_state(target)
    adjacent: list[set[int]] = [set() for _ in range(num_qubits)]
    for parent, child in edges:
        adjacent[parent].add(child)
        adjacent[child].add(parent)
    clustering: list[float] = []
    for vertex in range(num_qubits):
        ring = sorted(adjacent[vertex])
        degree = len(ring)
        if degree < 2:
            continue
        joined = sum(1 for i in range(degree) for j in range(i + 1, degree)
                     if ring[j] in adjacent[ring[i]])
        clustering.append(2.0 * joined / (degree * (degree - 1)))
    return {
        "edges": float(len(edges)),
        "mean_degree": 2.0 * len(edges) / num_qubits,
        "mean_clustering": (sum(clustering) / len(clustering) if clustering else 0.0),
        "min_degree": float(min(len(a) for a in adjacent)),
        "max_degree": float(max(len(a) for a in adjacent)),
    }


def graph_of_state(target: TargetSpecification) -> tuple[tuple[int, int], ...]:
    """Recover the graph from a graph-state specification's stabilizers.

    Every stabilizer of a graph state is ``X_v`` on exactly one qubit and ``Z`` on its
    neighbours, so the wiring is *in* the specification -- the construction families
    below never receive a graph, they read one. A specification that is not of that
    form (a ``Y``, two ``X``s, a ``Z`` on a vertex that does not list this one back) is
    refused by name rather than approximated, which is what stops a family from
    quietly building the wrong state for a target it half-recognises.
    """
    num_qubits = target.num_qubits
    neighbours: list[set[int]] = [set() for _ in range(num_qubits)]
    placed: set[int] = set()
    for labels, phase in target.stabilizers:
        if int(phase) % 4 != 0:
            raise SynthesisError(
                f"a graph-state stabilizer is a +1 eigenstate; {pauli_string(labels, phase)} "
                "has an imaginary sign")
        if any(label == Y for label in labels):
            raise SynthesisError(
                f"{pauli_string(labels, 0)} contains a Y, so it is not of the form "
                "X_v Z_N(v)")
        anchor = [index for index, label in enumerate(labels) if label == X]
        if len(anchor) != 1:
            raise SynthesisError(
                f"{pauli_string(labels, 0)} has {len(anchor)} X positions; a graph-state "
                "stabilizer has exactly one")
        vertex = anchor[0]
        if vertex in placed:
            raise SynthesisError(f"qubit {vertex} appears in two stabilizers")
        placed.add(vertex)
        neighbours[vertex] = {
            index for index, label in enumerate(labels) if index != vertex and label == Z
        }
    if len(placed) != num_qubits:  # pragma: no cover - every generator places one
        raise SynthesisError("the specification does not place every qubit")
    edges = set()
    for vertex, adjacent in enumerate(neighbours):
        for neighbour in adjacent:
            if vertex not in neighbours[neighbour]:
                raise SynthesisError(
                    f"the specification is asymmetric on ({vertex}, {neighbour}): one "
                    "stabilizer names the other but not the reverse")
            edges.add((min(vertex, neighbour), max(vertex, neighbour)))
    return tuple(sorted(edges))


def entangling_depth_lower_bound(target: TargetSpecification) -> dict[str, Any]:
    """A proven floor on the ``cz`` layers, and where it comes from.

    A qubit with ``d`` neighbours must take part in ``d`` ``cz`` gates, and no two of
    them can share a layer, so ``d`` layers are unavoidable. This is a lower bound on
    *entangling depth*, not on total depth (the ``h`` layer is the caller's choice),
    and it is only informative for a graph-state target.
    """
    edges = graph_of_state(target)
    degrees: dict[int, int] = dict.fromkeys(range(target.num_qubits), 0)
    for parent, child in edges:
        degrees[parent] += 1
        degrees[child] += 1
    degree = max(degrees.values(), default=0)
    busiest = sorted(qubit for qubit, value in degrees.items() if value == degree)
    return {
        "edges": len(edges),
        "max_degree": degree,
        "busiest_qubits": busiest[:4],
        "cz_layers_floor": degree,
        "reason": ("a qubit with d neighbours needs d cz gates, and they cannot share "
                   "a layer with one another"),
    }


def ring_graph_target(num_qubits: int) -> TargetSpecification:
    """A ring graph state: ``X_q Z_{q-1} Z_{q+1}`` on every qubit, a *different* state.

    Included so the synthesis record cannot be read as "GHZ works": the second target
    uses a different construction (``h`` everywhere plus a ``cz`` ring), a different
    stabilizer shape, and no declared support at width.
    """
    if num_qubits < 3:
        raise SynthesisError(f"a ring needs at least three qubits, got {num_qubits}")
    stabilizers = []
    for qubit in range(num_qubits):
        labels = [I] * num_qubits
        labels[qubit] = X
        # For n >= 3 the two neighbours of a qubit are always distinct qubits.
        labels[(qubit - 1) % num_qubits] = Z
        labels[(qubit + 1) % num_qubits] = Z
        stabilizers.append((tuple(labels), 0))
    return TargetSpecification(
        name="ring_graph",
        num_qubits=num_qubits,
        stabilizers=tuple(stabilizers),
        support=None,
        decides="the ring graph state, up to one global phase",
    )


# ---------------------------------------------------------------- dense oracle
_PAULI_CACHE: dict[int, np.ndarray] = {}


def _one_qubit_pauli(label: int) -> np.ndarray:
    if label == I:
        return np.eye(2, dtype=np.complex128)
    name = {X: "x", Y: "y", Z: "z"}[label]
    cached = _PAULI_CACHE.get(label)
    if cached is None:
        cached = np.asarray(matrix_for(name), dtype=np.complex128)
        _PAULI_CACHE[label] = cached
    return cached


def _pauli_matrix(labels: Sequence[int]) -> np.ndarray:
    matrix = _one_qubit_pauli(int(labels[0]))
    for label in labels[1:]:
        matrix = np.kron(matrix, _one_qubit_pauli(int(label)))
    return matrix


def specification_state_vector(target: TargetSpecification) -> np.ndarray:
    """The state the specification describes, by projection: an independent oracle.

    Starts from ``|0...0>`` and projects onto the ``+1`` eigenspace of each stabilizer
    in turn, ``(I + (-1)**phase P)/2``, renormalising. A specification whose
    eigenspaces do not intersect produces a zero vector at some step, and that is
    refused rather than returned as a small number.

    Above :data:`DENSE_SPEC_LIMIT` this refuses to materialise anything. The bound is
    a policy on the *referee*, and it is not lifted here or anywhere a verdict is
    reached: the only caller that reaches past it is :func:`dense_tier_cost`, which
    measures what the arithmetic costs and returns no verdict at all.
    """
    return _projected_state_vector(target, enforce_bound=True)


def _projected_state_vector(target: TargetSpecification, *,
                            enforce_bound: bool) -> np.ndarray:
    """Projection onto the specification's eigenspaces, with the bound optional.

    One implementation for both callers: the referee's oracle (bound enforced) and the
    cost probe (bound lifted), so the probe cannot drift into measuring different
    arithmetic from the tier it is measuring.
    """
    if enforce_bound and target.num_qubits > DENSE_SPEC_LIMIT:
        raise SynthesisError(
            f"materialising a {target.num_qubits}-qubit specification is 2**"
            f"{target.num_qubits} amplitudes; the dense bound is {DENSE_SPEC_LIMIT}")
    dimension = 2 ** target.num_qubits
    vector = np.zeros(dimension, dtype=np.complex128)
    vector[0] = 1.0
    for labels, phase in target.stabilizers:
        sign = 1.0 if int(phase) % 4 == 0 else -1.0
        projected = 0.5 * (vector + sign * (_pauli_matrix(labels) @ vector))
        norm = float(np.linalg.norm(projected))
        if norm < 1e-12:
            raise SynthesisError(
                f"the specification is not a state: stabilizer "
                f"{pauli_string(labels, phase)} has empty intersection with the "
                f"previous eigenspaces")
        vector = projected / norm
    _out: np.ndarray = vector
    return _out


@dataclasses.dataclass(frozen=True)
class DenseTierCost:
    """What one dense-tier decision costs, with **no verdict attached**.

    `adjudicate_specification` refuses the dense tier above :data:`DENSE_SPEC_LIMIT`,
    and that refusal is a policy: it says nothing about what the arithmetic would cost
    one qubit higher. This record carries the measurement the policy is based on, so
    the bound is a number with a wall behind it rather than a round decimal.

    ``pauli_operator_bytes`` is the dominant term and is not a rounding: the tier
    materialises each stabilizer as a dense ``2**n x 2**n`` matrix, which is
    ``16 * 4**n`` bytes, so the cost is quadratic in the Hilbert dimension, not linear.
    """

    qubits: int
    amplitudes: int
    seconds: float
    peak_traced_bytes: int
    distance: float
    within_tolerance: bool
    pauli_operator_bytes: int
    state_vector_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "amplitudes": self.amplitudes,
            "seconds": self.seconds,
            "peak_traced_bytes": self.peak_traced_bytes,
            "distance": self.distance,
            "within_tolerance": self.within_tolerance,
            "pauli_operator_bytes": self.pauli_operator_bytes,
            "state_vector_bytes": self.state_vector_bytes,
            "memory_model": "16 * 4**n for one Pauli operator, 16 * 2**n for the state",
        }


def dense_tier_cost(circuit: Circuit, target: TargetSpecification) -> DenseTierCost:
    """Run the dense tier's own arithmetic and measure it, at any width.

    The steps are exactly the dense branch of :func:`adjudicate_specification`: prepare
    the circuit's amplitudes in the specification's basis order, project ``|0...0>``
    onto the specification's eigenspaces, and minimise the phase-free distance. What
    is missing is the *verdict*: this returns seconds, peak traced bytes and the
    distance, and no `VALID`/`INVALID`, because a caller allowed to exceed the bound
    must not be able to turn that into an adjudication. No referee path calls it; its
    caller is `scripts/measure_vqc_synthesis.py`, which uses it to locate the width at
    which the tier stops being practical.

    Peak memory is measured with :mod:`tracemalloc`, which NumPy allocators report to,
    so it covers the ``2**n x 2**n`` Pauli matrix rather than only Python objects.
    """
    circuit.validate()
    if circuit.num_qubits != target.num_qubits:
        raise SynthesisError(
            f"the cost probe needs a {target.num_qubits}-qubit circuit, got "
            f"{circuit.num_qubits}")
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        start = time.perf_counter()
        prepared = _prepare_dense(circuit)
        expected = _projected_state_vector(target, enforce_bound=False)
        distance, _phase = _phase_free_distance(prepared, expected)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if started_tracing:
            tracemalloc.stop()
    qubits = target.num_qubits
    return DenseTierCost(
        qubits=qubits,
        amplitudes=2 ** qubits,
        seconds=seconds,
        peak_traced_bytes=int(peak),
        distance=distance,
        within_tolerance=bool(distance <= DENSE_TOLERANCE),
        pauli_operator_bytes=16 * 4 ** qubits,
        state_vector_bytes=16 * 2 ** qubits,
    )


# ---------------------------------------------------------------- constructions
@dataclasses.dataclass(frozen=True)
class Construction:
    """One circuit built from the specification, plus the wiring it used."""

    family: str
    circuit: Circuit
    edges: tuple[tuple[int, int], ...]
    notes: str = ""

    @property
    def operations(self) -> int:
        return len(self.circuit.ops)

    @property
    def depth(self) -> int:
        return self.circuit.depth()

    @property
    def two_qubit_gates(self) -> int:
        return int(self.circuit.resources()["two_qubit_gates"])

    @property
    def max_edge_span(self) -> int:
        """The longest wire a two-qubit gate reaches across: a connectivity cost."""
        return max((abs(parent - child) for parent, child in self.edges), default=0)

    @property
    def nearest_neighbour_only(self) -> bool:
        return all(abs(parent - child) == 1 for parent, child in self.edges)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "circuit_sha256": self.circuit.sha256(),
            "operations": self.operations,
            "depth": self.depth,
            "two_qubit_gates": self.two_qubit_gates,
            "edges": [list(edge) for edge in self.edges],
            "max_edge_span": self.max_edge_span,
            "nearest_neighbour_only": self.nearest_neighbour_only,
            "notes": self.notes,
        }


def _tree_circuit(num_qubits: int, edges: Sequence[tuple[int, int]], family: str,
                  notes: str) -> Construction:
    """`H` on the root, then `CX` along each edge: a GHZ state on any spanning tree.

    This is the construction the search *discovers*, not one it is handed: the root
    is the only node with no incoming edge, and the edges arrive parent-first, so the
    circuit is determined by the wiring alone.
    """
    root = edges[0][0] if edges else 0
    ops: list[Any] = [Gate("h", (root,))]
    ops.extend(Gate("cx", (parent, child)) for parent, child in edges)
    return Construction(
        family=family,
        circuit=Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0,
                        metadata={"generator": "vqc.synthesis.tree", "family": family}),
        edges=tuple(edges),
        notes=notes,
    )


def _ghz_chain(target: TargetSpecification, *, seed: int = 0) -> Construction:
    num_qubits = target.num_qubits
    edges = [(qubit, qubit + 1) for qubit in range(num_qubits - 1)]
    return _tree_circuit(num_qubits, edges, "chain",
                         "linear chain, nearest-neighbour only, depth n")


def _ghz_reverse_chain(target: TargetSpecification, *, seed: int = 0) -> Construction:
    num_qubits = target.num_qubits
    edges = [(qubit, qubit - 1) for qubit in range(num_qubits - 1, 0, -1)]
    return _tree_circuit(num_qubits, edges, "reverse_chain",
                         "the same chain with the root at the other end")


def _ghz_star(target: TargetSpecification, *, seed: int = 0) -> Construction:
    num_qubits = target.num_qubits
    edges = [(0, qubit) for qubit in range(1, num_qubits)]
    # The fan-out is shorter in wire length than a chain but not in depth: every `cx`
    # shares qubit 0 as its control, so the gates serialise and the depth is n. That
    # is a measurement this module reports rather than a property to assume.
    return _tree_circuit(num_qubits, edges, "star",
                         "all-to-one fan-out: span n-1, but the shared control makes it "
                         "depth n, not depth 2")


def _ghz_balanced(target: TargetSpecification, *, seed: int = 0) -> Construction:
    num_qubits = target.num_qubits
    edges = [((child - 1) // 2, child) for child in range(1, num_qubits)]
    return _tree_circuit(num_qubits, edges, "balanced_tree",
                         "binary-heap tree: depth about log2(n), span about n/2")


def _ghz_random_tree(target: TargetSpecification, *, seed: int = 0) -> Construction:
    num_qubits = target.num_qubits
    rng = random.Random(seed)
    edges = [(rng.randrange(child), child) for child in range(1, num_qubits)]
    return _tree_circuit(num_qubits, edges, "random_tree",
                         f"seeded random spanning tree (seed {seed})")


def _ring_ring(target: TargetSpecification, *, seed: int = 0) -> Construction:
    num_qubits = target.num_qubits
    ops: list[Any] = [Gate("h", (qubit,)) for qubit in range(num_qubits)]
    edges = [(qubit, (qubit + 1) % num_qubits) for qubit in range(num_qubits)]
    ops.extend(Gate("cz", edge) for edge in edges)
    return Construction(
        family="ring",
        circuit=Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0,
                        metadata={"generator": "vqc.synthesis.ring", "family": "ring"}),
        edges=tuple(edges),
        notes="h on every qubit, then the cz ring that closes the loop",
    )


def _ring_linear(target: TargetSpecification, *, seed: int = 0) -> Construction:
    """The same construction with the closing edge missing: a *refused* candidate."""
    num_qubits = target.num_qubits
    ops: list[Any] = [Gate("h", (qubit,)) for qubit in range(num_qubits)]
    edges = [(qubit, qubit + 1) for qubit in range(num_qubits - 1)]
    ops.extend(Gate("cz", edge) for edge in edges)
    return Construction(
        family="linear_cluster",
        circuit=Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0,
                        metadata={"generator": "vqc.synthesis.ring",
                                  "family": "linear_cluster"}),
        edges=tuple(edges),
        notes="a linear cluster: the ring's closing edge is missing",
    )


# -------------------------------------------------------------------- graph states
def _graph_circuit(target: TargetSpecification,
                   ordered_edges: Sequence[tuple[int, int]], family: str,
                   notes: str) -> Construction:
    """`h` on every qubit, then the graph's `cz` edges in the given order.

    The edge *set* is the specification's, so the only thing a family chooses is the
    order the commuting `cz`s are emitted in -- which is exactly what decides the
    depth, because the circuit's depth is the schedule of a sequential list. That is
    the whole of Experiment E's question, and it is why these families are schedules
    rather than wirings.
    """
    ops: list[Any] = [Gate("h", (qubit,)) for qubit in range(target.num_qubits)]
    ops.extend(Gate("cz", edge) for edge in ordered_edges)
    return Construction(
        family=family,
        circuit=Circuit(ops=tuple(ops), num_qubits=target.num_qubits, num_bits=0,
                        metadata={"generator": "vqc.synthesis.graph", "family": family}),
        edges=tuple(ordered_edges),
        notes=notes,
    )


def _breadth_first_tree(target: TargetSpecification,
                        edges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """A spanning tree in breadth-first order from qubit 0, as canonical edges."""
    adjacency: dict[int, list[int]] = {qubit: [] for qubit in range(target.num_qubits)}
    for parent, child in edges:
        adjacency[parent].append(child)
        adjacency[child].append(parent)
    reached = {0}
    queue = [0]
    tree: list[tuple[int, int]] = []
    while queue:
        vertex = queue.pop(0)
        for neighbour in sorted(adjacency[vertex]):
            if neighbour in reached:
                continue
            reached.add(neighbour)
            tree.append((min(vertex, neighbour), max(vertex, neighbour)))
            queue.append(neighbour)
    return tree


def _graph_raster(target: TargetSpecification, *, seed: int = 0) -> Construction:
    edges = sorted(graph_of_state(target))
    return _graph_circuit(target, edges, "graph_raster",
                          "edges in ascending wire order: the schedule a row-major walk "
                          "of the register produces")


def _graph_column_major(target: TargetSpecification, *, seed: int = 0) -> Construction:
    edges = sorted(graph_of_state(target), key=lambda edge: (max(edge), min(edge)))
    return _graph_circuit(target, edges, "graph_column_major",
                          "the same edges ordered by their higher endpoint: the "
                          "transposed walk, which is a genuinely different schedule")


def _graph_tree_first(target: TargetSpecification, *, seed: int = 0) -> Construction:
    graph = graph_of_state(target)
    tree = _breadth_first_tree(target, graph)
    rest = [edge for edge in sorted(graph) if edge not in set(tree)]
    return _graph_circuit(
        target, [*tree, *rest], "graph_tree_first",
        f"grow a breadth-first spanning tree first ({len(tree)} edges), then close the "
        f"remaining {len(rest)}: the schedule a builder that starts from one qubit emits")


def _graph_layers(target: TargetSpecification, *, seed: int = 0) -> Construction:
    """The minimum edge colouring: ``chi'`` layers, with the argument for that count.

    A layer of ``cz`` gates that do not touch is a matching, so a schedule is an edge
    colouring and the minimum schedule length is ``chi'(G)``. This family is the
    proven minimum, not the greedy peel: the peel is kept in `colouring` to measure
    what this saves, and stands only as the last resort if Vizing's construction fails
    its own check (in which case the notes say so rather than calling the result
    minimal).
    """
    graph = sorted(graph_of_state(target))
    schedule = minimum_edge_colouring(graph, num_nodes=target.num_qubits)
    peel = greedy_peel(graph)
    saved = len(peel) - schedule.num_colours
    verdict = (f"proven minimal ({schedule.reason})" if schedule.optimal
               else f"not proven minimal ({schedule.reason})")
    return _graph_circuit(
        target, [edge for layer in schedule.layers for edge in layer], "graph_layers",
        f"{schedule.num_colours} parallel cz layers against a degree floor of "
        f"{schedule.max_degree}; {verdict}; the greedy peel used {len(peel)} "
        f"layer(s), so this schedule saves {saved}")


def _graph_random_order(target: TargetSpecification, *, seed: int = 0) -> Construction:
    edges = sorted(graph_of_state(target))
    rng = random.Random(seed)
    rng.shuffle(edges)
    return _graph_circuit(target, edges, "graph_random_order",
                          f"the same edges in a seeded random order (seed {seed}): a "
                          "control, because it isolates the schedule from the state")


def _cluster_drop_edge(target: TargetSpecification, *, seed: int = 0) -> Construction:
    """The raster construction with one lattice edge missing: a *refused* candidate."""
    graph = sorted(graph_of_state(target))
    dropped = graph[len(graph) // 2]
    edges = [edge for edge in graph if edge != dropped]
    return _graph_circuit(
        target, edges, "cluster_drop_edge",
        f"the raster schedule with edge {dropped} removed: the stabilizers of its two "
        "endpoints can no longer be satisfied")


def _cluster_extra_edge(target: TargetSpecification, *, seed: int = 0) -> Construction:
    """Every lattice edge plus one that is not there: a *refused* candidate."""
    graph = set(graph_of_state(target))
    extra = None
    for parent in range(target.num_qubits):
        for child in range(parent + 2, target.num_qubits):
            if (parent, child) not in graph:
                extra = (parent, child)
                break
        if extra is not None:
            break
    if extra is None:  # pragma: no cover - a complete graph has no such pair
        raise SynthesisError("every pair is already an edge; no extra edge to add")
    edges = [*sorted(graph), extra]
    return _graph_circuit(
        target, edges, "cluster_extra_edge",
        f"the full edge set plus {extra}, which the specification does not name: a "
        "controlled phase the target does not have")


def _graph_swap_gate(target: TargetSpecification, *, seed: int = 0) -> Construction:
    """One `cz` replaced by `cx`: the same wiring, the wrong gate, a refusal."""
    graph = sorted(graph_of_state(target))
    swapped = graph[len(graph) // 2]
    ops: list[Any] = [Gate("h", (qubit,)) for qubit in range(target.num_qubits)]
    ops.extend(Gate("cx" if edge == swapped else "cz", edge) for edge in graph)
    return Construction(
        family="graph_swap_gate",
        circuit=Circuit(ops=tuple(ops), num_qubits=target.num_qubits, num_bits=0,
                        metadata={"generator": "vqc.synthesis.graph",
                                  "family": "graph_swap_gate"}),
        edges=tuple(graph),
        notes=f"edge {swapped} carries cx instead of cz: an entangling gate that is "
              "not the one the specification is built from")


#: The graph-state targets: one scheduling idea applied to two specifications, so a
#: family declared for both is a family, not a coincidence.
GRAPH_TARGETS = ("cluster_2d", "irregular_graph")
#: The graph targets the five *constructors* are declared for. Separate from
#: `GRAPH_TARGETS` because that tuple also scopes the refusal families: a
#: `watts_strogatz` target must be constructible, and must not acquire
#: `graph_swap_gate` -- a deliberately wrong circuit -- as one of its families.
SYNTHESIS_GRAPH_TARGETS = (*GRAPH_TARGETS, "watts_strogatz")

#: Families that are registered to be *refused*: near-misses that a resource-only
#: search would happily promote. They are built on purpose, adjudicated, and recorded
#: as `INVALID` -- a synthesis engine whose wrong constructions are never tested is
#: indistinguishable from one that accepts everything.
REFUSAL_FAMILIES = frozenset({"linear_cluster", "cluster_drop_edge", "cluster_extra_edge",
                             "graph_swap_gate"})

#: family name -> (target name(s) it builds, builder). A family asked to build another
#: target is a refusal, not a quietly wrong circuit. A family may serve several targets
#: when it reads the graph out of the specification instead of being handed one.
FAMILIES: dict[str, tuple[str | tuple[str, ...], Any]] = {
    "chain": ("ghz", _ghz_chain),
    "reverse_chain": ("ghz", _ghz_reverse_chain),
    "star": ("ghz", _ghz_star),
    "balanced_tree": ("ghz", _ghz_balanced),
    "random_tree": ("ghz", _ghz_random_tree),
    "ring": ("ring_graph", _ring_ring),
    "linear_cluster": ("ring_graph", _ring_linear),
    "graph_raster": (SYNTHESIS_GRAPH_TARGETS, _graph_raster),
    "graph_column_major": (SYNTHESIS_GRAPH_TARGETS, _graph_column_major),
    "graph_tree_first": (SYNTHESIS_GRAPH_TARGETS, _graph_tree_first),
    "graph_layers": (SYNTHESIS_GRAPH_TARGETS, _graph_layers),
    "graph_random_order": (SYNTHESIS_GRAPH_TARGETS, _graph_random_order),
    "cluster_drop_edge": (("cluster_2d",), _cluster_drop_edge),
    "cluster_extra_edge": (("cluster_2d",), _cluster_extra_edge),
    "graph_swap_gate": (GRAPH_TARGETS, _graph_swap_gate),
}


def family_targets(family: str) -> tuple[str, ...]:
    """The target names a family is declared for, whatever shape the registry entry has."""
    if family not in FAMILIES:
        raise SynthesisError(
            f"unknown construction family {family!r}; known: {sorted(FAMILIES)}")
    declared = FAMILIES[family][0]
    return (declared,) if isinstance(declared, str) else tuple(declared)


def build(family: str, target: TargetSpecification, *, seed: int = 0) -> Construction:
    """Build one construction family for `target`, or refuse by name."""
    targets = family_targets(family)
    if target.name not in targets:
        raise SynthesisError(
            f"family {family!r} constructs a {targets[0]!r} state, but the target is "
            f"{target.name!r}")
    _out: Construction = FAMILIES[family][1](target, seed=seed)
    return _out


def default_families(target: TargetSpecification) -> tuple[str, ...]:
    """The construction families that apply to this target, in a stable order."""
    return tuple(name for name in FAMILIES if target.name in family_targets(name))


# --------------------------------------------------------------------- refereeing
def _is_unitary(circuit: Circuit) -> bool:
    return all(isinstance(op, Gate) and not op.condition for op in circuit.ops)


def _state_of(circuit: Circuit) -> StabilizerState:
    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        assert isinstance(op, Gate)  # callers reject non-unitary ops first
        state.apply_gate(op.name, op.targets, op.params)
    return state


def _specification_order(raw: np.ndarray, num_qubits: int) -> np.ndarray:
    """Re-index amplitudes from the simulator's basis order to the specification's.

    :class:`StatevectorSimulator` documents its indexing as "bit i = qubit i": basis
    state ``sum(bit_q * 2**q)``. :func:`specification_state_vector` builds the state as
    a Kronecker product with qubit 0 in the **most** significant position, and
    :func:`dense_oracle_check` reads it that way. Comparing the two without this
    re-index compares a state with its own bit reversal -- which is invisible whenever
    the state is invariant under reversing its qubits, as a GHZ state, a ring, a
    square lattice and a uniform ``ry`` preparation all are, and which is wrong for
    every other state.

    Experiment E found this on the first asymmetric specification the referee had ever
    been given (an irregular graph state), where the naive comparison reported a
    distance of ``0.0605`` and the re-indexed one ``2.8e-17``. The conversion lives in
    one place so it can be tested, and
    ``test_the_basis_order_of_the_two_tiers_is_pinned_on_an_asymmetric_state`` pins it.
    """
    order = [int(format(index, f"0{num_qubits}b")[::-1], 2) for index in range(1 << num_qubits)]
    _out: np.ndarray = np.asarray(raw, dtype=np.complex128)[order]
    return _out


def _prepare_dense(circuit: Circuit) -> np.ndarray:
    """The prepared amplitudes, in the same basis order as the specification."""
    simulator = StatevectorSimulator(circuit.num_qubits)
    for op in circuit.ops:
        assert isinstance(op, Gate)  # callers reject non-unitary ops first
        apply_gate(simulator, op.name, op.targets, op.params)
    return _specification_order(np.asarray(simulator.state, dtype=np.complex128),
                                circuit.num_qubits)


def _phase_free_distance(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """``min_phi ||a - e^(i phi) e|| / sqrt(dim)`` and the minimising phase."""
    dimension = expected.shape[0]
    overlap = np.vdot(expected, actual)
    phase = 0.0 if abs(overlap) < 1e-15 else float(np.angle(overlap))
    rotated = np.exp(1j * phase) * expected
    return float(np.linalg.norm(actual - rotated)) / math.sqrt(dimension), phase


@dataclasses.dataclass(frozen=True)
class SpecificationReport:
    """The referee's verdict on one construction, against the specification alone."""

    specification_sha256: str
    candidate_sha256: str
    qubits: int
    #: ``"stabilizer"`` (exact, any size), ``"dense"`` (amplitudes, small) or
    #: ``"none"`` (nothing could decide it).
    tier: str
    checks: tuple[Check, ...]
    resources: Mapping[str, Any]
    verdict: str
    decides: str
    support_compared: int = 0

    def failed(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.passed is False)

    @property
    def promotable(self) -> bool:
        return self.verdict == "VALID"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "decides": self.decides,
            "specification_sha256": self.specification_sha256,
            "candidate_sha256": self.candidate_sha256,
            "qubits": self.qubits,
            "verdict": self.verdict,
            "support_compared": self.support_compared,
            "resources": dict(self.resources),
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


#: The tiers a caller may name, instead of letting the referee choose.
#: `"auto"` is the default: stabilizer for a Clifford circuit, dense for a
#: non-Clifford one within its bound. Naming a tier is how the two referees are run
#: against the *same* circuit, which is the only way their agreement is evidence.
TIERS = ("auto", "stabilizer", "dense")


def adjudicate_specification(circuit: Circuit, target: TargetSpecification, *,
                             tier: str = "auto") -> SpecificationReport:
    """Decide whether `circuit` prepares the state `target` describes.

    Two checks always run: the qubit count, and the stabilizer group (stabilizer tier)
    or the state vector (dense tier). A third runs when the target declares a support
    and the state is small enough to enumerate: the exact distribution, compared
    outcome by outcome. Nothing here looks at a reference circuit -- the only inputs
    are the circuit and the mathematics of the target.

    `tier` overrides the automatic choice. Asking for a tier that cannot decide the
    circuit (the stabilizer tier on a non-Clifford construction, the dense tier beyond
    its bound) raises :class:`SynthesisError`: a named tier that silently fell back
    would make :func:`referee_agreement` compare a referee with itself.
    """
    circuit.validate()
    if tier not in TIERS:
        raise SynthesisError(f"tier must be one of {TIERS}, got {tier!r}")
    checks: list[Check] = []

    same_width = circuit.num_qubits == target.num_qubits
    checks.append(Check(
        name="the construction acts on the specified number of qubits",
        requirement=f"exactly {target.num_qubits} qubits, so the comparison is meaningful",
        measured={"candidate": circuit.num_qubits, "specification": target.num_qubits},
        passed=same_width))
    if not same_width:
        return SpecificationReport(
            specification_sha256=target.sha256(), candidate_sha256=circuit.sha256(),
            qubits=circuit.num_qubits, tier="none", checks=tuple(checks),
            resources=circuit.resources(), verdict="INVALID", decides=target.decides)

    unitary = _is_unitary(circuit)
    clifford = unitary and is_clifford_circuit(circuit)
    support_compared = 0

    if tier != "auto":
        if tier == "stabilizer" and not clifford:
            raise SynthesisError(
                "the stabilizer tier cannot decide this circuit: "
                + ("it measures or resets, so it prepares a distribution rather than a "
                   "state" if not unitary else
                   "it is not Clifford, so its conjugated Pauli images are not Pauli "
                   "strings"))
        if tier == "dense":
            if not unitary:
                raise SynthesisError(
                    "the dense tier compares amplitudes and needs a unitary circuit; this "
                    "one measures or resets")
            if target.num_qubits > DENSE_SPEC_LIMIT:
                raise SynthesisError(
                    f"the dense tier is bounded at {DENSE_SPEC_LIMIT} qubits and this "
                    f"construction acts on {target.num_qubits}")
        use = tier
    elif clifford:
        use = "stabilizer"
    elif unitary and target.num_qubits <= DENSE_SPEC_LIMIT:
        use = "dense"
    else:
        use = "none"

    if use == "stabilizer":
        tier = use
        state = _state_of(circuit)
        failures: list[dict[str, Any]] = []
        for labels, phase in target.stabilizers:
            measured = state.phase_of(labels)
            if measured == int(phase) % 4:
                continue
            failures.append({"pauli": pauli_string(labels, 0),
                             "expected": f"i**{int(phase) % 4}",
                             "measured": ("absent from the group" if measured is None
                                           else f"i**{measured}")})
        checks.append(Check(
            name="the prepared state is the +1 eigenstate of every specified stabilizer",
            requirement=("all stabilizers of the specification stabilise the prepared "
                         "state with the specified sign; a missing generator means a "
                         "different state, and a wrong sign means the same group with a "
                         "different global sign"),
            measured={"stabilizers_checked": len(target.stabilizers),
                      "failed": failures[:4], "failed_count": len(failures),
                      "tier": "stabilizer"},
            passed=not failures))
        if target.support is not None and target.num_qubits <= 12:
            distribution = state.distribution()
            matches, evidence = target.support_matches(distribution)
            support_compared = len(target.support)
            checks.append(Check(
                name="the exact measurement distribution is the declared support",
                requirement=("every declared outcome has its declared probability and "
                             "no other outcome can occur, compared exactly (these are "
                             "powers of two, so there is no tolerance to tune)"),
                measured=evidence,
                passed=matches))
    elif use == "dense":
        tier = "dense"
        actual = _prepare_dense(circuit)
        expected = specification_state_vector(target)
        dist_value, phase = _phase_free_distance(actual, expected)
        checks.append(Check(
            name="the prepared amplitudes are the specification, up to one global phase",
            requirement=(f"min_phi ||psi_c - e^(i phi) psi_s|| / sqrt(2**n) <= "
                         f"{DENSE_TOLERANCE:g} against the state projected out of the "
                         f"specification's stabilizers"),
            measured={"dist_value": dist_value, "phi": phase, "tier": "dense",
                      "relative_phase_sensitive": True},
            passed=bool(dist_value <= DENSE_TOLERANCE)))
    else:
        tier = "none"
        reason = (
            "the circuit measures or resets, so it prepares a distribution rather than "
            "a state" if not unitary else
            f"the circuit is not Clifford and {target.num_qubits} qubits is above the "
            f"dense specification bound of {DENSE_SPEC_LIMIT}")
        checks.append(Check(
            name="a tier that can decide this construction",
            requirement="the stabilizer tier for Clifford circuits, the dense tier "
                        "for non-Clifford ones within its bound",
            measured={"reason": reason},
            passed=None))

    if any(check.passed is False for check in checks):
        verdict = "INVALID"
    elif all(check.passed is True for check in checks):
        verdict = "VALID"
    else:
        verdict = "UNRESOLVED"

    return SpecificationReport(
        specification_sha256=target.sha256(),
        candidate_sha256=circuit.sha256(),
        qubits=circuit.num_qubits,
        tier=tier,
        checks=tuple(checks),
        resources=circuit.resources(),
        verdict=verdict,
        decides=target.decides,
        support_compared=support_compared,
    )


# ------------------------------------------------------------------- agreement
@dataclasses.dataclass(frozen=True)
class RefereeAgreement:
    """Both referees on the same circuit: two representations, one question.

    The stabilizer tier answers with exact integer arithmetic on the tableau; the
    dense tier answers with ``2**n`` amplitudes and a phase-free distance. Neither
    consults the other, and neither is a re-implementation of the other, so a
    disagreement would mean the answer is representation-dependent -- which is what
    this record exists to rule out at every width both tiers can reach.
    """

    circuit_sha256: str
    qubits: int
    stabilizer_verdict: str
    dense_verdict: str
    stabilizer_checks_failed: tuple[str, ...]
    dense_checks_failed: tuple[str, ...]
    dense_distance: float | None

    @property
    def agree(self) -> bool:
        return self.stabilizer_verdict == self.dense_verdict

    def as_dict(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "circuit_sha256": self.circuit_sha256,
            "stabilizer_verdict": self.stabilizer_verdict,
            "dense_verdict": self.dense_verdict,
            "stabilizer_checks_failed": list(self.stabilizer_checks_failed),
            "dense_checks_failed": list(self.dense_checks_failed),
            "dense_distance": self.dense_distance,
            "agree": self.agree,
        }


def referee_agreement(circuit: Circuit, target: TargetSpecification) -> RefereeAgreement:
    """Run *both* specification referees on one Clifford circuit and compare.

    Only Clifford constructions within the dense bound can be decided twice, so
    anything else raises :class:`SynthesisError` naming the tier that could not run --
    an agreement that was only ever checked by one referee is not agreement.
    """
    stabilizer = adjudicate_specification(circuit, target, tier="stabilizer")
    dense = adjudicate_specification(circuit, target, tier="dense")
    distance = None
    for check in dense.checks:
        if check.measured.get("tier") == "dense" and "distance" in check.measured:
            distance = float(check.measured["distance"])
    return RefereeAgreement(
        circuit_sha256=circuit.sha256(),
        qubits=circuit.num_qubits,
        stabilizer_verdict=stabilizer.verdict,
        dense_verdict=dense.verdict,
        stabilizer_checks_failed=stabilizer.failed(),
        dense_checks_failed=dense.failed(),
        dense_distance=distance,
    )


# ---------------------------------------------------------------------- admission
@dataclasses.dataclass(frozen=True)
class SpecificationCandidate:
    """A synthesised circuit offered against a *specification*, not a reference.

    It carries the same fields the circuit candidate does -- an id, a circuit, a
    source and resource claims -- so it goes through the same admission policy in
    :mod:`apedp_dyn.vqc.discovery`: `VALID` and no unmet claim and sufficient
    evidence, or no promotion. The reference circuit is simply absent, which is the
    point of the experiment.
    """

    candidate_id: str
    circuit: Circuit
    specification: TargetSpecification
    source: str = "synthesis"
    claims: Mapping[str, int] = dataclasses.field(default_factory=dict)
    metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "specification": self.specification.as_dict(),
            "specification_sha256": self.specification.sha256(),
            "candidate_sha256": self.circuit.sha256(),
            "claims": dict(self.claims),
            "metadata": dict(self.metadata),
        }


def _claim_check(circuit: Circuit, claims: Mapping[str, int]) -> Check:
    """The same rule the equivalence referee applies: every claim is an upper bound.

    Kept local so this module does not import a private helper, and pinned against
    the other implementation by a test, so the two cannot drift apart silently.
    """
    actual = circuit.resources()
    unmet = {measure: {"claimed": claimed, "actual": actual.get(measure)}
             for measure, claimed in claims.items()
             if actual.get(measure) is None or actual[measure] > claimed}
    return Check(
        name="the construction meets the resource claim it is offered under",
        requirement="every claimed measure is at most the claimed value",
        measured={"claims": dict(claims), "unmet": unmet},
        passed=not unmet)


def adjudicate_specification_candidate(candidate: SpecificationCandidate
                                       ) -> CandidateVerdict:
    """Route a specification candidate through the specification referee, seal it."""
    report = adjudicate_specification(candidate.circuit, candidate.specification)
    checks: list[Check] = list(report.checks)
    claim_verdict = "NOT_ASSERTED"
    claim: Check | None = None
    if candidate.claims:
        claim = _claim_check(candidate.circuit, candidate.claims)
        checks.append(claim)
        claim_verdict = "MET" if claim.passed else "UNMET"

    # The semantic verdict comes from the specification referee alone. An unmet
    # resource claim is *not* a statement that the circuit computes something else:
    # folding the two together is the conflation `discovery.py` exists to prevent,
    # and this test caught it being reintroduced here.
    verdict = report.verdict

    reasons: list[str] = []
    for check in checks:
        if check.passed is True or check is claim:
            continue
        if check.passed is None:
            reasons.append(f"{check.name}: undecided "
                           f"({check.measured.get('reason', 'no reason recorded')})")
        else:
            reasons.append(f"{check.name}: failed")
    if claim is not None and claim.passed is False:
        unmet = claim.measured.get("unmet") or {}
        detail = ", ".join(f"{measure}: claimed {row['claimed']}, measured "
                          f"{row['actual']}" for measure, row in sorted(unmet.items()))
        reasons.append(f"the construction is right but does not improve its "
                       f"resource claim ({detail})")

    return CandidateVerdict(
        candidate=candidate,
        tier=report.tier,
        equivalence=None,
        distribution=None,
        checks=tuple(checks),
        verdict=verdict,
        refused_reasons=tuple(reasons),
        claim_verdict=claim_verdict,
    )


# ---------------------------------------------------------------------- synthesis
def _pareto(rows: Sequence[Mapping[str, Any]],
            keys: Sequence[str] = ("depth", "max_edge_span")) -> tuple[str, ...]:
    """The constructions nothing else beats on *every* key: the trade-off frontier."""
    frontier = []
    for row in rows:
        dominated = any(
            all(other[key] <= row[key] for key in keys)
            and any(other[key] < row[key] for key in keys)
            for other in rows if other is not row)
        if not dominated:
            frontier.append(row["family"])
    return tuple(frontier)


@dataclasses.dataclass(frozen=True)
class SynthesisOutcome:
    """Everything one synthesis run established, including what it refused."""

    target: TargetSpecification
    rows: tuple[Mapping[str, Any], ...]
    ranked: tuple[str, ...]
    pareto: tuple[str, ...]
    promoted: str | None
    refusal: str | None
    verdict: CandidateVerdict | None

    def valid(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(row for row in self.rows if row["verdict"] == "VALID")

    def as_dict(self) -> dict[str, Any]:
        return {
            "specification": self.target.as_dict(),
            "specification_sha256": self.target.sha256(),
            "constructions": [dict(row) for row in self.rows],
            "valid": [row["family"] for row in self.rows if row["verdict"] == "VALID"],
            "refused": [row["family"] for row in self.rows if row["verdict"] != "VALID"],
            "ranked": list(self.ranked),
            "pareto_frontier": list(self.pareto),
            "promoted": self.promoted,
            "promotion_refusal": self.refusal,
            "admission": None if self.verdict is None else {
                "candidate_id": self.verdict.candidate.candidate_id,
                "semantic_verdict": self.verdict.verdict,
                "resource_verdict": self.verdict.claim_verdict,
                "evidence_sufficiency": self.verdict.evidence,
                "promotion_decision": ("PROMOTED" if self.verdict.promotable
                                       else "REFUSED"),
                "promotable": self.verdict.promotable,
                "tier": self.verdict.tier,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def synthesize(target: TargetSpecification, *, families: Sequence[str] | None = None,
               seed: int = 0, claims: Mapping[str, int] | None = None,
               ) -> SynthesisOutcome:
    """Build every applicable construction, decide each, rank, then admit the best.

    "Best" is `(depth, operations, max_edge_span)`, and the report also carries the
    Pareto frontier, because the cheapest depth and the cheapest connectivity are
    different circuits: a star has depth 2 and spans the whole register, a chain is
    nearest-neighbour and `n` deep. Collapsing that into one number would be a
    preference dressed as a measurement.

    The admitted candidate is the best-ranked *valid* construction, offered under
    `claims` (default: its own measured `operations` and `depth`). A claim it does not
    meet is `UNMET` and the promotion is refused -- by the same policy, and with the
    same four fields, that a rewrites-loop candidate goes through.
    """
    names = tuple(families) if families is not None else default_families(target)
    if not names:
        raise SynthesisError(f"no construction family builds a {target.name!r} state")

    rows: list[dict[str, Any]] = []
    circuits: dict[str, Circuit] = {}
    for family in names:
        try:
            construction = build(family, target, seed=seed)
        except SynthesisError as error:
            rows.append({"family": family, "verdict": "NOT_CONSTRUCTIBLE",
                         "reason": str(error)})
            continue
        report = adjudicate_specification(construction.circuit, target)
        row = construction.as_dict()
        row["verdict"] = report.verdict
        row["tier"] = report.tier
        row["checks_failed"] = list(report.failed())
        row["first_failure"] = next(
            (check.measured for check in report.checks if check.passed is False), None)
        rows.append(row)
        circuits[family] = construction.circuit

    valid = [row for row in rows if row["verdict"] == "VALID"]
    ranked = tuple(row["family"] for row in sorted(
        valid, key=lambda row: (row["depth"], row["operations"], row["max_edge_span"])))

    if not ranked:
        return SynthesisOutcome(target=target, rows=tuple(rows), ranked=(), pareto=(),
                                promoted=None, refusal="no construction reproduced the "
                                                       "specification", verdict=None)

    frontier = _pareto(valid)
    best = ranked[0]
    circuit = circuits[best]
    candidate = SpecificationCandidate(
        candidate_id=f"spec-{target.name}-{target.num_qubits}q-{best}",
        circuit=circuit,
        specification=target,
        claims=dict(claims) if claims is not None else {
            "operations": circuit.resources()["operations"],
            "depth": circuit.resources()["depth"]},
        metadata={"family": best, "specification_sha256": target.sha256()},
    )
    verdict = adjudicate_specification_candidate(candidate)
    promoted: str | None
    refusal: str | None
    try:
        promoted = promote(verdict)
        refusal = None
    except PromotionRefused as error:
        promoted = None
        refusal = str(error)

    return SynthesisOutcome(target=target, rows=tuple(rows), ranked=ranked,
                            pareto=frontier, promoted=promoted, refusal=refusal,
                            verdict=verdict)


def specification_summary(target: TargetSpecification) -> dict[str, Any]:
    """The target as a record should cite it: hashes, stabilizers, support size."""
    return {
        "name": target.name,
        "qubits": target.num_qubits,
        "sha256": target.sha256(),
        "stabilizers": list(target.stabilizer_strings())[:4],
        "stabilizer_count": len(target.stabilizers),
        "declared_outcomes": None if target.support is None else len(target.support),
    }


def dense_oracle_check(target: TargetSpecification) -> dict[str, Any]:
    """Compare the specification's two readings of itself, on small widths.

    The tableau (used at any size) and the projection onto ``+1`` eigenspaces (used
    for amplitudes, at most :data:`DENSE_SPEC_LIMIT` qubits) are independent routes to
    the same state, so a disagreement between them would mean one of them is wrong --
    not that the specification is ambiguous. This is the check both referees rest on,
    run as a measurement of its own.
    """
    state = target.stabilizer_state()
    vector = specification_state_vector(target)
    dimension = vector.shape[0]
    probabilities = np.abs(vector) ** 2
    support = {}
    for index in range(dimension):
        if probabilities[index] <= 1e-15:
            continue
        bits = tuple(bool((index >> (target.num_qubits - 1 - qubit)) & 1)
                     for qubit in range(target.num_qubits))
        support[bits] = float(probabilities[index])
    agrees, evidence = (target.support_matches(support)
                        if target.support is not None else (True, {"skipped": True}))
    stabilizer_agreement = all(
        state.stabilizes(labels, sign=0 if int(phase) % 4 == 0 else 1)
        for labels, phase in target.stabilizers)
    return {
        "vectors_compared": 1,
        "non_zero_amplitudes": len(support),
        "stabilizer_group_agrees": bool(stabilizer_agreement),
        "support_agrees": bool(agrees),
        "evidence": evidence,
    }
