"""A shared target corpus: a seeded *sample* of graph states, not a handful of rows.

Status: experimental

The external comparison had a structural weakness that its own record named: the GHZ
and ring rows compare a construction grammar that *knows* those targets against a
general synthesizer, and the random-state row is the case where the grammar abstains.
Neither is a shared comparison, because only one side is being asked to do something
it was designed for.

This module is the corpus that fixes that -- and the review of it named the second
weakness: seven hand-picked instances are a *set*, not a *sample*, so a single row per
family could be the draw that flattered either engine. The corpus is therefore a
celled sample: every family at every width in :data:`SAMPLE_WIDTHS`, with
:data:`SAMPLE_REPLICATES` independent draws per cell, each draw reproducible from its
seed alone and each row reported alongside the spread of its family rather than in
place of it.

* ``regular_graph`` -- random connected ``d``-regular graph states (``d`` = 3, 4). No
  lattice boundary, no product structure, no ring: the grammar's graph families read
  the edge set and emit one fixed schedule of it, and Qiskit's synthesizer runs its
  general elimination on the stabilizers. Neither has a construction written for a
  random regular graph.
* ``random_graph`` -- the seeded spanning-tree-plus-extra-edges graph, which is this
  package's own generator for an *irregular* graph rather than a regular one, so the
  corpus is not the family Experiment E was measured on. Its degree is not fixed, so
  its structural spread is wider than the regular families' and that is recorded.
* ``random_stabilizer`` -- a stabilizer state with no graph structure at all. This is
  the corpus's honest asymmetry: the grammar abstains (it has no family for it) and
  Qiskit synthesizes. An abstention is recorded as an abstention, never as a loss.

Every target is a :class:`~apedp_dyn.vqc.synthesis.TargetSpecification` -- the same
object both engines are handed -- and every target is reproducible from its seed alone,
so the corpus is a fixed set of instances rather than a generator whose output changes
between runs. ``test_the_corpus_is_reproducible_from_its_seeds_alone`` pins that.

The seed rule is arithmetic, not a list of magic numbers::

    seed = SEED_BASE[family] + 10 * width + replicate

with ``replicate < 10`` and the bases far enough apart that no two cells collide. A
label names the cell, so a row in a record can be traced back to the draw that
produced it.

What this is not: graph space. The widths are five fixed tiers, the draws come from
one pairing-model generator, and instances at the same width are not independent of
that width. A spread over this sample is a spread over *these* instances -- it says
how much a single row would have moved, not how either engine behaves on the
distribution of all graph states. Nor is it a hardware corpus: the graphs are abstract
connectivity, with no calibration or error model.
"""
from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence
from typing import Any

from apedp_dyn.vqc.stabilizer import StabilizerState
from apedp_dyn.vqc.synthesis import (
    SynthesisError,
    TargetSpecification,
    Z,
    irregular_graph_target,
    seeded_graph_target,
)

__all__ = [
    "CORPUS",
    "GRAPH_FAMILIES",
    "SAMPLE_REPLICATES",
    "SAMPLE_WIDTHS",
    "SEED_BASE",
    "STABILIZER_WIDTHS",
    "CorpusTarget",
    "random_regular_graph_edges",
    "random_stabilizer_target",
    "sampled_graph_corpus",
    "seed_for",
    "shared_target_corpus",
]

#: The widths the sample is drawn at, in qubits.
SAMPLE_WIDTHS: tuple[int, ...] = (12, 20, 32, 48, 64)
#: Independent draws per ``(family, width)`` cell.
SAMPLE_REPLICATES: int = 5
#: The graph families, as ``(name, degree)``; ``None`` degree means irregular.
GRAPH_FAMILIES: tuple[tuple[str, int | None], ...] = (
    ("regular3", 3),
    ("regular4", 4),
    ("random_graph", None),
)
#: The widths the abstention anchors are drawn at. They carry no spread of their own --
#: the grammar abstains at every width -- so three instances are enough to show that
#: the abstention is structural rather than a property of one narrow state.
STABILIZER_WIDTHS: tuple[int, ...] = (12, 20, 32)
#: ``family -> seed base``; see the module docstring for the seed rule.
SEED_BASE: dict[str, int] = {
    "regular3": 20_000,
    "regular4": 30_000,
    "random_graph": 40_000,
    "random_stabilizer": 50_000,
}


def seed_for(family: str, width: int, replicate: int) -> int:
    """The seed for one cell of the sample, from the documented rule alone."""
    if family not in SEED_BASE:
        raise SynthesisError(f"unknown corpus family {family!r}")
    if not 0 <= replicate < 10:
        raise SynthesisError(f"a replicate must be in [0, 10), got {replicate}")
    return SEED_BASE[family] + 10 * width + replicate


def _connected(num_qubits: int, edges: Sequence[tuple[int, int]]) -> bool:
    neighbours: list[list[int]] = [[] for _ in range(num_qubits)]
    for parent, child in edges:
        neighbours[parent].append(child)
        neighbours[child].append(parent)
    reached = {0}
    stack = [0]
    while stack:
        vertex = stack.pop()
        for neighbour in neighbours[vertex]:
            if neighbour not in reached:
                reached.add(neighbour)
                stack.append(neighbour)
    return len(reached) == num_qubits


def random_regular_graph_edges(num_qubits: int, degree: int, *, seed: int,
                               max_attempts: int = 500) -> tuple[tuple[int, int], ...]:
    """A random simple connected ``degree``-regular graph, from a seed.

    The pairing model: give every vertex ``degree`` stubs, shuffle them, and pair them
    off. A draw with a self-loop or a repeated edge is rejected and redrawn, and so is
    a disconnected draw -- a graph state on a disconnected graph is a product state,
    which the target constructor refuses for the same reason. Rejecting rather than
    repairing matters for the corpus's claim: the graphs are *random* regular graphs,
    not regularised versions of something else, and the seed reproduces the accepted
    draw exactly.
    """
    if num_qubits < 4:
        raise SynthesisError(f"a regular graph needs at least four vertices, got {num_qubits}")
    if not 1 <= degree < num_qubits:
        raise SynthesisError(f"degree must be in [1, {num_qubits}), got {degree}")
    if (num_qubits * degree) % 2:
        raise SynthesisError(
            f"{num_qubits} vertices cannot all have degree {degree}: the sum must be even")
    rng = random.Random(seed)
    for _attempt in range(max_attempts):
        stubs = [vertex for vertex in range(num_qubits) for _ in range(degree)]
        rng.shuffle(stubs)
        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        simple = True
        for index in range(0, len(stubs), 2):
            first, second = stubs[index], stubs[index + 1]
            if first == second:
                simple = False
                break
            edge = (min(first, second), max(first, second))
            if edge in seen:
                simple = False
                break
            seen.add(edge)
            edges.append(edge)
        if not simple or not _connected(num_qubits, edges):
            continue
        return tuple(sorted(edges))
    raise SynthesisError(  # pragma: no cover - the corpus seeds succeed on the first draws
        f"no simple connected {degree}-regular graph on {num_qubits} vertices in "
        f"{max_attempts} draws of seed {seed}")


def random_stabilizer_target(num_qubits: int, *, seed: int) -> TargetSpecification:
    """A stabilizer state with no local construction family: the abstention target.

    Built by running a seeded Clifford circuit on ``|0...0>`` and reading the tableau's
    generators back as the specification. The circuit that produced it is *not* handed
    to the synthesiser -- there is nothing to hand it to -- so the state is a target,
    not a reference: this is the shape Experiment A was built to accept.
    """
    ops = []
    rng = random.Random(seed)
    for qubit in range(num_qubits):
        ops.append(("h", (qubit,)))
    for _ in range(3 * num_qubits):
        qubit = rng.randrange(num_qubits)
        if rng.random() < 0.45:
            ops.append((rng.choice(("s", "sdg", "h")), (qubit,)))
        else:
            other = rng.randrange(num_qubits)
            while other == qubit:
                other = rng.randrange(num_qubits)
            ops.append(("cx", (qubit, other)))
    state = StabilizerState(num_qubits)
    for name, targets in ops:
        state.apply_gate(name, targets, ())
    return TargetSpecification(
        name="random_stabilizer",
        num_qubits=num_qubits,
        stabilizers=tuple(zip(state.generators, state.phases, strict=True)),
        support=None,
        decides="the random stabilizer state, up to one global phase",
    )


@dataclasses.dataclass(frozen=True)
class CorpusTarget:
    """One corpus instance: its cell in the sample, its seed and the specification."""

    label: str
    kind: str
    seed: int
    target: TargetSpecification
    note: str
    #: The family the instance was drawn from: ``regular3``, ``regular4``,
    #: ``random_graph`` or ``random_stabilizer``. ``kind`` is the coarse class.
    family: str = ""
    #: The replicate index within its ``(family, width)`` cell.
    replicate: int = 0
    #: ``d`` for the regular families, ``None`` for an irregular or non-graph draw.
    degree: int | None = None

    @property
    def width(self) -> int:
        return self.target.num_qubits

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "kind": self.kind,
            "family": self.family,
            "seed": self.seed,
            "width": self.width,
            "replicate": self.replicate,
            "qubits": self.target.num_qubits,
            "target_sha256": self.target.sha256(),
            "stabilizers": len(self.target.stabilizers),
            "note": self.note,
        }
        if self.degree is not None:
            payload["degree"] = self.degree
        if self.kind != "random_stabilizer":
            # A graph state's stabilizer is X on its vertex and Z on its neighbours,
            # so the Z count is the degree and half the sum is the edge count.
            degrees = [sum(1 for label in labels if label == Z)
                       for labels, _phase in self.target.stabilizers]
            payload["edges"] = sum(degrees) // 2
            payload["min_degree"] = min(degrees)
            payload["max_degree"] = max(degrees)
            payload["mean_degree"] = sum(degrees) / len(degrees)
        return payload


def _regular_target(num_qubits: int, degree: int, *, replicate: int,
                    family: str) -> CorpusTarget:
    seed = seed_for(family, num_qubits, replicate)
    edges = random_regular_graph_edges(num_qubits, degree, seed=seed)
    target = irregular_graph_target(
        num_qubits, edges,
        decides=(f"the random {degree}-regular graph state on {num_qubits} vertices, "
                 f"up to one global phase"))
    return CorpusTarget(
        label=f"{family}_{num_qubits}q_{replicate}",
        kind="regular_graph",
        seed=seed,
        target=target,
        note=(f"pairing-model draw, cell ({family}, {num_qubits}q, replicate "
              f"{replicate}), seed {seed}: {len(edges)} edges, every vertex of degree "
              f"{degree}, no lattice, ring or tree structure"),
        family=family,
        replicate=replicate,
        degree=degree,
    )


def _random_graph_target(num_qubits: int, *, replicate: int) -> CorpusTarget:
    family = "random_graph"
    seed = seed_for(family, num_qubits, replicate)
    target = seeded_graph_target(num_qubits, seed=seed, extra_edge_probability=0.12)
    return CorpusTarget(
        label=f"{family}_{num_qubits}q_{replicate}",
        kind="random_graph",
        seed=seed,
        target=target,
        note=(f"seeded random spanning tree plus extra edges at p=0.12, cell "
              f"({family}, {num_qubits}q, replicate {replicate}), seed {seed}; an "
              f"irregular instance, not a regular family"),
        family=family,
        replicate=replicate,
    )


def _random_stabilizer_entry(num_qubits: int) -> CorpusTarget:
    family = "random_stabilizer"
    seed = seed_for(family, num_qubits, 0)
    return CorpusTarget(
        label=f"{family}_{num_qubits}q_0",
        kind="random_stabilizer",
        seed=seed,
        target=random_stabilizer_target(num_qubits, seed=seed),
        note=(f"seeded Clifford state at {num_qubits} qubits, seed {seed}: the graph "
              f"families do not apply, so the grammar abstains"),
        family=family,
    )


def sampled_graph_corpus(*, widths: Sequence[int] = SAMPLE_WIDTHS,
                         replicates: int = SAMPLE_REPLICATES,
                         ) -> tuple[CorpusTarget, ...]:
    """The sample, in a stable order: graph cells first, then the abstention anchors.

    ``widths x replicates`` draws from each graph family, followed by one random
    stabilizer state per width in :data:`STABILIZER_WIDTHS`. The order is a nested
    loop over ``(family, width, replicate)``, so a row's position identifies its cell
    and the record's summary can be regrouped without re-deriving anything.
    """
    entries: list[CorpusTarget] = []
    for family, degree in GRAPH_FAMILIES:
        for width in widths:
            for replicate in range(replicates):
                if degree is None:
                    entries.append(_random_graph_target(width, replicate=replicate))
                else:
                    entries.append(_regular_target(width, degree,
                                                   replicate=replicate, family=family))
    entries.extend(_random_stabilizer_entry(width) for width in STABILIZER_WIDTHS)
    return tuple(entries)


def shared_target_corpus() -> tuple[CorpusTarget, ...]:
    """The corpus both engines are handed: the sample, rebuilt from its seeds."""
    return sampled_graph_corpus()


#: The corpus as data, so a record can cite it without rebuilding it.
CORPUS = shared_target_corpus()
