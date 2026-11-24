"""Decoders: the interface, a lookup table, and exact minimum-weight matching.

Status: experimental

The repetition code in :mod:`apedp_dyn.vqc.logical` decodes with a table written for
distance 3: four syndromes, four corrections. That is a *decoder*, and it does not
generalise -- at distance 7 there are 64 syndromes and the minimum-weight answer is
no longer a table entry. This module names the interface both kinds implement, and
adds the second implementation:

:class:`LookupDecoder`
    a syndrome -> correction table, which is what the distance-3 code already had;
:class:`MinimumWeightDecoder`
    exact minimum-weight perfect matching on a defect graph.

One graph per error basis
-------------------------
A matching graph belongs to *one* error type. A CSS code needs two (X errors and Z
errors), and this module builds one at a time; the five-qubit code is not CSS, so it
is out of scope here and keeps its own decoder. That restriction is stated because
the alternative -- pretending one graph covers a non-CSS code -- is how a decoder
silently becomes wrong.

How the matching works
----------------------
Each syndrome bit that is lit is a *defect*: a node of the graph that must be
resolved. Errors correspond to paths, so the decoder's job is to pair defects up
(and to route the unpaired ones to a boundary) at minimum total weight, then apply
the corrections the path carried. The minimisation is exact T-join dynamic
programming over subsets of defects -- real minimum-weight perfect matching,
exponential in the number of defects and therefore stated as such. A surface-code
shot at a workable physical error rate has a handful of defects; a shot at
``p = 1/2`` has ``d`` of them, which is why the tests stop at distance 7.

The decoder is not the circuit
------------------------------
:func:`apedp_dyn.vqc.logical.repetition_code_circuit` builds its correction gates
*from* the decoder it is given, so the same code can be run with either decoder and
the two are compared on the same noise, not on two different experiments.
"""
from __future__ import annotations

import dataclasses
import heapq
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Correction",
    "Decoder",
    "DecoderRefusalError",
    "LookupDecoder",
    "MatchingGraph",
    "MinimumWeightDecoder",
    "repetition_lookup_decoder",
    "repetition_matching_graph",
    "repetition_mwpm_decoder",
]


class DecoderRefusalError(ValueError):
    """A syndrome this decoder will not guess at."""


@dataclasses.dataclass(frozen=True)
class Correction:
    """A Pauli correction over the data qubits, and where it came from."""

    pauli: str
    decoder: str
    syndrome: tuple[bool, ...]
    #: The matching weight the decoder minimised; `None` for a lookup table.
    weight: int | None = None

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(index for index, letter in enumerate(self.pauli) if letter != "I")

    @property
    def pauli_weight(self) -> int:
        return len(self.support)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pauli": self.pauli,
            "pauli_weight": self.pauli_weight,
            "support": list(self.support),
            "decoder": self.decoder,
            "syndrome": [bool(bit) for bit in self.syndrome],
            "matching_weight": self.weight,
        }


@runtime_checkable
class Decoder(Protocol):
    """What every decoder has to provide."""

    name: str
    code: str
    distance: int

    def decode(self, syndrome: Sequence[bool]) -> Correction:  # pragma: no cover
        ...


class LookupDecoder:
    """A table from syndrome to correction, with the table kept inspectable."""

    def __init__(self, table: Mapping[tuple[bool, ...], str], *,
                 code: str = "repetition-3", distance: int = 3,
                 name: str = "lookup") -> None:
        self.table = {tuple(bool(bit) for bit in key): value for key, value in table.items()}
        for correction in self.table.values():
            if len(correction) != distance:
                raise DecoderRefusalError(
                    f"correction {correction!r} does not cover {distance} data qubits")
            if any(letter not in "IXYZ" for letter in correction):
                raise DecoderRefusalError(f"correction {correction!r} is not a Pauli string")
        self.name = name
        self.code = code
        self.distance = int(distance)

    @property
    def syndrome_width(self) -> int:
        widths = {len(syndrome) for syndrome in self.table}
        return widths.pop() if len(widths) == 1 else 0

    def decode(self, syndrome: Sequence[bool]) -> Correction:
        key = tuple(bool(bit) for bit in syndrome)
        if key not in self.table:
            raise DecoderRefusalError(
                f"syndrome {''.join('1' if bit else '0' for bit in key)} is not in the "
                f"{self.name} table; a table cannot answer a code it was not built for")
        return Correction(pauli=self.table[key], decoder=self.name, syndrome=key)


@dataclasses.dataclass(frozen=True)
class _Edge:
    other: int
    weight: int
    pauli: str


class MatchingGraph:
    """A defect graph: nodes, weighted edges, and the corrections they carry."""

    def __init__(self, syndrome_bits: int) -> None:
        if syndrome_bits < 1:
            raise ValueError(f"a graph needs at least one syndrome bit, got {syndrome_bits}")
        self.syndrome_bits = int(syndrome_bits)
        self.boundaries: list[int] = []
        self.edges: list[tuple[int, int, int, str]] = []
        self._adjacency: dict[int, list[_Edge]] | None = None

    # ------------------------------------------------------------------ building
    def add_boundary(self, node: int) -> None:
        """Mark a node as a boundary: a defect may terminate there at its distance."""
        if node not in self.boundaries:
            self.boundaries.append(node)
        self._adjacency = None

    def add_edge(self, first: int, second: int, weight: int, pauli: str = "") -> None:
        if weight < 1:
            raise ValueError(f"edge weight must be positive, got {weight}")
        self.edges.append((int(first), int(second), int(weight), pauli))
        self._adjacency = None

    def syndrome_node(self, index: int) -> int:
        """The graph node that carries syndrome bit `index`."""
        return int(index)

    def nodes(self) -> tuple[int, ...]:
        labels = set(self.boundaries)
        for first, second, _weight, _pauli in self.edges:
            labels.update((first, second))
        return tuple(sorted(labels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "syndrome_bits": self.syndrome_bits,
            "nodes": list(self.nodes()),
            "boundaries": list(self.boundaries),
            "edges": [{"u": u, "v": v, "weight": w, "pauli": p}
                      for u, v, w, p in self.edges],
        }

    # ------------------------------------------------------------------ geometry
    def _neighbours(self) -> dict[int, list[_Edge]]:
        if self._adjacency is None:
            adjacency: dict[int, list[_Edge]] = {}
            for first, second, weight, pauli in self.edges:
                adjacency.setdefault(first, []).append(_Edge(second, weight, pauli))
                adjacency.setdefault(second, []).append(_Edge(first, weight, pauli))
            self._adjacency = adjacency
        return self._adjacency

    def shortest_paths(self, start: int) -> dict[int, tuple[int, str]]:
        """`{node: (distance, correction along a shortest path)}` from `start`."""
        adjacency = self._neighbours()
        best: dict[int, tuple[int, str]] = {start: (0, "")}
        queue: list[tuple[int, int]] = [(0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > best[node][0]:
                continue
            for edge in adjacency.get(node, ()):
                candidate = (distance + edge.weight,
                             _combine(best[node][1], edge.pauli))
                current = best.get(edge.other)
                if current is None or candidate[0] < current[0]:
                    best[edge.other] = candidate
                    heapq.heappush(queue, (*candidate[0:1], edge.other))
        return best

    def distance_to_boundary(self, node: int) -> tuple[int, str] | None:
        best: tuple[int, str] | None = None
        if node in self.boundaries:
            return (0, "")
        reachable = self.shortest_paths(node)
        for boundary in self.boundaries:
            candidate = reachable.get(boundary)
            if candidate is None:
                continue
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best


def _combine(left: str, right: str) -> str:
    """The product of two corrections in one error basis, where letters commute."""
    if not left:
        return right
    if not right:
        return left
    if len(left) != len(right):
        raise DecoderRefusalError(f"corrections of different lengths: {left!r}, {right!r}")
    out = []
    for first, second in zip(left, right, strict=True):
        if first == "I":
            out.append(second)
        elif second == "I" or second == first:
            out.append(first)
        else:
            raise DecoderRefusalError(
                f"{first!r} and {second!r} do not commute within one error basis")
    return "".join(out)


class MinimumWeightDecoder:
    """Exact minimum-weight perfect matching of the syndrome defects.

    The minimisation is T-join dynamic programming over subsets of defects: a defect
    either pairs with another defect or terminates on a boundary, recursively. It is
    exact -- not a greedy matching -- and its cost is `2**defects`, which is the
    honest price of exactness.
    """

    name = "mwpm"

    def __init__(self, graph: MatchingGraph, *, code: str = "repetition", distance: int,
                 name: str | None = None) -> None:
        self.graph = graph
        self.code = code
        self.distance = int(distance)
        if name:
            self.name = name
        self._pairwise: dict[tuple[int, int], tuple[int, str] | None] = {}

    # ------------------------------------------------------------------ geometry
    def _pair(self, first: int, second: int) -> tuple[int, str] | None:
        key = (min(first, second), max(first, second))
        if key not in self._pairwise:
            self._pairwise[key] = self.graph.shortest_paths(first).get(second)
        return self._pairwise[key]

    def decode(self, syndrome: Sequence[bool]) -> Correction:
        bits = tuple(bool(bit) for bit in syndrome)
        if len(bits) != self.graph.syndrome_bits:
            raise DecoderRefusalError(
                f"expected {self.graph.syndrome_bits} syndrome bits, got {len(bits)}")
        defects = [self.graph.syndrome_node(index) for index, lit in enumerate(bits) if lit]
        if not defects:
            return Correction(pauli="I" * self.distance, decoder=self.name, syndrome=bits,
                              weight=0)

        boundary = [self.graph.distance_to_boundary(node) for node in defects]
        pairs: dict[tuple[int, int], tuple[int, str] | None] = {}
        for left in range(len(defects)):
            for right in range(left + 1, len(defects)):
                pairs[(left, right)] = self._pair(defects[left], defects[right])

        memo: dict[int, tuple[int, tuple[tuple[int, str], ...]]] = {0: (0, ())}

        def solve(mask: int) -> tuple[int, tuple[tuple[int, str], ...]]:
            if mask in memo:
                return memo[mask]
            first = (mask & -mask).bit_length() - 1
            rest = mask ^ (1 << first)
            best: tuple[int, tuple[tuple[int, str], ...]] | None = None
            entry = boundary[first]
            if entry is not None:
                cost, choices = solve(rest)
                best = (entry[0] + cost, (entry, *choices))
            remaining = rest
            while remaining:
                second = (remaining & -remaining).bit_length() - 1
                remaining ^= 1 << second
                path = pairs[(first, second)]
                if path is None:
                    continue
                cost, choices = solve(rest ^ (1 << second))
                candidate = (path[0] + cost, (path, *choices))
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is None:
                raise DecoderRefusalError(
                    "the syndrome is not resolvable on this graph: some defects are "
                    "isolated from every other defect and from every boundary")
            memo[mask] = best
            return best

        weight, choices = solve((1 << len(defects)) - 1)
        correction = ""
        for _distance, pauli in choices:
            correction = _combine(correction, pauli) if correction else pauli
        if not correction:
            correction = "I" * self.distance
        if len(correction) != self.distance:
            raise DecoderRefusalError(
                f"the matching produced {len(correction)} letters for {self.distance} "
                f"data qubits")
        return Correction(pauli=correction, decoder=self.name, syndrome=bits, weight=weight)


# -------------------------------------------------------------- repetition codes
def repetition_matching_graph(distance: int, *, code: str = "bit_flip") -> MatchingGraph:
    """The defect graph of a distance-`distance` repetition code.

    Syndromes are the ``distance - 1`` parity checks in order, and the two ends of the
    chain are boundaries: an error at the end of the chain lights only one check, and
    that defect terminates on the boundary that explains it with weight one.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd and at least 3, got {distance}")
    letter = "X" if code == "bit_flip" else "Z"
    checks = distance - 1
    graph = MatchingGraph(checks)
    left, right = checks, checks + 1
    graph.add_boundary(left)
    graph.add_boundary(right)
    graph.add_edge(left, 0, 1, letter + "I" * (distance - 1))
    for index in range(checks - 1):
        labels = ["I"] * distance
        labels[index + 1] = letter
        graph.add_edge(index, index + 1, 1, "".join(labels))
    end = ["I"] * distance
    end[distance - 1] = letter
    graph.add_edge(checks - 1, right, 1, "".join(end))
    return graph


def repetition_mwpm_decoder(distance: int, *, code: str = "bit_flip",
                            name: str = "mwpm") -> MinimumWeightDecoder:
    return MinimumWeightDecoder(repetition_matching_graph(distance, code=code),
                                code=f"repetition-{distance}-{code}", distance=distance,
                                name=name)


def repetition_lookup_decoder(distance: int = 3, *, code: str = "bit_flip",
                              name: str = "lookup") -> LookupDecoder:
    """The distance-3 table, which is the decoder the code started with.

    Only distance 3: at distance 5 a syndrome can come from a weight-2 error and the
    single-error table has nothing to say, which is the reason the matching decoder
    exists.
    """
    if distance != 3:
        raise DecoderRefusalError(
            f"the lookup table is written for distance 3; distance {distance} has "
            f"syndromes no weight-1 table can answer -- use the matching decoder")
    letter = "X" if code == "bit_flip" else "Z"
    table: dict[tuple[bool, ...], str] = {tuple([False] * (distance - 1)): "I" * distance}
    for qubit in range(distance):
        syndrome = [False] * (distance - 1)
        for check in range(distance - 1):
            syndrome[check] = qubit in (check, check + 1)
        labels = ["I"] * distance
        labels[qubit] = letter
        table[tuple(syndrome)] = "".join(labels)
    return LookupDecoder(table, code=f"repetition-{distance}-{code}", distance=distance,
                         name=name)
