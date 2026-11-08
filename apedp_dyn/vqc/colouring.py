"""Minimum edge colourings: the schedule of parallel ``cz`` layers, with a proof.

Status: experimental

A graph-state circuit is ``h`` on every qubit followed by one ``cz`` per edge, and
commuting ``cz`` gates can share a layer exactly when the edges do not touch -- a
matching. The number of layers a schedule needs is therefore the number of colours in
an **edge colouring**, and the minimum possible is the graph's edge chromatic number
``chi'(G)``. This module computes a minimum colouring, not a heuristic one.

Why a new scheduler
-------------------
The previous family (`graph_layers`) peeled the edge set greedily: take any maximal
matching, remove it, repeat. That is a valid colouring, but nothing about a maximal
matching is minimum. On the shared corpus's 4-regular 20-vertex graph it used 6 layers
where 4 suffice, on the complete graph ``K5`` it used 7 where 5 suffice, and it can
miss ``Delta`` even when ``Delta`` colours exist. A greedy peel is a schedule, not an
optimum, and the old record could only claim it *attained* the degree floor on the
targets where it happened to.

The bound
---------
Vizing's theorem gives ``Delta <= chi'(G) <= Delta + 1`` for every simple graph, so
there are exactly two cases to decide:

* a ``Delta``-colouring exists -- then it is optimal, because no colouring can use
  fewer than ``Delta`` colours at a vertex of maximum degree;
* no ``Delta``-colouring exists (the graph is *class 2*, e.g. an odd cycle, the
  Petersen graph, or the complete graph on an odd number of vertices) -- then
  ``chi' = Delta + 1`` and a ``(Delta + 1)``-colouring is optimal.

This module searches for a ``Delta``-colouring by complete backtracking. If it finds
one, the optimality proof is the degree lower bound. If the search **exhausts** the
space without one, that exhaustion *is* the proof that ``chi' > Delta``; a
``Delta + 1`` colouring is then constructed and is optimal by the same theorem. Every
result says which of those two arguments it rests on, because "proven optimal" and
"the best I found" are not the same claim.

One refutation is cheaper than the search, and it is checked first. A layer is a
matching, so it holds at most ``floor(V / 2)`` edges and ``Delta`` layers cover at most
``Delta * floor(V / 2)``: when the graph has more edges than that, no ``Delta``-colouring
exists and no search can change it. That is what decides the complete graphs on an odd
number of vertices -- ``K9``'s 36 edges do not fit into eight layers of four, ``K11``'s
55 do not fit into ten of five -- with one multiplication instead of an exponential
search, and it is what makes that whole family a measured refutation rather than an
unfinished run.

The ``Delta + 1`` half is *constructed*, not searched
-----------------------------------------------------
Vizing's theorem is constructive: the proof of ``chi' <= Delta + 1`` is an algorithm
(fan rotations, coloured with the Misra-Gries form of the argument) that colours any
edge in one pass given ``Delta + 1`` colours. This module implements it as
:func:`vizing_colouring`, so a ``(Delta + 1)``-colouring is always available and never
has to be searched for. That matters exactly where the search cannot finish: deciding
whether a ``Delta``-colouring exists is NP-complete, so the ``Delta`` search carries a
node budget, and a graph that exhausts it (``K9`` at a small budget, say) still gets a
schedule at ``Delta + 1`` layers -- nine rather than the peel's fifteen -- instead of a
schedule with no bound at all. The construction is verified before it is returned: it
is a real edge colouring of the input or it raises.

What is not claimed
-------------------
A graph whose ``Delta`` search exhausts the budget cannot be classified, so it is
returned with ``optimal=False`` and the reason recorded; it is never reported as
minimal, even though its ``Delta + 1`` schedule is only one layer above the theorem's
lower bound. The greedy peel survives as a last resort for the case the construction's
self-check refuses, which is a bug in this module rather than a property of the input,
and is then labelled as a schedule.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

__all__ = [
    "ColouringConstructionError",
    "ColouringError",
    "EdgeColouring",
    "greedy_peel",
    "minimum_edge_colouring",
    "vizing_colouring",
]

#: Nodes the complete ``Delta``-colouring search may visit before it gives up. The
#: graphs this scheduler is used on are at most a few hundred edges; the budget is a
#: refusal threshold for pathological instances, not a tuning knob.
DEFAULT_BUDGET = 500_000


class ColouringError(ValueError):
    """An edge list that is not a simple graph this module can colour."""


class ColouringConstructionError(ColouringError):
    """The ``Delta + 1`` construction failed its own verification.

    Vizing's theorem says this cannot happen on a simple graph, so a raise here is a
    defect in this module: the partial colouring is discarded rather than returned.
    """


@dataclasses.dataclass(frozen=True)
class EdgeColouring:
    """A colouring of the edges into ``cz`` layers, and the argument for its size."""

    layers: tuple[tuple[tuple[int, int], ...], ...]
    num_colours: int
    max_degree: int
    lower_bound: int
    optimal: bool
    delta_refuted: bool
    nodes: int
    budget: int
    status: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "cz_layers": self.num_colours,
            "max_degree": self.max_degree,
            "lower_bound": self.lower_bound,
            "edge_chromatic_number": self.num_colours if self.optimal else None,
            "optimal": self.optimal,
            "delta_refuted": self.delta_refuted,
            "search_nodes": self.nodes,
            "search_budget": self.budget,
            "status": self.status,
            "reason": self.reason,
            "layer_sizes": [len(layer) for layer in self.layers],
        }


def _validate(edges: Sequence[tuple[int, int]], num_nodes: int | None) -> list[tuple[int, int]]:
    cleaned = []
    seen = set()
    for edge in edges:
        if len(edge) != 2:
            raise ColouringError(f"an edge needs two endpoints, got {edge}")
        first, second = int(edge[0]), int(edge[1])
        if first == second:
            raise ColouringError(f"a self-loop is not an edge of a graph state: {edge}")
        key = (min(first, second), max(first, second))
        if key in seen:
            raise ColouringError(f"edge {key} is listed twice")
        seen.add(key)
        cleaned.append(key)
    if not cleaned:
        raise ColouringError("a graph state needs at least one edge")
    largest = max(max(edge) for edge in cleaned)
    if num_nodes is not None and largest >= num_nodes:
        raise ColouringError(
            f"edge endpoint {largest} is out of range for {num_nodes} vertices")
    return sorted(cleaned)


def greedy_peel(edges: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """The previous scheduler: repeated maximal matchings, for comparison only.

    Kept so the record can report what the replacement saves rather than asserting
    that the old schedule was suboptimal. It is a valid colouring and is used as the
    fallback when the exact search exhausts its budget.
    """
    remaining = sorted(edges)
    layers: list[list[tuple[int, int]]] = []
    while remaining:
        used: set[int] = set()
        layer: list[tuple[int, int]] = []
        rest: list[tuple[int, int]] = []
        for edge in remaining:
            if edge[0] in used or edge[1] in used:
                rest.append(edge)
            else:
                used.update(edge)
                layer.append(edge)
        layers.append(layer)
        remaining = rest
    return layers


def _degrees(edges: Sequence[tuple[int, int]], num_nodes: int) -> list[int]:
    degree = [0] * num_nodes
    for first, second in edges:
        degree[first] += 1
        degree[second] += 1
    return degree


def _layer_capacity(num_nodes: int) -> int:
    """The most edges one layer can hold: a matching pairs vertices up, so at most
    ``floor(num_nodes / 2)``. Weaker than the true matching number on a sparse graph and
    never wrong, which is all a refutation needs."""
    return num_nodes // 2


class _BudgetExceeded(Exception):
    pass


def _search(edges: Sequence[tuple[int, int]], num_nodes: int, colours: int,
            budget: int) -> tuple[list[int] | None, int, bool]:
    """A complete backtracking search for a `colours`-edge-colouring.

    Returns ``(colouring_or_None, nodes, exhausted)``. ``None`` with
    ``exhausted=True`` means the whole space was searched and no colouring exists --
    the refutation that proves the graph is class 2. ``None`` with
    ``exhausted=False`` means the node budget was hit, so no conclusion is drawn.
    """
    count = len(edges)
    incident: list[list[int]] = [[] for _ in range(num_nodes)]
    for index, (first, second) in enumerate(edges):
        incident[first].append(index)
        incident[second].append(index)
    assignment = [-1] * count
    used = [0] * num_nodes
    remaining = [len(edges_at) for edges_at in incident]
    full = (1 << colours) - 1
    nodes = 0

    def candidates(index: int) -> int:
        first, second = edges[index]
        return full & ~(used[first] | used[second])

    def pick() -> tuple[int, int, int]:
        best, best_mask, best_count = -1, 0, colours + 1
        for index in range(count):
            if assignment[index] != -1:
                continue
            mask = candidates(index)
            size = mask.bit_count()
            if size < best_count:
                best, best_mask, best_count = index, mask, size
                if size == 0:
                    break
        return best, best_mask, best_count

    def descend(coloured: int, highest: int) -> bool:
        nonlocal nodes
        if coloured == count:
            return True
        nodes += 1
        if nodes > budget:
            raise _BudgetExceeded
        # Every uncoloured edge at a vertex needs a *different* colour that is still
        # free there, so a vertex with fewer free colours than uncoloured edges cannot
        # be completed. This is a necessary condition, so pruning on it loses nothing;
        # on dense graphs it is what makes the refutation searchable at all.
        for vertex in range(num_nodes):
            if remaining[vertex] > colours - used[vertex].bit_count():
                return False
        index, mask, size = pick()
        if size == 0:
            return False
        first, second = edges[index]
        # Colour symmetry: a colour never used so far is only worth trying once, and
        # the next new colour is `highest + 1`, never an arbitrary unused index.
        limit = min(colours - 1, highest + 1)
        while mask:
            bit = mask & -mask
            mask ^= bit
            colour = bit.bit_length() - 1
            if colour > limit:
                continue
            assignment[index] = colour
            used[first] |= bit
            used[second] |= bit
            remaining[first] -= 1
            remaining[second] -= 1
            if descend(coloured + 1, max(highest, colour)):
                return True
            assignment[index] = -1
            used[first] &= ~bit
            used[second] &= ~bit
            remaining[first] += 1
            remaining[second] += 1
        return False

    try:
        found = descend(0, -1)
    except _BudgetExceeded:
        return None, nodes, False
    return (assignment if found else None), nodes, True


def _layers_from(edges: Sequence[tuple[int, int]], assignment: Sequence[int],
                 colours: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    layers: list[list[tuple[int, int]]] = [[] for _ in range(colours)]
    for edge, colour in zip(edges, assignment, strict=True):
        layers[colour].append(edge)
    return tuple(tuple(sorted(layer)) for layer in layers)


# --------------------------------------------------------------------- Delta + 1
#
# Vizing's proof of ``chi' <= Delta + 1`` is an algorithm, and this is it in the
# Misra-Gries form. Colours are carried as ``vertex -> neighbour -> colour`` (every
# colour is used at most once per vertex, so a colour names at most one neighbour).


def _colours_at(colouring: dict[int, dict[int, int]], vertex: int) -> set[int]:
    return set(colouring.get(vertex, {}).values())


def _free_colours(colouring: dict[int, dict[int, int]], vertex: int,
                  colours: int) -> list[int]:
    taken = _colours_at(colouring, vertex)
    return [colour for colour in range(colours) if colour not in taken]


def _neighbour_with_colour(colouring: dict[int, dict[int, int]], vertex: int,
                           colour: int) -> int | None:
    for neighbour, other in colouring.get(vertex, {}).items():
        if other == colour:
            return neighbour
    return None


def _set_colour(colouring: dict[int, dict[int, int]], first: int, second: int,
                colour: int) -> None:
    colouring.setdefault(first, {})[second] = colour
    colouring.setdefault(second, {})[first] = colour


def _clear_colour(colouring: dict[int, dict[int, int]], first: int,
                  second: int) -> None:
    del colouring[first][second]
    del colouring[second][first]


def _maximal_fan(colouring: dict[int, dict[int, int]], x: int, y: int,
                 colours: int) -> list[int]:
    """A maximal fan at ``x``: ``[y, ...]`` where ``(x, fan[i])`` carries a colour that
    is free at ``fan[i - 1]``, extended until no such vertex is left."""
    fan = [y]
    in_fan = {y}
    while True:
        last = fan[-1]
        taken = _colours_at(colouring, last)
        extended = False
        for colour in range(colours):
            if colour in taken:
                continue
            candidate = _neighbour_with_colour(colouring, x, colour)
            if candidate is None or candidate in in_fan:
                continue
            fan.append(candidate)
            in_fan.add(candidate)
            extended = True
            break
        if not extended:
            return fan


def _invert_cd_path(colouring: dict[int, dict[int, int]], x: int, c: int,
                    d: int) -> None:
    """Swap ``c`` and ``d`` on the maximal ``c``/``d`` path through ``x``.

    The path is collected first and inverted afterwards, so the walk cannot be flipped
    while it is being read. A ``c``/``d`` subgraph is a disjoint union of paths and
    even cycles, so the component through ``x`` is exactly the path the argument needs
    -- including the case where it closes into a cycle.
    """
    if c == d:
        return
    component: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    seen_vertices = {x}
    stack = [x]
    while stack:
        vertex = stack.pop()
        for neighbour, colour in list(colouring.get(vertex, {}).items()):
            if colour not in (c, d):
                continue
            pair = (min(vertex, neighbour), max(vertex, neighbour))
            if pair in seen_edges:
                continue
            seen_edges.add(pair)
            component.append((vertex, neighbour))
            if neighbour not in seen_vertices:
                seen_vertices.add(neighbour)
                stack.append(neighbour)
    for first, second in component:
        _set_colour(colouring, first, second,
                    d if colouring[first][second] == c else c)


def _fan_condition(colouring: dict[int, dict[int, int]], x: int, previous: int,
                   current: int) -> bool:
    """Is the colour of ``(x, current)`` free at ``previous``?"""
    colour = colouring.get(x, {}).get(current)
    if colour is None:
        return False
    return colour not in _colours_at(colouring, previous)


def _fan_prefix_end(colouring: dict[int, dict[int, int]], x: int, fan: Sequence[int],
                    d: int) -> int:
    """The last index whose prefix ``fan[0..index]`` is a fan with ``d`` free at its tip.

    These are the conditions the rotation restores, checked on the colours *after* the
    path inversion, since that is what the placement then relies on.
    """
    end = -1
    for index in range(len(fan)):
        if d in _colours_at(colouring, fan[index]):
            continue
        if all(_fan_condition(colouring, x, fan[step], fan[step + 1])
               for step in range(index)):
            end = index
    return end


def _extend_colouring(colouring: dict[int, dict[int, int]], x: int, y: int,
                      colours: int) -> None:
    """Give the uncoloured edge ``(x, y)`` a colour, or raise.

    One fan, one inversion, one rotation: the Misra-Gries procedure colours a single
    edge in a single pass, given ``Delta + 1`` colours and a proper partial colouring.
    """
    fan = _maximal_fan(colouring, x, y, colours)
    # A colour free at x, and one free at the tip of the fan (Delta + 1 colours make
    # both exist: no vertex carries more than Delta coloured edges).
    c = _free_colours(colouring, x, colours)[0]
    d = _free_colours(colouring, fan[-1], colours)[0]
    _invert_cd_path(colouring, x, c, d)
    if d in _colours_at(colouring, x):
        raise ColouringConstructionError(
            f"inverting the {c}/{d} path at {x} did not free colour {d}; "
            f"Vizing's argument does not produce this on a simple graph")
    end = _fan_prefix_end(colouring, x, fan, d)
    if end < 0:
        raise ColouringConstructionError(
            f"no fan prefix around {x} both keeps its colours free and frees {d}")
    # Rotate: the colour of (x, fan[i + 1]) moves to (x, fan[i]), which frees the edge
    # (x, fan[end]) for `d`. Read every colour before clearing any of them, so no edge
    # is momentarily coloured twice.
    moved = [colouring[x][fan[step + 1]] for step in range(end)]
    for step in range(end + 1):
        if fan[step] in colouring.get(x, {}):
            _clear_colour(colouring, x, fan[step])
    for step in range(end):
        _set_colour(colouring, x, fan[step], moved[step])
    _set_colour(colouring, x, fan[end], d)


def vizing_colouring(edges: Sequence[tuple[int, int]], *,
                     num_nodes: int | None = None) -> list[list[tuple[int, int]]]:
    """A ``Delta + 1`` edge colouring, constructed by Vizing's fan argument.

    Returns layers in the same shape as :func:`greedy_peel`: a list of matchings whose
    union is the edge set. Empty layers are dropped, so ``len(result) <= Delta + 1``
    holds and equals the number of colours actually used.

    The construction is verified before it is returned -- proper, complete, within the
    bound -- because the alternative is a schedule that silently puts two gates on one
    qubit. A verification failure raises :class:`ColouringConstructionError` rather
    than returning the partial colouring.
    """
    graph = _validate(edges, num_nodes)
    vertices = num_nodes if num_nodes is not None else 1 + max(
        max(edge) for edge in graph)
    colours = max(_degrees(graph, vertices)) + 1
    colouring: dict[int, dict[int, int]] = {}
    for first, second in graph:
        _extend_colouring(colouring, first, second, colours)
    layers: list[list[tuple[int, int]]] = [[] for _ in range(colours)]
    for first, second in graph:
        colour = colouring.get(first, {}).get(second)
        if colour is None:
            raise ColouringConstructionError(f"edge {(first, second)} was left uncoloured")
        layers[colour].append((first, second))
    used = [sorted(layer) for layer in layers if layer]
    seen: set[tuple[int, int]] = set()
    for layer in used:
        touched: set[int] = set()
        for edge in layer:
            if edge in seen:
                raise ColouringConstructionError(f"edge {edge} was coloured twice")
            if touched & set(edge):
                raise ColouringConstructionError(
                    f"two edges touching in one layer: {sorted(touched)} and {edge}")
            touched.update(edge)
            seen.add(edge)
    if len(used) > colours:
        raise ColouringConstructionError(
            f"the construction used {len(used)} colours, above the Delta + 1 = {colours} "
            f"that Vizing's theorem allows")
    return used


def minimum_edge_colouring(edges: Sequence[tuple[int, int]], *,
                           num_nodes: int | None = None,
                           budget: int = DEFAULT_BUDGET) -> EdgeColouring:
    """A minimum ``cz``-layer schedule, with the argument for its minimality.

    Tries the greedy peel first: when it reaches ``Delta`` layers it is already
    optimal by the degree lower bound, and no search is run. Then the layer-capacity
    bound: when the edges do not fit into ``Delta`` layers at ``floor(V / 2)`` edges
    each, ``Delta`` is refuted by counting
    (``status: delta_refuted_by_layer_capacity``) and the constructed ``Delta + 1``
    colouring is optimal. Otherwise a complete search looks for a
    ``Delta``-colouring; if it finds one the result is optimal, and if it exhausts the
    space the graph is *proved* class 2, which makes the ``Delta + 1`` colouring from
    :func:`vizing_colouring` optimal too. A search that only runs out of budget decides
    nothing, so the result is marked ``optimal=False`` -- but it is still the
    constructed ``Delta + 1`` schedule, which Vizing bounds to at most one layer above
    the minimum, not the unbounded peel.
    """
    if budget < 1:
        raise ColouringError(f"budget must be positive, got {budget}")
    graph = _validate(edges, num_nodes)
    vertices = num_nodes if num_nodes is not None else 1 + max(
        max(edge) for edge in graph)
    degree = _degrees(graph, vertices)
    delta = max(degree)
    peel = greedy_peel(graph)

    def result(layers: Any, *, optimal: Any, delta_refuted: Any, nodes: Any, status: Any, reason: Any,
               lower_bound: Any) -> Any:
        return EdgeColouring(
            layers=tuple(tuple(layer) for layer in layers),
            num_colours=len(layers),
            max_degree=delta,
            lower_bound=lower_bound,
            optimal=optimal,
            delta_refuted=delta_refuted,
            nodes=nodes,
            budget=budget,
            status=status,
            reason=reason,
        )

    if len(peel) == delta:
        _out: EdgeColouring = result(
            peel, optimal=True, delta_refuted=False, nodes=0,
            status="greedy_attained_delta",
            reason=(f"the greedy peel already uses Delta = {delta} layers, and no "
                    f"colouring can use fewer (a maximum-degree vertex needs one per "
                    f"edge)"),
            lower_bound=delta)
        return _out

    # A layer is a matching, so it holds at most floor(V / 2) edges and Delta layers
    # cover at most Delta * floor(V / 2). When the edge count exceeds that, no
    # Delta-colouring exists at all and the graph is class 2 by counting -- no search,
    # no budget, and no doubt about which case it is. This is what decides the odd
    # complete graphs the search cannot: K9's 36 edges do not fit into 8 layers of 4,
    # and K11's 55 do not fit into 10 layers of 5.
    capacity = delta * _layer_capacity(vertices)
    if len(graph) > capacity:
        try:
            constructed = vizing_colouring(graph, num_nodes=vertices)
        except ColouringConstructionError as error:
            _out: EdgeColouring = result(
                peel, optimal=False, delta_refuted=True, nodes=0,
                status="construction_refused",
                reason=(f"a Delta = {delta} colouring cannot exist ({len(graph)} edges "
                        f"cannot fit into {delta} layers of at most "
                        f"{_layer_capacity(vertices)} edges), "
                        f"but the Delta + 1 construction failed its own verification "
                        f"({error}); the greedy peel is reported as a schedule"),
                lower_bound=delta + 1)
            return _out
        _out: EdgeColouring = result(
            constructed, optimal=True, delta_refuted=True, nodes=0,
            status="delta_refuted_by_layer_capacity",
            reason=(f"no Delta = {delta} colouring can exist: every layer is a matching "
                    f"on {vertices} vertices, so it holds at most "
                    f"{_layer_capacity(vertices)} edges and {delta} layers cover at most "
                    f"{capacity} of the graph's {len(graph)}; the Delta + 1 = {delta + 1} "
                    f"colouring Vizing's fan construction produced is therefore minimal "
                    f"(chi' = Delta + 1)"),
            lower_bound=delta + 1)
        return _out

    assignment, nodes, exhausted = _search(graph, vertices, delta, budget)
    if assignment is not None:
        layers = _layers_from(graph, assignment, delta)
        _out: EdgeColouring = result(
            layers, optimal=True, delta_refuted=False, nodes=nodes,
            status="delta_colouring_found",
            reason=(f"a Delta = {delta} colouring exists, searched in {nodes} nodes; "
                    f"Delta is a lower bound, so {delta} layers is minimal"),
            lower_bound=delta)
        return _out

    # Vizing's theorem is constructive: a Delta + 1 colouring is built, not searched
    # for, so neither remaining branch has to fall back on the peel. A graph at a
    # small budget still gets a schedule within one layer of the minimum -- K9 gets its
    # nine layers here where the peel uses fifteen.
    try:
        constructed = vizing_colouring(graph, num_nodes=vertices)
    except ColouringConstructionError as error:
        _out: EdgeColouring = result(
            peel, optimal=False, delta_refuted=exhausted, nodes=nodes,
            status="construction_refused",
            reason=(f"the Delta + 1 construction failed its own verification "
                    f"({error}); the greedy peel is reported as a schedule, not as an "
                    f"optimum"),
            lower_bound=delta + 1 if exhausted else delta)
        return _out

    if not exhausted:
        # Which of Delta and Delta + 1 is chi' was not decided. The construction is at
        # most one layer above the minimum and is reported as exactly that, never as a
        # minimum.
        _out: EdgeColouring = result(
            constructed, optimal=False, delta_refuted=False, nodes=nodes,
            status="delta_undecided_within_one",
            reason=(f"the Delta = {delta} search hit its {budget}-node budget without "
                    f"a colouring or a refutation, so whether chi' is Delta or "
                    f"Delta + 1 was not decided; Vizing's fan construction produced a "
                    f"Delta + 1 = {delta + 1} colouring, which is at most one layer "
                    f"above the minimum, but it is not called minimal"),
            lower_bound=delta)
        return _out

    # The Delta space was exhausted: no Delta-colouring exists, so chi' = Delta + 1 by
    # Vizing's theorem and the constructed colouring is optimal.
    _out: EdgeColouring = result(
        constructed, optimal=True, delta_refuted=True, nodes=nodes,
        status="delta_refuted",
        reason=(f"the Delta = {delta} search exhausted its space without a "
                f"colouring, so no Delta-colouring exists; Vizing's fan construction "
                f"then produced a Delta + 1 = {delta + 1} colouring, which the "
                f"refutation makes minimal (chi' = Delta + 1)"),
        lower_bound=delta + 1)
    return _out
