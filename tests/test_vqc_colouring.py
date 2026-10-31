"""Minimum edge colouring: a matching decomposition with an argument for its size.

The scheduler replaced a greedy peel, so the pins have to do more than check that a
colouring is valid:

* every layer really is a matching and every edge appears exactly once;
* the reported count equals the number of layers and, when ``optimal``, equals the
  proven lower bound (``Delta``, or ``Delta + 1`` after the ``Delta`` space was
  exhausted);
* class-2 graphs are recognised as class 2 by a **proof** -- a counting bound when the
  edges cannot fit into ``Delta`` layers, otherwise an exhaustive ``Delta`` search --
  never by assuming Vizing's extra colour is necessary, and the counting bound is
  cross-checked against the exhaustive search on every small graph;
* a search that hits its budget is never labelled optimal -- but it still returns a
  usable schedule, because the ``Delta + 1`` half of Vizing's theorem is constructive;
* that construction is a real colouring on every graph it is given, checked
  exhaustively on small graphs and on the families where fans actually rotate.
"""
from __future__ import annotations

import itertools

import pytest

from apedp_dyn.vqc import colouring as colouring_module
from apedp_dyn.vqc.colouring import (
    ColouringError,
    greedy_peel,
    minimum_edge_colouring,
    vizing_colouring,
)

C5 = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
PETERSEN = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5))
K5 = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
K9 = tuple((a, b) for a in range(9) for b in range(a + 1, 9))
K11 = tuple((a, b) for a in range(11) for b in range(a + 1, 11))
CUBE = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7))


def _assert_matching_decomposition(layers, edges):
    seen = set()
    for layer in layers:
        used: set[int] = set()
        for edge in layer:
            assert edge not in seen, f"{edge} appears twice"
            assert not (used & set(edge)), f"{edge} collides in one layer"
            used.update(edge)
            seen.add(edge)
    assert seen == {tuple(sorted(edge)) for edge in edges}


def test_a_delta_colouring_is_a_proved_minimum_and_a_real_schedule():
    schedule = minimum_edge_colouring(CUBE)
    assert schedule.optimal is True
    assert schedule.delta_refuted is False
    assert schedule.max_degree == 3
    assert schedule.num_colours == 3
    assert schedule.as_dict()["edge_chromatic_number"] == 3
    _assert_matching_decomposition(schedule.layers, CUBE)
    assert all(len(layer) > 0 for layer in schedule.layers)


def test_an_odd_cycle_is_class_two_by_counting_the_layers():
    """Five edges cannot fit into two matchings of two: `Delta` is refuted without a search."""
    schedule = minimum_edge_colouring(C5)
    assert schedule.max_degree == 2
    assert schedule.num_colours == 3
    assert schedule.optimal is True
    assert schedule.delta_refuted is True
    assert schedule.lower_bound == 3
    assert schedule.status == "delta_refuted_by_layer_capacity"
    assert schedule.nodes == 0                     # the search was never needed
    assert "every layer is a matching" in schedule.reason
    assert "no Delta = 2 colouring can exist" in schedule.reason
    _assert_matching_decomposition(schedule.layers, C5)


def test_the_petersen_graph_is_proved_class_two_by_exhausting_the_delta_space():
    """The layer count is tight here (15 edges, 3 layers of 5), so the search must run."""
    assert len(greedy_peel(PETERSEN)) == 4      # happens to be optimal here
    schedule = minimum_edge_colouring(PETERSEN)
    assert schedule.delta_refuted is True
    assert schedule.num_colours == schedule.max_degree + 1 == 4
    assert schedule.optimal is True
    assert schedule.status == "delta_refuted"
    assert schedule.nodes > 0                      # proved by exhaustion, not by counting
    assert "exhausted its space" in schedule.reason
    _assert_matching_decomposition(schedule.layers, PETERSEN)


def test_the_complete_graph_on_five_vertices_saves_two_layers():
    """Greedy peel uses seven layers here; the minimum is five."""
    assert len(greedy_peel(K5)) == 7
    schedule = minimum_edge_colouring(K5)
    assert schedule.optimal is True
    assert schedule.delta_refuted is True
    assert schedule.max_degree == 4
    assert schedule.num_colours == 5
    assert schedule.status == "delta_refuted_by_layer_capacity"
    assert schedule.nodes == 0
    assert len(greedy_peel(K5)) - schedule.num_colours == 2
    _assert_matching_decomposition(schedule.layers, K5)


def test_the_odd_complete_graphs_are_refuted_without_a_budget():
    """`K9` and `K11`: the family's refutation is measured, not left to an exhausted search.

    Their `Delta`-colouring search cannot finish (`K9` needs more than five million
    nodes, `K11` far more), so without the counting bound the family could only ever be
    reported as undecided. The bound refutes `Delta` at zero nodes -- 36 edges do not fit
    into eight layers of four, 55 do not fit into ten layers of five -- and the
    constructed `Delta + 1` schedule is then minimal by Vizing.
    """
    for graph, nodes, layers in ((K9, 9, 4), (K11, 11, 5)):
        assert len(greedy_peel(graph)) == 15
        for budget in (1, 200):
            schedule = minimum_edge_colouring(graph, budget=budget)
            assert schedule.max_degree == nodes - 1
            assert schedule.num_colours == schedule.max_degree + 1 == nodes
            assert schedule.status == "delta_refuted_by_layer_capacity"
            assert schedule.optimal is True and schedule.delta_refuted is True
            assert schedule.nodes == 0
            assert schedule.as_dict()["edge_chromatic_number"] == nodes
            _assert_matching_decomposition(schedule.layers, graph)
            assert all(len(layer) == layers for layer in schedule.layers)
        assert len(vizing_colouring(graph, num_nodes=nodes)) == nodes


def test_a_refutation_by_counting_does_not_depend_on_the_budget():
    """`K7` minus an edge is class 2 by the layer bound, so no budget can change the answer.

    The bound is checked before the search, which is what makes it a proof: the same
    seven layers and the same optimality come back at a one-node budget and at a budget
    the search would never finish inside.
    """
    graph = tuple((a, b) for a in range(7) for b in range(a + 1, 7)
                  if (a, b) != (0, 2))
    answers = []
    for budget in (1, 150):
        schedule = minimum_edge_colouring(graph, budget=budget)
        assert schedule.optimal is True
        assert schedule.delta_refuted is True
        assert schedule.status == "delta_refuted_by_layer_capacity"
        assert schedule.num_colours == schedule.max_degree + 1 == 7
        assert schedule.num_colours < len(greedy_peel(graph))
        assert schedule.as_dict()["edge_chromatic_number"] == 7
        _assert_matching_decomposition(schedule.layers, graph)
        answers.append(schedule.layers)
    assert answers[0] == answers[1]                 # the construction is deterministic


def test_a_budget_exhausted_search_still_returns_a_constructed_schedule():
    """`budget=1` decides nothing, so the answer is the `Delta + 1` schedule, not the peel.

    The Petersen graph is the case where the counting bound is tight: only the search can
    refute `Delta`, and a one-node budget stops it. Four layers is also the minimum, but
    this run did not prove that and does not claim it.
    """
    schedule = minimum_edge_colouring(PETERSEN, budget=1)
    assert schedule.optimal is False
    assert schedule.delta_refuted is False          # nothing was proved either way
    assert schedule.num_colours == schedule.max_degree + 1 == 4
    assert schedule.num_colours <= len(greedy_peel(PETERSEN)) == 4
    assert schedule.status == "delta_undecided_within_one"
    assert "at most one layer above the minimum" in schedule.reason
    assert "not called minimal" in schedule.reason
    assert schedule.as_dict()["edge_chromatic_number"] is None
    _assert_matching_decomposition(schedule.layers, PETERSEN)


def test_the_layer_bound_never_refutes_a_graph_that_has_a_delta_colouring():
    """Every small graph: a class-2 verdict from the bound must survive the full search.

    The bound is a counting argument, so it can only be trusted if it never disagrees
    with the exhaustive ``Delta`` search it replaces. On all 1,094 graphs of at most five
    vertices, each claimed refutation is re-derived by the complete search.
    """
    checked = 0
    for nodes in range(2, 6):
        pairs = list(itertools.combinations(range(nodes), 2))
        for mask in range(1, 1 << len(pairs)):
            graph = [pairs[index] for index in range(len(pairs)) if mask >> index & 1]
            schedule = minimum_edge_colouring(graph, num_nodes=nodes)
            if not schedule.delta_refuted:
                continue
            found, _, exhausted = colouring_module._search(
                graph, nodes, schedule.max_degree, 200_000)
            assert found is None and exhausted, (graph, schedule.status)
            checked += 1
    assert checked > 0


def test_the_construction_is_a_valid_colouring_of_every_small_graph():
    """Every simple graph on at most five vertices: matchings, and at most `Delta + 1` layers."""
    checked = 0
    for nodes in range(2, 6):
        pairs = list(itertools.combinations(range(nodes), 2))
        for mask in range(1, 1 << len(pairs)):
            graph = [pairs[index] for index in range(len(pairs)) if mask >> index & 1]
            layers = vizing_colouring(graph, num_nodes=nodes)
            _assert_matching_decomposition(layers, graph)
            degree = [0] * nodes
            for first, second in graph:
                degree[first] += 1
                degree[second] += 1
            assert len(layers) <= max(degree) + 1
            assert all(layers)                                   # no empty layer
            checked += 1
    assert checked == 1094


@pytest.mark.parametrize("name,graph", [
    ("complete_9", tuple((a, b) for a in range(9) for b in range(a + 1, 9))),
    ("cycle_9", tuple((index, (index + 1) % 9) for index in range(9))),
    ("petersen", PETERSEN),
    ("hypercube_4", tuple(sorted({(min(vertex, vertex ^ (1 << bit)),
                                   max(vertex, vertex ^ (1 << bit)))
                                  for vertex in range(16) for bit in range(4)}))),
    ("complete_bipartite_5x5", tuple((left, 5 + right)
                                     for left in range(5) for right in range(5))),
])
def test_the_construction_colours_the_structured_families(name, graph):
    """Fans, inversions and rotations all fire here; the layer count must stay bounded."""
    nodes = 1 + max(max(edge) for edge in graph)
    layers = vizing_colouring(graph, num_nodes=nodes)
    _assert_matching_decomposition(layers, graph)
    degree = [0] * nodes
    for first, second in graph:
        degree[first] += 1
        degree[second] += 1
    assert len(layers) <= max(degree) + 1, name


def test_the_construction_refuses_the_same_edge_lists_as_the_search():
    with pytest.raises(ColouringError, match="self-loop"):
        vizing_colouring(((0, 0), (0, 1)))
    with pytest.raises(ColouringError, match="twice"):
        vizing_colouring(((0, 1), (1, 0)))
    with pytest.raises(ColouringError, match="out of range"):
        vizing_colouring(((0, 1),), num_nodes=1)
    with pytest.raises(ColouringError, match="at least one edge"):
        vizing_colouring(())


def test_a_construction_that_fails_its_check_falls_back_to_the_labelled_peel(monkeypatch):
    """The peel is a last resort for a defect here, and is reported as a schedule only."""
    def broken(*args, **kwargs):
        raise colouring_module.ColouringConstructionError("injected")

    monkeypatch.setattr(colouring_module, "_extend_colouring", broken)
    schedule = minimum_edge_colouring(K5, budget=1)
    assert schedule.num_colours == len(greedy_peel(K5)) == 7
    assert schedule.status == "construction_refused"
    assert schedule.optimal is False
    assert "injected" in schedule.reason
    _assert_matching_decomposition(schedule.layers, K5)


def test_the_greedy_peel_is_valid_but_not_minimal():
    """Kept so the saving is measured against the schedule it replaced."""
    peel = greedy_peel(K5)
    _assert_matching_decomposition(peel, K5)
    assert len(peel) > minimum_edge_colouring(K5).num_colours


def test_an_edge_list_that_is_not_a_simple_graph_is_refused():
    with pytest.raises(ColouringError, match="self-loop"):
        minimum_edge_colouring(((0, 0), (0, 1)))
    with pytest.raises(ColouringError, match="twice"):
        minimum_edge_colouring(((0, 1), (1, 0)))
    with pytest.raises(ColouringError, match="out of range"):
        minimum_edge_colouring(((0, 1),), num_nodes=1)
    with pytest.raises(ColouringError, match="at least one edge"):
        minimum_edge_colouring(())
    with pytest.raises(ColouringError, match="budget"):
        minimum_edge_colouring(C5, budget=0)
