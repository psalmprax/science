"""Pins for the Watts-Strogatz graph source added to Experiment E.

The family exists to be a graph source that is *not* the lattice and *not* the seeded
tree: triangles, odd cycles, and a degree distribution set by the rewire. These pins keep
it honest -- reproducible from its seed, refused when its parameters are impossible, and
measured against the ring lattice and the random-rewired control rather than asserted to
be small-world.
"""
from __future__ import annotations

import pytest

from apedp_dyn.vqc.synthesis import (
    SynthesisError,
    adjudicate_specification,
    build,
    default_families,
    graph_of_state,
    small_world_shape,
    watts_strogatz_target,
)

GRAPH_FAMILIES = ("graph_raster", "graph_column_major", "graph_tree_first",
                  "graph_layers", "graph_random_order")


def test_the_draw_is_reproducible_from_its_seed():
    first = watts_strogatz_target(32, k=4, beta=0.1, seed=7)
    again = watts_strogatz_target(32, k=4, beta=0.1, seed=7)
    other = watts_strogatz_target(32, k=4, beta=0.1, seed=8)
    assert graph_of_state(first) == graph_of_state(again)
    assert first.sha256() == again.sha256()
    assert graph_of_state(first) != graph_of_state(other)


def test_zero_rewiring_is_the_ring_lattice():
    target = watts_strogatz_target(24, k=4, beta=0.0, seed=0)
    edges = set(graph_of_state(target))
    expected = set()
    for offset in (1, 2):
        for vertex in range(24):
            pair = tuple(sorted((vertex, (vertex + offset) % 24)))
            expected.add(pair)
    assert edges == expected
    assert len(edges) == 2 * 24  # 2n edges, every vertex of degree 4


def test_the_rewire_lowers_clustering_at_the_same_degree_and_seed():
    ring = small_world_shape(32, k=4, beta=0.0, seed=0)
    small_world = small_world_shape(32, k=4, beta=0.1, seed=0)
    randomised = small_world_shape(32, k=4, beta=1.0, seed=0)
    assert ring["mean_clustering"] > small_world["mean_clustering"]
    assert small_world["mean_clustering"] > randomised["mean_clustering"]
    # A degree-4 ring lattice has 3 joined pairs among each vertex's six neighbour
    # pairs -- (v-2,v-1), (v-1,v+1) and (v+1,v+2) -- so its clustering is 1/2 exactly.
    assert ring["mean_clustering"] == pytest.approx(0.5, abs=1e-9)
    assert ring["mean_degree"] == pytest.approx(4.0)


def test_the_generator_refuses_impossible_parameters():
    with pytest.raises(SynthesisError):
        watts_strogatz_target(10, k=3)  # odd ring degree
    with pytest.raises(SynthesisError):
        watts_strogatz_target(10, k=10)  # k not smaller than the graph
    with pytest.raises(SynthesisError):
        watts_strogatz_target(10, k=4, beta=1.5)
    with pytest.raises(SynthesisError):
        watts_strogatz_target(3, k=2)


def test_the_graph_families_apply_and_the_state_verifies():
    target = watts_strogatz_target(32, k=4, beta=0.1, seed=0)
    assert set(default_families(target)) == set(GRAPH_FAMILIES)
    construction = build("graph_layers", target)
    report = adjudicate_specification(construction.circuit, target)
    assert report.verdict == "VALID", report.failed()


def test_the_schedule_never_beats_the_degree_floor():
    target = watts_strogatz_target(32, k=4, beta=0.1, seed=0)
    degrees = dict.fromkeys(range(target.num_qubits), 0)
    for parent, child in graph_of_state(target):
        degrees[parent] += 1
        degrees[child] += 1
    floor = max(degrees.values())
    construction = build("graph_layers", target)
    layers = construction.circuit.depth() - 1
    assert layers >= floor
    # Vizing: Delta <= chi' <= Delta + 1, so the minimum schedule is one of the two.
    assert layers in (floor, floor + 1)


def test_a_rewired_graph_is_not_the_lattice_with_the_same_edge_count():
    ring = watts_strogatz_target(32, k=4, beta=0.0, seed=0)
    rewired = watts_strogatz_target(32, k=4, beta=0.1, seed=0)
    assert graph_of_state(ring) != graph_of_state(rewired)
    assert len(graph_of_state(rewired)) == len(graph_of_state(ring))
