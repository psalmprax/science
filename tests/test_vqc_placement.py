"""Architecture-aware placement: what a construction costs on a real topology.

Logical depth is the number Experiment A quotes; this file pins the second number,
the one the review asked for as Experiment D, and the properties that make it
trustworthy rather than decorative:

* a topology is a graph, and an unreachable node or a two-node ring is refused
  instead of routed around silently;
* every two-qubit gate in a routed circuit is between adjacent nodes -- the one
  promise a router makes, checked by walking the emitted circuit;
* routing is a *relabelling*, so swapping the routing swaps back makes the mapped
  circuit the same unitary, and the specification referee must therefore accept it
  with both of its tiers;
* the numbers are not wishful: all-to-all reproduces the logical depth exactly, a
  line reproduces a chain exactly, and every other pair is more expensive than the
  logical circuit, never less.
"""
from __future__ import annotations

import pytest

from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.placement import (
    CONNECTIVITIES,
    Connectivity,
    PlacementError,
    all_to_all,
    grid,
    line,
    place,
    ring,
)
from apedp_dyn.vqc.synthesis import (
    adjudicate_specification,
    build,
    ghz_target,
    referee_agreement,
    ring_graph_target,
)


# -------------------------------------------------------------------- topologies
def test_a_topology_is_a_graph_and_stays_connected():
    assert line(4).adjacent(1, 2) and not line(4).adjacent(1, 3)
    assert ring(4).adjacent(0, 3) and not line(4).adjacent(0, 3)
    assert grid(9).name == "grid_3x3" and len(grid(9).edges) == 12
    assert all_to_all(5).adjacent(0, 4) and all_to_all(5).degree(0) == 4
    assert grid(9).adjacent(0, 1) and grid(9).adjacent(0, 3)
    assert not grid(9).adjacent(0, 4)
    # A lattice that does not factor is refused rather than guessed at.
    with pytest.raises(PlacementError, match="factor"):
        grid(13)


def test_a_topology_that_cannot_be_routed_on_is_refused():
    with pytest.raises(PlacementError, match="not connected"):
        Connectivity("split", 4, [(0, 1), (2, 3)])
    with pytest.raises(PlacementError, match="at least three nodes"):
        ring(2)
    with pytest.raises(PlacementError, match="self-loop"):
        Connectivity("loop", 3, [(1, 1)])


def test_shortest_paths_are_shortest():
    assert line(6).distance(0, 5) == 5
    assert ring(6).distance(0, 5) == 1
    assert grid(9).distance(0, 8) == 4
    assert all_to_all(9).distance(0, 8) == 1
    assert line(5).shortest_path(0, 3) == (0, 1, 2, 3)


# ---------------------------------------------------------------------- routing
@pytest.mark.parametrize("topology", sorted(CONNECTIVITIES))
@pytest.mark.parametrize("strategy", ("greedy", "lookahead"))
def test_every_two_qubit_gate_in_a_routed_circuit_is_between_neighbours(topology,
                                                                        strategy):
    target = ghz_target(9)
    connectivity = CONNECTIVITIES[topology](9)
    placement = place(build("balanced_tree", target).circuit, connectivity,
                      strategy=strategy)
    for op in placement.circuit.ops:
        if isinstance(op, Gate) and len(op.targets) == 2:
            assert connectivity.adjacent(*op.targets), (topology, strategy, op)


@pytest.mark.parametrize("topology", sorted(CONNECTIVITIES))
@pytest.mark.parametrize("strategy", ("greedy", "lookahead"))
def test_a_routed_circuit_still_prepares_the_specification(topology, strategy):
    """The restore ladder is what makes the mapped circuit checkable, not just fast."""
    target = ghz_target(9)
    connectivity = CONNECTIVITIES[topology](9)
    placement = place(build("balanced_tree", target).circuit, connectivity,
                      strategy=strategy)
    assert placement.restored is True
    assert placement.permutation == tuple(range(9))
    report = adjudicate_specification(placement.circuit, target)
    assert report.verdict == "VALID", (topology, strategy, report.failed())
    # And both referees agree about the routed circuit, not just one of them.
    agreement = referee_agreement(placement.circuit, target)
    assert agreement.agree is True and agreement.dense_verdict == "VALID"


# ---------------------------------------------------------------- the lookahead
def test_the_router_accepts_only_the_strategies_it_measures():
    circuit = build("chain", ghz_target(6)).circuit
    with pytest.raises(PlacementError, match="strategy must be"):
        place(circuit, line(6), strategy="fastest")
    with pytest.raises(PlacementError, match="non-negative"):
        place(circuit, line(6), strategy="lookahead", lookahead=-1)


def test_the_lookahead_is_recorded_as_the_router_that_ran():
    circuit = build("balanced_tree", ghz_target(12)).circuit
    greedy = place(circuit, grid(12))
    lookahead = place(circuit, grid(12), strategy="lookahead")
    assert greedy.router == "greedy" and greedy.lookahead_window is None
    assert lookahead.router == "lookahead" and lookahead.lookahead_window == 8
    assert greedy.as_dict()["router"] == "greedy"
    assert lookahead.as_dict()["lookahead_window"] == 8
    # Same topology, same circuit, different router: the circuits must differ.
    assert greedy.circuit.sha256() != lookahead.circuit.sha256()


def test_the_lookahead_wins_where_the_greedy_router_pays_for_a_sweep():
    """A chain on a lattice: the greedy router sweeps a wire across the grid."""
    target = ghz_target(12)
    circuit = build("chain", target).circuit
    greedy = place(circuit, grid(12))
    lookahead = place(circuit, grid(12), strategy="lookahead")
    assert lookahead.mapped_depth < greedy.mapped_depth
    assert lookahead.swap_gates < greedy.swap_gates
    assert lookahead.two_qubit_gates < greedy.two_qubit_gates
    assert adjudicate_specification(lookahead.circuit, target).verdict == "VALID"


def test_the_lookahead_is_deterministic():
    circuit = build("balanced_tree", ghz_target(16)).circuit
    first = place(circuit, line(16), strategy="lookahead")
    second = place(circuit, line(16), strategy="lookahead")
    assert first.circuit.sha256() == second.circuit.sha256()
    assert first.swap_gates == second.swap_gates


# ---------------------------------------------------------------- swap accounting
def test_swap_accounting_separates_abstract_native_and_cx_equivalent_counts():
    """A SWAP is one operation, one native instruction, or three CNOTs."""
    target = ghz_target(12)
    circuit = build("balanced_tree", target).circuit
    placement = place(circuit, grid(12))
    assert placement.swap_gates > 0
    assert placement.total_swaps == 2 * placement.swap_gates
    assert placement.two_qubit_gates == (placement.logical_two_qubit_gates
                                         + 2 * placement.swap_gates)
    # With the default basis (a native SWAP) the native count is the abstract one.
    assert placement.swap_entanglers == 1
    assert placement.native_entanglers == placement.two_qubit_gates
    assert placement.cx_equivalent_two_qubit_gates == (
        placement.logical_two_qubit_gates + 3 * placement.total_swaps)
    assert placement.cx_equivalent_two_qubit_gates > placement.two_qubit_gates
    payload = placement.as_dict()
    assert payload["abstract_two_qubit_gates"] == payload["two_qubit_gates"]
    assert payload["swap_as_cnot"] == 3
    assert payload["cx_equivalent_two_qubit_gates"] == \
        placement.cx_equivalent_two_qubit_gates


def test_a_cnot_only_basis_is_a_declared_swap_cost():
    circuit = build("balanced_tree", ghz_target(12)).circuit
    native = place(circuit, grid(12))
    decomposed = place(circuit, grid(12), swap_entanglers=3)
    assert decomposed.native_entanglers == decomposed.cx_equivalent_two_qubit_gates
    assert decomposed.native_entanglers == native.cx_equivalent_two_qubit_gates
    assert decomposed.two_qubit_gates == native.two_qubit_gates
    with pytest.raises(PlacementError, match="at least one entangler"):
        place(circuit, grid(12), swap_entanglers=0)


def test_the_reported_depth_includes_restoration_and_the_restore_cost_is_separate():
    target = ghz_target(12)
    placement = place(build("balanced_tree", target).circuit, line(12))
    assert placement.restored is True
    # The reported mapped depth is the circuit that executes, restore ladder included.
    assert placement.mapped_depth == placement.circuit.depth()
    assert placement.depth_before_restore < placement.mapped_depth
    assert placement.restore_depth_cost == (placement.mapped_depth
                                            - placement.depth_before_restore)
    payload = placement.as_dict()
    assert payload["mapped_depth"] == placement.circuit.depth()
    assert payload["restore_depth_cost"] == placement.restore_depth_cost


def test_all_to_all_is_the_logical_circuit_and_a_line_is_a_free_chain():
    target = ghz_target(12)
    chain = build("chain", target).circuit
    free = place(chain, all_to_all(12))
    assert free.swap_gates == free.restore_swap_gates == 0
    assert free.mapped_depth == free.logical_depth == chain.depth()
    assert free.mapped_operations == free.logical_operations

    flat = place(chain, line(12))
    assert flat.swap_gates == 0 and flat.mapped_depth == flat.logical_depth
    assert flat.longest_hop == 1


def test_a_tree_pays_for_its_connectivity():
    """The finding Experiment D exists to report: depth 12 is not depth 12 on hardware."""
    target = ghz_target(12)
    tree = build("balanced_tree", target).circuit
    logical = tree.depth()
    for topology in ("line", "ring", "grid"):
        placement = place(tree, CONNECTIVITIES[topology](12))
        assert placement.logical_depth == logical
        assert placement.mapped_depth > logical, topology
        assert placement.two_qubit_gates > placement.logical_two_qubit_gates
        assert placement.routing_overhead > 0.0
        # A routed circuit never gets cheaper than the abstract one.
        assert placement.mapped_depth >= placement.depth_before_restore >= logical


def test_the_restore_ladder_is_reported_rather_than_hidden():
    target = ghz_target(12)
    tree = build("balanced_tree", target).circuit
    with_restore = place(tree, line(12))
    without = place(tree, line(12), restore=False)
    assert without.restored is False
    assert without.restore_swap_gates == 0
    assert without.permutation != tuple(range(12))
    assert with_restore.depth_before_restore == without.mapped_depth
    # Undoing the routing doubles the swap bill and is quoted separately.
    assert with_restore.restore_swap_gates == with_restore.swap_gates
    assert with_restore.mapped_depth > without.mapped_depth


def test_a_three_qubit_gate_on_a_line_is_refused_not_squeezed():
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("ccx", (0, 1, 2))), num_qubits=3,
                      num_bits=0)
    with pytest.raises(PlacementError, match="mutually adjacent"):
        place(circuit, line(3))
    # On an all-to-all device it is expressible as written.
    placement = place(circuit, all_to_all(3))
    assert placement.swap_gates == 0


def test_a_circuit_wider_than_the_device_is_refused():
    with pytest.raises(PlacementError, match="has 4 nodes"):
        place(build("chain", ghz_target(5)).circuit, line(4))


def test_a_larger_device_leaves_the_extra_wires_idle():
    placement = place(build("chain", ghz_target(4)).circuit, grid(9))
    assert placement.circuit.num_qubits == 9
    assert placement.permutation == tuple(range(4))
    assert placement.as_dict()["permutation_is_identity"] is True
    touched = {qubit for op in placement.circuit.ops for qubit in op.targets}
    assert touched <= set(range(4)), touched


def test_the_placement_record_carries_both_depths_and_its_hash():
    target = ghz_target(6)
    placement = place(build("chain", target).circuit, ring(6))
    record = placement.as_dict()
    assert record["logical_depth"] == placement.logical_depth
    assert record["mapped_depth"] == placement.mapped_depth
    assert record["total_swaps"] == placement.swap_gates + placement.restore_swap_gates
    assert record["circuit_sha256"] == placement.circuit.sha256()
    assert record["connectivity"] == "ring"
    assert record["nodes"] == 6
    assert record["edges"] == 6


def test_a_ring_graph_state_survives_a_lattice():
    """A second target, because a GHZ is permutation-symmetric and a ring is not."""
    target = ring_graph_target(6)
    placement = place(build("ring", target).circuit, grid(6))
    report = adjudicate_specification(placement.circuit, target)
    assert report.verdict == "VALID", report.failed()
    assert placement.swap_gates > 0
