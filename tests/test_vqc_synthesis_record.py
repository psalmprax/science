"""Pins for `benchmarks/quantum_compilation/vqc_synthesis.json`.

The record is the evidence for the review's Experiment A (construct from a
specification, no reference circuit) and Experiment B (the same at 10..1,000
qubits), plus the boundary of Experiment C. What is pinned here:

* no reference circuit anywhere in Experiment A -- the key is `null` and the note
  says what the only inputs were;
* every construction verifies *and* the wrong ones are refused with the failing
  stabilizer named, because a synthesis record that only shows successes is not
  evidence of a working referee;
* the ranking and the Pareto frontier are consistent with the measured rows, and the
  frontier keeps the nearest-neighbour chain even though it is not the shallowest --
  depth and connectivity trade off and the record says so instead of picking one;
* the admission is the same four-field policy as the rewrites loop, including the
  case where a valid construction is refused for missing its claim;
* Experiment C's boundary is where the record says it is: decided within the dense
  bound, `UNRESOLVED` above it.
* Experiment E's graph states are specifications, not circuits: the edge set is the
  target's, so the schedules agree on operations and connectivity and differ only in
  depth, and the peeled schedule's depth is checked against the degree floor.

Timings are not pinned; the recorded verdicts and hashes are.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_synthesis.json"
SCRIPT = ROOT / "scripts" / "measure_vqc_synthesis.py"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_the_record_is_the_schema_it_claims_and_every_check_passed(record):
    assert record["schema"] == "apedp.vqc.synthesis.v3"
    assert record["checks"] and all(record["checks"].values()), record["checks"]


def test_experiment_a_supplies_no_reference_circuit(record):
    experiment = record["experiment_a"]
    assert experiment["reference_circuit"] is None
    assert "the only inputs are the qubit count and the signed Pauli stabilizers" in \
        experiment["reference_circuit_note"]
    specification = experiment["specification"]
    assert specification["name"] == "ghz"
    assert specification["qubits"] == record["command"]["qubits"]
    assert specification["stabilizer_count"] == specification["qubits"]
    assert specification["sha256"]
    assert "circuit" not in specification


def test_every_construction_reaches_the_specification(record):
    experiment = record["experiment_a"]
    rows = {row["family"]: row for row in experiment["constructions"]}
    assert set(rows) == {"chain", "reverse_chain", "star", "balanced_tree", "random_tree"}
    for name, row in rows.items():
        assert row["verdict"] == "VALID", name
        assert row["tier"] == "stabilizer"
        assert row["checks_failed"] == []
        assert row["operations"] == row["two_qubit_gates"] + 1
        assert row["circuit_sha256"]
    # One two-qubit gate per non-root qubit, in every wiring.
    assert {row["two_qubit_gates"] for row in rows.values()} == {experiment["specification"]["qubits"] - 1}
    # The wiring is what differs, and the measurement sees it.
    assert rows["chain"]["nearest_neighbour_only"] is True
    assert rows["chain"]["max_edge_span"] == 1
    assert rows["balanced_tree"]["depth"] < rows["chain"]["depth"]
    assert rows["star"]["max_edge_span"] == experiment["specification"]["qubits"] - 1


def test_the_ranking_and_the_frontier_follow_the_measured_rows(record):
    experiment = record["experiment_a"]
    rows = {row["family"]: row for row in experiment["constructions"]}
    ranking = experiment["ranking"]
    assert len(ranking) == len(rows)
    assert rows[ranking[0]]["depth"] == min(row["depth"] for row in rows.values())
    assert experiment["pareto_frontier"]
    # The nearest-neighbour chain stays on the frontier even though it is not the
    # shallowest: that trade-off is the point of reporting a frontier.
    assert "chain" in experiment["pareto_frontier"]
    assert rows[ranking[0]]["max_edge_span"] > rows["chain"]["max_edge_span"]
    # A star is not on it: a chain is at least as deep and needs shorter wires.
    assert "star" not in experiment["pareto_frontier"]


def test_every_near_miss_is_refused_with_the_generator_that_failed(record):
    refusals = record["experiment_a"]["refusals"]
    assert len(refusals) == 4
    for row in refusals:
        assert row["verdict"] == "INVALID", row["mutation"]
        assert row["checks_failed"] == [
            "the prepared state is the +1 eigenstate of every specified stabilizer"]
        assert row["failed_count"] >= 1
        assert row["failed_generators"][0]["pauli"].startswith("+")


def test_the_admission_carries_the_four_fields_and_promoted_the_best(record):
    admission = record["experiment_a"]["admission"]
    assert admission["semantic_verdict"] == "VALID"
    assert admission["resource_verdict"] == "MET"
    assert admission["evidence_sufficiency"] == "SUFFICIENT"
    assert admission["promotion_decision"] == "PROMOTED"
    assert record["experiment_a"]["promoted"] == admission["candidate_id"]
    assert admission["candidate_id"].startswith("spec-ghz-")
    assert admission["tier"] == "stabilizer"


def test_the_second_target_is_a_different_state_and_the_near_miss_is_refused(record):
    second = record["second_target"]
    assert second["specification"]["name"] == "ring_graph"
    verdicts = {row["family"]: row["verdict"] for row in second["constructions"]}
    assert verdicts["ring"] == "VALID"
    assert verdicts["linear_cluster"] == "INVALID"
    assert second["promoted"] is not None and "ring" in second["promoted"]
    near_miss = second["near_miss"]
    assert near_miss["verdict"] == "INVALID"
    assert near_miss["family"] == "linear_cluster"


def test_a_valid_construction_that_misses_its_claim_is_not_promoted(record):
    claim = record["claim_refusal"]
    assert claim["claims"] == {"depth": 1}
    assert claim["semantic_verdict"] == "VALID"
    assert claim["resource_verdict"] == "UNMET"
    assert claim["evidence_sufficiency"] == "SUFFICIENT"
    assert claim["promoted"] is None
    assert "does not improve its resource claim" in claim["refusal"]
    assert "depth: claimed 1" in claim["refusal"]


def test_experiment_b_scales_to_a_thousand_qubits_and_verifies_every_row(record):
    experiment = record["experiment_b"]
    assert experiment["sizes"] == [10, 50, 100, 250, 500, 1000]
    rows = experiment["rows"]
    assert len(rows) == len(experiment["sizes"]) * 4
    assert {row["qubits"] for row in rows} == set(experiment["sizes"])
    for row in rows:
        assert row["verdict"] == "VALID", (row["qubits"], row["family"])
        assert row["tier"] == "stabilizer"
        assert row["operations"] == row["qubits"]
        assert row["stabilizers_checked"] == row["qubits"]
        assert row["verification_seconds"] > 0
    by_key = {(row["qubits"], row["family"]): row for row in rows}
    for size in experiment["sizes"]:
        assert by_key[(size, "chain")]["nearest_neighbour_only"] is True
        assert by_key[(size, "balanced_tree")]["depth"] < by_key[(size, "chain")]["depth"]
        assert by_key[(size, "balanced_tree")]["max_edge_span"] > \
            by_key[(size, "chain")]["max_edge_span"]


def test_experiment_c_boundary_is_where_the_record_says_it_is(record):
    boundary = record["experiment_c"]
    limit = boundary["dense_spec_limit"]
    rows = boundary["rows"]
    decided = [row for row in rows if row["tier"] == "dense"]
    undecided = [row for row in rows if row["tier"] == "none"]
    assert decided and undecided
    assert all(row["qubits"] <= limit and row["verdict"] == "VALID" for row in decided)
    assert all(row["qubits"] > limit and row["verdict"] == "UNRESOLVED" for row in undecided)
    assert all(row["is_clifford"] is False for row in rows)
    assert all(row["reason"] and "above the dense specification bound" in row["reason"]
               for row in undecided)
    # Decided exactly, not approximately: the distance is rounding.
    assert all(row["distance"] is not None and row["distance"] < 1e-12 for row in decided)
    assert "no non-Clifford search exists" in boundary["note"]


def test_experiment_d_prices_the_topology_each_construction_runs_on(record):
    """Logical depth is not hardware depth, and the record carries both numbers."""
    experiment = record["experiment_d"]
    assert experiment["qubits"] == record["command"]["qubits"]
    keys = {(row["family"], row["topology"], row["router"])
            for row in experiment["rows"]}
    assert keys == {(family, topology, router)
                    for family in ("chain", "balanced_tree")
                    for topology in ("line", "ring", "grid", "all_to_all")
                    for router in ("greedy", "lookahead")}
    for row in experiment["rows"]:
        assert row["verdict"] == "VALID", (row["family"], row["topology"], row["router"])
        assert row["tier"] == "stabilizer"
        assert row["stabilizers_checked"] == experiment["qubits"]
        assert row["restored"] is True and row["permutation_is_identity"] is True
        # A routed circuit is never cheaper than the abstract one, and the restore
        # ladder is on top of the routing rather than folded into it.
        assert row["mapped_depth"] >= row["depth_before_restore"] >= row["logical_depth"]
        assert row["total_swaps"] == row["swap_gates"] + row["restore_swap_gates"]
        assert row["restore_swap_gates"] == row["swap_gates"]
        assert row["two_qubit_gates"] >= row["logical_two_qubit_gates"]
        # The three bases are recomputable from the row, not independently typed.
        assert row["abstract_two_qubit_gates"] == row["two_qubit_gates"]
        assert row["total_swaps"] == 2 * row["swap_gates"]
        assert row["two_qubit_gates"] == (row["logical_two_qubit_gates"]
                                          + 2 * row["swap_gates"])
        assert row["native_entanglers"] == row["two_qubit_gates"]  # native SWAP basis
        assert row["cx_equivalent_two_qubit_gates"] == (
            row["logical_two_qubit_gates"] + 3 * row["total_swaps"])
        assert row["restore_depth_cost"] == (row["mapped_depth"]
                                             - row["depth_before_restore"])
        assert row["circuit_sha256"]
        if row["router"] == "greedy":
            assert row["lookahead_window"] is None
        else:
            assert row["lookahead_window"] == experiment["lookahead_window"] == 8
    rows = {(row["family"], row["topology"], row["router"]): row
            for row in experiment["rows"]}
    # All-to-all is the logical circuit, so the measurement has a known zero point.
    for family in ("chain", "balanced_tree"):
        for router in ("greedy", "lookahead"):
            assert rows[(family, "all_to_all", router)]["mapped_depth"] == \
                rows[(family, "all_to_all", router)]["logical_depth"]
            assert rows[(family, "all_to_all", router)]["total_swaps"] == 0
    # A chain is free on a line; a tree that is ten times shallower is not, even
    # after the lookahead router has done its work.
    for router in ("greedy", "lookahead"):
        assert rows[("chain", "line", router)]["swap_gates"] == 0
        assert rows[("chain", "line", router)]["mapped_depth"] == \
            rows[("chain", "line", router)]["logical_depth"]
    tree = rows[("balanced_tree", "line", "lookahead")]
    chain = rows[("chain", "line", "lookahead")]
    assert tree["logical_depth"] < chain["logical_depth"]
    assert tree["mapped_depth"] > chain["mapped_depth"]
    assert tree["longest_hop"] > 1
    assert "greedy" in experiment["greedy_router"]
    assert "lookahead" in experiment["lookahead_router"]


def test_experiment_d_compares_both_routers_row_by_row(record):
    """The lookahead is measured against the greedy numbers, not announced."""
    experiment = record["experiment_d"]
    assert experiment["routers"] == ["greedy", "lookahead"]
    comparisons = experiment["comparison"]
    assert len(comparisons) == 8
    assert {(row["family"], row["topology"]) for row in comparisons} == {
        (family, topology)
        for family in ("chain", "balanced_tree")
        for topology in ("line", "ring", "grid", "all_to_all")}
    for row in comparisons:
        greedy, lookahead = row["greedy"], row["lookahead"]
        # Never worse, and cheaper where the sweep of the greedy router is avoidable.
        assert lookahead["mapped_depth"] <= greedy["mapped_depth"], row
        assert lookahead["depth_before_restore"] <= greedy["depth_before_restore"]
        assert lookahead["swap_gates"] <= greedy["swap_gates"], row
        assert lookahead["abstract_two_qubit_gates"] <= \
            greedy["abstract_two_qubit_gates"], row
        assert lookahead["native_entanglers"] <= greedy["native_entanglers"], row
        assert lookahead["cx_equivalent_two_qubit_gates"] <= \
            greedy["cx_equivalent_two_qubit_gates"], row
        # The savings are recomputable from the two columns, not independently typed.
        assert row["mapped_depth_saved"] == greedy["mapped_depth"] - lookahead["mapped_depth"]
        assert row["swaps_saved"] == greedy["swap_gates"] - lookahead["swap_gates"]
        assert row["two_qubit_gates_saved"] == (
            greedy["abstract_two_qubit_gates"] - lookahead["abstract_two_qubit_gates"])
        assert row["mapped_depth_ratio"] == pytest.approx(
            lookahead["mapped_depth"] / greedy["mapped_depth"])
    summary = experiment["summary"]
    assert summary["rows_worse"] == 0
    assert summary["rows_improved"] + summary["rows_unchanged"] == len(comparisons)
    assert summary["rows_improved"] >= 1
    assert summary["mapped_depth_saved_total"] == sum(
        row["mapped_depth_saved"] for row in comparisons)
    assert summary["swaps_saved_total"] == sum(row["swaps_saved"] for row in comparisons)
    assert summary["best_mapped_depth_ratio"] == pytest.approx(
        min(row["mapped_depth_ratio"] for row in comparisons))


def test_experiment_d_reports_every_two_qubit_count_in_three_bases(record):
    """The accounting the review asked for: abstract, native, CNOT-equivalent."""
    experiment = record["experiment_d"]
    accounting = experiment["swap_accounting"]
    assert accounting["swap_entanglers"] == 1
    assert accounting["swap_as_cnot"] == 3
    assert "three bases" in accounting["note"]
    comparisons = {(row["family"], row["topology"]): row
                   for row in experiment["comparison"]}
    tree = comparisons[("balanced_tree", "grid")]["greedy"]
    # The reviewer's arithmetic: 99 + 2(547) abstract, 99 + 6(547) CX-equivalent.
    assert tree["abstract_two_qubit_gates"] == 1193
    assert tree["abstract_two_qubit_gates"] == 99 + 2 * 547
    assert tree["cx_equivalent_two_qubit_gates"] == 3381
    assert tree["cx_equivalent_two_qubit_gates"] == 99 + 6 * 547
    lookahead = comparisons[("balanced_tree", "grid")]["lookahead"]
    assert lookahead["cx_equivalent_two_qubit_gates"] == (
        lookahead["logical_two_qubit_gates"] + 3 * lookahead["total_swaps"])
    assert lookahead["cx_equivalent_two_qubit_gates"] < tree["cx_equivalent_two_qubit_gates"]
    saved = comparisons[("balanced_tree", "grid")]["cx_equivalent_saved"]
    assert saved == (tree["cx_equivalent_two_qubit_gates"]
                     - lookahead["cx_equivalent_two_qubit_gates"])
    assert experiment["summary"]["cx_equivalent_saved_total"] == sum(
        row["cx_equivalent_saved"] for row in experiment["comparison"])
    # Rows with no routing have no gap between the bases, which is the zero point.
    for row in experiment["comparison"]:
        for side in (row["greedy"], row["lookahead"]):
            if side["swap_gates"] == 0:
                assert side["cx_equivalent_two_qubit_gates"] == side["abstract_two_qubit_gates"]


def test_experiment_d_lookahead_moves_the_headline_numbers(record):
    """The finding: depth 12 is not depth 12, and neither is the greedy bill."""
    comparisons = {(row["family"], row["topology"]): row
                   for row in record["experiment_d"]["comparison"]}
    tree_grid = comparisons[("balanced_tree", "grid")]
    assert tree_grid["greedy"]["mapped_depth"] == 220
    assert tree_grid["lookahead"]["mapped_depth"] < 100
    assert tree_grid["mapped_depth_ratio"] < 0.5
    chain_grid = comparisons[("chain", "grid")]
    assert chain_grid["lookahead"]["mapped_depth"] < chain_grid["greedy"]["mapped_depth"]
    assert chain_grid["lookahead"]["swap_gates"] < chain_grid["greedy"]["swap_gates"]
    # Rows that need no routing are the zero point: the lookahead cannot improve them.
    for key, row in comparisons.items():
        if row["greedy"]["swap_gates"] == 0:
            assert row["lookahead"]["mapped_depth"] == row["greedy"]["mapped_depth"], key
            assert row["mapped_depth_saved"] == 0


def test_both_referees_answered_and_agreed_on_every_circuit(record):
    agreement = record["referee_agreement"]
    assert agreement["all_agree"] is True
    rows = agreement["rows"]
    accepted = [row for row in rows if row["stabilizer_verdict"] == "VALID"]
    assert agreement["accepted"] == len(accepted)
    assert agreement["refused"] == len(rows) - len(accepted) >= 1
    for row in rows:
        assert row["agree"] is True
        assert row["stabilizer_verdict"] == row["dense_verdict"]
        assert row["dense_distance"] is not None
    # The accepted ones are decided exactly: the distance is rounding, not tolerance.
    assert all(row["dense_distance"] < 1e-12 for row in accepted)
    # A wrong construction is in the corpus, so the agreement is not only on accepts.
    refused = next(row for row in rows if row["stabilizer_verdict"] == "INVALID")
    assert refused["dense_distance"] > 1e-9
    assert refused["family"] == "chain_mutated_cx_to_cz"


def test_the_wide_circuits_are_decided_twice_by_two_implementations(record):
    """400 and 900 qubits: the dense tier cannot run there, so a package Stim can.

    Two-tier agreement stops at ten qubits, so each wide row is this package's
    stabilizer tier against an implementation it did not write. The dense tier's
    refusal at those widths is in the corpus as rows of its own, and an abstention is
    never allowed to read as agreement.
    """
    wide = record["referee_agreement"]["wide"]
    assert [row["qubits"] for row in wide["rows"]] == [400, 900]
    assert [row["target"] for row in wide["rows"]] == ["cluster_20x20",
                                                       "cluster_30x30"]
    # A disagreement is a contradiction, not a row: the script refuses to write it.
    assert "raises and writes no record" in wide["disagreement_policy"]
    assert wide["all_agree"] is True
    timed = {(entry["target"], entry["family"]): entry
             for entry in record["experiment_e"]["rows"]}
    for row in wide["rows"]:
        assert row["stabilizer"]["tier"] == "stabilizer"
        assert row["stabilizer"]["verdict"] == "VALID"
        assert row["stabilizer"]["checks_failed"] == []
        referee = row["second_referee"]
        assert referee["referee"] == "stim"
        assert referee["qubits"] == row["qubits"]
        if referee["decides_the_circuit"]:
            assert referee["verdict"] == "VALID"
            assert referee["generators"] == referee["satisfied"] == row["qubits"]
            assert referee["failing"] == []
            assert row["agree"] is True
            assert "two independent decisions" in row["agree_note"]
        else:
            # An absent or refusing third party is an abstention with a reason, never
            # a pass and never a disagreement.
            assert referee["verdict"] in {"UNAVAILABLE", "REFUSED"}
            assert referee["reason"]
            assert row["agree"] is None
            assert "not compared" in row["agree_note"]
        # The circuit decided twice is the circuit Experiment E timed, not a lookalike.
        entry = timed[(row["target"], "graph_layers")]
        assert row["circuit_sha256"] == entry["circuit_sha256"]
        assert entry["verdict"] == row["stabilizer"]["verdict"]
        assert entry["qubits"] == row["qubits"]
    # The row that cannot exist: the dense tier raises at these widths, and the corpus
    # stores each refusal instead of dropping the case.
    assert [entry["qubits"] for entry in wide["dense_tier"]] == [400, 900]
    for dense in wide["dense_tier"]:
        assert dense["refused"] is True
        assert dense["tier"] == "dense" and dense["verdict"] == "REFUSED"
        assert dense["bound"] == record["experiment_c"]["dense_spec_limit"]
        assert "bounded at" in dense["reason"]
        assert str(dense["bound"]) in dense["reason"]


def test_the_wide_corpus_refuses_the_wrong_graph_constructions(record):
    """A third referee that is only ever seen accepting says nothing about refusal."""
    wide = record["referee_agreement"]["wide"]
    refusals = {row["family"]: row for row in wide["refusals"]}
    assert set(refusals) == {"cluster_drop_edge", "cluster_extra_edge",
                             "graph_swap_gate"}
    experiment = {row["family"]: row for row in record["experiment_e"]["refusals"]}
    for family, row in refusals.items():
        assert row["qubits"] == 400
        assert row["stabilizer"]["verdict"] == "INVALID"
        # The same construction family Experiment E refuses on a 12-qubit cluster is
        # refused here at 400 qubits, by the same referee and by the third party.
        assert experiment[family]["verdict"] == "INVALID"
        referee = row["second_referee"]
        if referee["decides_the_circuit"]:
            assert referee["verdict"] == "INVALID"
            assert row["agree"] is True
            assert referee["satisfied"] < referee["generators"]
            # The third party names the generators it failed, not just a count.
            assert referee["failing"]
            assert all(entry["expected_expectation"]
                       != entry["observed_expectation"] for entry in referee["failing"])
        else:
            assert row["agree"] is None
            assert referee["verdict"] in {"UNAVAILABLE", "REFUSED"}
            assert referee["reason"]


def test_the_specifications_two_readings_agree_everywhere_both_run(record):
    oracle = record["dense_oracle"]
    assert oracle["all_agree"] is True
    ghz_rows = [row for row in oracle["rows"] if row["target"] == "ghz"]
    ring_rows = [row for row in oracle["rows"] if row["target"] == "ring_graph"]
    assert len(ghz_rows) == record["experiment_c"]["dense_spec_limit"] - 2
    assert ring_rows and all(row["non_zero_amplitudes"] == 2 ** row["qubits"]
                             for row in ring_rows)


def test_experiment_e_specifies_graph_states_by_their_edges(record):
    """A graph state is the graph: the record cites edges, degrees and the floor."""
    experiment = record["experiment_e"]
    targets = {target["description"]: target for target in experiment["targets"]}
    assert set(targets) == {"cluster_4x4", "cluster_10x10", "cluster_20x20",
                            "cluster_30x30", "irregular_25q_seed7",
                            "irregular_64q_seed23", "complete_k9"}
    cluster = targets["cluster_10x10"]
    assert cluster["specification"]["name"] == "cluster_2d"
    assert cluster["specification"]["qubits"] == 100
    assert cluster["edges"] == 180
    assert cluster["max_degree"] == 4
    large = targets["cluster_20x20"]
    assert large["specification"]["qubits"] == 400
    assert large["specification"]["stabilizer_count"] == 400
    assert large["edges"] == 2 * 400 - 20 - 20 == 760
    assert large["max_degree"] == large["cz_layers_floor"] == 4
    # The largest target: 900 qubits on a 30x30 lattice, 1,740 edges, still degree 4.
    biggest = targets["cluster_30x30"]
    assert biggest["specification"]["qubits"] == 900
    assert biggest["specification"]["stabilizer_count"] == 900
    assert biggest["edges"] == 2 * 900 - 30 - 30 == 1740
    assert biggest["max_degree"] == biggest["cz_layers_floor"] == 4
    irregular = targets["irregular_64q_seed23"]
    assert irregular["specification"]["name"] == "irregular_graph"
    assert irregular["specification"]["qubits"] == 64
    # More edges than a tree: the draw is irregular, not a path in disguise.
    assert irregular["edges"] > irregular["specification"]["qubits"] - 1
    for target in targets.values():
        specification = target["specification"]
        assert specification["sha256"]
        assert specification["stabilizer_count"] == specification["qubits"]
        assert target["max_degree"] == target["cz_layers_floor"]
        assert target["promoted"] and target["promoted"].endswith("-graph_layers")
    # A cluster's degree is bounded by the lattice, whatever its size.
    assert targets["cluster_4x4"]["max_degree"] == 4
    # K9 is the class-2 case inside Experiment E: no 8-layer schedule exists, so the
    # measured minimum is one above the degree floor and the peel's 15 is shown against
    # a built, verified 9 in the same table as the clusters.
    k9 = targets["complete_k9"]
    assert k9["specification"]["qubits"] == 9
    assert k9["edges"] == 36
    assert k9["max_degree"] == k9["cz_layers_floor"] == 8
    assert k9["scheduler"]["cz_layers"] == k9["scheduler"]["edge_chromatic_number"] == 9
    assert k9["scheduler"]["greedy_cz_layers"] == 15
    assert k9["scheduler"]["layers_saved"] == 6
    assert k9["scheduler"]["delta_refuted"] is True
    assert k9["scheduler"]["status"] == "delta_refuted_by_layer_capacity"


def test_experiment_e_the_schedule_is_the_only_free_variable(record):
    """The specification fixes the wiring; only the depth is the family's to choose."""
    experiment = record["experiment_e"]
    schedules = {"graph_raster", "graph_column_major", "graph_tree_first",
                 "graph_layers", "graph_random_order"}
    rows = experiment["rows"]
    assert set(experiment["shared_edge_set"]["by_target"]) == {
        target["description"] for target in experiment["targets"]}
    for description, tally in experiment["shared_edge_set"]["by_target"].items():
        by_family = {row["family"]: row for row in rows if row["target"] == description}
        assert schedules <= set(by_family)
        assert len(tally["operations"]) == 1
        assert len(tally["two_qubit_gates"]) == 1
        assert len(tally["max_edge_span"]) == 1
        assert len(tally["depths"]) > 1, "the depth is what a schedule changes"
        for family in schedules:
            row = by_family[family]
            assert row["verdict"] == "VALID" and row["tier"] == "stabilizer"
            assert row["checks_failed"] == []
            assert row["depth"] == 1 + row["cz_layers"]
            assert row["two_qubit_gates"] == by_family["graph_layers"]["two_qubit_gates"]
            assert row["max_edge_span"] == by_family["graph_layers"]["max_edge_span"]


def test_experiment_e_places_the_schedules_and_the_advantage_survives(record):
    """The gap is measured again after routing: it never shrinks, and never inverts."""
    routing = record["experiment_e"]["routing"]
    summary = routing["summary"]
    assert routing["reference"] == "graph_layers"
    assert routing["topologies"] == ["line", "ring", "grid"]
    assert routing["routers"] == ["greedy", "lookahead"]
    assert summary["rows"] == 3 * 3 * 2 * 5 == len(routing["rows"]) == 90
    assert summary["comparisons"] == 72 == len(routing["comparisons"])
    # Three of the four cluster sizes are routed, and the fourth is named rather than
    # silently missing: the 900-qubit lattice is timed for build and decide only.
    assert routing["max_qubits"] == 400
    # Routing is the cluster corpus: K9 is a complete graph, not a lattice, so it is
    # timed and decided but not placed.
    assert set(routing["routed_targets"]) == {"cluster_4x4", "cluster_10x10",
                                              "cluster_20x20"}
    assert "complete_k9" not in {row["target"] for row in routing["rows"]}
    assert {row["target"] for row in routing["rows"]} == set(routing["routed_targets"])
    assert [entry["target"] for entry in routing["not_routed"]] == ["cluster_30x30"]
    assert all(entry["qubits"] > routing["max_qubits"] and entry["reason"]
               for entry in routing["not_routed"])
    assert summary["every_placed_circuit_verifies"] is True
    assert summary["survives_everywhere"] is True
    assert summary["inverted_cases"] == []
    # The advantage is recomputable from the row's own two depth columns.
    by_key = {(row["target"], row["topology"], row["router"], row["family"]): row
              for row in routing["rows"]}
    for row in routing["rows"]:
        assert row["verdict"] == "VALID" and row["tier"] == "stabilizer"
        assert row["circuit_sha256"]
        assert row["mapped_depth"] >= row["logical_depth"] >= 1
        assert row["depth_before_restore"] <= row["mapped_depth"]
        assert row["restore_depth_cost"] == (row["mapped_depth"]
                                              - row["depth_before_restore"])
        reference = by_key[(row["target"], row["topology"], row["router"],
                            "graph_layers")]
        assert row["logical_advantage"] == (row["logical_depth"]
                                            - reference["logical_depth"])
        assert row["mapped_advantage"] == (row["mapped_depth"]
                                           - reference["mapped_depth"])
    for row in routing["comparisons"]:
        assert row["survives"] is True
        assert row["mapped_advantage"] > 0
        assert row["surviving_fraction"] >= 1.0
        assert row["surviving_fraction"] == pytest.approx(
            row["mapped_advantage"] / row["logical_advantage"])
    assert summary["min_surviving_fraction"] >= 1.0
    assert summary["max_surviving_fraction"] > summary["median_surviving_fraction"]
    # The lattice is the control: the target's own wiring, so no schedule pays a swap.
    assert summary["lattice_needs_no_routing"] is True
    for row in routing["rows"]:
        if row["topology"] == "grid":
            assert row["total_swaps"] == 0
            assert row["mapped_depth"] == row["logical_depth"]
            assert row["surviving_fraction"] == 1.0
    # On a line the minimum schedule is the cheapest to route, which is why the gap grows.
    line_400 = [row for row in routing["rows"]
                if row["target"] == "cluster_20x20" and row["topology"] == "line"
                and row["router"] == "greedy"]
    reference = next(row for row in line_400 if row["family"] == "graph_layers")
    assert reference["mapped_depth"] == min(row["mapped_depth"] for row in line_400)
    assert reference["total_swaps"] == min(row["total_swaps"] for row in line_400)


def test_experiment_e_times_synthesis_and_verification_per_schedule(record):
    """Every row carries both wall clocks, and the 400-qubit ones are real numbers.

    A timed circuit is only the recorded one if it rebuilds to the same hash; the script
    refuses to write the record otherwise, and this pin checks the columns are present,
    positive and on the row's own circuit. Each column is also pinned as the **median of
    the recorded repeats** rather than whatever one run happened to report, because a
    single clock at 400-900 qubits moves by tens of percent between runs.
    """
    rows = record["experiment_e"]["rows"]
    timing = record["experiment_e"]["timing"]
    assert timing["statistic"] == "median"
    assert timing["repeats_per_phase_per_schedule"] == 3
    assert rows
    for row in rows:
        assert row["synthesis_seconds"] > 0.0, row["family"]
        assert row["verification_seconds"] > 0.0, row["family"]
        assert row["circuit_sha256"]
        assert len(row["synthesis_samples"]) == len(row["verification_samples"]) == 3
        assert all(sample > 0.0 for sample in [*row["synthesis_samples"],
                                               *row["verification_samples"]])
        # The median is recomputable from the row's own samples, so the quoted clock is
        # not a second measurement that happens to sit beside them.
        assert row["synthesis_seconds"] == statistics.median(row["synthesis_samples"])
        assert row["verification_seconds"] == statistics.median(
            row["verification_samples"])
        # A median of three is one of the three samples, not an average of them.
        assert row["synthesis_seconds"] in row["synthesis_samples"]
        assert row["verification_seconds"] in row["verification_samples"]
    large = [row for row in rows if row["target"] == "cluster_20x20"]
    schedules = [row for row in large
                 if row["family"].startswith("graph_") and row["verdict"] == "VALID"]
    assert len(schedules) == 5
    assert max(row["qubits"] for row in rows) == 900
    # Three lattice sizes are timed, so the clocks can be read as a ladder rather than
    # as one point: 100, 400 and 900 qubits, each with five valid schedules.
    ladder = sorted({row["qubits"] for row in rows
                     if row["target"].startswith("cluster_")
                     and row["family"].startswith("graph_")
                     and row["verdict"] == "VALID"})
    assert ladder == [16, 100, 400, 900]
    expected_schedules = {100: 5, 400: 5, 900: 5}
    for qubits, count in expected_schedules.items():
        timings = [row for row in rows if row["qubits"] == qubits
                   and row["family"].startswith("graph_") and row["verdict"] == "VALID"]
        assert len(timings) == count
        assert all(row["verification_seconds"] > row["synthesis_seconds"]
                   for row in timings)
    # The largest lattice costs more to decide than the 400-qubit one: the referee runs
    # one stabilizer check per qubit, so the ladder is measured rather than assumed.
    at_400 = min(row["verification_seconds"] for row in rows
                 if row["qubits"] == 400 and row["family"] == "graph_layers")
    at_900 = min(row["verification_seconds"] for row in rows
                 if row["qubits"] == 900 and row["family"] == "graph_layers")
    assert at_900 > at_400
    # Building a schedule is cheap next to deciding it: the referee runs 400 stabilizer
    # checks while the builder emits one h per qubit and one cz per edge.
    assert all(row["verification_seconds"] > row["synthesis_seconds"]
               for row in schedules)
    # Every family emits the same 1,160 operations and is decided by the same 400
    # checks, so the times are close: no depth is bought with time, and no timing
    # advantage is claimed for the minimum -- only its 4 layers, the fewest of the five.
    minimum = next(row for row in schedules if row["family"] == "graph_layers")
    assert minimum["cz_layers"] == min(row["cz_layers"] for row in schedules) == 4
    assert all(row["operations"] == minimum["operations"] == 1160 for row in schedules)
    assert all(row["two_qubit_gates"] == minimum["two_qubit_gates"] == 760
               for row in schedules)


def test_experiment_e_the_minimum_schedule_attains_the_degree_floor(record):
    """The floor is proven (a degree-`d` qubit needs `d` layers) and it is reached."""
    experiment = record["experiment_e"]
    rows = experiment["rows"]
    peeled = [row for row in rows if row["family"] == "graph_layers"]
    assert len(peeled) == len(experiment["targets"])
    for row in peeled:
        assert row["cz_layers"] == row["depth"] - 1
        if row["target"] == "complete_k9":
            # The floor is 8 and the minimum is 9, which is the class-2 proof rather
            # than a failure to reach the floor.
            assert row["attains_the_floor"] is False
            assert row["cz_layers"] == row["cz_layers_floor"] + 1 == 9
        else:
            assert row["attains_the_floor"] is True
            assert row["cz_layers"] == row["cz_layers_floor"]
    others = [row for row in rows if row["family"] not in ("graph_layers",
                                                            "cluster_drop_edge",
                                                            "cluster_extra_edge",
                                                            "graph_swap_gate")]
    assert others
    assert all(row["attains_the_floor"] is False for row in others)
    assert all(row["cz_layers"] > row["cz_layers_floor"] for row in others)


def test_experiment_e_records_the_scheduler_and_what_it_saves(record):
    """The replacement is a minimum colouring; the record says how much it saved."""
    experiment = record["experiment_e"]
    scheduler = experiment["scheduler"]
    assert scheduler["theorem"].startswith("Vizing")
    assert len(scheduler["targets"]) == len(experiment["targets"])
    for entry in scheduler["targets"]:
        assert entry["optimal"] is True
        assert entry["cz_layers"] >= entry["max_degree"]
        assert entry["edge_chromatic_number"] == entry["cz_layers"]
        assert entry["layers_saved"] == (entry["greedy_cz_layers"] - entry["cz_layers"])
        assert entry["layers_saved"] >= 0
    assert scheduler["total_layers_saved"] == sum(
        entry["layers_saved"] for entry in scheduler["targets"])
    # Six of the seven Experiment E targets are the case where the greedy peel was
    # already optimal; K9 is not, and its 6 saved layers are the class-2 proof showing
    # up in the measured table rather than only in the colouring block.
    assert scheduler["total_layers_saved"] == 6
    assert set(scheduler["targets_where_greedy_was_already_optimal"]) == {
        target["description"] for target in experiment["targets"]
        if target["description"] != "complete_k9"}


def test_experiment_e_proves_the_class_two_cases_it_adds(record):
    class_two = record["experiment_e"]["scheduler"]["class_two"]
    assert {row["description"] for row in class_two} == {
        "odd_cycle_c5", "petersen", "complete_k5", "complete_k9", "complete_k11",
        "complete_k13"}
    by_name = {row["description"]: row for row in class_two}
    for row in class_two:
        assert row["optimal"] is True and row["delta_refuted"] is True
        assert row["edge_chromatic_number"] == row["max_degree"] + 1
        assert row["depth"] == 1 + row["cz_layers"]
        assert row["verdict"] == "VALID"
        assert row["family"] == "graph_layers"
        # The proof kind is recorded, and it has to be one of the two real ones.
        assert row["status"] in {"delta_refuted", "delta_refuted_by_layer_capacity"}
        if row["status"] == "delta_refuted_by_layer_capacity":
            assert row["search_nodes"] == 0
            assert row["proof_kind"] == "class 2, refuted by layer capacity"
        else:
            assert row["search_nodes"] > 0
            assert row["proof_kind"] == "class 2, refuted by exhausted search"
    # The odd complete graphs are the family the search cannot finish -- K9 needs more
    # than five million nodes -- so their refutation must come from the counting bound,
    # and it must be a saving: 15 peel layers against a proven 9 and 11.
    for name in ("complete_k9", "complete_k11"):
        row = by_name[name]
        assert row["status"] == "delta_refuted_by_layer_capacity", name
        assert row["search_nodes"] == 0, name
        assert row["greedy_cz_layers"] == 15, name
        assert row["cz_layers"] == row["max_degree"] + 1, name
        assert row["layers_saved"] == 15 - row["cz_layers"], name
    assert by_name["complete_k9"]["layers_saved"] == 6
    assert by_name["complete_k11"]["layers_saved"] == 4
    # K13 is the family continued one step: 78 edges cannot fit into 12 layers of 6, so
    # the capacity bound refutes Delta at zero nodes and the peel's 15 falls to 13.
    k13 = by_name["complete_k13"]
    assert k13["cz_layers"] == 13 and k13["greedy_cz_layers"] == 15
    assert k13["layers_saved"] == 2
    assert k13["status"] == "delta_refuted_by_layer_capacity"
    assert k13["search_nodes"] == 0
    # The Petersen graph is the tight case: 15 edges fit exactly into three layers of
    # five, so the counting bound says nothing and the refutation is the search's.
    assert by_name["petersen"]["status"] == "delta_refuted"
    # At least one class-2 schedule is actually cheaper than the peel, otherwise the
    # replacement would only be a proof and not a saving.
    assert any(row["layers_saved"] > 0 for row in class_two)
    assert record["experiment_e"]["scheduler"]["class_two_layers_saved"] == sum(
        row["layers_saved"] for row in class_two)


def test_experiment_e_wrong_graph_constructions_are_refused_edge_by_edge(record):
    refusals = {row["family"]: row for row in record["experiment_e"]["refusals"]}
    assert set(refusals) == {"cluster_drop_edge", "cluster_extra_edge", "graph_swap_gate"}
    for family, row in refusals.items():
        assert row["verdict"] == "INVALID", family
        assert row["failed_count"] == 2, family
        assert row["checks_failed"] == [
            "the prepared state is the +1 eigenstate of every specified stabilizer"]
        generators = row["failed_generators"]
        assert len(generators) == 2
        assert all(entry["pauli"].startswith("+") for entry in generators)
        assert all(entry["measured"] == "absent from the group" for entry in generators)


def test_experiment_e_targets_are_not_the_same_state(record):
    """Both have 12 qubits, so the cross-comparison is at a shared width."""
    discrimination = record["experiment_e"]["discrimination"]
    assert discrimination["ghz_chain_against_the_cluster"] == "INVALID"
    assert discrimination["cluster_against_the_ghz"] == "INVALID"
    assert "shared width" in discrimination["note"]


def test_experiment_e_both_referees_agree_on_the_graph_states(record):
    agreement = record["experiment_e"]["referee_agreement"]
    assert agreement["all_agree"] is True
    rows = agreement["rows"]
    accepted = [row for row in rows if row["stabilizer_verdict"] == "VALID"]
    assert agreement["accepted"] == len(accepted) == len(rows) - 1
    assert agreement["refused"] == 1
    assert all(row["stabilizer_verdict"] == row["dense_verdict"] for row in rows)
    assert all(row["dense_distance"] < 1e-12 for row in accepted)
    refused = next(row for row in rows if row["stabilizer_verdict"] == "INVALID")
    assert refused["case"] == "cluster_2x3_dropped_edge"
    assert refused["dense_distance"] > 1e-9
    # The asymmetric cases are why this corpus exists: reversing the wires is an
    # automorphism of a cluster but not of a seeded irregular graph.
    assert any(row["case"].startswith("irregular") for row in accepted)


def test_a_third_referee_decides_the_circuits_as_unitaries(record):
    """QCEC on the same circuits as operators, which is a stricter question."""
    third = record["third_referee"]
    assert third["tool"] == "mqt.qcec"
    assert "unitaries" in third["question"]
    assert "h on every wire" in third["reference"]
    rows = third["rows"]
    assert third["decided"] == sum(1 for row in rows if row["decided"])
    assert third["decided"] + third["abstained"] == len(rows)
    assert third["agreements"] == sum(1 for row in rows if row["agrees"])
    for row in rows:
        assert row["circuit_sha256"] and row["reference_sha256"]
        assert row["our_verdict"] in {"VALID", "INVALID"}
        if row["decided"]:
            equivalent = row["third_party"]["verdict"] in (
                "equivalent", "equivalent_up_to_global_phase")
            # The two questions must coincide on this corpus: a state-level VALID is a
            # unitary-level equivalence, and a wrong construction is neither.
            assert row["agrees"] is ((row["our_verdict"] == "VALID") == equivalent)
        else:
            assert row["agrees"] is None
            assert row["abstention_reason"]
    # The five schedules of one edge set are one unitary; the wrong constructions are not.
    schedules = [row for row in rows if row["target"] == "cluster_4x4"]
    assert len(schedules) == 5
    assert {row["third_party"]["verdict"] for row in schedules} <= {
        "equivalent", "equivalent_up_to_global_phase"}
    wrong = [row for row in rows if row["family"] in (
        "cluster_drop_edge", "cluster_extra_edge", "graph_swap_gate")]
    assert len(wrong) == 3
    assert all(row["third_party"]["verdict"] == "not_equivalent" for row in wrong)
    assert all(row["our_verdict"] == "INVALID" for row in wrong)
    # The widest pair in the corpus is 900 qubits, and it was decided.
    widest = max(rows, key=lambda row: row["qubits"])
    assert widest["qubits"] == 900 and widest["decided"] is True
    assert third["abstentions"] == [
        {"target": row["target"], "family": row["family"], "qubits": row["qubits"],
         "verdict": row["third_party"]["verdict"],
         "reason": row["third_party"].get("reason")}
        for row in rows if not row["decided"]]


def test_sabre_is_swept_over_five_seeds_and_the_interval_is_reported(record):
    """One SABRE run is a draw; the spread between seeded runs is the measurement."""
    sweep = record["experiment_d"]["sabre_seeds"]
    if sweep["unavailable"]:
        assert sweep["unavailable"] == len(sweep["rows"])
        assert all(row["reason"] for row in sweep["rows"])
        return
    assert sweep["seeds"] == [0, 1, 2, 3, 4]
    assert sweep["summary"]["rows"] == 2 * 3 * 5 == len(sweep["rows"])
    assert sweep["summary"]["cases"] == 6
    for row in sweep["rows"]:
        assert row["status"] == "compared"
        assert row["verdict"] == "VALID"
        assert len(set(row["final_layout"])) == len(row["final_layout"]) == 100
        assert row["two_qubit_gates"] >= row["routing_swaps"]
        assert row["circuit_sha256"]
        # Qiskit keeps its layout, so the comparable column is ours before the restore
        # ladder; both routers' numbers are stored beside the foreign one.
        assert row["ours"]["lookahead_depth_before_restore"] > 0
        assert row["ours"]["greedy_depth_before_restore"] > 0
    intervals = sweep["intervals"]
    assert set(intervals) == {f"{family}/{topology}"
                              for family in ("chain", "balanced_tree")
                              for topology in ("line", "ring", "grid")}
    for key, entry in intervals.items():
        case = [row for row in sweep["rows"]
                if f"{row['family']}/{row['topology']}" == key]
        depths = [row["routed_depth"] for row in case]
        assert entry["min_depth"] == min(depths) and entry["max_depth"] == max(depths)
        assert entry["spread"] == entry["max_depth"] - entry["min_depth"]
        assert entry["distinct_circuits"] == len({row["circuit_sha256"] for row in case})
        assert entry["our_lookahead_depth_before_restore"] == \
            case[0]["ours"]["lookahead_depth_before_restore"]
    assert sweep["summary"]["cases_with_a_spread"] == sum(
        1 for entry in intervals.values() if entry["spread"] > 0)
    assert sweep["summary"]["largest_spread"] == max(
        entry["spread"] for entry in intervals.values())
    # The stochasticity is real: the chain on a line or ring needs no routing and is
    # seed-invariant, and every other case moves.
    assert sweep["summary"]["cases_with_a_spread"] == 4
    assert intervals["chain/line"]["spread"] == 0
    assert intervals["chain/ring"]["spread"] == 0
    assert sweep["summary"]["largest_spread"] == 24
    assert "depth_before_restore" in sweep["note"]


def test_the_regression_case_where_lookahead_is_worse_is_in_the_record(record):
    """A heuristic shown only winning is a claim, so the counterexample is pinned."""
    case = record["experiment_d"]["regression_case"]
    assert case["qubits"] == 9
    assert case["family"] == "balanced_tree" and case["topology"] == "line"
    assert case["greedy"]["mapped_depth"] == 17
    assert case["lookahead"]["mapped_depth"] == 19
    assert case["lookahead_is_deeper_by"] == 2
    assert case["lookahead"]["swap_gates"] == case["greedy"]["swap_gates"] + 1
    assert case["greedy"]["circuit_sha256"] != case["lookahead"]["circuit_sha256"]
    assert "not monotone" in case["note"]


def test_the_placement_frontier_is_nondominated_and_reports_its_own_size(record):
    """Depth and gate count are both costs; the frontier is not a score."""
    pareto = record["experiment_d"]["pareto"]
    assert pareto["axes"] == ["mapped_depth", "cx_equivalent_two_qubit_gates"]
    points = pareto["points"]
    assert points and pareto["distinct_points"] >= len(points)
    assert pareto["dominated"] == pareto["distinct_points"] - len(points)
    assert pareto["is_a_trade_off"] is (len(points) > 1)
    for point in points:
        for other in points:
            if other is point:
                continue
            assert not (other["mapped_depth"] <= point["mapped_depth"]
                        and other["cx_equivalent_two_qubit_gates"]
                        <= point["cx_equivalent_two_qubit_gates"]
                        and (other["mapped_depth"] < point["mapped_depth"]
                             or other["cx_equivalent_two_qubit_gates"]
                             < point["cx_equivalent_two_qubit_gates"]))
    # On this corpus the two costs are ordered identically: the frontier is one point,
    # which the record states as the result instead of calling it a trade-off.
    assert len(points) == 1 and pareto["is_a_trade_off"] is False
    assert points[0]["topology"] == "all_to_all"
    assert points[0]["family"] == "balanced_tree"
    assert "ordered identically" in pareto["note"]


def test_the_lookahead_window_is_swept_and_window_8_is_the_recorded_router(record):
    """The window was a choice, so all five are measured and one must be the record's."""
    sweep = record["experiment_d"]["window_sweep"]
    assert sweep["windows"] == [2, 4, 8, 16, 32]
    rows = sweep["rows"]
    assert sweep["summary"]["rows"] == 2 * 4 * 5 == len(rows)
    assert sweep["summary"]["cases"] == 8
    assert all(row["verdict"] == "VALID" for row in rows)
    assert all(row["circuit_sha256"] for row in rows)
    recorded = {(row["family"], row["topology"]): row
                for row in record["experiment_d"]["rows"]
                if row["router"] == "lookahead"}
    for row in rows:
        if row["window"] != 8:
            continue
        reference = recorded[(row["family"], row["topology"])]
        assert row["circuit_sha256"] == reference["circuit_sha256"]
        assert row["mapped_depth"] == reference["mapped_depth"]
        assert row["swap_gates"] == reference["swap_gates"]
    # All-to-all needs no routing, so the window cannot change it: the control row.
    for family in ("chain", "balanced_tree"):
        depths = {row["mapped_depth"] for row in rows
                  if row["family"] == family and row["topology"] == "all_to_all"}
        assert len(depths) == 1
    assert sweep["all_to_all_is_window_independent"] is True
    # Depth is not monotone in the window, and the record reports that rather than
    # claiming a window is better in general.
    assert sweep["depth_is_monotone_in_the_window"] is False
    assert sweep["summary"]["best_depth_total"] <= sweep["summary"]["at_window_8_depth_total"]
    assert set(sweep["best_window_by_case"]) == {
        f"{family}/{topology}" for family in ("chain", "balanced_tree")
        for topology in ("line", "ring", "grid", "all_to_all")}


def test_the_scheduler_carries_both_counting_bounds_and_the_overshoot_rate(record):
    scheduler = record["experiment_e"]["scheduler"]
    bounds = scheduler["bounds"]
    targets = {entry["description"]: entry for entry in bounds["targets"]}
    assert set(targets) == {target["description"]
                            for target in record["experiment_e"]["targets"]}
    for entry in targets.values():
        assert entry["matching_number"] > 0
        assert entry["layer_capacity"] == entry["max_degree"] * (entry["qubits"] // 2)
        assert entry["matching_lower_bound"] == -(-entry["edges"] // entry["matching_number"])
        assert entry["lower_bound"] == max(entry["max_degree"],
                                           entry["matching_lower_bound"])
        assert entry["layer_capacity_refutes_delta"] is (
            entry["edges"] > entry["layer_capacity"])
        assert entry["matching_bound_refutes_delta"] is (
            entry["matching_lower_bound"] > entry["max_degree"])
    # Six targets are class 1 and neither bound fires there; K9 is the seventh, and the
    # capacity bound is exactly what proves its extra layer. Both are reported.
    assert bounds["refuted_by_layer_capacity"] == 1
    assert bounds["refuted_by_matching"] == 0
    assert targets["complete_k9"]["layer_capacity_refutes_delta"] is True
    assert all(not entry["layer_capacity_refutes_delta"]
               for name, entry in targets.items() if name != "complete_k9")
    assert bounds["refuted_by_layer_capacity"] == sum(
        1 for entry in targets.values() if entry["layer_capacity_refutes_delta"])
    overshoot = scheduler["overshoot"]
    assert overshoot["graphs"] > 0
    assert overshoot["overshoot_rate"] == pytest.approx(
        overshoot["overshoot_graphs"] / overshoot["graphs"])
    assert overshoot["worst_overshoot_layers"] <= 1
    for row in overshoot["rows"]:
        assert row["graphs_with_a_proved_delta_colouring"] > 0
        assert row["overshoot_rate"] == pytest.approx(
            row["overshoot_graphs"] / row["graphs_with_a_proved_delta_colouring"])
        # Vizing bounds the construction at Delta + 1, so the overshoot is at most one.
        assert row["worst_overshoot_layers"] <= 1


def test_the_dense_tier_ceiling_is_measured_width_by_width(record):
    """The bound of ten is priced, not asserted: seconds and bytes at every width."""
    ceiling = record["dense_referee_ceiling"]
    assert ceiling["widths"] == [6, 8, 10, 11, 12, 13, 14, 15]
    limit = record["experiment_c"]["dense_spec_limit"]
    rows = {row["width"]: row for row in ceiling["rows"]}
    assert set(rows) == set(ceiling["widths"])
    for width, row in rows.items():
        if width <= limit:
            assert row["status"] == "measured"
            assert row["policy_tier"] == "dense"
            assert row["policy_verdict"] == "VALID"
            assert row["cost"]["within_tolerance"] is True
            assert row["cost"]["distance"] < 1e-12
        else:
            # The referee's answer is unchanged by the probe: no tier, no verdict.
            assert row["policy_tier"] == "none"
            assert row["policy_verdict"] == "UNRESOLVED"
            assert "above the dense specification bound" in row["policy_reason"]
    for width, row in rows.items():
        if row["cost"] is None:
            continue
        assert row["cost"]["qubits"] == width
        assert row["cost"]["amplitudes"] == 2 ** width
        # The model is quoted, and the measurement has to be at least that large.
        assert row["cost"]["pauli_operator_bytes"] == 16 * 4 ** width
        assert row["cost"]["state_vector_bytes"] == 16 * 2 ** width
        assert row["cost"]["peak_traced_bytes"] >= row["cost"]["pauli_operator_bytes"]
        assert row["cost"]["seconds"] > 0.0
        assert row["peak_rss_bytes"] >= row["cost"]["peak_traced_bytes"]
    # One more qubit is four times the operator and (measured) about four times the
    # time, which is why the wall arrives one width after the tier is comfortable.
    assert ceiling["growth"]
    for pair in ceiling["growth"]:
        assert 3.0 < pair["peak_ratio"] < 6.0, pair
        assert pair["time_ratio"] > 2.0, pair


def test_the_dense_tier_wall_is_memory_and_the_record_names_it(record):
    ceiling = record["dense_referee_ceiling"]
    wall = ceiling["wall"]
    assert wall is not None
    assert wall["status"] == "exceeded_memory"
    assert wall["width"] == max(ceiling["widths"])
    assert wall["pauli_operator_bytes"] == 16 * 4 ** wall["width"]
    assert wall["state_vector_bytes"] == 16 * 2 ** wall["width"]
    assert wall["pauli_operator_bytes"] > ceiling["memory_ceiling_bytes"]
    assert wall["operator_over_budget"] > 1.0
    assert "MemoryError" in wall["reason"]
    measured = [row for row in ceiling["rows"] if row["status"] == "measured"]
    assert {row["width"] for row in measured} == set(ceiling["widths"][:-1])
    assert max(row["width"] for row in measured) == ceiling["largest_measured_width"]
    assert ceiling["largest_measured_width"] < wall["width"]
    # Everything below the wall fitted the same declared budget, per width.
    assert ceiling["memory_ceiling_bytes"] == 8 * 2**30
    assert all(row["cost"]["peak_traced_bytes"] < ceiling["memory_ceiling_bytes"]
               for row in measured)


def test_the_dense_tier_price_is_quoted_with_its_host_and_budget(record):
    ceiling = record["dense_referee_ceiling"]
    assert ceiling["host"]["cpus"] and ceiling["host"]["memory_total_bytes"]
    assert ceiling["host"]["python"] and ceiling["host"]["platform"]
    assert ceiling["timeout_seconds"] > 0
    assert "quadratic in the Hilbert dimension" in ceiling["note"]
    assert "declared RLIMIT_AS ceiling" in ceiling["note"]


def test_the_record_pins_the_code_that_produced_it(record):
    reproducibility = record["reproducibility"]
    assert reproducibility["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    for key, relative in (("synthesis_sha256", "apedp_dyn/vqc/synthesis.py"),
                          ("placement_sha256", "apedp_dyn/vqc/placement.py"),
                          ("discovery_sha256", "apedp_dyn/vqc/discovery.py"),
                          ("equivalence_sha256", "apedp_dyn/vqc/equivalence.py"),
                          ("stabilizer_sha256", "apedp_dyn/vqc/stabilizer.py"),
                          ("colouring_sha256", "apedp_dyn/vqc/colouring.py")):
        assert reproducibility[key] == hashlib.sha256(
            (ROOT / relative).read_bytes()).hexdigest(), key


def test_the_limitations_say_it_is_a_grammar_not_a_synthesiser(record):
    text = " ".join(record["limitations"]).lower()
    for topic in ("grammar", "general synthesiser", "trade off", "experiment c",
                  "wall clock", "lookahead router", "initial-placement search",
                  "reorders gates", "swap ladder", "device calibration",
                  "referee agreement", "experiment e", "edge chromatic", "surface-code",
                  "adversarial", "dense-tier ceiling", "address-space budget",
                  "tracemalloc"):
        assert topic in text, topic


def test_the_depth_histogram_places_the_circuit_this_record_timed(record):
    """The other record's histogram quotes this record's circuit, not a lookalike.

    `vqc_external_comparison.json` places each Experiment E depth against the Qiskit
    corpus. That block rebuilds the minimum schedule itself, so the two records are
    independent measurements of one circuit: they must agree on its SHA-256, on its
    depth and on the qubit count, or one of them is quoting a different construction.
    """
    external = json.loads((ROOT / "benchmarks" / "quantum_compilation"
                           / "vqc_external_comparison.json").read_text(encoding="utf-8"))
    placed = {row["target"]: row for row in external["qiskit_depth_context"]["targets"]}
    timed = {row["target"]: row for row in record["experiment_e"]["rows"]
             if row["family"] == "graph_layers"}
    # The histogram record was written for the six graph targets it covers; K9 was added
    # to Experiment E afterwards and is not in it. The gap is named, not compared away.
    assert set(timed) - set(placed) == {"complete_k9"}
    assert not set(placed) - set(timed)
    for description, row in placed.items():
        timed_row = timed[description]
        assert row["circuit_sha256"] == timed_row["circuit_sha256"]
        assert row["depth"] == timed_row["depth"] == timed_row["cz_layers"] + 1
        assert row["qubits"] == timed_row["qubits"]
        assert row["verdict"] == timed_row["verdict"] == "VALID"
