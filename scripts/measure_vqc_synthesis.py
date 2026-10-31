#!/usr/bin/env python3
"""Synthesise circuits from a specification, with no reference circuit supplied.

The rewrites record (`measure_vqc_clifford_discovery.py`) measures *optimisation*:
a circuit is given, and local rewrites remove work from it. This record measures the
next capability, the one the review put first:

    **Experiment A** -- give the platform only the mathematics of the target
    (``|GHZ_n>`` as a stabilizer specification) and ask it to construct a circuit.
    Five construction families are built, each is decided against the specification
    alone, the ranking and the Pareto frontier over depth/operations/connectivity are
    reported, and four deliberately wrong constructions are refused with the failing
    stabilizer named. A second target (a ring graph state) is run so the result cannot
    be read as "GHZ works".

    **Experiment B** -- the same synthesis at 10, 50, 100, 250, 500 and 1,000 qubits,
    with operations, depth, connectivity, wall clock and the verification outcome of
    every row.

    **Experiment C** -- not built. What is measured here is its boundary: the dense
    specification referee *does* decide a non-Clifford construction (a GHZ prepared
    with ``ry(pi/2)``), at up to ``DENSE_SPEC_LIMIT`` qubits and not above, while no
    non-Clifford *search* exists. The record says so rather than implying more.

    **Experiment D** -- the same constructions placed on a real connectivity graph.
    Logical depth is what the abstract circuit costs; mapped depth is what it costs
    after the non-local edges have been paid for with SWAP routing on a line, a ring,
    a rectangular lattice and an all-to-all device. Every routed circuit is decided
    against the same specification, so the routing is verified rather than assumed.
    Both routers run on every row: the greedy router this record started with, and a
    lookahead router that schedules each swap into the earliest layer its nodes are
    free and breaks ties by what the swap does to the next eight two-qubit gates. The
    comparison is row by row, so "routing dominates the topology difference" is a
    measurement against a baseline rather than against an assumption.

    **Experiment E** -- graph-state specifications: a two-dimensional cluster on a
    square lattice at 16, 100, 400 and 900 qubits (10x10, 20x20 and 30x30 are the timing
    ladder), and an irregular graph drawn from a seed. A graph state names every edge,
    so the wiring is
    the target's and the only free variable is the *schedule* of commuting ``cz`` gates;
    five schedules are built for each target, the minimum-edge-colouring one is proved
    optimal against Vizing's bound, the greedy peel it replaced is measured against it,
    and three wrong graph constructions are refused with the failing stabilizers named.
    Each row carries the wall clock of building and of deciding that schedule, because
    from 100 qubits up the schedule's cost stops being negligible and grows with the
    lattice. The two referees are run
    on the targets at the widths both can reach, including an irregular graph whose
    asymmetry is what caught the dense tier's basis-order bug.

    **Referee agreement** -- the stabilizer tier and the dense tier, run against the
    *same* circuit at every width both can reach, so that "two referees" is a
    measurement and not a claim about the code layout. The widest rows are 400 and 900
    qubits, which is above where the dense tier exists at all: each is decided by the
    stabilizer tier and by Stim, the three wrong constructions are decided again and
    refused by both, and the dense tier's own refusal at both widths is recorded beside
    them rather than left as an absence.

    **The dense tier's ceiling** -- what one non-Clifford decision costs, width by
    width, in seconds and peak memory, until it stops being practical. The tier
    materialises each stabilizer as a dense ``2**n x 2**n`` Pauli operator, so its
    dominant cost is ``16 * 4**n`` bytes; each width runs in a fresh process under a
    declared address-space budget, and the record names the width at which that budget
    is exceeded instead of extrapolating from the bound of ten.

There is no reference circuit anywhere in Experiment A: a ``TargetSpecification`` is
a qubit count plus signed Pauli stabilizers, and the referee compares the prepared
state against those generators. `experiment_a.reference_circuit` is recorded as
``null`` on purpose, and the script asserts that the specification object has no
circuit attached.

Example::

    python scripts/measure_vqc_synthesis.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.vqc.circuit import Circuit, Gate  # noqa: E402
from apedp_dyn.vqc.colouring import (  # noqa: E402
    greedy_peel,
    minimum_edge_colouring,
    vizing_colouring,
)
from apedp_dyn.vqc.external import (  # noqa: E402
    ToolUnavailable,
    qcec_equivalence,
    qiskit_routing,
    relabelled_target,
)
from apedp_dyn.vqc.foreign_referee import stim_verdict  # noqa: E402
from apedp_dyn.vqc.placement import (  # noqa: E402
    CONNECTIVITIES,
    ROUTERS,
    SWAP_AS_CNOT,
    place,
)
from apedp_dyn.vqc.stabilizer import is_clifford_circuit  # noqa: E402
from apedp_dyn.vqc.synthesis import (  # noqa: E402
    DENSE_SPEC_LIMIT,
    REFUSAL_FAMILIES,
    SynthesisError,
    adjudicate_specification,
    build,
    cluster_state_target,
    dense_oracle_check,
    dense_tier_cost,
    entangling_depth_lower_bound,
    ghz_target,
    graph_of_state,
    irregular_graph_target,
    referee_agreement,
    ring_graph_target,
    seeded_graph_target,
    specification_summary,
    synthesize,
)

RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_synthesis.json"
SIZES = (10, 50, 100, 250, 500, 1000)
FAMILIES_BY_SIZE = ("chain", "balanced_tree", "star", "random_tree")
#: The constructions Experiment D places: the pair whose depths and connectivity
#: requirements trade off against each other.
PLACED_FAMILIES = ("chain", "balanced_tree")
#: Experiment E's specifications: a two-dimensional cluster at four sizes and an
#: irregular graph reproducible from its seed alone. The three largest lattices (100,
#: 400 and 900 qubits) are the timing ladder: the same five schedules at the same
#: degree-4 wiring, so build and decide times can be read as a function of width
#: rather than as one point. The 4x4 lattice stays as the small case where a reader
#: can check the schedules by eye.
CLUSTER_SHAPES = ((4, 4), (10, 10), (20, 20), (30, 30))
#: The lattices the referee-agreement corpus decides twice: 400 and 900 qubits, above
#: the dense tier's bound, so the second decision comes from an implementation this
#: package did not write rather than from a third tier.
WIDE_CLUSTER_SHAPES = ((20, 20), (30, 30))
IRREGULAR_CASES = ((25, 7, 0.15), (64, 23, 0.10))
#: Graphs whose edge chromatic number exceeds their maximum degree: an odd cycle
#: (Delta = 2, chi' = 3), the Petersen graph (Delta = 3, chi' = 4) and the complete graphs
#: on 5, 9 and 11 vertices (Delta = 4, 8, 10, chi' = Delta + 1). They are not Experiment E
#: targets -- they exist to show the scheduler proves the class-2 case rather than
#: assuming it. The odd complete graphs are the family that grows: their `Delta` search
#: cannot finish (K9 needs more than five million nodes) and they are refuted by the
#: layer-capacity bound instead, so the odd-regular family's refutation is *measured*
#: rather than left undecided, with the peel's 15 layers against a proven 9 and 11.
CLASS_TWO_GRAPHS = {
    "odd_cycle_c5": (5, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))),
    "petersen": (10, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                      (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                      (5, 7), (7, 9), (9, 6), (6, 8), (8, 5))),
    "complete_k5": (5, ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))),
    "complete_k9": (9, tuple((a, b) for a in range(9) for b in range(a + 1, 9))),
    "complete_k11": (11, tuple((a, b) for a in range(11) for b in range(a + 1, 11))),
    # K13 continues the odd-complete family one step past K11: 78 edges cannot fit into
    # 12 layers of 6, so the capacity bound refutes Delta at zero nodes again and the
    # saving over the peel is measured at the next size rather than extrapolated.
    "complete_k13": (13, tuple((a, b) for a in range(13) for b in range(a + 1, 13))),
}
#: Wall-clock repeats per schedule phase. One 0.2 s decision moves by tens of percent
#: between runs on a loaded host, which is the size of the differences being read off
#: these columns, so each phase is timed `TIMING_REPEATS` times on the same circuit and
#: the row quotes the **median**. Every sample is recorded beside the median, so the
#: statistic is recomputable from the record and a reader can see the spread it hid.
TIMING_REPEATS = 3
#: The graph-state construction families Experiment E measures: five schedules of the
#: same edge set. The refusal families are measured beside them, not among them.
GRAPH_FAMILIES = ("graph_raster", "graph_column_major", "graph_tree_first",
                  "graph_layers", "graph_random_order")
#: The topologies the cluster schedules are placed on, and the family whose depth the
#: others are measured against. `grid` is the cluster's own lattice, so it is the
#: control: a schedule placed on its own wiring needs no swaps at all.
ROUTING_TOPOLOGIES = ("line", "ring", "grid")
ROUTING_REFERENCE = "graph_layers"
#: Routing is measured up to this width. The 900-qubit lattice is timed for build and
#: decide and is deliberately *not* routed: the greedy router needs about 56 s for one
#: placement there (27,159 swaps under lookahead, 378,450 under greedy), and the
#: routing task runs thirty placements per lattice size, so routing it would multiply
#: the suite's runtime for a comparison already measured at 100 and 400 qubits. The
#: record names the target it did not route rather than shrinking the row count.
ROUTING_MAX_QUBITS = 400
#: The widths the dense tier is priced at. The first four are inside its declared
#: bound, where the referee itself runs; the rest are the policy bound lifted in a
#: separate process, so the wall is measured rather than extrapolated from one width.
DENSE_PROBE_WIDTHS = (6, 8, 10, 11, 12, 13, 14, 15)
#: A declared budget, not the host's: every width runs in a fresh process whose address
#: space is capped here, so an over-budget width fails cleanly instead of invoking the
#: kernel's OOM killer, and the record can say *which* budget it stopped at.
DENSE_PROBE_MEMORY_CEILING = 8 * 2**30
DENSE_PROBE_TIMEOUT_SECONDS = 300.0


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


def mutate(construction, operation: str) -> Circuit:
    """A construction that differs from the right one in exactly one way."""
    ops = list(construction.circuit.ops)
    if operation == "drop_the_hadamard":
        ops = [op for op in ops if getattr(op, "name", None) != "h"]
    elif operation == "reverse_the_middle_cx":
        index = len(ops) // 2
        ops[index] = Gate(ops[index].name, tuple(reversed(ops[index].targets)))
    elif operation == "extra_hadamard_on_the_last_qubit":
        ops.append(Gate("h", (construction.circuit.num_qubits - 1,)))
    elif operation == "replace_a_cx_with_a_cz":
        index = next(i for i, op in enumerate(ops) if getattr(op, "name", None) == "cx")
        ops[index] = Gate("cz", ops[index].targets)
    else:  # pragma: no cover - the caller names the mutations
        raise AssertionError(operation)
    return Circuit(ops=tuple(ops), num_qubits=construction.circuit.num_qubits,
                   num_bits=0)


def non_clifford_ghz(num_qubits: int) -> Circuit:
    """A GHZ prepared with `ry(pi/2)` instead of `h`: correct, and not Clifford."""
    ops: list = [Gate("ry", (0,), params=(math.pi / 2,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0)


def experiment_a(qubits: int) -> dict[str, Any]:
    """Construct from the specification, rank the wiring, refuse the near misses."""
    target = ghz_target(qubits)
    outcome, seconds = timed(synthesize, target)
    rows = []
    for row in outcome.rows:
        rows.append({
            "family": row["family"],
            "verdict": row["verdict"],
            "tier": row["tier"],
            "operations": row["operations"],
            "depth": row["depth"],
            "two_qubit_gates": row["two_qubit_gates"],
            "max_edge_span": row["max_edge_span"],
            "nearest_neighbour_only": row["nearest_neighbour_only"],
            "circuit_sha256": row["circuit_sha256"],
            "checks_failed": row["checks_failed"],
        })
    chain = build("chain", target)
    refusals = []
    for operation in ("drop_the_hadamard", "reverse_the_middle_cx",
                      "extra_hadamard_on_the_last_qubit", "replace_a_cx_with_a_cz"):
        report = adjudicate_specification(mutate(chain, operation), target)
        failed = next((check for check in report.checks if check.passed is False), None)
        refusals.append({
            "mutation": operation,
            "verdict": report.verdict,
            "checks_failed": list(report.failed()),
            "failed_generators": (None if failed is None
                                  else failed.measured.get("failed", [])[:2]),
            "failed_count": None if failed is None else failed.measured.get("failed_count"),
        })
    return {
        "specification": specification_summary(target),
        "reference_circuit": None,
        "reference_circuit_note": (
            "none was supplied: the only inputs are the qubit count and the signed "
            "Pauli stabilizers, and the referee compares the prepared state against "
            "those generators"),
        "synthesize_seconds": seconds,
        "constructions": rows,
        "ranking": list(outcome.ranked),
        "pareto_frontier": list(outcome.pareto),
        "admission": None if outcome.verdict is None else {
            "candidate_id": outcome.verdict.candidate.candidate_id,
            "semantic_verdict": outcome.verdict.verdict,
            "resource_verdict": outcome.verdict.claim_verdict,
            "evidence_sufficiency": outcome.verdict.evidence,
            "promotion_decision": ("PROMOTED" if outcome.verdict.promotable else "REFUSED"),
            "promoted": outcome.promoted,
            "tier": outcome.verdict.tier,
        },
        "promoted": outcome.promoted,
        "refusals": refusals,
    }


def second_target(qubits: int) -> dict[str, Any]:
    """The same machinery on a different state, so the result is not "GHZ works"."""
    target = ring_graph_target(qubits)
    outcome, seconds = timed(synthesize, target)
    return {
        "specification": specification_summary(target),
        "synthesize_seconds": seconds,
        "constructions": [{"family": row["family"], "verdict": row["verdict"],
                           "operations": row["operations"], "depth": row["depth"],
                           "max_edge_span": row["max_edge_span"]}
                          for row in outcome.rows],
        "promoted": outcome.promoted,
        "near_miss": next((row for row in outcome.rows
                           if row["family"] in REFUSAL_FAMILIES), None),
    }


def claim_refusal(qubits: int) -> dict[str, Any]:
    """A valid construction offered under a claim it cannot meet is still refused."""
    target = ghz_target(qubits)
    outcome = synthesize(target, claims={"depth": 1})
    assert outcome.verdict is not None
    return {
        "claims": {"depth": 1},
        "semantic_verdict": outcome.verdict.verdict,
        "resource_verdict": outcome.verdict.claim_verdict,
        "evidence_sufficiency": outcome.verdict.evidence,
        "promoted": outcome.promoted,
        "refusal": outcome.refusal,
    }


def experiment_b() -> list[dict[str, Any]]:
    """Synthesis and verification from 10 to 1,000 qubits, row by row."""
    rows = []
    for num_qubits in SIZES:
        target = ghz_target(num_qubits)
        for family in FAMILIES_BY_SIZE:
            construction = build(family, target)
            report, seconds = timed(adjudicate_specification, construction.circuit, target)
            rows.append({
                "qubits": num_qubits,
                "family": family,
                "operations": construction.operations,
                "depth": construction.depth,
                "two_qubit_gates": construction.two_qubit_gates,
                "max_edge_span": construction.max_edge_span,
                "nearest_neighbour_only": construction.nearest_neighbour_only,
                "verdict": report.verdict,
                "tier": report.tier,
                "stabilizers_checked": report.checks[1].measured["stabilizers_checked"],
                "verification_seconds": seconds,
                "circuit_sha256": construction.circuit.sha256(),
            })
    return rows


def non_clifford_boundary() -> dict[str, Any]:
    """Where verification crosses the Clifford boundary, and where it stops."""
    rows = []
    for num_qubits in (3, 8, DENSE_SPEC_LIMIT + 2):
        circuit = non_clifford_ghz(num_qubits)
        report = adjudicate_specification(circuit, ghz_target(num_qubits))
        rows.append({
            "qubits": num_qubits,
            "circuit_sha256": circuit.sha256(),
            "is_clifford": is_clifford_circuit(circuit),
            "tier": report.tier,
            "verdict": report.verdict,
            "reason": report.checks[-1].measured.get("reason"),
            "distance": report.checks[-1].measured.get("distance"),
        })
    return {
        "dense_spec_limit": DENSE_SPEC_LIMIT,
        "rows": rows,
        "note": ("the dense specification referee decides a non-Clifford construction "
                 "within its bound and says UNRESOLVED above it; no non-Clifford search "
                 "exists, so Experiment C is not demonstrated here"),
    }


def experiment_d(qubits: int) -> dict[str, Any]:
    """The same construction on four graphs, routed by greedy and by lookahead.

    `logical_depth` is the abstract circuit's depth, the number Experiment A quotes.
    `mapped_depth` is the depth of the circuit that can run on the topology, with the
    routing swaps **and** the swap ladder that restores the wire permutation; the
    pre-restore depth is reported beside it so the cost of being verifiable is
    visible rather than folded in.

    Both routers are run on every row, so the last section's claim that routing
    dominates the difference between topologies is a measurement against a baseline,
    not against an assumption. The lookahead router schedules each swap into the
    earliest layer its two nodes are free and breaks ties by what the swap does to the
    next eight two-qubit gates; the greedy router is unchanged, so its numbers are the
    ones this record reported before there was a lookahead to compare with.

    Every routed circuit is decided against the same specification, because a
    placement that is cheap and wrong is worse than one that is expensive.
    """
    target = ghz_target(qubits)
    rows = []
    comparisons = []
    for family in PLACED_FAMILIES:
        construction = build(family, target)
        for topology in ("line", "ring", "grid", "all_to_all"):
            connectivity = CONNECTIVITIES[topology](qubits)
            placements = {}
            for router in ROUTERS:
                placement, seconds = timed(place, construction.circuit, connectivity,
                                           strategy=router)
                report = adjudicate_specification(placement.circuit, target)
                placements[router] = placement
                rows.append({
                    "family": family,
                    "topology": topology,
                    "placement_seconds": seconds,
                    "verdict": report.verdict,
                    "tier": report.tier,
                    "stabilizers_checked": report.checks[1].measured["stabilizers_checked"],
                    **placement.as_dict(),
                })
            def accounting(placement: Any) -> dict[str, Any]:
                """Both depths and all three two-qubit bases, for one router."""
                return {
                    "mapped_depth": placement.mapped_depth,
                    "depth_before_restore": placement.depth_before_restore,
                    "restore_depth_cost": placement.restore_depth_cost,
                    "logical_two_qubit_gates": placement.logical_two_qubit_gates,
                    "abstract_two_qubit_gates": placement.two_qubit_gates,
                    "swap_gates": placement.swap_gates,
                    "total_swaps": placement.total_swaps,
                    "native_entanglers": placement.native_entanglers,
                    "cx_equivalent_two_qubit_gates":
                        placement.cx_equivalent_two_qubit_gates,
                }

            greedy, lookahead = placements["greedy"], placements["lookahead"]
            comparisons.append({
                "family": family,
                "topology": topology,
                "greedy": accounting(greedy),
                "lookahead": accounting(lookahead),
                "mapped_depth_saved": greedy.mapped_depth - lookahead.mapped_depth,
                "mapped_depth_ratio": lookahead.mapped_depth / greedy.mapped_depth,
                "swaps_saved": greedy.swap_gates - lookahead.swap_gates,
                "two_qubit_gates_saved": (greedy.two_qubit_gates
                                          - lookahead.two_qubit_gates),
                "cx_equivalent_saved": (greedy.cx_equivalent_two_qubit_gates
                                        - lookahead.cx_equivalent_two_qubit_gates),
            })
    return {
        "qubits": qubits,
        "routers": list(ROUTERS),
        "greedy_router": ("greedy: move the second operand along a shortest path to the "
                          "first, one swap at a time, no scheduling and no lookahead"),
        "lookahead_router": ("lookahead: schedule each swap into the earliest layer its two nodes "
                             "are free, then prefer the candidate that minimises the "
                             "wire distance of the next 8 two-qubit gates involving a "
                             "moved qubit, then the swap that moves the later-needed "
                             "qubit, then the greedy direction"),
        "lookahead_window": 8,
        "swap_accounting": {
            "swap_entanglers": 1,
            "swap_as_cnot": SWAP_AS_CNOT,
            "note": ("two-qubit counts are reported in three bases: abstract "
                     "operations (a swap is one instruction), native entanglers in the "
                     "declared basis (one per swap here, i.e. a native SWAP), and "
                     "CNOT-equivalent (a swap is three CNOTs). mapped_depth is the "
                     "total including the restore ladder; depth_before_restore is the "
                     "circuit before that ladder and restore_depth_cost is the "
                     "difference"),
        },
        "rows": rows,
        "comparison": comparisons,
        "window_sweep": experiment_d_window_sweep(qubits),
        "regression_case": lookahead_regression(),
        "pareto": placement_pareto(rows),
        "sabre_seeds": sabre_seed_sweep(qubits, rows),
        "summary": {
            "rows_improved": sum(1 for row in comparisons if row["mapped_depth_saved"] > 0),
            "rows_unchanged": sum(1 for row in comparisons
                                  if row["mapped_depth_saved"] == 0),
            "rows_worse": sum(1 for row in comparisons if row["mapped_depth_saved"] < 0),
            "mapped_depth_saved_total": sum(row["mapped_depth_saved"]
                                            for row in comparisons),
            "swaps_saved_total": sum(row["swaps_saved"] for row in comparisons),
            "cx_equivalent_saved_total": sum(row["cx_equivalent_saved"]
                                             for row in comparisons),
            "best_mapped_depth_ratio": min(row["mapped_depth_ratio"]
                                           for row in comparisons),
        },
    }


def _graph_targets() -> list[tuple[str, str, Any]]:
    """Experiment E's specifications: ``(description, target, note)``, target by target."""
    targets: list[tuple[str, str, Any]] = []
    for rows, cols in CLUSTER_SHAPES:
        targets.append((f"cluster_{rows}x{cols}",
                        cluster_state_target(rows, cols),
                        f"{rows}x{cols} square lattice, open boundary: "
                        f"2*rows*cols - rows - cols edges"))
    for qubits, seed, probability in IRREGULAR_CASES:
        targets.append((f"irregular_{qubits}q_seed{seed}",
                        seeded_graph_target(qubits, seed=seed,
                                            extra_edge_probability=probability),
                        f"seeded random spanning tree plus extra edges at p={probability} "
                        f"(seed {seed})"))
    # K9 is the class-2 case inside Experiment E: 36 edges, degree 8, and no 8-layer
    # schedule exists (36 > 8 * 4), so the measured minimum is 9 layers. It is a target
    # rather than only a colouring row so the peel's 15 layers are shown against a
    # built, verified schedule beside the clusters rather than in a separate table.
    targets.append(("complete_k9",
                    irregular_graph_target(9, CLASS_TWO_GRAPHS["complete_k9"][1],
                                           decides="the complete graph K9 state"),
                    "complete graph on 9 vertices: 36 edges, degree 8, edge "
                    "chromatic number 9"))
    return targets


def _scheduler_entry(description: str, target: Any) -> dict[str, Any]:
    """What the minimum edge colouring costs, next to what the greedy peel cost."""
    graph = sorted(graph_of_state(target))
    peel = greedy_peel(graph)
    schedule = minimum_edge_colouring(graph, num_nodes=target.num_qubits)
    return {
        "description": description,
        **schedule.as_dict(),
        "greedy_cz_layers": len(peel),
        "layers_saved": len(peel) - schedule.num_colours,
        "proof_kind": _proof_kind(schedule),
        "bounds": graph_bounds(description, target),
    }


def _proof_kind(schedule: Any) -> str:
    """Which argument the schedule's size rests on, named rather than implied."""
    if not schedule.delta_refuted:
        return "class 1"
    if schedule.status == "delta_refuted_by_layer_capacity":
        return "class 2, refuted by layer capacity"
    return "class 2, refuted by exhausted search"


def class_two_rows() -> list[dict[str, Any]]:
    """Graphs with ``chi' > Delta``: the scheduler must prove, not assume, the extra layer."""
    rows = []
    for name, (qubits, edges) in CLASS_TWO_GRAPHS.items():
        target = irregular_graph_target(qubits, edges,
                                        decides=f"the {name} graph state")
        entry = _scheduler_entry(name, target)
        construction = build("graph_layers", target)
        report = adjudicate_specification(construction.circuit, target)
        resources = construction.circuit.resources()
        rows.append({
            **entry,
            "family": "graph_layers",
            "qubits": qubits,
            "depth": int(resources["depth"]),
            "two_qubit_gates": int(resources["two_qubit_gates"]),
            "verdict": report.verdict,
            "tier": report.tier,
            "circuit_sha256": construction.circuit.sha256(),
        })
    return rows


def experiment_e() -> dict[str, Any]:
    """A cluster and an irregular graph, specified by their edges.

    A graph state is ``X_v Z_{N(v)}`` on every vertex, so the specification names
    every edge: the edge set, the operation count, the connectivity and the
    two-qubit count are all the target's, and the only thing a construction family
    chooses is the order the commuting ``cz`` gates are emitted in. The depth is
    therefore a measurement of the *schedule*, and it has a proven floor -- a qubit
    with ``d`` neighbours needs ``d`` ``cz`` gates that cannot share a layer -- so the
    record reports the depth, the floor, and whether the schedule attained it.

    Each row also carries the **wall clock of building and of deciding that one
    schedule**, measured on the circuit the row cites (its SHA-256 is re-derived and
    compared on every rebuild, so the timing cannot drift onto a different circuit).
    Each phase is timed `TIMING_REPEATS` times and the row quotes the **median**, with
    all the samples recorded beside it: at 400-900 qubits a single run moves by tens of
    percent, which is the size of the differences these columns are read for. The
    timings are one process on this host, not a benchmark of the referee against
    another implementation.
    """
    rows = []
    targets = []
    for description, target, note in _graph_targets():
        outcome, seconds = timed(synthesize, target)
        shape = entangling_depth_lower_bound(target)
        targets.append({
            "description": description,
            "note": note,
            "specification": specification_summary(target),
            "edges": shape["edges"],
            "max_degree": shape["max_degree"],
            "busiest_qubits": shape["busiest_qubits"],
            "cz_layers_floor": shape["cz_layers_floor"],
            "floor_reason": shape["reason"],
            "synthesize_seconds": seconds,
            "promoted": outcome.promoted,
            "ranked": list(outcome.ranked),
            "pareto_frontier": list(outcome.pareto),
            "scheduler": _scheduler_entry(description, target),
        })
        for row in outcome.rows:
            layers = row["depth"] - 1
            if row["verdict"] == "NOT_CONSTRUCTIBLE":
                build_samples = verify_samples = None
                build_seconds = verify_seconds = None
            else:
                # Time the two phases separately on a circuit rebuilt from the same
                # family and target: the row's hash proves it is the recorded one.
                # Every rebuild is checked, not just the first, so a cached second run
                # cannot quietly time a different circuit.
                build_samples = []
                verify_samples = []
                construction = None
                for _ in range(TIMING_REPEATS):
                    construction, seconds = timed(build, row["family"], target)
                    if construction.circuit.sha256() != row["circuit_sha256"]:
                        raise SystemExit(
                            f"refusing to time a different circuit: rebuilding "
                            f"{row['family']} on {description} produced "
                            f"{construction.circuit.sha256()}, the row records "
                            f"{row['circuit_sha256']}")
                    build_samples.append(seconds)
                for _ in range(TIMING_REPEATS):
                    _, seconds = timed(adjudicate_specification,
                                       construction.circuit, target)
                    verify_samples.append(seconds)
                build_seconds = statistics.median(build_samples)
                verify_seconds = statistics.median(verify_samples)
            rows.append({
                "target": description,
                "qubits": target.num_qubits,
                "family": row["family"],
                "verdict": row["verdict"],
                "tier": row["tier"],
                "operations": row["operations"],
                "two_qubit_gates": row["two_qubit_gates"],
                "depth": row["depth"],
                "cz_layers": layers,
                "cz_layers_floor": shape["cz_layers_floor"],
                "attains_the_floor": layers == shape["cz_layers_floor"],
                "max_edge_span": row["max_edge_span"],
                "nearest_neighbour_only": row["nearest_neighbour_only"],
                "checks_failed": row["checks_failed"],
                "circuit_sha256": row["circuit_sha256"],
                "synthesis_seconds": build_seconds,
                "verification_seconds": verify_seconds,
                "synthesis_samples": build_samples,
                "verification_samples": verify_samples,
            })

    # The two targets are different states, so a construction for one must fail the
    # other: otherwise "every family verifies" would be a statement about the referee's
    # tolerance rather than about the constructions.
    cluster = cluster_state_target(3, 4)
    ghz = ghz_target(12)
    discrimination = {
        "ghz_chain_against_the_cluster": adjudicate_specification(
            build("chain", ghz).circuit, cluster).verdict,
        "cluster_against_the_ghz": adjudicate_specification(
            build("graph_layers", cluster).circuit, ghz).verdict,
        "note": ("both specifications have 12 qubits, so the comparison is at a "
                 "shared width; a VALID here would mean the two targets were the "
                 "same state"),
    }

    # Refusals, with the failing generators named: a missing edge, an edge the
    # specification does not have, and one `cz` replaced by `cx`.
    refusals = []
    for family in ("cluster_drop_edge", "cluster_extra_edge", "graph_swap_gate"):
        report = adjudicate_specification(build(family, cluster).circuit, cluster)
        failed = next((check for check in report.checks if check.passed is False), None)
        refusals.append({
            "family": family,
            "verdict": report.verdict,
            "checks_failed": list(report.failed()),
            "failed_count": None if failed is None else failed.measured.get("failed_count"),
            "failed_generators": (None if failed is None
                                  else failed.measured.get("failed", [])[:2]),
        })

    # Both referees, at every width both can decide, on the two targets. The irregular
    # cases are asymmetric under reversing the wires, which the cluster is not: they
    # are the only specifications that can expose a basis-order mistake between the
    # tableau and the amplitudes, and one of them did.
    agreement = []
    for description, family, target in (
        ("cluster_2x4", "graph_layers", cluster_state_target(2, 4)),
        ("cluster_2x5", "graph_raster", cluster_state_target(2, 5)),
        ("irregular_path_3q", "graph_layers", irregular_graph_target(3, ((0, 1), (0, 2)))),
        ("irregular_seeded_9q", "graph_layers", seeded_graph_target(9, seed=3)),
        ("irregular_seeded_10q", "graph_column_major", seeded_graph_target(10, seed=5)),
        ("cluster_2x3_dropped_edge", "cluster_drop_edge", cluster_state_target(2, 3)),
    ):
        result = referee_agreement(build(family, target).circuit, target)
        agreement.append({"case": description, "family": family, **result.as_dict()})

    class_two = class_two_rows()
    scheduler = {
        "algorithm": (
            "complete backtracking over Delta-edge-colourings with colour symmetry "
            "breaking, preceded by a layer-capacity bound that refutes Delta when the "
            "edges cannot fit into Delta matchings; the Delta + 1 colouring is "
            "constructed by Vizing's fan argument, never searched for"),
        "theorem": "Vizing: Delta <= chi' <= Delta + 1",
        "optimality_rule": (
            "Delta colours is optimal by the degree lower bound; Delta + 1 is "
            "optimal when Delta is refuted, either by layer capacity (more edges than "
            "Delta layers of floor(V/2) can hold) or by an exhausted Delta search, "
            "both of which prove the graph class 2. A budget-exhausted search reports "
            "its constructed Delta + 1 candidate as a schedule, never as minimal"),
        "targets": [target["scheduler"] for target in targets],
        "total_layers_saved": sum(target["scheduler"]["layers_saved"]
                                  for target in targets),
        "targets_where_greedy_was_already_optimal": [
            target["description"] for target in targets
            if target["scheduler"]["layers_saved"] == 0],
        "class_two": class_two,
        "class_two_layers_saved": sum(row["layers_saved"] for row in class_two),
        "bounds": {
            "targets": [target["scheduler"]["bounds"] for target in targets],
            "refuted_by_layer_capacity": sum(
                1 for target in targets
                if target["scheduler"]["bounds"]["layer_capacity_refutes_delta"]),
            "refuted_by_matching": sum(
                1 for target in targets
                if target["scheduler"]["bounds"]["matching_bound_refutes_delta"]),
            "note": ("both counting bounds are reported on every target, not only "
                     "where one of them fires: the layer capacity `Delta * "
                     "floor(V/2)` and the matching bound `ceil(|E| / nu)` with `nu` "
                     "a maximum matching. Neither refutes `Delta` on a class-1 "
                     "graph, which is what makes their silence on these targets a "
                     "result rather than an unused test"),
        },
        "overshoot": construction_overshoot(),
    }

    return {
        "targets": targets,
        "scheduler": scheduler,
        "timing": {
            "statistic": "median",
            "repeats_per_phase_per_schedule": TIMING_REPEATS,
            "sample_columns": ["synthesis_samples", "verification_samples"],
            "why": (
                "a single wall clock at 400-900 qubits moves by tens of percent "
                "between runs on a loaded host, which is the size of the differences "
                "read off these columns; each phase is therefore timed three times on "
                "the same rebuilt circuit and the row quotes the median, with all "
                "three samples recorded beside it so the statistic is recomputable "
                "and the spread it hid is visible"),
        },
        "rows": rows,
        "shared_edge_set": {
            "note": ("a graph state names every edge, so the edge set is the "
                     "specification's and only the emission order is the family's: "
                     "the operation count, the two-qubit count and the connectivity "
                     "are identical across schedules, and only the depth moves"),
            "by_target": {
                target["description"]: {
                    "operations": sorted({row["operations"] for row in rows
                                          if row["target"] == target["description"]
                                          and row["family"] in GRAPH_FAMILIES}),
                    "two_qubit_gates": sorted({row["two_qubit_gates"] for row in rows
                                               if row["target"] == target["description"]
                                               and row["family"] in GRAPH_FAMILIES}),
                    "max_edge_span": sorted({row["max_edge_span"] for row in rows
                                              if row["target"] == target["description"]
                                              and row["family"] in GRAPH_FAMILIES}),
                    "depths": sorted({row["depth"] for row in rows
                                      if row["target"] == target["description"]
                                      and row["family"] in GRAPH_FAMILIES}),
                }
                for target in targets
            },
        },
        "routing": experiment_e_routing(),
        "second_graph_source": experiment_e_second_source(),
        "discrimination": discrimination,
        "refusals": refusals,
        "referee_agreement": {
            "rows": agreement,
            "all_agree": all(row["agree"] for row in agreement),
            "accepted": sum(1 for row in agreement
                            if row["stabilizer_verdict"] == "VALID"),
            "refused": sum(1 for row in agreement
                           if row["stabilizer_verdict"] == "INVALID"),
        },
    }


#: The Watts-Strogatz cells: (width, ring degree, rewiring probability, seed). Two
#: widths, two ring degrees, three seeds, so the family is measured at more than one size
#: and the result is not one lucky draw.
WATTS_STROGATZ_CELLS = ((32, 4, 0.1, 0), (48, 4, 0.1, 1), (64, 6, 0.1, 2))


def experiment_e_second_source() -> dict[str, Any]:
    """Experiment E on a second graph source: Watts-Strogatz, not a lattice.

    Every target in the measured table is either a square-lattice cluster or a seeded
    tree-plus-edges draw. Both are connected graphs, but both have a shape the scheduler
    could be reading: the cluster is planar, bipartite and degree-4, and the seeded draw
    is a tree with extra edges. A Watts-Strogatz graph is neither -- it has triangles,
    odd cycles and a degree distribution set by the rewire -- so a schedule that still
    attains its floor there is not attaining it because of the lattice.

    The contrast that makes it a *small-world* source is measured beside the rows, not
    asserted: local clustering at ``beta = 0`` (the ring lattice), at ``beta = 0.1``
    (the small-world regime) and at ``beta = 1`` (rewired toward a random graph) at the
    same width, degree and seed.
    """
    from apedp_dyn.vqc.synthesis import small_world_shape, watts_strogatz_target

    rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for width, degree, beta, seed in WATTS_STROGATZ_CELLS:
        target = watts_strogatz_target(width, k=degree, beta=beta, seed=seed)
        description = f"watts_strogatz_{width}q_k{degree}_b{beta:g}_s{seed}"
        shape = entangling_depth_lower_bound(target)
        outcome, seconds = timed(synthesize, target)
        targets.append({
            "description": description,
            "qubits": width,
            "k": degree,
            "beta": beta,
            "seed": seed,
            "specification": specification_summary(target),
            "edges": shape["edges"],
            "max_degree": shape["max_degree"],
            "cz_layers_floor": shape["cz_layers_floor"],
            "synthesize_seconds": seconds,
            "promoted": outcome.promoted,
            "ranked": list(outcome.ranked),
            "scheduler": _scheduler_entry(description, target),
        })
        for row in outcome.rows:
            layers = row["depth"] - 1
            rows.append({
                "target": description,
                "family": row["family"],
                "verdict": row["verdict"],
                "tier": row["tier"],
                "operations": row["operations"],
                "two_qubit_gates": row["two_qubit_gates"],
                "depth": row["depth"],
                "cz_layers": layers,
                "cz_layers_floor": shape["cz_layers_floor"],
                "attains_the_floor": layers == shape["cz_layers_floor"],
                "circuit_sha256": row["circuit_sha256"],
            })

    contrast = []
    for beta in (0.0, 0.1, 1.0):
        stats = small_world_shape(WATTS_STROGATZ_CELLS[0][0],
                                  k=WATTS_STROGATZ_CELLS[0][1], beta=beta,
                                  seed=WATTS_STROGATZ_CELLS[0][3])
        contrast.append({"beta": beta, **stats})

    return {
        "task": ("the Experiment E schedule measurement on a second graph source: "
                 "Watts-Strogatz, which has triangles, odd cycles and a rewire-set "
                 "degree distribution that neither the lattice nor the seeded tree has"),
        "cells": [{"width": width, "k": degree, "beta": beta, "seed": seed}
                  for width, degree, beta, seed in WATTS_STROGATZ_CELLS],
        "targets": targets,
        "rows": rows,
        "small_world_contrast": {
            "measured_on": {"width": WATTS_STROGATZ_CELLS[0][0],
                            "k": WATTS_STROGATZ_CELLS[0][1],
                            "seed": WATTS_STROGATZ_CELLS[0][3]},
            "rows": contrast,
            "what_it_shows": (
                "clustering falls as the rewire probability rises at the same width, "
                "degree and seed, so the middle row is the small-world regime rather "
                "than a random graph wearing its name"),
        },
    }


def experiment_e_routing() -> dict[str, Any]:
    """The cluster schedules placed on device topologies: what routing does to the gap.

    Experiment E measures depth where every qubit may talk to every other. A schedule's
    advantage over its alternatives is a *logical* number until the circuit is routed,
    and grouping gates into the fewest layers is not obviously the ordering that
    minimises SWAPs — so the advantage is measured again after routing, on a line, on a
    ring and on the cluster's own lattice. The lattice is the control: there the
    identity placement already respects every edge, so a schedule that keeps its
    advantage there has not had it tested by routing at all.

    For each topology and router the reference is the minimum schedule (`graph_layers`)
    and every other family is compared to it twice: the layers it is deeper *logically*
    and the layers it is deeper *mapped*. Their ratio is how much of the advantage
    survives; 1.0 means routing changed nothing, a value below 1.0 means routing paid
    back part of the gap, and above 1.0 means routing widened it. Every routed circuit
    is decided against the same specification, because a cheap placement that is wrong
    is worse than an expensive one.
    """
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    not_routed: list[dict[str, Any]] = []
    for description, target, _note in _graph_targets():
        if not description.startswith("cluster_"):
            continue
        if target.num_qubits > ROUTING_MAX_QUBITS:
            not_routed.append({
                "target": description,
                "qubits": target.num_qubits,
                "reason": ("above the routing width: timed for build and decide, not "
                           "placed, because the greedy router needs tens of seconds "
                           "per placement at this size"),
            })
            continue
        qubits = target.num_qubits
        for topology in ROUTING_TOPOLOGIES:
            connectivity = CONNECTIVITIES[topology](qubits)
            for router in ROUTERS:
                by_family: dict[str, dict[str, Any]] = {}
                for family in GRAPH_FAMILIES:
                    circuit = build(family, target).circuit
                    placement, seconds = timed(place, circuit, connectivity,
                                               strategy=router)
                    report = adjudicate_specification(placement.circuit, target)
                    row = {
                        "target": description,
                        "qubits": qubits,
                        "topology": topology,
                        "router": router,
                        "family": family,
                        "placement_seconds": seconds,
                        "verdict": report.verdict,
                        "tier": report.tier,
                        "logical_depth": placement.logical_depth,
                        "mapped_depth": placement.mapped_depth,
                        "depth_before_restore": placement.depth_before_restore,
                        "restore_depth_cost": placement.restore_depth_cost,
                        "logical_two_qubit_gates": placement.logical_two_qubit_gates,
                        "two_qubit_gates": placement.two_qubit_gates,
                        "cx_equivalent_two_qubit_gates":
                            placement.cx_equivalent_two_qubit_gates,
                        "swap_gates": placement.swap_gates,
                        "restore_swap_gates": placement.restore_swap_gates,
                        "total_swaps": placement.total_swaps,
                        "longest_hop": placement.longest_hop,
                        "restored": placement.restored,
                        "circuit_sha256": placement.circuit.sha256(),
                    }
                    by_family[family] = row
                reference = by_family[ROUTING_REFERENCE]
                for family, row in by_family.items():
                    if family == ROUTING_REFERENCE:
                        row["logical_advantage"] = 0
                        row["mapped_advantage"] = 0
                        row["surviving_fraction"] = 1.0
                        rows.append(row)
                        continue
                    logical = row["logical_depth"] - reference["logical_depth"]
                    mapped = row["mapped_depth"] - reference["mapped_depth"]
                    row["logical_advantage"] = logical
                    row["mapped_advantage"] = mapped
                    row["surviving_fraction"] = (mapped / logical) if logical else None
                    rows.append(row)
                    comparisons.append({
                        "target": description,
                        "qubits": qubits,
                        "topology": topology,
                        "router": router,
                        "family": family,
                        "logical_advantage": logical,
                        "mapped_advantage": mapped,
                        "surviving_fraction": row["surviving_fraction"],
                        "survives": mapped > 0,
                        "reference_swaps": reference["total_swaps"],
                        "family_swaps": row["total_swaps"],
                    })

    fractions = [row["surviving_fraction"] for row in comparisons
                 if row["surviving_fraction"] is not None]
    by_topology = {
        topology: {
            "comparisons": sum(1 for row in comparisons if row["topology"] == topology),
            "survives": sum(1 for row in comparisons
                            if row["topology"] == topology and row["survives"]),
            "min_surviving_fraction": min(
                (row["surviving_fraction"] for row in comparisons
                 if row["topology"] == topology and row["surviving_fraction"] is not None),
                default=None),
            "max_surviving_fraction": max(
                (row["surviving_fraction"] for row in comparisons
                 if row["topology"] == topology and row["surviving_fraction"] is not None),
                default=None),
        }
        for topology in ROUTING_TOPOLOGIES
    }
    grid_rows = [row for row in rows if row["topology"] == "grid"]
    return {
        "note": ("the five Experiment E schedules are placed on a line, a ring and the "
                 "cluster's own lattice under both routers; the advantage quoted for "
                 "each topology is measured against the minimum schedule and counted "
                 "in mapped layers, which include the routing swaps and the ladder "
                 "that restores the wire permutation"),
        "reference": ROUTING_REFERENCE,
        "max_qubits": ROUTING_MAX_QUBITS,
        "routed_targets": sorted({row["target"] for row in rows}),
        "not_routed": not_routed,
        "topologies": list(ROUTING_TOPOLOGIES),
        "routers": list(ROUTERS),
        "rows": rows,
        "comparisons": comparisons,
        "summary": {
            "rows": len(rows),
            "comparisons": len(comparisons),
            "survives_everywhere": all(row["survives"] for row in comparisons),
            "inverted_cases": [row for row in comparisons if not row["survives"]],
            "min_surviving_fraction": min(fractions) if fractions else None,
            "median_surviving_fraction": (sorted(fractions)[len(fractions) // 2]
                                          if fractions else None),
            "max_surviving_fraction": max(fractions) if fractions else None,
            "by_topology": by_topology,
            "lattice_needs_no_routing": bool(grid_rows) and all(
                row["total_swaps"] == 0 and row["mapped_depth"] == row["logical_depth"]
                for row in grid_rows),
            "every_placed_circuit_verifies": all(
                row["verdict"] == "VALID" for row in rows),
        },
    }


def _virtual_memory_bytes() -> int | None:
    """This process's address space, so a limit can be set above the interpreter."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - Linux is the target, not a requirement
        return None
    return None  # pragma: no cover


def dense_probe_child(width: int) -> int:
    """One width, one process: print the measurement as a single JSON line.

    A fresh process is what makes peak RSS attributable to a width: `ru_maxrss` is a
    high-water mark of the whole process, so measuring several widths in one process
    would report the largest peak for every smaller width.

    The address space is capped *after* the imports (NumPy and the package are part of
    the baseline, not of the measurement) at the documented budget, so a width whose
    Pauli operator does not fit raises `MemoryError` here rather than being killed by
    the kernel. A `MemoryError` above the bound is a measurement of the wall, not a
    failed run; the referee's own verdict is recorded beside it either way.
    """
    import resource

    baseline = _virtual_memory_bytes()
    enforced = baseline is not None and hasattr(resource, "RLIMIT_AS")
    if enforced:
        limit = baseline + DENSE_PROBE_MEMORY_CEILING
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    target = ghz_target(width)
    circuit = non_clifford_ghz(width)
    cost = None
    status = "measured"
    reason = None
    try:
        cost = dense_tier_cost(circuit, target).as_dict()
    except MemoryError as error:
        status = "exceeded_memory"
        reason = f"{type(error).__name__}: {error}"[:400]

    # The referee's policy answer at this width: the dense tier at or below its bound,
    # `UNRESOLVED` with no tier above it. It never materialises the state above the
    # bound, so this is cheap even at the width that just failed to allocate.
    policy = adjudicate_specification(circuit, target)
    payload = {
        "width": width,
        "status": status,
        "reason": reason,
        "cost": cost,
        "policy_verdict": policy.verdict,
        "policy_tier": policy.tier,
        "policy_reason": policy.checks[-1].measured.get("reason"),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "memory_ceiling_bytes": DENSE_PROBE_MEMORY_CEILING,
        "memory_ceiling_enforced": enforced,
        "circuit_sha256": circuit.sha256(),
        "specification_sha256": target.sha256(),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _meminfo(key: str) -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - the host block is best effort
        return None
    return None  # pragma: no cover


def dense_ceiling() -> dict[str, Any]:
    """Where the dense tier stops being practical, width by width, measured.

    The tier materialises every stabilizer as a dense ``2**n x 2**n`` Pauli matrix, so
    its dominant cost is ``16 * 4**n`` bytes -- quadratic in the Hilbert dimension,
    while the state it is applied to is ``16 * 2**n``. The declared bound of
    :data:`DENSE_SPEC_LIMIT` qubits is therefore a policy that has to be priced: this
    runs the same construction (a ``ry(pi/2)`` GHZ, which the stabilizer tier cannot
    decide because it is not Clifford) at each width, in a fresh process with a
    declared address-space ceiling, and records seconds, peak traced bytes and peak
    RSS. At or below the bound the referee's own verdict is recorded beside the price;
    above it the referee still refuses, and only the price is new.
    """
    rows = []
    for width in DENSE_PROBE_WIDTHS:
        command = [sys.executable, str(Path(__file__)), "--dense-probe", str(width)]
        try:
            # `command` is this interpreter and this file plus an integer: no input
            # reaches it from outside.
            completed = subprocess.run(command, capture_output=True, text=True,  # noqa: S603
                                       timeout=DENSE_PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            rows.append({"width": width, "status": "timeout", "cost": None,
                         "reason": f"exceeded {DENSE_PROBE_TIMEOUT_SECONDS:g} s"})
            continue
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0 or not lines:
            stderr = completed.stderr.strip().splitlines()
            rows.append({"width": width, "status": "failed", "cost": None,
                         "child_exit_code": completed.returncode,
                         "reason": (stderr[-1] if stderr else "no output")[:400]})
            continue
        payload = json.loads(lines[-1])
        payload["child_exit_code"] = completed.returncode
        rows.append(payload)

    measured = [row for row in rows if row["status"] == "measured"]
    stopped = [row for row in rows if row["status"] != "measured"]
    wall = None
    if stopped:
        first = min(stopped, key=lambda row: row["width"])
        wall = {
            "width": first["width"],
            "status": first["status"],
            "reason": first["reason"],
            "pauli_operator_bytes": 16 * 4 ** first["width"],
            "state_vector_bytes": 16 * 2 ** first["width"],
            "operator_over_budget": (16 * 4 ** first["width"]) / DENSE_PROBE_MEMORY_CEILING,
        }

    growth = []
    for previous, current in itertools.pairwise(measured):
        if previous["width"] < 10:
            continue
        growth.append({
            "from": previous["width"],
            "to": current["width"],
            "peak_ratio": (current["cost"]["peak_traced_bytes"]
                           / previous["cost"]["peak_traced_bytes"]),
            "time_ratio": current["cost"]["seconds"] / previous["cost"]["seconds"],
        })

    return {
        "widths": list(DENSE_PROBE_WIDTHS),
        "rows": rows,
        "growth": growth,
        "memory_ceiling_bytes": DENSE_PROBE_MEMORY_CEILING,
        "timeout_seconds": DENSE_PROBE_TIMEOUT_SECONDS,
        "wall": wall,
        "largest_measured_width": max((row["width"] for row in measured), default=None),
        "note": (
            "the tier materialises one 2**n x 2**n Pauli operator per stabilizer "
            "(16 * 4**n bytes), so peak memory is quadratic in the Hilbert dimension; "
            "each width is measured in a fresh process with a declared RLIMIT_AS "
            "ceiling, so a width that does not fit the budget is refused by the "
            "process rather than by the kernel, and the budget is the record's, not "
            "the host's"),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "memory_total_bytes": _meminfo("MemTotal"),
            "memory_available_bytes": _meminfo("MemAvailable"),
        },
    }


def dense_tier_refusal(circuit: Circuit, target: Any) -> dict[str, Any]:
    """What the dense tier answers when it is asked for a width above its bound.

    The refusal is a row of the corpus rather than a footnote. Above
    `DENSE_SPEC_LIMIT` the second in-package referee does not exist, and a corpus that
    only held widths both tiers can run would imply it exists everywhere. The call is
    actually made -- the same path `referee_agreement` uses, which raises rather than
    falling back -- so the row shows the bound being enforced instead of restating it.
    """
    try:
        report = adjudicate_specification(circuit, target, tier="dense")
    except SynthesisError as error:
        return {
            "tier": "dense",
            "qubits": target.num_qubits,
            "bound": DENSE_SPEC_LIMIT,
            "refused": True,
            "verdict": "REFUSED",
            "reason": str(error),
        }
    return {
        "tier": "dense",
        "qubits": target.num_qubits,
        "bound": DENSE_SPEC_LIMIT,
        "refused": False,
        "verdict": report.verdict,
        "reason": None,
    }


def third_party_row(target: Any, circuit: Circuit, *, target_name: str,
                    family: str | None = None) -> dict[str, Any]:
    """One circuit decided by the stabilizer tier and again by a foreign referee.

    Both verdicts are stored, and `agree` is computed only when the foreign referee
    actually decided the circuit: an abstention has no verdict to compare, and folding
    one into a boolean is how an unmeasured agreement gets recorded.
    """
    stabilizer = adjudicate_specification(circuit, target, tier="stabilizer")
    second_opinion = stim_verdict(circuit, target)
    decided = second_opinion.comparative
    row = {
        "target": target_name,
        "qubits": target.num_qubits,
        "circuit_sha256": circuit.sha256(),
        "stabilizer": {
            "tier": stabilizer.tier,
            "verdict": stabilizer.verdict,
            "checks_failed": list(stabilizer.failed()),
        },
        "second_referee": second_opinion.as_dict(),
        "agree": (stabilizer.verdict == second_opinion.verdict if decided else None),
        "agree_note": (
            "two independent decisions of one question" if decided else
            f"not compared: {second_opinion.referee} answered "
            f"{second_opinion.verdict} rather than deciding the circuit"),
    }
    if family is not None:
        row["family"] = family
    return row


def referee_agreement_rows() -> dict[str, Any]:
    """Both referees on the same circuit, at every width both can decide.

    The dense tier solves the specification for its state vector and compares
    amplitudes; the stabilizer tier tests group membership in exact integer
    arithmetic. They share the specification and nothing else, so an agreeing verdict
    is two independent answers to one question -- and a wrong construction is included
    so the agreement is not only ever tested on the accept side.

    Two-tier agreement stops at the dense tier's bound, which is why the wide rows pair
    the stabilizer tier with a second *implementation* instead of a third tier: the 400-
    and 900-qubit clusters are decided again by Stim, the three wrong graph constructions
    are decided again and refused by both, and the dense tier's refusals at both widths
    are recorded beside them.
    """
    rows = []
    cases = [("ghz", family, qubits) for qubits in range(2, DENSE_SPEC_LIMIT + 1)
             for family in ("chain", "balanced_tree")]
    cases += [("ring_graph", "ring", qubits) for qubits in range(3, 7)]
    for target_name, family, qubits in cases:
        target = ghz_target(qubits) if target_name == "ghz" else ring_graph_target(qubits)
        circuit = build(family, target).circuit
        agreement = referee_agreement(circuit, target)
        rows.append({"target": target_name, "family": family, **agreement.as_dict()})
    target = ghz_target(6)
    mutated = mutate(build("chain", target), "replace_a_cx_with_a_cz")
    rows.append({"target": "ghz", "family": "chain_mutated_cx_to_cz",
                 **referee_agreement(mutated, target).as_dict()})

    # The wide rows are not a third tier but a second implementation. The dense tier
    # stops at ten qubits by construction, so the only way to have a 400- or 900-qubit
    # circuit decided twice is to ask a referee this package did not write: Stim
    # simulates the same circuit as a tableau and evaluates the same frozen generators
    # as observable expectations. Each row is the circuit Experiment E builds and times,
    # compared by SHA-256.
    wide: dict[str, Any] = {
        "note": (
            "the two-tier corpus stops at the dense tier's bound, so its wide rows "
            "pair the stabilizer tier with an implementation outside this package "
            "(foreign_referee.py) rather than with a third tier; the rows say which "
            "they got, and 'agree' is null when that referee abstained instead of "
            "recording an agreement that was never measured"),
        "disagreement_policy": (
            "a comparative third-party verdict that differs from the stabilizer "
            "tier's is two independent answers contradicting each other, not a row to "
            "publish: the script raises and writes no record, so 'agree' is never "
            "false in a written record"),
        "rows": [],
        "refusals": [],
        "dense_tier": [],
    }
    for lattice_rows, lattice_cols in WIDE_CLUSTER_SHAPES:
        target = cluster_state_target(lattice_rows, lattice_cols)
        circuit = build("graph_layers", target).circuit
        wide["rows"].append(third_party_row(
            target, circuit, target_name=f"cluster_{lattice_rows}x{lattice_cols}",
            family="graph_layers"))
        # The row that cannot exist, recorded rather than left as an absence.
        wide["dense_tier"].append(dense_tier_refusal(circuit, target))
    # The same three wrong constructions Experiment E refuses, decided again by the
    # third party at the wide width: a corpus that only ever shows the foreign referee
    # accepting would say nothing about whether it refuses a wrong circuit at scale.
    refusal_target = cluster_state_target(20, 20)
    for family in ("cluster_drop_edge", "cluster_extra_edge", "graph_swap_gate"):
        circuit = build(family, refusal_target).circuit
        wide["refusals"].append(third_party_row(
            refusal_target, circuit, target_name="cluster_20x20", family=family))
    wide["all_agree"] = all(row["agree"] is not False
                            for row in [*wide["rows"], *wide["refusals"]])
    wide["comparative"] = sum(
        1 for row in [*wide["rows"], *wide["refusals"]]
        if row["second_referee"]["decides_the_circuit"])
    return {
        "rows": rows,
        "all_agree": all(row["agree"] for row in rows),
        "accepted": sum(1 for row in rows if row["stabilizer_verdict"] == "VALID"),
        "refused": sum(1 for row in rows if row["stabilizer_verdict"] == "INVALID"),
        "wide": wide,
    }


def dense_oracle() -> dict[str, Any]:
    """The specification's two readings of itself, at every width both can reach."""
    rows = []
    for num_qubits in range(3, DENSE_SPEC_LIMIT + 1):
        rows.append({"target": "ghz", "qubits": num_qubits,
                     **dense_oracle_check(ghz_target(num_qubits))})
    for num_qubits in range(3, 7):
        rows.append({"target": "ring_graph", "qubits": num_qubits,
                     **dense_oracle_check(ring_graph_target(num_qubits))})
    return {"rows": rows,
            "all_agree": all(row["stabilizer_group_agrees"] and row["support_agrees"]
                             for row in rows)}


# ---------------------------------------------------------------- third referee
#: The time limit QCEC is given per pair, in seconds. It is the tool's abstention
#: threshold, not a tuned parameter: a pair it cannot decide inside this window is
#: recorded as an abstention with its reason, which is a result about the tool.
QCEC_TIMEOUT_SECONDS = 20.0
#: The lookahead windows Experiment D's router is swept over. 8 is the value the record
#: reports, so the sweep contains its own consistency check.
LOOKAHEAD_WINDOWS = (2, 4, 8, 16, 32)


def specification_circuit(target: Any) -> Circuit:
    """The specification's canonical preparation, built here rather than by a family.

    A graph state is `h` on every wire followed by one `cz` per edge, and the edge set is
    read back out of the stabilizers, so this circuit follows from the specification
    alone. It is deliberately not one of the construction families: using one as the
    reference would make the third-party comparison partly a comparison with itself.
    """
    edges = sorted(graph_of_state(target))
    ops = [Gate("h", (qubit,)) for qubit in range(target.num_qubits)]
    ops.extend(Gate("cz", (first, second)) for first, second in edges)
    return Circuit(ops=tuple(ops), num_qubits=target.num_qubits, num_bits=0)


def third_referee_rows() -> dict[str, Any]:
    """QCEC on the synthesis circuits as *unitaries*, and where it abstains.

    Our referees decide a prepared state; QCEC decides whether two circuits compute the
    same unitary up to one global phase, which is a different and stricter question -- a
    schedule that permuted its wires would prepare the same state and fail this one. So
    the comparison is run where the two questions coincide by construction: every
    graph-state schedule emits the same `h` and the same commuting `cz` set on the same
    wires, making all of them one unitary, while the wrong constructions are not.

    A pair QCEC cannot decide inside its time limit is a row with a reason, not a gap:
    the abstention is the measurement at that width.
    """
    cases: list[tuple[str, str, Any]] = [
        ("cluster_4x4", family, cluster_state_target(4, 4)) for family in GRAPH_FAMILIES]
    irregular = seeded_graph_target(25, seed=7, extra_edge_probability=0.15)
    cases.append(("irregular_25q_seed7", "graph_layers", irregular))
    for lattice_rows, lattice_cols in WIDE_CLUSTER_SHAPES:
        cases.append((f"cluster_{lattice_rows}x{lattice_cols}", "graph_layers",
                      cluster_state_target(lattice_rows, lattice_cols)))
    wrong = cluster_state_target(3, 4)
    for family in ("cluster_drop_edge", "cluster_extra_edge", "graph_swap_gate"):
        cases.append(("cluster_3x4", family, wrong))

    rows = []
    for description, family, target in cases:
        candidate = build(family, target).circuit
        reference = specification_circuit(target)
        ours = adjudicate_specification(candidate, target).verdict
        row: dict[str, Any] = {
            "target": description,
            "family": family,
            "qubits": target.num_qubits,
            "circuit_sha256": candidate.sha256(),
            "reference_sha256": reference.sha256(),
            "our_verdict": ours,
        }
        start = time.perf_counter()
        try:
            answer = qcec_equivalence(reference, candidate,
                                      timeout=QCEC_TIMEOUT_SECONDS)
        except ToolUnavailable as error:
            row["third_party"] = {"status": "unavailable", "verdict": None,
                                  "reason": str(error),
                                  "seconds": time.perf_counter() - start}
            row["decided"] = False
            row["agrees"] = None
            row["abstention_reason"] = str(error)
        else:
            verdict = answer.verdict
            equivalent = verdict in ("equivalent", "equivalent_up_to_global_phase")
            row["third_party"] = answer.as_dict()
            row["decided"] = answer.decided
            row["agrees"] = (None if not answer.decided
                             else (ours == "VALID") == equivalent)
            row["abstention_reason"] = (
                None if answer.decided
                else f"QCEC returned {verdict!r} rather than a verdict")
        rows.append(row)
    decided = [row for row in rows if row["decided"]]
    return {
        "tool": "mqt.qcec",
        "question": ("U_candidate = e^(i phi) U_reference for one global phase: the "
                     "circuits as unitaries, not the states they prepare"),
        "reference": (
            "the specification's own preparation, built from the stabilizers here "
            "(h on every wire, one cz per edge in sorted order), so no construction "
            "family is compared with itself"),
        "timeout_seconds": QCEC_TIMEOUT_SECONDS,
        "rows": rows,
        "decided": len(decided),
        "abstained": len(rows) - len(decided),
        "abstentions": [
            {"target": row["target"], "family": row["family"],
             "qubits": row["qubits"], "verdict": row["third_party"]["verdict"],
             "reason": row["third_party"].get("reason")}
            for row in rows if not row["decided"]],
        "agreements": sum(1 for row in decided if row["agrees"]),
        "all_decided_rows_agree": all(row["agrees"] for row in decided),
    }


def _maximum_matching_size(edges: Sequence[tuple[int, int]], num_nodes: int) -> int:
    """The size of a maximum matching, by augmenting paths.

    Used only to compute a lower bound on the schedule: one layer holds at most a
    matching, so `ceil(|E| / nu)` layers are needed whatever the degree is. That bound
    dominates the layer-capacity one whenever a graph has no perfect matching.
    """
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for first, second in edges:
        neighbours[first].append(second)
        neighbours[second].append(first)
    match = [-1] * num_nodes

    def augment(node: int, seen: list[bool]) -> bool:
        for other in neighbours[node]:
            if seen[other]:
                continue
            seen[other] = True
            if match[other] == -1 or augment(match[other], seen):
                match[other] = node
                match[node] = other
                return True
        return False

    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, 4 * num_nodes + 100))
    try:
        size = 0
        for node in range(num_nodes):
            if match[node] == -1 and augment(node, [False] * num_nodes):
                size += 1
    finally:
        sys.setrecursionlimit(previous)
    return size


def graph_bounds(description: str, target: Any) -> dict[str, Any]:
    """The layers this target can be proved to need, by two counting arguments.

    * layer capacity: a layer is a matching on `V` vertices, so it holds at most
      `floor(V/2)` edges and `Delta` layers cover at most `Delta * floor(V/2)`;
    * matching number: a layer holds at most `nu` edges, so `ceil(|E| / nu)` layers are
      needed -- never weaker than the capacity bound when `nu < floor(V/2)`.

    Both are refutations of `Delta`, and on a class-1 graph neither should fire. The
    record reports both on every Experiment E target rather than only where one of them
    works, so the schedule's proof rests on the strongest bound available.
    """
    graph = sorted(graph_of_state(target))
    vertices = target.num_qubits
    degree = [0] * vertices
    for first, second in graph:
        degree[first] += 1
        degree[second] += 1
    delta = max(degree)
    matching = _maximum_matching_size(graph, vertices)
    capacity = delta * (vertices // 2)
    matching_bound = math.ceil(len(graph) / matching) if matching else None
    return {
        "description": description,
        "qubits": vertices,
        "edges": len(graph),
        "max_degree": delta,
        "matching_number": matching,
        "layer_capacity": capacity,
        "layer_capacity_refutes_delta": len(graph) > capacity,
        "matching_lower_bound": matching_bound,
        "matching_bound_refutes_delta": (matching_bound is not None
                                         and matching_bound > delta),
        "lower_bound": max(delta, matching_bound or 0),
    }


def construction_overshoot() -> dict[str, Any]:
    """How often the `Delta + 1` construction overshoots on graphs that are class 1.

    The construction is only asked for the `Delta + 1` half when the `Delta` search
    fails, so its overshoot is invisible in every schedule the record builds: the row
    quotes the colouring, not what the construction would have produced. Here it is
    called directly on graphs whose `Delta`-colouring this module proves exists, and
    the layers it uses are compared with `Delta`. The rate is the point -- Vizing's
    theorem bounds the construction at `Delta + 1`, and this measures how often it
    actually spends the extra layer where it did not have to.
    """
    rows = []
    for vertices, probability in ((10, 0.2), (10, 0.4), (12, 0.3)):
        graphs = overshoot = worst = 0
        for seed in range(40):
            rng = random.Random(seed)
            edges = sorted((first, second) for first in range(vertices)
                           for second in range(first + 1, vertices)
                           if rng.random() < probability)
            if not edges:
                continue
            schedule = minimum_edge_colouring(edges, num_nodes=vertices,
                                              budget=20_000)
            if schedule.status != "delta_colouring_found":
                continue
            graphs += 1
            used = len(vizing_colouring(edges, num_nodes=vertices))
            extra = used - schedule.max_degree
            overshoot += extra > 0
            worst = max(worst, extra)
        rows.append({
            "vertices": vertices,
            "edge_probability": probability,
            "graphs_with_a_proved_delta_colouring": graphs,
            "overshoot_graphs": overshoot,
            "worst_overshoot_layers": worst,
            "overshoot_rate": (overshoot / graphs if graphs else None),
        })
    graphs = sum(row["graphs_with_a_proved_delta_colouring"] for row in rows)
    overshoot = sum(row["overshoot_graphs"] for row in rows)
    return {
        "rows": rows,
        "graphs": graphs,
        "overshoot_graphs": overshoot,
        "overshoot_rate": (overshoot / graphs if graphs else None),
        "worst_overshoot_layers": max((row["worst_overshoot_layers"] for row in rows),
                                      default=0),
        "note": ("random graphs from a seeded generator, decided by this module's own "
                 "complete search at a 20,000-node budget; only graphs whose Delta "
                 "colouring is proved are counted, and the construction is timed "
                 "nowhere -- this is a rate over graphs, not a cost"),
    }


def experiment_d_window_sweep(qubits: int) -> dict[str, Any]:
    """The lookahead router at windows 2, 4, 8, 16 and 32, on every placement.

    The window is the only free parameter of the lookahead router and 8 was a choice, so
    the record now reports all five of them: depth, swaps, two-qubit gates and wall clock
    per window, including the all-to-all topology where no routing is needed and the
    window therefore cannot matter. That row is the control: if the window moved it, the
    measurement would be contaminated by something other than routing.
    """
    target = ghz_target(qubits)
    rows = []
    for family in PLACED_FAMILIES:
        construction = build(family, target)
        for topology in ("line", "ring", "grid", "all_to_all"):
            connectivity = CONNECTIVITIES[topology](qubits)
            for window in LOOKAHEAD_WINDOWS:
                placement, seconds = timed(place, construction.circuit, connectivity,
                                           strategy="lookahead", lookahead=window)
                report = adjudicate_specification(placement.circuit, target)
                rows.append({
                    "family": family,
                    "topology": topology,
                    "window": window,
                    "mapped_depth": placement.mapped_depth,
                    "depth_before_restore": placement.depth_before_restore,
                    "swap_gates": placement.swap_gates,
                    "total_swaps": placement.total_swaps,
                    "two_qubit_gates": placement.two_qubit_gates,
                    "cx_equivalent_two_qubit_gates":
                        placement.cx_equivalent_two_qubit_gates,
                    "verdict": report.verdict,
                    "seconds": seconds,
                    "circuit_sha256": placement.circuit.sha256(),
                })
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault((row["family"], row["topology"]), []).append(row)
    best = {f"{family}/{topology}": min(case,
                                        key=lambda row: (row["mapped_depth"], row["window"]))
            for (family, topology), case in by_case.items()}
    return {
        "qubits": qubits,
        "windows": list(LOOKAHEAD_WINDOWS),
        "rows": rows,
        "best_window_by_case": {
            key: {"window": row["window"], "mapped_depth": row["mapped_depth"],
                  "swap_gates": row["swap_gates"]}
            for key, row in best.items()},
        "depth_is_monotone_in_the_window": all(
            [row["mapped_depth"] for row in sorted(case, key=lambda r: r["window"])]
            == sorted(row["mapped_depth"] for row in case)
            for case in by_case.values()),
        "all_to_all_is_window_independent": all(
            len({row["mapped_depth"] for row in case}) == 1
            for (family, topology), case in by_case.items()
            if topology == "all_to_all"),
        "summary": {
            "cases": len(by_case),
            "rows": len(rows),
            "best_depth_total": sum(row["mapped_depth"] for row in best.values()),
            "at_window_8_depth_total": sum(
                row["mapped_depth"] for row in rows if row["window"] == 8),
        },
    }


#: The transpiler seeds Qiskit's SABRE router is run under. Fixed seeds, so the sweep
#: is reproducible; the spread between them is the routing randomness, quantified.
SABRE_SEEDS = (0, 1, 2, 3, 4)


def sabre_seed_sweep(qubits: int,
                     recorded: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Qiskit's own SABRE router under five seeds, on the routed corpus.

    The seeds are the tool's randomness, not ours: same circuit, same coupling map, same
    optimisation level, five seeds. Every routed circuit is decided by mapping the
    specification onto the final layout Qiskit reports (`relabelled_target`), so a depth
    interval is quoted only for circuits the referee accepted.

    The comparable column is `depth_before_restore`: Qiskit keeps its layout while this
    package's router restores the permutation, so comparing totals would charge SABRE for
    a swap ladder it never emits.
    """
    target = ghz_target(qubits)
    ours = {(row["family"], row["topology"], row["router"]): row for row in recorded}
    rows = []
    for family in PLACED_FAMILIES:
        construction = build(family, target)
        for topology in ("line", "ring", "grid"):
            connectivity = CONNECTIVITIES[topology](qubits)
            for seed in SABRE_SEEDS:
                row: dict[str, Any] = {"family": family, "topology": topology,
                                       "seed": seed}
                try:
                    foreign = qiskit_routing(
                        construction.circuit, target=target, num_qubits=qubits,
                        edges=connectivity.edges, device=f"vqc-{topology}",
                        device_version="synthetic connectivity", seed=seed,
                        optimization_level=1)
                except ToolUnavailable as error:
                    row.update({"status": "unavailable", "reason": str(error)})
                    rows.append(row)
                    continue
                relabelled = relabelled_target(target, foreign.final_layout, qubits)
                report = adjudicate_specification(foreign.circuit, relabelled)
                row.update({
                    "status": "compared",
                    "routed_depth": foreign.routed_depth,
                    "reported_depth": foreign.reported_depth,
                    "routing_swaps": foreign.routing_swaps,
                    "two_qubit_gates": foreign.two_qubit_gates,
                    "seconds": foreign.seconds,
                    "final_layout": list(foreign.final_layout),
                    "verdict": report.verdict,
                    "tier": report.tier,
                    "checks_failed": list(report.failed()),
                    "circuit_sha256": foreign.circuit.sha256(),
                    "ours": {
                        "greedy_depth_before_restore":
                            ours[(family, topology, "greedy")]["depth_before_restore"],
                        "lookahead_depth_before_restore":
                            ours[(family, topology, "lookahead")]["depth_before_restore"],
                    },
                })
                rows.append(row)
    compared = [row for row in rows if row["status"] == "compared"]
    cases: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in compared:
        cases.setdefault((row["family"], row["topology"]), []).append(row)
    intervals = {}
    for (family, topology), case in cases.items():
        depths = [row["routed_depth"] for row in case]
        intervals[f"{family}/{topology}"] = {
            "seeds": [row["seed"] for row in case],
            "min_depth": min(depths),
            "max_depth": max(depths),
            "spread": max(depths) - min(depths),
            "distinct_circuits": len({row["circuit_sha256"] for row in case}),
            "our_lookahead_depth_before_restore":
                case[0]["ours"]["lookahead_depth_before_restore"],
        }
    return {
        "tool": "qiskit",
        "router": "transpile.sabre",
        "seeds": list(SABRE_SEEDS),
        "optimisation_level": 1,
        "topologies": ["line", "ring", "grid"],
        "rows": rows,
        "compared": len(compared),
        "unavailable": len(rows) - len(compared),
        "intervals": intervals,
        "summary": {
            "rows": len(rows),
            "cases": len(intervals),
            "cases_with_a_spread": sum(1 for entry in intervals.values()
                                       if entry["spread"] > 0),
            "largest_spread": max((entry["spread"] for entry in intervals.values()),
                                  default=0),
            "min_depth": min((entry["min_depth"] for entry in intervals.values()),
                             default=None),
            "max_depth": max((entry["max_depth"] for entry in intervals.values()),
                             default=None),
        },
        "note": ("SABRE is run under five fixed seeds on the same coupling map and "
                 "optimisation level; the spread between them is the routing "
                 "randomness, not a difference between routers. Depths are compared "
                 "against this package's `depth_before_restore`, because Qiskit keeps "
                 "its final layout while this record's router restores it"),
    }


def lookahead_regression() -> dict[str, Any]:
    """The small case where looking ahead loses, recorded rather than omitted.

    The lookahead router improves every 100-qubit row, and it is a heuristic, so the
    record states the counterexample it already knew about: a 9-qubit balanced tree on a
    line, where the window sees too little to pay for the swap it chooses. A router that
    were only ever shown winning would be a claim, not a measurement.
    """
    target = ghz_target(9)
    connectivity = CONNECTIVITIES["line"](9)
    construction = build("balanced_tree", target)
    greedy, greedy_seconds = timed(place, construction.circuit, connectivity,
                                   strategy="greedy")
    lookahead, lookahead_seconds = timed(place, construction.circuit, connectivity,
                                         strategy="lookahead")
    return {
        "qubits": 9,
        "family": "balanced_tree",
        "topology": "line",
        "greedy": {"mapped_depth": greedy.mapped_depth,
                   "swap_gates": greedy.swap_gates,
                   "cx_equivalent_two_qubit_gates":
                       greedy.cx_equivalent_two_qubit_gates,
                   "seconds": greedy_seconds,
                   "circuit_sha256": greedy.circuit.sha256()},
        "lookahead": {"mapped_depth": lookahead.mapped_depth,
                      "swap_gates": lookahead.swap_gates,
                      "cx_equivalent_two_qubit_gates":
                          lookahead.cx_equivalent_two_qubit_gates,
                      "seconds": lookahead_seconds,
                      "circuit_sha256": lookahead.circuit.sha256()},
        "lookahead_is_deeper_by": (lookahead.mapped_depth
                                   - greedy.mapped_depth),
        "note": ("a heuristic is not monotone in its own parameters: at 100 qubits the "
                 "lookahead improves every routed row, and here it is deeper. The "
                 "record carries both numbers so neither can be quoted alone"),
    }


def placement_pareto(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The routed placements that no other placement beats on both axes.

    Depth and CNOT-equivalent two-qubit count are the two costs a compiler trades, and
    the sixteen placements are sorted into the frontier and the dominated set by those
    two numbers rather than by a score. A frontier of one point would mean the ranking
    were total; the record says how many points it actually has.
    """
    points = [{"family": row["family"], "topology": row["topology"],
               "router": row["router"], "mapped_depth": row["mapped_depth"],
               "cx_equivalent_two_qubit_gates":
                   row["cx_equivalent_two_qubit_gates"]}
              for row in rows]

    def dominates(better: dict[str, Any], worse: dict[str, Any]) -> bool:
        return (better["mapped_depth"] <= worse["mapped_depth"]
                and better["cx_equivalent_two_qubit_gates"]
                <= worse["cx_equivalent_two_qubit_gates"]
                and (better["mapped_depth"] < worse["mapped_depth"]
                     or better["cx_equivalent_two_qubit_gates"]
                     < worse["cx_equivalent_two_qubit_gates"]))

    # Two routers on a placement that needs no routing produce the same circuit, so the
    # frontier is taken over distinct (depth, gates) pairs: counting duplicates would
    # manufacture a trade-off out of one point measured twice.
    distinct: dict[tuple[int, int], dict[str, Any]] = {}
    for point in points:
        distinct.setdefault((point["mapped_depth"],
                             point["cx_equivalent_two_qubit_gates"]), point)
    candidates = list(distinct.values())
    frontier = [point for point in candidates
                if not any(dominates(other, point) for other in candidates
                           if other is not point)]
    return {
        "axes": ["mapped_depth", "cx_equivalent_two_qubit_gates"],
        "points": frontier,
        "distinct_points": len(candidates),
        "dominated": len(candidates) - len(frontier),
        "is_a_trade_off": len(frontier) > 1,
        "note": ("nondominated in (depth, CNOT-equivalent two-qubit gates) over distinct "
                 "pairs; no score collapses the two. A frontier of one point means "
                 "the two costs are ordered identically in this corpus, which is "
                 "reported as the result rather than as a missing trade-off"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qubits", type=int, default=100,
                        help="width for Experiment A and the second target")
    parser.add_argument("--output", type=Path, default=RECORD)
    parser.add_argument("--dense-probe", type=int, default=None,
                        help="internal: measure one dense-tier width and print its row")
    args = parser.parse_args()
    if args.dense_probe is not None:
        return dense_probe_child(args.dense_probe)

    a = experiment_a(args.qubits)
    rows = experiment_b()
    boundary = non_clifford_boundary()
    oracle = dense_oracle()
    placement = experiment_d(args.qubits)
    graphs = experiment_e()
    ceiling = dense_ceiling()
    agreement = referee_agreement_rows()
    third = third_referee_rows()

    large_cluster = next(
        (target for target in graphs["targets"] if target["description"] == "cluster_20x20"),
        None)
    wide = agreement["wide"]
    wide_timed = {(row["target"], row["family"]): row for row in graphs["rows"]}

    # A third-party verdict that differs from ours is two independent decisions
    # contradicting each other, not a row to publish. Stop before writing anything:
    # recording a disagreement in a file whose checks all pass would be the record
    # laundering it.
    for row in [*wide["rows"], *wide["refusals"]]:
        if row["agree"] is False:
            raise SystemExit(
                f"refusing to write: {row['second_referee']['referee']} "
                f"{row['second_referee']['version']} decided "
                f"{row['second_referee']['verdict']} on {row['target']} "
                f"({row['qubits']} qubits, {row.get('family')}), where this "
                f"package's stabilizer tier decided {row['stabilizer']['verdict']}; "
                f"two independent referees contradicting each other is not a "
                f"measurement this record may report")
    checks = {
        "experiment_a_reaches_the_target": all(row["verdict"] == "VALID"
                                               for row in a["constructions"]),
        "experiment_a_refuses_every_near_miss": all(row["verdict"] == "INVALID"
                                                    for row in a["refusals"]),
        "experiment_a_promoted": a["promoted"] is not None,
        "no_reference_circuit_supplied": a["reference_circuit"] is None,
        "experiment_b_every_row_verifies": all(row["verdict"] == "VALID" for row in rows),
        "experiment_b_reaches_a_thousand_qubits": max(row["qubits"] for row in rows) == 1000,
        "dense_oracle_agrees": oracle["all_agree"],
        "non_clifford_decided_within_the_bound": any(
            row["tier"] == "dense" and row["verdict"] == "VALID" for row in boundary["rows"]),
        "non_clifford_unresolved_above_the_bound": any(
            row["tier"] == "none" and row["verdict"] == "UNRESOLVED"
            for row in boundary["rows"]),
        "experiment_d_every_placement_verifies": all(
            row["verdict"] == "VALID" for row in placement["rows"]),
        "experiment_d_line_chain_needs_no_routing": all(
            row["swap_gates"] == 0 and row["mapped_depth"] == row["logical_depth"]
            for row in placement["rows"]
            if row["family"] == "chain" and row["topology"] == "line"),
        "experiment_d_all_to_all_is_the_logical_depth": all(
            row["mapped_depth"] == row["logical_depth"] and row["total_swaps"] == 0
            for row in placement["rows"] if row["topology"] == "all_to_all"),
        "experiment_d_lookahead_never_costs_more_depth": all(
            row["lookahead"]["mapped_depth"] <= row["greedy"]["mapped_depth"]
            for row in placement["comparison"]),
        "experiment_d_lookahead_saves_depth_somewhere": any(
            row["mapped_depth_saved"] > 0 for row in placement["comparison"]),
        "experiment_d_lookahead_never_costs_more_swaps": all(
            row["swaps_saved"] >= 0 for row in placement["comparison"]),
        "experiment_d_lookahead_saves_swaps_somewhere": any(
            row["swaps_saved"] > 0 for row in placement["comparison"]),
        "experiment_d_lookahead_never_costs_more_two_qubit_gates": all(
            row["two_qubit_gates_saved"] >= 0 for row in placement["comparison"]),
        "experiment_d_swap_accounting_adds_up": all(
            side["abstract_two_qubit_gates"] == side["logical_two_qubit_gates"]
            + 2 * side["swap_gates"]
            and side["total_swaps"] == 2 * side["swap_gates"]
            and side["cx_equivalent_two_qubit_gates"] == side["logical_two_qubit_gates"]
            + SWAP_AS_CNOT * side["total_swaps"]
            and side["native_entanglers"] == side["logical_two_qubit_gates"]
            + placement["swap_accounting"]["swap_entanglers"] * side["total_swaps"]
            and side["restore_depth_cost"] == side["mapped_depth"]
            - side["depth_before_restore"]
            for row in placement["comparison"]
            for side in (row["greedy"], row["lookahead"])),
        "experiment_d_cx_equivalent_exceeds_the_abstract_count": any(
            row["greedy"]["cx_equivalent_two_qubit_gates"]
            > row["greedy"]["abstract_two_qubit_gates"]
            for row in placement["comparison"]),
        "experiment_d_cx_equivalent_is_never_cheaper_under_lookahead": all(
            row["cx_equivalent_saved"] >= 0 for row in placement["comparison"]),
        "referee_tiers_agree": agreement["all_agree"],
        "referee_tiers_disagree_on_the_wrong_construction": any(
            row["stabilizer_verdict"] == "INVALID" for row in agreement["rows"]),
        "referee_agreement_reaches_four_hundred_and_nine_hundred_qubits": (
            [row["qubits"] for row in wide["rows"]] == [400, 900]
            and all(row["stabilizer"]["verdict"] == "VALID" for row in wide["rows"])),
        "every_wide_second_opinion_is_a_verdict_or_an_abstention": all(
            (row["second_referee"]["decides_the_circuit"]
             and row["agree"] is True
             and row["second_referee"]["generators"]
             == row["second_referee"]["satisfied"])
            or (not row["second_referee"]["decides_the_circuit"]
                and row["agree"] is None
                and row["second_referee"]["verdict"] in ("UNAVAILABLE", "REFUSED")
                and bool(row["second_referee"]["reason"]))
            for row in wide["rows"]),
        "no_wide_verdict_disagrees_with_the_stabilizer_tier": (
            wide["all_agree"]
            and all(row["agree"] is not False
                    for row in [*wide["rows"], *wide["refusals"]])),
        "the_wide_agreeing_circuits_are_the_timed_circuits": all(
            row["circuit_sha256"]
            == wide_timed[(row["target"], "graph_layers")]["circuit_sha256"]
            for row in wide["rows"]),
        "the_wide_refusals_are_refused_by_both": all(
            row["stabilizer"]["verdict"] == "INVALID"
            and (row["second_referee"]["verdict"] == "INVALID"
                 if row["second_referee"]["decides_the_circuit"]
                 else row["agree"] is None)
            for row in wide["refusals"]),
        "the_wide_refusals_name_the_generators_they_failed": all(
            row["second_referee"]["failing"]
            for row in wide["refusals"]
            if row["second_referee"]["decides_the_circuit"]),
        "the_dense_tier_refuses_every_wide_width": all(
            entry["refused"] and entry["bound"] == DENSE_SPEC_LIMIT
            and str(DENSE_SPEC_LIMIT) in entry["reason"]
            for entry in wide["dense_tier"]),
        "experiment_e_every_valid_graph_family_verifies": all(
            row["verdict"] == "VALID" for row in graphs["rows"]
            if row["family"] not in REFUSAL_FAMILIES),
        "experiment_e_every_graph_refusal_is_refused": all(
            row["verdict"] == "INVALID" for row in graphs["rows"]
            if row["family"] in REFUSAL_FAMILIES),
        "experiment_e_the_schedule_is_the_only_free_variable": all(
            len(tally["operations"]) == 1 and len(tally["two_qubit_gates"]) == 1
            and len(tally["max_edge_span"]) == 1 and len(tally["depths"]) > 1
            for tally in graphs["shared_edge_set"]["by_target"].values()),
        "experiment_e_every_minimum_schedule_is_the_floor_or_proven_class_two": all(
            row["attains_the_floor"]
            or (row["target"] == "complete_k9"
                and row["cz_layers"] == row["cz_layers_floor"] + 1)
            for row in graphs["rows"] if row["family"] == "graph_layers"),
        "experiment_e_the_class_two_target_is_in_the_measured_table": (
            any(target["description"] == "complete_k9" for target in graphs["targets"])
            and any(row["target"] == "complete_k9" and row["family"] == "graph_layers"
                    and row["cz_layers"] == 9 and row["cz_layers_floor"] == 8
                    for row in graphs["rows"])),
        "experiment_e_the_scheduler_saving_includes_that_target": any(
            target["scheduler"]["layers_saved"] > 0 for target in graphs["targets"]),
        "the_lookahead_regression_is_recorded": (
            placement["regression_case"]["lookahead_is_deeper_by"] > 0),
        "the_placement_frontier_is_nondominated": (
            placement["pareto"]["is_a_trade_off"]
            == (len(placement["pareto"]["points"]) > 1)
            and all(
                not (other["mapped_depth"] <= point["mapped_depth"]
                     and other["cx_equivalent_two_qubit_gates"]
                     <= point["cx_equivalent_two_qubit_gates"])
                or (other["mapped_depth"] == point["mapped_depth"]
                    and other["cx_equivalent_two_qubit_gates"]
                    == point["cx_equivalent_two_qubit_gates"])
                for point in placement["pareto"]["points"]
                for other in placement["pareto"]["points"] if other is not point)),
        "the_placement_frontier_partitions_the_placements": (
            placement["pareto"]["distinct_points"]
            >= len(placement["pareto"]["points"]) > 0
            and placement["pareto"]["dominated"]
            == placement["pareto"]["distinct_points"]
            - len(placement["pareto"]["points"])),
        "sabre_sweep_covers_every_case_and_seed": (
            placement["sabre_seeds"]["summary"]["rows"]
            == len(PLACED_FAMILIES) * len(placement["sabre_seeds"]["topologies"])
            * len(SABRE_SEEDS)
            and placement["sabre_seeds"]["seeds"] == list(SABRE_SEEDS)),
        "every_sabre_circuit_is_decided_on_its_layout": all(
            row["status"] != "compared"
            or (row["verdict"] == "VALID"
                and len(set(row["final_layout"])) == len(row["final_layout"]))
            for row in placement["sabre_seeds"]["rows"]),
        "sabre_intervals_recompute_from_the_rows": (
            placement["sabre_seeds"]["summary"]["cases_with_a_spread"]
            == sum(1 for entry in placement["sabre_seeds"]["intervals"].values()
                   if entry["spread"] > 0)
            and all(entry["spread"] == entry["max_depth"] - entry["min_depth"]
                    and entry["min_depth"] <= entry["max_depth"]
                    for entry in placement["sabre_seeds"]["intervals"].values())),
        "the_sabre_interval_is_measured_not_assumed": (
            placement["sabre_seeds"]["unavailable"] == 0
            or placement["sabre_seeds"]["unavailable"] == len(
                placement["sabre_seeds"]["rows"])),
        "experiment_e_schedule_is_proven_optimal": all(
            target["scheduler"]["optimal"] for target in graphs["targets"]),
        "experiment_e_scheduler_savings_recompute": (
            graphs["scheduler"]["total_layers_saved"]
            == sum(target["scheduler"]["layers_saved"]
                   for target in graphs["targets"])
            and all(target["scheduler"]["layers_saved"]
                    == target["scheduler"]["greedy_cz_layers"]
                    - target["scheduler"]["cz_layers"]
                    for target in graphs["targets"])),
        "class_two_graphs_prove_the_extra_layer": all(
            row["optimal"] and row["delta_refuted"]
            and row["edge_chromatic_number"] == row["max_degree"] + 1
            and row["verdict"] == "VALID"
            for row in graphs["scheduler"]["class_two"]),
        "class_two_schedule_saves_layers_somewhere": (
            graphs["scheduler"]["class_two_layers_saved"] > 0),
        # The odd complete graphs are the family whose Delta search cannot finish: the
        # refutation has to be a counting argument, not a run that stopped early, and it
        # has to be the proof the row's own status names.
        "odd_complete_graphs_are_refuted_by_layer_capacity": all(
            row["status"] == "delta_refuted_by_layer_capacity"
            and row["search_nodes"] == 0
            and row["layers_saved"] > 0
            for row in graphs["scheduler"]["class_two"]
            if row["description"] in {"complete_k5", "complete_k9", "complete_k11"}),
        "experiment_e_promotes_each_target": all(
            target["promoted"] for target in graphs["targets"]),
        "experiment_e_refusals_name_the_failing_generators": all(
            row["failed_count"] and row["failed_generators"]
            for row in graphs["refusals"]),
        "experiment_e_referees_agree": graphs["referee_agreement"]["all_agree"],
        "experiment_e_second_graph_source_verifies": all(
            row["verdict"] == "VALID" for row in graphs["second_graph_source"]["rows"]),
        "experiment_e_second_source_never_beats_its_floor": all(
            row["cz_layers"] >= row["cz_layers_floor"]
            for row in graphs["second_graph_source"]["rows"]),
        "experiment_e_second_source_is_a_small_world_draw": (
            graphs["second_graph_source"]["small_world_contrast"]["rows"][0]
            ["mean_clustering"]
            > graphs["second_graph_source"]["small_world_contrast"]["rows"][-1]
            ["mean_clustering"]),
        "experiment_e_agreement_corpus_contains_a_refusal": (
            graphs["referee_agreement"]["refused"] >= 1),
        "experiment_e_targets_are_different_states": all(
            verdict == "INVALID" for key, verdict in graphs["discrimination"].items()
            if key != "note"),
        "experiment_e_reaches_four_hundred_qubits": max(
            row["qubits"] for row in graphs["rows"]) >= 400,
        "experiment_e_every_schedule_carries_its_wall_clock": all(
            row["synthesis_seconds"] is not None
            and row["verification_seconds"] is not None
            and row["synthesis_seconds"] > 0.0 and row["verification_seconds"] > 0.0
            for row in graphs["rows"]),
        "experiment_e_every_wall_clock_is_a_median_of_three": all(
            len(row["synthesis_samples"]) == len(row["verification_samples"])
            == graphs["timing"]["repeats_per_phase_per_schedule"] == TIMING_REPEATS
            and row["synthesis_seconds"] == statistics.median(row["synthesis_samples"])
            and row["verification_seconds"]
            == statistics.median(row["verification_samples"])
            for row in graphs["rows"] if row["synthesis_seconds"] is not None),
        "experiment_e_no_timed_row_is_missing_its_samples": all(
            (row["synthesis_seconds"] is None) == (row["synthesis_samples"] is None)
            for row in graphs["rows"]),
        "experiment_e_routing_covers_every_schedule": (
            graphs["routing"]["summary"]["rows"]
            == len(graphs["routing"]["routed_targets"]) * len(ROUTING_TOPOLOGIES)
            * len(ROUTERS) * len(GRAPH_FAMILIES)),
        "experiment_e_routing_stops_at_its_declared_width": (
            all(row["qubits"] <= ROUTING_MAX_QUBITS
                for row in graphs["routing"]["rows"])
            and all(entry["qubits"] > ROUTING_MAX_QUBITS
                    for entry in graphs["routing"]["not_routed"])
            and bool(graphs["routing"]["not_routed"])),
        "experiment_e_the_timing_ladder_has_three_lattice_sizes": (
            sorted(target["specification"]["qubits"] for target in graphs["targets"]
                   if target["description"].startswith("cluster_")
                   and target["specification"]["qubits"] >= 100) == [100, 400, 900]
            and max(row["qubits"] for row in graphs["rows"]) == 900
            and all(row["verification_seconds"] > row["synthesis_seconds"]
                    for row in graphs["rows"] if row["qubits"] == 900)),
        "experiment_e_routing_every_placed_circuit_verifies": (
            graphs["routing"]["summary"]["every_placed_circuit_verifies"]),
        "experiment_e_routing_keeps_the_minimum_ahead": (
            graphs["routing"]["summary"]["survives_everywhere"]
            and not graphs["routing"]["summary"]["inverted_cases"]),
        "experiment_e_routing_never_shrinks_the_advantage": all(
            row["surviving_fraction"] is None or row["surviving_fraction"] >= 1.0
            for row in graphs["routing"]["comparisons"]),
        "experiment_e_routing_on_the_native_lattice_needs_no_swaps": (
            graphs["routing"]["summary"]["lattice_needs_no_routing"]),
        "experiment_e_the_400_qubit_cluster_is_the_lattice": (
            large_cluster is not None and large_cluster["edges"] == 760
            and large_cluster["max_degree"] == 4
            and large_cluster["cz_layers_floor"] == 4
            and large_cluster["specification"]["qubits"] == 400),
        "dense_referee_decides_every_width_within_its_bound": all(
            row["policy_verdict"] == "VALID" and row["cost"]["within_tolerance"]
            for row in ceiling["rows"]
            if row["status"] == "measured" and row["width"] <= DENSE_SPEC_LIMIT),
        "dense_referee_refuses_every_width_above_its_bound": all(
            row["policy_tier"] == "none" and row["policy_verdict"] == "UNRESOLVED"
            for row in ceiling["rows"] if row["width"] > DENSE_SPEC_LIMIT),
        "the_dense_tier_has_a_measured_memory_wall": (
            ceiling["wall"] is not None
            and ceiling["wall"]["status"] == "exceeded_memory"),
        "the_wall_is_where_one_operator_exceeds_the_budget": (
            ceiling["wall"] is not None
            and ceiling["wall"]["pauli_operator_bytes"]
            > ceiling["memory_ceiling_bytes"]
            and all(row["cost"]["peak_traced_bytes"] < ceiling["memory_ceiling_bytes"]
                    for row in ceiling["rows"]
                    if row["status"] == "measured"
                    and row["width"] < ceiling["wall"]["width"])),
        "the_measured_cost_grows_with_the_operator": bool(ceiling["growth"]) and all(
            pair["peak_ratio"] > 3.0 and pair["time_ratio"] > 2.0
            for pair in ceiling["growth"]),
        "third_referee_answers_every_row": all(
            row["decided"] or row["abstention_reason"] for row in third["rows"]),
        "third_referee_agrees_wherever_it_decided": third["all_decided_rows_agree"],
        "third_referee_counts_recompute": (
            third["decided"] == sum(1 for row in third["rows"] if row["decided"])
            and third["abstained"] == len(third["rows"]) - third["decided"]
            and third["agreements"] == sum(1 for row in third["rows"] if row["agrees"])),
        "third_referee_calls_the_wrong_constructions_different_unitaries": all(
            row["third_party"]["verdict"] == "not_equivalent"
            for row in third["rows"]
            if row["family"] in ("cluster_drop_edge", "cluster_extra_edge",
                                 "graph_swap_gate") and row["decided"]),
        "third_referee_is_asked_the_unitary_question": "unitaries" in third["question"],
        "window_sweep_covers_every_case_and_window": (
            placement["window_sweep"]["summary"]["rows"]
            == len(PLACED_FAMILIES) * 4 * len(LOOKAHEAD_WINDOWS)
            and placement["window_sweep"]["summary"]["cases"]
            == len(PLACED_FAMILIES) * 4
            and placement["window_sweep"]["windows"] == list(LOOKAHEAD_WINDOWS)),
        "every_window_circuit_verifies": all(
            row["verdict"] == "VALID" for row in placement["window_sweep"]["rows"]),
        "window_8_reproduces_the_recorded_lookahead": all(
            row["circuit_sha256"]
            == next(entry for entry in placement["rows"]
                    if entry["family"] == row["family"]
                    and entry["topology"] == row["topology"]
                    and entry["router"] == "lookahead")["circuit_sha256"]
            for row in placement["window_sweep"]["rows"] if row["window"] == 8),
        "all_to_all_is_independent_of_the_window": (
            placement["window_sweep"]["all_to_all_is_window_independent"]),
        "every_experiment_e_target_has_both_bounds": all(
            entry["lower_bound"] == max(entry["max_degree"],
                                        entry["matching_lower_bound"] or 0)
            and entry["layer_capacity"] == entry["max_degree"] * (entry["qubits"] // 2)
            for entry in graphs["scheduler"]["bounds"]["targets"]),
        "counting_bounds_recompute_from_the_targets": (
            graphs["scheduler"]["bounds"]["refuted_by_layer_capacity"]
            == sum(1 for entry in graphs["scheduler"]["bounds"]["targets"]
                   if entry["layer_capacity_refutes_delta"])
            and graphs["scheduler"]["bounds"]["refuted_by_matching"]
            == sum(1 for entry in graphs["scheduler"]["bounds"]["targets"]
                   if entry["matching_bound_refutes_delta"])),
        "overshoot_rates_recompute": (
            graphs["scheduler"]["overshoot"]["overshoot_rate"]
            == (graphs["scheduler"]["overshoot"]["overshoot_graphs"]
                / graphs["scheduler"]["overshoot"]["graphs"])
            and all(row["overshoot_rate"]
                    == (row["overshoot_graphs"]
                        / row["graphs_with_a_proved_delta_colouring"])
                    for row in graphs["scheduler"]["overshoot"]["rows"]
                    if row["graphs_with_a_proved_delta_colouring"])),
    }
    if not all(checks.values()):
        raise SystemExit(f"refusing to write a record with failed checks: {checks}")

    payload = {
        "schema": "apedp.vqc.synthesis.v3",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": {"qubits": args.qubits},
        "experiment_a": a,
        "second_target": second_target(args.qubits),
        "claim_refusal": claim_refusal(64),
        "experiment_b": {"sizes": list(SIZES), "rows": rows},
        "experiment_c": boundary,
        "experiment_d": placement,
        "experiment_e": graphs,
        "dense_referee_ceiling": ceiling,
        "referee_agreement": agreement,
        "third_referee": third,
        "dense_oracle": oracle,
        "checks": checks,
        "limitations": [
            "the construction space is a grammar of named wireings (chain, star, "
            "balanced binary tree, seeded random tree, ring), not a general "
            "synthesiser: the engine selects among constructions it knows how to "
            "build, and it cannot invent a topology that is not in the grammar",
            "two targets are specified (GHZ and a ring graph state); a third would "
            "need its own construction family, so 'synthesis from a specification' "
            "is demonstrated, not generalised",
            "the ranking is (depth, operations, max_edge_span) and the frontier is "
            "reported instead of a single winner, because depth and connectivity "
            "trade off; the ordering is a stated preference, not a measurement of "
            "quality",
            "non-Clifford synthesis (Experiment C) is not implemented: the dense "
            "specification referee can decide such a circuit up to "
            f"{DENSE_SPEC_LIMIT} qubits, but no search proposes one",
            "wall clock is one host and one process, and the 1,000-qubit rows are "
            "dominated by circuit construction rather than by the referee",
            "Experiment D's mapped depth is a routing bound, not a floor: the lookahead "
            "router improves every routed row in this record and a better one -- an "
            "initial-placement search, or a router that reorders commuting gates -- "
            "would improve it again; neither router is optimal",
            "the lookahead's window is eight two-qubit gates and its tie-break order is "
            "a choice, not a measurement: it scores future wire distance and schedules "
            "swaps by earliest free layer, and a different window or scoring would move "
            "the numbers, which is why both routers are reported row by row",
            "neither router reorders gates: the program order is routed as written, so "
            "commuting gates are never scheduled past each other even where a "
            "reordering would shorten the routing",
            "the lookahead is a heuristic, not a theorem of non-degradation: at small "
            "widths (a 9-qubit balanced tree on a line) it has been measured to route "
            "two layers deeper than greedy, so the 100-qubit rows are the evidence "
            "and they do not promise the same ordering everywhere",
            "the two-qubit counts are reported in three bases and none of them is a "
            "device measurement: abstract operations count a SWAP as one instruction, "
            "native entanglers assume the declared basis (one per SWAP here, i.e. a "
            "native SWAP), and CNOT-equivalent charges three CNOTs per SWAP. A "
            "CNOT-only backend therefore sees the third column, which for the greedy "
            "100-qubit lattice tree is 3,381 rather than 1,193",
            "mapped_depth is the total depth including the restore ladder, and "
            "depth_before_restore is the circuit before that ladder is appended; "
            "restore_depth_cost is the difference, reported so neither number can be "
            "mistaken for the other. No depth is decomposed into a native gate basis",
            "Experiment D includes the swap ladder that restores the wire permutation "
            "(shown separately as depth_before_restore and restore_swap_gates): a "
            "compiler that tracks its final layout can absorb that cost, and the "
            "verifiable circuit cannot",
            "the router assumes every node holds a qubit and every edge is one equally "
            "cheap two-qubit interaction; there is no device calibration, no error rate "
            "and no classical control cost in these numbers",
            "referee agreement is demonstrated only where both tiers can run -- Clifford "
            f"constructions up to {DENSE_SPEC_LIMIT} qubits -- and says nothing about "
            "a non-Clifford circuit, which only the dense tier can decide",
            "the 400- and 900-qubit agreement rows pair this package's stabilizer "
            "tier with Stim, not with the dense tier, which refuses above "
            f"{DENSE_SPEC_LIMIT} qubits and whose refusal is recorded as a row at "
            "each width. Stim decides the prepared state only, is Clifford-only, and "
            "reaches the circuit through this package's IR, so a shared translation "
            "bug would fool both referees. A missing Stim is UNAVAILABLE and the rows "
            "record no agreement rather than a weaker one, and a comparative "
            "disagreement stops the write instead of being recorded",
            "Experiment E's edge chromatic number is proved, not asserted, by an "
            "argument that carries a budget: the "
            "Delta-colouring search is complete, so a colouring it finds is minimal by "
            "the degree bound and a refutation of Delta proves the graph class 2 (then "
            "Delta + 1 is minimal by Vizing). Delta is refuted either by layer capacity "
            "-- more edges than Delta layers of floor(V/2) can hold, which is what "
            "decides the odd complete graphs -- or by exhausting the search space. "
            "Deciding the Delta case is NP-complete, so a search that only hits its node "
            "budget decides nothing: it reports the constructed Delta + 1 colouring (at "
            "most one layer above the minimum by Vizing) marked optimal=False rather "
            "than claiming a minimum",
            "the minimum schedule saves nothing on the four Experiment E targets: the "
            "greedy peel already attained Delta on all of them, so the record reports "
            "0 layers saved there and 12 on the class-2 graphs, where the peel's 15 "
            "layers on K9 and K11 are proved down to 9 and 11",
            "the cluster target is the graph state of a square lattice with qubits on "
            "its vertices, not the surface-code cluster whose qubits sit on the edges "
            "of a lattice, and no error-correction property is claimed for it",
            "the per-schedule wall clock is one run in this process on this host (the "
            "host is recorded with the dense-tier probe): synthesis is the time to "
            "build the circuit from the specification and verification is the time for "
            "the stabilizer referee to decide it, both measured on a circuit rebuilt "
            "from the same family and target and checked against the row's SHA-256. It "
            "is a measurement of this Python implementation on this machine, not a "
            "benchmark against another engine and not a hardware estimate. Both "
            "phases are timed on a rebuild in a warm process -- the same family has "
            "already been built and decided once for the outcome and the ranking -- so "
            "they are not cold-start numbers, and they move by tens of percent between "
            "runs under load",
            "the irregular targets are seeded random spanning trees with extra edges: "
            "reproducible and without regular structure, but not adversarial, and "
            "their degree floor is a property of the draw rather than of the family",
            "the routing measurement starts every schedule from the identity placement "
            "(no placement search), routes the program order as written (no gate "
            "reordering) and counts the restore ladder in the mapped depth, with both "
            "routers reported; a compiler that searches initial layouts or reorders "
            "commuting gates would move these numbers, and no device calibration or "
            "error rate is involved",
            "Experiment E's edge set is the specification's, so it measures schedule "
            "and verification, not a cheaper preparation: every family emits the same "
            "one h per qubit and one cz per edge, and none is a resource advance",
            "the dense-tier ceiling is measured with the policy bound lifted in a "
            "separate process and carries no verdict: the referee still answers "
            "UNRESOLVED above DENSE_SPEC_LIMIT, and only the price of the arithmetic "
            "is new",
            "the ceiling is a property of the declared 8 GiB address-space budget and "
            "this host: a larger budget reaches one more width at 16 * 4**n bytes per "
            "operator, and the record says which width its wall is at rather than "
            "claiming a universal limit",
            "the ceiling measures a non-Clifford GHZ construction; a target with the "
            "same number of stabilizers costs the same number of matrix-vector "
            "products, but only this construction is measured",
            "tracemalloc peak is the tier's own allocation high-water mark and peak "
            "RSS adds the interpreter and NumPy baseline, so a peak reached inside "
            "one process cannot be attributed to a single allocation",
            "the third referee decides the unitary, which is a stricter question than "
            "the one this package's referees answer: a schedule that permuted its "
            "wires would prepare the correct state and fail QCEC, so the comparison "
            "is run only on schedules that share one wire order and on constructions "
            "that are wrong as operators. A disagreement would mean the two "
            "questions came apart on a circuit, and none did",
            "QCEC decided every pair in this corpus, including the 900-qubit one, so "
            "the record holds no third-referee abstention: the abstention path exists "
            "and is pinned by tests, but this corpus does not exercise it and no "
            "claim is made about a pair the tool could not decide",
            "the lookahead window sweep is one host and one process per row; the best "
            "window is chosen by mapped depth with the smaller window breaking ties, "
            "and depth is not monotone in the window, so no ordering between windows "
            "is claimed. The window-8 column reproduces the Experiment D lookahead "
            "rows exactly, which is the sweep's own consistency check",
            "the ceiling prices the dense tier as implemented: each stabilizer is "
            "materialised as a dense 2**n x 2**n operator, so the wall is 16 * 4**n "
            "bytes. Applying the Pauli string as single-qubit gates and index "
            "permutations would use O(2**n) memory and move the wall by several "
            "qubits; that implementation is not measured here, and none of these "
            "numbers is a claim about what the question costs in principle",
        ],
        "reproducibility": {
            "script_sha256": sha256_of(Path(__file__)),
            "synthesis_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "synthesis.py"),
            "colouring_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "colouring.py"),
            "placement_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "placement.py"),
            "discovery_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "discovery.py"),
            "equivalence_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "equivalence.py"),
            "stabilizer_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "stabilizer.py"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": f"python scripts/measure_vqc_synthesis.py --qubits {args.qubits}",
        },
        "verdict": {
            "synthesis": "circuits were constructed from the specification with no "
                         "reference circuit and verified against it at 10..1000 qubits",
            "refusals": "four wrong constructions and one near-miss family were refused "
                        "with the failing stabilizer named",
            "admission": "the promoted construction went through the same four-field "
                         "policy as the rewrites loop, and an unmet claim was refused "
                         "without being called wrong",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"experiment A: {len(a['constructions'])} constructions at "
          f"{a['specification']['qubits']} qubits, promoted {a['promoted']}, "
          f"{len(a['refusals'])} near misses refused")
    print(f"experiment B: {len(rows)} rows, sizes {SIZES}, all verified")
    print(f"experiment D: {len(placement['rows'])} placements, all verified")
    print(f"experiment E: {len(graphs['targets'])} graph specifications, "
          f"{len(graphs['rows'])} constructions, "
          f"{sum(1 for row in graphs['rows'] if row['family'] == 'graph_layers')} "
          f"minimum schedules against the degree floor, "
          f"{graphs['scheduler']['total_layers_saved']} layers saved against the peel, "
          f"{graphs['scheduler']['class_two_layers_saved']} on the class-2 graphs, "
          f"largest {max(row['qubits'] for row in graphs['rows'])} qubits")
    for description, qubits in (("cluster_10x10", 100), ("cluster_20x20", 400),
                                ("cluster_30x30", 900)):
        ladder = [row for row in graphs["rows"]
                  if row["target"] == description and row["family"] in GRAPH_FAMILIES]
        if ladder:
            print(f"experiment E at {qubits:>3d} qubits (median of "
                  f"{TIMING_REPEATS}): synthesis "
                  f"{min(row['synthesis_seconds'] for row in ladder):.2f}-"
                  f"{max(row['synthesis_seconds'] for row in ladder):.2f} s, "
                  f"verification "
                  f"{min(row['verification_seconds'] for row in ladder):.2f}-"
                  f"{max(row['verification_seconds'] for row in ladder):.2f} s "
                  f"per schedule")
    routing = graphs["routing"]["summary"]
    print(f"experiment E routing: {routing['rows']} placements, "
          f"{routing['comparisons']} comparisons against the minimum schedule, "
          f"advantage survives in {sum(1 for row in graphs['routing']['comparisons'] if row['survives'])}"
          f"/{routing['comparisons']}, surviving fraction "
          f"{routing['min_surviving_fraction']:.2f}-{routing['max_surviving_fraction']:.2f}x")
    measured = [row for row in ceiling["rows"] if row["status"] == "measured"]
    print(f"dense ceiling: {len(measured)} widths measured, "
          f"largest {ceiling['largest_measured_width']} qubits, "
          f"wall at {ceiling['wall']['width'] if ceiling['wall'] else 'none'} "
          f"({ceiling['wall']['status'] if ceiling['wall'] else 'no wall'})")
    print(f"referee agreement: {len(agreement['rows'])} circuits, "
          f"{agreement['accepted']} accepted and {agreement['refused']} refused, "
          f"all agreeing")
    for row in wide["rows"]:
        referee = row["second_referee"]
        print(f"referee agreement at {row['qubits']} qubits: stabilizer "
              f"{row['stabilizer']['verdict']}, {referee['referee']} "
              f"{referee['verdict']} ({referee['satisfied']}/{referee['generators']} "
              f"generators), dense tier REFUSED")
    print(f"referee agreement refusals: "
          f"{len(wide['refusals'])} wrong constructions, both referees INVALID on "
          f"{sum(1 for row in wide['refusals'] if row['agree'] is True)}")
    abstained = ("qubits " + ", ".join(str(entry["qubits"])
                                        for entry in third["abstentions"])
                 if third["abstentions"] else "none")
    print(f"third referee (mqt.qcec, unitaries): {third['decided']} decided, "
          f"{third['abstained']} abstained ({abstained}), "
          f"agreements {third['agreements']}/{third['decided']}")
    sweep = placement["window_sweep"]
    print(f"lookahead window sweep: {sweep['summary']['rows']} placements over "
          f"windows {sweep['windows']}, best-depth total "
          f"{sweep['summary']['best_depth_total']} against "
          f"{sweep['summary']['at_window_8_depth_total']} at window 8")
    print(f"written: {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
