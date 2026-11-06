#!/usr/bin/env python3
"""Compare APEDP-VQC's synthesis with an external tool on the same specification.

The review before this record was explicit that the work has not yet demonstrated an
advantage over existing tools using identical benchmarks. This script is that
benchmark, with one design rule that shapes every row:

    a tool that did not run is not a tool that lost.

Each row is either a *matched* measurement -- the same frozen specification handed to
both engines, both results decided by this package's own specification referee -- or an
explicit abstention naming the reason. Nothing in between.

What is compared
----------------
**Task 1/2, synthesis.** The same signed Pauli stabilizers go to Qiskit's
`synth_circuit_from_stabilizers` and to this package's construction grammar. Qiskit's
circuit comes back through QASM 2, is re-imported, and is *decided* by the referee
against the same specification, so its gate count is quoted only for a circuit that was
independently verified as preparing the requested state. Metrics: operations, two-qubit
gates, depth, wall clock.

**Task 3, the capability asymmetry.** A random 30-qubit stabilizer state: Qiskit has an
algorithm for it, this package has no construction family and must abstain. That row is
the honest complement to the GHZ rows, where a grammar that knows the answer is being
compared with a general algorithm that does not.

**Task 4, the topology.** The same three 100-qubit circuits placed on a line by this
package's router, so the comparison includes what each circuit costs on hardware rather
than only what it costs in the abstract.

**Task 5, the shared corpus.** A seeded *sample* from
:mod:`apedp_dyn.vqc.corpus`, none of it GHZ, ring or lattice: random connected 3- and
4-regular graph states and a seeded irregular graph, at five widths with five
independent draws per ``(family, width)`` cell, plus a random stabilizer state at each
of three widths. No construction in either engine is specialised to these instances --
the grammar reads the edge set and emits a fixed schedule of it, Qiskit runs its
general elimination -- which is what makes the row a comparison rather than the
favourable case.

The sample exists because the earlier corpus was seven hand-picked instances, and a
single draw per family cannot tell a family's behaviour from the draw's. Every family
is therefore reported as a *distribution*: median with quartiles and range over the
instances in the cell, the share of instances on each side, a width-by-width trend,
and a bootstrap interval on the median ratio that resamples these instances (and says
so, rather than implying a statement about graph space). On the graph states the two
engines produce the *same gate count* on every matched instance; the difference is in
the schedule and is reported with its spread. On the random stabilizer states the
grammar abstains, and that abstention is not counted as a win.

**Task 7, each tool's own router.** The same logical circuits are routed onto a
recorded device coupling map (a qiskit-ibm-runtime snapshot, not a synthetic lattice)
by Qiskit's transpiler, and by both of this package's routers on the same device
graph. Qiskit keeps its final layout rather than restoring it, so its circuit is
decided against the target relabelled by that layout; this package restores and
reports both the routed depth and the total, so the restore cost is not hidden.

**Task 6, sampling.** Stim overlaps with execution, not synthesis, so it is given the
task it actually does: hand both engines the same signed stabilizers and the same shot
budget, and decide each sample set against the **exact** outcome distribution, which is
enumerable at 12 qubits. The two samplers are also compared with each other under a
null in which both are correct. The throughput ratio is recorded as an implementation
comparison -- Python per-shot tableau clones against Stim's compiled C++ sampler -- not
as a ceiling, and the workload width is set by the referee, not by what either engine
can hold.

**Task 8, equivalence.** MQT QCEC overlaps with the equivalence referee, not with
synthesis or sampling, so it is handed the pairs that referee decides -- plus the pairs
it *cannot* decide, which is the point. The corpus is built so the two disagree, and so
that every disagreement has a named cause: a pair whose difference is smaller than one
threshold and larger than the other, a pair that is equal standalone but not under
control, a pair that prepares the same state from different unitaries, a Clifford pair
above the dense bound, and a non-Clifford pair above it that this package refuses and
the other tool decides. Each pair's relation is fixed by the construction that built
it (``X Y Z = i I``, ``RZ(2*pi) = -I``, ``U_c = RX(theta) U_r``), so neither tool is the
arbiter of the other. Near its own threshold QCEC's answer is not a function of the
input, and the record reports the criteria of repeated identical calls rather than one
draw; where one tool abstains the abstention is recorded on that side and never scored.

What is not compared, and why
-----------------------------
QCEC is compared on the equivalence task only: it is not a synthesizer and it does not
sample. Its verdicts are about the *standalone unitary* question, so a pair that this
package decides under control or as a prepared state is recorded as a scope difference
rather than as an agreement or a disagreement. The 1,000-qubit rows elsewhere in this
repository are evidence about this package's own scaling, not a state-of-the-art claim:
Stim's published 20,000-qubit workload is orders of magnitude larger than anything
measured here, and the sampling rows say nothing about Stim's ceiling.

The asymmetry of the two engines is stated in the record rather than smoothed over: a
grammar that selects among constructions it can already build is not a synthesizer that
finds one, and beating it on the target it was written for is not a general advantage.

Example::

    python scripts/compare_vqc_external_tools.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.vqc.circuit import Circuit  # noqa: E402
from apedp_dyn.vqc.corpus import (  # noqa: E402
    GRAPH_FAMILIES,
    SAMPLE_REPLICATES,
    SAMPLE_WIDTHS,
    STABILIZER_WIDTHS,
    CorpusTarget,
    random_stabilizer_target,
    seed_for,
    shared_target_corpus,
)
from apedp_dyn.vqc.equivalence_corpus import (  # noqa: E402
    EQUIVALENCE_TOLERANCE,
    SCOPES,
    adjudicate_pair,
    classify,
    equivalence_pairs,
)
from apedp_dyn.vqc.equivalence_corpus import (  # noqa: E402
    summarise as equivalence_summary,
)
from apedp_dyn.vqc.external import (  # noqa: E402
    ROUTING_BASIS,
    ToolUnavailable,
    adjudicate_sampling,
    adjudicate_submission,
    apedp_sampling,
    qcec_equivalence,
    qiskit_routing,
    qiskit_synthesis,
    relabelled_target,
    sampling_pair,
    stim_sampling,
    tool_statuses,
)
from apedp_dyn.vqc.foreign_referee import (  # noqa: E402
    FOREIGN_REFEREES,
    ForeignVerdict,
    stim_verdict,
)
from apedp_dyn.vqc.marginals import (  # noqa: E402
    adjudicate_marginal,
    marginal_pair,
)
from apedp_dyn.vqc.memory import (  # noqa: E402
    MemoryUnavailable,
    measure_engine,
)
from apedp_dyn.vqc.placement import Connectivity, line, place  # noqa: E402
from apedp_dyn.vqc.synthesis import (  # noqa: E402
    I,
    SynthesisError,
    TargetSpecification,
    Z,
    adjudicate_specification,
    build,
    cluster_state_target,
    entangling_depth_lower_bound,
    ghz_target,
    ring_graph_target,
    seeded_graph_target,
    specification_summary,
    synthesize,
)

RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_external_comparison.json"
SIZES = (10, 50, 100)
RANDOM_STATE_QUBITS = 30
RANDOM_STATE_SEED = 30
TOPOLOGY = "line"
#: Sampling is refereed against the exact distribution, which is enumerable up to 12
#: qubits; the shot budget is large enough that the 0.999 null band is tight relative
#: to the spread of a correct sampler, and small enough that the Python sampler runs.
SAMPLING_QUBITS = 12
SAMPLING_SHOTS = 40_000
#: Near its own numerical threshold QCEC's verdict is not a function of the input, so
#: the tolerance family is asked the identical question this many times and the record
#: carries every criterion rather than one draw. One repeat everywhere else: a pair
#: away from the boundary is stable, which the repeat that is taken there confirms.
EQUIVALENCE_REPEATS = 15
SAMPLING_TARGETS = ("ghz", "ring_graph", "random_stabilizer_12q")
#: Above the enumerable width the referee is the exact marginal on chosen wire
#: subsets, so the wide block can go past 12 qubits without weakening the question it
#: asks: every tested subset is decided against its exact distribution, and the family
#: verdict is corrected for the number of subsets tested. The shot budget is smaller
#: than the 12-qubit block's because the subsets are smaller than the full state: 1,024
#: cells over 20,000 shots is a tighter ratio than 4,096 over 40,000, so the wide rows
#: are decided by a *stricter* band than the enumerable ones.
WIDE_SAMPLING_WIDTHS = (16, 24, 32)
WIDE_SAMPLING_SHOTS = 20_000
WIDE_SUBSET_SIZE = 10
#: The wide block makes ~190 subset decisions per run, so its false-positive rate is a
#: budget for the *family*: each subset's chi-square p-value is multiplied by the number
#: of subsets tested before it is compared with this, and the product is recorded. A
#: per-subset band would also do, but a quantile of 1e-3 over 190 decisions is not
#: estimable from a few hundred simulated draws -- which is why this block tests
#: p-values instead of quoting a simulated band it cannot resolve.
WIDE_SAMPLING_ALPHA = 1e-3
#: The one target whose every Z-basis marginal is uniform: a graph state. It is in the
#: block because the marginal referee cannot tell one graph state from another, and a
#: comparison that only ran the informative case would be hiding its own blind spot.
WIDE_UNIFORM_TARGET = "ring_graph"
#: One seed and one optimisation level, recorded, because Qiskit's SABRE is
#: stochastic: a different seed is a different router run, not the same one.
ROUTING_SEED = 1234
ROUTING_OPTIMISATION = 1


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(function, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


def local_row(target: TargetSpecification, family: str, *, promoted: bool,
              construction: Any | None = None, seconds: float | None = None
              ) -> dict[str, Any]:
    """One construction from this package's grammar, decided by the same referee.

    A caller that already built the construction -- the corpus, which routes the same
    circuit afterwards -- hands it in rather than rebuilding it, so the row and the
    routed circuit cannot come from two different builds.
    """
    if construction is None or seconds is None:
        construction, seconds = timed(build, family, target)
    from apedp_dyn.vqc.synthesis import adjudicate_specification

    report = adjudicate_specification(construction.circuit, target)
    resources = construction.circuit.resources()
    return {
        "engine": "apedp.synthesis",
        "variant": f"grammar:{family}",
        "status": "compared",
        "promoted": promoted,
        "target_sha256": target.sha256(),
        "qubits": target.num_qubits,
        "metrics": {
            "operations": int(resources["operations"]),
            "two_qubit_gates": int(resources["two_qubit_gates"]),
            "depth": int(resources["depth"]),
            "seconds": seconds,
        },
        "verification": {
            "verdict": report.verdict,
            "tier": report.tier,
            "checks_failed": list(report.failed()),
            "verified_by": "apedp_dyn.vqc.synthesis (specification referee)",
        },
        "circuit_sha256": construction.circuit.sha256(),
        "note": construction.notes,
    }


def external_row(target: TargetSpecification
                 ) -> tuple[dict[str, Any], Circuit | None]:
    """Qiskit's synthesis of the same specification, decided by our referee.

    The circuit is returned beside the row so the third-party referee can be handed the
    same object our referee decided, rather than a rebuild that might differ.
    """
    try:
        submission, seconds = timed(qiskit_synthesis, target)
    except ToolUnavailable as error:
        return ({"engine": "qiskit", "status": "unavailable", "reason": str(error),
                 "target_sha256": target.sha256()}, None)
    row, decide_seconds = timed(adjudicate_submission, submission, target)
    payload = row.as_dict()
    payload["status"] = "compared"
    payload["metrics"]["seconds"] = seconds
    payload["metrics"]["verification_seconds"] = decide_seconds
    payload["metrics"]["synthesis_seconds"] = seconds
    return payload, submission.circuit


def abstention_row(target: TargetSpecification) -> dict[str, Any]:
    """What this package does with a target it has no construction family for."""
    try:
        outcome = synthesize(target)
    except SynthesisError as error:
        return {"engine": "apedp.synthesis", "variant": "grammar:any", "status": "abstained",
                "reason": str(error), "target_sha256": target.sha256(),
                "qubits": target.num_qubits}
    return {"engine": "apedp.synthesis", "variant": "grammar:any", "status": "compared",
            "promoted": outcome.promoted, "target_sha256": target.sha256(),
            "qubits": target.num_qubits,
            "metrics": {"constructions_valid": len(outcome.valid())}}


def synthesis_task(target: TargetSpecification, *, families: tuple[str, ...],
                   promoted_family: str) -> dict[str, Any]:
    rows = []
    for family in families:
        rows.append(local_row(target, family, promoted=family == promoted_family))
    foreign, _circuit = external_row(target)
    rows.append(foreign)
    return {
        "task": "synthesise a circuit from a stabilizer specification",
        "target": specification_summary(target),
        "target_sha256": target.sha256(),
        "rows": rows,
    }


def random_state_task() -> dict[str, Any]:
    target = random_stabilizer_target(RANDOM_STATE_QUBITS, seed=RANDOM_STATE_SEED)
    foreign, _circuit = external_row(target)
    return {
        "task": "synthesise a circuit from a stabilizer specification "
                "with no local construction family",
        "target": specification_summary(target),
        "target_sha256": target.sha256(),
        "rows": [abstention_row(target), foreign],
    }


CORPUS_METRICS = ("operations", "two_qubit_gates", "depth")
#: Resamples for the bootstrap interval on a median ratio. Deterministic: the seed is
#: fixed, so a rebuild of the record reproduces the interval exactly.
CORPUS_BOOTSTRAP_RESAMPLES = 2000
CORPUS_BOOTSTRAP_SEED = 20_260_101


def _spread(values: list[float]) -> dict[str, Any] | None:
    """The whole distribution of a per-instance quantity, not one number from it.

    A median without a spread is the thing this record was told not to report: on a
    sample of graph states, the interesting question is how far a single draw moves the
    answer, and min/q1/median/q3/max is the cheapest honest answer to it.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return {"n": 1, "min": ordered[0], "q1": ordered[0], "median": ordered[0],
                "q3": ordered[0], "max": ordered[0], "iqr": 0.0,
                "mean": ordered[0], "stdev": 0.0}
    q1, median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "n": len(ordered),
        "min": ordered[0],
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": ordered[-1],
        "iqr": q3 - q1,
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered),
    }


def _median_interval(values: list[float]) -> list[float] | None:
    """A bootstrap 95% interval for the median, over the sampled instances.

    This resamples *these* instances. It says how much another draw from the same
    seeded generator could have moved the median, and nothing about graph space; the
    record's limitations say so, because an interval that is read as a confidence
    statement about all graph states would be a fabricated claim.
    """
    if len(values) < 2:
        return None
    rng = random.Random(CORPUS_BOOTSTRAP_SEED)
    medians = []
    for _ in range(CORPUS_BOOTSTRAP_RESAMPLES):
        resample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        medians.append(statistics.median(resample))
    medians.sort()
    lower = medians[int(0.025 * len(medians))]
    upper = medians[min(len(medians) - 1, math.ceil(0.975 * len(medians)))]
    return [lower, upper]


def _metric_spread(matched: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """One metric's per-instance distribution for both engines, plus the paired ratio."""
    grammar = [int(row["grammar"]["metrics"][metric]) for row in matched]
    foreign = [int(row["qiskit"]["metrics"][metric]) for row in matched]
    ratios = [float(row["comparison"]["ratios_qiskit_over_grammar"][metric])
              for row in matched]
    wins = {engine: sum(1 for row in matched
                        if row["comparison"]["winners"][metric] == engine)
            for engine in ("grammar", "qiskit", "tie")}
    return {
        "grammar": _spread(grammar),
        "qiskit": _spread(foreign),
        "ratio_qiskit_over_grammar": _spread(ratios),
        "ratio_median_bootstrap_95": _median_interval(ratios),
        "wins": wins,
        "share_instances_grammar_lower": (wins["grammar"] / len(matched)
                                          if matched else None),
        "paired_difference_qiskit_minus_grammar": _spread(
            [other - own for own, other in zip(grammar, foreign, strict=True)]),
    }


def _family_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One family's rows as a distribution: counts, per-metric spread, width trend."""
    matched = [row for row in rows if row["comparison"] is not None]
    abstained = [row["label"] for row in rows if row["grammar"]["status"] != "compared"]
    widths = sorted({int(row["width"]) for row in rows})
    by_width: dict[str, Any] = {}
    for width in widths:
        cell = [row for row in matched if int(row["width"]) == width]
        if not cell:
            continue
        by_width[f"{width}q"] = {
            "matched": len(cell),
            "median_depth": {
                "grammar": statistics.median(row["grammar"]["metrics"]["depth"]
                                             for row in cell),
                "qiskit": statistics.median(row["qiskit"]["metrics"]["depth"]
                                            for row in cell),
            },
            "depth_ratio_qiskit_over_grammar": _spread(
                [float(row["comparison"]["ratios_qiskit_over_grammar"]["depth"])
                 for row in cell]),
            "depth_wins": {engine: sum(1 for row in cell
                                       if row["comparison"]["winners"]["depth"]
                                       == engine)
                           for engine in ("grammar", "qiskit", "tie")},
        }
    block: dict[str, Any] = {
        "targets": len(rows),
        "matched": len(matched),
        "abstained": abstained,
        "widths": widths,
        "replicates_per_width": sorted(
            {len([row for row in rows if int(row["width"]) == width]) for width in widths}),
        "per_metric": {metric: _metric_spread(matched, metric)
                       for metric in CORPUS_METRICS},
        "by_width": by_width,
    }
    graph_rows = [row for row in rows if row["kind"] != "random_stabilizer"]
    if graph_rows:
        block["structure"] = {
            "edges": _spread([float(row["edges"]) for row in graph_rows]),
            "mean_degree": _spread([float(row["mean_degree"]) for row in graph_rows]),
        }
    return block


def _winner(grammar: int, foreign: int) -> str:
    if grammar < foreign:
        return "grammar"
    if foreign < grammar:
        return "qiskit"
    return "tie"


def corpus_grammar_row(entry: CorpusTarget) -> tuple[dict[str, Any], Circuit | None]:
    """The grammar's answer to one corpus instance, or the abstention it makes.

    The construction is returned beside the row so the routed task can hand the very
    circuit whose synthesis the row records to each engine's router. Rebuilding it
    from the family name would be a second synthesis that happens to agree, and a
    comparison that routes a circuit it did not measure is not the same comparison.
    """
    try:
        outcome = synthesize(entry.target)
    except SynthesisError as error:
        return ({"engine": "apedp.synthesis", "status": "abstained",
                 "reason": str(error), "target_sha256": entry.target.sha256()}, None)
    if not outcome.ranked:
        return ({"engine": "apedp.synthesis", "status": "refused",
                 "reason": outcome.refusal, "target_sha256": entry.target.sha256()},
                None)
    construction, seconds = timed(build, outcome.ranked[0], entry.target)
    row = local_row(entry.target, outcome.ranked[0],
                    promoted=outcome.promoted is not None,
                    construction=construction, seconds=seconds)
    row["families_valid"] = [r["family"] for r in outcome.valid()]
    row["pareto_frontier"] = list(outcome.pareto)
    return row, construction.circuit


def corpus_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The sample in numbers, recomputed from the rows so prose cannot drift.

    Every headline is a distribution over the sampled instances: a median with its
    quartiles and range, the share of instances on one side, and a width-by-width
    breakdown. A single row per family would let the draw that flattered an engine
    stand in for the family, which is exactly what the review of the seven-instance
    corpus objected to.
    """
    matched = [row for row in rows if row["comparison"] is not None]
    families = sorted({str(row["family"]) for row in rows})
    grammar_total = (sum(int(row["grammar"]["metrics"]["two_qubit_gates"])
                         for row in matched) if matched else None)
    foreign_total = (sum(int(row["qiskit"]["metrics"]["two_qubit_gates"])
                         for row in matched) if matched else None)
    return {
        "targets": len(rows),
        "matched": len(matched),
        "grammar_abstained": [row["label"] for row in rows
                              if row["grammar"]["status"] != "compared"],
        "qiskit_unavailable": [row["label"] for row in rows
                               if row["qiskit"]["status"] != "compared"],
        "families": families,
        "overall": _family_block(rows),
        "by_family": {family: _family_block([row for row in rows
                                             if row["family"] == family])
                      for family in families},
        "two_qubit_gates_agree_on_every_matched_instance": bool(matched) and all(
            row["grammar"]["metrics"]["two_qubit_gates"]
            == row["qiskit"]["metrics"]["two_qubit_gates"] for row in matched),
        "grammar_total_two_qubit_gates": grammar_total,
        "qiskit_total_two_qubit_gates": foreign_total,
    }


#: Experiment E's five specifications, re-derived here so the corpus histogram can be
#: read against the targets `docs/VQC_SYNTHESIS.md` §5.3 reports. The depth this block
#: quotes is the minimum schedule's (`graph_layers`), because that is the row §5.3's
#: table shows; the other four schedules of each target are that record's to report.
#: ``(description, qubits, factory)``: the width is declared beside the factory so a
#: reader (and a test) can tell how big a target is without building the specification,
#: which at 900 qubits is itself tens of megabytes of stabilizer text.
EXPERIMENT_E_CASES: tuple[tuple[str, int, Callable[[], TargetSpecification]], ...] = (
    ("cluster_4x4", 16, lambda: cluster_state_target(4, 4)),
    ("cluster_10x10", 100, lambda: cluster_state_target(10, 10)),
    ("cluster_20x20", 400, lambda: cluster_state_target(20, 20)),
    ("cluster_30x30", 900, lambda: cluster_state_target(30, 30)),
    ("irregular_25q_seed7", 25,
     lambda: seeded_graph_target(25, seed=7, extra_edge_probability=0.15)),
    ("irregular_64q_seed23", 64,
     lambda: seeded_graph_target(64, seed=23, extra_edge_probability=0.10)),
)
#: The histogram's bucket width. Ten is the coarsest stride that still separates the
#: corpus's own modes (10-19 holds 43 of 78 rows) while keeping the one 126-qubit row
#: from stretching a single bin over half the axis.
DEPTH_HISTOGRAM_BUCKET = 10


def qiskit_depth_context(corpus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where Experiment E's depths fall in the Qiskit corpus's depth histogram.

    The corpus rows already carry Qiskit's own circuit for the same specification, so
    their depth is a measurement this record has, not one it borrows. This block puts
    the five Experiment E depths beside that distribution *at the width each target
    is*, and says which of the five can be read that way: the corpus runs at 12, 20,
    32, 48 and 64 qubits, so only the 64-qubit irregular target has a cell at its own
    width. The other four are placed against the nearest cell and labelled by the gap,
    because a depth at 400 qubits cannot be compared with one at 12.

    Two things this is not. The corpus engine is `synth_circuit_from_stabilizers`,
    whose job is to build a circuit for an arbitrary stabilizer state and not to
    minimise depth, so a corpus depth is one engine's answer rather than a floor. And
    an Experiment E depth is the depth of a *named graph's* minimum schedule, fixed by
    the edge set (``1 + chi'``), while a corpus row is a random instance of a family
    no construction is specialised to. So the block reports a position in a
    distribution, and the caveats travel with it.
    """
    observed = [row for row in corpus_rows
                if row["qiskit"].get("status") == "compared"
                and "depth" in row["qiskit"].get("metrics", {})]
    if not observed:
        return {
            "status": "unavailable",
            "reason": next((str(row["qiskit"].get("reason")) for row in corpus_rows
                            if row["qiskit"].get("reason")), "qiskit produced no rows"),
            "targets": [],
            "buckets": [],
            "by_width": {},
            "overall": None,
            "like_for_like": [],
        }

    depths = [int(row["qiskit"]["metrics"]["depth"]) for row in observed]
    cells: dict[int, list[int]] = {}
    for row in observed:
        cells.setdefault(int(row["width"]), []).append(
            int(row["qiskit"]["metrics"]["depth"]))
    cells = {width: sorted(values) for width, values in sorted(cells.items())}

    low = (min(depths) // DEPTH_HISTOGRAM_BUCKET) * DEPTH_HISTOGRAM_BUCKET
    high = (max(depths) // DEPTH_HISTOGRAM_BUCKET) * DEPTH_HISTOGRAM_BUCKET
    buckets = [{
        "low": start,
        "high": start + DEPTH_HISTOGRAM_BUCKET - 1,
        "count": sum(1 for depth in depths if start <= depth <= start
                     + DEPTH_HISTOGRAM_BUCKET - 1),
    } for start in range(low, high + DEPTH_HISTOGRAM_BUCKET, DEPTH_HISTOGRAM_BUCKET)]

    targets: list[dict[str, Any]] = []
    for description, declared_qubits, make_target in EXPERIMENT_E_CASES:
        target = make_target()
        if target.num_qubits != declared_qubits:  # pragma: no cover - a typo guard
            raise SystemExit(f"{description} declares {declared_qubits} qubits and "
                             f"builds {target.num_qubits}")
        construction = build("graph_layers", target)
        depth = construction.circuit.depth()
        shape = entangling_depth_lower_bound(target)
        exact = target.num_qubits in cells
        cell_width = (target.num_qubits if exact else
                      min(cells, key=lambda width: (abs(width - target.num_qubits),
                                                    width)))
        values = cells[cell_width]
        at_or_below = sum(1 for value in values if value <= depth)
        targets.append({
            "target": description,
            "qubits": target.num_qubits,
            "edges": shape["edges"],
            "max_degree": shape["max_degree"],
            "cz_layers": shape["cz_layers_floor"],
            "depth": depth,
            "circuit_sha256": construction.circuit.sha256(),
            "verdict": adjudicate_specification(construction.circuit, target).verdict,
            "placement": {
                "cell_qubits": cell_width,
                "width_match": "exact" if exact else "nearest",
                "cell_width_over_target": (None if exact
                                           else cell_width / target.num_qubits),
                "n": len(values),
                "min": values[0],
                "median": statistics.median(values),
                "max": values[-1],
                "qiskit_depths_at_or_below": at_or_below,
                "share_at_or_below": at_or_below / len(values),
                "median_over_target": statistics.median(values) / max(1, depth),
            },
        })

    return {
        "status": "measured",
        "engine": "qiskit.synthesis.synth_circuit_from_stabilizers",
        "measured_by": "this package's specification referee for the depth's circuit",
        "corpus": {
            "module": "apedp_dyn.vqc.corpus",
            "targets": len(observed),
            "widths": sorted(cells),
            "families": sorted({str(row["family"]) for row in observed}),
        },
        "bucket_width": DEPTH_HISTOGRAM_BUCKET,
        "buckets": buckets,
        "overall": _spread(depths),
        "by_width": {str(width): {**_spread(values), "depths": values}
                     for width, values in cells.items()},
        "targets": targets,
        "like_for_like": [row["target"] for row in targets
                          if row["placement"]["width_match"] == "exact"],
        "caveats": [
            "the corpus engine builds any stabilizer state; it is not asked to minimise "
            "depth, so a corpus depth is one engine's answer rather than a floor",
            "an Experiment E depth is fixed by a named graph's edge set (1 + chi'), a "
            "corpus row is a random instance of a family no construction is specialised "
            "to: the histogram is context, not a like-for-like contest",
            "only the targets whose width the corpus also runs are compared directly; "
            "the rest name the nearest cell and the gap, because depth does not scale "
            "with width in a way that makes 400 qubits and 12 comparable",
        ],
    }


def third_party_row(ours: ForeignVerdict, theirs: ForeignVerdict,
                    our_referee: dict[str, str], *,
                    ours_circuit: Circuit, theirs_circuit: Circuit) -> dict[str, Any]:
    """One instance's two circuits, decided by the third-party referee.

    Agreement is only asserted where both referees reached a verdict: a Stim that
    abstained (absent, or a circuit outside a tableau simulator) is *undecided*, not a
    disagreement, and the record must not turn one into the other.
    """
    def agrees(foreign: str, ours_verdict: str) -> bool | None:
        if foreign not in ("VALID", "INVALID"):
            return None
        return (foreign == "VALID") == (ours_verdict == "VALID")

    ours_payload, theirs_payload = ours.as_dict(), theirs.as_dict()
    # The hash makes the second opinion citable: a reader can check that Stim decided
    # the circuit the synthesis row recorded, not a rebuild that happens to agree.
    ours_payload["circuit_sha256"] = ours_circuit.sha256()
    theirs_payload["circuit_sha256"] = theirs_circuit.sha256()
    return {
        "apedp": ours_payload,
        "qiskit": theirs_payload,
        "both_valid": ours.verdict == "VALID" and theirs.verdict == "VALID",
        "agrees_with_our_referee": {
            "apedp": agrees(ours.verdict, our_referee["apedp"]),
            "qiskit": agrees(theirs.verdict, our_referee["qiskit"]),
        },
    }


def third_party_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How the third-party referee decided a set of rows, recomputed from them."""
    blocks = [(row["label"], row["third_party"]) for row in rows
              if row.get("third_party")]
    if not blocks:
        return None
    verdicts = [block["apedp"]["verdict"] for _label, block in blocks]
    verdicts += [block["qiskit"]["verdict"] for _label, block in blocks]
    decided = ("VALID", "INVALID")
    versions = sorted({block[side]["version"] for _label, block in blocks
                       for side in ("apedp", "qiskit") if block[side]["version"]})
    return {
        "referee": "stim",
        "version": versions[0] if versions else None,
        "instances": len(blocks),
        "decisions": len(verdicts),
        "decided": sum(1 for verdict in verdicts if verdict in decided),
        "abstained": sum(1 for verdict in verdicts if verdict not in decided),
        "valid": sum(1 for verdict in verdicts if verdict == "VALID"),
        "invalid": sum(1 for verdict in verdicts if verdict == "INVALID"),
        "apedp_valid": sum(1 for _label, block in blocks
                           if block["apedp"]["verdict"] == "VALID"),
        "qiskit_valid": sum(1 for _label, block in blocks
                            if block["qiskit"]["verdict"] == "VALID"),
        "both_valid": sum(1 for _label, block in blocks if block["both_valid"]),
        "undecided": [label for label, block in blocks
                      if block["apedp"]["verdict"] not in decided
                      or block["qiskit"]["verdict"] not in decided],
        "not_both_valid": [label for label, block in blocks if not block["both_valid"]],
        "disagreements_with_our_referee": [
            label for label, block in blocks
            if False in block["agrees_with_our_referee"].values()],
    }


def shared_corpus_task() -> dict[str, Any]:
    """Both engines on the sampled corpus in :mod:`apedp_dyn.vqc.corpus`.

    The corpus is rebuilt here from its seeds rather than imported as data, and the
    check below compares the two builds: a corpus that is not reproducible from its
    seeds is not citable, and the record would rather refuse to write. The sample
    design is written down next to the rows, because a spread over a sample is only
    interpretable if the cells it was drawn from are named.
    """
    rows: list[dict[str, Any]] = []
    constructions: list[tuple[CorpusTarget, dict[str, Any], Circuit]] = []
    for entry in shared_target_corpus():
        grammar, circuit = corpus_grammar_row(entry)
        foreign, foreign_circuit = external_row(entry.target)
        row: dict[str, Any] = {**entry.as_dict(), "grammar": grammar, "qiskit": foreign,
                               "comparison": None}
        if grammar["status"] == "compared" and foreign["status"] == "compared":
            own, other = grammar["metrics"], foreign["metrics"]
            row["comparison"] = {
                "winners": {metric: _winner(own[metric], other[metric])
                            for metric in CORPUS_METRICS},
                "ratios_qiskit_over_grammar": {
                    metric: other[metric] / max(1, own[metric])
                    for metric in CORPUS_METRICS},
                "agrees_on_state": (grammar["verification"]["verdict"] == "VALID"
                                    and foreign["verification"]["verdict"] == "VALID"),
            }
        if circuit is not None:
            # The circuit routed below is the one this row measured, not a rebuild.
            constructions.append((entry, row, circuit))
        if (circuit is not None and foreign_circuit is not None
                and grammar["status"] == "compared"
                and foreign["status"] == "compared"):
            row["third_party"] = third_party_row(
                stim_verdict(circuit, entry.target),
                stim_verdict(foreign_circuit, entry.target),
                {"apedp": grammar["verification"]["verdict"],
                 "qiskit": foreign["verification"]["verdict"]},
                ours_circuit=circuit, theirs_circuit=foreign_circuit)
        rows.append(row)
    routing = corpus_routing_task(constructions)
    return {
        "task": ("synthesise a circuit for targets no construction in either engine "
                 "was written for, on a sample of graph states rather than a set of "
                 "chosen instances"),
        "design": {
            "sampler": ("every graph family at every width in widths, with "
                         "replicates independent seeded draws per cell"),
            "widths": list(SAMPLE_WIDTHS),
            "replicates_per_cell": SAMPLE_REPLICATES,
            "graph_families": [{"family": family, "degree": degree}
                               for family, degree in GRAPH_FAMILIES],
            "abstention_anchors": {
                "family": "random_stabilizer",
                "widths": list(STABILIZER_WIDTHS),
                "replicates": 1,
            },
            "expected_targets": (len(GRAPH_FAMILIES) * len(SAMPLE_WIDTHS)
                                 * SAMPLE_REPLICATES + len(STABILIZER_WIDTHS)),
            "seed_rule": ("seed = SEED_BASE[family] + 10 * width + replicate, so a "
                          "row's label names the cell that produced it"),
            "what_a_spread_here_means": (
                "the per-instance distribution over these draws. The widths are fixed "
                "tiers and the instances come from one generator, so this is not a "
                "sample of graph space and the bootstrap interval describes these "
                "instances only"),
        },
        "corpus": {
            "module": "apedp_dyn.vqc.corpus",
            "families": {
                "regular_graph": ("a random connected d-regular graph state from a "
                                  "pairing-model draw; no lattice, ring or tree"),
                "random_graph": ("seeded random spanning tree plus extra edges: an "
                                 "irregular instance, not a regular family"),
                "random_stabilizer": ("a seeded Clifford state with no graph "
                                      "structure, where the grammar must abstain"),
            },
            "reproducible_from_seeds": True,
        },
        "rows": rows,
        "summary": corpus_summary(rows),
        "routing": routing,
        "third_party": {
            "referee": "stim.TableauSimulator",
            "module": "apedp_dyn.vqc.foreign_referee",
            "what_it_decides": FOREIGN_REFEREES["stim"]["decides"],
            "what_it_does_not_decide": FOREIGN_REFEREES["stim"]["does_not_decide"],
            "synthesis": third_party_summary(rows),
            "routed": third_party_summary(routing.get("rows") or []),
            "status": "compared" if any(row.get("third_party") for row in rows) else
                      "unavailable",
        },
    }


#: Each engine's own router for the corpus sample. Ours is the lookahead router (the
#: one this package would use); Qiskit's is SabreSwap inside `transpile`, with its own
#: layout search. Using one router per engine is the point: routing both circuits with
#: this package's router measures this package's router.
CORPUS_ROUTER_OURS = "apedp.placement:lookahead"
CORPUS_ROUTER_QISKIT = (
    "qiskit.transpile (SabreSwap, its own initial layout search, layout kept)")
CORPUS_ROUTER_STRATEGY = "lookahead"


def corpus_routing_task(constructions: list[tuple[CorpusTarget, dict[str, Any], Circuit]]
                        ) -> dict[str, Any]:
    """The corpus sample routed by each engine's *own* router on one device.

    The synthesis rows say what each engine's circuit costs in the abstract. This task
    asks the question the abstract numbers cannot: once the device's connectivity is
    paid for, which circuit is shallower, with each engine using the router it ships?
    Qiskit's circuit goes through its transpiler on the recorded coupling map and keeps
    its final layout; this package's circuit goes through the lookahead router on the
    same graph and is restored to its logical wires. Both are decided by this
    package's referee -- Qiskit's against the target relabelled by its layout, this
    package's against the target padded with idle wires -- so a routed row whose
    circuit does not prepare the state is not counted, however shallow it is.

    Only instances the grammar could build are routed (the abstentions have no local
    circuit to route), and the totals are reported as a spread over the sample, per
    family and per width, for the same reason the synthesis rows are.
    """
    try:
        snapshot = _device_snapshot()
    except ToolUnavailable as error:
        return {"status": "unavailable", "reason": str(error), "rows": [],
                "summary": None, "router_ours": CORPUS_ROUTER_OURS,
                "router_qiskit": CORPUS_ROUTER_QISKIT}

    device = snapshot["device"]
    size = snapshot["size"]
    edges = snapshot["edges"]
    connectivity = snapshot["connectivity"]

    rows: list[dict[str, Any]] = []
    for entry, synthesis_row, circuit in constructions:
        placement, place_seconds = timed(place, circuit, connectivity,
                                         strategy=CORPUS_ROUTER_STRATEGY)
        padded = _padded_target(entry.target, size)
        own_report = adjudicate_specification(placement.circuit, padded)
        ours = {
            "router": CORPUS_ROUTER_OURS,
            "target_sha256": padded.sha256(),
            "depth_before_restore": placement.depth_before_restore,
            "mapped_depth": placement.mapped_depth,
            "routing_swaps": placement.swap_gates,
            "restore_swap_gates": placement.restore_swap_gates,
            "two_qubit_gates": placement.two_qubit_gates,
            "native_entanglers": placement.native_entanglers,
            "seconds": place_seconds,
            "verification": {"verdict": own_report.verdict, "tier": own_report.tier,
                             "checks_failed": list(own_report.failed())},
        }
        try:
            foreign = qiskit_routing(circuit, target=entry.target, num_qubits=size,
                                     edges=edges, device=device["name"],
                                     device_version=device["backend_version"],
                                     seed=ROUTING_SEED,
                                     optimization_level=ROUTING_OPTIMISATION)
        except ToolUnavailable as error:
            rows.append({
                "label": entry.label, "family": entry.family, "width": entry.width,
                "logical_target_sha256": entry.target.sha256(),
                "logical_circuit_sha256": synthesis_row["grammar"]["circuit_sha256"],
                "logical_depth": ours["depth_before_restore"],
                "ours": ours,
                "qiskit": {"status": "unavailable", "reason": str(error)},
                "comparison": None,
            })
            continue
        foreign_target = relabelled_target(entry.target, foreign.final_layout, size)
        foreign_report = adjudicate_specification(foreign.circuit, foreign_target)
        foreign_payload = foreign.as_dict()
        foreign_payload["logical_target_sha256"] = entry.target.sha256()
        foreign_payload["verification"] = {
            "verdict": foreign_report.verdict,
            "tier": foreign_report.tier,
            "checks_failed": list(foreign_report.failed()),
            "decided_against": ("the target relabelled by final_layout; Qiskit keeps "
                                "its layout rather than restoring it"),
            "decided_target_sha256": foreign_target.sha256(),
        }
        third_party = third_party_row(
            stim_verdict(placement.circuit, padded),
            stim_verdict(foreign.circuit, foreign_target),
            {"apedp": own_report.verdict, "qiskit": foreign_report.verdict},
            ours_circuit=placement.circuit, theirs_circuit=foreign.circuit)
        shallower = ours["depth_before_restore"]
        comparison = {
            "routed_depth_winner": (
                "apedp" if shallower < foreign.routed_depth
                else "qiskit" if foreign.routed_depth < shallower else "tie"),
            "depth_ratio_qiskit_over_apedp": (
                foreign.routed_depth / max(1, shallower)),
            "total_depth_ratio_qiskit_over_apedp_total": (
                foreign.routed_depth / max(1, ours["mapped_depth"])),
            "reported_depth_agrees": foreign.reported_depth == foreign.routed_depth,
            "agrees_on_state": (own_report.verdict == "VALID"
                                and foreign_report.verdict == "VALID"),
        }
        rows.append({
            "label": entry.label, "family": entry.family, "width": entry.width,
            "logical_target_sha256": entry.target.sha256(),
            "logical_circuit_sha256": synthesis_row["grammar"]["circuit_sha256"],
            "logical_depth": int(circuit.resources()["depth"]),
            "ours": ours,
            "qiskit": foreign_payload,
            "comparison": comparison,
            "third_party": third_party,
        })

    return {
        "status": "compared",
        "task": ("route the corpus sample onto a recorded device coupling map with "
                 "each engine's own router"),
        "device": device,
        "seed_transpiler": ROUTING_SEED,
        "optimization_level": ROUTING_OPTIMISATION,
        "basis_gates": list(ROUTING_BASIS),
        "router_ours": CORPUS_ROUTER_OURS,
        "router_qiskit": CORPUS_ROUTER_QISKIT,
        "rows": rows,
        "summary": corpus_routing_summary(rows),
    }


def corpus_routing_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The routed sample as a distribution, recomputed from its rows."""
    if not rows:
        return None
    matched = [row for row in rows if row["comparison"] is not None]

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        ratios = [row["comparison"]["depth_ratio_qiskit_over_apedp"] for row in subset]
        wins = {engine: sum(1 for row in subset
                            if row["comparison"]["routed_depth_winner"] == engine)
                for engine in ("apedp", "qiskit", "tie")}
        return {
            "matched": len(subset),
            "wins": wins,
            "share_instances_qiskit_shallower": (wins["qiskit"] / len(subset)
                                                 if subset else None),
            "depth_ratio_qiskit_over_apedp": _spread(ratios),
            "ratio_median_bootstrap_95": _median_interval(ratios),
            "apedp_routed_depth": _spread([row["ours"]["depth_before_restore"]
                                           for row in subset]),
            "qiskit_routed_depth": _spread([row["qiskit"]["routed_depth"]
                                            for row in subset]),
            "apedp_swaps": _spread([row["ours"]["routing_swaps"] for row in subset]),
            "qiskit_swaps": _spread([row["qiskit"]["routing_swaps"] for row in subset]),
        }

    families = sorted({str(row["family"]) for row in rows})
    widths = sorted({int(row["width"]) for row in rows})
    ratios = [row["comparison"]["depth_ratio_qiskit_over_apedp"] for row in matched]
    wins = {engine: sum(1 for row in matched
                        if row["comparison"]["routed_depth_winner"] == engine)
            for engine in ("apedp", "qiskit", "tie")}
    return {
        "instances": len(rows),
        "matched": len(matched),
        "wins": wins,
        "share_instances_qiskit_shallower": (wins["qiskit"] / len(matched)
                                             if matched else None),
        "depth_ratio_qiskit_over_apedp": _spread(ratios),
        "ratio_median_bootstrap_95": _median_interval(ratios),
        "by_family": {family: block([row for row in matched
                                      if row["family"] == family])
                      for family in families},
        "by_width": {f"{width}q": block([row for row in matched
                                          if int(row["width"]) == width])
                     for width in widths},
    }


def sampling_targets() -> tuple[tuple[str, TargetSpecification], ...]:
    """The three targets sampled by both engines, at the width the referee can decide."""
    corpus_state = next(entry.target for entry in shared_target_corpus()
                        if entry.label == "random_stabilizer_12q_0")
    return (
        ("ghz 12q", ghz_target(SAMPLING_QUBITS)),
        ("ring_graph 12q", ring_graph_target(SAMPLING_QUBITS)),
        ("random_stabilizer_12q", corpus_state),
    )


def sampling_memory(target: TargetSpecification, *, shots: int,
                    seeds: Mapping[str, int]) -> dict[str, Any]:
    """Peak memory of both engines on the workload the row already sampled.

    Each engine is measured in its own child interpreter with the peak counter reset
    after its imports, so the number is the sampling step's own resident growth rather
    than numpy's footprint. The child reports the count hash of the samples it drew and
    the record's checks require it to equal the hash of the recorded run, so the memory
    belongs to the workload in the record and not to a similar one. A platform that
    cannot reset the counter abstains with a reason instead of quoting the whole child.
    """
    memory: dict[str, Any] = {}
    for engine in ("grammar", "stim"):
        try:
            measurement = measure_engine(engine, target, shots=shots,
                                         seed=seeds[engine])
        except MemoryUnavailable as error:
            memory[engine] = {"status": "unavailable", "reason": str(error)}
        else:
            memory[engine] = measurement.as_dict()
    return memory


def sampling_task() -> dict[str, Any]:
    """The same specification, the same shots, sampled by both engines and decided.

    Each engine is handed the target's signed stabilizers only. The local sampler
    clones the tableau and measures it shot by shot; Stim builds its own tableau,
    synthesizes a preparation circuit and compiles a measurement sampler. Both sample
    sets are then decided against the exact distribution -- which is the referee, and
    is enumerable at this width -- and against each other under a null in which both
    samplers are correct. A missing Stim is an abstention with a reason, exactly as in
    the synthesis tasks.
    """
    rows: list[dict[str, Any]] = []
    for index, (label, target) in enumerate(sampling_targets()):
        own = apedp_sampling(target, shots=SAMPLING_SHOTS, seed=7001 + index)
        row: dict[str, Any] = {
            "label": label,
            "target": specification_summary(target),
            "target_sha256": target.sha256(),
            "qubits": target.num_qubits,
            "shots": SAMPLING_SHOTS,
            "grammar": adjudicate_sampling(own, target),
            "stim": None,
            "comparison": None,
            "memory": sampling_memory(target, shots=SAMPLING_SHOTS,
                                      seeds={"grammar": 7001 + index,
                                             "stim": 8001 + index}),
        }
        try:
            foreign = stim_sampling(target, shots=SAMPLING_SHOTS, seed=8001 + index)
        except ToolUnavailable as error:
            row["stim"] = {"status": "unavailable", "reason": str(error)}
            row["memory"]["stim"] = {"status": "unavailable", "reason": str(error)}
        else:
            row["stim"] = adjudicate_sampling(foreign, target)
            row["comparison"] = sampling_pair(own, foreign, target)
        rows.append(row)
    return {
        "task": "sample Z-basis outcomes from a stabilizer specification",
        "shots_per_run": SAMPLING_SHOTS,
        "referee": (
            "the exact outcome distribution of the specification, enumerated from its "
            "stabilizers at 12 qubits; a sample set is CONSISTENT when its "
            "total-variation distance is inside the 0.999 band of simulated correct "
            "samplers at the same shot count"),
        "two_sample_null": (
            "two independent correct samplers at the same shot count, so the pair is "
            "only reported as DISAGREE beyond sampling noise"),
        "memory_rule": (
            "each engine's sampling runs in a fresh child interpreter whose peak-RSS "
            "counter is reset after the engine's imports and the specification's "
            "rebuild, so the reported growth is the sampling step's own resident growth "
            "and the floor beside it is what the engine costs before sampling; the "
            "child's count hash must equal the recorded run's, so the memory belongs to "
            "these samples"),
        "rows": rows,
        "summary": sampling_summary(rows),
    }


def memory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median resident growth per engine, and every row that could not be measured."""
    growth: dict[str, float | None] = {}
    floors: dict[str, float | None] = {}
    for engine in ("grammar", "stim"):
        measurements = [row["memory"][engine] for row in rows
                        if "growth_mib" in row.get("memory", {}).get(engine, {})]
        growth[engine] = (statistics.median([m["growth_mib"] for m in measurements])
                          if measurements else None)
        floors[engine] = (statistics.median([m["floor_mib"] for m in measurements])
                          if measurements else None)
    return {
        "median_growth_mib": growth,
        "median_floor_mib": floors,
        "measured": sum(
            1 for row in rows for engine in ("grammar", "stim")
            if "growth_mib" in row.get("memory", {}).get(engine, {})),
        "unavailable": [
            {"label": row["label"], "engine": engine,
             "reason": row["memory"][engine]["reason"]}
            for row in rows for engine in ("grammar", "stim")
            if row.get("memory", {}).get(engine, {}).get("status") == "unavailable"],
    }


def sampling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The sampling block's numbers, recomputed from the rows that produced them."""
    medians = {}
    for engine in ("grammar", "stim"):
        rates = [row[engine]["samples_per_second"] for row in rows
                 if row[engine] is not None
                 and "samples_per_second" in row[engine]]
        medians[engine] = statistics.median(rates) if rates else None
    return {
        "memory": memory_summary(rows),
        "targets": len(rows),
        "matched": sum(1 for row in rows if row["comparison"] is not None),
        "stim_unavailable": [row["label"] for row in rows
                             if row["stim"].get("status") == "unavailable"],
        "grammar_consistent": sum(
            1 for row in rows if row["grammar"]["verdict"] == "CONSISTENT"),
        "stim_consistent": sum(
            1 for row in rows if row["stim"] is not None
            and row["stim"].get("verdict") == "CONSISTENT"),
        "pairs_agreeing": sum(
            1 for row in rows if row["comparison"] is not None
            and row["comparison"]["verdict"] == "AGREE"),
        "median_samples_per_second": medians,
    }


def wide_sampling_targets() -> tuple[tuple[str, TargetSpecification], ...]:
    """The targets sampled above the enumerable width, with their witnesses.

    GHZ is the informative case -- every 10-wire marginal has two supported outcomes --
    and a random stabilizer state is the shape that exists at any width. The ring graph
    is the referee's blind spot: a graph state is uniform in the Z basis, so every one
    of its marginals is uniform and no Z-basis marginal separates it from any other
    graph state. It is recorded as what it is instead of being left out.
    """
    targets: list[tuple[str, TargetSpecification]] = []
    for width in WIDE_SAMPLING_WIDTHS:
        targets.append((f"ghz {width}q", ghz_target(width)))
        targets.append((f"random_stabilizer {width}q", random_stabilizer_target(
            width, seed=seed_for("random_stabilizer", width, 0))))
    widest = max(WIDE_SAMPLING_WIDTHS)
    targets.append((f"{WIDE_UNIFORM_TARGET} {widest}q", ring_graph_target(widest)))
    return tuple(targets)


def wide_subsets(num_qubits: int) -> tuple[tuple[int, ...], ...]:
    """Every adjacent pair of wires, plus three 10-wire projections of the state.

    A pair is the smallest subset that carries a correlation, and covering the whole
    chain catches a wire that is wrong anywhere on the device rather than only inside a
    block; the 10-wire projections catch a wrong joint distribution over a region. A
    contiguous block, a strided block and a seeded draw are three different ways to be
    wrong, and the seeded one is recorded so the choice is reproducible rather than the
    draw that passed.

    Single-wire marginals are deliberately *not* used: from a GHZ or graph-state
    specification they are uniform by construction, so they would spend most of the
    family budget on subsets the state itself makes uninformative.
    """
    pairs = [(wire, wire + 1) for wire in range(num_qubits - 1)]
    # A step of one would make the strided draw the contiguous block again, and two
    # identical subsets would double-count in the Bonferroni correction.
    stride = max(2, num_qubits // WIDE_SUBSET_SIZE)
    wide = [
        tuple(range(WIDE_SUBSET_SIZE)),
        tuple(sorted(range(0, num_qubits, stride)))[:WIDE_SUBSET_SIZE],
        tuple(sorted(random.Random(90_000 + num_qubits).sample(
            range(num_qubits), WIDE_SUBSET_SIZE))),
    ]
    subsets = tuple(pairs + wide)
    if len(set(subsets)) != len(subsets):
        raise SystemExit(f"the wide subset design repeats a subset at {num_qubits} qubits")
    return subsets


def wide_sampling_task() -> dict[str, Any]:
    """The same comparison above 12 qubits, with exact marginals as the referee.

    Each engine is handed the same signed stabilizers and the same shot budget, and
    every sample set is decided against the exact distribution of each chosen subset of
    wires -- derived from the specification by linear algebra, not by either sampler --
    and then against the other engine under a null where both are correct. The full
    distribution is not the referee here because it is not enumerable above 12 qubits;
    the subsets are, so a `CONSISTENT` in this block means what it says for the tested
    projections and nothing about the wires nobody looked at.
    """
    rows: list[dict[str, Any]] = []
    for index, (label, target) in enumerate(wide_sampling_targets()):
        subsets = wide_subsets(target.num_qubits)
        own = apedp_sampling(target, shots=WIDE_SAMPLING_SHOTS, seed=7101 + index)
        row: dict[str, Any] = {
            "label": label,
            "target": specification_summary(target),
            "target_sha256": target.sha256(),
            "qubits": target.num_qubits,
            "shots": WIDE_SAMPLING_SHOTS,
            "subsets": [list(subset) for subset in subsets],
            "grammar": adjudicate_marginal(own, target, subsets, top=2,
                                           alpha=WIDE_SAMPLING_ALPHA),
            "stim": None,
            "comparison": None,
            "memory": sampling_memory(target, shots=WIDE_SAMPLING_SHOTS,
                                      seeds={"grammar": 7101 + index,
                                             "stim": 8101 + index}),
        }
        try:
            foreign = stim_sampling(target, shots=WIDE_SAMPLING_SHOTS, seed=8101 + index)
        except ToolUnavailable as error:
            row["stim"] = {"status": "unavailable", "reason": str(error)}
            row["memory"]["stim"] = {"status": "unavailable", "reason": str(error)}
        else:
            row["stim"] = adjudicate_marginal(foreign, target, subsets, top=2,
                                              alpha=WIDE_SAMPLING_ALPHA)
            row["comparison"] = marginal_pair(own, foreign, target, subsets,
                                              alpha=WIDE_SAMPLING_ALPHA)
        rows.append(row)
    return {
        "task": "sample Z-basis outcomes above the width where the full distribution is "
                "enumerable, decided by exact marginals of the specification",
        "shots_per_run": WIDE_SAMPLING_SHOTS,
        "alpha": WIDE_SAMPLING_ALPHA,
        "subset_design": (
            f"every adjacent pair of wires plus three {WIDE_SUBSET_SIZE}-wire "
            f"projections (contiguous, strided, seeded draw) of each target; each "
            f"subset's exact "
            f"distribution is derived from the specification by Gaussian elimination "
            f"over GF(2), so the referee is arithmetic rather than either sampler"),
        "referee": (
            "the exact Z-basis distribution of each chosen subset of wires; a sample "
            "set is CONSISTENT on a subset when a chi-square goodness-of-fit against "
            "that marginal has a p-value, multiplied by the number of subsets tested, "
            f"of at least {WIDE_SAMPLING_ALPHA:g}. The correction is Bonferroni and the "
            "product is recorded per subset, so the family-wise rate is the recorded "
            "one rather than the best of the subsets tested"),
        "two_sample_null": (
            "two independent correct samplers at the same shot count on the same "
            "subset, so the pair is only reported as DISAGREE beyond sampling noise"),
        "memory_rule": (
            "as in the enumerable block: a child interpreter per engine, the peak "
            "counter reset after imports and rebuild, growth reported against the floor "
            "it started from, and the child's count hash required to equal the recorded "
            "run's"),
        "what_it_does_not_decide": (
            "a projection, not the distribution: a state that differs from the target "
            "only on wires no tested subset covers passes, which is why the tested "
            "subsets are listed in every row, and every graph state is uniform in the "
            "Z basis, so no marginal of this kind separates one graph state from "
            "another -- the ring_graph row is in the block to show that (all of its "
            "marginals are uniform) rather than to claim a verdict it cannot support"),
        "rows": rows,
        "summary": wide_sampling_summary(rows),
    }


def wide_sampling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The wide block's numbers, recomputed from the rows that produced them."""
    medians = {}
    for engine in ("grammar", "stim"):
        rates = [row[engine]["samples_per_second"] for row in rows
                 if row[engine] is not None and "samples_per_second" in row[engine]]
        medians[engine] = statistics.median(rates) if rates else None
    decisions = sum(len(row["grammar"]["subsets"]) for row in rows)
    return {
        "targets": len(rows),
        "matched": sum(1 for row in rows if row["comparison"] is not None),
        "widest_qubits": max((int(row["qubits"]) for row in rows), default=0),
        "subset_decisions": decisions,
        "grammar_consistent_subsets": sum(
            1 for row in rows for subset in row["grammar"]["subsets"]
            if subset["verdict"] == "CONSISTENT"),
        "stim_consistent_subsets": sum(
            1 for row in rows if row["stim"] is not None
            for subset in row["stim"]["subsets"] if subset["verdict"] == "CONSISTENT"),
        "pair_subset_disagreements": [
            {"label": row["label"], "subset": subset["subset"],
             "family_p_value": subset["family_p_value"]}
            for row in rows if row["comparison"] is not None
            for subset in row["comparison"]["subsets"]
            if subset["verdict"] == "DISAGREE"],
        "uniform_subsets": sum(len(row["grammar"]["uniform_subsets"]) for row in rows),
        "memory": memory_summary(rows),
        "unresolved_subsets": sum(
            len(row["grammar"]["unresolved_subsets"])
            + (0 if row["stim"] is None else len(row["stim"]["unresolved_subsets"]))
            for row in rows),
        "smallest_family_p_value": {
            row["label"]: min(subset["family_p_value"]
                              for subset in row["grammar"]["subsets"])
            for row in rows},
        "stim_unavailable": [row["label"] for row in rows
                             if row["stim"].get("status") == "unavailable"],
        "median_samples_per_second": medians,
    }


def _padded_target(target: TargetSpecification, num_qubits: int) -> TargetSpecification:
    """The same state on a wider device: logical wires, then idle wires in ``|0>``."""
    if target.num_qubits > num_qubits:
        raise ValueError(f"target is wider ({target.num_qubits}) than the device ({num_qubits})")
    stabilizers = tuple(
        (tuple(labels) + (I,) * (num_qubits - len(labels)), phase)
        for labels, phase in target.stabilizers)
    stabilizers += tuple(
        (tuple(Z if index == qubit else I for index in range(num_qubits)), 0)
        for qubit in range(target.num_qubits, num_qubits))
    return TargetSpecification(
        name=f"{target.name}+idle", num_qubits=num_qubits, stabilizers=stabilizers,
        decides=(f"{target.decides} on the first {target.num_qubits} wires, every "
                 f"other wire in |0>"))


def _device_snapshot(backend_name: str = "FakeSherbrooke") -> dict[str, Any]:
    """The recorded coupling map both routing tasks use, or a reason it is absent.

    A snapshot shipped with qiskit-ibm-runtime, not a synthetic lattice: a recorded
    device's graph is the thing a routing claim is about, and a redrawn graph would
    make every routed depth a statement about a graph no device has. One helper, so
    the four heavy circuits and the corpus sample are routed onto the *same* device
    rather than two snapshots that might drift apart.
    """
    try:
        fake_provider = importlib.import_module("qiskit_ibm_runtime.fake_provider")
        backend = getattr(fake_provider, backend_name)()
    except (ImportError, AttributeError) as error:
        raise ToolUnavailable(
            f"a recorded device snapshot is not available "
            f"({type(error).__name__}: {error}); install `qiskit-ibm-runtime` to run "
            f"the routing comparison") from error

    edges = sorted({(min(int(a), int(b)), max(int(a), int(b)))
                    for a, b in backend.coupling_map.get_edges()})
    size = int(backend.num_qubits)
    device = {
        "name": str(backend.name),
        "backend_version": str(getattr(backend, "backend_version", "unknown")),
        "num_qubits": size,
        "edges": len(edges),
        "edge_list_sha256": hashlib.sha256(
            json.dumps([list(edge) for edge in edges]).encode("utf-8")).hexdigest(),
        "source": ("qiskit_ibm_runtime.fake_provider snapshot: a recorded device "
                   "coupling map, not a synthetic lattice"),
        "coupling_map": "undirected, from CouplingMap.get_edges()",
    }
    return {"device": device, "connectivity": Connectivity(device["name"], size, edges),
            "size": size, "edges": edges}


def device_spread_task() -> dict[str, Any]:
    """The heavy circuits on two recorded devices: how much the device moves the answer.

    One device is one graph, and a routing comparison on one graph is a statement about
    that graph. Both snapshots are recorded coupling maps from the same provider
    package, so the difference between the blocks is the device rather than a redraw of
    the same lattice: the same four logical circuits are routed by each of the three
    routers on both, every routed circuit is decided by the same referee, and the
    per-circuit depth ratio between the two devices is recorded beside the ratio between
    the routers.
    """
    blocks = {name: device_routing_task(name) for name in DEVICE_SNAPSHOTS}
    rows: list[dict[str, Any]] = []
    for name, block in blocks.items():
        others = [entry for other, entry in blocks.items() if other != name]
        for row in block.get("rows") or []:
            if row["comparison"] is None:
                rows.append({"circuit": row["circuit"], "device": name,
                             "status": "abstained", "depths": None,
                             "qiskit_over_apedp": None,
                             "ratio_qiskit_depth_against_other_device": None,
                             "verification": None})
                continue
            counterpart = next(
                (candidate for other in others for candidate in other.get("rows") or []
                 if candidate["circuit"] == row["circuit"] and candidate["comparison"]),
                None)
            rows.append({
                "circuit": row["circuit"],
                "device": name,
                "status": "routed",
                "depths": {
                    "qiskit": row["qiskit"]["routed_depth"],
                    "greedy": row["ours"]["greedy"]["depth_before_restore"],
                    "lookahead": row["ours"]["lookahead"]["depth_before_restore"],
                },
                "qiskit_over_apedp": row["comparison"]["depth_ratios_qiskit_over"],
                "ratio_qiskit_depth_against_other_device": (
                    row["qiskit"]["routed_depth"]
                    / max(1, counterpart["qiskit"]["routed_depth"])
                    if counterpart else None),
                "verification": {
                    "qiskit": row["qiskit"]["verification"]["verdict"],
                    **{strategy: row["ours"][strategy]["verification"]["verdict"]
                       for strategy in ("greedy", "lookahead")},
                },
            })
    routed = [row for row in rows if row["status"] == "routed"]
    return {
        "task": ("the same heavy circuits on two recorded device coupling maps, so a "
                 "routing result is not a single-graph result"),
        "devices": {name: block.get("device") for name, block in blocks.items()},
        "rows": rows,
        "summary": {
            "devices": len(blocks),
            "routed_rows": len(routed),
            "all_verified": all(
                all(verdict == "VALID" for verdict in row["verification"].values())
                for row in routed),
            "unavailable": [name for name, block in blocks.items()
                            if block.get("status") == "unavailable"],
            "depth_changed_between_devices": sum(
                1 for row in routed
                if row["ratio_qiskit_depth_against_other_device"] not in (None, 1.0)),
            "median_qiskit_over_apedp": {
                strategy: (statistics.median(
                    row["qiskit_over_apedp"][strategy] for row in routed)
                    if routed else None)
                for strategy in ("greedy", "lookahead")},
        },
    }


def device_routing_task(backend_name: str = "FakeSherbrooke") -> dict[str, Any]:
    """Each engine's own router on a recorded device coupling map.

    The existing topology task routes *both* circuits with this package's router on a
    line, which is the strongest form of the objection the record already carries.
    Here Qiskit is given a real device's coupling map (a snapshot shipped with
    qiskit-ibm-runtime, not a synthetic lattice) and its own transpiler, and its
    routed depth is compared with both of this package's routers on the same logical
    circuit and the same device graph. Qiskit's routed circuits are decided against
    the target relabelled by its own final layout, because it does not restore the
    layout; this package restores, and both its pre-restore and total depths are
    reported so the restore cost is not hidden.
    """
    try:
        snapshot = _device_snapshot(backend_name)
    except ToolUnavailable as error:
        return {"status": "unavailable", "reason": str(error), "device": None,
                "rows": [], "summary": None}

    device = snapshot["device"]
    size = snapshot["size"]
    edges = snapshot["edges"]
    connectivity = snapshot["connectivity"]

    ghz = ghz_target(100)
    ring = ring_graph_target(100)
    circuits = (
        ("ghz chain", ghz, build("chain", ghz).circuit),
        ("ghz balanced_tree", ghz, build("balanced_tree", ghz).circuit),
        ("ring 100", ring, build("ring", ring).circuit),
        ("qiskit synth ghz", ghz, qiskit_synthesis(ghz).circuit),
    )

    rows = []
    for label, target, logical in circuits:
        try:
            foreign = qiskit_routing(logical, target=target, num_qubits=size,
                                     edges=edges, device=device["name"],
                                     device_version=device["backend_version"],
                                     seed=ROUTING_SEED,
                                     optimization_level=ROUTING_OPTIMISATION)
        except ToolUnavailable as error:
            rows.append({"circuit": label,
                         "logical_target_sha256": target.sha256(),
                         "qiskit": {"status": "unavailable", "reason": str(error)},
                         "ours": {}, "comparison": None})
            continue
        foreign_target = relabelled_target(target, foreign.final_layout, size)
        foreign_report = adjudicate_specification(foreign.circuit, foreign_target)
        padded = _padded_target(target, size)
        ours = {}
        for strategy in ("greedy", "lookahead"):
            placement = place(logical, connectivity, strategy=strategy)
            report = adjudicate_specification(placement.circuit, padded)
            ours[strategy] = {
                "engine": f"apedp.placement:{strategy}",
                "target_sha256": padded.sha256(),
                "depth_before_restore": placement.depth_before_restore,
                "mapped_depth": placement.mapped_depth,
                "routing_swaps": placement.swap_gates,
                "two_qubit_gates": placement.two_qubit_gates,
                "verification": {"verdict": report.verdict, "tier": report.tier,
                                 "checks_failed": list(report.failed())},
            }
        foreign_payload = foreign.as_dict()
        foreign_payload["logical_target_sha256"] = target.sha256()
        foreign_payload["verification"] = {
            "verdict": foreign_report.verdict,
            "tier": foreign_report.tier,
            "checks_failed": list(foreign_report.failed()),
            "decided_against": ("the target relabelled by final_layout; Qiskit keeps "
                                "its layout rather than restoring it"),
            "decided_target_sha256": foreign_target.sha256(),
        }
        comparison = {
            "depth_ratios_qiskit_over": {
                strategy: foreign.routed_depth / max(1, ours[strategy][
                    "depth_before_restore"])
                for strategy in ("greedy", "lookahead")},
            "routed_depth_winners": {
                strategy: ("qiskit" if foreign.routed_depth
                           < ours[strategy]["depth_before_restore"]
                           else "apedp" if ours[strategy]["depth_before_restore"]
                           < foreign.routed_depth else "tie")
                for strategy in ("greedy", "lookahead")},
            "total_depth_ratios_qiskit_routed_over_apedp_total": {
                strategy: foreign.routed_depth / max(1, ours[strategy]["mapped_depth"])
                for strategy in ("greedy", "lookahead")},
            "reported_depth_agrees": foreign.reported_depth == foreign.routed_depth,
        }
        rows.append({
            "circuit": label,
            "logical_target_sha256": target.sha256(),
            "logical_depth": foreign.logical_depth,
            "qiskit": foreign_payload,
            "ours": ours,
            "comparison": comparison,
        })

    return {
        "task": ("route the same logical circuits onto a recorded device coupling map "
                 "with each tool's own router"),
        "device": device,
        "seed_transpiler": ROUTING_SEED,
        "optimization_level": ROUTING_OPTIMISATION,
        "basis_gates": list(ROUTING_BASIS),
        "rows": rows,
        "summary": {
            "circuits": len(rows),
            "matched": sum(1 for row in rows if row["comparison"] is not None),
            "qiskit_verified": sum(
                1 for row in rows if row["comparison"] is not None
                and row["qiskit"]["verification"]["verdict"] == "VALID"),
            "apedp_verified": sum(
                1 for row in rows for strategy in ("greedy", "lookahead")
                if row["comparison"] is not None
                and row["ours"][strategy]["verification"]["verdict"] == "VALID"),
            "routed_depth_wins": {
                strategy: {engine: sum(
                    1 for row in rows if row["comparison"] is not None
                    and row["comparison"]["routed_depth_winners"][strategy] == engine)
                    for engine in ("qiskit", "apedp", "tie")}
                for strategy in ("greedy", "lookahead")},
            "median_depth_ratio_qiskit_over": {
                strategy: (statistics.median(
                    row["comparison"]["depth_ratios_qiskit_over"][strategy]
                    for row in rows if row["comparison"] is not None)
                    if any(row["comparison"] is not None for row in rows) else None)
                for strategy in ("greedy", "lookahead")},
        },
    }


def topology_task(target: TargetSpecification, circuits: dict[str, Circuit]) -> dict[str, Any]:
    """The same circuits on a line: what each one costs once routing is priced."""
    connectivity = line(target.num_qubits)
    rows = []
    for engine, circuit in circuits.items():
        placement, seconds = timed(place, circuit, connectivity)
        from apedp_dyn.vqc.synthesis import adjudicate_specification

        report = adjudicate_specification(placement.circuit, target)
        rows.append({
            "engine": engine,
            "status": "compared",
            "target_sha256": target.sha256(),
            "topology": placement.connectivity,
            "logical_depth": placement.logical_depth,
            "mapped_depth": placement.mapped_depth,
            "routing_swaps": placement.swap_gates,
            "restore_swaps": placement.restore_swap_gates,
            "two_qubit_gates": placement.two_qubit_gates,
            "longest_hop": placement.longest_hop,
            "seconds": seconds,
            "verification": {"verdict": report.verdict, "tier": report.tier,
                             "checks_failed": list(report.failed()),
                             "verified_by": "apedp_dyn.vqc.synthesis "
                                            "(specification referee)"},
        })
    return {
        "task": "run the same 100-qubit GHZ circuits on a line",
        "topology": connectivity.as_dict(),
        "router": "apedp_dyn.vqc.placement (greedy, no scheduling, no lookahead)",
        "target_sha256": target.sha256(),
        "rows": rows,
    }


#: Repeats behind each referee's wall clock. A single call at these sizes moves by tens
#: of percent, which is the size of the ratio the record quotes, so the clock is a median
#: over repeated identical calls on both sides and every sample is kept.
CLOCK_REPEATS = 3

#: The recorded device snapshots the heavy circuits are routed onto. Two graphs, because a
#: routing result on one graph is a statement about that graph.
DEVICE_SNAPSHOTS = ("FakeSherbrooke", "FakeBrooklynV2")


def equivalence_task() -> dict[str, Any]:
    """The same pairs for both referees, on a corpus built so they disagree.

    Every pair is decided here on all three scopes this package offers, and the
    question the pair poses is the one compared with the other tool. All three are in
    the record because an abstention or a scope difference that is not written down
    reads as agreement: the pair that prepares one state from two unitaries is the
    case where a naive comparison would record a contradiction that is not there.
    """
    rows: list[dict[str, Any]] = []
    version: str | None = None
    for pair in equivalence_pairs():
        verdicts, first_seconds = timed(adjudicate_pair, pair,
                                        tolerance=EQUIVALENCE_TOLERANCE)
        our_clock: list[float] = [first_seconds]
        for _ in range(CLOCK_REPEATS - 1):
            _, seconds = timed(adjudicate_pair, pair, tolerance=EQUIVALENCE_TOLERANCE)
            our_clock.append(seconds)
        repeats = EQUIVALENCE_REPEATS if pair.family == "tolerance" else 1
        try:
            foreign = qcec_equivalence(pair.reference, pair.candidate, repeats=repeats)
            foreign_row = foreign.as_dict()
            version = foreign_row["version"] or version
            foreign_verdict: str | None = foreign.verdict
            foreign_criteria: tuple[str, ...] = foreign.criteria
        except ToolUnavailable as error:
            foreign_row = {
                "status": "unavailable",
                "reason": str(error),
                "verdict": None,
                "criteria": [],
                "considered_equivalent": [],
                "repeats": 0,
                "seconds": [],
                "checkers": [],
                "numerical_tolerance": None,
                "timeout": None,
                "seed": None,
                "decides": "nothing: the tool was not run",
            }
            foreign_verdict = None
            foreign_criteria = ()
        qcec_clock: list[float] = []
        try:
            clock_run, clock_seconds = timed(
                qcec_equivalence, pair.reference, pair.candidate, repeats=CLOCK_REPEATS)
            qcec_clock = list(clock_run.seconds) or [clock_seconds]
        except ToolUnavailable:
            qcec_clock = []
        our_median = statistics.median(our_clock)
        qcec_median = statistics.median(qcec_clock) if qcec_clock else None
        rows.append({
            **pair.as_dict(),
            "ours": {scope: verdict.as_dict() for scope, verdict in verdicts.items()},
            "qcec": foreign_row,
            "wall_clock": {
                "statistic": "median",
                "repeats": {"ours": CLOCK_REPEATS, "qcec": len(qcec_clock)},
                "ours_seconds_samples": our_clock,
                "qcec_seconds_samples": qcec_clock,
                "ours_median_seconds": our_median,
                "qcec_median_seconds": qcec_median,
                "ratio_ours_over_qcec": (our_median / qcec_median
                                         if qcec_median else None),
                "what_it_is": (
                    "the wall clock of deciding this pair, each referee timed on "
                    "repeated identical calls on one host and quoted as a median, "
                    "with every sample kept; the sizes differ by orders of magnitude "
                    "across the corpus, so the ratio is read per pair, never pooled"),
            },
            "agreement": classify(pair, verdicts, foreign_verdict=foreign_verdict,
                                  foreign_criteria=foreign_criteria),
        })

    compares = [row for row in rows if row["qcec"]["status"] == "compared"]
    return {
        "task": ("decide whether two circuits compute the same operation, on a corpus "
                 "constructed so the two referees disagree"),
        "referee": ("apedp_dyn.vqc.equivalence: the dense unitary and its controlled "
                    "lift up to 10 qubits, the stabilizer tier above it, plus the "
                    "prepared-state scope only the stabilizer tier can decide"),
        "tolerance": EQUIVALENCE_TOLERANCE,
        "foreign": {
            "tool": "mqt.qcec",
            "version": version,
            "numerical_tolerance": next((row["qcec"]["numerical_tolerance"]
                                         for row in compares
                                         if row["qcec"]["numerical_tolerance"]), None),
            "timeout": None,
            "repeats": EQUIVALENCE_REPEATS,
            "corpus_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" /
                                       "equivalence_corpus.py"),
            "note": ("QCEC runs its own default method selection and numerical "
                     "tolerance, untuned: a threshold chosen to agree with this "
                     "package would not be a second opinion"),
        },
        "design": {
            "families": sorted({pair.family for pair in equivalence_pairs()}),
            "scopes": list(SCOPES),
            "pairs": len(rows),
            "repeated_family": "tolerance",
            "repeats": EQUIVALENCE_REPEATS,
            "relation_is_fixed_by": ("the construction on each row, not by either "
                                     "referee"),
        },
        "clock": {
            "statistic": "median",
            "repeats_per_referee": CLOCK_REPEATS,
            "pairs_with_both_clocks": sum(
                1 for row in rows if row["wall_clock"]["qcec_median_seconds"]),
            "median_ratio_ours_over_qcec": statistics.median(
                [row["wall_clock"]["ratio_ours_over_qcec"] for row in rows
                 if row["wall_clock"]["ratio_ours_over_qcec"]]) if any(
                    row["wall_clock"]["ratio_ours_over_qcec"] for row in rows) else None,
            "why": (
                "a verdict is a decision, a clock is a measurement of this host: each "
                "pair is decided CLOCK_REPEATS times by both referees and the row "
                "quotes the median with all samples, so the ratio can be recomputed "
                "and the spread it hid is visible"),
        },
        "rows": rows,
        "summary": equivalence_summary(rows),
    }


def equivalence_verdict(equivalence: Mapping[str, Any]) -> str:
    """The record's sentence about the equivalence task, built from its own rows."""
    summary = equivalence["summary"]
    outcomes = summary["outcomes"]
    compared = [row for row in equivalence["rows"]
                if row["qcec"]["status"] == "compared"]
    if not compared:
        reason = next((row["qcec"].get("reason") for row in equivalence["rows"]
                       if row["qcec"].get("reason")), "no reason recorded")
        return (f"MQT QCEC could not be run here ({reason}); the corpus is recorded "
                f"as an abstention, not as a loss")
    tolerance = equivalence["tolerance"]
    qcec_tolerance = equivalence["foreign"].get("numerical_tolerance")
    unstable = outcomes.get("UNSTABLE", 0)
    criteria = sorted({criterion for row in equivalence["rows"]
                       if row["agreement"]["outcome"] == "UNSTABLE"
                       for criterion in row["qcec"]["criteria"]})
    sentence = (
        f"on {summary['pairs']} pairs whose relation is fixed by the construction "
        f"that built them, the two referees agree on {outcomes.get('AGREE', 0)} and "
        f"differ on {outcomes.get('DIFFER_BY_THRESHOLD', 0)} where a threshold decides")
    if unstable:
        sentence += (
            f", while on a further {unstable} the other tool has no single answer: "
            f"repeated identical calls returned {' and '.join(criteria)}")
    sentence += (
        f". Both apply a threshold and neither is the arbiter: this package decides at "
        f"{tolerance:g} on the per-amplitude Frobenius distance and QCEC at a numerical "
        f"tolerance of its own"
        + (f" ({qcec_tolerance:.2e}, as its configuration declares it)"
           if qcec_tolerance else "")
        + ", which are not the same quantity; the corpus fixes the exact answer on "
          "every ladder pair and both distances are recorded beside the verdicts. "
        + f"{outcomes.get('WE_ABSTAIN', 0)} pair(s) above the dense bound are decided "
          f"by QCEC and refused here (an abstention on this side), "
          f"{outcomes.get('QCEC_ABSTAINS', 0) + outcomes.get('BOTH_ABSTAIN', 0)} are "
          f"declined by QCEC, and {outcomes.get('SCOPE', 0)} prepares one state from "
          f"two different unitaries, where QCEC answers the unitary question and this "
          f"package's own unitary verdict agrees with it. "
        + (f"{len(summary['contradictions'])} divergence(s) are left unexplained."
           if summary["contradictions"] else
           "No divergence is left unexplained."))
    return sentence


def advantage_measured(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ratios, computed from the rows rather than written in prose.

    The prose in `verdict` is built from this block, so a run that changes the numbers
    cannot leave a sentence behind that says something else. Each row uses the
    *promoted* grammar construction -- the one this package would offer -- and the
    qiskit row from the same task, and records the ratio both ways round.
    """
    measured = []
    for task in tasks:
        if not task.get("target"):
            continue
        grammar = [row for row in task["rows"]
                   if row["engine"] == "apedp.synthesis" and row["status"] == "compared"]
        foreign = [row for row in task["rows"]
                   if row["engine"] != "apedp.synthesis" and row["status"] == "compared"]
        if not grammar or not foreign:
            continue
        chosen = next((row for row in grammar if row.get("promoted")), grammar[0])
        other = foreign[0]
        measured.append({
            "target": task["target"]["name"],
            "qubits": task["target"]["qubits"],
            "grammar_variant": chosen["variant"],
            "grammar": {"two_qubit_gates": chosen["metrics"]["two_qubit_gates"],
                        "operations": chosen["metrics"]["operations"],
                        "depth": chosen["metrics"]["depth"]},
            "qiskit": {"two_qubit_gates": other["metrics"]["two_qubit_gates"],
                       "operations": other["metrics"]["operations"],
                       "depth": other["metrics"]["depth"]},
            "two_qubit_gate_ratio_qiskit_over_grammar": (
                other["metrics"]["two_qubit_gates"]
                / max(1, chosen["metrics"]["two_qubit_gates"])),
            "depth_ratio_qiskit_over_grammar": (
                other["metrics"]["depth"] / max(1, chosen["metrics"]["depth"])),
            "agrees_on_state": (chosen["verification"]["verdict"] == "VALID"
                                and other["verification"]["verdict"] == "VALID"),
        })
    return measured


def _summary_line(measured: list[dict[str, Any]], target: str, qubits: int) -> str:
    row = next((entry for entry in measured
                if entry["target"] == target and entry["qubits"] == qubits), None)
    if row is None:  # pragma: no cover - the record always contains these
        return "not measured"
    return (f"{row['grammar']['two_qubit_gates']} two-qubit gates against "
            f"{row['qiskit']['two_qubit_gates']} "
            f"(x{row['two_qubit_gate_ratio_qiskit_over_grammar']:.0f}), depth "
            f"{row['grammar']['depth']} against {row['qiskit']['depth']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=RECORD)
    args = parser.parse_args()

    tasks = []
    for num_qubits in SIZES:
        tasks.append(synthesis_task(
            ghz_target(num_qubits),
            families=("chain", "balanced_tree"),
            promoted_family="balanced_tree"))
    for num_qubits in SIZES:
        tasks.append(synthesis_task(
            ring_graph_target(num_qubits),
            families=("ring",),
            promoted_family="ring"))
    tasks.append(random_state_task())

    # Task 4 needs the 100-qubit GHZ circuits again, from both engines.
    topology_target = ghz_target(SIZES[-1])
    topology_circuits = {
        "apedp.synthesis:grammar:chain": build("chain", topology_target).circuit,
        "apedp.synthesis:grammar:balanced_tree":
            build("balanced_tree", topology_target).circuit,
    }
    try:
        topology_circuits["qiskit:synth_circuit_from_stabilizers"] = \
            qiskit_synthesis(topology_target).circuit
    except ToolUnavailable:
        pass
    tasks.append(topology_task(topology_target, topology_circuits))
    corpus = shared_corpus_task()
    sampling = sampling_task()
    wide_sampling = wide_sampling_task()
    device_routing = device_routing_task()
    device_routing["second_device_spread"] = device_spread_task()
    equivalence = equivalence_task()

    statuses = [status.as_dict() for status in tool_statuses()]
    measured = advantage_measured(tasks)
    widest = max(SIZES)

    if not any(status["counts_as_a_comparison"] for status in statuses):
        # Refuse rather than overwrite a record that holds a real comparison with one
        # that holds none: an abstention inventory is a fine thing to *report*, and a
        # terrible thing to leave behind in place of measurements.
        raise SystemExit(
            "no external tool is available in this environment, so this run would "
            "replace the recorded comparison with abstentions. Install one with "
            "`pip install 'apedp[compare]'` to run the comparison, or leave the "
            "existing record alone.")

    # ---------------------------------------------------------------- the checks
    compared_rows = [row for task in tasks for row in task["rows"]
                     if row["status"] == "compared"]
    local_rows = [row for row in compared_rows if row["engine"] == "apedp.synthesis"]
    # Synthesis rows only: a topology row has no resource claim to check, and its
    # verdict is already in the `every_*_verifies` checks below.
    foreign_rows = [row for row in compared_rows if "resource_claim" in row]
    corpus_rows = corpus["rows"]
    corpus_matched = [row for row in corpus_rows if row["comparison"] is not None]
    depth_context = qiskit_depth_context(corpus_rows)
    depth_placed = depth_context["targets"]
    corpus_rebuilt = shared_target_corpus()
    corpus_routing = corpus["routing"]
    routing_rows = corpus_routing.get("rows") or []
    routing_matched = [row for row in routing_rows if row["comparison"] is not None]
    syntheses_by_label = {row["label"]: row for row in corpus_matched}
    third_party = corpus["third_party"]
    third_synthesis = third_party["synthesis"]
    third_routed = third_party["routed"]
    sampling_rows = sampling["rows"]
    sampling_matched = [row for row in sampling_rows if row["comparison"] is not None]
    wide_rows = wide_sampling["rows"]
    wide_matched = [row for row in wide_rows if row["comparison"] is not None]
    device_rows = device_routing["rows"]
    device_matched = [row for row in device_rows if row["comparison"] is not None]
    device_qiskit_wins = [row["circuit"] for row in device_matched
                         if row["comparison"]["routed_depth_winners"]["lookahead"]
                         == "qiskit"]
    device_apedp_wins = [row["circuit"] for row in device_matched
                         if row["comparison"]["routed_depth_winners"]["lookahead"]
                         == "apedp"]
    checks = {
        "every_local_row_verifies": all(
            row["verification"]["verdict"] == "VALID" for row in local_rows),
        "every_placed_circuit_verifies": all(
            row["verification"]["verdict"] == "VALID" for row in compared_rows
            if "topology" in row),
        "every_foreign_circuit_is_verified_by_our_referee": all(
            row["verification"]["verdict"] == "VALID" for row in foreign_rows),
        "every_foreign_claim_is_met_by_the_translated_circuit": all(
            row["resource_claim"]["verdict"] == "MET" for row in foreign_rows),
        "every_abstention_names_a_reason": all(
            row.get("reason") for row in
            [row for task in tasks for row in task["rows"]
             if row["status"] != "compared"]),
        "every_abstaining_tool_names_a_reason": all(
            status["reason"] for status in statuses
            if not status["counts_as_a_comparison"]),
        "no_unavailable_tool_is_counted_as_a_comparison": all(
            status["counts_as_a_comparison"] is False for status in statuses
            if status["status"] != "compared"),
        "the_capability_asymmetry_is_recorded": any(
            row["status"] == "abstained"
            for task in tasks if "no local construction family" in task["task"]
            for row in task["rows"]),
        "every_task_shares_one_specification": all(
            {row.get("target_sha256") for row in task["rows"]} == {task["target_sha256"]}
            for task in tasks),
        "at_least_one_foreign_circuit_ran": bool(foreign_rows),
        # ------------------------------------------------------ the shared corpus
        "the_shared_corpus_is_reproducible_from_its_seeds": (
            [(entry.label, entry.target.sha256()) for entry in shared_target_corpus()]
            == [(entry.label, entry.target.sha256()) for entry in corpus_rebuilt]),
        "every_corpus_row_shares_one_specification": all(
            row["target_sha256"] == row["grammar"].get("target_sha256")
            and row["target_sha256"] == row["qiskit"].get("target_sha256")
            for row in corpus_rows),
        "every_corpus_grammar_circuit_verifies": all(
            row["grammar"]["verification"]["verdict"] == "VALID"
            for row in corpus_matched),
        "every_corpus_foreign_circuit_verifies": all(
            row["qiskit"]["verification"]["verdict"] == "VALID"
            for row in corpus_matched),
        "no_corpus_abstention_is_scored_as_a_win": all(
            "metrics" not in row["grammar"] and row["comparison"] is None
            for row in corpus_rows if row["grammar"]["status"] != "compared"),
        "the_corpus_covers_a_comparison_and_an_abstention": (
            len(corpus_matched) >= 4 and bool(corpus["summary"]["grammar_abstained"])),
        "the_corpus_matches_its_declared_sample_design": (
            corpus["summary"]["targets"] == corpus["design"]["expected_targets"]
            and all(cell["replicates_per_width"] == [SAMPLE_REPLICATES]
                    for family, cell in corpus["summary"]["by_family"].items()
                    if family != "random_stabilizer")
            and all(cell["matched"] > 1 for cell in
                    corpus["summary"]["by_family"].values() if cell["matched"])),
        "every_corpus_row_names_its_cell_in_the_sample": all(
            row["family"] and str(row["label"]).startswith(row["family"])
            and int(row["replicate"]) < SAMPLE_REPLICATES
            for row in corpus_rows),
        "the_corpus_reports_a_spread_and_not_a_single_row": all(
            cell["per_metric"][metric]["ratio_qiskit_over_grammar"]["n"]
            == cell["matched"]
            and cell["per_metric"][metric]["ratio_qiskit_over_grammar"]["iqr"]
            is not None
            for cell in corpus["summary"]["by_family"].values() if cell["matched"]
            for metric in CORPUS_METRICS),
        "the_corpus_summary_recomputes_from_its_rows": (
            corpus_summary(corpus_rows) == corpus["summary"]),
        # ------------------------------------------- Experiment E against the corpus
        # An absent Qiskit is an abstention with a reason, not a histogram of nothing.
        "the_depth_histogram_accounts_for_every_corpus_row": (
            (depth_context["status"] == "unavailable"
             and bool(depth_context["reason"]) and not depth_context["buckets"])
            or (depth_context["status"] == "measured"
                and sum(bucket["count"] for bucket in depth_context["buckets"])
                == depth_context["overall"]["n"] == len(corpus_rows))),
        "the_depth_context_recomputes_from_its_rows": (
            qiskit_depth_context(corpus_rows) == depth_context),
        "every_experiment_e_target_is_placed_against_a_named_cell": all(
            row["placement"]["cell_qubits"] in depth_context["corpus"]["widths"]
            and row["placement"]["n"] > 0
            and row["placement"]["min"] <= row["placement"]["max"]
            for row in depth_placed),
        "the_placement_counts_only_the_cell_it_names": all(
            row["placement"]["qiskit_depths_at_or_below"]
            == sum(1 for value in depth_context["by_width"][
                       str(row["placement"]["cell_qubits"])]["depths"]
                   if value <= row["depth"])
            for row in depth_placed),
        "only_a_width_matched_target_is_called_like_for_like": (
            depth_context["status"] != "measured"
            or (depth_context["like_for_like"]
                == [row["target"] for row in depth_placed
                    if row["qubits"] in depth_context["corpus"]["widths"]]
                and len(depth_context["like_for_like"]) == 1)),
        # ------------------------------------------------------ the corpus routing
        "the_corpus_routing_covers_every_matched_instance": (
            (corpus_routing.get("status") == "unavailable"
             and bool(corpus_routing.get("reason")))
            or (len(routing_rows) == len(corpus_matched) == len(routing_matched))),
        "every_routed_corpus_circuit_verifies": all(
            row["comparison"] is None
            or (row["ours"]["verification"]["verdict"] == "VALID"
                and row["qiskit"]["verification"]["verdict"] == "VALID")
            for row in routing_rows),
        "every_corpus_routing_abstention_names_a_reason": all(
            row["comparison"] is not None or row["qiskit"].get("reason")
            for row in routing_rows),
        "the_routed_corpus_is_the_synthesised_corpus": all(
            row["logical_circuit_sha256"]
            == syntheses_by_label[row["label"]]["grammar"]["circuit_sha256"]
            for row in routing_rows),
        "each_engine_routed_its_own_circuit": all(
            row["ours"]["router"] == CORPUS_ROUTER_OURS
            and row["qiskit"].get("engine") == "qiskit"
            and row["qiskit"].get("note", {}).get("router", "").startswith("Qiskit")
            for row in routing_matched),
        "the_corpus_routing_summary_recomputes_from_its_rows": (
            corpus_routing_summary(routing_rows) == corpus_routing.get("summary")),
        # ------------------------------------------------- the third-party referee
        "every_corpus_output_is_decided_by_the_third_party_referee": (
            third_synthesis is not None and (
                (third_synthesis["decided"] == 0
                 and third_party["status"] == "unavailable")
                or (third_synthesis["both_valid"] == third_synthesis["instances"]
                    and not third_synthesis["undecided"]))),
        "the_third_party_referee_disagrees_with_our_referee_nowhere": (
            third_synthesis is not None
            and not third_synthesis["disagreements_with_our_referee"]),
        "the_routed_corpus_is_decided_by_the_third_party_referee": (
            third_routed is None
            or (third_routed["both_valid"] == third_routed["instances"]
                and not third_routed["undecided"])),
        "the_third_party_referee_decides_the_routed_sample_on_the_relabelled_target": (
            third_routed is None or third_routed["decided"] == third_routed["decisions"]),
        "every_third_party_abstention_names_a_reason": all(
            block[side]["reason"]
            for row in [*corpus_rows, *routing_rows] if row.get("third_party")
            for block in [row["third_party"]] for side in ("apedp", "qiskit")
            if block[side]["verdict"] not in ("VALID", "INVALID")),
        "the_third_party_summary_recomputes_from_its_rows": (
            third_party_summary(corpus_rows) == third_synthesis
            and third_party_summary(routing_rows) == third_routed),
        # ----------------------------------------------------------- the sampling
        "every_sampling_run_is_consistent_with_the_exact_distribution": all(
            row["grammar"]["verdict"] == "CONSISTENT"
            and (row["comparison"] is None
                 or row["stim"]["verdict"] == "CONSISTENT")
            for row in sampling_rows),
        "every_sampling_pair_agrees": all(
            row["comparison"]["verdict"] == "AGREE" for row in sampling_matched),
        "every_sampling_row_shares_one_specification": all(
            row["grammar"]["target_sha256"] == row["target_sha256"]
            and (row["comparison"] is None
                 or row["stim"]["target_sha256"] == row["target_sha256"])
            for row in sampling_rows),
        "every_sampling_abstention_names_a_reason": all(
            row["stim"].get("reason") for row in sampling_rows
            if row["comparison"] is None),
        "the_sampling_summary_recomputes_from_its_rows": (
            sampling_summary(sampling_rows) == sampling["summary"]),
        # ------------------------------------------------------ the sampling memory
        "every_sampling_memory_measurement_is_of_the_recorded_samples": all(
            row["memory"][engine]["counts_sha256"] == row[engine]["counts_sha256"]
            and row["memory"][engine]["target_sha256"] == row["target_sha256"]
            and row["memory"][engine]["shots"] == row["shots"]
            and row["memory"][engine]["growth_bytes"]
            == (row["memory"][engine]["window_peak_bytes"]
                - row["memory"][engine]["floor_bytes"])
            and row["memory"][engine]["peak_reset"] is True
            for row in [*sampling_rows, *wide_rows]
            for engine in ("grammar", "stim")
            if row.get("memory", {}).get(engine, {}).get("status") != "unavailable"),
        "every_sampling_memory_abstention_names_a_reason": all(
            row["memory"][engine].get("reason")
            for row in [*sampling_rows, *wide_rows]
            for engine in ("grammar", "stim")
            if row.get("memory", {}).get(engine, {}).get("status") == "unavailable"),
        "the_sampling_memory_summary_recomputes_from_its_rows": (
            memory_summary(sampling_rows) == sampling["summary"]["memory"]
            and memory_summary(wide_rows) == wide_sampling["summary"]["memory"]),
        # ------------------------------------------------------ the wide sampling
        "every_wide_row_is_above_the_enumerable_width": all(
            int(row["qubits"]) > SAMPLING_QUBITS for row in wide_rows),
        "every_wide_sampling_run_matches_its_exact_marginals": all(
            row["grammar"]["verdict"] == "CONSISTENT"
            and (row["comparison"] is None
                 or row["stim"]["verdict"] == "CONSISTENT")
            for row in wide_rows),
        "every_wide_sampling_pair_agrees": all(
            row["comparison"]["verdict"] == "AGREE" for row in wide_matched),
        "every_wide_row_shares_one_specification": all(
            row["grammar"]["target_sha256"] == row["target_sha256"]
            and all(subset["target_sha256"] == row["target_sha256"]
                    for subset in row["grammar"]["subsets"])
            and (row["comparison"] is None
                 or row["stim"]["target_sha256"] == row["target_sha256"])
            for row in wide_rows),
        "every_wide_sampling_abstention_names_a_reason": all(
            (row["stim"].get("reason") or row["comparison"] is not None)
            for row in wide_rows),
        "the_wide_block_tests_more_than_one_subset_per_target": all(
            len(row["grammar"]["subsets"]) > WIDE_SUBSET_SIZE for row in wide_rows),
        "the_wide_block_records_the_uniform_marginal_case": any(
            len(row["grammar"]["uniform_subsets"]) == len(row["grammar"]["subsets"])
            for row in wide_rows),
        "every_wide_row_is_corrected_for_the_number_of_subsets_tested": all(
            subset["family_p_value"] == min(1.0, subset["p_value"] * len(row[engine]["subsets"]))
            for row in wide_rows for engine in ("grammar", "stim")
            if row[engine] is not None for subset in row[engine]["subsets"]),
        "every_wide_subset_is_decided_by_a_valid_test": all(
            subset["verdict"] != "UNRESOLVED"
            for row in wide_rows for engine in ("grammar", "stim")
            if row[engine] is not None for subset in row[engine]["subsets"]),
        "no_wide_subset_fails_the_corrected_p_value": all(
            subset["verdict"] == "CONSISTENT"
            for row in wide_rows for engine in ("grammar", "stim")
            if row[engine] is not None for subset in row[engine]["subsets"]),
        "the_wide_sampling_summary_recomputes_from_its_rows": (
            wide_sampling_summary(wide_rows) == wide_sampling["summary"]),
        # ------------------------------------------------------- the device routing
        "every_device_routed_circuit_verifies": all(
            row["qiskit"]["verification"]["verdict"] == "VALID"
            and all(row["ours"][strategy]["verification"]["verdict"] == "VALID"
                    for strategy in ("greedy", "lookahead"))
            for row in device_matched),
        "every_device_row_relabels_the_same_logical_circuit": all(
            row["qiskit"]["logical_target_sha256"] == row["logical_target_sha256"]
            and len(set(row["qiskit"]["final_layout"])) == len(
                row["qiskit"]["final_layout"])
            for row in device_matched),
        "qiskit_reported_depth_matches_the_reimported_circuit": all(
            row["comparison"]["reported_depth_agrees"] for row in device_matched),
        "the_device_is_a_recorded_coupling_map": (
            device_routing.get("device", {}).get("source", "").find("recorded") >= 0
            and device_routing.get("device", {}).get("num_qubits", 0) >= 100
            and bool(device_routing.get("device", {}).get("edge_list_sha256"))),
        "at_least_one_device_router_ran": bool(device_matched),
        "the_device_spread_ran_on_more_than_one_graph": (
            device_routing["second_device_spread"]["summary"]["devices"]
            == len(DEVICE_SNAPSHOTS)
            and not device_routing["second_device_spread"]["summary"]["unavailable"]),
        "both_devices_verify_every_routed_circuit": (
            device_routing["second_device_spread"]["summary"]["routed_rows"] > 0
            and device_routing["second_device_spread"]["summary"]["all_verified"]),
        "the_equivalence_clocks_are_medians_over_repeats": all(
            row["wall_clock"]["repeats"]["ours"] == CLOCK_REPEATS
            and (row["qcec"]["status"] != "compared"
                 or row["wall_clock"]["repeats"]["qcec"] == CLOCK_REPEATS)
            for row in equivalence["rows"]),
        # ---------------------------------------------------------- the equivalence
        "every_equivalence_pair_answers_what_its_construction_requires": (
            not equivalence["summary"]["expected_met"]),
        "no_equivalence_disagreement_is_unexplained": (
            not equivalence["summary"]["contradictions"]),
        "the_equivalence_corpus_exercises_agreement_and_divergence": (
            "AGREE" in equivalence["summary"]["outcomes"]
            and len(equivalence["summary"]["outcomes"]) >= 3),
        "every_equivalence_pair_is_decided_on_all_three_scopes": all(
            list(row["ours"]) == list(SCOPES) for row in equivalence["rows"]),
        "every_equivalence_abstention_names_a_reason": all(
            (row["ours"][row["question"]]["verdict"] != "UNRESOLVED"
             or row["ours"][row["question"]]["reason"])
            and (row["qcec"]["status"] == "compared" or row["qcec"].get("reason"))
            for row in equivalence["rows"]),
        "the_repeated_family_is_asked_the_identical_question_more_than_once": all(
            row["qcec"]["repeats"] == EQUIVALENCE_REPEATS
            for row in equivalence["rows"] if row["family"] == "tolerance"),
        "an_unstable_equivalence_answer_carries_every_criterion": all(
            len(row["qcec"]["criteria"]) == row["qcec"]["repeats"]
            and len(set(row["qcec"]["criteria"])) > 1
            for row in equivalence["rows"]
            if row["agreement"]["outcome"] == "UNSTABLE"),
        "every_equivalence_verdict_is_compared_on_the_question_it_answers": all(
            row["agreement"]["compared"] is False
            for row in equivalence["rows"] if row["question"] != "standalone"),
        "qcec_is_a_comparison_on_the_equivalence_task": (
            any(status["tool"] == "mqt.qcec" and status["counts_as_a_comparison"]
                for status in statuses)
            and bool([row for row in equivalence["rows"]
                      if row["agreement"]["compared"]])),
        "the_equivalence_summary_recomputes_from_its_rows": (
            equivalence_summary(equivalence["rows"]) == equivalence["summary"]),
        "every_equivalence_pair_carries_the_construction_that_fixes_it": all(
            row["construction"] and row["relation"] for row in equivalence["rows"]),
    }
    if not all(checks.values()):
        raise SystemExit(f"refusing to write a record with failed checks: {checks}")

    overall_depth = corpus["summary"]["overall"]["per_metric"]["depth"][
        "ratio_qiskit_over_grammar"]
    third_reason = next((block[side]["reason"]
                         for row in [*corpus_rows, *routing_rows]
                         if row.get("third_party")
                         for block in [row["third_party"]]
                         for side in ("apedp", "qiskit")
                         if block[side]["verdict"] not in ("VALID", "INVALID")), "")
    payload = {
        "schema": "apedp.vqc.external_comparison.v8",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "A task-matched comparison with an external tool on the same frozen "
            "specification, with the foreign circuit decided by this package's own "
            "referee. A tool that did not run is recorded as an abstention, never as a "
            "loss."),
        "rule": ("every row is either a matched measurement (both engines given the same "
                 "target_sha256) or an abstention naming its reason"),
        "tasks": tasks,
        "shared_corpus": corpus,
        "qiskit_depth_context": depth_context,
        "sampling": sampling,
        "wide_sampling": wide_sampling,
        "device_routing": device_routing,
        "equivalence": equivalence,
        "tools": statuses,
        "advantage": {
            "claimed": False,
            "measured": measured,
            "corpus": corpus["summary"],
            "why": (
                "the GHZ and ring rows compare a construction grammar that knows those "
                "targets against a synthesizer that is general over stabilizer states, "
                "and the random-state row is the case where the grammar has nothing to "
                "say. A cheaper circuit for a target the grammar was written for is a "
                "measurement about that target, not a general advantage, and no claim "
                "of general superiority is made here. On the shared corpus, where no "
                f"construction is specialised, the two-qubit counts agree on every one "
                f"of {corpus['summary']['matched']} matched instances and the grammar's "
                f"only measured saving is depth -- median "
                f"{overall_depth['median']:.2f}x, quartiles "
                f"{overall_depth['q1']:.2f}-{overall_depth['q3']:.2f}, range "
                f"{overall_depth['min']:.2f}-{overall_depth['max']:.2f} across the "
                f"sample -- which is a schedule difference a rescheduling pass can "
                f"also produce, and the routed corpus is the measurement of that: with "
                f"each engine's own router on a recorded device the two land within a "
                f"few percent of each other at the median and Qiskit is the shallower "
                f"of the two on more instances than not"),
            "what_would_change_it": (
                "a third party verifying both the outputs and the resource counts "
                "(the prepared state is now decided by Stim as well, but the gate "
                "counts, depth and clocks are still this package's accounting), the "
                "comparison "
                "repeated on hardware connectivities with each tool's own router, and "
                "a sampling referee at wide widths that decides the whole outcome "
                "distribution rather than a family of projections (the wide block "
                "now decides exact marginals at 32 qubits, so the full distribution "
                "is still only decided at 12). The shared corpus, the Stim sampling "
                "task and the marginal referee are three of those conditions, not all "
                "of them"),
        },
        "checks": checks,
        "limitations": [
            "One host, one Qiskit version, one run per row: the wall clocks are single "
            "measurements, not medians, and they are here to show the scale of the "
            "synthesis cost rather than to be quoted as a ratio.",
            "Qiskit's synthesizer is general over stabilizer states and this package's "
            "grammar is not. The GHZ and ring rows are the favourable case for the "
            "grammar by construction, which is why the random-state row exists.",
            "Three random stabilizer states at three widths are three samples, not a "
            "distribution over stabilizer states; they exist to show that the "
            "grammar's abstention is structural rather than a property of one state.",
            "The line topology task still routes Qiskit's circuit with THIS "
            "package's router. The device routing task and the routed corpus are the "
            "counterparts: Qiskit is given a recorded device coupling map and its own "
            "transpiler, and this package's router runs on the same device graph. One "
            "seed and one optimisation level are recorded because SABRE is "
            "stochastic; each routed instance is one router run, not a median over "
            "seeds.",
            "The routed corpus compares each engine as shipped, which is not the same "
            "optimisation problem on both sides: Qiskit's transpiler searches an "
            "initial layout, routes and keeps its final layout, while this package "
            "starts from the identity placement, routes with its lookahead router and "
            "restores logical qubits to their original wires. The comparable number "
            "is the pre-restore routed depth; the total is reported beside it. A "
            "router-only difference -- a layout search this package does not do -- is "
            "therefore part of what the routed rows measure, and it is not priced.",
            "Qiskit keeps its final layout rather than restoring it, so its routed "
            "circuit is decided against the target relabelled by that layout. This "
            "package restores logical qubits to their original wires; both its "
            "pre-restore depth (the comparable routed number) and its total depth "
            "(the cost of fixed-wire verifiability) are reported rather than one "
            "number being chosen to flatter either engine.",
            "The device routing rows price a SWAP as one two-qubit gate on both sides. "
            "No CNOT-equivalent restatement is made in this task; that accounting "
            "belongs to the placement record, where its arithmetic is checked.",
            "Gate counts and depth are not fidelity, latency or error rate. There is no "
            "device calibration, error model or classical-control cost anywhere in this "
            "record.",
            "MQT QCEC is compared on the equivalence task only, and on one corpus: "
            "pairs whose relation is fixed by the construction that built them, of "
            "which seven are one angle apart on a tolerance ladder. It decides the "
            "standalone unitary question, so its verdicts are compared with this "
            "package's standalone verdict and never with the controlled or "
            "prepared-state ones -- a scope difference is recorded as a scope "
            "difference. Near its own numerical threshold its verdict changed between "
            "identical runs, so the record carries a distribution of criteria there "
            "instead of a verdict, and the threshold is reported as measured rather "
            "than as the value its configuration declares. It is given no timeout, its "
            "own default tolerance, and one corpus: it has not been run on synthesis, "
            "on routing, on sampling, or on any workload of this repository's own.",
            "The Stim comparison is sampling only, and its referee is exact "
            "arithmetic in two shapes: the full outcome distribution at 12 qubits, "
            "and the exact marginal on chosen wire subsets above that (16, 24 and 32 "
            "qubits). A marginal is a projection, not the distribution -- two "
            "different states can share every tested marginal, and every graph state "
            "is uniform in the Z basis, so no Z-basis marginal separates one graph "
            "state from another. The wide rows say 'these projections are right', "
            "not 'the state is right', and the uniform-marginal target is in the "
            "block so that limit is measured rather than asserted. It says nothing "
            "about Stim's ceiling: the throughput ratio compares a Python per-shot "
            "tableau clone with Stim's compiled C++ sampler, and Stim's published "
            "20,000-qubit surface-code workload is not addressed by any row here. No "
            "memory measurement was taken on either side.",
            "The two engines drew independent seeds, so the pair comparison is two "
            "independent samples of one state rather than paired trajectories.",
            "Peak memory is resident-set growth measured in a child interpreter per "
            "engine, with the peak counter reset after the engine's imports and the "
            "specification's rebuild, so it is the sampling step's own growth and not "
            "the ~165 MiB floor numpy and this package's imports cost before a shot is "
            "drawn. It is resident memory, not bytes allocated: an engine that recycles "
            "heap it already touched shows less than one that allocates fresh, and that "
            "is a property of the allocator as much as of the engine. It is one run per "
            "engine rather than a median, it measures no GPU and no allocator-internal "
            "bookkeeping, both engines' outcome tables are inside the window, and it is "
            "not a ceiling claim about Stim -- no 20,000-qubit workload is measured here.",
            "The shared corpus is a sample from this package's own generators: five "
            "widths, five independent draws per cell, three graph families, plus one "
            "random stabilizer state at each of three widths. That fixes the "
            "weakness of a hand-picked set -- every family is reported as a "
            "distribution -- but it is not graph space. The widths are five tiers, "
            "the draws come from one pairing-model generator and one tree-plus-edges "
            "generator, and instances at the same width share that width; no claim "
            "about the distribution of all graph states is made.",
            "The bootstrap intervals in the corpus summary resample the instances in "
            "this sample. They describe how much another draw from the same seeded "
            "generator could have moved a median, and are not confidence statements "
            "about graph space. The per-instance quartiles and ranges are reported "
            "beside them so the width of the distribution is visible without them.",
            "On the graph states both engines emit one entangler per edge, so the "
            "two-qubit counts agree by construction; the corpus measures schedule and "
            "abstention, not a gate-count advantage. The depth gap is a layering "
            "difference and a transpiler scheduling pass could close it, which is why "
            "it is recorded as a schedule result and not as synthesis superiority.",
            "Both engines' outputs are now decided twice: by this package's "
            "specification referee and by a third party -- Stim's tableau simulator, "
            "an independent implementation (`apedp_dyn.vqc.foreign_referee`). That "
            "second opinion "
            "covers the prepared state and nothing else: gate counts, depth and wall "
            "clock are still this package's accounting, and the bridge from this "
            "package's IR to Stim's circuit is ours, so a translation bug shared by "
            "both referees would not be caught. It is also Clifford-only -- a "
            "non-Clifford circuit is `REFUSED` rather than verified, which is an "
            "abstention and not a loss.",
        ],
        "reproducibility": {
            "script_sha256": sha256_of(Path(__file__)),
            "external_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "external.py"),
            "corpus_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "corpus.py"),
            "marginals_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "marginals.py"),
            "memory_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "memory.py"),
            "synthesis_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "synthesis.py"),
            "placement_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "placement.py"),
            "qasm_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "qasm.py"),
            "stim_version": next((status["version"] for status in statuses
                                  if status["tool"] == "stim"), None),
            "qiskit_ibm_runtime_version": importlib.metadata.version(
                "qiskit-ibm-runtime") if importlib.util.find_spec(
                    "qiskit_ibm_runtime") is not None else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": "python scripts/compare_vqc_external_tools.py",
        },
        "verdict": {
            "synthesis": (
                f"on a {widest}-qubit GHZ the grammar uses "
                f"{_summary_line(measured, 'ghz', widest)}; on a {widest}-qubit ring "
                f"state it uses {_summary_line(measured, 'ring_graph', widest)}. Both "
                f"sides are decided VALID against the same specification by the same "
                f"referee, so the difference is in the circuits, not in the verdict"),
            "target_dependence": (
                "the two-qubit-gate advantage is target-specific: large on GHZ, absent "
                "on the ring state, where a general Gaussian-elimination synthesizer "
                "reaches the same two-qubit count as the hand-written construction"),
            "asymmetry": (
                "on a random stabilizer state this package abstains and Qiskit "
                "synthesizes; that row is recorded because a comparison that only ran "
                "the favourable case would not be a comparison"),
            "abstentions": (
                "all three tools ran on the task each of them overlaps with, so there "
                "is no unavailable-tool abstention left to report; what remains are "
                "per-pair abstentions inside the equivalence task, where a tool that "
                "declined is recorded as declining rather than as losing. No claim of "
                "state-of-the-art performance rests on the large-width rows recorded "
                "elsewhere, and none of these tools was run on synthesis, sampling and "
                "equivalence alike: Qiskit on synthesis and routing, Stim on sampling, "
                "QCEC on equivalence"),
            "equivalence": equivalence_verdict(equivalence),
            "device_routing": (
                ("qiskit-ibm-runtime is not installed, so no recorded coupling map "
                 "was available; the routing comparison is an abstention, not a loss")
                if device_routing.get("status") == "unavailable" else
                f"on {device_routing['summary']['matched']} logical circuits routed "
                f"onto {device_routing.get('device', {}).get('name', 'a device')} "
                f"({device_routing.get('device', {}).get('num_qubits', 0)} qubits, "
                f"{device_routing.get('device', {}).get('edges', 0)} edges), Qiskit's "
                f"own transpiler wins {device_routing['summary']['routed_depth_wins']['lookahead']['qiskit']} "
                f"of {device_routing['summary']['matched']} routed-depth comparisons "
                f"against the lookahead router (median ratio "
                f"{device_routing['summary']['median_depth_ratio_qiskit_over']['lookahead']:.2f}x) "
                f"and {device_routing['summary']['routed_depth_wins']['greedy']['qiskit']} "
                f"of {device_routing['summary']['matched']} against greedy (median "
                f"{device_routing['summary']['median_depth_ratio_qiskit_over']['greedy']:.2f}x). "
                f"Which engine is shallower depends on the circuit: Qiskit wins on "
                f"{', '.join(device_qiskit_wins) or 'nothing'}, this package on "
                f"{', '.join(device_apedp_wins) or 'nothing'}. Both routers and both "
                f"routed and total depths are reported rather than the number that "
                f"would flatter one engine"),
            "sampling": (
                f"on {sampling['summary']['matched']} of {sampling['summary']['targets']} "
                f"12-qubit targets both samplers are CONSISTENT with the exact "
                f"distribution and the two sample sets AGREE on "
                f"{sampling['summary']['pairs_agreeing']} of "
                f"{sampling['summary']['matched']}; median throughput "
                f"{sampling['summary']['median_samples_per_second']['grammar']:,.0f} vs "
                f"{sampling['summary']['median_samples_per_second']['stim']:,.0f} "
                f"samples/s. That ratio is implementation, not ceiling: Python "
                f"tableau clones against a compiled C++ sampler"),
            "sampling_memory": (
                f"resident growth during the sampling step, measured in a child "
                f"interpreter whose peak counter is reset after its imports: at 12 "
                f"qubits the median is "
                f"{sampling['summary']['memory']['median_growth_mib']['grammar']:.1f} MiB "
                f"for this package and "
                f"{sampling['summary']['memory']['median_growth_mib']['stim']:.1f} MiB "
                f"for Stim, and above the enumerable width "
                f"{wide_sampling['summary']['memory']['median_growth_mib']['grammar']:.1f} "
                f"against "
                f"{wide_sampling['summary']['memory']['median_growth_mib']['stim']:.1f} "
                f"MiB, over an import floor of "
                f"{sampling['summary']['memory']['median_floor_mib']['grammar']:.0f} and "
                f"{sampling['summary']['memory']['median_floor_mib']['stim']:.0f} MiB "
                f"that is charged to neither engine. The number that moves is the "
                f"outcome table, not the width: a 32-qubit GHZ keeps two outcomes and "
                f"grows "
                f"{next(row['memory']['grammar']['growth_mib'] for row in wide_rows if row['label'] == 'ghz 32q'):.1f} "
                f"MiB here, while a 32-qubit random state with 20,000 distinct outcomes "
                f"grows "
                f"{max(row['memory']['grammar']['growth_mib'] for row in wide_rows):.1f} "
                f"MiB on both engines -- which is why the gap between the two samplers "
                f"is a few megabytes of implementation and not an order of magnitude, "
                f"and why neither number is a ceiling claim: Stim's 20,000-qubit "
                f"workload is not measured here"),
            "wide_sampling": (
                f"above the enumerable width the referee is the exact marginal on "
                f"{wide_sampling['summary']['subset_decisions']} wire subsets, up to "
                f"{wide_sampling['summary']['widest_qubits']} qubits: "
                f"{wide_sampling['summary']['grammar_consistent_subsets']} of "
                f"{wide_sampling['summary']['subset_decisions']} of this package's "
                f"marginals pass the Bonferroni-corrected chi-square and "
                f"{wide_sampling['summary']['stim_consistent_subsets']} of "
                f"{wide_sampling['summary']['subset_decisions']} of Stim's, with "
                f"{len(wide_sampling['summary']['pair_subset_disagreements'])} "
                f"disagreeing subsets between the two engines and "
                f"{wide_sampling['summary']['uniform_subsets']} subsets whose exact "
                f"marginal is uniform (the graph-state blind spot, recorded rather "
                f"than avoided). Median throughput "
                f"{wide_sampling['summary']['median_samples_per_second']['grammar']:,.0f} "
                f"vs "
                f"{wide_sampling['summary']['median_samples_per_second']['stim']:,.0f} "
                f"samples/s, the same implementation gap as at 12 qubits"),
            "third_party": (
                ("the third-party referee could not decide these circuits, so the "
                 "corpus rests on this package's referee: " + third_reason)
                if third_synthesis is None or not third_synthesis["decided"] else
                f"both engines' circuits are decided twice: by this package's "
                f"specification referee and by Stim's tableau simulator "
                f"({third_synthesis['version']}), which shares no code with it. On "
                f"{third_synthesis['instances']} matched corpus instances and "
                f"{third_routed['instances'] if third_routed else 0} routed ones the two "
                f"referees agree on every circuit -- "
                f"{third_synthesis['decisions'] + (third_routed['decisions'] if third_routed else 0)} "
                f"decisions, all VALID, no disagreement recorded -- so adjudication is "
                f"not self-referential. Stim decides the prepared state, not the gate "
                f"counts, and the bridge from this package's IR to Stim's circuit is "
                f"ours; both limits are in the limitations"),
            "corpus_routing": (
                ("the routed corpus sample is an abstention, not a loss: "
                 + str(corpus_routing.get("reason")))
                if corpus_routing.get("status") == "unavailable" else
                f"on {corpus_routing['summary']['matched']} corpus instances routed "
                f"onto {corpus_routing['device']['name']} "
                f"({corpus_routing['device']['num_qubits']} qubits, "
                f"{corpus_routing['device']['edges']} edges) with each engine's own "
                f"router -- Qiskit's transpiler with its own layout search and no "
                f"restore, this package's {CORPUS_ROUTER_STRATEGY} router with the "
                f"restore ladder reported separately -- Qiskit's routed circuit is "
                f"shallower on "
                f"{corpus_routing['summary']['share_instances_qiskit_shallower']:.0%} "
                f"of instances, median depth ratio "
                f"{corpus_routing['summary']['depth_ratio_qiskit_over_apedp']['median']:.2f}x "
                f"[quartiles "
                f"{corpus_routing['summary']['depth_ratio_qiskit_over_apedp']['q1']:.2f}-"
                f"{corpus_routing['summary']['depth_ratio_qiskit_over_apedp']['q3']:.2f}] "
                f"with its own spread by family: "
                + ", ".join(
                    f"{family} "
                    f"{block['depth_ratio_qiskit_over_apedp']['median']:.2f}x "
                    f"[{block['depth_ratio_qiskit_over_apedp']['q1']:.2f}-"
                    f"{block['depth_ratio_qiskit_over_apedp']['q3']:.2f}]"
                    for family, block in corpus_routing["summary"]["by_family"].items()
                    if block["matched"])
                + ". The abstract depth gap does not survive routing: the two "
                "routers land within "
                + f"{abs(1 - corpus_routing['summary']['depth_ratio_qiskit_over_apedp']['median']):.0%} "
                + "of each other at the median, Qiskit is shallower on "
                + f"{corpus_routing['summary']['wins']['qiskit']} of "
                + f"{corpus_routing['summary']['matched']} instances and this package "
                + f"on {corpus_routing['summary']['wins']['apedp']} -- partly because "
                + "Qiskit's transpiler searches an initial layout while this package "
                + "starts from the identity placement on a sparse device graph"),
            "shared_corpus": (
                f"on {corpus['summary']['matched']} of {corpus['summary']['targets']} "
                f"sampled targets both engines prepare the state; the grammar abstains "
                f"on the {len(corpus['summary']['grammar_abstained'])} random stabilizer "
                f"states. Two-qubit counts agree on every matched instance, so there is "
                f"no gate-count advantage to report. Depth is reported as a spread over "
                f"the sample, not as one row: median ratio "
                f"{overall_depth['median']:.2f}x (qiskit/grammar), quartiles "
                f"{overall_depth['q1']:.2f}-{overall_depth['q3']:.2f}, range "
                f"{overall_depth['min']:.2f}-{overall_depth['max']:.2f}; the grammar is "
                f"shallower on "
                f"{corpus['summary']['overall']['per_metric']['depth']['wins']['grammar']} "
                f"of {corpus['summary']['matched']} instances. Per family the median "
                f"depth ratio is " + ", ".join(
                    f"{family} "
                    f"{cell['per_metric']['depth']['ratio_qiskit_over_grammar']['median']:.2f}x"
                    for family, cell in corpus["summary"]["by_family"].items()
                    if cell["matched"]) + ". That is a schedule difference, not a "
                "gate-count one, and the spread is the size of the effect a single "
                "instance would have misreported"),
        },
    }

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for task in tasks:
        target = task.get("target")
        label = (f"{target['name']} {target['qubits']}q" if target
                 else f"{task.get('topology', {}).get('name', 'topology')} "
                      f"{task.get('topology', {}).get('nodes', '')}")
        print(f"\n{task['task']}  ({label})")
        for row in task["rows"]:
            if row["status"] == "compared" and "topology" in row:
                print(f"  {row['engine']:38s} logical {row['logical_depth']:>6} "
                      f"mapped {row['mapped_depth']:>6} "
                      f"swaps {row['routing_swaps']:>6} "
                      f"{row['verification']['verdict']}")
            elif row["status"] == "compared":
                metrics = row.get("metrics", {})
                print(f"  {row['engine']:16s} {row['variant'][:36]:38s} "
                      f"ops {metrics.get('operations', '-'):>6} "
                      f"2q {metrics.get('two_qubit_gates', '-'):>6} "
                      f"depth {metrics.get('depth', '-'):>5} "
                      f"{row['verification']['verdict']}")
            else:
                print(f"  {row['engine']:16s} {row['status']}: "
                      f"{str(row.get('reason'))[:90]}")
    print(f"\nshared corpus: {corpus['summary']['targets']} sampled instances, "
          f"{corpus['summary']['matched']} matched, "
          f"{len(corpus['summary']['grammar_abstained'])} abstentions")
    print(f"  {'family':16s} {'n':>3s}  {'grammar depth':>20s}  {'qiskit depth':>20s}  "
          f"{'ratio q/g':>22s}  wins g/q/t")
    for family, cell in corpus["summary"]["by_family"].items():
        depth = cell["per_metric"]["depth"]
        if not cell["matched"]:
            reason = next((str(row["grammar"].get("reason", "")) for row in corpus["rows"]
                           if row["family"] == family), "")
            print(f"  {family:16s} {cell['targets']:>3d}  grammar abstains on all of "
                  f"them: {reason[:48]}")
            continue
        own, other = depth["grammar"], depth["qiskit"]
        ratio, wins = depth["ratio_qiskit_over_grammar"], depth["wins"]
        print(f"  {family:16s} {cell['matched']:>3d}  "
              f"{own['median']:>8.1f} [{own['q1']:.0f}-{own['q3']:.0f}]  "
              f"{other['median']:>8.1f} [{other['q1']:.0f}-{other['q3']:.0f}]  "
              f"{ratio['median']:>6.2f} [{ratio['q1']:.2f}-{ratio['q3']:.2f}]  "
              f"{wins['grammar']:>3d}/{wins['qiskit']:>3d}/{wins['tie']:>3d}")
    print("  (median [interquartile range] over the instances in each family; the "
          "record carries the full rows)")
    if depth_context["status"] == "measured":
        print(f"\nqiskit corpus depth histogram ({depth_context['overall']['n']} rows, "
              f"bucket width {depth_context['bucket_width']})")
        for bucket in depth_context["buckets"]:
            bar = "#" * bucket["count"]
            print(f"  {bucket['low']:>3d}-{bucket['high']:<3d} {bucket['count']:>3d} {bar}")
        print(f"  {'target':22s} {'q':>4s} {'depth':>5s}  {'cell':>5s} {'n':>3s}  "
              f"{'min/median/max':>16s}  {'<= depth':>9s}")
        for row in depth_placed:
            place = row["placement"]
            star = "" if place["width_match"] == "exact" else "*"
            cell = f"{place['cell_qubits']}q{star}"
            print(f"  {row['target']:22s} {row['qubits']:>4d} {row['depth']:>5d}  "
                  f"{cell:>5s} {place['n']:>3d}  "
                  f"{place['min']:>4d}/{place['median']:>6.1f}/{place['max']:>4d}  "
                  f"{place['qiskit_depths_at_or_below']:>4d} "
                  f"({place['share_at_or_below']:.0%})")
        print("  (* nearest corpus width, not a width match: context only)")
    routing = corpus["routing"]
    if routing.get("status") == "unavailable":
        print(f"\ncorpus routing: abstention -- {routing['reason']}")
    else:
        print(f"\ncorpus routed with each engine's own router on "
              f"{routing['device']['name']} ({routing['device']['num_qubits']}q, "
              f"seed {routing['seed_transpiler']}, optimisation "
              f"{routing['optimization_level']})")
        print(f"  {'family':16s} {'n':>3s}  {'apedp':>7s}  {'qiskit':>7s}  "
              f"{'ratio q/a':>18s}  wins a/q/t")
        for family, block in routing["summary"]["by_family"].items():
            if not block["matched"]:
                continue
            ratio, wins = block["depth_ratio_qiskit_over_apedp"], block["wins"]
            print(f"  {family:16s} {block['matched']:>3d}  "
                  f"{block['apedp_routed_depth']['median']:>7.1f}  "
                  f"{block['qiskit_routed_depth']['median']:>7.1f}  "
                  f"{ratio['median']:>6.2f} [{ratio['q1']:.2f}-{ratio['q3']:.2f}]  "
                  f"{wins['apedp']:>3d}/{wins['qiskit']:>3d}/{wins['tie']:>3d}")
        print("  (routed depth before this package's restore ladder, so the numbers "
              "are comparable; the restore cost is in the record)")
    third = corpus["third_party"]
    if third["synthesis"] is None:
        print(f"\nthird-party referee: {third['status']} -- the corpus rests on this "
              f"package's referee alone")
    else:
        synthesis_third, routed_third = third["synthesis"], third["routed"]
        print(f"\nthird-party referee: {third['referee']} "
              f"({synthesis_third['version']}) -- "
              f"{synthesis_third['valid']}/{synthesis_third['decisions']} synthesis "
              f"decisions VALID"
              + (f", {routed_third['valid']}/{routed_third['decisions']} routed "
                 f"decisions VALID" if routed_third else "")
              + f"; disagreements with our referee: "
                f"{len(synthesis_third['disagreements_with_our_referee'])}")
    print("\nsampling: one specification, both engines, the exact distribution as referee")
    for row in sampling["rows"]:
        own = row["grammar"]
        if row["comparison"] is None:
            print(f"  {row['label']:24s} stim unavailable: "
                  f"{str(row['stim'].get('reason'))[:60]}")
            continue
        foreign = row["stim"]
        print(f"  {row['label']:24s} TV {own['tv_to_exact']:.4f} (band {own['band']:.4f}) "
              f"vs {foreign['tv_to_exact']:.4f} (band {foreign['band']:.4f})  "
              f"pair {row['comparison']['tv']:.4f} "
              f"(band {row['comparison']['band']:.4f})  "
              f"{own['samples_per_second']:,.0f} vs "
              f"{foreign['samples_per_second']:,.0f} samples/s")
    print("\nwide sampling: exact marginals as the referee, above the enumerable width")
    print(f"  {'target':26s} {'q':>3s} {'subsets':>8s} {'grammar':>12s} "
          f"{'stim':>12s} {'pair':>10s} {'uniform':>8s}")
    for row in wide_sampling["rows"]:
        own = row["grammar"]
        foreign = row["stim"] if row["comparison"] is not None else None
        print(f"  {row['label']:26s} {int(row['qubits']):>3d} "
              f"{len(own['subsets']):>8d} "
              f"{own['verdict']:>12s} "
              f"{(foreign['verdict'] if foreign else 'unavailable'):>12s} "
              f"{(row['comparison']['verdict'] if foreign else '-'):>10s} "
              f"{len(own['uniform_subsets']):>8d}")
    print("\ndevice routing: each tool's own router on a recorded coupling map")
    if device_routing.get("status") == "unavailable":
        print(f"  unavailable: {device_routing['reason']}")
    else:
        print(f"  {device_routing['device']['name']} "
              f"({device_routing['device']['num_qubits']}q, "
              f"{device_routing['device']['edges']} edges), seed "
              f"{device_routing['seed_transpiler']}, optimisation "
              f"{device_routing['optimization_level']}")
        for row in device_routing["rows"]:
            if row["comparison"] is None:
                print(f"  {row['circuit']:20s} qiskit unavailable: "
                      f"{str(row['qiskit'].get('reason'))[:60]}")
                continue
            own = row["qiskit"]
            print(f"  {row['circuit']:20s} qiskit {own['routed_depth']:>6} "
                  f"swaps {own['routing_swaps']:>5} | greedy "
                  f"{row['ours']['greedy']['depth_before_restore']:>6} "
                  f"(total {row['ours']['greedy']['mapped_depth']:>6}) | "
                  f"lookahead {row['ours']['lookahead']['depth_before_restore']:>6} "
                  f"(total {row['ours']['lookahead']['mapped_depth']:>6})")
    print("\nequivalence: the same pairs for both referees, from constructions that fix "
          "the answer")
    for row in equivalence["rows"]:
        foreign = row["qcec"]
        shown = foreign["verdict"] if foreign["status"] == "compared" else "unavailable"
        if shown == "unstable":
            shown = "unstable " + "/".join(sorted(set(foreign["criteria"])))
        print(f"  {row['label']:32s} {row['question']:11s} "
              f"ours {row['ours'][row['question']]['verdict']:11s} "
              f"qcec {shown:34s} {row['agreement']['outcome']}")
    print("  " + equivalence_verdict(equivalence))
    print(f"\nwritten: {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
