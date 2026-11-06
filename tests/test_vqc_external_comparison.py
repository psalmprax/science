"""External comparison: our referee on a foreign circuit, and abstention done honestly.

The review's first ask before any claim of differentiation was a matched comparison
with the tools that already do part of this. What that comparison is allowed to say is
bounded by two rules this file pins:

* **a foreign circuit is decided before it is quoted.** A submission carries the other
  tool's own operation count, and our specification referee decides the circuit. An
  unverified gate count is not evidence, and a wrong circuit is `INVALID` here no
  matter what it claims;
* **a tool that did not run is not a tool that lost.** An absent or unwired tool
  produces an abstention with a reason and is never counted as a comparison.

What is pinned about the record itself: every task shares one specification hash, every
non-comparative tool has a reason, the capability asymmetry (a random stabilizer state,
where this package abstains and Qiskit does not) is present, the measured ratios are
recomputable from the rows, and the prose verdict does not claim a general advantage.
The shared corpus gets its own pins: a celled *sample* neither engine is specialised
to -- every graph family at every declared width with the declared replicates, plus
one abstention anchor per anchor width -- with every row on one specification, every
circuit `VALID`, and a summary that recomputes from the rows. The summary is checked
as a distribution rather than as a headline: the per-family quartiles, ranges and
width breakdowns must be recomputable from the rows, and the two-qubit counts must
agree instance by instance, because on a graph state neither engine wins on gate count
and a single row would hide how much the depth result moves across draws.

The Qiskit-dependent tests are skipped when it is not installed. That is not a gap in
the pins: the record and its abstentions are pinned either way, and the adapter's own
behaviour is pinned against a stub when the real package is absent.

The Stim rows pin the execution task: both engines are handed the same signed
stabilizers, each sample set is decided against the **exact** distribution, and the two
sets are compared with each other under a null where both samplers are correct. A
sample set that is inconsistent must be reported as `INCONSISTENT`, not smoothed over,
and a width the exact referee cannot enumerate must be refused rather than compared
against an approximation.

The wide rows are the answer to that refusal: above 12 qubits the referee is the exact
marginal of chosen subsets of wires, derived from the specification by linear algebra
and therefore by neither sampler. Those pins are narrower on purpose -- every row must
be *above* the enumerable width, every subset decision must carry the Bonferroni-
corrected p-value it was taken at, no subset may abstain silently, and the graph-state
row must show the blind spot (every marginal uniform) rather than being left out. A
marginal decides a projection, so the record's prose is pinned not to claim more: the
full distribution is still only decided at 12 qubits.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.external import (
    TOOLS,
    Sampling,
    SamplingUnsupported,
    Submission,
    ToolUnavailable,
    adjudicate_sampling,
    adjudicate_submission,
    apedp_sampling,
    probe,
    qiskit_routing,
    qiskit_synthesis,
    relabelled_target,
    sampling_pair,
    stim_sampling,
    tool_statuses,
)
from apedp_dyn.vqc.placement import line, place
from apedp_dyn.vqc.stabilizer import I, X, Z
from apedp_dyn.vqc.synthesis import (
    SpecificationCandidate,
    TargetSpecification,
    adjudicate_specification,
    adjudicate_specification_candidate,
    build,
    ghz_target,
    ring_graph_target,
)

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_external_comparison.json"
SCRIPT = ROOT / "scripts" / "compare_vqc_external_tools.py"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


# ------------------------------------------------------- a foreign circuit is decided
def test_a_foreign_submission_is_decided_by_our_own_referee():
    target = ghz_target(6)
    # Stand in for another tool's output with a construction we can reason about.
    submission = Submission(tool="fixture", variant="chain",
                            circuit=build("chain", target).circuit,
                            claims={"operations": 6}, version="0")
    row = adjudicate_submission(submission, target)
    assert row.verdict == "VALID" and row.tier == "stabilizer"
    assert row.claim_verdict == "MET"
    assert row.verified is True
    assert row.operations == 6 and row.two_qubit_gates == 5
    payload = row.as_dict()
    assert payload["verification"]["verified_by"].startswith("apedp_dyn.vqc.synthesis")
    assert payload["circuit_sha256"] == submission.circuit.sha256()


def test_a_foreign_claim_the_circuit_does_not_meet_is_unmet_not_invalid():
    """The same semantic/resource split applies to a competitor as to a local candidate."""
    target = ghz_target(6)
    submission = Submission(tool="fixture", variant="chain",
                            circuit=build("chain", target).circuit,
                            claims={"two_qubit_gates": 1})
    row = adjudicate_submission(submission, target)
    assert row.verdict == "VALID"
    assert row.claim_verdict == "UNMET"
    # A valid circuit that misses its own advertised count is not promoted.
    assert row.promotable is False
    assert row.claim_verdict != row.verdict


def test_a_wrong_foreign_circuit_is_invalid_with_the_failing_generator():
    target = ghz_target(6)
    ops = list(build("chain", target).circuit.ops)
    index = next(i for i, op in enumerate(ops) if getattr(op, "name", None) == "cx")
    ops[index] = Gate("cz", ops[index].targets)
    wrong = Circuit(ops=tuple(ops), num_qubits=6, num_bits=0)
    row = adjudicate_submission(Submission(tool="fixture", variant="mutated", circuit=wrong),
                                target)
    assert row.verdict == "INVALID"
    assert row.checks_failed
    assert "the prepared state is the +1 eigenstate" in row.checks_failed[0]


def test_the_same_four_field_policy_is_what_decides_a_submission():
    """`adjudicate_submission` is a wrapper, not a second policy."""
    target = ghz_target(4)
    circuit = build("chain", target).circuit
    candidate = SpecificationCandidate(candidate_id="x", circuit=circuit, specification=target,
                                       claims={"operations": 4})
    verdict = adjudicate_specification_candidate(candidate)
    row = adjudicate_submission(Submission(tool="fixture", variant="chain", circuit=circuit,
                                           claims={"operations": 4}), target)
    assert (row.verdict, row.claim_verdict, row.evidence, row.promotable) == (
        verdict.verdict, verdict.claim_verdict, verdict.evidence, verdict.promotable)


# --------------------------------------------------------- the string-order finding
def test_the_string_order_difference_is_pinned_on_a_non_palindromic_target():
    """`X` on qubit 0 and `Z` on qubit 1: a mirror image is a different state.

    Qiskit reads the first character of a stabilizer string as the *highest*-indexed
    qubit; this package reads it as qubit 0. Unconverted, a GHZ and a ring state still
    verify -- both are palindromic -- so the discrepancy is invisible on exactly the
    targets a comparison would start with. This test uses a target where it is not.
    """
    pytest.importorskip("qiskit")
    target = TargetSpecification(name="x0z1", num_qubits=2,
                                 stabilizers=(((X, I), 0), ((I, Z), 0)))
    row = adjudicate_submission(qiskit_synthesis(target), target)
    assert row.verdict == "VALID", row.checks_failed

    # And the naive order really does produce the mirror image, which is why the
    # adapter reverses: this is the bug the test protects.
    from qiskit import qasm2
    from qiskit.synthesis import synth_circuit_from_stabilizers

    from apedp_dyn.vqc.qasm import from_qasm

    naive = from_qasm(qasm2.dumps(synth_circuit_from_stabilizers(
        list(target.stabilizer_strings()))))
    assert adjudicate_specification(naive, target).verdict == "INVALID"


def test_the_adapter_reverses_and_keeps_the_sign():
    from apedp_dyn.vqc.external import _qiskit_strings

    # A commuting, independent set that is *not* palindromic, with one negative sign.
    target = TargetSpecification(name="signed", num_qubits=3,
                                 stabilizers=(((X, I, Z), 0), ((I, Z, I), 2),
                                              ((Z, Z, X), 0)))
    strings = _qiskit_strings(target)
    assert strings[0] == "+ZIX"
    assert strings[1] == "-IZI"
    assert strings[2] == "+XZZ"


# ------------------------------------------------------------------- the abstentions
def test_a_tool_that_did_not_run_is_an_abstention_never_a_loss():
    for status in tool_statuses():
        assert status.name in TOOLS
        if status.status == "compared":
            continue
        assert status.reason, status.name
        assert status.comparative is False
        assert status.status in {"unavailable", "installed_not_wired"}


def test_an_absent_package_is_named_and_never_silently_skipped(monkeypatch):
    from apedp_dyn.vqc import external

    monkeypatch.setattr(external, "_installed",
                        lambda package: (False, None, f"{package!r} is not installed"))
    status = probe("qiskit")
    assert status.status == "unavailable"
    assert status.installed is False and status.version is None
    assert "not installed" in status.reason
    assert status.comparative is False
    # Installing but not wiring is a *different* state, and it says so.
    monkeypatch.setattr(external, "_installed", lambda package: (True, "9.9.9", "installed"))
    monkeypatch.setitem(external.TOOLS["stim"], "wired", False)
    assert probe("stim").status == "installed_not_wired"
    monkeypatch.setitem(external.TOOLS["stim"], "wired", True)
    assert probe("stim").status == "compared"


def test_a_missing_qiskit_is_a_refusal_with_a_reason(monkeypatch):
    from apedp_dyn.vqc import external

    def unavailable():
        raise ToolUnavailable("qiskit is not importable (ImportError: test)")

    monkeypatch.setattr(external, "_import_qiskit", unavailable)
    with pytest.raises(ToolUnavailable, match="not importable"):
        qiskit_synthesis(ghz_target(3))


def test_an_unwired_variant_is_refused_with_the_reason_it_is_unwired():
    pytest.importorskip("qiskit")
    with pytest.raises(ToolUnavailable, match="outside this package's gate table"):
        qiskit_synthesis(ghz_target(4), variant="lnn")


# --------------------------------------------------------------------- the record
def test_the_record_is_the_schema_it_claims_and_every_check_passed(record):
    assert record["schema"] == "apedp.vqc.external_comparison.v8"
    assert record["checks"] and all(record["checks"].values()), record["checks"]


def test_every_task_gives_both_engines_one_specification(record):
    for task in record["tasks"]:
        assert task["target_sha256"]
        assert {row.get("target_sha256") for row in task["rows"]} == {task["target_sha256"]}
        compared = [row for row in task["rows"] if row["status"] == "compared"]
        assert compared
        for row in compared:
            assert row["verification"]["verdict"] == "VALID", (task["task"], row["engine"])
            assert row["verification"]["verified_by"].startswith("apedp_dyn.vqc.synthesis")


def test_no_unavailable_tool_is_counted_as_a_comparison(record):
    statuses = record["tools"]
    assert {status["tool"] for status in statuses} == set(TOOLS)
    for status in statuses:
        assert status["reason"]
        if status["status"] != "compared":
            assert status["counts_as_a_comparison"] is False
    # At least one tool was actually compared, or the record is a page of abstentions.
    assert any(status["counts_as_a_comparison"] for status in statuses)
    assert record["checks"]["at_least_one_foreign_circuit_ran"] is True


def test_the_capability_asymmetry_is_recorded_rather_than_omitted(record):
    task = next(task for task in record["tasks"]
                if "no local construction family" in task["task"])
    abstained = [row for row in task["rows"] if row["status"] == "abstained"]
    compared = [row for row in task["rows"] if row["status"] == "compared"]
    assert len(abstained) == 1 and len(compared) == 1
    assert abstained[0]["engine"] == "apedp.synthesis"
    assert "no construction family" in abstained[0]["reason"]
    assert compared[0]["engine"] != "apedp.synthesis"
    assert compared[0]["verification"]["verdict"] == "VALID"
    assert compared[0]["metrics"]["two_qubit_gates"] > 0


def test_the_measured_ratios_are_recomputable_from_the_rows(record):
    measured = {row["target"]: row for row in record["advantage"]["measured"]}
    assert set(measured) == {"ghz", "ring_graph"}
    for row in measured.values():
        grammar, other = row["grammar"], row["qiskit"]
        assert row["two_qubit_gate_ratio_qiskit_over_grammar"] == pytest.approx(
            other["two_qubit_gates"] / max(1, grammar["two_qubit_gates"]))
        assert row["depth_ratio_qiskit_over_grammar"] == pytest.approx(
            other["depth"] / max(1, grammar["depth"]))
        assert row["agrees_on_state"] is True
    # The advantage is target-specific, and the record's own numbers say so: on GHZ it
    # is large, on the ring state a general synthesizer reaches the same count.
    assert measured["ghz"]["two_qubit_gate_ratio_qiskit_over_grammar"] > 10
    assert measured["ring_graph"]["two_qubit_gate_ratio_qiskit_over_grammar"] == pytest.approx(1.0)


def test_the_verdict_does_not_claim_a_general_advantage(record):
    advantage = record["advantage"]
    assert advantage["claimed"] is False
    assert "grammar" in advantage["why"]
    assert advantage["what_would_change_it"]
    verdict = record["verdict"]
    assert "target-specific" in verdict["target_dependence"]
    assert "abstains" in verdict["asymmetry"]
    # The synthesis sentence is built from the measured block, not from prose.
    assert str(measured_two_qubit := record["advantage"]["measured"][-1]["qiskit"][
        "two_qubit_gates"]) in verdict["synthesis"] or measured_two_qubit > 0
    assert "two-qubit gates against" in verdict["synthesis"]


def test_the_limitations_name_what_was_not_compared(record):
    text = " ".join(record["limitations"]).lower()
    for topic in ("qiskit's synthesizer is general",
                  "mqt qcec is compared on the equivalence task only",
                  "the line topology task still routes qiskit's circuit",
                  "one seed and one optimisation level", "single measurements",
                  "stim comparison is sampling only", "python per-shot tableau clone"):
        assert topic in text, topic


# -------------------------------------------------------------------- the routing
def test_qiskit_routes_onto_a_given_coupling_map_and_is_decided():
    """Qiskit's own transpiler, a given device graph, and our referee on the result."""
    pytest.importorskip("qiskit")
    target = ghz_target(4)
    logical = build("chain", target).circuit
    edges = [(index, index + 1) for index in range(5)]      # a six-node line
    routing = qiskit_routing(logical, target=target, num_qubits=6, edges=edges,
                             device="line6", seed=7)
    assert routing.qubits == 6
    assert routing.logical_qubits == 4
    assert routing.logical_depth == logical.resources()["depth"]
    # Qiskit keeps its layout; the same circuit is decided by relabelling the target.
    assert len(set(routing.final_layout)) == 4
    decided = relabelled_target(target, routing.final_layout, 6)
    assert adjudicate_specification(routing.circuit, decided).verdict == "VALID"
    # The depth quoted is measured on the re-imported circuit, and Qiskit agrees.
    assert routing.reported_depth == routing.routed_depth
    assert routing.routed_depth == routing.circuit.resources()["depth"]


def test_a_circuit_wider_than_the_device_is_refused():
    pytest.importorskip("qiskit")
    target = ghz_target(4)
    with pytest.raises(ToolUnavailable, match="device has 3"):
        qiskit_routing(build("chain", target).circuit, target=target, num_qubits=3,
                       edges=[(0, 1), (1, 2)], device="tiny")


def test_a_layout_that_is_not_a_permutation_is_refused():
    target = ghz_target(3)
    with pytest.raises(ValueError, match="entries"):
        relabelled_target(target, (0, 1), 4)
    with pytest.raises(ValueError, match="injective"):
        relabelled_target(target, (0, 0, 2), 4)
    with pytest.raises(ValueError, match="injective"):
        relabelled_target(target, (0, 1, 9), 4)


# ------------------------------------------------------------------- the sampling
def test_stim_samples_the_same_specification():
    """Same signed stabilizers, same shots, two samplers, one exact referee."""
    pytest.importorskip("stim")
    target = ghz_target(8)
    shots = 4_000
    own = apedp_sampling(target, shots=shots, seed=11)
    foreign = stim_sampling(target, shots=shots, seed=12)
    assert own.target_sha256 == foreign.target_sha256 == target.sha256()
    assert own.num_qubits == foreign.num_qubits == 8
    assert sum(own.counts.values()) == sum(foreign.counts.values()) == shots
    for samples in (own, foreign):
        row = adjudicate_sampling(samples, target)
        assert row["verdict"] == "CONSISTENT", (samples.tool, row)
        assert row["tv_to_exact"] <= row["band"]
        assert row["band"] == adjudicate_sampling(samples, target)["band"]
        assert row["counts_sha256"] == samples.counts_sha256()
    assert sampling_pair(own, foreign, target)["verdict"] == "AGREE"
    # Both adapters are deterministic under their own seeds, so the record's count
    # hash is a citation of the run rather than a snapshot of one process.
    assert apedp_sampling(target, shots=shots, seed=11).counts_sha256() == own.counts_sha256()
    assert stim_sampling(target, shots=shots, seed=12).counts_sha256() == foreign.counts_sha256()


def test_a_sample_set_that_is_not_the_state_is_inconsistent():
    """A skewed sample set is reported as such; this is the check with teeth."""
    target = ring_graph_target(4)
    shots = 4_000
    skewed = Sampling(tool="fixture", variant="skewed", version="0",
                      target_sha256=target.sha256(), num_qubits=4, shots=shots,
                      seed=0, counts={(False,) * 4: shots},
                      preparation_seconds=0.0, sampling_seconds=0.0)
    row = adjudicate_sampling(skewed, target)
    assert row["verdict"] == "INCONSISTENT"
    assert row["tv_to_exact"] > row["band"]


def test_sampling_is_refused_against_a_different_specification():
    own = apedp_sampling(ghz_target(4), shots=200, seed=1)
    with pytest.raises(ValueError, match="different specification"):
        adjudicate_sampling(own, ghz_target(5))


def test_a_width_the_exact_referee_cannot_enumerate_is_refused():
    """Above 12 qubits there is no exact distribution, so there is no comparison."""
    target = ghz_target(13)
    samples = apedp_sampling(target, shots=10, seed=0)
    with pytest.raises(SamplingUnsupported, match="12 qubits"):
        adjudicate_sampling(samples, target)


def test_a_missing_stim_is_a_refusal_with_a_reason(monkeypatch):
    from apedp_dyn.vqc import external

    def unavailable():
        raise ToolUnavailable("stim is not importable (ImportError: test)")

    monkeypatch.setattr(external, "_import_stim", unavailable)
    with pytest.raises(ToolUnavailable, match="not importable"):
        stim_sampling(ghz_target(4), shots=10)


# ------------------------------------------------------------------ shared corpus
def test_the_shared_corpus_is_both_matched_and_abstaining(record):
    corpus = record["shared_corpus"]
    rows = corpus["rows"]
    design = corpus["design"]
    assert corpus["summary"]["targets"] == len(rows) == design["expected_targets"]
    matched = [row for row in rows if row["comparison"] is not None]
    abstained = [row for row in rows if row["grammar"]["status"] != "compared"]
    assert len(matched) == corpus["summary"]["matched"]
    assert len(matched) + len(abstained) == len(rows)
    assert {row["label"] for row in abstained} == set(corpus["summary"]["grammar_abstained"])
    assert abstained, "a corpus that only ran the favourable case is not a comparison"
    # Every graph cell was actually drawn, and the abstention anchors are the only
    # family without a spread of its own.
    for family, cell in corpus["summary"]["by_family"].items():
        if family == "random_stabilizer":
            assert cell["matched"] == 0
            continue
        assert cell["matched"] == (len(design["widths"])
                                   * design["replicates_per_cell"])
        assert cell["replicates_per_width"] == [design["replicates_per_cell"]]
    assert set(corpus["summary"]["families"]) == {
        entry["family"] for entry in design["graph_families"]} | {"random_stabilizer"}
    for row in rows:
        assert row["grammar"]["target_sha256"] == row["target_sha256"]
        assert row["qiskit"]["target_sha256"] == row["target_sha256"]
    for row in matched:
        assert row["grammar"]["verification"]["verdict"] == "VALID"
        assert row["qiskit"]["verification"]["verdict"] == "VALID"
        assert row["comparison"]["agrees_on_state"] is True
    for row in abstained:
        assert row["grammar"]["reason"]
        assert "metrics" not in row["grammar"]


def _matching(rows: list[dict], *, family: str | None = None) -> list[dict]:
    return [row for row in rows if row["comparison"] is not None
            and (family is None or row["family"] == family)]


def _check_wins_and_spread(block: dict, matched: list[dict]) -> None:
    """One summary block against the rows it was computed from."""
    assert block["matched"] == len(matched)
    if not matched:
        # An abstaining family has a block with counts and no distribution: nothing was
        # compared, and a fabricated zero spread would be worse than an empty one.
        for metric in ("operations", "two_qubit_gates", "depth"):
            assert block["per_metric"][metric]["ratio_qiskit_over_grammar"] is None
            assert block["per_metric"][metric]["ratio_median_bootstrap_95"] is None
        return
    for metric in ("operations", "two_qubit_gates", "depth"):
        spread = block["per_metric"][metric]
        ratios = [row["comparison"]["ratios_qiskit_over_grammar"][metric]
                  for row in matched]
        for engine in ("grammar", "qiskit", "tie"):
            assert spread["wins"][engine] == sum(
                1 for row in matched
                if row["comparison"]["winners"][metric] == engine)
        assert sum(spread["wins"].values()) == len(matched)
        ratio = spread["ratio_qiskit_over_grammar"]
        assert ratio["n"] == len(matched)
        assert ratio["median"] == pytest.approx(statistics.median(ratios))
        assert ratio["min"] == pytest.approx(min(ratios))
        assert ratio["max"] == pytest.approx(max(ratios))
        if len(ratios) > 1:
            q1, median, q3 = statistics.quantiles(sorted(ratios), n=4,
                                                  method="inclusive")
            assert ratio["q1"] == pytest.approx(q1)
            assert ratio["q3"] == pytest.approx(q3)
            assert ratio["median"] == pytest.approx(median)
            assert ratio["iqr"] == pytest.approx(q3 - q1)
            assert ratio["min"] <= ratio["q1"] <= ratio["median"] <= ratio["q3"]
            assert ratio["q3"] <= ratio["max"]
            interval = spread["ratio_median_bootstrap_95"]
            assert interval is not None
            assert interval[0] <= ratio["median"] <= interval[1]
            assert ratio["min"] <= interval[0] <= interval[1] <= ratio["max"]
        for engine in ("grammar", "qiskit"):
            values = [row[engine]["metrics"][metric] for row in matched]
            assert spread[engine]["n"] == len(values)
            assert spread[engine]["median"] == pytest.approx(statistics.median(values))
            assert spread[engine]["min"] == min(values)
            assert spread[engine]["max"] == max(values)


def test_the_corpus_summary_recomputes_from_its_rows(record):
    rows = record["shared_corpus"]["rows"]
    summary = record["shared_corpus"]["summary"]
    matched = _matching(rows)
    _check_wins_and_spread(summary["overall"], matched)
    for family, block in summary["by_family"].items():
        _check_wins_and_spread(block, _matching(rows, family=family))
    assert summary["grammar_total_two_qubit_gates"] == sum(
        row["grammar"]["metrics"]["two_qubit_gates"] for row in matched)
    assert summary["qiskit_total_two_qubit_gates"] == sum(
        row["qiskit"]["metrics"]["two_qubit_gates"] for row in matched)


def _check_routing_block(block: dict, rows: list[dict]) -> None:
    """One routed block against the rows it was computed from."""
    matched = [row for row in rows if row["comparison"] is not None]
    assert block["matched"] == len(matched)
    if not matched:
        assert block["depth_ratio_qiskit_over_apedp"] is None
        return
    ratios = [row["comparison"]["depth_ratio_qiskit_over_apedp"] for row in matched]
    spread = block["depth_ratio_qiskit_over_apedp"]
    assert spread["n"] == len(ratios)
    assert spread["median"] == pytest.approx(statistics.median(ratios))
    assert spread["min"] == pytest.approx(min(ratios))
    assert spread["max"] == pytest.approx(max(ratios))
    if len(ratios) > 1:
        q1, _median, q3 = statistics.quantiles(sorted(ratios), n=4,
                                                method="inclusive")
        assert spread["q1"] == pytest.approx(q1)
        assert spread["q3"] == pytest.approx(q3)
        assert spread["iqr"] == pytest.approx(q3 - q1)
        interval = block["ratio_median_bootstrap_95"]
        assert interval[0] <= spread["median"] <= interval[1]
    for engine in ("apedp", "qiskit", "tie"):
        assert block["wins"][engine] == sum(
            1 for row in matched if row["comparison"]["routed_depth_winner"] == engine)
    assert sum(block["wins"].values()) == len(matched)
    assert block["share_instances_qiskit_shallower"] == pytest.approx(
        block["wins"]["qiskit"] / len(matched))
    assert block["apedp_routed_depth"]["median"] == pytest.approx(
        statistics.median(row["ours"]["depth_before_restore"] for row in matched))
    assert block["qiskit_routed_depth"]["median"] == pytest.approx(
        statistics.median(row["qiskit"]["routed_depth"] for row in matched))
    assert block["apedp_swaps"]["median"] == pytest.approx(
        statistics.median(row["ours"]["routing_swaps"] for row in matched))
    assert block["qiskit_swaps"]["median"] == pytest.approx(
        statistics.median(row["qiskit"]["routing_swaps"] for row in matched))


def test_the_corpus_is_adjudicated_by_a_third_party_as_well(record):
    """Both engines' circuits get a second opinion, from code this package did not write.

    The record's own limitation used to be that both outputs were decided by this
    package's referee; that is a self-referential adjudication of a comparison one side
    of which is this package. The third-party block is the correction, and these pins
    are what stop it from becoming a name in a field: every matched row carries Stim's
    verdict for both circuits, and the two referees must agree.
    """
    third = record["shared_corpus"]["third_party"]
    assert third["status"] == "compared", third.get("reason")
    assert third["referee"].startswith("stim")
    assert third["module"] == "apedp_dyn.vqc.foreign_referee"
    assert third["what_it_decides"]
    assert third["what_it_does_not_decide"]
    summary = third["synthesis"]
    matched = record["shared_corpus"]["summary"]["matched"]
    assert summary["instances"] == matched
    assert summary["referee"] == "stim"
    assert summary["version"]
    assert summary["decisions"] == 2 * summary["instances"]
    assert summary["decided"] == summary["decisions"]
    assert summary["abstained"] == 0
    assert summary["valid"] == summary["decisions"]
    assert summary["invalid"] == 0
    assert summary["both_valid"] == summary["instances"]
    assert summary["undecided"] == []
    assert summary["not_both_valid"] == []
    assert summary["disagreements_with_our_referee"] == []
    for row in record["shared_corpus"]["rows"]:
        if row["comparison"] is None:
            assert "third_party" not in row      # nothing was built, so nothing is decided
            continue
        block = row["third_party"]
        for side in ("apedp", "qiskit"):
            verdict = block[side]
            assert verdict["referee"] == "stim"
            assert verdict["verdict"] == "VALID"
            assert verdict["generators"] == verdict["satisfied"]
            assert verdict["qubits"] == row["width"]
            assert verdict["failing"] == []
        assert block["both_valid"] is True
        assert block["agrees_with_our_referee"] == {"apedp": True, "qiskit": True}
        # The second opinion is about the very circuit the row recorded, not a rebuild.
        assert block["apedp"]["circuit_sha256"] == row["grammar"]["circuit_sha256"]
        assert block["qiskit"]["circuit_sha256"] == row["qiskit"]["circuit_sha256"]
        # Our referee reached the same verdict on the same circuit.
        assert row["grammar"]["verification"]["verdict"] == "VALID"
        assert row["qiskit"]["verification"]["verdict"] == "VALID"


def test_the_third_party_summary_recomputes_from_its_rows(record):
    third = record["shared_corpus"]["third_party"]
    for key, rows in (("synthesis", record["shared_corpus"]["rows"]),
                      ("routed", record["shared_corpus"]["routing"]["rows"])):
        blocks = [row["third_party"] for row in rows if row.get("third_party")]
        summary = third[key]
        assert summary is not None
        assert summary["instances"] == len(blocks)
        assert summary["decisions"] == 2 * len(blocks)
        assert summary["valid"] == sum(
            1 for block in blocks for side in ("apedp", "qiskit")
            if block[side]["verdict"] == "VALID")
        assert summary["invalid"] == sum(
            1 for block in blocks for side in ("apedp", "qiskit")
            if block[side]["verdict"] == "INVALID")
        assert summary["both_valid"] == sum(1 for block in blocks
                                            if block["both_valid"])
        assert summary["undecided"] == [
            row["label"] for row in rows if row.get("third_party")
            and any(row["third_party"][side]["verdict"] not in ("VALID", "INVALID")
                    for side in ("apedp", "qiskit"))]
        assert summary["disagreements_with_our_referee"] == [
            row["label"] for row in rows if row.get("third_party")
            and False in row["third_party"]["agrees_with_our_referee"].values()]


def test_the_routed_corpus_is_adjudicated_by_the_third_party_too(record):
    """The routed circuits are where a shared referee would matter most."""
    routed = record["shared_corpus"]["third_party"]["routed"]
    assert routed is not None
    rows = record["shared_corpus"]["routing"]["rows"]
    assert routed["instances"] == record["shared_corpus"]["routing"]["summary"][
        "matched"]
    assert routed["undecided"] == [] and routed["not_both_valid"] == []
    for row in rows:
        block = row["third_party"]
        # Ours is decided against the padded target, Qiskit's against the relabelled one,
        # so a 127-qubit router row is still decided on every wire.
        assert len(block["apedp"]["circuit_sha256"]) == 64
        assert len(block["qiskit"]["circuit_sha256"]) == 64
        assert block["apedp"]["circuit_sha256"] != row["logical_circuit_sha256"]
        assert block["apedp"]["qubits"] == block["qiskit"]["qubits"]
        assert block["apedp"]["generators"] == block["apedp"]["qubits"]
        assert block["apedp"]["verdict"] == "VALID"
        assert block["qiskit"]["verdict"] == "VALID"
    assert "third party" in " ".join(record["limitations"])
    assert "bridge" in " ".join(record["limitations"])


def test_the_routed_corpus_uses_each_engines_own_router(record):
    """The sample routed by the router each engine ships, not ours on both sides.

    The synthesis rows measure circuits in the abstract, and the topology task routes
    both circuits with this package's router -- which measures this package's router.
    This block is the correction for the sample: Qiskit's own transpiler against this
    package's lookahead router on the same recorded coupling map, every routed circuit
    decided by this package's referee (Qiskit's against the target relabelled by its
    final layout), and the routed depths reported as a distribution.
    """
    corpus = record["shared_corpus"]
    routing = corpus["routing"]
    assert routing["status"] == "compared", routing.get("reason")
    assert routing["device"]["source"].startswith("qiskit_ibm_runtime")
    assert routing["device"]["num_qubits"] >= 100
    assert routing["router_ours"].startswith("apedp.placement")
    assert routing["router_qiskit"].startswith("qiskit")
    rows = routing["rows"]
    matched_synthesis = [row for row in corpus["rows"]
                         if row["comparison"] is not None]
    assert len(rows) == routing["summary"]["matched"] == len(matched_synthesis)
    by_label = {row["label"]: row for row in matched_synthesis}
    for row in rows:
        # The routed circuit is the one the synthesis row measured, not a rebuild.
        assert row["logical_circuit_sha256"] == by_label[row["label"]]["grammar"][
            "circuit_sha256"]
        assert row["logical_target_sha256"] == by_label[row["label"]]["target_sha256"]
        assert row["ours"]["router"] == routing["router_ours"]
        assert row["qiskit"]["engine"] == "qiskit"
        assert row["qiskit"]["variant"] == "transpile.sabre"
        assert row["qiskit"]["seed_transpiler"] == routing["seed_transpiler"]
        assert row["qiskit"]["optimization_level"] == routing["optimization_level"]
        assert row["ours"]["verification"]["verdict"] == "VALID"
        assert row["qiskit"]["verification"]["verdict"] == "VALID"
        assert row["comparison"]["agrees_on_state"] is True
        # Ours restores its layout, so the comparable number is the pre-restore depth.
        assert row["ours"]["mapped_depth"] >= row["ours"]["depth_before_restore"]
        assert row["ours"]["routing_swaps"] > 0
        expected = ("apedp" if row["ours"]["depth_before_restore"]
                    < row["qiskit"]["routed_depth"]
                    else "qiskit" if row["qiskit"]["routed_depth"]
                    < row["ours"]["depth_before_restore"] else "tie")
        assert row["comparison"]["routed_depth_winner"] == expected


def test_the_corpus_routing_summary_recomputes_from_its_rows(record):
    routing = record["shared_corpus"]["routing"]
    rows = routing["rows"]
    summary = routing["summary"]
    _check_routing_block({"matched": summary["matched"],
                          "wins": summary["wins"],
                          "share_instances_qiskit_shallower": summary[
                              "share_instances_qiskit_shallower"],
                          "depth_ratio_qiskit_over_apedp": summary[
                              "depth_ratio_qiskit_over_apedp"],
                          "ratio_median_bootstrap_95": summary[
                              "ratio_median_bootstrap_95"],
                          "apedp_routed_depth": {"median": statistics.median(
                              row["ours"]["depth_before_restore"] for row in rows)},
                          "qiskit_routed_depth": {"median": statistics.median(
                              row["qiskit"]["routed_depth"] for row in rows)},
                          "apedp_swaps": {"median": statistics.median(
                              row["ours"]["routing_swaps"] for row in rows)},
                          "qiskit_swaps": {"median": statistics.median(
                              row["qiskit"]["routing_swaps"] for row in rows)}},
                         rows)
    for family, block in summary["by_family"].items():
        _check_routing_block(block, [row for row in rows if row["family"] == family])
    for width, block in summary["by_width"].items():
        _check_routing_block(block, [row for row in rows
                                     if f"{row['width']}q" == width])


def test_the_corpus_reports_spread_not_single_rows(record):
    """Every family carries a distribution, and the width trend is broken out.

    The review of the seven-instance corpus was that one row per family lets the
    flattering draw stand in for the family. This pins the replacement: quartiles,
    range and the per-instance share for each sampled family, plus a per-width
    median, so a single number cannot be read off the record without its spread.
    """
    corpus = record["shared_corpus"]
    summary = corpus["summary"]
    design_widths = corpus["design"]["widths"]
    family_blocks = [block for family, block in summary["by_family"].items()
                     if family != "random_stabilizer"]
    assert len(family_blocks) == 3
    for block in family_blocks:
        depth = block["per_metric"]["depth"]
        assert depth["ratio_qiskit_over_grammar"]["n"] == block["matched"] > 1
        assert depth["ratio_qiskit_over_grammar"]["iqr"] >= 0
        assert 0.0 <= depth["share_instances_grammar_lower"] <= 1.0
        assert set(block["by_width"]) == {f"{width}q" for width in design_widths}
        for cell in block["by_width"].values():
            assert cell["matched"] == corpus["design"]["replicates_per_cell"]
            assert cell["median_depth"]["grammar"] > 0
            assert cell["median_depth"]["qiskit"] > 0
            assert cell["depth_ratio_qiskit_over_grammar"]["n"] == cell["matched"]
        assert block["structure"]["edges"]["n"] == block["matched"]
    overall = summary["overall"]["per_metric"]["depth"]
    assert overall["ratio_qiskit_over_grammar"]["n"] == summary["matched"]
    assert overall["ratio_median_bootstrap_95"] is not None


def test_the_corpus_does_not_show_a_gate_count_advantage(record):
    """On targets neither engine is specialised to, gate counts are a tie.

    This pins the committed record's result, not a law: both engines emit one entangler
    per edge of a graph state, and the grammar's only saving is depth. The record says
    the depth gap is a scheduling difference a transpiler pass could close, which is
    why it is not counted as synthesis superiority -- and it reports that gap with its
    spread rather than as a single row.
    """
    summary = record["shared_corpus"]["summary"]
    assert summary["two_qubit_gates_agree_on_every_matched_instance"] is True
    for block in [summary["overall"], *summary["by_family"].values()]:
        if not block["matched"]:
            continue
        wins = block["per_metric"]["two_qubit_gates"]["wins"]
        assert wins["tie"] == block["matched"]
        assert wins["grammar"] == wins["qiskit"] == 0
        assert block["per_metric"]["two_qubit_gates"][
            "ratio_qiskit_over_grammar"]["median"] == 1.0
    assert (summary["grammar_total_two_qubit_gates"]
            == summary["qiskit_total_two_qubit_gates"])
    assert "transpiler scheduling pass" in " ".join(record["limitations"])
    assert "schedule difference" in record["verdict"]["shared_corpus"]
    assert "third party" in record["advantage"]["what_would_change_it"]


def test_the_record_carries_the_matched_sampling_task(record):
    sampling = record["sampling"]
    assert sampling["shots_per_run"] == 40_000
    assert sampling["summary"]["targets"] == len(sampling["rows"]) == 3
    matched = [row for row in sampling["rows"] if row["comparison"] is not None]
    assert len(matched) == sampling["summary"]["matched"] == 3
    for row in matched:
        assert row["grammar"]["target_sha256"] == row["target_sha256"]
        assert row["stim"]["target_sha256"] == row["target_sha256"]
        assert row["grammar"]["verdict"] == "CONSISTENT"
        assert row["stim"]["verdict"] == "CONSISTENT"
        assert row["comparison"]["verdict"] == "AGREE"
        assert row["grammar"]["counts_sha256"] and row["stim"]["counts_sha256"]
        assert row["grammar"]["shots"] == row["stim"]["shots"] == 40_000
        assert row["grammar"]["samples_per_second"] > 0
        assert row["stim"]["samples_per_second"] > 0
    assert (sampling["summary"]["median_samples_per_second"]["stim"]
            > sampling["summary"]["median_samples_per_second"]["grammar"])
    assert record["reproducibility"]["stim_version"]


def test_the_record_carries_the_wide_sampling_block(record):
    """Above the enumerable width the referee is exact marginals, not an approximation."""
    wide = record["wide_sampling"]
    assert wide["shots_per_run"] == 20_000
    assert wide["alpha"] == 1e-3
    assert "chi-square" in wide["referee"]
    assert "Bonferroni" in wide["referee"]
    assert "a projection, not the distribution" in wide["what_it_does_not_decide"]
    assert "graph state" in wide["what_it_does_not_decide"]
    rows = wide["rows"]
    assert wide["summary"]["targets"] == len(rows) == 7
    assert all(row["qubits"] > record["sampling"]["rows"][0]["qubits"] for row in rows)
    for row in rows:
        assert row["qubits"] > 12
        assert row["shots"] == 20_000
        assert len(row["subsets"]) == len(row["grammar"]["subsets"]) > 12
        assert row["grammar"]["verdict"] == "CONSISTENT"
        assert row["grammar"]["target_sha256"] == row["target_sha256"]
        assert row["grammar"]["referee"] == "exact Z-basis marginals of the specification"
        assert row["stim"]["verdict"] == "CONSISTENT"
        assert row["comparison"]["verdict"] == "AGREE"
        for subset in row["grammar"]["subsets"]:
            # The correction is arithmetic: p multiplied by the family size, capped at 1.
            assert subset["family_p_value"] == min(
                1.0, subset["p_value"] * len(row["grammar"]["subsets"]))
            assert subset["verdict"] != "UNRESOLVED"
            assert subset["target_sha256"] == row["target_sha256"]
            assert subset["dof"] == subset["support"] - 1
            assert subset["min_expected"] >= 5.0
            assert subset["outcomes_outside_the_exact_support"] == 0
            assert _bits_in(subset["top_outcomes"], subset["subset_size"])
        for subset in row["comparison"]["subsets"]:
            assert subset["verdict"] == "AGREE"
            assert subset["family_p_value"] == min(
                1.0, subset["p_value"] * len(row["comparison"]["subsets"]))
    labels = {row["label"]: row for row in rows}
    # GHZ's pair marginals keep the correlation: a two-outcome marginal with p = 1/2.
    for label in ("ghz 16q", "ghz 24q", "ghz 32q"):
        summaries = labels[label]["grammar"]["subsets"]
        pairs = [s for s in summaries if s["subset_size"] == 2]
        assert len(pairs) == labels[label]["qubits"] - 1
        assert all(s["support"] == 2 for s in pairs)
        assert all(s["probability"] == pytest.approx(0.5) for s in pairs)
        assert labels[label]["grammar"]["uniform_subsets"] == []
    # The blind spot is recorded, not avoided: a graph state is uniform on every subset.
    ring = labels["ring_graph 32q"]["grammar"]
    assert len(ring["uniform_subsets"]) == len(ring["subsets"])
    assert all(subset["is_uniform"] for subset in ring["subsets"])
    assert wide["summary"]["uniform_subsets"] == sum(
        len(row["grammar"]["uniform_subsets"]) for row in rows)


def _bits_in(top_outcomes: list[dict], size: int) -> bool:
    return all(len(outcome["bits"]) == size for outcome in top_outcomes)


def test_the_wide_sampling_summary_recomputes_from_its_rows(record):
    """The wide block's numbers are derived from its rows, not asserted beside them."""
    wide = record["wide_sampling"]
    rows, summary = wide["rows"], wide["summary"]
    matched = [row for row in rows if row["comparison"] is not None]
    assert summary["matched"] == len(matched)
    assert summary["widest_qubits"] == max(row["qubits"] for row in rows)
    assert summary["subset_decisions"] == sum(len(row["grammar"]["subsets"])
                                              for row in rows)
    assert summary["grammar_consistent_subsets"] == sum(
        1 for row in rows for subset in row["grammar"]["subsets"]
        if subset["verdict"] == "CONSISTENT")
    assert summary["stim_consistent_subsets"] == sum(
        1 for row in rows for subset in row["stim"]["subsets"]
        if subset["verdict"] == "CONSISTENT")
    assert summary["pair_subset_disagreements"] == [
        {"label": row["label"], "subset": subset["subset"],
         "family_p_value": subset["family_p_value"]}
        for row in matched for subset in row["comparison"]["subsets"]
        if subset["verdict"] == "DISAGREE"]
    assert summary["unresolved_subsets"] == sum(
        len(row["grammar"]["unresolved_subsets"])
        + len(row["stim"]["unresolved_subsets"]) for row in rows)
    assert summary["smallest_family_p_value"] == {
        row["label"]: min(subset["family_p_value"] for subset in row["grammar"]["subsets"])
        for row in rows}
    assert summary["median_samples_per_second"]["stim"] > \
        summary["median_samples_per_second"]["grammar"]
    assert summary["stim_unavailable"] == []


def test_the_wide_prose_does_not_claim_the_whole_distribution(record):
    """A marginal decides a projection; the record must say so where it is read."""
    verdict = record["verdict"]["wide_sampling"]
    assert "marginal" in verdict
    assert "projection" in " ".join(record["limitations"])
    assert "graph state" in " ".join(record["limitations"])
    assert "whole outcome distribution" in record["advantage"]["what_would_change_it"]
    limitations = " ".join(record["limitations"])
    assert "16, 24 and 32" in limitations
    assert record["reproducibility"]["marginals_sha256"] == hashlib.sha256(
        (ROOT / "apedp_dyn/vqc/marginals.py").read_bytes()).hexdigest()


def _check_memory(row: dict, engine: str) -> None:
    """One memory measurement against the row whose samples it was taken on."""
    memory = row["memory"][engine]
    if memory.get("status") == "unavailable":
        assert memory["reason"]
        return
    assert memory["engine"] == engine
    assert memory["shots"] == row["shots"]
    assert memory["target_sha256"] == row["target_sha256"]
    # The number belongs to these samples, not to a similar workload: the child that
    # measured the memory also drew them, and its hash is the recorded one.
    if engine == "grammar" or row[engine] is not None:
        assert memory["counts_sha256"] == row[engine]["counts_sha256"]
    assert memory["growth_bytes"] == memory["window_peak_bytes"] - memory["floor_bytes"]
    assert memory["floor_bytes"] > 0
    assert memory["growth_bytes"] >= 0
    assert memory["peak_reset"] is True
    assert "VmHWM" in memory["method"] and "reset" in memory["method"]
    assert memory["growth_mib"] == pytest.approx(memory["growth_bytes"] / 2 ** 20)
    # The floor is the interpreter and numpy, not the engine: it must dwarf nothing and
    # must be reported beside the growth so a reader cannot mistake one for the other.
    assert memory["floor_bytes"] > 64 * 2 ** 20


def _check_memory_summary(block: dict, rows: list[dict]) -> None:
    summary = block["summary"]["memory"]
    for engine in ("grammar", "stim"):
        values = [row["memory"][engine]["growth_mib"] for row in rows
                  if "growth_mib" in row["memory"][engine]]
        assert summary["measured"] == sum(
            len([1 for row in rows if "growth_mib" in row["memory"][engine]])
            for engine in ("grammar", "stim"))
        assert summary["median_growth_mib"][engine] == (
            pytest.approx(statistics.median(values)) if values else None)
    assert summary["unavailable"] == [
        {"label": row["label"], "engine": engine,
         "reason": row["memory"][engine]["reason"]}
        for row in rows for engine in ("grammar", "stim")
        if row["memory"][engine].get("status") == "unavailable"]


def test_the_record_carries_peak_memory_for_both_engines(record):
    """Peak resident growth per engine, on the workloads whose samples are recorded."""
    rows = [*record["sampling"]["rows"], *record["wide_sampling"]["rows"]]
    assert len(rows) == 10
    for row in rows:
        for engine in ("grammar", "stim"):
            _check_memory(row, engine)
    assertion = "memory_rule"
    assert assertion in record["sampling"] and assertion in record["wide_sampling"]
    assert "reset" in record["sampling"][assertion]
    _check_memory_summary(record["sampling"], record["sampling"]["rows"])
    _check_memory_summary(record["wide_sampling"], record["wide_sampling"]["rows"])
    assert record["checks"]["every_sampling_memory_measurement_is_of_the_recorded_samples"]
    assert record["checks"]["every_sampling_memory_abstention_names_a_reason"]
    assert record["checks"]["the_sampling_memory_summary_recomputes_from_its_rows"]
    assert record["reproducibility"]["memory_sha256"] == hashlib.sha256(
        (ROOT / "apedp_dyn/vqc/memory.py").read_bytes()).hexdigest()


def test_the_memory_claim_is_bounded_where_it_is_read(record):
    """Peak RSS is not a ceiling, not a median, and not bytes allocated -- say so."""
    limitations = " ".join(record["limitations"])
    assert "resident-set growth" in limitations
    assert "not bytes allocated" in limitations
    assert "not a ceiling claim" in limitations
    verdict = record["verdict"]["sampling_memory"]
    assert "outcome table" in verdict
    assert "MiB" in verdict
    # The growth is small beside the import floor it is reported against, in every row.
    for row in [*record["sampling"]["rows"], *record["wide_sampling"]["rows"]]:
        for engine in ("grammar", "stim"):
            memory = row["memory"][engine]
            assert memory["growth_bytes"] < memory["floor_bytes"] / 4


def test_the_record_lists_all_three_tools_as_compared(record):
    """Each tool on the task it overlaps with, and none of them as an abstention."""
    statuses = {status["tool"]: status for status in record["tools"]}
    for tool in ("qiskit", "stim", "mqt.qcec"):
        assert statuses[tool]["status"] == "compared", tool
        assert statuses[tool]["counts_as_a_comparison"] is True, tool
        assert statuses[tool]["version"], tool
    # The equivalence task is what QCEC was wired for, so its rows are in the record.
    assert statuses["mqt.qcec"]["wired"] is True
    assert any(row["agreement"]["compared"] for row in record["equivalence"]["rows"])


def test_the_record_carries_the_device_routing_comparison(record):
    routing = record["device_routing"]
    assert routing["device"]["source"].startswith("qiskit_ibm_runtime")
    assert "recorded device" in routing["device"]["source"]
    assert routing["device"]["num_qubits"] >= 100
    assert routing["device"]["edges"] > 0
    rows = routing["rows"]
    assert routing["summary"]["circuits"] == len(rows) == 4
    assert routing["summary"]["matched"] == 4
    for row in rows:
        foreign = row["qiskit"]
        assert foreign["verification"]["verdict"] == "VALID"
        assert foreign["logical_target_sha256"] == row["logical_target_sha256"]
        assert foreign["qiskit_reported_depth"] == foreign["routed_depth"]
        assert len(set(foreign["final_layout"])) == len(foreign["final_layout"])
        assert foreign["routing_swaps"] > 0
        for strategy in ("greedy", "lookahead"):
            ours = row["ours"][strategy]
            assert ours["verification"]["verdict"] == "VALID"
            assert ours["mapped_depth"] >= ours["depth_before_restore"]
            expected = ("qiskit" if foreign["routed_depth"]
                        < ours["depth_before_restore"] else
                        "apedp" if ours["depth_before_restore"]
                        < foreign["routed_depth"] else "tie")
            assert row["comparison"]["routed_depth_winners"][strategy] == expected
    # Neither engine sweeps: the record would be suspect if it did.
    wins = routing["summary"]["routed_depth_wins"]["lookahead"]
    assert wins["qiskit"] > 0 and wins["apedp"] > 0
    assert wins["qiskit"] + wins["apedp"] + wins["tie"] == 4


def test_the_recorded_device_map_matches_the_snapshot():
    """The coupling map in the record is the device's, not a redrawn graph."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    backend = FakeSherbrooke()
    edges = sorted({(min(int(first), int(second)), max(int(first), int(second)))
                    for first, second in backend.coupling_map.get_edges()})
    device = json.loads(RECORD.read_text())["device_routing"]["device"]
    assert device["edge_list_sha256"] == hashlib.sha256(
        json.dumps([list(edge) for edge in edges]).encode("utf-8")).hexdigest()
    assert device["num_qubits"] == backend.num_qubits
    assert device["edges"] == len(edges)


def test_the_topology_task_routes_a_foreign_circuit_too(record):
    task = next(task for task in record["tasks"] if "on a line" in task["task"])
    rows = {row["engine"]: row for row in task["rows"]}
    assert rows["apedp.synthesis:grammar:chain"]["routing_swaps"] == 0
    assert rows["apedp.synthesis:grammar:chain"]["mapped_depth"] == 100
    foreign = rows["qiskit:synth_circuit_from_stabilizers"]
    assert foreign["routing_swaps"] > 0
    assert foreign["mapped_depth"] > foreign["logical_depth"]
    assert all(row["verification"]["verdict"] == "VALID" for row in task["rows"])
    assert task["topology"]["name"] == "line"


def test_the_record_pins_the_sources_it_ran(record):
    reproducibility = record["reproducibility"]
    assert reproducibility["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    for key, relative in (("external_sha256", "apedp_dyn/vqc/external.py"),
                          ("corpus_sha256", "apedp_dyn/vqc/corpus.py"),
                          ("synthesis_sha256", "apedp_dyn/vqc/synthesis.py"),
                          ("placement_sha256", "apedp_dyn/vqc/placement.py"),
                          ("qasm_sha256", "apedp_dyn/vqc/qasm.py")):
        assert reproducibility[key] == hashlib.sha256(
            (ROOT / relative).read_bytes()).hexdigest(), key


def test_a_placed_foreign_circuit_still_verifies():
    """The router is not allowed to be correct only on circuits this package built."""
    pytest.importorskip("qiskit")
    target = ghz_target(12)
    placement = place(qiskit_synthesis(target).circuit, line(12))
    assert placement.swap_gates > 0
    assert adjudicate_specification(placement.circuit, target).verdict == "VALID"


# ------------------------- Experiment E's depths against the Qiskit depth histogram
@pytest.fixture(scope="module")
def comparison_script():
    """The measurement script, so a recorded depth is checked against the code."""
    spec = importlib.util.spec_from_file_location(
        "compare_vqc_external_tools", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _corpus_depths(rows: list[dict]) -> list[int]:
    return [row["qiskit"]["metrics"]["depth"] for row in rows]


def test_the_qiskit_depth_histogram_accounts_for_every_corpus_row(record):
    """A histogram that drops rows is a shape that does not exist."""
    block = record["qiskit_depth_context"]
    rows = record["shared_corpus"]["rows"]
    depths = _corpus_depths(rows)
    assert block["status"] == "measured"
    assert block["engine"] == "qiskit.synthesis.synth_circuit_from_stabilizers"
    assert block["corpus"]["targets"] == len(rows) == block["overall"]["n"]
    assert sum(bucket["count"] for bucket in block["buckets"]) == len(depths)
    assert block["overall"]["min"] == min(depths) == block["buckets"][0]["low"]
    assert block["overall"]["max"] == max(depths)
    width = block["bucket_width"]
    for left, right in zip(block["buckets"], block["buckets"][1:], strict=False):
        assert right["low"] == left["high"] + 1
    for bucket in block["buckets"]:
        assert bucket["high"] - bucket["low"] + 1 == width
        inside = [depth for depth in depths
                  if bucket["low"] <= depth <= bucket["high"]]
        assert bucket["count"] == len(inside), bucket
    # An empty bucket is a bucket: the gap between 59 and 90 is part of the shape, and
    # dropping it would turn a long tail into a short one.
    assert any(bucket["count"] == 0 for bucket in block["buckets"])


def test_each_width_cell_carries_the_depths_it_summarises(record):
    block = record["qiskit_depth_context"]
    rows = record["shared_corpus"]["rows"]
    assert sorted(int(width) for width in block["by_width"]) == block["corpus"]["widths"]
    for width, cell in block["by_width"].items():
        values = sorted(row["qiskit"]["metrics"]["depth"] for row in rows
                        if str(row["width"]) == width)
        assert cell["depths"] == values
        assert cell["n"] == len(values)
        assert cell["min"] == values[0] and cell["max"] == values[-1]
        assert cell["median"] == statistics.median(values)
    assert sum(cell["n"] for cell in block["by_width"].values()) == block["overall"]["n"]


def test_only_a_width_matched_target_is_compared_like_for_like(record):
    block = record["qiskit_depth_context"]
    widths = block["corpus"]["widths"]
    placed = block["targets"]
    assert len(placed) == 6
    assert block["like_for_like"] == [row["target"] for row in placed
                                      if row["qubits"] in widths]
    # Exactly one Experiment E target shares a width with the corpus, and the record
    # says so rather than letting four extrapolations pass as comparisons.
    assert block["like_for_like"] == ["irregular_64q_seed23"]
    for row in placed:
        place = row["placement"]
        assert place["cell_qubits"] in widths
        assert place["n"] == block["by_width"][str(place["cell_qubits"])]["n"]
        if place["width_match"] == "exact":
            assert place["cell_qubits"] == row["qubits"]
            assert place["cell_width_over_target"] is None
            continue
        assert place["cell_qubits"] != row["qubits"]
        nearest = min(widths, key=lambda width: (abs(width - row["qubits"]), width))
        assert place["cell_qubits"] == nearest
        assert place["cell_width_over_target"] == pytest.approx(
            nearest / row["qubits"])


def test_each_placement_is_recomputed_from_the_cell_it_names(record):
    block = record["qiskit_depth_context"]
    for row in block["targets"]:
        place = row["placement"]
        values = block["by_width"][str(place["cell_qubits"])]["depths"]
        at_or_below = [value for value in values if value <= row["depth"]]
        assert place["qiskit_depths_at_or_below"] == len(at_or_below)
        assert place["share_at_or_below"] == pytest.approx(
            len(at_or_below) / len(values))
        assert place["min"] == values[0] and place["max"] == values[-1]
        assert place["median"] == statistics.median(values)
        assert place["median_over_target"] == pytest.approx(
            statistics.median(values) / row["depth"])


def test_the_placed_depths_are_the_minimum_schedule_depths(record, comparison_script):
    """The depth in the histogram block is a circuit, not a transcription."""
    block = record["qiskit_depth_context"]
    placed = {row["target"]: row for row in block["targets"]}
    assert set(placed) == {name for name, _qubits, _make
                           in comparison_script.EXPERIMENT_E_CASES}
    for description, qubits, make_target in comparison_script.EXPERIMENT_E_CASES:
        row = placed[description]
        assert row["qubits"] == qubits
        if qubits > 400:
            # The 900-qubit specification is not even constructed here: its stabilizer
            # text alone is tens of megabytes, which would move this process's resident
            # set out from under the memory-window pins that run after this file. The
            # circuit is pinned against the record that timed it, in
            # `tests/test_vqc_synthesis_record.py`; here its own columns must be
            # consistent with each other.
            assert row["depth"] == 1 + row["cz_layers"]
            assert row["verdict"] == "VALID"
            continue
        target = make_target()
        construction = build("graph_layers", target)
        assert row["depth"] == construction.circuit.depth()
        assert row["cz_layers"] == row["depth"] - 1
        assert row["circuit_sha256"] == construction.circuit.sha256()
        assert row["verdict"] == adjudicate_specification(
            construction.circuit, target).verdict == "VALID"
