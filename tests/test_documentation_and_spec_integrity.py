"""Documentation & Configuration Integrity Test Suite.

Automates the verification of:
  1. Cryptographic SHA-256 integrity of canonical preregistered specifications.
  2. Runtime code consumption of frozen constants with zero drift.
  3. Consistency between documentation text, canonical spec, and runtime implementations.
  4. Detection of broken file links in documentation.
  5. Detection of orphaned source modules missing from DOCUMENTATION.md index.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import apedp_dyn.epistemic.novelty_adjudication as na
import apedp_dyn.epistemic.promotion_lifecycle as pl
import apedp_dyn.sandbox.calibrated_benchmark_suite as cbs
import apedp_dyn.sandbox.canonical_ledger as cl
from apedp_dyn.conf.canonical_spec import (
    CONTINUOUS_FACTOR_ACTIVE_MIN,
    DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    DSR_THRESHOLD,
    MAX_LITERATURE_R2,
    PINNED_SPEC_SHA256,
    REFEREE_MAX_FPR,
    REFEREE_MIN_CAR,
    REFEREE_MIN_VRR,
    SPEC_PATH,
    STOCK_SPLIT_INVARIANCE_TOLERANCE,
    TACTICAL_ACTIVE_MAX,
    TACTICAL_ACTIVE_MIN,
    TAIL_RISK_MIN_EVENTS,
    load_and_verify_canonical_spec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# Pinned documentation-debt ledger (2026-10-19). Debt was reduced from 125 to
# 67 by indexing the core packages and the benchmark problem catalog, and to ZERO
# by indexing the remaining historical sandbox harnesses. The ledger is now empty:
# any new undocumented module fails this test, and there is no pin to hide behind.
PINNED_UNINDEXED_COUNT = 0
PINNED_UNINDEXED_MODULES = frozenset(
    """
""".split()
)


def test_canonical_spec_hash_integrity():
    """Verify that canonical spec file matches its pinned cryptographic SHA-256 hash."""
    assert SPEC_PATH.exists(), f"Canonical specification missing at {SPEC_PATH}"
    content = SPEC_PATH.read_bytes()
    computed_hash = hashlib.sha256(content).hexdigest()
    assert computed_hash == PINNED_SPEC_SHA256, (
        f"Canonical spec has been mutated without updating pinned hash!\n"
        f"Expected: {PINNED_SPEC_SHA256}\nComputed: {computed_hash}"
    )

    # Calling loader with enforce_hash=True must succeed
    spec_data = load_and_verify_canonical_spec(enforce_hash=True)
    assert spec_data["spec_id"] == "APEDPX_PREREGISTERED_CANONICAL_V1"
    assert spec_data["thresholds"]["tactical_active_fraction_max"] == 0.90
    assert spec_data["thresholds"]["pruning_holdout_f_test_significance_level"] == 0.05
    assert spec_data["thresholds"]["pruning_holdout_r2_base_floor"] == 0.20


def test_runtime_code_matches_canonical_spec():
    """Ensure runtime functions consume canonical thresholds with zero drift."""
    # 1. canonical_ledger: assert_strategy_class_compliance defaults
    sig_comp = inspect.signature(cl.assert_strategy_class_compliance)
    params_comp = sig_comp.parameters
    assert params_comp["min_active_tactical"].default == TACTICAL_ACTIVE_MIN
    assert params_comp["max_active_tactical"].default == TACTICAL_ACTIVE_MAX
    assert params_comp["min_events"].default == TAIL_RISK_MIN_EVENTS
    assert params_comp["min_active_continuous"].default == CONTINUOUS_FACTOR_ACTIVE_MIN

    # 2. canonical_ledger: assert_stock_split_invariance default atol
    sig_split = inspect.signature(cl.assert_stock_split_invariance)
    assert sig_split.parameters["atol"].default == STOCK_SPLIT_INVARIANCE_TOLERANCE

    # 3. novelty_adjudication: defaults
    sig_dm = inspect.signature(na.evaluate_predictive_incrementality)
    assert sig_dm.parameters["alpha"].default == DIEBOLD_MARIANO_P_VALUE_THRESHOLD

    sig_novelty = inspect.signature(na.evaluate_structural_novelty)
    assert sig_novelty.parameters["novelty_threshold_r2"].default == MAX_LITERATURE_R2

    # 4. promotion_lifecycle: stage defaults
    tracker = pl.PromotionTracker(candidate_id="test_candidate")
    sig_s3 = inspect.signature(tracker.verify_stage_3)
    assert sig_s3.parameters["dm_alpha"].default == DIEBOLD_MARIANO_P_VALUE_THRESHOLD
    assert sig_s3.parameters["max_r2_ceiling"].default == MAX_LITERATURE_R2

    sig_s5 = inspect.signature(tracker.verify_stage_5)
    assert sig_s5.parameters["min_dsr_hurdle"].default == DSR_THRESHOLD

    # 5. calibrated_benchmark_suite: referee contract constants
    assert cbs.REFEREE_MAX_FPR == REFEREE_MAX_FPR == 0.05
    assert cbs.REFEREE_MIN_CAR == REFEREE_MIN_CAR == 0.70
    assert cbs.REFEREE_MIN_VRR == REFEREE_MIN_VRR == 0.75


def test_documentation_constants_match_runtime_code():
    """Check that documentation and orientation files describe the correct tactical window [5%, 90%]."""
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    readme_path = REPO_ROOT / "README.md"
    arch_path = REPO_ROOT / "ARCHITECTURE.md"

    for file_path in [doc_path, readme_path, arch_path]:
        assert file_path.exists(), f"File {file_path} must exist"
        text = file_path.read_text(encoding="utf-8")
        # Ensure the outdated 95% threshold is not present in tactical window context
        assert "active in [5%, 95%]" not in text, f"Outdated 95% tactical threshold found in {file_path}"
        assert "[5%, 95%]" not in text, f"Outdated [5%, 95%] tactical threshold found in {file_path}"
        # Ensure the corrected 90% threshold is mentioned
        assert ("90%" in text or "90\\%" in text), f"Expected 90% tactical bound not mentioned in {file_path}"


def test_documentation_links_exist_on_disk():
    """Validate every repository-relative link in DOCUMENTATION.md.

    Links are resolved from ``docs/`` (ARCHITECTURE.md §4.2). Absolute
    ``file:///home/<user>/...`` links are rejected outright: they are not
    portable and this test used to honor only that non-portable form.
    """
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    content = doc_path.read_text(encoding="utf-8")

    assert "file:///" not in content, (
        "Absolute file:// links are forbidden in the live documentation index "
        "(use repository-relative links; see ARCHITECTURE.md §4.2)"
    )

    links = re.findall(r"\]\(([^)\s]+)\)", content)
    checked = 0
    for link in links:
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = (doc_path.parent / link.split("#", 1)[0]).resolve()
        assert target.exists(), (
            f"Broken relative link in DOCUMENTATION.md: {link} does not exist at {target}"
        )
        checked += 1
    assert checked >= 200, f"documentation link check covered too few links ({checked})"


def test_orphan_module_detection():
    """Detect production modules in apedp_dyn missing from DOCUMENTATION.md.

    The scan is recursive over the whole ``apedp_dyn`` tree (the original
    four-directory, non-recursive scan missed 143/250 modules, including the
    entire ``epistemic/active_disambiguation`` and most of ``discovery_l2``,
    which are now indexed). The pinned debt ledger was fully repaid on
    2026-10-19 (125 -> 67 -> 0): every ``apedp_dyn`` module is now indexed.
    Adding a module without indexing it fails this test; there is no pin left
    to mask a regression.

    Third-party checkouts vendored inside the sandbox for a build are not APEDP
    modules: ``kickmix/_qualtran_src`` is an upstream git repository (759 files),
    indexed in DOCUMENTATION.md as a tree rather than file by file. Any nested
    repository (a ``.git`` below the scan root) is therefore skipped, which is
    what keeps the scan on code this repository actually authors.
    """
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    doc_text = doc_path.read_text(encoding="utf-8")

    vendored_roots = {marker.parent for marker in (REPO_ROOT / "apedp_dyn").rglob(".git")}

    def authored(py_file: Path) -> bool:
        return ("__pycache__" not in py_file.parts
                and not py_file.name.startswith("__")
                and not any(root in py_file.parents for root in vendored_roots))

    all_modules = sorted(
        py_file for py_file in (REPO_ROOT / "apedp_dyn").rglob("*.py") if authored(py_file)
    )
    unindexed = {
        str(py_file.relative_to(REPO_ROOT))
        for py_file in all_modules
        if str(py_file.relative_to(REPO_ROOT)) not in doc_text
    }

    added = sorted(unindexed - PINNED_UNINDEXED_MODULES)
    repaid = sorted(PINNED_UNINDEXED_MODULES - unindexed)
    assert not added, (
        "New undocumented modules detected (index them in DOCUMENTATION.md "
        "or the pin may not grow):\n" + "\n".join(f" - {a}" for a in added)
    )
    assert not repaid, (
        "Documentation debt was paid down; shrink PINNED_UNINDEXED_MODULES "
        "to match (debt must never mask a regression):\n"
        + "\n".join(f" - {r}" for r in repaid)
    )
    assert len(PINNED_UNINDEXED_MODULES) == PINNED_UNINDEXED_COUNT


def test_registry_status_tags_match_code_status_labels():
    """Registry Status tags and module docstring `Status:` lines must agree.

    AGENTS.md section 3 defines exactly four canonical status tags. The
    DOCUMENTATION.md module registry must use only those tags, and wherever a
    linked module carries an explicit `Status:` line in its docstring, it must
    match the registry tag: documentation cannot certify a stability level
    the code does not declare.
    """
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    content = doc_path.read_text(encoding="utf-8")

    valid_tags = {"production", "experimental", "audit-only", "deprecated"}
    row_re = re.compile(
        r"\|\s*\[`([^`]+)`\]\((\.\./[^)]+)\)[^|]*\|[^|]*\|\s*`([a-z-]+)`\s*\|"
    )
    status_re = re.compile(r"Status:\s*(production|experimental|audit-only|deprecated)\b")

    checked = 0
    for _display, rel, tag in row_re.findall(content):
        assert tag in valid_tags, f"Non-canonical status tag `{tag}` for {rel} in DOCUMENTATION.md"
        target = (doc_path.parent / rel).resolve()
        if not target.exists() or target.suffix != ".py":
            continue
        head = target.read_text(encoding="utf-8")[:4000]
        m = status_re.search(head)
        if m is None:
            continue
        assert m.group(1) == tag, (
            f"Status mismatch for {rel}: DOCUMENTATION.md says `{tag}`, "
            f"module docstring says `{m.group(1)}`"
        )
        checked += 1

    assert checked >= 10, f"registry status consistency check covered too few modules ({checked})"


def test_mmlm_graph_state_figures_in_the_index_match_the_record():
    """The 54-win / 308-gate figures are quoted in DOCUMENTATION.md; they are measured.

    The table sentence could drift from the artifact without any test noticing, which is
    the failure the record was created to prevent.
    """
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "benchmarks" / "quantum_compilation" / "vqc_mmlm_graph_state.json").read_text()
    )
    index = (root / "docs" / "DOCUMENTATION.md").read_text()

    assert index.count("54 wins / 21 ties / 0 losses") >= 1
    assert f"{record['wins']} wins / {record['ties']} ties / {record['losses']} losses" in index
    assert f"{record['saved_two_qubit_gates']} two-qubit gates saved" in index
    assert f"{record['certified_valid']}/{record['targets']} certified" in index
    # The gate totals are quoted with thousands separators; check both ends.
    assert f"{record['baseline_two_qubit_gates']:,}" in index
    assert f"{record['mmlm_two_qubit_gates']:,}" in index

    # The headline is a total over three families that are not comparable. The split is
    # quoted beside it and generated from the same record, because "54 wins" reads as a
    # uniform result when 77% of it is one family and the regular families thin with width.
    split = ", ".join(
        f"`{name}` **{block['wins']}/{block['targets']} wins, "
        f"{block['saved_two_qubit_gates']} gates**"
        for name, block in sorted(record["families"].items())
    )
    assert split in index, f"the family split is not quoted as recorded: {split}"
    assert sum(b["wins"] for b in record["families"].values()) == record["wins"]
    assert (
        sum(b["saved_two_qubit_gates"] for b in record["families"].values())
        == record["saved_two_qubit_gates"]
    )
    for family, bands in record["by_width"].items():
        assert sum(b["targets"] for b in bands.values()) == record["families"][family]["targets"]
    # The width cut is the part no total can show: regular4 saves nothing at 32 qubits.
    assert f"`regular4` {record['by_width']['regular4']['32']['saved_two_qubit_gates']} gates at 32q" in index
    assert (
        record["by_width"]["regular4"]["64"]["saved_two_qubit_gates"]
        < record["by_width"]["random_graph"]["64"]["saved_two_qubit_gates"]
    )


def test_dqi_target_closures_are_recorded_in_the_campaign_docs():
    """The two closures are prose where a reader will meet the target, and pinned here."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    reachability = (root / "docs" / "BENCHMARK2_TARGET_REACHABILITY.md").read_text()
    compatibility = (root / "docs" / "DQI_OFFICIAL_COMPATIBILITY.md").read_text()

    for text in (reachability, compatibility):
        assert "permanently unsourced" in text
    assert "96 > 32" in reachability
    assert "1,853" in compatibility
    assert (
        root / "benchmarks" / "quantum_compilation" / "dqi_design_projection.json"
    ).exists()


def test_the_register_floor_model_is_recorded_where_the_target_is():
    """The one-parameter floor and its 64-qubit shortfall are next to the target table."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "docs"
    reachability = (root / "BENCHMARK2_TARGET_REACHABILITY.md").read_text()
    compatibility = (root / "DQI_OFFICIAL_COMPATIBILITY.md").read_text()

    assert "peak = 1,853 + 8k" in reachability
    assert "zero-register floor" in compatibility
    assert "64 qubits" in reachability


def test_the_data_path_floor_is_recorded_with_the_target():
    """The 55-vs-64 accounting sits next to the target table and stays there."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "docs"
    reachability = (root / "BENCHMARK2_TARGET_REACHABILITY.md").read_text()
    compatibility = (root / "DQI_OFFICIAL_COMPATIBILITY.md").read_text()

    assert "23 < 64" in reachability
    assert "model_data_path_floor" in reachability
    assert "23 against the 64" in compatibility
    # Recycling is closed on both grounds and the projection is rescored at the verifier peak.
    assert "closed on both grounds" in reachability
    assert "0.898×" in reachability
    assert "1.9006e12" in reachability and "1.9006e12" in compatibility


def test_the_frontier_rung4_prices_rung3s_own_residual():
    """§7.9 rung 4 must keep *rung 3's* residual, not the six-lever arm's 472,768.2.

    Substituting 472,768.2 into the campaign arm's row made rung 4 read 1.084× -- a loss --
    when the arm's own residual is 299,953.2 and freeing it scores 0.688×. That substitution
    is what let the cheapest family look inadmissible, so the figures in the note are pinned
    against the model that computes them.
    """
    import pathlib

    from apedp_dyn.quantum_compilation.dqi_gluzdov_v6 import product_axis_decision

    root = pathlib.Path(__file__).resolve().parents[1] / "docs"
    reachability = (root / "BENCHMARK2_TARGET_REACHABILITY.md").read_text()
    axis = product_axis_decision()
    rung4 = axis["rung4_corrected"]

    assert axis["campaign_residual_toffolis"] == 299953.2
    assert f"{rung4['t_mean_toffolis']:,.1f}" in reachability
    assert f"{rung4['score']:.4e}".replace("e+12", "e12") in reachability
    assert f"{rung4['vs_denominator']:.3f}×" in reachability
    # The replaced row is retained for audit rather than deleted, and labelled as the
    # six-lever residual it actually was.
    assert "as tabulated (six-lever residual)" in reachability
    assert f"{axis['tabulated_overstatement_toffolis']:,.1f}" in reachability
    # The product axis, which is where the remaining gap actually lives.
    assert f"{axis['products_allowed_fraction'] * 100:.1f}%" in reachability
    assert "product_axis_decision" in reachability

    import pathlib

    campaign = (
        pathlib.Path(__file__).resolve().parents[1] / "docs" / "EEA_SCHEDULE_CAMPAIGN.md"
    ).read_text()
    assert "route is now closed" in campaign
    assert "0.898×" in campaign


def test_the_single_array_packing_closure_matches_its_artifact():
    """The contract says the packing route is closed; the arm totals come from the record.

    The route was killed by a classical probe rather than by building a decoder, so the prose and
    the numbers have only this comparison keeping them in step: regenerate the artifact with a
    different arm total and the contract doc has to move with it.
    """
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "benchmarks" / "quantum_compilation" / "dqi_single_array_packing.json").read_text()
    )
    contract = (root / "docs" / "DQI_SINGLE_ARRAY_DECODER_CONTRACT.md").read_text()

    arms = record["arms"]
    three = arms["three_region"]
    assert three["failed"] == 0, "the three-region core must decode every planted pattern"
    boundary = record["failure_boundary"]
    assert boundary["failed_to_decode"] == boundary["trials"], (
        "the weight-l+1 boundary must fail for every trial, or a false pass cannot be seen"
    )
    for name in ("two_region_naive", "two_region_swapped"):
        arm = arms[name]
        assert arm["first_failure"]["weight"] == 1, (
            f"{name} must fail at the first weight, not merely lose a few patterns"
        )
        assert f"{arm['decoded']}/{three['decoded']}" in contract, (
            f"the contract doc does not quote {name}'s {arm['decoded']}/{three['decoded']}"
        )
    assert f"**{three['decoded']}/{three['decoded']}**" in contract
    assert "PACKING_ROUTE_CLOSED" in contract
    assert "weight 1" in contract
    assert "1,001" in contract


def test_the_replay_route_price_matches_its_artifact():
    """The contract's replay price comes from a build, so the doc must quote the artifact.

    `17,143` per round and the two score multiples are the numbers that killed the route; if the
    artifact is regenerated at a different `n_iters` the contract section has to be rewritten rather
    than left quoting a superseded price.
    """
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "benchmarks" / "quantum_compilation" / "dqi_single_array_replay_cost.json").read_text()
    )
    contract = (root / "docs" / "DQI_SINGLE_ARRAY_DECODER_CONTRACT.md").read_text()

    per_round = record["per_round"]["both_eea_passes"]
    bill = record["replay_bill"]
    assert per_round == round(sum(record["marginal_toffoli_per_round_step"]) / 3)
    assert bill["compute_only"]["multiple_of_eea"] > 1.0, "the route must be priced as a loss"
    assert bill["with_bennett_uncompute"]["multiple_of_eea"] > bill["compute_only"]["multiple_of_eea"]
    assert 222 * (222 + 1) // 2 == bill["rounds_units_to_produce_B_t_at_every_t"]

    assert f"{per_round:,}" in contract
    assert f"{bill['compute_only']['score_multiple_of_reference']:.2f}× worse" in contract
    assert f"{bill['with_bennett_uncompute']['score_multiple_of_reference']:.2f}× worse" in contract
    assert "REPLAY_ROUTE_CLOSED" in contract


def test_the_frontier_restatement_agrees_with_the_projection():
    """The frontier section is final-state prose; it must not drift from the projection.

    Every width route in it is closed by an artifact, and the one number a reader will quote —
    the rung-3 score — comes from the same record the rest of the campaign is scored against.
    """
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    projection = json.loads(
        (root / "benchmarks" / "quantum_compilation" / "dqi_design_projection.json").read_text()
    )
    reachability = (root / "docs" / "BENCHMARK2_TARGET_REACHABILITY.md").read_text()
    arm = json.loads(
        (root / "benchmarks" / "quantum_compilation"
         / "dqi_campaign_arm_submission_score.json").read_text()
    )
    replay = json.loads(
        (root / "benchmarks" / "quantum_compilation" / "dqi_single_array_replay_cost.json").read_text()
    )

    trade = projection["single_array_trade"]
    assert trade["route_status"] == "CLOSED_ON_MEASUREMENT"
    assert trade["peak_with_one_array"] == 1001
    assert "CLOSED_ON_MEASUREMENT" in reachability
    assert "withdrawn" in trade["closure"]["headline"], (
        "the closure must say the 0.284x trade is withdrawn, not merely that it is closed"
    )

    # The frontier arm's own verified numbers are what the section quotes. The doc writes the
    # score without a signed exponent and the Toffoli count to one decimal, so match that form.
    score = f"{arm['verified']['score']:.4e}".replace("e+", "e")
    assert score in reachability, score
    assert f"{arm['verified']['toffoli_count']:,.1f}" in reachability
    assert f"{arm['verified']['num_qubits']:,}" in reachability

    # And the replay price it rules the route out on.
    assert f"{replay['per_round']['both_eea_passes']:,}" in reachability
    assert f"{replay['replay_bill']['with_bennett_uncompute']['score_multiple_of_reference']:.2f}×" in reachability


def test_the_single_array_trade_is_recorded():
    """The bracketed trade and the two design docs are reachable from the note with the target."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    reachability = (root / "docs" / "BENCHMARK2_TARGET_REACHABILITY.md").read_text()
    compatibility = (root / "docs" / "DQI_OFFICIAL_COMPATIBILITY.md").read_text()

    assert "**1,001**" in reachability
    assert "0.284×" in reachability and "0.786×" in reachability
    assert "378" in reachability and "46,656" in reachability
    assert "1,001" in compatibility

    contract = (root / "docs" / "DQI_SINGLE_ARRAY_DECODER_CONTRACT.md").read_text()
    bodies = (root / "docs" / "DQI_RECYCLED_PASS_BODIES.md").read_text()
    assert "Derivation lemma" in contract and "Falsifiers" in contract
    assert "1,885" in bodies and "leading-zero" in bodies


def test_the_recycled_substitution_status_is_recorded():
    import pathlib

    bodies = (
        pathlib.Path(__file__).resolve().parents[1] / "docs" / "DQI_RECYCLED_PASS_BODIES.md"
    ).read_text()
    assert "derive_n_a" in bodies
    assert "1,893 → 1,885" in bodies
    assert "reads no `n_a`" in bodies
    # The build reaches the peak and fails equivalence; both are stated, neither is dropped.
    assert "is a peak, not a decoder" in bodies
    assert "52/64" in bodies and "fail" in bodies
    assert "dqi_recycled_flag_body_build.json" in bodies
    assert "dqi_recycled_degrees_window.json" in bodies
    assert "dqi_recycled_equivalence.json" in bodies

    equivalence = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "quantum_compilation"
         / "dqi_recycled_equivalence.json").read_text()
    )
    assert equivalence["arms"]["reference"]["value_errors"] == 0
    assert equivalence["equivalent_to_reference"]["update_flag_substituted"] is False
    assert equivalence["equivalent_to_reference"]["all_sites_substituted"] is False
    record = json.loads((pathlib.Path(__file__).resolve().parents[1] / "benchmarks" /
                         "quantum_compilation" / "dqi_recycled_flag_body_build.json").read_text())
    assert record["peak_unchanged"] is True
    assert record["recycled_flag_peak_qubits"] == record["reference_peak_qubits"]
    assert record["read_sites_substituted"] == 1
    assert record["accounting_peak_qubits"] == 1885
