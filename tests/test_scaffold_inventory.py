"""Scaffold inventory: every tier claim in docs/SCAFFOLD_INVENTORY.md, pinned.

The rule these tests enforce is narrow and absolute: a constant may not be
returned under a name that says "prediction". Where the honest answer needs a
model that is not present, the call must raise.

Status: audit-only
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from apedp import applied_synthetic_bio
from apedp.applied_climate import CarbonMineralization, ClimateModeling, DirectAirCapture
from apedp.applied_neuromorphic import NeuromorphicArchitect, NeuromorphicChipDesigner
from apedp.applied_synthetic_bio import (
    SCOPE,
    CRISPRDesigner,
    MetabolicPathwayOptimizer,
)
from apedp.ingestion import ingestion_academic_apis
from apedp.ingestion.ingestion_web_equations import (
    EXTRACTION_IS_LOSSY,
    _extract_formula_strings,
)
from apedp_dyn.distributed.cluster_launcher import (
    SUBMITS_JOBS,
    ClusterJobSpec,
    generate_slurm_script,
)
from apedp_dyn.distributed.cluster_launcher import (
    describe as describe_cluster,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# The refused call: a constant was being returned as a prediction
# --------------------------------------------------------------------------


GUIDE = "GAGTCCGAGCAGAAGAAGAA"
FILLER = "A" * 24           # no G at all, so filler cannot form a PAM


def _mutate(sequence: str, positions: tuple[int, ...]) -> str:
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    characters = list(sequence)
    for index in positions:
        characters[index] = swap[characters[index]]
    return "".join(characters)


def _off_target_reference() -> str:
    from apedp.applied_synthetic_bio import reverse_complement

    return (
        FILLER
        + GUIDE + "AGG"                              # exact, forward
        + FILLER
        + _mutate(GUIDE, (19,)) + "CGG"               # one mismatch
        + FILLER
        + _mutate(GUIDE, (0, 10)) + "TGG"             # two mismatches
        + FILLER
        + reverse_complement(GUIDE + "AGG")          # exact, reverse strand
        + FILLER
    )


def test_off_target_search_refuses_without_a_reference():
    """Regression: returned ``predicted_off_targets: 12`` and ``confidence: high``.

    A genome's size cannot produce its off-target count, so the call still raises
    when no sequence is supplied. With a sequence, it searches.
    """
    with pytest.raises(NotImplementedError, match="reference sequence"):
        CRISPRDesigner.predict_off_targets(GUIDE)
    with pytest.raises(TypeError, match="genome_size_mb was removed"):
        CRISPRDesigner.predict_off_targets(GUIDE, 3000)


def test_off_target_search_counts_real_sites_on_both_strands():
    result = CRISPRDesigner.predict_off_targets(GUIDE, _off_target_reference())

    # The guide contains CCG, which is an NGG PAM on the minus strand, so the
    # scan legitimately finds more PAM-adjacent sites than the four built here.
    assert result["sites_scanned"] >= 4
    assert result["exact_matches"] == 2                  # forward and reverse strand
    assert result["predicted_off_targets"] == 2          # one and two mismatches
    assert result["counts_by_mismatch"] == {0: 2, 1: 1, 2: 1, 3: 0, 4: 0}
    assert result["high_risk_sites"] == 3
    assert result["medium_risk_sites"] == 1
    assert result["low_risk_sites"] == 0
    assert {site["strand"] for site in result["exact_sites"]} == {"+", "-"}
    assert result["score_basis"] == "hamming_mismatch_count"
    assert result["evidence"] == "seeded_hamming_search"
    assert "genome_wide_scan" in result["not_modelled"]


def test_off_target_sites_are_ranked_and_located():
    result = CRISPRDesigner.predict_off_targets(GUIDE, _off_target_reference())
    sites = result["top_sites"]
    assert [site["mismatches"] for site in sites] == [1, 2]
    reference = _off_target_reference()
    for site in sites:
        start = site["forward_start"]
        assert reference[start:start + result["guide_length"]] == site["protospacer"]
        assert reference[start + result["guide_length"]:start + result["guide_length"] + 3] == site["pam"]


def test_off_target_search_needs_a_pam_adjacent_site():
    """A guide present without an adjacent PAM yields no exact candidate site."""
    reference = FILLER + GUIDE + "AAA" + FILLER
    result = CRISPRDesigner.predict_off_targets(GUIDE, reference)
    assert result["exact_matches"] == 0
    assert result["predicted_off_targets"] == 0


def test_off_target_max_mismatches_bounds_the_search():
    reference = _off_target_reference()
    strict = CRISPRDesigner.predict_off_targets(GUIDE, reference, max_mismatches=1)
    assert strict["predicted_off_targets"] == 1
    assert strict["counts_by_mismatch"] == {0: 2, 1: 1}


@pytest.mark.parametrize(
    ("guide", "reference"),
    [
        ("GAGTCCGAGCAGAAGAAGX", FILLER),          # non-DNA guide
        ("ACG", FILLER),                          # shorter than 8 nt
        (GUIDE, "ACGTZ"),                         # non-DNA reference
        (GUIDE, ""),                              # empty reference
    ],
)
def test_off_target_search_rejects_input_it_cannot_scan(guide, reference):
    with pytest.raises(ValueError):
        CRISPRDesigner.predict_off_targets(guide, reference)


def _bulged_reference() -> str:
    """The guide with one extra base after position 10: the reference bulges."""
    return FILLER + GUIDE[:11] + "A" + GUIDE[11:] + "TGG" + FILLER


def _indel_reference() -> str:
    """The guide missing one base after position 10: a single deletion."""
    return FILLER + GUIDE[:11] + GUIDE[12:] + "AGG" + FILLER


def test_off_target_search_tolerates_a_one_base_bulge_and_says_so():
    result = CRISPRDesigner.predict_off_targets(GUIDE, _bulged_reference())

    assert result["bulge_tolerance"] is True
    assert result["bulge_off_targets"] >= 1
    site = next(s for s in result["top_gapped_sites"] if s["kind"] == "bulge")
    # The reference carries the extra base, so the unpaired base is the target's.
    assert site["unpaired_base"] == "target"
    assert site["mismatches"] <= result["max_mismatches"]
    assert 1 <= site["gap_position"] <= result["guide_length"] - 2


def test_off_target_search_tolerates_a_single_indel_and_says_so():
    result = CRISPRDesigner.predict_off_targets(GUIDE, _indel_reference())

    assert result["indel_tolerance"] is True
    assert result["indel_off_targets"] >= 1
    site = next(s for s in result["top_gapped_sites"] if s["kind"] == "indel")
    # The reference is missing a base, so the guide is the one with the extra one.
    assert site["unpaired_base"] == "guide"


def test_gapped_tolerances_are_separable_and_switchable():
    reference = _bulged_reference() + "T" * 12 + _indel_reference()
    both = CRISPRDesigner.predict_off_targets(GUIDE, reference)
    neither = CRISPRDesigner.predict_off_targets(
        GUIDE, reference, tolerate_bulges=False, tolerate_indels=False)
    bulge_only = CRISPRDesigner.predict_off_targets(GUIDE, reference, tolerate_indels=False)
    indel_only = CRISPRDesigner.predict_off_targets(GUIDE, reference, tolerate_bulges=False)

    assert neither["bulge_off_targets"] == 0 and neither["indel_off_targets"] == 0
    assert bulge_only["indel_off_targets"] == 0 and bulge_only["bulge_off_targets"] >= 1
    assert indel_only["bulge_off_targets"] == 0 and indel_only["indel_off_targets"] >= 1
    # Turning a tolerance off may never change the mismatch-only count.
    assert {r["predicted_off_targets"] for r in (both, neither, bulge_only, indel_only)} == {
        both["predicted_off_targets"]}
    assert both["total_off_targets"] == (both["predicted_off_targets"]
                                         + both["bulge_off_targets"]
                                         + both["indel_off_targets"])
    assert both["counts_by_kind"] == {"mismatch_only": both["predicted_off_targets"],
                                     "bulge": both["bulge_off_targets"],
                                     "indel": both["indel_off_targets"]}


def test_gapped_tolerance_does_not_relabel_the_old_counts():
    """``predicted_off_targets`` stays mismatch-only; gapped sites are extra."""
    result = CRISPRDesigner.predict_off_targets(GUIDE, _off_target_reference())
    assert result["predicted_off_targets"] == 2
    assert result["counts_by_mismatch"] == {0: 2, 1: 1, 2: 1, 3: 0, 4: 0}
    assert result["total_off_targets"] == (result["predicted_off_targets"]
                                           + result["bulge_off_targets"]
                                           + result["indel_off_targets"])
    assert "bulges_longer_than_one_base" in result["not_modelled"]
    assert "bulge_loops" not in result["not_modelled"]


def test_position_weights_ramp_towards_the_pam():
    from apedp.applied_synthetic_bio import position_weighted_score, position_weights

    weights = position_weights(20)
    assert weights[0] == pytest.approx(0.25)
    assert weights[-1] == pytest.approx(1.0)
    assert all(later > earlier for earlier, later in pairwise(weights))
    assert sum(weights) == pytest.approx(12.5)

    # Normalised to mismatch equivalents: one PAM-proximal mismatch is four
    # times one distal mismatch, and both stay on the 0..20 scale.
    assert position_weighted_score("A" * 20, "A" * 19 + "C") == 0.08
    assert position_weighted_score("A" * 20, "C" + "A" * 19) == 0.02
    assert position_weighted_score("A" * 20, "A" * 20) == 0.0
    with pytest.raises(ValueError, match="same length"):
        position_weighted_score("AAAA", "AAAAA")


def test_hamming_and_weighted_rankings_are_reported_side_by_side():
    result = CRISPRDesigner.predict_off_targets(GUIDE, _off_target_reference())

    # The Hamming order is unchanged: mismatch count first, seed second.
    assert [site["mismatches"] for site in result["top_sites"]] == [1, 2]
    assert all("position_weighted_score" in site for site in result["top_sites"])

    comparison = result["ranking_comparison"]
    assert comparison["sites_compared"] == result["predicted_off_targets"]
    assert comparison["concordant_pairs"] + comparison["discordant_pairs"] == 1
    # One 1-mismatch site (PAM-proximal, score 0.08) and one 2-mismatch site
    # (both distal, score 0.0663): the two orders are exactly reversed.
    assert comparison["kendall_tau"] == -1.0
    assert comparison["cross_band_discordant_pairs"] == 1
    assert comparison["within_band_discordant_pairs"] == 0

    (disagreement,) = comparison["disagreements"]
    assert disagreement["hamming_prefers"]["mismatches"] == 1
    assert disagreement["weighted_prefers"]["mismatches"] == 2


def _same_count_reference() -> str:
    """Two 2-mismatch sites: seed ordering and weighted ordering disagree.

    At (10, 11) both mismatches sit just outside the 8 nt seed but well into the
    weighted ramp (score 0.106). At (0, 19) one mismatch is deep in the seed, so
    the seed tie-break prefers it (0.1). Hamming orders by seed mismatches, the
    weighted score does not, so this pair is a within-band disagreement.
    """
    return (FILLER + GUIDE + "TGG" + FILLER
            + _mutate(GUIDE, (10, 11)) + "AGG" + FILLER
            + _mutate(GUIDE, (0, 19)) + "CGG" + FILLER)


def test_within_band_disagreements_are_counted_apart_from_cross_band_ones():
    result = CRISPRDesigner.predict_off_targets(GUIDE, _same_count_reference())
    comparison = result["ranking_comparison"]

    assert comparison["within_band_discordant_pairs"] >= 1
    assert any(
        entry["hamming_prefers"]["mismatches"] == entry["weighted_prefers"]["mismatches"]
        for entry in comparison["disagreements"]
    ), comparison["disagreements"]


def test_guide_rna_gc_content_is_arithmetic_and_the_score_is_labelled():
    result = CRISPRDesigner.design_guide_rna("GGCC")
    assert result["gc_content_percent"] == 100.0
    assert result["on_target_efficiency"] == 100.0          # 85 + 50*0.5, clipped
    assert result["score_basis"] == "linear_gc_rule"
    assert result["off_target_scan"] == "not_performed"
    assert result["evidence"] == "gc_heuristic"
    assert "off_target_search" in result["not_modelled"]


def test_metabolic_titer_is_no_longer_a_prediction():
    result = MetabolicPathwayOptimizer.optimize_pathway("ethanol")
    assert result["theoretical_yield_g_g"] == 0.9
    assert result["assumed_recovery_fraction"] == 0.85
    assert result["optimized_yield_g_g"] == pytest.approx(0.9 * 0.85)
    assert result["titer_computed_g_l"] is None
    assert result["titer_reference_g_l"] == 50
    assert result["evidence"] == "literature_lookup"
    assert "titer" in result["not_modelled"]


def test_module_scope_declarations_resolve():
    for key, entry in SCOPE.items():
        class_name, _, member = key.partition(".")
        cls = getattr(applied_synthetic_bio, class_name)
        assert hasattr(cls, member), key
        assert ("computes" in entry and "evidence" in entry) or entry.get("refuses"), key


# --------------------------------------------------------------------------
# Neuromorphic: reference specs, and requirements that are not silently dropped
# --------------------------------------------------------------------------


def test_chip_advantage_is_a_claim_not_a_measured_property():
    design = NeuromorphicChipDesigner.design_chip()
    assert design["advantage_measured"] is False
    assert design["measured"] is False
    assert design["evidence"] == "reference_spec"
    assert "1000x" in design["claimed_advantage"]
    assert "advantage_over_gpu" not in design


def test_architect_echoes_requirements_it_cannot_honour():
    """Regression: ``requirements`` was accepted and never read."""
    requirements = {"power_mw": 1, "neurons": 5e5, "latency_ms": 2}
    design = NeuromorphicArchitect.design_system(requirements)
    assert design["requirements_echo"] == requirements
    assert design["requirements_used"] == []
    assert design["evidence"] == "reference_spec"


# --------------------------------------------------------------------------
# Climate: a scenario table is not a model
# --------------------------------------------------------------------------


def test_sea_level_figure_is_flagged_as_an_assumed_scaling():
    projection = ClimateModeling.project_temperature("ssp245")
    assert projection["sea_level_rise_is_modelled"] is False
    assert projection["sea_level_rise_assumed_m"] == pytest.approx(2.4 * 0.2)
    assert "assumed" in projection["sea_level_rise_basis"]
    assert "sea_level" in projection["not_modelled"]
    assert projection["evidence"] == "literature_lookup"


def test_dac_removal_potential_is_a_reference_number():
    design = DirectAirCapture.analyze_dac_technology("solid_sorbent")
    assert design["co2_removal_potential_is_computed"] is False
    assert design["co2_removal_potential_reference_gigatons_yr"] == 10
    assert design["energy_requirement_kwh_tCO2"] == 1000
    assert "lifecycle_assessment" in design["not_modelled"]


def test_mineralization_costs_are_reference_values():
    result = CarbonMineralization.analyze_mineralization("olivine")
    assert result["co2_sequestration_capacity_kg_per_kg"] == 0.5
    assert result["cost_usd_tco2_reference"] == 50
    assert "cost_usd_tco2" not in result


# --------------------------------------------------------------------------
# Infrastructure that does not do what its name suggests
# --------------------------------------------------------------------------


def test_academic_api_module_is_configuration_only():
    assert ingestion_academic_apis.CONNECTS_TO_SERVICES is False
    assert ingestion_academic_apis.AUTHENTICATION == "none"
    described = ingestion_academic_apis.describe()
    assert described["connects_to_services"] is False
    assert described["n_targets"] == len(ingestion_academic_apis.INSTITUTIONAL_TARGETS)
    assert "arXiv" in ingestion_academic_apis.list_targets()


def test_web_equation_extraction_returns_a_flat_formula_intact():
    """Regression: the displaystyle branch matched nothing.

    The pattern was double-escaped, so ``\\\\s*`` searched for a literal backslash
    followed by a run of ``s`` -- text real markup never contains. Every result
    the harvester ever produced came from the ``alt=`` branch, and the branch its
    name advertises was dead. Escaping is now generated with ``re.escape``.
    """
    html = "<span class='mwe-math-element'>{\\displaystyle E = mc^2}</span>"
    assert _extract_formula_strings(html) == ["E = mc^2"]


def test_nested_formula_survives_intact():
    """Brace-balanced extraction: a nested group is no longer cut in half.

    The flat character class stopped at the first closing brace, so a formula
    carrying its own braces could never come back whole. The scan counts depth.
    """
    assert EXTRACTION_IS_LOSSY is True
    bs = chr(92)
    inner = "{" + bs + "displaystyle x^{2} = y^{2} + z^{2}}"
    assert _extract_formula_strings("<span>" + inner + "</span>") == [
        "x^{2} = y^{2} + z^{2}"
    ]

    fraction = "{" + bs + "displaystyle a = " + bs + "frac{1}{2}}"
    assert _extract_formula_strings("<span>" + fraction + "</span>") == [
        "a = " + bs + "frac{1}{2}"
    ]


def test_alt_attribute_branch_still_works():
    """The branch that was carrying the module, kept honest."""
    html = '<img alt="{\\displaystyle z^2 = x^2 + y^2}">'
    extracted = _extract_formula_strings(html)
    assert any("z^2 = x^2 + y^2" in formula for formula in extracted), extracted


def test_cluster_launcher_renders_but_never_submits():
    assert SUBMITS_JOBS is False
    described = describe_cluster()
    assert described["submits_jobs"] is False
    assert described["network_access"] is False
    assert described["schedulers"] == ["slurm", "kubernetes", "ray"]

    script = generate_slurm_script(ClusterJobSpec(total_partitions=4, max_parallel=2))
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --array=0-3%2" in script


# --------------------------------------------------------------------------
# The inventory itself must exist on disk (AGENTS.md 1.4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative_path",
    ["docs/SCAFFOLD_INVENTORY.md", "docs/BIOPHARMA_SCOPE.md"],
)
def test_inventory_documents_are_on_disk(relative_path):
    path = REPO_ROOT / relative_path
    assert path.is_file(), relative_path
    assert path.read_text(encoding="utf-8").strip(), relative_path


def test_inventory_page_names_the_refused_calls():
    text = (REPO_ROOT / "docs/SCAFFOLD_INVENTORY.md").read_text(encoding="utf-8")
    for refused in ("predict_off_targets", "optimize_confinement", "discover_targets"):
        assert refused in text, refused


# --------------------------------------------------------------------------
# Second sweep: an absent corpus, a computed module, and a flagged placeholder
# --------------------------------------------------------------------------


def test_patent_corpus_is_bundled_with_its_coverage_declared():
    """The corpus is now real, and small — which is reported, not implied.

    The earlier finding was that an audit against an absent corpus and one
    against a searched corpus looked alike. The corpus now exists; the same
    concern applies to its size, so ``corpus_status`` states the record count,
    the searchable count and the one-claim-per-record scope. The provenance and
    transcription pins live in ``tests/test_patent_corpus_real.py``.
    """
    from apedp import patent_prior_art as patent

    status = patent.corpus_status()
    assert status["bundled"] is True
    assert status["records"] == 29
    assert status["searchable_claims"] == 15
    assert status["decisive_claims"] == 8
    assert status["operand_only_claims"] == 7
    assert status["records_with_formula"] == 7
    assert status["max_claims_per_record"] == 4
    assert status["unsearchable_records"] == 22
    assert status["synthetic_records"] == 0
    assert status["requires_caller_supplied_corpus"] is False
    assert status["claims_scanned_for_formulas"] == 501
    assert status["formulas_found_in_claim_text"] == 0
    assert "one to four per record" in status["reason"].lower()
    assert patent.BUNDLED_CORPUS_PATH is not None
    assert patent.BUNDLED_CORPUS_PATH.exists()


def test_missing_patent_corpus_names_the_absence():
    from apedp.patent_prior_art import load_patent_corpus

    with pytest.raises(FileNotFoundError, match="patent corpus not found") as excinfo:
        load_patent_corpus(REPO_ROOT / "data" / "patents" / "absent-corpus.jsonl")
    assert "bundled at" in str(excinfo.value)


def test_empty_patent_index_reports_no_searchable_claims():
    """An empty corpus must not answer NOT_FOUND, which would read as "searched"."""
    from apedp.patent_prior_art import NO_SEARCHABLE_CLAIMS, PatentPriorArtIndex

    index = PatentPriorArtIndex([])
    result = index.audit("a*x/(1+b*x)")
    assert result.status == NO_SEARCHABLE_CLAIMS
    assert result.searchable_claims == 0
    assert "corpus-relative" in result.epistemic_disclaimer


def test_electrical_module_computes_and_states_its_ceiling():
    from apedp.applied_electrical import PowerSystems

    result = PowerSystems.analyze_grid(source_impedance_pu=0.2)
    assert result["three_phase_fault_mva"] == 500.0            # 100 MVA base / 0.2 pu
    assert "not a load-flow solution" in result["note"]


def test_audio_transport_declares_synthetic_scope():
    from apedp_dyn.audio import spectrogram_transport as audio

    doc = (audio.__doc__ or "").lower()
    assert "synthetic" in doc
    assert "no claim is made" in doc


def test_position_sizing_is_computed_and_the_demo_flag_is_retired():
    """The flag existed because the number was invented; the number is not.

    ``epistemic_position_sizing`` mapped a verdict to leverage, and the flag told
    a reader so. The rule now takes a forecast, a covariance matrix and a fee, so
    there is nothing to disclaim — the pin is that the flag is gone and the size
    responds to its inputs.
    """
    import dataclasses

    from apedp_dyn.sandbox import finance_l2_demo as demo

    assert not hasattr(demo, "POSITION_SIZING_IS_DEMO_ONLY")
    assert not hasattr(demo, "epistemic_position_sizing")
    field_names = {field.name for field in dataclasses.fields(demo.FinanceComparisonResult)}
    assert "position_sizing_is_demo_only" not in field_names
    assert "sizing" in field_names

    decision = demo.size_positions([2e-3, 2e-3], [[1e-4, 0.0], [0.0, 4e-4]], fee_bps=0.0)
    assert decision.admitted is True
    assert decision.weights[0] > decision.weights[1]   # covariance moves the split
    assert decision.cost == 0.0
    thicker = demo.size_positions([2e-3, 2e-3], [[1e-4, 0.0], [0.0, 4e-4]], fee_bps=30.0)
    assert thicker.gross_exposure < decision.gross_exposure  # cost moves the size


def test_finance_demo_insights_no_longer_assert_unobserved_outcomes():
    """Regression: the strings claimed UNIQUE_SUPPORT and 50% sizing unconditionally."""
    import inspect

    from apedp_dyn.sandbox import finance_l2_demo as demo

    source = inspect.getsource(demo)
    assert "reducing position size to 50%" not in source
    assert "converting an abstention into UNIQUE_SUPPORT" not in source


def test_harvested_nested_formula_survives_the_equation_pipeline():
    """The consumer path: a balanced nested formula must reach sympy intact.

    The equation pipeline's offline test injects its own strings, so nothing was
    checking that what the harvester actually returns can be normalised and
    parsed. ``x^{2} = y^{2} + z^{2}`` is the case that could not survive the old
    extractor at all: it was cut at the first inner brace.
    """
    from apedp.equations.normalize import clean_latex, latex_to_sympy_str

    bs = chr(92)
    html = "<span class='mwe-math-element'>{" + bs + "displaystyle x^{2} = y^{2} + z^{2}}</span>"
    harvested = _extract_formula_strings(html)
    assert harvested == ["x^{2} = y^{2} + z^{2}"], harvested

    ok, expr_str, detail = latex_to_sympy_str(clean_latex(harvested[0]))
    assert ok, detail
    assert "x**2" in expr_str, expr_str


# --------------------------------------------------------------------------
# Third sweep: the remaining apedp modules
# --------------------------------------------------------------------------


def _owner_of(module, attribute):
    """The class in a module that carries ``attribute``."""
    import inspect

    for candidate in vars(module).values():
        if inspect.isclass(candidate) and hasattr(candidate, attribute):
            return candidate
    raise AssertionError(f"no class in {module.__name__} has {attribute}")


def test_anscombe_scenario_returns_the_statistics_it_computed():
    """Regression: it computed curvature, asserted on it, then returned a string."""
    from apedp import scenarios

    result = _owner_of(scenarios, "scenario_20_anscombe_metrology")().scenario_20_anscombe_metrology()
    curvature = result["curvature_a_by_dataset"]
    assert set(curvature) >= {"dataset_1", "dataset_2"}
    assert abs(curvature["dataset_2"]) > result["distinguishing_curvature_threshold"]
    assert "dataset_2" in result["distinguishing_datasets"]
    assert set(result["statistics"]) == set(curvature)


def test_publication_text_is_never_ready_for_discovery_and_says_why():
    from apedp.source_ingestion import PublicationText

    text = PublicationText(path="x.pdf", file_hash="0", page_count=1,
                           text="y = m x", warnings=())
    assert text.ready_for_discovery is False
    assert "unverified" in text.readiness_reason


def test_expression_template_grammar_is_a_fixed_small_set():
    from apedp import model_synthesis

    templates = _owner_of(model_synthesis, "_exp_templates")._exp_templates()
    assert len(templates) == 4
    assert all(isinstance(template, tuple) and len(template) == 2 for template in templates)


def test_amplitude_convention_note_is_declared_as_convention():
    from apedp.quantum_checks import amplitude_convention_note

    note = amplitude_convention_note()
    assert "imaginary-unit" in note["bare_I"]
    assert "convention" in amplitude_convention_note.__doc__.lower()


def test_third_sweep_is_recorded_in_the_inventory():
    text = (REPO_ROOT / "docs/SCAFFOLD_INVENTORY.md").read_text(encoding="utf-8")
    for entry in ("Third sweep", "scenario_20_anscombe_metrology", "readiness_reason",
                  "applied_defects.py"):
        assert entry in text, entry


# --------------------------------------------------------------------------
# Static-table census
# --------------------------------------------------------------------------


def _census():
    """Run the census script and return {module: {tables}} with relative paths."""
    import importlib.util

    script = REPO_ROOT / "scripts" / "audit_static_tables.py"
    spec = importlib.util.spec_from_file_location("audit_static_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefix = str(REPO_ROOT / "apedp")
    raw = module.census(REPO_ROOT / "apedp")
    return {
        str(path).replace(prefix, "apedp").replace("\\", "/"):
            {table for _, tables in entries for table in tables}
        for path, entries in raw.items()
    }


def _recorded_tables(root_name):
    """The recorded census for one root, read from the file the check itself uses.

    One record, two readers: ``scripts/audit_static_tables.py --check`` runs on it
    in pre-commit and this suite compares the live tree to it, so the gate and the
    test cannot disagree about what has been recorded.
    """
    import json

    path = REPO_ROOT / "scripts" / "static_table_census.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))["recorded"]
    prefix = root_name.rstrip("/") + "/"
    return {
        module: set(tables)
        for module, tables in recorded.items()
        if module.startswith(prefix)
    }


EXPECTED_STATIC_TABLES = _recorded_tables("apedp")


def test_static_table_census_is_exactly_what_the_inventory_records():
    """A new static table must be recorded before it can ship."""
    assert _census() == EXPECTED_STATIC_TABLES


def test_census_excludes_dataclass_fields_returned_through_self():
    """Regression: every ``to_dict`` returning a tuple-defaulted field was a hit."""
    census = _census()
    for false_positive in ("apedp/ablation.py", "apedp/epistemic.py",
                           "apedp/evidence_report.py", "apedp/source_audit.py"):
        assert false_positive not in census, false_positive


def test_census_modules_are_named_in_the_inventory():
    text = (REPO_ROOT / "docs/SCAFFOLD_INVENTORY.md").read_text(encoding="utf-8")
    assert "Static-table census" in text
    for module in EXPECTED_STATIC_TABLES:
        assert module in text, module


def test_census_check_passes_on_the_recorded_tree():
    """The hook's own command: exit 0 means the tree matches the record."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/audit_static_tables.py", "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "unrecorded_tables" in result.stdout


def test_census_check_fails_on_an_unrecorded_table(tmp_path):
    """The behaviour the hook exists for: a new table exits 1 and is named."""
    import json
    import subprocess
    import sys

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "stub.py").write_text(
        'INTENTS = {"a": 1}\n\n\ndef list_intents():\n    return INTENTS\n',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    # An absolute root keeps the fixture outside the repository, which is the
    # case the check must handle without writing into the tree it guards.
    baseline.write_text(json.dumps({"roots": [str(package)], "recorded": {}}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/audit_static_tables.py", "--check",
         "--baseline", str(baseline)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "pkg/stub.py" in result.stdout
    assert "docs/SCAFFOLD_INVENTORY.md" in result.stderr


def test_census_check_reports_a_relocated_table_as_drift(tmp_path):
    """A table that moves between modules is a stale record, not a match."""
    import importlib.util

    script = REPO_ROOT / "scripts" / "audit_static_tables.py"
    spec = importlib.util.spec_from_file_location("audit_static_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found = {"pkg/stub.py": {"INTENTS"}}
    report = module.drift(found, {"pkg/other.py": {"INTENTS"}})
    assert report["unrecorded_modules"] == ["pkg/stub.py"]
    assert report["vanished_modules"] == ["pkg/other.py"]

    assert module.drift({"pkg/stub.py": {"INTENTS", "MORE"}},
                        {"pkg/stub.py": {"INTENTS"}})["unrecorded_tables"] == {
        "pkg/stub.py": ["MORE"]}
    assert not any(module.drift({"a": {"T"}}, {"a": {"T"}}).values())


def test_pre_commit_runs_the_census_check():
    """The gate is wired where it runs before review, and unconditionally."""
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "static-table-census" in text
    assert "scripts/audit_static_tables.py --check" in text
    hook = text.split("- id: static-table-census", 1)[1].split("- id:", 1)[0]
    assert "always_run: true" in hook
    assert "pass_filenames: false" in hook


# --------------------------------------------------------------------------
# Scope scan: the census rule turned into a gate
# --------------------------------------------------------------------------


def _scope_scan():
    import importlib.util

    script = REPO_ROOT / "scripts" / "scope_scan.py"
    spec = importlib.util.spec_from_file_location("scope_scan", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCOPE_FIXTURE = '''
"""Fixture module for the scope scan."""

REFERENCE_ROWS = [{"grade": "A"}, {"grade": "B"}]


def predict_yield(x):
    return {"yield": 0.9}


def estimate_cost(x):
    return {"cost": 2.0 * x}


def forecast_regime(x):
    if x > 0:
        return "UP"
    return "DOWN"


def search_targets(query):
    return [dict(row) for row in REFERENCE_ROWS]


def design_reactor(power):
    return {"q": power / 50.0}


def optimize_confinement(config):
    raise NotImplementedError("no transport model")


def helper_lookup():
    return {"a": 1}
'''


def _scan_fixture(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "fixture.py").write_text(SCOPE_FIXTURE, encoding="utf-8")
    return package


def test_scope_scan_flags_a_constant_under_a_predictive_name(tmp_path):
    """The four cases that separate a fabricated answer from a computed one."""
    findings = _scope_scan().scan(_scan_fixture(tmp_path), display="pkg")
    found = {(finding["function"], tuple(finding["bases"])) for finding in findings}
    assert found == {
        # Answers without reading its own input, and the answer is a literal.
        ("predict_yield", ("literal",)),
        # A search-shaped name over a module-level table.
        ("search_targets", ("table:REFERENCE_ROWS",)),
    }


def test_scope_scan_leaves_computation_refusal_and_predicates_alone(tmp_path):
    """The exclusions that make the gate usable rather than noisy."""
    findings = _scope_scan().scan(_scan_fixture(tmp_path), display="pkg")
    names = {finding["function"] for finding in findings}
    assert "estimate_cost" not in names          # returns a function of its input
    assert "design_reactor" not in names         # a literal-shaped dict that computes
    assert "forecast_regime" not in names        # literal labels chosen by branching
    assert "optimize_confinement" not in names   # raises: a refusal, not a constant
    assert "helper_lookup" not in names          # a table, under a name that claims nothing


def test_scope_scan_cli_fails_on_an_unrecorded_constant(tmp_path):
    """The repo check itself: a new predictive constant must exit non-zero."""
    import subprocess
    import sys

    package = _scan_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/scope_scan.py", str(package)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "predict_yield" in result.stdout


def test_scope_scan_cli_passes_on_the_recorded_tree():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/scope_scan.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_scope_scan_findings_are_exactly_the_recorded_baseline():
    """New and vanished both fail, because either means the record is stale."""
    scan_module = _scope_scan()
    baseline = scan_module.read_baseline()
    found = set()
    for root in scan_module.DEFAULT_ROOTS:
        for finding in scan_module.scan(REPO_ROOT / root, display=root):
            key = (finding["module"], finding["function"])
            assert tuple(finding["bases"]) == tuple(baseline[key]["bases"]), key
            found.add(key)
    assert found == set(baseline)


def test_scope_scan_baseline_entries_are_explained_and_documented():
    baseline = _scope_scan().read_baseline()
    text = (REPO_ROOT / "docs/SCAFFOLD_INVENTORY.md").read_text(encoding="utf-8")
    assert baseline
    for (module, function), entry in baseline.items():
        why = str(entry.get("why", "")).strip()
        assert len(why.split()) >= 15, function
        assert module in text, module
        assert function in text, function


# --------------------------------------------------------------------------
# Fourth sweep: apedp_dyn, by the same two signals
# --------------------------------------------------------------------------


def _dyn_census():
    """Static-table census over apedp_dyn, excluding the vendored verification harness."""
    import importlib.util

    script = REPO_ROOT / "scripts" / "audit_static_tables.py"
    spec = importlib.util.spec_from_file_location("audit_static_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = REPO_ROOT / "apedp_dyn"
    vendored = ("apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation",)
    found = {}
    for path, entries in module.census(root).items():
        relative = str(path).replace(str(REPO_ROOT) + "/", "").replace("\\", "/")
        if relative.startswith(vendored):
            continue
        found[relative] = {table for _, tables in entries for table in tables}
    return found


EXPECTED_STATIC_TABLES_APEDP_DYN = _recorded_tables("apedp_dyn")


def test_dyn_static_table_census_is_exactly_what_the_inventory_records():
    """The same pin as apedp: a new table in apedp_dyn must be recorded first."""
    assert _dyn_census() == EXPECTED_STATIC_TABLES_APEDP_DYN


def test_dyn_census_counts_the_two_false_positives_it_records():
    """Both are listed, and both are in the census rather than explained away."""
    census = _dyn_census()
    assert census["apedp_dyn/grammar/candidate_ir.py"] == {"parent_candidates"}
    assert census["apedp_dyn/vqc/native.py"] == {"_LAST_FAILURE", "_TALLIES"}


def test_dyn_cost_models_ignore_only_the_inputs_their_formula_has_no_term_for():
    """Measured, not asserted: each model responds to one input and not the others."""
    import numpy as np

    from apedp_dyn.sandbox.execution_cost_models import (
        AlmgrenChrissImpactCostModel,
        FixedPlusSlippageCostModel,
        LimitOrderQueueCostModel,
    )

    positions = np.array([0.0, 1.0, 1.0, 0.0, -1.0])
    prices = np.array([100.0, 100.5, 99.8, 100.2, 100.0])
    n = positions.size

    fixed = FixedPlusSlippageCostModel()
    assert np.allclose(
        fixed.compute_cost_drag(positions, prices, volumes=np.full(n, 1e3)),
        fixed.compute_cost_drag(positions, prices, volumes=np.full(n, 1e9)),
    )
    assert not np.allclose(
        fixed.compute_cost_drag(positions, prices, spreads=np.full(n, 1e-5)),
        fixed.compute_cost_drag(positions, prices, spreads=np.full(n, 1e-2)),
    )

    impact = AlmgrenChrissImpactCostModel()
    assert np.allclose(
        impact.compute_cost_drag(positions, prices, spreads=np.full(n, 1e-5)),
        impact.compute_cost_drag(positions, prices, spreads=np.full(n, 1e-2)),
    )

    queue = LimitOrderQueueCostModel()
    assert np.allclose(
        queue.compute_cost_drag(positions, prices),
        queue.compute_cost_drag(positions, prices * 10.0),
    )


def test_dyn_dead_knobs_are_measured_where_they_can_be_reached():
    """Four readings from the sweep, each a call that moves a knob and sees nothing."""
    import numpy as np

    from apedp_dyn.conf.dimensionless import discover_pi_groups
    from apedp_dyn.epistemic.shape_inference import infer_shape_prior
    from apedp_dyn.robotics.cbf_certificate import CBFQPCertificateAssumptions, certified_cbf_guard
    from apedp_dyn.sandbox.independent_proposer_benchmark import (
        generate_proposer_1_symbolic_cases,
    )

    # variable_units never reaches the group
    exponents = np.array([[1.0, 1.0], [1.0, -2.0], [1.0, -1.0]])
    names = ("a", "b", "c")
    bare = discover_pi_groups(names, exponents)
    with_units = discover_pi_groups(
        names, exponents, variable_units={"a": "N", "b": "m", "c": "s"})
    assert [g.name for g in bare] == [g.name for g in with_units]

    # improvement_margin / min_flat_fraction / min_run_length never reach the prior
    rng = np.random.default_rng(0)
    x = np.repeat(np.linspace(0.0, 1.0, 40), 4)
    y = 2.0 * x + 1.0 + 0.02 * rng.standard_normal(x.size)
    tight = infer_shape_prior(
        x, y, improvement_margin=0.01, min_flat_fraction=0.1, min_run_length=2.0)
    loose = infer_shape_prior(
        x, y, improvement_margin=0.99, min_flat_fraction=0.99, min_run_length=20.0)
    assert tight == loose

    # headings never reach the barrier QP
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    nominal = np.array([[0.3, 0.0], [-0.2, 0.0], [0.0, 0.2]])
    assumptions = CBFQPCertificateAssumptions(
        v_max=1.0, r_safe=0.3, gamma=1.0, dt=0.1, max_agents=3, tol=1e-9)
    straight, straight_diag = certified_cbf_guard(
        positions, np.array([0.0, 0.0, 0.0]), nominal, assumptions)
    turned, turned_diag = certified_cbf_guard(
        positions, np.array([3.0, 1.5, -2.0]), nominal, assumptions)
    assert np.allclose(straight, turned)
    assert straight_diag == turned_diag

    # the benchmark seed drives nothing
    assert (generate_proposer_1_symbolic_cases(seed=1)
            == generate_proposer_1_symbolic_cases(seed=2))


def test_every_census_module_in_apedp_dyn_is_named_in_the_inventory():
    """Twenty-two modules, twenty-two names: seventeen gated, five excluded by record.

    The vendored five are not this repository's tables, so the gate leaves them
    out. They are still written down, because an exclusion nobody can audit is
    indistinguishable from an omission — and a sixth one now fails this test.
    """
    import importlib.util

    script = REPO_ROOT / "scripts" / "audit_static_tables.py"
    spec = importlib.util.spec_from_file_location("audit_static_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    text = (REPO_ROOT / "docs/SCAFFOLD_INVENTORY.md").read_text(encoding="utf-8")
    recorded = _recorded_tables("apedp_dyn")
    assert len(recorded) == 18

    for module_path in recorded:
        # The census table writes paths relative to the package root.
        assert module_path.replace("apedp_dyn/", "") in text, module_path

    vendored = "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
    reported = []
    for path in module.census(REPO_ROOT / "apedp_dyn"):
        relative = str(path).replace(str(REPO_ROOT) + "/", "").replace("\\", "/")
        if relative.startswith(vendored):
            reported.append(relative)
    reported.sort()
    assert len(reported) == 5, reported
    for path in reported:
        assert path in text, path
        assert path not in recorded, path


def test_dyn_sweep_is_recorded_in_the_inventory():
    text = (REPO_ROOT / "docs/SCAFFOLD_INVENTORY.md").read_text(encoding="utf-8")
    assert "Fourth sweep: `apedp_dyn`" in text
    for entry in (
        "compute_cost_drag",
        "infer_shape_prior",
        "check_with_pi_groups",
        "certified_cbf_guard",
        "simulate_type2_fusion",
        "run_eseds_trap",
        "square_window_coefficients",
        "discover_pi_groups",
        "ablation",
        "base_provenance",
    ):
        assert entry in text, entry


def test_scope_scan_covers_both_packages_and_skips_vendored_code():
    scan_module = _scope_scan()
    assert scan_module.DEFAULT_ROOTS == ("apedp", "apedp_dyn")
    for vendored in ("_qualtran_src", "__pycache__", "tests", "benchmark_corpus"):
        assert vendored in scan_module.SKIPPED_PARTS, vendored
    flagged = {finding["module"] for finding in scan_module.scan(REPO_ROOT / "apedp_dyn", display="apedp_dyn")}
    assert not any("_qualtran_src" in module for module in flagged)
