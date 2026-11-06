"""The semantic-model comparison record: pinned so its numbers cannot drift.

`scripts/compare_region_split_to_semantics.py` runs a pretrained ADE20K
segmentation model (SegFormer-B0) on the ten audit photographs, cuts three bands
where the *class composition* changes most, and scores that split against the
same hand labels and tolerance the texture and depth arms used.

The recorded outcome is narrow on purpose. Semantics is the best signal of the
three by median boundary error (0.115 against texture's 0.182 and depth's 0.228)
and it fixes **no verdicts** — the close/partial/wrong counts are identical to
texture's. These tests pin both halves, so a better model has to regenerate the
record rather than quietly contradict it, and so the win cannot be reported as
more than it is.

The record also carries a structural measurement: a class map cannot be handed
to the texture splitter, because ADE20K ids are nominal and differencing them
measures the vocabulary's numbering. That splitter does not fail loudly — it
returns boundaries and only its evidence gate refuses them — so the measurement
is pinned too.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "region_mask_audit" / "semantic_comparison.json"
SCRIPT = ROOT / "scripts" / "compare_region_split_to_semantics.py"


@pytest.fixture(scope="module")
def record() -> dict:
    if not RECORD.exists():
        pytest.skip(f"{RECORD.name} has not been generated")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _module():
    """Load the comparison script as a module (it is not on the import path)."""
    pytest.importorskip("PIL")
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT.name} is missing")
    spec = importlib.util.spec_from_file_location("compare_semantics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_record_names_a_real_pretrained_model_with_pinned_weights(record):
    assert record["schema"] == "region_mask_semantic_comparison/v1"
    model = record["model"]
    assert model["hub_repo"] == "nvidia/segformer-b0-finetuned-ade-512-512"
    assert model["num_classes"] == 150
    # A comparison against weights nobody can identify is not reproducible.
    weights = model.get("weights") or {}
    assert weights.get("sha256"), model
    assert weights.get("bytes", 0) > 1_000_000
    for key in ("torch", "transformers", "device"):
        assert model.get(key), key
    # The vocabulary digest pins which class names the bands were read in.
    assert len(model["id2label_sha256"]) == 64


def test_the_three_arms_are_held_to_one_ruler(record):
    fixed = record["fixed_across_arms"]
    assert fixed["analysis_size"] == 256
    assert fixed["tolerance_height_fraction"] == 0.06
    assert fixed["min_band_fraction"] == 0.15
    # The evidence thresholds are imported from region_masks, not re-declared,
    # so a change to the gate must move this record with it.
    from apedp_dyn.video.region_masks import (
        MAX_HALF_DISAGREEMENT,
        MIN_CUT_DECISIVENESS,
    )
    assert fixed["min_cut_decisiveness"] == MIN_CUT_DECISIVENESS
    assert fixed["max_half_disagreement"] == MAX_HALF_DISAGREEMENT
    assert "two per-row scores" in fixed["two_cut_selection_rule"]


def test_semantics_has_the_lowest_median_error_but_changes_no_verdict(record):
    summary = record["summary"]
    assert summary["photos"] == 10
    assert summary["labelled"] == 7
    semantic, texture, depth = (summary["semantic"], summary["texture"],
                                summary["depth"])
    assert semantic["median_abs_error"] < texture["median_abs_error"]
    assert texture["median_abs_error"] < depth["median_abs_error"]
    assert summary["ranking_by_median_abs_error"] == [
        "semantic", "texture", "depth"]
    # The honest half: the same photographs pass and fail, under every signal.
    for arm in (semantic, texture):
        assert arm["scored"] == 7
        assert (arm["close"], arm["partial"], arm["wrong"]) == (2, 1, 4)
    assert summary["verdict_changed_vs_texture"] == []


def test_the_evidence_gate_still_refuses_most_semantic_splits(record):
    """Semantics is better, not settled: six of ten splits are still refused."""
    assert record["summary"]["semantic_reliable"] == 4
    assert record["summary"]["semantic_reliable_and_close"] == 1
    for row in record["photos"]:
        evidence = row["semantic_evidence"]
        assert evidence["reliable"] == (not evidence["reasons"]), row["file"]
        if not evidence["reliable"]:
            assert evidence["reasons"], row["file"]


def test_the_bands_record_what_content_they_separated(record):
    """A split is only meaningful if the bands hold different content.

    The adapted third gate check refuses when all three bands share one
    dominant class; the record has to show the dominant classes it judged, or
    that refusal would be unauditable.
    """
    for row in record["photos"]:
        for band in ("upper", "middle", "lower"):
            content = row["semantic_bands"][band]
            assert content["dominant"], (row["file"], band)
            assert isinstance(content["reported_classes"], dict)
        if row["semantic_evidence"]["distinct_dominant_classes"] == 1:
            assert not row["semantic_evidence"]["reliable"], row["file"]
        if row["file"] == "mountain_lake_reflection.jpg":
            # Three clean regimes, and ADE20K's vocabulary should name them.
            assert "sky" in row["semantic_bands"]["upper"]["reported_classes"]
            assert "mountain" in row["semantic_bands"]["middle"][
                "reported_classes"]


def test_a_class_map_is_recorded_as_invalid_input_to_the_texture_splitter(record):
    """Nominal ids, not intensities: the splitter accepts it and the gate refuses.

    This is why the semantic arm needs its own score rather than the same
    splitter with a different array, as the depth arm could use.
    """
    note = record["class_arm_structural_note"]
    assert "nominal" in note["why_not_the_same_splitter"]
    measured = note["invalid_input_measurement"]
    assert measured["label_ids_are_nominal"] is True
    assert measured["texture_split_reliable"] is False
    assert measured["texture_split_cut_decisiveness"] == 0.0
    assert measured["texture_split_reasons"], measured
    # It returns boundaries all the same -- which is the point of recording it.
    assert set(measured["texture_split_boundaries"]) == {
        "upper_to_middle", "middle_to_lower"}


def test_the_splitter_cuts_where_the_class_composition_changes():
    """A synthetic three-region frame: the cuts must land on the real edges."""
    module = _module()
    id2label = {0: "sky", 1: "water", 2: "sand"}
    class_map = np.zeros((256, 256), dtype=np.int32)
    class_map[100:180, :] = 1
    class_map[180:, :] = 2

    split = module.semantic_split(class_map, id2label)

    assert abs(split["boundaries"]["upper_to_middle"] - 100 / 256) < 0.03
    assert abs(split["boundaries"]["middle_to_lower"] - 180 / 256) < 0.03
    assert split["evidence"]["reliable"] is True, split["evidence"]
    assert split["evidence"]["distinct_dominant_classes"] == 3
    assert next(iter(split["bands"]["upper"]["dominant"])) == "sky"
    assert next(iter(split["bands"]["middle"]["dominant"])) == "water"
    assert next(iter(split["bands"]["lower"]["dominant"])) == "sand"


def test_a_uniform_class_map_is_refused_rather_than_split():
    """One class everywhere: two cuts would separate nothing, so the gate says so."""
    module = _module()
    id2label = {0: "sky", 1: "water"}
    class_map = np.zeros((256, 256), dtype=np.int32)

    split = module.semantic_split(class_map, id2label)

    assert split["evidence"]["reliable"] is False
    assert split["evidence"]["distinct_dominant_classes"] == 1
    assert any("dominated by" in reason
               for reason in split["evidence"]["reasons"])
