"""The depth-model comparison record: pinned so its numbers cannot drift.

`scripts/compare_region_split_to_depth.py` runs the same band splitter on a
pretrained monocular depth model (MiDaS) and on RGB texture, scores both against
the same hand labels, and writes `data/region_mask_audit/model_comparison.json`.

The recorded outcome is a negative one: the depth signal did **not** beat the
texture heuristic on these ten photographs, and the splitter's own evidence gate
refused every depth split. These tests pin that, so replacing the signal with a
better model or a fairer input size has to regenerate the record rather than
quietly contradict it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "region_mask_audit" / "model_comparison.json"


@pytest.fixture(scope="module")
def record() -> dict:
    if not RECORD.exists():
        pytest.skip(f"{RECORD.name} has not been generated")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_the_record_names_a_real_pretrained_model_with_pinned_weights(record):
    assert record["schema"] == "region_mask_model_comparison/v1"
    model = record["model"]
    assert model["hub_repo"] == "intel-isl/MiDaS"
    assert model["model"] == "MiDaS_small"
    # A comparison against weights nobody can identify is not reproducible.
    weights = model.get("weights") or {}
    assert weights.get("sha256"), model
    assert weights.get("bytes", 0) > 10_000_000
    for key in ("torch", "torchvision", "timm", "device"):
        assert model.get(key), key


def test_both_signals_went_through_the_same_splitter_and_labels(record):
    assert record["analysis_size"] == 256
    assert record["tolerance_height_fraction"] == 0.06
    photos = record["photos"]
    assert len(photos) == 10
    for row in photos:
        for key in ("depth_boundaries", "texture_boundaries"):
            assert set(row[key]) == {"upper_to_middle", "middle_to_lower"}, key
        assert row["depth_evidence"], row["file"]


def test_the_recorded_outcome_is_that_depth_did_not_beat_texture(record):
    texture, depth = record["summary"]["texture"], record["summary"]["depth"]
    assert texture["scored"] == depth["scored"] == 7
    assert texture["median_abs_error"] < depth["median_abs_error"]
    assert texture["close"] > depth["close"]


def test_the_depth_signal_is_dominated_by_a_vantage_ramp(record):
    """The measured reason the swap failed.

    A step-seeking split needs structure that is not a ramp; MiDaS's row-mean
    depth is closer to a straight line down the frame (median R^2 0.83) than the
    texture profile is (0.24), and it rises toward the bottom in every photo.
    """
    ramp = record["summary"]["ramp_dominance"]
    assert ramp["depth_median_ramp_r2"] > 0.7, ramp
    assert ramp["depth_median_ramp_r2"] > ramp["texture_median_ramp_r2"], ramp
    for row in record["photos"]:
        assert row["depth_ramp"]["ramp_spearman"] > 0.7, row["file"]


def test_every_depth_split_was_refused_by_the_evidence_gate(record):
    """The strongest part of the finding: the gate never endorsed a depth split.

    The depth maps are smooth, so a cut through them is usually not a cut at all
    -- which is exactly what the half-disagreement and ordering checks report.
    """
    assert record["summary"]["depth_reliable"] == 0
    for row in record["photos"]:
        assert row["depth_evidence"]["reliable"] is False
        assert row["depth_evidence"]["reasons"], row["file"]
