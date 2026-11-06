"""The photograph audit of the derived region masks.

``apedp_dyn.video.region_masks`` was only ever tested against synthetic latents
in which the textured rows were placed by the test. These tests pin the audit
against real photographs: that the images are present and unmodified, that each
one carries a licence and a source, that a hand label exists for every one of
them, and that the recorded verdicts are the ones the module actually produces.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "region_mask_audit"


def _load_json(name: str) -> dict:
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def record() -> dict:
    return _load_json("audit_record.json")


def test_the_audit_record_covers_ten_photographs(record):
    assert record["schema"] == "region_mask_audit/v1"
    assert len(record["photos"]) == 10
    assert record["summary"]["photos"] == 10


def test_every_audited_photo_is_on_disk_and_unmodified(record):
    """A record that names a picture which is not there proves nothing."""
    for row in record["photos"]:
        path = AUDIT / "photos" / row["file"]
        assert path.exists(), f"{row['file']} is recorded but missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"], f"{row['file']} changed since the audit"


def test_every_photo_has_a_hand_label_and_no_label_is_orphaned(record):
    labelled = _load_json("ground_truth.json")["labelled"]
    assert {row["file"] for row in record["photos"]} == set(labelled)


def test_provenance_names_a_source_and_a_licence_for_each_photo():
    provenance = _load_json("PROVENANCE.json")
    assert provenance["counts"]["photos"] == len(provenance["photos"]) > 0
    for photo in provenance["photos"]:
        assert photo["page_title"], "an unattributed photograph is not auditable"
        assert photo["description_url"]
        assert photo["licence"], photo["page_title"]
        assert photo["bytes"] >= 4096
        assert photo["sha256"]


def test_a_file_that_is_not_a_photograph_is_refused_with_a_reason():
    """The first fetch returned an oil painting for "snow field"."""
    reasons = [r["reason"] for r in _load_json("PROVENANCE.json")["refusals"]]
    assert any("not a photograph" in reason for reason in reasons), reasons


def test_the_split_is_wrong_on_more_photographs_than_it_lands_close(record):
    summary = record["summary"]
    assert summary["scored"] == 7
    assert summary["close"] == 2
    assert summary["partial"] == 1
    assert summary["wrong"] == 4
    # Median error over the seven scored photographs. If this ever drops below
    # a tenth of the frame height the record is describing a different module.
    assert summary["median_abs_error"] > 0.15


def test_photographs_without_three_bands_still_get_two_boundaries(record):
    """The finding: there is no output that says "this frame has no bands".

    Three of the ten photographs have one transition or none, and the splitter
    returns two interior boundaries for every one of them.
    """
    undefined = [row for row in record["photos"]
                 if row["comparison"]["verdict"] == "undefined"]
    assert len(undefined) == 3
    for row in undefined:
        assert row["comparison"]["boundaries_returned"] == 2
        assert row["comparison"]["reason"]
        assert row["expected"] is None, "a scene with no bands must not be labelled"


def test_the_recorded_boundaries_reproduce_from_the_photograph(record):
    from apedp_dyn.video.region_masks import derive_region_masks

    rows = {row["file"]: row for row in record["photos"]}
    for name in ("mountain_lake_reflection.jpg", "sand_dunes.jpg"):
        with Image.open(AUDIT / "photos" / name) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.asarray(image, dtype=np.float64) / 255.0
        got = derive_region_masks(arr.transpose(2, 0, 1))["boundaries"]
        assert got == rows[name]["boundaries"], name


def test_the_gate_is_recorded_for_every_photograph(record):
    for row in record["photos"]:
        gate = row["gate"]
        assert isinstance(gate["reliable"], bool)
        assert isinstance(gate["reasons"], list)
        assert gate["reliable"] is (not gate["reasons"])


def test_the_gate_never_accepts_a_split_that_is_wrong(record):
    """The point of a refusal gate: it may be timid, it may not be wrong.

    It is timid -- it rejects one split that was in fact within tolerance --
    which is the acceptable direction. Every split it accepts is one the hand
    labels agree with.
    """
    accepted = [row for row in record["photos"] if row["gate"]["reliable"]]
    assert accepted, "a gate that accepts nothing proves nothing"
    for row in accepted:
        assert row["comparison"]["verdict"] in ("close", "partial"), (
            f"gate accepted {row['file']} whose boundaries are "
            f"{row['comparison']['verdict']}")
    assert record["summary"]["gate_accepts_that_are_wrong"] == []


def test_a_frame_with_no_ordered_bands_is_refused_when_asked():
    """Textured above and below with a smooth middle has no ordered stack."""
    from apedp_dyn.video.region_masks import (
        UnusableRegionSplit,
        derive_region_masks,
    )

    rng = np.random.default_rng(0)
    frame = np.zeros((3, 64, 64), dtype=np.float64)
    frame[:, :16, :] = rng.normal(0.0, 1.0, size=(3, 16, 64))
    frame[:, 48:, :] = rng.normal(0.0, 1.0, size=(3, 16, 64))

    with pytest.raises(UnusableRegionSplit, match="no defensible three-band"):
        derive_region_masks(frame, require_reliable=True)

    # Asked politely, it returns the split and says why it is not supported.
    regions = derive_region_masks(frame)
    assert regions["evidence"]["reliable"] is False
    assert regions["evidence"]["texture_order"] == "non_monotone"
    assert any("non-monotone" in reason
               for reason in regions["evidence"]["reasons"])
    assert set(regions["masks"]) == {"upper", "middle", "lower"}
