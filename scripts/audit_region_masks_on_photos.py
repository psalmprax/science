"""Audit the derived region masks against real photographs.

``apedp_dyn.video.region_masks`` splits a frame into three bands by row-texture
energy and lets the caller name them. Its own docstring is careful: "what the
bands mean is a decision this module does not make". That care is worth nothing
without evidence about how often the split lands somewhere defensible, so this
script runs it on photographs and records every place the bands come out wrong.

For each image it reports what the split *did* (boundaries, band energies) and
three diagnostics that predict when a row split cannot work:

    texture_order      which way the energy runs: increasing / decreasing /
                       non_monotone. A photograph whose subject is smoother
                       than its background reverses the order the animator
                       assumes when it names the upper band "sky".
    step_contrast      the two chosen row steps over the median row step. A cut
                       at 1.2x the median is a cut through noise.
    diagonal_shift     the same cut search run on the left half and the right
                       half separately, reporting how far apart they land. A
                       horizontal boundary gives the same cut on both sides; a
                       diagonal horizon does not, and a row split has no way to
                       follow one.

Where a hand labelling exists (``ground_truth.json``, read from the pictures
and not from the module) the record also reports the boundary error in height
fractions and a verdict: ``close`` (both within ``--tolerance``), ``partial``
(one of them), ``wrong`` (neither).

Usage::

    python3 scripts/audit_region_masks_on_photos.py \
        --photos data/region_mask_audit/photos \
        --ground-truth data/region_mask_audit/ground_truth.json \
        --out data/region_mask_audit/audit_record.json

    # or preview the tiles with a height ruler, for labelling by eye
    python3 scripts/audit_region_masks_on_photos.py \
        --photos data/region_mask_audit/photos --contact-sheet /tmp/sheet.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from apedp_dyn.video.region_masks import (
    _smooth,
    derive_region_masks,
    row_texture_energy,
)

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _load(path: Path, size: int) -> np.ndarray:
    """Load as a (3, size, size) float64 array in [0, 1], EXIF-oriented."""
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float64) / 255.0
    return arr.transpose(2, 0, 1)


def _order(means: tuple[float, float, float]) -> str:
    upper, middle, lower = means
    if upper < middle < lower:
        return "increasing"
    if upper > middle > lower:
        return "decreasing"
    return "non_monotone"


def _best_two_cuts(
    energy: np.ndarray, min_band_fraction: float,
) -> tuple[int, int, float]:
    """The (b1, b2, score) the splitter would choose for this profile."""
    h = energy.size
    min_len = max(1, round(min_band_fraction * h))
    step = np.abs(np.diff(energy, prepend=energy[0]))
    best = (-np.inf, min_len, h - min_len)
    for b1 in range(min_len, h - 2 * min_len + 1):
        for b2 in range(b1 + min_len, h - min_len + 1):
            score = step[b1] + step[b2]
            if score > best[0]:
                best = (score, b1, b2)
    return best[1], best[2], best[0]


def _diagnose(
    latent: np.ndarray, regions: dict, min_band_fraction: float, smoothing: int,
) -> dict[str, Any]:
    h = latent.shape[1]
    b1 = round(regions["boundaries"]["upper_to_middle"] * h)
    b2 = round(regions["boundaries"]["middle_to_lower"] * h)
    energy = _smooth(row_texture_energy(latent), smoothing)
    step = np.abs(np.diff(energy, prepend=energy[0]))
    median_step = float(np.median(step)) or 1e-12
    spread = float(energy.max() - energy.min())
    range_scale = spread if spread > 0 else 1e-12

    width = latent.shape[2]
    halves = {}
    for side, sl in (("left", slice(0, width // 2)),
                     ("right", slice(width // 2, width))):
        half_energy = _smooth(row_texture_energy(latent[:, :, sl]), smoothing)
        hb1, hb2, _ = _best_two_cuts(half_energy, min_band_fraction)
        halves[side] = (hb1, hb2)
    shifts = {
        "upper_to_middle": abs(halves["left"][0] - halves["right"][0]) / h,
        "middle_to_lower": abs(halves["left"][1] - halves["right"][1]) / h,
    }

    means = (
        regions["energy"]["upper_band_mean"],
        regions["energy"]["middle_band_mean"],
        regions["energy"]["lower_band_mean"],
    )
    return {
        "texture_order": _order(means),
        "band_energy_means": dict(zip(("upper", "middle", "lower"), means,
                                     strict=True)),
        "band_energy_spread_fraction": round(spread / (energy.max() or 1.0), 4),
        "chosen_step_over_median": round(
            float(step[b1] + step[b2]) / median_step, 3),
        "largest_step_over_profile_spread": round(
            float(step.max()) / range_scale, 3),
        "halves": {
            "left": {"upper_to_middle": round(halves["left"][0] / h, 4),
                     "middle_to_lower": round(halves["left"][1] / h, 4)},
            "right": {"upper_to_middle": round(halves["right"][0] / h, 4),
                      "middle_to_lower": round(halves["right"][1] / h, 4)},
            "diagonal_shift": {k: round(v, 4) for k, v in shifts.items()},
            "max_shift": round(max(shifts.values()), 4),
        },
    }


def _verdict(
    got: dict[str, float], expected: dict[str, float], tolerance: float,
) -> dict[str, Any]:
    errors = {
        key: round(got[key] - expected[key], 4)
        for key in ("upper_to_middle", "middle_to_lower")
        if key in expected
    }
    within = {k: abs(v) <= tolerance for k, v in errors.items()}
    if not within:
        label = "unlabelled"
    elif all(within.values()):
        label = "close"
    elif any(within.values()):
        label = "partial"
    else:
        label = "wrong"
    return {
        "errors": errors,
        "abs_errors": {k: round(abs(v), 4) for k, v in errors.items()},
        "max_abs_error": round(max((abs(v) for v in errors.values()), default=0.0), 4),
        "within_tolerance": within,
        "verdict": label,
    }


def audit_photo(
    path: Path, size: int, tolerance: float, min_band_fraction: float,
    smoothing: int, expected: dict[str, float] | None,
    applicable: bool = True,
) -> dict[str, Any]:
    latent = _load(path, size)
    regions = derive_region_masks(
        latent, smoothing=smoothing, min_band_fraction=min_band_fraction)
    diagnostics = _diagnose(latent, regions, min_band_fraction, smoothing)
    row: dict[str, Any] = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "analysed_size": [size, size],
        "boundaries": regions["boundaries"],
        "diagnostics": diagnostics,
        # The module's own verdict on its split, recorded so the audit can ask
        # whether the gate is right, not just whether the boundaries are.
        "gate": {
            "reliable": regions["evidence"]["reliable"],
            "reasons": regions["evidence"]["reasons"],
            "cut_decisiveness": regions["evidence"]["cut_decisiveness"],
            "half_disagreement": regions["evidence"]["half_disagreement"],
            "texture_order": regions["evidence"]["texture_order"],
        },
    }
    if not applicable:
        # The photograph has no three-band horizontal structure. The module
        # still returned two boundaries, which is the finding: there is no
        # output that says "this frame has no bands to split".
        row["expected"] = None
        row["comparison"] = {
            "verdict": "undefined",
            "boundaries_returned": 2,
            "reason": "scene has no three-band horizontal structure; the "
                      "splitter returned two boundaries anyway",
        }
    elif expected:
        row["expected"] = expected
        row["comparison"] = _verdict(
            regions["boundaries"], expected, tolerance)
    else:
        row["expected"] = None
        row["comparison"] = {"verdict": "unlabelled"}
    return row


def contact_sheet(paths: list[Path], out: Path, size: int) -> None:
    """Tiles with a 10 %-of-height ruler, so boundaries can be read by eye."""
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    label_h = 16
    tile_w, tile_h = size, size + label_h
    gap = 10
    sheet = Image.new(
        "RGB", (columns * tile_w + (columns + 1) * gap,
                rows * tile_h + (rows + 1) * gap), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as handle:
            tile = ImageOps.exif_transpose(handle).convert("RGB")
            tile = tile.resize((size, size), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + gap)
        sheet.paste(tile, (x, y))
        for tenth in range(1, 10):
            ty = y + round(tile_h * tenth / 10)
            draw.line([(x, ty), (x + 10, ty)], fill=(255, 0, 0), width=1)
            draw.text((x + 12, ty - 5), f".{tenth}", fill=(255, 60, 60))
        draw.text((x + 2, y + tile_h + 3), path.stem[:34], fill=(220, 220, 220))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", default="data/region_mask_audit/photos")
    parser.add_argument("--ground-truth",
                        default="data/region_mask_audit/ground_truth.json")
    parser.add_argument("--out", default="data/region_mask_audit/audit_record.json")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=0.06,
                        help="a boundary within this fraction of the height is "
                             "'close'; 0.06 is 15 rows at 256 px")
    parser.add_argument("--min-band-fraction", type=float, default=0.15)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--contact-sheet", default=None)
    args = parser.parse_args()

    photos_dir = Path(args.photos)
    paths = sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not paths:
        raise SystemExit(f"no photographs found in {photos_dir}")

    if args.contact_sheet:
        contact_sheet(paths, Path(args.contact_sheet), args.size)
        print(f"wrote contact sheet to {args.contact_sheet}")

    truth: dict[str, Any] = {}
    truth_path = Path(args.ground_truth)
    if truth_path.exists():
        truth = json.loads(truth_path.read_text(encoding="utf-8")).get(
            "labelled", {})

    rows = []
    for path in paths:
        entry = truth.get(path.name, {})
        applicable = entry.get("applicable", bool(entry.get("expected")))
        row = audit_photo(
            path, args.size, args.tolerance, args.min_band_fraction,
            args.smoothing, entry.get("expected") if entry else None,
            applicable=applicable)
        if entry:
            row["ground_truth_note"] = entry.get("note")
            row["scene_regimes"] = entry.get("regimes")
        rows.append(row)
        verdict = row["comparison"]["verdict"]
        print(f"  {path.stem:22s} {verdict:11s} "
              f"b1={row['boundaries']['upper_to_middle']:<6} "
              f"b2={row['boundaries']['middle_to_lower']:<6} "
              f"{row['diagnostics']['texture_order']:13s} "
              f"shift={row['diagnostics']['halves']['max_shift']}")

    verdicts = [r["comparison"]["verdict"] for r in rows]
    scored = [r for r in rows
              if r["comparison"]["verdict"] in ("close", "partial", "wrong")]
    wrong = [r["file"] for r in scored
             if r["comparison"]["verdict"] == "wrong"]
    undefined = [r["file"] for r in rows
                 if r["comparison"]["verdict"] == "undefined"]
    record = {
        "schema": "region_mask_audit/v1",
        "photos_dir": str(args.photos),
        "method": (
            "derive_region_masks run on the RGB image resized to "
            f"{args.size}x{args.size}; ground truth hand-labelled from the "
            "pictures, not from the module"),
        "tolerance_height_fraction": args.tolerance,
        "min_band_fraction": args.min_band_fraction,
        "smoothing": args.smoothing,
        "photos": rows,
        "summary": {
            "photos": len(rows),
            "scored": len(scored),
            "close": verdicts.count("close"),
            "partial": verdicts.count("partial"),
            "wrong": verdicts.count("wrong"),
            "undefined": verdicts.count("undefined"),
            "unlabelled": verdicts.count("unlabelled"),
            "wrong_files": wrong,
            "undefined_files": undefined,
            "max_abs_error": max(
                (r["comparison"]["max_abs_error"] for r in scored), default=None),
            "median_abs_error": (
                round(float(np.median([
                    r["comparison"]["max_abs_error"] for r in scored])), 4)
                if scored else None),
            "non_monotone": [r["file"] for r in rows
                             if r["diagnostics"]["texture_order"] == "non_monotone"],
            "diagonal_shift_over_tolerance": [
                r["file"] for r in rows
                if r["diagnostics"]["halves"]["max_shift"] > args.tolerance],
            "weak_cuts": [
                r["file"] for r in rows
                if r["diagnostics"]["chosen_step_over_median"] < 2.0],
            "gate_accepts": [
                r["file"] for r in rows if r["gate"]["reliable"]],
            "gate_accepts_that_are_wrong": [
                r["file"] for r in rows if r["gate"]["reliable"]
                and r["comparison"]["verdict"] == "wrong"],
            "gate_rejects_that_were_close": [
                r["file"] for r in rows if not r["gate"]["reliable"]
                and r["comparison"]["verdict"] == "close"],
        },
        "limitations": (
            f"{len(rows)} photographs from one source, one resize, one "
            "smoothing window, labelled by one pair of eyes at about 0.03 "
            "height-fraction resolution. The band meaning is still assigned by "
            "the caller, so this audit measures whether the boundaries sit on a "
            "horizontal texture change, not whether the upper band is a sky."),
    }
    Path(args.out).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} photos: scored={len(scored)} close={verdicts.count('close')} "
          f"partial={verdicts.count('partial')} wrong={verdicts.count('wrong')} "
          f"undefined={verdicts.count('undefined')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
