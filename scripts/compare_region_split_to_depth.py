"""Compare the row-texture band split against a pretrained monocular depth model.

`apedp_dyn.video.region_masks` finds three horizontal bands by cutting at the two
sharpest steps in per-row *texture* energy. This script asks what happens when
the same splitter is given *depth* instead: MiDaS (Ranftl et al., "Towards Robust
Monocular Depth Estimation with Mixture of Density and Scale", 2022) predicts a
relative inverse-depth map per pixel, and the two sharpest steps in the per-row
mean depth are, in principle, exactly the boundaries the texture heuristic is
trying to find.

Design note — the algorithm is held fixed on purpose. The depth map is stacked
into three identical channels and handed to `derive_region_masks` unchanged, so
the same `min_band_fraction`, the same smoothing and the same argmax-of-two-steps
rule apply. Any difference in the resulting boundaries is therefore attributable
to the signal (depth versus RGB texture), not to a different boundary search. The
evidence gate built into `derive_region_masks` also runs, so a depth split that
is unreliable says so.

Labels and the verdict rule come from the photograph audit
(`data/region_mask_audit/ground_truth.json`, hand-labelled; 0.06 of the frame
height counts as close), and the texture verdicts are read back from
`audit_record.json`, so the two methods are scored by one rule against one set of
labels.

Interpreter: this needs torch, torchvision and timm, which the base interpreter
does not have. Build the pinned environment once:

    python3 -m venv --system-site-packages .venv-region-models
    .venv-region-models/bin/pip install --no-deps \\
        --index-url https://download.pytorch.org/whl/cpu "torchvision==0.18.1+cpu"
    .venv-region-models/bin/pip install --no-deps "timm==0.9.16"

then

    .venv-region-models/bin/python scripts/compare_region_split_to_depth.py

Output: `data/region_mask_audit/model_comparison.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from audit_region_masks_on_photos import _verdict  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from apedp_dyn.video.region_masks import (  # noqa: E402
    derive_region_masks,
    row_texture_energy,
)

AUDIT = ROOT / "data" / "region_mask_audit"
MODEL = {"hub_repo": "intel-isl/MiDaS", "name": "MiDaS_small",
         "transform": "small_transform",
         "citation": "Ranftl et al., MiDaS v3.1 (2022)",
         "output_semantics": "relative inverse depth; larger = nearer"}


def _require_backends():
    try:
        import timm
        import torch
        import torchvision
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{exc.name} is not available in this interpreter. The depth "
            "comparison needs torch, torchvision and timm; see the docstring "
            "for the pinned .venv-region-models setup.") from exc
    return torch, torchvision, timm


def load_model():
    torch, torchvision, timm = _require_backends()
    model = torch.hub.load(MODEL["hub_repo"], MODEL["name"],
                           trust_repo=True, verbose=False).eval()
    transforms = torch.hub.load(MODEL["hub_repo"], "transforms",
                                trust_repo=True, verbose=False)
    provenance = {
        "hub_repo": MODEL["hub_repo"],
        "model": MODEL["name"],
        "transform": MODEL["transform"],
        "citation": MODEL["citation"],
        "output_semantics": MODEL["output_semantics"],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": "cpu",
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    weights = _weights_digest(torch)
    if weights:
        provenance["weights"] = weights
    return model, transforms.small_transform, provenance


def ramp_fit(profile: np.ndarray) -> dict:
    """How much of a row profile is just a linear top-to-bottom ramp.

    This is the diagnostic that explains the result. A step-seeking split needs
    structure that is *not* a ramp; a monocular depth network carries a strong
    vantage prior (the bottom of a photograph is usually nearer), so its row-mean
    output can be almost linear whatever the scene contains. Measured rather than
    asserted: R^2 of the profile on row index, its rank correlation, and the
    total rise across the frame.
    """
    x = np.arange(profile.size, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, profile, rcond=None)
    fitted = design @ coefficients
    ss_res = float(np.sum((profile - fitted) ** 2))
    ss_tot = float(np.sum((profile - profile.mean()) ** 2))
    centered = profile - profile.mean()
    denom = float(np.sqrt(np.sum(centered ** 2) * np.sum((x - x.mean()) ** 2)))
    return {
        "ramp_r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
        "ramp_spearman": (round(float(np.sum(centered * (x - x.mean())) / denom), 4)
                          if denom > 0 else 0.0),
        "rise_across_frame": round(float(coefficients[1]) * profile.size, 3),
    }


def _weights_digest(torch) -> dict | None:
    """SHA-256 of the cached MiDaS_small checkpoint, when it is on disk."""
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    candidates = sorted(cache.glob("*MiDaS*small*")) + sorted(
        cache.glob("*midas*small*"))
    for path in candidates:
        if path.is_file():
            return {"file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return None


def depth_rows(model, transform, image_rgb: np.ndarray, size: int,
               torch) -> np.ndarray:
    """Relative inverse depth, resized to the audit's analysis size."""
    batch = transform(image_rgb)
    if isinstance(batch, dict):          # some MiDaS transforms return a map
        batch = batch["image"]
    if not hasattr(batch, "unsqueeze"):  # numpy CHW
        batch = torch.from_numpy(np.asarray(batch))
    # The transform's output arity is version-dependent: 3-D (CHW) needs a batch
    # dimension, 4-D is already batched. Assuming one of them produced a 5-D
    # tensor and a conv2d shape error on the first run.
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    if batch.ndim != 4:
        raise ValueError(f"unexpected model input shape {tuple(batch.shape)}")
    batch = batch.float()
    with torch.no_grad():
        prediction = model(batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(size, size), mode="bicubic",
            align_corners=False)
    return prediction.squeeze().cpu().numpy().astype(np.float64)


def run(size: int, limit: int | None = None) -> dict:
    torch, _tv, _timm = _require_backends()
    model, transform, provenance = load_model()
    labels = json.loads((AUDIT / "ground_truth.json").read_text(encoding="utf-8"))
    labelled = labels["labelled"]
    texture = {row["file"]: row
               for row in json.loads(
                   (AUDIT / "audit_record.json").read_text(encoding="utf-8"))["photos"]}
    tolerance = labels.get("tolerance_height_fraction", 0.06)

    names = sorted(labelled)
    if limit:
        names = names[:limit]
    rows = []
    for name in names:
        path = AUDIT / "photos" / name
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.float64) / 255.0
        started = time.time()
        depth = depth_rows(model, transform,
                           (rgb * 255.0).astype(np.uint8), size, torch)
        seconds = time.time() - started
        split = derive_region_masks(np.repeat(depth[None, ...], 3, axis=0))
        expected = labelled[name].get("expected")
        applicable = labelled[name].get("applicable", bool(expected))
        comparison = (_verdict(split["boundaries"], expected, tolerance)
                      if applicable and expected
                      else {"verdict": "undefined"})
        profile = depth.mean(axis=1)
        texture_profile = row_texture_energy(np.asarray(rgb).transpose(2, 0, 1))
        rows.append({
            "file": name,
            "scene_regimes": labelled[name].get("regimes"),
            "applicable": applicable,
            "depth_boundaries": split["boundaries"],
            "depth_evidence": split["evidence"],
            "texture_boundaries": texture[name]["boundaries"],
            "texture_verdict": texture[name]["comparison"]["verdict"],
            "texture_max_abs_error": texture[name]["comparison"].get(
                "max_abs_error"),
            "depth_comparison": comparison,
            "depth_verdict": comparison["verdict"],
            "depth_max_abs_error": comparison.get("max_abs_error"),
            "depth_range": [round(float(depth.min()), 4),
                            round(float(depth.max()), 4)],
            "row_profile_every_16": [round(float(v), 3)
                                     for v in profile[::16]],
            "top_vs_bottom_mean": [round(float(profile[:16].mean()), 3),
                                   round(float(profile[-16:].mean()), 3)],
            "depth_ramp": ramp_fit(profile),
            "texture_ramp": ramp_fit(texture_profile),
            "seconds": round(seconds, 2),
        })

    def _scored(prefix: str) -> dict:
        key = f"{prefix}_verdict"
        scored = [r for r in rows
                  if r[key] in ("close", "partial", "wrong")]
        errors = [r[f"{prefix}_max_abs_error"] for r in scored
                  if r[f"{prefix}_max_abs_error"] is not None]
        return {
            "scored": len(scored),
            "close": sum(r[key] == "close" for r in scored),
            "partial": sum(r[key] == "partial" for r in scored),
            "wrong": sum(r[key] == "wrong" for r in scored),
            "median_abs_error": round(float(np.median(errors)), 4) if errors else None,
            "max_abs_error": round(float(max(errors)), 4) if errors else None,
        }

    return {
        "schema": "region_mask_model_comparison/v1",
        "generated_by": "scripts/compare_region_split_to_depth.py",
        "model": provenance,
        "method": (
            "the same derive_region_masks splitter run on the MiDaS relative "
            "inverse-depth map instead of RGB, at the audit's analysis size; "
            "hand labels and tolerance reused from ground_truth.json"),
        "analysis_size": size,
        "tolerance_height_fraction": tolerance,
        "photos": rows,
        "summary": {
            "photos": len(rows),
            "ramp_dominance": {
                "depth_median_ramp_r2": round(float(np.median(
                    [r["depth_ramp"]["ramp_r2"] for r in rows])), 4),
                "texture_median_ramp_r2": round(float(np.median(
                    [r["texture_ramp"]["ramp_r2"] for r in rows])), 4),
                "depth_ramp_r2_above_0.9": [
                    r["file"] for r in rows
                    if r["depth_ramp"]["ramp_r2"] > 0.9],
                "note": (
                    "A profile that is a linear ramp has no step for the splitter "
                    "to cut; the depth signal being ramp-dominated is why its cuts "
                    "are arbitrary and its evidence gate refuses them."),
            },
            "labelled": sum(1 for r in rows if r["applicable"]),
            "texture": _scored("texture"),
            "depth": _scored("depth"),
            "depth_reliable": sum(1 for r in rows
                                  if r["depth_evidence"]["reliable"]),
            "verdict_changed": [
                {"file": r["file"], "texture": r["texture_verdict"],
                 "depth": r["depth_verdict"]}
                for r in rows
                if r["texture_verdict"] != r["depth_verdict"]],
        },
        "limitations": (
            "Ten photographs, one resize, one hand labeller. MiDaS_small is the "
            "smallest variant, run on CPU at the audit's 256 px analysis size; a "
            "larger depth model would change the numbers. The splitter, the "
            "minimum band height and the tolerance are inherited unchanged, so "
            "this compares signals, not algorithms."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out",
                        default=str(AUDIT / "model_comparison.json"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    record = run(args.size, args.limit)
    for row in record["photos"]:
        print(f"  {row['file'][:26]:28s} texture={row['texture_verdict']:10s}"
              f" depth={row['depth_verdict']:10s}"
              f" depth_b={row['depth_boundaries']}"
              f" reliable={row['depth_evidence']['reliable']}")
    summary = record["summary"]
    print(f"\ntexture: {summary['texture']}")
    print(f"depth  : {summary['depth']}")
    print(f"depth splits judged reliable: {summary['depth_reliable']}/{summary['photos']}")
    if not args.no_write:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
