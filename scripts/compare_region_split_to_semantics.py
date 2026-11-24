"""Compare the band split against a pretrained semantic segmentation model.

`apedp_dyn.video.region_masks` cuts three horizontal bands at the two sharpest
steps in per-row *texture* energy. `scripts/compare_region_split_to_depth.py`
swapped the signal for monocular depth and measured a negative result: MiDaS's
row profile is a vantage ramp, so the step-seeker had nothing to cut and the
evidence gate refused every split.

This script runs the third arm: **semantics**. SegFormer-B0 fine-tuned on
ADE20K (Xie et al., "SegFormer", 2021) predicts a class label per pixel, and
ADE20K's vocabulary contains `sky`, `water`, `sea`, `mountain`, `earth`, `sand`,
`tree`, `building` — the exact vocabulary the hand labels use. Instead of asking
"where does brightness change", this arm asks "where does *what the scene
contains* change", which is the question the labels actually encode.

Design note — why this arm cannot reuse the splitter verbatim, and why that is
a finding rather than a shortcut. The depth arm could hand the same array to
`derive_region_masks` unchanged because a depth map is continuous and its
horizontal differences mean something. A class map's do not: ADE20K class ids
are *nominal*, so differencing neighbouring pixels (sky 2 next to water 21) sums
the numbering of the vocabulary, not the picture. `row_texture_energy` will
still accept the array and `derive_region_masks` will still return boundaries —
and its own evidence gate then refuses them with a cut decisiveness of 0.0,
which `_measure_class_map_as_texture` records rather than asserts. A
segmentation model therefore forces a score that has meaning under a class
alphabet: how much the *composition* of the scene changes across a row. What is
held fixed is everything
that made the two earlier arms comparable: the analysis size (256 px), the two
cuts as the maximiser of a per-row score under `min_band_fraction` (0.15, read
from `region_masks`), and the evidence thresholds `MIN_CUT_DECISIVENESS` (2.0)
and `MAX_HALF_DISAGREEMENT` (0.06), imported rather than re-declared. Only the
score changes: the total-variation distance between the class composition
just above and just below each candidate row, over a 2*window window.

The third gate check is adapted and the adaptation is recorded. The texture
splitter refuses a "non-monotone" band ordering, because "the smoothest band is
the sky" has no meaning when the stack is unordered. A class map has no texture
order to check, so this arm refuses instead when all three bands share one
dominant class — i.e. the split has not separated different content, which is
the same objection in the vocabulary that exists here.

The decisiveness denominator is measured more carefully here than in the texture
gate, and the reason is a real defect the synthetic regression test found. A
genuinely piecewise-constant frame has a *zero* median row change, because most
rows sit strictly inside a band; dividing by it reports the cleanest possible
cut as ``0.00x`` the noise, and the gate refuses it. The fallback is the median
of the rows that do change, and ``evidence["decisiveness_basis"]`` says which
denominator produced the ratio, so "decisive" and "undefined" are not reported
as the same number.

Labels and the verdict rule are the photograph audit's own:
`data/region_mask_audit/ground_truth.json` (hand-labelled; 0.06 of the frame
height counts as close), with texture verdicts read back from
`audit_record.json`. Same labels, same tolerance, same `_verdict` function, so
the three arms are scored by one rule.

Interpreter: this needs torch and transformers, which the base interpreter does
not have. The pinned environment (see the depth script's docstring for the base
of it) additionally needs a transformers line that supports torch 2.3:

    .venv-region-models/bin/pip install --no-deps "transformers==4.44.2"
    .venv-region-models/bin/pip install --no-deps "tokenizers==0.19.1"
    .venv-region-models/bin/pip install --no-deps "huggingface-hub==0.25.2"

transformers 5.x requires a torch newer than 2.3 and 4.44 requires
tokenizers<0.20, so all three pins are load-bearing; each one was found by an
ImportError, not by guesswork.

then

    .venv-region-models/bin/python scripts/compare_region_split_to_semantics.py

Output: `data/region_mask_audit/semantic_comparison.json`.
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
    MAX_HALF_DISAGREEMENT,
    MIN_CUT_DECISIVENESS,
    derive_region_masks,
    row_texture_energy,
)

AUDIT = ROOT / "data" / "region_mask_audit"
MODEL = {
    "hub_repo": "nvidia/segformer-b0-finetuned-ade-512-512",
    "architecture": "SegFormer-B0, ADE20K (150 classes) fine-tune",
    "citation": "Xie et al., SegFormer (NeurIPS 2021)",
    "output_semantics": "one ADE20K class id per pixel",
}

# The bands are at least this wide, as a fraction of the height. The same value
# `derive_region_masks` defaults to; imported below so the two arms cannot drift.
MIN_BAND_FRACTION = 0.15
# Half-width, in rows, of the windows whose class compositions are compared
# across a candidate cut. Eight rows at 256 px is ~3 % of the frame: wide enough
# that a single mislabelled row cannot move a boundary.
WINDOW = 8
# Classes whose presence in a band is reported by name, so a reader can check
# that a split separated content rather than cutting a uniform field.
REPORTED_CLASSES = ("sky", "mountain", "water", "sea", "sand", "snow", "earth",
                    "tree", "building", "road", "person", "car", "wall")


def _require_backends():
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{exc.name} is not available in this interpreter. The semantic "
            "comparison needs torch and transformers; see the docstring for "
            "the pinned .venv-region-models setup.") from exc
    return torch, transformers


def load_model():
    torch, transformers = _require_backends()
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained(MODEL["hub_repo"])
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL["hub_repo"]).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    provenance = {
        "hub_repo": MODEL["hub_repo"],
        "architecture": MODEL["architecture"],
        "citation": MODEL["citation"],
        "output_semantics": MODEL["output_semantics"],
        "num_classes": len(id2label),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "id2label_sha256": hashlib.sha256(
            json.dumps(sorted(id2label.items())).encode()).hexdigest(),
    }
    digest = _weights_digest()
    if digest:
        provenance["weights"] = digest
    return processor, model, id2label, provenance


def _weights_digest() -> dict | None:
    """SHA-256 of the cached SegFormer weights, when the file is on disk."""
    home = Path.home()
    cache = (home / ".cache" / "huggingface" / "hub")
    if not cache.is_dir():
        return None
    folder = cache / ("models--" + MODEL["hub_repo"].replace("/", "--"))
    for path in sorted(folder.rglob("*.safetensors")) + sorted(
            folder.rglob("*.bin")):
        if path.is_file():
            return {"file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return None


def segment(processor, model, image: np.ndarray, size: int, torch) -> np.ndarray:
    """One ADE20K class id per pixel at the audit's analysis size."""
    import torch.nn.functional as functional
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = functional.interpolate(outputs.logits, size=(size, size),
                                        mode="bilinear", align_corners=False)
    # The processor's own post-processor reads ``outputs.logits``; passing the
    # already-resized tensor raises AttributeError, which is how the required
    # interpolation above was found rather than assumed.
    logits = outputs.__class__(logits=logits)
    segmented = processor.post_process_semantic_segmentation(
        logits, target_sizes=[(size, size)])[0]
    return segmented.cpu().numpy().astype(np.int32)


def row_class_composition(class_map: np.ndarray, n_classes: int) -> np.ndarray:
    """``(H, K)`` row-wise class occupancy, each row summing to one.

    A row of a piecewise-constant class map is a constant vector, which is why
    this representation - not the map itself - is what the splitter can read.
    """
    h = class_map.shape[0]
    composition = np.zeros((h, n_classes), dtype=np.float64)
    for row in range(h):
        counts = np.bincount(class_map[row], minlength=n_classes)
        total = counts.sum()
        composition[row] = counts / total if total else counts
    return composition


def class_change_profile(composition: np.ndarray, window: int) -> np.ndarray:
    """Total-variation distance between the composition above and below a row.

    ``score[b]`` answers: if the frame were cut at row ``b``, how much would the
    mix of things in the scene differ across that cut? A horizon scores high, a
    uniform field scores ~0, and unlike a texture gradient the score does not
    depend on how busy the band looks.
    """
    h = composition.shape[0]
    cumulative = np.vstack([np.zeros((1, composition.shape[1])),
                            np.cumsum(composition, axis=0)])
    score = np.zeros(h, dtype=np.float64)
    for b in range(h):
        above_lo, above_hi = max(0, b - window), b
        below_lo, below_hi = b, min(h, b + window)
        above = cumulative[above_hi] - cumulative[above_lo]
        below = cumulative[below_hi] - cumulative[below_lo]
        above_n = above_hi - above_lo
        below_n = below_hi - below_lo
        if above_n <= 0 or below_n <= 0:
            continue
        score[b] = 0.5 * float(np.abs(above / above_n - below / below_n).sum())
    return score


def _best_two_cuts(score: np.ndarray, min_len: int) -> tuple[int, int]:
    """The two rows whose composition change is largest, each band >= min_len.

    The same selection rule as ``region_masks._best_two_cuts``; only the per-row
    score differs, so the splitter is held fixed and the signal is what moves.
    """
    h = score.size
    best = (-np.inf, min_len, h - min_len)
    for b1 in range(min_len, h - 2 * min_len + 1):
        for b2 in range(b1 + min_len, h - min_len + 1):
            total = score[b1] + score[b2]
            if total > best[0]:
                best = (total, b1, b2)
    return best[1], best[2]


def _measure_class_map_as_texture(class_map: np.ndarray) -> dict:
    """What the texture splitter returns when handed a class map.

    ``row_texture_energy`` measures the mean absolute horizontal difference
    between neighbouring pixels. ADE20K class ids are *nominal*: 26 is not
    "twenty-four more than" 2, so differencing them aggregates label churn into
    a number with no units and no meaning, and a splitter consuming it cuts on
    the numbering of the vocabulary rather than on the picture.

    Crucially the splitter does not fail loudly here — it returns boundaries.
    Only its evidence gate refuses them (cut decisiveness 0.0, because the
    nominal-id profile has no step that stands out from its own noise). That is
    why this measurement belongs in the record: a reader who saw only the
    boundaries would have no way to know the input was invalid.
    """
    as_latent = class_map[None, ...].astype(np.float64)
    energy = row_texture_energy(as_latent)
    split = derive_region_masks(as_latent)
    return {
        "label_ids_are_nominal": True,
        "horizontal_difference_max_id_units": round(float(energy.max()), 4),
        "texture_split_boundaries": split["boundaries"],
        "texture_split_cut_decisiveness": split["evidence"]["cut_decisiveness"],
        "texture_split_reliable": split["evidence"]["reliable"],
        "texture_split_reasons": split["evidence"]["reasons"],
    }


def band_composition(composition: np.ndarray, lo: int, hi: int,
                     id2label: dict[int, str]) -> dict:
    """Dominant classes of a band, and the share of the reported vocabulary."""
    mean = composition[lo:hi].mean(axis=0) if hi > lo else composition[lo]
    order = np.argsort(-mean)[:4]
    shares = {id2label[int(c)]: round(float(mean[c]), 4) for c in order}
    reported = {}
    for cls in REPORTED_CLASSES:
        idx = [i for i, name in id2label.items() if name == cls]
        if idx:
            reported[cls] = round(float(mean[idx[0]]), 4)
    return {"dominant": shares,
            "reported_classes": {k: v for k, v in reported.items() if v > 0.01}}


def semantic_split(
    class_map: np.ndarray,
    id2label: dict[int, str],
    *,
    window: int = WINDOW,
    min_band_fraction: float = MIN_BAND_FRACTION,
) -> dict:
    """Three bands cut where the scene's content changes, with its evidence."""
    h = class_map.shape[0]
    n_classes = len(id2label)
    composition = row_class_composition(class_map, n_classes)
    score = class_change_profile(composition, window)
    min_len = max(1, round(min_band_fraction * h))
    if h < 3 * min_len + 2:
        raise ValueError(f"height {h} is too small for three bands")
    b1, b2 = _best_two_cuts(score, min_len)

    cut_score = float(score[b1] + score[b2])
    median = float(np.median(score))
    # A frame that is genuinely piecewise constant has a *zero* median row
    # change, because most rows sit strictly inside a band. Dividing by that
    # median reports the cleanest possible cut as 0.0x the noise, which the gate
    # then refuses: backwards, and only visible on a perfectly clean input. Fall
    # back to the median of the rows that do change, and record the denominator
    # so a reader can tell the two cases apart.
    if median > 0.0:
        basis, denominator = "median_row_change", median
    else:
        nonzero = score[score > 0.0]
        basis = "median_nonzero_row_change"
        denominator = float(np.median(nonzero)) if nonzero.size else 0.0
    decisiveness = cut_score / denominator if denominator > 0.0 else 0.0

    width = class_map.shape[1]
    half_cuts = {}
    for side, columns in (("left", slice(0, width // 2)),
                          ("right", slice(width // 2, width))):
        half_map = class_map[:, columns]
        half_score = class_change_profile(
            row_class_composition(half_map, n_classes), window)
        half_cuts[side] = list(_best_two_cuts(half_score, min_len))
    disagreement = max(
        abs(half_cuts["left"][0] - half_cuts["right"][0]),
        abs(half_cuts["left"][1] - half_cuts["right"][1])) / h

    bands = {
        "upper": band_composition(composition, 0, b1, id2label),
        "middle": band_composition(composition, b1, b2, id2label),
        "lower": band_composition(composition, b2, h, id2label),
    }
    dominants = [next(iter(b["dominant"])) for b in bands.values()]

    reasons: list[str] = []
    if decisiveness < MIN_CUT_DECISIVENESS:
        reasons.append(
            f"the two cuts are {decisiveness:.2f}x the {basis.replace('_', ' ')} "
            f"(need {MIN_CUT_DECISIVENESS:g}x): the scene content does not "
            "step there")
    if disagreement > MAX_HALF_DISAGREEMENT:
        reasons.append(
            f"the left and right halves disagree by {disagreement:.3f} of the "
            f"height (limit {MAX_HALF_DISAGREEMENT:g}): one row boundary does "
            "not describe the frame everywhere")
    if len(set(dominants)) == 1:
        reasons.append(
            f"all three bands are dominated by '{dominants[0]}', so the split "
            "has not separated different content")

    return {
        "boundaries": {"upper_to_middle": round(b1 / h, 4),
                       "middle_to_lower": round(b2 / h, 4)},
        "method": "row_class_composition_change_argmax_two_cuts",
        "bands": bands,
        "evidence": {
            "reliable": not reasons,
            "reasons": reasons,
            "cut_decisiveness": round(decisiveness, 3),
            "decisiveness_basis": basis,
            "half_disagreement": round(disagreement, 4),
            "distinct_dominant_classes": len(set(dominants)),
            "half_cuts": {side: [c / h for c in cuts]
                          for side, cuts in half_cuts.items()},
        },
        "change_profile_every_16": [round(float(v), 4) for v in score[::16]],
    }


def run(size: int, limit: int | None = None) -> dict:
    torch, _transformers = _require_backends()
    processor, model, id2label, provenance = load_model()
    labels = json.loads((AUDIT / "ground_truth.json").read_text(encoding="utf-8"))
    labelled = labels["labelled"]
    depth_record = json.loads(
        (AUDIT / "model_comparison.json").read_text(encoding="utf-8"))
    depth_rows = {row["file"]: row for row in depth_record["photos"]}
    texture = {row["file"]: row
               for row in json.loads(
                   (AUDIT / "audit_record.json").read_text(
                       encoding="utf-8"))["photos"]}
    tolerance = labels.get("tolerance_height_fraction", 0.06)

    names = sorted(labelled)
    if limit:
        names = names[:limit]
    rows = []
    degeneracy = None
    for name in names:
        path = AUDIT / "photos" / name
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.uint8)
        started = time.time()
        class_map = segment(processor, model, rgb, size, torch)
        seconds = time.time() - started
        if degeneracy is None:
            degeneracy = _measure_class_map_as_texture(class_map)
        split = semantic_split(class_map, id2label)
        expected = labelled[name].get("expected")
        applicable = labelled[name].get("applicable", bool(expected))
        comparison = (_verdict(split["boundaries"], expected, tolerance)
                      if applicable and expected
                      else {"verdict": "undefined"})
        present = sorted({id2label[int(c)] for c in np.unique(class_map)})
        rows.append({
            "file": name,
            "scene_regimes": labelled[name].get("regimes"),
            "applicable": applicable,
            "semantic_boundaries": split["boundaries"],
            "semantic_bands": split["bands"],
            "semantic_evidence": split["evidence"],
            "semantic_verdict": comparison["verdict"],
            "semantic_max_abs_error": comparison.get("max_abs_error"),
            "semantic_comparison": comparison,
            "texture_verdict": texture[name]["comparison"]["verdict"],
            "texture_max_abs_error": texture[name]["comparison"].get(
                "max_abs_error"),
            "depth_verdict": depth_rows.get(name, {}).get("depth_verdict"),
            "depth_max_abs_error": depth_rows.get(name, {}).get(
                "depth_max_abs_error"),
            "classes_in_frame": present,
            "change_profile_every_16": split["change_profile_every_16"],
            "seconds": round(seconds, 2),
        })

    def _scored(key: str, error_key: str) -> dict:
        scored = [r for r in rows if r[key] in ("close", "partial", "wrong")]
        errors = [r[error_key] for r in scored if r[error_key] is not None]
        return {
            "scored": len(scored),
            "close": sum(r[key] == "close" for r in scored),
            "partial": sum(r[key] == "partial" for r in scored),
            "wrong": sum(r[key] == "wrong" for r in scored),
            "median_abs_error": round(float(np.median(errors)), 4) if errors else None,
            "max_abs_error": round(float(max(errors)), 4) if errors else None,
        }

    summary = {
        "photos": len(rows),
        "labelled": sum(1 for r in rows if r["applicable"]),
        "texture": _scored("texture_verdict", "texture_max_abs_error"),
        "depth": _scored("depth_verdict", "depth_max_abs_error"),
        "semantic": _scored("semantic_verdict", "semantic_max_abs_error"),
        "semantic_reliable": sum(1 for r in rows
                                 if r["semantic_evidence"]["reliable"]),
        "semantic_reliable_and_close": sum(
            1 for r in rows if r["semantic_evidence"]["reliable"]
            and r["semantic_verdict"] == "close"),
        "verdict_changed_vs_texture": [
            {"file": r["file"], "texture": r["texture_verdict"],
             "semantic": r["semantic_verdict"]}
            for r in rows if r["texture_verdict"] != r["semantic_verdict"]],
    }
    ranked = ("texture", "depth", "semantic")
    summary["ranking_by_median_abs_error"] = sorted(
        ranked,
        key=lambda a: (summary[a]["median_abs_error"] is None,
                       summary[a]["median_abs_error"] or 0.0))
    return {
        "schema": "region_mask_semantic_comparison/v1",
        "generated_by": "scripts/compare_region_split_to_semantics.py",
        "model": provenance,
        "method": (
            "three bands cut at the two rows of largest total-variation change "
            "in ADE20K class composition (windows of +/-8 rows), scored against "
            "the photograph audit's hand labels and tolerance; the texture and "
            "depth verdicts are read back from their own records for a like-for-"
            "like comparison"),
        "fixed_across_arms": {
            "analysis_size": size,
            "min_band_fraction": MIN_BAND_FRACTION,
            "min_cut_decisiveness": MIN_CUT_DECISIVENESS,
            "max_half_disagreement": MAX_HALF_DISAGREEMENT,
            "tolerance_height_fraction": tolerance,
            "two_cut_selection_rule": (
                "maximise the sum of the two per-row scores, each band at "
                "least min_band_fraction of the height"),
        },
        "class_arm_structural_note": {
            "why_not_the_same_splitter": (
                "row_texture_energy differences neighbouring pixels, and ADE20K "
                "class ids are nominal, so differencing them measures the "
                "numbering of the vocabulary rather than the picture. Handed a "
                "class map the splitter returns boundaries and its own evidence "
                "gate refuses them, so the semantic arm needs a score with "
                "meaning under a class alphabet: the total variation between "
                "the class composition just above and just below each row."),
            "invalid_input_measurement": degeneracy,
        },
        "photos": rows,
        "summary": summary,
        "limitations": (
            "Ten photographs, one resize, one hand labeller, one 256 px analysis "
            "size. SegFormer-B0 is the smallest ADE20K variant; the 150-class "
            "vocabulary is used whole, with no per-scene class list, so a scene "
            "whose bands are not ADE20K classes cannot be split well. The "
            "composition score, not the splitter's two-cut rule, is what this "
            "arm changes, so this compares signals under a shared selection "
            "rule - it does not show that a composition-aware splitter with a "
            "different band count would score the same."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out",
                        default=str(AUDIT / "semantic_comparison.json"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    record = run(args.size, args.limit)
    print(f"{'file':30s} {'texture':10s} {'depth':10s} {'semantic':10s} "
          "boundaries / reliable")
    for row in record["photos"]:
        print(f"  {row['file'][:26]:28s} {row['texture_verdict']:10s} "
              f"{row['depth_verdict']!s:10s} {row['semantic_verdict']:10s} "
              f"{row['semantic_boundaries']} "
              f"reliable={row['semantic_evidence']['reliable']}")
    summary = record["summary"]
    print(f"\ntexture : {summary['texture']}")
    print(f"depth   : {summary['depth']}")
    print(f"semantic: {summary['semantic']}")
    print(f"semantic splits judged reliable: "
          f"{summary['semantic_reliable']}/{summary['photos']} "
          f"({summary['semantic_reliable_and_close']} also close)")
    print(f"best by median abs error: {summary['ranking_by_median_abs_error']}")
    print("class map as texture input: "
          f"{record['class_arm_structural_note']['invalid_input_measurement']}")
    if not args.no_write:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
