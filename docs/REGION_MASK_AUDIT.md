# Region-mask audit against photographs

`apedp_dyn.video.region_masks` replaced three hard-coded bands with a
measurement: per-row texture energy, cut at the two sharpest steps. That is an
improvement over a constant, and the module is careful to say it names nothing —
"texture rank only; the caller names the bands".

Careful wording is not evidence. This record is the evidence: the split run on
ten photographs, compared against band boundaries labelled by eye from the
pictures.

## What was run

- **Script:** `scripts/audit_region_masks_on_photos.py`
- **Photographs:** ten, fetched by `scripts/fetch_region_mask_audit_photos.py`
  from the Wikimedia Commons API, each with its Commons page, licence, artist
  and a SHA-256 of the bytes written (`data/region_mask_audit/PROVENANCE.json`).
  Files are under a public-domain, CC0, CC BY or CC BY-SA licence.
- **Analysis:** each image resized to 256×256 RGB and passed to
  `derive_region_masks` with the default `smoothing=5` and
  `min_band_fraction=0.15`, which is what the animator uses.
- **Ground truth:** `data/region_mask_audit/ground_truth.json`, labelled by hand
  from a contact sheet with a ruler at every 10 % of height. Labelling
  uncertainty is about 0.03 of the frame height. A boundary within 0.06 (15 rows
  at 256 px) counts as *close*.
- **Record:** `data/region_mask_audit/audit_record.json`.

## Result

| Photograph | Verdict | Boundary error (b1, b2) | What happened |
|---|---|---|---|
| `beach_palm_trees` | **close** | +0.010, +0.054 | Sky/fronds → water → sand; both cuts land on the real transitions. |
| `mountain_lake_reflection` | **close** | +0.013, −0.005 | Cloud, ridge, lagoon; the sharpest steps in the frame are the right ones. |
| `city_skyline_night` | partial | +0.182, +0.022 | The street-level cut is right; the night-sky band is 8 % of the height and the 15 % floor makes it unreachable. |
| `breaking_waves` | **wrong** | +0.161, −0.066 | Cloud texture above out-energies the horizon, so b1 lands at 0.37 inside the sea. |
| `sand_dunes` | **wrong** | +0.280, +0.188 | Sand is texture-free, so the smoothest rows are at the bottom and the lower cut is dragged into the dune. |
| `snow_field` | **wrong** | +0.197, +0.192 | Same failure as the dune: snow is the smoothest content and attracts a cut. |
| `portrait_outdoors` | **wrong** | +0.270, +0.251 | Boardwalk planks and water highlights in the bottom third pull both cuts down. |
| `forest_path` | *undefined* | — | A corridor: texture changes left-to-right around the path, not top-to-bottom. Two boundaries returned anyway. |
| `sea_sunset_horizon` | *undefined* | — | Sky and sea, one transition at 0.45. Two boundaries returned: 0.27 and 0.72. |
| `aerial_coastline` | *undefined* | — | Sky, sea and a diagonal land spit; the strongest texture is the window frame. Two boundaries returned. |

**2 close, 1 partial, 4 wrong, 3 with no structure to be right about.** Median
boundary error over the seven scored photographs: **0.18** of the frame height;
worst 0.28.

## The three findings

1. **The three-band model mis-specifies real scenes.** Three of ten photographs
   do not have three stacked horizontal regions, and the module returns two
   interior boundaries for every one of them. There is no output that says *this
   frame has no bands to split*, so a caller cannot tell a confident split from
   an invented one.
2. **Smoothness is not emptiness.** Both photographs scored worst — a dune, a
   snowfield — fail in the same direction: the largest smooth region is at the
   bottom of the frame, so it attracts the lower cut regardless of what it is.
   Texture rank measures contrast, not content.
3. **Diagonal structure cannot be followed.** Running the same cut search on the
   left and right halves separately gives boundaries that disagree by more than
   the tolerance on five of the ten photographs (worst: 0.45 on
   `portrait_outdoors`). Where the halves disagree, a single row boundary is
   describing something that is not there. That diagnostic is in the record as
   `halves.diagonal_shift` and costs nothing to check at runtime.

## The gate, and how well it predicts

The audit's findings 1 and 3 are cheap to check at run time, so
`derive_region_masks` now measures them and reports them in an `evidence` block:
`cut_decisiveness` (the two chosen steps over the median row step),
`half_disagreement` (the same cut search on the left and right halves), and
`texture_order`. `reliable` is true only when all three pass, and `reasons`
names the ones that failed. With `require_reliable=True` it raises
`UnusableRegionSplit` instead.

Run against the ten hand-labelled photographs, the gate accepts **one**:

| | |
|---|---|
| accepted | `beach_palm_trees` — verdict **close** |
| accepted but wrong | **none** |
| rejected but close | `mountain_lake_reflection` |
| rejected | the other 8 |

That is the direction a gate should err in: it never certifies a split the hand
labels call wrong, at the cost of being timid. The one false alarm is
informative rather than embarrassing — at Jökulsárlón the left half of the frame
is icebergs and the right half is open water, so the two halves genuinely locate
different cuts, and a single horizontal boundary really does not describe the
frame everywhere. Rejecting it is the gate being right about the frame and the
score being right about the numbers.

Note also what never fired: `cut_decisiveness` is 10x or more on every
photograph, because row-texture profiles are smooth and any cut through them
stands out. On this sample the three-band model, not the cut detection, is what
fails.

## What this does not measure

Naming. The audit scores only whether the boundaries sit on a genuine horizontal
transition; it says nothing about whether the upper band is a sky, because the
module does not claim that. Ten photographs from one source, one resize and one
smoothing window is a first sample, not a survey.

## Replacing texture with depth: measured, not assumed

`scripts/compare_region_split_to_depth.py` runs the **same** splitter on a
pretrained monocular depth model — MiDaS_small (Ranftl et al.), with the
checkpoint pinned by SHA-256 in the record — instead of on RGB texture. Same
256 px analysis size, same `smoothing`, same `min_band_fraction`, same hand
labels, same 0.06 tolerance. Only the signal changes.

| | texture | depth |
|---|---|---|
| scored (of 7 applicable) | 7 | 7 |
| close | **2** | 0 |
| partial | 1 | 4 |
| wrong | 4 | **3** |
| median boundary error | **0.182** | 0.228 |
| worst | 0.280 | 0.325 |
| splits the evidence gate endorsed | 1 of 10 | **0 of 10** |

Depth did not win, and the losing margin is not the interesting part. The
measured reason is:

> **MiDaS's row-mean depth is close to a straight line down the frame.**
> R² of the row profile on row index is **0.827** median for depth against
> **0.240** for texture, and the rank correlation is positive in all ten
> photographs (+0.83 to +0.96): the prediction says "nearer as you go down"
> almost regardless of what the scene contains, which is the image vantage
> prior rather than scene structure.

A splitter that cuts at the two largest **steps** has nothing to cut in a smooth
monotone field. Its boundaries land wherever the ramp's curvature happens to be
largest, and the evidence gate refused **all ten** depth splits — which is
exactly the behaviour finding 1 asked for, arriving unasked. The gate is
therefore not a formality: it is what stops this pipeline from presenting
gradient artefacts as bands.

The reading is narrow and worth stating narrowly: swapping the signal while
keeping the algorithm does not help. The algorithm assumes piecewise-constant
bands, and neither signal reliably has them. Getting the bands right needs
per-pixel **semantics** (a class map in which "sky" is a class) or a splitter
that models a gradient instead of seeking steps. The semantic half of that is
tested in the next section.

Reproducing the comparison needs `torch`, `torchvision` and `timm`, which the
base interpreter does not have:

```bash
python3 -m venv --system-site-packages .venv-region-models
.venv-region-models/bin/pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cpu "torchvision==0.18.1+cpu"
.venv-region-models/bin/pip install --no-deps "timm==0.9.16"
.venv-region-models/bin/python scripts/compare_region_split_to_depth.py
```

## Replacing texture with semantics: measured

`scripts/compare_region_split_to_semantics.py` is the third arm. SegFormer-B0
fine-tuned on ADE20K (Xie et al., 2021), weights pinned by SHA-256, predicts one
class per pixel; the bands are cut at the two rows where the **class
composition** changes most — the total variation between the composition in an
8-row window above and below each candidate row. Same 256 px analysis size,
same `min_band_fraction=0.15`, same two-cut selection rule, same evidence
thresholds (`MIN_CUT_DECISIVENESS` and `MAX_HALF_DISAGREEMENT` imported from
`region_masks`, not re-declared), same hand labels, same 0.06 tolerance.

| | texture | depth | semantic |
|---|---|---|---|
| scored (of 7 applicable) | 7 | 7 | 7 |
| close | 2 | 0 | 2 |
| partial | 1 | 4 | 1 |
| wrong | 4 | 3 | 4 |
| median boundary error | 0.182 | 0.228 | **0.115** |
| worst | 0.280 | 0.325 | 0.268 |
| splits the evidence gate endorsed | 1 of 10 | 0 of 10 | **4 of 10** (1 also close) |
| verdicts changed against texture | — | 3 | **0** |

Semantics is the best signal on this set and it **fixes nothing**. Median error
falls 37 % and the worst case improves slightly, but the photographs that were
close are still close, the four that were wrong are still wrong, and not one
verdict flipped. The gain is in where inside a band a cut lands, not in whether
the split describes the frame.

The failures are the same four, for the two causes the first audit named.
`city_skyline_night` puts its true upper boundary at 0.08 — under the 15 %
minimum band height, unreachable by construction under every signal.
`breaking_waves`, `sand_dunes`, `portrait_outdoors` and `snow_field` each have
three regimes and still get exactly two cuts, so a scene whose real structure is
sky | sea | beach has its second cut placed wherever the sand begins to differ
from the wet rock rather than where the labels place it. Choosing a better
signal cannot fix a mis-specified band model.

### A class map cannot be fed to the texture splitter

Recorded because it explains why this arm changes the splitter and the depth arm
did not. `row_texture_energy` differences neighbouring pixels; ADE20K class ids
are **nominal**, so differencing them measures the numbering of the vocabulary
(a sky pixel beside a water pixel reads as 19, whatever the picture shows).
Handed the class map, the splitter does not fail — it returns boundaries at
0.18/0.75 — and only its own evidence gate refuses them, with a cut
decisiveness of 0.0. That measurement is in the record
(`class_arm_structural_note.invalid_input_measurement`), because a reader shown
only the boundaries would have no way to know the input was meaningless.

Reproducing needs the pinned environment from above plus three pins that a
torch 2.3 CPU build forces (`transformers` 5.x requires a newer torch, 4.44
requires `tokenizers<0.20`, and 4.44 requires `huggingface-hub<1.0`):

```bash
.venv-region-models/bin/pip install --no-deps "transformers==4.44.2"
.venv-region-models/bin/pip install --no-deps "tokenizers==0.19.1"
.venv-region-models/bin/pip install --no-deps "huggingface-hub==0.25.2"
.venv-region-models/bin/python scripts/compare_region_split_to_semantics.py
```

## What would raise the ceiling

Two of the three suggestions are now tested and neither rescued the split: depth
did not help (0.228 median, ramp-dominated) and semantics helped the error
without changing a verdict (0.115 median, the same 2/1/4). What is left is not
another signal:

- **A band count chosen from the evidence.** Every frame still gets exactly
  three bands, so sky-over-sea with one horizon is split twice and a
  four-regime scene is under-split. This is the largest remaining gap, and it is
  a change to the return type rather than a measurement. The composition
  profile the semantic arm now records is the natural place to read a band
  count off: it has one clear peak on `sea_sunset_horizon` and three on
  `mountain_lake_reflection`.
- **A layered representation**, which optimises the split against the whole
  video instead of against one static frame.

## Reproducing

```bash
python3 scripts/fetch_region_mask_audit_photos.py --out data/region_mask_audit
python3 scripts/audit_region_masks_on_photos.py \
    --photos data/region_mask_audit/photos \
    --ground-truth data/region_mask_audit/ground_truth.json \
    --out data/region_mask_audit/audit_record.json
pytest tests/test_region_mask_photo_audit.py -v
```

To audit your own photographs, point `--photos` at a directory of images; every
verdict not covered by `ground_truth.json` is recorded as `unlabelled` rather
than guessed.
