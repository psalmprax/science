# JetFlow-Video v0.3 Confirmation Report — Collision/Occlusion World

**Prereg:** `docs/PREREGISTRATION_JETFLOW_VIDEO_V03_COLLISION.md` (PREREG-JETFLOW-VIDEO-V03-COLLISION-2026-10-19)
**Harness:** `benchmarks/run_jetflow_video_v03_confirmation.py`; per-seed artifacts:
`benchmarks/jetflow_video_v03_confirmation/seed_{407,508,509}_results.json`;
machine-readable verdicts: `benchmarks/jetflow_video_v03_confirmation_results.json`.
**Runs committed BEFORE this report** (no post-hoc world/threshold edits).

## Headline

**The fluid-domain JetFlow advantage does NOT transfer to the collision world.**
H1-collision CONFIRMED on 3/3 seeds — the matched diffusion baseline wins SSIM
and temporal consistency on every confirmation seed. Recorded per the prereg as a
BOUNDARY claim: the published fluid-domain advantage is **domain-scoped to
continuum transport**; no unscoped generalization claim is made on either side.

## Per-hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1-collision (domain-scoping) | **CONFIRMED** (3/3 seeds, required ≥ 2) | diffusion SSIM 0.064/0.048/0.053 > jet 0.010/0.007/0.018; diffusion temporal 0.082/0.092/0.084 < jet 0.244/0.220/0.220 |
| H2 (jet PSNR floor ≥ 10.0 dB) | **CONFIRMED** (3/3) | jet2 PSNR 13.62 / 13.26 / 13.86 dB |
| H3 (train ratio ≤ 2.0) | **CONFIRMED** (3/3) | 1.23 / 1.30 / 1.54 |
| H4 (structural passes: jet2 == 8, jet4 == 16) | **CONFIRMED** (3/3) after **ERR-H4-UNITS** | recorded per-clip passes 9.0 / 16.0 on every seed |

## Erratum disclosure (ERR-H4-UNITS)

The prereg text derived the structural counts as `round(passes * n_gen) - 1`,
mis-stating the metric's units: `model_passes_per_clip` is **already** per-clip
(the jet2 recording of 9.0 carries the one warmup GENERATION amortized inside the
count; jet4's instrumentation window has no warmup). The corrected derivation is
`round(passes) - 1` (jet2) and `round(passes)` (jet4). The registered NUMBERS
(8 and 16, from the v0.2 registration) are unchanged; verdicts were recomputed
from the SAME committed per-seed artifacts — **no benchmark re-run, no threshold
moved, no seed changed** (`evaluated_from_committed_artifacts: true` in the JSON).

## What was NOT done

- No re-tuning toward the verdicts; the exploratory seed 404 stayed excluded.
- The instruments are the inherited honest labels: FVD-proxy is a
  ranked-separation instrument, NOT I3D-FVD; PSNR/SSIM are nearest-holdout
  similarity, not identity.
- Latency is reported (machine-dependent), never gated.

## Boundary summary for the knowledge record

JetFlow's measured advantages on the fluid world (temporal smoothness 0.275 vs
0.365+; SSIM 0.514 vs 0.477; 5.6× faster training) now stand WITH an explicit
scope: **continuum transport**. On the rigid-body/occlusion world: diffusion
dominates on every pixel/consistency instrument; JetFlow keeps its structural
pass-count economy (8-9 vs 8 matched passes — H3 ceiling respected). This is the
shape of an honest, bounded claim rather than a general one.
