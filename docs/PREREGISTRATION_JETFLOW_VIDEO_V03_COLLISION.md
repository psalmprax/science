# Preregistration: JetFlow-Video v0.3 — Collision/Occlusion World (Second Problem Family)

**Document ID:** PREREG-JETFLOW-VIDEO-V03-COLLISION-2026-10-19
**Status:** exploratory seed-404 artifact committed BEFORE this freeze; margins below are derived
from it and disclosed as such (weaker provenance than prereg-first, stated rather than hidden).
**Relation:** a NEW prereg per the v0.2 deviation rule — a new world (collision/occlusion,
`apedp_dyn/video/collision_domain.py`) plus any protocol change is a v0.3, never a patch of v0.2.
Nothing in the v0.2 fluid protocol (thresholds, seeds {101,202,303}, instruments) is touched.

## World (frozen)

Layered 2D rigid-body scenes from `make_collision_clip` (deterministic, RandomState,
byte-reproducible): same-layer objects collide elastically (equal mass) and bounce off walls;
cross-layer objects pass through each other's tracks with the nearer layer genuinely occluding
the farther (parallax occlusion). World-shape coverage (pinned by
`tests/test_dyn_collision_domain.py`): per 20-clip family ≥ 3 collision-event clips and
≥ 5 occlusion-event clips — dataset bounds, not discovery thresholds.

## A. Matched-benchmark collision-world protocol (frozen now)

| Item | Frozen value |
|---|---|
| Harness | `benchmarks/run_jetflow_vs_diffusion_matched.py --domain collision` |
| Training | `train_steps=300` fixed-step for both architectures (no wall-clock calibration) |
| Clips | `n_clips=10`; manifest `collision_domain_manifest(n_clips=10, holdout_fraction=0.2, holdout_seed=13)` |
| Holdout | `collision_clip_0003`, `collision_clip_0005` (identical across architectures and rows) |
| Confirmation seeds | **{407, 508, 509}** — seed 404 is the EXPLORATORY source of margins, excluded from confirmation |
| Shapes | latent `(6, 12, 8, 8)`, pixels `(6, 3, 16, 16)` |
| Inference | CFG = 3.0, identical reverse-time convention, shared frozen wavelet VAE, `n_gen=8`, `threads=4` |
| Instruments | `nearest_psnr_db_mean`, `nearest_ssim_mean`, `temporal_consistency_mean`, `fvd_proxy_swd_vs_holdout`, `model_passes_per_clip`, training wall-clock |

## Exploratory seed-404 results (disclosed provenance, NOT confirmatory)

| Row | SSIM | PSNR dB | Temp. smooth | SWD vs holdout | passes/clip |
|---|---|---|---|---|---|
| `jet_tt_2step` | 0.017 | 13.68 | 0.234 | 0.476 | 9.0 |
| `diffusion_matched` (K=4) | **0.065** | **20.76** | **0.078** | **0.167** | 8.0 |
| `diffusion_frontier_K48` | 0.066 | 20.86 | 0.084 | 0.155 | 96.0 |
| `jet_tt_4step` | 0.019 | 15.12 | 0.183 | 0.397 | 16.0 |
| `jet_tt_2step_distilled` | 0.005 | 10.95 | 0.394 | 0.782 | n/a |

Real-anchor SWD (train vs holdout): 0.122. Param counts: jet 216,525 vs baseline 216,396
(matched within 10%). Training wall-clock: jet 11.12 s vs baseline 8.23 s.

## Hypotheses and pre-committed criteria (exploratory point estimates in parentheses)

- **H1-collision (domain-scoping of the advantage).** The fluid-domain ordering does NOT
  transfer: per seed, `diffusion_matched` SSIM > `jet_tt_2step` SSIM **and**
  `diffusion_matched` temporal consistency < `jet_tt_2step` temporal consistency
  (exploratory 0.065 vs 0.017; 0.078 vs 0.234). Required on **≥ 2 of 3 seeds**.
  CONFIRMING H1-collision means the published fluid-domain advantage is **domain-scoped to
  continuum transport** — it is a boundary claim on the existing claim, not an Advantage
  claim for either architecture on this world. NOT CONSIDERED a failure of the framework;
  it disconfirms an unscoped generalization claim.
- **H2 (JetFlow floor on the new world).** `jet_tt_2step` PSNR ≥ **10.0 dB** on every seed
  (exploratory 13.68 dB) — guards against degradation beyond the domain's difficulty.
- **H3 (no training blow-up).** jet training wall-clock / baseline training wall-clock
  **≤ 2.0** on every seed (exploratory ratio 1.35). This is a sanity ceiling, NOT an
  advantage claim: no ≤ 0.5 ratio is registered on this world (the exploratory ratio
  disconfirms it; claiming otherwise would be dishonest).
- **H4 (instrumented pass counts, deterministic).** As in v0.2 with the counter reset after
  warmup: `jet_tt_2step` == 8 structural passes/clip (the artifact's amortized row reads
  9.0), `jet_tt_4step` == 16, on every seed.

## Decision rule

- H1-collision on ≥ 2/3 seeds; H2–H4 on all seeds.
- ANY failure ⇒ the corresponding claim is NOT confirmed and is reported as a negative
  result; no seed, threshold, instrument, or training setting may be adjusted without a
  v0.4 prereg. The model is not re-tuned to green.
- Publication duty: whatever the outcome, the collision-world rows enter the benchmark
  record alongside the fluid rows with identical instrument honesty
  (FVD-proxy is a ranked-separation instrument, NOT I3D-FVD).

## Non-goals

No photorealism or external-model claims; no real-video datasets; no claim that the
layered-world generator's occlusion semantics match human depth perception; the
latent-metric/pixel-metric correspondence question remains out of scope.
