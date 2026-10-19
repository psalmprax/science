# Preregistration: JetFlow-Video v0.2 — Fresh-Seed Confirmation & Architecture Invariants

**Document ID:** PREREG-JETFLOW-VIDEO-V02-2026-10-17
**Status:** DESIGN FROZEN at commit time; confirmation harness and artifacts committed SEPARATELY after.
**Relation:** the JetFlow-Video stack (matched-benchmark line) and the DC-STT neural transport
path (`apedp_dyn/video/`, `packages/jetflow-video/`) had no registered confirmation protocol.
The seed-0 matched benchmark (`benchmarks/jetflow_vs_diffusion_matched_results.json`) and the
fixed-seed unit batteries were **exploratory**: they motivated the v0.2 remediation of the seven
structural defects and produced the point estimates used to set the margins below.
**Disclosure:** the v0.2 implementation already existed uncommitted in the working tree when this
prereg was frozen; the confirmation harness does not exist yet, and the signatures below are
derived from the exploratory seed-0 run (disclosed per hypothesis). This is a weaker provenance
than the ideal prereg-first workflow and is stated rather than hidden.
**Deviation rule:** changing a seed, threshold, world or instrument after this document is frozen
means the confirmation is re-run under a NEW prereg — it is never quietly patched.

## What is registered (and what is NOT touched)

- Confirmation A: fresh-seed replication of the matched benchmark on the frozen protocol below
  (`benchmarks/run_jetflow_vs_diffusion_matched.py::run_matched_benchmark`).
- Confirmation B: architecture invariants on fresh random tensors (causality, SE(3), text
  encoder, Noether projector, TT compression).
- Confirmation C: periodic-transport parity (torch / NumPy fallback / Rust vs generator formula).
- NOT touched: no threshold in `rounded`-instrument code, no change to the exploratory artifact,
  no retroactive re-labeling of seed-0 results as confirmatory.

## A. Matched-benchmark fresh-seed replication (frozen)

Seeds: **{101, 202, 303}** — one full run per seed; the exploratory seed 0 is NOT part of the
confirmation. Protocol (fixed now):

| Item | Frozen value |
|---|---|
| Training | `train_steps=300` for both architectures (fixed-step mode; no wall-clock calibration) |
| Clips | `n_clips=10`; manifest `fluid_domain_manifest(n_clips=10, holdout_fraction=0.2, holdout_seed=13)` |
| Holdout | `fluid_clip_0003`, `fluid_clip_0005` (identical across architectures and rows) |
| Shapes | latent `(6, 12, 8, 8)`, pixels `(6, 3, 16, 16)`, 8 train clips |
| Inference | CFG = 3.0, identical reverse-time convention, shared frozen wavelet VAE, `n_gen=8`, `threads=4` |
| Instruments | `nearest_psnr_db_mean`, `nearest_ssim_mean`, `temporal_consistency_mean`, `fvd_proxy_swd_vs_holdout`, `model_passes_per_clip`, training wall-clock |

Hypotheses and pre-committed criteria (exploratory seed-0 point estimates in parentheses):

- **H1 (architecture ordering).** Per seed: `jet_tt_2step` SSIM > `diffusion_matched` SSIM
  **and** `jet_tt_2step` temporal consistency < `diffusion_matched` temporal consistency
  (exploratory 0.514 vs 0.477; 0.275 vs 0.365). Required on **≥ 2 of 3 seeds**.
- **H2 (remediation floor).** `jet_tt_2step` PSNR ≥ **9.5 dB** on every seed (exploratory 10.74 dB
  post-fix vs 7.49 dB pre-fix; the floor sits well above the documented defect regime).
- **H3 (training-cost claim).** jet training wall-clock / baseline training wall-clock **≤ 0.5** on
  every seed (exploratory 14.4 s vs 80.8 s ≈ 0.18; the 0.5 ceiling leaves headroom for machine variance).
- **H4 (structural counts, deterministic).** On the confirmation harness (counter reset AFTER the
  one warmup generation call): `jet_tt_2step` per-clip backbone passes == **8** and `jet_tt_4step`
  == **16** on every seed. Disclosed caveat: the committed exploratory artifact records 9.0 for
  `jet_tt_2step` because its counter includes the one-time warmup amortized over `n_gen = 8`
  (8 + 8/8 = 9.0); the structural per-clip count under the 2-step CFG path is 8.
  `jet_tt_2step_distilled` passes are **not** registered (not instrumented in the exploratory artifact).

No latency-millisecond thresholds are registered (machine-dependent); latency is reported, not gated.

## B. Architecture invariants on fresh tensors (seeds {2001, 2002, 2003})

All must hold on all seeds; failures are reportable per-invariant.

1. **Causality (`CausalConv3d` / `CausalResBlock3D`).** Perturbing input frames at any
   `t ≥ τ` leaves every output at `t < τ` **bitwise unchanged**.
2. **SE(3) exactness (`depth_se3.py`).** `se3_exp_map` returns `‖RᵀR − I‖∞ < 1e-5` and
   `|det R − 1| < 1e-5`; Plücker orthogonality `max |d·(o×d)| < 1e-5`; predicted depth within the
   engine's declared `[z_min, z_max]`.
3. **Compositional text encoder (`text_encoder.py`).** `encode_prompt_deterministic` is exactly
   reproducible; distinct token orderings differ (`max|Δ| > 1e-3`); negation gating holds on the
   declared prompt pairs (similarity(negated, affirmative) < similarity(synonym paraphrase)).
4. **Noether projector.** Post-projection divergence residual < 5% of the pre-projection scale and
   projector idempotence within 5%.
5. **Tensor-Train compression.** At the registered `(4, 12, 8, 8)` shape: compression ratio > 5
   and reconstruction cosine similarity > 0.95.

## C. Periodic transport parity (seed 3001)

On random wrap-heavy fields: torch transport step, NumPy fallback and the Rust kernel
(`video_warp_bilinear_2d_periodic`) all match the generator formula within **1e-12**
(max-abs), and the periodic warp differs from the clamped warp by > 1e-3
(identity check that the periodic path is actually exercised).

## Decision rule

- B and C: ALL invariants pass on ALL seeds.
- A: H1 on ≥ 2/3 seeds; H2–H4 on all seeds.
- ANY failure ⇒ the corresponding claim is **NOT confirmed**; it is reported as a negative
  result per this prereg, and no seed, threshold, instrument or training setting may be adjusted
  without a v2 prereg. The model is not re-tuned to green during confirmation.

## Metrics and artifacts (planned, committed after the run)

- `benchmarks/run_jetflow_video_v02_confirmation.py`
- `benchmarks/jetflow_video_v02_confirmation_results.json` — per-seed rows keyed by the frozen
  instrument names, probe-calibration metadata, and run provenance (git HEAD, thread count).
- `docs/JETFLOW_VIDEO_V02_CONFIRMATION_REPORT.md` — per-hypothesis verdicts on both sides
  (confirmed / not confirmed), including any failure at raw numbers.

## Non-goals

No photorealism or external-model generation-quality claims; no real-video datasets; no latency
SLA; no claim that the seven-defect remediation generalizes beyond the registered worlds; the
hypothesis that latent-space metrics predict human-perceived quality is explicitly out of scope.
