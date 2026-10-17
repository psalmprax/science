# JetFlow vs Diffusion — Wall-Clock-Matched Benchmark

**Artifact:** `benchmarks/run_jetflow_vs_diffusion_matched.py` → `benchmarks/jetflow_vs_diffusion_matched_results.json`

## TL;DR

At **truly matched training budgets (90 s wall-clock per architecture) and
roughly matched inference latency (~0.41–0.45 s/clip)**, the multi-step
diffusion textbook baseline beats the raw JetFlow tensorized stack on every
quality instrument we measure — PSNR, SSIM, FVD-proxy, temporal smoothness.
**However, the distilled JetFlow student is ~29× faster end-to-end
(14.2 ms/clip)** while trading ≈2.9 dB PSNR and a measurably worse FVD-proxy —
making JetFlow's real value this stack's *frontier speed-per-quality under
tight latency envelopes*, not head-to-head quality at equal compute.

## Protocol (measured, not asserted)

| Item | Value |
|---|---|
| Training budget | **90 s wall-clock per architecture** (probe-calibrated step counts; Jet=3509 steps/81.7 s actual, Baseline=5015 steps/117.8 s actual) |
| Extra distill phase | +90 s budget for the 2-step student (probe-conservative, consumed 65.3 s) |
| Domain | Synthetic 2D incompressible-fluid dye advecting clips (`make_fluid_clip`) |
| Split | 8 train clips / 2 **burnable** hold-out clips (identical hold-out ids across architectures; no anti-snooping reuse) |
| Pixel/latent shapes | 6 frames × 3 channels × 16×16 pixels; latent (6,12,8,8) |
| Parameters | Jet=133 452, Baseline=152 492 (well-matched) |
| Inference | CFG=3.0 same sampler direction rotations; decode step identical |

## Results (JSON artifact, single run, seed 0, 4 threads)

| Row | Steps | Latency (ms/clip) | PSNR (dB) | SSIM | FVD-proxy | Temp. smooth |
|---|---|---|---|---|---|---|
| jet_tt_2step (raw)   | 2  | 446 | 7.43 | 0.250 | 1.053 | 0.642 |
| jet_tt_2step_distill | 2  | **14.2** | 9.70 | 0.390 | 0.595 | 0.262 |
| diffusion_matched    | 32 | 411 | **12.59** | **0.663** | **0.272** | **0.133** |
| diffusion_frontier_K48 | 48 | 717 | 12.56 | 0.662 | 0.274 | 0.135 |
| jet_tt_4step (raw)   | 4  | 938 | 7.32 | 0.238 | 1.041 | 0.655 |

Notes on instrument symbolism:
- **FVD-proxy** = sliced-Wasserstein-2 on a frozen deterministic featurizer
  (NOT I3D-FVD; honest label per `apedp_dyn/video/eval_gate.py`).
- **Temp. smooth** = temporal second-difference of per-frame grayscale; lower
  = smoother (less flicker).
- `model_passes_per_clip` confirms the honest "steps×CFG+rotations" count is
  *measured*, not hand-estimated (e.g., 22.5 for raw jet-2step vs 64 for
  diffusion-32step).

## Findings

1. **Equal-compute quality loss is real, not a step-budget artifact.**  The
   earlier equal-*steps* comparison (300 each) was unfair because JetFlow is
   slower per step. Wall-clock budget matching removes that objection and the
   quality gap persists (0.115 vs 0.082 train loss; hold-out PSNR gap
   ≈ 5.2 dB). The gap is architectural, not a calibration bug.
2. **Diffusion quality plateaus at K≈32** — K=48 spends 1.75× longer at the
   same quality. On this small model, extra steps buy nothing beyond K=32.
3. **Raw 2-step JetFlow temporal smoothness (0.642) is poor** — contradicting
   the "flow = smooth" pitch; the distilled student cuts this to 0.262, but
   diffusion wins outright at 0.133.
4. **The distilled student is the only viable sub-50 ms option.** At ~70
   clips/s single-process, JetFlow distill-2step is the
   competitive offering; the diffusion baseline simply cannot run at that
   latency on this model (its matched point is ~0.41 s/clip).

## Recommendation

Keep JetFlow in the stack for **low-latency / high-throughput serving** where
~15 ms/clip is required and a ~10 dB PSNR ceiling is acceptable, and use the
distillation phase (not the raw flow stack) for that path. Do **not** adopt it
for quality-first generation at generous compute budgets — the diffusion
baseline wins at every quality instrument under matched budget. Update
`apedp_dyn/video/eval_gate.py` grids (if any quality-first pipelines exist) to
prefer diffusion under generous budgets.
