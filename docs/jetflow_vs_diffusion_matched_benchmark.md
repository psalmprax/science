# JetFlow vs Diffusion — Matched Benchmark (Post-Remediation)

**Artifact:** [`benchmarks/run_jetflow_vs_diffusion_matched.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_jetflow_vs_diffusion_matched.py) → [`benchmarks/jetflow_vs_diffusion_matched_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/jetflow_vs_diffusion_matched_results.json)

## TL;DR

Following the remediation of the 7 structural flaws in the JetFlow-Video stack (adding 3D $(\tau, y, x)$ Fourier coordinate encodings + local 2D 4-neighbor spatial stencil mixing in [`JetFlowBackbone`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/model.py), subband-preserving temporal mass conservation in [`NoetherConservationEngine`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/noether.py), 2-evaluation fused directional material derivatives in [`JetFlowStepper`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/stepper.py), full-precision latent decoding before Tensor-Train compression, and eliminating the AdaLN-Zero / CFG unconditional bias saddle point):

1. **`jet_tt_2step` PSNR improved by +3.25 dB** (from `7.49 dB` to **`10.74 dB`**), and **SSIM improved from `0.254` (and `0.073` in fixed-step mode) to `0.514`**, surpassing both `diffusion_matched` (`0.477` SSIM) and `diffusion_frontier_K48` (`0.492` SSIM).
2. **Temporal consistency (second-difference flicker, lower is smoother) improved from `0.637` to `0.275` (`0.244` at 4 steps)**, beating `diffusion_matched` (`0.365`) and `diffusion_frontier_K48` (`0.372`).
3. **Model forward passes per clip dropped by $2.5\times$** (from `20` passes to **`8` passes** for 2-step CFG generation), and 300-step training wall-clock time was **14.4 s** for JetFlow vs **80.8 s** for the matched UNet baseline (**5.6× faster training** to a lower final flow-matching loss: `0.1528` vs `0.1631`).

## Protocol (measured, not asserted)

| Item | Value |
|---|---|
| Training budget | **300 steps per architecture** (JetFlow consumed **14.40 s** → final loss `0.1528`; Diffusion UNet baseline consumed **80.75 s** → final loss `0.1631`) |
| Domain | Synthetic 2D incompressible-fluid dye advecting clips (`make_fluid_clip`) |
| Split | 8 train clips / 2 hold-out clips (`fluid_clip_0003`, `fluid_clip_0005`; never fit during training) |
| Pixel/latent shapes | 6 frames × 3 channels × 16×16 pixels; latent `(6, 12, 8, 8)` |
| Parameters | JetFlow = `138,780`, Diffusion Baseline = `152,492` (matched within 10%) |
| Inference | CFG = `3.0`, identical reverse-time convention $t: 1 \to 0$, shared frozen `ContinuousWaveletVAE` |

## Results (`benchmarks/jetflow_vs_diffusion_matched_results.json`, seed 0, 4 threads)

| Row | Steps | Model Passes/Clip | PSNR (dB) | SSIM | FVD-proxy SWD | Temp. smooth (↓) |
|---|---|---|---|---|---|---|
| `jet_tt_2step` (remediated) | 2 | **8.0** | 10.74 | **0.514** | 0.518 | **0.275** |
| `jet_tt_2step` *(pre-fix v2)* | 2 | 20.0 | 7.49 | 0.254 | 1.061 | 0.637 |
| `jet_tt_2step_distilled` | 2 | 4.0 | 8.63 | 0.341 | 0.802 | 0.538 |
| `diffusion_matched` | 2 | 4.0 | **10.97** | 0.477 | **0.511** | 0.365 |
| `diffusion_frontier_K48` | 48 | 96.0 | 10.92 | 0.492 | 0.517 | 0.372 |
| `jet_tt_4step` (remediated) | 4 | 16.0 | 10.44 | 0.495 | 0.517 | **0.244** |

## Second problem family (collision/occlusion, v0.3 prereg) — honest boundary result

The matched-benchmark protocol now carries a second world (`--domain collision`,
`docs/PREREGISTRATION_JETFLOW_VIDEO_V03_COLLISION.md`, exploratory seed 404):
layered rigid-body scenes with elastic same-layer collisions and genuine cross-layer
parallax occlusion (`apedp_dyn/video/collision_domain.py`). The exploratory run is a
disclosed NEGATIVE result for advantage transfer: on the collision world, the matched
diffusion baseline wins on every instrument (SSIM 0.065 vs 0.017, PSNR 20.8 vs 13.7 dB,
temporal smoothness 0.078 vs 0.234, SWD vs holdout 0.167 vs 0.476) and trains faster
(8.2 s vs 11.1 s at 300 fixed steps). Conclusion recorded per the prereg: the
fluid-domain advantage is DOMAIN-SCOPED to continuum transport; no unscoped
generalization claim is made. Confirmation seeds {407, 508, 509} are frozen in the
v0.3 prereg to replicate the ordering (H1-collision).

Notes on instrument symbolism:
- **FVD-proxy SWD** = sliced-Wasserstein-2 on a frozen deterministic featurizer (NOT I3D-FVD; honest label per [`apedp_dyn/video/eval_gate.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/eval_gate.py)). Real-vs-holdout anchor SWD = `0.388`.
- **Temp. smooth** = temporal second-difference of per-frame grayscale; lower = smoother (less flicker).
- `model_passes_per_clip` is directly instrumented on `backbone.forward`. For `jet_tt_2step` the recorded 9.0 includes the one-time warmup generation amortized over `n_gen = 8`; the structural per-clip count under the 2-step CFG path is **8** (the confirmation protocol resets the counter after warmup; see `docs/PREREGISTRATION_JETFLOW_VIDEO_V02.md`).

