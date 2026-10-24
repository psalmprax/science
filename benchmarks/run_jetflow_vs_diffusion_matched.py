r"""JetFlow-Video TT Architecture vs Standard Diffusion — Matched CPU Budget.

Status: experimental (benchmark harness — writes a versioned JSON artifact)

Honest apples-to-apples comparison of the JetFlow-Video stack (4-core Tensor-
Train video latent + 2nd-order Jet2 curvature stepper + Noether solenoidal
projector + physical steering) against a deliberately standard compact
diffusion baseline (spatio-temporal factorized flow-matching UNet + plain
reverse-time Euler sampler), on ONE shared latent space, ONE shared decoder,
ONE shared held-out split, and ONE shared optimization protocol, with the
inference budget matched by measured CPU wall-clock per generated clip.

Pipeline discipline (AGENTS.md 1.2 at the data-split level):
- Both models train ONLY on manifest train_ids of the fluid domain
  (holdout_fraction=0.2, holdout_seed=13). Holdout clips are never fit.
- Both share the frozen ContinuousWaveletVAE (encoder for training latents,
  decoder for RGB); the decoder contributes no learned parameters and is
  identical for both models, so pixel metrics are attributable to the two
  generative backbones alone.
- One shared objective (flow matching, distill.py convention) and one shared
  AdamW schedule; the baseline is param-matched by width search (+/-15%),
  with both exact parameter counts recorded.
- Inference budget is matched by wall-clock on a fixed thread count: the
  baseline's step count K is chosen so its median per-clip latency is closest
  to the JetFlow 2-step median. A K=32 frontier row and a 4-step JetFlow row
  are reported for reference beyond the matched point.

Metrics use the repo's own eval-gate instruments (kernel-backed when the
native module is present): PSNR / SSIM vs the nearest held-out real clip,
temporal consistency, and the sliced-Wasserstein FVD proxy (honest label:
ranked-separation instrument, NOT I3D-FVD).

Run:
    python benchmarks/run_jetflow_vs_diffusion_matched.py \
        [--train-steps 300] [--n-clips 10] [--n-gen 8] [--threads 4] [--out ...json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "packages" / "jetflow-video")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from jetflow_video.bridge import ContinuousWaveletVAE, VideoVAEBridge
from jetflow_video.diffusion_baseline import (
    DiffusionVideoUNet,
    count_parameters,
    euler_sample_baseline,
    pick_matched_config,
    train_flow_model_baseline,
)
from jetflow_video.distill import distill_solution, euler_sample_reverse, train_flow_model
from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel

from apedp_dyn.video.eval_gate import (
    compute_psnr,
    compute_ssim,
    fvd_proxy,
    measure_temporal_consistency,
)
from apedp_dyn.video.fluid_domain import fluid_domain_manifest, make_domain_set


def _load_domain_latents(domain: str, n_clips: int, t: int, h: int, w: int):
    """Manifest-disciplined split for the SELECTED domain: both models fit ONLY
    train ids. ``domain="fluid"`` preserves the original exploratory path
    byte-for-byte (same manifest call, same holdout seeding)."""
    if domain == "fluid":
        man = fluid_domain_manifest(n_clips=n_clips, holdout_fraction=0.2, holdout_seed=13,
                                    t=t, h=h, w=w)
        ids = list(man["train_ids"]) + list(man["holdout_ids"])
        clips_all = make_domain_set(tuple(man["train_ids"]), tuple(man["holdout_ids"]), t=t, h=h, w=w)
    elif domain == "collision":
        from apedp_dyn.video.collision_domain import collision_domain_manifest, make_domain_set as _mk
        man = collision_domain_manifest(n_clips=n_clips, holdout_fraction=0.2, holdout_seed=13,
                                        t=t, h=h, w=w)
        ids = list(man["train_ids"]) + list(man["holdout_ids"])
        clips_all = _mk(tuple(man["train_ids"]), tuple(man["holdout_ids"]), t=t, h=h, w=w)
    else:
        raise ValueError(f"unknown domain: {domain}")
    clips = clips_all  # already keyed by manifest ids, holdouts included
    vae = ContinuousWaveletVAE(latent_channels=12, downsample_factor=2)
    latents = {}
    for cid in man["train_ids"]:
        arr = torch.from_numpy(np.asarray(clips[cid][-t:], dtype=np.float32))
        latents[cid] = torch.stack([vae.encode_frame(arr[i]) for i in range(arr.shape[0])], dim=0)
    return torch.stack(list(latents.values()), dim=0).double(), man, clips, vae


def _load_latents(n_clips: int, t: int, h: int, w: int):
    """Fluid-domain wrapper retained for byte-compatible exploratory use."""
    return _load_domain_latents("fluid", n_clips, t, h, w)


def _calibrated_train_steps(train_fn, model, train_lat, budget_s: float, seed: int) -> tuple[int, list[float]]:
    """Measure per-step cost on a probe run, then return the step count that
    consumes ≈ the same wall-clock budget for this architecture (matched
    TRAINING budget = equal seconds, not equal epochs/steps). The probe is
    long enough (200 steps) that first-call warm-up cost is diluted to <1% of
    the estimate (verified below by the achieved-time check in the JSON).

    The returned count is padded ×1.25 upward-bound: real-run per-step cost can
    drift from the probe in either direction, so the budget axis is enforced
    by the trainer's own ``deadline_s`` (probe is for step-count shape, the
    seconds are enforced in-loop)."""
    probe_steps = 200
    t0 = time.perf_counter()
    _probe = train_fn(model, train_lat, steps=probe_steps, lr=1e-3, seed=seed)
    per_step = (time.perf_counter() - t0) / float(probe_steps)
    steps = max(50, int(1.25 * budget_s / max(1e-6, per_step)))
    return steps, _probe


def _median(xs: list[float]) -> float:
    return float(np.median(xs))


def _to_np(t: object) -> np.ndarray:
    """Grad-free torch->numpy conversion for instrumentation (values identical)."""
    tt = torch.asarray(t, dtype=torch.float64, requires_grad=False)  # type: ignore[arg-type]
    return tt.detach().cpu().numpy()


def _instrument_fwd(model: object, counter: dict[str, int]) -> None:
    """Count actual model-forward passes (JetFlow wraps its backbone)."""
    inner = model.backbone if hasattr(model, "backbone") else model  # type: ignore[attr-defined]
    orig = inner.forward

    def counted(*args: object, **kwargs: object):
        counter["n"] += 1
        return orig(*args, **kwargs)  # type: ignore[misc]

    inner.forward = counted  # type: ignore[method-assign]


def _nearest_holdout_metrics(gen_rgb: np.ndarray, holdout_rgb: list[np.ndarray]) -> dict[str, float]:
    """Visual similarity to the MOST similar held-out real clip (max over holdout)."""
    psnrs = [compute_psnr(gen_rgb, real) for real in holdout_rgb]
    ssims = [compute_ssim(gen_rgb, real) for real in holdout_rgb]
    best = int(np.argmax(psnrs))
    return {"nearest_psnr_db": float(psnrs[best]), "nearest_ssim": float(ssims[best])}


def run_matched_benchmark(train_steps: int = 300, n_clips: int = 10, n_gen: int = 8,
                          threads: int = 4, seed: int = 0, train_budget_s: float | None = None,
                          out_path: str | None = None, domain: str = "fluid") -> dict:
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_lat, man, clips, _vae = _load_domain_latents(domain, n_clips, t=6, h=16, w=16)
    shape = tuple(train_lat.shape[1:])  # (T, C, H', W')
    bridge = VideoVAEBridge(latent_channels=12, downsample_factor=2,
                            device=torch.device("cpu"))

    holdout_rgb = [np.asarray(clips[cid], dtype=np.float64) for cid in man["holdout_ids"]]
    train_real = [np.asarray(clips[cid], dtype=np.float64) for cid in man["train_ids"]]

    # ---- build: JetFlow teacher config, param-matched baseline ---------------
    jet = JetFlowVideoModel(JetFlowModelConfig(latent_channels=12, hidden_dim=48, num_layers=2)).double()
    n_jet = count_parameters(jet)
    cfg = pick_matched_config(n_jet)
    base = DiffusionVideoUNet(cfg).double()
    n_base = count_parameters(base)

    # ---- train under the identical protocol, timing both -------------------
    # Matched-budget mode: probe per-arch throughput on a fresh throwaway model,
    # then give each architecture the SAME wall-clock seconds (the step counts
    # legitimately differ; the seconds are the budget axis).
    probe_jet = JetFlowVideoModel(JetFlowModelConfig(latent_channels=12, hidden_dim=48, num_layers=2)).double()
    probe_base = DiffusionVideoUNet(cfg).double()
    if train_budget_s is not None:
        jet_steps, _ = _calibrated_train_steps(train_flow_model, probe_jet, train_lat, train_budget_s, seed)
        base_steps, _ = _calibrated_train_steps(train_flow_model_baseline, probe_base, train_lat, train_budget_s, seed)
        budget_note = f"wall-clock matched: {train_budget_s:.0f}s per arch (jet steps={jet_steps}, baseline steps={base_steps})"
        train_steps_jet, train_steps_base = jet_steps, base_steps
    else:
        train_steps_jet = train_steps_base = int(train_steps)
        budget_note = f"fixed-step mode: {train_steps} steps for both"

    t0 = time.perf_counter()
    jet_deadline = train_budget_s if train_budget_s is not None else None
    jet_losses = train_flow_model(jet, train_lat, steps=train_steps_jet, lr=1e-3, seed=seed,
                                  deadline_s=jet_deadline)
    jet_train_s = time.perf_counter() - t0
    jet_achieved_steps = len(jet_losses)
    t0 = time.perf_counter()
    base_losses = train_flow_model_baseline(base, train_lat, steps=train_steps_base, lr=1e-3, seed=seed,
                                            deadline_s=jet_deadline)
    base_train_s = time.perf_counter() - t0
    jet.train()
    base.train()
    jet.eval()
    base.eval()

    # ---- shared per-clip conditions (text tokens are drawn once per seed) --
    conds: dict[int, torch.Tensor] = {}
    g = torch.Generator().manual_seed(9000 + seed)
    for k in range(n_gen):
        conds[k] = torch.randn(2, 64, generator=g, dtype=torch.float64)

    # ---- generate with the JetFlow stack (2-step and 4-step) ---------------
    def _gen_jet(num_steps: int, counter: dict[str, int]) -> tuple[list[np.ndarray], list[float]]:
        vids, lats_ms = [], []
        jet.generate_video(conds[0], num_steps=num_steps, shape=shape)  # warmup
        for k in range(n_gen):
            t0 = time.perf_counter()
            _tt, rgb = jet.generate_video(conds[k], num_steps=num_steps, shape=shape)
            lats_ms.append((time.perf_counter() - t0) * 1e3)
            vids.append(_to_np(rgb))
        return vids, lats_ms

    cc2: dict[str, int] = {"n": 0}
    _instrument_fwd(jet, cc2)
    t_gen0 = time.perf_counter()
    jet2, jet2_ms = _gen_jet(2, cc2)
    t_gen_jet2 = time.perf_counter() - t_gen0
    jet2_passes = cc2["n"] / max(1, n_gen)

    tt_4 = None
    jet4_ms: list[float] = []
    jet4: list[np.ndarray] = []
    cc4: dict[str, int] = {"n": 0}
    _instrument_fwd(jet, cc4)
    for k in range(n_gen):
        t0 = time.perf_counter()
        tt_crop_v, rgb4 = jet.generate_video(conds[k], num_steps=4, shape=shape)
        jet4_ms.append((time.perf_counter() - t0) * 1e3)
        jet4.append(_to_np(rgb4))
        tt_4 = tt_crop_v
    jet4_passes = cc4["n"] / max(1, n_gen)

    # ---- generate with the diffusion baseline across a step grid -----------
    # ---- JetFlow student distilled to few steps (the stack's shipped claim)
    # The distilled student costs an EXTRA equal wall-clock budget on top of
    # the teacher's; that asymmetry is stated in the JSON rather than hidden.
    student = JetFlowVideoModel(JetFlowModelConfig(latent_channels=12, hidden_dim=32, num_layers=1)).double()
    n_student = count_parameters(student)
    t0 = time.perf_counter()
    _probe_d = distill_solution(jet, student, train_lat, distill_steps=20, teacher_steps=16,
                                student_steps=2, lr=5e-4, seed=seed + 1)
    per_d_step = (time.perf_counter() - t0) / 20.0
    if train_budget_s is not None:
        distill_steps = max(50, int(train_budget_s / max(1e-6, per_d_step)))
    else:
        distill_steps = 100
    d_losses = distill_solution(jet, student, train_lat, distill_steps=distill_steps,
                                teacher_steps=16, student_steps=2, lr=5e-4,
                                seed=seed + 1,
                                deadline_s=train_budget_s if train_budget_s is not None else None)
    distill_train_s = time.perf_counter() - t0 + per_d_step * 0  # probe included
    student.eval()

    from jetflow_video.tensor_train import compress_video_latent_torch
    stud_vids, stud_lat_ms = [], []
    student.generate_video(conds[0], num_steps=2, shape=shape)  # warmup
    for k in range(n_gen):
        t0 = time.perf_counter()
        z_stud = euler_sample_reverse(student, conds[k], shape, num_steps=2, seed=222 + k,
                                      guidance_scale=3.0)
        tt_stud = compress_video_latent_torch(z_stud, max_rank=16)
        rgb_stud = bridge.decode_tt_to_video(tt_stud)
        stud_lat_ms.append((time.perf_counter() - t0) * 1e3)
        stud_vids.append(_to_np(rgb_stud))
    stud_med = _median(stud_lat_ms)

    step_grid = (2, 4, 8, 12, 16, 24, 32, 48)
    base_rows: dict[int, tuple[list[np.ndarray], list[float], float]] = {}
    for k_steps in step_grid:
        vids, ms = [], []
        if int(k_steps) == int(step_grid[0]):
            base0 = euler_sample_baseline(base, conds[0], shape, num_steps=k_steps, seed=111, guidance_scale=3.0)
            bridge.decode_latent_to_video(base0)
        ccb: dict[str, int] = {"n": 0}
        _instrument_fwd(base, ccb)
        for k in range(n_gen):
            t0 = time.perf_counter()
            z0 = euler_sample_baseline(base, conds[k], shape, num_steps=k_steps, seed=222 + k,
                                       guidance_scale=3.0)
            ms.append((time.perf_counter() - t0) * 1e3)
            rgb0 = bridge.decode_latent_to_video(z0)
            vids.append(_to_np(rgb0))
        base_rows[int(k_steps)] = (vids, ms, ccb["n"] / max(1, n_gen))

    # decode-one-clip cost for the shared VAE (excluded from sampler budgets)
    z_ref = base_rows[step_grid[0]][0][0]  # decoded first baseline clip shape check
    _ = z_ref.shape
    t0 = time.perf_counter()
    bridge.decode_latent_to_video(euler_sample_baseline(base, conds[0], shape, num_steps=2, seed=5, guidance_scale=1.0))
    decode_ms = (time.perf_counter() - t0) * 1e3

    jet2_med = _median(jet2_ms)

    matched_k = min(step_grid, key=lambda kk: abs(_median(base_rows[int(kk)][1]) - jet2_med))
    matched_lat_ms = base_rows[int(matched_k)][1]
    matched_gap_pct = abs(_median(matched_lat_ms) - jet2_med) / max(1e-12, jet2_med) * 100.0

    frontier_k = step_grid[-1]

    def _metrics(vids: list[np.ndarray], cx: float) -> dict:
        nm = [_nearest_holdout_metrics(v, holdout_rgb) for v in vids]
        tc = float(np.mean([measure_temporal_consistency(v) for v in vids]))
        fvd_ho = fvd_proxy(holdout_rgb, vids, seed=7)["swd"]
        fvd_tr = fvd_proxy(train_real, vids, seed=7)["swd"]
        return {
            "nearest_psnr_db_mean": float(np.mean([m["nearest_psnr_db"] for m in nm])),
            "nearest_ssim_mean": float(np.mean([m["nearest_ssim"] for m in nm])),
            "temporal_consistency_mean": tc,
            "fvd_proxy_swd_vs_holdout": float(fvd_ho),
            "fvd_proxy_swd_vs_train": float(fvd_tr),
            "decode_ms_per_clip": cx,
        }

    real_anchor = fvd_proxy(train_real, holdout_rgb, seed=7)["swd"]

    rows = {
        "jet_tt_2step": {
            "latency_ms_per_clip_median": jet2_med,
            "latency_ms_per_clip_all": [round(x, 3) for x in jet2_ms],
            "model_passes_per_clip": jet2_passes,
            "total_generation_s": t_gen_jet2,
            "params": n_jet,
            **_metrics(jet2, decode_ms),
            "tt_compression_ratio": float(tt_4.compression_ratio if tt_4 is not None else 0.0),
        },
        "diffusion_matched": {
            "steps": int(matched_k),
            "latency_ms_per_clip_median": _median(matched_lat_ms),
            "latency_ms_per_clip_all": [round(x, 3) for x in matched_lat_ms],
            "match_gap_pct": matched_gap_pct,
            "model_passes_per_clip": base_rows[int(matched_k)][2],
            "params": n_base,
            **_metrics(base_rows[int(matched_k)][0], decode_ms),
        },
        f"diffusion_frontier_K{int(frontier_k)}": {
            "steps": int(frontier_k),
            "latency_ms_per_clip_median": _median(base_rows[int(frontier_k)][1]),
            "model_passes_per_clip": base_rows[int(frontier_k)][2],
            "params": n_base,
            **_metrics(base_rows[int(frontier_k)][0], decode_ms),
        },
        "jet_tt_4step": {
            "latency_ms_per_clip_median": _median(jet4_ms),
            "model_passes_per_clip": jet4_passes,
            "params": n_jet,
            **_metrics(jet4, decode_ms),
        },
        "jet_tt_2step_distilled": {
            "latency_ms_per_clip_median": stud_med,
            "train_budget_s": float(distill_train_s),
            "distill_steps": int(distill_steps),
            "distill_final_mse": d_losses[-1],
            "params": n_student,
            **_metrics(stud_vids, decode_ms),
        },
    }
    # Row order for stable doc table
    frontier_name = f"diffusion_frontier_K{int(frontier_k)}"
    ordered = ("jet_tt_2step", "jet_tt_2step_distilled", "diffusion_matched", frontier_name, "jet_tt_4step")
    step_curve = [{"steps": kk, "latency_ms_median": _median(base_rows[int(kk)][1]),
                   "model_passes_per_clip": base_rows[int(kk)][2],
                   **_metrics(base_rows[int(kk)][0], decode_ms)} for kk in step_grid]

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "convention_note": "flow param t=0 data / t=1 noise; sampling integrates t: 1 -> 0 (distill.py convention)",
        "honest_labels": {
            "fvd": "sliced-Wasserstein proxy over frozen deterministic features — ranked separation, NOT I3D-FVD",
            "nearest_psnr_ssim": "visual similarity to the most-similar held-out real clip, not identity",
            "decode_ms": "shared decoder cost measured once, excluded from sampler budgets",
        },
        "protocol": {
            "n_clips": int(n_clips), "n_train": len(man["train_ids"]), "n_holdout": len(man["holdout_ids"]),
            "domain_split": {"train": list(man["train_ids"]), "holdout": list(man["holdout_ids"])},
            "latent_shape": list(shape), "pixel_shape": [6, 3, 16, 16],
            "train_steps": {"jet": int(train_steps_jet), "baseline": int(train_steps_base)},
            "achieved_steps": {"jet": int(jet_achieved_steps), "baseline": len(base_losses)},
            "train_lr": 1e-3,
            "budget_mode": budget_note,
            "train_time_s_jet": jet_train_s, "train_time_s_baseline": base_train_s,
            "extra_distill_phase": {
                "note": "the distilled 2-step student consumes one MORE equal second-budget after the teacher",
                "budget_s": float(train_budget_s) if train_budget_s is not None else "fixed-steps mode",
                "achieved_s": float(distill_train_s),
                "per_d_step_probe_s": float(per_d_step),
                "achieved_steps": len(d_losses),
            },
            "final_train_loss_jet": jet_losses[-1], "final_train_loss_baseline": base_losses[-1],
            "threads": int(threads), "seed": int(seed),
            "params_jet": n_jet, "params_baseline": n_base,
            "baseline_width": int(cfg.width), "baseline_blocks": int(cfg.num_res_blocks),
            "guidance_scales": {"jet": 3.0, "baseline": 3.0},
            "matched_k": int(matched_k), "matched_gap_pct": matched_gap_pct,
        },
        "rows": [{"name": nm, **rows[nm]} for nm in ordered],
        "baseline_step_curve": step_curve,
        "real_anchor": {"fvd_proxy_swd_train_vs_holdout": real_anchor},
        "training": {
            "jet_losses_head": [round(x, 6) for x in jet_losses[:5]],
            "jet_losses_tail": [round(x, 6) for x in jet_losses[-5:]],
            "baseline_losses_head": [round(x, 6) for x in base_losses[:5]],
            "baseline_losses_tail": [round(x, 6) for x in base_losses[-5:]],
        },
    }

    dst = out_path or str(_REPO / "benchmarks" / "jetflow_vs_diffusion_matched_results.json")
    Path(dst).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="fluid", choices=("fluid", "collision"),
                    help="problem family: fluid (original) or collision (second family, v0.3 prereg)")
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--train-budget-s", type=float, default=None,
                    help="equal wall-clock seconds per arch; overrides --train-steps")
    ap.add_argument("--n-clips", type=int, default=10)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    res = run_matched_benchmark(
        train_steps=args.train_steps, n_clips=args.n_clips, n_gen=args.n_gen,
        threads=args.threads, seed=args.seed, train_budget_s=args.train_budget_s,
        out_path=args.out, domain=args.domain)
    print(json.dumps(res["protocol"], indent=2))
    for r in res["rows"]:
        print({k: r.get(k) for k in ("name", "latency_ms_per_clip_median", "steps",
                                     "nearest_psnr_db_mean", "nearest_ssim_mean",
                                     "fvd_proxy_swd_vs_holdout", "temporal_consistency_mean")})
