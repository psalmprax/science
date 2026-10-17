r"""JetFlow-Video Distillation Benchmark (teacher -> 2-4-step student).

Status: experimental (benchmark harness — writes a versioned JSON artifact)

The honest replacement for the previous "2 steps beats 50 steps" integrator
benchmark: a small flow-matching teacher is TRAINED on the fluid domain,
sampled at high step count as the reference, and a student is solution-
distilled so its few-step sampler reproduces the teacher's trajectories.
Curves = endpoint RMSE(student, teacher) over (teacher_steps, student_steps)
grid, including the untrained-student baseline.

Run:
    python benchmarks/run_jetflow_distill_benchmark.py \
        [--teacher-steps 200] [--distill-steps 100] [--out ...json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "packages" / "jetflow-video")):
    if p not in sys.path:
        sys.path.insert(0, p)

from apedp_dyn.video.eval_gate import held_out_split
from apedp_dyn.video.fluid_domain import fluid_domain_manifest, make_domain_set
from jetflow_video.bridge import ContinuousWaveletVAE
from jetflow_video.distill import (
    distill_solution,
    distillation_curves,
    train_flow_model,
)
from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel

import torch


def _holdout_train_latents(n_clips: int, t: int, h: int, w: int):
    """Manifest-disciplined split: distillation fits ONLY train ids."""
    man = fluid_domain_manifest(n_clips=n_clips, holdout_fraction=0.2, holdout_seed=13,
                                t=t, h=h, w=w)
    clips = make_domain_set(tuple(man["train_ids"]), tuple(man["holdout_ids"]), t=t, h=h, w=w)
    vae = ContinuousWaveletVAE(latent_channels=12, downsample_factor=2)
    latents = []
    for cid in man["train_ids"]:
        arr = torch.from_numpy(clips[cid][-t:].astype(np.float32))  # keep last t frames
        latents.append(torch.stack([vae.encode_frame(arr[i]) for i in range(arr.shape[0])], dim=0))
    return torch.stack(latents, dim=0).double(), man


def run_distill_benchmark(teacher_steps: int = 200, distill_steps: int = 100,
                          n_clips: int = 10, out_path: str | None = None,
                          seed: int = 0) -> dict:
    torch.manual_seed(seed)
    train_lat, man = _holdout_train_latents(n_clips=n_clips, t=6, h=16, w=16)
    shape = tuple(train_lat.shape[1:])

    teacher = JetFlowVideoModel(JetFlowModelConfig(
        latent_channels=12, hidden_dim=48, num_layers=2)).double()
    t_losses = train_flow_model(teacher, train_lat, steps=teacher_steps, lr=1e-3, seed=seed)

    student = JetFlowVideoModel(JetFlowModelConfig(
        latent_channels=12, hidden_dim=32, num_layers=1)).double()
    base = distillation_curves(teacher, student, shape=shape, n_eval=6,
                               teacher_step_grid=(8, 16), student_step_grid=(2, 4),
                               seed=123)

    d_losses = distill_solution(teacher, student, train_lat, distill_steps=distill_steps,
                                teacher_steps=16, student_steps=2, lr=5e-4,
                                seed=seed + 1, progress_every=max(1, distill_steps // 4))

    after = distillation_curves(teacher, student, shape=shape, n_eval=6,
                                teacher_step_grid=(8, 16), student_step_grid=(2, 4),
                                seed=123)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "convention_note": "flow param t=0 data / t=1 noise; sampling integrates t: 1 -> 0",
        "domain_split": {"train": list(man["train_ids"]), "holdout": list(man["holdout_ids"])},
        "teacher": {"hidden": 48, "layers": 2, "train_steps": teacher_steps,
                    "final_train_loss": t_losses[-1]},
        "student": {"hidden": 32, "layers": 1, "distill_steps": distill_steps,
                    "final_distill_mse": d_losses[-1]},
        "curves": {
            "untrained_baseline": base["rows"],
            "post_distillation": after["rows"],
        },
        "key_deltas": {
            "t16_s2_rmse_before": [r for r in base["rows"] if r["teacher_steps"] == 16 and r["student_steps"] == 2][0]["rmse_mean"],
            "t16_s2_rmse_after": [r for r in after["rows"] if r["teacher_steps"] == 16 and r["student_steps"] == 2][0]["rmse_mean"],
        },
    }
    dst = out_path or str(_REPO / "benchmarks" / "jetflow_distill_curves.json")
    Path(dst).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["key_deltas"], indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-steps", type=int, default=200)
    ap.add_argument("--distill-steps", type=int, default=100)
    ap.add_argument("--n-clips", type=int, default=10)
    ap.add_argument("--out", type=str, default=None, dest="out_path")
    args = ap.parse_args()
    run_distill_benchmark(**vars(args))
