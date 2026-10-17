r"""Teacher -> Student Solution Distillation (2-4 Step Consistency).

Status: experimental

This module makes the "2 steps beats N steps to 50 steps" claim HONEST for the
first time: the previous benchmark compared integrators on an analytic field
they were TOLD. Here a small flow-matching teacher is actually TRAINED,
sampled at a high step count (the expensive-but-accurate reference), and a
smaller student is optimized so its OWN few-step sampler reproduces the
teacher's sampled trajectories. Curves report student endpoint fidelity vs
the teacher as a function of step counts, against the untrained baseline.

Sampling convention (FIXED here): the trainer defines
    z_t = (1 - t) * z_data + t * z_noise,   target v* = z_noise - z_data
so sampling goes from noise (t = 1) DOWN to data (t = 0):
    z <- z + dt * v(z, t)   with timesteps 1 -> 0.
``model.generate_video`` previously integrated the opposite direction
(found bug) — it has been fixed in place to this same convention; this
module keeps its own consistent sampler.

Batch note: ``JetFlowBackbone``'s conditioning path assumes batch size 1
(text-token batch and z batch are combined naively). All functions here loop
over clips one at a time (b = 1) to stay correct.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def flow_matching_loss(
    model,
    z_data: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Plain flow-matching loss on a single clip (b = 1)."""
    if z_data.ndim != 5 or z_data.shape[0] != 1:
        raise ValueError(f"Expected (1, T, C, H, W) latents, got {tuple(z_data.shape)}")
    noise = torch.randn_like(z_data)
    t = float(torch.empty(1).uniform_(0.05, 0.95).item())
    z_t = (1.0 - t) * z_data + t * noise
    target_v = noise - z_data
    pred_v = model(z_t, t, text_tokens=prompt_tokens)
    return {"flow_loss": F.mse_loss(pred_v, target_v), "t": t}


def train_flow_model(
    model,
    clips_latents: torch.Tensor,
    steps: int,
    lr: float = 1e-3,
    seed: int = 0,
    progress_every: int = 0,
) -> List[float]:
    """Train a flow model on stacked clip latents (N, T, C, H, W), b=1 loops."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    n = clips_latents.shape[0]
    dev = clips_latents.device
    dtype = clips_latents.dtype
    losses = []
    for step in range(steps):
        idx = rng.randint(0, n)
        z = clips_latents[idx: idx + 1].to(dev)
        text = torch.randn(1, 2, 64, device=z.device, dtype=dtype)
        opt.zero_grad(set_to_none=True)
        d = flow_matching_loss(model, z, text)
        d["flow_loss"].backward()
        opt.step()
        losses.append(float(d["flow_loss"].item()))
        if progress_every and (step + 1) % progress_every == 0:
            print(f"[train_flow_model] step {step + 1}/{steps} loss={losses[-1]:.4f}")
    return losses


@torch.no_grad()
def euler_sample_reverse(
    model,
    prompt_tokens: torch.Tensor,
    shape: Tuple[int, int, int, int],
    num_steps: int,
    seed: int = 0,
    guidance_scale: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    trajectory: bool = False,
):
    """Reverse-time Euler sampler consistent with the trainer parameterization.

    Integrates t: 1 -> 0 (noise -> data) with z <- z + dt * v.
    """
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(shape, generator=gen, device=dev, dtype=dtype)
    if prompt_tokens.ndim == 1:
        text = prompt_tokens.view(1, 1, -1).to(device=dev, dtype=z.dtype)
    elif prompt_tokens.ndim == 2:
        text = prompt_tokens.view(1, prompt_tokens.shape[0], -1).to(device=dev, dtype=z.dtype)
    else:
        text = prompt_tokens.to(device=dev, dtype=z.dtype)
    traj = []
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        t_curr = float(ts[i])
        t_next = float(ts[i + 1])
        dt = t_next - t_curr  # negative: toward data
        v = model(z, t_curr, text_tokens=text)
        if guidance_scale is not None and guidance_scale > 1.0:
            v_u = model(z, t_curr, text_tokens=None)
            v = v_u + guidance_scale * (v - v_u)
        z = z + dt * v
        if trajectory:
            traj.append(z.clone())
    return (z, traj) if trajectory else z


def _student_sample_trainable(
    student,
    text: torch.Tensor,
    shape: Tuple[int, int, int, int],
    num_steps: int,
    seed: int,
    dtype: torch.dtype,
):
    """Differentiable few-step reverse-time sampler (b = 1); a direct Euler
    integration with gradients flowing through the velocity predictions.

    Uses the same deterministic noise draw as ``euler_sample_reverse`` for the
    same (shape, seed, dtype), so the student starts from the exact noise the
    teacher's reference trajectory started from.
    """
    dev = next(student.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(shape, generator=gen, device=dev, dtype=dtype)
    if text.ndim == 3:
        text = text.to(device=dev, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        dt = float(ts[i + 1]) - float(ts[i])
        v = student(z, float(ts[i]), text_tokens=text)
        z = z + dt * v
    return z


def distill_solution(
    teacher,
    student,
    clips_latents: torch.Tensor,
    distill_steps: int = 200,
    teacher_steps: int = 8,
    student_steps: int = 2,
    lr: float = 5e-4,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    progress_every: int = 0,
) -> List[float]:
    """Solution distillation: student's few-step endpoint == teacher's
    many-step endpoint from the SAME initial noise."""
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    losses = []
    clip_shape = tuple(clips_latents[0].shape)  # (T, C, H, W)
    for step in range(distill_steps):
        noise_seed = int(rng.randint(0, 2 ** 31 - 1))
        text = torch.randn(1, 2, 64, dtype=dtype)
        with torch.no_grad():
            z_teacher = euler_sample_reverse(
                teacher, text, clip_shape, num_steps=teacher_steps,
                seed=noise_seed, dtype=dtype,
            )
        opt.zero_grad(set_to_none=True)
        z_stud = _student_sample_trainable(
            student, text, clip_shape, num_steps=student_steps,
            seed=noise_seed, dtype=dtype,
        )
        loss = F.mse_loss(z_stud, z_teacher)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if progress_every and (step + 1) % progress_every == 0:
            print(f"[distill] step {step + 1}/{distill_steps} mse={losses[-1]:.6f}")
    return losses


def distillation_curves(
    teacher,
    student,
    shape: Tuple[int, int, int, int],
    n_eval: int = 8,
    teacher_step_grid: Sequence[int] = (4, 8, 16),
    student_step_grid: Sequence[int] = (1, 2, 4),
    seed: int = 123,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, object]:
    """End-point fidelity grid: RMSE(student endpoint, teacher endpoint) per
    (teacher_steps, student_steps) pair."""
    teacher.eval()
    was_training = student.training
    student.eval()
    rows = []
    for t_steps in teacher_step_grid:
        for s_steps in student_step_grid:
            dists = []
            for k in range(n_eval):
                text = torch.randn(1, 2, 64, dtype=dtype)
                seed_k = seed * 1000 + k
                z_ref = euler_sample_reverse(teacher, text, shape, num_steps=t_steps, seed=seed_k, dtype=dtype)
                z_stu = euler_sample_reverse(student, text, shape, num_steps=s_steps, seed=seed_k, dtype=dtype)
                dists.append(float(F.mse_loss(z_stu, z_ref).sqrt().item()))
            rows.append({"teacher_steps": int(t_steps), "student_steps": int(s_steps),
                         "rmse_mean": float(np.mean(dists)), "rmse_std": float(np.std(dists))})
    if was_training:
        student.train()
    return {"rows": rows}
