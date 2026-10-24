r"""Anchored (real-first-frame) conditioning + teacher->student distillation tests.

Pins (torch, CPU, b=1 loop pattern):
1. Anchor discipline: anchored corruption keeps frame 0 clean at every t;
   the loss excludes the anchor frame; gradients flow to parameters.
2. Anchored generation respects the anchor bitwise after every step, in the
   trainer-consistent sampling direction (t: 1 -> 0).
3. Training on the fluid domain reduces the flow loss below the frozen init.
4. Distillation: the few-step student reaches lower endpoint RMSE vs the
   teacher than the untrained student baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "packages" / "jetflow-video"
for p in (str(_REPO), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from jetflow_video.bridge import ContinuousWaveletVAE
from jetflow_video.distill import (
    distill_solution,
    distillation_curves,
    euler_sample_reverse,
    train_flow_model,
)
from jetflow_video.first_frame import (
    anchored_flow_matching_loss,
    generate_anchored_video,
    sample_anchored_z_t,
)
from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel

from apedp_dyn.video.fluid_domain import make_fluid_clip


def _fluid_latents(n_clips: int = 4, t: int = 6, h: int = 16, w: int = 16, seed0: int = 100) -> torch.Tensor:
    vae = ContinuousWaveletVAE(latent_channels=12, downsample_factor=2)
    latents = []
    for k in range(n_clips):
        vid = make_fluid_clip(seed0 + k, t=t + 2, h=h * 2, w=w * 2)
        arr = torch.from_numpy(vid[-t:].astype(np.float32))
        lat = torch.stack([vae.encode_frame(arr[i]) for i in range(arr.shape[0])], dim=0)
        latents.append(lat)
    return torch.stack(latents, dim=0).double()  # (N, T, 12, 16, 16)


@pytest.fixture(scope="module")
def fluid_latents() -> torch.Tensor:
    torch.manual_seed(0)
    return _fluid_latents()


def _tiny_model(hidden: int = 32, layers: int = 1, seed: int = 0) -> JetFlowVideoModel:
    torch.manual_seed(seed)
    cfg = JetFlowModelConfig(latent_channels=12, hidden_dim=hidden, num_layers=layers, jet_order=2)
    return JetFlowVideoModel(cfg).double()


def test_anchor_discipline_corruption_loss_and_grads(fluid_latents):
    z = fluid_latents[0:1]
    for t in (0.1, 0.5, 0.9):
        noise = torch.randn_like(z)
        z_t = sample_anchored_z_t(z, t, noise)
        np.testing.assert_array_equal(z_t[:, 0].numpy(), z[:, 0].numpy())
        expected_rest = (1.0 - t) * z[:, 1:] + t * noise[:, 1:]
        np.testing.assert_allclose(z_t[:, 1:].numpy(), expected_rest.numpy(), rtol=1e-12)

    model = _tiny_model(seed=1)
    text = torch.randn(2, 64).double()
    d = anchored_flow_matching_loss(model, z, text)
    assert torch.isfinite(d["flow_loss"]).item() and d["flow_loss"].item() > 0.0
    d["flow_loss"].backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert gnorm > 0.0, gnorm


def test_anchored_generation_respects_anchor_bitwise(fluid_latents):
    z = fluid_latents[0:1]
    anchor = z[0, 0].clone()  # (12, 16, 16) clean data latent
    model = _tiny_model(seed=2)
    text = torch.randn(2, 64).double()
    z_gen = generate_anchored_video(model, text, anchor, residual_frames=5,
                                    num_steps=4, guidance_scale=0.0, trace_seed=7)
    assert z_gen.shape == (1, 6, 12, 16, 16)
    np.testing.assert_array_equal(z_gen[0, 0].numpy(), anchor.numpy())


def test_flow_training_reduces_loss_on_fluid_domain(fluid_latents):
    model = _tiny_model(seed=3)
    z = fluid_latents[0:1]
    text = torch.randn(2, 64).double()
    frozen = [float(anchored_flow_matching_loss(model, z, text)["flow_loss"].item())
              for _ in range(20)]
    losses = train_flow_model(model, fluid_latents[:3], steps=60, lr=1e-3, seed=0)
    assert float(np.mean(losses[-10:])) < float(np.mean(frozen)), (
        float(np.mean(losses[-10:])), float(np.mean(frozen))
    )


def test_distillation_improves_few_step_fidelity(fluid_latents):
    teacher = _tiny_model(hidden=32, layers=1, seed=4)
    train_flow_model(teacher, fluid_latents[:3], steps=40, lr=1e-3, seed=1)
    student = _tiny_model(hidden=32, layers=1, seed=5)
    shape = tuple(fluid_latents.shape[1:])

    base = distillation_curves(teacher, student, shape=shape, n_eval=4,
                               teacher_step_grid=(8,), student_step_grid=(2,), seed=123)
    base_rmse = base["rows"][0]["rmse_mean"]

    distill_solution(teacher, student, fluid_latents[:3], distill_steps=25,
                     teacher_steps=8, student_steps=2, lr=5e-4, seed=2, progress_every=10)

    after = distillation_curves(teacher, student, shape=shape, n_eval=4,
                                teacher_step_grid=(8,), student_step_grid=(2,), seed=123)
    after_rmse = after["rows"][0]["rmse_mean"]
    assert after_rmse < base_rmse, (after_rmse, base_rmse)


def test_reverse_time_sampler_direction_matches_training():
    model = _tiny_model(seed=6)
    text = torch.randn(1, 2, 64).double()
    z = euler_sample_reverse(model, text, (6, 12, 16, 16), num_steps=4, seed=9)
    assert torch.isfinite(z).all()
    assert float(z.std()) > 0.0
