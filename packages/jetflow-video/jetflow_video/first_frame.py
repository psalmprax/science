r"""Real-First-Frame Conditioning (Anchored Flow Matching).

Status: experimental

Photorealism lever #1: condition generation on a REAL first frame. A real
anchor frame injects true texture/lighting/parallax in pixel space; the
model only learns to predict motion past it.

Training:
    Given clean latents z_data (B, T, C, H, W), sample t ~ U(0.05, 0.95),
    corrupt non-anchor frames  z_t = (1 - t) * z_data + t * z_noise, but keep
    frame 0 EXACTLY CLEAN (the anchor). The flow-matching target remains
    v* = z_noise - z_data; the loss EXCLUDES the anchor frame's prediction
    (predicting noise displacement for a clean input is vacuous) so no signal
    leaks that would teach the network to expect clean non-anchor frames.

Generation:
    Integrate the ODE from noise to data in the direction CONSISTENT with
    the training parameterization (t: 1 -> 0, dz/dt = v), and re-impose the
    clean anchor latent on frame 0 after every step (inpainting-style
    conditioning, RePaint-alike). Frame 0 is therefore pixel-exact by
    construction; everything after it is learned motion.

NOTE on the package sampler convention: ``model.generate_video`` used to
integrate t: 0 -> 1 from pure noise, contradicting
``trainer.compute_loss`` (z_t = (1 - t) * z_data + t * z_noise, i.e.
t = 0 is DATA). This was fixed in place: ``generate_video`` now follows
the TRAINER convention (noise at t = 1, integrate DOWN to t = 0 with a
negative dt) exactly like the sampler in this module, so sampling
matches what the network is trained to predict.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def sample_anchored_z_t(
    z_data: torch.Tensor,
    t: float,
    noise: torch.Tensor,
    keep_anchor_clean: bool = True,
) -> torch.Tensor:
    """Corrupt latents for flow-matching while pinning frame 0 clean.

    z_data, noise: (B, T, C, H, W). Returns z_t. t in [0, 1]: 0 = data, 1 = noise.
    """
    if z_data.shape != noise.shape:
        raise ValueError(f"Shape mismatch {z_data.shape} vs {noise.shape}")
    z_t = (1.0 - t) * z_data + t * noise
    if keep_anchor_clean:
        # Anchor: frame 0 stays at the real data latent (no corruption)
        z_t = z_t.clone()
        z_t[:, 0] = z_data[:, 0]
    return z_t


def anchored_flow_matching_loss(
    model,
    z_data: torch.Tensor,
    prompt_tokens: torch.Tensor,
    guidance_scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Jet2-free (plain flow matching) anchored loss over non-anchor frames.

    Returns dict with flow_loss (scalar tensor) and t sampled. Excludes the
    anchor frame from both input corruption and the loss target.
    """
    if z_data.ndim != 5 or z_data.shape[0] != 1:
        raise ValueError(f"Expected (1, T, C, H, W) latents (b=1 loop pattern), got {tuple(z_data.shape)}")
    t_dim, c, h, w = z_data.shape[1], z_data.shape[2], z_data.shape[3], z_data.shape[4]
    if t_dim < 2:
        raise ValueError("Need at least 2 frames: an anchor + at least one frame to animate")

    noise = torch.randn_like(z_data)
    t = float(torch.empty(1).uniform_(0.05, 0.95).item())
    z_t = sample_anchored_z_t(z_data, t, noise, keep_anchor_clean=True)
    target_v = noise - z_data  # (1, T, C, H, W)

    text = prompt_tokens
    if text.ndim == 1:
        text = text.view(1, 1, -1)
    elif text.ndim == 2:
        text = text.view(1, text.shape[0], -1)
    text = text.to(device=z_data.device, dtype=z_data.dtype)

    pred_v = model(z_t, t, text_tokens=text)
    if pred_v.shape != target_v.shape:
        raise ValueError(f"Model returned velocity of shape {tuple(pred_v.shape)}, expected {tuple(target_v.shape)}")

    # Non-anchor frames only
    flow_loss = F.mse_loss(pred_v[:, 1:], target_v[:, 1:])
    return {"flow_loss": flow_loss, "t": t, "pred_norm": float(pred_v[:, 1:].norm().item())}


@torch.no_grad()
def generate_anchored_video(
    model,
    prompt_embedding: torch.Tensor,
    anchor_latent: torch.Tensor,
    residual_frames: int,
    num_steps: int = 8,
    guidance_scale: float = 2.0,
    trace_seed: int = 0,
    return_trajectory: bool = False,
):
    """Generate a video latent anchored on a REAL first frame.

    anchor_latent: (C, H, W) clean data latent of the real frame.
    Returns (z_final, trajectory_frames|None), z_final shape (T, C, H, W) with
    frame 0 pixel-identical (in latent space) to the anchor at every step.
    """
    device = anchor_latent.device
    c, h, w = anchor_latent.shape

    gen = torch.Generator(device=device).manual_seed(trace_seed)
    z = torch.randn((1, residual_frames, c, h, w), generator=gen, device=device, dtype=anchor_latent.dtype)
    anchor = anchor_latent.view(1, 1, c, h, w)  # (1, 1, C, H, W)

    text_tokens = prompt_embedding
    if text_tokens.ndim == 1:
        text_tokens = text_tokens.view(1, 1, -1).to(device=device, dtype=anchor_latent.dtype)
    elif text_tokens.ndim == 2:
        text_tokens = text_tokens.view(1, text_tokens.shape[0], -1).to(device=device, dtype=anchor_latent.dtype)
    else:
        text_tokens = text_tokens.to(device=device, dtype=anchor_latent.dtype)

    z_final = torch.cat([anchor, z], dim=1)  # (1, T, C, H, W)

    trajectory = []
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        t_curr = float(ts[i])
        t_next = float(ts[i + 1])
        dt = t_next - t_curr  # negative: noise (t=1) -> data (t=0)
        v = model(z_final, t_curr, text_tokens=text_tokens)
        if guidance_scale and guidance_scale > 1.0:
            v_uncond = model(z_final, t_curr, text_tokens=None)
            v = v_uncond + guidance_scale * (v - v_uncond)
        z_final = z_final + dt * v
        # Re-impose the anchor every step (strict conditioning)
        z_final[:, 0] = anchor[0]
        if return_trajectory:
            trajectory.append(z_final.clone())

    if return_trajectory:
        return z_final, trajectory
    return z_final
