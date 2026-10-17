r"""Compact Diffusion Video Baseline for matched-compute comparison.

Status: experimental (benchmark-harness companion — audit-only weighting)

A deliberately standard, dependency-free alternative to the JetFlow-Video
stack for the matched-compute benchmark (``run_jetflow_vs_diffusion_matched.py``):
a spatio-temporal FACTORIZED flow-matching UNet — per-frame residual UNet at
fullname resolution plus one cheap temporal self-attention layer — the
off-the-shelf compact video-diffusion recipe, without the TT cores, jet
stepper, Noether projector, or physical steering of the main stack.

Parameterization conventions match ``distill.py`` EXACTLY so the comparison
uses one shared objective:
  z_t = (1 - t) * z_data + t * z_noise,   target v* = z_noise - z_data,
sampling integrates t: 1 -> 0 with z <- z + dt * v.

The parameter count is NOT free to drift: ``pick_matched_config`` searches a
width/block grid for the configuration whose trainable parameter count is
closest to the JetFlow model it is being compared against, and the runner
records the exact counts of both.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionVideoConfig:
    """Hyperparameters of the compact factorized video UNet baseline."""
    latent_channels: int = 12
    frames: int = 6
    width: int = 40
    num_res_blocks: int = 4
    text_dim: int = 64
    fourier_frequencies: int = 8
    guidance_scale: float = 3.0
    levels: int = 1  # single spatial level (latents are tiny, 8x8)


class FourierTimeEmbedding(nn.Module):
    """Same frequency scaffold family as the JetFlow time embedding."""

    def __init__(self, hidden_dim: int, num_frequencies: int = 8) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = torch.exp(-torch.arange(num_frequencies, dtype=torch.float32)
                          * (math.log(10000.0) / max(1, num_frequencies - 1)))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: float | torch.Tensor, n: int) -> torch.Tensor:
        if isinstance(t, (float, int)):
            t_tensor = torch.tensor([float(t)], dtype=torch.float32)
        else:
            t_tensor = torch.as_tensor(t, dtype=torch.float32).view(-1, 1)
        angles = (t_tensor * self.freqs.unsqueeze(0)) * (2.0 * math.pi)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(fourier).repeat(n, 1)


class ResBlock(nn.Module):
    """Fullname residual conv block with FiLM (scale/shift) time+text conditioning."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.film = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.norm = nn.GroupNorm(min(8, width), width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        y = self.norm(x) * (1.0 + gamma.view(b, c, 1, 1)) + beta.view(b, c, 1, 1)
        y = F.silu(self.conv1(y))
        y = self.conv2(y)
        return x + y


class TemporalAttentionBlock(nn.Module):
    """Self-attention across the small frame axis (factorized-st video baseline)."""

    def __init__(self, width: int, frames: int, num_heads: int = 4) -> None:
        super().__init__()
        self.frames = frames
        self.film = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W), cond: (B*T, width) — frame-major per batch element.
        rows, c, h, w = x.shape
        b = rows // self.frames
        tokens = (
            x.reshape(b, self.frames, c, h, w)
            .permute(0, 3, 4, 1, 2)          # (B, H, W, T, C)
            .reshape(b * h * w, self.frames, c)
        )
        attn_out, _ = self.attn(tokens, tokens, tokens)   # (B*H*W, T, C)
        gamma, beta = self.film(cond.reshape(b, self.frames, -1)).chunk(2, dim=-1)
        g2 = (
            gamma.unsqueeze(2).unsqueeze(2)  # (B, T, 1, 1, C)
            .expand(b, self.frames, h, w, c)
            .reshape(b * h * w, self.frames, c)
        )
        out_tokens = tokens * (1.0 + g2) + (
            beta.unsqueeze(2).unsqueeze(2)  # (B, T, 1, 1, C)
            .expand(b, self.frames, h, w, c)
            .reshape(b * h * w, self.frames, c)
        )
        attn_proj = (
            attn_out.reshape(b, h, w, self.frames, c)
            .permute(0, 3, 4, 1, 2)          # (B, T, C, H, W)
            .reshape(rows, c, h, w)
        )
        out = (
            out_tokens.reshape(b, h, w, self.frames, c)
            .permute(0, 3, 4, 1, 2)          # (B, T, C, H, W)
            .reshape(rows, c, h, w)
        )
        # Near-identity at load: FiLM and the attention out-projection are both
        # zero-initialized (AdaLN-Zero discipline as in the JetFlow stack).
        return x + out + attn_proj


class DiffusionVideoUNet(nn.Module):
    """Compact per-frame UNet + temporal attention, flow-matching velocity model."""

    def __init__(self, config: DiffusionVideoConfig) -> None:
        super().__init__()
        self.config = config
        w = config.width
        self.time_embed = FourierTimeEmbedding(w, config.fourier_frequencies)
        self.text_proj = nn.Linear(config.text_dim, w)
        self.input_proj = nn.Conv2d(config.latent_channels, w, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(w) for _ in range(config.num_res_blocks)])
        self.temporal = TemporalAttentionBlock(w, config.frames)
        self.output_norm = nn.GroupNorm(min(8, w), w)
        self.output_proj = nn.Conv2d(w, config.latent_channels, 3, padding=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z: torch.Tensor,
        t: float | torch.Tensor,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Predict v_theta(z, t); z: (B, T, C, H, W) or (T, C, H, W), returns same rank."""
        orig_ndim = z.ndim
        if orig_ndim == 4:
            z = z.unsqueeze(0)
        b, t_frames, c, h, w_in = z.shape

        # Per-frame factorization: (B, T, C, H, W) -> (B*T, C, H, W)
        x = z.permute(0, 1, 2, 3, 4).reshape(b * t_frames, c, h, w_in)
        x = self.input_proj(x)

        t_emb = self.time_embed(t, t_frames)  # (b*T, w) — frame rows repeat the same t
        # text pooled per clip b then broadcast per frame
        if text_tokens is not None:
            tok_pooled = self.text_proj(text_tokens.mean(dim=1))  # (b, w)
            cond = t_emb + tok_pooled.repeat_interleave(t_frames, dim=0)
        else:
            cond = t_emb

        for block in self.blocks:
            x = block(x, cond)

        x = self.temporal(x, cond)
        x = self.output_norm(x)
        v = self.output_proj(x)  # (b*T, C, H, W)

        v = v.reshape(b, t_frames, c, h, w_in)
        return v.squeeze(0) if orig_ndim == 4 else v


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


_WIDTH_GRID = (32, 40, 48, 56, 64, 72, 80, 96, 112, 128)


def pick_matched_config(
    target_params: int,
    latent_channels: int = 12,
    frames: int = 6,
    blocks: int = 4,
    max_rel_gap: float = 0.15,
) -> DiffusionVideoConfig:
    """Search the width grid for the config closest to ``target_params``.

    Returns the nearest match and raises if none is within the tolerance band —
    the matched-budget claim degrades silently otherwise, so failing loudly is
    the honest choice.
    """
    best: tuple[float, DiffusionVideoConfig] | None = None
    for w in _WIDTH_GRID:
        cfg = DiffusionVideoConfig(latent_channels=latent_channels, frames=frames,
                                   width=w, num_res_blocks=blocks)
        n = count_parameters(DiffusionVideoUNet(cfg))
        gap = abs(n - target_params) / max(1, target_params)
        if best is None or gap < best[0]:
            best = (gap, cfg)
    assert best is not None
    gap, cfg = best
    if gap > max_rel_gap:
        raise ValueError(
            f"No baseline config within {max_rel_gap:.0%} of target {target_params} "
            f"(closest {count_parameters(DiffusionVideoUNet(cfg))} at width {cfg.width})")
    return cfg


FLOW_LOSS_KEY = "flow_loss"


def flow_matching_loss_baseline(
    model: DiffusionVideoUNet,
    z_data: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Same flow-matching objective as ``distill.flow_matching_loss`` (b = 1)."""
    if z_data.ndim != 5 or z_data.shape[0] != 1:
        raise ValueError(f"Expected (1, T, C, H, W) latents, got {tuple(z_data.shape)}")
    noise = torch.randn_like(z_data)
    t = float(torch.empty(1).uniform_(0.05, 0.95).item())
    z_t = (1.0 - t) * z_data + t * noise
    target_v = noise - z_data
    pred_v = model(z_t, t, text_tokens=prompt_tokens)
    return {FLOW_LOSS_KEY: F.mse_loss(pred_v, target_v), "t": t}


def train_flow_model_baseline(
    model: DiffusionVideoUNet,
    clips_latents: torch.Tensor,
    steps: int,
    lr: float = 1e-3,
    seed: int = 0,
    progress_every: int = 0,
    deadline_s: float | None = None,
) -> list[float]:
    """Identical optimization protocol as ``distill.train_flow_model`` (b=1 loops).

    ``deadline_s`` (optional) stops the loop at a wall-clock deadline so a
    caller can enforce an equal-time training budget regardless of any
    pre-calibrated step-count estimate (probe drift protection).
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    n = clips_latents.shape[0]
    dev = clips_latents.device
    dtype = clips_latents.dtype
    losses: list[float] = []
    deadline = (time.perf_counter() + deadline_s) if deadline_s else None
    for step in range(steps):
        idx = int(rng.randint(0, n))
        z = clips_latents[idx: idx + 1].to(dev)
        text = torch.randn(1, 2, 64, device=z.device, dtype=dtype)
        opt.zero_grad(set_to_none=True)
        d = flow_matching_loss_baseline(model, z, text)
        d[FLOW_LOSS_KEY].backward()
        opt.step()
        losses.append(float(d[FLOW_LOSS_KEY].item()))
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if progress_every and (step + 1) % progress_every == 0:
            print(f"[train_flow_model_baseline] step {step + 1}/{steps} loss={losses[-1]:.4f}")
    return losses


@torch.no_grad()
def euler_sample_baseline(
    model: DiffusionVideoUNet,
    prompt_tokens: torch.Tensor,
    shape: tuple[int, int, int, int],
    num_steps: int,
    seed: int = 0,
    guidance_scale: float | None = None,
    dtype: torch.dtype = torch.float64,
    trajectory: bool = False,
):
    """Standard Euler flow-matching sampler: identical noise draw + convention
    as ``distill.euler_sample_reverse`` so the comparison is apples-to-apples."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(shape, generator=gen, device=dev, dtype=dtype)
    if prompt_tokens.ndim == 1:
        text = prompt_tokens.view(1, 1, -1).to(device=dev, dtype=z.dtype)
    elif prompt_tokens.ndim == 2:
        text = prompt_tokens.view(1, prompt_tokens.shape[0], -1).to(device=dev, dtype=z.dtype)
    else:
        text = prompt_tokens.to(device=dev, dtype=z.dtype)
    s = guidance_scale if guidance_scale is not None else model.config.guidance_scale
    traj: list[torch.Tensor] = []
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        t_curr = float(ts[i])
        dt = float(ts[i + 1]) - float(ts[i])  # negative: toward data
        v = model(z, t_curr, text_tokens=text)
        if s > 1.0:
            v_u = model(z, t_curr, text_tokens=None)
            v = v_u + s * (v - v_u)
        z = z + dt * v
        if trajectory:
            traj.append(z.clone())
    return (z, traj) if trajectory else z
