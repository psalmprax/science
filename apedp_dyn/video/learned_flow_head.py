r"""Learned Flow Head for DC-STT: a small CNN estimating the advection
velocity field v(x) from two consecutive frames.

Status: experimental

Purpose (why this module exists): the DC-STT transport engine
(`apedp_dyn/video/decoupled_transport.py`) executes the continuous transport
PDE given an analytic, hand-coded per-scene flow field. That is a
reconstruction-class capability. To transport REAL content — frames whose
velocity field is unknown, not planted — the pipeline needs a component that
*infers* v(x) from pixel evidence. This module is that component, trained
ENTIRELY ON CPU, using the same wall-clock deadline discipline as the
matched-compute benchmarks:

    prediction: v_t(x) = head(z_{t-1}, z_t)
    self-supervised target: advect(z_t, -v_t) ≈ z_{t+1}
    loss = advection-consistency MSE + lambda_div * div(v)^2         + lambda_smooth * TV(v)

Honest scope (stated, not hidden):
- The current training domain is the repo's synthetic fluid clips
  (`fluid_domain.make_domain_set`) — "real content" here means frames whose
  flow is NOT provided to the transport engine (the head must infer it from
  two frames and generalize across random vortex configurations); it does NOT
  mean photoreal foilage/faces etc. Photorealism remains unclaimed.
- The 2 advected scalar channels (indices 1:3) are the training target; dye
  renormalization in the clip generator makes the mapping approximate, so a
  nonzero floor on advection-consistency MSE is expected and reported.
- Inference convention: training warp and DC-STT rollout both define z_{t+1}
  = Warp(z_t, -dt*v_t) with bilateral-equivalent `grid_sample` semantics; the
  rollout config uses nu=0.0 (pure advection, no viscosity) so the learned
  dynamics match the trained ones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowHeadConfig:
    """Small-CNN flow estimator configuration (kept tiny for CPU budgets)."""

    context_channels: int = 2      # advected scalar channels fed to the head
    hidden_dim: int = 24
    n_layers: int = 3
    flow_scale: float = 4.0        # tanh output scale bound (px/frame)
    lambda_div: float = 0.10       # soft incompressibility penalty
    lambda_smooth: float = 0.01    # TV smoothness penalty on v


class LearnedFlowHead(nn.Module):
    """Predicts per-pixel velocity v(x)=(v_x, v_y) from two context frames.

    Input : (B, 2C, H, W)  — stacked previous + current frames
    Output: (B, 2, H, W)   — (v_x, v_y), each in (-flow_scale, flow_scale)
    """

    def __init__(self, cfg: FlowHeadConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FlowHeadConfig()
        c_in = self.cfg.context_channels * 2
        layers: list[nn.Module] = []
        c = c_in
        for _ in range(self.cfg.n_layers):
            layers.append(nn.Conv2d(c, self.cfg.hidden_dim, 3, padding=1))
            layers.append(nn.GELU())
            c = self.cfg.hidden_dim
        layers.append(nn.Conv2d(c, 2, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_prev, z_curr], dim=1)
        raw = self.net(x)
        return self.cfg.flow_scale * torch.tanh(raw)


def warp_frames(z: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor,
                dt: float = 1.0) -> torch.Tensor:
    """Backward warp z (B,C,H,W) by displacement -dt*(vx,vy) via grid_sample.

    Matches DC-STT's `video_warp_2d` convention: sample z at the upstream
    position x - dt*v (semi-Lagrangian backtrace), border padding. ``vx``
    must be (B,H,W) — broadcasting handles the H,W meshgrid; the caller is
    responsible for batch shape (see training/rollout). Used in training
    loss and rollout; sign-convention parity against the Rust-kernel NumPy
    fallback is pinned in tests/test_dyn_learned_flow_head.py.
    """
    b, _, h, w = z.shape
    if vx.shape != (b, h, w) or vy.shape != (b, h, w):
        raise ValueError(f"Expected vx/vy ({b}, {h}, {w}), got {vx.shape}/{vy.shape}")
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=z.dtype), torch.arange(w, dtype=z.dtype), indexing="ij"
    )
    # pixel-center convention (align_corners=False): pixel i maps to
    # (2i+1)/size - 1; equivalent to the bridge fallback's sx = x + disp with
    # bilinear floor weights and border clamping (parity pinned in tests).
    nx = 2.0 * (xx + 0.5 - dt * vx) / w - 1.0
    ny = 2.0 * (yy + 0.5 - dt * vy) / h - 1.0
    grid = torch.stack([nx, ny], dim=-1)
    return F.grid_sample(z, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _divergence(v: torch.Tensor) -> torch.Tensor:
    """Central-difference div v for (B,2,H,W) on interior cells: d(vx)/dx + d(vy)/dy."""
    dvx_dx = (v[:, 0, :, 2:] - v[:, 0, :, :-2]) * 0.5   # (B, H,   W-2) centered at w
    dvy_dy = (v[:, 1, 2:, :] - v[:, 1, :-2, :]) * 0.5   # (B, H-2, W)   centered at h
    return dvx_dx[:, 1:-1, :] + dvy_dy[:, :, 1:-1]      # (B, H-2, W-2), aligned


def flow_consistency_loss(model: LearnedFlowHead,
                          z_prev: torch.Tensor, z_curr: torch.Tensor,
                          z_next: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Self-supervised advection-consistency objective for one transition.

    All tensors (B,C,H,W); C should equal the head's context_channels.
    """
    v = model(z_prev, z_curr)
    z_pred = warp_frames(z_curr, v[:, 0], v[:, 1], dt=dt)
    adv_loss = F.mse_loss(z_pred, z_next)
    div_sq = float(model.cfg.lambda_div) * torch.mean(_divergence(v) ** 2)
    tv = float(model.cfg.lambda_smooth) * (
        torch.mean(torch.abs(torch.diff(v, dim=-1)))
        + torch.mean(torch.abs(torch.diff(v, dim=-2)))
    )
    return adv_loss + div_sq + tv


def train_flow_head(model: LearnedFlowHead,
                    clips: np.ndarray,
                    steps: int,
                    lr: float = 1e-3,
                    seed: int = 0,
                    progress_every: int = 0,
                    deadline_s: float | None = None,
                    dt: float = 1.0) -> list[float]:
    """Train a LearnedFlowHead self-supervised on (T,C,H,W) clips on CPU.

    ``clips`` cent be a single clip or a stack of clips shape either
    (T,C,H,W) or a dict from ``make_domain_set``; dict values are stacked.
    Channels 1:3 (the advected scalars) are the head's domain. Optimizer
    AdamW, deterministic RandomState frame sampling, optional wall-clock
    ``deadline_s`` identical in spirit to the trainer deadline contract.
    Returns the per-step advection-consistency loss.
    """
    if isinstance(clips, dict):
        arr = np.concatenate([np.asarray(v, np.float64) for v in clips.values()], axis=0)
    else:
        arr = np.asarray(clips, np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Expected (T,C,H,W) clips, got {arr.shape}")
    # keep only the advected scalar channels for head training
    if arr.shape[1] >= 3:
        arr = arr[:, 1:3]
    data = torch.from_numpy(arr)

    model.double()   # head params default fp32; the domain data is float64
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    deadline = (time.perf_counter() + deadline_s) if deadline_s else None
    losses: list[float] = []
    t_total = data.shape[0]
    for step in range(steps):
        ti = int(rng.randint(1, t_total - 1))
        z_prev = data[ti - 1: ti]
        z_curr = data[ti: ti + 1]
        z_next = data[ti + 1: ti + 2]
        opt.zero_grad(set_to_none=True)
        loss = flow_consistency_loss(model, z_prev, z_curr, z_next, dt=dt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if progress_every and (step + 1) % progress_every == 0:
            print(f"[learned_flow_head] step {step + 1}/{steps} loss={losses[-1]:.6f}")
    return losses


@torch.no_grad()
def rollout_dcstt_head(head: LearnedFlowHead,
                       transport,          # DecoupledSpatioTemporalTransport
                       context: np.ndarray,
                       horizon: int,
                       dt: float = 1.0,
                       input_channels: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Roll DC-STT forward from 2 context frames using head-predicted flow.

    ``context`` is (2,C,H,W). ``input_channels`` selects which context
    channels the HEAD sees (default (1, 2): the advected dye scalars it was
    trained on); ALL channels are transported. Returns (horizon+2, C, H, W)
    covering the two context frames plus every transported frame, i.e. row
    i corresponds to clip column t = i (row 0/1 are the context frames).
    The transport config must use nu=0.0 so the rollout matches the
    pure-advection training dynamics (module docstring).
    """
    head.double()
    head.eval()
    ctx = np.asarray(context, dtype=np.float64)
    z_prev = torch.from_numpy(ctx[0][list(input_channels)]).unsqueeze(0)
    z_curr = torch.from_numpy(ctx[1][list(input_channels)]).unsqueeze(0)
    z_state = torch.from_numpy(ctx[1]).unsqueeze(0)
    frames = [ctx[0].copy(), ctx[1].copy()]
    for _ in range(horizon):
        v = head(z_prev, z_curr)
        z_np, _ = transport.transport_step(z_state.squeeze(0).numpy(),
                                           v[0, 0].numpy(), v[0, 1].numpy(), dt=dt)
        z_state_new = torch.from_numpy(z_np).unsqueeze(0)
        z_prev, z_curr = z_curr, z_state_new[:, list(input_channels)]
        z_state = z_state_new
        frames.append(z_state.squeeze(0).numpy().copy())
    return np.stack(frames, axis=0)
