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
    predict_gain: bool = False     # + per-channel (s, b) affine after the warp


class LearnedFlowHead(nn.Module):
    """Predicts per-pixel velocity v(x)=(v_x, v_y) from two context frames.

    Input : (B, 2C, H, W)  — stacked previous + current frames
    Output: (B, 2, H, W)   — (v_x, v_y), each in (-flow_scale, flow_scale)

    With ``cfg.predict_gain`` the network additionally emits a per-channel
    affine correction (s_c, b_c): the predicted next frame is
    ``s_c * warp(z_curr, v)_c + b_c``. This is the closed-form family of the
    fluid generator's per-step map — semi-Lagrangian advection, joint dye
    renormalization and the per-clip per-channel stretch collapse to exactly
    an affine post-warp map (the renorm max also involves an in-generator
    third dye channel that is NEVER stored in frames, so (s, b) must be
    inferred from pixels, not computed).
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
        out_ch = 2 + (2 * self.cfg.context_channels if self.cfg.predict_gain else 0)
        layers.append(nn.Conv2d(c, out_ch, 3, padding=1))
        self.net = nn.Sequential(*layers)
        if self.cfg.predict_gain:
            with torch.no_grad():
                last = self.net[-1]
                # s = softplus(raw_mean) starts at exactly 1.0, b at 0.0:
                # softplus^{-1}(1) = log(e - 1); the head thus starts as the
                # plain no-gain model and moves only with evidence.
                last.bias[2:2 + self.cfg.context_channels].fill_(0.5413248546129181)
                last.bias[2 + self.cfg.context_channels:].zero_()

    def predict_flow_and_gain(self, z_prev: torch.Tensor, z_curr: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (v, s, b): flow (B,2,H,W) and, when ``predict_gain``, the
        per-channel affine correction (B,C) applied after the warp."""
        x = torch.cat([z_prev, z_curr], dim=1)
        raw = self.net(x)
        v = self.cfg.flow_scale * torch.tanh(raw[:, :2])
        if not self.cfg.predict_gain:
            return v, None, None
        c = self.cfg.context_channels
        s = F.softplus(raw[:, 2:2 + c].mean(dim=(2, 3)))
        b = raw[:, 2 + c:2 + 2 * c].mean(dim=(2, 3))
        return v, s, b

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return self.predict_flow_and_gain(z_prev, z_curr)[0]


def warp_frames(z: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor,
                dt: float = 1.0, periodic: bool = False) -> torch.Tensor:
    """Backward warp z (B,C,H,W) by displacement -dt*(vx,vy).

    Matches DC-STT's `video_warp_2d` convention: sample z at the upstream
    position x - dt*v (semi-Lagrangian backtrace). ``vx`` must be (B,H,W) —
    broadcasting handles the H,W meshgrid; the caller is responsible for
    batch shape (see training/rollout). Used in training loss and rollout;
    sign-convention parity against the Rust-kernel NumPy fallback is pinned
    in tests/test_dyn_learned_flow_head.py.

    ``periodic=True``: wrap-around (toroidal) boundary conditions matching
    the fluid generator's `floor(q) % size` backtrace indexing — the source
    coordinate is floored first, the integer index taken modulo, frac =
    q - floor(q). ``periodic=False`` (default): grid_sample, border padding.
    Train and rollout boundary treatment must agree; the fluid benchmark
    sets both periodic.
    """
    b, _, h, w = z.shape
    if vx.shape != (b, h, w) or vy.shape != (b, h, w):
        raise ValueError(f"Expected vx/vy ({b}, {h}, {w}), got {vx.shape}/{vy.shape}")
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=z.dtype), torch.arange(w, dtype=z.dtype), indexing="ij"
    )
    if periodic:
        qx = xx - dt * vx
        qy = yy - dt * vy
        fx = torch.floor(qx)
        fy = torch.floor(qy)
        x0 = torch.remainder(fx, w).long()
        y0 = torch.remainder(fy, h).long()
        x1 = (x0 + 1) % w
        y1 = (y0 + 1) % h
        wx = (qx - fx).unsqueeze(1)
        wy = (qy - fy).unsqueeze(1)

        def _gather(yi: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            # Flattened spatial gather: (B,H,W) row/col indices -> H*W flat
            # index, expanded over channels. (Naive z[:, :, yi, xi] advanced
            # indexing pushes the index batch dim to the FRONT; two-stage
            # gathers along H then W re-index the remaining axis incorrectly.)
            z_flat = z.reshape(b, z.shape[1], h * w)
            flat = (yi * w + xi).reshape(b, 1, h * w).expand_as(z_flat)
            return torch.gather(z_flat, 2, flat).reshape(b, z.shape[1], h, w)

        top = _gather(y0, x0) * (1.0 - wx) + _gather(y0, x1) * wx
        bottom = _gather(y1, x0) * (1.0 - wx) + _gather(y1, x1) * wx
        return top * (1.0 - wy) + bottom * wy
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


def _advance(z_curr: torch.Tensor, v: torch.Tensor,
             s: torch.Tensor | None, b: torch.Tensor | None,
             dt: float, periodic: bool) -> torch.Tensor:
    """Predicted next frame: warp, then the head's per-channel affine (if any)."""
    z_pred = warp_frames(z_curr, v[:, 0], v[:, 1], dt=dt, periodic=periodic)
    if s is not None:
        z_pred = s.unsqueeze(-1).unsqueeze(-1) * z_pred + b.unsqueeze(-1).unsqueeze(-1)
    return z_pred


def flow_consistency_loss(model: LearnedFlowHead,
                          z_prev: torch.Tensor, z_curr: torch.Tensor,
                          z_next: torch.Tensor, dt: float = 1.0,
                          periodic: bool = False,
                          return_pred: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Self-supervised advection-consistency objective for one transition.

    All tensors (B,C,H,W); C should equal the head's context_channels.
    ``periodic`` selects the warp boundary treatment (see :func:`warp_frames`);
    it must match the transport config used at rollout. If the head was built
    with ``cfg.predict_gain`` its per-channel affine (s, b) is applied after
    the warp (see :meth:`LearnedFlowHead.predict_flow_and_gain`).
    """
    v, s, b = model.predict_flow_and_gain(z_prev, z_curr)
    z_pred = _advance(z_curr, v, s, b, dt, periodic)
    adv_loss = F.mse_loss(z_pred, z_next)
    div_sq = float(model.cfg.lambda_div) * torch.mean(_divergence(v) ** 2)
    tv = float(model.cfg.lambda_smooth) * (
        torch.mean(torch.abs(torch.diff(v, dim=-1)))
        + torch.mean(torch.abs(torch.diff(v, dim=-2)))
    )
    total = adv_loss + div_sq + tv
    if return_pred:
        return total, z_pred
    return total


def flow_supervision_loss(model: LearnedFlowHead,
                          z_prev: torch.Tensor, z_curr: torch.Tensor,
                          vx_teacher: torch.Tensor, vy_teacher: torch.Tensor,
                          z_next: torch.Tensor, dt: float = 1.0,
                          periodic: bool = False,
                          return_pred: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Teacher-supervised flow objective (Stage 3): consistency + flow MSA.

    Adds ``MSE(v_head, v_teacher)`` to :func:`flow_consistency_loss`. The
    teacher is the fluid-domain generator's own ANALYTIC field, made
    available for TRAIN clips only (``fluid_domain.make_fluid_clip(...,
    return_flow=True)``): the generator advects its dye backtracing with
    ``u = dpsi/dy`` (y-velocity) and ``v = -dpsi/dx`` (x-velocity), so the
    caller MUST pass ``vx_teacher = v_gen, vy_teacher = u_gen``. That swap is
    pinned in tests/test_dyn_learned_flow_head.py.

    The supervised term fills exactly the gap the 2x-budget rung exposed:
    a dense per-pixel target replaces the (floor-bound) single-advection
    residual self-supervision, so the gradient is not tied to the dye
    renormalization floor.
    """
    v, s, b = model.predict_flow_and_gain(z_prev, z_curr)
    z_pred = _advance(z_curr, v, s, b, dt, periodic)
    adv_loss = F.mse_loss(z_pred, z_next)
    div_sq = float(model.cfg.lambda_div) * torch.mean(_divergence(v) ** 2)
    tv = float(model.cfg.lambda_smooth) * (
        torch.mean(torch.abs(torch.diff(v, dim=-1)))
        + torch.mean(torch.abs(torch.diff(v, dim=-2)))
    )
    flow_mse = F.mse_loss(v[:, 0], vx_teacher) + F.mse_loss(v[:, 1], vy_teacher)
    total = adv_loss + div_sq + tv + flow_mse
    if return_pred:
        return total, z_pred
    return total


def train_flow_head(model: LearnedFlowHead,
                    clips: np.ndarray,
                    steps: int,
                    lr: float = 1e-3,
                    seed: int = 0,
                    progress_every: int = 0,
                    deadline_s: float | None = None,
                    dt: float = 1.0,
                    flow_targets: np.ndarray | None = None,
                    frames_per_clip: int | None = None,
                    periodic: bool = False,
                    rollout_steps: int = 1) -> list[float]:
    """Train a LearnedFlowHead self-supervised on (T,C,H,W) clips on CPU.

    ``clips`` can be a single clip or a stack of clips shape either
    (T,C,H,W) or a dict from ``make_domain_set``; dict values are stacked.
    Channels 1:3 (the advected scalars) are the head's domain. Optimizer
    AdamW, deterministic RandomState frame sampling, optional wall-clock
    ``deadline_s`` identical in spirit to the trainer deadline contract.
    Returns the per-step total loss (advection-consistency [+ flow MSE]).

    ``flow_targets`` (optional, Stage 3 supervision): (T, 2, H, W) teacher
    fields in the SAME concatenated clip order as the stacked ``clips``,
    channel 0 = vx (x-velocity), channel 1 = vy (y-velocity), px/frame.
    Built from ``make_fluid_clip(seed, return_flow=True)`` as
    ``np.stack([v_gen, u_gen])`` per frame (the swap pinned in tests).
    Train-clip ids only; never built from holdout clips.

    Teacher alignment: for the sampled triple (ti-1, ti, ti+1) the target is
    ``flow[ti+1]`` — the field the generator itself used to advect frame ti
    into ti+1 — matching both the advection-consistency loss and the rollout
    convention (head(z_{ti-1}, z_{ti}) advances z_ti). ``frames_per_clip``:
    when the stacked array concatenates equal-length clips, pass the per-clip
    frame count so sampled transitions never straddle clip boundaries (dict
    input derives it automatically). ``periodic``: toroidal warp boundary in
    the loss — must match the rollout transport's ``periodic_warp`` (the
    fluid domain is periodic; see the benchmark wiring).

    ``rollout_steps`` (exposure-bias training): when > 1 the model is trained
    on consecutive transitions it generates ITSELF — the per-step loss is the
    same objective, but each forward feeds the model's own previous prediction
    and gradients flow through the whole unrolled window (BPTT). Sampled
    windows never run past a clip end (ti + rollout_steps <= clip last index).
    """
    clip_lengths: list[int] | None = None
    if isinstance(clips, dict):
        vals = [np.asarray(v, np.float64) for v in clips.values()]
        clip_lengths = [int(v.shape[0]) for v in vals]
        arr = np.concatenate(vals, axis=0)
    else:
        arr = np.asarray(clips, np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Expected (T,C,H,W) clips, got {arr.shape}")
    # keep only the advected scalar channels for head training
    if arr.shape[1] >= 3:
        arr = arr[:, 1:3]
    data = torch.from_numpy(arr)

    flow = None
    if flow_targets is not None:
        flow = np.asarray(flow_targets, np.float64)
        if flow.shape != (arr.shape[0], 2, arr.shape[2], arr.shape[3]):
            raise ValueError(
                f"flow_targets {flow.shape} must match stacked clips (T,2,H,W); "
                "channel 0 = vx (x-velocity), channel 1 = vy (y-velocity)"
            )
        flow = torch.from_numpy(flow)

    k = int(rollout_steps)
    if k < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")

    model.double()   # head params default fp32; the domain data is float64
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    deadline = (time.perf_counter() + deadline_s) if deadline_s else None
    losses: list[float] = []
    t_total = data.shape[0]
    if clip_lengths is None and frames_per_clip is not None and int(frames_per_clip) >= k + 2:
        clip_lengths = [int(frames_per_clip)] * (t_total // int(frames_per_clip))
    valid_tis: list[int] = []
    if clip_lengths:
        start = 0
        for length in clip_lengths:
            if length >= k + 2:
                valid_tis.extend(range(start + 1, start + length - k))
            start += length
    if not valid_tis:
        valid_tis = list(range(1, t_total - k))
    if not valid_tis:
        raise ValueError(
            f"no training windows fit: need at least {k + 2} consecutive frames "
            f"(got t_total={t_total}, rollout_steps={k})"
        )
    valid_arr = np.asarray(valid_tis, dtype=np.int64)
    for step in range(steps):
        ti = int(valid_arr[rng.randint(0, valid_arr.shape[0])])
        z_prev = data[ti - 1: ti]
        z_curr = data[ti: ti + 1]
        opt.zero_grad(set_to_none=True)
        # Multi-step unroll: each step reuses the SAME per-step objective
        # (advection consistency [+ flow MSE]); the model's own prediction
        # becomes the next step's input, so drift over k steps is directly
        # penalized rather than only single-transition error.
        total: torch.Tensor | None = None
        for j in range(k):
            z_next = data[ti + 1 + j: ti + 2 + j]
            if flow is None:
                loss, z_pred = flow_consistency_loss(model, z_prev, z_curr, z_next,
                                                     dt=dt, periodic=periodic, return_pred=True)
            else:
                loss, z_pred = flow_supervision_loss(model, z_prev, z_curr,
                                                     flow[ti + 1 + j: ti + 2 + j, 0],
                                                     flow[ti + 1 + j: ti + 2 + j, 1],
                                                     z_next, dt=dt, periodic=periodic, return_pred=True)
            total = loss if total is None else total + loss
            if j + 1 < k:
                # advance with the graph-carrying prediction (computed in flow loss above)
                z_prev, z_curr = z_curr, z_pred
        total.backward()
        opt.step()
        losses.append(float(total.item()))
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
                       input_channels: tuple[int, ...] = (1, 2),
                       ch0_policy: str = "transport") -> np.ndarray:
    """Roll DC-STT forward from 2 context frames using head-predicted flow.

    ``context`` is (2,C,H,W). ``input_channels`` selects which context
    channels the HEAD sees (default (1, 2): the advected dye scalars it was
    trained on); ALL channels are transported unless ``ch0_policy`` says
    otherwise. Returns (horizon+2, C, H, W) covering the two context frames
    plus every transported frame, i.e. row i corresponds to clip column
    t = i (row 0/1 are the context frames). The transport config must use
    nu=0.0 so the rollout matches the pure-advection training dynamics
    (module docstring), and its ``periodic_warp`` must match the training
    loss boundary treatment.

    If the head was built with ``cfg.predict_gain``, its per-channel affine
    (s, b) is applied to the dye channels after every transport step — the
    rollout then executes the exact prediction map the loss trained.

    ``ch0_policy`` — how the vorticity channel (index 0) is advanced:
    - "transport" (legacy): advected by the head flow like the dye — wrong
      for this domain: ch0 = |curl(v)| is a DERIVED field, not an advected
      scalar; measured 9x worse than persistence on 1-step advection.
    - "persist": frozen at the last context frame (|curl| evolves smoothly;
      the strongest trivial policy on this channel).
    """
    if ch0_policy not in ("transport", "persist"):
        raise ValueError(f"ch0_policy must be 'transport' or 'persist', got {ch0_policy!r}")
    head.double()
    head.eval()
    ctx = np.asarray(context, dtype=np.float64)
    ch = list(input_channels)
    z_prev = torch.from_numpy(ctx[0][ch]).unsqueeze(0)
    z_curr = torch.from_numpy(ctx[1][ch]).unsqueeze(0)
    z_state = torch.from_numpy(ctx[1]).unsqueeze(0)
    frames = [ctx[0].copy(), ctx[1].copy()]
    for _ in range(horizon):
        v, s, b = head.predict_flow_and_gain(z_prev, z_curr)
        z_np, _ = transport.transport_step(z_state.squeeze(0).numpy(),
                                           v[0, 0].numpy(), v[0, 1].numpy(), dt=dt)
        z_state_new = torch.from_numpy(z_np).unsqueeze(0)
        if s is not None:
            z_state_new[:, ch] = (s[:, :, None, None] * z_state_new[:, ch]
                                  + b[:, :, None, None])
        if ch0_policy == "persist":
            z_state_new[:, 0] = z_state[:, 0]
        z_prev, z_curr = z_curr, z_state_new[:, ch]
        z_state = z_state_new
        frames.append(z_state.squeeze(0).numpy().copy())
    return np.stack(frames, axis=0)
