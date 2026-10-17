"""Unit tests for the learned flow head (DC-STT inferred-flow transport).

Covers: warp backtrace convention (peak moves by +dv under vx>0, exact
energy conservation), exact parity with the Rust-bridge NumPy fallback warp
(sign convention), deadline-bounded training, direct-vs-rollout consistency
at zero horizon, and the no-train-python inversion: for a clip frozen in
time (z1 == z2), the self-supervised optimum is v = 0 and the rollout must
reproduce the context frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apedp.rust_bridge import RustKernelBridge
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport,
    DecoupledTransportConfig,
)
from apedp_dyn.video.fluid_domain import make_fluid_clip
from apedp_dyn.video.learned_flow_head import (
    FlowHeadConfig,
    LearnedFlowHead,
    flow_consistency_loss,
    flow_supervision_loss,
    rollout_dcstt_head,
    train_flow_head,
    warp_frames,
)


def test_warp_backtrace_moves_peak_by_plus_v_with_exact_energy() -> None:
    z = torch.zeros(1, 2, 16, 16, dtype=torch.float64)
    z[0, 0, 8, 8] = 1.0
    vx = torch.full((1, 16, 16), 3.0, dtype=torch.float64)
    vy = torch.zeros(1, 16, 16, dtype=torch.float64)
    w = warp_frames(z, vx, vy, dt=1.0)
    row = w[0, 0, 8, :]
    # out(x) = in(x - dt*v): the delta lands at col 8 + 3 = 11
    assert int(row.argmax()) == 11
    assert abs(float(row.sum()) - 1.0) < 1e-9


def test_warp_matches_rust_bridge_numpy_fallback() -> None:
    znp = np.asarray(torch.randn(3, 16, 16, dtype=torch.float64))
    rng_vx, rng_vy = np.random.RandomState(1), np.random.RandomState(2)
    vx, vy = rng_vx.randn(16, 16), rng_vy.randn(16, 16)
    bridge = RustKernelBridge(force_fallback=True)
    # DC-STT convention: transport passes disp = -dt * v (dt=1 here)
    ref = bridge.video_warp_2d(znp, -vx, -vy)
    mine = warp_frames(
        torch.from_numpy(znp[None]),
        torch.from_numpy(vx).unsqueeze(0),
        torch.from_numpy(vy).unsqueeze(0),
    ).numpy()[0]
    assert float(np.max(np.abs(ref - mine))) < 1e-12


def test_train_flow_head_deadline_bounded() -> None:
    cfg_tiny = FlowHeadConfig(hidden_dim=4, n_layers=1, context_channels=2)
    head = LearnedFlowHead(cfg_tiny)
    clips = np.random.RandomState(0).randn(6, 2, 16, 16)
    t0 = __import__("time").perf_counter()
    losses = train_flow_head(head, clips, steps=100000, lr=1e-3, seed=0, deadline_s=1.0)
    dt_s = __import__("time").perf_counter() - t0
    assert 0 < len(losses) < 100000, "deadline did not bound the loop"
    assert dt_s < 15.0, f"deadline overrun beyond one-step bound: {dt_s:.1f}s"


def test_training_decreases_loss_on_frozen_pair() -> None:
    # two identical frames z1 == z2, third frame equal too: optimal v = 0
    frames = np.repeat(np.random.RandomState(3).randn(1, 2, 16, 16), 3, axis=0)
    head = LearnedFlowHead(FlowHeadConfig(hidden_dim=8, n_layers=2, flow_scale=4.0))
    losses = train_flow_head(head, frames, steps=60, lr=1e-3, seed=0)
    assert losses[-1] < losses[0]
    # final flow must be near zero (tanh head at optimum ~0)
    z = torch.from_numpy(frames[1:2])
    v = head(torch.from_numpy(frames[0:1]), z)
    assert float(v.abs().max()) < 0.5


def test_rollout_shape_and_frozen_consistency() -> None:
    head = LearnedFlowHead(FlowHeadConfig(hidden_dim=4, n_layers=1))
    head.double()
    solver = DecoupledSpatioTemporalTransport(
        DecoupledTransportConfig(use_rust=True, nu=0.0, enable_laplacian_dissipation=False)
    )
    ctx = np.random.RandomState(5).randn(2, 3, 16, 16)
    out = rollout_dcstt_head(head, solver, ctx, horizon=3)
    assert out.shape == (3 + 2, 3, 16, 16)
    # first two rollout rows are the context frames verbatim
    assert np.array_equal(out[0], ctx[0])
    assert np.array_equal(out[1], ctx[1])


def test_flow_consistency_loss_zero_for_perfect_constant_translation() -> None:
    """A uniform +3px/frame shift between frames can be advected exactly: the
    pure advection term of the loss must be (near) zero for the true v."""
    torch.manual_seed(0)
    z1 = torch.rand(1, 2, 32, 32, dtype=torch.float64)
    v = torch.zeros(1, 2, 32, 32, dtype=torch.float64)
    v[:, 0] = 3.0
    z2 = warp_frames(z1, v[:, 0], v[:, 1], dt=1.0)   # ground-truth next frame

    class _FixedHead(torch.nn.Module):
        def forward(self, a, b):
            return v

    model = LearnedFlowHead(FlowHeadConfig(hidden_dim=4, n_layers=1, lambda_smooth=0.0))
    model.forward = _FixedHead().forward.__get__(model)  # type: ignore[method-assign]
    loss = flow_consistency_loss(model, z1, z1, z2, dt=1.0)
    # smooth term off; advection term ~0; only tiny div-terms from finite diff at borders
    assert float(loss) < 0.05


def test_teacher_flow_matches_the_generator_semantics() -> None:
    """Stage 3 teacher convention, pinned hard: (vx, vy) = (v_gen, u_gen).

    The generator advects with back_y = y - u (u = dpsi/dy, the Y-velocity)
    and back_x = x - v (v = -dpsi/dx, the X-velocity); the head's warp uses
    warp_frames(z, vx, vy) with out(x) = in(x - v). So feeding warp the
    SWAPPED pair must advect the two dye channels like the generator did:
    next-frame channels 1:3 ~ warp(current channels, vx=v_gen, vy=u_gen).
    A silent (u, v)/xy swap would fail BOTH assertions.
    """
    clip, u_gen, v_gen = make_fluid_clip(4, t=4, h=32, w=32, return_flow=True)
    assert clip.shape == (4, 3, 32, 32)
    assert u_gen.shape == (4, 32, 32) == v_gen.shape
    # the analytic field is grid-divergence-free by construction (the SAME
    # mixed-difference stencil pinned in tests/test_dyn_fluid_domain.py)
    dux = (np.roll(u_gen[0], -1, axis=1) - np.roll(u_gen[0], 1, axis=1)) * 0.5
    dvy = (np.roll(v_gen[0], -1, axis=0) - np.roll(v_gen[0], 1, axis=0)) * 0.5
    assert float(np.abs(dux + dvy).max()) < 1e-10

    # advection agreement in head units (dt=1, no diffusion in this probe).
    # TEMPORAL alignment is pinned too: transition 1 -> 2 is generated with
    # flow[2] (frames[t+1] = advect(frames[t], flow[t+1])), so flow[2] must
    # beat the stale flow[1] — this is the index the trainer and rollout use.
    src = clip[1, 1:3]

    def _warp_at(k: int):
        return warp_frames(
            torch.from_numpy(src[None]),
            torch.from_numpy(v_gen[k]).unsqueeze(0),
            torch.from_numpy(u_gen[k]).unsqueeze(0),
            dt=1.0,
        ).numpy()[0]

    # the generator's own diffusion blur adds curvature; interior agreement
    # up to transport blur is the pin (border pixels wrapped by % h, excluded)
    d = np.abs(_warp_at(2)[0, 4:-4, 4:-4] - clip[2, 1, 4:-4, 4:-4]).max()
    d_prev = np.abs(_warp_at(1)[0, 4:-4, 4:-4] - clip[2, 1, 4:-4, 4:-4]).max()
    assert d < 0.1, f"swapped teacher warp disagrees with generator advection: {d}"
    assert d < d_prev, (
        f"temporal alignment wrong: flow[2] ({d:.4f}) must beat flow[1] ({d_prev:.4f})"
    )
    # and the UNSWAPPED pair must disagree much more (anti-alias of a swap)
    warped_sw = warp_frames(
        torch.from_numpy(src[None]),
        torch.from_numpy(u_gen[2]).unsqueeze(0),
        torch.from_numpy(v_gen[2]).unsqueeze(0),
        dt=1.0,
    ).numpy()[0]
    d_sw = np.abs(warped_sw[0, 4:-4, 4:-4] - clip[2, 1, 4:-4, 4:-4]).max()
    assert d_sw > d, "unswapped pair accidentally matches; the pin is vacuous"


def test_flow_supervision_beats_self_supervision_on_known_flow() -> None:
    """With a TRUE flow target available, the supervised objective must drive
    the head's flow MSE far below the self-supervised objective's ability to
    pin v within a fixed step budget (dense target vs single residual)."""
    torch.manual_seed(1)
    clip, u_gen, v_gen = make_fluid_clip(7, t=6, h=24, w=24, return_flow=True)
    flow = np.stack([v_gen, u_gen], axis=1)  # (T, 2, H, W) head-unit swap

    def _flow_mse(head):
        # pair (clip[0], clip[1]) is the ti=1 transition: teacher flow[2]
        z1 = torch.from_numpy(clip[0:1, 1:3])
        z2 = torch.from_numpy(clip[1:2, 1:3])
        v = head(z1, z2)
        vt = torch.from_numpy(flow[2:3])
        return float(((v - vt) ** 2).mean())

    cfg = FlowHeadConfig(hidden_dim=8, n_layers=2, lambda_div=0.0, lambda_smooth=0.0)
    hs = LearnedFlowHead(cfg).double()
    ls = train_flow_head(hs, clip, steps=80, lr=3e-3, seed=0,
                         flow_targets=flow)
    assert ls[-1] < ls[0], "supervised trainer did not descend"
    hm = LearnedFlowHead(cfg).double()
    lm = train_flow_head(hm, clip, steps=80, lr=3e-3, seed=0)
    assert _flow_mse(hs) < _flow_mse(hm), (
        f"supervised {_flow_mse(hs):.3e} must beat self-supervised {_flow_mse(hm):.3e} "
        "at equal budget when a true dense target exists"
    )


def test_train_flow_head_rejects_mismatched_flow_targets() -> None:
    head = LearnedFlowHead(FlowHeadConfig(hidden_dim=4, n_layers=1))
    clips = np.random.RandomState(0).randn(6, 2, 16, 16)
    bad = np.random.RandomState(1).randn(6, 2, 15, 16)
    import pytest
    with pytest.raises(ValueError, match="flow_targets"):
        train_flow_head(head, clips, steps=2, flow_targets=bad)
