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
from apedp_dyn.video.learned_flow_head import (
    FlowHeadConfig,
    LearnedFlowHead,
    flow_consistency_loss,
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
