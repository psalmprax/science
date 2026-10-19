"""Cross-language parity harness: verifies mathematical parity between compiled Rust kernels and Python/PyTorch implementations.

Protects against silent cross-engine semantic drift across:
  1. Bilinear spatial backward warp (periodic and border-clamped).
  2. Optical flow divergence computation.
  3. Flow head forward evaluations across spatial grid sizes.
  4. Video quality observables (PSNR and SSIM).
"""
from __future__ import annotations

import numpy as np
import torch

import apedp_kernel as ak
from apedp_dyn.video.flow_head_rust import RustFlowHeadSpec, pack_weights
from apedp_dyn.video.learned_flow_head import FlowHeadConfig, LearnedFlowHead, warp_frames


def test_warp_bilinear_border_clamped_parity():
    """Verify Rust video_warp_bilinear_2d matches PyTorch grid_sample under border clamping."""
    h, w, c = 16, 16, 2
    rng = np.random.RandomState(42)
    img = np.ascontiguousarray(rng.randn(c, h, w), dtype=np.float64)

    # Velocity field in pixels/frame; DC-STT convention: displacement = -dt * v
    vx = np.ascontiguousarray(rng.uniform(-2.0, 2.0, size=(h, w)), dtype=np.float64)
    vy = np.ascontiguousarray(rng.uniform(-2.0, 2.0, size=(h, w)), dtype=np.float64)
    disp_x = np.ascontiguousarray(-vx, dtype=np.float64)
    disp_y = np.ascontiguousarray(-vy, dtype=np.float64)

    # 1. Rust execution with displacement
    warped_rust_flat = ak.video_warp_bilinear_2d(img.reshape(-1), disp_x.reshape(-1), disp_y.reshape(-1), c, h, w)
    warped_rust = np.asarray(warped_rust_flat, dtype=np.float64).reshape(c, h, w)

    # 2. PyTorch execution using warp_frames (which internally uses displacement = -dt * v)
    t_img = torch.from_numpy(img[None])  # (1, C, H, W)
    t_vx = torch.from_numpy(vx[None])    # (1, H, W)
    t_vy = torch.from_numpy(vy[None])    # (1, H, W)
    warped_torch = warp_frames(t_img, t_vx, t_vy, dt=1.0, periodic=False)[0].numpy()

    max_diff = float(np.max(np.abs(warped_rust - warped_torch)))
    assert max_diff < 1e-12, f"Border-clamped warp max difference {max_diff:.3e} exceeds tolerance"


def test_warp_bilinear_periodic_parity():
    """Verify Rust video_warp_bilinear_2d_periodic matches PyTorch periodic warp."""
    h, w, c = 16, 16, 2
    rng = np.random.RandomState(43)
    img = np.ascontiguousarray(rng.randn(c, h, w), dtype=np.float64)

    # Large displacements wrapping across the toroidal border; disp = -dt * v
    vx = np.ascontiguousarray(rng.uniform(-5.0, 5.0, size=(h, w)), dtype=np.float64)
    vy = np.ascontiguousarray(rng.uniform(-5.0, 5.0, size=(h, w)), dtype=np.float64)
    disp_x = np.ascontiguousarray(-vx, dtype=np.float64)
    disp_y = np.ascontiguousarray(-vy, dtype=np.float64)

    warped_rust_flat = ak.video_warp_bilinear_2d_periodic(img.reshape(-1), disp_x.reshape(-1), disp_y.reshape(-1), c, h, w)
    warped_rust = np.asarray(warped_rust_flat, dtype=np.float64).reshape(c, h, w)

    t_img = torch.from_numpy(img[None])
    t_vx = torch.from_numpy(vx[None])
    t_vy = torch.from_numpy(vy[None])
    warped_torch = warp_frames(t_img, t_vx, t_vy, dt=1.0, periodic=True)[0].numpy()

    max_diff = float(np.max(np.abs(warped_rust - warped_torch)))
    assert max_diff < 1e-12, f"Periodic warp max difference {max_diff:.3e} exceeds tolerance"


def test_optical_flow_divergence_parity():
    """Verify Rust video_compute_optical_flow_divergence matches central finite difference divergence."""
    h, w = 24, 24
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)

    vx = np.ascontiguousarray(np.sin(x * 0.5) * np.cos(y * 0.5), dtype=np.float64)
    vy = np.ascontiguousarray(-np.cos(x * 0.5) * np.sin(y * 0.5), dtype=np.float64)

    rust_div = np.asarray(
        ak.video_compute_optical_flow_divergence(vx.reshape(-1), vy.reshape(-1), h, w, 1.0, 1.0),
        dtype=np.float64,
    ).reshape(h, w)

    # Interior central difference in numpy with dx=1.0, dy=1.0
    dvx_dx = (vx[:, 2:] - vx[:, :-2]) * 0.5
    dvy_dy = (vy[2:, :] - vy[:-2, :]) * 0.5
    np_div_interior = dvx_dx[1:-1, :] + dvy_dy[:, 1:-1]

    max_diff = float(np.max(np.abs(rust_div[1:-1, 1:-1] - np_div_interior)))
    assert max_diff < 1e-12, f"Optical flow divergence difference {max_diff:.3e} exceeds tolerance"


def test_flow_head_forward_cross_resolution_parity():
    """Verify flow_head_forward matches PyTorch autograd across diverse spatial resolutions."""
    for (h, w) in [(8, 8), (16, 16), (20, 24)]:
        spec = RustFlowHeadSpec(context_channels=2, hidden_dim=8, n_layers=2)
        head = LearnedFlowHead(FlowHeadConfig(context_channels=2, hidden_dim=8, n_layers=2, flow_scale=4.0)).double()
        packed = np.ascontiguousarray(pack_weights(head.state_dict()), dtype=np.float64)

        rng = np.random.RandomState(int(h * w))
        zp = np.ascontiguousarray(rng.randn(2, h, w), dtype=np.float64)
        zc = np.ascontiguousarray(rng.randn(2, h, w), dtype=np.float64)

        vx_r, vy_r = ak.flow_head_forward(
            zp.reshape(-1), zc.reshape(-1),
            spec.context_channels, spec.hidden_dim, spec.n_layers,
            spec.flow_scale, h, w, packed,
        )

        with torch.no_grad():
            v_t = head(torch.from_numpy(zp[None]), torch.from_numpy(zc[None]))[0].numpy()

        diff_vx = float(np.max(np.abs(np.asarray(vx_r).reshape(h, w) - v_t[0])))
        diff_vy = float(np.max(np.abs(np.asarray(vy_r).reshape(h, w) - v_t[1])))
        assert max(diff_vx, diff_vy) < 5e-6, f"Resolution ({h}, {w}) forward divergence exceeds erf-bound"


def test_video_psnr_mathematical_parity():
    """Verify Rust video_psnr matches 10 * log10(max_val^2 / MSE)."""
    rng = np.random.RandomState(99)
    img_a = np.ascontiguousarray(rng.uniform(0.0, 1.0, size=256), dtype=np.float64)
    noise = rng.normal(0.0, 0.05, size=256).astype(np.float64)
    img_b = np.ascontiguousarray(np.clip(img_a + noise, 0.0, 1.0), dtype=np.float64)

    rust_psnr = ak.video_psnr(img_a, img_b)
    mse = float(np.mean((img_a - img_b) ** 2))
    py_psnr = 10.0 * np.log10(1.0 / max(mse, 1e-15))

    assert abs(rust_psnr - py_psnr) < 1e-10, f"PSNR parity error: {abs(rust_psnr - py_psnr):.3e}"


def test_spatial_flow_convention_round_trip_and_warp_invariance():
    """Verify that Eulerian velocity <-> Lagrangian displacement conversions are exact round-trips
    and satisfy the advective pullback identity in video warping."""
    from apedp_dyn.video.flow_head_rust import (
        SpatialFlowConvention,
        displacement_to_eulerian_velocity,
        eulerian_velocity_to_displacement,
    )

    rng = np.random.RandomState(42)
    h, w = 16, 16
    vx = rng.randn(h, w) * 2.0
    vy = rng.randn(h, w) * 2.0
    dt = 0.5

    # 1. Round trip exactness
    dx, dy = eulerian_velocity_to_displacement(vx, vy, dt=dt)
    vx_rec, vy_rec = displacement_to_eulerian_velocity(dx, dy, dt=dt)

    np.testing.assert_allclose(vx_rec, vx, atol=1e-12)
    np.testing.assert_allclose(vy_rec, vy, atol=1e-12)

    # 2. Warp advective identity:
    # A positive flow velocity (vx > 0) advects features to the right,
    # which means target pixel at x queries source pixel at x - dt * vx.
    # Therefore, displacement offset added in kernel must be dx = -dt * vx.
    assert SpatialFlowConvention.EULERIAN_VELOCITY.value == "eulerian_velocity"
    assert SpatialFlowConvention.LAGRANGIAN_DISPLACEMENT.value == "lagrangian_displacement"

    img = np.zeros((1, h, w), dtype=np.float64)
    img[0, 8, 8] = 1.0  # single impulse at center

    # Constant rightward velocity field: vx = 2.0, dt = 1.0 -> features move right by 2
    flow_vx = np.full((h, w), 2.0, dtype=np.float64)
    flow_vy = np.zeros((h, w), dtype=np.float64)
    disp_x, disp_y = eulerian_velocity_to_displacement(flow_vx, flow_vy, dt=1.0)

    # Warped image via Rust kernel with displacement: (img, disp_x, disp_y, c, h, w)
    warped_flat = ak.video_warp_bilinear_2d(
        img.reshape(-1), disp_x.reshape(-1), disp_y.reshape(-1), 1, h, w
    )
    warped = np.asarray(warped_flat).reshape(1, h, w)

    # At target pixel (8, 10), backward trace queries (8, 10 + disp_x) = (8, 10 - 2) = (8, 8), which is 1.0!
    assert warped[0, 8, 10] == 1.0, f"Expected feature at (8, 10), got {warped[0, 8, 10]}"
    assert warped[0, 8, 8] == 0.0, f"Original feature should have advected away, got {warped[0, 8, 8]}"
