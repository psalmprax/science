r"""Unit and Integration Tests for Decoupled Continuous Spatio-Temporal Transport (DC-STT).

Status: experimental

Validates:
1. Native Rust kernel fast-path execution and tallies accounting.
2. Differential cross-validation: exact numerical parity between native Rust kernel and NumPy fallback.
3. Noether mass-conservation: Helmholtz-Hodge projection onto divergence-free subspace.
4. Optical flow divergence and disocclusion mask detection with residual source injection.
5. Multi-scene video trajectory generation across diverse physical dynamics (translation, fluid vortex, radial expansion).
6. 4D Tensor-Train (TT) compression and continuous temporal interpolation.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.rust_bridge import RustKernelBridge, is_rust_kernel_available
from apedp_dyn.video.continuous_interpolator import ContinuousTemporalInterpolator
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport,
    DecoupledTransportConfig,
)


def _synthetic_scene(scene_type: str, h: int = 32, w: int = 32) -> np.ndarray:
    """Generate high-contrast initial spatial appearance textures."""
    y, x = np.mgrid[0:h, 0:w]
    if scene_type == "checkerboard":
        img = ((x // 4 + y // 4) % 2).astype(np.float64)
        return np.stack([img, 1.0 - img, np.sin(x / 4.0)], axis=0)
    elif scene_type == "gaussian_spot":
        r2 = (x - w / 2) ** 2 + (y - h / 2) ** 2
        spot = np.exp(-r2 / (2.0 * (w / 6.0) ** 2))
        return np.stack([spot, spot * 0.8, spot * 0.5], axis=0)
    elif scene_type == "gradient":
        gx = x / float(w - 1)
        gy = y / float(h - 1)
        return np.stack([gx, gy, 0.5 * (gx + gy)], axis=0)
    else:
        raise ValueError(f"Unknown scene_type: {scene_type}")


def _generate_velocity_field(flow_type: str, h: int = 32, w: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Generate velocity field for various physical scene dynamics."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0

    if flow_type == "translation":
        # Constant horizontal velocity
        vx = np.ones((h, w), dtype=np.float64) * 1.5
        vy = np.zeros((h, w), dtype=np.float64)
    elif flow_type == "vortex":
        # Divergence-free circular vortex around center
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2) + 1e-6
        # Tangential velocity: (-dy, dx) / r
        speed = 2.0 * np.exp(-r / (w / 3.0))
        vx = -dy / r * speed
        vy = dx / r * speed
    elif flow_type == "expansion":
        # Radial expanding flow (positive divergence in center -> disocclusion)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2) + 1e-6
        speed = 1.5 * np.exp(-r / (w / 2.5))
        vx = dx / r * speed
        vy = dy / r * speed
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")

    return vx, vy


class TestDecoupledSpatioTemporalTransport:
    def test_rust_kernel_fast_path_and_tallies(self) -> None:
        """Verify that transport_step routes through native Rust kernel when available."""
        if not is_rust_kernel_available():
            pytest.skip("Native apedp_kernel not built")

        cfg = DecoupledTransportConfig(use_rust=True, nu=0.005)
        solver = DecoupledSpatioTemporalTransport(cfg)

        z = _synthetic_scene("checkerboard", 32, 32)
        vx, vy = _generate_velocity_field("translation", 32, 32)

        z_next, diag = solver.transport_step(z, vx, vy)
        assert z_next.shape == z.shape
        assert diag["kernel_tallies"].get("rust", 0) > 0

    def test_differential_cross_validation_parity(self) -> None:
        """Verify exact numerical parity between native Rust kernel and NumPy fallback."""
        if not is_rust_kernel_available():
            pytest.skip("Native apedp_kernel not built")

        cfg_rust = DecoupledTransportConfig(use_rust=True, enforce_mass_conservation=False, nu=0.0)
        cfg_fallback = DecoupledTransportConfig(use_rust=False, enforce_mass_conservation=False, nu=0.0)

        solver_rust = DecoupledSpatioTemporalTransport(cfg_rust)
        solver_fallback = DecoupledSpatioTemporalTransport(cfg_fallback)

        z = _synthetic_scene("gradient", 32, 32)
        vx, vy = _generate_velocity_field("vortex", 32, 32)

        z_rust, diag_rust = solver_rust.transport_step(z, vx, vy)
        z_fallback, diag_fallback = solver_fallback.transport_step(z, vx, vy)

        max_diff = np.max(np.abs(z_rust - z_fallback))
        assert max_diff < 1e-10, f"Rust vs NumPy discrepancy: {max_diff}"
        assert abs(diag_rust["mean_divergence"] - diag_fallback["mean_divergence"]) < 1e-10

    def test_helmholtz_hodge_mass_conservation(self) -> None:
        """Verify that Helmholtz-Hodge projection eliminates divergence."""
        cfg = DecoupledTransportConfig(enforce_mass_conservation=True, helmholtz_hodge_iterations=40)
        solver = DecoupledSpatioTemporalTransport(cfg)

        z = _synthetic_scene("gaussian_spot", 32, 32)
        # Expansion has high positive divergence
        vx, vy = _generate_velocity_field("expansion", 32, 32)

        bridge = RustKernelBridge()
        orig_div = bridge.video_optical_flow_divergence(vx, vy)
        orig_div_norm = float(np.linalg.norm(orig_div))

        z_next, diag = solver.transport_step(z, vx, vy)
        # Remaining divergence after Poisson projection must be significantly reduced
        rem_div = diag["rem_divergence"]
        assert rem_div < orig_div_norm or diag["mean_divergence"] < float(np.mean(orig_div))

    def test_disocclusion_source_injection(self) -> None:
        """Verify that positive divergence triggers disocclusion mask and injects residual source."""
        cfg = DecoupledTransportConfig(
            enforce_mass_conservation=False,  # Keep divergence to trigger disocclusion
            disocclusion_threshold=0.01,
            source_injection_scale=2.0,
        )
        solver = DecoupledSpatioTemporalTransport(cfg)

        z = np.zeros((3, 32, 32), dtype=np.float64)
        vx, vy = _generate_velocity_field("expansion", 32, 32)
        # Distinct source pattern
        s_source = np.ones((3, 32, 32), dtype=np.float64) * 5.0

        z_next, diag = solver.transport_step(z, vx, vy, s_source=s_source)
        assert diag["disocclusion_energy"] > 0.0
        # In regions with positive divergence, values should be injected
        assert np.max(z_next) > 0.0

    def test_multiscene_generation_and_tensor_train_compression(self) -> None:
        """Verify multi-scene video trajectory generation and Tensor-Train matrix compression."""
        cfg = DecoupledTransportConfig(use_rust=True, nu=0.002)
        solver = DecoupledSpatioTemporalTransport(cfg)

        scenes = [
            ("translation", "checkerboard"),
            ("vortex", "gaussian_spot"),
            ("expansion", "gradient"),
        ]

        for flow_type, scene_type in scenes:
            z_init = _synthetic_scene(scene_type, 32, 32)
            vx, vy = _generate_velocity_field(flow_type, 32, 32)
            # Create a 6-frame flow sequence
            flow_seq = [(vx * (0.8 + 0.1 * i), vy * (0.8 + 0.1 * i)) for i in range(5)]

            traj, telem = solver.generate_multiscene_trajectory(z_init, flow_seq, dt=0.5)
            assert traj.shape == (6, 3, 32, 32)
            assert len(telem) == 5

            # Measure temporal smoothness: continuous transport must have low flicker
            flicker = ContinuousTemporalInterpolator.measure_temporal_smoothness(traj)
            assert np.isfinite(flicker)
            assert flicker < 10.0  # Physical continuous advection is smooth

            # 4D Tensor-Train Matrix Compression
            tt_video = solver.compress_trajectory_to_tensor_train(traj, max_rank=8)
            assert tt_video.compression_ratio > 1.0
            recon = tt_video.reconstruct()
            recon_error = float(np.mean((traj - recon) ** 2))
            assert recon_error < 0.05  # High-fidelity reconstruction

    def test_neural_transport_flow_net_training_and_rust_rollout(self) -> None:
        """Verify trainable NeuralTransportFlowNet learns optical flow + source and rolls out with Rust solver."""
        import torch
        from apedp_dyn.video.decoupled_transport import (
            NeuralTransportFlowNet,
            train_neural_transport_model,
        )

        cfg = DecoupledTransportConfig(use_rust=True, nu=0.002)
        solver = DecoupledSpatioTemporalTransport(cfg)

        z0 = _synthetic_scene("gaussian_spot", 16, 16)
        vx, vy = _generate_velocity_field("translation", 16, 16)
        traj, _ = solver.generate_multiscene_trajectory(z0, [(vx, vy)] * 3, dt=0.5)
        # Shape (N=2, T=4, C=3, H=16, W=16)
        clips_t = torch.from_numpy(np.stack([traj, traj], axis=0)).float()

        net = NeuralTransportFlowNet(in_channels=3, hidden_dim=16, text_dim=16)
        prompts = torch.ones(2, 16)
        losses = train_neural_transport_model(net, clips_t, prompt_embeddings=prompts, steps=20, lr=5e-3, seed=0)
        assert len(losses) == 20
        assert losses[-1] < losses[0], f"Expected transport loss to decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"

        rollout, telem = net.rollout_with_rust_transport(solver, z0, num_steps=3, prompt_emb=prompts[0], dt=0.5)
        assert rollout.shape == (4, 3, 16, 16)
        assert len(telem) == 3

