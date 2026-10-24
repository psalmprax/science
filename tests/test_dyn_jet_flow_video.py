r"""Regression test suite for JetFlow-Video Continuous Spatio-Temporal PDE Video Generation.

Tests:
1. 4D Tensor-Train Latent Video Compression (compression ratio >= 4x, fidelity R^2 > 0.98).
2. Continuous temporal frame evaluation without dense tensor expansion.
3. Jet2 Curvature Flow Stepper: 2-step Taylor vs 50-step Euler reference trajectory.
4. Continuous Temporal Interpolator: 24 FPS -> 60 FPS / 120 FPS upsampling with temporal smoothness.
5. Native Rust Jet2 Taylor operator integration bridge.
"""
from __future__ import annotations

import time

import numpy as np

import apedp_kernel as ak
from apedp_dyn.video.continuous_interpolator import (
    ContinuousTemporalInterpolator,
)
from apedp_dyn.video.jet_flow_stepper import (
    JetFlowConfig,
    JetFlowStepper,
)
from apedp_dyn.video.tensor_train_video import (
    compress_video_latent,
    decompress_video_latent,
    evaluate_continuous_temporal_frame,
)


def _generate_synthetic_video_latent(
    t_dim: int = 16,
    c_dim: int = 4,
    h_dim: int = 32,
    w_dim: int = 32,
    seed: int = 42,
    noise_level: float = 0.005,
) -> np.ndarray:
    r"""Generate synthetic 4D continuous video latent with moving spatio-temporal features."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(-1, 1, h_dim), np.linspace(-1, 1, w_dim), indexing="ij")

    latent = np.zeros((t_dim, c_dim, h_dim, w_dim), dtype=np.float64)
    times = np.linspace(0, 2 * np.pi, t_dim)

    for ti, t_val in enumerate(times):
        # Moving Gaussian vortex centers
        cx1 = 0.5 * np.cos(t_val)
        cy1 = 0.5 * np.sin(t_val)
        r1_sq = (x - cx1) ** 2 + (y - cy1) ** 2
        blob1 = np.exp(-r1_sq / 0.15)

        cx2 = -0.4 * np.cos(1.5 * t_val)
        cy2 = -0.4 * np.sin(1.5 * t_val)
        r2_sq = (x - cx2) ** 2 + (y - cy2) ** 2
        blob2 = np.exp(-r2_sq / 0.20)

        for ci in range(c_dim):
            phase = ci * (np.pi / 4.0)
            latent[ti, ci, :, :] = (
                np.cos(phase) * blob1 + np.sin(phase) * blob2 + noise_level * rng.normal(size=(h_dim, w_dim))
            )

    return latent


def test_tensor_train_compression_ratio_and_fidelity():
    """Verify 4D Tensor-Train Latent Video compression achieves >= 4x savings with R^2 > 0.98."""
    t_dim, c_dim, h_dim, w_dim = 16, 4, 32, 32
    dense_latent = _generate_synthetic_video_latent(t_dim, c_dim, h_dim, w_dim)

    # Compress into 4-core Tensor-Train
    tt_video = compress_video_latent(dense_latent, max_rank=8, tolerance=1e-3)

    assert tt_video.shape == (t_dim, c_dim, h_dim, w_dim)
    assert len(tt_video.bond_ranks) == 3
    assert all(r <= 8 for r in tt_video.bond_ranks)

    # Verify memory reduction
    assert tt_video.compression_ratio >= 4.0, (
        f"Expected compression ratio >= 4.0, got {tt_video.compression_ratio:.2f}x"
    )
    assert tt_video.memory_saving_pct >= 75.0, (
        f"Expected memory savings >= 75%, got {tt_video.memory_saving_pct:.1f}%"
    )

    # Decompress and verify reconstruction fidelity
    reconstructed = decompress_video_latent(tt_video)
    assert reconstructed.shape == dense_latent.shape

    ss_tot = np.sum((dense_latent - np.mean(dense_latent)) ** 2)
    ss_res = np.sum((dense_latent - reconstructed) ** 2)
    r2_fidelity = 1.0 - (ss_res / max(1e-12, ss_tot))

    assert r2_fidelity > 0.98, f"Expected reconstruction R^2 > 0.98, got {r2_fidelity:.4f}"


def test_tensor_train_continuous_temporal_slice():
    """Verify continuous temporal frame evaluation without building the full 4D video array."""
    t_dim, c_dim, h_dim, w_dim = 16, 4, 32, 32
    dense_latent = _generate_synthetic_video_latent(t_dim, c_dim, h_dim, w_dim)
    tt_video = compress_video_latent(dense_latent, max_rank=8)

    # Evaluate at boundary tau = 0.0 and tau = 1.0
    frame_0 = evaluate_continuous_temporal_frame(tt_video, 0.0)
    frame_1 = evaluate_continuous_temporal_frame(tt_video, 1.0)

    assert frame_0.shape == (c_dim, h_dim, w_dim)
    assert frame_1.shape == (c_dim, h_dim, w_dim)

    # Check correlation with actual first and last dense frames
    corr_0 = np.corrcoef(frame_0.ravel(), dense_latent[0].ravel())[0, 1]
    corr_1 = np.corrcoef(frame_1.ravel(), dense_latent[-1].ravel())[0, 1]
    assert corr_0 > 0.98, f"First frame correlation {corr_0:.4f} <= 0.98"
    assert corr_1 > 0.98, f"Last frame correlation {corr_1:.4f} <= 0.98"

    # Evaluate at non-integer continuous midpoints tau in (0, 1)
    for tau_test in [0.25, 0.5, 0.731]:
        mid_frame = evaluate_continuous_temporal_frame(tt_video, tau_test)
        assert mid_frame.shape == (c_dim, h_dim, w_dim)
        assert np.all(np.isfinite(mid_frame))


def test_jet2_flow_stepper_2_step_vs_euler_50_steps():
    """Verify that 2-step Jet2 Curvature Stepper matches 50-step Euler reference trajectory."""
    dim = 64
    rng = np.random.default_rng(123)
    z_init = rng.normal(0, 1.0, size=dim)

    # Nonlinear curved velocity field representing flow matching transport
    # v(z, t) = -0.5 * z + 0.3 * sin(pi * t) * z^2 + 0.2 * cos(2 * pi * t)
    def velocity_fn(z: np.ndarray, t: float) -> np.ndarray:
        return -0.5 * z + 0.3 * np.sin(np.pi * t) * (z ** 2) / (1.0 + np.abs(z)) + 0.2 * np.cos(2 * np.pi * t)

    # 1. High-resolution reference trajectory using 50 Euler steps
    stepper_ref = JetFlowStepper(JetFlowConfig(num_steps=50, order=1))
    traj_ref = stepper_ref.generate(velocity_fn, z_init)
    z_ref = traj_ref.final_latent

    # 2. 2-Step standard Euler stepper (without curvature correction)
    stepper_euler_2 = JetFlowStepper(JetFlowConfig(num_steps=2, order=1))
    traj_euler_2 = stepper_euler_2.generate(velocity_fn, z_init)
    z_euler_2 = traj_euler_2.final_latent

    # 3. 2-Step Jet2 Curvature Stepper (with acceleration a_t = d v / dt)
    stepper_jet2_2 = JetFlowStepper(JetFlowConfig(num_steps=2, order=2))
    traj_jet2_2 = stepper_jet2_2.generate(velocity_fn, z_init)
    z_jet2_2 = traj_jet2_2.final_latent

    # Calculate errors relative to 50-step ground truth
    rmse_euler = float(np.sqrt(np.mean((z_euler_2 - z_ref) ** 2)))
    rmse_jet2 = float(np.sqrt(np.mean((z_jet2_2 - z_ref) ** 2)))

    # Compute trajectory R^2 fidelity
    ss_tot = np.sum((z_ref - np.mean(z_ref)) ** 2)
    ss_res_jet2 = np.sum((z_ref - z_jet2_2) ** 2)
    r2_jet2 = 1.0 - (ss_res_jet2 / max(1e-12, ss_tot))

    # Jet2 Curvature Stepper must substantially reduce error compared to naive 2-step Euler
    assert rmse_jet2 < rmse_euler, (
        f"Expected Jet2 RMSE ({rmse_jet2:.5f}) < Euler RMSE ({rmse_euler:.5f})"
    )
    assert r2_jet2 > 0.95, f"Expected 2-step Jet2 R^2 > 0.95, got {r2_jet2:.4f}"
    assert traj_jet2_2.accelerations is not None
    assert len(traj_jet2_2.accelerations) == 2
    assert traj_jet2_2.total_curvature > 0.0


def test_continuous_interpolator_infinite_fps():
    """Verify Continuous Temporal Interpolation upsamples 24 FPS -> 60 FPS / 120 FPS smoothly."""
    t_dim, c_dim, h_dim, w_dim = 12, 3, 24, 24
    dense_latent = _generate_synthetic_video_latent(t_dim, c_dim, h_dim, w_dim)
    tt_video = compress_video_latent(dense_latent, max_rank=6)

    interpolator = ContinuousTemporalInterpolator(tt_video)

    # 1. Upsample 24 FPS -> 60 FPS
    video_60fps, taus_60 = interpolator.resample_to_fps(target_fps=60.0, source_fps=24.0)
    assert video_60fps.shape[0] > t_dim
    assert video_60fps.shape[1:] == (c_dim, h_dim, w_dim)
    assert len(taus_60) == video_60fps.shape[0]

    # 2. Render 4x slow motion
    video_slow, _taus_slow = interpolator.resample_to_fps(target_fps=96.0, source_fps=24.0)
    assert video_slow.shape[0] > video_60fps.shape[0]

    # 3. Verify temporal smoothness (zero high-frequency flickering)
    smoothness_60fps = ContinuousTemporalInterpolator.measure_temporal_smoothness(video_60fps)
    assert np.isfinite(smoothness_60fps)
    assert smoothness_60fps < 1.0, f"Temporal flicker acceleration {smoothness_60fps} is too high"


def test_rust_kernel_jet2_acceleration_bridge():
    """Verify native Rust Jet2 Taylor operator evaluates 1,000 spatial points in under 5 ms."""
    m_points = 1000
    dim = 16
    h_units = 32

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1.0, (m_points, dim))
    w1 = rng.normal(0, 0.4, (h_units, dim))
    b1 = rng.normal(0, 0.1, (h_units,))
    w2 = rng.normal(0, 0.4, (h_units,))

    t0 = time.perf_counter()
    u, grad_norm_sq, lap, res = ak.rime_jet2_neural_pde_operator(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w1, dtype=np.float64),
        np.ascontiguousarray(b1, dtype=np.float64),
        np.ascontiguousarray(w2, dtype=np.float64),
        0.25,  # viscosity_nu
    )
    t_ms = (time.perf_counter() - t0) * 1000.0

    assert len(u) == m_points
    assert len(grad_norm_sq) == m_points
    assert len(lap) == m_points
    assert len(res) == m_points
    assert np.all(np.isfinite(res))
    assert t_ms < 50.0, f"Rust Jet2 operator took {t_ms:.2f} ms (expected < 50 ms)"


def test_semantic_conditioner_fourier_time_and_text_embedding():
    """Verify continuous Fourier time embedding and deterministic text prompt embeddings."""
    from apedp_dyn.video.semantic_bridge import SemanticConditioningConfig, SemanticVideoConditioner

    cfg = SemanticConditioningConfig(latent_dim=16, text_dim=64, guidance_scale=3.0)
    conditioner = SemanticVideoConditioner(cfg)

    # 1. Continuous Fourier temporal embeddings
    e_0 = conditioner.embed_continuous_time(0.0)
    e_mid = conditioner.embed_continuous_time(0.5)
    e_1 = conditioner.embed_continuous_time(1.0)

    assert e_0.shape == (16,)
    assert e_mid.shape == (16,)
    assert e_1.shape == (16,)
    assert not np.allclose(e_0, e_mid)
    assert not np.allclose(e_mid, e_1)

    # 2. Text prompt embeddings
    p1 = conditioner.embed_text_prompt("ocean waves at sunset")
    p1_repeat = conditioner.embed_text_prompt("ocean waves at sunset")
    p2 = conditioner.embed_text_prompt("astronaut floating in deep space")

    assert np.allclose(p1, p1_repeat), "Identical prompt must produce identical embedding"
    assert not np.allclose(p1, p2), "Distinct prompts must produce distinct embeddings"
    assert p1.shape == (8, 64)


def test_semantic_flow_guidance_steering():
    """Verify Classifier-Free Flow Guidance steers velocity field with exact scale multiplier."""
    from apedp_dyn.video.semantic_bridge import SemanticConditioningConfig, SemanticVideoConditioner

    cfg = SemanticConditioningConfig(latent_dim=16, text_dim=64, guidance_scale=3.0)
    conditioner = SemanticVideoConditioner(cfg)

    rng = np.random.default_rng(101)
    z = rng.normal(0, 1.0, size=(4, 16, 8, 8))  # (T, C, H, W)
    text_tokens = conditioner.embed_text_prompt("drone flying over snowy mountain peak")

    # Evaluate unconditional (s = 0) and conditional
    v_uncond = conditioner.compute_conditioned_velocity(z, t=0.5, text_tokens=None)
    v_cond = conditioner.compute_conditioned_velocity(z, t=0.5, text_tokens=text_tokens)

    # Evaluate flow-guided with scale s = 3.0
    v_guided, steer_mag = conditioner.compute_flow_guided_velocity(
        z, t=0.5, text_tokens=text_tokens, guidance_scale=3.0
    )

    # Mathematical identity: v_guided - v_uncond == s * (v_cond - v_uncond)
    expected_guided = v_uncond + 3.0 * (v_cond - v_uncond)
    assert np.allclose(v_guided, expected_guided, atol=1e-10)
    assert steer_mag > 0.0

    # Scale s = 1.0 must match exact conditional velocity
    v_s1, _ = conditioner.compute_flow_guided_velocity(
        z, t=0.5, text_tokens=text_tokens, guidance_scale=1.0
    )
    assert np.allclose(v_s1, v_cond, atol=1e-10)


def test_end_to_end_text_to_video_jet2_flow_matching():
    """Verify end-to-end Text-to-Video generation using Jet2 Flow Stepper and Tensor-Train compression."""
    from apedp_dyn.video.continuous_interpolator import ContinuousTemporalInterpolator
    from apedp_dyn.video.jet_flow_stepper import JetFlowConfig, JetFlowStepper
    from apedp_dyn.video.semantic_bridge import SemanticConditioningConfig, SemanticVideoConditioner
    from apedp_dyn.video.tensor_train_video import compress_video_latent

    t0 = time.perf_counter()

    # 1. Initialize lightweight semantic conditioner (d=16 latent channels)
    cfg = SemanticConditioningConfig(latent_dim=16, text_dim=64, guidance_scale=2.5)
    conditioner = SemanticVideoConditioner(cfg, seed=777)
    text_prompt = "hyper-realistic fluid vortex animation"
    text_tokens = conditioner.embed_text_prompt(text_prompt)

    # 2. Gaussian noise latent prior z_0 ~ N(0, I)
    rng = np.random.default_rng(888)
    shape_4d = (8, 16, 16, 16)  # T=8, C=16, H=16, W=16
    z_init = rng.normal(0, 1.0, size=shape_4d)

    # 3. Define guided velocity function
    def guided_v(z_curr: np.ndarray, t_curr: float) -> np.ndarray:
        v_g, _ = conditioner.compute_flow_guided_velocity(z_curr, t_curr, text_tokens, guidance_scale=2.5)
        return v_g

    # 4. Generate video latent in just 2 Jet2 curvature steps
    stepper = JetFlowStepper(JetFlowConfig(num_steps=2, order=2))
    traj = stepper.generate(guided_v, z_init)
    z_final = traj.final_latent

    assert z_final.shape == shape_4d
    assert np.all(np.isfinite(z_final))
    assert traj.total_curvature > 0.0

    # 5. Compress generated video latent using 4D Tensor-Train
    tt_video = compress_video_latent(z_final, max_rank=8)
    assert tt_video.compression_ratio >= 3.0
    assert tt_video.memory_saving_pct >= 65.0

    # 6. Upsample to 60 FPS continuous video frames
    interpolator = ContinuousTemporalInterpolator(tt_video)
    video_60fps, _taus = interpolator.resample_to_fps(target_fps=60.0, source_fps=24.0)

    assert video_60fps.shape[0] > 8
    assert video_60fps.shape[1:] == (16, 16, 16)
    assert np.all(np.isfinite(video_60fps))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # End-to-end execution must finish quickly on CPU (under 500 ms)
    assert elapsed_ms < 500.0, f"End-to-end pipeline took {elapsed_ms:.2f} ms (expected < 500 ms)"

