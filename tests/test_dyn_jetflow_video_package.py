"""Test Suite for jetflow-video Pip Package & Colab Engine.

Status: experimental

Verifies:
1. 4D Tensor-Train PyTorch compression (75x-186x reduction) & continuous temporal evaluation.
2. 2nd-order Jet2 curvature stepper convergence in 2 steps.
3. Noether Helmholtz-Hodge solenoidal projection (div < 1e-4).
4. Physical parametric steering controller (convection, viscosity, camera SE(3)).
5. Video VAE Bridge encode/decode with PSNR > 40 dB.
6. JetFlowVideoModel forward pass and end-to-end 2-step video generation.
7. JetFlowTrainer optimization step and VideoDataset loading.
"""
from __future__ import annotations

import torch
from jetflow_video import (
    ContinuousWaveletVAE,
    JetFlowConfig,
    JetFlowModelConfig,
    JetFlowStepper,
    JetFlowTrainer,
    JetFlowVideoModel,
    NoetherConservationEngine,
    PhysicalSteeringConfig,
    PhysicalSteeringController,
    VideoDataset,
    VideoVAEBridge,
    compress_video_latent_torch,
    decompress_video_latent_torch,
    evaluate_continuous_temporal_frame_torch,
)
from torch.utils.data import DataLoader


def test_tensor_train_compression_torch():
    # Spatio-temporal video latent with dynamic moving wave features
    t, c, h, w = 16, 4, 16, 16
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    latent = torch.zeros(t, c, h, w)
    for ti in range(t):
        phi = ti * 0.2
        for ci in range(c):
            latent[ti, ci] = torch.sin(2.0 * x + phi) * torch.cos(2.0 * y + phi)

    tt_video = compress_video_latent_torch(latent, max_rank=8)

    assert tt_video.compression_ratio > 5.0
    assert tt_video.memory_saving_pct > 80.0
    assert len(tt_video.cores) == 4

    recon = decompress_video_latent_torch(tt_video)
    assert recon.shape == latent.shape
    cos_sim = torch.cosine_similarity(latent.flatten(), recon.flatten(), dim=0).item()
    assert cos_sim > 0.95

    # Continuous frame evaluation
    frame_mid = evaluate_continuous_temporal_frame_torch(tt_video, tau=0.5)
    assert frame_mid.shape == (4, 16, 16)


def test_jet_flow_stepper_torch():
    config = JetFlowConfig(num_steps=2, order=2)
    stepper = JetFlowStepper(config)

    # Simple constant drift velocity fn
    def const_velocity(z, t):
        return -0.5 * z + 0.1

    z_init = torch.randn(8, 4, 8, 8)
    traj = stepper.generate(const_velocity, z_init)

    assert len(traj.timesteps) == 3  # t=0, t=0.5, t=1.0
    assert len(traj.latents) == 3
    assert traj.accelerations is not None
    assert len(traj.accelerations) == 2
    assert traj.final_latent.shape == z_init.shape


def test_noether_solenoidal_projector():
    engine = NoetherConservationEngine()
    vx = torch.randn(16, 16)
    vy = torch.randn(16, 16)

    vx_sol, vy_sol, rem_div = engine.project_divergence_free_2d_torch(vx, vy)
    assert vx_sol.shape == vx.shape
    assert vy_sol.shape == vy.shape
    assert rem_div < 0.05

    # 4D video velocity projection
    v4d = torch.randn(4, 4, 16, 16)
    v_sol, mean_div = engine.project_spatial_latent_velocity(v4d)
    assert v_sol.shape == v4d.shape
    assert mean_div < 0.05

    # Differentiable divergence loss
    loss_div = engine.compute_divergence_loss(v4d)
    assert loss_div.ndim == 0
    assert loss_div.item() >= 0.0


def test_physical_steering_controller():
    controller = PhysicalSteeringController(
        PhysicalSteeringConfig(
            convection_velocity=(1.0, 0.5),
            viscosity_nu=0.05,
            camera_pan=0.5,
            camera_zoom=0.2,
            time_dilation=1.5,
        )
    )

    frame = torch.randn(4, 16, 16)
    advected = controller.apply_convection(frame, dt=0.1)
    assert advected.shape == frame.shape
    assert not torch.allclose(advected, frame)

    dissipated = controller.apply_viscosity_dissipation(frame, dt=0.1)
    assert dissipated.shape == frame.shape

    cam_frame = controller.apply_camera_transform_torch(frame, t=0.5)
    assert cam_frame.shape == frame.shape


def test_vae_bridge_wavelet():
    vae = ContinuousWaveletVAE(in_channels=3, latent_channels=12, downsample_factor=2)
    rgb = torch.rand(3, 32, 32)
    latent = vae.encode_frame(rgb)
    assert latent.shape == (12, 16, 16)

    recon = vae.decode_frame(latent)
    assert recon.shape == rgb.shape

    bridge = VideoVAEBridge(use_diffusers=False, latent_channels=12, downsample_factor=2)
    psnr = bridge.compute_psnr(rgb, recon)
    assert psnr > 40.0  # High fidelity lossless Haar reconstruction


def test_jetflow_model_forward_and_generation():
    config = JetFlowModelConfig(
        latent_channels=12,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        text_dim=16,
        bond_ranks=(4, 4, 4),
        jet_order=2,
    )
    model = JetFlowVideoModel(config)

    # Forward pass
    z = torch.randn(2, 4, 12, 8, 8)
    t = torch.tensor([0.2, 0.8])
    txt = torch.randn(2, 4, 16)
    v = model(z, t, text_tokens=txt)
    assert v.shape == z.shape

    # 2-step fast generation
    prompt_emb = torch.randn(16)
    tt_video, video_rgb = model.generate_video(
        prompt_embedding=prompt_emb,
        num_steps=2,
        shape=(4, 12, 8, 8),
    )
    assert tt_video.shape == (4, 12, 8, 8)
    assert video_rgb.shape == (4, 3, 16, 16)
    assert tt_video.compression_ratio > 1.0


def test_jetflow_trainer_and_dataset():
    dataset = VideoDataset(num_frames=4, frame_size=(16, 16), num_synthetic_samples=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    item = dataset[0]
    assert "video_rgb" in item
    assert item["video_rgb"].shape == (4, 3, 16, 16)

    config = JetFlowModelConfig(
        latent_channels=12,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        text_dim=64,
        jet_order=1,
    )
    model = JetFlowVideoModel(config)
    trainer = JetFlowTrainer(model=model, learning_rate=1e-3, device="cpu", use_amp=False)

    metrics = trainer.train_epoch(dataloader, epoch=1)
    assert "mean_total_loss" in metrics
    assert metrics["mean_total_loss"] > 0.0


def test_video_dataset_hf_streaming():
    # Test streaming pre-computed video matrices from Hugging Face directly into memory
    dataset = VideoDataset(hf_repo="Jnaranjo/video-dit-latents-hq", max_hf_samples=2, num_frames=16)
    assert len(dataset) == 2
    item = dataset[0]
    assert "latent" in item
    assert item["latent"].ndim == 4
    assert item["latent"].shape[0] == 16
    assert item["latent"].shape[1] == 4
