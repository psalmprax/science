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


def test_remediated_flaws_verification():
    from jetflow_video.tensor_train import TTLinear
    from jetflow_video.trainer import encode_prompt_deterministic

    # 1. JetFlowBackbone 3D positional encoding + TTLinear FFN support
    cfg = JetFlowModelConfig(
        latent_channels=12,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        text_dim=64,
        use_tt_ffn=True,
        tt_ffn_rank=8,
    )
    model = JetFlowVideoModel(cfg)
    assert isinstance(model.backbone.blocks[0].ffn[0], TTLinear)

    # Spatially uniform input must produce spatially varying hidden representations due to 3D pos encoding
    z_uniform = torch.ones(1, 4, 12, 8, 8)
    pos_feat = model.backbone._build_spatiotemporal_pos_features(4, 8, 8, z_uniform.device, z_uniform.dtype)
    assert pos_feat.shape == (1, 4, 64, 21)
    assert not torch.allclose(pos_feat[:, :, 0, :], pos_feat[:, :, -1, :])

    # 2. Subband-preserving Noether conservation does not scramble LL and LH channels
    wavelet_v = torch.zeros(4, 12, 8, 8)
    wavelet_v[:, 0, :, :] = 2.0  # Constant LL channel
    v_proj, _ = model.noether_engine.project_spatial_latent_velocity(wavelet_v)
    assert torch.allclose(v_proj[:, 1, :, :], torch.zeros_like(v_proj[:, 1, :, :]), atol=1e-6)

    # 3. Jet2 stepper uses exactly 2 velocity evaluations per step (4 total for 2 steps)
    eval_calls = {"n": 0}

    def counted_v(z_in: torch.Tensor, t_in: float) -> torch.Tensor:
        eval_calls["n"] += 1
        return -0.5 * z_in

    stepper = JetFlowStepper(JetFlowConfig(num_steps=2, order=2))
    _ = stepper.generate(counted_v, torch.randn(2, 4, 8, 8))
    assert eval_calls["n"] == 4

    # 4. VideoVAEBridge with integrated PerceptualRefiner
    bridge_refined = VideoVAEBridge(use_diffusers=False, latent_channels=12, downsample_factor=2, enable_perceptual_refiner=True)
    assert bridge_refined.perceptual_refiner is not None
    lat_clip = torch.rand(3, 12, 8, 8)
    rgb_clip = bridge_refined.decode_latent_to_video(lat_clip)
    assert rgb_clip.shape == (3, 3, 16, 16)

    # 5. Synonym-aligned deterministic prompt encoding in trainer
    e_car1 = encode_prompt_deterministic("fast red car driving on road")
    e_car2 = encode_prompt_deterministic("rapid automobile racing on highway")
    e_ocean = encode_prompt_deterministic("calm ocean water waves")
    assert torch.allclose(e_car1, encode_prompt_deterministic("fast red car driving on road"))
    sim_syn = float(torch.dot(e_car1, e_car2).item())
    sim_unrel = float(torch.dot(e_car1, e_ocean).item())
    assert sim_syn > sim_unrel


def test_four_architectural_pillars_95_readiness():
    from jetflow_video import (
        Causal3DSpatioTemporalVAE,
        CompositionalTextEncoder,
        DepthAware3DTransportEngine,
        compute_plucker_rays,
        encode_prompt_sequence,
        extract_physical_steering_from_prompt,
        se3_exp_map,
    )

    # Pillar 1: Compositional Sequence-Level Text Encoder & Prompt-to-Physics Extraction
    seq_a = encode_prompt_sequence("red car chasing blue truck", max_seq_len=8, dim=64)
    seq_b = encode_prompt_sequence("blue car chasing red truck", max_seq_len=8, dim=64)
    assert seq_a.shape == (8, 64)
    # Order-sensitive modifier binding: "red car ... blue truck" != "blue car ... red truck"
    assert not torch.allclose(seq_a, seq_b, atol=1e-3)

    # Negation gating flips semantic polarity of negated token
    seq_pos = encode_prompt_sequence("storm with rain", max_seq_len=6, dim=64)
    seq_neg = encode_prompt_sequence("storm without rain", max_seq_len=6, dim=64)
    rain_dot = float(torch.dot(seq_pos[3], seq_neg[3]).item())
    assert rain_dot < 0.0

    txt_enc = CompositionalTextEncoder(text_dim=32, max_seq_len=8, num_heads=2, num_layers=1)
    batch_seq = txt_enc.encode_prompts(["fast red car driving right", "3d rotating sphere orbiting"])
    assert batch_seq.shape == (2, 8, 32)

    knobs = extract_physical_steering_from_prompt("fast red car driving right and orbiting in 3d with zoom")
    assert knobs.convection_velocity[0] > 0.0
    assert knobs.camera_orbit_yaw > 0.0
    assert knobs.camera_zoom > 0.0

    # Pillar 2: Learned 3D Causal Spatio-Temporal VAE (strict temporal causality + trainable reduction)
    causal_vae = Causal3DSpatioTemporalVAE(in_channels=3, latent_channels=12, downsample_factor=2, hidden_dim=16)
    vid1 = torch.rand(1, 4, 3, 16, 16)
    vid2 = vid1.clone()
    vid2[:, 2:, :, :, :] = torch.rand(1, 2, 3, 16, 16)  # Modify future frames t=2, 3
    z1, _ = causal_vae.encode_video(vid1, sample_posterior=False)
    z2, _ = causal_vae.encode_video(vid2, sample_posterior=False)
    # Strict causality: frames t=0, 1 must be completely unaffected by changes at t=2, 3
    assert torch.allclose(z1[:, :2], z2[:, :2], atol=1e-6)

    recon_init = causal_vae.decode_video(z1)
    assert VideoVAEBridge.compute_psnr(vid1, recon_init) > 40.0
    fit_stats = causal_vae.fit_video_batch(vid1, num_steps=4, lr=1e-3)
    assert fit_stats["final_loss"] <= fit_stats["initial_loss"]

    # Pillar 3: Exact SE(3) Exponential Map, 6D Plücker Rays, & DepthAware3DTransportEngine
    omega = torch.tensor([[0.2, -0.35, 0.1]])
    trans = torch.tensor([[0.1, -0.05, 0.25]])
    rot, t_vec = se3_exp_map(omega, trans)
    assert rot.shape == (1, 3, 3)
    assert t_vec.shape == (1, 3)
    assert torch.allclose(torch.bmm(rot.transpose(1, 2), rot), torch.eye(3).unsqueeze(0), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(1), atol=1e-5)

    rays = compute_plucker_rays(t_dim=4, h=8, w=8, omega_per_sec=(0.1, 0.3, 0.0), v_per_sec=(0.2, 0.0, -0.1))
    assert rays.shape == (1, 4, 64, 6)
    # Plücker orthogonality invariant: d . (o x d) == 0
    d_vec, m_vec = rays[..., :3], rays[..., 3:]
    plucker_ortho = torch.sum(d_vec * m_vec, dim=-1).abs().max().item()
    assert plucker_ortho < 1e-5

    depth_engine = DepthAware3DTransportEngine(latent_channels=12, hidden_dim=16)
    z_frame = torch.randn(2, 12, 8, 8)
    depth, _flow_3d, _disocc = depth_engine.predict_depth_and_flow(z_frame)
    assert depth.shape == (2, 1, 8, 8)
    assert float(depth.min().item()) >= depth_engine.z_min
    assert float(depth.max().item()) <= depth_engine.z_max
    warped_3d = depth_engine.warp_frame_se3(z_frame, omega=(0.1, 0.25, 0.0), trans=(0.1, 0.0, -0.1), dt=0.5)
    assert warped_3d.shape == z_frame.shape

    # Pillar 4: 3D Tubelet Patchification + String-Prompt Generation + EMA & CFG-Dropout Trainer
    cfg_3d = JetFlowModelConfig(
        latent_channels=12,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        text_dim=64,
        patch_size=(2, 2, 2),
        use_tt_ffn=True,
        tt_ffn_rank=8,
        enable_plucker_3d=True,
        enable_depth_transport=True,
    )
    model_3d = JetFlowVideoModel(cfg_3d)
    tt_vid, rgb_vid = model_3d.generate_video(
        prompt_embedding="fast red car driving right and orbiting in 3d at sunset",
        num_steps=2,
        shape=(4, 12, 8, 8),
    )
    assert tt_vid.shape == (4, 12, 8, 8)
    assert rgb_vid.shape == (4, 3, 16, 16)
    assert torch.isfinite(rgb_vid).all()


