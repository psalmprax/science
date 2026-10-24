r"""Standalone Training & Video Generation Script for Google Colab and Local GPUs.

Usage:
    python train_colab.py --epochs 3 --batch_size 2 --device cuda --num_samples 8
"""
from __future__ import annotations

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from jetflow_video import (
    JetFlowVideoModel,
    JetFlowModelConfig,
    JetFlowTrainer,
    VideoDataset,
    PhysicalSteeringConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Train JetFlow-Video on Google Colab or Local GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu/mps)")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of procedural video sequences to train on")
    parser.add_argument("--frames", type=int, default=8, help="Frames per video clip")
    parser.add_argument("--frame_size", type=int, default=32, help="Spatial frame dimension (e.g. 32, 64)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Transformer hidden embedding dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of spatio-temporal transformer layers")
    parser.add_argument("--latent_dir", type=str, default=None, help="Directory of pre-cached latent files (.pt)")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory of raw video files (.avi, .mp4)")
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face dataset repository to stream video matrices from (e.g. Jnaranjo/video-dit-latents-hq)")
    parser.add_argument("--gen_frames", type=int, default=None, help="Number of frames for video generation (e.g. 60 for 5s at 12fps, 120 for 10s)")
    parser.add_argument("--fps", type=int, default=12, help="Playback frames per second for exported MP4 video")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints and videos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[JetFlow] Using execution device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # 1. Dataset & DataLoader
    if args.hf_repo:
        print(f"[JetFlow] Streaming pre-computed video matrices directly from Hugging Face: {args.hf_repo}")
        dataset = VideoDataset(
            hf_repo=args.hf_repo,
            max_hf_samples=args.num_samples,
            num_frames=args.frames,
            frame_size=(args.frame_size, args.frame_size),
        )
    elif args.latent_dir:
        print(f"[JetFlow] Initializing VideoDataset from cached latents in: {args.latent_dir}")
        dataset = VideoDataset(latent_dir=args.latent_dir, num_frames=args.frames, frame_size=(args.frame_size, args.frame_size))
    elif args.video_dir:
        print(f"[JetFlow] Initializing VideoDataset from raw video files in: {args.video_dir}")
        dataset = VideoDataset(video_dir=args.video_dir, num_frames=args.frames, frame_size=(args.frame_size, args.frame_size))
    else:
        print(f"[JetFlow] Initializing VideoDataset with {args.num_samples} procedural samples ({args.frames} frames @ {args.frame_size}x{args.frame_size})...")
        dataset = VideoDataset(num_frames=args.frames, frame_size=(args.frame_size, args.frame_size), num_synthetic_samples=args.num_samples)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Dynamic Channel Detection & Model Initialization
    sample_0 = dataset[0]
    if "latent" in sample_0:
        detected_channels = sample_0["latent"].shape[1]
    else:
        detected_channels = 12
    print(f"[JetFlow] Detected latent channels: {detected_channels}")

    config = JetFlowModelConfig(
        latent_channels=detected_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=4,
        text_dim=64,
        guidance_scale=3.0,
        bond_ranks=(8, 8, 8),
        jet_order=2,
    )
    print(f"[JetFlow] Constructing JetFlowVideoModel (Order-2 Jet2 Curvature, Solenoidal Noether Projection)...")
    model = JetFlowVideoModel(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[JetFlow] Total trainable parameters: {total_params:,}")

    # 3. Trainer Setup
    trainer = JetFlowTrainer(model=model, learning_rate=args.lr, device=device)

    # 4. Training Loop
    print(f"\n[JetFlow] Starting Jet2 Flow Matching Training for {args.epochs} epochs...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch=epoch)
        print(
            f"Epoch {epoch:02d} | "
            f"Total Loss: {metrics['mean_total_loss']:.4f} | "
            f"Flow Loss: {metrics['mean_flow_loss']:.4f} | "
            f"Acc Loss: {metrics['mean_acc_loss']:.4f} | "
            f"Div Loss: {metrics['mean_div_loss']:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"[JetFlow] Training completed in {elapsed:.2f} seconds ({elapsed / max(1, args.epochs):.2f}s per epoch).")

    # 5. Save Checkpoint
    ckpt_path = os.path.join(args.output_dir, "jetflow_video_latest.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[JetFlow] Model checkpoint saved to: {ckpt_path}")

    # 6. Fast 2-Step Video Generation
    print(f"\n[JetFlow] Running 2-Step Curvature Video Latent Generation...")
    model.eval()
    prompt_emb = torch.randn(64, device=device)
    physics = PhysicalSteeringConfig(
        convection_velocity=(0.2, 0.0),
        viscosity_nu=0.01,
        time_dilation=1.0,
    )

    if "latent" in sample_0:
        gen_h = sample_0["latent"].shape[2]
        gen_w = sample_0["latent"].shape[3]
    else:
        gen_h = args.frame_size // 2
        gen_w = args.frame_size // 2

    num_gen = args.gen_frames if args.gen_frames is not None else args.frames
    t0 = time.time()
    with torch.no_grad():
        tt_video, generated_rgb = model.generate_video(
            prompt_embedding=prompt_emb,
            num_steps=2,
            shape=(num_gen, detected_channels, gen_h, gen_w),
            physical_knobs=physics,
        )
    gen_time = time.time() - t0
    print(f"[JetFlow] 2-Step Generation completed in {gen_time * 1000.0:.2f} ms ({num_gen / gen_time:.1f} FPS)!")
    print(f"[JetFlow] Tensor-Train Compression Ratio: {tt_video.compression_ratio:.1f}x ({tt_video.memory_saving_pct:.2f}% VRAM saved)")
    print(f"[JetFlow] Generated Video Tensor Shape: {tuple(generated_rgb.shape)} (Min: {generated_rgb.min():.2f}, Max: {generated_rgb.max():.2f})")

    # Export frames or MP4 if imageio is available
    out_video_path = os.path.join(args.output_dir, "generated_sample.mp4")
    try:
        import imageio
        video_np = (generated_rgb.cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype("uint8")
        imageio.mimwrite(out_video_path, video_np, fps=args.fps)
        duration_sec = num_gen / float(args.fps)
        print(f"[JetFlow] Sample video ({num_gen} frames, {duration_sec:.1f}s @ {args.fps} FPS) exported to: {out_video_path}")
    except Exception as e:
        print(f"[JetFlow] Video export note: {e}")


if __name__ == "__main__":
    main()
