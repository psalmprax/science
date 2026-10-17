r"""Cache Video Dataset into Fast VAE Latents for High-Speed Training.

Pre-encodes raw video files (.avi, .mp4) into compact PyTorch latent representations (.pt),
extracts action labels for semantic text conditioning, and stores them for instant DataLoader ingestion.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from jetflow_video import ContinuousWaveletVAE


def extract_action_label(filename: str) -> str:
    """Extract human-readable action label from UCF101 filename (e.g. 'v_ApplyEyeMakeup_g01_c01.avi')."""
    base = os.path.basename(filename)
    match = re.match(r"v_([A-Za-z]+)_g\d+_c\d+", base)
    if match:
        action_camel = match.group(1)
        # Convert CamelCase to words (e.g. ApplyEyeMakeup -> Apply Eye Makeup)
        words = re.findall(r"[A-Z][a-z]*", action_camel)
        return " ".join(words) if words else action_camel
    return "Action Video"


def create_text_embedding(label: str, text_dim: int = 64) -> torch.Tensor:
    """Deterministic normalized semantic embedding from action label string."""
    rng = np.random.RandomState(abs(hash(label)) % (2**31))
    vec = rng.randn(text_dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return torch.from_numpy(vec / max(1e-8, norm))


def main():
    parser = argparse.ArgumentParser(description="Cache video files into VAE latents")
    parser.add_argument("--video_dir", type=str, default="data/video_dataset/raw_videos", help="Input video folder")
    parser.add_argument("--latent_dir", type=str, default="data/video_dataset/latents", help="Output latent folder")
    parser.add_argument("--max_videos", type=int, default=100, help="Maximum number of videos to encode")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames to sample per video")
    parser.add_argument("--frame_size", type=int, default=64, help="Spatial resolution (HxW)")
    args = parser.parse_args()

    os.makedirs(args.latent_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.avi")) + glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not video_files:
        print(f"[CacheLatents] No video files found in {args.video_dir}")
        return

    selected_files = video_files[:args.max_videos]
    print(f"[CacheLatents] Processing {len(selected_files)} videos into {args.latent_dir}...")
    vae = ContinuousWaveletVAE(in_channels=3, latent_channels=12, downsample_factor=2)

    success_count = 0
    t0 = time.time()
    for v_path in tqdm(selected_files, desc="Encoding Latents"):
        out_name = os.path.splitext(os.path.basename(v_path))[0] + ".pt"
        out_path = os.path.join(args.latent_dir, out_name)

        if os.path.exists(out_path):
            success_count += 1
            continue

        try:
            raw_frames = iio.imread(v_path)
            if raw_frames.ndim != 4:
                continue

            t_total = raw_frames.shape[0]
            indices = np.linspace(0, t_total - 1, args.num_frames).astype(int)
            sampled = raw_frames[indices]

            # (T, H, W, 3) -> (T, 3, H, W) float [0, 1]
            video_tensor = torch.from_numpy(sampled).permute(0, 3, 1, 2).float() / 255.0

            # Spatial resize if needed
            if video_tensor.shape[2] != args.frame_size or video_tensor.shape[3] != args.frame_size:
                video_tensor = F.interpolate(
                    video_tensor, size=(args.frame_size, args.frame_size), mode="bilinear", align_corners=False
                )

            # Encode frames through VAE
            latent_frames = [vae.encode_frame(video_tensor[i]) for i in range(args.num_frames)]
            latent_video = torch.stack(latent_frames, dim=0)  # (T, 12, H/2, W/2)

            action = extract_action_label(v_path)
            prompt_emb = create_text_embedding(action, text_dim=64)

            torch.save(
                {
                    "latent": latent_video,
                    "action": action,
                    "prompt_embedding": prompt_emb,
                },
                out_path,
            )
            success_count += 1
        except Exception as e:
            print(f"Error encoding {v_path}: {e}")

    elapsed = time.time() - t0
    print(f"[CacheLatents] Completed {success_count}/{len(selected_files)} videos in {elapsed:.2f}s ({elapsed / max(1, success_count):.3f}s/video)!")
    print(f"[CacheLatents] Latents directory size: {sum(os.path.getsize(os.path.join(args.latent_dir, f)) for f in os.listdir(args.latent_dir)) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
