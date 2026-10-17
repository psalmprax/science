r"""Universal Video VAE Bridge & Invertible Continuous Wavelet Autoencoder.

Bridges pixel-space video tensors (T, 3, H, W) with latent tensors (T, C, H', W') and
low-rank TensorTrainVideo representations:
1. ContinuousWaveletVAE: Fast, zero-dependency PyTorch Haar/spline multi-scale latent autoencoder (PSNR > 40 dB).
2. DiffusersVAEBridge: Integration with HuggingFace diffusers AutoencoderKL (SD, SDXL, CogVideoX).
3. VideoVAEBridge: Unified interface with direct continuous temporal frame decoding from TT cores.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn

from jetflow_video.tensor_train import (
    TensorTrainVideo,
    compress_video_latent_torch,
    decompress_video_latent_torch,
    evaluate_continuous_temporal_frame_torch,
)


class ContinuousWaveletVAE:
    """Invertible Continuous Multi-Scale Wavelet Video Latent Autoencoder in PyTorch."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 12, downsample_factor: int = 2) -> None:
        if downsample_factor not in (2, 4):
            raise ValueError(f"downsample_factor must be 2 or 4, got {downsample_factor}")
        self.in_channels = in_channels
        self.downsample_factor = downsample_factor
        self.expected_latent_channels = in_channels * (downsample_factor ** 2)
        self.latent_channels = latent_channels

    def encode_frame(self, frame_rgb: torch.Tensor) -> torch.Tensor:
        r"""Encode (3, H, W) or (B, 3, H, W) RGB tensor into latent (C, H', W')."""
        orig_ndim = frame_rgb.ndim
        if orig_ndim == 3:
            frame_rgb = frame_rgb.unsqueeze(0)

        b, c, h, w = frame_rgb.shape
        f = self.downsample_factor
        if h % f != 0 or w % f != 0:
            raise ValueError(f"Frame dimensions ({h}, {w}) must be divisible by {f}")

        h_prime = h // f
        w_prime = w // f

        if f == 2:
            x00 = frame_rgb[:, :, 0::2, 0::2]
            x01 = frame_rgb[:, :, 0::2, 1::2]
            x10 = frame_rgb[:, :, 1::2, 0::2]
            x11 = frame_rgb[:, :, 1::2, 1::2]

            ll = 0.5 * (x00 + x01 + x10 + x11)
            lh = 0.5 * (x00 - x01 + x10 - x11)
            hl = 0.5 * (x00 + x01 - x10 - x11)
            hh = 0.5 * (x00 - x01 - x10 + x11)

            # Stack along channel dimension: [LL, LH, HL, HH] for each input channel
            sub_bands = []
            for ch in range(c):
                sub_bands.extend([ll[:, ch:ch+1], lh[:, ch:ch+1], hl[:, ch:ch+1], hh[:, ch:ch+1]])
            latent = torch.cat(sub_bands, dim=1)  # (B, 12, H/2, W/2)
        else:
            # 4x4 spatial blocks
            blocks = []
            for di in range(4):
                for dj in range(4):
                    blocks.append(frame_rgb[:, :, di::4, dj::4])
            latent = torch.cat(blocks, dim=1)  # (B, 48, H/4, W/4)

        if latent.shape[1] > self.latent_channels:
            latent = latent[:, :self.latent_channels]
        elif latent.shape[1] < self.latent_channels:
            pad = torch.zeros(
                (b, self.latent_channels - latent.shape[1], h_prime, w_prime),
                device=latent.device,
                dtype=latent.dtype,
            )
            latent = torch.cat([latent, pad], dim=1)

        return latent.squeeze(0) if orig_ndim == 3 else latent

    def decode_frame(self, latent: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        r"""Decode latent tensor (C, H', W') or (B, C, H', W') to RGB (3, H, W)."""
        orig_ndim = latent.ndim
        if orig_ndim == 3:
            latent = latent.unsqueeze(0)

        b, c_lat, h_prime, w_prime = latent.shape
        f = self.downsample_factor
        h = h_prime * f if target_shape is None else target_shape[0]
        w = w_prime * f if target_shape is None else target_shape[1]

        if c_lat == 4:
            # Calibrated SD / Video-DiT Latent-to-RGB linear projection matrix
            # Latent channels correspond to [Luminance, Chroma A, Chroma B, Texture]
            proj_w = torch.tensor([
                [0.298,  0.207,  0.208],
                [0.187,  0.286,  0.173],
                [-0.158, 0.189,  0.264],
                [-0.184, -0.271, 0.473],
            ], device=latent.device, dtype=latent.dtype)
            rgb_low = torch.einsum("bchw,cd->bdhw", latent, proj_w)
            rgb_min = rgb_low.amin(dim=(-2, -1), keepdim=True)
            rgb_max = rgb_low.amax(dim=(-2, -1), keepdim=True)
            rgb_norm = (rgb_low - rgb_min) / (rgb_max - rgb_min + 1e-6)
            rgb = torch.nn.functional.interpolate(
                rgb_norm, size=(h, w), mode="bilinear", align_corners=False
            )
            rgb = torch.clamp(rgb, 0.0, 1.0)
            return rgb.squeeze(0) if orig_ndim == 3 else rgb

        if f == 2:
            if c_lat < self.expected_latent_channels:
                pad = torch.zeros(
                    (b, self.expected_latent_channels - c_lat, h_prime, w_prime),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                latent_full = torch.cat([latent, pad], dim=1)
            else:
                latent_full = latent[:, :self.expected_latent_channels]

            rgb_channels = []
            for ch in range(self.in_channels):
                base_idx = ch * 4
                ll = latent_full[:, base_idx + 0]
                lh = latent_full[:, base_idx + 1]
                hl = latent_full[:, base_idx + 2]
                hh = latent_full[:, base_idx + 3]

                x00 = 0.5 * (ll + lh + hl + hh)
                x01 = 0.5 * (ll - lh + hl - hh)
                x10 = 0.5 * (ll + lh - hl - hh)
                x11 = 0.5 * (ll - lh - hl + hh)

                recon_ch = torch.zeros((b, h, w), device=latent.device, dtype=latent.dtype)
                recon_ch[:, 0::2, 0::2] = x00
                recon_ch[:, 0::2, 1::2] = x01
                recon_ch[:, 1::2, 0::2] = x10
                recon_ch[:, 1::2, 1::2] = x11
                rgb_channels.append(recon_ch)

            rgb = torch.stack(rgb_channels, dim=1)
        else:
            if c_lat < self.expected_latent_channels:
                pad = torch.zeros(
                    (b, self.expected_latent_channels - c_lat, h_prime, w_prime),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                latent_full = torch.cat([latent, pad], dim=1)
            else:
                latent_full = latent[:, :self.expected_latent_channels]

            rgb_channels = []
            for ch in range(self.in_channels):
                base_idx = ch * 16
                recon_ch = torch.zeros((b, h, w), device=latent.device, dtype=latent.dtype)
                idx = 0
                for di in range(4):
                    for dj in range(4):
                        recon_ch[:, di::4, dj::4] = latent_full[:, base_idx + idx]
                        idx += 1
                rgb_channels.append(recon_ch)

            rgb = torch.stack(rgb_channels, dim=1)

        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb.squeeze(0) if orig_ndim == 3 else rgb


class DiffusersVAEBridge:
    """Wrapper around HuggingFace Diffusers AutoencoderKL for Video Latent Encoding/Decoding."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/sd-vae-ft-mse",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        scaling_factor: float = 0.18215,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.scaling_factor = scaling_factor
        self.latent_channels = 4
        self.downsample_factor = 8

        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype).to(self.device)
        self.vae.eval()

    @torch.no_grad()
    def encode_frame(self, frame_rgb: torch.Tensor) -> torch.Tensor:
        r"""Encode RGB frame (3, H, W) normalized [0, 1] to latent (4, H/8, W/8)."""
        orig_ndim = frame_rgb.ndim
        if orig_ndim == 3:
            frame_rgb = frame_rgb.unsqueeze(0)

        # Map [0, 1] to [-1, 1]
        x = (frame_rgb.to(device=self.device, dtype=self.dtype) * 2.0) - 1.0
        latent_dist = self.vae.encode(x).latent_dist
        latent = latent_dist.sample() * self.scaling_factor

        return latent.squeeze(0) if orig_ndim == 3 else latent

    @torch.no_grad()
    def decode_frame(self, latent: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        r"""Decode latent (4, H/8, W/8) to RGB frame (3, H, W) normalized [0, 1]."""
        orig_ndim = latent.ndim
        if orig_ndim == 3:
            latent = latent.unsqueeze(0)

        z = (latent.to(device=self.device, dtype=self.dtype) / self.scaling_factor)
        dec = self.vae.decode(z).sample
        rgb = (dec * 0.5 + 0.5).clamp(0.0, 1.0)

        return rgb.squeeze(0) if orig_ndim == 3 else rgb


class VideoVAEBridge:
    """Universal Spatio-Temporal Video VAE Bridge for JetFlow-Video."""

    def __init__(
        self,
        use_diffusers: bool = False,
        diffusers_model: str = "stabilityai/sd-vae-ft-mse",
        latent_channels: int = 12,
        downsample_factor: int = 2,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.use_diffusers = use_diffusers

        if use_diffusers:
            try:
                self.vae = DiffusersVAEBridge(
                    pretrained_model_name_or_path=diffusers_model,
                    device=self.device,
                )
                self.latent_channels = self.vae.latent_channels
                self.downsample_factor = self.vae.downsample_factor
            except Exception as e:
                print(f"[VideoVAEBridge] Failed to load diffusers VAE: {e}. Falling back to ContinuousWaveletVAE.")
                self.use_diffusers = False
                self.vae = ContinuousWaveletVAE(
                    in_channels=3,
                    latent_channels=latent_channels,
                    downsample_factor=downsample_factor,
                )
                self.latent_channels = latent_channels
                self.downsample_factor = downsample_factor
        else:
            self.vae = ContinuousWaveletVAE(
                in_channels=3,
                latent_channels=latent_channels,
                downsample_factor=downsample_factor,
            )
            self.latent_channels = latent_channels
            self.downsample_factor = downsample_factor

    def encode_video_to_latent(self, video_rgb: torch.Tensor) -> torch.Tensor:
        r"""Encode RGB video (T, 3, H, W) to latent (T, C, H', W')."""
        if video_rgb.ndim != 4:
            raise ValueError(f"Expected 4D video tensor (T, 3, H, W), got {video_rgb.shape}")

        t = video_rgb.shape[0]
        latent_frames = [self.vae.encode_frame(video_rgb[i]) for i in range(t)]
        return torch.stack(latent_frames, dim=0)

    def decode_latent_to_video(self, latent_video: torch.Tensor) -> torch.Tensor:
        r"""Decode latent video (T, C, H', W') to RGB video (T, 3, H, W)."""
        if latent_video.ndim != 4:
            raise ValueError(f"Expected 4D latent tensor (T, C, H', W'), got {latent_video.shape}")

        t = latent_video.shape[0]
        rgb_frames = [self.vae.decode_frame(latent_video[i]) for i in range(t)]
        return torch.stack(rgb_frames, dim=0)

    def encode_video_to_tt(
        self,
        video_rgb: torch.Tensor,
        max_rank: int = 16,
    ) -> TensorTrainVideo:
        r"""Encode RGB video (T, 3, H, W) directly into a low-rank TensorTrainVideo."""
        latent = self.encode_video_to_latent(video_rgb)
        return compress_video_latent_torch(latent, max_rank=max_rank)

    def decode_tt_to_video(self, tt_video: TensorTrainVideo) -> torch.Tensor:
        r"""Decompress TensorTrainVideo and decode back to RGB video (T, 3, H, W)."""
        latent = decompress_video_latent_torch(tt_video)
        return self.decode_latent_to_video(latent)

    def decode_continuous_frame_rgb(
        self,
        tt_video: TensorTrainVideo,
        tau: float,
    ) -> torch.Tensor:
        r"""Directly decode an arbitrary continuous timestamp tau in [0, 1] into an RGB frame.

        Evaluates temporal spline contraction in O(R^2 C + R^2 H + RW) without 4D expansion.
        """
        latent_frame = evaluate_continuous_temporal_frame_torch(tt_video, tau)
        return self.vae.decode_frame(latent_frame)

    @staticmethod
    def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
        mse = torch.mean((original - reconstructed) ** 2).item()
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * math.log10(1.0 / mse))
