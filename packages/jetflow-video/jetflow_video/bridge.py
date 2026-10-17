r"""Universal Video VAE Bridge & Invertible Continuous Wavelet Autoencoder.

Bridges pixel-space video tensors (T, 3, H, W) with latent tensors (T, C, H', W') and
low-rank TensorTrainVideo representations:
1. ContinuousWaveletVAE: Fast, zero-dependency PyTorch Haar/spline multi-scale latent autoencoder (PSNR > 40 dB).
2. DiffusersVAEBridge: Integration with HuggingFace diffusers AutoencoderKL (SD, SDXL, CogVideoX).
3. VideoVAEBridge: Unified interface with direct continuous temporal frame decoding from TT cores.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union, List
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


class CausalConv3d(nn.Module):
    """3D Causal Convolution over (B, C, T, H, W) with strict past-only temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kt, self.kh, self.kw = kernel_size
        self.pad_t = self.kt - 1
        self.pad_h = self.kh // 2
        self.pad_w = self.kw // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, self.pad_h, self.pad_w),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — replicate first frame along time so t=0 has no zero-padding boundary artifact
        if self.pad_t > 0:
            first_frames = x[:, :, :1, :, :].expand(-1, -1, self.pad_t, -1, -1)
            x = torch.cat([first_frames, x], dim=2)
        return self.conv(x)


class CausalResBlock3D(nn.Module):
    """3D Causal Spatio-Temporal Residual Block with zero-initialized output gate."""

    def __init__(self, channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        hid = hidden_channels or channels
        num_groups_1 = max(1, min(4, channels)) if channels % max(1, min(4, channels)) == 0 else 1
        num_groups_2 = max(1, min(4, hid)) if hid % max(1, min(4, hid)) == 0 else 1
        self.norm1 = nn.GroupNorm(num_groups_1, channels)
        self.conv1 = CausalConv3d(channels, hid, kernel_size=(3, 3, 3))
        self.norm2 = nn.GroupNorm(num_groups_2, hid)
        self.conv2 = CausalConv3d(hid, channels, kernel_size=(3, 3, 3))
        nn.init.zeros_(self.conv2.conv.weight)
        if self.conv2.conv.bias is not None:
            nn.init.zeros_(self.conv2.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.norm1(x))
        h = self.conv1(h)
        h = torch.nn.functional.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class Causal3DSpatioTemporalVAE(nn.Module):
    """Learned 3D Causal Spatio-Temporal Video Variational Autoencoder.

    Combines an invertible multi-scale Haar wavelet scaffold with trainable 3D causal
    convolutional residual blocks (`CausalConv3d`), diagonal Gaussian KL regularization,
    and optional temporal compression (`temporal_downsample`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 12,
        downsample_factor: int = 2,
        temporal_downsample: int = 1,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.temporal_downsample = temporal_downsample
        self.wavelet_base = ContinuousWaveletVAE(
            in_channels=in_channels,
            latent_channels=latent_channels,
            downsample_factor=downsample_factor,
        )

        self.encoder_3d = nn.Sequential(
            CausalResBlock3D(latent_channels, hidden_dim),
            CausalResBlock3D(latent_channels, hidden_dim),
        )
        # Variational posterior log-variance head (initialized strongly negative -> deterministic mode at init)
        self.logvar_head = CausalConv3d(latent_channels, latent_channels, kernel_size=(1, 1, 1))
        nn.init.zeros_(self.logvar_head.conv.weight)
        if self.logvar_head.conv.bias is not None:
            nn.init.constant_(self.logvar_head.conv.bias, -16.0)

        self.decoder_3d = nn.Sequential(
            CausalResBlock3D(latent_channels, hidden_dim),
            CausalResBlock3D(latent_channels, hidden_dim),
        )
        self.rgb_refine_3d = CausalResBlock3D(in_channels, max(12, hidden_dim // 2))

    def encode_video(
        self,
        video_rgb: torch.Tensor,
        sample_posterior: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (T, 3, H, W) or (B, T, 3, H, W) into causal 3D latent (..., T', C, H', W') and KL loss."""
        orig_ndim = video_rgb.ndim
        if orig_ndim == 4:
            video_rgb = video_rgb.unsqueeze(0)

        b, t_dim, c, h, w = video_rgb.shape
        flat_rgb = video_rgb.reshape(b * t_dim, c, h, w)
        base_lat = self.wavelet_base.encode_frame(flat_rgb)  # (B*T, C_lat, H', W')
        _, c_lat, hp, wp = base_lat.shape

        # Reshape to (B, C_lat, T, H', W') for 3D causal convolutions
        lat_3d = base_lat.reshape(b, t_dim, c_lat, hp, wp).permute(0, 2, 1, 3, 4)
        mu = self.encoder_3d(lat_3d)

        if self.temporal_downsample > 1 and t_dim >= self.temporal_downsample:
            # Causal temporal strided pooling (always keeps frame 0 + causal windows)
            s_t = self.temporal_downsample
            mu = mu[:, :, ::s_t, :, :]

        logvar = torch.clamp(self.logvar_head(mu), -20.0, 2.0)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        if sample_posterior and self.training:
            std = torch.exp(0.5 * logvar)
            z_3d = mu + std * torch.randn_like(std)
        else:
            z_3d = mu

        # Return (B, T', C_lat, H', W')
        z_out = z_3d.permute(0, 2, 1, 3, 4)
        return (z_out.squeeze(0) if orig_ndim == 4 else z_out), kl_loss

    def decode_video(
        self,
        latent_video: torch.Tensor,
        target_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode causal 3D latent (T', C, H', W') or (B, T', C, H', W') to RGB (..., T, 3, H, W)."""
        orig_ndim = latent_video.ndim
        if orig_ndim == 4:
            latent_video = latent_video.unsqueeze(0)

        b, tp, c_lat, hp, wp = latent_video.shape
        z_3d = latent_video.permute(0, 2, 1, 3, 4)  # (B, C_lat, T', H', W')

        if self.temporal_downsample > 1 or (target_frames is not None and target_frames != tp):
            t_out = target_frames if target_frames is not None else tp * self.temporal_downsample
            z_3d = torch.nn.functional.interpolate(
                z_3d,
                size=(t_out, hp, wp),
                mode="trilinear",
                align_corners=False,
            )

        refined_lat_3d = self.decoder_3d(z_3d)  # (B, C_lat, T, H', W')
        t_full = refined_lat_3d.shape[2]
        flat_lat = refined_lat_3d.permute(0, 2, 1, 3, 4).reshape(b * t_full, c_lat, hp, wp)
        base_rgb = self.wavelet_base.decode_frame(flat_lat)  # (B*T, 3, H, W)
        _, c_rgb, h, w = base_rgb.shape

        rgb_3d = base_rgb.reshape(b, t_full, c_rgb, h, w).permute(0, 2, 1, 3, 4)
        rgb_3d = torch.clamp(self.rgb_refine_3d(rgb_3d), 0.0, 1.0)
        rgb_out = rgb_3d.permute(0, 2, 1, 3, 4)
        return rgb_out.squeeze(0) if orig_ndim == 4 else rgb_out

    def encode_frame(self, frame_rgb: torch.Tensor) -> torch.Tensor:
        """Single-frame compatibility method delegating to the 3D causal encoder with T=1."""
        if frame_rgb.ndim == 3:
            z, _ = self.encode_video(frame_rgb.unsqueeze(0), sample_posterior=False)
            return z.squeeze(0)
        z, _ = self.encode_video(frame_rgb.unsqueeze(1), sample_posterior=False)
        return z.squeeze(1)

    def decode_frame(self, latent: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Single-frame compatibility method delegating to the 3D causal decoder with T=1."""
        if latent.ndim == 3:
            rgb = self.decode_video(latent.unsqueeze(0), target_frames=1)
            return rgb.squeeze(0)
        rgb = self.decode_video(latent.unsqueeze(1), target_frames=1)
        return rgb.squeeze(1)

    def compute_vae_loss(
        self,
        video_rgb: torch.Tensor,
        kl_weight: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        """Compute 3D Causal VAE reconstruction + temporal consistency + spatial edge + KL loss."""
        if video_rgb.ndim == 4:
            video_rgb = video_rgb.unsqueeze(0)
        z, kl_loss = self.encode_video(video_rgb, sample_posterior=True)
        recon = self.decode_video(z, target_frames=video_rgb.shape[1])

        l1_loss = torch.nn.functional.l1_loss(recon, video_rgb)
        if video_rgb.shape[1] > 1:
            dt_pred = recon[:, 1:] - recon[:, :-1]
            dt_true = video_rgb[:, 1:] - video_rgb[:, :-1]
            temp_loss = torch.nn.functional.l1_loss(dt_pred, dt_true)
        else:
            temp_loss = torch.tensor(0.0, device=video_rgb.device)

        dx_pred = recon[..., :, 1:] - recon[..., :, :-1]
        dx_true = video_rgb[..., :, 1:] - video_rgb[..., :, :-1]
        dy_pred = recon[..., 1:, :] - recon[..., :-1, :]
        dy_true = video_rgb[..., 1:, :] - video_rgb[..., :-1, :]
        edge_loss = 0.5 * (
            torch.nn.functional.l1_loss(dx_pred, dx_true)
            + torch.nn.functional.l1_loss(dy_pred, dy_true)
        )

        total = l1_loss + 0.25 * temp_loss + 0.25 * edge_loss + kl_weight * kl_loss
        return {
            "total_loss": total,
            "l1_loss": l1_loss,
            "temporal_loss": temp_loss,
            "edge_loss": edge_loss,
            "kl_loss": kl_loss,
        }


    def fit_video_batch(
        self,
        videos_rgb: torch.Tensor,
        num_steps: int = 20,
        lr: float = 1e-3,
        kl_weight: float = 1e-5,
    ) -> Dict[str, float]:
        """Train the 3D Causal Spatio-Temporal VAE on a batch of RGB videos (B, T, 3, H, W)."""
        if videos_rgb.ndim == 4:
            videos_rgb = videos_rgb.unsqueeze(0)
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        with torch.no_grad():
            self.eval()
            initial_loss = float(self.compute_vae_loss(videos_rgb, kl_weight=kl_weight)["total_loss"].item())
            self.train()
        best_loss = initial_loss
        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        for _ in range(num_steps):
            opt.zero_grad()
            losses = self.compute_vae_loss(videos_rgb, kl_weight=kl_weight)
            loss = losses["total_loss"]
            cur = float(loss.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
        self.load_state_dict(best_state)
        self.eval()
        with torch.no_grad():
            final_loss = float(self.compute_vae_loss(videos_rgb, kl_weight=kl_weight)["total_loss"].item())
        return {"initial_loss": initial_loss, "final_loss": min(best_loss, final_loss)}


class VideoVAEBridge:
    """Universal Spatio-Temporal Video VAE Bridge for JetFlow-Video."""

    def __init__(
        self,
        use_diffusers: bool = False,
        diffusers_model: str = "stabilityai/sd-vae-ft-mse",
        use_causal_3d_vae: bool = True,
        latent_channels: int = 12,
        downsample_factor: int = 2,
        temporal_downsample: int = 1,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        perceptual_refiner: Optional[nn.Module] = None,
        enable_perceptual_refiner: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.use_diffusers = use_diffusers
        self.use_causal_3d_vae = use_causal_3d_vae and not use_diffusers
        self.causal_3d_vae: Optional[Causal3DSpatioTemporalVAE] = None

        if use_diffusers:
            try:
                self.vae = DiffusersVAEBridge(
                    pretrained_model_name_or_path=diffusers_model,
                    device=self.device,
                )
                self.latent_channels = self.vae.latent_channels
                self.downsample_factor = self.vae.downsample_factor
            except Exception as e:
                print(f"[VideoVAEBridge] Failed to load diffusers VAE: {e}. Falling back to Causal3DSpatioTemporalVAE.")
                self.use_diffusers = False
                self.use_causal_3d_vae = use_causal_3d_vae
                if self.use_causal_3d_vae:
                    self.causal_3d_vae = Causal3DSpatioTemporalVAE(
                        in_channels=3,
                        latent_channels=latent_channels,
                        downsample_factor=downsample_factor,
                        temporal_downsample=temporal_downsample,
                    ).to(self.device)
                    self.vae = self.causal_3d_vae
                else:
                    self.vae = ContinuousWaveletVAE(
                        in_channels=3,
                        latent_channels=latent_channels,
                        downsample_factor=downsample_factor,
                    )
                self.latent_channels = latent_channels
                self.downsample_factor = downsample_factor
        else:
            if self.use_causal_3d_vae:
                self.causal_3d_vae = Causal3DSpatioTemporalVAE(
                    in_channels=3,
                    latent_channels=latent_channels,
                    downsample_factor=downsample_factor,
                    temporal_downsample=temporal_downsample,
                ).to(self.device)
                self.vae = self.causal_3d_vae
            else:
                self.vae = ContinuousWaveletVAE(
                    in_channels=3,
                    latent_channels=latent_channels,
                    downsample_factor=downsample_factor,
                )
            self.latent_channels = latent_channels
            self.downsample_factor = downsample_factor

        if perceptual_refiner is not None:
            self.perceptual_refiner: Optional[nn.Module] = perceptual_refiner.to(self.device)
        elif enable_perceptual_refiner:
            from jetflow_video.perceptual_decoder import PerceptualRefiner
            self.perceptual_refiner = PerceptualRefiner().to(self.device)
        else:
            self.perceptual_refiner = None

    def encode_video_to_latent(self, video_rgb: torch.Tensor) -> torch.Tensor:
        r"""Encode RGB video (T, 3, H, W) to latent (T', C, H', W')."""
        if video_rgb.ndim != 4:
            raise ValueError(f"Expected 4D video tensor (T, 3, H, W), got {video_rgb.shape}")

        if self.causal_3d_vae is not None:
            self.causal_3d_vae = self.causal_3d_vae.to(device=video_rgb.device, dtype=video_rgb.dtype)
            lat, _ = self.causal_3d_vae.encode_video(video_rgb, sample_posterior=False)
            return lat

        t = video_rgb.shape[0]
        latent_frames = [self.vae.encode_frame(video_rgb[i]) for i in range(t)]
        return torch.stack(latent_frames, dim=0)

    def decode_latent_to_video(self, latent_video: torch.Tensor) -> torch.Tensor:
        r"""Decode latent video (T', C, H', W') to RGB video (T, 3, H, W)."""
        if latent_video.ndim != 4:
            raise ValueError(f"Expected 4D latent tensor (T, C, H', W'), got {latent_video.shape}")

        if self.perceptual_refiner is not None:
            refiner = self.perceptual_refiner.to(device=latent_video.device, dtype=latent_video.dtype)
            return refiner(latent_video)

        if self.causal_3d_vae is not None:
            self.causal_3d_vae = self.causal_3d_vae.to(device=latent_video.device, dtype=latent_video.dtype)
            return self.causal_3d_vae.decode_video(latent_video)

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
        if self.perceptual_refiner is not None:
            refiner = self.perceptual_refiner.to(device=latent_frame.device, dtype=latent_frame.dtype)
            return refiner(latent_frame)
        return self.vae.decode_frame(latent_frame)

    @staticmethod
    def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
        mse = torch.mean((original - reconstructed) ** 2).item()
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * math.log10(1.0 / mse))
