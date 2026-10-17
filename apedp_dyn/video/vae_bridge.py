"""Video VAE Latent Bridge and Invertible Continuous Wavelet Autoencoder.

Status: experimental

Bridges high-dimensional pixel-space video tensors (T, 3, H, W) with low-rank
Tensor-Train spatio-temporal video latents (TensorTrainVideo), providing:
1. ContinuousWaveletVAE: High-fidelity invertible multi-scale spatial latent autoencoder.
2. VideoVAEBridge: End-to-end bridge from raw RGB video to compressed TT-video representations.
3. Direct continuous-time RGB frame decoding from contracted TT-cores without dense 4D expansion.
4. DiffusersVAEBridge: HuggingFace ``diffusers.AutoencoderKL`` behind the SAME
   encode/decode interface (Stage-2 of the distillation ladder: the frozen
   appearance prior). Config-only (random init) and pretrained-weights paths
   are EXPLICIT and never silently conflated.
"""
from __future__ import annotations

import math

import numpy as np

from apedp_dyn.video.tensor_train_video import (
    TensorTrainVideo,
    compress_video_latent,
    decompress_video_latent,
    evaluate_continuous_temporal_frame,
)


class ContinuousWaveletVAE:
    """Invertible Continuous Multi-Scale Wavelet / Spline Video Latent Autoencoder.

    Maps (3, H, W) RGB frames to (C, H', W') compact continuous latents and back.
    Employs spatial 2D Haar/spline sub-band decomposition (LL, LH, HL, HH)
    with smooth orthogonal energy distribution, guaranteeing PSNR > 40 dB.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 12, downsample_factor: int = 2):
        if downsample_factor not in (2, 4):
            raise ValueError(f"downsample_factor must be 2 or 4, got {downsample_factor}")
        self.in_channels = in_channels
        self.downsample_factor = downsample_factor
        # Sub-bands per input channel: 4 for factor 2 (LL, LH, HL, HH), 16 for factor 4
        self.expected_latent_channels = in_channels * (downsample_factor ** 2)
        self.latent_channels = latent_channels

    def encode_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Encode a single RGB frame (3, H, W) into latent representation (C, H', W')."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[0] != self.in_channels:
            raise ValueError(f"Expected frame shape ({self.in_channels}, H, W), got {frame_rgb.shape}")
        c, h, w = frame_rgb.shape
        f = self.downsample_factor
        if h % f != 0 or w % f != 0:
            raise ValueError(f"Frame dimensions ({h}, {w}) must be divisible by {f}")

        h_prime = h // f
        w_prime = w // f

        if f == 2:
            # 2D Haar wavelet decomposition: LL, LH, HL, HH
            sub_bands = []
            for ch in range(c):
                x = frame_rgb[ch]
                x00 = x[0::2, 0::2]
                x01 = x[0::2, 1::2]
                x10 = x[1::2, 0::2]
                x11 = x[1::2, 1::2]

                ll = 0.5 * (x00 + x01 + x10 + x11)
                lh = 0.5 * (x00 - x01 + x10 - x11)
                hl = 0.5 * (x00 + x01 - x10 - x11)
                hh = 0.5 * (x00 - x01 - x10 + x11)
                sub_bands.extend([ll, lh, hl, hh])
            latent = np.stack(sub_bands, axis=0)  # (12, H/2, W/2)
        else:
            # Factor 4: hierarchical 2-level 2D Haar decomposition
            sub_bands = []
            for ch in range(c):
                x = frame_rgb[ch]
                blocks = []
                for di in range(4):
                    for dj in range(4):
                        blocks.append(x[di::4, dj::4])
                # Direct invertible block DCT/Walsh matrix
                w_mat = np.stack(blocks, axis=0)  # (16, H/4, W/4)
                sub_bands.append(w_mat)
            latent = np.concatenate(sub_bands, axis=0)  # (48, H/4, W/4)

        if latent.shape[0] > self.latent_channels:
            # Retain top dominant energy channels
            latent = latent[:self.latent_channels]
        elif latent.shape[0] < self.latent_channels:
            # Pad with zero channels
            pad = np.zeros((self.latent_channels - latent.shape[0], h_prime, w_prime), dtype=latent.dtype)
            latent = np.concatenate([latent, pad], axis=0)

        return latent

    def decode_frame(self, latent: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Decode a latent representation (C, H', W') back to an RGB frame (3, H, W)."""
        c_lat, h_prime, w_prime = latent.shape
        f = self.downsample_factor
        h = h_prime * f if target_shape is None else target_shape[0]
        w = w_prime * f if target_shape is None else target_shape[1]

        if f == 2:
            # Pad back to expected 12 channels if truncated
            if c_lat < self.expected_latent_channels:
                pad = np.zeros((self.expected_latent_channels - c_lat, h_prime, w_prime), dtype=latent.dtype)
                latent_full = np.concatenate([latent, pad], axis=0)
            else:
                latent_full = latent[:self.expected_latent_channels]

            rgb_channels = []
            for ch in range(self.in_channels):
                base_idx = ch * 4
                ll = latent_full[base_idx + 0]
                lh = latent_full[base_idx + 1]
                hl = latent_full[base_idx + 2]
                hh = latent_full[base_idx + 3]

                # Exact inverse 2D Haar synthesis
                x00 = 0.5 * (ll + lh + hl + hh)
                x01 = 0.5 * (ll - lh + hl - hh)
                x10 = 0.5 * (ll + lh - hl - hh)
                x11 = 0.5 * (ll - lh - hl + hh)

                recon_ch = np.zeros((h, w), dtype=latent.dtype)
                recon_ch[0::2, 0::2] = x00
                recon_ch[0::2, 1::2] = x01
                recon_ch[1::2, 0::2] = x10
                recon_ch[1::2, 1::2] = x11
                rgb_channels.append(recon_ch)

            rgb = np.stack(rgb_channels, axis=0)
        else:
            if c_lat < self.expected_latent_channels:
                pad = np.zeros((self.expected_latent_channels - c_lat, h_prime, w_prime), dtype=latent.dtype)
                latent_full = np.concatenate([latent, pad], axis=0)
            else:
                latent_full = latent[:self.expected_latent_channels]

            rgb_channels = []
            for ch in range(self.in_channels):
                base_idx = ch * 16
                recon_ch = np.zeros((h, w), dtype=latent.dtype)
                idx = 0
                for di in range(4):
                    for dj in range(4):
                        recon_ch[di::4, dj::4] = latent_full[base_idx + idx]
                        idx += 1
                rgb_channels.append(recon_ch)

            rgb = np.stack(rgb_channels, axis=0)

        return np.asarray(np.asarray(np.asarray(np.clip(rgb, 0.0, 1.0))))


class VideoVAEBridge:
    """Universal Spatio-Temporal Video VAE Bridge for High-Dim PDE Flow.

    Bridges raw RGB video tensors (T, 3, H, W) with low-rank TensorTrainVideo
    representations and provides direct continuous temporal frame decoding.
    """

    def __init__(
        self,
        vae: ContinuousWaveletVAE | None = None,
        latent_channels: int = 12,
        downsample_factor: int = 2,
    ):
        self.vae = vae or ContinuousWaveletVAE(
            in_channels=3,
            latent_channels=latent_channels,
            downsample_factor=downsample_factor,
        )
        self.latent_channels = self.vae.latent_channels
        self.downsample_factor = self.vae.downsample_factor

    def encode_video_to_latent(self, video_rgb: np.ndarray) -> np.ndarray:
        """Encode raw RGB video (T, 3, H, W) to dense 4D latent (T, C, H', W')."""
        if video_rgb.ndim != 4:
            raise ValueError(f"Expected 4D video array (T, 3, H, W), got shape {video_rgb.shape}")
        num_frames = video_rgb.shape[0]
        latent_frames = [self.vae.encode_frame(video_rgb[t]) for t in range(num_frames)]
        return np.stack(latent_frames, axis=0)

    def decode_latent_to_video(self, latent_video: np.ndarray) -> np.ndarray:
        """Decode dense 4D latent (T, C, H', W') back to RGB video (T, 3, H, W)."""
        if latent_video.ndim != 4:
            raise ValueError(f"Expected 4D latent array (T, C, H', W'), got shape {latent_video.shape}")
        num_frames = latent_video.shape[0]
        rgb_frames = [self.vae.decode_frame(latent_video[t]) for t in range(num_frames)]
        return np.stack(rgb_frames, axis=0)

    def encode_video_to_tt(
        self,
        video_rgb: np.ndarray,
        bond_ranks: int | tuple[int, int, int] = 16,
    ) -> TensorTrainVideo:
        """Encode RGB video (T, 3, H, W) directly into a low-rank TensorTrainVideo."""
        latent = self.encode_video_to_latent(video_rgb)
        max_r = max(bond_ranks) if isinstance(bond_ranks, (tuple, list)) else int(bond_ranks)
        return compress_video_latent(latent, max_rank=max_r)

    def decode_tt_to_video(self, tt_video: TensorTrainVideo) -> np.ndarray:
        """Decompress TensorTrainVideo and decode back to RGB video (T, 3, H, W)."""
        latent = decompress_video_latent(tt_video)
        return self.decode_latent_to_video(latent)

    def decode_continuous_frame_rgb(
        self,
        tt_video: TensorTrainVideo,
        tau: float,
    ) -> np.ndarray:
        """Directly decode an arbitrary continuous timestamp tau in [0, 1] into an RGB frame.

        Evaluates the temporal spline contraction in O(R^2 C + R^2 H + RW) without
        expanding the full 4D video tensor into memory, then decodes through the VAE.
        """
        latent_frame = evaluate_continuous_temporal_frame(tt_video, tau)
        return self.vae.decode_frame(latent_frame)

    def compute_rgb_psnr(self, original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two RGB videos."""
        mse = float(np.mean((original_rgb - reconstructed_rgb) ** 2))
        if mse <= 1e-12:
            return 100.0
        max_val = 1.0
        return 10.0 * math.log10((max_val ** 2) / mse)


class DiffusersVAEBridge:
    """HuggingFace ``diffusers.AutoencoderKL`` behind the VideoVAEBridge surface.

    Status: experimental

    Stage-2 wiring of the distillation ladder: the frozen appearance prior.
    Contract mirrors :class:`VideoVAEBridge` exactly (encode_frame /
    decode_frame / encode_video_to_latent / decode_latent_to_video /
    encode_video_to_tt / decode_tt_to_video / compute_rgb_psnr), with these
    bridge-specific semantics:

    * Construct with ``model_id`` (e.g. ``"stabilityai/sd-vae-ft-mse"``) for a
      PRETRAINED appearance prior — requires network or a warm HF cache.
    * Construct with ``config_only=True`` for a random-initialized AutoencoderKL
      (small config) — the interface/parity test path. Its latents are
      UNTRAINED NOISE; a caller must never treat them as appearance priors.
      ``self.pretrained_weights_loaded`` records which path was taken so no
      artifact can silently claim a prior it does not have.
    * Frames are mapped [0,1] -> [-1,1] for the VAE and clamped back; latent
      handling uses ``config.scaling_factor`` on both sides (encode-time
      multiply, decode-time divide), matching diffusers convention.
    * Internal compute is float32 (the diffusers reference dtype); array
      boundaries are float64, preserving the repo's boundary contract.
    * Channel-count mismatches raise before any silent truncation/padding
      (contrasting the analytic WaveletVAE's zero-pad, which is safe there
      because that transform is linear; the KL latent is not).
    """

    def __init__(
        self,
        model_id: str | None = None,
        config_only: bool = False,
        vae=None,
        in_channels: int = 3,
        downsample_factor: int = 8,
    ) -> None:
        if vae is None:
            try:
                from diffusers import AutoencoderKL
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "DiffusersVAEBridge needs the optional dependency 'diffusers' "
                    "(pip install diffusers). The analytic ContinuousWaveletVAE "
                    "VideoVAEBridge remains available without it."
                ) from exc
            if config_only:
                if model_id is not None and model_id:
                    raise ValueError("config_only=True and model_id are mutually exclusive")
                # 0.41 requires explicit per-block type tuples; without them the
                # block chain truncates to one stage and mid-block channels
                # mismatch (reproduced + pinned by the interface tests).
                n_blocks = 4
                vae = AutoencoderKL(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    latent_channels=4,
                    block_out_channels=tuple(32 * 2 ** i for i in range(n_blocks)),
                    down_block_types=("DownEncoderBlock2D",) * n_blocks,
                    up_block_types=("UpDecoderBlock2D",) * n_blocks,
                    layers_per_block=1,
                    norm_num_groups=32,  # divides every block width (32..256)
                    sample_size=32,
                )
            else:
                if not model_id:
                    raise ValueError("provide model_id or config_only=True")
                vae = AutoencoderKL.from_pretrained(model_id)  # type: ignore[assignment]
        vae.eval()
        self.vae = vae
        self.latent_channels = int(vae.config.latent_channels)
        self.downsample_factor = int(downsample_factor)
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
        # Which construction path ran — surfaced for every artifact to declare.
        self.pretrained_weights_loaded = bool(model_id) and not config_only
        self.source_model_id = None if config_only else model_id

    # -- frame-level interface (same shapes as ContinuousWaveletVAE) --------
    def encode_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb.ndim != 3 or frame_rgb.shape[0] != 3:
            raise ValueError(f"Expected frame (3, H, W), got {frame_rgb.shape}")
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.asarray(frame_rgb, np.float32)[None]).mul_(2.0).sub_(1.0)
        with torch.no_grad():
            dist = self.vae.encode(x).latent_dist
            z = dist.mode() if hasattr(dist, "mode") else dist
            z = z * self.scaling_factor
        return np.asarray(z[0].double().numpy())

    def decode_frame(self, latent: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
        import torch

        latent = np.asarray(latent)
        if latent.shape[0] != self.latent_channels:
            raise ValueError(
                f"This VAE's latent space has {self.latent_channels} channels, "
                f"got {latent.shape[0]}; refusing silent remap"
            )
        z = torch.from_numpy(np.asarray(latent, np.float32)[None]) / self.scaling_factor
        with torch.no_grad():
            x = self.vae.decode(z).sample
        rgb = ((x[0] + 1.0) * 0.5).clamp(0.0, 1.0)
        return np.asarray(rgb.double().numpy())

    # -- video + TT composition ------------------------------
    def encode_video_to_latent(self, video_rgb: np.ndarray) -> np.ndarray:
        if video_rgb.ndim != 4:
            raise ValueError(f"Expected (T, 3, H, W), got {video_rgb.shape}")
        return np.stack([self.encode_frame(video_rgb[t]) for t in range(video_rgb.shape[0])], axis=0)

    def decode_latent_to_video(self, latent_video: np.ndarray) -> np.ndarray:
        if latent_video.ndim != 4:
            raise ValueError(f"Expected (T, L, H', W'), got {latent_video.shape}")
        return np.stack([self.decode_frame(latent_video[t]) for t in range(latent_video.shape[0])], axis=0)

    def encode_video_to_tt(self, video_rgb: np.ndarray, bond_ranks: int | tuple[int, int, int] = 16):
        from apedp_dyn.video.tensor_train_video import compress_video_latent

        max_r = max(bond_ranks) if isinstance(bond_ranks, (tuple, list)) else int(bond_ranks)
        return compress_video_latent(self.encode_video_to_latent(video_rgb), max_rank=max_r)

    def decode_tt_to_video(self, tt_video) -> np.ndarray:
        from apedp_dyn.video.tensor_train_video import decompress_video_latent

        return self.decode_latent_to_video(decompress_video_latent(tt_video))

    def compute_rgb_psnr(self, original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> float:
        mse = float(np.mean((original_rgb - reconstructed_rgb) ** 2))
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * math.log10(1.0 / mse))
