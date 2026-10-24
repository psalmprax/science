r"""Perceptual Decoder: zero-residual-init finetuning on top of the Haar VAE.

Status: experimental

Photorealism lever #2: the Haar wavelet synthesis has NOTHING LEARNED — pixel
sharpness is capped by the exact inverse transform. This module adds a small
residual conv decoder whose LAST layer is zero-initialized, so training
STARTS at bit-exact Haar decode parity and can only ADD learned refinement
(monotone upgrade path — no regression cliff if training is stopped early).

Training objective (patch-LSGAN + reconstruction):
    L = L1(proj, true) + lambda_grad * L_grad(proj, true) + lambda_adv * L_gan
where L_grad is the image-gradient difference (texture/detail loss) and the
patch discriminator supplies the adversarial component that pushes local
texture statistics toward the real domain (this is the same mechanism that
makes SD's pixel decoders photoreal).

CPU constraints honored: channels 16-32, 3-4 conv layers, 32-64 px frames.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_synthesis(latent: torch.Tensor, in_channels: int = 3, downsample_factor: int = 2) -> torch.Tensor:
    """Exact inverse 2D Haar synthesis from sub-band latents (f=2 path).

    latent: (B, C_lat, H', W') with C_lat >= in_channels * 4 bands.
    Returns (B, 3, H, W) base reconstruction (the Haar-VAE decode).
    """
    b, c_lat, hp, wp = latent.shape
    f = downsample_factor
    if f != 2:
        raise NotImplementedError("Perceptual refiner currently supports downsample_factor=2")
    expected = in_channels * 4
    if c_lat < expected:
        pad = torch.zeros((b, expected - c_lat, hp, wp), device=latent.device, dtype=latent.dtype)
        latent_full = torch.cat([latent, pad], dim=1)
    else:
        latent_full = latent[:, :expected]
    h, w = hp * 2, wp * 2
    rgb_channels = []
    for ch in range(in_channels):
        base = ch * 4
        ll, lh, hl, hh = latent_full[:, base + 0], latent_full[:, base + 1], latent_full[:, base + 2], latent_full[:, base + 3]
        x00 = 0.5 * (ll + lh + hl + hh)
        x01 = 0.5 * (ll - lh + hl - hh)
        x10 = 0.5 * (ll + lh - hl - hh)
        x11 = 0.5 * (ll - lh - hl + hh)
        recon = torch.zeros((b, h, w), device=latent.device, dtype=latent.dtype)
        recon[:, 0::2, 0::2] = x00
        recon[:, 0::2, 1::2] = x01
        recon[:, 1::2, 0::2] = x10
        recon[:, 1::2, 1::2] = x11
        rgb_channels.append(recon)
    return torch.stack(rgb_channels, dim=1)


class PerceptualRefiner(nn.Module):
    """Haar base + zero-init residual refinement (starts AT Haar parity)."""

    def __init__(self, latent_channels: int = 12, downsample_factor: int = 2,
                 in_channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        f = downsample_factor
        # Residual trunk on latent resolution, upsampled to pixels
        self.trunk = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),  # residual delta
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode (B, C, H', W') latents to RGB (B, 3, H, W), [0, 1] clamped."""
        base = _haar_synthesis(latent, in_channels=self.in_channels, downsample_factor=self.downsample_factor)
        f_up = F.interpolate(latent, size=base.shape[-2:], mode="nearest")
        delta = self.trunk(f_up)
        return torch.clamp(base + delta, 0.0, 1.0)

    def base_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Pure Haar decode (no learned refinement) — for parity comparisons."""
        base = _haar_synthesis(latent, in_channels=self.in_channels, downsample_factor=self.downsample_factor)
        return torch.clamp(base, 0.0, 1.0)


class PatchDiscriminator(nn.Module):
    """Minimal 70x70-style patch discriminator for CPU budgets."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout) if False else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.net = nn.Sequential(
            block(in_channels, base_channels),      # -> /2
            block(base_channels, base_channels * 2), # -> /4
            block(base_channels * 2, base_channels * 4),  # -> /8
            nn.Conv2d(base_channels * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gradient_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    dpx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dtx = true[:, :, 1:, :] - true[:, :, :-1, :]
    dpy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dty = true[:, :, :, 1:] - true[:, :, :, :-1]
    return F.l1_loss(dpx, dtx) + F.l1_loss(dpy, dty)


def train_perceptual_stack(
    refiner: PerceptualRefiner,
    disc: PatchDiscriminator,
    real_videos: List[np.ndarray],
    steps: int = 60,
    lr_gen: float = 2e-4,
    lr_disc: float = 2e-4,
    batch_frames: int = 8,
    lambda_grad: float = 0.5,
    lambda_adv: float = 0.1,
    seed: int = 3,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train the refiner + patch discriminator on real RGB frames.

    real_videos: list of (T, 3, H, W) numpy arrays in [0, 1] (e.g. fluid domain
    clips). Inputs for the refiner are the Haar latents ENCODED from the same
    real frames (encoder stays fixed). Returns loss history.
    """
    from jetflow_video.bridge import ContinuousWaveletVAE

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    refiner = refiner.to(dev)
    disc = disc.to(dev)
    opt_g = torch.optim.AdamW(refiner.parameters(), lr=lr_gen)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_disc)
    vae = ContinuousWaveletVAE(latent_channels=refiner.latent_channels,
                               downsample_factor=refiner.downsample_factor)
    rng = np.random.RandomState(seed)

    # Pool real frames
    frames = np.concatenate([np.asarray(v, dtype=np.float32) for v in real_videos], axis=0)  # (N, 3, H, W)
    n = frames.shape[0]

    hist = {"recon_l1": [], "grad_loss": [], "adv_g": [], "adv_d": []}
    for step in range(steps):
        idx = rng.choice(n, size=batch_frames, replace=False)
        true = torch.from_numpy(frames[idx]).to(dev)  # (B, 3, H, W)

        with torch.no_grad():
            lat = torch.stack([vae.encode_frame(true[i]) for i in range(true.shape[0])], dim=0)
        proj = refiner(lat)

        # Discriminator step (LSGAN form)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            proj_det = refiner(lat)
        score_fake = disc(proj_det)
        score_real = disc(true)
        d_loss = torch.mean((score_real - 1.0) ** 2) + torch.mean(score_fake ** 2)
        d_loss.backward()
        opt_d.step()

        # Generator step
        opt_g.zero_grad(set_to_none=True)
        proj = refiner(lat)
        score_fake = disc(proj)
        recon_l1 = F.l1_loss(proj, true)
        gl = _gradient_loss(proj, true)
        adv = torch.mean((score_fake - 1.0) ** 2)
        g_loss = recon_l1 + lambda_grad * gl + lambda_adv * adv
        g_loss.backward()
        opt_g.step()

        hist["recon_l1"].append(float(recon_l1.item()))
        hist["grad_loss"].append(float(gl.item()))
        hist["adv_g"].append(float(adv.item()))
        hist["adv_d"].append(float(d_loss.item()))

    return hist


class RefinerVAEBridge:
    """VAE bridge whose decoder is the learned PerceptualRefiner (Haar parity
    at init; encoder remains the fixed invertible Haar transform)."""

    def __init__(self, refiner: PerceptualRefiner, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.refiner = refiner.to(self.device)
        self.latent_channels = refiner.latent_channels
        self.downsample_factor = refiner.downsample_factor
        self._vae = None

    @property
    def encoder(self):
        if self._vae is None:
            from jetflow_video.bridge import ContinuousWaveletVAE
            self._vae = ContinuousWaveletVAE(latent_channels=self.refiner.latent_channels,
                                             downsample_factor=self.refiner.downsample_factor)
        return self._vae

    def encode_video_to_latent(self, video_rgb: torch.Tensor) -> torch.Tensor:
        if video_rgb.ndim != 4:
            raise ValueError(f"Expected (T, 3, H, W) video, got {video_rgb.shape}")
        lat = []
        for i in range(video_rgb.shape[0]):
            v = video_rgb[i].to(self.device)
            lat.append(self.encoder.encode_frame(v) if not isinstance(v, torch.Tensor) else self.encoder.encode_frame(v))
        return torch.stack(lat, dim=0)

    @torch.no_grad()
    def decode_latent_to_video(self, latent_video: torch.Tensor) -> torch.Tensor:
        if latent_video.ndim != 4:
            raise ValueError(f"Expected (T, C, H', W') latents, got {latent_video.shape}")
        out = []
        for i in range(latent_video.shape[0]):
            out.append(self.refiner(latent_video[i].unsqueeze(0).to(self.device)).squeeze(0))
        return torch.stack(out, dim=0)
