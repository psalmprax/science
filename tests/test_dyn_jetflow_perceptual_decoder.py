r"""Perceptual decoder (zero-init residual on Haar base) tests.

Pins (float32, CPU):
1. At initialization the refiner decodes EXACTLY like the Haar VAE
   (zero-residual parity — no regression cliff).
2. Training the patch-GAN stack on fluid-domain frames decreases the
   discriminator loss and the generator adversarial component, without
   regressing reconstruction, and outputs stay finite and in [0, 1].
3. Round-trip bridge: encode video -> decode with the refiner preserves
   shapes and value range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "packages" / "jetflow-video"
for p in (str(_REPO), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from jetflow_video.bridge import ContinuousWaveletVAE
from jetflow_video.perceptual_decoder import (
    PatchDiscriminator,
    PerceptualRefiner,
    RefinerVAEBridge,
    train_perceptual_stack,
)

from apedp_dyn.video.fluid_domain import make_fluid_clip


def _fluid_frames(n_clips: int = 3, t: int = 6, h: int = 32, w: int = 32) -> list:
    return [make_fluid_clip(200 + k, t=t, h=h, w=w) for k in range(n_clips)]


def test_zero_init_residual_starts_at_haar_parity():
    torch.manual_seed(0)
    refiner = PerceptualRefiner(latent_channels=12)
    lat = torch.rand(2, 12, 16, 16)
    ref_out = refiner(lat).detach().numpy()
    base = refiner.base_decode(lat).detach().numpy()
    np.testing.assert_array_equal(ref_out, base)


def test_gan_training_progress_on_fluid_frames():
    torch.manual_seed(1)
    refiner = PerceptualRefiner(latent_channels=12)
    disc = PatchDiscriminator()
    frames = _fluid_frames()
    hist = train_perceptual_stack(refiner, disc, frames, steps=30, batch_frames=8, seed=3)

    q = max(1, len(hist["adv_d"]) // 4)
    assert float(np.mean(hist["adv_d"][-q:])) < float(np.mean(hist["adv_d"][:q])), hist["adv_d"]
    assert float(np.mean(hist["adv_g"][-q:])) < float(np.mean(hist["adv_g"][:q])), hist["adv_g"]
    # Reconstruction must not regress beyond the L1-vs-adversarial tradeoff
    assert float(np.mean(hist["recon_l1"][-q:])) <= float(np.mean(hist["recon_l1"][:q])) + 0.05, hist["recon_l1"]

    vae = ContinuousWaveletVAE(latent_channels=12, downsample_factor=2)
    x = torch.from_numpy(frames[0][0].astype(np.float32)).unsqueeze(0)
    lat = vae.encode_frame(x[0])
    out = refiner(lat.unsqueeze(0)).detach()
    assert out.shape == (1, 3, 32, 32)
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_refiner_bridge_roundtrip():
    torch.manual_seed(2)
    refiner = PerceptualRefiner(latent_channels=12)
    bridge = RefinerVAEBridge(refiner)
    vid = torch.from_numpy(_fluid_frames(1, t=4)[0].astype(np.float32))
    lat = bridge.encode_video_to_latent(vid)
    assert lat.shape == (4, 12, 16, 16)
    rgb = bridge.decode_latent_to_video(lat)
    assert rgb.shape == (4, 3, 32, 32)
    assert 0.0 <= float(rgb.min()) and float(rgb.max()) <= 1.0
