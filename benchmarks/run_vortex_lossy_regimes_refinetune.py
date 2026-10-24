r"""Lossy-regime perceptual finetune: can the LSGAN refiner buy back SSIM?

Follow-up to `benchmarks/vortex_perceptual_finetune_results.json`, which measured
ZERO SSIM gain on the lossless vortex clip (the wavelet VAE is near-invertible,
so nothing was there to restore). This script repeats the finetune in TWO
genuinely lossy regimes, both paired-supervised (degraded latent -> clean GT):

A. JPEG-compressed frames: frames at quality Q_train (PIL, 4:2:0), wave-latents
   re-encoded from the decoded JPEG frames; baseline = plain Haar decode of the
   degraded latents (== the JPEG frame itself, measured).
B. Noise-degraded latents: Gaussian noise at sigma = SN*std(wave) added to clean
   latents; baseline = plain Haar decode of noisy latents.

Targets are ALWAYS the clean vortex clip. Loss composition mirrors
`train_perceptual_stack` (recon L1 + gradient penalty lambda_grad=0.5 + LSGAN
lambda_adv=0.1, AdamW 2e-4 both sides, no extra knobs invented).

Heldout: evaluation at an UNSEEN degradation strength per regime (different JPEG
quality / different sigma). Honest scope: one 8-frame synthetic clip; the refiner
can only be probed at degradation strengths near the trained one, and the clip
content itself is in-distribution. Status: audit-only.
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apedp_dyn.video.eval_gate import compute_ssim, compute_psnr
from jetflow_video.bridge import ContinuousWaveletVAE
from jetflow_video.perceptual_decoder import PatchDiscriminator, PerceptualRefiner

OUT_DIR = Path(__file__).resolve().parent
STEPS = 60
SEED = 3
LR = 2e-4
LAMBDA_GRAD = 0.5
LAMBDA_ADV = 0.1


def build_vortex_clip() -> torch.Tensor:
    xx, yy = np.meshgrid(np.linspace(-3, 3, 64), np.linspace(-3, 3, 64))
    frames = []
    for ts in range(8):
        x1, y1 = -1.0 + 0.2 * ts, -0.5 + 0.1 * ts
        x2, y2 = 1.0 - 0.2 * ts, 0.5 - 0.1 * ts
        flow = np.exp(-((xx - x1) ** 2 + (yy - y1) ** 2) / 0.8) - np.exp(
            -((xx - x2) ** 2 + (yy - y2) ** 2) / 0.8
        )
        r, g, b = (flow + 1) / 2, np.sin(flow * np.pi) * 0.5 + 0.5, np.cos(flow * np.pi) * 0.5 + 0.5
        frames.append(np.stack([r, g, b], axis=0))
    return torch.tensor(np.stack(frames), dtype=torch.float32)


def jpeg_roundtrip(frame: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG-compress (4:2:0) and decode one (3, H, W) frame tensor."""
    from PIL import Image

    arr = (frame.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)  # PIL 4:2:0 chroma subsampling
    buf.seek(0)
    back = np.asarray(Image.open(buf), dtype=np.float32) / 255.0
    return torch.tensor(back.transpose(2, 0, 1))


def haar_decode(vae: ContinuousWaveletVAE, lat: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([vae.decode_frame(lat[i:i + 1]) for i in range(lat.shape[0])], dim=0).clamp(0, 1)


def refiner_decode(refiner: PerceptualRefiner, lat: torch.Tensor) -> torch.Tensor:
    refiner.eval()
    with torch.no_grad():
        return refiner(lat).clamp(0, 1)


def gradient_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    dpx, dtx = pred[:, :, 1:, :] - pred[:, :, :-1, :], true[:, :, 1:, :] - true[:, :, :-1, :]
    dpy, dty = pred[:, :, :, 1:] - pred[:, :, :, :-1], true[:, :, :, 1:] - true[:, :, :, :-1]
    return F.l1_loss(dpx, dtx) + F.l1_loss(dpy, dty)


def train_paired(
    lat: torch.Tensor, target: torch.Tensor, batch_frames: int = 8
) -> tuple[PerceptualRefiner, dict]:
    """Paired-supervised LSGAN loop mirroring train_perceptual_stack's composition."""
    torch.manual_seed(SEED)
    refiner = PerceptualRefiner(latent_channels=12, downsample_factor=2)
    disc = PatchDiscriminator(in_channels=3)
    opt_g = torch.optim.AdamW(refiner.parameters(), lr=LR)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=LR, weight_decay=0.01)
    hist = {"recon_l1": [], "grad": [], "adv_g": []}
    t0 = time.perf_counter()
    for _ in range(STEPS):
        lat_b, true_b = lat, target
        proj = refiner(lat_b)
        # discriminator step (LSGAN)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            proj_det = refiner(lat_b)
        score_fake, score_real = disc(proj_det), disc(true_b)
        d_loss = torch.mean((score_real - 1.0) ** 2) + torch.mean(score_fake ** 2)
        d_loss.backward()
        opt_d.step()
        # generator step
        opt_g.zero_grad(set_to_none=True)
        proj = refiner(lat_b)
        recon_l1 = F.l1_loss(proj, true_b)
        grad = gradient_loss(proj, true_b)
        adv = torch.mean((disc(proj) - 1.0) ** 2)
        g_loss = recon_l1 + LAMBDA_GRAD * grad + LAMBDA_ADV * adv
        g_loss.backward()
        opt_g.step()
        hist["recon_l1"].append(float(recon_l1))
        hist["grad"].append(float(grad))
        hist["adv_g"].append(float(adv))
    hist["wall_s"] = round(time.perf_counter() - t0, 1)
    return refiner, hist


def score(rgb: torch.Tensor, gt: torch.Tensor) -> dict:
    return {"ssim": round(float(compute_ssim(gt.numpy(), rgb.numpy())), 6),
            "psnr_db": round(float(compute_psnr(gt.numpy(), rgb.numpy())), 3)}


def main() -> None:
    torch.manual_seed(SEED)
    gt = build_vortex_clip()  # clean (8, 3, 64, 64)
    vae = ContinuousWaveletVAE(in_channels=3, latent_channels=12, downsample_factor=2)
    with torch.no_grad():
        wave = torch.cat([vae.encode_frame(gt[i:i + 1]) for i in range(8)], dim=0)
    lat_std = float(wave.std())
    results: dict = {"seed": SEED, "steps": STEPS, "latent_std": lat_std}

    # sanity: invertibility of the instrument itself (baseline must reproduce the degradation)
    results["invertibility_check"] = score(haar_decode(vae, wave), gt)

    # ---------- regime A: JPEG ----------
    q_train = 15
    jpeg_frames = torch.cat([jpeg_roundtrip(gt[i], q_train).unsqueeze(0) for i in range(8)])
    with torch.no_grad():
        wave_jpeg = torch.cat([vae.encode_frame(jpeg_frames[i:i + 1]) for i in range(8)])
    pre_jpeg = score(haar_decode(vae, wave_jpeg), gt)   # == the JPEG frame itself (inverted VAE)
    ref_j, hist_j = train_paired(wave_jpeg, gt)
    post_jpeg = score(refiner_decode(ref_j, wave_jpeg), gt)
    # heldout: unseen quality
    q_eval = 60
    jpeg_held = torch.cat([jpeg_roundtrip(gt[i], q_eval).unsqueeze(0) for i in range(8)])
    with torch.no_grad():
        wave_jpeg_held = torch.cat([vae.encode_frame(jpeg_held[i:i + 1]) for i in range(8)])
    pre_held_j = score(haar_decode(vae, wave_jpeg_held), gt)
    post_held_j = score(refiner_decode(ref_j, wave_jpeg_held), gt)
    results["jpeg"] = {
        "q_train": q_train, "baseline": pre_jpeg, "refined": post_jpeg,
        "ssim_gain": round(post_jpeg["ssim"] - pre_jpeg["ssim"], 6),
        "psnr_gain_db": round(post_jpeg["psnr_db"] - pre_jpeg["psnr_db"], 3),
        "heldout_q_eval": q_eval, "heldout_baseline": pre_held_j, "heldout_refined": post_held_j,
        "heldout_ssim_gain": round(post_held_j["ssim"] - pre_held_j["ssim"], 6),
        "hist_recon_l1_first_last": [hist_j["recon_l1"][0], hist_j["recon_l1"][-1]],
        "hist_grad_first_last": [hist_j["grad"][0], hist_j["grad"][-1]],
        "train_wall_s": hist_j["wall_s"],
    }

    # ---------- regime B: noisy latents ----------
    sigma_n = 0.15 * lat_std
    g = torch.Generator().manual_seed(SEED)
    noisy = wave + sigma_n * torch.randn(wave.shape, generator=g)
    pre_n = score(haar_decode(vae, noisy), gt)
    ref_n, hist_n = train_paired(noisy, gt)
    post_n = score(refiner_decode(ref_n, noisy), gt)
    sigma_held = 0.05 * lat_std
    g2 = torch.Generator().manual_seed(SEED + 1)
    noisy_held = wave + sigma_held * torch.randn(wave.shape, generator=g2)
    pre_n_held = score(haar_decode(vae, noisy_held), gt)
    post_n_held = score(refiner_decode(ref_n, noisy_held), gt)
    results["noisy_latent"] = {
        "sigma_train": sigma_n, "baseline": pre_n, "refined": post_n,
        "ssim_gain": round(post_n["ssim"] - pre_n["ssim"], 6),
        "psnr_gain_db": round(post_n["psnr_db"] - pre_n["psnr_db"], 3),
        "heldout_sigma": sigma_held, "heldout_baseline": pre_n_held, "heldout_refined": post_n_held,
        "heldout_ssim_gain": round(post_n_held["ssim"] - pre_n_held["ssim"], 6),
        "hist_recon_l1_first_last": [hist_n["recon_l1"][0], hist_n["recon_l1"][-1]],
        "hist_grad_first_last": [hist_n["grad"][0], hist_n["grad"][-1]],
        "train_wall_s": hist_n["wall_s"],
    }

    out = OUT_DIR / "vortex_lossy_regimes_refinetune_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
