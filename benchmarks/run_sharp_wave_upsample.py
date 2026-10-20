r"""Checkerboard-artifact test: ConvTranspose3d vs Upsample+Conv SharperVideoDecoder.

Repeats the adjudication pipeline (`run_sharp_wave_pipeline.py`) for 80 epochs with
ONE controlled change: the transposed-conv upsampler is replaced by
trilinear Upsample((1,2,2)) + Conv3d (identical parameter count 8*32*27+32).

Control design: the encoder and the JetFlow backbone are constructed in the SAME
RNG order as the base harness (the base decoder is still instantiated, merely
unused, preserving the parameter-init random stream) -> encoder and backbone
weights are bit-identical to the base run; the decoder IS the only variable.

Instruments (eval_gate expects (T,3,H,W); pipeline tensors are (B,C,T,H,W)):
- SSIM / PSNR of the final reconstruction vs target, same as base run.
- checkerboard-mode metric: mean abs value of the diagonal Laplacian
  |f00 + f11 - f01 - f10| over H/W 2x2 groups (isolates the (pi,pi) grid mode
  that transposed convolutions classically imprint), computed for BOTH the
  base checkpoint's reconstruction and this variant's.

Artifacts:
- benchmarks/sharp_wave_upsample_results.json
- benchmarks/sharp_wave_upsample_checkpoint.pt (variant encoder/decoder/model)
Status: audit-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))

import run_sharp_wave_pipeline as base  # noqa: E402  (module-level defs only; main() guarded)

from apedp_dyn.video.eval_gate import compute_psnr, compute_ssim  # noqa: E402

DEVICE = "cpu"
EPOCHS = 80
OUT = _BENCH / "sharp_wave_upsample_results.json"


class SharperVideoDecoderUpsampled(nn.Module):
    """Same output geometry as the base decoder; trilinear upsample instead of ConvTranspose3d."""

    def __init__(self, latent_channels: int = 8, out_channels: int = 3) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def checker_metric(rgb_tw: np.ndarray) -> float:
    """Mean |f00 + f11 - f01 - f10| over 2x2 groups of the (T,3,H,W) frames (checker mode energy)."""
    f = np.asarray(rgb_tw, dtype=np.float64)
    x00, x01 = f[:, :, 0:-1:2, 0:-1:2], f[:, :, 0:-1:2, 1::2]
    x10, x11 = f[:, :, 1::2, 0:-1:2], f[:, :, 1::2, 1::2]
    return float(np.mean(np.abs((x00 + x11) - (x01 + x10))))


def to_tw_layout(t_bcth: torch.Tensor) -> np.ndarray:
    """(B,C,T,H,W) -> (T,3,H,W) numpy for the eval_gate instrument."""
    return t_bcth[0].permute(1, 0, 2, 3).numpy()


def main() -> None:
    torch.manual_seed(3)
    sharp_encoder = base.SharperVideoEncoder(3, 8).to(DEVICE)
    base_decoder_rng_placeholder = base.SharperVideoDecoder(8, 3).to(DEVICE)  # RNG parity, unused
    model = base.JetFlowVideoModel(base.JetFlowModelConfig(latent_channels=8, hidden_dim=128)).to(DEVICE)
    decoder_up = SharperVideoDecoderUpsampled(8, 3).to(DEVICE)

    sharp_parameters = list(sharp_encoder.parameters()) + list(model.parameters()) + list(decoder_up.parameters())
    sharp_optimizer = optim.Adam(sharp_parameters, lr=1e-3)
    mse_loss, l1_loss = nn.MSELoss(), nn.L1Loss()

    target_wave_video = base.build_clip().permute(0, 2, 1, 3, 4).to(DEVICE)
    text_embeddings = model.text_encoder.encode_prompts(
        "sharp moving wave front", device=DEVICE, dtype=torch.float32
    ).detach()
    stepper = base.JetFlowStepper(base.JetFlowConfig(num_steps=5, order=2))

    t_start = time.perf_counter()
    print("--- Checkerboard test: Upsample+Conv decoder, 80 epochs ---", flush=True)
    for epoch in range(1, EPOCHS + 1):
        sharp_optimizer.zero_grad()
        latents = sharp_encoder(target_wave_video)
        latents_jet_fmt = latents.permute(0, 2, 1, 3, 4).to(DEVICE)

        def local_sharp_ode_fn(x, t_val):
            t_tensor = torch.tensor([t_val], device=DEVICE)
            return model(x, t_tensor, text_tokens=text_embeddings)

        pred_latents_device, _, _ = stepper.step(local_sharp_ode_fn, latents_jet_fmt, t=0.0, dt=0.5)
        reconstructed_wave = decoder_up(pred_latents_device.permute(0, 2, 1, 3, 4))
        loss = mse_loss(reconstructed_wave, target_wave_video) + 0.3 * l1_loss(reconstructed_wave, target_wave_video)
        loss.backward()
        sharp_optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}/80 | Sharp Compound Loss: {float(loss.item()):.6f}", flush=True)

    wall = time.perf_counter() - t_start
    results = {
        "variant": "Upsample(trilinear,(1,2,2))+Conv3d vs base ConvTranspose3d",
        "param_count_equal": True,
        "rng_control": "encoder+backbone bit-identical by construction order",
        "epochs": EPOCHS,
        "loss_last": float(loss.item()),
        "wall_seconds": round(wall, 1),
        "device": DEVICE,
    }
    OUT.write_text(json.dumps(results, indent=2))

    torch.save({"encoder": sharp_encoder.state_dict(), "decoder": decoder_up.state_dict(),
                "model": model.state_dict()}, _BENCH / "sharp_wave_upsample_checkpoint.pt")

    # instruments on (T,3,H,W)
    gt_tw = to_tw_layout(target_wave_video)
    rec_up = to_tw_layout(reconstructed_wave.detach())
    results["ssim_vs_target"] = round(float(compute_ssim(gt_tw, rec_up)), 6)
    results["psnr_vs_target_db"] = round(float(compute_psnr(gt_tw, rec_up)), 3)
    results["checker_metric_variant"] = checker_metric(rec_up)

    # base-checkpoint reconstruction for the paired checker comparison
    ck_base = torch.load(_BENCH / "sharp_wave_pipeline_checkpoint.pt", map_location=DEVICE)
    base_dec = base.SharperVideoDecoder(8, 3).to(DEVICE)
    base_dec.load_state_dict(ck_base["decoder"])
    base_dec.eval()
    with torch.no_grad():
        base_encoder = base.SharperVideoEncoder(3, 8).to(DEVICE)
        base_encoder.load_state_dict(ck_base["encoder"])
        lat = base_encoder(target_wave_video)
        m2 = base.JetFlowVideoModel(base.JetFlowModelConfig(latent_channels=8, hidden_dim=128))
        m2.load_state_dict(ck_base["model"])
        lat_jet = lat.permute(0, 2, 1, 3, 4)
        pred, _, _ = base.JetFlowStepper(base.JetFlowConfig(num_steps=5, order=2)).step(
            lambda x, t: m2(x, torch.tensor([t], device=DEVICE),
                            text_tokens=model.text_encoder.encode_prompts(
                                "sharp moving wave front", device=DEVICE, dtype=torch.float32).detach()),
            lat_jet, t=0.0, dt=0.5)
        rec_base = base_dec(pred.permute(0, 2, 1, 3, 4))
    results["checker_metric_base"] = checker_metric(to_tw_layout(rec_base))
    results["ssim_base"] = round(float(compute_ssim(gt_tw, to_tw_layout(rec_base))), 6)
    results["psnr_base_db"] = round(float(compute_psnr(gt_tw, to_tw_layout(rec_base))), 3)

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
