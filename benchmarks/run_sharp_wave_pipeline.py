r"""Adjudication harness for the pasted 'Sharper' video pipeline notebook.

Runs the user-pasted 80-epoch pipeline (SharperVideoEncoder/SharpDecoder +
JetFlowVideoModel ODE step + MSE + 0.3*L1 compound loss) with ONLY these
disclosed adaptations:
1. `text_embeddings` was undefined in the snippet (NameError at epoch 1);
   built via the model's own text encoder.
2. `new_video_tensor` was pasted-out-of-context; reconstructed as an 8-frame
   wave-front clip (cos/sin channels, same analytically-类比 generator style).
3. Device cpu (this host has no CUDA); identical shapes/kernel geometry.
4. `text_embeddings` DETACHED: the undetached text-encoder output (whose params are
   optimized) is a non-leaf in every epoch's graph -> epoch 2 crashes with
   "backward through the graph a second time" (measured). detach() = constant text tokens.

Output: `benchmarks/sharp_wave_pipeline_results.json` — full loss history,
final SSIM/PSNR of the reconstructed wave vs target (decode-roundtrip quality),
and wall-clock. Status: audit-only.
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

from apedp_dyn.video.eval_gate import compute_ssim, compute_psnr
from jetflow_video import JetFlowConfig, JetFlowModelConfig, JetFlowStepper, JetFlowVideoModel

OUT = Path(__file__).resolve().parent / "sharp_wave_pipeline_results.json"
DEVICE = "cpu"
EPOCHS = 80


class SharperVideoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, latent_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
        )
    def forward(self, x):
        return self.encoder(x)


class SharperVideoDecoder(nn.Module):
    def __init__(self, latent_channels=8, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(x)


def build_clip():
    xx, yy = np.meshgrid(np.linspace(-3, 3, 64), np.linspace(-3, 3, 64))
    frames = []
    for ts in range(8):
        flow = np.cos(2.5 * np.pi * ts / 8) * np.exp(-(xx ** 2 + yy ** 2) / 3.5)
        frames.append(np.stack([(flow + 1) / 2, np.sin(flow * np.pi) * 0.5 + 0.5, np.cos(flow * np.pi) * 0.5 + 0.5]))
    return torch.tensor(np.stack(frames), dtype=torch.float32).unsqueeze(0)


def main():
    torch.manual_seed(3)
    sharp_encoder = SharperVideoEncoder(3, 8).to(DEVICE)
    sharp_decoder = SharperVideoDecoder(8, 3).to(DEVICE)
    model = JetFlowVideoModel(JetFlowModelConfig(latent_channels=8, hidden_dim=128)).to(DEVICE)
    sharp_parameters = list(sharp_encoder.parameters()) + list(model.parameters()) + list(sharp_decoder.parameters())
    sharp_optimizer = optim.Adam(sharp_parameters, lr=1e-3)
    mse_loss, l1_loss = nn.MSELoss(), nn.L1Loss()

    target_wave_video = build_clip().permute(0, 2, 1, 3, 4).to(DEVICE)
    text_embeddings = model.text_encoder.encode_prompts("sharp moving wave front", device=DEVICE, dtype=torch.float32).detach()
    stepper = JetFlowStepper(JetFlowConfig(num_steps=5, order=2))

    loss_history = []
    t_start = time.perf_counter()
    print("--- Training Sharp High-Capacity Video Pipeline on Wave Front ---", flush=True)
    for epoch in range(1, EPOCHS + 1):
        sharp_optimizer.zero_grad()
        latents = sharp_encoder(target_wave_video)
        latents_jet_fmt = latents.permute(0, 2, 1, 3, 4).to(DEVICE)

        def local_sharp_ode_fn(x, t_val):
            t_tensor = torch.tensor([t_val], device=DEVICE)
            return model(x, t_tensor, text_tokens=text_embeddings)

        pred_latents_device, _, _ = stepper.step(local_sharp_ode_fn, latents_jet_fmt, t=0.0, dt=0.5)
        pred_latents = pred_latents_device.to(DEVICE)
        reconstructed_wave = sharp_decoder(pred_latents.permute(0, 2, 1, 3, 4))
        loss = mse_loss(reconstructed_wave, target_wave_video) + 0.3 * l1_loss(reconstructed_wave, target_wave_video)
        loss.backward()
        sharp_optimizer.step()
        item = float(loss.item())
        loss_history.append(item)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}/80 | Sharp Compound Loss: {item:.6f}", flush=True)

    wall = time.perf_counter() - t_start
    # loss history FIRST: no downstream failure can erase the expensive phase
    results = {
        "adaptations": [
            "text_embeddings built via model.text_encoder (undefined in snippet)",
            "new_video_tensor reconstructed as 8-frame wave clip (out of context)",
            "device cpu (no CUDA on host)",
            "text_embeddings.detach(): undetached encoder output made epoch-2 backward crash (measured)",
        ],
        "instrument_fix": "eval_gate scores (T,3,H,W); pipeline tensors are (B,C,T,H,W) -> permuted before scoring",
        "epochs": EPOCHS,
        "loss_history": loss_history,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "wall_seconds": round(wall, 1),
        "device": DEVICE,
    }
    OUT.write_text(json.dumps(results, indent=2))
    # checkpoint (harness variable is `model`, not the snippet's `sharp_video_flow_model`)
    torch.save({"encoder": sharp_encoder.state_dict(), "decoder": sharp_decoder.state_dict(),
                "model": model.state_dict()},
               Path(__file__).resolve().parent / "sharp_wave_pipeline_checkpoint.pt")
    # final-quality instruments on the (T,3,H,W) layout the gate expects
    gt_tw = target_wave_video[0].permute(1, 0, 2, 3).numpy()          # (8,3,64,64)
    rec_tw = reconstructed_wave[0].detach().permute(1, 0, 2, 3).numpy()
    ssim = float(compute_ssim(gt_tw, rec_tw))
    psnr = float(compute_psnr(gt_tw, rec_tw))
    results["ssim_vs_target"] = round(ssim, 6)
    results["psnr_vs_target_db"] = round(psnr, 3)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[Saved] {OUT}")
    print(f"final: SSIM {ssim:.6f} | PSNR {psnr:.2f} dB | wall {wall:.0f} s")


if __name__ == "__main__":
    main()
