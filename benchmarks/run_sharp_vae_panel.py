r"""Visual side-by-side panel: repo VAE decode vs Sharp-pipeline reconstruction.

Frames per row exported from applied checkpoints:
- row 1: ground-truth wave clip frames (source)
- row 2: repo Causal3DSpatioTemporalVAE decode (encode -> decode roundtrip)
- row 3: Sharp-pipeline reconstruction (encoder -> Jet2 ODE step -> decoder,
         weights loaded from benchmarks/sharp_wave_pipeline_checkpoint.pt)

Each row is written as 8 individual PNGs (panel_row{r}_frame{i:02d}.png) and one
stacked contact sheet (sharp_vae_panel.png), plus measured SSIM/PSNR per path in
results JSON. Status: audit-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image  # noqa: E402  (needed by save_strip below)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))

import run_sharp_wave_pipeline as base  # noqa: E402
from jetflow_video.bridge import Causal3DSpatioTemporalVAE  # noqa: E402
from apedp_dyn.video.eval_gate import compute_psnr, compute_ssim  # noqa: E402

OUT = _BENCH / "sharp_vae_panel"
DEVICE = "cpu"


def save_strip(frames: np.ndarray, tag: str) -> list[str]:
    """frames: (T, 3, H, W) uint8 -> write per-frame PNGs and return paths."""
    print(f"[save_strip] {tag}: type={type(frames).__name__} dtype={frames.dtype} shape={tuple(frames.shape)} elem0={tuple(np.asarray(frames[0]).shape)}", flush=True)
    paths = []
    for i in range(frames.shape[0]):
        p = OUT / f"panel_{tag}_frame{i:02d}.png"
        Image.fromarray(frames[i].transpose(1, 2, 0)).save(p)  # (T,3,H,W) strips -> per-frame (H,W,3) RGB
        paths.append(str(p))
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    video = base.build_clip()               # (1, T, C, H, W) source, B-first
    gt_tw = video[0]                        # (T, 3, H, W)

    # ---- row 2: repo VAE roundtrip ----
    vae = Causal3DSpatioTemporalVAE(in_channels=3, latent_channels=12)
    with torch.no_grad():
        lat, _ = vae.encode_video(video)
        vae_recon = vae.decode_video(lat, target_frames=video.shape[1])   # (1, T, 3, H, W)
    vae_tw = vae_recon[0]

    # ---- row 3: sharp pipeline from checkpoint ----
    ck = torch.load(_BENCH / "sharp_wave_pipeline_checkpoint.pt", map_location=DEVICE)
    enc = base.SharperVideoEncoder(3, 8).to(DEVICE)
    dec = base.SharperVideoDecoder(8, 3).to(DEVICE)
    model = base.JetFlowVideoModel(base.JetFlowModelConfig(latent_channels=8, hidden_dim=128))
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"]); model.load_state_dict(ck["model"])
    with torch.no_grad():
        target_wave = video.permute(0, 2, 1, 3, 4)      # (B, C, T, H, W)
        lat_s = enc(target_wave)
        lat_jet = lat_s.permute(0, 2, 1, 3, 4)
        pred, _, _ = base.JetFlowStepper(base.JetFlowConfig(num_steps=5, order=2)).step(
            lambda x, t: model(x, torch.tensor([t], device=DEVICE),
                               text_tokens=model.text_encoder.encode_prompts(
                                   "sharp moving wave front", device=DEVICE, dtype=torch.float32).detach()),
            lat_jet, t=0.0, dt=0.5)
        sharp_recon = dec(pred.permute(0, 2, 1, 3, 4))   # (1, 3, T, H, W)
    sharp_tw = sharp_recon[0].permute(1, 0, 2, 3)        # -> (T, 3, H, W)

    # ---- instruments per path ----
    results = {
        "gt_layout_note": "all tensors aligned to (T,3,H,W) before scoring",
    }
    for name, tw in [("repo_vae", vae_tw), ("sharp_pipeline", sharp_tw)]:
        results[name] = {"ssim": round(float(compute_ssim(gt_tw.numpy(), tw.numpy())), 6),
                         "psnr_db": round(float(compute_psnr(gt_tw.numpy(), tw.numpy())), 3)}

    # ---- PNG export: per-frame files + stacked panel ----
    rows = [("gt", (gt_tw.numpy() * 255).round().astype(np.uint8)),
            ("repo_vae", (vae_tw.numpy() * 255).round().astype(np.uint8)),
            ("sharp_pipeline", (sharp_tw.numpy() * 255).round().astype(np.uint8))]
    paths = []
    for tag, strip in rows:
        print(f"[export] {tag}: {tuple(strip.shape)}", flush=True)
        paths.extend(save_strip(np.ascontiguousarray(strip), tag))

    from PIL import Image  # (also available at module import; kept for explicit grouping)
    panels = []
    for tag, strip in rows:
        panels.append(np.concatenate([np.asarray(Image.open(p)) for p in paths if f"panel_{tag}_" in Path(p).name], axis=1))
    sheet = np.concatenate(panels, axis=0)
    Image.fromarray(sheet).save(OUT / "sharp_vae_panel.png")

    results["files"] = {"panel": str(OUT / "sharp_vae_panel.png"), "per_frame_count": len(paths)}
    (OUT / "panel_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
