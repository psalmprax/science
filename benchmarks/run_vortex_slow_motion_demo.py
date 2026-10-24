r"""Create the claimed 8 -> 24-frame continuous slow-motion demo as a real artifact.

Reproduces the "dense frame generation from 8 source frames" claim with the
REPO's actual mechanism (learned in the 2026-10-20 verification session):
`VideoVAEBridge.encode_video_to_tt` (Tensor-Train compression, rank 16) followed by
`VideoVAEBridge.decode_continuous_frame_rgb` (C^1 Catmull-Rom spline over the
temporal TT core, evaluated on device without dense unrolling, O(R^2 C + R^2 H + RW)).

Outputs under `benchmarks/vortex_slow_motion/`:
- 24 PNG frames (`slow_frame_00..23.png`)
- a looping GIF at 12 FPS playback (`vortex_slow_motion_12fps.gif`)
- a contact sheet (`slow_motion_contact_sheet.png`, 2 rows x 12 columns)
- results JSON (`slow_motion_results.json`)

Status: audit-only (verification harness; the underlying interpolator stack is `experimental`)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "packages" / "jetflow-video")):
    if p not in sys.path:
        sys.path.insert(0, p)

from apedp_dyn.video.continuous_interpolator import ContinuousTemporalInterpolator
from jetflow_video.bridge import VideoVAEBridge

OUT_DIR = Path(__file__).resolve().parent / "vortex_slow_motion"
N_SOURCE_FRAMES = 8
N_RENDERED_FRAMES = 24
PLAYBACK_FPS = 12.0  # 83 ms per frame at 12 FPS playback, matching the recorded claim


def build_vortex_clip() -> torch.Tensor:
    """Deterministic two-vortex dye-transport clip, unchanged from the debugging session."""
    xx, yy = np.meshgrid(np.linspace(-3, 3, 64), np.linspace(-3, 3, 64))
    frames = []
    for ts in range(N_SOURCE_FRAMES):
        x1, y1 = -1.0 + 0.2 * ts, -0.5 + 0.1 * ts
        x2, y2 = 1.0 - 0.2 * ts, 0.5 - 0.1 * ts
        flow = np.exp(-((xx - x1) ** 2 + (yy - y1) ** 2) / 0.8) - np.exp(
            -((xx - x2) ** 2 + (yy - y2) ** 2) / 0.8
        )
        r, g, b = (flow + 1) / 2, np.sin(flow * np.pi) * 0.5 + 0.5, np.cos(flow * np.pi) * 0.5 + 0.5
        frames.append(np.stack([r, g, b], axis=0))
    return torch.tensor(np.stack(frames), dtype=torch.float32)  # (T, 3, H, W)


def main() -> None:
    import time

    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic source clip (8 frames). Section 1.1 of the video debugging record.
    video = build_vortex_clip()

    # 2) Encode into the Tensor-Train latent (rank 16, the artifact-standard rank).
    bridge = VideoVAEBridge(
        use_diffusers=False, use_causal_3d_vae=True,
        latent_channels=12, downsample_factor=2, temporal_downsample=1,
    )
    with torch.no_grad():
        tt = bridge.encode_video_to_tt(video, max_rank=16)

    # 3) Render 24 continuous frames without retraining: decoded from the TT cores.
    taus = np.linspace(0.0, 1.0, N_RENDERED_FRAMES)
    t_render = time.perf_counter()
    rendered = []
    with torch.no_grad():
        for tau in taus:
            rendered.append(bridge.decode_continuous_frame_rgb(tt, tau=float(tau)))
    rendered = torch.stack(rendered)  # (24, 3, H, W)
    render_ms = (time.perf_counter() - t_render) * 1000.0

    # 4) Flicker metric on the RENDERED sequence: ||D_t^2 u|| (same instrument as the
    #    repo's FPS benchmark), computed on the array the viewer watches.
    smoothness = ContinuousTemporalInterpolator.measure_temporal_smoothness(
        rendered.numpy().astype(np.float64)
    )

    # 5) Save artifacts: PNGs, contact sheet (2 rows x 12), and a 12 FPS looping GIF.
    from PIL import Image

    paths = []
    for i in range(N_RENDERED_FRAMES):
        img = (rendered[i].numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        p = OUT_DIR / f"slow_frame_{i:02d}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))

    thumbs = [np.asarray(Image.open(p)) for p in paths]
    top = np.concatenate(thumbs[:12], axis=1)
    bottom = np.concatenate(thumbs[12:], axis=1)
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(OUT_DIR / "slow_motion_contact_sheet.png")

    rgb_gif = [
        Image.fromarray((r.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8))
        for r in rendered
    ]
    gif_path = OUT_DIR / "vortex_slow_motion_12fps.gif"
    rgb_gif[0].save(
        gif_path,
        save_all=True,
        append_images=rgb_gif[1:],
        duration=int(1000.0 / PLAYBACK_FPS),  # ms per frame
        loop=0,
    )

    # 6) Results record (measured, no invented thresholds).
    results = {
        "demo": "vortex_slow_motion_8to24",
        "interpolator": "VideoVAEBridge.decode_continuous_frame_rgb (C^1 Catmull-Rom spline over the TT temporal core)",
        "source_frames": N_SOURCE_FRAMES,
        "rendered_frames": N_RENDERED_FRAMES,
        "playback_fps": PLAYBACK_FPS,
        "tt_bond_ranks": list(tt.bond_ranks),
        "tt_compression_ratio": tt.compression_ratio,
        "tt_mem_saving_pct": tt.memory_saving_pct,
        "total_render_ms": round(render_ms, 2),
        "ms_per_frame": round(render_ms / N_RENDERED_FRAMES, 3),
        "temporal_smoothness_dt2_l2": float(smoothness),
        "frames_saved": len(paths),
        "wall_total_s": round(time.perf_counter() - t_start, 2),
        "artifacts": {
            "gif": str(gif_path),
            "contact_sheet": str(OUT_DIR / "slow_motion_contact_sheet.png"),
            "results_json": str(OUT_DIR / "slow_motion_results.json"),
        },
    }
    (OUT_DIR / "slow_motion_results.json").write_text(json.dumps(results, indent=2))

    print(f"repo-rendered {N_RENDERED_FRAMES} continuous frames from {N_SOURCE_FRAMES} source frames")
    print(f"  TT rank: {tt.bond_ranks} | compression: {tt.compression_ratio:.1f}x | mem saved: {tt.memory_saving_pct:.1f}%")
    print(f"  render: {render_ms:.1f} ms total ({render_ms / N_RENDERED_FRAMES:.2f} ms/frame)")
    print(f"  temporal smoothness ||D_t^2 u||: {smoothness:.3e}")
    print(f"[Saved] 24 PNGs + contact sheet + 12 FPS GIF + results -> {OUT_DIR}")


if __name__ == "__main__":
    main()
