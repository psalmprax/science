r"""JetFlow-Video CPU Export & Latency Benchmark.

Status: experimental (benchmark harness — writes a versioned JSON artifact)

Pipeline:
1. Export the velocity backbone (JetFlowBackbone) to ONNX (opset 17).
   NOTE: the artifact is a STATIC traced graph per clip configuration
   (frames, spatial size): traced nn.MultiheadAttention reshapes bake in
   intermediate constants, so a dynamic temporal axis is NOT supported;
   regenerate the graph per clip length. The parity gate below verifies the
   exporter generalizes across TWO configs rather than promising dynamic axes.
2. Dynamic int8 quantization weights-only via onnxruntime.quantization.
3. Numeric parity check: ORT fp32 and ORT int8 vs torch eager.
4. Latency: torch eager / ORT fp32 / ORT int8 per forward (n reps, warmup).
5. Sizes: fp32 vs int8 model byte sizes.

Honest-reporting note (established this round): at small hidden sizes int8
dynamic quantization is primarily a SIZE win; the fastest CPU executor can be
ORT fp32. The JSON artifact reports measured latencies, not marketing.

Run:
    python benchmarks/run_jetflow_cpu_export_benchmark.py \
        [--hidden 32] [--layers 1] [--frames 6] [--lat 16] [--reps 25] [--out ...json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "packages" / "jetflow-video"
for p in (str(_REPO), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel


def export_backbone_to_onnx(model: JetFlowVideoModel, out_path: str, *,
                            frames: int, latent_channels: int, lat: int) -> str:
    """Export the velocity backbone as a STATIC traced graph for one clip
    configuration (frames, lat). Regenerate per clip length (see module note)."""
    bb = model.backbone
    bb.eval()
    z = torch.randn(1, frames, latent_channels, lat, lat)
    t = torch.tensor([0.5])
    txt = torch.randn(1, 2, 64)
    torch.onnx.export(
        bb, (z, t, txt), out_path, opset_version=17, dynamo=False,
        input_names=["z", "t", "text_tokens"], output_names=["v"],
    )
    return out_path


def quantize_onnx(in_path: str, out_path: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
    return out_path


def _bench(fn, n: int) -> float:
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def run_benchmark(hidden: int = 32, layers: int = 1, frames: int = 6, lat: int = 16,
                  reps: int = 25, out_path: str | None = None, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = JetFlowVideoModel(JetFlowModelConfig(
        latent_channels=12, hidden_dim=hidden, num_layers=layers)).eval()
    n_params = sum(p.numel() for p in model.parameters())

    tmp_dir = "/tmp"
    fp32_path = os.path.join(tmp_dir, "jetflow_backbone_fp32.onnx")
    int8_path = os.path.join(tmp_dir, "jetflow_backbone_int8.onnx")
    export_backbone_to_onnx(model, fp32_path, frames=frames, latent_channels=12, lat=lat)
    quantize_onnx(fp32_path, int8_path)

    import onnxruntime as ort
    sess32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    z = torch.randn(1, frames, 12, lat, lat)
    txt = torch.randn(1, 2, 64)
    feeds = {"z": z.numpy(), "t": np.array([0.5], dtype=np.float32), "text_tokens": txt.numpy()}
    with torch.no_grad():
        v_t = model.backbone(z, 0.5, text_tokens=txt).numpy()
    v32 = sess32.run(None, feeds)[0]
    v8 = sess8.run(None, feeds)[0]

    # Cross-config exporter parity: a SECOND static graph at a different clip
    # length must also reproduce torch (exporter generality, not a dynamic axis).
    fp32_alt = os.path.join(tmp_dir, "jetflow_backbone_fp32_alt.onnx")
    export_backbone_to_onnx(model, fp32_alt, frames=frames * 2, latent_channels=12, lat=lat)
    sess_alt = ort.InferenceSession(fp32_alt, providers=["CPUExecutionProvider"])
    z16 = torch.randn(1, frames * 2, 12, lat, lat)
    feeds16 = {"z": z16.numpy(), "t": np.array([0.25], dtype=np.float32), "text_tokens": txt.numpy()}
    with torch.no_grad():
        v_t16 = model.backbone(z16, 0.25, text_tokens=txt).numpy()
    v32_16 = sess_alt.run(None, feeds16)[0]

    with torch.no_grad():
        ms_eager = _bench(lambda: model.backbone(z, 0.5, text_tokens=txt), reps)
    ms32 = _bench(lambda: sess32.run(None, feeds), reps)
    ms8 = _bench(lambda: sess8.run(None, feeds), reps)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"hidden_dim": hidden, "num_layers": layers, "frames": frames,
                   "latent_shape": [12, lat, lat], "n_params": int(n_params), "reps": reps},
        "sizes_bytes": {"fp32_onnx": os.path.getsize(fp32_path), "int8_onnx": os.path.getsize(int8_path)},
        "parity": {
            "ort_fp32_vs_torch_max_abs": float(np.abs(v32 - v_t).max()),
            "ort_int8_vs_torch_max_abs": float(np.abs(v8 - v_t).max()),
            "ort_fp32_alt_config_vs_torch_max_abs": float(np.abs(v32_16 - v_t16).max()),
        },
        "latency_ms": {"torch_eager": float(ms_eager), "ort_fp32": float(ms32), "ort_int8": float(ms8)},
    }

    dst = out_path or str(_REPO / "benchmarks" / "jetflow_cpu_export_benchmark_results.json")
    Path(dst).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--lat", type=int, default=16)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--out", type=str, default=None, dest="out_path")
    args = ap.parse_args()
    run_benchmark(**vars(args))
