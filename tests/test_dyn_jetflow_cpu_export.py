r"""ONNX int8 export + CPU latency benchmark tests.

Pins:
1. ONNX export runs (static traced graph per clip config; traced attention
   cannot carry a dynamic temporal axis) and ORT fp32 reproduces torch eager
   within tight tolerance at the traced config; a SECOND export at a different
   temporal length also reproduces torch (exporter generality).
2. Dynamic int8 quantization shrinks the artifact and stays within tolerance
   of torch on this tiny model.
3. Latencies are finite and positive; artifact writes with the documented
   structure.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "packages" / "jetflow-video")):
    if p not in sys.path:
        sys.path.insert(0, p)

from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel

from benchmarks.run_jetflow_cpu_export_benchmark import (
    export_backbone_to_onnx,
    quantize_onnx,
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = JetFlowVideoModel(JetFlowModelConfig(latent_channels=12, hidden_dim=32, num_layers=1)).eval()
    td = tmp_path_factory.mktemp("onnx")
    fp32 = export_backbone_to_onnx(model, str(td / "fp32.onnx"), frames=6, latent_channels=12, lat=16)
    int8 = quantize_onnx(fp32, str(td / "int8.onnx"))
    return model, fp32, int8


def test_export_parity_traced_config_and_second_config(exported, tmp_path):
    model, fp32, _int8 = exported
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    torch.manual_seed(1)
    # Traced configuration: exact parity
    z = torch.randn(1, 6, 12, 16, 16)
    txt = torch.randn(1, 2, 64)
    feeds = {"z": z.numpy(), "t": np.array([0.5], dtype=np.float32), "text_tokens": txt.numpy()}
    with torch.no_grad():
        v_t = model.backbone(z, 0.5, text_tokens=txt).numpy()
    v = sess.run(None, feeds)[0]
    assert np.abs(v - v_t).max() < 1e-5, np.abs(v - v_t).max()

    # Exporter generality: a fresh static graph at T=9 also matches torch
    alt = export_backbone_to_onnx(model, str(tmp_path / "alt_t9.onnx"),
                                  frames=9, latent_channels=12, lat=16)
    sess9 = ort.InferenceSession(alt, providers=["CPUExecutionProvider"])
    z9 = torch.randn(1, 9, 12, 16, 16)
    feeds9 = {"z": z9.numpy(), "t": np.array([0.25], dtype=np.float32), "text_tokens": txt.numpy()}
    with torch.no_grad():
        v_t9 = model.backbone(z9, 0.25, text_tokens=txt).numpy()
    v9 = sess9.run(None, feeds9)[0]
    assert np.abs(v9 - v_t9).max() < 1e-5, np.abs(v9 - v_t9).max()


def test_int8_smaller_and_within_tolerance(exported):
    model, fp32, int8 = exported
    assert os.path.getsize(int8) < os.path.getsize(fp32)
    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    z = torch.randn(1, 6, 12, 16, 16)
    txt = torch.randn(1, 2, 64)
    feeds = {"z": z.numpy(), "t": np.array([0.5], dtype=np.float32), "text_tokens": txt.numpy()}
    with torch.no_grad():
        v_t = model.backbone(z, 0.5, text_tokens=txt).numpy()
    v = sess.run(None, feeds)[0]
    # Tiny model: dynamic weight-int8 quantization error stays well under 1e-2
    assert np.abs(v - v_t).max() < 1e-2, np.abs(v - v_t).max()


def test_latency_benchmark_artifact(tmp_path):
    import importlib
    bench = importlib.import_module("benchmarks.run_jetflow_cpu_export_benchmark")
    out = tmp_path / "cpu_export.json"
    res = bench.run_benchmark(hidden=32, layers=1, frames=6, lat=16, reps=5, out_path=str(out))
    assert out.exists()
    for key in ("timestamp_utc", "config", "sizes_bytes", "parity", "latency_ms"):
        assert key in json.loads(out.read_text())
    for v in res["latency_ms"].values():
        assert v > 0.0
    assert res["sizes_bytes"]["int8_onnx"] < res["sizes_bytes"]["fp32_onnx"]
