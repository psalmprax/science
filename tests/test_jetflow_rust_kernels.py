"""Rust/Python parity for the native JetFlow kernels."""
from __future__ import annotations

import pytest
import torch

apedp_kernel = pytest.importorskip("apedp_kernel")
from jetflow_video import stepper, tensor_train

pytestmark = pytest.mark.skipif(
    not (
        hasattr(apedp_kernel, "jetflow_compute_acceleration")
        and hasattr(apedp_kernel, "jetflow_step_integration")
        and hasattr(apedp_kernel, "jetflow_tensor_train_compress")
    ),
    reason="apedp_kernel built without JetFlow kernels",
)


def _velocity(z: torch.Tensor, t: float) -> torch.Tensor:
    return torch.sin(z) + t * z


def test_acceleration_matches_python_reference(monkeypatch):
    torch.manual_seed(0)
    z = torch.randn(2, 3, 4, dtype=torch.float64)
    v = _velocity(z, 0.3)
    cfg = stepper.JetFlowConfig(num_steps=2, order=2)
    s = stepper.JetFlowStepper(cfg)

    rust_out = s.compute_acceleration_torch(_velocity, z, 0.3, v, dt=0.1)
    monkeypatch.setattr(stepper, "_RUST_AVAILABLE", False)
    py_out = s.compute_acceleration_torch(_velocity, z, 0.3, v, dt=0.1)

    assert rust_out.shape == z.shape
    assert torch.allclose(rust_out, py_out, atol=1e-10)


def test_step_matches_python_reference(monkeypatch):
    torch.manual_seed(1)
    z = torch.randn(5, 2, dtype=torch.float64)
    s = stepper.JetFlowStepper(stepper.JetFlowConfig(num_steps=2, order=2))

    rust_z, _, _ = s.step(_velocity, z, 0.2, 0.25)
    monkeypatch.setattr(stepper, "_RUST_AVAILABLE", False)
    py_z, _, _ = s.step(_velocity, z, 0.2, 0.25)

    assert torch.allclose(rust_z, py_z, atol=1e-10)


def test_tt_compress_rust_matches_python_reconstruction(monkeypatch):
    torch.manual_seed(2)
    latent = torch.randn(4, 3, 5, 6, dtype=torch.float64)

    rust_tt = tensor_train.compress_video_latent_torch(latent, max_rank=64, tolerance=0.0)
    rust_err = (tensor_train.decompress_video_latent_torch(rust_tt) - latent).abs().max()

    monkeypatch.setattr(tensor_train, "_RUST_TT_AVAILABLE", False)
    py_tt = tensor_train.compress_video_latent_torch(latent, max_rank=64, tolerance=0.0)
    py_err = (tensor_train.decompress_video_latent_torch(py_tt) - latent).abs().max()

    assert rust_tt.bond_ranks == py_tt.bond_ranks
    assert rust_err < 1e-10
    assert py_err < 1e-10
    assert [c.shape for c in rust_tt.cores] == [c.shape for c in py_tt.cores]


def test_tt_truncated_rank_respects_max_rank():
    torch.manual_seed(3)
    latent = torch.randn(4, 3, 5, 6, dtype=torch.float64)
    tt = tensor_train.compress_video_latent_torch(latent, max_rank=2, tolerance=0.0)
    assert max(tt.bond_ranks) <= 2
