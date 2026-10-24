"""Rust bridge contracts for the Noether fast-path retained from PR #31."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from jetflow_video import noether


@pytest.fixture
def rust_kernel(monkeypatch):
    kernel = Mock()
    monkeypatch.setattr(noether, "apedp_kernel", SimpleNamespace(jetflow_noether_project=kernel))
    monkeypatch.setattr(noether, "_RUST_NOETHER_AVAILABLE", True)
    return kernel


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "channels_last", "negative"])
def test_rust_success_round_trip(rust_kernel, dtype, device, layout):
    velocity = torch.arange(128, dtype=torch.float64, device=device).reshape(2, 4, 4, 4).to(dtype)
    if layout == "transposed":
        velocity = velocity.transpose(-1, -2)
    elif layout == "channels_last":
        velocity = velocity.contiguous(memory_format=torch.channels_last)
    elif layout == "negative":
        velocity = torch._neg_view(velocity)
    original = velocity.clone()

    def project(array):
        assert array.shape == tuple(velocity.shape)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        np.testing.assert_array_equal(array, original.double().cpu().numpy())
        array.fill(7.0)
        return array, 0.125

    rust_kernel.side_effect = project
    engine = noether.NoetherConservationEngine()
    engine.project_divergence_free_2d_torch = Mock(side_effect=AssertionError("Python path used"))
    projected, residual = engine.project_spatial_latent_velocity(velocity)
    rust_kernel.assert_called_once()
    assert projected.shape == velocity.shape
    assert projected.device == velocity.device
    assert projected.dtype == velocity.dtype
    assert torch.equal(projected, torch.full_like(velocity, 7.0))
    assert torch.equal(velocity, original)
    assert residual == 0.125


@pytest.mark.parametrize("failure", [
    RuntimeError("native failure"),
    TypeError("FFI type failure"),
    OSError("native adapter failure"),
    (np.zeros((1, 2, 4, 4)), 0.0),
    (object(), 0.0),
    (np.full((2, 3, 4, 4), "invalid"), 0.0),
    (np.zeros((2, 3, 4, 4), dtype=object), 0.0),
    (np.zeros((2, 3, 4, 4), dtype=np.complex128), 0.0),
    (np.full((2, 3, 4, 4), np.nan), 0.0),
    (np.full((2, 3, 4, 4), np.inf), 0.0),
    (np.zeros((2, 3, 4, 4)), np.nan),
    (np.zeros((2, 3, 4, 4)), -1.0),
    (np.zeros((2, 3, 4, 4)), [0.0]),
    (np.zeros((2, 3, 4, 4)), "invalid"),
    (np.zeros((2, 3, 4, 4)),),
    None,
])
def test_rust_failure_matches_python(rust_kernel, monkeypatch, caplog, failure):
    engine = noether.NoetherConservationEngine()
    velocity = torch.arange(96, dtype=torch.float64).reshape(2, 3, 4, 4)
    monkeypatch.setattr(noether, "_RUST_NOETHER_AVAILABLE", False)
    expected, expected_residual = engine.project_spatial_latent_velocity(velocity)
    monkeypatch.setattr(noether, "_RUST_NOETHER_AVAILABLE", True)

    def project(array):
        array.fill(-999.0)
        if isinstance(failure, Exception):
            raise failure
        return failure

    rust_kernel.side_effect = project
    with caplog.at_level(logging.DEBUG, logger=noether.__name__):
        actual, residual = engine.project_spatial_latent_velocity(velocity)
    rust_kernel.assert_called_once()
    torch.testing.assert_close(actual, expected)
    assert residual == expected_residual
    assert "using PyTorch" in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG


@pytest.mark.parametrize("output_layout", ["list", "negative_strides", "readonly", "big_endian"])
def test_array_convertible_outputs(rust_kernel, output_layout):
    output = np.arange(64, dtype=np.float64).reshape(1, 4, 4, 4)
    if output_layout == "list":
        output = output.tolist()
    elif output_layout == "negative_strides":
        output = output[..., ::-1]
    elif output_layout == "readonly":
        output.setflags(write=False)
    elif output_layout == "big_endian":
        output = output.astype(">f8")
    rust_kernel.return_value = output, 0.0
    projected, _ = noether.NoetherConservationEngine().project_spatial_latent_velocity(
        torch.zeros(1, 4, 4, 4, dtype=torch.float64),
    )
    np.testing.assert_array_equal(projected.numpy(), output)


def test_output_dtype_overflow_falls_back(rust_kernel):
    rust_kernel.return_value = np.full((1, 2, 4, 4), np.finfo(np.float64).max), 0.0
    projected, residual = noether.NoetherConservationEngine().project_spatial_latent_velocity(
        torch.zeros(1, 2, 4, 4, dtype=torch.float32),
    )
    assert torch.count_nonzero(projected) == 0
    assert residual == 0.0


@pytest.mark.parametrize("config", [
    noether.NoetherConfig(preserve_wavelet_subbands=True),
    noether.NoetherConfig(dx=2.0),
    noether.NoetherConfig(dy=2.0),
])
def test_unsupported_config_uses_python(rust_kernel, monkeypatch, config):
    velocity = torch.arange(64, dtype=torch.float64).reshape(2, 2, 4, 4)
    engine = noether.NoetherConservationEngine(config)
    actual, residual = engine.project_spatial_latent_velocity(velocity)
    rust_kernel.assert_not_called()
    monkeypatch.setattr(noether, "_RUST_NOETHER_AVAILABLE", False)
    expected, expected_residual = engine.project_spatial_latent_velocity(velocity)
    torch.testing.assert_close(actual, expected)
    assert residual == expected_residual


def test_training_preserves_autograd(rust_kernel):
    velocity = torch.ones(1, 2, 4, 4, requires_grad=True)
    projected, _ = noether.NoetherConservationEngine().project_spatial_latent_velocity(velocity)
    projected.sum().backward()
    rust_kernel.assert_not_called()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()


def test_no_grad_allows_rust(rust_kernel):
    velocity = torch.ones(1, 2, 4, 4, requires_grad=True)
    rust_kernel.return_value = np.zeros(velocity.shape), 0.0
    with torch.no_grad():
        noether.NoetherConservationEngine().project_spatial_latent_velocity(velocity)
    rust_kernel.assert_called_once()


def test_invalid_dimensions_do_not_call_rust(rust_kernel):
    with pytest.raises(ValueError, match="Expected 4D tensor"):
        noether.NoetherConservationEngine().project_spatial_latent_velocity(torch.zeros(2, 4, 4))
    rust_kernel.assert_not_called()


def test_empty_tensor_skips_rust(rust_kernel):
    velocity = torch.empty(0, 2, 4, 4)
    projected, residual = noether.NoetherConservationEngine().project_spatial_latent_velocity(velocity)
    rust_kernel.assert_not_called()
    assert projected.shape == velocity.shape
    assert residual == 0.0


@pytest.mark.parametrize("unsupported", ["sparse", "integer", "complex"])
def test_unsupported_input_uses_python(rust_kernel, unsupported):
    velocity = torch.ones(1, 1, 4, 4)
    if unsupported == "sparse":
        velocity = velocity.to_sparse()
    elif unsupported == "integer":
        velocity = velocity.to(torch.int64)
    else:
        velocity = velocity.to(torch.complex64)
    projected, residual = noether.NoetherConservationEngine().project_spatial_latent_velocity(velocity)
    rust_kernel.assert_not_called()
    if unsupported == "sparse":
        projected, velocity = projected.to_dense(), velocity.to_dense()
    assert torch.equal(projected, velocity)
    assert residual == 0.0


def test_numpy_conversion_failure_uses_python(rust_kernel, monkeypatch, caplog):
    velocity = torch.zeros(1, 2, 4, 4)
    monkeypatch.setattr(torch.Tensor, "numpy", Mock(side_effect=RuntimeError("conversion failed")))
    with caplog.at_level(logging.DEBUG, logger=noether.__name__):
        projected, residual = noether.NoetherConservationEngine().project_spatial_latent_velocity(
            velocity,
        )
    rust_kernel.assert_not_called()
    assert torch.equal(projected, velocity)
    assert residual == 0.0
    assert "using PyTorch" in caplog.text


@pytest.mark.parametrize("available,kernel", [(False, object()), (True, None)])
def test_missing_extension_uses_python(monkeypatch, available, kernel):
    monkeypatch.setattr(noether, "_RUST_NOETHER_AVAILABLE", available)
    monkeypatch.setattr(noether, "apedp_kernel", kernel)
    projected, residual = noether.NoetherConservationEngine().project_spatial_latent_velocity(
        torch.zeros(1, 2, 4, 4),
    )
    assert torch.count_nonzero(projected) == 0
    assert residual == 0.0
