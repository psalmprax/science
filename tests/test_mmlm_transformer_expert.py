"""The GPU-scale transformer expert, checked on CPU with a tiny model: it learns, looks only backwards, and its bits add up."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from apedp_mmlm import transformer_expert as te

TEXT = (b"The referee promotes a law only when no distinct rival survives. "
        b"The search proposes laws; the referee tests them. ") * 80


@pytest.fixture(scope="module")
def trained():
    return te.train(TEXT[:7000], TEXT[7000:], seconds=5, d_model=32, layers=1, heads=2, context=32, batch=16,
                    eval_every=50, eval_bytes=1000)


def test_training_learns_and_keeps_the_best_weights(trained):
    model, info = trained
    assert info["steps"] > 50 and info["parameters"] == sum(p.numel() for p in model.parameters())
    assert info["best_valid_bits_per_byte"] < 4.0 < 8.0  # far below the 8 bits of a uniform guess
    assert info["best_valid_bits_per_byte"] == min(point["valid_bits_per_byte"] for point in info["valid_curve"])
    assert te.bits_per_byte(model, TEXT[7000:8000], torch.device("cpu")) == pytest.approx(info["best_valid_bits_per_byte"])


def test_bit_probabilities_multiply_back_to_the_byte_probability_in_every_window(trained):
    model, _ = trained
    data = TEXT[7000:7400]
    bits = te.bit_probabilities(model, data, batch=4)
    array = np.frombuffer(data, dtype=np.uint8)
    assert bits.shape == (400, 8) and np.allclose(bits[0], 0.5) and np.all((bits > 0) & (bits < 1))
    with torch.no_grad():
        for t, start in ((10, 0), (200, 176), (399, 368)):  # the first window, a later one, the last byte
            logp = torch.log_softmax(model(torch.tensor(list(data[start : start + 32]))[None])[0], dim=-1)
            along_path = sum(math.log(bits[t, j] if (array[t] >> (7 - j)) & 1 else 1 - bits[t, j]) for j in range(8))
            assert along_path == pytest.approx(float(logp[t - start - 1, array[t]]), abs=1e-4)


def test_the_expert_does_not_see_the_future(trained):
    model, _ = trained
    data = TEXT[7000:7400]
    altered = data[:200] + bytes(reversed(data[200:]))
    a, b = te.bit_probabilities(model, data, batch=4), te.bit_probabilities(model, altered, batch=4)
    assert np.allclose(a[:199], b[:199], atol=1e-5) and np.allclose(a[:200, 0], b[:200, 0], atol=1e-5)


def test_weights_round_trip(trained, tmp_path):
    model, _ = trained
    te.save(model, tmp_path / "w.pt")
    again = te.load(tmp_path / "w.pt")
    assert again.config == model.config
    assert np.allclose(te.bit_probabilities(model, TEXT[:200]), te.bit_probabilities(again, TEXT[:200]))


def test_width_must_divide_into_heads():
    with pytest.raises(ValueError):
        te.ByteTransformer(d_model=30, heads=4)
