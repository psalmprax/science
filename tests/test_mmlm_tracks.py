"""Behavioural checks for the MMLM tracks over the GF(2^12) decomposition.

These pin the properties the tracks claim: every reported product is produced by the
decomposition and checked against the field model, the float ablation is fitted rather
than rigged, and each track states a scope narrow enough to be falsified.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.quantum_compilation.gf2mul_llm_adjustable import (
    MMLMMultiplier,
    MMLMTrackA,
    MMLMTrackB,
    MMLMTrackC,
)

PAIRS = [
    (0, 0),
    (1, 1),
    (1, 2),
    (3, 5),
    (1234, 2345),
    (0xFFF, 0xFFF),
    (0xA53, 0x65C),
    (4094, 7),
]

TRACKS = [MMLMTrackA, MMLMTrackB, MMLMTrackC]


def test_bilinear_decomposition_reproduces_field_products_on_a_sample():
    multiplier = MMLMMultiplier()
    checked = 0
    for left in range(0, 4096, 61):
        for right in (0, 1, 2, 3, 7, 42, 1023, 2048, 4094, 4095):
            result = multiplier.bilinear_multiply(left, right)
            assert result["exact"], (left, right, result["product"], result["expected"])
            checked += 1
    assert checked >= 500


@pytest.mark.parametrize("track_cls", TRACKS)
def test_every_track_returns_the_exact_product(track_cls):
    track = track_cls()
    for left, right in PAIRS:
        out = track.process_with_tensor(left, second_operand=right)
        assert out["exact"], (out["track"], left, right, out["product"], out["expected_product"])
        assert out["product"] == out["expected_product"]
        assert out["product"] == track.gf.mul(left, right)


def test_packed_operands_are_split_and_multiplied():
    left, right = 1234, 2345
    packed = np.array(
        [[(left >> i) & 1 for i in range(12)] + [(right >> i) & 1 for i in range(12)]],
        dtype=float,
    )
    track = MMLMTrackA()
    out = track.process_with_tensor(packed)
    assert out["operands"] == {"left": left, "right": right}
    assert out["exact"]
    assert out["product"] == track.gf.mul(left, right)


def test_track_b_is_deterministic_and_keeps_the_exact_path_when_float_errs():
    first = MMLMTrackB(seed=7).process_with_tensor(1234, second_operand=2345)
    second = MMLMTrackB(seed=7).process_with_tensor(1234, second_operand=2345)
    assert first["result"] == second["result"]
    if first["continuous"]["hamming_errors"]:
        assert first["uses_continuous"] is False
        assert first["result"]["path"] == "gf2_bilinear"


def test_track_b_float_model_is_fitted_at_the_same_rank():
    out = MMLMTrackB().process_with_tensor(1234, second_operand=2345)
    continuous = out["continuous"]
    assert continuous["path"] == "float_fitted_bilinear"
    assert continuous["atoms"] == 54
    assert continuous["fit_samples"] >= 1000
    assert continuous["fit_hamming_errors"] >= 0
    assert continuous["float_macs"] == 3 * 54 * 12


def test_track_c_energy_equals_the_bit_counts_it_reports():
    out = MMLMTrackC().process_with_tensor(1234, second_operand=2345)
    bits = [(out["product"] >> i) & 1 for i in range(12)]
    ssd = out["ssd_result"]
    assert ssd["energy"] == float(ssd["active_products"])

    htm = out["htm_result"]
    assert htm["energy"] == float(sum(bits) ** 2)
    assert htm["parity"] == sum(bits) % 2

    energy = out["energy_result"]
    assert energy["combined_energy"] == ssd["energy"] + htm["energy"]


@pytest.mark.parametrize("track_cls", TRACKS)
def test_tracks_state_a_scope_they_have_not_overreached(track_cls):
    out = track_cls().process_with_tensor(3, second_operand=5)
    scope = out["scope"]
    assert scope["unit"] == "one GF(2^12) product"
    assert scope["models_tokens"] is False
    assert scope["implements_attention"] is False
    assert "language model" not in scope["claim"].lower()
