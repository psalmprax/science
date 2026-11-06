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


def test_batch_path_agrees_with_the_scalar_path():
    multiplier = MMLMMultiplier()
    pairs = [
        (left, right)
        for left in (0, 1, 7, 1234, 4095)
        for right in (0, 3, 42, 4094)
    ]
    batch = multiplier.bilinear_multiply_batch(
        [left for left, _ in pairs], [right for _, right in pairs]
    )
    assert batch["pairs"] == len(pairs)
    for (left, right), product, exact in zip(
        pairs, batch["products"], batch["exact"]
    ):
        scalar = multiplier.bilinear_multiply(left, right)
        assert product == scalar["product"], (left, right)
        assert exact is True, (left, right)


@pytest.mark.parametrize("track_cls", TRACKS)
def test_inverse_is_a_composition_verified_on_the_whole_group(track_cls):
    """Inversion runs 21 products per element and is checked by x * x^-1 == 1."""
    track = track_cls()
    result = track.invert_batch(range(1, 4096))
    assert result["elements"] == 4095
    assert all(result["exact"]), result["inverses"][:5]
    assert set(result["checked"]) == {1}
    assert result["products_per_element"] == 21
    assert result["bilinear_products"] == 21 * 4095
    assert len(set(result["inverses"])) == 4095
    assert track.gf.mul(0x5A3, result["inverses"][0x5A3 - 1]) == 1


def test_inverse_refuses_zero_and_out_of_field_elements():
    track = MMLMTrackA()
    with pytest.raises(ValueError):
        track.invert_batch([0])
    with pytest.raises(ValueError):
        track.invert_batch([4096])


@pytest.mark.parametrize("track_cls", TRACKS)
def test_matvec_composes_products_and_matches_the_field_model(track_cls):
    track = track_cls()
    rng = np.random.default_rng(11)
    matrix = rng.integers(0, 4096, size=(6, 9))
    vector = rng.integers(1, 4096, size=9)
    out = track.gf_matvec(matrix, vector)
    assert out["exact"], (out["result"], out["expected"])
    assert out["entries"] == 54
    assert out["products"] == int(np.count_nonzero(matrix))
    assert out["products"] + out["zero_products_skipped"] == out["entries"]
    for row, value in zip(matrix, out["result"]):
        total = 0
        for entry, element in zip(row, vector):
            total ^= track.gf.mul(int(entry), int(element))
        assert value == total


def test_matvec_rejects_a_short_row():
    with pytest.raises(ValueError):
        MMLMTrackA().gf_matvec([[1, 2, 3], [4, 5]], [1, 2])


@pytest.mark.parametrize("track_cls", TRACKS)
def test_tracks_state_a_scope_they_have_not_overreached(track_cls):
    out = track_cls().process_with_tensor(3, second_operand=5)
    scope = out["scope"]
    assert scope["unit"] == "GF(2^12) arithmetic"
    assert scope["compositions"] == ["product", "inverse", "matvec"]
    assert scope["models_tokens"] is False
    assert scope["implements_attention"] is False
    assert "language model" not in scope["claim"].lower()
