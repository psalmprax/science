"""Unit and integration tests for Benchmark III (dfthcsos) signed 4-address banking architecture.

Status: audit-only

Tests:
1. Signed Givens tree invariant: cos(theta) > 0 ==> bit9 == bit8 in 10-bit representation.
2. Exhaustive 9-bit compression/decompression round-trip on all 512 valid 10-bit words.
3. 4-address banking table packing and lossless inversion on full (25, 3976, 3) arrays.
4. Exact 128-qubit cap satisfaction (12 + 30 + 81 + 5 = 128) with 0 headroom.
5. All 25 compiled ANGLE_k circuits strictly satisfy num_qubits <= 128.
6. Forensic audit and numerical resolution of the 4,363 Toffoli discrepancy (69,607 vs. 65,244).
7. Impact on chemistry cost (2.266B vs 2.124B) and portfolio geometric mean (6.53 vs 6.68).
8. Physical Kickmix circuit compilation and measurement when available.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np

from apedp_dyn.quantum_compilation.dfthcsos_signed_4bank import (
    ADDRESS_COUNT,
    BASE_WORD_BITS,
    COMPRESSED_DELTA_BITS_PER_ROW,
    HEADROOM,
    J,
    K_CIRCUITS,
    PEAK_ANGLE_QUBITS,
    PROMOTION_STATUS_HOLD,
    QUBIT_CAP,
    SCRATCH_QUBITS,
    SELECTION_QUBITS,
    compile_all_25_angle_circuits,
    compress_signed_word,
    decompress_signed_word,
    design_projection_signed_4bank,
    pack_4address_bank,
    reconstruct_4address_bank,
    resolve_4363_discrepancy,
    validate_signed_angle_word,
    verify_cos_positive_invariant,
)
from apedp_dyn.quantum_compilation.dfthcsos_adapter import (
    extract_j3_signed_words,
)


def _synthetic_valid_words(seed: int = 20261026) -> np.ndarray:
    """Generate (25, 3976, 3) synthetic words strictly obeying bit9 == bit8."""
    rng = np.random.default_rng(seed)
    signed = rng.integers(-256, 256, size=(K_CIRCUITS, ADDRESS_COUNT, J), dtype=np.int16)
    return (signed & 0x3FF).astype(np.uint16)


def test_signed_givens_invariant_mathematical_proof():
    """Verify cos(theta) > 0 implies bit9 == bit8 in 10-bit quantization."""
    # Positive angles in [0, 1.55] (strict interior of [0, pi/2))
    thetas_pos = np.linspace(0.0, 1.55, 100)
    assert np.all(verify_cos_positive_invariant(thetas_pos))

    # Negative angles in [-1.55, -0.01] (strict interior of (-pi/2, 0))
    thetas_neg = np.linspace(-1.55, -0.01, 100)
    assert np.all(verify_cos_positive_invariant(thetas_neg))

    # Exact endpoints
    assert verify_cos_positive_invariant(0.0)[0] is np.True_

    # Out of domain: angles with cos(theta) <= 0 must raise ValueError
    with pytest.raises(ValueError, match="cos.*strictly positive"):
        verify_cos_positive_invariant(1.6)

    with pytest.raises(ValueError, match="cos.*strictly positive"):
        verify_cos_positive_invariant(np.pi)


def test_exhaustive_signed_word_compression_codec():
    """Exhaustively verify 9-bit compression and decompression on all 1,024 10-bit words."""
    codes = np.arange(1024, dtype=np.uint16)
    valid_mask = ((codes >> 9) & 1) == ((codes >> 8) & 1)
    valid_codes = codes[valid_mask]
    invalid_codes = codes[~valid_mask]

    # Exactly 512 words in [0, 256) U [768, 1024) satisfy bit9 == bit8
    assert len(valid_codes) == 512
    assert len(invalid_codes) == 512

    for w in valid_codes:
        assert validate_signed_angle_word(int(w)) is True
        c9 = compress_signed_word(int(w))
        assert 0 <= c9 < 512
        w10 = decompress_signed_word(c9)
        assert w10 == int(w)

    for w in invalid_codes:
        assert validate_signed_angle_word(int(w)) is False
        with pytest.raises(ValueError, match="bit9 == bit8"):
            compress_signed_word(int(w))


def test_4bank_packing_and_reconstruction_roundtrip():
    """Verify 4-address banking lossless roundtrip on full (25, 3976, 3) table."""
    words = _synthetic_valid_words()
    base_words, delta_bank = pack_4address_bank(words)

    assert base_words.shape == (25, 994, 3)
    assert delta_bank.shape == (25, 994, 3, 3)
    assert base_words.dtype == np.uint16
    assert delta_bank.dtype == np.uint16

    # Delta bank words fit within 9 bits [0, 512)
    assert np.all(delta_bank < 512)

    # Invert and verify exact table identity
    reconstructed = reconstruct_4address_bank(base_words, delta_bank)
    assert np.array_equal(reconstructed, words)


def test_exact_128_qubit_cap_and_zero_headroom():
    """Verify exact 128-qubit cap and 0 headroom allocation."""
    assert SELECTION_QUBITS == 12
    assert BASE_WORD_BITS == 30
    assert COMPRESSED_DELTA_BITS_PER_ROW == 81
    assert SCRATCH_QUBITS == 5
    assert PEAK_ANGLE_QUBITS == 12 + 30 + 81 + 5 == 128
    assert QUBIT_CAP == 128
    assert HEADROOM == 0


def test_all_25_compiled_circuits_strictly_satisfy_128_qubits(tmp_path):
    """Verify that compiling all 25 ANGLE_k circuits strictly respects <= 128 qubits."""
    words = _synthetic_valid_words()
    res = compile_all_25_angle_circuits(words, output_dir=tmp_path)

    assert res["status"] == "PASS"
    assert res["num_circuits"] == 25
    assert res["max_circuit_qubits"] <= 128
    assert res["within_cap"] is True
    assert res["headroom"] >= 0
    assert len(res["circuits"]) == 25

    for circ_meta in res["circuits"]:
        assert circ_meta["num_qubits"] <= 128
        assert circ_meta["within_cap"] is True


def test_forensic_resolution_of_4363_toffoli_discrepancy():
    """Formally resolve the 4,363 Toffoli discrepancy between literal formula and external claim."""
    res = resolve_4363_discrepancy()

    assert res["status"] == "RESOLVED_AUDIT_DISCREPANCY"
    assert res["literal_formula_result"] == 69607
    assert res["claimed_result"] == 65244
    assert res["gap_toffolis"] == 4363

    # Breakdown derivation
    assert res["breakdown"]["circuits_K"] == 25
    assert res["breakdown"]["toffolis_per_pass"] == 1129
    assert res["breakdown"]["passes"] == 2
    assert res["breakdown"]["angle_toffolis"] == 56450
    assert res["breakdown"]["non_angle_toffolis"] == 13157
    assert 56450 + 13157 == 69607

    # Cost derivation
    assert math.isclose(res["cost_impact"]["claimed_cost"], 2123803639.40, rel_tol=1e-6)
    assert math.isclose(res["cost_impact"]["literal_cost"], 2266003679.75, rel_tol=1e-6)
    assert math.isclose(res["cost_impact"]["cost_penalty"], 142200040.35, rel_tol=1e-6)

    # Gain factors
    assert math.isclose(res["cost_impact"]["claimed_gain"], 1.5800, rel_tol=1e-4)
    assert math.isclose(res["cost_impact"]["literal_gain"], 1.4808, rel_tol=1e-3)

    # Portfolio geometric mean impact
    gm_claim = res["portfolio_geometric_mean"]["with_claimed_t_65244"]
    gm_literal = res["portfolio_geometric_mean"]["with_literal_t_69607"]
    assert gm_claim >= 6.67
    assert gm_literal < 6.54
    assert res["promotion_gate"]["decision"] == PROMOTION_STATUS_HOLD


def test_design_projection_signed_4bank():
    """Verify architectural design projection specification."""
    proj = design_projection_signed_4bank()
    assert proj["status"] == "ANALYTICAL_AND_TABLE_VERIFIED"
    assert proj["design"] == "signed_4_address_banking"
    assert proj["angle_circuits_K"] == 25
    assert proj["address_count"] == 3976
    assert proj["block_size_B"] == 4
    assert proj["bank_rows_per_batch"] == 994
    assert proj["angle_circuit_width_components"]["peak_qubits"] == 128
    assert proj["toffolis"]["literal_toffolis_total"] == 69607
    assert proj["toffolis"]["discrepancy"] == 4363
    assert proj["promotion_gate"]["status"] == PROMOTION_STATUS_HOLD


def test_dfthcsos_adapter_quantization():
    """Verify extract_j3_signed_words with bit-width quantization on real competition data."""
    npz_path = Path("apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/sample_submission/dfthcsos/submission.npz")
    if not npz_path.exists():
        pytest.skip("sample_submission/dfthcsos/submission.npz not present")

    words10 = extract_j3_signed_words(npz_path, target_brot=10, source_brot=12)
    assert words10.shape == (25, 3976, 3)
    assert words10.dtype == np.uint16
    assert words10.max() < 1024
    assert words10.min() >= 0
