import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dfthcsos_rm_block2_j4.py"
spec = spec_from_file_location("dfthcsos_rm_block2_j4", SCRIPT)
assert spec and spec.loader
mod = module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def synthetic_table():
    # Deterministic test-only table; not data from the competition submission.
    rng = np.random.default_rng(20261025)
    return rng.integers(0, 1 << mod.WORD_BITS,
                        size=(mod.ANGLE_BATCH_COUNT, mod.ADDRESS_COUNT, mod.J),
                        dtype=np.uint16)


def test_design_profile_is_honest_and_consistent():
    p = mod.design_projection()
    assert p["status"] == "ANALYTICAL_PROJECTION_ONLY"
    assert p["angle_batch_count"] == 19
    assert p["projected_Q_total"] == 682
    assert p["angle_base_width_before_buffer"] == 71
    assert p["buffer_payload_bits_for_four_word_delta"] == 48
    assert p["peak_angle_qubits_before_phase_fixup"] == 119
    assert p["extra_phase_fixup_qubits_allowed"] == 9
    assert p["projected_candidate_QROM_Toffolis_all_19_batches"] == 19 * (1988 - 1)
    assert p["promotion_gate"]["official_verifier_status"] == "NOT_RUN"


def test_pair_transform_reconstructs_every_row_all_batches():
    table = synthetic_table()
    packed = mod.pair_adjacent_addresses(table)
    assert packed.shape == (19, 1988, 8)
    reconstructed = mod.reconstruct_addresses(packed)
    assert np.array_equal(reconstructed, table)
    report = mod.profile_actual_table(table, packed)
    assert report["lookup_equivalence_all_addresses"] is True
    assert report["compiled_kmb"] is False
    assert report["official_verifier_run"] is False


def test_delta_is_bitwise_xor_and_odd_row_decodes():
    table = synthetic_table()
    packed = mod.pair_adjacent_addresses(table)
    base = packed[:, :, :4]
    delta = packed[:, :, 4:]
    assert np.array_equal(delta, np.bitwise_xor(table[:, 0::2, :], table[:, 1::2, :]))
    assert np.array_equal(np.bitwise_xor(base, delta), table[:, 1::2, :])
    assert np.array_equal(base, table[:, 0::2, :])


def test_rejects_wrong_shape_or_non_12_bit_values():
    with np.testing.assert_raises(ValueError):
        mod.validate_batch_xor_words(np.zeros((10, 3976, 4), dtype=np.uint16))
    bad = synthetic_table()
    bad[0, 0, 0] = 4096
    with np.testing.assert_raises(ValueError):
        mod.validate_batch_xor_words(bad)


def test_reconstruction_rejects_invalid_packed_shape():
    with np.testing.assert_raises(ValueError):
        mod.reconstruct_addresses(np.zeros((19, 1988, 7), dtype=np.uint16))
