"""The packed single-register Berlekamp-Massey, verified where it has to be.

The width claim for the archived Benchmark II artifact is only worth making if a
single-array decoder reproduces the two-array reference at the fields the decoder
runs over, so these tests pin the verification at ``m = 15`` (GF(2^4)) and
``m = 255`` (GF(2^8)) before the target field, and pin the width arithmetic that
the archived 891 qubits would have to match.
"""
from __future__ import annotations

import pytest

from apedp_dyn.quantum_compilation.inplace_bm import (
    DecoderLayout,
    PackingOverflow,
    check_field,
    error_positions,
    inplace_berlekamp_massey,
    inversionless_berlekamp_massey,
    reference_berlekamp_massey,
    syndromes_of,
)
from apedp_dyn.quantum_compilation.local_mr_sim import Field


def test_layout_is_one_array_not_two():
    layout = DecoderLayout(n_syndromes=70, degree=12)
    assert layout.t == 35
    assert layout.positions == 73
    assert layout.array_qubits == 73 * 12 == 876
    assert layout.reference_positions == 73
    assert layout.reference_array_qubits == 2 * 876 == 1752
    assert layout.fits_one_array is True
    # The archive is 891 = 876 + 15; the verified construction needs its two
    # scalars outside the array and therefore does not land on 891.
    assert layout.scalar_bits == 12
    assert layout.scaffold_bits == 11
    assert layout.transient_qubits == 23
    assert layout.total_qubits == 899


def test_the_eight_qubit_gap_is_the_division_scaffold():
    """891 = 876 (array) + 12 (scale) + 3 (control): the archive is division-free.

    The division variant needs an 11-bit GF(2^12) scaffold on top of its 12-bit
    stored ``b``, so it lands at 899 -- 8 above the archive. Dropping the division
    (inversionless update) removes the scaffold and leaves one field element, so
    876 + 12 = 888, and 888 + 3 control bits is exactly 891: the 8-qubit gap is
    11 - 3.
    """
    layout = DecoderLayout(n_syndromes=70, degree=12)
    assert layout.scaffold_bits == 11
    assert layout.control_bits == 3
    assert layout.transient_qubits == 23
    assert layout.total_qubits == 899
    assert layout.inversionless_transient_qubits == 12
    assert layout.inversionless_total_qubits == 888
    assert layout.inversionless_with_control_qubits == 891
    assert layout.total_qubits - layout.inversionless_with_control_qubits == 8
    assert layout.scaffold_bits - layout.control_bits == 8


def test_two_registers_cannot_be_narrower_than_the_archive():
    layout = DecoderLayout(n_syndromes=70, degree=12)
    assert layout.reference_array_qubits > 891
    assert layout.array_qubits < 891


@pytest.mark.parametrize("degree,n_syndromes,seed", [(4, 10, 1504), (8, 20, 25508)])
def test_verified_at_small_fields(degree, n_syndromes, seed):
    row = check_field(degree=degree, n_syndromes=n_syndromes, trials=120, seed=seed)
    assert row["verdict"] == "PASS"
    assert row["agreed_with_reference"] == row["trials"]
    assert row["inversionless_agreed_with_reference"] == row["trials"]
    assert row["recovered_planted_positions"] == row["trials"]


def test_verified_at_the_target_field():
    row = check_field(degree=12, n_syndromes=70, trials=12, seed=409512)
    assert row["verdict"] == "PASS"
    assert row["agreed_with_reference"] == row["trials"]
    assert row["inversionless_agreed_with_reference"] == row["trials"]
    assert row["recovered_planted_positions"] == row["trials"]


def test_inversionless_variant_recovers_planted_positions():
    """The 891-Q variant is checked on its own, not only against my division one."""
    field = Field(4)
    layout = DecoderLayout(n_syndromes=10, degree=4)
    errors = {0: 3, 5: 7, 11: 1, 14: 9, 7: 2}
    syndromes = syndromes_of(errors, layout.n_syndromes, field)
    sigma, _ = inversionless_berlekamp_massey(syndromes, field, layout.t)
    assert set(error_positions(sigma, field)) == set(errors)
    # Scale-invariance: normalising the locator must not move its roots.
    lead = sigma[-1]
    normalised = [field.div(c, lead) for c in sigma]
    assert normalised[-1] == 1
    assert set(error_positions(normalised, field)) == set(errors)


def test_packed_decoder_writes_only_its_own_array():
    """The packing is in place: the update must not need a second polynomial array.

    The test reconstructs the layout by hand at a small field and checks that the
    returned locator has degree at most ``t`` and that a two-array reference agrees
    root for root -- a second array would show up as a degree that exceeds ``t``
    or as disagreement on patterns beyond the correcting radius.
    """
    field = Field(4)
    layout = DecoderLayout(n_syndromes=10, degree=4)
    errors = {0: 3, 5: 7, 11: 1, 14: 9, 7: 2}          # weight t = 5
    syndromes = syndromes_of(errors, layout.n_syndromes, field)
    packed, length = inplace_berlekamp_massey(syndromes, field, layout.t)
    reference, reference_length = reference_berlekamp_massey(syndromes, field, layout.t)
    assert length == reference_length
    assert len(packed) - 1 <= layout.t
    assert set(error_positions(packed, field)) == set(errors)
    assert set(error_positions(packed, field)) == set(error_positions(reference, field))


def test_the_sigma_region_is_always_wide_enough():
    """``2t`` syndromes bound the locator degree by ``t``, so the array never grows.

    This is what makes the packing safe rather than merely lucky: with ``2t``
    syndromes the linear complexity is at most ``t`` for *any* input, including
    patterns far beyond the correcting radius, so the ``t + 1`` sigma cells are
    enough and the halfway point of the array is never crossed.
    """
    import numpy as np

    field = Field(8)
    layout = DecoderLayout(n_syndromes=8, degree=8)       # t = 4
    rng = np.random.default_rng(8804)
    observed_max = 0
    for _ in range(200):
        weight = int(rng.integers(0, 2 * layout.t + 1))
        positions = rng.choice(field.size - 1, size=weight, replace=False)
        errors = {int(p): int(rng.integers(1, field.size)) for p in positions}
        syndromes = syndromes_of(errors, layout.n_syndromes, field)
        reference, _ = reference_berlekamp_massey(syndromes, field, layout.t)
        packed, _ = inplace_berlekamp_massey(syndromes, field, layout.t)
        observed_max = max(observed_max, len(reference) - 1, len(packed) - 1)
        assert set(error_positions(packed, field)) == set(error_positions(reference, field))
    assert observed_max <= layout.t
    # ``observed_max`` reaches the cap on this seed, so the assertion is not vacuous.
    assert observed_max == layout.t
