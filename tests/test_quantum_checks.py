"""Quantum checks: convention note, commutator consistency, box spectra."""
from apedp.quantum_checks import (
    amplitude_convention_note,
    particle_in_box,
    verify_canonical_xp,
)


def test_amplitude_convention_documented():
    note = amplitude_convention_note()
    assert "imaginary" in note["bare_I"]
    assert "electronics" in note["rule"]


def test_canonical_commutator_consistent():
    out = verify_canonical_xp()
    assert out["status"] == "consistent", out
    assert out["antisymmetry_holds"] is True, out


def test_particle_in_box_matches_analytic():
    out = particle_in_box(n_levels=3, n_grid=200)
    assert out["max_rel_err"] is not None, out
    assert out["max_rel_err"] < 1e-3, out
    assert len(out["computed"]) == 3
