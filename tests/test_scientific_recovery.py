"""Fast canonical known-truth recovery checks."""

import pytest

from benchmarks.known_truth_challenge import _run_core, build_worlds


@pytest.mark.scientific_regression
def test_canonical_quadratic_recovery_meets_preregistered_criteria():
    world = build_worlds(noise_seed=0)[0]
    result = _run_core([world], noise_seed=0)

    assert result["protocol"]["noise_seed"] == 0
    assert result["protocol"]["recovery_r2"] == 0.99
    assert result["protocol"]["coef_tol"] == 0.05
    assert result["rates"]["rime_static"]["recovery_rate"] == 1.0
    assert result["rates"]["rime_static"]["false_promotion"] == 0

    for cell in result["cells"]:
        assert cell["verdict"] == "recovered_structure"
        assert cell["structure"] == world.world.d
        assert cell["holdout_r2"] >= 0.99
        assert cell["holdout_r2_reversed"] >= 0.99
        assert cell["coef_relerr"] <= 0.05
