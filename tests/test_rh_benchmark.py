"""Unit and Integration Tests for APEDP-RH Mathematical Discovery Benchmark."""
# RECONSTRUCTION NOTE: deleted from the worktree by a concurrent session and
# restored here from bytecode forensics (tests/__pycache__/...pytest-9.1.1.pyc):
# seeds/ranges/thresholds/messages below are recovered from compiled constants.
# The rh_dataset / rh_adversarial library modules were deleted alongside the
# tests and are NOT restorable from bytecode with numeric fidelity; tests that
# need them skip (loudly) until the modules are restored, at which point the
# bodies below must be re-verified before un-skipping. test_proof_boundary_*
# and test_identifiability_* run against the surviving mathematical_layer
# engine and are fully verified.
from __future__ import annotations

import numpy as np
import pytest

from apedp.mathematical.mathematical_layer import (
    MathematicalAssuranceEngine,
    ProofBoundaryStatus,
)

_RH_MISSING_REASON = (
    "DATA LOSS (2026-09-27): apedp/mathematical/rh_dataset.py and "
    "rh_adversarial.py deleted from worktree; not in git, only bytecode "
    "remains. Restore the modules, then re-verify dependents before relying."
)


def test_zeta_zeros_caching_and_properties():
    """Verify that computed zeta zeros lie on critical strip with positive imaginary parts."""
    rh_dataset = pytest.importorskip("apedp.mathematical.rh_dataset", reason=_RH_MISSING_REASON)
    zeros = rh_dataset.generate_or_load_zeta_zeros(50)
    assert len(zeros) == 50
    for z in zeros:
        assert 0.0 < z.beta < 1.0
        assert z.beta == pytest.approx(0.5, abs=1e-12)
        assert z.gamma > 14.0


def test_proof_boundary_guard_enforcement():
    """Verify that MathematicalAssuranceEngine strictly denies claims of mathematical proof."""
    engine = MathematicalAssuranceEngine()
    gamma = np.linspace(14.0, 100.0, 100)
    audit_honest = engine.audit_mathematical_candidate(
        "gamma",
        {"gamma": (14.0, 100.0)},
        sample_observations={"gamma": gamma},
        claimed_proof=False,
    )
    assert audit_honest.is_admissible
    assert audit_honest.proof_status == ProofBoundaryStatus.EMPIRICALLY_SUPPORTED
    assert "Formal proof remains unestablished" in audit_honest.rationale
    audit_overconfident = engine.audit_mathematical_candidate(
        "gamma",
        {"gamma": (14.0, 100.0)},
        sample_observations={"gamma": gamma},
        claimed_proof=True,
    )
    assert audit_overconfident.proof_status == ProofBoundaryStatus.PROOF_NOT_ESTABLISHED
    assert "Proof boundary guard enforced" in audit_overconfident.rationale


def test_identifiability_resolution_limits():
    """Verify that hypotheses differing by less than numerical resolution are flagged as tied."""
    engine = MathematicalAssuranceEngine()
    distinguishable, winner, _tied = engine.assess_identifiability_resolution(
        {"true_rh": 0.0001, "alt": 0.0002}, 1e-12
    )
    assert distinguishable
    assert winner == "true_rh"
    distinguishable2, _winner2, tied2 = engine.assess_identifiability_resolution(
        {"true_rh": 2e-15, "alt": 1e-15}, 1e-12
    )
    assert not distinguishable2
    assert len(tied2) == 2


def test_adversarial_fake_rh_generators():
    """Verify that Fake-RH worlds introduce real structural deviations away from beta=0.5."""
    rh_adversarial = pytest.importorskip("apedp.mathematical.rh_adversarial", reason=_RH_MISSING_REASON)
    n_arr = np.arange(1, 101)
    gamma_arr = np.linspace(14.0, 250.0, 100)
    adv = rh_adversarial.RiemannAdversarialEngine()
    fake1 = adv.get_fake_rh_1(n_arr, gamma_arr)
    assert np.max(np.abs(np.asarray(fake1) - 0.5)) > 0.0009
    fake2 = adv.get_fake_rh_2(n_arr, gamma_arr)
    assert np.max(np.asarray(fake2)) > 1.5
    assert np.min(np.asarray(fake2)) < -1.0
    assert float(np.mean(np.asarray(fake2)[:50])) == pytest.approx(0.509, abs=0.01)
    fake3 = adv.get_fake_rh_3(n_arr, gamma_arr)
    assert len(fake3) == 50
    assert abs(float(np.mean(fake3)) - 0.51) < 0.05
