"""Tests for Active Singularity and Pole Probing Engine.

PROVENANCE: re-authored, not recovered. This file existed only as
``tests/__pycache__/test_singularity_probing.cpython-312-pytest-*.pyc`` --
compiled, never committed, and therefore invisible to git, to review, and
to CI. The source was lost; the bytecode was not.

No decompiler exists for CPython 3.12 (``decompyle3`` and ``uncompyle6``
both stop well below it, and the ``pycdc`` distribution on PyPI is an
unrelated CRM package), so the assertions here were reconstructed from the
bytecode's code objects: the docstrings, the literal constants, the keyword
names, and the attribute access sequence are all preserved in a ``.pyc`` and
all survived. What did not survive, and is therefore absent here, is the
original formatting, comments, and variable names.

The reconstruction is validated two ways. Every test passes against
today's source. And each assertion was checked against the bytecode's own
control flow rather than inferred from the module it tests, so a test that
passed for the wrong reason would not have been written this way.

This matters more than usual right now: ``apedp/singularity_probing.py``
had five silent exception handlers and one that fabricated a divergence
magnitude on failure. The file that covered it was in this state.
"""
from __future__ import annotations

import numpy as np

from apedp.singularity_probing import ActiveSingularityProbeEngine


def test_active_singularity_avoided_pole_gap_detection():
    """Verify that an avoided pole in a data gap (P19 archetype) is flagged."""
    # Sampling stops at 2.1 and resumes at 2.9; the candidate's pole sits at
    # 2.5001, inside the hole. The data cannot distinguish a smooth function
    # from a divergent one there, which is the whole point of P19.
    x = np.array([1.0, 1.4, 1.8, 2.1, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0])
    y = 2.0 * x + 1.0
    report = ActiveSingularityProbeEngine().probe_candidate_singularity(
        candidate_expr="2.0*x + 1.0 / (x - 2.5001)",
        competing_expressions=[],
        observed_data={"x": x, "y": y},
    )
    assert report.has_singularity_risk is True
    assert report.risk_type == "AVOIDED_POLE_IN_GAP"
    assert report.gap_details is not None
    assert abs(report.gap_details["pole_location"] - 2.5001) < 0.0001
    assert report.gap_details["gap_left"] == 2.1
    assert report.gap_details["gap_right"] == 2.9


def test_active_singularity_unobserved_critical_regime_detection():
    """Verify that an asymptotic divergence with unobserved near-critical data (P20 archetype) is flagged."""
    # Observations live at T = 308..350; the critical point is T = 300, and
    # the competing smooth law is linear. Claiming a critical singularity
    # from data that never approached it is unevidenced.
    T = np.linspace(308.0, 350.0, 30)
    chi = 1.5 * T - 300.0
    report = ActiveSingularityProbeEngine().probe_candidate_singularity(
        candidate_expr="1.5 / ((T - 300.0)**1.2)",
        competing_expressions=["0.5 - 0.001*T"],
        observed_data={"T": T, "chi": chi},
    )
    assert report.has_singularity_risk is True
    assert report.risk_type == "UNOBSERVED_CRITICAL_REGIME"
    assert report.gap_details is not None
    assert abs(report.gap_details["pole_location"] - 300.0) < 0.0001


def test_active_singularity_safe_canonical_physics():
    """Verify that well-sampled physical laws (Newton law, Runge function) are NOT falsely flagged.

    The over-correction guard for this file. Both expressions are genuinely
    singular somewhere, and neither should be flagged, because each is
    sampled densely across its singularity and is a canonical physics law
    rather than a claim about unseen critical behaviour. A probe that fires
    here is a probe that cannot be trusted when it stays quiet.

    The competing expression in each case is load-bearing, not decoration.
    Check B only raises UNOBSERVED_CRITICAL_REGIME when a *smooth alternative*
    exists to compete against, so passing an empty list changes the code
    path under test. An equivalence check against the original bytecode
    caught that: with the lists dropped, this test passed two mutants the
    original caught -- a gap threshold lowered to 0, and the root-dedup
    tolerance set to 0.
    """
    engine = ActiveSingularityProbeEngine()

    r = np.linspace(1.0, 10.0, 30)
    F = 10.0 / r**2
    data_newton = {"r": r, "y": F}

    rep_newton = engine.probe_candidate_singularity(
        candidate_expr="10.0 / r**2",
        competing_expressions=["5.0 / r"],
        observed_data=data_newton,
    )
    assert rep_newton.has_singularity_risk is False

    x = np.linspace(-1.0, 1.0, 30)
    y_runge = 1.0 / (1.0 + 25.0 * x**2)
    data_runge = {"x": x, "y": y_runge}

    rep_runge = engine.probe_candidate_singularity(
        candidate_expr="1.0 / (1.0 + 25.0*x**2)",
        competing_expressions=["1.0 - 5.0*x**2"],
        observed_data=data_runge,
    )
    assert rep_runge.has_singularity_risk is False
