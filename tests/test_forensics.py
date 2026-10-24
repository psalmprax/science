"""Tests for P1 forensics and P2 evidence reporting (engine-untouched)."""
from __future__ import annotations

import numpy as np

from apedp.domain_forensics import (
    asymptotic_behavior,
    coefficient_sensitivity,
    derivative_runs,
    domain_report,
    pole_scan,
)
from apedp.evidence_report import CandidateEvidence, build_evidence_report


def test_pole_scan_finds_singularity_location():
    report = pole_scan(lambda x: 1.0 / (x - 0.88), np.linspace(0.5, 1.5, 101))
    assert report["finite_fraction"] < 1.0
    assert any(abs(location - 0.88) < 0.02 for location in report["pole_locations"])


def test_pole_scan_clean_on_smooth():
    report = pole_scan(lambda x: x**2 + 1.0, np.linspace(0.5, 3.0, 50))
    assert report["finite_fraction"] == 1.0
    assert report["pole_locations"] == []


def test_derivative_runs_separates_smooth_from_oscillatory():
    grid = np.linspace(0.0, 6.0, 400)
    assert derivative_runs(lambda x: x**2, grid) == 0
    assert derivative_runs(lambda x: np.sin(3.0 * x) + 0.5 * np.sin(7.0 * x), grid) > 10


def test_asymptotic_classes():
    grid = np.linspace(1.0, 5.0, 50)
    assert asymptotic_behavior(lambda x: 5.0 + 0.0 * x, grid) == "bounded"
    assert asymptotic_behavior(lambda x: x**3, grid) == "polynomial"
    assert asymptotic_behavior(lambda x: np.exp(2.0 * x), grid) == "exponential"
    assert asymptotic_behavior(lambda x: np.sin(8.0 * x), grid) == "oscillatory"
    assert asymptotic_behavior(lambda x: 1.0 / (x - 30.0), grid) == "singular"
    assert asymptotic_behavior(lambda x: np.exp(-2.0 * x), grid) == "decaying"


def test_coefficient_sensitivity_separates_stable_from_touchy():
    grid = np.linspace(0.5, 3.0, 60)

    def factory(params):
        a = params["a"]
        return lambda x, a=a: a * x**2

    calm = coefficient_sensitivity(factory, {"a": 2.0}, grid)
    assert calm is not None and calm < 2.0

    def touchy_factory(params):
        a = params["a"]
        return lambda x, a=a: a * np.exp(a * x)

    wild = coefficient_sensitivity(touchy_factory, {"a": 5.0}, grid)
    assert wild is not None and wild > calm


def test_domain_report_assembles_notes():
    grid = np.linspace(0.5, 1.5, 101)
    report = domain_report(lambda x: 1.0 / (x - 0.88), grid)
    assert report.finite_fraction < 1.0
    assert any("non-finite" in note for note in report.notes)
    clean = domain_report(lambda x: x**2 + 1.0, np.linspace(0.5, 3.0, 50))
    assert clean.finite_fraction == 1.0
    assert clean.asymptotic_class == "polynomial"


def test_evidence_report_non_uniqueness():
    report = build_evidence_report(
        [
            CandidateEvidence("shomate", {"valMSE": 0.3}, True, ("poor far-OOD",)),
            CandidateEvidence("quadratic", {"valMSE": 4.4}, True, ("worse fit",)),
            CandidateEvidence("nest", {"valMSE": 0.01}, False, ("dim FAIL",)),
        ]
    )
    assert report["n_admissible"] == 2
    assert "non-unique" in report["assessment"]
    assert len(report["sections"]) == 3


def test_evidence_report_empty_and_single():
    assert "No admissible candidate" in build_evidence_report([])["assessment"]
    single = build_evidence_report([CandidateEvidence("a", {"m": 1.0}, True, ())])
    assert "Single admissible candidate" in single["assessment"]


def test_asymptotic_details_carry_ambiguity():
    import numpy as np

    from apedp.domain_forensics import asymptotic_behavior

    grid = np.linspace(1.0, 5.0, 50)
    detailed = asymptotic_behavior(lambda x: x**3, grid, details=True)
    assert detailed["class"] == "polynomial"
    assert 0.0 <= detailed["confidence"] <= 1.0
    assert "runner_up" in detailed
    # Narrow-margin case reports ambiguity rather than a hard label.
    soft = asymptotic_behavior(lambda x: 5.0 + 0.0 * x, grid, details=True)
    assert soft["class"] == "bounded"


def test_candidate_evidence_profile_gates_unresolved():
    from apedp.evidence_report import candidate_evidence_profile

    blocked = candidate_evidence_profile(
        "nest",
        mathematical={"finite": True, "real": True},
        dimensional={"status": "UNRESOLVED"},
        empirical={"summary": "good fit"},
        scientific={"summary": "oscillatory"},
    )
    assert blocked["selectable"] is False
    assert "unresolved" in blocked["report"]
    ready = candidate_evidence_profile(
        "shomate",
        mathematical={"finite": True, "real": True},
        dimensional={"status": "VALID"},
        empirical={"summary": "good fit"},
        scientific={"summary": "stable"},
    )
    assert ready["selectable"] is True
