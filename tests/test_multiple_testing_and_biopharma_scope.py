"""Regression tests for exact p-value correction and biopharma evidence labels."""

from __future__ import annotations

import math

import pytest

from apedp.applied_biopharma import (
    ADMETPredictor,
    DiseaseBiomarkerAnalyzer,
    DrugRepurposingEngine,
    UniversalDrugTargetDiscovery,
)
from apedp_dyn.epistemic.multiple_testing import adjust_p_values


@pytest.mark.parametrize(
    ("method", "expected_adjusted", "expected_rejections", "effective_threshold"),
    [
        ("bh", (0.02, 0.04, 0.04, 0.008), (True, True, True, True), 0.04),
        ("by", (0.01 * 4 / 2 * 2.0833333333333335,
               0.04 * 2.0833333333333335,
               0.03 * 4 / 3 * 2.0833333333333335,
               0.002 * 4 * 2.0833333333333335),
         (True, False, False, True), 0.01),
        ("holm", (0.03, 0.06, 0.06, 0.008), (True, False, False, True), 0.01),
    ],
)
def test_multiple_testing_adjustments_preserve_input_order(
    method, expected_adjusted, expected_rejections, effective_threshold
):
    result = adjust_p_values((0.01, 0.04, 0.03, 0.002), alpha=0.05, method=method)

    assert result.adjusted_p_values == pytest.approx(expected_adjusted)
    assert result.rejected == expected_rejections
    assert result.effective_threshold == pytest.approx(effective_threshold)


def test_by_is_no_less_conservative_than_bh():
    p_values = (0.001, 0.01, 0.03, 0.2)
    bh = adjust_p_values(p_values, alpha=0.05, method="bh")
    by = adjust_p_values(p_values, alpha=0.05, method="by")

    assert all(by_p >= bh_p for by_p, bh_p in zip(
        by.adjusted_p_values, bh.adjusted_p_values, strict=True
    ))


@pytest.mark.parametrize(
    ("p_values", "alpha", "method"),
    [
        ((), 0.05, "bh"),
        ((0.1, float("nan")), 0.05, "bh"),
        ((0.1, float("inf")), 0.05, "bh"),
        ((-0.1,), 0.05, "bh"),
        ((1.1,), 0.05, "bh"),
        ((True,), 0.05, "bh"),
        ((0.1,), 0.0, "bh"),
        ((0.1,), 1.0, "bh"),
        ((0.1,), 0.05, "bonferroni"),
    ],
)
def test_multiple_testing_rejects_invalid_inputs(p_values, alpha, method):
    with pytest.raises(ValueError):
        adjust_p_values(p_values, alpha=alpha, method=method)


def test_no_rejections_has_no_effective_threshold():
    result = adjust_p_values((0.8, 0.9), alpha=0.05, method="holm")

    assert result.rejected == (False, False)
    assert result.effective_threshold is None


def test_omics_are_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="omics_data is not analyzed"):
        UniversalDrugTargetDiscovery.discover_targets(
            "oncology", omics_data={"KRAS": 3.2}
        )


def test_static_target_examples_disclose_evidence_scope():
    targets = UniversalDrugTargetDiscovery.discover_targets("oncology")

    assert targets
    assert all(target["evidence"] == "static_example" for target in targets)
    assert all(not target["computed_from_omics"] for target in targets)


def test_admet_output_is_labeled_as_a_rule_screen():
    result = ADMETPredictor.predict_properties(300, 2.0, 2, 4, 3, 60)

    assert result["evidence"] == "rule_screen"
    assert "Lipinski rule-of-five" in result["checks_applied"]
    assert "hERG" in result["not_modelled"]


@pytest.mark.parametrize(
    "descriptors",
    [
        (float("nan"), 2.0, 2, 4, 3, 60),
        (300, float("inf"), 2, 4, 3, 60),
        (-1, 2.0, 2, 4, 3, 60),
        (300, 2.0, -1, 4, 3, 60),
    ],
)
def test_admet_rejects_invalid_descriptors(descriptors):
    with pytest.raises(ValueError):
        ADMETPredictor.predict_properties(*descriptors)


def test_biomarker_screen_reports_unassessed_markers_and_scope():
    result = DiseaseBiomarkerAnalyzer.analyze_biomarkers(
        "oncology", {"CEA": 3.0, "unlisted marker": 1.0}
    )

    assert result["analyzed_markers"] == []
    assert result["not_assessed_markers"] == ["CEA", "unlisted marker"]
    assert result["evidence"] == "rule_screen"
    assert "not a clinical risk estimate" in result["interpretation"]


def test_biomarker_screen_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        DiseaseBiomarkerAnalyzer.analyze_biomarkers(
            "cardiovascular", {"LDL-C": math.nan}
        )


def test_repurposing_candidates_are_not_claimed_as_computed_predictions():
    candidates = DrugRepurposingEngine.discover_repurposing_candidates("oncology")

    assert candidates
    assert all(candidate["evidence"] == "static_example" for candidate in candidates)
    assert all(
        not candidate["computed_from_patient_or_omics_data"]
        for candidate in candidates
    )
