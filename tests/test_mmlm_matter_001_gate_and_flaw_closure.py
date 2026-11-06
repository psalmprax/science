"""Closure tests for the eight gaps found while validating the cognates.

One test per fix, so each claim is pinned where it was made:

* circuit continuity is tracked from the joint angles, not the branch label;
* a truncated trace is never scored as if it were the whole path;
* a discrete distance publishes its sampling stability and its resolution;
* the cognates are compared as point sets, not only point for point;
* the measurement settings are pinned alongside the thresholds;
* the Phase 0 gate refuses to run without a preregistration and records every
  verdict, rejection included;
* proposers and model adapters cannot import the referee kinematics.
"""

from __future__ import annotations

import ast
from itertools import pairwise
from pathlib import Path

import pytest

from mmlm_matter_001.evaluation import VERDICT_KIND, VerdictLedger
from mmlm_matter_001.kinematics import (
    BASE_PATH_SAMPLES,
    COGNATE_TOLERANCE_MM,
    FourBar,
    check_cognates,
    check_path,
    circuit_breaks,
    cognate_fixtures,
    joint_gap_deg,
    roberts_cognates,
    symmetric_hausdorff,
)
from mmlm_matter_001.models import (
    ModelContract,
    P1SelectionOutput,
    P2DesignOutput,
    p1_prompt,
    p2_prompt,
)
from mmlm_matter_001.pipeline import STUB_MODEL, GateError, run_gate
from mmlm_matter_001.protocol import (
    Phase0Preregistration,
    PreregistrationError,
    ProtocolDriftError,
    freeze_protocol,
    load_protocol,
)
from mmlm_matter_001.schemas import PublicTaskView

ROOT = Path(__file__).resolve().parents[1]
CRANK_ROCKER = FourBar(
    ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
    coupler_point_mm=(20.0, 0.0))
NON_GRASHOF = FourBar(
    ground_mm=10.0, input_mm=40.0, coupler_mm=12.0, output_mm=11.0)
FORBIDDEN = "mmlm_matter_001.kinematics"


# --------------------------------------------------------------------- helpers


def _preregistration(**overrides) -> Phase0Preregistration:
    values = {
        "grashof_margin_mm": 0.5,
        "function_tolerance_deg": 0.25,
        "rationale": "Phase 0 thresholds and their measurement settings, pinned.",
        "path_samples": 37,
    }
    values.update(overrides)
    return Phase0Preregistration(**values)


def _p1_contract(**overrides) -> ModelContract:
    values = {
        "provider": "stub",
        "model": STUB_MODEL,
        "prompt": p1_prompt(),
        "schema": P1SelectionOutput,
        "max_output_tokens": 512,
    }
    values.update(overrides)
    return ModelContract.freeze(**values)


def _p2_contract(**overrides) -> ModelContract:
    values = {
        "provider": "stub",
        "model": STUB_MODEL,
        "prompt": p2_prompt(),
        "schema": P2DesignOutput,
        "max_output_tokens": 512,
    }
    values.update(overrides)
    return ModelContract.freeze(**values)


def _contracts() -> dict[str, ModelContract]:
    return {"P1": _p1_contract(), "P2": _p2_contract()}


def _path_task(points, *, tolerance: float = 2.0) -> PublicTaskView:
    return PublicTaskView(
        task_id="T1-closure",
        family="T1",
        target_kind="ordered planar coupler path",
        tolerance=tolerance,
        input_range_deg=(0.0, 360.0),
        target_points=tuple(points),
    )


def _function_task(tolerance: float) -> PublicTaskView:
    samples = tuple(
        (float(angle), CRANK_ROCKER.solve(float(angle), branch="open").theta4_deg)
        for angle in range(100, 141, 5)
    )
    return PublicTaskView(
        task_id="T2-closure",
        family="T2",
        target_kind="input-output angle function",
        tolerance=tolerance,
        input_range_deg=(100.0, 140.0),
        function_samples=samples,
    )


def _own_path(samples: int = 37):
    """The mechanism's own curve, traced on one circuit rather than by label."""
    return CRANK_ROCKER.trace_continuous((0.0, 360.0), samples=samples, strict=False)


def _ambiguous_input(mechanism: FourBar) -> float:
    """An input where the two assembly roots carry the same branch label."""
    for angle in range(0, 360):
        roots = mechanism.poses(float(angle))
        if len(roots) == 2 and roots[0].branch == roots[1].branch:
            return float(angle)
    raise AssertionError("the fixture has no input whose two roots share a label")


def _tampered_cognate(mechanism: FourBar):
    """The right cognate of the crank-rocker, whose roots share a label."""
    return roberts_cognates(mechanism).right.mechanism


# ------------------------------------------------------- flaw 1: circuit roots


def test_a_branch_label_cannot_distinguish_two_assemblies():
    mechanism = _tampered_cognate(dict(cognate_fixtures())["crank-rocker"])
    angle = _ambiguous_input(mechanism)
    roots = mechanism.poses(angle)

    assert roots[0].branch == roots[1].branch
    assert joint_gap_deg(roots[0], roots[1]) > 1e-6
    # The label returns one of them and cannot say which one it is.
    assert mechanism.solve(angle, branch=roots[0].branch) == roots[0]
    # Continuity returns the assembly next to where you came from.
    assert mechanism.pose_near(angle, roots[0]) == roots[0]
    assert mechanism.pose_near(angle, roots[1]) == roots[1]


def test_a_continuous_trace_follows_one_assembly_through_a_shared_label():
    mechanism = _tampered_cognate(dict(cognate_fixtures())["crank-rocker"])
    angle = _ambiguous_input(mechanism)
    seed = mechanism.poses(angle)[1]

    trace = mechanism.trace_continuous(
        (angle, angle + 20.0), samples=21, seed=seed, strict=False)

    assert trace.continuity_tracked is True
    assert trace.complete is True
    assert joint_gap_deg(trace.poses[0], seed) <= 1e-6
    assert circuit_breaks(trace.poses) == 0
    for first, second in pairwise(trace.poses):
        assert joint_gap_deg(first, second) < 90.0


def test_check_path_can_judge_continuity_from_the_joints():
    trace = _own_path(73)

    check = check_path(CRANK_ROCKER, _path_task(trace.points_mm), samples=73,
                       continuity=True)

    assert check.continuity_tracked is True
    assert check.circuit_continuous is True
    assert check.branch_switches == 0
    assert check.passed is True, check.reasons
    assert check.record()["continuity_tracked"] is True


# --------------------------------------------------- flaw 2: partial traces


def test_a_truncated_trace_is_never_scored():
    task = _path_task(_own_path().points_mm)

    check = check_path(NON_GRASHOF, task, samples=37)

    assert check.passed is False
    assert check.complete is False
    assert "partial_trace_not_scored" in check.reasons
    assert check.frechet_mm is None
    assert check.record()["frechet_mm"] is None


# ---------------------------------------------- flaw 3: sampling stability


def test_the_reported_distance_publishes_its_sampling_stability():
    trace = _own_path(13)

    check = check_path(
        CRANK_ROCKER, _path_task(trace.points_mm, tolerance=10.0), samples=13,
        refine=True, continuity=True)

    assert check.passed is True, check.reasons
    assert check.refinement_stable is True
    assert check.samples_tested[0] == 13
    assert len(check.samples_tested) > 1
    # The number moves because a finer grid exposes the target polyline's own
    # interpolation error; that is reported, not hidden.
    assert check.frechet_change_mm is not None
    assert check.frechet_change_mm >= 0.0
    assert check.resolution_mm is not None
    assert check.resolution_mm > 0.0
    assert check.record()["samples_tested"] == list(check.samples_tested)
    assert check.record()["refinement_stable"] is True


# ------------------------------------------------- flaw 4: set-level cognates


def test_the_cognates_agree_as_point_sets_not_only_point_for_point():
    mechanism = dict(cognate_fixtures())["crank-rocker"]

    report = check_cognates(
        roberts_cognates(mechanism), input_range_deg=(0.0, 355.0), samples=72,
        tolerance_mm=COGNATE_TOLERANCE_MM)

    assert report.passed, report.details
    assert set(report.hausdorff_mm) == {"left", "right"}
    assert max(report.hausdorff_mm.values()) <= COGNATE_TOLERANCE_MM
    assert "curve_set_gap" not in report.reasons
    assert report.record()["hausdorff_mm"]["left"] == report.hausdorff_mm["left"]


def test_hausdorff_separates_a_shifted_point_set():
    first = [(0.0, 0.0), (1.0, 0.0)]
    second = [(0.0, 2.0), (1.0, 2.0)]

    assert symmetric_hausdorff(first, first) == pytest.approx(0.0)
    assert symmetric_hausdorff(first, second) == pytest.approx(2.0)


# -------------------------------------------- flaw 6: pinned measurement settings


def test_the_measurement_settings_are_preregistered_with_the_thresholds():
    pinned = _preregistration()

    assert pinned.require_path_samples(37) == 37
    with pytest.raises(PreregistrationError):
        pinned.require_path_samples(42)
    assert _preregistration(path_samples=42).digest != pinned.digest
    assert (pinned.effective_settings()["loop_residual_tolerance_mm"]
            == pinned.loop_residual_tolerance_mm)


def test_an_older_preregistration_record_loads_with_the_measurement_defaults():
    legacy = {
        "version": "1.0",
        "grashof_margin_mm": 0.5,
        "function_tolerance_deg": 0.25,
        "rationale": "written before the measurement settings were pinned",
    }

    restored = Phase0Preregistration.from_record(legacy)

    assert restored.path_samples == BASE_PATH_SAMPLES
    assert (restored.effective_settings()
            == _preregistration(path_samples=BASE_PATH_SAMPLES).effective_settings())


def test_the_bundle_carries_the_measurement_settings_and_detects_a_change(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_protocol(
        path, _contracts(), pin_models=False,
        preregistration=_preregistration(path_samples=37))
    loaded = load_protocol(path)

    assert loaded.require_preregistration().path_samples == 37
    with pytest.raises(ProtocolDriftError):
        loaded.verify_preregistration(_preregistration(path_samples=42))


# -------------------------------------------------------- flaws 5 and 8: gate


def test_the_gate_refuses_to_run_without_a_preregistration():
    task = _path_task(_own_path().points_mm)

    with pytest.raises(GateError):
        run_gate(CRANK_ROCKER, task, preregistration=None)


def test_the_gate_refuses_a_sample_count_that_is_not_preregistered():
    task = _path_task(_own_path().points_mm)

    with pytest.raises(PreregistrationError):
        run_gate(CRANK_ROCKER, task, preregistration=_preregistration(), samples=42)


def test_the_gate_refuses_a_bundle_pinned_to_other_thresholds(tmp_path):
    path = tmp_path / "protocol.json"
    frozen = freeze_protocol(
        path, _contracts(), pin_models=False,
        preregistration=_preregistration(path_samples=37))
    task = _path_task(_own_path().points_mm)

    with pytest.raises(ProtocolDriftError):
        run_gate(
            CRANK_ROCKER, task, preregistration=_preregistration(path_samples=42),
            protocol=frozen)


def test_a_passing_candidate_is_judged_and_recorded(tmp_path):
    task = _path_task(_own_path().points_mm)
    ledger_path = tmp_path / "verdicts.jsonl"

    verdict = run_gate(
        CRANK_ROCKER, task, preregistration=_preregistration(), ledger=ledger_path,
        verify_cognates=True)

    assert verdict.passed is True, verdict.reasons
    assert verdict.samples == 37
    assert verdict.grashof["is_grashof"] is True
    assert verdict.loop_closure["verified"] is True
    assert verdict.loop_closure["checked"] >= 37
    assert verdict.cognates is not None
    assert verdict.cognates["passed"] is True
    assert verdict.effective_settings["samples_used"] == 37

    rows = VerdictLedger(ledger_path).rows()
    assert len(rows) == 1
    assert rows[0]["kind"] == VERDICT_KIND
    assert rows[0]["candidate_id"] == task.task_id
    assert rows[0]["passed"] is True
    assert rows[0]["preregistration_digest"] == _preregistration().digest
    assert VerdictLedger(ledger_path).summary()["accepted"] == 1


def test_a_refused_candidate_is_recorded_with_its_reasons(tmp_path):
    trace = _own_path()
    shifted = [(x + 5.0, y) for x, y in trace.points_mm]
    ledger_path = tmp_path / "verdicts.jsonl"

    verdict = run_gate(
        CRANK_ROCKER, _path_task(shifted, tolerance=0.5),
        preregistration=_preregistration(), ledger=ledger_path)

    assert verdict.passed is False
    assert "frechet_exceeds_tolerance" in verdict.reasons
    row = VerdictLedger(ledger_path).rows()[0]
    assert row["passed"] is False
    assert "frechet_exceeds_tolerance" in row["reasons"]
    summary = VerdictLedger(ledger_path).summary()
    assert summary["refused"] == 1
    assert summary["reasons"]["frechet_exceeds_tolerance"] == 1


def test_a_marginal_grashof_design_is_refused_by_the_pinned_margin():
    marginal = FourBar(
        ground_mm=10.5, input_mm=10.0, coupler_mm=10.2, output_mm=10.4)

    verdict = run_gate(
        marginal, _path_task(_own_path().points_mm),
        preregistration=_preregistration())

    assert verdict.passed is False
    assert "grashof_margin_not_met" in verdict.reasons
    assert verdict.grashof["is_grashof"] is False


def test_a_t2_task_is_scored_against_the_pinned_tolerance():
    verdict = run_gate(
        CRANK_ROCKER, _function_task(0.25), preregistration=_preregistration())

    assert verdict.passed is True, verdict.reasons
    assert verdict.function is not None
    assert verdict.function["max_error_deg"] <= 0.25


def test_a_t2_task_cannot_re_open_the_pinned_tolerance():
    verdict = run_gate(
        CRANK_ROCKER, _function_task(0.5), preregistration=_preregistration())

    assert verdict.passed is False
    assert "tolerance_not_preregistered" in verdict.reasons


# ---------------------------------------------------- flaw 7: import boundary


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_proposers_and_model_adapters_cannot_import_referee_kinematics():
    files = sorted((ROOT / "mmlm_matter_001" / "proposers").glob("*.py"))
    files += sorted((ROOT / "mmlm_matter_001" / "models").glob("*.py"))
    assert len(files) >= 4, "the boundary test must not be vacuous"

    offenders = {
        str(path.relative_to(ROOT)): sorted(
            name for name in _imported_modules(path)
            if name == FORBIDDEN or name.startswith(FORBIDDEN + ".")
        )
        for path in files
    }

    assert {path: names for path, names in offenders.items() if names} == {}
