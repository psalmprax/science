"""Roberts–Chebyshev cognates, pinned Phase 0 thresholds and per-pose verification.

Three separate claims are pinned here:

* the three cognates of a four-bar trace one coupler curve within the declared
  tolerance, and each cognate's curve comes from *its own* closed form, not from
  the construction reproducing itself;
* the Grashof margin and the T2 angular tolerance are preregistered inside the
  frozen protocol bundle, and a run cannot substitute its own numbers for them;
* every pose a T1 path check accepted is re-solved from blind seeds and its loop
  closure confirmed independently.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from mmlm_matter_001.kinematics import (
    COGNATE_TOLERANCE_MM,
    FourBar,
    MechanismError,
    check_cognates,
    check_path,
    cognate_fixtures,
    roberts_cognates,
    verify_poses,
)
from mmlm_matter_001.models import (
    ModelContract,
    P1SelectionOutput,
    P2DesignOutput,
    p1_prompt,
    p2_prompt,
)
from mmlm_matter_001.pipeline import STUB_MODEL
from mmlm_matter_001.protocol import (
    Phase0Preregistration,
    PreregistrationError,
    ProtocolDriftError,
    ProtocolError,
    ProtocolIntegrityError,
    freeze_protocol,
    load_protocol,
)
from mmlm_matter_001.schemas import PublicTaskView

CRANK_ROCKER = FourBar(
    ground_mm=40.0,
    input_mm=12.0,
    coupler_mm=45.0,
    output_mm=30.0,
    coupler_point_mm=(20.0, 0.0),
)


def _p1_contract(**overrides) -> ModelContract:
    values = {
        "provider": "stub",
        "model": STUB_MODEL,
        "prompt": p1_prompt(),
        "schema": P1SelectionOutput,
        "max_output_tokens": 512,
        "price_input_usd_per_mtok": 0.0,
        "price_output_usd_per_mtok": 0.0,
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


def _preregistration(**overrides) -> Phase0Preregistration:
    values = {
        "grashof_margin_mm": 0.5,
        "function_tolerance_deg": 0.25,
        "rationale": "Phase 0 thresholds pinned before hidden tasks are opened.",
    }
    values.update(overrides)
    return Phase0Preregistration(**values)


def _solvable(mechanism: FourBar, angle: float) -> bool:
    for branch in ("open", "crossed"):
        try:
            mechanism.solve(angle, branch=branch)
        except MechanismError:
            continue
        return True
    return False


def _solvable_range(mechanism: FourBar, *, span: float = 30.0) -> tuple[float, float]:
    """A contiguous input interval the mechanism assembles on, for a fixture."""
    for start in range(0, 360, 10):
        if all(
            _solvable(mechanism, start + step)
            for step in range(0, int(span) + 1, 5)
        ):
            return float(start), float(start) + span
    raise AssertionError("no contiguous solvable input range found for the fixture")


# ------------------------------------------------------------------- cognates


@pytest.mark.parametrize("name,mechanism", cognate_fixtures())
def test_the_three_cognates_trace_one_coupler_curve(name, mechanism):
    cognates = roberts_cognates(mechanism)
    low, high = _solvable_range(mechanism)

    report = check_cognates(
        cognates, input_range_deg=(low, high), samples=31,
        tolerance_mm=COGNATE_TOLERANCE_MM,
    )

    assert report.passed, report.details
    assert report.focal_drift_mm <= COGNATE_TOLERANCE_MM
    assert max(report.coupler_gap_mm.values()) <= COGNATE_TOLERANCE_MM
    assert max(report.frechet_mm.values()) <= COGNATE_TOLERANCE_MM
    assert set(report.inversion) == {"original", "left", "right"}


def test_the_cognates_are_dimensionally_dissimilar_to_the_original():
    mechanism = dict(cognate_fixtures())["crank-rocker"]
    cognates = roberts_cognates(mechanism)
    original_links = cognates.original.mechanism.link_lengths_mm
    for cognate in (cognates.left, cognates.right):
        assert cognate.mechanism.link_lengths_mm != original_links


def test_the_construction_does_not_depend_on_the_reference_pose():
    mechanism = dict(cognate_fixtures())["crank-rocker"]
    angles = [float(angle) for angle in range(0, 360, 5) if _solvable(mechanism, angle)]
    first = roberts_cognates(mechanism, angles[0])
    second = roberts_cognates(mechanism, angles[-1])

    assert math.dist(first.focal_pivot_mm, second.focal_pivot_mm) <= COGNATE_TOLERANCE_MM
    for left, right in ((first.left, second.left), (first.right, second.right)):
        for name, length in left.mechanism.link_lengths_mm.items():
            assert right.mechanism.link_lengths_mm[name] == pytest.approx(length)


def test_a_tampered_cognate_fails_the_shared_curve_check():
    mechanism = dict(cognate_fixtures())["crank-rocker"]
    cognates = roberts_cognates(mechanism)
    low, high = _solvable_range(mechanism)
    broken = replace(
        cognates,
        left=replace(
            cognates.left,
            mechanism=replace(
                cognates.left.mechanism,
                input_mm=cognates.left.mechanism.input_mm + 1.0,
            ),
        ),
    )

    report = check_cognates(
        broken, input_range_deg=(low, high), samples=21,
        tolerance_mm=COGNATE_TOLERANCE_MM,
    )

    # A wrong crank length no longer produces only a reason: the nearest assembly
    # is reported, so the gap is a measured number rather than "not assemblable".
    assert report.passed is False
    assert "coupler_point_gap" in report.reasons
    assert report.coupler_gap_mm["left"] > COGNATE_TOLERANCE_MM
    assert report.coupler_angle_gap_deg["left"] >= 0.0


def test_a_coupler_point_on_a_moving_pivot_is_refused():
    degenerate = FourBar(
        ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
        coupler_point_mm=(0.0, 0.0),
    )
    with pytest.raises(MechanismError):
        roberts_cognates(degenerate, 0.0)


# ------------------------------------------------------------ preregistration


def test_the_preregistration_is_pinned_inside_the_bundle(tmp_path):
    path = tmp_path / "protocol.json"
    pinned = _preregistration()

    freeze_protocol(
        path, _contracts(), pin_models=False, preregistration=pinned)
    loaded = load_protocol(path)

    assert loaded.require_preregistration() == pinned
    assert loaded.require_preregistration().digest == pinned.digest
    loaded.verify_preregistration(pinned)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["preregistration"]["grashof_margin_mm"] == 0.5
    assert raw["preregistration"]["function_tolerance_deg"] == 0.25


def test_a_bundle_pinned_to_other_thresholds_refuses_to_run(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_protocol(
        path, _contracts(), pin_models=False, preregistration=_preregistration())

    loaded = load_protocol(path)
    with pytest.raises(ProtocolDriftError):
        loaded.verify_preregistration(_preregistration(grashof_margin_mm=0.75))
    with pytest.raises(ProtocolDriftError):
        loaded.verify_preregistration(_preregistration(function_tolerance_deg=1.0))


def test_a_bundle_without_a_preregistration_cannot_claim_one(tmp_path):
    path = tmp_path / "protocol.json"
    frozen = freeze_protocol(path, _contracts(), pin_models=False)

    assert frozen.preregistration is None
    with pytest.raises(ProtocolError):
        frozen.require_preregistration()


def test_editing_the_pinned_margin_by_hand_is_detected(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_protocol(
        path, _contracts(), pin_models=False, preregistration=_preregistration())

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["preregistration"]["grashof_margin_mm"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ProtocolIntegrityError):
        load_protocol(path)


def test_a_task_cannot_re_open_the_t2_tolerance():
    pinned = _preregistration(function_tolerance_deg=0.25)

    assert pinned.require_function_tolerance(0.25) == 0.25
    with pytest.raises(PreregistrationError):
        pinned.require_function_tolerance(0.5)


def test_the_grashof_test_uses_only_the_pinned_margin():
    # Slack = 0.1 mm: clears the strict inequality, fails a 0.5 mm margin.
    marginal = FourBar(
        ground_mm=10.5, input_mm=10.0, coupler_mm=10.2, output_mm=10.4)

    assert marginal.check_grashof(margin_mm=0.0).is_grashof is True
    verdict = _preregistration(grashof_margin_mm=0.5).grashof(marginal)
    assert verdict.is_grashof is False
    assert verdict.reason == "grashof_margin_not_met"


# ------------------------------------------------- loop closure second opinion


def test_every_pose_of_an_accepted_path_closes_its_loop():
    trace = CRANK_ROCKER.trace((100.0, 260.0), samples=17, branch="open")

    verification = verify_poses(CRANK_ROCKER, trace.poses)

    assert verification.verified is True
    assert verification.checked == 17
    assert verification.disagreements == 0
    assert verification.max_angle_gap_deg <= 1e-6
    assert verification.max_residual_mm <= 1e-8


def test_a_pose_that_does_not_close_is_reported():
    trace = CRANK_ROCKER.trace((100.0, 260.0), samples=5, branch="open")
    tampered = list(trace.poses)
    tampered[2] = replace(
        tampered[2], theta3_deg=tampered[2].theta3_deg + 1.0)

    verification = verify_poses(CRANK_ROCKER, tampered)

    assert verification.verified is False
    assert verification.disagreements == 1


def test_check_path_reports_the_loop_closure_second_opinion():
    trace = CRANK_ROCKER.trace((100.0, 260.0), samples=21, branch="open")
    task = PublicTaskView(
        task_id="T1-loop-closure",
        family="T1",
        target_kind="ordered planar coupler path",
        tolerance=0.5,
        input_range_deg=(100.0, 260.0),
        target_points=trace.points_mm,
    )

    check = check_path(
        CRANK_ROCKER, task, branch="open", samples=21, verify_loop_closure=True)

    assert check.passed is True, check.reasons
    assert "loop_closure_disagrees" not in check.reasons
    assert check.loop_closure_verified is True
    assert check.loop_closure_checked == 21
    assert check.max_loop_residual_mm is not None
    assert check.max_loop_residual_mm <= 1e-8
