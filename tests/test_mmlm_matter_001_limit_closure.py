"""The remaining limits: capability gaps, the hardened boundary and T2 selection.

Each test pins one claim that was previously only documented:

* an answer this checkout cannot verify is recorded as ``unverifiable``, not as a
  candidate failure, so a benchmark denominator stays honest;
* the runtime boundary walks the stack and checks source paths, so rewriting
  ``__name__`` does not launder a proposer's call;
* a T2 task is scored on one assembly across its declared inputs, not per-sample
  by branch label;
* the cognate angular tolerance is pinned with the other measurement settings.
"""

from __future__ import annotations

import pytest

from mmlm_matter_001.evaluation import VerdictLedger
from mmlm_matter_001.kinematics import (
    BoundaryError,
    FourBar,
    MechanismError,
    check_cognates,
    check_function,
    check_path,
    cognate_fixtures,
    roberts_cognates,
)
from mmlm_matter_001.pipeline import CAPABILITY_REASONS, run_gate
from mmlm_matter_001.protocol import Phase0Preregistration
from mmlm_matter_001.schemas import PublicTaskView, TopologyEntry

CRANK_ROCKER = FourBar(
    ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
    coupler_point_mm=(20.0, 0.0))


def _pinned(**overrides) -> Phase0Preregistration:
    values = {
        "grashof_margin_mm": 0.5,
        "function_tolerance_deg": 0.25,
        "rationale": "Phase 0 thresholds and measurement settings.",
        "path_samples": 37,
    }
    values.update(overrides)
    return Phase0Preregistration(**values)


def _path_task_for(
    mechanism: FourBar, *, family: str = "T1", tolerance: float = 2.0,
) -> PublicTaskView:
    trace = mechanism.trace((0.0, 360.0), samples=37)
    return PublicTaskView(
        task_id=f"{family}-limit", family=family,
        target_kind="ordered planar coupler path", tolerance=tolerance,
        input_range_deg=(0.0, 360.0), target_points=trace.points_mm)


def _path_task(*, family: str = "T3", tolerance: float = 2.0) -> PublicTaskView:
    return _path_task_for(CRANK_ROCKER, family=family, tolerance=tolerance)


def _topology(**overrides) -> TopologyEntry:
    values = {
        "topology_id": "T4-000",
        "link_count": 4,
        "joint_count": 4,
        "inversion": "crank-rocker",
        "ground_link": "ground",
        "input_link": "input",
        "output_link": "output",
        "parent_chain_id": "4R-0001",
        "graph_hash": "a" * 64,
    }
    values.update(overrides)
    return TopologyEntry(**values)


# ------------------------------------------------------------- verdict states


def test_the_three_verdict_states_are_distinct():
    accepted = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_pinned(), topology=_topology())
    refused = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_pinned(),
        topology=_topology(inversion="double-rocker"))
    unverifiable = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_pinned(),
        topology=_topology(
            topology_id="T6-000", link_count=6, joint_count=7))

    assert (accepted.status, refused.status, unverifiable.status) == (
        "accepted", "refused", "unverifiable")
    assert accepted.passed is True
    assert refused.passed is False
    # A structure this checkout cannot verify is not a candidate failure.
    assert unverifiable.passed is False
    assert set(unverifiable.reasons) <= CAPABILITY_REASONS
    assert unverifiable.record()["status"] == "unverifiable"


def test_the_ledger_counts_unverifiable_apart_from_failures(tmp_path):
    path = tmp_path / "verdicts.jsonl"

    run_gate(CRANK_ROCKER, _path_task(), preregistration=_pinned(),
             topology=_topology(), ledger=path)
    run_gate(CRANK_ROCKER, _path_task(), preregistration=_pinned(),
             topology=_topology(
                 topology_id="T6-000", link_count=6, joint_count=7), ledger=path)

    summary = VerdictLedger(path).summary()
    assert summary["verdicts"] == 2
    assert summary["accepted"] == 1
    assert summary["refused"] == 0
    assert summary["unverifiable"] == 1
    # The denominator a pass rate may use, with the capability gap removed.
    assert summary["eligible"] == 1


# --------------------------------------------------------------- boundary


def test_a_spoofed_module_name_is_caught_by_the_source_path():
    namespace = {
        "__name__": "tests.spoofed_caller",
        "__file__": "mmlm_matter_001/proposers/sneaky.py",
        "check_path": check_path,
        "mechanism": CRANK_ROCKER,
        "task": _path_task(),
    }

    with pytest.raises(BoundaryError):
        exec("check_path(mechanism, task, samples=5)", namespace)  # noqa: S102


def _referee_helper(task: PublicTaskView):
    return check_path(CRANK_ROCKER, task, samples=5)


def test_a_proposer_frame_deeper_in_the_stack_is_refused():
    namespace = {
        "__name__": "mmlm_matter_001.proposers.p2_llm_direct",
        "helper": _referee_helper,
        "task": _path_task(),
    }

    with pytest.raises(BoundaryError):
        exec("helper(task)", namespace)  # noqa: S102


def test_a_referee_driver_more_than_one_frame_away_is_allowed():
    check = _referee_helper(_path_task())

    assert check.frechet_mm is not None


# ------------------------------------------------ T2 assembly selection


def _chained_function_inputs(mechanism: FourBar):
    """A five-input declared chain that passes through a shared-label input.

    The labels hold across the whole chain, which is exactly the case where a
    per-sample label lookup *could* hop: both assemblies at the ambiguous input
    answer to the same branch name.
    """
    for angle in range(10, 350):
        roots = mechanism.poses(float(angle))
        if len(roots) != 2 or roots[0].branch != roots[1].branch:
            continue
        inputs = [angle - 8.0 + 2.0 * index for index in range(5)]
        try:
            chain = [mechanism.pose_on(inputs[0], roots[0].branch)]
            for value in inputs[1:]:
                chain.append(mechanism.pose_near(value, chain[-1]))
        except MechanismError:
            continue
        if len({pose.branch for pose in chain}) == 1:
            return inputs, chain
    return None


def test_a_t2_task_is_scored_on_one_assembly_across_its_inputs():
    mechanism = roberts_cognates(
        dict(cognate_fixtures())["crank-rocker"]).right.mechanism
    fixture = _chained_function_inputs(mechanism)
    assert fixture is not None, "no chained shared-label region on this link set"
    inputs, chain = fixture

    task = PublicTaskView(
        task_id="T2-continuity",
        family="T2",
        target_kind="input-output angle function",
        tolerance=1e-6,
        input_range_deg=(inputs[0], inputs[-1]),
        function_samples=tuple(
            (value, pose.theta4_deg)
            for value, pose in zip(inputs, chain, strict=True)),
    )

    check = check_function(mechanism, task, branch=chain[0].branch)

    assert check.passed is True, check.reasons
    assert check.max_error_deg is not None
    assert check.max_error_deg <= 1e-6


def test_a_t2_task_advances_by_continuity_after_one_seeded_pose(monkeypatch):
    """The rule itself: one label-seeded pose, then nearest-root for the rest.

    On the shipped fixtures the label lookup happens to agree with continuity at
    every shared-label input (measured: both approach directions return the same
    root), so this pins the selection rule rather than reproducing a hop.
    """
    mechanism = roberts_cognates(
        dict(cognate_fixtures())["crank-rocker"]).right.mechanism
    fixture = _chained_function_inputs(mechanism)
    assert fixture is not None, "no chained shared-label region on this link set"
    inputs, chain = fixture

    original = type(mechanism).pose_near
    followed: list[float] = []

    def recording(self, angle, reference):
        followed.append(float(angle))
        return original(self, angle, reference)

    monkeypatch.setattr(type(mechanism), "pose_near", recording)

    task = PublicTaskView(
        task_id="T2-continuity-rule",
        family="T2",
        target_kind="input-output angle function",
        tolerance=1e-6,
        input_range_deg=(inputs[0], inputs[-1]),
        function_samples=tuple(
            (value, pose.theta4_deg)
            for value, pose in zip(inputs, chain, strict=True)),
    )

    check = check_function(mechanism, task, branch=chain[0].branch)

    assert check.passed is True, check.reasons
    # Five declared inputs: one seeded by branch, four followed by continuity.
    assert followed == pytest.approx(inputs[1:])


# ------------------------------------------- pinned cognate angle tolerance


def test_the_cognate_angle_tolerance_is_pinned_and_reported():
    pinned = _pinned()
    mechanism = dict(cognate_fixtures())["crank-rocker"]

    report = check_cognates(
        roberts_cognates(mechanism), input_range_deg=(0.0, 355.0), samples=72,
        tolerance_mm=pinned.cognate_tolerance_mm,
        angle_tolerance_deg=pinned.cognate_angle_tolerance_deg)

    assert report.passed, report.details
    assert report.angle_tolerance_deg == pinned.cognate_angle_tolerance_deg
    assert "cognate_angle_gap" not in report.reasons
    assert report.record()["angle_tolerance_deg"] == pinned.cognate_angle_tolerance_deg

    verdict = run_gate(
        mechanism, _path_task_for(mechanism, family="T1"), preregistration=pinned,
        verify_cognates=True)
    assert verdict.passed is True, verdict.reasons
    assert (verdict.effective_settings["cognate_angle_tolerance_deg"]
            == pytest.approx(pinned.cognate_angle_tolerance_deg))
