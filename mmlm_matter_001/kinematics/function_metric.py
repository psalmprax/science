"""T2 function-generation metric (blueprint §5.7).

A T2 task declares an input/output angle relationship: ordered
``(input_deg, output_deg)`` samples over a specified input range. The checker
solves the mechanism at each declared input on the requested assembly branch and
compares the achieved output angle ``theta4`` with the declared one.

The error is angular, so it is wrapped: a mechanism returning ``-179.9`` for a
declared ``179.9`` is off by ``0.2`` degrees, not ``359.8``. Both the maximum and
the RMS error are reported, the tolerance comes from the task and is never tuned
here, unreachable inputs and branch changes are rejections rather than large
errors, and the samples that do not cover the declared input range are refused —
"evaluated the whole range" is a claim the result has to support.

The declared samples *are* the Freudenstein relationship, so comparing the
achieved output angle with them is the residual expressed in the metric the task
preregisters (§5.7).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from mmlm_matter_001.kinematics.boundary import assert_referee_caller
from mmlm_matter_001.kinematics.fourbar import (
    FourBar,
    MechanismError,
    Pose,
    angle_difference_deg,
    circuit_switches,
    poses_share_circuit,
)
from mmlm_matter_001.kinematics.loop_closure import (
    ANGLE_TOLERANCE_DEG,
    RESIDUAL_TOLERANCE_MM,
    verify_poses,
)
from mmlm_matter_001.schemas import PublicTaskView

#: Machine-readable reasons a function check can fail.
FUNCTIONS_REASONS = frozenset({
    "no_function_samples",
    "unreachable_input",
    "range_not_covered",
    "branch_change",
    "function_error_exceeds_tolerance",
    "loop_closure_disagrees",
})

#: How far the declared samples may fall short of the task's input range before
#: the range counts as not evaluated. A numerical guard, not a protocol value.
_COVERAGE_SLACK_DEG = 1e-9


@dataclass(frozen=True, slots=True)
class FunctionPoint:
    """One declared input/output pair and what the mechanism actually did."""

    input_deg: float
    target_output_deg: float
    achieved_output_deg: float | None
    error_deg: float | None


@dataclass(frozen=True, slots=True)
class FunctionCheck:
    """The referee's verdict for a T2 function-generation task."""

    passed: bool
    max_error_deg: float | None
    rms_error_deg: float | None
    tolerance_deg: float
    reasons: tuple[str, ...]
    branch: str
    circuit_continuous: bool
    branch_switches: int
    reachable_samples: int
    sample_count: int
    range_covered: bool
    points: tuple[FunctionPoint, ...]
    loop_closure_verified: bool | None = None
    loop_closure_checked: int = 0
    max_loop_residual_mm: float | None = None

    def record(self) -> dict[str, object]:
        """Everything §5.7 says to retain with a reported error."""
        return {
            "passed": self.passed,
            "max_error_deg": self.max_error_deg,
            "rms_error_deg": self.rms_error_deg,
            "tolerance_deg": self.tolerance_deg,
            "reasons": list(self.reasons),
            "branch": self.branch,
            "circuit_continuous": self.circuit_continuous,
            "branch_switches": self.branch_switches,
            "reachable_samples": self.reachable_samples,
            "sample_count": self.sample_count,
            "range_covered": self.range_covered,
            "loop_closure_verified": self.loop_closure_verified,
            "loop_closure_checked": self.loop_closure_checked,
            "max_loop_residual_mm": self.max_loop_residual_mm,
            "points": [
                {
                    "input_deg": point.input_deg,
                    "target_output_deg": point.target_output_deg,
                    "achieved_output_deg": point.achieved_output_deg,
                    "error_deg": point.error_deg,
                }
                for point in self.points
            ],
        }


def _empty_function_check(task: PublicTaskView, reason: str) -> FunctionCheck:
    return FunctionCheck(
        passed=False,
        max_error_deg=None,
        rms_error_deg=None,
        tolerance_deg=task.tolerance,
        reasons=(reason,),
        branch="open",
        circuit_continuous=False,
        branch_switches=0,
        reachable_samples=0,
        sample_count=0,
        range_covered=False,
        points=(),
    )


def score_function_samples(
    task: PublicTaskView,
    poses: Sequence[Pose | None],
    *,
    branch: str = "open",
    range_covered: bool = True,
) -> FunctionCheck:
    """Score already-solved poses against a T2 task's declared relationship.

    ``poses`` aligns one-to-one with ``task.function_samples``; ``None`` marks an
    input the mechanism could not assemble. Split out from :func:`check_function`
    so a recorded early run can be re-scored without re-solving it.
    """
    targets = tuple(task.function_samples)
    if not targets:
        return _empty_function_check(task, "no_function_samples")
    if len(poses) != len(targets):
        raise ValueError(
            f"got {len(poses)} poses for {len(targets)} declared samples; every declared "
            "input needs exactly one pose or None")
    usable = [pose for pose in poses if pose is not None]
    try:
        continuous = poses_share_circuit(usable) if usable else False
    except MechanismError:
        continuous = False

    reasons: list[str] = []
    if len(usable) != len(poses):
        reasons.append("unreachable_input")
    if not continuous:
        reasons.append("branch_change")
    if not range_covered:
        reasons.append("range_not_covered")

    points: list[FunctionPoint] = []
    errors: list[float] = []
    for (input_deg, target_output_deg), pose in zip(targets, poses, strict=True):
        if pose is None:
            points.append(FunctionPoint(
                input_deg=input_deg,
                target_output_deg=target_output_deg,
                achieved_output_deg=None,
                error_deg=None,
            ))
            continue
        error = abs(angle_difference_deg(target_output_deg, pose.theta4_deg))
        errors.append(error)
        points.append(FunctionPoint(
            input_deg=input_deg,
            target_output_deg=target_output_deg,
            achieved_output_deg=pose.theta4_deg,
            error_deg=error,
        ))

    max_error = max(errors) if errors else None
    rms_error = (
        math.sqrt(sum(error * error for error in errors) / len(errors))
        if errors else None
    )
    if max_error is not None and max_error > task.tolerance:
        reasons.append("function_error_exceeds_tolerance")

    return FunctionCheck(
        passed=not reasons,
        max_error_deg=max_error,
        rms_error_deg=rms_error,
        tolerance_deg=task.tolerance,
        reasons=tuple(reasons),
        branch=branch,
        circuit_continuous=continuous,
        branch_switches=circuit_switches(usable),
        reachable_samples=len(usable),
        sample_count=len(poses),
        range_covered=range_covered,
        points=tuple(points),
    )


def check_function(
    mechanism: FourBar,
    task: PublicTaskView,
    *,
    branch: str = "open",
    verify_loop_closure: bool = False,
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    residual_tolerance_mm: float = RESIDUAL_TOLERANCE_MM,
) -> FunctionCheck:
    """Evaluate a mechanism over a T2 task's declared input range (§5.7).

    The tolerance is the task's preregistered angular tolerance in degrees; this
    function never decides what "close enough" means.    ``verify_loop_closure=True``
    asks the independent numerical solver to confirm that every pose scored here
    actually closes its loop, so a T2 pass is not resting on the closed form
    alone.

    The declared inputs are scored on one continuous assembly: the first is
    solved on the requested branch, and each later one takes the root nearest the
    pose before it. Picking every sample independently by branch label can return
    the other assembly where the two roots share a label, which is a branch hop a
    real mechanism cannot make.
    """
    assert_referee_caller("check_function")
    targets = tuple(task.function_samples)
    if not targets:
        return _empty_function_check(task, "no_function_samples")

    poses: list[Pose | None] = []
    previous: Pose | None = None
    for input_deg, _ in targets:
        try:
            pose = (
                mechanism.pose_on(input_deg, branch) if previous is None
                else mechanism.pose_near(input_deg, previous)
            )
        except MechanismError:
            poses.append(None)
            previous = None
            continue
        poses.append(pose)
        previous = pose

    declared = [input_deg for input_deg, _ in targets]
    low, high = task.input_range_deg
    covered = (
        min(declared) <= low + _COVERAGE_SLACK_DEG
        and max(declared) >= high - _COVERAGE_SLACK_DEG
    )
    check = score_function_samples(
        task, tuple(poses), branch=branch, range_covered=covered)
    if not verify_loop_closure:
        return check

    usable = [pose for pose in poses if pose is not None]
    if not usable:
        return replace(check, loop_closure_verified=False)
    try:
        verification = verify_poses(
            mechanism,
            usable,
            angle_tolerance_deg=angle_tolerance_deg,
            residual_tolerance_mm=residual_tolerance_mm,
        )
    except MechanismError:
        reasons = tuple(dict.fromkeys((*check.reasons, "loop_closure_disagrees")))
        return replace(
            check, passed=False, reasons=reasons, loop_closure_verified=False)
    reasons = check.reasons
    if not verification.verified:
        reasons = tuple(dict.fromkeys((*check.reasons, "loop_closure_disagrees")))
    return replace(
        check,
        passed=not reasons,
        reasons=reasons,
        loop_closure_verified=verification.verified,
        loop_closure_checked=verification.checked,
        max_loop_residual_mm=verification.max_residual_mm,
    )


def coverage_gap_deg(task: PublicTaskView) -> tuple[float, float]:
    """How far the declared samples fall short of the task's input range."""
    targets = tuple(task.function_samples)
    if not targets:
        return (task.input_range_deg[1] - task.input_range_deg[0], 0.0)
    declared = [input_deg for input_deg, _ in targets]
    low, high = task.input_range_deg
    return (max(0.0, min(declared) - low), max(0.0, high - max(declared)))
