"""Discrete Fréchet path metric and the T1 path check (blueprint §5.6).

Point-to-curve error lets a mechanism pass by hitting the right points in the
wrong order; the discrete Fréchet distance couples the two ordered sequences, so
order is part of the score. This is the metric §5.6 mandates and it is what
:func:`check_path` reports — with the target and generated samples retained, and
never described as a continuous certificate.

Three limits are handled here rather than left implicit:

* A trace that stopped early is not scored. Previously the samples that did
  assemble were aligned and given a Fréchet number, which reads as "the path was
  this close" when the mechanism may never reach the rest of the target. A
  partial trace now reports ``frechet_mm=None`` and
  ``partial_trace_not_scored``.
* A discrete distance is a function of the sampling. ``check_path(refine=True)``
  re-runs at increasing sample counts and reports whether the accept/reject
  decision survived, how much the number moved, and which counts were tested, so
  a tolerance that only just passes cannot hide inside an unlucky grid.
* Circuit continuity can be judged from the joints instead of the branch label.
  ``check_path(continuity=True)`` traces the nearest root at every step and
  reports geometric breaks, because the label is not a circuit invariant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from mmlm_matter_001.kinematics.boundary import assert_referee_caller
from mmlm_matter_001.kinematics.fourbar import (
    DEFAULT_MAX_JUMP_DEG,
    FourBar,
    MechanismError,
    Trace,
    circuit_breaks,
    circuit_switches,
    mean_toggle_spacing_deg,
    poses_share_circuit,
)
from mmlm_matter_001.kinematics.loop_closure import (
    ANGLE_TOLERANCE_DEG,
    RESIDUAL_TOLERANCE_MM,
    verify_poses,
)
from mmlm_matter_001.schemas import PublicTaskView

Point = tuple[float, float]

#: Machine-readable reasons a path check can fail.
PATH_REASONS = frozenset({
    "no_target_points",
    "unreachable_input",
    "partial_trace_not_scored",
    "branch_change",
    "target_order",
    "frechet_exceeds_tolerance",
    "refinement_unstable",
})

#: Sampling-stability settings, not protocol thresholds: how far refinement goes
#: before it stops asking whether the decision is grid-dependent. A gate pins the
#: starting sample count together with the tolerances it scores against.
DEFAULT_GROWTH = 2
DEFAULT_MAX_DOUBLINGS = 3
#: Default grid for a T1 path check. A gate does not use this default: it pins
#: the sample count in the preregistration and passes that.
BASE_PATH_SAMPLES = 181


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def discrete_frechet(target: Sequence[Point], generated: Sequence[Point]) -> float:
    """Exact discrete Fréchet distance between two ordered point sequences.

    Dynamic programming over the coupling graph, ``O(len(target) * len(generated))``.
    Raising on an empty sequence rather than returning 0 keeps "no data" from
    looking like a perfect match.
    """
    if not target or not generated:
        raise ValueError("the discrete Fréchet distance needs two non-empty sequences")
    for name, sequence in (("target", target), ("generated", generated)):
        for point in sequence:
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise ValueError(f"{name} holds a non-finite (x, y) pair: {point!r}")

    columns = len(generated)
    previous = [0.0] * columns
    first = target[0]
    for column in range(columns):
        step = _distance(first, generated[column])
        previous[column] = max(previous[column - 1], step) if column else step
    for row in range(1, len(target)):
        current = [0.0] * columns
        current[0] = max(previous[0], _distance(target[row], generated[0]))
        for column in range(1, columns):
            step = _distance(target[row], generated[column])
            current[column] = max(
                step,
                min(previous[column], previous[column - 1], current[column - 1]),
            )
        previous = current
    return previous[-1]


def max_chord_mm(points: Sequence[Point]) -> float | None:
    """Longest straight step between adjacent samples.

    The finest feature a polyline with this sampling can express; a deviation
    smaller than this is below the resolution of the grid, not evidence of
    agreement. ``None`` for fewer than two points.
    """
    if len(points) < 2:
        return None
    return max(
        _distance(points[index], points[index + 1])
        for index in range(len(points) - 1)
    )


@dataclass(frozen=True, slots=True)
class PathCheck:
    """The referee's verdict for a T1 path task."""

    passed: bool
    frechet_mm: float | None
    tolerance_mm: float
    reasons: tuple[str, ...]
    branch: str
    circuit_continuous: bool
    branch_switches: int
    order_ok: bool
    orientation_reversed: bool
    complete: bool
    target_points: int
    generated_points: int
    sample_count: int
    mean_sample_spacing_deg: float
    target_mm: tuple[Point, ...]
    generated_mm: tuple[Point, ...]
    loop_closure_verified: bool | None = None
    loop_closure_checked: int = 0
    max_loop_residual_mm: float | None = None
    continuity_tracked: bool = False
    #: The target's longest chord: the finest deviation this grid can resolve.
    resolution_mm: float | None = None
    #: Sample counts actually scored, finest last, when refinement was asked for.
    samples_tested: tuple[int, ...] = ()
    #: How far the distance moved between the requested grid and the finest one.
    frechet_change_mm: float | None = None
    #: Whether the accept/reject decision survived every refinement.
    refinement_stable: bool | None = None

    def record(self) -> dict[str, object]:
        """Everything §5.6 says to retain with a reported error."""
        return {
            "passed": self.passed,
            "frechet_mm": self.frechet_mm,
            "tolerance_mm": self.tolerance_mm,
            "reasons": list(self.reasons),
            "branch": self.branch,
            "circuit_continuous": self.circuit_continuous,
            "branch_switches": self.branch_switches,
            "order_ok": self.order_ok,
            "orientation_reversed": self.orientation_reversed,
            "complete": self.complete,
            "target_points": self.target_points,
            "generated_points": self.generated_points,
            "sample_count": self.sample_count,
            "mean_sample_spacing_deg": self.mean_sample_spacing_deg,
            "target_mm": [list(point) for point in self.target_mm],
            "generated_mm": [list(point) for point in self.generated_mm],
            "loop_closure_verified": self.loop_closure_verified,
            "loop_closure_checked": self.loop_closure_checked,
            "max_loop_residual_mm": self.max_loop_residual_mm,
            "continuity_tracked": self.continuity_tracked,
            "resolution_mm": self.resolution_mm,
            "samples_tested": list(self.samples_tested),
            "frechet_change_mm": self.frechet_change_mm,
            "refinement_stable": self.refinement_stable,
        }


def _closest_index(points: Sequence[Point], probe: Point) -> int:
    best_index, best_distance = 0, math.inf
    for index, point in enumerate(points):
        distance = _distance(point, probe)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def _cycle_from(points: Sequence[Point], start: int, *, reverse: bool = False) -> tuple[Point, ...]:
    """Read a closed sequence once from `start`, forwards or backwards."""
    count = len(points)
    stride = -1 if reverse else 1
    return tuple(points[(start + stride * index) % count] for index in range(count))


def _empty_check(task: PublicTaskView, trace: Trace | None, branch: str, samples: int,
                 reason: str) -> PathCheck:
    return PathCheck(
        passed=False,
        frechet_mm=None,
        tolerance_mm=task.tolerance,
        reasons=(reason,),
        branch=branch,
        circuit_continuous=False,
        branch_switches=0,
        order_ok=False,
        orientation_reversed=False,
        complete=False,
        target_points=len(task.target_points),
        generated_points=0 if trace is None else len(trace.poses),
        sample_count=samples,
        mean_sample_spacing_deg=0.0 if trace is None else mean_toggle_spacing_deg(trace.poses),
        target_mm=tuple(task.target_points),
        generated_mm=() if trace is None else trace.points_mm,
    )


def _check_once(
    mechanism: FourBar,
    task: PublicTaskView,
    *,
    branch: str,
    samples: int,
    allow_reversed: bool,
    verify_loop_closure: bool,
    loop_angle_tolerance_deg: float,
    loop_residual_tolerance_mm: float,
    continuity: bool,
    max_jump_deg: float,
) -> PathCheck:
    if not task.target_points:
        return _empty_check(task, None, branch, samples, "no_target_points")
    try:
        # The sweep is continuity-tracked in both modes. A label-per-sample sweep
        # is not a safe default: it can hop to the other assembly where the two
        # roots share a label. `continuity` selects how the result is judged.
        trace = mechanism.trace(
            task.input_range_deg, samples=samples, branch=branch, strict=False)
    except MechanismError:
        return _empty_check(task, None, branch, samples, "unreachable_input")

    reasons: list[str] = []
    if not trace.complete:
        reasons.append("unreachable_input")
    if continuity:
        breaks = circuit_breaks(trace.poses, max_jump_deg=max_jump_deg)
        continuous = trace.complete and breaks == 0
        switches = breaks
    else:
        try:
            continuous = poses_share_circuit(trace.poses)
        except MechanismError:
            continuous = False
        switches = circuit_switches(trace.poses)
    if not continuous:
        reasons.append("branch_change")

    target = tuple(task.target_points)
    generated = trace.points_mm
    order_ok = True
    orientation_reversed = False
    distance: float | None = None
    aligned: tuple[Point, ...] = generated
    if not trace.complete:
        # A truncated trace describes a mechanism that never reaches the rest of
        # the target. Scoring the part that did assemble would report a distance
        # for a path the mechanism cannot follow.
        reasons.append("partial_trace_not_scored")
        order_ok = False
    elif generated:
        # A closed target (first and last event are the same point within the
        # task tolerance) is compared over a whole cycle; an open target is
        # trimmed after its last event, because the return stroke is
        # unconstrained (§5.6).
        closed = _distance(target[0], target[-1]) <= task.tolerance
        start = _closest_index(generated, target[0])
        if closed:
            # A closed path has no preferred end, so both directions are read
            # from the same start and compared without a rotation penalty.
            forward = _cycle_from(generated, start)
            backward = _cycle_from(generated, start, reverse=True)
        else:
            last, best = start, math.inf
            for index in range(start, len(generated)):
                candidate = _distance(generated[index], target[-1])
                if candidate < best:
                    last, best = index, candidate
            forward = generated[start:last + 1] if last > start else generated[start:start + 1]
            backward = tuple(reversed(forward))
        distance = discrete_frechet(target, forward)
        reversed_distance = discrete_frechet(target, backward)
        if reversed_distance < distance:
            if allow_reversed:
                distance = reversed_distance
                orientation_reversed = True
                aligned = backward
            else:
                order_ok = False
                reasons.append("target_order")
                aligned = forward
        else:
            aligned = forward
        if distance > task.tolerance:
            reasons.append("frechet_exceeds_tolerance")

    verification = None
    if verify_loop_closure and trace.poses:
        try:
            verification = verify_poses(
                mechanism,
                trace.poses,
                angle_tolerance_deg=loop_angle_tolerance_deg,
                residual_tolerance_mm=loop_residual_tolerance_mm,
            )
        except MechanismError:
            verification = None
        if verification is None or not verification.verified:
            reasons.append("loop_closure_disagrees")

    return PathCheck(
        passed=not reasons,
        frechet_mm=distance,
        tolerance_mm=task.tolerance,
        reasons=tuple(reasons),
        branch=branch,
        circuit_continuous=continuous,
        branch_switches=switches,
        order_ok=order_ok,
        orientation_reversed=orientation_reversed,
        complete=trace.complete,
        target_points=len(target),
        generated_points=len(aligned),
        sample_count=samples,
        mean_sample_spacing_deg=mean_toggle_spacing_deg(trace.poses),
        target_mm=target,
        generated_mm=tuple(aligned),
        loop_closure_verified=None if verification is None else verification.verified,
        loop_closure_checked=0 if verification is None else verification.checked,
        max_loop_residual_mm=(
            None if verification is None else verification.max_residual_mm),
        continuity_tracked=trace.continuity_tracked,
        resolution_mm=max_chord_mm(target),
    )


def check_path(
    mechanism: FourBar,
    task: PublicTaskView,
    *,
    branch: str = "open",
    samples: int = BASE_PATH_SAMPLES,
    allow_reversed: bool = False,
    verify_loop_closure: bool = False,
    loop_angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    loop_residual_tolerance_mm: float = RESIDUAL_TOLERANCE_MM,
    continuity: bool = False,
    max_jump_deg: float = DEFAULT_MAX_JUMP_DEG,
    refine: bool = False,
    growth: int = DEFAULT_GROWTH,
    max_doublings: int = DEFAULT_MAX_DOUBLINGS,
) -> PathCheck:
    """Score a mechanism's coupler path against a T1 task (§5.6).

    The tolerance comes from the task; this function never decides what counts
    as close enough. Reversal is off unless the task explicitly allows it.

    ``refine=True`` re-scores at ``growth``x the sample count up to
    ``max_doublings`` times and asks a narrower question than "did the number
    stop moving": did the **accept/reject decision** change? The task's target is
    a fixed polyline, so a finer generated sampling exposes the target's own
    interpolation error — the reported distance is expected to move. What must
    not happen is a verdict that flips with grid density, so a change of decision
    is refused as ``refinement_unstable``. The reported distance stays the one at
    the requested grid, which is the preregistered measurement.

    ``resolution_mm`` on the verdict is the target's own longest chord: the finest
    feature its sampling can express, and therefore the finest deviation this
    check can distinguish from zero.
    """
    if samples < 2:
        raise MechanismError(f"samples must be at least 2, got {samples!r}")
    assert_referee_caller("check_path")
    if refine and (growth < 2 or max_doublings < 1):
        raise MechanismError(
            f"refinement needs growth >= 2 and max_doublings >= 1, got "
            f"{growth!r} and {max_doublings!r}")

    options = {
        "branch": branch,
        "allow_reversed": allow_reversed,
        "verify_loop_closure": verify_loop_closure,
        "loop_angle_tolerance_deg": loop_angle_tolerance_deg,
        "loop_residual_tolerance_mm": loop_residual_tolerance_mm,
        "continuity": continuity,
        "max_jump_deg": max_jump_deg,
    }
    current = _check_once(mechanism, task, samples=samples, **options)
    if not refine:
        return current

    tolerance = current.tolerance_mm
    tested = [samples]
    distances = [current.frechet_mm]
    decisions = [
        None if current.frechet_mm is None else current.frechet_mm <= tolerance]
    for _ in range(max_doublings):
        candidate = _check_once(
            mechanism, task, samples=tested[-1] * growth, **options)
        tested.append(candidate.sample_count)
        distances.append(candidate.frechet_mm)
        decisions.append(
            None if candidate.frechet_mm is None else candidate.frechet_mm <= tolerance)
        if candidate.frechet_mm is None:
            break

    known = [distance for distance in distances if distance is not None]
    change = None if len(known) < 2 else abs(known[-1] - known[0])
    stable = (
        None if any(decision is None for decision in decisions)
        else len(set(decisions)) == 1
    )
    reasons = list(current.reasons)
    if stable is False and "refinement_unstable" not in reasons:
        reasons.append("refinement_unstable")
    return replace(
        current,
        passed=not reasons,
        reasons=tuple(reasons),
        samples_tested=tuple(tested),
        frechet_change_mm=change,
        refinement_stable=stable,
    )
