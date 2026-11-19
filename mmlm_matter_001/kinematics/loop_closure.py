"""Independent numerical loop-closure solver for four-bars (blueprint §5.5, §5.8).

§5.8 requires an independent numerical loop-closure implementation that agrees
with the closed form across random feasible four-bars, and §5.5 requires a
separate loop-closure checker for adjudication rather than one solver doing both
jobs. This module is that independent implementation, and it shares no algebra
with :mod:`mmlm_matter_001.kinematics.fourbar`:

* It solves the two scalar loop-closure equations in ``(theta3, theta4)`` by
  damped Newton iteration on the residual, not by the Freudenstein tangent
  substitution.
* Its starting points come from a coarse grid over the joint-angle torus, so it
  is not seeded with the closed-form answer it is meant to check.
* Its only shared code is the branch convention (``branch_of_theta3``), which is
  a definition of which assembly is which, not a solution method.

Near a dead centre the Jacobian becomes singular: ``det J`` vanishes when the
coupler and output link are collinear. A plain Newton step has no direction to
move in there, so the iteration uses a ridged (Levenberg) normal-equation step
that stays solvable at a tangency. Convergence is still judged only by the
loop-closure residual, and a start that does not reach it is never returned as a
root.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from mmlm_matter_001.kinematics.fourbar import (
    FourBar,
    MechanismError,
    Pose,
    angle_difference_deg,
    branch_of_theta3,
    wrap_deg,
)

#: Solver settings, not protocol thresholds: how hard Newton tries and how
#: closely the numerical and closed-form answers must agree to count as agreeing.
RESIDUAL_TOLERANCE_MM = 1e-10
ANGLE_TOLERANCE_DEG = 1e-6
MAX_ITERATIONS = 120
GRID_TICKS = 16
MAX_STARTS = 20
MAX_STEP_RAD = 0.5
#: Levenberg ridge, relative to the size of the normal-equation matrix.
RIDGE_RELATIVE = 1e-12


class LoopClosureError(MechanismError):
    """The numerical solver could not be applied as asked."""


@dataclass(frozen=True, slots=True)
class NumericalSolution:
    """One root of the loop-closure equations."""

    theta2_deg: float
    theta3_deg: float
    theta4_deg: float
    residual_mm: float
    iterations: int
    branch: str
    converged: bool
    singular: bool = False


@dataclass(frozen=True, slots=True)
class CrossCheckReport:
    """Agreement between the closed form and the independent numerical solver."""

    mechanisms: int
    angles: int
    comparisons: int
    missing_branches: int
    max_theta3_error_deg: float | None
    max_theta4_error_deg: float | None
    max_residual_mm: float | None
    tolerance_deg: float
    tolerance_mm: float
    passed: bool
    details: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "mechanisms": self.mechanisms,
            "angles": self.angles,
            "comparisons": self.comparisons,
            "missing_branches": self.missing_branches,
            "max_theta3_error_deg": self.max_theta3_error_deg,
            "max_theta4_error_deg": self.max_theta4_error_deg,
            "max_residual_mm": self.max_residual_mm,
            "tolerance_deg": self.tolerance_deg,
            "tolerance_mm": self.tolerance_mm,
            "passed": self.passed,
            "details": list(self.details),
        }


@dataclass(frozen=True, slots=True)
class PoseVerification:
    """A second opinion on the poses an accepted path is built from."""

    verified: bool
    checked: int
    disagreements: int
    max_angle_gap_deg: float
    max_residual_mm: float
    details: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "checked": self.checked,
            "disagreements": self.disagreements,
            "max_angle_gap_deg": self.max_angle_gap_deg,
            "max_residual_mm": self.max_residual_mm,
            "details": list(self.details),
        }


def verify_poses(
    mechanism: FourBar,
    poses: Sequence[Pose],
    *,
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    residual_tolerance_mm: float = RESIDUAL_TOLERANCE_MM,
) -> PoseVerification:
    """Re-solve every pose's input from blind seeds and compare.

    The path check scores the coupler curve; it does not ask whether each pose it
    accepted actually closes its loop. This does, with the independent solver and
    no seed taken from the pose, so a closed form that is self-consistent but
    wrong (or a pose list that was assembled by hand) cannot pass quietly.
    """
    if not isinstance(mechanism, FourBar):
        raise LoopClosureError(
            f"verifying poses needs a FourBar, got {type(mechanism).__name__}")
    if not poses:
        raise LoopClosureError("verifying poses needs at least one pose")
    details: list[str] = []
    checked = disagreements = 0
    max_gap = max_residual = 0.0
    for pose in poses:
        solutions = [
            solution
            for solution in solve_numerically(mechanism, pose.theta2_deg)
            if solution.converged
        ]
        if not solutions:
            disagreements += 1
            if len(details) < 5:
                details.append(
                    f"input {pose.theta2_deg:.6g} deg: the numerical solver found "
                    "no root")
            continue
        best = min(
            solutions,
            key=lambda solution: max(
                abs(angle_difference_deg(pose.theta3_deg, solution.theta3_deg)),
                abs(angle_difference_deg(pose.theta4_deg, solution.theta4_deg)),
            ),
        )
        gap = max(
            abs(angle_difference_deg(pose.theta3_deg, best.theta3_deg)),
            abs(angle_difference_deg(pose.theta4_deg, best.theta4_deg)),
        )
        checked += 1
        max_gap = max(max_gap, gap)
        max_residual = max(max_residual, best.residual_mm)
        if gap > angle_tolerance_deg or best.residual_mm > residual_tolerance_mm:
            disagreements += 1
            if len(details) < 5:
                details.append(
                    f"input {pose.theta2_deg:.6g} deg: the numerical solver "
                    f"disagrees by {gap:.3g} deg")
    return PoseVerification(
        verified=disagreements == 0 and checked == len(poses),
        checked=checked,
        disagreements=disagreements,
        max_angle_gap_deg=max_gap,
        max_residual_mm=max_residual,
        details=tuple(details),
    )


def residuals(
    mechanism: FourBar,
    theta2_deg: float,
    theta3_deg: float,
    theta4_deg: float,
) -> tuple[float, float]:
    """The loop-closure residual in millimetres.

    ``a e^{i theta2} + b e^{i theta3} - c e^{i theta4} - d = 0`` written as two
    real equations. Both must vanish for an assembled configuration.
    """
    theta2 = math.radians(theta2_deg)
    theta3 = math.radians(theta3_deg)
    theta4 = math.radians(theta4_deg)
    a, b = mechanism.input_mm, mechanism.coupler_mm
    c, d = mechanism.output_mm, mechanism.ground_mm
    return (
        a * math.cos(theta2) + b * math.cos(theta3) - c * math.cos(theta4) - d,
        a * math.sin(theta2) + b * math.sin(theta3) - c * math.sin(theta4),
    )


def _newton(
    mechanism: FourBar,
    theta2_deg: float,
    theta3_deg: float,
    theta4_deg: float,
    *,
    tolerance_mm: float,
    max_iterations: int,
) -> NumericalSolution:
    b, c = mechanism.coupler_mm, mechanism.output_mm
    theta3, theta4 = theta3_deg, theta4_deg
    for iteration in range(1, max_iterations + 1):
        first, second = residuals(mechanism, theta2_deg, theta3, theta4)
        residual = math.hypot(first, second)
        if residual <= tolerance_mm:
            return NumericalSolution(
                theta2_deg=wrap_deg(theta2_deg),
                theta3_deg=wrap_deg(theta3),
                theta4_deg=wrap_deg(theta4),
                residual_mm=residual,
                iterations=iteration,
                branch=branch_of_theta3(theta3),
                converged=True,
            )
        radians3, radians4 = math.radians(theta3), math.radians(theta4)
        j11 = -b * math.sin(radians3)
        j12 = c * math.sin(radians4)
        j21 = b * math.cos(radians3)
        j22 = -c * math.cos(radians4)
        # Ridged normal equations: solvable even where det(J) = 0, which is
        # exactly a tangency between the coupler and output circles.
        gradient3 = j11 * first + j21 * second
        gradient4 = j12 * first + j22 * second
        hessian11 = j11 * j11 + j21 * j21
        hessian12 = j11 * j12 + j21 * j22
        hessian22 = j12 * j12 + j22 * j22
        ridge = RIDGE_RELATIVE * max(hessian11 + hessian22, 1.0)
        matrix_a, matrix_c = hessian11 + ridge, hessian22 + ridge
        determinant = matrix_a * matrix_c - hessian12 * hessian12
        if determinant <= 0.0 or not math.isfinite(determinant):
            return NumericalSolution(
                theta2_deg=wrap_deg(theta2_deg),
                theta3_deg=wrap_deg(theta3),
                theta4_deg=wrap_deg(theta4),
                residual_mm=residual,
                iterations=iteration,
                branch=branch_of_theta3(theta3),
                converged=False,
                singular=True,
            )
        step3 = (-gradient3 * matrix_c + hessian12 * gradient4) / determinant
        step4 = (-gradient4 * matrix_a + hessian12 * gradient3) / determinant
        length = math.hypot(step3, step4)
        if length > MAX_STEP_RAD:
            step3 *= MAX_STEP_RAD / length
            step4 *= MAX_STEP_RAD / length
        theta3 += math.degrees(step3)
        theta4 += math.degrees(step4)
    first, second = residuals(mechanism, theta2_deg, theta3, theta4)
    return NumericalSolution(
        theta2_deg=wrap_deg(theta2_deg),
        theta3_deg=wrap_deg(theta3),
        theta4_deg=wrap_deg(theta4),
        residual_mm=math.hypot(first, second),
        iterations=max_iterations,
        branch=branch_of_theta3(theta3),
        converged=False,
    )


def _starting_points(mechanism: FourBar, theta2_deg: float) -> list[tuple[float, float]]:
    """Rank a coarse grid of joint configurations by residual, best first."""
    scored: list[tuple[float, float, float]] = []
    for third in range(GRID_TICKS):
        for fourth in range(GRID_TICKS):
            theta3 = 360.0 * third / GRID_TICKS
            theta4 = 360.0 * fourth / GRID_TICKS
            first, second = residuals(mechanism, theta2_deg, theta3, theta4)
            scored.append((math.hypot(first, second), theta3, theta4))
    scored.sort(key=lambda item: item[0])
    starts: list[tuple[float, float]] = []
    for _, theta3, theta4 in scored:
        if all(
            abs(angle_difference_deg(theta3, other3)) > 20.0
            or abs(angle_difference_deg(theta4, other4)) > 20.0
            for other3, other4 in starts
        ):
            starts.append((theta3, theta4))
        if len(starts) >= MAX_STARTS:
            break
    return starts


def solve_numerically(
    mechanism: FourBar,
    theta2_deg: float,
    *,
    tolerance_mm: float = RESIDUAL_TOLERANCE_MM,
    max_iterations: int = MAX_ITERATIONS,
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
) -> tuple[NumericalSolution, ...]:
    """All loop-closure roots at one input angle, from blind multistart Newton."""
    if not isinstance(mechanism, FourBar):
        raise LoopClosureError(
            f"the numerical solver needs a FourBar, got {type(mechanism).__name__}")
    if not math.isfinite(theta2_deg):
        raise LoopClosureError(f"theta2_deg must be finite, got {theta2_deg!r}")
    solutions: list[NumericalSolution] = []
    for theta3, theta4 in _starting_points(mechanism, theta2_deg):
        candidate = _newton(
            mechanism, theta2_deg, theta3, theta4,
            tolerance_mm=tolerance_mm, max_iterations=max_iterations)
        if not candidate.converged:
            continue
        if all(
            abs(angle_difference_deg(candidate.theta3_deg, other.theta3_deg))
            > angle_tolerance_deg
            or abs(angle_difference_deg(candidate.theta4_deg, other.theta4_deg))
            > angle_tolerance_deg
            for other in solutions
        ):
            solutions.append(candidate)
    return tuple(solutions)


def random_grashof_mechanisms(
    count: int,
    *,
    seed: int = 0,
    margin_mm: float = 1.0,
    length_range_mm: tuple[float, float] = (10.0, 60.0),
    attempts_per_mechanism: int = 400,
) -> tuple[FourBar, ...]:
    """Deterministic random crank-rockers with a strict full-rotation input crank.

    Only the crank-rocker inversion with the shortest link as the input crank is
    drawn. Both of its assembly circuits exist for every input angle, so a
    cross-check sample never silently compares against a configuration that does
    not exist; a drag-link has a full-rotation input but can lose one branch
    mid-cycle, and a double-rocker has no full rotation at all. Mechanisms must
    also clear ``margin_mm`` of Grashof slack.
    """
    if count < 1:
        raise LoopClosureError(f"count must be positive, got {count!r}")
    low, high = length_range_mm
    if not (0.0 < low < high):
        raise LoopClosureError(
            f"length_range_mm must be positive and increasing, got {length_range_mm!r}")
    rng = random.Random(seed)
    mechanisms: list[FourBar] = []
    for _ in range(count):
        for _attempt in range(attempts_per_mechanism):
            lengths = [rng.uniform(low, high) for _ in range(4)]
            candidate = FourBar(
                ground_mm=lengths[0],
                input_mm=lengths[1],
                coupler_mm=lengths[2],
                output_mm=lengths[3],
                coupler_point_mm=(rng.uniform(0.0, lengths[2]), rng.uniform(-5.0, 5.0)),
            )
            grashof = candidate.check_grashof(margin_mm=margin_mm)
            if (
                grashof.is_grashof
                and grashof.inversion == "crank-rocker"
                and grashof.shortest_link == "input"
            ):
                mechanisms.append(candidate)
                break
        else:
            raise LoopClosureError(
                f"could not draw {count} full-rotation crank-rockers in "
                f"{attempts_per_mechanism} attempts each; widen length_range_mm")
    return tuple(mechanisms)


def cross_check(
    *,
    mechanism_count: int = 8,
    angles: int = 8,
    seed: int = 0,
    tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    tolerance_mm: float = RESIDUAL_TOLERANCE_MM,
    margin_mm: float = 1.0,
) -> CrossCheckReport:
    """Compare the closed form with the independent solver on random mechanisms.

    Agrees only if, for every mechanism and input angle, both assembly branches
    are found by both solvers and the coupler and output angles match within
    ``tolerance_deg``. A missing branch is a failure, not a skipped sample.
    """
    if angles < 1:
        raise LoopClosureError(f"angles must be positive, got {angles!r}")
    mechanisms = random_grashof_mechanisms(mechanism_count, seed=seed, margin_mm=margin_mm)
    comparisons = missing = 0
    max_theta3 = max_theta4 = max_residual = 0.0
    details: list[str] = []
    for mechanism_index, mechanism in enumerate(mechanisms):
        for index in range(angles):
            # Offset off exact grid points so no sample lands on a dead centre.
            theta2_deg = 360.0 * (index + 0.37) / angles
            numerical = {
                solution.branch: solution
                for solution in solve_numerically(
                    mechanism, theta2_deg, tolerance_mm=tolerance_mm)
                if solution.converged
            }
            # At a tangency the two closed-form branches coincide into one pose,
            # so the comparison is between the poses each solver finds rather
            # than against a fixed expectation of two labelled branches.
            closed_poses = []
            for branch in ("open", "crossed"):
                try:
                    pose = mechanism.solve(theta2_deg, branch=branch)
                except MechanismError:
                    continue
                if all(
                    abs(angle_difference_deg(pose.theta3_deg, other.theta3_deg)) > tolerance_deg
                    or abs(angle_difference_deg(pose.theta4_deg, other.theta4_deg))
                    > tolerance_deg
                    for other in closed_poses
                ):
                    closed_poses.append(pose)
            if not closed_poses:
                missing += 1
                if len(details) < 5:
                    details.append(
                        f"mechanism {mechanism_index} at {theta2_deg:.6g} deg: closed form "
                        f"has no assembly")
                continue
            unmatched = list(numerical.values())
            for expected in closed_poses:
                match = next(
                    (
                        solution for solution in unmatched
                        if abs(angle_difference_deg(
                            expected.theta3_deg, solution.theta3_deg)) <= tolerance_deg
                        and abs(angle_difference_deg(
                            expected.theta4_deg, solution.theta4_deg)) <= tolerance_deg
                    ),
                    None,
                )
                if match is None:
                    missing += 1
                    if len(details) < 5:
                        details.append(
                            f"mechanism {mechanism_index} at {theta2_deg:.6g} deg: numerical "
                            f"solver found no pose matching the closed-form branch "
                            f"{expected.branch}")
                    continue
                unmatched.remove(match)
                comparisons += 1
                error3 = abs(angle_difference_deg(expected.theta3_deg, match.theta3_deg))
                error4 = abs(angle_difference_deg(expected.theta4_deg, match.theta4_deg))
                max_theta3 = max(max_theta3, error3)
                max_theta4 = max(max_theta4, error4)
                max_residual = max(max_residual, match.residual_mm)
                if len(details) < 5 and max(error3, error4) > tolerance_deg:
                    details.append(
                        f"mechanism {mechanism_index} at {theta2_deg:.6g} deg "
                        f"{expected.branch}: theta3 differs by {error3:.3g} deg, "
                        f"theta4 by {error4:.3g} deg")
            for extra in unmatched:
                missing += 1
                if len(details) < 5:
                    details.append(
                        f"mechanism {mechanism_index} at {theta2_deg:.6g} deg: numerical "
                        f"solver found an extra {extra.branch} pose")
    passed = (
        comparisons > 0
        and missing == 0
        and max_theta3 <= tolerance_deg
        and max_theta4 <= tolerance_deg
        and max_residual <= tolerance_mm
    )
    return CrossCheckReport(
        mechanisms=len(mechanisms),
        angles=angles,
        comparisons=comparisons,
        missing_branches=missing,
        max_theta3_error_deg=max_theta3,
        max_theta4_error_deg=max_theta4,
        max_residual_mm=max_residual,
        tolerance_deg=tolerance_deg,
        tolerance_mm=tolerance_mm,
        passed=passed,
        details=tuple(details),
    )
