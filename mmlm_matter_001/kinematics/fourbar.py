"""Closed-form four-bar kinematics (blueprint §5.1–§5.4).

Coordinates and convention, fixed once for the whole experiment:

* All lengths in millimetres, all angles in degrees at the API boundary.
* Origin at the ground pivot ``O2``; the ground link runs along ``+x`` to the
  ground pivot ``O4`` at ``(d, 0)``. The plane is right-handed, ``+y`` up.
* Link order: ``input`` (``a``, ``O2A``), ``coupler`` (``b``, ``AB``), ``output``
  (``c``, ``O4B``), ``ground`` (``d``).
* Angles are measured counter-clockwise from ``+x``. ``theta2`` is the input
  crank angle, ``theta3`` the coupler direction ``A->B`` and ``theta4`` the
  output direction ``O4->B``.
* The assembly branch is the sign of ``sin(theta3)``: ``open`` for a positive
  sign (coupler above the ``A`` pivot), ``crossed`` for a negative one, and
  ``toggle`` at the dead-centre positions where the sign vanishes and the two
  branches meet.

Position analysis is the standard Freudenstein closed form solved through the
half-angle tangent, which keeps the sign of the square-root term that
distinguishes the two assembly branches (§5.3). Nothing here iterates, so there
is no convergence tolerance to tune and no chance of a branch flip hiding inside
a solver. Toggle angles are closed form as well (§5.3, last paragraph).

This is referee-side code: it reports pass/fail, which a proposer may read but
never produces.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

#: Guard for "numerically exactly equal". Not a protocol threshold: Grashof
#: margins and path tolerances arrive from the task/configuration (§5.2).
_ZERO = 1e-12

BRANCHES = ("open", "crossed")
_TOGGLE = "toggle"

#: Largest joint step a continuous sweep may take between adjacent samples.
#: A solver guard, not a protocol threshold: the gate pins its value.
DEFAULT_MAX_JUMP_DEG = 90.0

#: Inversions that can traverse a full input rotation under a strict Grashof
#: inequality, with the shortest link in the required position (§5.2).
FULL_ROTATION_INPUT = frozenset({"crank-rocker", "double-crank"})


class MechanismError(ValueError):
    """A mechanism description that cannot be assembled."""


class UnreachableError(MechanismError):
    """The circle intersection is empty at this input angle (§5.3)."""


class BranchError(MechanismError):
    """The requested assembly branch has no pose at this input angle."""


def _require_positive(name: str, value: float) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise MechanismError(f"{name} must be a finite positive length in mm, got {value!r}")
    return float(value)


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to ``(-180, 180]``."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def angle_difference_deg(first: float, second: float) -> float:
    """Signed smallest rotation from `first` to `second`, in degrees."""
    return wrap_deg(second - first)


def branch_of_theta3(theta3_deg: float) -> str:
    """Assembly branch from the coupler direction: the sign of ``sin(theta3)``.

    This is the §5.4 sign test, expressed once so the closed form and the
    independent numerical solver cannot drift apart on it.
    """
    value = math.sin(math.radians(theta3_deg))
    if abs(value) <= 1e-12:
        return _TOGGLE
    return "open" if value > 0.0 else "crossed"


@dataclass(frozen=True, slots=True)
class GrashofResult:
    """Grashof verdict for a candidate link set (§5.2)."""

    is_grashof: bool
    inversion: str
    shortest_link: str
    longest_link: str
    slack_mm: float
    margin_mm: float
    reason: str

    @property
    def full_rotation_input(self) -> bool:
        return self.is_grashof and self.inversion in FULL_ROTATION_INPUT


@dataclass(frozen=True, slots=True)
class Pose:
    """One assembled configuration."""

    theta2_deg: float
    theta3_deg: float
    theta4_deg: float
    branch: str
    pivot_a_mm: tuple[float, float]
    pivot_b_mm: tuple[float, float]
    coupler_point_mm: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Trace:
    """The ordered coupler path traversed over an input interval."""

    input_range_deg: tuple[float, float]
    branch: str
    samples: int
    poses: tuple[Pose, ...]
    complete: bool
    missing_deg: tuple[float, float] | None = None
    #: True when the poses were selected by continuity instead of by label.
    continuity_tracked: bool = False

    @property
    def points_mm(self) -> tuple[tuple[float, float], ...]:
        return tuple(pose.coupler_point_mm for pose in self.poses)


@dataclass(frozen=True, slots=True)
class ToggleReport:
    """Input and output dead centres, classified separately (§5.3)."""

    input_dead_centre_deg: tuple[float, ...]
    output_dead_centre_deg: tuple[float, ...]
    input_locked: bool
    notes: tuple[str, ...]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class FourBar:
    """A revolute four-bar with a declared coupler point."""

    ground_mm: float
    input_mm: float
    coupler_mm: float
    output_mm: float
    coupler_point_mm: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("ground_mm", "input_mm", "coupler_mm", "output_mm"):
            _require_positive(name, getattr(self, name))
        point = self.coupler_point_mm
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise MechanismError(
                f"coupler_point_mm must be a finite (x, y) pair, got {point!r}")

    # ------------------------------------------------------------------ links
    @property
    def link_lengths_mm(self) -> dict[str, float]:
        return {
            "ground": self.ground_mm,
            "input": self.input_mm,
            "coupler": self.coupler_mm,
            "output": self.output_mm,
        }

    def record(self) -> dict[str, object]:
        return {
            "link_lengths_mm": self.link_lengths_mm,
            "coupler_point_mm": list(self.coupler_point_mm),
        }

    # ---------------------------------------------------------------- Grashof
    def check_grashof(self, *, margin_mm: float = 0.0) -> GrashofResult:
        """Strict Grashof test with an explicit margin (§5.2).

        ``margin_mm`` is the task's declared ``delta_g``; passing ``0.0`` asks
        for the strict inequality with no safety margin. Equality is never
        accepted as a robust full-rotation design.
        """
        if not math.isfinite(margin_mm) or margin_mm < 0:
            raise MechanismError(f"margin_mm must be finite and non-negative, got {margin_mm!r}")
        links = self.link_lengths_mm
        shortest = min(links, key=links.__getitem__)
        longest = max(links, key=links.__getitem__)
        remaining = [name for name in links if name not in {shortest, longest}]
        long_pair = links[shortest] + links[longest]
        other_pair = links[remaining[0]] + links[remaining[1]]
        slack = other_pair - long_pair

        if slack < -_ZERO:
            inversion, reason = "non-grashof", "non_grashof"
        elif abs(slack) <= _ZERO:
            inversion, reason = "change-point", "change_point"
        elif shortest == "ground":
            inversion, reason = "double-crank", "strict_grashof"
        elif shortest in {"input", "output"}:
            inversion, reason = "crank-rocker", "strict_grashof"
        else:
            inversion, reason = "double-rocker", "strict_grashof"

        is_grashof = slack > margin_mm
        if is_grashof is False and reason == "strict_grashof":
            reason = "grashof_margin_not_met"
        return GrashofResult(
            is_grashof=is_grashof,
            inversion=inversion,
            shortest_link=shortest,
            longest_link=longest,
            slack_mm=slack,
            margin_mm=margin_mm,
            reason=reason,
        )

    # ------------------------------------------------------- position analysis
    def poses(self, theta2_deg: float) -> tuple[Pose, ...]:
        """Every assembly at one input angle, without choosing a branch.

        The two signed-square-root roots share a ``sin(theta3)`` sign for some
        link sets, so the branch label does not always identify one of them;
        callers that know which *configuration* they expect (the cognate check
        does) need the roots themselves.
        """
        if not math.isfinite(theta2_deg):
            raise MechanismError(f"theta2_deg must be finite, got {theta2_deg!r}")
        a, b, c, d = self.input_mm, self.coupler_mm, self.output_mm, self.ground_mm
        theta2 = math.radians(theta2_deg)
        cos2, sin2 = math.cos(theta2), math.sin(theta2)
        # Loop closure eliminated for theta3, modulus squared of
        # b e^{i theta3} = d + c e^{i theta4} - a e^{i theta2}:
        #   (cos t2 - K1) cos t4 + sin(t2) sin t4 + (K2 cos t2 - K3) = 0
        k1, k2 = d / a, d / c
        k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
        coefficient_a = cos2 - k1
        coefficient_b = sin2
        coefficient_c = k2 * cos2 - k3
        return tuple(self._poses_from_theta4(
            coefficient_a, coefficient_b, coefficient_c, theta2_deg, theta2))

    def solve(self, theta2_deg: float, *, branch: str = "open") -> Pose:
        """Solve both moving pivots at one input angle."""
        if branch not in BRANCHES:
            raise MechanismError(f"branch must be one of {BRANCHES}, got {branch!r}")
        candidates = self.poses(theta2_deg)
        if not candidates:
            raise UnreachableError(
                f"no assembly at input {theta2_deg!r} deg: the output circle does not reach the "
                f"coupler circle")
        matched = [pose for pose in candidates if pose.branch == branch]
        if not matched:
            toggle = [pose for pose in candidates if pose.branch == _TOGGLE]
            if toggle:
                # At a dead centre the two branches coincide; the pose is the same.
                return toggle[0]
            raise BranchError(
                f"branch {branch!r} does not exist at input {theta2_deg!r} deg; "
                f"available: {sorted({pose.branch for pose in candidates})}")
        return matched[0]

    def pose_on(self, theta2_deg: float, branch: str) -> Pose:
        """The pose at `theta2_deg` on `branch`, or the only pose available.

        At a dead centre the two branches meet, so a label with no pose of its
        own is not an error: the single available pose is that pose.
        """
        if branch not in BRANCHES:
            raise MechanismError(f"branch must be one of {BRANCHES}, got {branch!r}")
        candidates = self.poses(theta2_deg)
        if not candidates:
            raise UnreachableError(
                f"no assembly at input {theta2_deg!r} deg: the output circle does not "
                f"reach the coupler circle")
        for pose in candidates:
            if pose.branch == branch:
                return pose
        return candidates[0]

    def pose_near(self, theta2_deg: float, reference: Pose) -> Pose:
        """The assembly at `theta2_deg` closest to `reference` in joint space.

        Selecting a root by branch label is not enough, because the two roots
        share a label for some link sets, so a label match can return the other
        assembly. What a sweep needs is continuity, and that is a property of
        the joints, not of the label.
        """
        candidates = self.poses(theta2_deg)
        if not candidates:
            raise UnreachableError(
                f"no assembly at input {theta2_deg!r} deg: the output circle does not "
                f"reach the coupler circle")
        return min(candidates, key=lambda pose: joint_gap_deg(reference, pose))

    def trace_continuous(
        self,
        input_range_deg: tuple[float, float],
        *,
        samples: int = 181,
        seed: Pose | None = None,
        max_jump_deg: float = DEFAULT_MAX_JUMP_DEG,
        strict: bool = True,
    ) -> Trace:
        """Trace one physical circuit by following the nearest root.

        A sweep assembled by label can hop to the other assembly where the two
        roots share a label, and a real circuit break can look like a small
        label change. This walks the roots instead: each sample takes the
        assembly closest to the previous one, and a step larger than
        `max_jump_deg` stops the trace rather than being accepted as the next
        sample. `seed` fixes the circuit; without one the first root at the low
        end is used.
        """
        if samples < 2:
            raise MechanismError(f"samples must be at least 2, got {samples!r}")
        low, high = input_range_deg
        if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
            raise MechanismError(
                f"input_range_deg must be finite and increasing, got {input_range_deg!r}")
        if not (math.isfinite(max_jump_deg) and max_jump_deg > 0.0):
            raise MechanismError(
                f"max_jump_deg must be finite and positive, got {max_jump_deg!r}")
        poses: list[Pose] = []
        previous = seed
        for index in range(samples):
            theta2_deg = low + (high - low) * index / (samples - 1)
            try:
                if previous is None:
                    candidates = self.poses(theta2_deg)
                    if not candidates:
                        raise UnreachableError(
                            f"no assembly at input {theta2_deg!r} deg")
                    pose = min(candidates, key=lambda item: abs(item.theta3_deg))
                else:
                    pose = self.pose_near(theta2_deg, previous)
            except UnreachableError:
                if strict:
                    raise
                return Trace(
                    input_range_deg=(low, high),
                    branch=poses[0].branch if poses else "open",
                    samples=samples,
                    poses=tuple(poses),
                    complete=False,
                    missing_deg=(theta2_deg, high),
                    continuity_tracked=True,
                )
            if previous is not None:
                gap = joint_gap_deg(previous, pose)
                if gap > max_jump_deg:
                    if strict:
                        raise BranchError(
                            f"circuit break at input {theta2_deg!r} deg: the nearest "
                            f"assembly is {gap:.4g} deg from the previous pose")
                    return Trace(
                        input_range_deg=(low, high),
                        branch=poses[0].branch,
                        samples=samples,
                        poses=tuple(poses),
                        complete=False,
                        missing_deg=(theta2_deg, high),
                        continuity_tracked=True,
                    )
            poses.append(pose)
            previous = pose
        return Trace(
            input_range_deg=(low, high),
            branch=poses[0].branch,
            samples=samples,
            poses=tuple(poses),
            complete=True,
            continuity_tracked=True,
        )

    def _poses_from_theta4(
        self,
        coefficient_a: float,
        coefficient_b: float,
        coefficient_c: float,
        theta2_deg: float,
        theta2: float,
    ) -> list[Pose]:
        """Both signed-square-root branches of the output-angle equation."""
        roots = _solve_cosine_sine(
            coefficient_a, coefficient_b, coefficient_c,
            f"input {theta2_deg!r} deg")
        a, c, d = self.input_mm, self.output_mm, self.ground_mm
        pivot_a = (a * math.cos(theta2), a * math.sin(theta2))
        poses: list[Pose] = []
        for theta4 in roots:
            pivot_b = (d + c * math.cos(theta4), c * math.sin(theta4))
            theta3 = math.atan2(pivot_b[1] - pivot_a[1], pivot_b[0] - pivot_a[0])
            poses.append(self._pose(theta2_deg, theta2, theta3, theta4, pivot_a, pivot_b))
        return poses

    def _pose(
        self,
        theta2_deg: float,
        theta2: float,
        theta3: float,
        theta4: float,
        pivot_a: tuple[float, float],
        pivot_b: tuple[float, float],
    ) -> Pose:
        px, py = self.coupler_point_mm
        cos3, sin3 = math.cos(theta3), math.sin(theta3)
        coupler_point = (
            pivot_a[0] + px * cos3 - py * sin3,
            pivot_a[1] + px * sin3 + py * cos3,
        )
        branch = branch_of_theta3(math.degrees(theta3))
        return Pose(
            theta2_deg=wrap_deg(theta2_deg),
            theta3_deg=wrap_deg(math.degrees(theta3)),
            theta4_deg=wrap_deg(math.degrees(theta4)),
            branch=branch,
            pivot_a_mm=pivot_a,
            pivot_b_mm=pivot_b,
            coupler_point_mm=coupler_point,
        )

    # ---------------------------------------------------------------- tracing
    def trace(
        self,
        input_range_deg: tuple[float, float],
        *,
        samples: int = 181,
        branch: str = "open",
        strict: bool = True,
        circuit: bool = True,
    ) -> Trace:
        """Trace the coupler point over an input interval on one assembly.

        ``circuit=True`` (the default) follows the assembly nearest the previous
        pose, seeded on `branch` at the low end. That is what a physical sweep
        does, and it is the only selection rule that cannot hop to the other
        assembly where the two roots share a branch label.

        ``circuit=False`` reproduces the old label-per-sample sweep and is kept
        only for forensic comparison: on some link sets it returns a path a real
        mechanism cannot follow, which is precisely the defect that made a
        cognate check disagree with itself.
        """
        if branch not in BRANCHES:
            raise MechanismError(f"branch must be one of {BRANCHES}, got {branch!r}")
        if samples < 2:
            raise MechanismError(f"samples must be at least 2, got {samples!r}")
        low, high = input_range_deg
        if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
            raise MechanismError(
                f"input_range_deg must be finite and increasing, got {input_range_deg!r}")
        if circuit:
            try:
                seed = self.pose_on(low, branch)
            except UnreachableError:
                if strict:
                    raise
                return Trace(
                    input_range_deg=(low, high),
                    branch=branch,
                    samples=samples,
                    poses=(),
                    complete=False,
                    missing_deg=(low, high),
                    continuity_tracked=True,
                )
            return self.trace_continuous(
                (low, high), samples=samples, seed=seed, strict=strict)

        poses: list[Pose] = []
        for index in range(samples):
            theta2_deg = low + (high - low) * index / (samples - 1)
            try:
                poses.append(self.solve(theta2_deg, branch=branch))
            except (UnreachableError, BranchError):
                if strict:
                    raise
                return Trace(
                    input_range_deg=(low, high),
                    branch=branch,
                    samples=samples,
                    poses=tuple(poses),
                    complete=False,
                    missing_deg=(theta2_deg, high),
                    continuity_tracked=False,
                )
        return Trace(
            input_range_deg=(low, high),
            branch=branch,
            samples=samples,
            poses=tuple(poses),
            complete=True,
            continuity_tracked=False,
        )

    # ---------------------------------------------------------------- toggles
    def toggles(self, *, margin_mm: float = 0.0) -> ToggleReport:
        """Closed-form input and output dead centres (§5.3)."""
        grashof = self.check_grashof(margin_mm=margin_mm)
        a, b, c, d = self.input_mm, self.coupler_mm, self.output_mm, self.ground_mm
        input_angles = _toggle_angles(
            adjacent_mm=d, opposite_mm=c, swept_mm=a, slider_mm=b)
        output_angles = _toggle_angles(
            adjacent_mm=d, opposite_mm=a, swept_mm=b, slider_mm=c)
        notes: list[str] = []
        input_locked = not grashof.full_rotation_input
        if input_angles and input_locked:
            notes.append(
                f"input drive locks at {[round(value, 6) for value in input_angles]} deg: "
                f"the {grashof.inversion} inversion cannot drive through a toggle")
        elif input_angles:
            notes.append(
                "input dead centres are pass-through positions for a full-rotation input")
        if output_angles:
            notes.append(
                "output dead centres are limit positions; normal for a crank-rocker and not "
                "by itself a rejection")
        return ToggleReport(
            input_dead_centre_deg=input_angles,
            output_dead_centre_deg=output_angles,
            input_locked=input_locked,
            notes=tuple(notes),
        )


def _toggle_angles(
    *, adjacent_mm: float, opposite_mm: float, swept_mm: float, slider_mm: float
) -> tuple[float, ...]:
    """Input angles where two links become collinear, from the cosine rule."""
    angles: list[float] = []
    for reach in (swept_mm + slider_mm, abs(swept_mm - slider_mm)):
        if reach <= _ZERO:
            continue
        cosine = (
            adjacent_mm * adjacent_mm + reach * reach - opposite_mm * opposite_mm
        ) / (2.0 * adjacent_mm * reach)
        if abs(cosine) > 1.0:
            continue
        value = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        for angle in (value, -value):
            if all(abs(angle_difference_deg(angle, other)) > 1e-9 for other in angles):
                angles.append(wrap_deg(angle))
    return tuple(sorted(angles))


def _solve_cosine_sine(
    coefficient_a: float, coefficient_b: float, coefficient_c: float, where: str
) -> list[float]:
    """Roots of ``A cos(t) + B sin(t) + C = 0`` via the half-angle tangent.

    Returns both signed-square-root roots, which are the two assembly branches.
    """
    discriminant = (
        coefficient_a * coefficient_a + coefficient_b * coefficient_b
        - coefficient_c * coefficient_c
    )
    if discriminant < -_ZERO:
        return []
    root = math.sqrt(max(discriminant, 0.0))
    denominator = coefficient_c - coefficient_a
    tangents: list[float] = []
    if abs(denominator) > _ZERO:
        for sign in (1.0, -1.0):
            tangents.append((-coefficient_b + sign * root) / denominator)
    else:
        # A == C: the quadratic degenerates to 2Bt + 2A = 0 plus the t -> inf root.
        if abs(coefficient_b) <= _ZERO:
            raise MechanismError(
                f"degenerate output-angle equation at {where}: it vanishes identically")
        tangents.append(-coefficient_a / coefficient_b)
        tangents.append(math.inf)
    roots: list[float] = []
    for tangent in tangents:
        theta = math.pi if math.isinf(tangent) else 2.0 * math.atan(tangent)
        wrapped = wrap_deg(math.degrees(theta))
        if all(abs(angle_difference_deg(wrapped, other)) > 1e-9 for other in roots):
            roots.append(wrapped)
    return [math.radians(value) for value in roots]


def poses_share_circuit(poses: Sequence[Pose]) -> bool:
    """True when every pose sits on one continuous assembly circuit (§5.4).

    The branch label *is* the sign of the square-root term, so a changed label
    between adjacent poses is a branch swap. Toggle poses are ignored: at a dead
    centre the sign vanishes because the two branches meet, which is not a
    disassembly.
    """
    labels = [pose.branch for pose in poses if pose.branch != _TOGGLE]
    if not labels:
        raise MechanismError("cannot establish a circuit from toggle poses only")
    return len(set(labels)) == 1


def joint_gap_deg(first: Pose, second: Pose) -> float:
    """Largest wrapped difference of the two moving joint angles, in degrees."""
    return max(
        abs(angle_difference_deg(first.theta3_deg, second.theta3_deg)),
        abs(angle_difference_deg(first.theta4_deg, second.theta4_deg)),
    )


def circuit_breaks(poses: Sequence[Pose], *, max_jump_deg: float = 90.0) -> int:
    """Adjacent poses whose joints jump further than a continuous sweep allows.

    Unlike :func:`circuit_switches` this reads the geometry, not the label: a
    real circuit break shows up as a large joint jump even when the label
    happens to be unchanged.
    """
    if not (math.isfinite(max_jump_deg) and max_jump_deg > 0.0):
        raise MechanismError(
            f"max_jump_deg must be finite and positive, got {max_jump_deg!r}")
    return sum(
        1
        for first, second in pairwise(poses)
        if joint_gap_deg(first, second) > max_jump_deg
    )


def circuit_switches(poses: Sequence[Pose]) -> int:
    """How many times the branch label changes between adjacent poses."""
    switches = 0
    previous: str | None = None
    for pose in poses:
        if pose.branch == _TOGGLE:
            continue
        if previous is not None and pose.branch != previous:
            switches += 1
        previous = pose.branch
    return switches


def mean_toggle_spacing_deg(poses: Sequence[Pose]) -> float:
    """Mean angular spacing of a pose sequence; provenance for a reported error."""
    if len(poses) < 2:
        return 0.0
    steps = [
        abs(angle_difference_deg(poses[index].theta2_deg, poses[index + 1].theta2_deg))
        for index in range(len(poses) - 1)
    ]
    return _mean(steps)
