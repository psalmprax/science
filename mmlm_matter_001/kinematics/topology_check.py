"""Topology validity for a T3 candidate (blueprint §2.2, §2.4, §2.5).

A T3 answer is a topology, an inversion and dimensions. The path metric says
whether the dimensions trace the target; it says nothing about whether the
topology the candidate claims is the one it implemented. This checker is the other
half: a planar revolute chain, one degree of freedom, three distinct link roles,
and the inversion the entry declares must be the inversion the dimensions actually
produce. A crank-rocker link set cannot honestly claim to implement a
double-rocker entry, however well it traces the target.

The universe the blueprint evaluates is 6-, 8- and 10-link chains (§2.5). This
repository implements four-bar kinematics and no general chain solver, so a larger
entry is refused as unverifiable rather than passed on the strength of its
metadata: an unverifiable topology is not a verified one, and grading it as a pass
would be the exact claim the blueprint forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

from mmlm_matter_001.kinematics.boundary import assert_referee_caller
from mmlm_matter_001.kinematics.fourbar import FourBar, MechanismError
from mmlm_matter_001.schemas import TopologyEntry

#: Machine-readable reasons a topology check can fail.
TOPOLOGY_REASONS = frozenset({
    "topology_unsupported_structure",
    "topology_dof_not_one",
    "topology_link_roles_invalid",
    "topology_inversion_mismatch",
})

#: The only structure this checkout can verify: a four-bar, revolute only.
SUPPORTED_LINKS = 4
SUPPORTED_JOINTS = 4
ONE_DOF = 1


def planar_revolute_dof(*, link_count: int, joint_count: int) -> int:
    """Degrees of freedom of a planar chain of revolute lower pairs.

    Grübler's criterion: ``3 (n - 1) - 2 j`` for a single-loop planar chain with
    only one-degree-of-freedom joints.
    """
    if link_count < 1 or joint_count < 0:
        raise MechanismError(
            f"a chain needs at least one link and no negative joints, got "
            f"{link_count!r} links and {joint_count!r} joints")
    return 3 * (link_count - 1) - 2 * joint_count


@dataclass(frozen=True, slots=True)
class TopologyCheck:
    """The referee's verdict on the topology half of a T3 answer."""

    passed: bool
    topology_id: str
    link_count: int
    joint_count: int
    dof: int
    declared_inversion: str
    achieved_inversion: str
    reasons: tuple[str, ...]
    details: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "topology_id": self.topology_id,
            "link_count": self.link_count,
            "joint_count": self.joint_count,
            "dof": self.dof,
            "declared_inversion": self.declared_inversion,
            "achieved_inversion": self.achieved_inversion,
            "reasons": list(self.reasons),
            "details": list(self.details),
        }


def check_topology(
    topology: TopologyEntry,
    mechanism: FourBar,
    *,
    margin_mm: float = 0.0,
) -> TopologyCheck:
    """Check that `mechanism` implements `topology` as declared (§2.2, §2.4).

    ``margin_mm`` is passed to the Grashof test, so the inversion is judged under
    the same margin the gate preregistered rather than at equality.
    """
    assert_referee_caller("check_topology")
    if not isinstance(topology, TopologyEntry):
        raise MechanismError(
            f"check_topology needs a TopologyEntry, got {type(topology).__name__}")
    if not isinstance(mechanism, FourBar):
        raise MechanismError(
            f"check_topology needs a FourBar, got {type(mechanism).__name__}")

    reasons: list[str] = []
    details: list[str] = []
    dof = planar_revolute_dof(
        link_count=topology.link_count, joint_count=topology.joint_count)

    if (topology.link_count, topology.joint_count) != (SUPPORTED_LINKS, SUPPORTED_JOINTS):
        reasons.append("topology_unsupported_structure")
        details.append(
            f"{topology.topology_id} declares {topology.link_count} links and "
            f"{topology.joint_count} joints; this checkout verifies only four-bars "
            f"({SUPPORTED_LINKS} links, {SUPPORTED_JOINTS} joints)")
    if dof != ONE_DOF:
        reasons.append("topology_dof_not_one")
        details.append(
            f"{topology.topology_id} has {dof} degrees of freedom; the mechanism "
            f"universe is one-degree-of-freedom chains (§2.2)")

    roles = (topology.ground_link, topology.input_link, topology.output_link)
    if len(set(roles)) != 3 or any(not role for role in roles):
        reasons.append("topology_link_roles_invalid")
        details.append(
            f"{topology.topology_id} declares ground/input/output as {roles}; three "
            "distinct links are required")

    grashof = mechanism.check_grashof(margin_mm=margin_mm)
    if topology.link_count == SUPPORTED_LINKS and topology.inversion != grashof.inversion:
        reasons.append("topology_inversion_mismatch")
        details.append(
            f"{topology.topology_id} declares {topology.inversion!r} but the "
            f"dimensions produce {grashof.inversion!r} (slack {grashof.slack_mm:.6g} mm, "
            f"margin {margin_mm} mm)")

    return TopologyCheck(
        passed=not reasons,
        topology_id=topology.topology_id,
        link_count=topology.link_count,
        joint_count=topology.joint_count,
        dof=dof,
        declared_inversion=topology.inversion,
        achieved_inversion=grashof.inversion,
        reasons=tuple(reasons),
        details=tuple(details),
    )


__all__ = [
    "ONE_DOF",
    "SUPPORTED_JOINTS",
    "SUPPORTED_LINKS",
    "TOPOLOGY_REASONS",
    "TopologyCheck",
    "check_topology",
    "planar_revolute_dof",
]
