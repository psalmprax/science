r"""Multi-Agent Collaborative Robotics & Collision-Avoidance Transport.

Status: experimental

Scientific Scope & Algorithmic Foundation:
- Models cooperative multi-agent non-holonomic kinematic systems (inspired by RoboParty
  RP1 and PingpongDuo dual-robot collaborative coordination).
- Extends single-unicycle closed-form arc integration to N-agent systems with:
    1. Repulsive Artificial Potential Fields:
       Continuous Khatib potential gradient pushing agents apart when inter-agent distance
       falls below warning radius $r_{warning}$. Note: This is an artificial repulsive potential
       heuristic, not a certified quadratic-programming Control Barrier Function (CBF-QP)
       with formal forward-invariance proofs. A tangential breakaway term resolves the
       symmetric head-on deadlock a purely radial field cannot.
    2. Dynamic Role Allocation & Turn-Taking:
       In cooperative double-agent tasks (e.g. ping-pong doubles or payload handoff),
       agents dynamically switch between "Active Striker" and "Support/Recovery" modes
       to prevent deadlock and territorial contention. Support agents receive
       formation-indexed offsets (none share a waypoint), so a fleet cannot collapse
       into a single point.
    3. Closed-loop Lyapunov distance convergence toward dynamic cooperative waypoints.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from apedp_dyn.robotics.trajectory_transport import exact_unicycle_step, wrap_angle

_TWO_PI = 2.0 * math.pi


class AgentRole(str, Enum):
    """Dynamic role assignment for multi-agent coordination."""
    ACTIVE_STRIKER = "active_striker"
    SUPPORT_RECOVERY = "support_recovery"


@dataclass
class MultiAgentState:
    """State of an N-agent robotic system."""
    poses: np.ndarray  # Shape: (N, 3), [x, y, theta]
    velocities: np.ndarray  # Shape: (N,), linear velocities
    angular_velocities: np.ndarray  # Shape: (N,), angular velocities
    roles: list[AgentRole]  # Length N


@dataclass
class MultiAgentCoordinationResult:
    """Trajectory and safety log for multi-agent navigation."""
    trajectories: np.ndarray  # Shape: (T, N, 3)
    minimum_inter_agent_distances: np.ndarray  # Shape: (T,)
    safety_violation: bool
    target_reached: bool
    final_distances_to_target: np.ndarray  # Shape: (N,)


def compute_cbf_repulsive_force(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    r_safe: float = 0.5,
    r_warning: float = 1.0,
    k_rep: float = 2.0,
    heading_i: float | None = None,
) -> np.ndarray:
    """Compute repulsive safety gradient between agent i and agent j.

    When a heading is supplied and the agents are oriented head-on (their
    relative displacement is anti-parallel to their heading), a purely radial
    repulsion has no lateral component and the smooth control law
    ``v ~ max(0, cos(heading_error))`` drives *both* agents to speed 0: a
    symmetric, permanent deadlock. A fixed-sign tangential component breaks the
    symmetry so each agent slides past the other.
    """
    diff = pos_i - pos_j
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        # Avoid division by zero on exact collision
        return np.array([k_rep, 0.0])
    if dist >= r_warning:
        return np.zeros(2)

    # Repulsive vector pushing pos_i away from pos_j
    direction = diff / dist
    magnitude = k_rep * (1.0 / max(r_safe, dist) - 1.0 / r_warning) * (1.0 / (dist ** 2.0))
    force = magnitude * direction

    if heading_i is not None:
        heading_vec = np.array([np.cos(heading_i), np.sin(heading_i)])
        # Anti-parallel test: radical repulsion opposes the agent's motion.
        anti_par = float(-direction @ heading_vec)  # > 0 when diff opposes heading
        if anti_par > 0.7:
            perp = np.array([-direction[1], direction[0]])
            # Fixed (rightward) sign breaks the head-on symmetry for THIS agent;
            # the other agent's own heading resolves to the opposite tangential
            # side, so the pair separates rather than both freezing.
            force = force + anti_par * magnitude * perp
    return force


def pairwise_repulsive_forces(
    positions: np.ndarray,
    headings: np.ndarray | None,
    r_safe: float = 0.5,
    r_warning: float = 1.0,
    k_rep: float = 2.0,
) -> np.ndarray:
    """Vectorized all-pairs repulsive-force sweep, EXACTLY mirroring
    :func:`compute_cbf_repulsive_force` per pair (same elementwise IEEE ops,
    same branch masks), summed over j ≠ i.

    Per-pair semantics replicated exactly:
      * dist < 1e-9 (degenerate collision): force = [k_rep, 0.0]
      * dist >= r_warning: zeros (the distance filter a proximity structure
        would replicate more expensively — the mask IS the filter)
      * else magnitude = k_rep * (1/max(r_safe, dist) - 1/r_warning) *
        (1/dist²), direction = (p_i - p_j)/dist, with the head-on
        tangential term added when -direction·heading > 0.7.

    Returns [N, 2] summed force per agent. Per-pair equality with the
    scalar function is pinned (branch structure AND values); the per-agent
    SUM differs only at float summation-order ulp level (numpy pairwise
    summation vs sequential +=), pinned at <=1e-12 relative.
    """
    p = np.asarray(positions, dtype=float)
    n = p.shape[0]
    diff = p[:, None, :] - p[None, :, :]                     # [N, N, 2]
    dist = np.sqrt((diff * diff).sum(-1))                    # [N, N] (raw)
    offdiag = ~np.eye(n, dtype=bool)
    degen = (dist < 1e-9) & offdiag
    near = (dist < r_warning) & offdiag & ~degen

    # Branch 1: degenerate collision -> the FIXED [k_rep, 0.0] vector.
    f = np.zeros((n, n, 2), dtype=float)                     # [N, N, 2]
    f[degen] = np.array([k_rep, 0.0])

    # Branch 2: active repulsion, elementwise ops in the scalar order.
    d = dist[near]                                           # [P,]
    dirv = diff[near] / d[:, None]
    mag = k_rep * (1.0 / np.maximum(r_safe, d) - 1.0 / r_warning) * (1.0 / (d ** 2.0))
    force = mag[:, None] * dirv                              # [P, 2]
    if headings is not None:
        th = np.asarray(headings, dtype=float)
        hv = np.stack([np.cos(th), np.sin(th)], axis=1)      # [N, 2]
        rows_i, _rows_j = np.nonzero(near)
        anti = -(dirv * hv[rows_i]).sum(axis=1)              # -direction·heading
        hit = anti > 0.7
        perp = np.stack([-dirv[:, 1], dirv[:, 0]], axis=1)
        force[hit] += (anti[hit] * mag[hit])[:, None] * perp[hit]
    f[near] = force

    # Sum over j != i (masked). Diagonal contributions are always zero:
    # degen/near are both off-diagonal, and far pairs stay zeros.
    return f.sum(axis=1)


def multi_agent_step(
    poses: np.ndarray,
    targets: np.ndarray,
    roles: Sequence[AgentRole],
    dt: float = 0.05,
    r_safe: float = 0.5,
    r_warning: float = 1.0,
    v_max: float = 1.0,
    omega_max: float = 2.0,
    k_att: float = 1.5,
    k_rep: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute a single collision-avoiding kinematic step across N agents.

    Returns:
        next_poses: Shape (N, 3)
        linear_velocities: Shape (N,)
        angular_velocities: Shape (N,)
    """
    n_agents = poses.shape[0]
    next_poses = np.zeros_like(poses)
    lin_vels = np.zeros(n_agents)
    ang_vels = np.zeros(n_agents)

    positions = poses[:, :2]
    # ONE vectorized all-pairs sweep up front (exact per-pair mirror of
    # compute_cbf_repulsive_force; the per-pair Python loop was the
    # measured bottleneck: ~6 µs/invocation, 5 ms budget crossing at N=32).
    rep2d = pairwise_repulsive_forces(
        positions, poses[:, 2], r_safe=r_safe, r_warning=r_warning, k_rep=k_rep,
    )

    for i in range(n_agents):
        pose_i = poses[i]
        p_i = pose_i[:2]
        theta_i = pose_i[2]
        target_i = targets[i]
        role_i = roles[i]

        # 1. Attractive force toward assigned target / waypoint
        d_target = target_i - p_i
        dist_target = float(np.linalg.norm(d_target))
        if dist_target > 1e-6:
            f_att = k_att * (d_target / dist_target) * min(dist_target, 1.0)
        else:
            f_att = np.zeros(2)

        # 2. Artificial-potential repulsive force from other agents —
        #    ONE vectorized all-pairs sweep (exact mirror of the per-pair
        #    scalar rule, see pairwise_repulsive_forces); the per-pair
        #    Python loop it replaces was the measured bottleneck (~6 µs
        #    per invocation, 5 ms budget crossing at N=32).
        f_rep = rep2d[i]

        # Net nominal steering vector
        # Support recovery agent gives right of way by scaling down aggressive forward drive
        role_multiplier = 1.0 if role_i == AgentRole.ACTIVE_STRIKER else 0.5
        f_total = f_att + f_rep

        total_norm = float(np.linalg.norm(f_total))
        if total_norm > 1e-6:
            desired_heading = np.arctan2(f_total[1], f_total[0])
            heading_error = wrap_angle(desired_heading - theta_i)

            # Smooth unicycle control law
            v_cmd = role_multiplier * v_max * max(0.0, np.cos(heading_error)) * min(1.0, total_norm)
            omega_cmd = np.clip(2.0 * heading_error, -omega_max, omega_max)
        else:
            v_cmd = 0.0
            omega_cmd = 0.0

        next_poses[i] = exact_unicycle_step(pose_i, v_cmd, omega_cmd, dt)
        lin_vels[i] = v_cmd
        ang_vels[i] = omega_cmd

    return next_poses, lin_vels, ang_vels


def simulate_multi_agent_coordination(
    initial_poses: np.ndarray,
    target: np.ndarray,
    support_offset: np.ndarray | None = None,
    max_steps: int = 200,
    dt: float = 0.05,
    r_safe: float = 0.4,
    r_warning: float = 0.8,
) -> MultiAgentCoordinationResult:
    """Simulate cooperative multi-agent coordination with dynamic role assignment.

    In ping-pong doubles / cooperative interception:
      - The agent closest to the target becomes the ACTIVE_STRIKER.
      - Every other agent becomes SUPPORT_RECOVERY, each targeting a UNIQUE
        formation waypoint around the target.
    """
    if support_offset is None:
        support_offset = np.array([-1.0, -1.0])

    n_agents = initial_poses.shape[0]
    traj = np.zeros((max_steps, n_agents, 3))
    min_distances = np.zeros(max_steps)

    current_poses = initial_poses.copy()
    safety_violated = False

    for step in range(max_steps):
        traj[step] = current_poses

        # Compute all pairwise inter-agent distances
        dists = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                d = float(np.linalg.norm(current_poses[i, :2] - current_poses[j, :2]))
                dists.append(d)
                if d < r_safe:
                    safety_violated = True
        min_distances[step] = min(dists) if dists else float("inf")

        # Dynamic Role Allocation based on proximity to target
        dist_to_target = np.array([float(np.linalg.norm(current_poses[i, :2] - target)) for i in range(n_agents)])
        striker_idx = int(np.argmin(dist_to_target))

        # Formation-indexed support waypoints: giving every non-striker the SAME
        # target + support_offset made the whole support fleet converge on one
        # point, so N - 1 robots fought for an identical slot. Each support
        # agent holds slot k of a small ring around the target, oriented by the
        # caller's support_offset direction (falls back to -x). Single-support
        # (N = 2) keeps exactly the historical offset.
        support_ids = [i for i in range(n_agents) if i != striker_idx]
        base_dir = support_offset / (float(np.linalg.norm(support_offset)) or 1.0)
        support_radial = float(np.linalg.norm(support_offset))
        roles: list[AgentRole] = []
        targets = np.zeros((n_agents, 2))
        base_angle = float(np.arctan2(base_dir[1], base_dir[0]))
        for i in range(n_agents):
            if i == striker_idx:
                roles.append(AgentRole.ACTIVE_STRIKER)
                targets[i] = target
            else:
                roles.append(AgentRole.SUPPORT_RECOVERY)
                slot = support_ids.index(i)
                if len(support_ids) == 1:
                    targets[i] = target + support_offset
                else:
                    angle = base_angle + _TWO_PI * slot / len(support_ids)
                    targets[i] = target + support_radial * np.array([np.cos(angle), np.sin(angle)])

        # Check termination condition
        if dist_to_target[striker_idx] < 0.15:
            # Active striker reached target
            traj = traj[: step + 1]
            min_distances = min_distances[: step + 1]
            return MultiAgentCoordinationResult(
                trajectories=traj,
                minimum_inter_agent_distances=min_distances,
                safety_violation=safety_violated,
                target_reached=True,
                final_distances_to_target=dist_to_target,
            )

        current_poses, _, _ = multi_agent_step(
            poses=current_poses,
            targets=targets,
            roles=roles,
            dt=dt,
            r_safe=r_safe,
            r_warning=r_warning,
        )

    final_dists = np.array([float(np.linalg.norm(current_poses[i, :2] - target)) for i in range(n_agents)])
    return MultiAgentCoordinationResult(
        trajectories=traj,
        minimum_inter_agent_distances=min_distances,
        safety_violation=safety_violated,
        target_reached=bool(np.min(final_dists) < 0.2),
        final_distances_to_target=final_dists,
    )
