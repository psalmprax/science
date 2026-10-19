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
       with formal forward-invariance proofs.
    2. Dynamic Role Allocation & Turn-Taking:
       In cooperative double-agent tasks (e.g. ping-pong doubles or payload handoff),
       agents dynamically switch between "Active Striker" and "Support/Recovery" modes
       to prevent deadlock and territorial contention.
    3. Closed-loop Lyapunov distance convergence toward dynamic cooperative waypoints.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from apedp_dyn.robotics.trajectory_transport import exact_unicycle_step, wrap_angle


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
) -> np.ndarray:
    """Compute repulsive safety gradient between agent i and agent j."""
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
    return magnitude * direction


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

        # 2. Artificial-potential repulsive force from other agents
        f_rep = np.zeros(2)
        for j in range(n_agents):
            if i != j:
                f_rep += compute_cbf_repulsive_force(
                    p_i, positions[j], r_safe=r_safe, r_warning=r_warning, k_rep=k_rep
                )

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
    support_offset: np.ndarray = np.array([-1.0, -1.0]),
    max_steps: int = 200,
    dt: float = 0.05,
    r_safe: float = 0.4,
    r_warning: float = 0.8,
) -> MultiAgentCoordinationResult:
    """Simulate cooperative multi-agent coordination with dynamic role assignment.

    In ping-pong doubles / cooperative interception:
      - The agent closest to the target becomes the ACTIVE_STRIKER.
      - The other agent becomes SUPPORT_RECOVERY targeting target + support_offset.
    """
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

        roles: list[AgentRole] = []
        targets = np.zeros((n_agents, 2))
        for i in range(n_agents):
            if i == striker_idx:
                roles.append(AgentRole.ACTIVE_STRIKER)
                targets[i] = target
            else:
                roles.append(AgentRole.SUPPORT_RECOVERY)
                targets[i] = target + support_offset

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
