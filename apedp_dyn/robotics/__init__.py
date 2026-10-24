r"""Robot Trajectory Transport Modules.

Status: experimental
"""
from __future__ import annotations

from apedp_dyn.robotics.multi_agent_transport import (
    AgentRole,
    MultiAgentCoordinationResult,
    MultiAgentState,
    compute_cbf_repulsive_force,
    multi_agent_step,
    pairwise_repulsive_forces,
    simulate_multi_agent_coordination,
)
from apedp_dyn.robotics.trajectory_transport import (
    NavigationResult,
    exact_unicycle_step,
    goal_feedback_velocity_fn,
    navigate_to_goal,
    propagate_with_stepper,
    simulate_unicycle_exact,
    unicycle_velocity_fn,
    wrap_angle,
)

__all__ = [
    "AgentRole",
    "MultiAgentCoordinationResult",
    "MultiAgentState",
    "NavigationResult",
    "compute_cbf_repulsive_force",
    "exact_unicycle_step",
    "goal_feedback_velocity_fn",
    "multi_agent_step",
    "navigate_to_goal",
    "pairwise_repulsive_forces",
    "propagate_with_stepper",
    "simulate_multi_agent_coordination",
    "simulate_unicycle_exact",
    "unicycle_velocity_fn",
    "wrap_angle",
]
