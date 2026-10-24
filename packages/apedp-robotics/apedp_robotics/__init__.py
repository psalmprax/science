"""APEDP-Robotics: Autonomous Robotics Trajectory Transport & Control."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package can be imported if running within monorepo
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from apedp_dyn.robotics.multi_agent_transport import (
    AgentRole,
    MultiAgentCoordinationResult,
    MultiAgentState,
    compute_cbf_repulsive_force,
    multi_agent_step,
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

__version__ = "0.1.0"

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
    "propagate_with_stepper",
    "simulate_multi_agent_coordination",
    "simulate_unicycle_exact",
    "unicycle_velocity_fn",
    "wrap_angle",
]
