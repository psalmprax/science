r"""Robot Trajectory Transport: unicycle kinematics with Jet2 steering integration.

Status: experimental

Scope & honesty ceiling:
- Implements the kinematic (first-order, nonholonomic) differential-drive robot
  model ``[x, y, theta]`` with exact closed-form arc stepping and drives it with
  the repository's continuous transport stepper
  (``apedp_dyn.video.jet_flow_stepper``) under both open-loop commands and a
  smooth feedback law.
- Purely kinematic: no wheel dynamics, no friction/slip, no obstacles, no
  sensors. Tests verify arc analytics, step-size convergence against the exact
  solution, global goal convergence from random starts, and Lyapunov descent of
  the goal distance. Manipulator arms and dynamic (torque-level) control are out
  of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

_TWO_PI = 2.0 * np.pi


def wrap_angle(angle: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    return (angle + np.pi) % _TWO_PI - np.pi


@dataclass
class NavigationResult:
    """Closed-loop navigation trace ending at the goal."""

    trajectory: np.ndarray
    distances: np.ndarray
    success: bool
    steps: int
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))


def exact_unicycle_step(pose: np.ndarray, linear_velocity: float, angular_velocity: float, dt: float) -> np.ndarray:
    """Exact integration of constant ``(v, omega)`` over ``dt`` (arc or straight)."""
    x, y, theta = pose
    if abs(angular_velocity) < 1e-12:
        return np.array([x + dt * linear_velocity * np.cos(theta), y + dt * linear_velocity * np.sin(theta), theta])
    theta_next = theta + angular_velocity * dt
    radius = linear_velocity / angular_velocity
    return np.array(
        [
            x + radius * (np.sin(theta_next) - np.sin(theta)),
            y - radius * (np.cos(theta_next) - np.cos(theta)),
            theta_next,
        ]
    )


def simulate_unicycle_exact(
    pose0: np.ndarray,
    controls: Sequence[tuple[float, float]],
    dt: float,
) -> np.ndarray:
    """Exact pose trajectory under a control sequence ``[(v, omega), ...]``."""
    trajectory = [np.asarray(pose0, dtype=np.float64).copy()]
    for linear_velocity, angular_velocity in controls:
        trajectory.append(exact_unicycle_step(trajectory[-1], linear_velocity, angular_velocity, dt))
    return np.stack(trajectory)


def unicycle_velocity_fn(linear_velocity: float, angular_velocity: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """Constant-command unicycle RHS ``dz/dt`` for the transport stepper."""

    def velocity_fn(z: np.ndarray, _t: float) -> np.ndarray:
        x_component = linear_velocity * np.cos(z[2])
        y_component = linear_velocity * np.sin(z[2])
        return np.array([x_component, y_component, angular_velocity])

    return velocity_fn


def propagate_with_stepper(
    pose0: np.ndarray,
    controls: Sequence[tuple[float, float]],
    dt: float,
    order: int = 2,
) -> np.ndarray:
    """Pose trajectory from the JetFlowStepper under piecewise-constant commands."""
    from apedp_dyn.video.jet_flow_stepper import JetFlowConfig, JetFlowStepper

    stepper = JetFlowStepper(JetFlowConfig(num_steps=1, order=order))
    poses = [np.asarray(pose0, dtype=np.float64).copy()]
    for linear_velocity, angular_velocity in controls:
        pose, _, _ = stepper.step(
            unicycle_velocity_fn(linear_velocity, angular_velocity), poses[-1], 0.0, dt
        )
        poses.append(pose)
    return np.stack(poses)


def goal_feedback_velocity_fn(
    goal_xy: np.ndarray,
    position_tolerance: float,
    linear_gain: float = 1.0,
    angular_gain: float = 2.0,
    max_linear_velocity: float = 2.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Smooth polar feedback law ``v = k_v rho cos(alpha)``, ``omega = k_w alpha``."""

    goal_xy = np.asarray(goal_xy, dtype=np.float64)

    def velocity_fn(z: np.ndarray, _t: float) -> np.ndarray:
        offset = goal_xy - z[:2]
        rho = float(np.linalg.norm(offset))
        if rho < position_tolerance:
            return np.zeros(3)
        alpha = wrap_angle(np.arctan2(offset[1], offset[0]) - z[2])
        linear_velocity = min(linear_gain * rho * np.cos(alpha), max_linear_velocity)
        linear_velocity = max(linear_velocity, 0.0)
        angular_velocity = angular_gain * alpha
        return np.array(
            [
                linear_velocity * np.cos(z[2]),
                linear_velocity * np.sin(z[2]),
                angular_velocity,
            ]
        )

    return velocity_fn


def navigate_to_goal(
    start_pose: np.ndarray,
    goal_xy: np.ndarray,
    step_size: float = 0.05,
    max_steps: int = 4000,
    position_tolerance: float = 0.01,
    linear_gain: float = 1.0,
    angular_gain: float = 2.0,
    order: int = 2,
) -> NavigationResult:
    """Drive the unicycle to ``goal_xy`` with the transport stepper (closed loop)."""
    from apedp_dyn.video.jet_flow_stepper import JetFlowConfig, JetFlowStepper

    stepper = JetFlowStepper(JetFlowConfig(num_steps=1, order=order))
    velocity_fn = goal_feedback_velocity_fn(
        goal_xy, position_tolerance, linear_gain=linear_gain, angular_gain=angular_gain
    )
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    current = np.asarray(start_pose, dtype=np.float64).copy()
    trajectory = [current.copy()]
    distances = [float(np.linalg.norm(goal_xy - current[:2]))]
    success = distances[0] < position_tolerance
    steps = 0
    while not success and steps < max_steps:
        current, _, _ = stepper.step(velocity_fn, current, 0.0, step_size)
        trajectory.append(current.copy())
        distances.append(float(np.linalg.norm(goal_xy - current[:2])))
        steps += 1
        success = distances[-1] < position_tolerance
    return NavigationResult(
        trajectory=np.stack(trajectory),
        distances=np.asarray(distances),
        success=success,
        steps=steps,
        goal=goal_xy,
    )
