r"""Robot trajectory transport: arc analytics, stepper convergence, goal convergence.

Status: experimental
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.robotics.trajectory_transport import (
    exact_unicycle_step,
    navigate_to_goal,
    propagate_with_stepper,
    simulate_unicycle_exact,
    wrap_angle,
)

_START = np.array([0.0, 0.0, 0.0])


def _full_path_error(order: int, n_steps: int) -> float:
    dt = (2.0 * np.pi) / n_steps
    controls = [(1.0, 1.0)] * n_steps
    exact = simulate_unicycle_exact(_START, controls, dt)
    approximate = propagate_with_stepper(_START, controls, dt, order=order)
    position_error = np.linalg.norm(approximate[:, :2] - exact[:, :2], axis=1)
    angle_error = np.abs(np.vectorize(wrap_angle)(approximate[:, 2] - exact[:, 2]))
    return float(np.max(position_error + angle_error))


def test_exact_step_matches_arc_analytics():
    step = exact_unicycle_step(_START, 1.0, 1.0, 0.1)
    centre = np.array([0.0, 1.0])
    assert abs(np.linalg.norm(step[:2] - centre) - 1.0) < 1e-12
    assert abs(step[2] - 0.1) < 1e-12
    straight = exact_unicycle_step(_START, 2.0, 0.0, 0.5)
    assert np.allclose(straight, [1.0, 0.0, 0.0], atol=1e-12)


def test_full_circle_stepper_path_accuracy():
    n_steps = 80
    dt = (2.0 * np.pi) / n_steps
    controls = [(1.0, 1.0)] * n_steps
    exact = simulate_unicycle_exact(_START, controls, dt)
    assert abs(wrap_angle(float(exact[-1, 2]))) < 1e-10
    assert np.max(np.abs(exact[-1, :2] - _START[:2])) < 1e-10

    jet2 = propagate_with_stepper(_START, controls, dt, order=2)
    error = np.max(np.linalg.norm(jet2[:, :2] - exact[:, :2], axis=1))
    assert error < 5e-3, error


def test_stepper_convergence_orders():
    counts = (40, 80, 160)
    euler = [_full_path_error(1, n) for n in counts]
    jet2 = [_full_path_error(2, n) for n in counts]
    euler_ratios = [euler[i] / euler[i + 1] for i in range(len(counts) - 1)]
    jet2_ratios = [jet2[i] / jet2[i + 1] for i in range(len(counts) - 1)]
    assert all(1.8 < ratio < 2.2 for ratio in euler_ratios), euler_ratios
    assert all(3.6 < ratio < 4.4 for ratio in jet2_ratios), jet2_ratios
    assert jet2[0] < euler[0] / 10.0, (jet2[0], euler[0])


def test_navigate_to_goal_converges_from_random_starts():
    rng = np.random.default_rng(11)
    for _ in range(20):
        bearing = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.5, 3.0)
        start = np.array(
            [radius * np.cos(bearing), radius * np.sin(bearing), rng.uniform(-np.pi, np.pi)]
        )
        result = navigate_to_goal(start, np.zeros(2), step_size=0.02, max_steps=6000)
        assert result.success, (start, result.distances[-1])
        assert result.steps < 1000, result.steps
        assert result.distances[-1] < 0.011, result.distances[-1]


def test_navigation_lyapunov_distance_nonincreasing():
    rng = np.random.default_rng(23)
    for _ in range(10):
        bearing = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.5, 3.0)
        start = np.array(
            [radius * np.cos(bearing), radius * np.sin(bearing), rng.uniform(-np.pi, np.pi)]
        )
        result = navigate_to_goal(start, np.zeros(2), step_size=0.02, max_steps=6000)
        increments = np.diff(result.distances)
        assert float(np.max(increments)) <= 0.0, float(np.max(increments))
        assert result.distances[-1] < 0.011, result.distances[-1]
        assert result.distances[-1] < 0.5 * result.distances[0]


def test_goal_directly_behind_converges():
    result = navigate_to_goal(_START, np.array([-2.0, 0.0]), step_size=0.02, max_steps=6000)
    assert result.success, result.distances[-1]
    assert result.distances[-1] < 0.011
