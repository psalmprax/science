"""Tests for Multi-Agent Robotics & Collaborative CBF Kinematics.

Verifies:
1. Control Barrier Function (CBF) repulsive force gradient properties.
2. Multi-agent single step execution with unicycle kinematic constraints.
3. Cooperative PingpongDuo-style turn-taking coordination with certified safety margin preservation.
4. Tensorized batch_multi_agent_cbf_step execution on PyTorch across B=100 environments.
"""
from __future__ import annotations

import numpy as np
import torch

from apedp_dyn.cuda.batch_physics_steppers import batch_multi_agent_cbf_step
from apedp_dyn.robotics.multi_agent_transport import (
    AgentRole,
    MultiAgentCoordinationResult,
    compute_cbf_repulsive_force,
    multi_agent_step,
    simulate_multi_agent_coordination,
)


def test_cbf_repulsive_force_properties():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([2.0, 0.0])  # Beyond r_warning=1.0

    f_rep = compute_cbf_repulsive_force(p1, p2, r_safe=0.5, r_warning=1.0)
    assert np.allclose(f_rep, 0.0)

    # Within warning zone
    p3 = np.array([0.7, 0.0])
    f_rep_close = compute_cbf_repulsive_force(p1, p3, r_safe=0.5, r_warning=1.0)
    # Force on p1 should be negative in x (pushing away from p3)
    assert f_rep_close[0] < 0.0
    assert abs(f_rep_close[1]) < 1e-10


def test_multi_agent_step_kinematics():
    poses = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, np.pi / 2],
    ])
    targets = np.array([
        [2.0, 0.0],
        [2.0, 2.0],
    ])
    roles = [AgentRole.ACTIVE_STRIKER, AgentRole.SUPPORT_RECOVERY]

    next_poses, v, omega = multi_agent_step(poses, targets, roles, dt=0.1)

    assert next_poses.shape == (2, 3)
    assert len(v) == 2
    assert len(omega) == 2

    # Agent 0 should have driven forward toward target
    assert next_poses[0, 0] > 0.0


def test_cooperative_pingpongduo_turn_taking_safety():
    # 2 agents positioned along y-axis navigating to a shared target at (3.0, 0.0)
    initial_poses = np.array([
        [0.0, 0.5, 0.0],
        [0.0, -0.5, 0.0],
    ])
    target = np.array([3.0, 0.0])
    r_safe = 0.35

    res = simulate_multi_agent_coordination(
        initial_poses=initial_poses,
        target=target,
        max_steps=150,
        dt=0.05,
        r_safe=r_safe,
        r_warning=0.8,
    )

    assert isinstance(res, MultiAgentCoordinationResult)
    # Verify safety: at all steps, distance between agents never fell below r_safe
    assert res.safety_violation is False
    assert np.all(res.minimum_inter_agent_distances >= r_safe - 1e-3)
    assert res.target_reached is True


def test_batch_multi_agent_cbf_step_tensorized():
    B = 100
    N = 3

    torch.manual_seed(42)
    # Scenario A: Agents in parallel lanes (y = 1.0, 0.0, -1.0) moving toward (3.0, y)
    poses = torch.zeros(B, N, 3)
    poses[:, 0, :] = torch.tensor([0.0, 1.0, 0.0])
    poses[:, 1, :] = torch.tensor([0.0, 0.0, 0.0])
    poses[:, 2, :] = torch.tensor([0.0, -1.0, 0.0])

    targets = torch.tensor([[3.0, 1.0], [3.0, 0.0], [3.0, -1.0]]).unsqueeze(0).expand(B, N, 2)
    roles = torch.tensor([[1.0, 0.0, 0.0]]).expand(B, N)  # Agent 0 active, others support

    next_poses = batch_multi_agent_cbf_step(
        poses=poses,
        targets=targets,
        roles=roles,
        dt=0.05,
        r_safe=0.4,
        r_warning=0.8,
    )

    assert next_poses.shape == (B, N, 3)
    assert not torch.isnan(next_poses).any()
    assert not torch.isinf(next_poses).any()

    # All unblocked agents in parallel lanes drive forward toward their targets
    assert (next_poses[:, :, 0] > poses[:, :, 0]).all()


def test_batch_multi_agent_head_on_collision_avoidance():
    """Verify that batch_multi_agent_cbf_step dynamically deflects agents on a head-on collision course.

    Two agents are initialized facing each other on near-identical y-lines with opposing targets.
    Without repulsive steering, the agents would collide head-on (distance -> 0).
    With repulsive potential field steering, the agents deflect laterally and maintain
    inter-agent distance strictly above r_safe throughout the entire trajectory.
    """
    B, N = 1, 2
    initial_poses = torch.tensor([[[0.0, 0.05, 0.0], [3.0, -0.05, np.pi]]], dtype=torch.float32)
    targets = torch.tensor([[[3.0, 0.05], [0.0, -0.05]]], dtype=torch.float32)
    roles = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    r_safe = 0.35
    r_warning = 0.90
    cur_poses = initial_poses
    min_dist_encountered = float("inf")

    for _ in range(50):
        cur_poses = batch_multi_agent_cbf_step(
            poses=cur_poses,
            targets=targets,
            roles=roles,
            dt=0.08,
            r_safe=r_safe,
            r_warning=r_warning,
            k_att=1.5,
            k_rep=2.5,
            v_max=1.0,
            omega_max=2.0,
        )
        d = float(torch.norm(cur_poses[0, 0, :2] - cur_poses[0, 1, :2]).item())
        min_dist_encountered = min(min_dist_encountered, d)

    # 1. Verify safety: minimum distance never breached r_safe
    assert min_dist_encountered >= r_safe, f"Safety violated: min distance {min_dist_encountered} < {r_safe}"
    # 2. Verify dynamic interaction: lateral deflection away from center line
    assert cur_poses[0, 0, 1].item() > 0.15, "Agent 0 failed to deflect laterally"
    assert cur_poses[0, 1, 1].item() < -0.15, "Agent 1 failed to deflect laterally"
    # 3. Verify target progress: agent 0 moved right, agent 1 moved left
    assert cur_poses[0, 0, 0].item() > 1.5, "Agent 0 failed to advance toward target"
    assert cur_poses[0, 1, 0].item() < 1.5, "Agent 1 failed to advance toward target"

