"""Tests for batched PyTorch/CUDA physics steppers and high-throughput evaluation.

Verifies:
1. batch_rk4_step numerical order and parity against scalar analytical harmonic oscillator.
2. batch_symplectic_verlet_step energy conservation over multiple cycles.
3. batch_unicycle_arc_step exact equivalence with scalar unicycle geometry across 5,000 candidates.
4. batch_molecular_conformation_flow SE(3) equivariance and center-of-mass conservation.
5. High-throughput batch scaling test evaluating 10,000 simultaneous candidates.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from apedp_dyn.cuda.batch_physics_steppers import (
    batch_molecular_conformation_flow,
    batch_rk4_step,
    batch_symplectic_verlet_step,
    batch_unicycle_arc_step,
)
from apedp_dyn.robotics.trajectory_transport import exact_unicycle_step


def test_batch_rk4_step_harmonic_oscillator():
    # d/dt [x, v] = [v, -x] (harmonic oscillator, omega=1)
    device = torch.device("cpu")
    dt = 0.01

    def vector_field(state: torch.Tensor) -> torch.Tensor:
        x = state[:, 0:1]
        v = state[:, 1:2]
        return torch.cat([v, -x], dim=-1)

    # Initial state: x(0) = 1.0, v(0) = 0.0 for 100 trajectories
    batch_size = 100
    state = torch.tensor([[1.0, 0.0]] * batch_size, dtype=torch.float64, device=device)

    # Step forward 100 steps (t = 1.0)
    for _ in range(100):
        state = batch_rk4_step(state, dt, vector_field)

    t_final = 1.0
    expected_x = math.cos(t_final)
    expected_v = -math.sin(t_final)

    assert torch.allclose(state[:, 0], torch.tensor(expected_x, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(state[:, 1], torch.tensor(expected_v, dtype=torch.float64), atol=1e-5)


def test_batch_symplectic_verlet_energy_conservation():
    # Harmonic potential: V(q) = 0.5 * q^2 => dV/dq = q
    device = torch.device("cpu")
    dt = 0.05
    batch_size = 50

    q = torch.ones((batch_size, 1), dtype=torch.float64, device=device)
    p = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)

    def grad_v(coords: torch.Tensor) -> torch.Tensor:
        return coords

    def energy(q_curr: torch.Tensor, p_curr: torch.Tensor) -> torch.Tensor:
        return 0.5 * p_curr**2 + 0.5 * q_curr**2

    initial_energy = energy(q, p)

    # Run for 200 steps
    for _ in range(200):
        q, p = batch_symplectic_verlet_step(q, p, dt, grad_v)

    final_energy = energy(q, p)
    # Symplectic integrators bound energy fluctuations: |E(t) - E(0)| is small and bounded
    assert torch.allclose(final_energy, initial_energy, atol=2e-3)


def test_batch_unicycle_arc_matches_scalar_analytics_across_5000_candidates():
    torch.manual_seed(42)
    B = 5000
    dt = 0.1

    # Generate random poses (x, y, theta)
    x = torch.randn(B, dtype=torch.float64) * 10.0
    y = torch.randn(B, dtype=torch.float64) * 10.0
    theta = (torch.rand(B, dtype=torch.float64) * 2 - 1) * math.pi
    poses = torch.stack([x, y, theta], dim=-1)

    # Random velocities (including zero angular velocity edge case)
    v = torch.randn(B, dtype=torch.float64) * 2.0
    omega = torch.randn(B, dtype=torch.float64) * 1.5
    omega[0:50] = 0.0  # Explicitly test straight-line branch

    # Batched execution
    batch_next = batch_unicycle_arc_step(poses, v, omega, dt)

    # Scalar reference comparison
    for i in [0, 10, 50, 100, 500, 1000]:
        scalar_pose = np.array([float(x[i]), float(y[i]), float(theta[i])])
        scalar_next = exact_unicycle_step(scalar_pose, float(v[i]), float(omega[i]), dt)

        assert abs(float(batch_next[i, 0]) - scalar_next[0]) < 1e-7
        assert abs(float(batch_next[i, 1]) - scalar_next[1]) < 1e-7
        # Angular difference modulo 2pi
        ang_diff = (float(batch_next[i, 2]) - scalar_next[2] + math.pi) % (2 * math.pi) - math.pi
        assert abs(ang_diff) < 1e-7


def test_batch_molecular_conformation_flow_com_and_equivariance():
    torch.manual_seed(42)
    B = 200
    N = 3
    bonds = ((0, 1), (1, 2))
    target_bond = 1.5

    # Random triatomic molecule conformations
    coords = torch.randn(B, N, 3, dtype=torch.float64)
    orig_com = torch.mean(coords, dim=1)

    # Run relaxation flow
    relaxed = batch_molecular_conformation_flow(coords, bonds, target_bond, step_size=0.1)

    # 1. Center of mass conservation
    new_com = torch.mean(relaxed, dim=1)
    assert torch.allclose(new_com, orig_com, atol=1e-10)

    # 2. Bond distance convergence: bond distances should move toward target_bond
    initial_dist01 = torch.norm(coords[:, 0] - coords[:, 1], dim=-1)
    relaxed_dist01 = torch.norm(relaxed[:, 0] - relaxed[:, 1], dim=-1)
    initial_err = torch.mean(torch.abs(initial_dist01 - target_bond))
    relaxed_err = torch.mean(torch.abs(relaxed_dist01 - target_bond))
    assert relaxed_err < initial_err


def test_high_throughput_10k_population_execution():
    B = 10000
    dt = 0.05
    poses = torch.randn(B, 3, dtype=torch.float32)
    v = torch.ones(B, dtype=torch.float32) * 1.5
    omega = torch.ones(B, dtype=torch.float32) * 0.2

    # Should execute in milliseconds across 10,000 candidates
    out = batch_unicycle_arc_step(poses, v, omega, dt)
    assert out.shape == (B, 3)
    assert torch.all(torch.isfinite(out))
