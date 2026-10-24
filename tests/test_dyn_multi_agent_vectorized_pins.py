"""Vectorized pair-force pins: behavior-identical to the legacy scalar loop.

`multi_agent_step`'s per-pair Python loop (the measured ~6 µs/invocation
bottleneck crossing the 5 ms real-time budget at N=32) was replaced by ONE
vectorized all-pairs sweep (`pairwise_repulsive_forces`). Behavior
identity is pinned at THREE levels:

1. Per-pair: each (i, j) off-diagonal cell of the matrix kernel equals the
   scalar `compute_cbf_repulsive_force` exactly — including every branch:
   degenerate collision (< 1e-9 → the fixed [k_rep, 0.0] vector), far
   (>= r_warning → zeros), near (active with head-on tangential terms), the
   head-on HIT (anti > 0.7) and MISS (anti <= 0.7) paths, and the exact
   r_warning boundary (`>=` → zeros vs `<` → active).
2. Per-agent sums: equality with a sequential legacy accumulation at
   <= 1e-12 relative (float summation order may differ by ulps; values
   do not differ semantically).
3. Whole-step: `multi_agent_step` outputs (poses, linear + angular
   velocities) equal an INLINE legacy reference loop (the pre-vectorization
   semantics, re-implemented here from the scalar function) on randomized
   fixtures with mixed roles across N sizes.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.robotics.multi_agent_transport import (
    AgentRole,
    compute_cbf_repulsive_force,
    exact_unicycle_step,
    multi_agent_step,
    pairwise_repulsive_forces,
    wrap_angle,
)

RWS = (0.5, 1.0)  # r_safe, r_warning


def _scalar_force(pi, pj, heading_i, r_safe, r_warning, k_rep=2.0):
    """Verbatim copy of compute_cbf_repulsive_force semantics for the
    per-pair branch check (kept independent of the module so a future
    edit of the scalar function cannot silently weaken the pin)."""
    diff = np.asarray(pi, dtype=float) - np.asarray(pj, dtype=float)
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return np.array([k_rep, 0.0])
    if dist >= r_warning:
        return np.zeros(2)
    direction = diff / dist
    magnitude = k_rep * (1.0 / max(r_safe, dist) - 1.0 / r_warning) * (1.0 / (dist ** 2.0))
    force = magnitude * direction
    if heading_i is not None:
        heading_vec = np.array([np.cos(heading_i), np.sin(heading_i)])
        anti_par = float(-direction @ heading_vec)
        if anti_par > 0.7:
            perp = np.array([-direction[1], direction[0]])
            force = force + anti_par * magnitude * perp
    return force


def _legacy_loop_forces(positions, headings, r_safe, r_warning, k_rep=2.0):
    """Sequential-accumulation force sums — the pre-vectorization inner loop."""
    n = len(positions)
    out = np.zeros((n, 2))
    for i in range(n):
        acc = np.zeros(2)
        for j in range(n):
            if i != j:
                acc += compute_cbf_repulsive_force(
                    positions[i], positions[j], r_safe=r_safe,
                    r_warning=r_warning, k_rep=k_rep, heading_i=headings[i])
        out[i] = acc
    return out


@pytest.mark.parametrize("seed, n", [(0, 6), (1, 11), (2, 25)])
def test_pairwise_force_cells_match_scalar_rule(seed, n):
    """Each (i, j) pair of a two-agent subsystem matches the scalar rule."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-2.0, 2.0, size=(n, 2))
    th = rng.uniform(-np.pi, np.pi, size=n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            expect = _scalar_force(pos[i], pos[j], th[i], *RWS)
            two_pos = np.stack([pos[i], pos[j]])
            two = pairwise_repulsive_forces(
                two_pos, np.array([th[i], th[j]]), *RWS)
            assert np.allclose(two[0], expect, rtol=1e-12, atol=1e-14), (
                i, j, two[0], expect)


def test_pairwise_sum_matches_sequential_legacy_accumulation():
    """Per-agent sums equal the sequential legacy += accumulation."""
    rng = np.random.default_rng(7)
    pos = rng.uniform(-2.0, 2.0, size=(12, 2))
    th = rng.uniform(-np.pi, np.pi, size=12)
    got = pairwise_repulsive_forces(pos, th, *RWS)
    legacy = _legacy_loop_forces(pos, th, *RWS)
    assert np.allclose(got, legacy, rtol=1e-12, atol=1e-14)


def test_pairwise_branches_degenerate_far_boundary_head_on():
    k_rep = 2.0
    # degenerate: coincident positions → each agent gets (N-1) × [k_rep, 0]
    pos = np.zeros((3, 2))
    th = np.zeros(3)
    got = pairwise_repulsive_forces(pos, th, *RWS, k_rep)
    legacy = _legacy_loop_forces(pos, th, *RWS, k_rep)
    expect = np.array([k_rep, 0.0]) + np.array([k_rep, 0.0])
    assert np.allclose(got[0], expect, atol=1e-14)
    assert np.allclose(got, legacy, rtol=1e-12, atol=1e-14)

    # far: zeros
    pos2 = np.array([[0.0, 0.0], [5.0, 0.0]])
    assert np.allclose(
        pairwise_repulsive_forces(pos2, np.zeros(2), *RWS, k_rep), 0.0)

    # boundary: distance exactly r_warning is OUTSIDE the active band
    r = RWS[1]
    pos3 = np.array([[0.0, 0.0], [r, 0.0]])
    assert np.allclose(
        pairwise_repulsive_forces(pos3, np.zeros(2), *RWS, k_rep), 0.0)
    # strictly inside → the ACTIVE branch (nonzero, exact scalar equality),
    # though the magnitude vanishes smoothly toward the boundary because
    # magnitude ∝ (1/max(r_safe, d) − 1/r_warning) → boundary continuity.
    pos4 = np.array([[0.0, 0.0], [r - 1e-9, 0.0]])
    f_in = pairwise_repulsive_forces(pos4, None, *RWS, k_rep)
    assert np.abs(f_in[0]).max() > 0.0
    s_in = _scalar_force(pos4[0], pos4[1], None, *RWS, k_rep)
    assert np.allclose(f_in[0], s_in, rtol=1e-12, atol=1e-14)

    # head-on HIT: direction opposes heading by >0.7 → tangential term present
    pos5 = np.array([[0.0, 0.0], [0.8, 0.0]])
    th5 = np.array([0.0, np.pi])
    f5 = pairwise_repulsive_forces(pos5, th5, *RWS, k_rep)
    s5 = _scalar_force(pos5[0], pos5[1], th5[0], *RWS, k_rep)
    assert np.allclose(f5[0], s5, rtol=1e-12, atol=1e-14)
    assert abs(f5[0, 1]) > 1e-6

    # head-on MISS: heading perpendicular (anti = -direction·heading < 0.7)
    th6 = np.array([np.pi / 2, 0.0])
    f6 = pairwise_repulsive_forces(pos5, th6, *RWS, k_rep)
    s6 = _scalar_force(pos5[0], pos5[1], th6[0], *RWS, k_rep)
    assert np.allclose(f6[0], s6, rtol=1e-12, atol=1e-14)
    assert abs(f6[0, 1]) <= 1e-15  # no tangential term


@pytest.mark.parametrize("n", [2, 5, 8, 33])
def test_multi_agent_step_matches_legacy_reference(n):
    """Whole-step outputs equal the inline legacy reference semantics."""
    rng = np.random.default_rng(100 + n)
    poses = np.column_stack([rng.uniform(-2, 2, size=(n, 2)),
                             rng.uniform(-np.pi, np.pi, size=n)])
    targets = rng.uniform(-1.5, 1.5, size=(n, 2))
    roles = [AgentRole.ACTIVE_STRIKER if i % 3 == 0 else AgentRole.SUPPORT_RECOVERY
             for i in range(n)]

    got_pos, got_v, got_om = multi_agent_step(poses, targets, roles, dt=0.05)

    # inline legacy reference (pre-vectorization semantics, module constants)
    k_att, v_max = 1.5, 1.0
    omega_max, dt = 2.0, 0.05
    ref_pos = np.zeros_like(poses)
    ref_v = np.zeros(n)
    ref_om = np.zeros(n)
    positions = poses[:, :2]
    rep_ref = _legacy_loop_forces(positions, poses[:, 2], *RWS)
    for i in range(n):
        p_i = poses[i, :2]
        theta_i = poses[i, 2]
        d_target = targets[i] - p_i
        dist_target = float(np.linalg.norm(d_target))
        if dist_target > 1e-6:
            f_att = k_att * (d_target / dist_target) * min(dist_target, 1.0)
        else:
            f_att = np.zeros(2)
        f_rep = rep_ref[i]
        role_multiplier = 1.0 if roles[i] == AgentRole.ACTIVE_STRIKER else 0.5
        f_total = f_att + f_rep
        total_norm = float(np.linalg.norm(f_total))
        if total_norm > 1e-6:
            desired_heading = np.arctan2(f_total[1], f_total[0])
            heading_error = wrap_angle(desired_heading - theta_i)
            v_cmd = role_multiplier * v_max * max(0.0, np.cos(heading_error)) * min(1.0, total_norm)
            omega_cmd = np.clip(2.0 * heading_error, -omega_max, omega_max)
        else:
            v_cmd = 0.0
            omega_cmd = 0.0
        ref_pos[i] = exact_unicycle_step(poses[i], v_cmd, omega_cmd, dt)
        ref_v[i] = v_cmd
        ref_om[i] = omega_cmd

    assert np.allclose(got_pos, ref_pos, rtol=1e-12, atol=1e-13), (
        np.max(np.abs(got_pos - ref_pos)))
    assert np.allclose(got_v, ref_v, rtol=1e-12, atol=1e-14)
    assert np.allclose(got_om, ref_om, rtol=1e-12, atol=1e-14)
