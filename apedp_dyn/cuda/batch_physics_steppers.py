"""GPU & Tensorized Batch Physics Steppers for Large-Scale Hypothesis Search.

Eliminates the CPU-to-GPU PCIe memory bottleneck by executing high-order
physical integration natively on PyTorch CUDA/MPS/CPU tensors across
$10^4$ to $10^6$ candidate hypotheses in parallel.

Supported Batched Steppers:
1. batch_rk4_step: Vectorized 4th-order Runge-Kutta integrator for nonlinear dynamical ODEs.
2. batch_symplectic_verlet_step: Symplectic Hamiltonian Verlet integrator conserving phase-space energy.
3. batch_unicycle_arc_step: Exact circular arc nonholonomic unicycle kinematics with zero-branching vectorized arcs.
4. batch_molecular_conformation_flow: SE(3)-equivariant molecular relaxation gradient flow across B conformations.
5. PyTorch Vectorized Tensor Operations: Vectorized batched tensor math on CUDA/MPS/CPU devices, with optional Triton environment check.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

# Check for Triton availability
try:
    import triton  # type: ignore[import-untyped]  # noqa: F401 (kernel scope, availability probe)
    import triton.language as tl  # type: ignore[import-untyped]  # noqa: F401 (kernel scope, availability probe)
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def is_triton_available() -> bool:
    """Return True if Triton is installed and CUDA is available."""
    return HAS_TRITON and torch.cuda.is_available()


def get_default_physics_device() -> torch.device:
    """Return default hardware accelerator (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_rk4_step(
    state: torch.Tensor,
    dt: float,
    vector_field_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Vectorized 4th-order Runge-Kutta integrator across a batch of candidates.

    Args:
        state: Tensor of shape [B, D] representing B dynamical states of dimension D.
        dt: Integration time step.
        vector_field_fn: Callable mapping Tensor[B, D] -> Tensor[B, D] (f(y)).

    Returns:
        New state Tensor[B, D] after time step dt with O(dt^4) local error.
    """
    k1 = vector_field_fn(state)
    k2 = vector_field_fn(state + 0.5 * dt * k1)
    k3 = vector_field_fn(state + 0.5 * dt * k2)
    k4 = vector_field_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def batch_symplectic_verlet_step(
    q: torch.Tensor,
    p: torch.Tensor,
    dt: float,
    grad_v_fn: Callable[[torch.Tensor], torch.Tensor],
    mass: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symplectic Velocity-Verlet integrator conserving Hamiltonian phase space.

    Args:
        q: Coordinates Tensor of shape [B, D].
        p: Conjugate momenta Tensor of shape [B, D].
        dt: Time step.
        grad_v_fn: Gradient of potential energy dV/dq mapping Tensor[B, D] -> Tensor[B, D].
        mass: Particle mass (default: 1.0).

    Returns:
        tuple (q_next, p_next) after time step dt.
    """
    # Half-step momentum update: p(t + dt/2) = p(t) - (dt/2) * dV/dq(q(t))
    grad_v_curr = grad_v_fn(q)
    p_half = p - 0.5 * dt * grad_v_curr

    # Full-step position update: q(t + dt) = q(t) + dt * p(t + dt/2) / m
    q_next = q + dt * (p_half / mass)

    # Final half-step momentum update: p(t + dt) = p(t + dt/2) - (dt/2) * dV/dq(q(t + dt))
    grad_v_next = grad_v_fn(q_next)
    p_next = p_half - 0.5 * dt * grad_v_next

    return q_next, p_next


def batch_unicycle_arc_step(
    poses: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    dt: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact circular arc unicycle kinematics computed simultaneously across B candidates.

    Evaluates the exact geometry:
        theta_next = theta + omega * dt
        If |omega| < eps:
            x_next = x + v * cos(theta) * dt
            y_next = y + v * sin(theta) * dt
        Else:
            x_next = x + (v / omega) * (sin(theta_next) - sin(theta))
            y_next = y - (v / omega) * (cos(theta_next) - cos(theta))

    Args:
        poses: Tensor of shape [B, 3] with columns (x, y, theta).
        v: Tensor of shape [B] linear velocities.
        omega: Tensor of shape [B] angular velocities.
        dt: Integration time step.
        eps: Threshold below which exact straight-line limit is applied.

    Returns:
        Tensor of shape [B, 3] with new poses.
    """
    theta = poses[:, 2]

    theta_next = theta + omega * dt

    # Safe division clamp for arc computation
    safe_omega = torch.where(torch.abs(omega) < eps, torch.full_like(omega, eps), omega)
    radius = v / safe_omega

    # Circular arc updates
    dx_arc = radius * (torch.sin(theta_next) - torch.sin(theta))
    dy_arc = -radius * (torch.cos(theta_next) - torch.cos(theta))

    # Straight-line limit updates
    dx_line = v * torch.cos(theta) * dt
    dy_line = v * torch.sin(theta) * dt

    # Vectorized condition selection without host sync
    is_straight = (torch.abs(omega) < eps).unsqueeze(-1)

    delta_arc = torch.stack([dx_arc, dy_arc, omega * dt], dim=-1)
    delta_line = torch.stack([dx_line, dy_line, omega * dt], dim=-1)

    delta = torch.where(is_straight, delta_line, delta_arc)

    new_poses = poses + delta
    # Normalize angles to [-pi, pi]
    new_poses[:, 2] = torch.atan2(torch.sin(new_poses[:, 2]), torch.cos(new_poses[:, 2]))
    return new_poses


def batch_molecular_conformation_flow(
    coords: torch.Tensor,
    bonds: tuple[tuple[int, int], ...],
    target_length: float,
    step_size: float = 0.05,
) -> torch.Tensor:
    """Batched SE(3)-equivariant molecular relaxation gradient flow.

    Args:
        coords: Tensor of shape [B, N, 3] representing B molecules with N atoms.
        bonds: Tuple of atom index pairs ((i, j), ...).
        target_length: Target bond distance r0.
        step_size: Gradient descent step size.

    Returns:
        Tensor of shape [B, N, 3] with relaxed coordinates.
    """
    grad = torch.zeros_like(coords)

    for i, j in bonds:
        # Vector from atom j to atom i: shape [B, 3]
        diff = coords[:, i, :] - coords[:, j, :]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        
        # Harmonic bond force: F = 2 * (dist - r0) * (diff / dist)
        force = 2.0 * (dist - target_length) * (diff / dist)

        grad[:, i, :] += force
        grad[:, j, :] -= force

    # Gradient step
    new_coords = coords - step_size * grad

    # Conserve center of mass: subtract batch COM drift
    com = torch.mean(new_coords, dim=1, keepdim=True)
    orig_com = torch.mean(coords, dim=1, keepdim=True)
    new_coords = new_coords - com + orig_com

    return new_coords


def batch_multi_agent_cbf_step(
    poses: torch.Tensor,
    targets: torch.Tensor,
    roles: torch.Tensor,
    dt: float,
    r_safe: float = 0.5,
    r_warning: float = 1.0,
    k_att: float = 1.5,
    k_rep: float = 2.0,
    v_max: float = 1.0,
    omega_max: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Vectorized multi-agent kinematics with repulsive artificial potential collision avoidance on CUDA/CPU.

    Evaluates N non-holonomic mobile robots concurrently across B candidate environments.
    Note: Uses an artificial repulsive potential field (Khatib repulsive gradient) for
    inter-agent collision deflection; this is not a certified quadratic-programming
    Control Barrier Function (CBF-QP) with formal forward-invariance certificates.

    Args:
        poses: Tensor of shape [B, N, 3] with columns (x, y, theta).
        targets: Tensor of shape [B, N, 2] target positions.
        roles: Tensor of shape [B, N], 1.0 for active striker, 0.0 for support/recovery.
        dt: Integration step size.
        r_safe: Minimum inter-agent safety distance.
        r_warning: Repulsion engagement threshold.
        k_att: Attraction gain to target.
        k_rep: Repulsive potential gain.
        v_max: Maximum linear speed.
        omega_max: Maximum turning speed.
        eps: Tolerance for straight-line vs arc geometry.

    Returns:
        Tensor of shape [B, N, 3] with next poses.
    """
    B, N, _ = poses.shape
    positions = poses[:, :, :2]  # [B, N, 2]
    thetas = poses[:, :, 2]      # [B, N]

    # 1. Attractive forces toward targets
    d_target = targets - positions  # [B, N, 2]
    dist_target = torch.norm(d_target, dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]
    f_att = k_att * (d_target / dist_target) * torch.clamp(dist_target, max=1.0)

    # 2. Control barrier repulsive forces across all agent pairs
    # positions.unsqueeze(2): [B, N, 1, 2], positions.unsqueeze(1): [B, 1, N, 2]
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2] (diff[b, i, j] = pos_i - pos_j)
    dist_pair = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, N, 1]

    # Mask self-interaction (diagonal i == j)
    eye_mask = torch.eye(N, device=poses.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
    
    in_warning = (dist_pair < r_warning) & ~eye_mask
    safe_dist = torch.clamp(dist_pair, min=r_safe)
    rep_magnitude = k_rep * (1.0 / safe_dist - 1.0 / r_warning) * (1.0 / (dist_pair ** 2))
    f_rep_pairs = torch.where(in_warning, rep_magnitude * (diff / dist_pair), torch.zeros_like(diff))

    # Tangential breakaway (mirrors multi_agent_transport.py:92-101): on
    # head-on/collinear approaches the radial repulsion is exactly
    # anti-parallel to the attraction, so the net force collapses and both
    # agents deadlock. When repulsion strongly opposes the CURRENT HEADING
    # (anti-parallel component > 0.7), inject the CPU reference's fixed
    # rightward tangential component; the opposing agent resolves the
    # mirrored heading to the opposite side, breaking the symmetry.
    heading_vec = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)  # [B, N, 2]
    rep_dir = diff / dist_pair  # [B, N, N, 2] unit repulsion direction
    # anti_par[b, i, j] > 0 when repulsion opposes agent i's heading
    anti_par = -(rep_dir * heading_vec.unsqueeze(2)).sum(dim=-1)  # [B, N, N]
    perp = torch.stack([rep_dir[..., 1], -rep_dir[..., 0]], dim=-1)  # [B, N, N, 2]
    tangential = torch.where(
        (anti_par > 0.7).unsqueeze(-1),
        anti_par.unsqueeze(-1) * rep_magnitude * perp,
        torch.zeros_like(perp),
    )
    tangential = torch.where(in_warning, tangential, torch.zeros_like(tangential))
    f_rep_pairs = f_rep_pairs + tangential
    f_rep = torch.sum(f_rep_pairs, dim=2)  # [B, N, 2]

    # 3. Role-weighted net force
    role_multiplier = torch.where(roles > 0.5, torch.tensor(1.0, device=poses.device), torch.tensor(0.5, device=poses.device)).unsqueeze(-1)
    f_total = f_att + f_rep
    total_norm = torch.norm(f_total, dim=-1).clamp(min=1e-8)  # [B, N]

    desired_heading = torch.atan2(f_total[:, :, 1], f_total[:, :, 0])  # [B, N]
    heading_error = torch.atan2(torch.sin(desired_heading - thetas), torch.cos(desired_heading - thetas))

    # Kinematic control commands
    v_cmd = role_multiplier.squeeze(-1) * v_max * torch.clamp(torch.cos(heading_error), min=0.0) * torch.clamp(total_norm, max=1.0)
    omega_cmd = torch.clamp(2.0 * heading_error, -omega_max, omega_max)

    # 4. Batch arc stepping
    # Flatten [B, N] to [B*N] to reuse exact arc kinematics
    flat_poses = poses.reshape(B * N, 3)
    flat_v = v_cmd.reshape(B * N)
    flat_omega = omega_cmd.reshape(B * N)

    next_flat_poses = batch_unicycle_arc_step(flat_poses, flat_v, flat_omega, dt, eps=eps)
    return next_flat_poses.reshape(B, N, 3)

