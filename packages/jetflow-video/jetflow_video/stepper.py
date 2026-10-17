r"""PyTorch-Native Jet2 Curvature Flow Stepper for 2-Step / 4-Step Video Generation.

Integrates continuous probability flow ODEs:
  d z_t / dt = v_\theta(z_t, t, c)
from t = 0 (prior noise) to t = 1 (sharp video latent) using 2nd-order Jet2 Taylor steps:
  z_{t + \Delta t} = z_t + \Delta t * v_t + 0.5 * (\Delta t)^2 * a_t + O(\Delta t^3)
where a_t = \partial_t v + (\nabla_z v) * v is the convective acceleration computed
via efficient directional finite differences or forward-mode differentiation without
constructing large Jacobian matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch


@dataclass(frozen=True)
class JetFlowConfig:
    """Configuration for Jet2 Curvature Flow Stepper."""
    num_steps: int = 2               # Number of integration steps (2, 4, 8)
    order: int = 2                   # 1 = Euler (1st order), 2 = Jet2 Taylor (2nd order)
    eps_fd: float = 1e-3             # Epsilon for directional convective derivative
    adaptive_step: bool = False      # Whether to adapt step size by curvature
    t_start: float = 0.0             # Initial noise time (t=0)
    t_end: float = 1.0               # Final clean video time (t=1)


@dataclass
class JetFlowTrajectory:
    """Full trajectory recorded during continuous ODE integration."""
    timesteps: List[float]
    latents: List[torch.Tensor]
    velocities: List[torch.Tensor]
    accelerations: Optional[List[torch.Tensor]]
    final_latent: torch.Tensor
    total_curvature: float


class JetFlowStepper:
    r"""PyTorch GPU/CPU 2nd-Order Jet2 Curvature Flow Stepper.

    Achieves photorealistic video latent integration in 2 to 4 steps by accounting
    for trajectory curvature along the continuous probability path.
    """

    def __init__(self, config: Optional[JetFlowConfig] = None) -> None:
        self.config = config or JetFlowConfig()

    def compute_acceleration_torch(
        self,
        velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        z: torch.Tensor,
        t: float,
        v_current: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute total convective acceleration along the flow trajectory:
        a_t = \partial_t v + (\nabla_z v) * v
        using directional derivatives without constructing the full Jacobian matrix.
        """
        eps = self.config.eps_fd

        # 1. Partial temporal derivative: \partial_t v
        t_plus = min(1.0, t + eps)
        t_minus = max(0.0, t - eps)
        dt_span = t_plus - t_minus

        if dt_span > 1e-8:
            v_t_plus = velocity_fn(z, t_plus)
            v_t_minus = velocity_fn(z, t_minus)
            v_t_dot = (v_t_plus - v_t_minus) / dt_span
        else:
            v_t_dot = torch.zeros_like(v_current)

        # 2. Convective spatial derivative: (\nabla_z v) * v
        v_norm = torch.linalg.norm(v_current)
        if v_norm > 1e-8:
            h_disp = eps / v_norm.item()
            z_plus = z + h_disp * v_current
            z_minus = z - h_disp * v_current

            v_spatial_plus = velocity_fn(z_plus, t)
            v_spatial_minus = velocity_fn(z_minus, t)
            convective_dot = (v_spatial_plus - v_spatial_minus) / (2.0 * h_disp)
        else:
            convective_dot = torch.zeros_like(v_current)

        return v_t_dot + convective_dot

    def step(
        self,
        velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        z_t: torch.Tensor,
        t: float,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""Perform a single continuous ODE integration step from t to t + dt.

        Returns:
            (z_next, velocity, acceleration_or_none)
        """
        v_t = velocity_fn(z_t, t)

        if self.config.order == 1:
            z_next = z_t + dt * v_t
            return z_next, v_t, None

        # 2nd-order Jet2 Taylor expansion
        a_t = self.compute_acceleration_torch(velocity_fn, z_t, t, v_t)
        z_next = z_t + dt * v_t + 0.5 * (dt ** 2) * a_t
        return z_next, v_t, a_t

    def generate(
        self,
        velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        z_init: torch.Tensor,
    ) -> JetFlowTrajectory:
        r"""Integrate continuous flow from z_init (t=0) to video latent (t=1)."""
        z_curr = z_init.clone()
        n_steps = max(1, self.config.num_steps)
        t_grid = torch.linspace(self.config.t_start, self.config.t_end, n_steps + 1).tolist()

        timesteps: List[float] = [float(t_grid[0])]
        latents: List[torch.Tensor] = [z_curr.clone()]
        velocities: List[torch.Tensor] = []
        accelerations: List[torch.Tensor] = []
        curvatures: List[float] = []

        for step_idx in range(n_steps):
            t_curr = float(t_grid[step_idx])
            t_next = float(t_grid[step_idx + 1])
            dt = t_next - t_curr

            z_next, v_t, a_t = self.step(velocity_fn, z_curr, t_curr, dt)

            velocities.append(v_t)
            if a_t is not None:
                accelerations.append(a_t)
                curvatures.append(float(torch.mean(torch.abs(a_t)).item()))
            else:
                curvatures.append(0.0)

            z_curr = z_next
            timesteps.append(t_next)
            latents.append(z_curr.clone())

        total_curvature = float(sum(curvatures)) if curvatures else 0.0

        return JetFlowTrajectory(
            timesteps=timesteps,
            latents=latents,
            velocities=velocities,
            accelerations=accelerations if self.config.order >= 2 else None,
            final_latent=z_curr,
            total_curvature=total_curvature,
        )
