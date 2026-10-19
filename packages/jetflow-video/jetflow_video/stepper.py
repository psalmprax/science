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
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Compute total convective acceleration along the flow trajectory:
        a_t = d/dt v(z(t), t) = \partial_t v + (\nabla_z v) * v
        via a single fused directional derivative along the characteristic ray
        (z + h * v_t, t + h), requiring only 1 additional velocity evaluation
        instead of 4 separate partial-derivative probes.
        """
        eps = float(self.config.eps_fd)
        # Step in the integration direction so t + h stays inside [0, 1]
        if dt is not None and dt < 0.0:
            h = -eps if (t - eps) >= 0.0 else eps
        else:
            h = eps if (t + eps) <= 1.0 else -eps

        t_probe = max(0.0, min(1.0, t + h))
        h_eff = t_probe - t
        if abs(h_eff) <= 1e-9:
            return torch.zeros_like(v_current)

        z_probe = z + h_eff * v_current
        v_probe = velocity_fn(z_probe, t_probe)
        return (v_probe - v_current) / h_eff

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

        # 2nd-order Jet2 Taylor expansion using fused material derivative (2 evals/step)
        a_t = self.compute_acceleration_torch(velocity_fn, z_t, t, v_t, dt=dt)
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
