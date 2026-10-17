r"""Jet-Flow Curvature Stepper for High-Precision 2-Step / 4-Step Video Generation.

Status: experimental

Resolves the 50-step diffusion bottleneck by computing continuous trajectory curvature
(total acceleration a_t = d v / dt = \partial_t v + (grad_z v) * v) using 2-Jet (Jet2) Taylor
expansions, achieving high-fidelity video latent generation in 1 to 4 steps.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from types import ModuleType

ak: ModuleType | None = None
try:
    import apedp_kernel as ak
except ImportError:
    pass  # `ak` is Optional-typed; use sites check `ak is not None`


@dataclass(frozen=True)
class JetFlowConfig:
    """Configuration for Jet2 Curvature Flow Stepper."""
    num_steps: int = 2               # Number of sampling steps (e.g. 2, 4, 8)
    order: int = 2                   # 1 = Euler (1st order), 2 = Jet2 Taylor (2nd order)
    eps_fd: float = 1e-4             # Step size for directional derivative calculation
    adaptive_step: bool = False      # Whether to adaptively refine step size on high curvature
    t_start: float = 0.0             # Initial noise time (t=0)
    t_end: float = 1.0               # Final clean video time (t=1)


@dataclass(frozen=True)
class JetFlowTrajectory:
    """Full time trajectory produced by JetFlowStepper."""
    timesteps: list[float]
    latents: list[np.ndarray]
    velocities: list[np.ndarray]
    accelerations: list[np.ndarray] | None
    final_latent: np.ndarray
    total_curvature: float


class JetFlowStepper:
    r"""High-order continuous flow matching ODE stepper for video generation.

    Integrates the probability transport equation:
      d z_t / dt = v_\theta(z_t, t, c)
    from t=0 (prior noise) to t=1 (sharp video latent) using 2nd-order Jet2 Taylor steps:
      z_{t + \Delta t} = z_t + \Delta t * v_t + 0.5 * (\Delta t)^2 * a_t + O(\Delta t^3)
    where a_t = \partial_t v + (\nabla_z v) * v is the convective acceleration.
    """

    def __init__(self, config: JetFlowConfig | None = None) -> None:
        self.config = config or JetFlowConfig()

    def compute_acceleration(
        self,
        velocity_fn: Callable[[np.ndarray, float], np.ndarray],
        z: np.ndarray,
        t: float,
        v_current: np.ndarray,
    ) -> np.ndarray:
        r"""Compute total convective acceleration along the flow trajectory:
        a_t = \partial_t v + (\nabla_z v) * v
        using directional derivatives without constructing the full Jacobian matrix.
        """
        eps = self.config.eps_fd

        # 1. Partial temporal derivative: \partial_t v
        # Central difference in time: (v(z, t + eps) - v(z, t - eps)) / (2 * eps)
        t_plus = min(1.0, t + eps)
        t_minus = max(0.0, t - eps)
        dt_span = t_plus - t_minus
        if dt_span > 1e-8:
            v_t_plus = velocity_fn(z, t_plus)
            v_t_minus = velocity_fn(z, t_minus)
            v_t_dot = (v_t_plus - v_t_minus) / dt_span
        else:
            v_t_dot = np.zeros_like(v_current)

        # 2. Convective spatial derivative: (\nabla_z v) * v
        # Directional derivative of v along the direction of vector v itself:
        # D_v v(z) = [v(z + eps * v) - v(z - eps * v)] / (2 * eps)
        v_norm = np.linalg.norm(v_current)
        if v_norm > 1e-8:
            # Scale displacement so eps is proportional to unit direction
            h_disp = eps / v_norm
            z_plus = z + h_disp * v_current
            z_minus = z - h_disp * v_current

            v_spatial_plus = velocity_fn(z_plus, t)
            v_spatial_minus = velocity_fn(z_minus, t)
            convective_dot = (v_spatial_plus - v_spatial_minus) / (2.0 * h_disp)
        else:
            convective_dot = np.zeros_like(v_current)

        # Total acceleration along trajectory: a_t = v_t_dot + convective_dot
        return np.asarray(v_t_dot + convective_dot)

    def step(
        self,
        velocity_fn: Callable[[np.ndarray, float], np.ndarray],
        z_t: np.ndarray,
        t: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        r"""Execute a single continuous integration step from t to t + dt.

        Returns:
        - (z_next, velocity, acceleration_or_none)
        """
        v_t = velocity_fn(z_t, t)

        if self.config.order == 1:
            if ak is not None and hasattr(ak, "rime_jet2_video_flow_step"):
                z_flat = np.ascontiguousarray(z_t.ravel(), dtype=np.float64)
                v_flat = np.ascontiguousarray(v_t.ravel(), dtype=np.float64)
                z_next_flat, _ = ak.rime_jet2_video_flow_step(
                    z_flat, v_flat, v_flat, v_flat, v_flat, v_flat, float(dt), 1.0, 1.0, 1
                )
                return z_next_flat.reshape(z_t.shape), v_t, None
            z_next = z_t + dt * v_t
            return z_next, v_t, None

        # Second-order Jet2 Taylor step with acceleration
        if ak is not None and hasattr(ak, "rime_jet2_video_flow_step"):
            eps = self.config.eps_fd
            t_plus = min(1.0, t + eps)
            t_minus = max(0.0, t - eps)
            dt_span = max(1e-8, t_plus - t_minus)

            v_t_plus = velocity_fn(z_t, t_plus)
            v_t_minus = velocity_fn(z_t, t_minus)

            v_norm = np.linalg.norm(v_t)
            if v_norm > 1e-8:
                h_disp = eps / v_norm
                z_plus = z_t + h_disp * v_t
                z_minus = z_t - h_disp * v_t
                v_sp_plus = velocity_fn(z_plus, t)
                v_sp_minus = velocity_fn(z_minus, t)
                two_h = 2.0 * h_disp
            else:
                v_sp_plus = v_t
                v_sp_minus = v_t
                two_h = 1.0

            z_flat = np.ascontiguousarray(z_t.ravel(), dtype=np.float64)
            z_next_flat, a_flat = ak.rime_jet2_video_flow_step(
                z_flat,
                np.ascontiguousarray(v_t.ravel(), dtype=np.float64),
                np.ascontiguousarray(v_t_plus.ravel(), dtype=np.float64),
                np.ascontiguousarray(v_t_minus.ravel(), dtype=np.float64),
                np.ascontiguousarray(v_sp_plus.ravel(), dtype=np.float64),
                np.ascontiguousarray(v_sp_minus.ravel(), dtype=np.float64),
                float(dt),
                float(dt_span),
                float(two_h),
                2,
            )
            return z_next_flat.reshape(z_t.shape), v_t, a_flat.reshape(z_t.shape)

        a_t = self.compute_acceleration(velocity_fn, z_t, t, v_t)
        z_next = z_t + dt * v_t + 0.5 * (dt ** 2) * a_t
        return z_next, v_t, a_t

    def generate(
        self,
        velocity_fn: Callable[[np.ndarray, float], np.ndarray],
        z_init: np.ndarray,
    ) -> JetFlowTrajectory:
        r"""Integrate continuous flow from z_init (t=0) to video latent (t=1).

        Parameters:
        - velocity_fn: function (z, t) -> velocity array matching z shape.
        - z_init: initial noise array at t=0 (e.g. from standard normal N(0, I)).

        Returns:
        - JetFlowTrajectory containing all intermediate states and final latent.
        """
        z_curr = np.ascontiguousarray(z_init, dtype=np.float64).copy()
        n_steps = max(1, self.config.num_steps)
        t_grid = np.linspace(self.config.t_start, self.config.t_end, n_steps + 1)

        timesteps: list[float] = [float(t_grid[0])]
        latents: list[np.ndarray] = [z_curr.copy()]
        velocities: list[np.ndarray] = []
        accelerations: list[np.ndarray] = []
        curvatures: list[float] = []

        for step_idx in range(n_steps):
            t_curr = float(t_grid[step_idx])
            t_next = float(t_grid[step_idx + 1])
            dt = t_next - t_curr

            z_next, v_t, a_t = self.step(velocity_fn, z_curr, t_curr, dt)

            velocities.append(v_t)
            if a_t is not None:
                accelerations.append(a_t)
                curvatures.append(float(np.mean(np.abs(a_t))))
            else:
                curvatures.append(0.0)

            z_curr = z_next
            timesteps.append(t_next)
            latents.append(z_curr.copy())

        total_curvature = float(np.sum(curvatures)) if curvatures else 0.0

        return JetFlowTrajectory(
            timesteps=timesteps,
            latents=latents,
            velocities=velocities,
            accelerations=accelerations if self.config.order >= 2 else None,
            final_latent=z_curr,
            total_curvature=total_curvature,
        )
