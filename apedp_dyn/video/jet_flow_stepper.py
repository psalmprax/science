r"""Jet-Flow Curvature Stepper for High-Precision 2-Step / 4-Step Video Generation.

Status: experimental

Resolves the 50-step diffusion bottleneck by computing continuous trajectory curvature
(total acceleration a_t = d v / dt = \partial_t v + (grad_z v) * v) using 2-Jet (Jet2) Taylor
expansions, achieving high-fidelity video latent generation in 1 to 4 steps.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import numpy as np

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
        dt: float | None = None,
    ) -> np.ndarray:
        r"""Compute total convective acceleration along the flow trajectory:
        a_t = d/dt v(z(t), t) = \partial_t v + (\nabla_z v) * v
        via a single fused directional derivative along the characteristic ray
        (z + h * v_t, t + h) without constructing the full Jacobian matrix.
        """
        eps = float(self.config.eps_fd)
        h = -eps if (dt is not None and dt < 0.0) else eps
        t_lo = min(self.config.t_start, self.config.t_end)
        t_hi = max(self.config.t_start, self.config.t_end)

        if t_lo <= t <= t_hi:
            if t + h > t_hi:
                h_eff = -eps if (t - eps >= t_lo) else (t_hi - t)
            elif t + h < t_lo:
                h_eff = eps if (t + eps <= t_hi) else (t_lo - t)
            else:
                h_eff = h
        else:
            h_eff = h

        t_probe = t + h_eff
        if abs(h_eff) <= 1e-9:
            return np.zeros_like(v_current)

        z_probe = z + h_eff * v_current
        v_probe = velocity_fn(z_probe, t_probe)
        return np.asarray((v_probe - v_current) / h_eff)

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

        # Second-order Jet2 Taylor step using fused directional material derivative (2 evals/step)
        if ak is not None and hasattr(ak, "rime_jet2_video_flow_step"):
            eps = float(self.config.eps_fd)
            h = -eps if dt < 0.0 else eps
            t_lo = min(self.config.t_start, self.config.t_end)
            t_hi = max(self.config.t_start, self.config.t_end)

            if t_lo <= t <= t_hi:
                if t + h > t_hi:
                    h_eff = -eps if (t - eps >= t_lo) else (t_hi - t)
                elif t + h < t_lo:
                    h_eff = eps if (t + eps <= t_hi) else (t_lo - t)
                else:
                    h_eff = h
            else:
                h_eff = h

            t_probe = t + h_eff
            if abs(h_eff) > 1e-9:
                v_probe = velocity_fn(z_t + h_eff * v_t, t_probe)
                dt_span = h_eff
            else:
                v_probe = v_t
                dt_span = 1.0

            z_flat = np.ascontiguousarray(z_t.ravel(), dtype=np.float64)
            v_flat = np.ascontiguousarray(v_t.ravel(), dtype=np.float64)
            v_probe_flat = np.ascontiguousarray(v_probe.ravel(), dtype=np.float64)
            z_next_flat, a_flat = ak.rime_jet2_video_flow_step(
                z_flat,
                v_flat,
                v_probe_flat,
                v_flat,
                v_flat,
                v_flat,
                float(dt),
                float(dt_span),
                1.0,
                2,
            )
            return z_next_flat.reshape(z_t.shape), v_t, a_flat.reshape(z_t.shape)

        a_t = self.compute_acceleration(velocity_fn, z_t, t, v_t, dt=dt)
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
