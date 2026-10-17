r"""PyTorch-Native Physical Parametric Steering and SE(3) Camera Motion Controller.

Provides continuous, physical parametric steering knobs for generative video latents:
1. Convection velocity c = (cx, cy): Wind, current, and smoke drift.
2. Fluid viscosity nu: Laplacian dissipative smoothing controlling laminar vs turbulent motion.
3. Continuous Camera SE(3) Motion: Differentiable GPU affine transforms for pan, tilt, zoom, and roll.
4. Continuous Time Dilation gamma: Real-time speed modulation, slow-motion, and time-reversal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicalSteeringConfig:
    """Explicit Physical Control Parameters for Generative Video Flows."""
    convection_velocity: Tuple[float, float] = (0.0, 0.0)  # (cx, cy)
    viscosity_nu: float = 0.0                             # nu >= 0
    camera_pan: float = 0.0                               # dx per second
    camera_tilt: float = 0.0                              # dy per second
    camera_zoom: float = 0.0                              # zoom velocity (d scale / dt)
    camera_roll: float = 0.0                              # angular velocity (rad/s)
    time_dilation: float = 1.0                            # gamma (1.0 = normal, 0.25 = slow-mo, -1.0 = reverse)


class PhysicalSteeringController:
    """PyTorch Controller applying physical operators to video latents and velocity fields."""

    def __init__(self, config: Optional[PhysicalSteeringConfig] = None) -> None:
        self.config = config or PhysicalSteeringConfig()

    def apply_convection(
        self,
        field: torch.Tensor,
        dt: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> torch.Tensor:
        r"""Apply convective advection: Z_{new} = Z - dt * (cx * dZ/dx + cy * dZ/dy).

        Accepts 3D (C, H, W) or 4D (T, C, H, W) tensors.
        """
        c_x = cx if cx is not None else self.config.convection_velocity[0]
        c_y = cy if cy is not None else self.config.convection_velocity[1]

        if abs(c_x) < 1e-12 and abs(c_y) < 1e-12:
            return field.clone()

        grad_x = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / (2.0 * dx)
        grad_y = (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2)) / (2.0 * dy)

        advection = c_x * grad_x + c_y * grad_y
        return field - (dt * advection)

    def apply_viscosity_dissipation(
        self,
        field: torch.Tensor,
        dt: float,
        nu: Optional[float] = None,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> torch.Tensor:
        r"""Apply dissipative Laplacian diffusion: Z_{new} = Z + dt * nu * \nabla^2 Z."""
        nu_val = nu if nu is not None else self.config.viscosity_nu
        if nu_val <= 1e-12:
            return field.clone()

        dx_sq = dx * dx
        dy_sq = dy * dy
        d2x = (torch.roll(field, -1, dims=-1) - 2.0 * field + torch.roll(field, 1, dims=-1)) / dx_sq
        d2y = (torch.roll(field, -1, dims=-2) - 2.0 * field + torch.roll(field, 1, dims=-2)) / dy_sq
        lap = d2x + d2y
        return field + (dt * nu_val * lap)

    def apply_camera_transform_torch(
        self,
        frame: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        r"""Apply continuous Lie-group SE(3) camera motion (pan, tilt, zoom, roll) using PyTorch grid_sample.

        frame: (C, H, W) or (B, C, H, W).
        """
        orig_ndim = frame.ndim
        if orig_ndim == 3:
            frame = frame.unsqueeze(0)  # (1, C, H, W)

        b, c, h, w = frame.shape
        device = frame.device
        dtype = frame.dtype

        pan = self.config.camera_pan * t
        tilt = self.config.camera_tilt * t
        zoom = 1.0 + self.config.camera_zoom * t
        zoom = max(0.2, min(5.0, zoom))
        roll = self.config.camera_roll * t

        if abs(pan) < 1e-6 and abs(tilt) < 1e-6 and abs(zoom - 1.0) < 1e-6 and abs(roll) < 1e-6:
            return frame.squeeze(0) if orig_ndim == 3 else frame

        cos_t = math.cos(roll)
        sin_t = math.sin(roll)
        inv_z = 1.0 / zoom

        # 2x3 affine matrix for torch affine_grid:
        tx = -pan if abs(pan) <= 2.0 else -pan * 2.0 / max(1, w)
        ty = -tilt if abs(tilt) <= 2.0 else -tilt * 2.0 / max(1, h)
        theta = torch.tensor(
            [
                [
                    [inv_z * cos_t, -inv_z * sin_t, tx],
                    [inv_z * sin_t,  inv_z * cos_t, ty],
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(b, 1, 1)

        grid = F.affine_grid(theta, frame.size(), align_corners=False)
        transformed = F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=False)

        return transformed.squeeze(0) if orig_ndim == 3 else transformed

    def steer_velocity_field(
        self,
        base_velocity: torch.Tensor,
        z_current: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        r"""Combine base generative flow velocity with physical convective and dissipative terms.

        v_steered = gamma * v_base - (c \cdot \nabla) z + nu * \nabla^2 z
        """
        gamma = self.config.time_dilation
        v_steered = gamma * base_velocity

        cx, cy = self.config.convection_velocity
        if abs(cx) > 1e-12 or abs(cy) > 1e-12:
            grad_x = (torch.roll(z_current, -1, dims=-1) - torch.roll(z_current, 1, dims=-1)) / 2.0
            grad_y = (torch.roll(z_current, -1, dims=-2) - torch.roll(z_current, 1, dims=-2)) / 2.0
            convective_velocity = -(cx * grad_x + cy * grad_y)
            v_steered = v_steered + convective_velocity

        nu = self.config.viscosity_nu
        if nu > 1e-12:
            d2x = torch.roll(z_current, -1, dims=-1) - 2.0 * z_current + torch.roll(z_current, 1, dims=-1)
            d2y = torch.roll(z_current, -1, dims=-2) - 2.0 * z_current + torch.roll(z_current, 1, dims=-2)
            dissipative_velocity = nu * (d2x + d2y)
            v_steered = v_steered + dissipative_velocity

        return v_steered
