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
    camera_orbit_yaw: float = 0.0                         # 3D out-of-plane yaw rotation (rad/s)
    camera_orbit_pitch: float = 0.0                       # 3D out-of-plane pitch rotation (rad/s)
    camera_dolly_z: float = 0.0                           # 3D optical-axis dolly translation (dz/s)
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
        depth_engine: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        r"""Apply continuous Lie-group SE(3) camera motion (pan, tilt, zoom, roll, 3D orbit yaw/pitch/dolly).

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
        yaw = self.config.camera_orbit_yaw * t
        pitch = self.config.camera_orbit_pitch * t
        dolly = self.config.camera_dolly_z * t

        has_2d = abs(pan) >= 1e-6 or abs(tilt) >= 1e-6 or abs(zoom - 1.0) >= 1e-6 or abs(roll) >= 1e-6
        has_3d = abs(yaw) >= 1e-6 or abs(pitch) >= 1e-6 or abs(dolly) >= 1e-6

        if not has_2d and not has_3d:
            return frame.squeeze(0) if orig_ndim == 3 else frame

        transformed = frame
        if has_3d:
            if depth_engine is not None and hasattr(depth_engine, "warp_frame_se3"):
                transformed = depth_engine.warp_frame_se3(
                    transformed,
                    omega=(float(pitch), float(yaw), float(roll)),
                    trans=(float(pan), float(tilt), float(-dolly)),
                    dt=1.0,
                )
                return transformed.squeeze(0) if orig_ndim == 3 else transformed
            else:
                from jetflow_video.depth_se3 import se3_exp_map
                omega_t = torch.tensor([[pitch, yaw, roll]], device=device, dtype=dtype)
                trans_t = torch.tensor([[pan, tilt, -dolly]], device=device, dtype=dtype)
                rot, t_vec = se3_exp_map(omega_t, trans_t)
                yy = (torch.arange(h, device=device, dtype=dtype) / max(1, h - 1)) * 2.0 - 1.0
                xx = (torch.arange(w, device=device, dtype=dtype) / max(1, w - 1)) * 2.0 - 1.0
                gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                # Radial pseudo-depth profile (foreground object closer at center)
                pseudo_z = 1.0 + 0.35 * (gx * gx + gy * gy)
                f = 1.5
                pts = torch.stack([gx * pseudo_z / f, gy * pseudo_z / f, pseudo_z], dim=0).reshape(1, 3, h * w)
                pts_src = torch.bmm(rot.transpose(-1, -2), pts - t_vec.unsqueeze(-1)).reshape(1, 3, h, w)
                zs = torch.clamp(pts_src[:, 2], min=0.2)
                us = (f * pts_src[:, 0] / zs) / zoom
                vs = (f * pts_src[:, 1] / zs) / zoom
                grid_3d = torch.stack([us, vs], dim=-1).expand(b, h, w, 2)
                transformed = F.grid_sample(transformed, grid_3d, mode="bilinear", padding_mode="border", align_corners=False)
                return transformed.squeeze(0) if orig_ndim == 3 else transformed

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

        grid = F.affine_grid(theta, transformed.size(), align_corners=False)
        transformed = F.grid_sample(transformed, grid, mode="bilinear", padding_mode="border", align_corners=False)

        return transformed.squeeze(0) if orig_ndim == 3 else transformed

    def steer_velocity_field(
        self,
        base_velocity: torch.Tensor,
        z_current: torch.Tensor,
        t: float,
        dt: float,
        depth_engine: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        r"""Combine base generative flow velocity with physical convective, dissipative, and 3D SE(3) terms.

        v_steered = gamma * v_base - (c \cdot \nabla) z + nu * \nabla^2 z + v_3d_se3
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

        if depth_engine is not None and hasattr(depth_engine, "compute_3d_velocity_correction"):
            has_3d = (
                abs(self.config.camera_orbit_yaw) > 1e-6
                or abs(self.config.camera_orbit_pitch) > 1e-6
                or abs(self.config.camera_dolly_z) > 1e-6
            )
            if has_3d:
                v_3d = depth_engine.compute_3d_velocity_correction(
                    z_current,
                    omega=(self.config.camera_orbit_pitch, self.config.camera_orbit_yaw, self.config.camera_roll),
                    trans=(self.config.camera_pan, self.config.camera_tilt, -self.config.camera_dolly_z),
                )
                v_steered = v_steered + v_3d

        return v_steered
