"""Physical Parametric Steering Knobs and Camera SE(3) Flow Controller.

Status: experimental

Provides direct, continuous parametric controls ('physics knobs') over generative video:
1. Convection velocity c = (cx, cy): Directional wind and current drift.
2. Fluid viscosity nu: Dissipative Laplacian damping controlling turbulence vs laminar motion.
3. Continuous Camera SE(3) motion: Rigid pan, tilt, zoom, and roll on spatial coordinates.
4. Continuous time dilation gamma: Real-time speed modulation, slow-motion, and exact time-reversal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import apedp_kernel as ak
    _HAS_RUST_LAPLACIAN = hasattr(ak, "rime_laplacian_dissipation_2d")
except ImportError:
    # `ak` stays unbound when the native module is absent; uses are flag-gated.
    _HAS_RUST_LAPLACIAN = False


@dataclass(frozen=True)
class PhysicalSteeringConfig:
    """Explicit Physical Control Parameters for Generative Video Flows."""
    convection_velocity: tuple[float, float] = (0.0, 0.0)  # (cx, cy)
    viscosity_nu: float = 0.0                             # nu >= 0
    camera_pan: float = 0.0                               # dx per second
    camera_tilt: float = 0.0                              # dy per second
    camera_zoom: float = 0.0                              # zoom velocity (d scale / dt)
    camera_roll: float = 0.0                              # angular velocity (d theta / dt in rad/s)
    time_dilation: float = 1.0                            # gamma (1.0 = normal, 0.25 = slow-mo, -1.0 = reverse)


class PhysicalSteeringController:
    """Controller applying explicit physical differential operators to video latents."""

    def __init__(self, config: PhysicalSteeringConfig | None = None):
        self.config = config or PhysicalSteeringConfig()

    def apply_convection(
        self,
        field: np.ndarray,
        dt: float,
        cx: float | None = None,
        cy: float | None = None,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> np.ndarray:
        """Apply convective spatial advection: Z_{new} = Z - dt * (cx * dZ/dx + cy * dZ/dy).

        Accepts 3D (C, H, W) or 4D (T, C, H, W) fields.
        """
        c_x = cx if cx is not None else self.config.convection_velocity[0]
        c_y = cy if cy is not None else self.config.convection_velocity[1]

        if abs(c_x) < 1e-12 and abs(c_y) < 1e-12:
            return field.copy()

        # Spatial central differences
        grad_x = (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / (2.0 * dx)
        grad_y = (np.roll(field, -1, axis=-2) - np.roll(field, 1, axis=-2)) / (2.0 * dy)

        advection = c_x * grad_x + c_y * grad_y
        return np.asarray(field - (dt * advection))

    def apply_viscosity_dissipation(
        self,
        field: np.ndarray,
        dt: float,
        nu: float | None = None,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> np.ndarray:
        """Apply dissipative Laplacian diffusion: Z_{new} = Z + dt * nu * \nabla^2 Z.

        Accelerated via native Rust AVX2/Rayon kernel when available.
        """
        nu_val = nu if nu is not None else self.config.viscosity_nu
        if nu_val <= 1e-12:
            return field.copy()

        orig_shape = field.shape
        if field.ndim == 3:
            c, h, w = orig_shape
            field_3d = field
        elif field.ndim == 4:
            t, c, h, w = orig_shape
            # Flatten (T, C) as effective channels
            field_3d = field.reshape((t * c, h, w))
            c = t * c
        else:
            raise ValueError(f"Expected 3D or 4D field, got shape {orig_shape}")

        if _HAS_RUST_LAPLACIAN:
            field_c = np.ascontiguousarray(field_3d, dtype=np.float64).ravel()
            out_flat = ak.rime_laplacian_dissipation_2d(field_c, h, w, c, nu_val, dt, dx, dy)
            out = np.array(out_flat, dtype=np.float64).reshape(orig_shape)
            return out

        # Vectorized NumPy fallback
        dx_sq = dx * dx
        dy_sq = dy * dy
        d2x = (np.roll(field_3d, -1, axis=-1) - 2.0 * field_3d + np.roll(field_3d, 1, axis=-1)) / dx_sq
        d2y = (np.roll(field_3d, -1, axis=-2) - 2.0 * field_3d + np.roll(field_3d, 1, axis=-2)) / dy_sq
        lap = d2x + d2y
        out_3d = field_3d + (dt * nu_val * lap)
        return np.asarray(out_3d.reshape(orig_shape))

    def apply_camera_transform(
        self,
        frame: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Apply continuous Lie-group SE(3) camera motion (pan, tilt, zoom, roll) to a frame (C, H, W)."""
        if frame.ndim != 3:
            raise ValueError(f"Expected 3D frame (C, H, W), got {frame.shape}")

        c, h, w = frame.shape
        pan = self.config.camera_pan * t
        tilt = self.config.camera_tilt * t
        zoom = 1.0 + self.config.camera_zoom * t
        zoom = max(0.2, min(5.0, zoom))  # Guard against inversion or singularity
        roll = self.config.camera_roll * t

        if abs(pan) < 1e-6 and abs(tilt) < 1e-6 and abs(zoom - 1.0) < 1e-6 and abs(roll) < 1e-6:
            return frame.copy()

        # Coordinate grid centered at frame center
        cy, cx = (h - 1.0) * 0.5, (w - 1.0) * 0.5
        y_coords, x_coords = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing='ij')

        # Translate to origin
        x_rel = x_coords - cx
        y_rel = y_coords - cy

        # Inverse rotation & zoom for back-sampling
        cos_theta = math.cos(-roll)
        sin_theta = math.sin(-roll)
        inv_zoom = 1.0 / zoom

        x_rot = (cos_theta * x_rel - sin_theta * y_rel) * inv_zoom
        y_rot = (sin_theta * x_rel + cos_theta * y_rel) * inv_zoom

        # Add pan and tilt shifts
        src_x = x_rot + cx - pan
        src_y = y_rot + cy - tilt

        # Bilinear interpolation
        x0 = np.floor(src_x).astype(np.int64)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(np.int64)
        y1 = y0 + 1

        wx1 = src_x - x0
        wx0 = 1.0 - wx1
        wy1 = src_y - y0
        wy0 = 1.0 - wy1

        # Periodic wrapping for seamless boundary flow
        x0_mod = np.mod(x0, w)
        x1_mod = np.mod(x1, w)
        y0_mod = np.mod(y0, h)
        y1_mod = np.mod(y1, h)

        transformed = np.zeros_like(frame)
        for ci in range(c):
            f = frame[ci]
            i00 = f[y0_mod, x0_mod]
            i01 = f[y0_mod, x1_mod]
            i10 = f[y1_mod, x0_mod]
            i11 = f[y1_mod, x1_mod]
            transformed[ci] = wy0 * (wx0 * i00 + wx1 * i01) + wy1 * (wx0 * i10 + wx1 * i11)

        return transformed

    def steer_velocity_field(
        self,
        base_velocity: np.ndarray,
        z_current: np.ndarray,
        t: float,
        dt: float,
    ) -> np.ndarray:
        r"""Combine base generative flow velocity with physical convective and dissipative terms.

        v_steered = gamma * v_base - (c \cdot \nabla) z + nu * \nabla^2 z
        """
        gamma = self.config.time_dilation
        v_steered = gamma * base_velocity

        # Physical convection component
        cx, cy = self.config.convection_velocity
        if abs(cx) > 1e-12 or abs(cy) > 1e-12:
            grad_x = (np.roll(z_current, -1, axis=-1) - np.roll(z_current, 1, axis=-1)) / 2.0
            grad_y = (np.roll(z_current, -1, axis=-2) - np.roll(z_current, 1, axis=-2)) / 2.0
            convective_velocity = -(cx * grad_x + cy * grad_y)
            v_steered += convective_velocity

        # Physical dissipative viscosity component
        nu = self.config.viscosity_nu
        if nu > 1e-12:
            d2x = np.roll(z_current, -1, axis=-1) - 2.0 * z_current + np.roll(z_current, 1, axis=-1)
            d2y = np.roll(z_current, -1, axis=-2) - 2.0 * z_current + np.roll(z_current, 1, axis=-2)
            dissipative_velocity = nu * (d2x + d2y)
            v_steered += dissipative_velocity

        return v_steered
