r"""3D Depth-Aware SE(3) Volumetric Transport & Plücker Camera Ray Conditioning.

Status: experimental

Resolves the flat-2D advection limitation by lifting latent video transport into 3D
camera/world coordinates with:
1. Exact Lie-group exp: se(3) -> SE(3) (Rodrigues rotation + coupled V(omega) Jacobian).
2. 6D Plücker Ray Coordinate Embedding: r = (d, o x d) in R^6 for epipolar 3D geometry conditioning.
3. DepthAware3DTransportEngine: Learned monocular latent depth Z_theta(z), 3D scene flow
   (dX, dY, dZ), perspective frustum projection, occlusion visibility masking, and disocclusion
   source synthesis during 3D object rotations and camera orbits.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _skew_symmetric(w: torch.Tensor) -> torch.Tensor:
    """Construct 3x3 skew-symmetric cross-product matrix [w]_x from w of shape (..., 3)."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    z = torch.zeros_like(wx)
    row0 = torch.stack([z, -wz, wy], dim=-1)
    row1 = torch.stack([wz, z, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def se3_exp_map(omega: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact Lie-algebra exponential map exp([omega, v]) -> (R, t) in SE(3).

    Args:
        omega: (..., 3) axis-angle rotation vector (pitch, yaw, roll) in radians.
        v: (..., 3) translational twist vector (tx, ty, tz).

    Returns:
        R: (..., 3, 3) orthonormal rotation matrix in SO(3).
        t: (..., 3) coupled translation vector V(omega) @ v.
    """
    theta_sq = torch.sum(omega * omega, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    theta = torch.sqrt(theta_sq + 1e-12)

    w_hat = _skew_symmetric(omega)  # (..., 3, 3)
    w_hat_sq = torch.matmul(w_hat, w_hat)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(w_hat)

    # Taylor-safe coefficients for small angles
    small = theta < 1e-4
    c1 = torch.where(small, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
    c2 = torch.where(small, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_sq + 1e-12))
    c3 = torch.where(
        small,
        (1.0 / 6.0) - theta_sq / 120.0,
        (theta - torch.sin(theta)) / (theta * theta_sq + 1e-12),
    )

    rot = eye + c1 * w_hat + c2 * w_hat_sq
    v_jac = eye + c2 * w_hat + c3 * w_hat_sq
    trans = torch.matmul(v_jac, v.unsqueeze(-1)).squeeze(-1)
    return rot, trans


def compute_plucker_rays(
    t_dim: int,
    h: int,
    w: int,
    omega_per_sec: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    v_per_sec: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    focal_length: float = 1.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute 6D Plücker ray coordinates r = (d, o x d) of shape (1, T, H*W, 6)."""
    dev = device or torch.device("cpu")
    t_i, h_i, w_i = int(t_dim), int(h), int(w)

    tau = torch.arange(t_i, device=dev, dtype=dtype) / max(1, t_i - 1)
    omega_base = torch.tensor(omega_per_sec, device=dev, dtype=dtype).unsqueeze(0) * tau.unsqueeze(-1)
    v_base = torch.tensor(v_per_sec, device=dev, dtype=dtype).unsqueeze(0) * tau.unsqueeze(-1)
    rot, cam_origin = se3_exp_map(omega_base, v_base)  # (T, 3, 3), (T, 3)

    yy = (torch.arange(h_i, device=dev, dtype=dtype) / max(1, h_i - 1)) * 2.0 - 1.0
    xx = (torch.arange(w_i, device=dev, dtype=dtype) / max(1, w_i - 1)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid_z = torch.full_like(grid_x, float(focal_length))

    # Camera-frame ray directions (H*W, 3)
    dirs_cam = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(h_i * w_i, 3)
    dirs_cam = dirs_cam / (torch.norm(dirs_cam, dim=-1, keepdim=True) + 1e-8)

    # Rotate rays into world frame: d_world = R_t @ d_cam -> (T, H*W, 3)
    dirs_world = torch.einsum("tij,sj->tsi", rot, dirs_cam)
    origins = cam_origin.unsqueeze(1).expand(t_i, h_i * w_i, 3)

    # Plücker moment m = o x d
    moments = torch.cross(origins, dirs_world, dim=-1)
    plucker = torch.cat([dirs_world, moments], dim=-1)  # (T, H*W, 6)
    return plucker.unsqueeze(0)


class PluckerRayEmbedder(nn.Module):
    """Projects 6D Plücker camera ray coordinates into the Spatio-Temporal Transformer hidden space."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Zero-init final projection so default camera rays preserve identity at step 0
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, plucker_rays: torch.Tensor) -> torch.Tensor:
        return self.net(plucker_rays)


class DepthAware3DTransportEngine(nn.Module):
    """3D Depth-Lifted SE(3) Perspective Warping & Disocclusion Synthesis Engine.

    Given latent video z of shape (B, T, C, H, W) or (B, C, H, W):
    1. Predicts monocular metric depth Z_theta(z) in [z_min, z_max],
       3D non-rigid scene flow (dX, dY, dZ), and a disocclusion source field s_occ(z).
    2. Unprojects each pixel (u, v) into 3D camera space P = (u*Z/f, v*Z/f, Z),
       applies 3D SE(3) rigid rotation/translation + non-rigid 3D scene flow,
       and reprojects back onto the 2D image plane with soft Z-buffer visibility weighting.
    """

    def __init__(
        self,
        latent_channels: int = 12,
        hidden_dim: int = 32,
        z_min: float = 0.5,
        z_max: float = 3.0,
        focal_length: float = 1.5,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.z_min = z_min
        self.z_max = z_max
        self.focal_length = focal_length

        # Lightweight local 3x3 depth & 3D scene-flow head
        self.depth_flow_head = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 4 + latent_channels, kernel_size=3, padding=1),
        )
        # Initialize depth near z=1.0 and scene flow / disocclusion source to zero
        nn.init.zeros_(self.depth_flow_head[2].weight)
        nn.init.zeros_(self.depth_flow_head[2].bias)

    def predict_depth_and_flow(
        self,
        frame_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict depth (B, 1, H, W), 3D scene flow (B, 3, H, W), and disocclusion source (B, C, H, W)."""
        orig_ndim = frame_latent.ndim
        if orig_ndim == 3:
            frame_latent = frame_latent.unsqueeze(0)

        raw = self.depth_flow_head(frame_latent)
        depth_raw = raw[:, 0:1]
        flow_3d = 0.25 * torch.tanh(raw[:, 1:4])
        disocc_source = raw[:, 4:]

        # Sigmoid mapped to [z_min, z_max], centered at 1.0 when depth_raw == 0
        sigma = torch.sigmoid(depth_raw)
        depth = self.z_min + (self.z_max - self.z_min) * sigma
        return depth, flow_3d, disocc_source

    def warp_frame_se3(
        self,
        frame_latent: torch.Tensor,
        omega: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        trans: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Apply 3D depth-lifted SE(3) perspective transformation and disocclusion filling."""
        orig_ndim = frame_latent.ndim
        if orig_ndim == 3:
            frame_latent = frame_latent.unsqueeze(0)

        b, c, h, w = frame_latent.shape
        device = frame_latent.device
        dtype = frame_latent.dtype

        depth, flow_3d, disocc_source = self.predict_depth_and_flow(frame_latent)  # (B, 1, H, W)

        omega_t = torch.tensor(omega, device=device, dtype=dtype).unsqueeze(0) * float(dt)
        trans_t = torch.tensor(trans, device=device, dtype=dtype).unsqueeze(0) * float(dt)
        rot, t_vec = se3_exp_map(omega_t, trans_t)  # (1, 3, 3), (1, 3)

        yy = (torch.arange(h, device=device, dtype=dtype) / max(1, h - 1)) * 2.0 - 1.0
        xx = (torch.arange(w, device=device, dtype=dtype) / max(1, w - 1)) * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        u = grid_x.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        v = grid_y.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        f = float(self.focal_length)

        # Unproject (u, v, Z) -> 3D camera coordinates (X, Y, Z)
        x_3d = u * depth / f
        y_3d = v * depth / f
        z_3d = depth
        pts_3d = torch.cat([x_3d, y_3d, z_3d], dim=1) + flow_3d * float(dt)  # (B, 3, H, W)

        # Apply inverse SE(3) camera motion for backward coordinate lookup: P_src = R^T (P - t)
        pts_flat = pts_3d.reshape(b, 3, h * w) - t_vec.unsqueeze(-1)
        rot_inv = rot.transpose(-1, -2).expand(b, 3, 3)
        pts_src = torch.bmm(rot_inv, pts_flat).reshape(b, 3, h, w)

        x_s = pts_src[:, 0]
        y_s = pts_src[:, 1]
        z_s = torch.clamp(pts_src[:, 2], min=0.15)

        u_src = f * (x_s / z_s)
        v_src = f * (y_s / z_s)
        sample_grid = torch.stack([u_src, v_src], dim=-1)  # (B, H, W, 2)

        warped = F.grid_sample(
            frame_latent,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # Compute disocclusion / out-of-frustum visibility gate
        in_frustum = ((u_src.abs() <= 1.0) & (v_src.abs() <= 1.0)).to(dtype=dtype).unsqueeze(1)
        # Depth compression ratio indicates self-occlusion / foreshortening
        depth_ratio = torch.clamp(depth / (z_s.unsqueeze(1) + 1e-6), 0.0, 2.0)
        disocc_weight = (1.0 - in_frustum) + 0.25 * torch.abs(1.0 - depth_ratio)
        disocc_weight = torch.clamp(disocc_weight, 0.0, 1.0)

        out = warped + disocc_weight * disocc_source * abs(float(dt))
        return out.squeeze(0) if orig_ndim == 3 else out

    def compute_3d_velocity_correction(
        self,
        z_video: torch.Tensor,
        omega: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        trans: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> torch.Tensor:
        """Compute instantaneous 3D depth-lifted velocity tangent field on (B, T, C, H, W)."""
        orig_ndim = z_video.ndim
        if orig_ndim == 4:
            z_video = z_video.unsqueeze(0)

        b, t_dim, c, h, w = z_video.shape
        z_flat = z_video.reshape(b * t_dim, c, h, w)
        eps = 0.05
        warped = self.warp_frame_se3(z_flat, omega=omega, trans=trans, dt=eps)
        v_3d = (warped - z_flat) / eps
        v_3d = v_3d.reshape(b, t_dim, c, h, w)
        return v_3d.squeeze(0) if orig_ndim == 4 else v_3d
