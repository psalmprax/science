r"""PyTorch-Native Noether Invariant Conservation and Solenoidal Projection Engine.

Guarantees physical conservation laws on generative video flows:
1. Divergence-free solenoidal velocity field projection via 2D/3D Helmholtz-Hodge decomposition
   (\nabla \cdot v = 0), preventing object deformation and 'melting' artifacts.
2. Differentiable spatial divergence loss for end-to-end training.
3. Noether invariant monitoring: divergence residual, kinetic energy drift, mass persistence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import apedp_kernel
except (ImportError, OSError):
    apedp_kernel = None
    logger.debug("Rust Noether kernel unavailable; using PyTorch", exc_info=True)
except Exception:  # noqa: BLE001 -- optional native module initialization must not prevent fallback
    apedp_kernel = None
    logger.debug("Rust Noether kernel initialization failed; using PyTorch", exc_info=True)

_RUST_NOETHER_AVAILABLE = callable(getattr(apedp_kernel, "jetflow_noether_project", None))


@dataclass(frozen=True)
class NoetherConfig:
    """Configuration for Noether Conservation and Invariant Enforcement."""
    enforce_divergence_free: bool = True
    dx: float = 1.0
    dy: float = 1.0
    divergence_tolerance: float = 1e-2
    preserve_wavelet_subbands: bool = False
    temporal_flicker_damping: float = 0.25


class NoetherConservationEngine:
    """PyTorch Solenoidal Projector and Physical Invariant Referee."""

    def __init__(self, config: Optional[NoetherConfig] = None) -> None:
        self.config = config or NoetherConfig()

    def project_divergence_free_2d_torch(
        self,
        vx: torch.Tensor,
        vy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        r"""Project 2D velocity fields (H, W) or (B, H, W) to solenoidal space via Spectral Poisson Solver.

        Solves \nabla^2 \phi = \nabla \cdot v in frequency space, then sets v_sol = v - \nabla \phi.
        Returns:
            vx_sol, vy_sol, rem_div_norm
        """
        orig_ndim = vx.ndim
        if orig_ndim == 2:
            vx = vx.unsqueeze(0)
            vy = vy.unsqueeze(0)

        b, h, w = vx.shape
        device = vx.device
        dtype = vx.dtype
        dx = self.config.dx
        dy = self.config.dy

        # 2D FFT wavenumbers
        kx = 2.0 * torch.pi * torch.fft.fftfreq(w, d=dx, device=device)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(h, d=dy, device=device)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

        # Exact central difference Fourier symbols: s_x = sin(k_x * dx) / dx
        sx = torch.sin(kx_grid * dx) / dx
        sy = torch.sin(ky_grid * dy) / dy
        s_sq = sx ** 2 + sy ** 2
        s_sq[0, 0] = 1.0  # Guard against DC division by zero

        # Spectral Leray projection
        vx_hat = torch.fft.fft2(vx)
        vy_hat = torch.fft.fft2(vy)
        div_hat = sx * vx_hat + sy * vy_hat

        phi_hat = div_hat / s_sq
        phi_hat[:, 0, 0] = 0.0

        vx_sol_hat = vx_hat - sx * phi_hat
        vy_sol_hat = vy_hat - sy * phi_hat

        vx_sol = torch.real(torch.fft.ifft2(vx_sol_hat)).to(dtype)
        vy_sol = torch.real(torch.fft.ifft2(vy_sol_hat)).to(dtype)

        # Residual divergence norm via central differences
        rem_dvx = (torch.roll(vx_sol, -1, dims=2) - torch.roll(vx_sol, 1, dims=2)) / (2.0 * dx)
        rem_dvy = (torch.roll(vy_sol, -1, dims=1) - torch.roll(vy_sol, 1, dims=1)) / (2.0 * dy)
        rem_div_norm = float(torch.sqrt(torch.mean((rem_dvx + rem_dvy) ** 2)).item())

        if orig_ndim == 2:
            return vx_sol.squeeze(0), vy_sol.squeeze(0), rem_div_norm
        return vx_sol, vy_sol, rem_div_norm

    def project_wavelet_temporal_conservation(
        self,
        velocity_field: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        r"""Enforce subband-preserving temporal mass conservation on wavelet latents (T, C, H, W).

        Unlike 2D vector-field Leray projection (which pairs channel 2k and 2k+1 and would
        illegally mix low-pass LL and high-pass LH wavelet subbands), this projector operates
        within each channel independently, damping unphysical frame-to-frame spatial DC mass
        flicker while preserving the learned spatial denoising vector field.
        """
        if velocity_field.ndim != 4:
            raise ValueError(f"Expected 4D tensor (T, C, H, W), got {velocity_field.shape}")

        t, _c, _h, _w = velocity_field.shape
        if t <= 1:
            return velocity_field.clone(), 0.0

        projected = velocity_field.clone()
        # Spatial mean per (frame, channel): shape (T, C, 1, 1)
        frame_means = projected.mean(dim=(-2, -1), keepdim=True)
        # Clip-wide temporal mean per channel: shape (1, C, 1, 1)
        clip_means = frame_means.mean(dim=0, keepdim=True)
        flicker = frame_means - clip_means

        damping = float(self.config.temporal_flicker_damping)
        projected = projected - damping * flicker
        rem_flicker = float(torch.sqrt(torch.mean(((1.0 - damping) * flicker) ** 2)).item())
        return projected, rem_flicker

    def project_spatial_latent_velocity(
        self,
        velocity_field: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        r"""Project 4D spatio-temporal video velocity (T, C, H, W) to physical invariant space.

        - When `config.preserve_wavelet_subbands` is True (used for multi-subband wavelet VAEs
          where channels represent LL, LH, HL, HH frequency bands rather than 2D (vx, vy) pairs),
          enforces subband-preserving temporal mass conservation without cross-channel mixing.
        - Otherwise, pairs channels (2k, 2k+1) into 2D spatial vector fields and projects onto
          the divergence-free solenoidal subspace (\nabla \cdot v = 0).
        """
        if velocity_field.ndim != 4:
            raise ValueError(f"Expected 4D tensor (T, C, H, W), got {velocity_field.shape}")

        if self.config.preserve_wavelet_subbands:
            return self.project_wavelet_temporal_conservation(velocity_field)

        # The optional FFI bridge is inference-only and assumes the default grid.
        defaults = NoetherConfig()
        if (
            _RUST_NOETHER_AVAILABLE
            and apedp_kernel is not None
            and velocity_field.layout == torch.strided
            and velocity_field.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            and all(size > 0 for size in velocity_field.shape)
            and not (torch.is_grad_enabled() and velocity_field.requires_grad)
            and self.config.dx == defaults.dx
            and self.config.dy == defaults.dy
        ):
            try:
                # Isolate caller storage even if the native implementation mutates its input.
                v_cpu = velocity_field.detach().to(
                    device="cpu", dtype=torch.float64, copy=True,
                    memory_format=torch.contiguous_format,
                ).resolve_neg().numpy()
                v_out, div_r = apedp_kernel.jetflow_noether_project(v_cpu)
                v_array = np.asarray(v_out)
                residual = np.asarray(div_r)
                if (
                    v_array.shape != tuple(velocity_field.shape)
                    or v_array.dtype.kind != "f"
                    or not np.isfinite(v_array).all()
                    or residual.ndim != 0
                    or residual.dtype.kind not in "fiu"
                    or not np.isfinite(residual)
                    or residual < 0
                ):
                    raise ValueError("Invalid Rust Noether output shape, dtype, or residual")
                v_out_t = torch.from_numpy(
                    np.require(v_array, dtype=np.float64, requirements=["C", "W"])
                ).to(device=velocity_field.device, dtype=velocity_field.dtype)
                if not torch.isfinite(v_out_t).all():
                    raise ValueError("Rust Noether output overflowed the original dtype")
                return v_out_t, float(residual)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                logger.debug("Rust Noether projection failed; using PyTorch", exc_info=True)
            except Exception:  # noqa: BLE001 -- optional native/array adapters may raise arbitrary errors
                logger.debug("Unexpected Rust Noether failure; using PyTorch", exc_info=True)

        t, c, h, w = velocity_field.shape
        projected = velocity_field.clone()
        total_rem_div = 0.0
        pairs = 0

        for ti in range(t):
            for ci in range(0, c - 1, 2):
                vx = velocity_field[ti, ci]
                vy = velocity_field[ti, ci + 1]
                vx_sol, vy_sol, rem_div = self.project_divergence_free_2d_torch(vx, vy)
                projected[ti, ci] = vx_sol
                projected[ti, ci + 1] = vy_sol
                total_rem_div += rem_div
                pairs += 1

        mean_div = total_rem_div / max(1, pairs)
        return projected, mean_div

    def compute_divergence_loss(self, velocity_field: torch.Tensor) -> torch.Tensor:
        r"""Differentiable physical invariant loss for PyTorch neural training.

        When `preserve_wavelet_subbands` is True, penalizes second-order temporal jitter
        and frame-to-frame spatial mass drift across wavelet channels. Otherwise computes
        2D spatial divergence loss \mathbb{E}_{t, c} [\|\partial_x v_x + \partial_y v_y\|^2].
        """
        v = velocity_field if velocity_field.ndim == 5 else velocity_field.unsqueeze(0)
        b, t, c, h, w = v.shape

        if self.config.preserve_wavelet_subbands:
            if t <= 1:
                return torch.tensor(0.0, device=velocity_field.device, dtype=velocity_field.dtype)
            frame_means = v.mean(dim=(-2, -1))  # (B, T, C)
            temporal_mass_diff = frame_means[:, 1:, :] - frame_means[:, :-1, :]
            return torch.mean(temporal_mass_diff ** 2)

        dx, dy = self.config.dx, self.config.dy

        div_losses = []
        for ci in range(0, c - 1, 2):
            vx = v[:, :, ci, :, :]
            vy = v[:, :, ci + 1, :, :]
            dvx = (torch.roll(vx, -1, dims=-1) - torch.roll(vx, 1, dims=-1)) / (2.0 * dx)
            dvy = (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2)) / (2.0 * dy)
            div = dvx + dvy
            div_losses.append(torch.mean(div ** 2))

        if not div_losses:
            return torch.tensor(0.0, device=velocity_field.device, dtype=velocity_field.dtype)
        return torch.stack(div_losses).mean()

    def evaluate_noether_metrics(
        self,
        video_latent: torch.Tensor,
        velocity_field: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        r"""Compute Noether conservation metrics across spatio-temporal video latents.

        Returns:
            mass_drift: Relative drift in total latent mass \int |z|^2 d\Omega
            kinetic_energy_drift: Relative drift in kinetic energy \int |v|^2 d\Omega
            mean_divergence: Mean spatial divergence norm \|\nabla \cdot v\|
        """
        if video_latent.ndim != 4:
            raise ValueError(f"Expected 4D tensor (T, C, H, W), got {video_latent.shape}")

        t = video_latent.shape[0]
        masses = [float(torch.sum(video_latent[ti] ** 2).item()) for ti in range(t)]
        initial_mass = max(1e-12, masses[0])
        mass_drift = float(max(abs(m - initial_mass) for m in masses) / initial_mass)

        ke_drift = 0.0
        mean_div = 0.0

        if velocity_field is not None and velocity_field.shape == video_latent.shape:
            ke_series = [float((0.5 * torch.sum(velocity_field[ti] ** 2)).item()) for ti in range(t)]
            init_ke = max(1e-12, ke_series[0])
            ke_drift = float(max(abs(k - init_ke) for k in ke_series) / init_ke)

            loss_div = self.compute_divergence_loss(velocity_field)
            mean_div = float(torch.sqrt(loss_div).item())

        return {
            "mass_drift": mass_drift,
            "kinetic_energy_drift": ke_drift,
            "mean_divergence": mean_div,
        }
