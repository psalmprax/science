r"""4D Spatio-Temporal Tensor-Train (TT / MPS) Latent Video Representation.

Status: experimental

Factorizes 4D continuous video latent tensors (Time, Channels, Height, Width)
into low-rank matrix product state (MPS / Tensor-Train) cores:
  u(t, c, h, w) = G^(0)(t) * G^(1)(c) * G^(2)(h) * G^(3)(w)
reducing memory footprint by 10x-100x while enabling continuous-time querying
at arbitrary frame rates without unrolling the full dense 4D array.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class TensorTrainVideo:
    """Compact 4D Tensor-Train representation of a video latent tensor."""
    # Cores:
    # core_t: (1, T, R1)
    # core_c: (R1, C, R2)
    # core_h: (R2, H, R3)
    # core_w: (R3, W, 1)
    core_t: np.ndarray
    core_c: np.ndarray
    core_h: np.ndarray
    core_w: np.ndarray
    shape: tuple[int, int, int, int]  # (T, C, H, W)
    bond_ranks: tuple[int, int, int]  # (R1, R2, R3)

    @property
    def dense_elements(self) -> int:
        """Total number of elements in uncompressed dense tensor."""
        t, c, h, w = self.shape
        return t * c * h * w

    @property
    def compressed_elements(self) -> int:
        """Total number of elements stored across TT cores."""
        return int(
            self.core_t.size + self.core_c.size + self.core_h.size + self.core_w.size
        )

    @property
    def compression_ratio(self) -> float:
        """Ratio of dense elements to compressed TT elements (e.g. 15.0 = 15x smaller)."""
        return float(self.dense_elements) / max(1, self.compressed_elements)

    @property
    def memory_saving_pct(self) -> float:
        """Percentage of memory saved (e.g. 93.5%)."""
        return (1.0 - (self.compressed_elements / self.dense_elements)) * 100.0


def compress_video_latent(
    latent: np.ndarray,
    max_rank: int = 16,
    tolerance: float = 1e-4,
) -> TensorTrainVideo:
    r"""Compress a 4D video latent tensor into a 4-core Tensor-Train (TT-SVD).

    Parameters:
    - latent: 4D numpy array of shape (T, C, H, W).
    - max_rank: Maximum bond rank across cores R_k <= max_rank.
    - tolerance: Relative truncation threshold for singular values.

    Returns:
    - TensorTrainVideo instance storing factorized cores.
    """
    arr = np.ascontiguousarray(latent, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (T, C, H, W), got {arr.ndim}D with shape {arr.shape}")

    t_dim, c_dim, h_dim, w_dim = arr.shape
    if min(t_dim, c_dim, h_dim, w_dim) < 1:
        raise ValueError("All dimensions of video latent must be >= 1")

    # Step 1: Unfold along axis 0 (Time) -> shape (T, C * H * W)
    mat0 = arr.reshape(t_dim, c_dim * h_dim * w_dim)
    u0, s0, vt0 = np.linalg.svd(mat0, full_matrices=False)

    # Determine rank R1
    thresh0 = tolerance * (s0[0] if len(s0) > 0 else 1.0)
    r1 = min(max_rank, int((s0 > thresh0).sum()))
    r1 = max(1, r1)

    core_t = u0[:, :r1].reshape(1, t_dim, r1)  # (1, T, R1)
    rem1 = (np.diag(s0[:r1]) @ vt0[:r1, :]).reshape(r1 * c_dim, h_dim * w_dim)

    # Step 2: Unfold along axis 1 (Channels) -> shape (R1 * C, H * W)
    u1, s1, vt1 = np.linalg.svd(rem1, full_matrices=False)
    thresh1 = tolerance * (s1[0] if len(s1) > 0 else 1.0)
    r2 = min(max_rank, int((s1 > thresh1).sum()))
    r2 = max(1, r2)

    core_c = u1[:, :r2].reshape(r1, c_dim, r2)  # (R1, C, R2)
    rem2 = (np.diag(s1[:r2]) @ vt1[:r2, :]).reshape(r2 * h_dim, w_dim)

    # Step 3: Unfold along axis 2 (Height) -> shape (R2 * H, W)
    u2, s2, vt2 = np.linalg.svd(rem2, full_matrices=False)
    thresh2 = tolerance * (s2[0] if len(s2) > 0 else 1.0)
    r3 = min(max_rank, int((s2 > thresh2).sum()))
    r3 = max(1, r3)

    core_h = u2[:, :r3].reshape(r2, h_dim, r3)  # (R2, H, R3)
    core_w = (np.diag(s2[:r3]) @ vt2[:r3, :]).reshape(r3, w_dim, 1)  # (R3, W, 1)

    return TensorTrainVideo(
        core_t=core_t,
        core_c=core_c,
        core_h=core_h,
        core_w=core_w,
        shape=(t_dim, c_dim, h_dim, w_dim),
        bond_ranks=(r1, r2, r3),
    )


def decompress_video_latent(tt_video: TensorTrainVideo) -> np.ndarray:
    r"""Decompress the Tensor-Train cores back into the dense 4D array (T, C, H, W)."""
    # G^(0): (1, T, R1) -> (T, R1)
    g0 = tt_video.core_t.reshape(tt_video.shape[0], tt_video.bond_ranks[0])
    # G^(1): (R1, C, R2)
    g1 = tt_video.core_c
    # G^(2): (R2, H, R3)
    g2 = tt_video.core_h
    # G^(3): (R3, W, 1) -> (R3, W)
    g3 = tt_video.core_w.reshape(tt_video.bond_ranks[2], tt_video.shape[3])

    # Efficient sequential contraction:
    # 1. contract G^(0) with G^(1): (T, R1) x (R1, C, R2) -> (T, C, R2)
    tc_r2 = np.tensordot(g0, g1, axes=([1], [0]))  # (T, C, R2)

    # 2. contract with G^(2): (T, C, R2) x (R2, H, R3) -> (T, C, H, R3)
    tch_r3 = np.tensordot(tc_r2, g2, axes=([2], [0]))  # (T, C, H, R3)

    # 3. contract with G^(3): (T, C, H, R3) x (R3, W) -> (T, C, H, W)
    dense = np.tensordot(tch_r3, g3, axes=([3], [0]))  # (T, C, H, W)
    return dense


def evaluate_continuous_temporal_frame(
    tt_video: TensorTrainVideo,
    tau: float,
) -> np.ndarray:
    r"""Evaluate a single video frame at an arbitrary continuous time coordinate tau in [0.0, 1.0].

    Instead of integer frames, uses cubic spline interpolation on the temporal core G^(0)
    and contracts directly with (C, H, W) cores in O(R^2 C + R^2 H + R W) operations,
    without building the full 4D video tensor in memory.

    Returns:
    - 3D array of shape (C, H, W) corresponding to continuous frame at tau.
    """
    t_dim, _c_dim, _h_dim, w_dim = tt_video.shape
    r1, _r2, r3 = tt_video.bond_ranks

    # Temporal knots in normalized time [0.0, 1.0]
    t_knots = np.linspace(0.0, 1.0, t_dim)
    # Extract temporal matrix of shape (T, R1)
    g0_mat = tt_video.core_t.reshape(t_dim, r1)

    # Fit cubic spline across each bond rank column
    tau_clamped = float(np.clip(tau, 0.0, 1.0))
    if t_dim >= 4:
        spline = CubicSpline(t_knots, g0_mat, axis=0, bc_type="natural")
        g0_tau = spline(tau_clamped)  # shape (R1,)
    else:
        # Linear fallback for short sequences
        g0_tau = np.zeros(r1, dtype=np.float64)
        for k in range(r1):
            g0_tau[k] = np.interp(tau_clamped, t_knots, g0_mat[:, k])

    # Contract g0_tau with remaining spatial cores:
    # g0_tau (R1,) x G^(1) (R1, C, R2) -> (C, R2)
    c_r2 = np.tensordot(g0_tau, tt_video.core_c, axes=([0], [0]))  # (C, R2)
    # (C, R2) x G^(2) (R2, H, R3) -> (C, H, R3)
    ch_r3 = np.tensordot(c_r2, tt_video.core_h, axes=([1], [0]))  # (C, H, R3)
    # (C, H, R3) x G^(3) (R3, W, 1) -> (C, H, W)
    g3 = tt_video.core_w.reshape(r3, w_dim)
    frame = np.tensordot(ch_r3, g3, axes=([2], [0]))  # (C, H, W)
    return frame
