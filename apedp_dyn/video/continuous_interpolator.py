r"""Continuous Temporal Interpolator for Infinite FPS and Zero-Flicker Slow Motion.

Status: experimental

Leverages the continuous mathematical formulation of the Tensor-Train video representation
to render video frames at arbitrary continuous timestamps tau in [0.0, 1.0], enabling:
- 24 FPS -> 60 FPS / 120 FPS / 240 FPS frame rate upsampling.
- Artifact-free slow motion with continuous physical advection.
- Temporal smoothness verification (measuring ||D_t^2 u|| to ensure zero flicker).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from apedp_dyn.video.tensor_train_video import TensorTrainVideo, evaluate_continuous_temporal_frame


class ContinuousTemporalInterpolator:
    """Continuous temporal frame generator for Tensor-Train video representations."""

    def __init__(self, tt_video: TensorTrainVideo) -> None:
        self.tt_video = tt_video

    def sample_at_timestamps(self, timestamps: Sequence[float] | np.ndarray) -> np.ndarray:
        r"""Sample video frames at an arbitrary sequence of continuous timestamps tau_k in [0.0, 1.0].

        Parameters:
        - timestamps: sequence of normalized time coordinates tau in [0.0, 1.0].

        Returns:
        - 4D numpy array of shape (N_out, C, H, W).
        """
        if len(timestamps) == 0:
            raise ValueError("timestamps list cannot be empty")

        frames: list[np.ndarray] = []
        for tau in timestamps:
            frame = evaluate_continuous_temporal_frame(self.tt_video, float(tau))
            frames.append(frame)

        return np.stack(frames, axis=0)

    def resample_to_fps(
        self,
        target_fps: float,
        source_fps: float = 24.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Resample the video to a new target frame rate (e.g. 24 -> 60 or 120 FPS).

        Parameters:
        - target_fps: desired output frame rate.
        - source_fps: original reference frame rate.

        Returns:
        - (interpolated_video_4d, timestamps_array)
        """
        if target_fps <= 0 or source_fps <= 0:
            raise ValueError("Target and source FPS must be positive numbers")

        num_orig_frames = self.tt_video.shape[0]
        duration_sec = (num_orig_frames - 1) / source_fps if num_orig_frames > 1 else 1.0
        num_target_frames = max(2, round(duration_sec * target_fps) + 1)

        taus = np.linspace(0.0, 1.0, num_target_frames)
        interpolated = self.sample_at_timestamps(taus)
        return interpolated, taus

    def render_slow_motion(self, slowdown_factor: float = 4.0) -> np.ndarray:
        r"""Render smooth slow motion by a specified factor (e.g. 4.0x or 8.0x slowdown).

        Parameters:
        - slowdown_factor: float > 1.0 specifying expansion factor.

        Returns:
        - 4D numpy array of shape (N_slow, C, H, W).
        """
        if slowdown_factor <= 1.0:
            raise ValueError("slowdown_factor must be > 1.0")

        num_orig_frames = self.tt_video.shape[0]
        n_slow_frames = round((num_orig_frames - 1) * slowdown_factor) + 1
        taus = np.linspace(0.0, 1.0, n_slow_frames)
        return self.sample_at_timestamps(taus)

    @staticmethod
    def measure_temporal_smoothness(video_4d: np.ndarray) -> float:
        r"""Measure temporal flickering / acceleration: ||D_t^2 u||_2.

        A smooth continuous video has low acceleration between consecutive frames,
        whereas flickering videos show massive high-frequency oscillations.

        Returns:
        - Mean squared second temporal derivative across all pixels and channels.
        """
        arr = np.asarray(video_4d, dtype=np.float64)
        if arr.ndim != 4 or arr.shape[0] < 3:
            return 0.0

        # Discrete second derivative along time axis: u_{t+1} - 2 u_t + u_{t-1}
        d2 = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
        return float(np.mean(d2 ** 2))
