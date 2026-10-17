r"""Audio Spectrogram Field Transport: chirp ridge transport and Jet2 prediction.

Status: experimental

Scope & honesty ceiling:
- Applies the repository's continuous spatiotemporal transport primitive
  (``apedp_dyn.video.jet_flow_stepper``) to *audio*: the magnitude spectrogram is
  treated as a 1D field along the frequency axis whose energy is advected by the
  instantaneous-frequency velocity ``c(t) = df_r/dt`` of the dominant chirp.
- Everything here is verified on *synthetic* chirps with analytically known
  instantaneous-frequency laws. No claim is made about music, speech, source
  separation, or codec-grade reconstruction. Real-audio evaluation would require
  a separate preregistered experiment.
- The ridge tracker uses parabolic sub-bin interpolation on a hand-rolled STFT;
  the transport velocity field is estimated from the tracked ridge series, not
  from generator internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ChirpSpec:
    """Chirp with ``f(t) = start_freq + linear_rate * t + 0.5 * quadratic_rate * t**2``."""

    start_freq: float
    linear_rate: float
    quadratic_rate: float = 0.0
    amplitude: float = 1.0
    phase: float = 0.0

    def instantaneous_frequency(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.start_freq + self.linear_rate * t + 0.5 * self.quadratic_rate * t * t

    def phase_at(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return (
            _TWO_PI
            * (
                self.start_freq * t
                + 0.5 * self.linear_rate * t * t
                + (1.0 / 6.0) * self.quadratic_rate * t * t * t
            )
            + self.phase
        )


def synthesize_chirps(
    specs: Sequence[ChirpSpec],
    sample_rate: float,
    duration: float,
) -> np.ndarray:
    """Sum of phase-continuous chirps sampled on ``[0, duration)``."""
    n_samples = int(round(sample_rate * duration))
    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)
    for spec in specs:
        signal += spec.amplitude * np.sin(spec.phase_at(t))
    return signal


def stft_magnitude(
    signal: np.ndarray,
    sample_rate: float,
    window_size: int = 256,
    hop: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann-window magnitude spectrogram ``(n_freq, n_frames)``.

    Returns ``(magnitude, frequencies_hz, frame_times_s)``; frame times mark the
    window centres.
    """
    if window_size % 2 != 0:
        raise ValueError("window_size must be even")
    if hop <= 0:
        raise ValueError("hop must be positive")
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < window_size:
        raise ValueError("signal shorter than one window")

    window = np.hanning(window_size)
    n_frames = 1 + (signal.size - window_size) // hop
    magnitude = np.empty((window_size // 2 + 1, n_frames), dtype=np.float64)
    for frame in range(n_frames):
        start = frame * hop
        segment = signal[start : start + window_size] * window
        magnitude[:, frame] = np.abs(np.fft.rfft(segment))

    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    times = (np.arange(n_frames) * hop + window_size / 2.0) / sample_rate
    return magnitude, freqs, times


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-15:
        return float(index)
    return float(index) + 0.5 * (left - right) / denominator


def ridge_instantaneous_frequency(magnitude: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Sub-bin dominant frequency per frame via parabolic ridge interpolation."""
    magnitude = np.asarray(magnitude, dtype=np.float64)
    n_freq, n_frames = magnitude.shape
    bin_width = float(frequencies[1] - frequencies[0])
    ridge = np.empty(n_frames, dtype=np.float64)
    for frame in range(n_frames):
        peak = int(np.argmax(magnitude[:, frame]))
        refined = _parabolic_peak(magnitude[:, frame], peak)
        ridge[frame] = frequencies[0] + refined * bin_width
    return ridge


@dataclass
class InstantaneousFrequencyModel:
    """Quadratic-in-time fit of a tracked ridge, with analytic velocity ``df/dt``."""

    coefficients: np.ndarray
    t_origin: float

    def frequency(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t, dtype=np.float64) - self.t_origin
        return np.polyval(self.coefficients, t)

    def velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t, dtype=np.float64) - self.t_origin
        return np.polyval(np.polyder(self.coefficients), t)


def fit_instantaneous_frequency_model(
    times: np.ndarray,
    ridge_frequencies: np.ndarray,
    degree: int = 2,
) -> InstantaneousFrequencyModel:
    """Least-squares polynomial IF model on the tracked ridge."""
    times = np.asarray(times, dtype=np.float64)
    origin = float(times[0])
    coefficients = np.polyfit(times - origin, np.asarray(ridge_frequencies, dtype=np.float64), degree)
    return InstantaneousFrequencyModel(coefficients=coefficients, t_origin=origin)


def frequency_advection_velocity_fn(
    model: InstantaneousFrequencyModel,
    frequency_step: float,
):
    """Velocity field ``-c(t) * d/d_f`` for spectrogram transport via JetFlowStepper."""

    def velocity_fn(z: np.ndarray, t: float) -> np.ndarray:
        gradient = np.gradient(np.asarray(z, dtype=np.float64), frequency_step)
        return -float(model.velocity(t)) * gradient

    return velocity_fn


def predict_next_frame(
    magnitude: np.ndarray,
    frame_index: int,
    times: np.ndarray,
    model: InstantaneousFrequencyModel,
    frequency_step: float,
    order: int = 2,
):
    """Transport-predict frame ``frame_index + 1`` from ``frame_index``.

    Wraps ``apedp_dyn.video.jet_flow_stepper.JetFlowStepper`` with the ridge
    velocity field; ``order=1`` is Euler, ``order=2`` is the Jet2 Taylor step.
    """
    from apedp_dyn.video.jet_flow_stepper import JetFlowConfig, JetFlowStepper

    config = JetFlowConfig(num_steps=1, order=order)
    stepper = JetFlowStepper(config)
    dt = float(times[frame_index + 1] - times[frame_index])
    velocity_fn = frequency_advection_velocity_fn(model, frequency_step)
    prediction, _, _ = stepper.step(velocity_fn, magnitude[:, frame_index], float(times[frame_index]), dt)
    return prediction


def prediction_r_squared(prediction: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination of a predicted frame against the true frame."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    residual = float(np.sum((prediction - target) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total
