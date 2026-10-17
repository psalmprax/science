r"""Audio Field-Transport Modules.

Status: experimental
"""
from __future__ import annotations

from apedp_dyn.audio.spectrogram_transport import (
    ChirpSpec,
    InstantaneousFrequencyModel,
    fit_instantaneous_frequency_model,
    frequency_advection_velocity_fn,
    predict_next_frame,
    prediction_r_squared,
    ridge_instantaneous_frequency,
    stft_magnitude,
    synthesize_chirps,
)

__all__ = [
    "ChirpSpec",
    "InstantaneousFrequencyModel",
    "fit_instantaneous_frequency_model",
    "frequency_advection_velocity_fn",
    "predict_next_frame",
    "prediction_r_squared",
    "ridge_instantaneous_frequency",
    "stft_magnitude",
    "synthesize_chirps",
]
