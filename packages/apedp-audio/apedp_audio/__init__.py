"""APEDP-Audio: Autonomous Audio Spectrogram Field Transport & Ridge Tracking."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package can be imported if running within monorepo
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

__version__ = "0.1.0"

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
