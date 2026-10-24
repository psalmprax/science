r"""Audio spectrogram transport: chirp ridge tracking, IF model, Jet2 prediction.

Status: experimental
"""
from __future__ import annotations

import numpy as np

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

SAMPLE_RATE = 8000.0
DURATION = 1.0
WINDOW = 256
HOP = 64


def _analysis(spec: ChirpSpec):
    signal = synthesize_chirps([spec], SAMPLE_RATE, DURATION)
    magnitude, frequencies, times = stft_magnitude(signal, SAMPLE_RATE, WINDOW, HOP)
    ridge = ridge_instantaneous_frequency(magnitude, frequencies)
    model = fit_instantaneous_frequency_model(times, ridge, degree=2)
    return magnitude, frequencies, times, ridge, model


def test_stft_ridge_tracks_linear_chirp_instantaneous_frequency():
    spec = ChirpSpec(start_freq=700.0, linear_rate=400.0)
    _, frequencies, times, ridge, _ = _analysis(spec)
    assert times.size >= 100
    true_if = spec.instantaneous_frequency(times)
    bin_width = float(frequencies[1] - frequencies[0])
    max_error = float(np.max(np.abs(ridge - true_if)))
    assert max_error < bin_width / 5.0, f"ridge max error {max_error} Hz vs bin {bin_width} Hz"


def test_if_model_recovers_velocity_and_acceleration():
    linear = ChirpSpec(start_freq=700.0, linear_rate=400.0)
    _, _frequencies, times, _, linear_model = _analysis(linear)
    mid_time = float(times[times.size // 2])
    velocity_error = abs(float(linear_model.velocity(mid_time)) - linear.linear_rate)
    assert velocity_error < 1.0, velocity_error

    accelerating = ChirpSpec(start_freq=500.0, linear_rate=300.0, quadratic_rate=2000.0)
    _, _, times_q, _, quad_model = _analysis(accelerating)
    mid_time_q = float(times_q[times_q.size // 2])
    expected_velocity = accelerating.linear_rate + accelerating.quadratic_rate * mid_time_q
    assert abs(float(quad_model.velocity(mid_time_q)) - expected_velocity) < 5.0
    acceleration = float(np.polyval(np.polyder(quad_model.coefficients, 2), 0.0))
    relative_error = abs(acceleration - accelerating.quadratic_rate) / accelerating.quadratic_rate
    assert relative_error < 0.05, (acceleration, accelerating.quadratic_rate)


def test_transport_prediction_matches_next_frame():
    spec = ChirpSpec(start_freq=700.0, linear_rate=400.0)
    magnitude, frequencies, times, _, model = _analysis(spec)
    frequency_step = float(frequencies[1] - frequencies[0])
    scores = [
        prediction_r_squared(predict_next_frame(magnitude, k, times, model, frequency_step, order=2), magnitude[:, k + 1])
        for k in range(2, times.size - 2)
    ]
    assert float(np.mean(scores)) > 0.995, float(np.mean(scores))
    assert float(np.min(scores)) > 0.99, float(np.min(scores))


def test_r_squared_zero_variance_target_returns_nan_not_crash():
    """Regression: a constant target made R^2 raise ZeroDivisionError."""
    constant = np.full(64, 0.25)
    assert np.isnan(prediction_r_squared(constant, constant))
    assert np.isnan(prediction_r_squared(np.zeros(64), constant))


def test_accelerated_chirp_transport_is_stable_and_order_invariant():
    """Frequency-axis advection is integrated with a stable upwind scheme.

    A centered Euler/Jet2 attribute-transport stepper is unconditionally
    unstable for pure advection (von Neumann: |G|^2 = 1 + nu^2 sin^2(k df) > 1
    for every wavenumber), so ``predict_next_frame`` integrates S_t = -c S_f
    upwind under an explicit CFL bound and projects magnitudes to their
    physical range S >= 0. ``order`` is therefore interface-compatible only:
    both orders must yield the SAME stable, accurate, non-negative prediction.
    The second-order advantage of the generic stepper is pinned where it is
    still meaningful -- at the stepper level -- in
    ``test_accelerated_ridge_transport_order2_error``.
    """
    spec = ChirpSpec(start_freq=500.0, linear_rate=300.0, quadratic_rate=2000.0)
    magnitude, frequencies, times, _, model = _analysis(spec)
    frequency_step = float(frequencies[1] - frequencies[0])

    def scores_for(order: int) -> np.ndarray:
        return np.array(
            [
                prediction_r_squared(
                    predict_next_frame(magnitude, k, times, model, frequency_step, order=order),
                    magnitude[:, k + 1],
                )
                for k in range(2, times.size - 2)
            ]
        )

    euler = scores_for(1)
    jet2 = scores_for(2)
    assert np.allclose(euler, jet2), "upwind integration must not depend on the order flag"
    assert float(np.mean(jet2)) > 0.98, float(np.mean(jet2))
    assert float(np.min(jet2)) > 0.95, float(np.min(jet2))
    for k in range(2, times.size - 2):
        prediction = predict_next_frame(magnitude, k, times, model, frequency_step, order=2)
        assert float(np.min(prediction)) >= 0.0, "spectral magnitudes must stay non-negative"


def test_accelerated_ridge_transport_order2_error():
    from apedp_dyn.video.jet_flow_stepper import JetFlowConfig, JetFlowStepper

    frequencies = np.linspace(0.0, 4000.0, 512)
    initial = np.exp(-0.5 * ((frequencies - 1000.0) / 150.0) ** 2)
    velocity_model = InstantaneousFrequencyModel(
        coefficients=np.array([900.0, 300.0, 1000.0]), t_origin=0.0
    )
    step = 0.02
    shift = 300.0 * step + 900.0 * step * step
    exact = np.exp(-0.5 * ((frequencies - 1000.0 - shift) / 150.0) ** 2)

    velocity_fn = frequency_advection_velocity_fn(velocity_model, float(frequencies[1] - frequencies[0]))
    errors: dict[int, float] = {}
    for order in (1, 2):
        stepper = JetFlowStepper(JetFlowConfig(num_steps=1, order=order))
        prediction, _, _ = stepper.step(velocity_fn, initial, 0.0, step)
        errors[order] = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))
        assert prediction_r_squared(prediction, exact) > 0.9999
    assert errors[2] < 5e-4, errors
    assert errors[2] < errors[1] / 3.0, errors
