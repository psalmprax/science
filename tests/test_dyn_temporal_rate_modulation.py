"""Pins for the Hypothesis B temporal-rate modulation null test.

The tests cover the scientific contract, not just the code path: a null run
must stay null, a planted common-mode injection must be recovered, an
oscillator-specific injection must be rejected as heterogeneous, and the
detector must meet its preregistered false-positive and power floors.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from apedp_dyn.conf.canonical_spec import (
    TEMPORAL_RATE_CALIBRATION_FPR_MAX,
    TEMPORAL_RATE_CALIBRATION_POWER_MIN,
    TEMPORAL_RATE_FAMILYWISE_ALPHA,
    TEMPORAL_RATE_HARMONICS,
    TEMPORAL_RATE_HETEROGENEITY_ALPHA,
    TEMPORAL_RATE_MIN_CALIBRATION_SNR,
    TEMPORAL_RATE_MIN_CLOCKS,
    TEMPORAL_RATE_MIN_COMPOSITIONS,
    TEMPORAL_RATE_MIN_DRIVE_CYCLES,
    TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE,
    TEMPORAL_RATE_SURROGATE_DRAWS,
    load_and_verify_canonical_spec,
)
from apedp_dyn.epistemic.temporal_rate_modulation import (
    CLAIM_BOUNDARY,
    ClockChannel,
    ClockDataset,
    TemporalRateProtocolError,
    TemporalRateVerdict,
    adjudicate_temporal_rate_modulation,
    run_detector_calibration,
    simulate_clock_dataset,
)

CALIBRATION_FREQUENCY_HZ = 0.83
CONFIRMATION_FREQUENCY_HZ = 1.0
DURATION_S = 128.0
SAMPLE_RATE_HZ = 16.0
N_SURROGATES = 256


def _calibration(**overrides):
    kwargs = {
        "dataset_id": "calibration",
        "drive_frequency_hz": CALIBRATION_FREQUENCY_HZ,
        "duration_s": DURATION_S,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "seed": 1,
    }
    kwargs.update(overrides)
    return simulate_clock_dataset(**kwargs)


def _confirmation(dataset_id="confirmation", seed=2, **overrides):
    kwargs = {
        "dataset_id": dataset_id,
        "drive_frequency_hz": CONFIRMATION_FREQUENCY_HZ,
        "duration_s": DURATION_S,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "seed": seed,
    }
    kwargs.update(overrides)
    return simulate_clock_dataset(**kwargs)


@pytest.fixture(scope="module")
def calibration():
    return _calibration()


def test_thresholds_are_preregistered_and_bound(calibration):
    spec = load_and_verify_canonical_spec()
    thresholds = spec["thresholds"]
    assert thresholds["temporal_rate_min_clocks"] == TEMPORAL_RATE_MIN_CLOCKS == 2
    assert (
        thresholds["temporal_rate_min_compositions"]
        == TEMPORAL_RATE_MIN_COMPOSITIONS
        == 2
    )
    assert thresholds["temporal_rate_min_drive_cycles"] == TEMPORAL_RATE_MIN_DRIVE_CYCLES
    assert thresholds["temporal_rate_harmonics"] == TEMPORAL_RATE_HARMONICS
    assert thresholds["temporal_rate_surrogate_draws"] == TEMPORAL_RATE_SURROGATE_DRAWS
    assert (
        thresholds["temporal_rate_familywise_alpha"]
        == TEMPORAL_RATE_FAMILYWISE_ALPHA
        == 0.01
    )
    assert (
        thresholds["temporal_rate_heterogeneity_alpha"]
        == TEMPORAL_RATE_HETEROGENEITY_ALPHA
        == 0.01
    )
    assert (
        thresholds["temporal_rate_min_fractional_amplitude"]
        == TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE
        == 1e-18
    )
    assert (
        thresholds["temporal_rate_min_calibration_snr"]
        == TEMPORAL_RATE_MIN_CALIBRATION_SNR
        == 10.0
    )
    assert (
        thresholds["temporal_rate_calibration_fpr_max"]
        == TEMPORAL_RATE_CALIBRATION_FPR_MAX
        == 0.05
    )
    assert (
        thresholds["temporal_rate_calibration_power_min"]
        == TEMPORAL_RATE_CALIBRATION_POWER_MIN
        == 0.8
    )

    signature = inspect.signature(adjudicate_temporal_rate_modulation)
    assert signature.parameters["harmonics"].default is None
    assert signature.parameters["n_surrogates"].default is None
    assert calibration.channels[0].composition != calibration.channels[1].composition


def test_null_run_is_null_consistent(calibration):
    result = adjudicate_temporal_rate_modulation(
        calibration,
        _confirmation(seed=2),
        n_surrogates=N_SURROGATES,
        seed=10,
    )
    assert result.verdict is TemporalRateVerdict.NULL_CONSISTENT
    assert result.p_value > TEMPORAL_RATE_FAMILYWISE_ALPHA
    assert result.n_cycles >= TEMPORAL_RATE_MIN_DRIVE_CYCLES


def test_common_mode_injection_is_recovered(calibration):
    result = adjudicate_temporal_rate_modulation(
        calibration,
        _confirmation(
            dataset_id="common-mode",
            seed=3,
            anomaly_amplitude=1e-15,
            anomaly_phase_rad=1.1,
        ),
        n_surrogates=N_SURROGATES,
        seed=10,
    )
    assert result.verdict is TemporalRateVerdict.COMMON_MODE_ANOMALY_CANDIDATE
    assert result.p_value < TEMPORAL_RATE_FAMILYWISE_ALPHA
    assert result.heterogeneity_p_value >= TEMPORAL_RATE_HETEROGENEITY_ALPHA
    assert result.best_harmonic == 1
    assert result.common_mode_amplitude == pytest.approx(1e-15, rel=0.05)
    assert result.calibration_dataset_id == "calibration"
    assert result.confirmation_dataset_id == "common-mode"
    assert result.harmonics == TEMPORAL_RATE_HARMONICS
    assert len(result.canonical_spec_sha256) == 64
    assert set(result.calibration_snr) == {"clock_0", "clock_1"}
    assert set(result.calibration_harmonic_snr) == {"clock_0", "clock_1"}
    assert all(
        len(values) == TEMPORAL_RATE_HARMONICS
        for values in result.calibration_harmonic_snr.values()
    )
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert "backward time travel" in result.claim_boundary


def test_oscillator_specific_injection_is_rejected(calibration):
    result = adjudicate_temporal_rate_modulation(
        calibration,
        _confirmation(
            dataset_id="oscillator-specific",
            seed=4,
            anomaly_amplitude=1e-15,
            anomaly_common_mode=False,
            anomaly_clock_index=0,
        ),
        n_surrogates=N_SURROGATES,
        seed=10,
    )
    assert result.verdict is TemporalRateVerdict.OSCILLATOR_SPECIFIC_ARTIFACT
    assert result.heterogeneity_p_value < TEMPORAL_RATE_HETEROGENEITY_ALPHA
    assert len(result.per_clock_amplitudes) == 2


def test_calibration_and_confirmation_must_be_disjoint(calibration):
    with pytest.raises(TemporalRateProtocolError, match="dataset_id"):
        adjudicate_temporal_rate_modulation(
            calibration, calibration, n_surrogates=32
        )

    same_frequency = _confirmation(
        dataset_id="same-frequency",
        drive_frequency_hz=CALIBRATION_FREQUENCY_HZ,
    )
    with pytest.raises(TemporalRateProtocolError, match="same drive frequency"):
        adjudicate_temporal_rate_modulation(
            calibration, same_frequency, n_surrogates=32
        )

    reused = ClockDataset(
        dataset_id="reused",
        time_s=calibration.time_s,
        drive=calibration.drive,
        channels=calibration.channels,
        drive_frequency_hz=0.9 * calibration.drive_frequency_hz,
    )
    with pytest.raises(TemporalRateProtocolError, match="reused"):
        adjudicate_temporal_rate_modulation(calibration, reused, n_surrogates=32)


def test_insufficient_cycles_is_refused(calibration):
    short = _confirmation(
        dataset_id="short", duration_s=10.0, sample_rate_hz=SAMPLE_RATE_HZ
    )
    result = adjudicate_temporal_rate_modulation(
        calibration, short, n_surrogates=32
    )
    assert result.verdict is TemporalRateVerdict.INSUFFICIENT_EVIDENCE
    assert "below the preregistered minimum" in result.rational_justification


def test_single_composition_cannot_support_a_common_mode_claim():
    calibration = _calibration(compositions=("Sr", "Sr"))
    confirmation = _confirmation(
        dataset_id="single-composition", compositions=("Sr", "Sr")
    )
    result = adjudicate_temporal_rate_modulation(
        calibration, confirmation, n_surrogates=32
    )
    assert result.verdict is TemporalRateVerdict.INSUFFICIENT_EVIDENCE
    assert "physically distinct clock compositions" in result.rational_justification


def test_single_clock_cannot_support_a_common_mode_claim():
    calibration = _calibration(
        compositions=("Sr",), susceptibilities=(1e-15 + 0j,)
    )
    confirmation = _confirmation(
        dataset_id="single-clock",
        compositions=("Sr",),
        susceptibilities=(1e-15 + 0j,),
    )
    result = adjudicate_temporal_rate_modulation(
        calibration, confirmation, n_surrogates=32
    )
    assert result.verdict is TemporalRateVerdict.INSUFFICIENT_EVIDENCE
    assert "requires at least" in result.rational_justification


def test_weak_calibration_is_refused():
    calibration = _calibration(susceptibilities=(0.0, 0.0))
    confirmation = _confirmation(
        dataset_id="weak-calibration", susceptibilities=(0.0, 0.0)
    )
    result = adjudicate_temporal_rate_modulation(
        calibration, confirmation, n_surrogates=32
    )
    assert result.verdict is TemporalRateVerdict.INSUFFICIENT_EVIDENCE
    assert "minimum calibration SNR" in result.rational_justification


def test_calibration_transfer_has_no_drive_phasor_bias():
    from apedp_dyn.epistemic.temporal_rate_modulation import (
        _calibrate_clocks,
    )

    true = (2e-15 + 1e-15j, -1e-15 + 3e-15j)
    with_drive = _calibration(noise_sigma=0.0, seed=19)
    models = _calibrate_clocks(with_drive, TEMPORAL_RATE_HARMONICS)
    for channel, expected in zip(with_drive.channels, true, strict=True):
        recovered = models[channel.clock_id].chi[0]
        assert recovered.real == pytest.approx(expected.real, rel=1e-10)
        assert recovered.imag == pytest.approx(expected.imag, rel=1e-10)

    time_s = np.arange(2048, dtype=float) / SAMPLE_RATE_HZ
    theta = 2.0 * np.pi * CALIBRATION_FREQUENCY_HZ * time_s
    phase = 0.7
    expected = true[0]
    phased = ClockDataset(
        dataset_id="phased-calibration",
        time_s=time_s,
        drive=np.cos(theta + phase),
        drive_frequency_hz=CALIBRATION_FREQUENCY_HZ,
        channels=(
            ClockChannel(
                clock_id="clock_0",
                composition="Sr optical lattice",
                fractional_frequency=np.real(
                    expected * np.exp(1j * (theta + phase))
                ),
            ),
        ),
    )
    phased_model = _calibrate_clocks(phased, TEMPORAL_RATE_HARMONICS)["clock_0"]
    recovered = phased_model.chi[0]
    assert recovered.real == pytest.approx(expected.real, rel=1e-10)
    assert recovered.imag == pytest.approx(expected.imag, rel=1e-10)


def test_deterministic_replay(calibration):
    confirmation = _confirmation(
        dataset_id="replay", seed=3, anomaly_amplitude=1e-15
    )
    first = adjudicate_temporal_rate_modulation(
        calibration, confirmation, n_surrogates=64, seed=99
    )
    second = adjudicate_temporal_rate_modulation(
        calibration, confirmation, n_surrogates=64, seed=99
    )
    assert first.verdict is second.verdict
    assert first.statistic == second.statistic
    assert first.p_value == second.p_value
    assert first.heterogeneity_p_value == second.heterogeneity_p_value


@pytest.mark.slow
def test_detector_calibration_meets_fpr_and_power_floors():
    from apedp_dyn.conf.canonical_spec import TEMPORAL_RATE_CALIBRATION_TRIALS

    result = run_detector_calibration(
        n_trials=TEMPORAL_RATE_CALIBRATION_TRIALS,
        n_surrogates=128,
        seed=6,
        duration_s=DURATION_S,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    assert result.n_calibration_draws == result.n_null_trials
    assert result.false_positive_rate <= result.fpr_max
    assert result.detection_power >= result.power_min
    assert 0.0 <= result.false_positive_rate_upper_95 <= 1.0
    assert 0.0 <= result.detection_power_lower_95 <= 1.0
    assert result.n_null_insufficient == 0
    assert result.n_injection_insufficient == 0
    assert result.passed


def test_simulated_channels_are_finite_and_seeded():
    first = _confirmation(seed=7)
    second = _confirmation(seed=7)
    for left, right in zip(first.channels, second.channels, strict=True):
        assert np.all(np.isfinite(left.fractional_frequency))
        assert np.array_equal(
            left.fractional_frequency, right.fractional_frequency
        )


def test_clock_channel_rejects_invalid_metadata():
    values = np.zeros(8)
    with pytest.raises(TemporalRateProtocolError, match="non-empty"):
        ClockChannel(
            clock_id=" ", composition="Sr", fractional_frequency=values
        )
    with pytest.raises(TemporalRateProtocolError, match="empty composition"):
        ClockChannel(
            clock_id="clock_0", composition=" ", fractional_frequency=values
        )
    with pytest.raises(TemporalRateProtocolError, match="systematic"):
        ClockChannel(
            clock_id="clock_0",
            composition="Sr",
            fractional_frequency=values,
            systematic_susceptibility_sigma=float("nan"),
        )


def test_clock_dataset_rejects_invalid_identity_and_duplicate_clocks():
    values = np.zeros(8)
    time_s = np.arange(8, dtype=float)
    channel = ClockChannel(
        clock_id="clock_0", composition="Sr", fractional_frequency=values
    )
    with pytest.raises(TemporalRateProtocolError, match="dataset_id"):
        ClockDataset(
            dataset_id=" ",
            time_s=time_s,
            drive=np.ones(8),
            channels=(channel,),
            drive_frequency_hz=1.0,
        )
    with pytest.raises(TemporalRateProtocolError, match="duplicate clock ids"):
        ClockDataset(
            dataset_id="duplicate",
            time_s=time_s,
            drive=np.ones(8),
            channels=(channel, channel),
            drive_frequency_hz=1.0,
        )


def test_clock_dataset_rejects_non_uniform_sampling():
    time_s = np.array([0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.0])
    channel = ClockChannel(
        clock_id="clock_0",
        composition="Sr",
        fractional_frequency=np.zeros(8),
    )
    with pytest.raises(TemporalRateProtocolError, match="uniformly sampled"):
        ClockDataset(
            dataset_id="non-uniform",
            time_s=time_s,
            drive=np.ones(8),
            channels=(channel,),
            drive_frequency_hz=1.0,
        )


def test_calibration_and_confirmation_harmonic_sets_must_be_disjoint(calibration):
    overlapping = _confirmation(
        dataset_id="harmonic-overlap",
        drive_frequency_hz=2.0 * CALIBRATION_FREQUENCY_HZ,
    )
    with pytest.raises(TemporalRateProtocolError, match="harmonic sets overlap"):
        adjudicate_temporal_rate_modulation(
            calibration, overlapping, n_surrogates=32
        )


def _single_harmonic_dataset(
    *,
    dataset_id: str,
    drive_frequency_hz: float,
    seed: int,
) -> ClockDataset:
    n_samples = round(DURATION_S * SAMPLE_RATE_HZ)
    time_s = np.arange(n_samples, dtype=float) / SAMPLE_RATE_HZ
    theta = 2.0 * np.pi * drive_frequency_hz * time_s
    rng = np.random.default_rng(seed)
    channels = tuple(
        ClockChannel(
            clock_id=f"clock_{index}",
            composition=composition,
            fractional_frequency=(
                2e-15 * np.cos(theta + 0.3 * index)
                + 1e-16 * rng.normal(size=n_samples)
            ),
        )
        for index, composition in enumerate(("Sr", "Yb+"))
    )
    return ClockDataset(
        dataset_id=dataset_id,
        time_s=time_s,
        drive=np.cos(theta),
        channels=channels,
        drive_frequency_hz=drive_frequency_hz,
    )


def test_calibration_reports_per_harmonic_snr():
    calibration = _single_harmonic_dataset(
        dataset_id="fundamental-only-calibration",
        drive_frequency_hz=CALIBRATION_FREQUENCY_HZ,
        seed=11,
    )
    confirmation = _single_harmonic_dataset(
        dataset_id="fundamental-only-confirmation",
        drive_frequency_hz=CONFIRMATION_FREQUENCY_HZ,
        seed=12,
    )
    result = adjudicate_temporal_rate_modulation(
        calibration, confirmation, n_surrogates=32
    )
    assert result.verdict is not TemporalRateVerdict.INSUFFICIENT_EVIDENCE
    for values in result.calibration_harmonic_snr.values():
        assert len(values) == TEMPORAL_RATE_HARMONICS
        assert values[0] >= TEMPORAL_RATE_MIN_CALIBRATION_SNR
        assert values[1] < TEMPORAL_RATE_MIN_CALIBRATION_SNR
        assert values[2] < TEMPORAL_RATE_MIN_CALIBRATION_SNR
