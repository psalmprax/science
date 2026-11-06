"""Hypothesis B: temporal-rate modulation null test.

Status: experimental

This module adjudicates one preregistered question. After subtracting each
clock's ordinary, calibration-measured response to a periodic drive, is there
a phase-locked fractional-frequency modulation that is *common* to clocks of
at least two different physical compositions?

The discriminator is intentionally narrow:

* A genuine modulation of the rate at which time is measured would appear in
  every clock coupled to the same field, with the same sign and phase. That is
  the ``COMMON_MODE_ANOMALY_CANDIDATE`` verdict.
* A modulation confined to one oscillator, or one with inconsistent amplitude
  or phase across clocks, is an ``OSCILLATOR_SPECIFIC_ARTIFACT``.
* No resolved phase-locked residual is ``NULL_CONSISTENT``.

The calibration and confirmation runs must be distinct, and they must use
different drive frequencies, so that the ordinary susceptibility fit cannot
absorb the very effect being tested. Dataset ids are the holdout key: reusing one across calibration and
confirmation raises ``TemporalRateProtocolError``. The fundamental and
harmonic frequencies of the two runs must also be spectrally resolvable and
disjoint; otherwise the calibration can absorb a confirmation harmonic.

The analysis assumes a uniformly sampled time axis and a drive that is a pure
sinusoid at ``drive_frequency_hz``. Non-uniformly sampled runs, non-sinusoidal
drives, and non-stationary drive waveforms are outside the supported protocol.

Claim boundary: even a common-mode candidate is only an anomalous clock-rate
modulation at the tested drive harmonics. It is not evidence of backward time
travel, closed timelike curves, or a non-metric "time". A candidate can also
be produced by an unmodeled common-mode systematic. Promotion requires
independent replication with dissimilar clocks, a bounded systematics budget,
and a physical coupling model. The simulation helpers in this module calibrate
and falsify the detector; they are not measurements of nature.

Preregistered thresholds are imported from
``apedp_dyn.conf.canonical_spec``; no experiment threshold is defined here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.stats import beta

from apedp_dyn.conf.canonical_spec import (
    PINNED_SPEC_SHA256,
    TEMPORAL_RATE_CALIBRATION_FPR_MAX,
    TEMPORAL_RATE_CALIBRATION_POWER_MIN,
    TEMPORAL_RATE_CALIBRATION_TRIALS,
    TEMPORAL_RATE_FAMILYWISE_ALPHA,
    TEMPORAL_RATE_HARMONICS,
    TEMPORAL_RATE_HETEROGENEITY_ALPHA,
    TEMPORAL_RATE_INJECTION_FRACTIONAL_AMPLITUDE,
    TEMPORAL_RATE_MIN_CALIBRATION_SNR,
    TEMPORAL_RATE_MIN_CLOCKS,
    TEMPORAL_RATE_MIN_COMPOSITIONS,
    TEMPORAL_RATE_MIN_DRIVE_CYCLES,
    TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE,
    TEMPORAL_RATE_MIN_SAMPLES_PER_CYCLE,
    TEMPORAL_RATE_SURROGATE_DRAWS,
)

CLAIM_BOUNDARY = (
    "A COMMON_MODE_ANOMALY_CANDIDATE is an anomalous common-mode clock-rate "
    "modulation at the tested drive harmonics, not evidence of backward time "
    "travel or a non-metric 'time'. A candidate can also be produced by an "
    "unmodeled common-mode systematic. Promotion requires independent "
    "replication with dissimilar clocks, a bounded systematics budget, and a "
    "physical coupling model."
)

_UNIFORM_TIME_TOLERANCE = 1e-6
_CALIBRATION_CONFIDENCE = 0.95


class TemporalRateProtocolError(ValueError):
    """Raised when calibration/confirmation data violate the preregistered protocol."""


class TemporalRateVerdict(str, Enum):
    """Epistemic verdicts for the temporal-rate modulation null test."""

    NULL_CONSISTENT = "NULL_CONSISTENT"
    COMMON_MODE_ANOMALY_CANDIDATE = "COMMON_MODE_ANOMALY_CANDIDATE"
    OSCILLATOR_SPECIFIC_ARTIFACT = "OSCILLATOR_SPECIFIC_ARTIFACT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ClockChannel:
    """One clock's fractional-frequency series and systematics budget.

    ``systematic_susceptibility_sigma`` is the one-sigma uncertainty, in
    fractional frequency per unit drive amplitude, of the ordinary
    susceptibility model at the confirmation frequency. It must include
    frequency-dependence and shielding-model uncertainty, not only the
    calibration fit's statistical error. The 0.0 default is only appropriate
    for synthetic controls with no unmodeled systematics; real data must
    supply a defensible positive budget.
    """

    clock_id: str
    composition: str
    fractional_frequency: np.ndarray
    systematic_susceptibility_sigma: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.clock_id).strip():
            raise TemporalRateProtocolError("clock_id must be a non-empty string")
        if not str(self.composition).strip():
            raise TemporalRateProtocolError(
                f"clock '{self.clock_id}' has an empty composition"
            )
        y = np.asarray(self.fractional_frequency, dtype=float).ravel()
        if y.size < 2:
            raise TemporalRateProtocolError(
                f"clock '{self.clock_id}' has fewer than two samples"
            )
        if not np.all(np.isfinite(y)):
            raise TemporalRateProtocolError(
                f"clock '{self.clock_id}' contains non-finite fractional frequencies"
            )
        if (
            not np.isfinite(self.systematic_susceptibility_sigma)
            or self.systematic_susceptibility_sigma < 0.0
        ):
            raise TemporalRateProtocolError(
                f"clock '{self.clock_id}' has an invalid systematic uncertainty"
            )
        object.__setattr__(self, "fractional_frequency", y)


@dataclass(frozen=True)
class ClockDataset:
    """One synchronised drive-plus-clocks run."""

    dataset_id: str
    time_s: np.ndarray
    drive: np.ndarray
    channels: tuple[ClockChannel, ...]
    drive_frequency_hz: float

    def __post_init__(self) -> None:
        if not str(self.dataset_id).strip():
            raise TemporalRateProtocolError("dataset_id must be a non-empty string")
        t = np.asarray(self.time_s, dtype=float).ravel()
        drive = np.asarray(self.drive, dtype=float).ravel()
        channels = tuple(self.channels)
        if t.size < 2 or drive.size != t.size:
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' has mismatched time/drive arrays"
            )
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(drive)):
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' contains non-finite time or drive values"
            )
        if np.any(np.diff(t) <= 0.0):
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' time axis is not strictly increasing"
            )
        steps = np.diff(t)
        median_step = float(np.median(steps))
        if np.max(np.abs(steps - median_step)) > _UNIFORM_TIME_TOLERANCE * median_step:
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' time axis is not uniformly "
                "sampled; resample before adjudication"
            )
        if not np.isfinite(self.drive_frequency_hz) or self.drive_frequency_hz <= 0.0:
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' has an invalid drive frequency"
            )
        if not channels:
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' has no clock channels"
            )
        clock_ids = [channel.clock_id for channel in channels]
        if len(set(clock_ids)) != len(clock_ids):
            raise TemporalRateProtocolError(
                f"dataset '{self.dataset_id}' contains duplicate clock ids"
            )
        for channel in channels:
            if channel.fractional_frequency.size != t.size:
                raise TemporalRateProtocolError(
                    f"clock '{channel.clock_id}' length does not match dataset "
                    f"'{self.dataset_id}'"
                )
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "drive", drive)
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class TemporalRateAdjudication:
    """Result of one calibration/confirmation adjudication."""

    verdict: TemporalRateVerdict
    statistic: float
    p_value: float
    heterogeneity_statistic: float
    heterogeneity_p_value: float
    common_mode_amplitude: float
    common_mode_phase_rad: float
    best_harmonic: int
    per_clock_amplitudes: dict[str, float]
    n_cycles: float
    n_surrogates: int
    calibration_dataset_id: str
    confirmation_dataset_id: str
    harmonics: int
    familywise_alpha: float
    heterogeneity_alpha: float
    min_fractional_amplitude: float
    calibration_snr: dict[str, float]
    calibration_harmonic_snr: dict[str, tuple[float, ...]]
    canonical_spec_sha256: str
    rational_justification: str
    claim_boundary: str = CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "heterogeneity_statistic": self.heterogeneity_statistic,
            "heterogeneity_p_value": self.heterogeneity_p_value,
            "common_mode_amplitude": self.common_mode_amplitude,
            "common_mode_phase_rad": self.common_mode_phase_rad,
            "best_harmonic": self.best_harmonic,
            "per_clock_amplitudes": dict(self.per_clock_amplitudes),
            "n_cycles": self.n_cycles,
            "n_surrogates": self.n_surrogates,
            "calibration_dataset_id": self.calibration_dataset_id,
            "confirmation_dataset_id": self.confirmation_dataset_id,
            "harmonics": self.harmonics,
            "familywise_alpha": self.familywise_alpha,
            "heterogeneity_alpha": self.heterogeneity_alpha,
            "min_fractional_amplitude": self.min_fractional_amplitude,
            "calibration_snr": dict(self.calibration_snr),
            "calibration_harmonic_snr": {
                clock_id: list(values)
                for clock_id, values in self.calibration_harmonic_snr.items()
            },
            "canonical_spec_sha256": self.canonical_spec_sha256,
            "rational_justification": self.rational_justification,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class TemporalRateCalibrationResult:
    """False-positive rate and injection power of the detector itself."""

    false_positive_rate: float
    false_artifact_rate: float
    any_false_claim_rate: float
    detection_power: float
    n_null_trials: int
    n_injection_trials: int
    n_calibration_draws: int
    n_null_insufficient: int
    n_injection_insufficient: int
    injection_amplitude: float
    fpr_max: float
    power_min: float
    false_positive_rate_upper_95: float
    false_artifact_rate_upper_95: float
    any_false_claim_rate_upper_95: float
    detection_power_lower_95: float
    median_null_p_value: float
    median_injection_p_value: float
    passed: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "false_positive_rate": self.false_positive_rate,
            "false_artifact_rate": self.false_artifact_rate,
            "any_false_claim_rate": self.any_false_claim_rate,
            "detection_power": self.detection_power,
            "n_null_trials": self.n_null_trials,
            "n_injection_trials": self.n_injection_trials,
            "n_calibration_draws": self.n_calibration_draws,
            "n_null_insufficient": self.n_null_insufficient,
            "n_injection_insufficient": self.n_injection_insufficient,
            "injection_amplitude": self.injection_amplitude,
            "fpr_max": self.fpr_max,
            "power_min": self.power_min,
            "false_positive_rate_upper_95": (
                self.false_positive_rate_upper_95
            ),
            "false_artifact_rate_upper_95": (
                self.false_artifact_rate_upper_95
            ),
            "any_false_claim_rate_upper_95": (
                self.any_false_claim_rate_upper_95
            ),
            "detection_power_lower_95": self.detection_power_lower_95,
            "median_null_p_value": self.median_null_p_value,
            "median_injection_p_value": self.median_injection_p_value,
            "passed": self.passed,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class _ClockCalibration:
    clock_id: str
    chi: np.ndarray
    sigma: np.ndarray
    harmonic_snr: np.ndarray
    snr: float


def _harmonic_indices(harmonics: int) -> list[tuple[int, int]]:
    return [(2 + 2 * (k - 1), 3 + 2 * (k - 1)) for k in range(1, harmonics + 1)]


def _drive_phasor(dataset: ClockDataset) -> complex:
    theta = 2.0 * np.pi * dataset.drive_frequency_hz * dataset.time_s
    design = np.column_stack(
        (
            np.ones_like(dataset.time_s),
            dataset.time_s - dataset.time_s.mean(),
            np.cos(theta),
            np.sin(theta),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, dataset.drive, rcond=None)
    phasor = complex(coefficients[2], -coefficients[3])
    if not np.isfinite(phasor.real) or not np.isfinite(phasor.imag):
        raise TemporalRateProtocolError(
            f"dataset '{dataset.dataset_id}' produced a non-finite drive phasor"
        )
    if abs(phasor) <= np.finfo(float).eps:
        raise TemporalRateProtocolError(
            f"dataset '{dataset.dataset_id}' has no resolvable drive component at "
            f"{dataset.drive_frequency_hz} Hz"
        )
    return phasor


def _design_matrix(
    time_s: np.ndarray,
    drive_frequency_hz: float,
    phasor: complex,
    harmonics: int,
    *,
    scale_by_amplitude: bool,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = abs(phasor)
    theta = 2.0 * np.pi * drive_frequency_hz * time_s + np.angle(phasor)
    columns = [np.ones_like(time_s), time_s - time_s.mean()]
    for k in range(1, harmonics + 1):
        cos_k = np.cos(k * theta)
        sin_k = np.sin(k * theta)
        if scale_by_amplitude:
            cos_k = amplitude * cos_k
            sin_k = amplitude * sin_k
        columns.append(cos_k)
        columns.append(sin_k)
    return np.column_stack(columns), theta


def _lag1_variance_inflation(residual: np.ndarray) -> float:
    centred = residual - residual.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0:
        return 1.0
    rho = float(np.dot(centred[:-1], centred[1:]) / denominator)
    if not np.isfinite(rho) or rho <= 0.0:
        return 1.0
    rho = min(rho, 0.95)
    return (1.0 + rho) / (1.0 - rho)


def _ols(
    target: np.ndarray,
    design: np.ndarray,
    xtx_inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    beta = xtx_inverse @ (design.T @ target)
    residual = target - design @ beta
    dof = max(1, design.shape[0] - design.shape[1])
    sigma_squared = float(np.dot(residual, residual) / dof)
    covariance = (
        sigma_squared * _lag1_variance_inflation(residual) * xtx_inverse
    )
    return beta, residual, sigma_squared, covariance


def _calibrate_clocks(
    calibration: ClockDataset,
    harmonics: int,
) -> dict[str, _ClockCalibration]:
    phasor = _drive_phasor(calibration)
    design, _ = _design_matrix(
        calibration.time_s,
        calibration.drive_frequency_hz,
        phasor,
        harmonics,
        scale_by_amplitude=True,
    )
    xtx_inverse = np.linalg.pinv(design.T @ design)
    fitted: dict[str, _ClockCalibration] = {}
    for channel in calibration.channels:
        beta, _, _, covariance = _ols(
            channel.fractional_frequency, design, xtx_inverse
        )
        chi = np.empty(harmonics, dtype=complex)
        sigma = np.empty(harmonics, dtype=float)
        for index, (ia, ib) in enumerate(_harmonic_indices(harmonics)):
            chi[index] = float(beta[ia]) - 1j * float(beta[ib])
            variance = float(covariance[ia, ia] + covariance[ib, ib])
            sigma[index] = np.sqrt(
                max(variance, 0.0) + channel.systematic_susceptibility_sigma**2
            )
        harmonic_snr = np.divide(
            np.abs(chi),
            sigma,
            out=np.full_like(sigma, np.inf),
            where=sigma > 0.0,
        )
        fitted[channel.clock_id] = _ClockCalibration(
            clock_id=channel.clock_id,
            chi=chi,
            sigma=sigma,
            harmonic_snr=harmonic_snr,
            snr=float(harmonic_snr[0]),
        )
    return fitted


def _confirmation_residuals(
    confirmation: ClockDataset,
    calibration: dict[str, _ClockCalibration],
    harmonics: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]:
    phasor = _drive_phasor(confirmation)
    amplitude = abs(phasor)
    design, theta = _design_matrix(
        confirmation.time_s,
        confirmation.drive_frequency_hz,
        phasor,
        harmonics,
        scale_by_amplitude=False,
    )
    xtx_inverse = np.linalg.pinv(design.T @ design)
    residuals = np.empty(
        (len(confirmation.channels), confirmation.time_s.size), dtype=float
    )
    for row, channel in enumerate(confirmation.channels):
        model = calibration[channel.clock_id]
        predicted = np.zeros_like(confirmation.time_s)
        for k in range(1, harmonics + 1):
            predicted += amplitude * np.real(
                model.chi[k - 1] * np.exp(1j * k * theta)
            )
        residuals[row] = channel.fractional_frequency - predicted
    return residuals, theta, design, xtx_inverse, phasor


def _project_amplitudes(
    residuals: np.ndarray,
    design: np.ndarray,
    xtx_inverse: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    projection = xtx_inverse @ design.T
    betas = residuals @ projection.T
    amplitudes = np.empty((residuals.shape[0], harmonics), dtype=complex)
    for index, (ia, ib) in enumerate(_harmonic_indices(harmonics)):
        amplitudes[:, index] = betas[:, ia] - 1j * betas[:, ib]
    return amplitudes


def _fit_confirmation_amplitudes(
    residuals: np.ndarray,
    design: np.ndarray,
    xtx_inverse: np.ndarray,
    harmonics: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projection = xtx_inverse @ design.T
    betas = residuals @ projection.T
    fitted = betas @ design.T
    fit_residuals = residuals - fitted
    dof = max(1, design.shape[0] - design.shape[1])
    sigma_squared = np.sum(fit_residuals**2, axis=1) / dof
    inflation = np.array(
        [_lag1_variance_inflation(row) for row in fit_residuals]
    )
    sigma_squared = sigma_squared * inflation
    amplitudes = np.empty((residuals.shape[0], harmonics), dtype=complex)
    sigmas = np.empty((residuals.shape[0], harmonics), dtype=float)
    for index, (ia, ib) in enumerate(_harmonic_indices(harmonics)):
        amplitudes[:, index] = betas[:, ia] - 1j * betas[:, ib]
        sigmas[:, index] = np.sqrt(
            np.maximum(
                sigma_squared * (xtx_inverse[ia, ia] + xtx_inverse[ib, ib]),
                0.0,
            )
        )
    return amplitudes, sigmas, fit_residuals


def _combine_clocks(
    amplitudes: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = sigmas[sigmas > 0.0]
    floor = float(positive.min()) * 1e-6 if positive.size else 1.0
    weights = 1.0 / np.maximum(sigmas, floor) ** 2
    denominator = weights.sum(axis=0)
    common = (weights * amplitudes).sum(axis=0) / denominator
    sigma_common = 1.0 / np.sqrt(denominator)
    statistic = np.abs(common) / sigma_common
    heterogeneity = np.sum(
        weights * np.abs(amplitudes - common) ** 2, axis=0
    )
    return statistic, heterogeneity, common


def _surrogate_residuals(
    residuals: np.ndarray,
    theta: np.ndarray,
    amplitude: float,
    calibration_sigmas: np.ndarray,
    null_sigmas: np.ndarray,
    harmonics: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = residuals.shape[1]
    spectra = np.fft.rfft(residuals, axis=1)
    phase_increments = rng.uniform(0.0, 2.0 * np.pi, size=spectra.shape[1])
    phase_increments[0] = 0.0
    if n_samples % 2 == 0:
        phase_increments[-1] = 0.0
    surrogate = np.fft.irfft(
        np.abs(spectra) * np.exp(1j * (np.angle(spectra) + phase_increments)),
        n=n_samples,
        axis=1,
    )
    for row in range(residuals.shape[0]):
        for index in range(harmonics):
            null_sigma = float(null_sigmas[row, index])
            if null_sigma > 0.0:
                noise = null_sigma * (
                    rng.normal() + 1j * rng.normal()
                ) / np.sqrt(2.0)
                surrogate[row] += np.real(
                    noise * np.exp(1j * (index + 1) * theta)
                )
            calibration_sigma = float(calibration_sigmas[row, index])
            if calibration_sigma <= 0.0:
                continue
            error = calibration_sigma * (
                rng.normal() + 1j * rng.normal()
            ) / np.sqrt(2.0)
            surrogate[row] += amplitude * np.real(
                error * np.exp(1j * (index + 1) * theta)
            )
    return surrogate


def validate_temporal_rate_pair(
    calibration: ClockDataset,
    confirmation: ClockDataset,
    *,
    harmonics: int | None = None,
) -> None:
    harmonics = TEMPORAL_RATE_HARMONICS if harmonics is None else int(harmonics)
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1")
    if calibration.dataset_id == confirmation.dataset_id:
        raise TemporalRateProtocolError(
            "calibration and confirmation share dataset_id "
            f"'{calibration.dataset_id}'; the confirmation slice is not independent"
        )
    frequency_scale = max(
        abs(calibration.drive_frequency_hz),
        abs(confirmation.drive_frequency_hz),
        1.0,
    )
    if (
        abs(calibration.drive_frequency_hz - confirmation.drive_frequency_hz)
        <= 1e-12 * frequency_scale
    ):
        raise TemporalRateProtocolError(
            "calibration and confirmation use the same drive frequency; the "
            "ordinary susceptibility fit would absorb the effect under test"
        )
    calibration_duration = float(
        calibration.time_s[-1] - calibration.time_s[0]
    )
    confirmation_duration = float(
        confirmation.time_s[-1] - confirmation.time_s[0]
    )
    resolution = 1.0 / min(calibration_duration, confirmation_duration)
    calibration_harmonics = [
        k * calibration.drive_frequency_hz for k in range(1, harmonics + 1)
    ]
    confirmation_harmonics = [
        k * confirmation.drive_frequency_hz for k in range(1, harmonics + 1)
    ]
    overlaps = [
        (calibration_frequency, confirmation_frequency)
        for calibration_frequency in calibration_harmonics
        for confirmation_frequency in confirmation_harmonics
        if abs(calibration_frequency - confirmation_frequency) <= resolution
    ]
    if overlaps:
        calibration_frequency, confirmation_frequency = overlaps[0]
        raise TemporalRateProtocolError(
            "calibration and confirmation harmonic sets overlap at "
            f"{calibration_frequency:g} Hz and {confirmation_frequency:g} Hz "
            f"(within the {resolution:g} Hz Rayleigh resolution); the "
            "calibration could absorb a confirmation harmonic"
        )
    calibration_by_id = {channel.clock_id: channel for channel in calibration.channels}
    confirmation_by_id = {
        channel.clock_id: channel for channel in confirmation.channels
    }
    if set(calibration_by_id) != set(confirmation_by_id):
        raise TemporalRateProtocolError(
            "calibration and confirmation must contain the same clock ids"
        )
    for clock_id, confirmation_channel in confirmation_by_id.items():
        calibration_channel = calibration_by_id[clock_id]
        if calibration_channel.composition != confirmation_channel.composition:
            raise TemporalRateProtocolError(
                f"clock '{clock_id}' composition differs between calibration and "
                "confirmation"
            )
        if (
            np.array_equal(
                calibration_channel.fractional_frequency,
                confirmation_channel.fractional_frequency,
            )
            and float(np.ptp(calibration_channel.fractional_frequency)) > 0.0
        ):
            raise TemporalRateProtocolError(
                f"clock '{clock_id}' carries an identical non-constant series in "
                "calibration and confirmation; the confirmation slice was reused"
            )


def _insufficient_reason(
    calibration: ClockDataset,
    confirmation: ClockDataset,
    harmonics: int,
) -> str | None:
    for dataset in (calibration, confirmation):
        duration = float(dataset.time_s[-1] - dataset.time_s[0])
        cycles = dataset.drive_frequency_hz * duration
        if cycles < TEMPORAL_RATE_MIN_DRIVE_CYCLES:
            return (
                f"dataset '{dataset.dataset_id}' spans {cycles:.2f} drive cycles, "
                f"below the preregistered minimum {TEMPORAL_RATE_MIN_DRIVE_CYCLES:g}"
            )
        sample_rate = (dataset.time_s.size - 1) / duration
        samples_per_cycle = sample_rate / dataset.drive_frequency_hz
        if samples_per_cycle < TEMPORAL_RATE_MIN_SAMPLES_PER_CYCLE:
            return (
                f"dataset '{dataset.dataset_id}' has {samples_per_cycle:.2f} samples "
                f"per drive cycle, below {TEMPORAL_RATE_MIN_SAMPLES_PER_CYCLE:g}"
            )
        if sample_rate < 2.0 * harmonics * dataset.drive_frequency_hz:
            return (
                f"dataset '{dataset.dataset_id}' is undersampled for {harmonics} "
                "harmonics"
            )
    if len(confirmation.channels) < TEMPORAL_RATE_MIN_CLOCKS:
        return (
            f"{len(confirmation.channels)} clocks supplied; a common-mode claim "
            f"requires at least {TEMPORAL_RATE_MIN_CLOCKS}"
        )
    compositions = {channel.composition for channel in confirmation.channels}
    if len(compositions) < TEMPORAL_RATE_MIN_COMPOSITIONS:
        return (
            f"a common-mode claim requires at least "
            f"{TEMPORAL_RATE_MIN_COMPOSITIONS} physically distinct clock "
            f"compositions; only {len(compositions)} supplied"
        )
    return None


def _refusal(
    verdict: TemporalRateVerdict,
    reason: str,
    *,
    calibration_dataset_id: str,
    confirmation_dataset_id: str,
    harmonics: int,
    n_cycles: float = float("nan"),
    n_surrogates: int = 0,
    calibration_snr: dict[str, float] | None = None,
    calibration_harmonic_snr: dict[str, tuple[float, ...]] | None = None,
) -> TemporalRateAdjudication:
    return TemporalRateAdjudication(
        verdict=verdict,
        statistic=float("nan"),
        p_value=float("nan"),
        heterogeneity_statistic=float("nan"),
        heterogeneity_p_value=float("nan"),
        common_mode_amplitude=float("nan"),
        common_mode_phase_rad=float("nan"),
        best_harmonic=0,
        per_clock_amplitudes={},
        n_cycles=n_cycles,
        n_surrogates=n_surrogates,
        calibration_dataset_id=calibration_dataset_id,
        confirmation_dataset_id=confirmation_dataset_id,
        harmonics=harmonics,
        familywise_alpha=TEMPORAL_RATE_FAMILYWISE_ALPHA,
        heterogeneity_alpha=TEMPORAL_RATE_HETEROGENEITY_ALPHA,
        min_fractional_amplitude=TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE,
        calibration_snr=dict(calibration_snr or {}),
        calibration_harmonic_snr=dict(calibration_harmonic_snr or {}),
        canonical_spec_sha256=PINNED_SPEC_SHA256,
        rational_justification=reason,
    )


def adjudicate_temporal_rate_modulation(
    calibration: ClockDataset,
    confirmation: ClockDataset,
    *,
    harmonics: int | None = None,
    n_surrogates: int | None = None,
    seed: int = 0,
) -> TemporalRateAdjudication:
    """Adjudicate a blinded confirmation run against a calibration run.

    Parameters
    ----------
    calibration:
        Independent run used only to estimate each clock's ordinary complex
        susceptibility to the drive. Its drive frequency must differ from the
        confirmation frequency unless the effect under test is known to be
        absent during calibration.
    confirmation:
        The blinded run. Constructed so that its dataset id and clock series
        are disjoint from calibration.
    harmonics:
        Number of drive harmonics tested. Defaults to the canonical value.
    n_surrogates:
        Shared-phase Fourier surrogates used to calibrate the null
        distribution. Defaults to the canonical value.
    seed:
        Seed for the surrogate generator. Deterministic replay is pinned.
    """
    harmonics = TEMPORAL_RATE_HARMONICS if harmonics is None else int(harmonics)
    n_surrogates = (
        TEMPORAL_RATE_SURROGATE_DRAWS
        if n_surrogates is None
        else int(n_surrogates)
    )
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")

    validate_temporal_rate_pair(
        calibration, confirmation, harmonics=harmonics
    )
    n_cycles = confirmation.drive_frequency_hz * float(
        confirmation.time_s[-1] - confirmation.time_s[0]
    )
    reason = _insufficient_reason(calibration, confirmation, harmonics)
    if reason is not None:
        return _refusal(
            TemporalRateVerdict.INSUFFICIENT_EVIDENCE,
            reason,
            calibration_dataset_id=calibration.dataset_id,
            confirmation_dataset_id=confirmation.dataset_id,
            harmonics=harmonics,
            n_cycles=n_cycles,
            n_surrogates=n_surrogates,
        )

    calibrated = _calibrate_clocks(calibration, harmonics)
    weak = [
        f"{clock_id} (SNR={model.snr:.2f})"
        for clock_id, model in calibrated.items()
        if model.snr < TEMPORAL_RATE_MIN_CALIBRATION_SNR
    ]
    if weak:
        return _refusal(
            TemporalRateVerdict.INSUFFICIENT_EVIDENCE,
            "calibration cannot resolve the ordinary drive response for "
            + ", ".join(weak)
            + f"; minimum calibration SNR is {TEMPORAL_RATE_MIN_CALIBRATION_SNR:g}",
            calibration_dataset_id=calibration.dataset_id,
            confirmation_dataset_id=confirmation.dataset_id,
            harmonics=harmonics,
            n_cycles=n_cycles,
            n_surrogates=n_surrogates,
            calibration_snr={
                clock_id: float(model.snr)
                for clock_id, model in calibrated.items()
            },
            calibration_harmonic_snr={
                clock_id: tuple(float(value) for value in model.harmonic_snr)
                for clock_id, model in calibrated.items()
            },
        )

    residuals, theta, design, xtx_inverse, phasor = _confirmation_residuals(
        confirmation, calibrated, harmonics
    )
    amplitudes, sigmas, fit_residuals = _fit_confirmation_amplitudes(
        residuals, design, xtx_inverse, harmonics
    )
    statistic, heterogeneity, common = _combine_clocks(amplitudes, sigmas)
    best = int(np.argmax(statistic))
    observed_statistic = float(statistic[best])
    observed_heterogeneity = float(heterogeneity[best])

    calibration_sigmas = np.array(
        [
            calibrated[channel.clock_id].sigma
            for channel in confirmation.channels
        ]
    )
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(n_surrogates, dtype=float)
    null_heterogeneities = np.empty(n_surrogates, dtype=float)
    for draw in range(n_surrogates):
        surrogate = _surrogate_residuals(
            fit_residuals,
            theta,
            abs(phasor),
            calibration_sigmas,
            sigmas,
            harmonics,
            rng,
        )
        surrogate_amplitudes = _project_amplitudes(
            surrogate, design, xtx_inverse, harmonics
        )
        surrogate_statistic, surrogate_heterogeneity, _ = _combine_clocks(
            surrogate_amplitudes, sigmas
        )
        null_statistics[draw] = float(np.max(surrogate_statistic))
        null_heterogeneities[draw] = float(np.max(surrogate_heterogeneity))

    p_value = float(
        (1 + np.sum(null_statistics >= observed_statistic))
        / (1 + n_surrogates)
    )
    heterogeneity_p_value = float(
        (1 + np.sum(null_heterogeneities >= observed_heterogeneity))
        / (1 + n_surrogates)
    )
    amplitude = float(abs(common[best]))
    phase = float(np.angle(common[best]))
    per_clock = {
        channel.clock_id: float(abs(amplitudes[row, best]))
        for row, channel in enumerate(confirmation.channels)
    }

    if heterogeneity_p_value < TEMPORAL_RATE_HETEROGENEITY_ALPHA:
        verdict = TemporalRateVerdict.OSCILLATOR_SPECIFIC_ARTIFACT
        reason = (
            "per-clock residual amplitudes are inconsistent "
            f"(heterogeneity p={heterogeneity_p_value:.4g} < "
            f"{TEMPORAL_RATE_HETEROGENEITY_ALPHA:g}); the effect does not "
            "replicate as a common-mode clock-rate modulation"
        )
    elif p_value < TEMPORAL_RATE_FAMILYWISE_ALPHA:
        if amplitude >= TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE:
            verdict = TemporalRateVerdict.COMMON_MODE_ANOMALY_CANDIDATE
            reason = (
                f"common-mode fractional-frequency modulation "
                f"{amplitude:.3g} at {best + 1}x the drive frequency survives "
                f"the surrogate null (p={p_value:.4g} < "
                f"{TEMPORAL_RATE_FAMILYWISE_ALPHA:g}) and is consistent across "
                "clock compositions"
            )
        else:
            verdict = TemporalRateVerdict.INSUFFICIENT_EVIDENCE
            reason = (
                f"statistically resolved modulation {amplitude:.3g} is below the "
                f"preregistered physical amplitude floor "
                f"{TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE:g}"
            )
    else:
        verdict = TemporalRateVerdict.NULL_CONSISTENT
        reason = (
            "no phase-locked common-mode residual at the tested drive "
            f"harmonics survives the surrogate null "
            f"(p={p_value:.4g} >= {TEMPORAL_RATE_FAMILYWISE_ALPHA:g}); "
            f"best candidate amplitude {amplitude:.3g}"
        )

    return TemporalRateAdjudication(
        verdict=verdict,
        statistic=observed_statistic,
        p_value=p_value,
        heterogeneity_statistic=observed_heterogeneity,
        heterogeneity_p_value=heterogeneity_p_value,
        common_mode_amplitude=amplitude,
        common_mode_phase_rad=phase,
        best_harmonic=best + 1,
        per_clock_amplitudes=per_clock,
        n_cycles=n_cycles,
        n_surrogates=n_surrogates,
        calibration_dataset_id=calibration.dataset_id,
        confirmation_dataset_id=confirmation.dataset_id,
        harmonics=harmonics,
        familywise_alpha=TEMPORAL_RATE_FAMILYWISE_ALPHA,
        heterogeneity_alpha=TEMPORAL_RATE_HETEROGENEITY_ALPHA,
        min_fractional_amplitude=TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE,
        calibration_snr={
            clock_id: float(model.snr)
            for clock_id, model in calibrated.items()
        },
        calibration_harmonic_snr={
            clock_id: tuple(float(value) for value in model.harmonic_snr)
            for clock_id, model in calibrated.items()
        },
        canonical_spec_sha256=PINNED_SPEC_SHA256,
        rational_justification=reason,
    )


def simulate_clock_dataset(
    *,
    dataset_id: str,
    drive_frequency_hz: float,
    duration_s: float,
    sample_rate_hz: float,
    compositions: tuple[str, ...] = ("Sr optical lattice", "Yb+ ion"),
    susceptibilities: tuple[complex, ...] = (2e-15 + 1e-15j, -1e-15 + 3e-15j),
    anomaly_amplitude: float = 0.0,
    anomaly_common_mode: bool = True,
    anomaly_clock_index: int = 0,
    anomaly_signs: tuple[float, ...] | None = None,
    anomaly_phase_rad: float = 0.0,
    noise_sigma: float = 1e-16,
    drift_per_s: float = 0.0,
    systematic_susceptibility_sigmas: tuple[float, ...] | None = None,
    seed: int = 0,
) -> ClockDataset:
    """Generate a synthetic run for detector calibration and falsification tests.

    The simulator is deliberately simple: a cosine drive, a linear ordinary
    susceptibility, optional linear drift, white noise, and an optional
    injected modulation. It exists to measure the detector's false-positive
    rate and power, and to pin that oscillator-specific injections are
    rejected. It is not a physical model of any clock.
    """
    n_samples = round(duration_s * sample_rate_hz)
    if n_samples < 2:
        raise ValueError("duration_s * sample_rate_hz must give at least two samples")
    if len(compositions) != len(susceptibilities):
        raise ValueError("compositions and susceptibilities must have equal length")
    if anomaly_signs is not None and len(anomaly_signs) != len(compositions):
        raise ValueError("anomaly_signs must match the number of clocks")
    if (
        systematic_susceptibility_sigmas is not None
        and len(systematic_susceptibility_sigmas) != len(compositions)
    ):
        raise ValueError(
            "systematic_susceptibility_sigmas must match the number of clocks"
        )
    if not anomaly_common_mode and not 0 <= anomaly_clock_index < len(compositions):
        raise ValueError("anomaly_clock_index is out of range")

    time_s = np.arange(n_samples, dtype=float) / sample_rate_hz
    theta = 2.0 * np.pi * drive_frequency_hz * time_s
    drive = np.cos(theta)
    rng = np.random.default_rng(seed)
    channels: list[ClockChannel] = []
    for index, (composition, susceptibility) in enumerate(
        zip(compositions, susceptibilities, strict=True)
    ):
        fractional = np.real(susceptibility * np.exp(1j * theta))
        fractional = fractional + drift_per_s * (time_s - time_s.mean())
        fractional = fractional + noise_sigma * rng.normal(size=n_samples)
        if anomaly_signs is not None:
            sign = float(anomaly_signs[index])
        elif anomaly_common_mode or index == anomaly_clock_index:
            sign = 1.0
        else:
            sign = 0.0
        if anomaly_amplitude != 0.0 and sign != 0.0:
            fractional = fractional + sign * anomaly_amplitude * np.cos(
                theta + anomaly_phase_rad
            )
        systematic = (
            float(systematic_susceptibility_sigmas[index])
            if systematic_susceptibility_sigmas is not None
            else 0.0
        )
        channels.append(
            ClockChannel(
                clock_id=f"clock_{index}",
                composition=composition,
                fractional_frequency=fractional,
                systematic_susceptibility_sigma=systematic,
            )
        )
    return ClockDataset(
        dataset_id=dataset_id,
        time_s=time_s,
        drive=drive,
        channels=tuple(channels),
        drive_frequency_hz=drive_frequency_hz,
    )


def _clopper_pearson_upper(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    return float(
        beta.ppf(_CALIBRATION_CONFIDENCE, successes + 1, trials - successes)
    )


def _clopper_pearson_lower(successes: int, trials: int) -> float:
    if trials <= 0 or successes <= 0:
        return 0.0
    if successes >= trials:
        return float(
            beta.ppf(1.0 - _CALIBRATION_CONFIDENCE, successes, 1)
        )
    return float(
        beta.ppf(
            1.0 - _CALIBRATION_CONFIDENCE,
            successes,
            trials - successes + 1,
        )
    )


def run_detector_calibration(
    *,
    n_trials: int | None = None,
    n_surrogates: int | None = None,
    injection_amplitude: float | None = None,
    seed: int = 0,
    drive_frequency_hz: float = 1.0,
    calibration_frequency_hz: float | None = None,
    duration_s: float = 128.0,
    sample_rate_hz: float = 16.0,
    noise_sigma: float = 1e-16,
) -> TemporalRateCalibrationResult:
    """Measure the detector's false-positive rate and injection power.

    A detector that cannot separate a planted common-mode injection from its
    own null is not evidence for anything. This harness generates independent
    calibration and confirmation runs per trial, so the measured null rejection
    rate is unconditional on one lucky calibration realization. The
    preregistered gate is the point-estimate ceiling/power floor. The exact
    one-sided 95% Clopper-Pearson upper bound on the false-positive rate and
    lower bound on power are reported alongside it; with only 128 trials, that
    bound is a diagnostic, not a second preregistered gate. Artifact and
    any-claim rates are reported separately because an artifact verdict is a
    rejection of common mode, not a discovery claim. At familywise alpha
    ``0.01``, ``n_surrogates`` must be at least 100 for any candidate to be
    reachable. The result is a property of the analysis, not of nature.
    """
    trials = (
        TEMPORAL_RATE_CALIBRATION_TRIALS if n_trials is None else int(n_trials)
    )
    surrogates = (
        TEMPORAL_RATE_SURROGATE_DRAWS
        if n_surrogates is None
        else int(n_surrogates)
    )
    amplitude = (
        TEMPORAL_RATE_INJECTION_FRACTIONAL_AMPLITUDE
        if injection_amplitude is None
        else float(injection_amplitude)
    )
    if trials < 1:
        raise ValueError("n_trials must be at least 1")
    if surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")
    calibration_frequency = (
        drive_frequency_hz * 0.83
        if calibration_frequency_hz is None
        else float(calibration_frequency_hz)
    )

    false_positives = 0
    false_artifacts = 0
    any_false_claims = 0
    detections = 0
    null_insufficient = 0
    injection_insufficient = 0
    null_p_values: list[float] = []
    injection_p_values: list[float] = []
    for trial in range(trials):
        calibration = simulate_clock_dataset(
            dataset_id=f"calibration-{trial}",
            drive_frequency_hz=calibration_frequency,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            noise_sigma=noise_sigma,
            seed=seed + 5000 + trial,
        )
        null_run = simulate_clock_dataset(
            dataset_id=f"null-{trial}",
            drive_frequency_hz=drive_frequency_hz,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            noise_sigma=noise_sigma,
            seed=seed + 1000 + trial,
        )
        null_result = adjudicate_temporal_rate_modulation(
            calibration,
            null_run,
            n_surrogates=surrogates,
            seed=seed + 2000 + trial,
        )
        if null_result.verdict == TemporalRateVerdict.COMMON_MODE_ANOMALY_CANDIDATE:
            false_positives += 1
        if null_result.verdict == TemporalRateVerdict.OSCILLATOR_SPECIFIC_ARTIFACT:
            false_artifacts += 1
        if null_result.verdict in {
            TemporalRateVerdict.COMMON_MODE_ANOMALY_CANDIDATE,
            TemporalRateVerdict.OSCILLATOR_SPECIFIC_ARTIFACT,
        }:
            any_false_claims += 1
        if null_result.verdict == TemporalRateVerdict.INSUFFICIENT_EVIDENCE:
            null_insufficient += 1
        if np.isfinite(null_result.p_value):
            null_p_values.append(null_result.p_value)

        injection_run = simulate_clock_dataset(
            dataset_id=f"injection-{trial}",
            drive_frequency_hz=drive_frequency_hz,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            anomaly_amplitude=amplitude,
            noise_sigma=noise_sigma,
            seed=seed + 3000 + trial,
        )
        injection_result = adjudicate_temporal_rate_modulation(
            calibration,
            injection_run,
            n_surrogates=surrogates,
            seed=seed + 4000 + trial,
        )
        if injection_result.verdict == TemporalRateVerdict.COMMON_MODE_ANOMALY_CANDIDATE:
            detections += 1
        if injection_result.verdict == TemporalRateVerdict.INSUFFICIENT_EVIDENCE:
            injection_insufficient += 1
        if np.isfinite(injection_result.p_value):
            injection_p_values.append(injection_result.p_value)

    false_positive_rate = false_positives / trials
    false_artifact_rate = false_artifacts / trials
    any_false_claim_rate = any_false_claims / trials
    detection_power = detections / trials
    fpr_upper = _clopper_pearson_upper(false_positives, trials)
    artifact_upper = _clopper_pearson_upper(false_artifacts, trials)
    any_upper = _clopper_pearson_upper(any_false_claims, trials)
    power_lower = _clopper_pearson_lower(detections, trials)
    passed = (
        false_positive_rate <= TEMPORAL_RATE_CALIBRATION_FPR_MAX
        and detection_power >= TEMPORAL_RATE_CALIBRATION_POWER_MIN
        and null_insufficient == 0
        and injection_insufficient == 0
    )
    return TemporalRateCalibrationResult(
        false_positive_rate=false_positive_rate,
        false_artifact_rate=false_artifact_rate,
        any_false_claim_rate=any_false_claim_rate,
        detection_power=detection_power,
        n_null_trials=trials,
        n_injection_trials=trials,
        n_calibration_draws=trials,
        n_null_insufficient=null_insufficient,
        n_injection_insufficient=injection_insufficient,
        injection_amplitude=amplitude,
        fpr_max=TEMPORAL_RATE_CALIBRATION_FPR_MAX,
        power_min=TEMPORAL_RATE_CALIBRATION_POWER_MIN,
        false_positive_rate_upper_95=fpr_upper,
        false_artifact_rate_upper_95=artifact_upper,
        any_false_claim_rate_upper_95=any_upper,
        detection_power_lower_95=power_lower,
        median_null_p_value=float(np.median(null_p_values))
        if null_p_values
        else float("nan"),
        median_injection_p_value=float(np.median(injection_p_values))
        if injection_p_values
        else float("nan"),
        passed=bool(passed),
        seed=seed,
    )
