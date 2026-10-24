"""Noise-scale instrument calibration (frozen v1).

The holdout noise estimator (third differences, gates/intervention.py)
measures smooth oscillatory trajectories with a stable upward bias: across
the frozen 36-cell dynamics grid (DUFF/COUP/STAB x 1/3/5/10% x 3 reps),
sigma_hat / sigma_true has median 1.82, range [1.67, 2.00]. The injected
noise is the calibration standard; the factor below is metrology, frozen
before reuse, never refit to outcomes.

CAL_SIGMA = 1.8 (median 1.82 rounded to one decimal; n=36).

Scope discipline: the calibrated scale is used ONLY where the question is
absolute adequacy against observational uncertainty (sole-survivor Guard 1).
tau_epi keeps the raw (inflated) sigma: overestimating the resolution floor
errs toward abstention, which is the safe direction for a comparative
indistinguishability margin. Changing tau would alter every comparative
adjudication; that is out of scope for this calibration.
"""
from __future__ import annotations

CAL_SIGMA = 1.8
CALIBRATION_VERSION = "1.0.0-grid36"


def calibrated_sigma(sigma_hat: float) -> float:
    """Best estimate of observational noise from the biased estimator."""
    return float(sigma_hat) / CAL_SIGMA
