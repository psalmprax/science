"""Frozen cross-regime benchmark: supported-in-R1 does not imply R2.

Nanoscale Hall-Petch fitted supra-threshold, scored on unseen supra-dc
(interpolation) and sub-dc inverted (extrapolation) holds. Recorded
measurements (noise_sigma=0.02, seed=42):

  interpolation: HP 3.56 vs constant 10.53 (HP leads 3x)
  extrapolation: HP 93.80 vs constant 14.30 (constant leads 6.6x)

The benchmark requires the transfer module to report CROSS_DOMAIN_FAILED
with leadership reversal and ~26x error blowup -- while the epistemic
verdict on either regime alone remains MULTI_HYPOTHESIS (no automatic
downgrade to INSUFFICIENT_EVIDENCE; interpolation evidence stands).
"""
from __future__ import annotations

from apedp_dyn.epistemic.transfer_scope import TransferScope, TransferStatus

# Frozen measurements from the M4 cross-regime experiment.
INTERP_BEST = 3.56
INTERP_RIVAL = 10.53
EXTRA_BEST = 93.80
EXTRA_RIVAL = 14.30


def _scope() -> TransferScope:
    return TransferScope(candidate_id="hp_global", rival_id="constant_baseline",
                         interp_rmse_best=INTERP_BEST, interp_rmse_rival=INTERP_RIVAL,
                         extra_rmse_best=EXTRA_BEST, extra_rmse_rival=EXTRA_RIVAL)


class TestCrossRegimeTransfer:
    def test_leadership_reversal_detected(self):
        assert _scope().leadership_reversed is True

    def test_error_blowup_recorded(self):
        scope = _scope()
        assert scope.error_blowup == 93.80 / 3.56
        assert scope.error_blowup > 5.0

    def test_cross_domain_failed(self):
        assert _scope().status == TransferStatus.CROSS_DOMAIN_FAILED

    def test_supported_ranking_alone_is_not_promotion(self):
        # Interpolation leadership (3.56 < 10.53) must not read as support
        # for the extrapolation regime: the record carries both scopes.
        rec = _scope().record()
        assert rec["transfer"]["status"] == TransferStatus.CROSS_DOMAIN_FAILED
        assert rec["promotion"]["status"] == "NOT_PROMOTED_BY_THIS_MODULE"
        assert rec["evidence"]["scope"] == "interpolation_and_extrapolation"

    def test_supported_domain_without_blowup(self):
        scope = TransferScope(candidate_id="hp_global", rival_id="constant_baseline",
                              interp_rmse_best=3.56, interp_rmse_rival=10.53,
                              extra_rmse_best=4.10, extra_rmse_rival=11.02)
        assert scope.leadership_reversed is False
        assert scope.status == TransferStatus.CROSS_DOMAIN_SUPPORTED
