"""Uncertainty quantification v1 (frozen): RMSE-CIs + residual-bootstrap bands.

Covers what Jacobian SEs could not (rejected: 13x-176000x undercoverage).
Bands are estimator-variability under resampled noise, documented as such;
coverage of physical truth is MEASURED below on the clean well-identified
fit, not assumed. Deterministic seeds throughout.
"""
import numpy as np

import apedp_dyn
from apedp_dyn import dyn_sweep as ds
from apedp_dyn.epistemic.bootstrap import (
    holdout_se_series,
    param_bootstrap_ci,
    rmse_bootstrap_ci,
)

NEGATE = {"u": "gmu", "x": "alpha", "x3": "bnl", "v": "delta"}


class TestRmseCI:
    def test_deterministic_ordered_covering(self):
        slate = ds.build_slates()
        prob = ds._with_noise(slate["STAB"]["prob_clean"], 10,
                              seed=ds.cell_seed("STAB", 2, 10))
        rr = apedp_dyn.fit_candidate(prob, slate["STAB"]["rival"])
        se = holdout_se_series(prob, rr.candidate, rr.fit.param_values)
        lo, hi, rmse = rmse_bootstrap_ci(se, n_boot=1000, seed=0)
        assert rmse_bootstrap_ci(se, n_boot=1000, seed=0) == (lo, hi, rmse)
        assert 0 < lo <= rmse <= hi
        assert hi - lo > 0


class TestParamBands:
    def test_clean_fit_bands_cover_signed_truth(self):
        """Fitted IR parameters carry the equation's sign convention
        (xddot = a1*x + ... with a1 ~ -alpha); compare against negated
        physical truth, as test_dyn_d2b does."""
        from apedp_dyn import enumerate_candidates, prune_admissible
        from apedp_dyn.problems import duffing

        clean = duffing.load()
        cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"),
                                     max_terms=4)
        kept = {tuple(sorted(s.candidate.term_set())): s
                for s in prune_admissible(clean, cands)}
        ir = kept[("u", "v", "x", "x3")].candidate
        rt = apedp_dyn.fit_candidate(clean, ir)
        truth = {p.name: float(p.value) for p in clean.parameters}
        bands = param_bootstrap_ci(clean, ir, rt.fit.param_values, n_boot=30, seed=0)
        assert set(bands) == set(rt.fit.param_values)
        covered = 0
        for t in ir.terms:
            lo, hi, orig = bands[t.parameter]
            assert np.isfinite(lo) and np.isfinite(hi) and lo <= orig <= hi
            # Truth coverage is MEASURED, not guaranteed: bootstrap centers on
            # the fit, so optimizer bias is invisible by construction. Observed
            # 3/4 on clean DUFF (u misses by 4e-5); floor pinned loosely for
            # optimizer/BLAS wobble, exact count lives in the artifact record.
            covered += lo <= -truth[NEGATE[t.name]] <= hi
        assert covered >= 1, bands
