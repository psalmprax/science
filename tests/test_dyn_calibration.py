"""Noise-scale instrument calibration (frozen v1).

The third-difference sigma estimator inflates ~1.8x on smooth oscillatory
trajectories (median sigma_hat/sigma_true = 1.82 over the frozen 36-cell
grid, range [1.67, 2.00]). CAL_SIGMA is metrology against the injected-noise
standard, not a tuning knob: Guard 1 asks absolute adequacy and needs the
best scale, while tau_epi keeps the raw (conservative) scale.
"""
import numpy as np

from apedp_dyn import dyn_sweep as ds
from apedp_dyn.epistemic.calibration import (
    CAL_SIGMA,
    CALIBRATION_VERSION,
    calibrated_sigma,
)
from apedp_dyn.gates.intervention import observation_scale


class TestCalibrationFrozen:
    def test_factor_and_version_pinned(self):
        assert CAL_SIGMA == 1.8
        assert CALIBRATION_VERSION == "1.0.0-grid36"
        assert calibrated_sigma(0.06) == 0.06 / 1.8

    def test_inflation_stable_on_subset(self):
        """Independent spot-check (1 problem x 2 levels x 1 rep, no fitting):
        the estimator bias is stable, justifying a single frozen factor."""
        slate = ds.build_slates()
        clean = slate["DUFF"]["prob_clean"]
        Yc = np.asarray(clean.holdout["truth_y"], dtype=float)
        sig_true = float(np.sqrt(np.mean(np.std(Yc, axis=0) ** 2)))
        ratios = []
        for pct in (3, 10):
            prob = ds._with_noise(clean, pct, seed=ds.cell_seed("DUFF", 0, pct))
            Yn = np.asarray(prob.holdout["truth_y"], dtype=float)
            sig_hat = observation_scale(Yn)["noise_sigma"]
            ratios.append(sig_hat / (pct / 100.0 * sig_true))
        assert all(1.5 <= r <= 2.2 for r in ratios), ratios

    def test_recompute_36cell_median_matches_frozen_cal_sigma(self):
        """Recompute the 36-cell median from code so that any drift fails loudly."""
        slate = ds.build_slates()
        ratios = []
        for p in ("DUFF", "COUP", "STAB"):
            clean = slate[p]["prob_clean"]
            Yc = np.asarray(clean.holdout["truth_y"], dtype=float)
            sig_true = float(np.sqrt(np.mean(np.std(Yc, axis=0) ** 2)))
            for pct in (1, 3, 5, 10):
                for rep in (0, 1, 2):
                    prob = ds._with_noise(clean, pct, seed=ds.cell_seed(p, rep, pct))
                    s = observation_scale(np.asarray(prob.holdout["truth_y"], dtype=float))
                    ratios.append(s["noise_sigma"] / (pct / 100.0 * sig_true))
        assert len(ratios) == 36
        computed_median = float(np.median(ratios))
        assert round(computed_median, 1) == CAL_SIGMA



class TestCalibrationTransfer:
    def test_inflation_stable_across_problems(self):
        """CAL_SIGMA transfers beyond the grid: pendulum + MSD + van der Pol
        (relaxation-oscillation limit cycle) spot-checks land in [1.5, 2.2].
        An earlier transient VDP load failure never reproduced; pinned by
        test_vdp_loads_finite_regression."""
        import numpy as np

        from apedp_dyn import dyn_sweep as ds
        from apedp_dyn.gates.intervention import observation_scale
        from apedp_dyn.problems import mass_spring_damper, pendulum, van_der_pol

        for mod, name in ((pendulum, "PEND"), (mass_spring_damper, "MSD"),
                          (van_der_pol, "VDP")):
            clean = mod.load()
            Yc = np.asarray(clean.holdout["truth_y"], dtype=float)
            sig_true = float(np.sqrt(np.mean(np.std(Yc, axis=0) ** 2)))
            for pct in (3, 10):
                prob = ds._with_noise(clean, pct, seed=ds.cell_seed(name, 0, pct))
                Yn = np.asarray(prob.holdout["truth_y"], dtype=float)
                ratio = observation_scale(Yn)["noise_sigma"] / (pct / 100.0 * sig_true)
                assert 1.5 <= ratio <= 2.2, (name, pct, ratio)


class TestTauDecomposition:
    def test_amplitude_floor_binds_on_grid(self):
        """Measured over all 36 v4 cells: the REL_FLOOR*RMS term binds 36/36;
        the noise-floor term never binds at n=400. tau_epi is effectively
        amplitude-relative here — pinned so a future n/noise regime that flips
        the balance is visible."""
        import numpy as np

        from apedp_dyn import dyn_sweep as ds
        from apedp_dyn.epistemic.adjudicate import REL_FLOOR
        from apedp_dyn.gates.intervention import observation_scale

        slate = ds.build_slates()
        amp = 0
        for p in ("DUFF", "COUP", "STAB"):
            clean = slate[p]["prob_clean"]
            Yc = np.asarray(clean.holdout["truth_y"], dtype=float)
            rms = float(np.sqrt(np.mean(Yc ** 2)))
            for pct in (1, 3, 5, 10):
                for rep in (0, 1, 2):
                    prob = ds._with_noise(clean, pct, seed=ds.cell_seed(p, rep, pct))
                    s = observation_scale(np.asarray(prob.holdout["truth_y"]))
                    if REL_FLOOR * rms >= s["noise_sigma"] / np.sqrt(s["n_samples"]):
                        amp += 1
        assert amp == 36, amp

    def test_noise_floor_can_bind(self):
        """Unit-level: with tiny n and large sigma, the noise term binds —
        the component is live code, just never binding at grid scale."""
        from apedp_dyn.epistemic.adjudicate import REL_FLOOR, resolution_floor

        assert resolution_floor(0.01, 1.0, 4) == 1.0 / 2.0  # noise binds
        assert resolution_floor(10.0, 0.0, 100) == REL_FLOOR * 10.0  # amplitude binds
