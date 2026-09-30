"""Phase 0: Zero-False-Promotion Hardening Battery.

Core Acceptance Criterion:
    No UNIQUE promotion without satisfying every mandatory evidence gate.

Tests adversarial failure modes and edge cases to ensure the adjudication
layer makes false promotion structurally impossible:
  1. Sole survivor with poor fidelity (rejected by Guard 1 absolute ceiling).
  2. Sole survivor with structured residuals (rejected by Guard 2 whiteness flag).
  3. High noise / swamped signal (holdout resolution floor enforces MULTI_HYPOTHESIS).
  4. Near-collinear mechanisms (unidentifiability enforces MULTI_HYPOTHESIS).
  5. Extrapolation / near-OOD gate 4 failure (no promotion of least-bad models).
  6. Misspecified candidate pool (all models inadequate; refused UNIQUE).
  7. Structurally equivalent functional forms (gauge equivalence; refused UNIQUE).
  8. Disparate parameter scales (scale-free resolution floor invariance).
  9. Point-estimate RMSE winner lacking epistemic margin (refused UNIQUE).
  10. Bootstrap confidence interval straddling zero (flipped to MULTI_HYPOTHESIS).
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    EpistemicStatus,
    adjudicate,
    resolution_floor,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _make_candidate(
    candidate_id: str,
    term_names: tuple[str, ...],
    train_loss: float,
    holdout_rmse: float,
    holdout_residuals: list[float] | None = None,
    obs_rms: float = 1.0,
    noise_sigma: float = 0.05,
    n_samples: int = 100,
    passed_gate4: bool = True,
    data_origin: str = "simulation",
) -> ScoredCandidate:
    terms = tuple(Term(name, f"p{i}") for i, name in enumerate(term_names))
    c = CandidateDynamics(lhs="y", terms=terms, candidate_id=candidate_id)
    g4 = {
        "passed": passed_gate4,
        "holdout_obs_rms": obs_rms,
        "holdout_noise_sigma": noise_sigma,
        "holdout_n_samples": n_samples,
        "rmse": holdout_rmse,
        "holdout_residuals": holdout_residuals,
        "holdout_sq_error": [r ** 2 for r in holdout_residuals] if holdout_residuals else None,
    }
    return ScoredCandidate(c, {"data_origin": data_origin, "gate4_status": g4},
                           train_loss=train_loss, holdout_rmse=holdout_rmse)


class TestZeroFalsePromotionHardening:
    def test_sole_survivor_poor_fidelity_rejected_by_guard1(self):
        """Sole survivor whose holdout RMSE exceeds Guard 1 (3.0*sigma/CAL) must be rejected."""
        # sigma = 0.06 -> ceiling = 3.0 * (0.06 / 1.8) = 0.10. RMSE = 0.25 fails.
        c = _make_candidate("sole_poor", ("one", "x"), train_loss=0.01, holdout_rmse=0.25,
                            noise_sigma=0.06, n_samples=200)
        adj = adjudicate([c], sole_survivor_rule=True)
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.promoted is None
        assert "failed the pre-specified absolute holdout-fidelity contract" in adj.reason

    def test_sole_survivor_structured_residuals_rejected_by_guard2(self):
        """Sole survivor with low RMSE but structured residuals (|rho1| > 0.5) must be rejected."""
        # Clean RMSE 0.05 passes Guard 1 (ceiling 0.10).
        # Residuals have strong AR(1) structure: rho1 ~ 0.85.
        rng = np.random.RandomState(42)
        n = 200
        structured_res = [0.0] * n
        for i in range(1, n):
            structured_res[i] = 0.85 * structured_res[i - 1] + rng.normal(0, 0.02)

        c = _make_candidate("sole_structured", ("one", "x"), train_loss=0.01, holdout_rmse=0.05,
                            holdout_residuals=structured_res, noise_sigma=0.06, n_samples=n)
        adj = adjudicate([c], resid_sole=structured_res, sole_survivor_rule=True)
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.promoted is None
        assert "carries structured holdout residuals" in adj.reason

    def test_high_noise_swamped_regime_forces_multi_hypothesis(self):
        """When noise swamps observational resolution, models cannot be uniquely separated."""
        # obs_rms = 1.0, noise_sigma = 0.80 -> tau_epi ~ 0.05 * 1.0 = 0.05.
        # Two models differ by Delta = 0.01 < tau_epi.
        c1 = _make_candidate("model_a", ("one", "x"), train_loss=0.50, holdout_rmse=0.81,
                             noise_sigma=0.80, obs_rms=1.0)
        c2 = _make_candidate("model_b", ("one", "x2"), train_loss=0.51, holdout_rmse=0.82,
                             noise_sigma=0.80, obs_rms=1.0)
        adj = adjudicate([c1, c2])
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None
        assert "observationally indistinguishable" in adj.reason

    def test_near_collinear_mechanisms_force_multi_hypothesis(self):
        """Distinct functional families that are collinear on sample data cannot earn UNIQUE_SUPPORT."""
        # Delta = 0.002, tau = 0.05 (obs_rms = 1.0).
        c_mech1 = _make_candidate("grain_dispersion", ("one", "d_inv_half"), train_loss=0.02,
                                  holdout_rmse=0.040, obs_rms=1.0)
        c_mech2 = _make_candidate("zener_pinning", ("one", "phi_half"), train_loss=0.021,
                                  holdout_rmse=0.042, obs_rms=1.0)
        adj = adjudicate([c_mech1, c_mech2])
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None

    def test_extrapolation_ood_gate4_failure_refuses_promotion(self):
        """Candidate with great training fit but failed gate 4 must not be promoted."""
        c_failed_g4 = _make_candidate("overfit_cheat", ("one", "x", "x3", "x5"),
                                      train_loss=0.0001, holdout_rmse=10.5,
                                      passed_gate4=False)
        adj = adjudicate([c_failed_g4])
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.promoted is None
        assert "no promotion of least-bad models" in adj.reason

    def test_misspecified_grammar_abstains(self):
        """When all candidates in a misspecified pool perform poorly, none earn UNIQUE_SUPPORT."""
        # True physics is cubic, but grammar only offers constant and linear.
        # Both fail fidelity ceiling or resolution.
        c_lin = _make_candidate("linear_bad", ("one", "x"), train_loss=0.40, holdout_rmse=0.45,
                                noise_sigma=0.03, obs_rms=1.0)
        # Ceiling = 3 * (0.03 / 1.8) = 0.05. RMSE 0.45 >> 0.05 -> INSUFFICIENT.
        adj_sole = adjudicate([c_lin], sole_survivor_rule=True)
        assert adj_sole.status == EpistemicStatus.INSUFFICIENT_EVIDENCE

    def test_structurally_equivalent_functional_forms_refuse_unique(self):
        """Two mathematically equivalent functional forms have identical holdout RMSE (Delta = 0);
        APEDP must refuse UNIQUE_SUPPORT and return MULTI_HYPOTHESIS."""
        c_form1 = _make_candidate("sin_phase", ("sin_x_plus_phase",), train_loss=0.05,
                                  holdout_rmse=0.050, obs_rms=1.0)
        c_form2 = _make_candidate("cos_form", ("cos_x",), train_loss=0.05,
                                  holdout_rmse=0.050, obs_rms=1.0)
        adj = adjudicate([c_form1, c_form2])
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None
        assert adj.delta_rmse == 0.0

    def test_disparate_parameter_scales_maintain_scale_invariance(self):
        """Micro-scale (1e-6) and macro-scale (1e6) signals are assessed scale-invariantly."""
        # Micro problem: obs_rms = 1e-6, noise_sigma = 5e-8
        tau_micro = resolution_floor(1e-6, 5e-8, 100)
        # Macro problem: obs_rms = 1e6, noise_sigma = 5e4
        tau_macro = resolution_floor(1e6, 5e4, 100)
        # Relative floors must match exactly
        assert abs(tau_micro / 1e-6 - tau_macro / 1e6) < 1e-12

    def test_rmse_winner_lacking_margin_abstains(self):
        """Point-estimate winner with margin smaller than resolution floor must abstain."""
        # obs_rms = 2.0 -> tau_epi = 0.05 * 2.0 = 0.10.
        # Delta = 0.03 <= tau_epi.
        c_best = _make_candidate("slight_winner", ("one", "x"), train_loss=0.08,
                                 holdout_rmse=0.12, obs_rms=2.0)
        c_rival = _make_candidate("close_rival", ("one", "x2"), train_loss=0.085,
                                  holdout_rmse=0.15, obs_rms=2.0)
        adj = adjudicate([c_best, c_rival])
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None

    def test_bootstrap_ci_straddling_zero_flips_to_multi(self):
        """Point margin Delta > tau_epi that does not survive block bootstrap must flip to MULTI."""
        # Model 1 has lower overall RMSE (Delta = 0.07 > tau = 0.05),
        # but errors alternate across blocks so bootstrap CI straddles zero.
        n = 100
        # Alternating residuals: on even blocks model 1 is better, on odd blocks model 2 is better
        r_best = np.zeros(n)
        r_rival = np.zeros(n)
        for i in range(n):
            if (i // 10) % 2 == 0:
                r_best[i] = 0.02
                r_rival[i] = 0.40
            else:
                r_best[i] = 0.30
                r_rival[i] = 0.02

        rmse_1 = float(np.sqrt(np.mean(r_best ** 2)))
        rmse_2 = float(np.sqrt(np.mean(r_rival ** 2)))

        c1 = _make_candidate("model_1", ("one", "x"), train_loss=0.02, holdout_rmse=rmse_1,
                             holdout_residuals=r_best.tolist(), obs_rms=1.0)
        c2 = _make_candidate("model_2", ("one", "x3"), train_loss=0.03, holdout_rmse=rmse_2,
                             holdout_residuals=r_rival.tolist(), obs_rms=1.0)

        adj = adjudicate([c1, c2], resid_best=r_best ** 2, resid_rival=r_rival ** 2,
                         bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=42)
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None
        assert adj.bootstrap_branch == "multi"
        assert adj.bootstrap_lo is not None and adj.bootstrap_lo <= 0.0
