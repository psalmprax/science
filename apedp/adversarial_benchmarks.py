"""APEDP Adversarial Science Benchmark Suite.

A dedicated adversarial benchmark designed to rigorously test whether APEDP
can be fooled by seductive, overfitted, confounded, or unphysical equations.

Evaluates 6 canonical adversarial traps:
  Trap A: Dimensional Decoy (dimensionally homogeneous, but violates parity/positivity).
  Trap B: Overfitting Trap (R^2 ~ 1 on training data, catastrophic on locked holdout).
  Trap C: Extrapolation Trap (fits calibration interval, singular pole in extrapolation).
  Trap D: Confounded Mechanisms (identical observables in low-range -> UNDERDETERMINED + x*).
  Trap E: Synthetic Novelty Trap (rediscovers known law -> KNOWN_MODEL, refuses false novelty).
  Trap F: Hidden Contradiction (plausible constraints with proven mathematical impossibility -> REFUSE).

Design contract
---------------
Every trap invokes the *production* component that is claimed to defend against
it and derives ``passed_audit`` from that component's returned data:

  A -> ``IndependentAdversarialFalsifier`` (positivity invariant +
       ``DimensionalAssuranceEngine`` homogeneity probe)
  B -> ``IndependentAdversarialFalsifier`` empirical axis on a locked holdout
  C -> ``IndependentAdversarialFalsifier`` denominator-root probe
  D -> ``TestableModelPackage.compute_uncertainty_weighted_discrimination``
  E -> ``PriorArtAuditor.audit_novelty``
  F -> ``ContradictoryConstraintEngine.check_incompatibility`` +
       ``IndependentAdversarialFalsifier`` constraint check

``passed_audit`` is therefore a measurement of the running system. When a
component fails to catch its trap, the trap reports ``passed_audit=False``
rather than asserting the expected outcome, and ``refusal_to_overclaim_score``
drops accordingly. Ground truth for each trap is established independently
(real fits, real holdout residuals, real limit arguments) before the system's
verdict is compared against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.epistemic import (
    NoveltyAuditResult,
    NoveltyStatus,
    PriorArtAuditor,
    TestableModelPackage,
)
from apedp.falsification import (
    ContradictoryConstraintEngine,
    FalsificationAxis,
    IndependentAdversarialFalsifier,
)
from apedp.gated_assurance import (
    EpistemicActionStance,
    ScientificImprovementMetric,
    SpecificationFeasibility,
    map_feasibility_to_stance,
)


def _clamp01(value: float) -> float:
    """Clamp a measured quantity into [0.0, 1.0] for metric sub-scores."""
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


class AdversarialTrapType(str, Enum):
    """The 6 canonical adversarial traps testing epistemic integrity."""
    DIMENSIONAL_DECOY = "DIMENSIONAL_DECOY"
    OVERFITTING_TRAP = "OVERFITTING_TRAP"
    EXTRAPOLATION_TRAP = "EXTRAPOLATION_TRAP"
    CONFOUNDED_MECHANISMS = "CONFOUNDED_MECHANISMS"
    SYNTHETIC_NOVELTY_TRAP = "SYNTHETIC_NOVELTY_TRAP"
    HIDDEN_CONTRADICTION = "HIDDEN_CONTRADICTION"


@dataclass(frozen=True)
class AdversarialTrapResult:
    """Outcome of evaluating an adversarial trap."""
    trap_id: str
    trap_type: AdversarialTrapType
    description: str
    passed_audit: bool
    epistemic_stance: EpistemicActionStance
    defense_mechanism: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trap_id": self.trap_id,
            "trap_type": self.trap_type.value,
            "description": self.description,
            "passed_audit": self.passed_audit,
            "epistemic_stance": self.epistemic_stance.value,
            "defense_mechanism": self.defense_mechanism,
            "details": self.details,
        }


class AdversarialScienceBenchmarkSuite:
    """Executes the 6-trap adversarial science benchmark suite.

    Each trap is measured, not asserted: the suite drives the real assurance
    components and records their verdicts plus the evidence behind them.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.falsifier = IndependentAdversarialFalsifier()

    # ------------------------------------------------------------------ Trap A
    def audit_trap_a_dimensional_decoy(self) -> AdversarialTrapResult:
        """Trap A: dimensionally homogeneous kinetic-energy decoy E_k = 0.5*m*v**3/c.

        The decoy carries exactly [J] under SI units, so the dimensional gate
        alone accepts it. It is odd under spatial inversion and therefore
        predicts negative kinetic energy for v < 0.

        Ground truth: kinetic energy must be non-negative on the admitted domain,
        so the model is inadmissible over v in the declared interval.

        Routed through: ``IndependentAdversarialFalsifier`` with the ``positivity``
        structural invariant, plus the ``DimensionalAssuranceEngine`` homogeneity
        probe (which must NOT fire, since homogeneity is what makes this a decoy).
        """
        report = self.falsifier.falsify(
            "0.5 * m_mass * v_phys**3 / c_light",
            variable_domains={"m_mass": (1.0, 10.0), "v_phys": (-10.0, 10.0)},
            physical_invariants=["positivity"],
            variable_units={"m_mass": "kilogram", "v_phys": "meter / second"},
            expected_output_unit="joule",
        )

        positivity = [
            c for c in report.counterexamples if c.probe_name == "PositivityViolationProbe"
        ]
        dimensional = [
            c for c in report.counterexamples
            if c.probe_name in ("DimensionalHomogeneityProbe", "DimensionalInferenceProbe")
        ]

        # Measurement: the dimensional gate must accept the decoy, and the
        # positivity invariant must reject it.
        dimensionally_homogeneous = not dimensional
        violation_detected = bool(positivity)
        passed = dimensionally_homogeneous and violation_detected and report.cannot_repair

        negative_value = positivity[0].metric_value if positivity else None
        return AdversarialTrapResult(
            trap_id="ADV-TRAP-A",
            trap_type=AdversarialTrapType.DIMENSIONAL_DECOY,
            description=(
                "Dimensional decoy E_k = 0.5*m_mass*v_phys**3/c_light satisfies SI "
                "homogeneity ([J]) but is odd in v, so it predicts negative kinetic "
                "energy on the negative-velocity branch"
            ),
            passed_audit=passed,
            epistemic_stance=map_feasibility_to_stance(SpecificationFeasibility.INCONSISTENT),
            defense_mechanism=(
                "IndependentAdversarialFalsifier.PositivityViolationProbe "
                "(DimensionalAssuranceEngine alone accepts the decoy)"
            ),
            details={
                "dimensionally_homogeneous": dimensionally_homogeneous,
                "dimensional_counterexamples": [c.probe_name for c in dimensional],
                "positivity_violation_detected": violation_detected,
                "negative_energy_value": negative_value,
                "system_verdict": report.verdict,
                "cannot_repair": report.cannot_repair,
                "falsification_report": report.to_dict(),
            },
        )

    # ------------------------------------------------------------------ Trap B
    def audit_trap_b_overfitting_trap(self) -> AdversarialTrapResult:
        """Trap B: seductive degree-9 polynomial, R^2 > 0.999 on train, explosive on holdout.

        Ground truth: the clean law is y = 20*x / (2 + x). The holdout is a
        locked partition (interior points plus extrapolation beyond the training
        grid) that must be evaluated without refitting.

        Routed through: the ``IndependentAdversarialFalsifier`` empirical axis
        (``_probe_empirical_fidelity``) on the locked holdout, plus the real
        ``ScientificImprovementMetric`` populated from measured quantities.
        """
        # --- Ground truth and the seductive candidate (real fits, real noise) ---
        x_train = np.linspace(1.0, 8.0, 10)
        y_clean = 20.0 * x_train / (2.0 + x_train)
        y_train = y_clean + self.rng.normal(0, 0.05, size=len(x_train))

        x_holdout = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0, 10.0])
        y_holdout_true = 20.0 * x_holdout / (2.0 + x_holdout)

        coeffs = np.polyfit(x_train, y_train, deg=9)
        poly_pred_train = np.polyval(coeffs, x_train)
        poly_pred_holdout = np.polyval(coeffs, x_holdout)

        ss_tot_train = float(np.sum((y_train - np.mean(y_train)) ** 2))
        r2_train = (
            float(1.0 - np.sum((y_train - poly_pred_train) ** 2) / ss_tot_train)
            if ss_tot_train > 0 else float("nan")
        )
        mse_holdout = float(np.mean((y_holdout_true - poly_pred_holdout) ** 2))

        # --- Route the candidate through the real falsifier on the holdout ---
        x_sym = sp.Symbol("x")
        poly_expr: sp.Expr = sp.Add(*[
            sp.Float(float(c)) * x_sym ** power for power, c in enumerate(reversed(coeffs))
        ])
        report = self.falsifier.falsify(
            poly_expr,
            variable_domains={"x": (float(x_train[0]), float(x_train[-1]))},
            empirical_data={"x": x_holdout, "y": y_holdout_true},
        )

        empirical = [c for c in report.counterexamples if c.axis == FalsificationAxis.EMPIRICAL]
        numerical = [
            c for c in report.counterexamples
            if c.axis in (FalsificationAxis.NUMERICAL, FalsificationAxis.BOUNDARY)
        ]
        empirical_metrics = report.detailed_metrics.get("empirical", {})
        holdout_r2 = float(empirical_metrics.get("r_squared", float("nan")))

        # --- Improvement metric sub-scores derived from measurements, not literals ---
        holdout_coverage = float(np.mean(
            np.abs(poly_pred_holdout - y_holdout_true) / np.abs(y_holdout_true) <= 0.05
        ))
        # Parsimony: 10 free parameters fitted to 10 points leaves no residual
        # degrees of freedom.
        parsimony = _clamp01(1.0 - (len(coeffs) / max(len(x_train), 1)))
        improvement = ScientificImprovementMetric(
            i_fit=_clamp01(holdout_r2),
            i_stability=0.0 if numerical else 1.0,
            i_physicality=1.0,  # no structural invariants were requested for this trap
            i_generality=_clamp01(holdout_coverage),
            i_complexity=parsimony,
            i_falsifiability=1.0 if empirical else 0.0,
        )

        caught_out_of_sample = bool(empirical)
        seductive_in_sample = bool(np.isfinite(r2_train) and r2_train > 0.99)
        passed = caught_out_of_sample and seductive_in_sample and not improvement.is_net_positive

        return AdversarialTrapResult(
            trap_id="ADV-TRAP-B",
            trap_type=AdversarialTrapType.OVERFITTING_TRAP,
            description=(
                "Degree-9 polynomial achieves R^2 > 0.999 on the training grid but "
                "explodes on the locked holdout"
            ),
            passed_audit=passed,
            epistemic_stance=map_feasibility_to_stance(SpecificationFeasibility.INCONSISTENT),
            defense_mechanism=(
                "IndependentAdversarialFalsifier.EmpiricalResidualProbe on the locked "
                "holdout; ScientificImprovementMetric built from measured holdout error"
            ),
            details={
                "r2_train": r2_train,
                "mse_holdout": mse_holdout,
                "holdout_r_squared": holdout_r2,
                "holdout_coverage_within_5pct": holdout_coverage,
                "n_train_points": len(x_train),
                "n_free_parameters": len(coeffs),
                "is_net_positive": improvement.is_net_positive,
                "improvement_metric": improvement.to_dict(),
                "system_verdict": report.verdict,
                "empirical_counterexamples": [c.probe_name for c in empirical],
                "falsification_report": report.to_dict(),
            },
        )

    # ------------------------------------------------------------------ Trap C
    def audit_trap_c_extrapolation_trap(self) -> AdversarialTrapResult:
        """Trap C: a model smooth on its calibration window with an essential pole inside
        the claimed operational envelope: f(x) = 100 / (10 - x) claimed valid on [0, 15].

        Ground truth: the denominator vanishes at x = 10.0, strictly inside the
        claimed domain, so the claimed envelope is invalid.

        Routed through: ``IndependentAdversarialFalsifier`` denominator-root probe
        (``_probe_mathematical_counterexamples``).
        """
        claimed_domain = (0.0, 15.0)
        report = self.falsifier.falsify(
            "100 / (10 - x)",
            variable_domains={"x": claimed_domain},
            critical_boundaries={"x": [10.0]},
        )

        poles = sorted({
            float(c.counterexample_point["x"])
            for c in report.counterexamples
            if c.probe_name == "DenominatorRootProbe" and "x" in c.counterexample_point
        })
        pole_in_claimed_envelope = any(claimed_domain[0] <= p <= claimed_domain[1] for p in poles)
        passed = pole_in_claimed_envelope and not report.survived

        return AdversarialTrapResult(
            trap_id="ADV-TRAP-C",
            trap_type=AdversarialTrapType.EXTRAPOLATION_TRAP,
            description=(
                "Model f(x) = 100/(10-x) is regular on the calibration window [0, 8] "
                "but has a fatal pole at x = 10 inside the claimed envelope [0, 15]"
            ),
            passed_audit=passed,
            epistemic_stance=map_feasibility_to_stance(SpecificationFeasibility.OUT_OF_DOMAIN),
            defense_mechanism=(
                "IndependentAdversarialFalsifier.DenominatorRootProbe against the "
                "claimed operational envelope"
            ),
            details={
                "poles_detected": poles,
                "claimed_domain": claimed_domain,
                "pole_in_claimed_envelope": pole_in_claimed_envelope,
                "system_verdict": report.verdict,
                "falsification_report": report.to_dict(),
            },
        )

    # ------------------------------------------------------------------ Trap D
    def audit_trap_d_confounded_mechanisms(self) -> AdversarialTrapResult:
        """Trap D: two distinct mechanisms indistinguishable inside the observed window.
        y1(t) = 1 / (1 + 0.1*t)  (second-order kinetics)
        y2(t) = exp(-0.1*t)       (first-order kinetics)

        Ground truth: on t in [0, 0.5] the two laws differ by less than the
        declared sensor noise, so the data cannot separate them; the separating
        experiment lies outside the observed window.

        Routed through: ``TestableModelPackage.compute_uncertainty_weighted_discrimination``,
        which returns the maximal-discrimination coordinate x* and its SNR.
        """
        sensor_noise = 0.05

        t_low = np.linspace(0.0, 0.5, 20)
        y1_low = 1.0 / (1.0 + 0.1 * t_low)
        y2_low = np.exp(-0.1 * t_low)
        max_discrepancy_low = float(np.max(np.abs(y1_low - y2_low)))

        t_grid = np.linspace(0.0, 20.0, 500)
        t_star, snr_star, raw_star = TestableModelPackage.compute_uncertainty_weighted_discrimination(
            eval_f1=lambda t: 1.0 / (1.0 + 0.1 * t),
            eval_f2=lambda t: np.exp(-0.1 * t),
            x_grid=t_grid,
            uncertainty_fn=lambda t: np.full_like(t, sensor_noise),
        )

        # Measurement: the observed window cannot separate the mechanisms, and the
        # real active-experiment search finds a separating point outside it.
        indistinguishable_in_window = max_discrepancy_low < sensor_noise
        separating_experiment_exists = t_star > 5.0 and snr_star > 2.0
        passed = indistinguishable_in_window and separating_experiment_exists

        return AdversarialTrapResult(
            trap_id="ADV-TRAP-D",
            trap_type=AdversarialTrapType.CONFOUNDED_MECHANISMS,
            description=(
                "First- vs second-order decay are indistinguishable on t in [0, 0.5] "
                f"(max discrepancy {max_discrepancy_low:.2e} < noise {sensor_noise})"
            ),
            passed_audit=passed,
            epistemic_stance=map_feasibility_to_stance(SpecificationFeasibility.UNDERDETERMINED),
            defense_mechanism=(
                "TestableModelPackage.compute_uncertainty_weighted_discrimination: "
                "reports underdetermination and returns the maximal-discrimination x*"
            ),
            details={
                "max_discrepancy_low_window": max_discrepancy_low,
                "sensor_noise": sensor_noise,
                "indistinguishable_in_window": indistinguishable_in_window,
                "optimal_discrimination_t_star": t_star,
                "max_snr_at_t_star": snr_star,
                "raw_discrepancy_at_t_star": raw_star,
            },
        )

    # ------------------------------------------------------------------ Trap E
    def audit_trap_e_synthetic_novelty_trap(self) -> AdversarialTrapResult:
        """Trap E: an unlabelled synthesis that rediscovers the Langmuir /
        Michaelis-Menten law must not be reported as novel.

        Ground truth: v = 15*x / (2.5 + x) is algebraically the classical rational
        saturation family, catalogued in the reference corpus.

        Routed through: ``PriorArtAuditor.audit_novelty``.
        """
        novelty_result: NoveltyAuditResult = PriorArtAuditor.audit_novelty(
            "15.0 * x / (2.5 + x)", variable_name="x"
        )
        known = novelty_result.status == NoveltyStatus.KNOWN_MODEL
        passed = known

        return AdversarialTrapResult(
            trap_id="ADV-TRAP-E",
            trap_type=AdversarialTrapType.SYNTHETIC_NOVELTY_TRAP,
            description=(
                "Synthesized candidate v = 15*x/(2.5+x) audited against prior art to "
                "prevent a false novelty claim"
            ),
            passed_audit=passed,
            # Correctly classifying a rediscovery is a successful solve, not a refusal.
            epistemic_stance=map_feasibility_to_stance(SpecificationFeasibility.SOLVABLE),
            defense_mechanism=(
                "PriorArtAuditor.audit_novelty: refused false novelty and matched the "
                "rational saturation family"
            ),
            details=novelty_result.to_dict(),
        )

    # ------------------------------------------------------------------ Trap F
    def audit_trap_f_hidden_contradiction(self) -> AdversarialTrapResult:
        """Trap F: innocent-looking constraints that are jointly unsatisfiable.

        Declared requirements:
          (1) contact equilibrium,  lim_{x->0+} f(x) = 0
          (2) strict inverse-square lower bound,  f(x) >= k/x**2 for all x > 0

        Ground truth (established independently below): as x -> 0+, requirement
        (2) forces f(x) >= k/x**2 -> +inf, which contradicts requirement (1). The
        admissible function space is empty, so no regularization can repair it.

        Routed through: ``ContradictoryConstraintEngine.check_incompatibility`` via
        ``IndependentAdversarialFalsifier.falsify(..., constraints=...)``.
        """
        constraints = [
            "contact_zero: lim_{x->0+} f(x) = 0",
            "strict_lower_bound: f(x) >= k/x**2 for all x > 0",
        ]

        # Independent establishment of ground truth: evaluate the imposed lower
        # bound on a shrinking sequence and confirm it diverges while the contact
        # requirement holds the value at 0.
        k = 1.0
        shrinking_x = np.array([1e-2, 1e-3, 1e-4, 1e-6])
        lower_bounds = k / shrinking_x ** 2
        diverges = bool(np.all(np.diff(lower_bounds) > 0)) and bool(
            np.isfinite(lower_bounds).all() and lower_bounds[-1] > 1e6
        )
        contact_requires_zero = True  # requirement (1), by definition
        is_mathematically_impossible = diverges and contact_requires_zero

        certificate = ContradictoryConstraintEngine.check_incompatibility(constraints)
        report = self.falsifier.falsify(
            "1 / x",
            variable_domains={"x": (0.1, 10.0)},
            constraints=constraints,
        )

        system_refused = (
            certificate is not None
            and certificate.is_infeasible
            and report.verdict == "REFUSED_INCOMPATIBLE_CONSTRAINTS"
        )
        passed = system_refused and is_mathematically_impossible

        return AdversarialTrapResult(
            trap_id="ADV-TRAP-F",
            trap_type=AdversarialTrapType.HIDDEN_CONTRADICTION,
            description=(
                "Specification {lim_{x->0+} f(x) = 0, f(x) >= k/x**2 on x > 0} is "
                "mathematically incompatible"
            ),
            passed_audit=passed,
            epistemic_stance=map_feasibility_to_stance(SpecificationFeasibility.INCONSISTENT),
            defense_mechanism=(
                "ContradictoryConstraintEngine.check_incompatibility: issued a "
                "non-existence proof and refused the specification"
            ),
            details={
                "is_mathematically_impossible": is_mathematically_impossible,
                "lower_bound_samples": {
                    f"{x_val:g}": float(bound) for x_val, bound in zip(shrinking_x, lower_bounds)
                },
                "system_refused": system_refused,
                "infeasibility_certificate": certificate.to_dict() if certificate else None,
                "system_verdict": report.verdict,
                "falsification_report": report.to_dict(),
            },
        )

    # --------------------------------------------------------------- Aggregation
    def run_all_traps(self) -> dict[str, Any]:
        """Runs the complete 6-trap adversarial suite and computes Refusal-to-Overclaim Score.

        The score is the fraction of traps whose defence was *verified against the
        running system*; it is not fixed by construction.
        """
        traps = [
            self.audit_trap_a_dimensional_decoy(),
            self.audit_trap_b_overfitting_trap(),
            self.audit_trap_c_extrapolation_trap(),
            self.audit_trap_d_confounded_mechanisms(),
            self.audit_trap_e_synthetic_novelty_trap(),
            self.audit_trap_f_hidden_contradiction(),
        ]

        total_traps = len(traps)
        passed_traps = sum(1 for t in traps if t.passed_audit)
        refusal_to_overclaim_score = float(passed_traps / total_traps) if total_traps > 0 else 0.0

        return {
            "total_traps": total_traps,
            "passed_traps": passed_traps,
            "failed_traps": total_traps - passed_traps,
            "failed_trap_ids": [t.trap_id for t in traps if not t.passed_audit],
            "refusal_to_overclaim_score": refusal_to_overclaim_score,
            "all_traps_defended": passed_traps == total_traps,
            "results": [t.to_dict() for t in traps],
        }
