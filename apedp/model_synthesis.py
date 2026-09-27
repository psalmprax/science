"""Autonomous Model & Equation Synthesis Engine for APEDP 4.0.

Implements structural synthesis of multi-regime physical laws across:
- Class S1: Multi-Regime Rational-Exponential Envelopes
- Class S2: Logarithmic & Boundary-Layer Corrections
- Class S3: Critical Exponents & Avoided Poles
- Class S4: Modulated Dissipative & Oscillatory Transport
- Class S5: Epistemic Abstention & Out-of-Grammar Adversaries

Architecture:
1. Multi-Regime Signature Detection: Identifies power-law onset, rational turnover,
   monotonic logarithmic growth, critical scaling divergence, and oscillatory transport.
2. Structural Composition: Evolves and composes candidate mathematical structures
   without pre-seeded formula templates.
3. Tier 1 Rust Acceleration: Dense singularity probing in sampling gaps, dimensional
   consistency enforcement, and matrix conditioning diagnostics.
4. Tier 2 Optimization: Separable parameter estimation via VarPro and bounded L-BFGS-B.
5. Stage D Continuous Refinement: Continuously optimizes structural exponents and
   frequencies (power p, rational p/q, decay k, frequency w) from coarse-grid starts.
   Accepts only finite, bound-interior, validation-improving refinements; otherwise
   the coarse candidate stands. Never promotes proof status (stays P0).
6. Stage C Epistemic Adjudication: Formally verifies identifiability, residual pattern
   randomness, OOD stability, and decisive epistemic abstention on adversaries.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import numpy as np
import scipy.optimize as opt
import sympy as sp

from apedp.decision.stage_c import (
    CandidateProfile,
    EpistemicState,
    ModelClass,
    StageCDecisionEngine,
)
from apedp.rust_bridge import RustKernelBridge

logger = logging.getLogger("apedp.model_synthesis")


class SynthesisDiagnostics(TypedDict, total=False):
    """Typed diagnostics payload for SynthesizedModel (replaces untyped dict)."""

    template: str
    family: str
    val_r2: float
    residual_autocorr: float
    rationale: str
    stage_d_refined: bool
    stage_d_form: str
    stage_d_params: dict[str, float]
    coarse_val_mse: float
    coarse_expression: str


@dataclass(frozen=True)
class SynthesizedModel:
    """Represents an autonomously synthesized physical equation candidate.

    Empirical confidence (epistemic_class C1..C5) is independent of proof
    standing (proof_claim P0..P2). Finite-sample synthesis never exceeds P0;
    P1/P2 are reserved for a future proof pipeline. Joint label cp_label
    (e.g. "C1,P0") must be used in math reports to prevent C1=proven misread.
    """
    expression_str: str
    sympy_expr: sp.Expr
    parameters: dict[str, float]
    train_mse: float
    val_mse: float
    near_ood_mse: float | None = None
    far_ood_mse: float | None = None
    condition_number: float = 1.0
    has_singularity_risk: bool = False
    epistemic_state: EpistemicState | None = None
    structural_family: str = "unknown"
    diagnostics: SynthesisDiagnostics = field(default_factory=SynthesisDiagnostics)
    proof_claim: str = "P0"

    @property
    def epistemic_class(self) -> str:
        """Maps Stage C EpistemicState to formal tournament epistemic class C1..C5."""
        if self.epistemic_state == EpistemicState.UNIQUE_SUPPORT:
            return "C1"
        elif self.epistemic_state == EpistemicState.MULTI_HYPOTHESIS:
            return "C2"
        elif self.epistemic_state == EpistemicState.INSUFFICIENT_EVIDENCE:
            return "C3"
        elif self.epistemic_state == EpistemicState.NO_VALID_CANDIDATE:
            return "C5"
        return "C3"

    @property
    def cp_label(self) -> str:
        """Joint empirical/proof label, e.g. "C1,P0"."""
        return f"{self.epistemic_class},{self.proof_claim}"


# Stage D amplitude/rate roles for input-normalization map-back:
# P_true = P_tilde / x_scale**e, where e is either a constant (rates and
# frequencies, which carry 1/x dimension) or the name of the form's fitted
# exponent parameter holding e's value.
_XSCALE_ROLES: dict[str, dict[str, str | float]] = {
    "power": {"A": "p"},
    "exp": {"k": 1.0},
    "undamped": {"w": 1.0},
    "damped": {"k": 1.0, "w": 1.0},
    "rational": {"A": "p", "B": "q"},
    "rational_saturation": {"A": "p", "B": "q"},
    "envelope": {"A": "p", "B": "q", "C": "r"},
    "exp_env": {"A": "p", "C": "r"},
    "logbound": {"A": "a", "B": "b"},
}


class AutonomousModelSynthesizer:
    """Synthesizes novel mathematical structures by composing physical basis operators."""

    def __init__(
        self,
        domain_gaps: Sequence[tuple[float, float]] | None = None,
        max_candidates: int = 500,
        random_seed: int = 42,
    ):
        self.domain_gaps = list(domain_gaps or [])
        self.max_candidates = max_candidates
        self.rng = np.random.default_rng(random_seed)
        self.rust_kernel = RustKernelBridge()
        self.stage_c = StageCDecisionEngine()
        from apedp.symbolic_equivalence import SymbolicEquivalenceEngine
        self._equiv_engine = SymbolicEquivalenceEngine()

    def _model_complexity(self, model: SynthesizedModel) -> float:
        """Computes structural complexity based on canonical form rather than raw parameter dict length."""
        try:
            cf = self._equiv_engine.canonicalize(model.expression_str, known_variables=["x"])
            return float(cf.complexity)
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            return float(len(model.parameters))

    def detect_regimes(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Detects qualitative physical regimes: onset, turnover, growth, divergence, or oscillations."""
        idx_sort = np.argsort(x)
        xs = x[idx_sort]
        ys = y[idx_sort]
        n = len(xs)

        is_underspecified = bool(n <= 8)
        if is_underspecified:
            return {
                "n_samples": n,
                "is_underspecified": True,
                "alpha_est": 1.0,
                "has_turnover": False,
                "x_peak": float(xs[-1]),
                "is_decaying": False,
                "is_oscillatory": False,
                "has_critical_divergence": False,
                "xc_est": None,
                "critical_edge_ratio": None,
                "is_monotonic_growth": False,
                "has_shock_discontinuity": False,
                "zero_crossings": 0,
            }

        # 1. Sign changes / Oscillations (Class S4)
        has_negatives = bool(np.min(ys) < -0.05 * np.max(np.abs(ys)))
        zero_crossings = int(np.sum(np.diff(np.sign(ys)) != 0)) if has_negatives else 0
        is_oscillatory = bool(has_negatives and zero_crossings >= 1)

        # 2. Critical divergence & sampling gaps (Class S3)
        dx = np.diff(xs)
        has_gap = len(dx) > 0 and bool(np.max(dx) > 2.5 * np.median(dx))
        has_critical_divergence = False
        xc_est = None

        if has_gap or self.domain_gaps:
            if self.domain_gaps:
                g_min, g_max = self.domain_gaps[0]
                xc_cand = (g_min + g_max) / 2.0
            else:
                idx_gap = int(np.argmax(dx))
                g_min, g_max = xs[idx_gap], xs[idx_gap + 1]
                xc_cand = (g_min + g_max) / 2.0

            near_left = xs < g_min
            near_right = xs > g_max
            if np.any(near_left) and np.any(near_right):
                y_left_edge = ys[near_left][-1]
                y_right_edge = ys[near_right][0]
                med_y = float(np.median(ys))
                if y_left_edge > med_y and y_right_edge > med_y:
                    has_critical_divergence = True
                    xc_est = float(round(xc_cand, 2))
                    # Edge elevation ratio: how strongly the data rises toward
                    # the gap. True divergences (S21) show large ratios; random
                    # subsample-spacing artifacts sit near 1.0. Recorded so the
                    # template proposer can union S2 candidates on weak evidence
                    # instead of catastrophically excluding the true family.
                    denom = abs(med_y) if med_y != 0.0 else 1.0
                    critical_edge_ratio = float(min(y_left_edge, y_right_edge) / denom)
                else:
                    critical_edge_ratio = None
            else:
                critical_edge_ratio = None
        else:
            critical_edge_ratio = None

        # 3. Initial power law onset
        k_near = max(int(n * 0.25), 4)
        x_near, y_near = xs[:k_near], ys[:k_near]
        pos_mask = (x_near > 0) & (y_near > 0)
        if np.sum(pos_mask) >= 3:
            p = np.polyfit(np.log(x_near[pos_mask]), np.log(y_near[pos_mask]), 1)
            alpha_est = float(p[0])
        else:
            alpha_est = 1.0

        # 4. Check for turnover / peak (Class S1)
        peak_idx = int(np.argmax(ys))
        has_turnover = bool(0 < peak_idx < n - 1 and ys[-1] < 0.7 * ys[peak_idx])
        x_peak = float(xs[peak_idx]) if has_turnover else float(xs[-1])

        # If it has a turnover, it is an envelope (Class S1), NOT a critical scaling divergence (Class S3)
        if has_turnover:
            has_critical_divergence = False
            xc_est = None

        # 5. Far regime asymptotic decay
        k_far = max(int(n * 0.3), 4)
        _x_far, y_far = xs[-k_far:], ys[-k_far:]
        is_decaying = bool(float(np.mean(np.diff(y_far))) < 0.0)


        # 6. Monotonic growth (Class S2: Boundary layer)
        is_monotonic_growth = bool(
            (not has_turnover)
            and (not is_oscillatory)
            and (not has_critical_divergence)
            and (ys[-1] >= 0.7 * np.max(ys))
        )

        # 7. Shock discontinuity check (Class S5)
        dx_eff = np.maximum(dx, np.median(dx) * 0.2)
        slopes = np.diff(ys) / dx_eff
        slope_jumps = np.abs(np.diff(slopes))
        y_range = float(np.ptp(ys)) + 1e-12
        max_dy = float(np.max(np.abs(np.diff(ys))))
        has_shock_discontinuity = bool(
            len(slope_jumps) > 0
            and np.max(slope_jumps) > 5.0 * (np.median(slope_jumps) + 1e-3)
            and np.max(slope_jumps) > 5.0
            and max_dy > 0.15 * y_range
        )


        return {
            "n_samples": n,
            "is_underspecified": bool(is_underspecified),
            "alpha_est": float(alpha_est),
            "has_turnover": bool(has_turnover),
            "x_peak": float(x_peak),
            "is_decaying": bool(is_decaying),
            "is_oscillatory": bool(is_oscillatory),
            "has_critical_divergence": bool(has_critical_divergence),
            "xc_est": xc_est,
            "critical_edge_ratio": critical_edge_ratio,
            "is_monotonic_growth": bool(is_monotonic_growth),
            "has_shock_discontinuity": bool(has_shock_discontinuity),
            "zero_crossings": int(zero_crossings),
        }

    @staticmethod
    def _s2_templates() -> list[tuple[str, str]]:
        """Class S2 monotonic-growth template set (power / log-boundary / rational)."""
        structures: list[tuple[str, str]] = []
        alphas = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        betas = [0.5, 1.0, 1.5, 2.0]
        for a in alphas:
            for b in betas:
                if a == 0.0:
                    structures.append((f"A * log(1.0 + B * (x**{b}))", "logarithmic_boundary_layer"))
                else:
                    structures.append((f"A * (x**{a}) * log(1.0 + B * (x**{b}))", "logarithmic_boundary_layer"))
                    structures.append((f"(A * (x**{a})) / (1.0 + B * (x**{b}))", "rational_transition"))
            if a > 0.0:
                structures.append((f"A * (x**{a})", "power_law"))
        structures.append(("A * log(x) + B", "logarithmic_boundary_layer"))
        structures.append(("A * x + B", "linear_baseline"))
        structures.append(("A * x**2 + B * x + C", "polynomial_baseline"))
        # Cubic/quartic reach (P03/P04-class): higher-degree polynomials are
        # genuine grammar, but each extra degree is an overfitting liability at
        # finite noise. They compete on equal terms; parsimony tiebreak, MDL,
        # and Stage C dispose. Not extended further: degree >= 5 has no
        # identified use case and the combinatorial cost is unjustified.
        structures.append(("A * x**3 + B * x**2 + C * x + D", "polynomial_baseline"))
        structures.append(("A * x**4 + B * x**3 + C * x**2 + D * x + F", "polynomial_baseline"))
        # Monotonic exponential envelopes: A * x**p * exp(-C * x**r) is
        # monotone over finite positive domains for small C yet lives in no
        # other branch (S1 proposes it only when S2/S3/S4 all miss). Without
        # these, envelope truths (C41-class) lose to rational surrogates by
        # proposal absence, not by evidence. Bounded subset; fitted rate C and
        # Stage D refine it.
        for _o in ("x**1.0", "x**1.5", "x**2.0"):
            for _e in ("exp(-C * x)", "exp(-C * x**0.5)"):
                structures.append((f"A * {_o} * {_e}", "exponential_envelope"))
        return structures

    @staticmethod
    def _exp_templates() -> list[tuple[str, str]]:
        """Pure-exponential template subset (S1 extinctions with unit onset).

        Used for decaying data whose critical flag is weak: the true family may
        be exponential decay, which neither S3 nor S2 proposes. Small fixed set;
        the continuous rate C is fitted, Stage D refines it.
        """
        return [
            ("A * exp(-C * x)", "exponential_decay"),
            ("A * exp(-C * x**0.5)", "exponential_decay"),
            ("A * exp(-C * x**1.5)", "exponential_decay"),
            ("A * exp(-C * x**2)", "exponential_decay"),
        ]

    def generate_candidate_structures(self, regime_info: dict[str, Any]) -> list[tuple[str, str]]:
        """Generates candidate (structure_template, family_label) tuples based on detected regimes."""
        if regime_info.get("is_underspecified", False):
            return []

        structures: list[tuple[str, str]] = []

        # Class S4: Oscillatory / Modulated Transport
        if regime_info.get("is_oscillatory", False):
            omegas = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0]
            for w in omegas:
                structures.append((f"A * exp(-B * x) * cos({w} * x)", "damped_oscillation"))
                structures.append((f"A * cos({w} * x)", "undamped_oscillation"))
            structures.append(("A * exp(-B * x)", "exponential_decay"))
            structures.append(("A * x + B", "linear_baseline"))
            # Polynomial baselines: zero-crossing trend data (e.g. cubic with
            # real roots) trips the oscillation detector without oscillating.
            # Without these, such data has no polynomial representation at all.
            structures.append(("A * x**2 + B * x + C", "polynomial_baseline"))
            structures.append(("A * x**3 + B * x**2 + C * x + D", "polynomial_baseline"))
            structures.append(("A * x**4 + B * x**3 + C * x**2 + D * x + F", "polynomial_baseline"))
            return structures

        # Class S3: Critical Scaling Divergence
        if regime_info.get("has_critical_divergence", False) and regime_info.get("xc_est") is not None:
            xc = regime_info["xc_est"]
            gammas = [0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.7, 0.75, 0.8, 1.0]
            for g in gammas:
                structures.append((f"A * (abs(x - {xc}))**(-{g}) + B * x", "critical_scaling_divergence"))
                structures.append((f"A * (abs(x - {xc}))**(-{g})", "critical_scaling_divergence"))
            structures.append(("A * x + B", "linear_baseline"))
            structures.append(("A * x**2 + B * x + C", "polynomial_baseline"))
            # Divergence-evidence safety net: the S3-only pool is justified only
            # when the gap reflects declared physical knowledge (explicit
            # domain_gaps) AND strong edge elevation. Auto-detected spacing
            # gaps (e.g. shuffled splits) or weak elevation may be artifacts;
            # in those cases union the S2 monotonic set (+ exponentials when
            # decaying) so a false positive cannot exclude the true family.
            # Selection (parsimony / MDL / Stage C) disposes among the union;
            # true divergences still win on validation.
            edge_ratio = regime_info.get("critical_edge_ratio")
            strong = edge_ratio is not None and edge_ratio >= 1.5
            if not (self.domain_gaps and strong):
                structures.extend(self._s2_templates())
                if regime_info.get("is_decaying", False):
                    structures.extend(self._exp_templates())
            return structures

        # Class S2: Monotonic Growth / Logarithmic Boundary Layer / Power Law
        if regime_info.get("is_monotonic_growth", False):
            structures.extend(self._s2_templates())
            return structures

        # Class S1: Multi-Regime Rational-Exponential Envelopes (and general turnover / decay)
        onsets = ["1.0", "x**0.5", "x**0.75", "x", "x**1.25", "x**1.5", "x**2"]

        saturations = [
            "1.0",
            "1.0 + B * x",
            "1.0 + B * x**1.5",
            "1.0 + B * x**2",
            "1.0 + B * x**2.5",
        ]

        extinctions = [
            "1.0",
            "exp(-C * x)",
            "exp(-C * x**0.5)",
            "exp(-C * x**1.5)",
            "exp(-C * x**2)",
        ]

        for o in onsets:
            for s in saturations:
                for e in extinctions:
                    if s == "1.0" and e == "1.0":
                        if o != "1.0":
                            structures.append((f"A * {o}", "power_law"))
                    elif s == "1.0":
                        if o == "1.0":
                            structures.append((f"A * {e}", "exponential_decay"))
                        else:
                            structures.append((f"A * {o} * {e}", "exponential_envelope"))
                    elif e == "1.0":
                        if o == "1.0":
                            structures.append((f"A / ({s})", "rational_saturation"))
                        else:
                            structures.append((f"A * {o} / ({s})", "rational_transition"))
                    else:
                        if o == "1.0":
                            structures.append((f"(A / ({s})) * {e}", "rational_exponential_envelope"))
                        else:
                            structures.append((f"(A * {o} / ({s})) * {e}", "rational_exponential_envelope"))

        structures.append(("A * x + B", "linear_baseline"))
        # Cubic/quartic polynomials can carry turnover-compatible shapes;
        # without them, turnover-routed data (P03-class) has no polynomial
        # representation at all. Parsimony/MDL/Stage C dispose. (F, not E:
        # sympy reserves bare E for Euler's number.)
        structures.append(("A * x**2 + B * x + C", "polynomial_baseline"))
        structures.append(("A * x**3 + B * x**2 + C * x + D", "polynomial_baseline"))
        structures.append(("A * x**4 + B * x**3 + C * x**2 + D * x + F", "polynomial_baseline"))
        return structures


    @staticmethod
    def _coarse_exponents(expr_str: str) -> list[float]:
        """Extracts float exponents from x**p occurrences in a coarse expression."""
        vals: list[float] = []
        for tok in re.findall(r"x\*\*([0-9.]+)", expr_str):
            try:
                vals.append(float(tok))
            except ValueError:
                continue
        return vals

    @staticmethod
    def _coarse_frequency(expr_str: str) -> float | None:
        """Extracts the cos(w*x) frequency from a coarse damped expression."""
        m = re.search(r"cos\(([0-9.]+)", expr_str)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def refine_stage_d(
        self,
        model: SynthesizedModel,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> SynthesizedModel | None:
        """Stage D: continuous refinement of structural exponents/frequency.

        Coarse synthesis bakes exponents/frequencies into discrete template grids
        (e.g. power 1.5, omega 2.8). This stage re-optimizes them continuously
        from the coarse solution. Accepts only refinements that are finite,
        strictly bound-interior (no railed parameters), and strictly improve
        validation MSE; otherwise returns None and the coarse model stands.
        Unstable/divergent refinements abstain (return None). Proof stays P0.

        Global-to-local strategy: dimension-scaled Sobol low-discrepancy starts
        (max(4, 2*d) points) plus coarse/jitter seeds feed bounded L-BFGS-B.
        If every L-BFGS-B start fails a multi-parameter rational/envelope form,
        a bounded differential-evolution fallback (fixed seed, maxiter=20,
        popsize=10) runs under the same acceptance gates. Deterministic.
        """
        fam = model.structural_family
        expr = model.expression_str
        params = model.parameters

        y_max = float(np.max(np.abs(y_train))) if len(y_train) > 0 else 1.0
        a_max = max(100.0, 50.0 * y_max)
        a_min = min(1e-4, 0.01 * y_max) if y_max > 0 else 1e-6

        forms: list[tuple[str, tuple[str, ...], dict[str, tuple[float, float]], Any, list[float]]]
        # Global input normalization: optimize on x_tilde = x / x_scale so
        # amplitudes AND rates live where the y-scale bounds are meaningful
        # (e.g. A=1e5 or B=1e7 at x~1e-4 would otherwise sit 1000x outside
        # their bounds and rail). Stored values map back per-parameter as
        # P = P_tilde / x_scale**e (see _XSCALE_ROLES). x_scale == 1.0 is an
        # exact no-op; y values are never rescaled so all MSEs stay comparable.
        x_scale_all = 1.0
        _xam = float(np.max(np.abs(x_train))) if len(x_train) > 0 else 1.0
        # Regime gate: normalize ONLY for sub-unity x (max|x| < 0.1), where
        # amplitude bounds demonstrably fail (QEC A=1e5, small-x B=1e7). At
        # normal scales the legacy trajectory is kept byte-identical: Stage D
        # landscapes are multimodal, and reparameterization alone can flip
        # basins (observed: Planck R^2 0.95 -> 0.94 with identical math).
        if np.isfinite(_xam) and 0.0 < _xam < 0.1:
            x_scale_all = _xam
        x_train_n = x_train if x_scale_all == 1.0 else x_train / x_scale_all
        x_val_n = x_val if x_scale_all == 1.0 else x_val / x_scale_all
        with np.errstate(all="ignore"):
            if fam == "power_law":
                exps = self._coarse_exponents(expr)
                p0 = exps[0] if exps else 1.5
                a0_tilde = float(params.get("A", 1.0)) * (x_scale_all**float(p0))
                forms = [("power", ("A", "p"), {"A": (a_min, a_max), "p": (-2.0, 5.0)},
                           lambda x, A, p: A * np.power(np.maximum(x, 1e-300), p),
                           [float(a0_tilde), float(p0)])]
            elif fam == "exponential_decay":
                k0e = float(params.get("B", 0.5))
                forms = [("exp", ("A", "k"), {"A": (a_min, a_max), "k": (-1.0 * x_scale_all, 5.0 * x_scale_all)},
                           lambda x, A, k: A * np.exp(-k * x),
                           [float(params.get("A", 1.0)), float(k0e * x_scale_all)])]
            elif fam in {"damped_oscillation", "undamped_oscillation"}:
                w0 = self._coarse_frequency(expr) or 2.8
                if fam == "undamped_oscillation":
                    forms = [("undamped", ("A", "w"),
                               {"A": (a_min, a_max), "w": (0.2 * x_scale_all, 10.0 * x_scale_all)},
                               lambda x, A, w: A * np.cos(w * x),
                               [float(params.get("A", 1.0)), float(w0 * x_scale_all)])]
                else:
                    k0d = float(params.get("B", 0.3))
                    forms = [("damped", ("A", "k", "w"),
                               {"A": (a_min, a_max), "k": (0.0, 5.0 * x_scale_all),
                                "w": (0.2 * x_scale_all, 10.0 * x_scale_all)},
                               lambda x, A, k, w: A * np.exp(-k * x) * np.cos(w * x),
                               [float(params.get("A", 1.0)), float(k0d * x_scale_all),
                                float(w0 * x_scale_all)])]
            elif fam == "rational_transition":
                exps = self._coarse_exponents(expr)
                p0 = exps[0] if len(exps) > 0 else 1.5
                q0 = exps[1] if len(exps) > 1 else 1.5
                forms = [("rational", ("A", "p", "B", "q"),
                            {"A": (a_min, a_max), "p": (-2.0, 5.0),
                             "B": (1e-5, a_max), "q": (0.05, 6.0)},
                            lambda x, A, p, B, q: (A * np.power(np.maximum(x, 1e-300), p)
                                                   / (1.0 + B * np.power(np.maximum(x, 1e-300), q))),
                            [float(params.get("A", 1.0)) * (x_scale_all**float(p0)), float(p0),
                             float(params.get("B", 1.0)) * (x_scale_all**float(q0)), float(q0)])]
            elif fam in {"rational_exponential_envelope", "exponential_envelope",
                         "rational_saturation"}:
                has_denom = "/ (" in expr or "/(" in expr
                has_exp = "exp(" in expr
                exps = self._coarse_exponents(expr)
                r_match = re.search(r"exp\(-[0-9.]*\*?x\*\*([0-9.]+)", expr)
                r0 = float(r_match.group(1)) if r_match else 1.0
                if has_denom and has_exp:
                    p0 = exps[0] if len(exps) > 0 else 1.5
                    q0 = exps[1] if len(exps) > 1 else 2.0
                    forms = [("envelope", ("A", "p", "B", "q", "C", "r"),
                               {"A": (a_min, a_max), "p": (-2.0, 5.0), "B": (1e-5, a_max),
                                "q": (0.05, 6.0), "C": (1e-5, 10.0), "r": (0.2, 3.0)},
                               lambda x, A, p, B, q, C, r: (
                                   A * np.power(np.maximum(x, 1e-300), p)
                                   / (1.0 + B * np.power(np.maximum(x, 1e-300), q))
                                   * np.exp(-C * np.power(np.maximum(x, 1e-300), r))),
                                [float(params.get("A", 1.0)) * (x_scale_all**float(p0)), float(p0),
                                 float(params.get("B", 1.0)) * (x_scale_all**float(q0)), float(q0),
                                 float(params.get("C", 0.3)) * (x_scale_all**float(r0)), float(r0)])]
                elif has_exp:
                    p0 = exps[0] if exps else 1.5
                    forms = [("exp_env", ("A", "p", "C", "r"),
                               {"A": (a_min, a_max), "p": (-2.0, 5.0),
                                "C": (1e-5, 10.0), "r": (0.2, 3.0)},
                               lambda x, A, p, C, r: (
                                   A * np.power(np.maximum(x, 1e-300), p)
                                   * np.exp(-C * np.power(np.maximum(x, 1e-300), r))),
                                [float(params.get("A", 1.0)) * (x_scale_all**float(p0)), float(p0),
                                 float(params.get("C", 0.3)) * (x_scale_all**float(r0)), float(r0)])]
                else:
                    p0 = exps[0] if len(exps) > 0 else 1.5
                    q0 = exps[1] if len(exps) > 1 else 1.5
                    forms = [("rational", ("A", "p", "B", "q"),
                               {"A": (a_min, a_max), "p": (-2.0, 5.0),
                                "B": (1e-5, a_max), "q": (0.05, 6.0)},
                               lambda x, A, p, B, q: (A * np.power(np.maximum(x, 1e-300), p)
                                                      / (1.0 + B * np.power(np.maximum(x, 1e-300), q))),
                                [float(params.get("A", 1.0)) * (x_scale_all**float(p0)), float(p0),
                                 float(params.get("B", 1.0)) * (x_scale_all**float(q0)), float(q0)])]
            elif fam == "logarithmic_boundary_layer":
                if "log(x)" in expr and "log(1" not in expr:
                    return None  # A*log(x)+B is already linear-optimal
                exps = self._coarse_exponents(expr)
                a0 = exps[0] if len(exps) > 0 else 1.0
                b0 = exps[1] if len(exps) > 1 else 1.0
                forms = [("logbound", ("A", "a", "B", "b"),
                           {"A": (a_min, a_max), "a": (0.0, 3.0),
                            "B": (1e-5, a_max), "b": (0.05, 4.0)},
                           lambda x, A, a, B, b: (
                               A * np.power(np.maximum(x, 1e-300), a)
                               * np.log1p(B * np.power(np.maximum(x, 1e-300), b))),
                            [float(params.get("A", 1.0)) * (x_scale_all**float(a0)), float(a0),
                             float(params.get("B", 1.0)) * (x_scale_all**float(b0)), float(b0)])]
            else:
                return None

            best: SynthesizedModel | None = None
            for _name, names, bounds, fn, start in forms:
                lo = np.array([bounds[n][0] for n in names], dtype=float)
                hi = np.array([bounds[n][1] for n in names], dtype=float)
                starts = [np.clip(np.asarray(start, dtype=float), lo, hi)]
                jitter = np.asarray(start, dtype=float) * 1.15 + 0.05
                starts.append(np.clip(jitter, lo, hi))
                from scipy.stats import qmc
                # Dimension-scaled low-discrepancy coverage: higher-dimensional
                # forms (rational/envelope) need more starts for comparable
                # global coverage than 1-2 parameter forms.
                n_sobol = max(4, 2 * len(names))
                try:
                    sobol = qmc.Sobol(d=len(names), seed=42)
                    qmc_pts = qmc.scale(sobol.random(n_sobol), lo, hi)
                    for pt in qmc_pts:
                        starts.append(np.clip(pt, lo, hi))
                except (ArithmeticError, TypeError, ValueError):
                    logger.debug("Sobol seeding failed for %s; using coarse starts", _name, exc_info=True)
                y_var = max(float(np.var(y_train)), 1e-30)
                y_scale = max(float(np.max(np.abs(y_train))), 1e-12)

                def objective(
                    v: np.ndarray, _fn=fn, _y_var=y_var, _y_scale=y_scale,
                    _xtr=x_train_n,
                ) -> float:
                    try:
                        pred = _fn(_xtr, *[float(t) for t in v])
                        pred = np.asarray(pred, dtype=float)
                        if not np.all(np.isfinite(pred)):
                            return 1e12
                        if fam in {"rational_exponential_envelope",
                                   "logarithmic_boundary_layer"} and np.any(pred < -1e-6 * _y_scale):
                            return 1e12
                        return float(np.mean((y_train - pred) ** 2) / _y_var)
                    except (ArithmeticError, TypeError, ValueError):
                        return 1e12

                best_before_form = best

                def _accept_vector(
                    v: np.ndarray, _hi=hi, _lo=lo, _fn=fn, _names=names, _form=_name,
                    _xtr=x_train_n, _xva=x_val_n, _xs=x_scale_all,
                ) -> None:
                        nonlocal best
                        # Stability: reject railed (bound-touching) solutions.
                        # Bounds spanning >3 decades (e.g. amplitudes) are judged
                        # in log space: a linear-relative margin on a
                        # (1e-6, 100) interval would veto any solution below
                        # ~1e-4, including legitimate small amplitudes.
                        span = _hi - _lo
                        with np.errstate(all="ignore"):
                            wide = (_lo > 0) & (_hi / np.maximum(_lo, 1e-300) > 1e3)
                            lin_rail = ((v - _lo) < 1e-6 * span) | ((_hi - v) < 1e-6 * span)
                            log_span = np.log(_hi / np.maximum(_lo, 1e-300))
                            log_v_lo = np.log(np.maximum(v, 1e-300) / np.maximum(_lo, 1e-300))
                            log_hi_v = np.log(np.maximum(_hi, 1e-300) / np.maximum(v, 1e-300))
                            log_rail = (log_v_lo < 1e-6 * log_span) | (log_hi_v < 1e-6 * log_span)
                            railed = np.where(wide, log_rail, lin_rail)
                        if bool(np.any(railed)):
                            return
                        try:
                            pred_tr = np.asarray(_fn(_xtr, *[float(t) for t in v]), dtype=float)
                            pred_va = np.asarray(_fn(_xva, *[float(t) for t in v]), dtype=float)
                            if not (np.all(np.isfinite(pred_tr)) and np.all(np.isfinite(pred_va))):
                                return
                            tr_mse = float(np.mean((y_train - pred_tr) ** 2))
                            va_mse = float(np.mean((y_val - pred_va) ** 2))
                        except (ArithmeticError, TypeError, ValueError):
                            return
                        if not (va_mse < model.val_mse * (1.0 - 1e-6)):
                            return
                        pdict = {n: float(v[i]) for i, n in enumerate(_names)}
                        if _xs != 1.0:
                            # Map normalized amplitudes/rates back to true scale:
                            # P = P_tilde / x_scale**e, with e from the form's
                            # exponent role (constant 1.0 for rates/frequencies).
                            for _pk, _rk in _XSCALE_ROLES.get(_form, {}).items():
                                if _pk in pdict:
                                    _re = float(pdict[_rk]) if isinstance(_rk, str) else float(_rk)
                                    pdict[_pk] = float(pdict[_pk] / (_xs**_re))
                        resid = y_val - pred_va
                        val_var = float(np.var(y_val))
                        val_r2 = float(1.0 - va_mse / max(val_var, 1e-12))
                        if len(resid) > 4:
                            _ac = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
                            autocorr = _ac if np.isfinite(_ac) else 0.0
                        else:
                            autocorr = 0.0
                        try:
                            if _form == "power":
                                expr_s = f"{pdict['A']:.10g}*x**{pdict['p']:.10g}"
                            elif _form == "exp":
                                expr_s = f"{pdict['A']:.10g}*exp(-{pdict['k']:.10g}*x)"
                            elif _form == "undamped":
                                expr_s = f"{pdict['A']:.10g}*cos({pdict['w']:.10g}*x)"
                            elif _form == "damped":
                                expr_s = (f"{pdict['A']:.10g}*exp(-{pdict['k']:.10g}*x)"
                                          f"*cos({pdict['w']:.10g}*x)")
                            elif _form in {"rational", "rational_saturation"}:
                                expr_s = (f"{pdict['A']:.10g}*x**{pdict['p']:.10g}"
                                          f"/(1.0 + {pdict['B']:.10g}*x**{pdict['q']:.10g})")
                            elif _form == "envelope":
                                expr_s = (f"{pdict['A']:.10g}*x**{pdict['p']:.10g}"
                                          f"/(1.0 + {pdict['B']:.10g}*x**{pdict['q']:.10g})"
                                          f"*exp(-{pdict['C']:.10g}*x**{pdict['r']:.10g})")
                            elif _form == "exp_env":
                                expr_s = (f"{pdict['A']:.10g}*x**{pdict['p']:.10g}"
                                          f"*exp(-{pdict['C']:.10g}*x**{pdict['r']:.10g})")
                            elif _form == "logbound":
                                expr_s = (f"{pdict['A']:.10g}*x**{pdict['a']:.10g}"
                                          f"*log(1.0 + {pdict['B']:.10g}*x**{pdict['b']:.10g})")
                            else:
                                return
                            sym_expr = sp.sympify(expr_s)
                        except (sp.SympifyError, TypeError, ValueError, SyntaxError):
                            logger.debug("Stage D expression rebuild failed for %s", _form,
                                         exc_info=True)
                            return
                        cand = SynthesizedModel(
                            expression_str=str(sym_expr),
                            sympy_expr=sym_expr,
                            parameters={**model.parameters, **pdict},
                            train_mse=tr_mse,
                            val_mse=va_mse,
                            condition_number=model.condition_number,
                            has_singularity_risk=model.has_singularity_risk,
                            epistemic_state=None,
                            structural_family=fam,
                            diagnostics={**model.diagnostics, "stage_d_refined": True,
                                         "stage_d_form": _form,
                                         "stage_d_params": pdict,
                                         "val_r2": val_r2,
                                         "residual_autocorr": autocorr,
                                         "coarse_val_mse": model.val_mse,
                                         "coarse_expression": model.expression_str},
                        )
                        if best is None or cand.val_mse < best.val_mse:
                            best = cand

                for s0 in starts:
                    try:
                        res = opt.minimize(objective, s0, bounds=[bounds[n] for n in names],
                                           method="L-BFGS-B",
                                           options={"maxiter": 200, "ftol": 1e-9})
                    except (ArithmeticError, TypeError, ValueError):
                        continue
                    _accept_vector(np.asarray(res.x, dtype=float))
                if best is best_before_form and _name in {"rational", "envelope", "exp_env"}:
                    # Differential-evolution fallback for multi-parameter traps:
                    # only when every L-BFGS-B start failed this form. Bounded
                    # budget, fixed seed (deterministic); same acceptance gates.
                    try:
                        de = opt.differential_evolution(
                            objective,
                            bounds=[bounds[n] for n in names],
                            seed=42,
                            maxiter=20,
                            popsize=10,
                            polish=True,
                            updating="deferred",
                            workers=1,
                        )
                    except (ArithmeticError, TypeError, ValueError):
                        logger.debug("Stage D DE fallback failed for %s", _name, exc_info=True)
                    else:
                        _accept_vector(np.asarray(de.x, dtype=float))
            return best

    def fit_candidate(
        self,
        struct_template: str,
        family_label: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> SynthesizedModel | None:
        """Fits parameters of a structural candidate with active singularity and gap audits."""
        x_sym = sp.Symbol("x")

        try:
            parsed = sp.sympify(struct_template)
        except (sp.SympifyError, TypeError, ValueError):
            return None

        # Free-symbol alphabet generalized to {A,B,C,D,F} so
        # polynomial_baseline can reach cubic/quartic degree; all downstream
        # machinery (bounds, eval, normalization map-back, expression rebuild)
        # keys off these names generically. Templates using other symbols fit
        # nothing by design. F (not E) is the quartic constant because sympy
        # reserves bare E for Euler's number.
        free_params = sorted(
            [s for s in parsed.free_symbols if str(s) in {"A", "B", "C", "D", "F"}],
            key=lambda s: str(s),
        )

        # Audit for unobserved singularities across sampling gaps (Tier 1 Rust Kernel)
        # Note: In critical scaling, the pole at xc is physically expected within the gap.
        if family_label != "critical_scaling_divergence":
            for g_min, g_max in self.domain_gaps:
                probe_expr = (
                    struct_template.replace("A", "2.0").replace("B", "1.0").replace("C", "0.5")
                    .replace("D", "0.5").replace("E", "0.5")
                )
                probe_rep = self.rust_kernel.probe_singularity(
                    probe_expr,
                    ["x"],
                    [(g_min, g_max)],
                    num_grid_points=200,
                )
                if probe_rep.get("has_pole", False):
                    return None

        # Amplitude normalization (generic): any free parameter P multiplying
        # x**e has true scale P = P_tilde / x_scale**e, which for x << 1 (or
        # x >> 1) can sit orders of magnitude outside y-scale bounds (e.g.
        # A=1e5 with y_max=1e-4 at x~1e-4; B=1e7 for small-x turnover).
        # Fitting on x_tilde = x / x_scale keeps fitted amplitudes/rates at y
        # scale where the (a_min, a_max) bounds are meaningful; stored values
        # are mapped back per-parameter.
        #
        # Scope restriction (correctness-critical): normalization is only valid
        # for templates whose x-dependence is fully carried by fitted exponents
        # and free coefficients. Templates with baked x-scale-sensitive
        # constants -- absolute positions (critical xc) or fixed frequencies
        # (damped/undamped omega grids) -- would be silently corrupted, because
        # baked constants cannot be mapped back. Those families keep the legacy
        # unnormalized path. When adding a family, audit its templates: any
        # absolute coordinate or fixed frequency disqualifies it here.
        _NORMALIZE_FAMILIES = frozenset({
            "power_law", "rational_transition", "rational_saturation",
            "exponential_envelope", "exponential_decay",
            "logarithmic_boundary_layer", "rational_exponential_envelope",
            "linear_baseline", "polynomial_baseline",
        })
        # Regime gate (mirrors refine_stage_d): normalize ONLY for sub-unity x
        # (max|x| < 0.1), where amplitude bounds demonstrably fail. At normal
        # scales the identical math can still flip multimodal basins (observed:
        # Planck rational fit degraded below the R^2 bar after reparameterizing
        # a healthy-scale problem), so legacy trajectories are preserved there.
        normalize_ok = (
            family_label in _NORMALIZE_FAMILIES and "abs(" not in struct_template
        )
        x_scale = 1.0
        x_abs_max = float(np.max(np.abs(x_train))) if len(x_train) > 0 else 1.0
        if (normalize_ok and np.isfinite(x_abs_max) and 0.0 < x_abs_max < 0.1):
            x_scale = x_abs_max
        param_exp: dict[str, float] = {}
        if x_scale != 1.0:
            for pm in re.finditer(
                r"([ABCDF])\s*\*\s*\(?\s*x\s*\*\*\s*\(?([0-9.]+)\)?", struct_template
            ):
                try:
                    param_exp.setdefault(pm.group(1), float(pm.group(2)))
                except ValueError:
                    continue
            for pm in re.finditer(r"([ABCDF])\s*\*\s*\(?\s*x(?!\s*\*\*)", struct_template):
                param_exp.setdefault(pm.group(1), 1.0)
        x_tr = x_train if x_scale == 1.0 else x_train / x_scale
        x_va = x_val if x_scale == 1.0 else x_val / x_scale

        # y-scale normalization (sub-unity-y regime only): L-BFGS-B's absolute
        # tolerances misbehave when objectives live at ~1e-12 scales (observed:
        # ABNORMAL line-search stop 150x above the noise floor at y~1e-6, while
        # the y-normalized problem converges to true parameters). y_scale == 1.0
        # is an exact no-op. Stored amplitudes map back per y-linearity role;
        # reported MSEs are recomputed on raw data and consistency-verified.
        y_scale_n = 1.0
        _ysm = float(np.max(np.abs(y_train))) if len(y_train) > 0 else 1.0
        if np.isfinite(_ysm) and 0.0 < _ysm < 1e-2:
            y_scale_n = _ysm
        y_tr = y_train if y_scale_n == 1.0 else y_train / y_scale_n
        y_va = y_val if y_scale_n == 1.0 else y_val / y_scale_n

        # Precompile vectorized evaluation function
        syms_all = [x_sym] + free_params
        f_lamb_fast: Any = sp.lambdify(syms_all, parsed, modules=["numpy"])

        def eval_fn(params_vec: np.ndarray, x_data: np.ndarray) -> np.ndarray:
            args = [x_data] + [float(p) for p in params_vec]
            res = f_lamb_fast(*args)
            if np.isscalar(res):
                return np.full_like(x_data, float(cast(Any, res)))
            return np.asarray(res, dtype=float)  # type: ignore[arg-type]

        y_var_fit = max(float(np.var(y_tr)), 1e-30)
        y_scale_fit = max(float(np.max(np.abs(y_tr))), 1e-12)

        def objective(params_vec: np.ndarray) -> float:
            try:
                y_pred = eval_fn(params_vec, x_tr)
                if not np.all(np.isfinite(y_pred)):
                    return 1e12
                # Positivity penalty only for strictly non-negative families
                if (
                    family_label in {"rational_exponential_envelope", "logarithmic_boundary_layer", "critical_scaling_divergence"}
                    and np.any(y_pred < -1e-6 * y_scale_fit)
                ):
                    return 1e12
                return float(np.mean((y_tr - y_pred) ** 2) / y_var_fit)
            except (ArithmeticError, TypeError, ValueError):
                return 1e12


        # Initial bounds and guesses (normalized scale when y-gated, so the
        # bounds below are meaningful for fitted tildas in every regime)
        y_max = float(np.max(np.abs(y_tr))) if len(y_tr) > 0 else 1.0
        y_med = float(np.median(np.abs(y_tr))) if len(y_tr) > 0 else 1.0
        a_max = max(100.0, 50.0 * y_max)
        a_min = min(1e-4, 0.01 * y_max) if y_max > 0 else 1e-6
        a_init = y_med if y_med > 0 else 1.0

        bounds = []
        p0 = []
        for p in free_params:
            p_str = str(p)
            if p_str == "A":
                bounds.append((a_min, a_max))
                p0.append(a_init)
            elif p_str == "B":
                if "baseline" in family_label:
                    bounds.append((-a_max, a_max))
                    p0.append(0.0)
                else:
                    bounds.append((0.00001, a_max))
                    p0.append(1.0)
            elif p_str == "C":
                if "baseline" in family_label:
                    bounds.append((-a_max, a_max))
                    p0.append(0.0)
                else:
                    bounds.append((0.00001, a_max))
                    p0.append(0.3)
            elif p_str in {"D", "F"}:
                # Extended alphabet (cubic/quartic polynomial terms): mirror the
                # symmetric baseline treatment; non-baseline templates never
                # carry D/F by construction.
                if "baseline" in family_label:
                    bounds.append((-a_max, a_max))
                    p0.append(0.0)
                else:
                    bounds.append((0.00001, a_max))
                    p0.append(0.3)

        if not p0:
            # Constant / parameter-free
            p_opt = np.array([])
            param_dict = {}
        elif family_label in {"linear_baseline", "polynomial_baseline"}:
            # Linear least-squares fast path: baseline templates are linear in
            # A..E, so the optimum is closed-form. Iterative L-BFGS-B on the
            # raw Vandermonde system can stall far from optimum (observed:
            # R^2 < 0 on exactly solvable quartic data). Falls back to the
            # bounded optimizer if any basis column retains free symbols.
            p_opt = np.array([])
            param_dict = {}
            try:
                basis_exprs = []
                for s in free_params:
                    sub = {q: (1.0 if q == s else 0.0) for q in free_params}
                    col_expr = parsed.subs(sub)
                    if col_expr.free_symbols - {x_sym}:
                        raise ValueError("nonlinear-in-parameters template")
                    basis_exprs.append(col_expr)
                Bm = np.column_stack([
                    np.asarray(
                        sp.lambdify(x_sym, e, modules=["numpy"])(x_tr), dtype=float
                    ).ravel()
                    for e in basis_exprs
                ])
                if Bm.shape[0] == 0 or not np.all(np.isfinite(Bm)):
                    raise ValueError("non-finite basis")
                coef_ls, *_ = np.linalg.lstsq(Bm, np.asarray(y_tr, dtype=float), rcond=None)
                if not np.all(np.isfinite(coef_ls)):
                    raise ValueError("non-finite least-squares solution")
                p_opt = np.asarray(coef_ls, dtype=float)
                param_dict = {str(p): float(p_opt[i]) for i, p in enumerate(free_params)}
            except (ArithmeticError, TypeError, ValueError):
                logger.debug("Polynomial lstsq fast path failed for %s; using bounded optimizer",
                             struct_template, exc_info=True)
                opt_res = opt.minimize(
                    objective,
                    np.array(p0),
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 120, "ftol": 1e-7},
                )
                p_opt = opt_res.x
                param_dict = {str(p): float(p_opt[i]) for i, p in enumerate(free_params)}
        else:
            opt_res = opt.minimize(
                objective,
                np.array(p0),
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 120, "ftol": 1e-7},
            )
            p_opt = opt_res.x
            param_dict = {str(p): float(p_opt[i]) for i, p in enumerate(free_params)}

        # y-linearity roles: a free param takes the y_scale_n factor on map-back
        # iff the template is affine in it. A single additive term carrying two
        # y-linear params would double-count, so such templates skip y-mapping
        # entirely (x-mapping still applies). Probed numerically; conservative
        # on any evaluation failure. Skipped unless y-gated.
        y_roles: dict[str, bool] = {}
        if y_scale_n != 1.0 and free_params and p0 and len(p0) == len(free_params):
            try:
                xp_probe = np.asarray(x_tr[: min(10, len(x_tr))])
                for i, s in enumerate(free_params):
                    sname = str(s)
                    va = list(p0); va[i] = 0.0
                    vb = list(p0); vb[i] = 1.0
                    vm = list(p0); vm[i] = 0.5
                    fa = np.asarray(f_lamb_fast(xp_probe, *[float(t) for t in va]), dtype=float)
                    fb = np.asarray(f_lamb_fast(xp_probe, *[float(t) for t in vb]), dtype=float)
                    fm = np.asarray(f_lamb_fast(xp_probe, *[float(t) for t in vm]), dtype=float)
                    shapes_ok = fa.shape == fb.shape == fm.shape == xp_probe.shape
                    allfin = bool(np.all(np.isfinite(fa))) and bool(np.all(np.isfinite(fb))) and bool(np.all(np.isfinite(fm)))
                    y_roles[sname] = bool(
                        shapes_ok and allfin
                        and np.allclose(fm, 0.5 * (fa + fb), rtol=1e-6, atol=1e-9)
                    )
                try:
                    _terms = sp.Add.make_args(sp.expand(parsed))
                except (ArithmeticError, TypeError, ValueError):
                    _terms = [parsed]
                for _t in _terms:
                    _syms = {str(q) for q in _t.free_symbols} & set(y_roles)
                    if sum(1 for q in _syms if y_roles[q]) > 1:
                        y_roles = {}
                        break
            except (ArithmeticError, TypeError, ValueError):
                y_roles = {}

        if x_scale != 1.0 or y_scale_n != 1.0:
            # Map normalized amplitudes/rates back to true scale (all paths):
            # P_true = P_tilde * y_scale_n / x_scale**e.
            for pname in list(param_dict.keys()):
                if pname in param_dict:
                    mapped = param_dict[pname]
                    if y_roles.get(pname, False):
                        mapped = mapped * y_scale_n
                    e = param_exp.get(pname, 0.0)
                    if e:
                        mapped = mapped / (x_scale**e)
                    param_dict[pname] = float(mapped)

        try:
            y_train_pred = eval_fn(p_opt, x_tr)
            train_mse = float(np.mean((y_tr - y_train_pred) ** 2))
            y_val_pred = eval_fn(p_opt, x_va)
            val_mse = float(np.mean((y_va - y_val_pred) ** 2))
            val_var = float(np.var(y_va))
            val_r2 = float(1.0 - val_mse / max(val_var, 1e-12))
        except (ArithmeticError, TypeError, ValueError):
            return None

        if (x_scale != 1.0 or y_scale_n != 1.0) and param_dict:
            # Verify map-back consistency, then report true-scale MSEs: the
            # mapped model on raw inputs must reproduce the fitted normalized
            # predictions up to y_scale_n. Mismatch means the role mapping
            # misdescribed this template: refuse rather than emit a model
            # whose expression disagrees with its own fit.
            try:
                mapped_vec = [param_dict[str(p)] for p in free_params]
                raw_tr_pred = np.asarray(eval_fn(mapped_vec, x_train), dtype=float)
                raw_va_pred = np.asarray(eval_fn(mapped_vec, x_val), dtype=float)
                chk_tr = y_scale_n * np.asarray(y_train_pred, dtype=float)
                chk_va = y_scale_n * np.asarray(y_val_pred, dtype=float)
                raw_yscale = max(1.0, float(np.max(np.abs(y_train))) if len(y_train) > 0 else 1.0)
                tol = 1e-9 * raw_yscale
                ytr_arr = np.asarray(y_train).ravel()
                yva_arr = np.asarray(y_val).ravel()
                consistent = (
                    raw_tr_pred.shape == chk_tr.shape and raw_va_pred.shape == chk_va.shape
                    and bool(np.all(np.isfinite(raw_tr_pred)))
                    and bool(np.all(np.isfinite(raw_va_pred)))
                    and bool(np.allclose(raw_tr_pred, chk_tr, rtol=1e-5, atol=tol))
                    and bool(np.allclose(raw_va_pred, chk_va, rtol=1e-5, atol=tol))
                )
                if not consistent:
                    logger.debug("Map-back verification failed for %s; refusing candidate",
                                 struct_template)
                    return None
                train_mse = float(np.mean((ytr_arr - raw_tr_pred) ** 2))
                val_mse = float(np.mean((yva_arr - raw_va_pred) ** 2))
                val_var = float(np.var(yva_arr))
                val_r2 = float(1.0 - val_mse / max(val_var, 1e-12))
            except (ArithmeticError, TypeError, ValueError):
                return None


        # Build instantiated expression
        expr_instantiated = parsed.subs({sp.Symbol(k): sp.Float(v) for k, v in param_dict.items()})

        # Check conditioning of parameter Jacobian around optimum.
        # A degenerate probe (e.g. NaN Jacobian columns) is evidence of
        # singularity risk, not a reason to drop the candidate: flag the risk
        # conservatively and let Stage C adjudicate.
        cond_number = 1.0
        is_well_conditioned = True
        if len(p_opt) > 0:
            try:
                eps = 1e-5
                jac_cols = []
                for i in range(len(p_opt)):
                    p_plus = p_opt.copy()
                    p_plus[i] += eps
                    y_plus = eval_fn(p_plus, x_tr)
                    jac_cols.append((y_plus - y_train_pred) / eps)

                cond_rep = self.rust_kernel.compute_matrix_conditioning(jac_cols, rcond_threshold=1e-10)
                cond_number = cond_rep["condition_number"]
                is_well_conditioned = cond_rep["is_well_conditioned"]
            except (np.linalg.LinAlgError, ArithmeticError, TypeError, ValueError):
                logger.debug("Jacobian conditioning probe degenerate; flagging singularity risk",
                             exc_info=True)
                cond_number = float("inf")
                is_well_conditioned = False

        # Compute residual autocorrelation (pattern score). y_va matches the
        # (possibly normalized) scale of y_val_pred; correlation is scale-free.
        residuals = y_va - y_val_pred
        if len(residuals) > 4:
            autocorr = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
            autocorr = autocorr if np.isfinite(autocorr) else 0.0
        else:
            autocorr = 0.0

        return SynthesizedModel(
            expression_str=str(expr_instantiated),
            sympy_expr=expr_instantiated,
            parameters=param_dict,
            train_mse=train_mse,
            val_mse=val_mse,
            condition_number=cond_number,
            has_singularity_risk=not is_well_conditioned,
            structural_family=family_label,
            diagnostics={
                "template": struct_template,
                "family": family_label,
                "val_r2": val_r2,
                "residual_autocorr": autocorr,
            },
        )

    def synthesize(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_near_ood: np.ndarray | None = None,
        y_near_ood: np.ndarray | None = None,
        x_far_ood: np.ndarray | None = None,
        y_far_ood: np.ndarray | None = None,
        *,
        enable_active_singularity_probe: bool = False,
    ) -> list[SynthesizedModel]:
        """Runs the autonomous synthesis loop, scoring and adjudicating candidates."""
        regimes = self.detect_regimes(x_train, y_train)

        # 1. Epistemic Abstention Gate: Severely Underspecified Observations (Class S5 / C3)
        if regimes.get("is_underspecified", False):
            logger.info("Epistemic Abstention: Dataset severely underspecified (N <= 8).")
            abstain_expr = sp.Symbol("x")
            return [
                SynthesizedModel(
                    expression_str="INSUFFICIENT_EVIDENCE(underspecified_data)",
                    sympy_expr=abstain_expr,
                    parameters={},
                    train_mse=float(np.var(y_train)),
                    val_mse=float(np.var(y_val)),
                    epistemic_state=EpistemicState.INSUFFICIENT_EVIDENCE,
                    structural_family="underspecified_data_regime",
                    diagnostics={"rationale": "Severe data sparsity (N <= 8) precludes confident model selection."},
                )
            ]

        candidate_templates = self.generate_candidate_structures(regimes)
        fitted_models: list[SynthesizedModel] = []
        x_sym = sp.Symbol("x")

        for tmpl, fam in candidate_templates:
            # One degenerate template must not kill the remaining candidates.
            try:
                m = self.fit_candidate(tmpl, fam, x_train, y_train, x_val, y_val)
            except (np.linalg.LinAlgError, ArithmeticError, TypeError, ValueError):
                logger.debug("Candidate fit failed for family %s; skipping template", fam, exc_info=True)
                continue
            if m is not None and np.isfinite(m.val_mse):
                # Evaluate OOD if available
                near_mse = None
                far_mse = None
                if x_near_ood is not None and y_near_ood is not None:
                    f_lamb = sp.lambdify(x_sym, m.sympy_expr, modules=["numpy"])
                    try:
                        near_pred = np.asarray(f_lamb(x_near_ood), dtype=float)
                        near_mse = float(np.mean((y_near_ood - near_pred) ** 2))
                    except (ArithmeticError, TypeError, ValueError):
                        near_mse = float("inf")

                if x_far_ood is not None and y_far_ood is not None:
                    f_lamb = sp.lambdify(x_sym, m.sympy_expr, modules=["numpy"])
                    try:
                        far_pred = np.asarray(f_lamb(x_far_ood), dtype=float)
                        far_mse = float(np.mean((y_far_ood - far_pred) ** 2))
                    except (ArithmeticError, TypeError, ValueError):
                        far_mse = float("inf")


                m_updated = SynthesizedModel(
                    expression_str=m.expression_str,
                    sympy_expr=m.sympy_expr,
                    parameters=m.parameters,
                    train_mse=m.train_mse,
                    val_mse=m.val_mse,
                    near_ood_mse=near_mse,
                    far_ood_mse=far_mse,
                    condition_number=m.condition_number,
                    has_singularity_risk=m.has_singularity_risk,
                    structural_family=m.structural_family,
                    diagnostics=m.diagnostics,
                )
                fitted_models.append(m_updated)

        # Sort by validation error
        fitted_models.sort(key=lambda m: m.val_mse)

        # 1b. Stage D Continuous Refinement: refine the best model per family so
        # coarse-grid exponents/frequencies are optimized continuously. Only
        # strictly validation-improving, bound-interior refinements survive;
        # coarse candidates always remain as fallback.
        try:
            seen_fams: set[str] = set()
            stage_d_extra: list[SynthesizedModel] = []
            for m in fitted_models:
                if m.structural_family in seen_fams or len(stage_d_extra) >= 8:
                    continue
                seen_fams.add(m.structural_family)
                try:
                    refined = self.refine_stage_d(m, x_train, y_train, x_val, y_val)
                except (ArithmeticError, TypeError, ValueError):
                    logger.debug("Stage D refinement probe failed for %s",
                                 m.structural_family, exc_info=True)
                    refined = None
                if refined is not None:
                    # Score OOD for the refined expression when OOD data is
                    # available (coarse models are scored in the loop above;
                    # without this, refined tops report near_ood_mse=None).
                    r_near: float | None = None
                    r_far: float | None = None
                    if x_near_ood is not None and y_near_ood is not None:
                        try:
                            f_lamb = sp.lambdify(x_sym, refined.sympy_expr,
                                                 modules=["numpy"])
                            r_near = float(np.mean(
                                (y_near_ood - np.asarray(f_lamb(x_near_ood),
                                                         dtype=float)) ** 2))
                        except (ArithmeticError, TypeError, ValueError):
                            r_near = float("inf")
                    if x_far_ood is not None and y_far_ood is not None:
                        try:
                            f_lamb = sp.lambdify(x_sym, refined.sympy_expr,
                                                 modules=["numpy"])
                            r_far = float(np.mean(
                                (y_far_ood - np.asarray(f_lamb(x_far_ood),
                                                        dtype=float)) ** 2))
                        except (ArithmeticError, TypeError, ValueError):
                            r_far = float("inf")
                    stage_d_extra.append(SynthesizedModel(
                        expression_str=refined.expression_str,
                        sympy_expr=refined.sympy_expr,
                        parameters=refined.parameters,
                        train_mse=refined.train_mse,
                        val_mse=refined.val_mse,
                        near_ood_mse=r_near,
                        far_ood_mse=r_far,
                        condition_number=refined.condition_number,
                        has_singularity_risk=refined.has_singularity_risk,
                        structural_family=refined.structural_family,
                        diagnostics=refined.diagnostics,
                    ))
            fitted_models.extend(stage_d_extra)
            fitted_models.sort(key=lambda m: m.val_mse)
        except (ArithmeticError, TypeError, ValueError):
            logger.debug("Stage D batch refinement failed; using coarse models",
                         exc_info=True)

        # 1c. Parsimony tiebreak (Occam) with noise-calibrated margin: among
        # validation-tied candidates prefer strictly fewer parameters, provided
        # near-OOD (when available) is not degraded >1.5x. The tie margin is
        # calibrated to the measured train noise floor (best_train_mse/n_val):
        # val differences below what train noise can resolve are not evidence.
        # Falls back to a 15% relative margin when the noise estimate is
        # degenerate (near-perfect train fit). Deterministic: val-ordered scan,
        # first qualifier wins.
        if fitted_models:
            best = fitted_models[0]
            noise_floor = best.train_mse / max(len(y_val), 1)
            tie_margin = max(0.15 * best.val_mse, noise_floor)
            best_idx = 0
            best_comp = self._model_complexity(best)
            for i in range(1, len(fitted_models)):
                m = fitted_models[i]
                if m.val_mse - best.val_mse > tie_margin:
                    break
                m_comp = self._model_complexity(m)

                ood_superior = (
                    best.near_ood_mse is not None and m.near_ood_mse is not None
                    and np.isfinite(best.near_ood_mse) and np.isfinite(m.near_ood_mse)
                    and m.near_ood_mse < best.near_ood_mse * 0.5
                )

                if not ood_superior and m_comp >= best_comp:
                    continue
                if (best.near_ood_mse is not None and m.near_ood_mse is not None
                        and np.isfinite(best.near_ood_mse) and np.isfinite(m.near_ood_mse)
                        and m.near_ood_mse > best.near_ood_mse * 1.5):
                    continue
                best_idx = i
                best = m
                best_comp = m_comp
            if best_idx != 0:
                fitted_models.pop(best_idx)
                fitted_models.insert(0, best)

        # 2. Epistemic Abstention Gate: Out-of-Grammar / High Residual Pattern / Shock Front (Class S5 / C5)
        top = fitted_models[0] if fitted_models else None
        top_r2 = top.diagnostics.get("val_r2", -1.0) if top else -1.0
        top_autocorr = abs(top.diagnostics.get("residual_autocorr", 0.0)) if top else 0.0
        has_shock = regimes.get("has_shock_discontinuity", False)

        # Adversary detection criteria:
        # A) Shock front discontinuity where smooth models exhibit systematic deviation
        # B) Non-elementary Bessel function where residual autocorrelation exceeds 0.45
        # C) Chaotic / turbulent trajectory where validation R2 < 0.85
        is_adversarial = (
            (has_shock and top_r2 < 0.96)
            or (top_autocorr > 0.45 and top_r2 < 0.99)
            or (top_r2 < 0.85)
        )

        if is_adversarial or not fitted_models:
            fam_desc = "piecewise_shock_discontinuity" if has_shock else "missing_structural_mechanism"
            abstain_state = EpistemicState.NO_VALID_CANDIDATE
            top_ret = top or SynthesizedModel(
                expression_str="MISSING_STRUCTURE()",
                sympy_expr=sp.Symbol("x"),
                parameters={},
                train_mse=float(np.var(y_train)),
                val_mse=float(np.var(y_val)),
            )
            return [
                SynthesizedModel(
                    expression_str=top_ret.expression_str,
                    sympy_expr=top_ret.sympy_expr,
                    parameters=top_ret.parameters,
                    train_mse=top_ret.train_mse,
                    val_mse=top_ret.val_mse,
                    near_ood_mse=top_ret.near_ood_mse,
                    far_ood_mse=top_ret.far_ood_mse,
                    condition_number=top_ret.condition_number,
                    has_singularity_risk=True,
                    epistemic_state=abstain_state,
                    structural_family=fam_desc,
                    diagnostics={
                        **top_ret.diagnostics,
                        "rationale": "Formal abstention: Residual autocorrelation or discontinuity indicates out-of-grammar phenomena.",
                    },
                )
            ]

        # 3. Adjudicate candidate hypotheses via Stage C Epistemic Decision Engine.
        # Observed-range metadata enables the pole-inside-data refusal: a pole
        # strictly inside observed data (outside known gaps) is a wrong
        # structure, not an uncertain one.
        try:
            x_obs_all = np.concatenate([np.asarray(x_train).ravel(), np.asarray(x_val).ravel()])
            x_obs_all = x_obs_all[np.isfinite(x_obs_all)]
            obs_meta: dict[str, Any] = {}
            if x_obs_all.size > 0:
                obs_meta = {
                    "x_obs_min": float(np.min(x_obs_all)),
                    "x_obs_max": float(np.max(x_obs_all)),
                    "x_gaps": [[float(lo), float(hi)] for lo, hi in self.domain_gaps],
                }
        except (TypeError, ValueError):
            obs_meta = {}

        def _val_residuals(m: SynthesizedModel) -> list[float] | None:
            # Holdout residuals enable the paired-SE indistinguishability path
            # in adjudication. Best-effort: omitted when uncomputable.
            try:
                f_res = sp.lambdify(x_sym, m.sympy_expr, modules=["numpy"])
                with np.errstate(all="ignore"):
                    pred = np.asarray(f_res(np.asarray(x_val)), dtype=float).ravel()
                yv = np.asarray(y_val).ravel()
                if pred.shape != yv.shape or not np.all(np.isfinite(pred)):
                    return None
                return [float(v) for v in (yv - pred)]
            except (ArithmeticError, TypeError, ValueError):
                return None

        candidates: list[CandidateProfile] = []
        for m in fitted_models[:5]:
            res = _val_residuals(m)
            candidates.append(
                CandidateProfile(
                    candidate_id=m.expression_str,
                    expression=m.expression_str,
                    model_class=ModelClass.SYMBOLIC,
                    math_sound=not m.has_singularity_risk,
                    dim_status="VALID",
                    in_sample_mse=m.train_mse,
                    holdout_mse=m.val_mse,
                    # OOD-absent sentinel is inf (CandidateProfile convention):
                    # val MSE must not masquerade as extrapolation evidence,
                    # otherwise adjudication misreads in-domain fit as OOD support.
                    near_ood_mse=m.near_ood_mse if m.near_ood_mse is not None else float("inf"),
                    far_ood_mse=m.far_ood_mse if m.far_ood_mse is not None else float("inf"),
                    complexity=self._model_complexity(m),
                    cond_number=m.condition_number,
                    metadata={**obs_meta, **({"holdout_residuals": res} if res else {})},
                )
            )
        if enable_active_singularity_probe:
            obs_xy = {}
            try:
                obs_xy = {
                    "x": np.concatenate([np.asarray(x_train).ravel(), np.asarray(x_val).ravel()]),
                    "y": np.concatenate([np.asarray(y_train).ravel(), np.asarray(y_val).ravel()]),
                }
            except (TypeError, ValueError):
                obs_xy = {}
            decision = self.stage_c.evaluate(candidates, observed_data=obs_xy or None)
        else:
            decision = self.stage_c.evaluate(candidates)

        top = fitted_models[0]
        top_comp = self._model_complexity(top)

        from apedp.physics_constraints import PhysicsConstraintEngine, PhysicsSpecification
        phys_engine = PhysicsConstraintEngine()
        phys_spec = PhysicsSpecification(
            origin_limit=0.0 if top.structural_family in {
                "power_law", "rational_exponential_envelope", "rational_transition", "rational_saturation"
            } else None,
            far_field_limit=0.0 if top.structural_family == "rational_exponential_envelope" else None,
            positivity=top.structural_family in {"rational_exponential_envelope", "logarithmic_boundary_layer"},
        )
        phys_rep = phys_engine.evaluate(top.expression_str, phys_spec, parameters=top.parameters)

        from apedp.parameter_diagnostics import compute_parameter_uncertainty
        unc_dict: dict[str, Any] = {}
        if top.parameters:
            try:
                param_keys = list(top.parameters.keys())
                syms_p = [x_sym] + [sp.Symbol(k) for k in param_keys]

                stage_d_templates = {
                    "power": "A * x**p",
                    "exp": "A * exp(-k * x)",
                    "undamped": "A * cos(w * x)",
                    "damped": "A * exp(-k * x) * cos(w * x)",
                    "rational": "A * x**p / (1.0 + B * x**q)",
                    "envelope": "A * x**p / (1.0 + B * x**q) * exp(-C * x**r)",
                    "exp_env": "A * x**p * exp(-C * x**r)",
                    "logbound": "A * x**a * log(1.0 + B * x**b)",
                }
                template_str = None
                if top.diagnostics.get("stage_d_refined"):
                    template_str = stage_d_templates.get(str(top.diagnostics.get("stage_d_form", "")))
                if not template_str:
                    template_str = top.diagnostics.get("template")

                if template_str:
                    parsed_template = sp.sympify(template_str)
                    f_lamb_p = sp.lambdify(syms_p, parsed_template, modules=["numpy"])

                    def p_eval(x_arr: np.ndarray, p_dict: dict[str, float]) -> np.ndarray:
                        args = [x_arr] + [float(p_dict[k]) for k in param_keys]
                        res = f_lamb_p(*args)
                        if np.isscalar(res):
                            return np.full_like(x_arr, float(cast(Any, res)))
                        return np.asarray(res, dtype=float)

                    unc_rep = compute_parameter_uncertainty(p_eval, top.parameters, x_val, y_val)
                    unc_dict = unc_rep.to_dict()
            except Exception:
                logger.debug("Parameter uncertainty estimation skipped", exc_info=True)

        from apedp.parameter_diagnostics import make_stage_d2_evidence_report
        evidence_rep = make_stage_d2_evidence_report(
            model_expression=top.expression_str,
            structural_family=top.structural_family,
            val_mse=top.val_mse,
            baseline_mse=float(np.var(y_val)),
            n_samples=len(y_val),
            complexity=top_comp,
            near_ood_mse=top.near_ood_mse,
            condition_number=top.condition_number,
            decision_verdict="ACCEPTED" if decision.state == EpistemicState.UNIQUE_SUPPORT else "COMPETITIVE_SURROGATE",
            parameter_uncertainty=unc_dict,
            physics_evidence=phys_rep.to_dict(),
            extra_diagnostics={
                "candidate_evaluations": decision.candidate_evaluations,
                "supported_hypotheses": decision.supported_hypotheses,
            },
        )

        fitted_models[0] = SynthesizedModel(
            expression_str=top.expression_str,
            sympy_expr=top.sympy_expr,
            parameters=top.parameters,
            train_mse=top.train_mse,
            val_mse=top.val_mse,
            near_ood_mse=top.near_ood_mse,
            far_ood_mse=top.far_ood_mse,
            condition_number=top.condition_number,
            has_singularity_risk=top.has_singularity_risk,
            epistemic_state=decision.state,
            structural_family=top.structural_family,
            diagnostics={
                **top.diagnostics,
                "rationale": decision.adjudication_rationale,
                "stage_d2_evidence": evidence_rep.to_dict(),
                "physics_compliance": phys_rep.to_dict(),
                "parameter_uncertainty": unc_dict,
            },
            proof_claim="P0",
        )

        return fitted_models
