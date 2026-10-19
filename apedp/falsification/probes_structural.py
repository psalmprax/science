"""Structural-invariant axis: positivity, monotonicity, ceilings, conservation."""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy as sp

from apedp.evidence import EvidenceTier
from apedp.safety import quiet_float_warnings

from .constants import ENVELOPE_DIVERGENCE_RATIO
from .counterexamples import Counterexample
from .taxonomy import AXIS_EVIDENCE_TIER, FailureClass, FalsificationAxis

logger = logging.getLogger("apedp.falsification")


@dataclass(frozen=True)
class _EnvelopeScan:
    """One model's behaviour across its declared envelope, measured once.

    Every invariant below reads these numbers rather than re-deriving them, so no
    check can decide against a different sample than the check before it, and the
    extremes (`lowest`, `highest`, `widest`) are computed from one definition of
    "the model's behaviour" instead of fifteen.
    """

    samples: list[tuple[dict[str, float], float | None]]
    finite: list[tuple[dict[str, float], float]]
    lowest: tuple[dict[str, float], float] | None
    highest: tuple[dict[str, float], float] | None
    widest: tuple[dict[str, float], float] | None
    first_unevaluated: dict[str, float]
    boundary_divergence: float | None

    @classmethod
    def from_samples(
        cls, samples: list[tuple[dict[str, float], float | None]]
    ) -> _EnvelopeScan:
        finite = [(point, value) for point, value in samples if value is not None]
        magnitudes = sorted(abs(value) for _, value in finite)
        median_magnitude = magnitudes[len(magnitudes) // 2] if magnitudes else 0.0
        return cls(
            samples=samples,
            finite=finite,
            lowest=min(finite, key=lambda pv: pv[1]) if finite else None,
            highest=max(finite, key=lambda pv: pv[1]) if finite else None,
            widest=max(finite, key=lambda pv: abs(pv[1])) if finite else None,
            first_unevaluated=next((point for point, value in samples if value is None), {}),
            boundary_divergence=(
                magnitudes[-1] / median_magnitude if median_magnitude > 0 else None
            ),
        )


@dataclass
class _StructuralContext:
    """Everything the individual invariant checks read, gathered once per call."""

    expr: sp.Expr
    invariants: list[str]
    domains: dict[str, tuple[float, float]]
    syms: list[sp.Symbol]
    evaluate: Callable[..., Any]
    scan: _EnvelopeScan
    probe_errors: list[dict[str, Any]]
    variable_units: dict[str, str] | None
    expected_output_unit: str | None


class _StructuralProbesMixin:
    """Structural-invariant axis: positivity, monotonicity, ceilings, conservation."""

    @quiet_float_warnings()
    def _probe_structural_invariants(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]],
        invariants: list[str],
        variable_units: dict[str, str] | None = None,
        expected_output_unit: str | None = None
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Tests physical invariants such as positivity, probability ceilings, and dimensional consistency."""
        ces: list[Counterexample] = []
        syms = list(expr.free_symbols)
        prepared = self._prepare_structural_envelope(expr, domains)
        if prepared is None:
            # A constant with no finite real value cannot decide any invariant. It
            # is reported as unevaluable -- an error on the record -- rather than
            # counted as a check that passed.
            return ces, {
                "invariants_declared": len(invariants),
                "invariants_checked": 0,
                "probe_errors": [{
                    "probe": "EnvelopeSweep",
                    "error": "NoRealValues",
                    "detail": f"constant model {expr} has no finite real value",
                    "unresolved": True,
                }],
            }
        evaluate, samples, probe_errors = prepared

        scan = _EnvelopeScan.from_samples(samples)
        metrics: dict[str, Any] = {
            # Declared labels and executed checks are reported separately: the two
            # differ whenever an invariant has no probe, and an invariant that could
            # not be evaluated is never counted as one that passed.
            "invariants_declared": len(invariants),
            "invariants_checked": 0,
            "envelope_samples": len(samples),
            "envelope_samples_unevaluated": len(samples) - len(scan.finite),
            "boundary_divergence": scan.boundary_divergence,
            "probe_errors": list(probe_errors),
        }
        context = _StructuralContext(
            expr=expr,
            invariants=invariants,
            domains=domains,
            syms=syms,
            evaluate=evaluate,
            scan=scan,
            probe_errors=metrics["probe_errors"],
            variable_units=variable_units,
            expected_output_unit=expected_output_unit,
        )

        # Order is the check numbering, and counterexamples are appended in that
        # order. A check returns None when its invariant was not declared (nothing
        # counted), otherwise the counterexamples it found (possibly none).
        for check in (
            self._invariant_dimensional_homogeneity,
            self._invariant_positivity,
            self._invariant_probability_ceiling,
            self._invariant_third_law_thermodynamics,
            self._invariant_lorentz_causality,
            self._invariant_second_law_thermodynamics,
            self._invariant_cosmic_censorship,
            self._invariant_unitarity_bound,
            self._invariant_current_density_ceiling,
            self._invariant_knudsen_continuum,
            self._invariant_velocity_saturation_ceiling,
            self._invariant_yield_strength_ceiling,
            self._invariant_stokes_laminar_ceiling,
            self._invariant_third_law_heat_capacity,
            self._invariant_saturation_ceiling,
        ):
            found = check(context)
            if found is None:
                continue
            metrics["invariants_checked"] += 1
            ces.extend(found)

        self._finalize_structural_metrics(metrics, scan, invariants, expr)
        return ces, metrics

    def _prepare_structural_envelope(
        self, expr: sp.Expr, domains: dict[str, tuple[float, float]]
    ) -> tuple[
        Callable[..., Any],
        list[tuple[dict[str, float], float | None]],
        list[dict[str, Any]],
    ] | None:
        """Evaluate the model over its declared envelope, ready for the invariant checks.

        Every invariant is decided from the model's own behaviour across that
        envelope: symbol *names* are never consulted, so renaming a variable cannot
        disable a check, and a violation away from the declared bounds
        (e.g. f = x**2 - 0.5 on [-1, 1]) is not invisible.

        A model with no free symbols is still a model: it is evaluated once at the
        single empty point rather than skipped. The previous early return declared
        the invariants, evaluated none of them, and reported `invariants_checked = 0`
        with an *empty* probe-error list -- a hole the report never mentioned (found
        by tests/test_structural_invariant_coverage.py: `-1.0` with `positivity` was
        neither refuted nor flagged).

        Returns ``None`` when a constant model has no finite real value, i.e. when no
        invariant can be decided; the caller must report that rather than pass it.
        """
        syms = list(expr.free_symbols)
        if syms:
            evaluate = sp.lambdify(syms, expr, modules=["numpy", "math"])
            samples, probe_errors = self._sample_envelope(evaluate, syms, domains)
            return evaluate, samples, probe_errors

        constant_value = self._constant_envelope_value(expr)
        if constant_value is None:
            return None

        def evaluate(*_args: Any, _value: float = constant_value) -> float:
            return _value

        return evaluate, [({}, constant_value)], []

    @staticmethod
    def _constant_envelope_value(expr: sp.Expr) -> float | None:
        """The single value of a symbol-free model, or None if it has no finite real one."""
        try:
            value = float(expr)
        except (TypeError, ValueError, ArithmeticError):
            return None
        return value if math.isfinite(value) else None

    def _invariant_dimensional_homogeneity(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 1: Dimensional consistency."""
        if not (context.variable_units and context.expected_output_unit):
            return None
        for s_name, u_str in context.variable_units.items():
            self.dim_engine.register_variable(s_name, u_str)
        try:
            inferred = self.dim_engine.check_expression_consistency(context.expr)
            expected_unit_obj = self.dim_engine.registry(context.expected_output_unit)
        except (TypeError, ValueError, AttributeError, NotImplementedError, sp.SympifyError) as e:
            # A probe that could not run is an unevaluated check, not evidence
            # that the model is inconsistent.
            context.probe_errors.append(
                {"probe": "DimensionalInferenceProbe", "error": type(e).__name__, "detail": str(e)}
            )
            logger.warning("Dimensional invariant probe failed", exc_info=True)
            return None
        if inferred.dimensionality == expected_unit_obj.dimensionality:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="DimensionalHomogeneityProbe", failure_class=FailureClass.DIMENSIONAL_MISMATCH,
            counterexample_point={"inferred": str(inferred), "expected": context.expected_output_unit},
            violation_description=(
                f"Dimensional mismatch: expression has units '{inferred}' but expected "
                f"'{context.expected_output_unit}'."
            ),
            severity="FATAL"
        )]

    def _invariant_positivity(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 2: Positivity Invariant (e.g. Temperature, Mass, Concentration, Energy)."""
        if not any(
            inv in ["positivity", "strictly_positive", "non_negative"] for inv in context.invariants
        ):
            return None
        lowest = context.scan.lowest
        if lowest is None or lowest[1] >= -1e-12:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="PositivityViolationProbe", failure_class=FailureClass.UNPHYSICAL_NEGATIVE,
            counterexample_point=lowest[0],
            violation_description=f"UNPHYSICAL_NEGATIVE: Quantity evaluated to {lowest[1]:.4e} < 0, violating non-negativity.",
            severity="FATAL",
            metric_value=lowest[1]
        )]

    def _invariant_probability_ceiling(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 3: Probability Ceiling (p <= 1.0 or depolarizing ceiling)."""
        if "probability_bound" not in context.invariants and "probability" not in context.invariants:
            return None
        highest = context.scan.highest
        if highest is None or highest[1] <= 1.00001:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="ProbabilityCeilingProbe", failure_class=FailureClass.UNPHYSICAL_PROBABILITY,
            counterexample_point=highest[0],
            violation_description=f"UNPHYSICAL_PROBABILITY: Predicted probability {highest[1]:.3f} exceeds absolute ceiling 1.0.",
            severity="FATAL",
            metric_value=highest[1]
        )]

    def _invariant_third_law_thermodynamics(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 4: Thermodynamics 3rd Law (negative Kelvin Temperature).

        Previously this fired on the *name* of the variable plus a negative declared
        bound, without ever evaluating the model. It is decided by the model's own
        values: a thermodynamic quantity below absolute zero is a negative one.
        """
        if "third_law_thermodynamics" not in context.invariants:
            return None
        lowest = context.scan.lowest
        if lowest is None or lowest[1] >= 0.0:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="ThirdLawThermodynamicsProbe", failure_class=FailureClass.THIRD_LAW_TEMPERATURE,
            counterexample_point=lowest[0],
            violation_description=f"UNPHYSICAL_NEGATIVE_TEMPERATURE_3RD_LAW_THERMODYNAMICS: Model evaluates to {lowest[1]:.4e} over its declared envelope, below absolute zero.",
            severity="FATAL",
            metric_value=lowest[1]
        )]

    def _invariant_lorentz_causality(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 5: Lorentz Causality.

        Previously this compared a declared upper bound against its own private copy
        of c. It is decided by the model: a subluminal formulation stays real-valued
        across its envelope, while a tachyonic one is imaginary (non-finite) over the
        domain it claims.
        """
        if "lorentz_causality" not in context.invariants:
            return None
        if len(context.scan.finite) >= len(context.scan.samples):
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="LorentzCausalityProbe", failure_class=FailureClass.ACASUAL_NOT_REAL_VALUED,
            counterexample_point=context.scan.first_unevaluated,
            violation_description="SUPERLUMINAL_LORENTZ_VIOLATION_ACAUSAL: Model is not real-valued (imaginary/Nan) somewhere in its declared envelope, as required by subluminal causality.",
            severity="FATAL"
        )]

    def _invariant_second_law_thermodynamics(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 6: Second Law of Thermodynamics (Entropy Production >= 0).

        The name gate (sigma_entropy / dS_dt / entropy_prod) is gone: when the caller
        declares the second law, the model's own output is what must not be negative
        anywhere in the envelope.
        """
        if "second_law_thermodynamics" not in context.invariants:
            return None
        lowest = context.scan.lowest
        if lowest is None or lowest[1] >= 0:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="SecondLawThermodynamicsProbe", failure_class=FailureClass.NEGATIVE_ENTROPY_PRODUCTION,
            counterexample_point=lowest[0],
            violation_description="NEGATIVE_ENTROPY_PRODUCTION_2ND_LAW_VIOLATION: Entropy production rate is negative.",
            severity="FATAL",
            metric_value=lowest[1]
        )]

    def _invariant_cosmic_censorship(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 7: Cosmic Censorship & Negative Mass Curvature Inversion.

        Previously this fired on a symbol whose *name* contained "mass"/"m_" together
        with a negative declared bound. It is decided by the model: a positive-energy
        law must not evaluate negative, nor grow without bound toward the edge of the
        envelope it declares (an uncensored curvature divergence).
        """
        if "cosmic_censorship" not in context.invariants and "positive_energy" not in context.invariants:
            return None
        scan = context.scan
        if scan.lowest is not None and scan.lowest[1] < 0.0:
            return [Counterexample(
                axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                probe_name="CosmicCensorshipProbe",
                counterexample_point=scan.lowest[0],
                failure_class=FailureClass.NAKED_SINGULARITY,
                violation_description=f"NAKED_SINGULARITY_INVERSION: Positive-energy model evaluates to {scan.lowest[1]:.4e} < 0 inside its declared envelope.",
                severity="FATAL",
                metric_value=scan.lowest[1]
            )]
        if scan.boundary_divergence is not None and scan.boundary_divergence > ENVELOPE_DIVERGENCE_RATIO:
            return [Counterexample(
                axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                probe_name="CosmicCensorshipProbe",
                counterexample_point=scan.widest[0] if scan.widest else {},
                failure_class=FailureClass.NAKED_SINGULARITY,
                violation_description=f"NAKED_SINGULARITY_INVERSION: Model diverges toward the edge of its declared envelope (|f| grows {scan.boundary_divergence:.1e}x), an uncensored curvature divergence.",
                severity="FATAL",
                metric_value=scan.boundary_divergence
            )]
        return []

    def _invariant_unitarity_bound(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 8: High-Energy Unitarity Bound (Froissart Bound)."""
        if "unitarity_bound" not in context.invariants:
            return None
        highest = context.scan.highest
        if highest is None or highest[1] <= 1e4:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="UnitarityBoundProbe", failure_class=FailureClass.UNITARITY_BOUND,
            counterexample_point=highest[0],
            violation_description=f"UNITARITY_VIOLATION: Cross-section {highest[1]:.2e} diverges with s, violating Froissart/unitarity bound at high energies.",
            severity="FATAL",
            metric_value=highest[1]
        )]

    def _invariant_current_density_ceiling(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 9: Current Density Ceiling (Electrochemistry Fickian Mass Transport Limit)."""
        return self._envelope_ceiling(
            context,
            declared=("current_density_ceiling",),
            threshold=1e4,
            probe_name="CurrentDensityCeilingProbe",
            failure_class=FailureClass.CURRENT_DENSITY_CEILING,
            description="UNPHYSICAL_CURRENT_DIVERGENCE: Current density {value:.2e} A/m^2 exceeds mass-transport diffusion ceiling (1e4 A/m^2).",
        )

    def _invariant_knudsen_continuum(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 10: Knudsen Continuum Validity (Fluid Dynamics Micro/Nanoscale Breakdown).

        This one is a *regime* statement -- the smallest scale the envelope claims
        against the 68 nm mean free path -- so it is decided from the declared scales
        with no name matching. The previous implementation additionally grepped the
        model's printed text for magic constants ("68e-9", "R**3", "kn", ...) to guess
        whether a slip correction was already present; that text search is removed. If
        an envelope reaches continuum-invalid scales, the continuum invariant is
        violated and the remedy is a domain restriction, which is what the caller is
        told.
        """
        if "knudsen_continuum" not in context.invariants:
            return None
        scales = [lo for lo, _ in context.domains.values() if lo > 0.0]
        smallest_scale = min(scales) if scales else None
        if smallest_scale is None:
            return []
        lambda_mfp = 68e-9  # Mean free path (68 nm)
        kn = lambda_mfp / smallest_scale
        if kn <= 0.01:
            return []
        widest = context.scan.widest
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="KnudsenContinuumBreakdownProbe", failure_class=FailureClass.CONTINUUM_BREAKDOWN,
            counterexample_point=widest[0] if widest else {},
            violation_description=f"CONTINUUM_BREAKDOWN_KNUDSEN: At R={smallest_scale:.1e}m, Knudsen number Kn={kn:.1f} > 0.01. Continuum no-slip formulation is invalid without slip correction.",
            severity="FATAL",
            metric_value=kn
        )]

    def _invariant_velocity_saturation_ceiling(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 11: Carrier / Velocity Saturation Ceiling."""
        return self._envelope_ceiling(
            context,
            declared=("velocity_saturation_ceiling", "carrier_velocity_ceiling"),
            threshold=1e5,
            probe_name="VelocitySaturationCeilingProbe",
            failure_class=FailureClass.VELOCITY_SATURATION_CEILING,
            description="VELOCITY_SATURATION_EXCEEDED: Predicted velocity {value:.2e} m/s exceeds optical phonon / saturation ceiling (1e5 m/s).",
        )

    def _invariant_yield_strength_ceiling(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 12: Stress / Yield Ceiling (Continuum Plasticity Breakdown)."""
        return self._envelope_ceiling(
            context,
            declared=("stress_saturation_ceiling", "yield_strength_ceiling"),
            threshold=1e9,
            probe_name="YieldStrengthCeilingProbe",
            failure_class=FailureClass.YIELD_STRESS_CEILING,
            description="YIELD_STRESS_EXCEEDED: Linear stress {value:.2e} Pa exceeds maximum material yield ceiling (1e9 Pa).",
        )

    def _invariant_stokes_laminar_ceiling(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 13: Laminar Flow Drag Ceiling (Stokes Law Breakdown)."""
        return self._envelope_ceiling(
            context,
            declared=("reynolds_laminar_ceiling", "stokes_drag_ceiling"),
            threshold=1e-3,
            probe_name="StokesLaminarCeilingProbe",
            failure_class=FailureClass.STOKES_REGIME_BREAKDOWN,
            description="STOKES_REGIME_BREAKDOWN: Drag force {value:.2e} N exceeds Stokes laminar creeping flow regime (Re >> 1).",
        )

    def _invariant_third_law_heat_capacity(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 14: Third Law Heat Capacity Floor (C_v -> 0 as T -> 0).

        The probe no longer keys off a variable *named* like a temperature; it applies
        to any non-negative quantity, whose floor is zero, and evaluates the model at
        that zero limit.
        """
        if "third_law_heat_capacity" not in context.invariants:
            return None
        names = [str(s) for s in context.syms]
        zero_floored = [
            name for name in names
            if context.domains.get(name, (1.0, 1.0))[0] >= 0.0
        ]
        if not zero_floored:
            return []
        test_pts = [
            0.0 if name in zero_floored else context.domains.get(name, (1.0, 1.0))[0]
            for name in names
        ]
        try:
            res_zero = float(context.evaluate(*test_pts))
        except (TypeError, ValueError, ArithmeticError, AttributeError) as exc:
            context.probe_errors.append({
                "probe": "ThirdLawHeatCapacityProbe",
                "error": type(exc).__name__,
                "detail": str(exc),
                "unresolved": True,
            })
            logger.warning("Third-law heat capacity probe failed", exc_info=True)
            res_zero = None
        if res_zero is None or abs(res_zero) <= 1e-3:
            return []
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name="ThirdLawHeatCapacityProbe", failure_class=FailureClass.THIRD_LAW_HEAT_CAPACITY,
            counterexample_point=dict(zip(names, test_pts, strict=False)),
            violation_description=f"3RD_LAW_THERMODYNAMICS_VIOLATION: Heat capacity C_v(0) = {res_zero:.2e} != 0 at absolute zero.",
            severity="FATAL",
            metric_value=res_zero
        )]

    def _invariant_saturation_ceiling(
        self, context: _StructuralContext
    ) -> list[Counterexample] | None:
        """Check 15: Recombination / Quenching Saturation Ceiling."""
        return self._envelope_ceiling(
            context,
            declared=("saturation_ceiling",),
            threshold=1e9,
            probe_name="RateSaturationCeilingProbe",
            failure_class=FailureClass.RATE_SATURATION_CEILING,
            description="UNBOUNDED_RATE_DIVERGENCE: Reaction/recombination rate {value:.2e} exceeds physical saturation ceiling.",
        )

    @staticmethod
    def _envelope_ceiling(
        context: _StructuralContext,
        *,
        declared: tuple[str, ...],
        threshold: float,
        probe_name: str,
        failure_class: FailureClass,
        description: str,
    ) -> list[Counterexample] | None:
        """A magnitude ceiling checked at the envelope's largest |f|.

        The five ceilings share one shape: declared label, a bound, and a magnitude
        measured somewhere in the envelope. Keeping the shape in one place means a
        new ceiling cannot quietly measure a different point than its siblings.
        """
        if not any(inv in declared for inv in context.invariants):
            return None
        widest = context.scan.widest
        if widest is None or abs(widest[1]) <= threshold:
            return []
        value = widest[1]
        return [Counterexample(
            axis=FalsificationAxis.STRUCTURAL_INVARIANT,
            probe_name=probe_name,
            failure_class=failure_class,
            counterexample_point=widest[0],
            violation_description=description.format(value=value),
            severity="FATAL",
            metric_value=value,
        )]

    def _finalize_structural_metrics(
        self,
        metrics: dict[str, Any],
        scan: _EnvelopeScan,
        invariants: list[str],
        expr: sp.Expr,
    ) -> None:
        """Record unevaluable declarations and the tier this axis earned."""
        if not scan.finite and invariants:
            # Nothing could be decided: the declared invariants were never evaluated
            # against a real value, so they are unevaluable -- an error to surface,
            # not checks that passed by default.
            metrics["invariants_checked"] = 0
            metrics["probe_errors"].append({
                "probe": "EnvelopeSweep",
                "error": "NoRealValues",
                "detail": "model returned no real value anywhere in the declared envelope",
                "unresolved": True,
            })
            logger.warning(
                "No real values in the declared envelope for %s; %d declared invariant(s) unevaluable",
                expr,
                len(invariants),
            )

        # Declared invariants that were never evaluated are a hole in the evidence
        # and collapse the tier; an axis with no invariants to check is silent.
        if metrics["invariants_checked"] > 0:
            metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.STRUCTURAL_INVARIANT].value
        elif invariants:
            metrics["evidence_tier"] = EvidenceTier.UNRESOLVED.value
