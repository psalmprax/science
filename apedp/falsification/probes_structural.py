"""Structural-invariant axis: positivity, monotonicity, ceilings, conservation."""
from __future__ import annotations

import logging
import math
from typing import Any

import sympy as sp

from apedp.evidence import EvidenceTier
from apedp.safety import quiet_float_warnings

from .constants import ENVELOPE_DIVERGENCE_RATIO
from .counterexamples import Counterexample
from .taxonomy import AXIS_EVIDENCE_TIER, FailureClass, FalsificationAxis

logger = logging.getLogger("apedp.falsification")

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
        if not syms:
            # A model with no free symbols is still a model. The previous early
            # return declared the invariants, evaluated none of them, and reported
            # `invariants_checked = 0` with an *empty* probe-error list -- a hole the
            # report never mentioned (found by tests/test_structural_invariant_
            # coverage.py: `-1.0` with `positivity` was neither refuted nor flagged).
            # The constant is evaluated once instead, so the same checks below
            # decide it; a non-finite constant is reported as unevaluable, exactly
            # as a model that returns no real value across its envelope is.
            try:
                constant_value: float | None = float(expr)
            except (TypeError, ValueError, ArithmeticError):
                constant_value = None
            if constant_value is None or not math.isfinite(constant_value):
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

            def fn(*_args: Any, _value: float = constant_value) -> float:
                return _value

            samples = [({}, constant_value)]
            probe_errors: list[dict[str, Any]] = []
        else:
            fn = sp.lambdify(syms, expr, modules=["numpy", "math"])

            # Every invariant below is decided from the model's own behaviour across
            # the declared envelope. Symbol *names* are never consulted, so renaming a
            # variable cannot disable a check, and a violation away from the declared
            # bounds (e.g. f = x**2 - 0.5 on [-1, 1]) is no longer invisible.
            samples, probe_errors = self._sample_envelope(fn, syms, domains)
        finite = [(pt, v) for pt, v in samples if v is not None]
        lowest = min(finite, key=lambda pv: pv[1]) if finite else None
        highest = max(finite, key=lambda pv: pv[1]) if finite else None
        widest = max(finite, key=lambda pv: abs(pv[1])) if finite else None
        first_unevaluated = next((pt for pt, v in samples if v is None), {})
        magnitudes = sorted(abs(v) for _, v in finite)
        median_magnitude = magnitudes[len(magnitudes) // 2] if magnitudes else 0.0
        boundary_divergence = (
            magnitudes[-1] / median_magnitude if median_magnitude > 0 else None
        )

        metrics: dict[str, Any] = {
            # Declared labels and executed checks are reported separately: the two
            # differ whenever an invariant has no probe, and an invariant that could
            # not be evaluated is never counted as one that passed.
            "invariants_declared": len(invariants),
            "invariants_checked": 0,
            "envelope_samples": len(samples),
            "envelope_samples_unevaluated": len(samples) - len(finite),
            "boundary_divergence": boundary_divergence,
            "probe_errors": list(probe_errors),
        }

        # Check 1: Dimensional consistency
        if variable_units and expected_output_unit:
            for s_name, u_str in variable_units.items():
                self.dim_engine.register_variable(s_name, u_str)
            try:
                inferred = self.dim_engine.check_expression_consistency(expr)
                expected_unit_obj = self.dim_engine.registry(expected_output_unit)
                metrics["invariants_checked"] += 1
                is_dim_compat = inferred.dimensionality == expected_unit_obj.dimensionality
                if not is_dim_compat:
                    ces.append(Counterexample(
                        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                        probe_name="DimensionalHomogeneityProbe", failure_class=FailureClass.DIMENSIONAL_MISMATCH,
                        counterexample_point={"inferred": str(inferred), "expected": expected_output_unit},
                        violation_description=f"Dimensional mismatch: expression has units '{inferred}' but expected '{expected_output_unit}'.",
                        severity="FATAL"
                    ))
            except (TypeError, ValueError, AttributeError, NotImplementedError, sp.SympifyError) as e:
                # A probe that could not run is an unevaluated check, not evidence
                # that the model is inconsistent.
                metrics["probe_errors"].append(
                    {"probe": "DimensionalInferenceProbe", "error": type(e).__name__, "detail": str(e)}
                )
                logger.warning("Dimensional invariant probe failed", exc_info=True)

        # Check 2: Positivity Invariant (e.g. Temperature, Mass, Concentration, Energy)
        if any(inv in ["positivity", "strictly_positive", "non_negative"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < -1e-12:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="PositivityViolationProbe", failure_class=FailureClass.UNPHYSICAL_NEGATIVE,
                    counterexample_point=lowest[0],
                    violation_description=f"UNPHYSICAL_NEGATIVE: Quantity evaluated to {lowest[1]:.4e} < 0, violating non-negativity.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))

        # Check 3: Probability Ceiling (p <= 1.0 or depolarizing ceiling)
        if "probability_bound" in invariants or "probability" in invariants:
            metrics["invariants_checked"] += 1
            if highest is not None and highest[1] > 1.00001:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="ProbabilityCeilingProbe", failure_class=FailureClass.UNPHYSICAL_PROBABILITY,
                    counterexample_point=highest[0],
                    violation_description=f"UNPHYSICAL_PROBABILITY: Predicted probability {highest[1]:.3f} exceeds absolute ceiling 1.0.",
                    severity="FATAL",
                    metric_value=highest[1]
                ))

        # Check 4: Thermodynamics 3rd Law (negative Kelvin Temperature). Previously
        # this fired on the *name* of the variable plus a negative declared bound,
        # without ever evaluating the model. It is now decided by the model's own
        # values: a thermodynamic quantity below absolute zero is a negative one.
        if "third_law_thermodynamics" in invariants:
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < 0.0:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="ThirdLawThermodynamicsProbe", failure_class=FailureClass.THIRD_LAW_TEMPERATURE,
                    counterexample_point=lowest[0],
                    violation_description=f"UNPHYSICAL_NEGATIVE_TEMPERATURE_3RD_LAW_THERMODYNAMICS: Model evaluates to {lowest[1]:.4e} over its declared envelope, below absolute zero.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))

        # Check 5: Lorentz Causality. Previously this compared a declared upper
        # bound against its own private copy of c. It is now decided by the model:
        # a subluminal formulation stays real-valued across its envelope, while a
        # tachyonic one is imaginary (non-finite) over the domain it claims.
        if "lorentz_causality" in invariants:
            metrics["invariants_checked"] += 1
            if len(finite) < len(samples):
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="LorentzCausalityProbe", failure_class=FailureClass.ACASUAL_NOT_REAL_VALUED,
                    counterexample_point=first_unevaluated,
                    violation_description="SUPERLUMINAL_LORENTZ_VIOLATION_ACAUSAL: Model is not real-valued (imaginary/Nan) somewhere in its declared envelope, as required by subluminal causality.",
                    severity="FATAL"
                ))

        # Check 6: Second Law of Thermodynamics (Entropy Production >= 0). The name
        # gate (sigma_entropy / dS_dt / entropy_prod) is gone: when the caller
        # declares the second law, the model's own output is what must not be
        # negative anywhere in the envelope.
        if "second_law_thermodynamics" in invariants:
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < 0:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="SecondLawThermodynamicsProbe", failure_class=FailureClass.NEGATIVE_ENTROPY_PRODUCTION,
                    counterexample_point=lowest[0],
                    violation_description="NEGATIVE_ENTROPY_PRODUCTION_2ND_LAW_VIOLATION: Entropy production rate is negative.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))

        # Check 7: Cosmic Censorship & Negative Mass Curvature Inversion. Previously
        # this fired on a symbol whose *name* contained "mass"/"m_" together with a
        # negative declared bound. It is now decided by the model: a positive-energy
        # law must not evaluate negative, nor grow without bound toward the edge of
        # the envelope it declares (an uncensored curvature divergence).
        if "cosmic_censorship" in invariants or "positive_energy" in invariants:
            metrics["invariants_checked"] += 1
            if lowest is not None and lowest[1] < 0.0:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="CosmicCensorshipProbe",
                    counterexample_point=lowest[0],
                    failure_class=FailureClass.NAKED_SINGULARITY,
                    violation_description=f"NAKED_SINGULARITY_INVERSION: Positive-energy model evaluates to {lowest[1]:.4e} < 0 inside its declared envelope.",
                    severity="FATAL",
                    metric_value=lowest[1]
                ))
            elif boundary_divergence is not None and boundary_divergence > ENVELOPE_DIVERGENCE_RATIO:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="CosmicCensorshipProbe",
                    counterexample_point=widest[0] if widest else {},
                    failure_class=FailureClass.NAKED_SINGULARITY,
                    violation_description=f"NAKED_SINGULARITY_INVERSION: Model diverges toward the edge of its declared envelope (|f| grows {boundary_divergence:.1e}x), an uncensored curvature divergence.",
                    severity="FATAL",
                    metric_value=boundary_divergence
                ))

        # Check 8: High-Energy Unitarity Bound (Froissart Bound)
        if "unitarity_bound" in invariants:
            metrics["invariants_checked"] += 1
            if highest is not None and highest[1] > 1e4:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="UnitarityBoundProbe", failure_class=FailureClass.UNITARITY_BOUND,
                    counterexample_point=highest[0],
                    violation_description=f"UNITARITY_VIOLATION: Cross-section {highest[1]:.2e} diverges with s, violating Froissart/unitarity bound at high energies.",
                    severity="FATAL",
                    metric_value=highest[1]
                ))

        # Check 9: Current Density Ceiling (Electrochemistry Fickian Mass Transport Limit)
        if "current_density_ceiling" in invariants:
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e4:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="CurrentDensityCeilingProbe", failure_class=FailureClass.CURRENT_DENSITY_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"UNPHYSICAL_CURRENT_DIVERGENCE: Current density {widest[1]:.2e} A/m^2 exceeds mass-transport diffusion ceiling (1e4 A/m^2).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 10: Knudsen Continuum Validity (Fluid Dynamics Micro/Nanoscale Breakdown).
        # This one is a *regime* statement -- the smallest scale the envelope claims
        # against the 68 nm mean free path -- so it is decided from the declared
        # scales with no name matching. The previous implementation additionally
        # grepped the model's printed text for magic constants ("68e-9", "R**3",
        # "kn", ...) to guess whether a slip correction was already present; that
        # text search is removed. If an envelope reaches continuum-invalid scales,
        # the continuum invariant is violated and the remedy is a domain
        # restriction, which is what the caller is told.
        if "knudsen_continuum" in invariants:
            metrics["invariants_checked"] += 1
            scales = [lo for lo, _ in domains.values() if lo > 0.0]
            smallest_scale = min(scales) if scales else None
            if smallest_scale is not None:
                lambda_mfp = 68e-9  # Mean free path (68 nm)
                kn = lambda_mfp / smallest_scale
                if kn > 0.01:
                    ces.append(Counterexample(
                        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                        probe_name="KnudsenContinuumBreakdownProbe", failure_class=FailureClass.CONTINUUM_BREAKDOWN,
                        counterexample_point=widest[0] if widest else {},
                        violation_description=f"CONTINUUM_BREAKDOWN_KNUDSEN: At R={smallest_scale:.1e}m, Knudsen number Kn={kn:.1f} > 0.01. Continuum no-slip formulation is invalid without slip correction.",
                        severity="FATAL",
                        metric_value=kn
                    ))

        # Check 11: Carrier / Velocity Saturation Ceiling
        if any(inv in ["velocity_saturation_ceiling", "carrier_velocity_ceiling"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e5:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="VelocitySaturationCeilingProbe", failure_class=FailureClass.VELOCITY_SATURATION_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"VELOCITY_SATURATION_EXCEEDED: Predicted velocity {widest[1]:.2e} m/s exceeds optical phonon / saturation ceiling (1e5 m/s).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 12: Stress / Yield Ceiling (Continuum Plasticity Breakdown)
        if any(inv in ["stress_saturation_ceiling", "yield_strength_ceiling"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e9:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="YieldStrengthCeilingProbe", failure_class=FailureClass.YIELD_STRESS_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"YIELD_STRESS_EXCEEDED: Linear stress {widest[1]:.2e} Pa exceeds maximum material yield ceiling (1e9 Pa).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 13: Laminar Flow Drag Ceiling (Stokes Law Breakdown)
        if any(inv in ["reynolds_laminar_ceiling", "stokes_drag_ceiling"] for inv in invariants):
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e-3:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="StokesLaminarCeilingProbe", failure_class=FailureClass.STOKES_REGIME_BREAKDOWN,
                    counterexample_point=widest[0],
                    violation_description=f"STOKES_REGIME_BREAKDOWN: Drag force {widest[1]:.2e} N exceeds Stokes laminar creeping flow regime (Re >> 1).",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        # Check 14: Third Law Heat Capacity Floor (C_v -> 0 as T -> 0). The probe no
        # longer keys off a variable *named* like a temperature; it applies to any
        # non-negative quantity, whose floor is zero, and evaluates the model at
        # that zero limit.
        if "third_law_heat_capacity" in invariants:
            metrics["invariants_checked"] += 1
            zero_floored = [str(s) for s in syms if domains.get(str(s), (1.0, 1.0))[0] >= 0.0]
            if zero_floored:
                test_pts = [
                    0.0 if str(sym) in zero_floored else domains.get(str(sym), (1.0, 1.0))[0]
                    for sym in syms
                ]
                try:
                    res_zero = float(fn(*test_pts))
                except (TypeError, ValueError, ArithmeticError, AttributeError) as exc:
                    metrics["probe_errors"].append({
                        "probe": "ThirdLawHeatCapacityProbe",
                        "error": type(exc).__name__,
                        "detail": str(exc),
                        "unresolved": True,
                    })
                    logger.warning("Third-law heat capacity probe failed", exc_info=True)
                    res_zero = None
                if res_zero is not None and abs(res_zero) > 1e-3:
                    ces.append(Counterexample(
                        axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                        probe_name="ThirdLawHeatCapacityProbe", failure_class=FailureClass.THIRD_LAW_HEAT_CAPACITY,
                        counterexample_point=dict(zip([str(s) for s in syms], test_pts, strict=False)),
                        violation_description=f"3RD_LAW_THERMODYNAMICS_VIOLATION: Heat capacity C_v(0) = {res_zero:.2e} != 0 at absolute zero.",
                        severity="FATAL",
                        metric_value=res_zero
                    ))

        # Check 15: Recombination / Quenching Saturation Ceiling
        if "saturation_ceiling" in invariants:
            metrics["invariants_checked"] += 1
            if widest is not None and abs(widest[1]) > 1e9:
                ces.append(Counterexample(
                    axis=FalsificationAxis.STRUCTURAL_INVARIANT,
                    probe_name="RateSaturationCeilingProbe", failure_class=FailureClass.RATE_SATURATION_CEILING,
                    counterexample_point=widest[0],
                    violation_description=f"UNBOUNDED_RATE_DIVERGENCE: Reaction/recombination rate {widest[1]:.2e} exceeds physical saturation ceiling.",
                    severity="FATAL",
                    metric_value=widest[1]
                ))

        if not finite and invariants:
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
        return ces, metrics
