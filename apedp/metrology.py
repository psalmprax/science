"""APEDP Metrology: GUM Analytical Uncertainty Propagation, Sensor Telemetry, and Orthogonal Metrology."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import PHYSICAL_CONSTANTS


def has_singularity_risk(expr: sp.Expr) -> bool:
    """Whether first-order GUM propagation is unreliable on ``expr``.

    First-order Taylor propagation divides by the evaluated value and its
    derivatives, so it degrades exactly where those derivatives blow up: poles,
    branch cuts, and the boundary of a log/tan domain.

    The check used to key on ``Pow`` exponents equal to -1 or -2 plus the mere
    presence of ``log``/``sqrt`` in the expression. That missed every other
    degree of the same defect -- ``1/x**3``, ``x**Rational(-1, 2)``,
    ``x**Rational(1, 3)`` (a cube-root branch cut) and ``tan(x)`` were all
    classified as safe, so the advisory warning never fired for them. The rule
    here is structural rather than a list of special cases: *any* power whose
    exponent is not a positive integer is a pole or a branch cut, and the
    transcendental functions with an unbounded derivative are named explicitly.
    ``exp`` and ``sin``/``cos`` are deliberately absent -- they are entire.

    Advisory only: it decides whether to attach a warning, never whether the
    propagation itself is allowed.
    """
    for power in expr.atoms(sp.Pow):
        exponent = power.args[1]
        # A symbolic exponent (``x**n``) can be anything, so it is a risk too.
        if not (exponent.is_Integer and exponent > 0):
            return True
    return bool(expr.has(
        sp.log, sp.tan, sp.asin, sp.acos, sp.acot,
        sp.bessely, sp.besselk, sp.gamma, sp.zeta,
    ))


@dataclass
class UncertaintyReport:
    evaluated_value: float
    combined_uncertainty: float
    expanded_uncertainty_95: float
    relative_uncertainty: float
    budget: dict[str, float]  # parameter name -> variance contribution
    percentage_budget: dict[str, float]  # parameter name -> percentage of total variance
    covariance_budget: dict[tuple[str, str], float] = field(default_factory=dict)


class AnalyticalUncertaintyPropagator:
    """GUM (JCGM 100:2008) analytical uncertainty propagation + GUM Supplement 1 Monte Carlo.

    ``propagate_uncertainty`` is first-order Taylor only and is unreliable
    near singularities (division/log/sqrt of an uncertain parameter) — the
    exact regime the repair layer targets. Use
    ``propagate_uncertainty_monte_carlo`` for those cases.
    """

    @staticmethod
    def propagate_uncertainty(
        expr: sp.Expr,
        param_values: dict[str, float],
        param_uncertainties: dict[str, float] | None = None,
        correlations: dict[tuple[str, str], float] | None = None,
        covariance_matrix: np.ndarray | list[list[float]] | dict[tuple[str, str] | str, float] | None = None,
        variable_order: list[str] | None = None
    ) -> UncertaintyReport:
        """
        Propagates standard uncertainties analytically using GUM first-order Taylor expansion partial derivatives.
        Supports standard variances, correlation coefficients, and arbitrary covariance matrices.
        """
        # Defensively copy param_values to prevent mutating caller dict
        param_values = dict(param_values)
        symbols = sorted(expr.free_symbols, key=str)
        # Auto-resolve universal constants if present in expression
        constants_map = PHYSICAL_CONSTANTS.values_dict()
        for sym in symbols:
            name = str(sym)
            if name not in param_values and name in constants_map:
                param_values[name] = constants_map[name]

        missing = [str(s) for s in symbols if str(s) not in param_values]
        if missing:
            raise ValueError(f"Missing parameter values for symbol(s): {missing}")

        sub_dict = {s: param_values[str(s)] for s in symbols if str(s) in param_values}

        # Evaluate the base function at nominal points
        evaluated_val = float(expr.subs(sub_dict))

        # First-order sensitivity coefficients c_i = dy/dx_i
        sensitivities: dict[str, float] = {}
        for sym in symbols:
            name = str(sym)
            if name in param_values:
                deriv = sp.diff(expr, sym)
                sensitivities[name] = float(deriv.subs(sub_dict))

        # Resolve order of variables
        if variable_order is not None:
            order = [v for v in variable_order if v in sensitivities]
        else:
            order = [str(s) for s in symbols if str(s) in sensitivities]

        # Extract standard uncertainties from param_uncertainties or covariance_matrix diagonal
        u_dict: dict[str, float] = dict(param_uncertainties or {})
        if covariance_matrix is not None:
            if isinstance(covariance_matrix, dict):
                for k, cov_val in covariance_matrix.items():
                    if isinstance(k, (tuple, list)) and len(k) == 2:
                        p1, p2 = k
                    else:
                        p1 = p2 = k
                    if p1 == p2 and p1 not in u_dict:
                        u_dict[str(p1)] = float(np.sqrt(max(0.0, float(cov_val))))
            else:
                cov_arr = np.asarray(covariance_matrix, dtype=float)
                if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
                    raise ValueError(f"Covariance matrix must be 2D square array, got shape {cov_arr.shape}")
                for idx, name in enumerate(order):
                    if name not in u_dict and idx < cov_arr.shape[0]:
                        var_val = cov_arr[idx, idx]
                        u_dict[name] = float(np.sqrt(max(0.0, var_val)))

        variance_budget: dict[str, float] = {}
        covariance_budget: dict[tuple[str, str], float] = {}
        variance_sum = 0.0

        for name in order:
            if name in u_dict and name in sensitivities:
                c_i = sensitivities[name]
                u_i = u_dict[name]
                term_var = (c_i ** 2) * (u_i ** 2)
                variance_budget[name] = term_var
                variance_sum += term_var

        # Handle direct covariance matrix input
        if covariance_matrix is not None:
            if isinstance(covariance_matrix, dict):
                seen_pairs = set()
                for k, cov_val in covariance_matrix.items():
                    if isinstance(k, (tuple, list)) and len(k) == 2:
                        p1, p2 = str(k[0]), str(k[1])
                        if p1 in sensitivities and p2 in sensitivities and p1 != p2:
                            pair = (p1, p2) if p1 <= p2 else (p2, p1)
                            if pair in seen_pairs:
                                continue
                            seen_pairs.add(pair)
                            c1 = sensitivities[p1]
                            c2 = sensitivities[p2]
                            cov_term = 2.0 * c1 * c2 * float(cov_val)
                            covariance_budget[pair] = cov_term
                            variance_sum += cov_term
            else:
                cov_arr = np.asarray(covariance_matrix, dtype=float)
                n = min(len(order), cov_arr.shape[0], cov_arr.shape[1])
                for i in range(n):
                    for j in range(i + 1, n):
                        p1, p2 = order[i], order[j]
                        c1, c2 = sensitivities[p1], sensitivities[p2]
                        cov_val = cov_arr[i, j]
                        cov_term = 2.0 * c1 * c2 * float(cov_val)
                        pair = (p1, p2)
                        covariance_budget[pair] = cov_term
                        variance_sum += cov_term

        # Handle correlation dictionary: 2 * sum_{i < j} c_i * c_j * u_i * u_j * r_ij
        elif correlations:
            seen_corr_pairs = set()
            for (p1, p2), r_12 in correlations.items():
                if p1 in sensitivities and p2 in sensitivities and p1 in u_dict and p2 in u_dict:
                    pair = (p1, p2) if p1 <= p2 else (p2, p1)
                    if pair in seen_corr_pairs:
                        continue
                    seen_corr_pairs.add(pair)
                    c1 = sensitivities[p1]
                    c2 = sensitivities[p2]
                    u1 = u_dict[p1]
                    u2 = u_dict[p2]
                    cov_term = 2.0 * c1 * c2 * u1 * u2 * float(r_12)
                    covariance_budget[pair] = cov_term
                    variance_sum += cov_term

        combined_u = float(np.sqrt(max(0.0, variance_sum)))
        expanded_u = 2.0 * combined_u  # k=2 for ~95% coverage interval
        rel_u = (combined_u / abs(evaluated_val)) if abs(evaluated_val) > 1e-15 else float('nan')

        pct_budget = {}
        if variance_sum > 0:
            pct_budget = {k: (v / variance_sum) * 100.0 for k, v in variance_budget.items()}

        return UncertaintyReport(
            evaluated_value=evaluated_val,
            combined_uncertainty=combined_u,
            expanded_uncertainty_95=expanded_u,
            relative_uncertainty=rel_u,
            budget=variance_budget,
            percentage_budget=pct_budget,
            covariance_budget=covariance_budget
        )

    @staticmethod
    def propagate_uncertainty_monte_carlo(
        expr: sp.Expr,
        param_values: dict[str, float],
        param_uncertainties: dict[str, float] | None = None,
        num_samples: int = 20000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """GUM Supplement 1 Monte Carlo propagation via Gaussian sampling.

        Draws ``num_samples`` joint samples (independent normals;
        zero-uncertainty params held fixed), evaluates ``expr`` with
        ``lambdify``, and returns ``mean``/``std``/percentiles. Non-finite
        draws (e.g. division by a sampled zero) are dropped and counted in
        ``non_finite_fraction`` — a large fraction signals the first-order
        GUM result is untrustworthy.
        """
        param_values = dict(param_values)
        symbols = sorted(expr.free_symbols, key=str)
        constants_map = PHYSICAL_CONSTANTS.values_dict()
        for sym in symbols:
            name = str(sym)
            if name not in param_values and name in constants_map:
                param_values[name] = constants_map[name]
        missing = [str(s) for s in symbols if str(s) not in param_values]
        if missing:
            raise ValueError(f"Missing parameter values for symbol(s): {missing}")
        u_dict = dict(param_uncertainties or {})
        rng = np.random.RandomState(seed)
        names = [str(s) for s in symbols]
        draws = np.column_stack([
            rng.normal(
                loc=float(param_values[n]),
                scale=float(u_dict.get(n, 0.0)),
                size=num_samples,
            )
            if float(u_dict.get(n, 0.0)) > 0
            else np.full(num_samples, float(param_values[n]))
            for n in names
        ])
        syms = [sp.Symbol(n) for n in names]
        f = sp.lambdify(syms, expr, modules=["numpy"])
        try:
            vals = np.asarray(f(*[draws[:, i] for i in range(len(names))]), dtype=float)
            if vals.shape != (num_samples,):
                vals = np.full(num_samples, float(vals), dtype=float)
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ValueError(f"Monte Carlo evaluation failed: {e}") from e
        finite = vals[np.isfinite(vals)]
        non_finite_frac = 1.0 - finite.size / max(num_samples, 1)
        if finite.size == 0:
            raise ValueError("All Monte Carlo draws non-finite; model singular over input distribution")
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "median": float(np.median(finite)),
            "p2_5": float(np.percentile(finite, 2.5)),
            "p97_5": float(np.percentile(finite, 97.5)),
            "non_finite_fraction": float(non_finite_frac),
            "num_samples": int(num_samples),
            "num_finite": int(finite.size),
        }


class MetrologySensorStream:
    """Simulates multi-sensor streaming telemetry with orthogonal measurement diversity."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def stream_gravity_experiment_data(self, num_samples: int = 50) -> dict[str, np.ndarray]:
        """
        Simulates experimental telemetry from dual orthogonal sensors:
        - Primary: strain-gauge balance force sensor with 2% gaussian noise + slight drift.
        - Orthogonal: laser-interferometric acceleration sensor with 1% noise and zero drift.
        """
        m1 = self.rng.uniform(5.0, 50.0, num_samples)      # kg
        m2 = self.rng.uniform(100.0, 1000.0, num_samples)  # kg
        r = self.rng.uniform(1.0, 10.0, num_samples)       # meters

        G = 6.67430e-11
        F_true = G * m1 * m2 / (r ** 2)

        # Primary sensor noise + systematic drift
        t_seq = np.linspace(0, 10, num_samples)
        drift = 0.005 * np.sin(t_seq)
        primary_noise = self.rng.normal(0, 0.02, num_samples)
        F_primary = F_true * (1.0 + primary_noise + drift)

        # Orthogonal sensor
        ortho_noise = self.rng.normal(0, 0.01, num_samples)
        F_orthogonal = F_true * (1.0 + ortho_noise)

        # Combined standard uncertainty estimate
        u_c = F_true * 0.02

        return {
            'm_phys': m1,
            'm_target': m2,
            'r_dist': r,
            'force_true': F_true,
            'force_primary': F_primary,
            'force_orthogonal': F_orthogonal,
            'uncertainty_u_c': u_c
        }

    def stream_anscombe_metrology_experiment(self) -> dict[str, dict[str, Any]]:
        """
        Metrology Dilemma: Anscombe's Quartet.
        Four distinct physical processes that yield virtually identical summary statistics
        (mean, variance, correlation, linear regression), but reveal drastically different
        physical behavior when audited through multi-sensor residual tracking.
        """
        # Classic Anscombe datasets
        d1_x = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
        d1_y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

        d2_x = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
        d2_y = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])

        d3_x = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
        d3_y = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])

        d4_x = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0])
        d4_y = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])

        return {
            'dataset_1': {'x': d1_x, 'y': d1_y, 'nature': 'Linear response with normal error'},
            'dataset_2': {'x': d2_x, 'y': d2_y, 'nature': 'Non-linear quadratic relation'},
            'dataset_3': {'x': d3_x, 'y': d3_y, 'nature': 'Linear relation with outlier'},
            'dataset_4': {'x': d4_x, 'y': d4_y, 'nature': 'Clustered boundary lever point'}
        }


class CalibrationTraceabilityChain:
    """Maintains ISO/IEC 17025 metrological traceability chain: Sensor -> Standard -> Procedure -> Record."""

    def __init__(self):
        self.chain_records: list[dict[str, Any]] = []

    def log_calibration(
        self,
        sensor_id: str,
        standard_id: str,
        procedure_name: str,
        drift_compensation_factor: float,
        uncertainty_budget: float,
        timestamp: str
    ) -> str:
        record_id = f"CAL-{sensor_id}-{len(self.chain_records) + 1:04d}"
        record = {
            'record_id': record_id,
            'sensor_id': sensor_id,
            'standard_id': standard_id,
            'procedure_name': procedure_name,
            'drift_compensation_factor': drift_compensation_factor,
            'calibration_uncertainty': uncertainty_budget,
            'timestamp': timestamp,
            'status': 'VALID'
        }
        self.chain_records.append(record)
        return record_id

    def verify_traceability(self, sensor_id: str) -> bool:
        """Verifies an unbroken calibration link to a recognized standard."""
        records = [r for r in self.chain_records if r['sensor_id'] == sensor_id and r['status'] == 'VALID']
        return len(records) > 0

    def revoke_calibration(self, record_id: str, reason: str = "") -> bool:
        """Revokes a calibration record (e.g. after standard drift or calibration defect)."""
        for r in self.chain_records:
            if r['record_id'] == record_id:
                r['status'] = 'REVOKED'
                r['revocation_reason'] = reason
                return True
        return False

    def get_calibration_record(self, record_id: str) -> dict[str, Any] | None:
        """Retrieves a specific calibration record by ID."""
        for r in self.chain_records:
            if r['record_id'] == record_id:
                return dict(r)
        return None

