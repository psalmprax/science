"""Evidence Acquisition Engine for APEDP (Active Scientific Discrimination).

Answers the core epistemic question:
"What experiment or observation would maximally reduce uncertainty between competing hypotheses?"

Given surviving admissible hypotheses (e.g. H1 = Shomate, H2 = Quadratic):
1. Evaluates predictive divergence Delta(x) = |y_H1(x) - y_H2(x)| across the valid physical domain.
2. Identifies x* = argmax Delta(x) subject to domain bounds.
3. Partitions the domain into INTERPOLATION_GAP, NEAR_OOD, and FAR_OOD regimes.
4. Generates an actionable, discriminating experimental design proposal.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscriminationPoint:
    """A specific coordinates vector where competing hypotheses maximally diverge."""

    coordinates: dict[str, float]
    max_divergence: float
    relative_divergence: float
    predictions: dict[str, float]
    regime: str  # "INTERPOLATION_GAP", "NEAR_OOD", "FAR_OOD", "UNSPECIFIED"
    discriminating_pair: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": dict(self.coordinates),
            "max_divergence": float(self.max_divergence),
            "relative_divergence": float(self.relative_divergence),
            "predictions": dict(self.predictions),
            "regime": self.regime,
            "discriminating_pair": list(self.discriminating_pair),
        }


@dataclass(frozen=True)
class DiscriminationReport:
    """Actionable scientific proposal identifying where to collect the next evidence."""

    candidate_names: tuple[str, ...]
    primary_recommendation: DiscriminationPoint | None
    regime_recommendations: dict[str, DiscriminationPoint]
    rationale: str
    variable_domains: dict[str, tuple[float, float]]
    observed_domain: dict[str, tuple[float, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_names": list(self.candidate_names),
            "primary_recommendation": (
                self.primary_recommendation.to_dict() if self.primary_recommendation else None
            ),
            "regime_recommendations": {
                k: v.to_dict() for k, v in self.regime_recommendations.items()
            },
            "rationale": self.rationale,
            "variable_domains": {k: list(v) for k, v in self.variable_domains.items()},
            "observed_domain": (
                {k: list(v) for k, v in self.observed_domain.items()} if self.observed_domain else None
            ),
        }


class EvidenceAcquisitionEngine:
    """Calculates optimal points of maximal epistemic divergence among hypotheses."""

    def __init__(self, default_grid_points: int = 250):
        self.default_grid_points = default_grid_points

    @staticmethod
    def _compile_model(
        model_spec: str | sp.Expr | Callable | dict[str, Any],
        variables: list[str],
    ) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
        """Converts model specifications into safe vectorized callables."""
        if callable(model_spec):
            def safe_callable_eval(inputs: dict[str, np.ndarray]) -> np.ndarray:
                try:
                    res = model_spec(**inputs)
                except TypeError:
                    res = model_spec(inputs)
                arr = np.asarray(res, dtype=float)
                if arr.ndim == 0:
                    arr = np.full(len(inputs[variables[0]]), float(arr))
                return arr

            return safe_callable_eval

        if isinstance(model_spec, dict):
            expr_raw = model_spec.get("expression")
            params = model_spec.get("parameters", {})
        else:
            expr_raw = model_spec
            params = {}

        if isinstance(expr_raw, str):
            parsed = safe_parse_expr(expr_raw)
        elif isinstance(expr_raw, sp.Expr):
            parsed = expr_raw
        else:
            raise TypeError(f"Unsupported model expression type: {type(expr_raw)}")

        if params:
            substitutions = {sp.Symbol(k): v for k, v in params.items()}
            parsed = parsed.subs(substitutions)

        sym_vars = [sp.Symbol(v) for v in variables]
        fn = sp.lambdify(sym_vars, parsed, modules=["numpy"])

        def evaluator(inputs: dict[str, np.ndarray]) -> np.ndarray:
            args = [inputs[v] for v in variables]
            with np.errstate(all="ignore"):
                raw_out = fn(*args)
            arr = np.asarray(raw_out, dtype=float)
            if arr.ndim == 0:
                arr = np.full(len(inputs[variables[0]]), float(arr))
            return arr

        return evaluator

    def _classify_regime(
        self,
        coords: dict[str, float],
        observed_domain: dict[str, tuple[float, float]] | None,
    ) -> str:
        if not observed_domain:
            return "UNSPECIFIED"

        all_in_domain = True
        near_ood = True

        for var, val in coords.items():
            if var not in observed_domain:
                continue
            obs_min, obs_max = observed_domain[var]
            obs_span = max(abs(obs_max - obs_min), 1e-6)

            if not (obs_min <= val <= obs_max):
                all_in_domain = False

            # Near OOD defined as within 1.5x of observed span outside boundaries
            near_min = obs_min - 0.5 * obs_span
            near_max = obs_max + 0.5 * obs_span
            if not (near_min <= val <= near_max):
                near_ood = False

        if all_in_domain:
            return "INTERPOLATION_GAP"
        if near_ood:
            return "NEAR_OOD"
        return "FAR_OOD"

    def find_discriminating_regime(
        self,
        # Read-only, so `Mapping`: a `dict[str, str]` (the caller's shape) is not
        # assignable to a dict with a wider value type, and nothing here writes.
        models: Mapping[str, str | sp.Expr | Callable | dict[str, Any]],
        variable_domains: dict[str, tuple[float, float]],
        observed_domain: dict[str, tuple[float, float]] | None = None,
        n_points: int | None = None,
    ) -> DiscriminationReport:
        """Finds domain regions where predictions between competing models diverge maximally."""
        names = sorted(models.keys())
        if len(names) < 2:
            return DiscriminationReport(
                candidate_names=tuple(names),
                primary_recommendation=None,
                regime_recommendations={},
                rationale="At least two competing hypotheses are required to calculate discrimination.",
                variable_domains=variable_domains,
                observed_domain=observed_domain,
            )

        variables = sorted(variable_domains.keys())
        grid_n = n_points or self.default_grid_points

        # Generate sample coordinates
        if len(variables) == 1:
            var = variables[0]
            low, high = variable_domains[var]
            grid_vals = np.linspace(low, high, grid_n)
            eval_grid = {var: grid_vals}
            n_eval = grid_n
        else:
            # Multi-dimensional grid
            per_var = max(2, round(grid_n ** (1.0 / len(variables))))
            axes = [np.linspace(variable_domains[v][0], variable_domains[v][1], per_var) for v in variables]
            mesh = np.meshgrid(*axes, indexing="ij")
            eval_grid = {v: mesh[i].ravel() for i, v in enumerate(variables)}
            n_eval = len(mesh[0].ravel())

        # Compile and evaluate all models
        compiled = {name: self._compile_model(models[name], variables) for name in names}
        predictions: dict[str, np.ndarray] = {}
        for name in names:
            try:
                preds = compiled[name](eval_grid)
                predictions[name] = preds
            except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
                logger.warning(f"Model {name} failed evaluation on grid: {exc}")
                predictions[name] = np.full(n_eval, np.nan)

        # Pairwise and ensemble divergence
        max_div_per_point = np.zeros(n_eval)
        rel_div_per_point = np.zeros(n_eval)
        best_pair_per_point: list[tuple[str, str]] = [("", "")] * n_eval

        for idx in range(n_eval):
            pt_preds = {name: float(predictions[name][idx]) for name in names}
            finite_preds = {k: v for k, v in pt_preds.items() if np.isfinite(v)}

            if len(finite_preds) < 2:
                continue

            max_d = -1.0
            top_pair = ("", "")
            pred_items = list(finite_preds.items())

            for i in range(len(pred_items)):
                for j in range(i + 1, len(pred_items)):
                    name_a, val_a = pred_items[i]
                    name_b, val_b = pred_items[j]
                    d = abs(val_a - val_b)
                    if d > max_d:
                        max_d = d
                        top_pair = (name_a, name_b)

            max_div_per_point[idx] = max_d
            best_pair_per_point[idx] = top_pair

            mean_scale = np.mean([abs(v) for v in finite_preds.values()])
            rel_div_per_point[idx] = max_d / (mean_scale + 1e-9)

        # Partition points into regimes
        regime_candidates: dict[str, list[int]] = {
            "INTERPOLATION_GAP": [],
            "NEAR_OOD": [],
            "FAR_OOD": [],
            "UNSPECIFIED": [],
        }

        for idx in range(n_eval):
            if max_div_per_point[idx] <= 0.0:
                continue
            coords = {v: float(eval_grid[v][idx]) for v in variables}
            reg = self._classify_regime(coords, observed_domain)
            regime_candidates[reg].append(idx)

        # Select best point in each regime
        regime_recommendations: dict[str, DiscriminationPoint] = {}
        for reg, indices in regime_candidates.items():
            if not indices:
                continue
            best_idx = max(indices, key=lambda i: max_div_per_point[i])
            coords = {v: float(eval_grid[v][best_idx]) for v in variables}
            regime_recommendations[reg] = DiscriminationPoint(
                coordinates=coords,
                max_divergence=float(max_div_per_point[best_idx]),
                relative_divergence=float(rel_div_per_point[best_idx]),
                predictions={name: float(predictions[name][best_idx]) for name in names if np.isfinite(predictions[name][best_idx])},
                regime=reg,
                discriminating_pair=best_pair_per_point[best_idx],
            )

        # Determine primary recommendation: prefer NEAR_OOD or INTERPOLATION_GAP if divergence is significant
        primary: DiscriminationPoint | None = None
        if "NEAR_OOD" in regime_recommendations and regime_recommendations["NEAR_OOD"].max_divergence > 1e-5:
            primary = regime_recommendations["NEAR_OOD"]
        elif "INTERPOLATION_GAP" in regime_recommendations and regime_recommendations["INTERPOLATION_GAP"].max_divergence > 1e-5:
            primary = regime_recommendations["INTERPOLATION_GAP"]
        elif "FAR_OOD" in regime_recommendations:
            primary = regime_recommendations["FAR_OOD"]
        elif "UNSPECIFIED" in regime_recommendations:
            primary = regime_recommendations["UNSPECIFIED"]

        # Synthesize clear, readable rationale
        if primary is None:
            rationale = "The competing hypotheses do not diverge meaningfully over the tested domain."
        else:
            coord_str = ", ".join(f"{k} = {v:.4g}" for k, v in primary.coordinates.items())
            pair_a, pair_b = primary.discriminating_pair
            val_a = primary.predictions.get(pair_a, 0.0)
            val_b = primary.predictions.get(pair_b, 0.0)

            rationale = (
                f"Measurements near {coord_str} ({primary.regime}) provide maximal discrimination "
                f"between '{pair_a}' ({val_a:.4g}) and '{pair_b}' ({val_b:.4g}), with divergence Delta = {primary.max_divergence:.4g} "
                f"({primary.relative_divergence:.1%} relative)."
            )
            if "INTERPOLATION_GAP" in regime_recommendations and primary.regime != "INTERPOLATION_GAP":
                in_gap = regime_recommendations["INTERPOLATION_GAP"]
                gap_coords = ", ".join(f"{k} = {v:.4g}" for k, v in in_gap.coordinates.items())
                rationale += (
                    f" Within the observed domain, maximal divergence occurs at {gap_coords} "
                    f"(Delta = {in_gap.max_divergence:.4g})."
                )

        return DiscriminationReport(
            candidate_names=tuple(names),
            primary_recommendation=primary,
            regime_recommendations=regime_recommendations,
            rationale=rationale,
            variable_domains=variable_domains,
            observed_domain=observed_domain,
        )

    find_maximal_divergence = find_discriminating_regime

    def find_bayesian_eig_regime(
        self,
        # Read-only, so `Mapping`: a `dict[str, str]` (the caller's shape) is not
        # assignable to a dict with a wider value type, and nothing here writes.
        models: Mapping[str, str | sp.Expr | Callable | dict[str, Any]],
        variable_domains: dict[str, tuple[float, float]],
        model_priors: dict[str, float] | None = None,
        parameter_uncertainties: dict[str, float] | None = None,
        sensor_noise: float | Callable[[dict[str, float]], float] = 1.0,
        cost_function: Callable[[dict[str, float]], float] | None = None,
        cost_weight: float = 0.0,
        observed_domain: dict[str, tuple[float, float]] | None = None,
        n_points: int | None = None,
    ) -> BayesianEIGReport:
        """AHD-v2: Optimal experimental design via Bayesian Expected Information Gain (EIG).

        Computes the expected reduction in model-space entropy:
            EIG(x) = 0.5 * ln(1 + V_between(x) / V_within(x))
        Penalizes measurement points where models diverge solely due to unidentifiable
        parameter extrapolation variance or high sensor noise.
        """
        names = sorted(models.keys())
        if len(names) < 2:
            return BayesianEIGReport(
                candidate_names=tuple(names),
                primary_recommendation=None,
                regime_recommendations={},
                model_priors=model_priors or {},
                rationale="At least two competing hypotheses are required to compute Bayesian EIG.",
                variable_domains=variable_domains,
                observed_domain=observed_domain,
            )

        # Normalize model prior probabilities
        if model_priors is None:
            priors = {name: 1.0 / len(names) for name in names}
        else:
            total_prior = sum(model_priors.get(n, 1.0) for n in names)
            priors = {n: model_priors.get(n, 1.0) / max(total_prior, 1e-12) for n in names}

        param_unc = parameter_uncertainties or dict.fromkeys(names, 0.05)

        variables = sorted(variable_domains.keys())
        grid_n = n_points or self.default_grid_points

        if len(variables) == 1:
            var = variables[0]
            low, high = variable_domains[var]
            grid_vals = np.linspace(low, high, grid_n)
            eval_grid = {var: grid_vals}
            n_eval = grid_n
        else:
            per_var = max(2, round(grid_n ** (1.0 / len(variables))))
            axes = [np.linspace(variable_domains[v][0], variable_domains[v][1], per_var) for v in variables]
            mesh = np.meshgrid(*axes, indexing="ij")
            eval_grid = {v: mesh[i].ravel() for i, v in enumerate(variables)}
            n_eval = len(mesh[0].ravel())

        # Compile and evaluate models
        compiled = {name: self._compile_model(models[name], variables) for name in names}
        predictions: dict[str, np.ndarray] = {}
        for name in names:
            try:
                predictions[name] = compiled[name](eval_grid)
            except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
                logger.warning(f"Model {name} failed evaluation: {exc}")
                predictions[name] = np.full(n_eval, np.nan)

        eig_scores = np.zeros(n_eval)
        net_utilities = np.zeros(n_eval)
        v_betweens = np.zeros(n_eval)
        v_withins = np.zeros(n_eval)
        point_param_stds: list[dict[str, float]] = []

        for idx in range(n_eval):
            coords = {v: float(eval_grid[v][idx]) for v in variables}
            pt_preds = {name: float(predictions[name][idx]) for name in names}
            finite_preds = {k: v for k, v in pt_preds.items() if np.isfinite(v)}

            if len(finite_preds) < 2:
                point_param_stds.append({})
                continue

            # Compute sensor noise variance at coords
            if callable(sensor_noise):
                sigma_meas = float(sensor_noise(coords))
            else:
                sigma_meas = float(sensor_noise)
            sigma_meas_sq = max(sigma_meas ** 2, 1e-12)

            # Extrapolation distance factor for parameter uncertainty growth
            extrap_factor = 1.0
            if observed_domain:
                for v in variables:
                    if v in observed_domain:
                        obs_min, obs_max = observed_domain[v]
                        val = coords[v]
                        if val < obs_min:
                            extrap_factor += (obs_min - val) / max(obs_max - obs_min, 1e-6)
                        elif val > obs_max:
                            extrap_factor += (val - obs_max) / max(obs_max - obs_min, 1e-6)

            # Predictive variance per model: sensor noise + parameter spread
            model_stds: dict[str, float] = {}
            for name in names:
                base_u = param_unc.get(name, 0.05)
                # Parameter spread expands away from observed support
                model_stds[name] = float(base_u * extrap_factor)
            point_param_stds.append(model_stds)

            # Calculate between-model and within-model variances
            active_weights = {k: priors[k] for k in finite_preds}
            weight_sum = sum(active_weights.values())
            norm_weights = {k: v / weight_sum for k, v in active_weights.items()}

            mean_pred = sum(norm_weights[k] * finite_preds[k] for k in finite_preds)
            v_between = sum(norm_weights[k] * (finite_preds[k] - mean_pred) ** 2 for k in finite_preds)
            v_within = sum(
                norm_weights[k] * (sigma_meas_sq + model_stds[k] ** 2) for k in finite_preds
            )

            # Bayesian Expected Information Gain
            eig = 0.5 * np.log(1.0 + v_between / max(v_within, 1e-12))

            # Net utility with cost penalty
            cost = 0.0
            if cost_function is not None:
                cost = float(cost_function(coords))
            utility = eig - cost_weight * cost

            eig_scores[idx] = eig
            net_utilities[idx] = utility
            v_betweens[idx] = v_between
            v_withins[idx] = v_within

        # Partition points into regimes
        regime_candidates: dict[str, list[int]] = {
            "INTERPOLATION_GAP": [],
            "NEAR_OOD": [],
            "FAR_OOD": [],
            "UNSPECIFIED": [],
        }

        for idx in range(n_eval):
            if eig_scores[idx] <= 1e-6:
                continue
            coords = {v: float(eval_grid[v][idx]) for v in variables}
            reg = self._classify_regime(coords, observed_domain)
            regime_candidates[reg].append(idx)

        # Select best point in each regime based on net utility
        regime_recommendations: dict[str, EIGPoint] = {}
        for reg, indices in regime_candidates.items():
            if not indices:
                continue
            best_idx = max(indices, key=lambda i: net_utilities[i])
            coords = {v: float(eval_grid[v][best_idx]) for v in variables}
            regime_recommendations[reg] = EIGPoint(
                coordinates=coords,
                expected_information_gain=float(eig_scores[best_idx]),
                net_utility=float(net_utilities[best_idx]),
                between_model_variance=float(v_betweens[best_idx]),
                within_model_variance=float(v_withins[best_idx]),
                predictions={name: float(predictions[name][best_idx]) for name in names if np.isfinite(predictions[name][best_idx])},
                parameter_std_devs=point_param_stds[best_idx],
                regime=reg,
            )

        # Determine primary recommendation (highest net utility across regimes)
        primary: EIGPoint | None = None
        if regime_recommendations:
            primary = max(regime_recommendations.values(), key=lambda pt: pt.net_utility)

        if primary is None:
            rationale = "No regime provides significant expected information gain over current knowledge."
        else:
            coord_str = ", ".join(f"{k} = {v:.4g}" for k, v in primary.coordinates.items())
            rationale = (
                f"Optimal Bayesian experiment at {coord_str} ({primary.regime}): "
                f"Yields maximum expected information gain EIG = {primary.expected_information_gain:.3f} nats "
                f"(between-model variance {primary.between_model_variance:.4g} vs within-model variance {primary.within_model_variance:.4g})."
            )
            if "INTERPOLATION_GAP" in regime_recommendations and primary.regime != "INTERPOLATION_GAP":
                in_gap = regime_recommendations["INTERPOLATION_GAP"]
                gap_coords = ", ".join(f"{k} = {v:.4g}" for k, v in in_gap.coordinates.items())
                rationale += (
                    f" Within the observed domain, optimal observation is at {gap_coords} "
                    f"(EIG = {in_gap.expected_information_gain:.3f} nats)."
                )

        return BayesianEIGReport(
            candidate_names=tuple(names),
            primary_recommendation=primary,
            regime_recommendations=regime_recommendations,
            model_priors=priors,
            rationale=rationale,
            variable_domains=variable_domains,
            observed_domain=observed_domain,
        )


@dataclass(frozen=True)
class EIGPoint:
    """A specific coordinates vector scored by Bayesian Expected Information Gain."""

    coordinates: dict[str, float]
    expected_information_gain: float  # EIG in nats
    net_utility: float  # EIG - cost_penalty
    between_model_variance: float
    within_model_variance: float
    predictions: dict[str, float]
    parameter_std_devs: dict[str, float]
    regime: str  # "INTERPOLATION_GAP", "NEAR_OOD", "FAR_OOD", "UNSPECIFIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": dict(self.coordinates),
            "expected_information_gain": float(self.expected_information_gain),
            "net_utility": float(self.net_utility),
            "between_model_variance": float(self.between_model_variance),
            "within_model_variance": float(self.within_model_variance),
            "predictions": dict(self.predictions),
            "parameter_std_devs": dict(self.parameter_std_devs),
            "regime": self.regime,
        }


@dataclass(frozen=True)
class BayesianEIGReport:
    """Actionable optimal experimental design proposal based on Bayesian EIG (AHD-v2)."""

    candidate_names: tuple[str, ...]
    primary_recommendation: EIGPoint | None
    regime_recommendations: dict[str, EIGPoint]
    model_priors: dict[str, float]
    rationale: str
    variable_domains: dict[str, tuple[float, float]]
    observed_domain: dict[str, tuple[float, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_names": list(self.candidate_names),
            "primary_recommendation": (
                self.primary_recommendation.to_dict() if self.primary_recommendation else None
            ),
            "regime_recommendations": {
                k: v.to_dict() for k, v in self.regime_recommendations.items()
            },
            "model_priors": dict(self.model_priors),
            "rationale": self.rationale,
            "variable_domains": {k: list(v) for k, v in self.variable_domains.items()},
            "observed_domain": (
                {k: list(v) for k, v in self.observed_domain.items()} if self.observed_domain else None
            ),
        }


__all__ = [
    "BayesianEIGReport",
    "DiscriminationPoint",
    "DiscriminationReport",
    "EIGPoint",
    "EvidenceAcquisitionEngine",
]
