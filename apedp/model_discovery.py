"""Model-level discovery built on APEDP's equation and assurance primitives."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import BayesianRefinementEngine, MultiVariableSRE
from apedp.safety import safe_parse_expr
from apedp.scientific_layers import ExperimentDesign, design_experiments


@dataclass
class ModelCandidateResult:
    """Auditable result for one model structure."""
    name: str
    expression: str
    train_r_squared: float
    holdout_r_squared: float | None
    train_bic: float
    residual_standard_uncertainty: float
    model_form_uncertainty: float
    dimensional_valid: bool
    domain_status: str
    parameter_count: int
    evaluator: Callable | None = field(default=None, repr=False, compare=False)

    @property
    def generalizes(self) -> bool:
        return (
            self.dimensional_valid
            and self.domain_status == "PASS"
            and self.holdout_r_squared is not None
            and np.isfinite(self.holdout_r_squared)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "train_r_squared": self.train_r_squared,
            "holdout_r_squared": self.holdout_r_squared,
            "train_bic": self.train_bic,
            "residual_standard_uncertainty": self.residual_standard_uncertainty,
            "model_form_uncertainty": self.model_form_uncertainty,
            "dimensional_valid": self.dimensional_valid,
            "domain_status": self.domain_status,
            "parameter_count": self.parameter_count,
            "generalizes": self.generalizes,
        }


@dataclass
class ModelDiscoveryResult:
    """Ranked model structures plus the selected candidate and next experiment."""
    selected: ModelCandidateResult | None
    ranked: list[ModelCandidateResult]
    selection_basis: str
    experiment_design: ExperimentDesign | None = None
    regimes: dict[str, Any] | None = None
    outputs: dict[str, ModelDiscoveryResult] | None = None
    structure_posterior: dict[str, float] | None = None
    parameter_bootstrap: dict[str, list[str]] | None = None
    posterior_method: str | None = None
    exact_log_evidence: dict[str, float] | None = None
    parameter_posterior: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "ranked": [candidate.to_dict() for candidate in self.ranked],
            "selection_basis": self.selection_basis,
            "experiment_design": self.experiment_design.to_dict() if self.experiment_design else None,
            "regimes": self.regimes,
            "outputs": {name: result.to_dict() for name, result in (self.outputs or {}).items()}
            if self.outputs else None,
            "structure_posterior": self.structure_posterior,
            "parameter_bootstrap": self.parameter_bootstrap,
            "posterior_method": self.posterior_method,
            "exact_log_evidence": self.exact_log_evidence,
            "parameter_posterior": self.parameter_posterior,
        }


class ModelDiscoveryEngine:
    """Compare model structures and produce an auditable model package."""

    def __init__(self, variables: list[str], target_unit: str = "dimensionless",
                 variable_units: dict[str, str] | None = None):
        self.variables = list(variables)
        self.target_unit = target_unit
        self.engine = DimensionalAssuranceEngine()
        self.platform = ClosedLoopAssuranceSystem(self.engine)
        for name in self.variables:
            self.engine.register_variable(name, (variable_units or {}).get(name, "dimensionless"))

    def _parse_and_validate(self, expression: str | sp.Expr) -> tuple[sp.Expr, bool]:
        expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
        # Unknown fitted parameters are dimensionless unless callers encode their
        # units in the expression's known variables; this keeps model structure
        # comparison usable while still enforcing the target dimensionality.
        for symbol in expr.free_symbols:
            if symbol not in self.engine.symbol_units:
                self.engine.register_variable(str(symbol), "dimensionless")
        computed = self.engine.check_expression_consistency(expr)
        return expr, computed.dimensionality == self.engine.registry(self.target_unit).dimensionality

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return BayesianRefinementEngine.compute_r_squared(y_true, y_pred)

    def _evaluator(self, expression: sp.Expr) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
        symbols = [sp.Symbol(name) for name in sorted(self.variables)]
        constants = {sp.Symbol(name): value for name, value in self.platform.constants_values.items()}
        function = sp.lambdify(symbols, expression.subs(constants), modules=["numpy"])

        def evaluate(data: dict[str, np.ndarray]) -> np.ndarray:
            values = [np.asarray(data[name], dtype=float) for name in sorted(self.variables)]
            prediction = function(*values)
            if not isinstance(prediction, np.ndarray):
                prediction = np.full_like(values[0], float(prediction), dtype=float)
            prediction = np.asarray(prediction, dtype=float)
            if prediction.shape != values[0].shape or not np.all(np.isfinite(prediction)):
                raise ValueError("model evaluator returned a non-finite or mismatched prediction")
            return prediction

        return evaluate

    def discover(
        self,
        candidates: dict[str, str | sp.Expr],
        x_train: dict[str, np.ndarray],
        y_train: np.ndarray,
        *,
        x_holdout: dict[str, np.ndarray] | None = None,
        y_holdout: np.ndarray | None = None,
        design_grid: dict[str, np.ndarray] | None = None,
        uncertainty_fn: Callable | None = None,
        domain: dict[str, tuple[float, float]] | None = None,
    ) -> ModelDiscoveryResult:
        """Fit, validate, rank, and package competing equation structures."""
        y_train = np.asarray(y_train, dtype=float).ravel()
        if y_train.size == 0 or not np.all(np.isfinite(y_train)):
            raise ValueError("y_train must contain finite values")
        if set(x_train) != set(self.variables):
            raise ValueError("x_train keys must match the declared variables")
        if x_holdout is not None and set(x_holdout) != set(self.variables):
            raise ValueError("x_holdout keys must match the declared variables")
        # Fail fast on undeclared symbols: this engine fits one coefficient
        # per additive term itself (num_constants=0), so free parameters
        # like a/b/c are unfillable and previously died later with
        # "Cannot convert expression to float" -> inf -> silent skip.
        allowed = set(self.variables) | set(self.platform.constants_values)
        offenders: list = []
        for name, source in candidates.items():
            try:
                probe = (safe_parse_expr(source) if isinstance(source, str)
                         else source)
            except (TypeError, ValueError, ArithmeticError):
                continue
            for symbol in probe.free_symbols:
                if str(symbol) not in allowed:
                    offenders.append(f"{name}: '{symbol}'")
        if offenders:
            raise ValueError(
                "undeclared symbol(s) " + ", ".join(sorted(set(offenders))) +
                "; use bare forms like 'x**2 + x' (the engine fits one "
                "coefficient per additive term itself) or register symbols "
                "via platform constants")
        results: list[ModelCandidateResult] = []
        for name, source in candidates.items():
            try:
                expression, dimensional_valid = self._parse_and_validate(source)
                if not dimensional_valid:
                    continue
                sre = MultiVariableSRE(
                    self.platform,
                    independent_vars=self.variables,
                    target_unit_str=self.target_unit,
                    inner_fit=False,
                    guided=False,
                    num_constants=0,
                )
                fitted, _ = sre.optimize_parameters(expression, x_train, y_train)
                evaluator = self._evaluator(fitted)
                train_pred = evaluator(x_train)
                train_score = BayesianRefinementEngine.score_candidate(fitted, y_train, train_pred)
                residual = y_train - train_pred
                holdout_score = None
                model_form = 0.0
                if x_holdout is not None and y_holdout is not None:
                    holdout_pred = evaluator(x_holdout)
                    holdout_score = self._r2(np.asarray(y_holdout, dtype=float), holdout_pred)
                    model_form = abs(train_score["r_squared"] - holdout_score)
                status = "PASS"
                if domain:
                    for variable, (low, high) in domain.items():
                        values = np.asarray(x_train[variable], dtype=float)
                        if np.any(values < low) or np.any(values > high):
                            status = "OUTSIDE_DECLARED_DOMAIN"
                            break
                results.append(ModelCandidateResult(
                    name=name,
                    expression=str(fitted),
                    train_r_squared=float(train_score["r_squared"]),
                    holdout_r_squared=float(holdout_score) if holdout_score is not None else None,
                    train_bic=float(train_score["bic"]),
                    residual_standard_uncertainty=float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0,
                    model_form_uncertainty=float(model_form),
                    dimensional_valid=True,
                    domain_status=status,
                    parameter_count=max(1, len(fitted.free_symbols)),
                    evaluator=evaluator,
                ))
            except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                continue
        if not results:
            return ModelDiscoveryResult(None, [], "no dimensionally valid candidates")
        if x_holdout is not None and y_holdout is not None:
            ranked = sorted(results, key=lambda item: (
                item.holdout_r_squared if item.holdout_r_squared is not None else -np.inf,
                -item.model_form_uncertainty,
                -item.train_r_squared,
            ), reverse=True)
            basis = "holdout R², then lower train/holdout gap"
        else:
            ranked = sorted(results, key=lambda item: item.train_bic)
            basis = "training BIC (lower is better)"
        selected = ranked[0]
        experiment = None
        # The generic designer currently operates on a one-dimensional design
        # coordinate. Multi-dimensional design grids remain valid model inputs,
        # but require an explicit geometry/constraint policy and are not silently
        # flattened into a misleading 1D experiment.
        if len(ranked) > 1 and len(self.variables) == 1:
            variable = self.variables[0]
            grid_data = design_grid or x_train
            xs = np.asarray(grid_data[variable], dtype=float)
            evaluators = [
                (lambda points, evaluator=candidate.evaluator: evaluator({variable: points}))
                for candidate in ranked if candidate.evaluator
            ]
            experiment = design_experiments(
                evaluators,
                xs,
                uncertainty_fn=uncertainty_fn,
                n_points=min(3, len(ranked)),
            )
        return ModelDiscoveryResult(selected, ranked, basis, experiment_design=experiment)

    def _design_matrix(self, expression: sp.Expr, data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
        """Evaluate additive structural terms as a linear design matrix."""
        parts = list(sp.Add.make_args(expression))
        symbols = [sp.Symbol(name) for name in sorted(self.variables)]
        constants = {sp.Symbol(name): value for name, value in self.platform.constants_values.items()}
        columns = []
        labels = []
        for part in parts:
            function = sp.lambdify(symbols, part.subs(constants), modules=["numpy"])
            values = [np.asarray(data[name], dtype=float) for name in sorted(self.variables)]
            column = np.asarray(function(*values), dtype=float)
            if column.ndim == 0:
                column = np.full(len(values[0]), float(column))
            if column.shape != values[0].shape or not np.all(np.isfinite(column)):
                raise ValueError("candidate term produced invalid design values")
            columns.append(column.ravel())
            labels.append(str(part))
        return np.column_stack(columns), labels

    def discover_exact_bayesian(
        self,
        candidates: dict[str, str | sp.Expr],
        x_train: dict[str, np.ndarray],
        y_train: np.ndarray,
        *,
        noise_std: float,
        prior_scale: float = 10.0,
        x_holdout: dict[str, np.ndarray] | None = None,
        y_holdout: np.ndarray | None = None,
    ) -> ModelDiscoveryResult:
        """Compute exact finite-candidate evidence for linear outer parameters.

        Each supplied symbolic structure is treated as a finite model and its
        additive terms receive an independent Gaussian coefficient prior
        ``N(0, prior_scale²)``. With known Gaussian observation noise, the
        coefficients are analytically integrated out and their Gaussian
        posterior mean/covariance are returned. This is exact for the supplied
        finite candidate set and stated prior, not exact inference over an
        unbounded symbolic program space.
        """
        if not np.isfinite(noise_std) or noise_std <= 0.0:
            raise ValueError("noise_std must be finite and positive")
        if not np.isfinite(prior_scale) or prior_scale <= 0.0:
            raise ValueError("prior_scale must be finite and positive")
        base = self.discover(candidates, x_train, y_train,
                             x_holdout=x_holdout, y_holdout=y_holdout)
        log_evidence: dict[str, float] = {}
        parameter_posterior: dict[str, dict[str, Any]] = {}
        y = np.asarray(y_train, dtype=float).ravel()
        n = len(y)
        for name, source in candidates.items():
            try:
                expression, dimensional_valid = self._parse_and_validate(source)
                if not dimensional_valid:
                    continue
                design, terms = self._design_matrix(expression, x_train)
                covariance = noise_std ** 2 * np.eye(n) + prior_scale ** 2 * (design @ design.T)
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0 or not np.isfinite(logdet):
                    continue
                solved = np.linalg.solve(covariance, y)
                log_evidence[name] = float(-0.5 * (n * np.log(2.0 * np.pi) + logdet + y @ solved))
                posterior_mean = prior_scale ** 2 * design.T @ solved
                posterior_covariance = prior_scale ** 2 * np.eye(design.shape[1]) - (
                    prior_scale ** 4 * design.T @ np.linalg.solve(covariance, design)
                )
                parameter_posterior[name] = {
                    "terms": terms,
                    "mean": [float(value) for value in posterior_mean],
                    "std": [float(np.sqrt(max(0.0, value))) for value in np.diag(posterior_covariance)],
                    "covariance": posterior_covariance.tolist(),
                }
            except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
                continue
        if not log_evidence:
            raise ValueError("no candidate supports finite exact Gaussian marginal likelihood")
        names = list(log_evidence)
        values = np.asarray([log_evidence[name] for name in names])
        weights = np.exp(values - np.max(values))
        weights /= np.sum(weights)
        base.structure_posterior = {name: float(weight) for name, weight in zip(names, weights)}
        base.exact_log_evidence = log_evidence
        base.parameter_posterior = parameter_posterior
        base.posterior_method = (
            "exact finite-candidate Gaussian marginal likelihood with "
            "known noise and independent Gaussian coefficient priors"
        )
        return base

    def discover_bayesian(
        self,
        candidates: dict[str, str | sp.Expr],
        x_train: dict[str, np.ndarray],
        y_train: np.ndarray,
        *,
        x_holdout: dict[str, np.ndarray] | None = None,
        y_holdout: np.ndarray | None = None,
        bootstrap_samples: int = 16,
        seed: int = 42,
    ) -> ModelDiscoveryResult:
        """Estimate joint structure/parameter uncertainty.

        Structure probabilities use the normalized BIC approximation
        ``P(M|D) ∝ exp(-BIC/2)``. Parameter uncertainty is represented by
        bootstrap refits of each structure, returned as fitted expression
        strings. This is an auditable approximation—not exact integration over
        symbolic structures—and its method is carried in the result.
        """
        if bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        base = self.discover(candidates, x_train, y_train,
                             x_holdout=x_holdout, y_holdout=y_holdout)
        if not base.ranked:
            base.posterior_method = "no posterior: no valid candidates"
            return base
        bics = np.asarray([candidate.train_bic for candidate in base.ranked], dtype=float)
        finite = np.isfinite(bics)
        if not np.any(finite):
            raise ValueError("cannot form a structure posterior from non-finite BIC values")
        log_weights = -0.5 * (bics - np.min(bics[finite]))
        log_weights[~finite] = -np.inf
        weights = np.exp(log_weights - np.max(log_weights[finite]))
        weights = weights / np.sum(weights)
        posterior = {candidate.name: float(weight)
                     for candidate, weight in zip(base.ranked, weights)}
        rng = np.random.default_rng(seed)
        parameter_bootstrap: dict[str, list[str]] = {candidate.name: [] for candidate in base.ranked}
        n = len(np.asarray(y_train))
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, n, size=n)
            boot_x = {key: np.asarray(value)[indices] for key, value in x_train.items()}
            boot_y = np.asarray(y_train)[indices]
            for name, source in candidates.items():
                boot = self.discover({name: source}, boot_x, boot_y)
                if boot.selected is not None:
                    parameter_bootstrap.setdefault(name, []).append(boot.selected.expression)
        base.structure_posterior = posterior
        base.parameter_bootstrap = parameter_bootstrap
        base.posterior_method = "BIC structure pseudo-posterior + nonparametric bootstrap refits"
        return base

    def discover_regimes(
        self,
        candidates: dict[str, str | sp.Expr],
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        split_variable: str,
        *,
        min_segment_size: int = 8,
    ) -> dict[str, Any]:
        """Search quantile split points and fit separate models per regime."""
        if split_variable not in self.variables:
            raise ValueError(f"unknown split variable: {split_variable}")
        order = np.argsort(np.asarray(x_data[split_variable], dtype=float))
        n = len(order)
        best: dict[str, Any] | None = None
        for index in range(min_segment_size, n - min_segment_size + 1):
            left = order[:index]
            right = order[index:]
            left_result = self.discover(candidates, {key: np.asarray(value)[left] for key, value in x_data.items()},
                                         np.asarray(y_data)[left])
            right_result = self.discover(candidates, {key: np.asarray(value)[right] for key, value in x_data.items()},
                                          np.asarray(y_data)[right])
            if left_result.selected is None or right_result.selected is None:
                continue
            score = left_result.selected.train_bic + right_result.selected.train_bic
            if best is None or score < best["combined_bic"]:
                best = {"split_variable": split_variable, "threshold": float(np.asarray(x_data[split_variable])[order[index]]),
                        "combined_bic": float(score), "left": left_result.to_dict(), "right": right_result.to_dict()}
        return best or {"status": "no_valid_regime_split"}

    def discover_multi_output(
        self,
        candidates: dict[str, str | sp.Expr],
        x_data: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> ModelDiscoveryResult:
        """Discover one auditable model per output while sharing input structure."""
        outputs = {
            name: self.discover(candidates, x_data, values, **kwargs)
            for name, values in targets.items()
        }
        selected = [result.selected for result in outputs.values() if result.selected]
        return ModelDiscoveryResult(
            selected=selected[0] if len(selected) == 1 else None,
            ranked=selected,
            selection_basis="independent output model selection",
            outputs=outputs,
        )
