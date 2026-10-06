"""Bayesian hypothesis comparison and model evidence refinement.

Extracted from discovery.py to provide modular, robust model evaluation,
information criteria (BIC, AIC, AICc), marginal likelihood integration
(quadrature, Laplace), and Kass-Raftery evidence grading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp

from apedp.family_detection import STANDARD_RATIONALS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvidence:
    """The marginal likelihood ``p(D|M)`` of one model, and how it was obtained.

    ``p(D|M) = ∫ p(D|θ,M) p(θ|M) dθ``: the likelihood is *integrated over the
    prior*, not evaluated at a point. A ratio of summed pointwise likelihoods has
    neither, so it is a different (weaker) statistic with a different home --
    :meth:`BayesianRefinementEngine.compute_likelihood_ratio`.

    ``method`` is the approximation used (``"quadrature"``, ``"laplace"``,
    ``"bic"``); ``prior_description`` records the prior the integral was taken
    against. Both travel with the number so a verdict can state its own basis.
    """

    log_evidence: float
    method: str
    n_params: int
    prior_description: str = "unspecified"

    def __str__(self) -> str:
        return (
            f"log p(D|M) = {self.log_evidence:.4f} "
            f"[{self.method}, {self.n_params} param(s), prior: {self.prior_description}]"
        )


def _log_integrand(
    log_likelihood_fn: Callable[[np.ndarray], float],
    prior_logpdf: Callable[[np.ndarray], float],
    point: np.ndarray | list[float],
) -> float:
    """``log p(D|θ) + log p(θ)`` at one parameter point."""
    p = np.asarray(point, dtype=float)
    return float(log_likelihood_fn(p)) + float(prior_logpdf(p))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Composite trapezoid rule on a uniform or non-uniform grid.

    Written out rather than calling ``np.trapz``, which NumPy 2 removed.
    """
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _log_integrate(log_values: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule on ``exp(log_values)`` without leaving log space.

    Non-finite values are treated as a zero integrand (a hard support boundary),
    not as an annihilation of their neighbourhood.
    """
    finite = np.isfinite(log_values)
    if not bool(np.any(finite)):
        return float("-inf")
    m = float(np.max(log_values[finite]))
    area = _trapezoid(np.where(finite, np.exp(log_values - m), 0.0), x)
    if area <= 0.0 or not np.isfinite(area):
        return float("-inf")
    return float(m + np.log(area))


def _finite_difference_hessian(
    f: Callable[[np.ndarray], float],
    theta: np.ndarray,
    rel_step: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``theta`` (step scaled per parameter)."""
    k = int(theta.size)
    h = rel_step * np.maximum(np.abs(theta), 1.0)
    hessian = np.zeros((k, k))
    f0 = float(f(theta))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        hessian[i, i] = (f(theta + ei) - 2.0 * f0 + f(theta - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            hessian[i, j] = hessian[j, i] = (
                f(theta + ei + ej)
                - f(theta + ei - ej)
                - f(theta - ei + ej)
                + f(theta - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return hessian


def _snap_coefficient(value: float) -> sp.Expr:
    for target, rational_val in STANDARD_RATIONALS:
        if abs(value - target) < 1e-3:
            return rational_val
    return sp.Float(value)


def _dim_dict(unit) -> dict[str, float]:
    """Pint dimensionality as a plain {dimension: exponent} dict."""
    return {str(k): float(v) for k, v in unit.dimensionality.items() if float(v) != 0.0}


def _dim_add(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
        if out[k] == 0.0:
            del out[k]
    return out


def _dim_scale(a: dict[str, float], p: float) -> dict[str, float]:
    return {k: v * p for k, v in a.items() if v * p != 0.0}


class BayesianRefinementEngine:
    """Bayesian hypothesis comparison and model selection engine.

    .. note::
        ``compute_bayes_factor`` is a legacy heuristic (ratio of summed
        likelihoods). It is kept for backwards compatibility but must NOT
        be used for discovery decisions. Use ``gaussian_log_likelihood`` +
        ``compute_bic``/``compute_aic`` + ``select_best_model`` instead,
        which implement a real Gaussian-noise log-likelihood and standard
        information-criterion selection.
    """

    @staticmethod
    def gaussian_log_likelihood(
        y_true: np.ndarray, y_pred: np.ndarray, noise_std: float | None = None
    ) -> tuple[float, float]:
        """Gaussian log-likelihood with ML noise estimate when not supplied.

        Returns ``(log_likelihood, noise_std_used)``. ``log_likelihood`` is
        ``-n/2 * (log(2*pi*sigma^2) + 1)`` at the ML sigma, the standard
        form used for BIC/AIC comparison.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        resid = y_true - y_pred
        n = int(y_true.size)
        if n == 0:
            raise ValueError("Empty data for likelihood computation")
        if noise_std is None:
            sigma2 = float(np.mean(resid**2))
            sigma2 = max(sigma2, 1e-300)
        else:
            sigma2 = max(float(noise_std) ** 2, 1e-300)
        log_lik = float(-0.5 * n * (np.log(2.0 * np.pi * sigma2) + np.mean(resid**2) / sigma2))
        return log_lik, float(np.sqrt(sigma2))

    @staticmethod
    def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of determination. Returns ``nan`` for degenerate inputs."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0 or not np.all(np.isfinite(y_pred)):
            return float("nan")
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 0 or not np.isfinite(ss_tot):
            return float("nan")
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def is_degenerate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        expression: sp.Expr | None = None,
        min_r_squared: float = 0.1,
    ) -> tuple[bool, str]:
        """Decide whether a candidate is degenerate and must be refused.

        A candidate is degenerate when predictions are non-finite, constant
        (zero variance while the target varies), R² is NaN, or R² is below
        ``min_r_squared``. Returns ``(is_degenerate, reason)``.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != y_true.shape:
            return True, f"shape mismatch {y_pred.shape} vs {y_true.shape}"
        if y_true.size == 0:
            return True, "no data points to score"
        if not np.all(np.isfinite(y_pred)):
            return True, "non-finite predictions"
        if float(np.std(y_pred)) <= 1e-12 and float(np.std(y_true)) > 1e-12:
            return True, "constant prediction for varying target"
        r2 = BayesianRefinementEngine.compute_r_squared(y_true, y_pred)
        if not np.isfinite(r2):
            return True, "R² is NaN/non-finite"
        if r2 < min_r_squared:
            return True, f"R²={r2:.4f} below threshold {min_r_squared}"
        if expression is not None and expression in (sp.Integer(0), sp.Float(0.0)):
            return True, "fallback zero expression"
        return False, f"OK (R²={r2:.4f})"

    @staticmethod
    def select_best_model(candidates: list[dict]) -> dict | None:
        """Select the best candidate by BIC (lower is better).

        Each candidate dict must carry ``bic`` (float, finite). Ties break
        on lower ``complexity``. Returns ``None`` when no finite-BIC
        candidate exists — the caller must then refuse to persist.
        """
        finite = [c for c in candidates if np.isfinite(float(c.get("bic", np.inf)))]
        if not finite:
            return None
        return min(finite, key=lambda c: (float(c["bic"]), int(c.get("complexity", 0))))

    # ------------------------------------------------------------------
    # Model evidence
    # ------------------------------------------------------------------

    @staticmethod
    def marginal_likelihood(
        log_likelihood_fn: Callable[[np.ndarray], float],
        prior_logpdf: Callable[[np.ndarray], float],
        *,
        bounds: np.ndarray | None = None,
        map_params: np.ndarray | None = None,
        grid_points: int = 401,
        prior_description: str = "unspecified",
    ) -> ModelEvidence:
        """Integrate the likelihood against the prior: ``log ∫ p(D|θ) p(θ) dθ``.

        This is the quantity a Bayes factor is a ratio of. Supply either
        ``bounds`` (shape ``(n_params, 2)``; 1-2 parameters, nested trapezoid
        quadrature) or ``map_params`` (the MAP estimate; Gaussian/Laplace
        approximation around it). The returned :class:`ModelEvidence` records
        which approximation was used, because the number is only as trustworthy
        as the method.
        """
        if bounds is not None:
            b = np.asarray(bounds, dtype=float)
            if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] < 1:
                raise ValueError(f"bounds must have shape (n_params, 2), got {b.shape}")
            if bool(np.any(b[:, 1] <= b[:, 0])):
                raise ValueError(f"every bound must have upper > lower, got {b.tolist()}")
            n_params = int(b.shape[0])
            if n_params > 2:
                raise ValueError(
                    f"grid quadrature is implemented for 1-2 parameters, got {n_params}; "
                    "pass map_params for the Laplace approximation"
                )
            n = max(int(grid_points), 3)
            axes = [np.linspace(b[i, 0], b[i, 1], n) for i in range(n_params)]
            if n_params == 1:
                vals = np.array([_log_integrand(log_likelihood_fn, prior_logpdf, [x]) for x in axes[0]])
                log_ev = _log_integrate(vals, axes[0])
            else:
                grid = np.empty((n, n))
                for i, x in enumerate(axes[0]):
                    for j, y in enumerate(axes[1]):
                        grid[i, j] = _log_integrand(log_likelihood_fn, prior_logpdf, [x, y])
                inner = np.array([_log_integrate(grid[i], axes[1]) for i in range(n)])
                log_ev = _log_integrate(inner, axes[0])
            return ModelEvidence(log_ev, "quadrature", n_params, prior_description)

        if map_params is None:
            raise ValueError(
                "marginal_likelihood needs bounds (quadrature) or map_params (Laplace); "
                "a likelihood without a prior and an integral is not a marginal likelihood"
            )

        theta = np.asarray(map_params, dtype=float).ravel()
        if theta.size < 1:
            raise ValueError("map_params must be non-empty")

        def log_posterior(p: np.ndarray) -> float:
            return _log_integrand(log_likelihood_fn, prior_logpdf, p)

        f_here = log_posterior(theta)
        step = 1e-4 * np.maximum(np.abs(theta), 1.0)
        for i in range(theta.size):
            offset = np.zeros(theta.size)
            offset[i] = step[i]
            if log_posterior(theta + offset) > f_here or log_posterior(theta - offset) > f_here:
                raise ValueError(
                    "Laplace approximation needs map_params to be a mode of the log-posterior, but "
                    f"moving parameter {i} by {step[i]:g} from {theta.tolist()} increases it; "
                    "optimise first, or pass bounds for quadrature"
                )

        hessian = _finite_difference_hessian(log_posterior, theta)
        neg_det = float(np.linalg.det(-hessian))
        if not np.isfinite(neg_det) or neg_det <= 0.0:
            raise ValueError(
                "Laplace approximation needs a negative-definite log-posterior Hessian "
                f"at the MAP; det(-H)={neg_det:g} for {theta.tolist()} (is it a mode?)"
            )
        log_ev = log_posterior(theta) + 0.5 * theta.size * np.log(2.0 * np.pi) - 0.5 * np.log(neg_det)
        return ModelEvidence(float(log_ev), "laplace", int(theta.size), prior_description)

    @staticmethod
    def evidence_from_bic(
        log_likelihood_max: float,
        num_params: int,
        num_samples: int,
        prior_description: str = "flat over the parameter support (asymptotic)",
    ) -> ModelEvidence:
        """Asymptotic evidence proxy: ``log p(D|M) ≈ log L_max - (k/2) ln n``.

        Labelled ``"bic"`` so a verdict resting on it can say so. Asymptotic in
        ``n``; not valid for small samples.
        """
        if num_params < 0:
            raise ValueError(f"num_params must be >= 0, got {num_params}")
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        lam = 0.5 * float(num_params) * float(np.log(float(num_samples)))
        return ModelEvidence(float(log_likelihood_max) - lam, "bic", int(num_params), prior_description)

    @staticmethod
    def kass_raftery_verdict(two_ln_bf: float, method_note: str = "") -> str:
        """Kass-Raftery grade for ``2 ln BF``, with the evidence method appended.

        The scale applies to a Bayes factor. It is not applied to a
        summed-likelihood ratio (see :meth:`compute_likelihood_ratio`).
        """
        if two_ln_bf > 10.0:
            verdict = "Very Strong / Decisive evidence for Model 1"
        elif two_ln_bf > 6.0:
            verdict = "Strong evidence for Model 1"
        elif two_ln_bf > 2.0:
            verdict = "Positive evidence for Model 1"
        elif two_ln_bf >= -2.0:
            verdict = "Inconclusive / Weak evidence"
        elif two_ln_bf >= -6.0:
            verdict = "Positive evidence for Model 2"
        elif two_ln_bf >= -10.0:
            verdict = "Strong evidence for Model 2"
        else:
            verdict = "Very Strong / Decisive evidence for Model 2"
        return f"{verdict} ({method_note})" if method_note else verdict

    @staticmethod
    def compute_bayes_factor(
        evidence_1: ModelEvidence,
        evidence_2: ModelEvidence,
    ) -> tuple[float, str, float]:
        """Bayes factor B_12 = p(D|M1) / p(D|M2) from two marginal likelihoods.

        Returns ``(BF, Kass-Raftery verdict naming the method, log10 BF)``.
        Non-finite evidence raises: an unmeasurable quantity has no ratio.
        """
        for label, ev in (("model 1", evidence_1), ("model 2", evidence_2)):
            if not np.isfinite(ev.log_evidence):
                raise ValueError(
                    f"{label} log-evidence is not finite ({ev.log_evidence}); "
                    "a Bayes factor cannot be formed from an unmeasured marginal likelihood"
                )
        diff = float(evidence_1.log_evidence - evidence_2.log_evidence)
        bf = float(np.exp(np.clip(diff, -700.0, 700.0)))
        log_bf = float(diff / np.log(10.0))
        note = f"{evidence_1.method} vs {evidence_2.method}"
        return bf, BayesianRefinementEngine.kass_raftery_verdict(2.0 * diff, note), log_bf

    @staticmethod
    def compute_bayes_factor_from_log_evidence(
        log_evidence_1: float,
        log_evidence_2: float,
        method_1: str = "quadrature",
        method_2: str = "quadrature",
    ) -> tuple[float, str, float]:
        """Bayes factor from two *log marginal likelihoods*, underflow-safe.

        Both inputs must be ``log ∫ p(D|θ) p(θ) dθ``. Passing a maximised log
        likelihood here is the error this signature exists to prevent; use
        :meth:`compute_likelihood_ratio` if that is what you have.
        """
        return BayesianRefinementEngine.compute_bayes_factor(
            ModelEvidence(float(log_evidence_1), method_1, 0),
            ModelEvidence(float(log_evidence_2), method_2, 0),
        )

    @staticmethod
    def compute_likelihood_ratio(
        model_1_likelihoods: np.ndarray,
        model_2_likelihoods: np.ndarray,
        regularizer: float = 1e-15,
    ) -> tuple[float, str, float]:
        """Ratio of *summed pointwise likelihoods* -- **not** a Bayes factor.

        Kept because it is what a caller handing over likelihood arrays has
        actually computed, and it is a useful description of two evaluated
        models. It has no prior and no integral over parameters, so it cannot be
        read on the Kass-Raftery scale; the description reports direction only.
        Returns ``(ratio, description, log10 ratio)``.
        """
        ev1 = float(np.sum(model_1_likelihoods)) + regularizer
        ev2 = float(np.sum(model_2_likelihoods)) + regularizer
        if ev1 < 0.0 or ev2 < 0.0:
            raise ValueError(
                f"summed likelihoods must be non-negative, got {ev1:g} and {ev2:g}; "
                "supply likelihoods, not log-likelihoods"
            )
        ratio = ev1 / ev2 if ev2 > 0.0 else float("inf")
        log_ratio = float(np.log10(ratio)) if np.isfinite(ratio) else float("inf")
        caveat = "NOT a Bayes factor: no prior, no parameter integral"
        if ratio > 1.0 + 1e-9:
            desc = f"summed likelihood ratio favours Model 1 ({caveat})"
        elif ratio < 1.0 - 1e-9:
            desc = f"summed likelihood ratio favours Model 2 ({caveat})"
        else:
            desc = f"summed likelihood ratios are indistinguishable ({caveat})"
        return ratio, desc, log_ratio

    @staticmethod
    def compute_bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
        """Bayesian Information Criterion: BIC = k*ln(n) - 2*ln(L). Lower is better."""
        return float(num_params * np.log(max(num_samples, 1)) - 2.0 * log_likelihood)

    @staticmethod
    def compute_aic(log_likelihood: float, num_params: int) -> float:
        """Akaike Information Criterion: AIC = 2k - 2*ln(L). Lower is better."""
        return float(2.0 * num_params - 2.0 * log_likelihood)

    @staticmethod
    def compute_aicc(log_likelihood: float, num_params: int, num_samples: int) -> float:
        """Corrected Akaike Information Criterion for small sample sizes."""
        aic = BayesianRefinementEngine.compute_aic(log_likelihood, num_params)
        denom = num_samples - num_params - 1
        if denom > 0:
            return float(aic + (2.0 * num_params * (num_params + 1)) / denom)
        return float(aic)

    @staticmethod
    def score_candidate(
        expression: sp.Expr,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_params: int | None = None,
    ) -> dict:
        """Score one fitted candidate with R² / log-likelihood / BIC / AIC.

        ``num_params`` defaults to the number of free symbols in
        ``expression`` (a conservative complexity proxy). Returns a dict
        with ``r_squared``, ``log_likelihood``, ``bic``, ``aic``,
        ``complexity`` and ``degenerate``/``degenerate_reason``.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if num_params is None:
            try:
                num_params = len(set(expression.free_symbols))
            except AttributeError:
                num_params = 1
        num_params = max(int(num_params), 1)
        n = int(y_true.size)
        if y_true.shape == y_pred.shape:
            r2 = BayesianRefinementEngine.compute_r_squared(y_true, y_pred)
        else:
            r2 = float("nan")
        degen, reason = BayesianRefinementEngine.is_degenerate(y_true, y_pred, expression)
        try:
            log_lik, _ = BayesianRefinementEngine.gaussian_log_likelihood(y_true, y_pred)
        except (TypeError, ValueError):
            log_lik = float("-inf")
        try:
            complexity = len(list(sp.preorder_traversal(expression)))
        except (AttributeError, TypeError, ValueError):
            complexity = num_params
        if np.isfinite(log_lik):
            bic = BayesianRefinementEngine.compute_bic(log_lik, num_params, n)
            aic = BayesianRefinementEngine.compute_aic(log_lik, num_params)
        else:
            bic, aic = float("inf"), float("inf")
        return {
            "expression": expression,
            "r_squared": float(r2),
            "log_likelihood": float(log_lik),
            "bic": float(bic),
            "aic": float(aic),
            "complexity": int(complexity),
            "num_params": int(num_params),
            "degenerate": bool(degen),
            "degenerate_reason": reason,
        }
