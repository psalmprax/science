"""APEDP Discovery: Multi-Variable Symbolic Regression Engine & Bayesian Model Selection."""
from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem
from apedp.safety import quiet_float_warnings

logger = logging.getLogger(__name__)

# Rational coefficients worth snapping fitted values to.
STANDARD_RATIONALS: list[tuple[float, sp.Expr]] = [
    (0.5, sp.Rational(1, 2)),
    (0.25, sp.Rational(1, 4)),
    (0.75, sp.Rational(3, 4)),
    (1.0 / 3.0, sp.Rational(1, 3)),
    (2.0 / 3.0, sp.Rational(2, 3)),
    (1.0, sp.Integer(1)),
    (2.0, sp.Integer(2)),
    (3.0, sp.Integer(3)),
    (4.0, sp.Integer(4)),
    (-1.0, sp.Integer(-1)),
    (-0.5, sp.Rational(-1, 2)),
]


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
            # ``np.std``/``np.var`` warn ("Degrees of freedom <= 0") and return
            # nan on empty input. Name the condition rather than computing
            # statistics over nothing, which is how this surfaced as a numeric
            # warning in the first place.
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

        Raises ``ValueError`` when neither is usable rather than guessing: an
        un-integrated likelihood is a different statistic, not this one.
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
                # Not a SymPy expression (a raw string or array was passed): fall
                # back to the minimal complexity proxy rather than failing.
                num_params = 1
        num_params = max(int(num_params), 1)
        n = int(y_true.size)
        # A shape disagreement makes every statistic below undefined, but
        # ``compute_r_squared`` raises only on *incompatible* shapes and silently
        # broadcasts compatible ones (e.g. (1,) against (n,)), returning a
        # meaningless figure. Check explicitly and report the pair as unscoreable
        # instead of letting it reach the arithmetic. ``is_degenerate`` already
        # names the mismatch and refuses such a candidate.
        if y_true.shape == y_pred.shape:
            r2 = BayesianRefinementEngine.compute_r_squared(y_true, y_pred)
        else:
            r2 = float("nan")
        degen, reason = BayesianRefinementEngine.is_degenerate(y_true, y_pred, expression)
        try:
            log_lik, _ = BayesianRefinementEngine.gaussian_log_likelihood(y_true, y_pred)
        except (TypeError, ValueError):
            # The likelihood function raises ValueError for shape mismatch and
            # empty data; np.asarray raises TypeError/ValueError for input that
            # cannot be cast to float. Both mean "no likelihood available".
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


class MultiVariableSRE:
    """Dimensional-invariant guided evolutionary symbolic regression engine.

    Division of labor: ``MultiVariableSRE`` (this class) is the primary
    dimensionally-gated evolutionary engine used by the pipeline and API.
    ``apedp.symbolic_regression.SymbolicRegressor`` is a legacy standalone
    tournament/crossover engine kept for backwards compatibility and small
    dimensionless demos; prefer this class for all dimensional work.

    Optional extensions (on by default; pass False to restore the legacy
    minimal operator set):

    - ``allow_exp``: include ``exp(x)`` in generated candidates. The argument
      is required to be dimensionless (same rule the dimensional gatekeeper
      enforces), so generation retries with plain ``Mul``/``Pow`` when no
      dimensionless inner expression is found.
    - ``allow_trig``: include ``sin(x)``/``cos(x)`` with dimensionless
      arguments (wave phases, oscillations).
    - ``allow_log``: include ``log(x)`` with dimensionless arguments
      (entropy, Arrhenius linearizations).
    - ``allow_sqrt``: include explicit ``sqrt(x)`` nodes (in addition to
      ``Pow(x, 1/2)`` fractional exponents, which are always enabled).
      On by default; dimensionally halves the argument's dimensions.
    - ``extra_const_values``: mapping of ``name -> (value, unit_str)`` for
      physical constants absent from the global registry (e.g. Faraday's
      constant). Each entry is registered on the engine, added to the
      platform's constant values, and appended to the building blocks.
    - ``template_priors``: list of ``str``/``Expr`` physics-informed seed
      expressions (e.g. ``"A_freq*exp(-E_act/(R_gas*T_temp))"``). Entries
      that parse and pass the dimensional gatekeeper are prepended to the
      initial population; invalid entries are skipped with a warning.
    - ``inner_fit``: hill-climb the values of ephemeral random constants
      (``C0``, ``C1``, ...) that appear inside nonlinearities such as
      ``exp``. The legacy scheme fixes them at 1.0 and only fits outer
      linear coefficients, which cannot learn inner scales like the 19.5
      in ``exp(19.5*x)``. Uses only the standard library + numpy.
    - ``guided``: seed the population with dimensionally valid-by-construction
      candidates (unit-vector monomials of the target dimensionality, same-dim
      sums, dimensionless exp factors). Unguided random trees hit a specific
      cross-unit target roughly 0.03% of the time per candidate, so discovery
      outcomes were seed luck; guided seeding proposes structures such as
      ``m*v**2``, ``I*R`` or ``G*m1*m2/r**2`` directly.
    """

    def __init__(
        self,
        assurance_system: ClosedLoopAssuranceSystem,
        independent_vars: list[str] | None = None,
        target_unit_str: str = "dimensionless",
        extra_constants: list[str] | None = None,
        variables: list[str | sp.Symbol] | None = None,
        num_constants: int = 3,
        allow_exp: bool = True,
        allow_trig: bool = True,
        allow_log: bool = True,
        allow_sqrt: bool = True,
        extra_const_values: dict[str, tuple[float, str]] | None = None,
        template_priors: list[str | sp.Expr] | None = None,
        inner_fit: bool = True,
        inner_steps: int = 40,
        guided: bool = True,
    ):
        self.system = assurance_system
        vars_input = independent_vars if independent_vars is not None else (variables or [])
        self.var_names = [str(v) for v in vars_input]
        self.vars = [sp.Symbol(v) for v in self.var_names]
        self.target_unit_str = target_unit_str
        self.target_unit = self.system.engine.registry(target_unit_str)

        # Constant symbols available for discovery
        default_const_names = ['G_grav', 'c_light', 'sigma_sb', 'h_bar', 'k_B', 'R_gas', 'g_earth']
        const_names = extra_constants or default_const_names
        self.constants = [sp.Symbol(c) for c in const_names if c in self.system.constants_values]
        # Ephemeral random constants: dimensionless placeholders that become
        # constant features during coefficient fitting (so `C0 + x**2` can
        # discover `3 + 2*x**2`). Avoid naming variables `C0`, `C1`, ...
        self.erc_symbols = [sp.Symbol(f"C{i}") for i in range(max(0, num_constants))]
        for sym in self.erc_symbols:
            self.system.engine.register_variable(str(sym), "dimensionless")
        self.building_blocks = self.vars + self.constants + self.erc_symbols
        self.operators = [sp.Add, sp.Mul]
        self.allow_exp = bool(allow_exp)
        self.allow_trig = bool(allow_trig)
        self.allow_log = bool(allow_log)
        self.allow_sqrt = bool(allow_sqrt)
        self.inner_fit = bool(inner_fit)
        self.inner_steps = max(0, int(inner_steps))
        self.guided = bool(guided)
        for name, spec in dict(extra_const_values or {}).items():
            try:
                value, unit_str = spec
            except (TypeError, ValueError) as e:
                raise ValueError(f"extra_const_values[{name!r}] must be a (value, unit_str) pair.") from e
            try:
                self.system.engine.register_variable(name, unit_str)
            except (ValueError, TypeError, AttributeError) as e:
                raise ValueError(f"Cannot register extra constant '{name}' with unit '{unit_str}'.") from e
            sym = sp.Symbol(name)
            self.system.constants_values[name] = float(value)
            if sym not in self.building_blocks:
                self.building_blocks.append(sym)
        self.template_exprs: list[sp.Expr] = []
        for prior in template_priors or []:
            try:
                if isinstance(prior, str):
                    from apedp.safety import safe_parse_expr

                    expr = safe_parse_expr(prior)
                else:
                    expr = sp.sympify(prior)
                unit = self.system.engine.check_expression_consistency(expr)
                if unit.dimensionality == self.target_unit.dimensionality:
                    self.template_exprs.append(expr)
                else:
                    logger.warning("Template prior %r is dimensionally inconsistent, skipping.", prior)
            except (ValueError, TypeError) as e:
                logger.warning("Template prior %r could not be parsed, skipping: %s", prior, e)
        # Populated by run_discovery; empty until a search has been run.
        self.stagnation_report: dict[str, Any] = {}

    def _generate_candidate_base(self, depth: int = 0, max_depth: int = 3) -> sp.Expr:
        # Seeded scientific search (seed=42 is a tested feature); secrets-style
        # non-determinism would break reproducibility, so stdlib random is correct here.
        if depth >= max_depth or random.random() < 0.3:  # nosec B311
            return random.choice(self.building_blocks)  # nosec B311

        choices: list = list(self.operators + [sp.Pow])
        if self.allow_sqrt:
            choices.append(sp.sqrt)
        op = random.choice(choices)  # nosec B311
        if op == sp.Pow:
            base = self._generate_candidate(depth + 1, max_depth)
            # Half- and third-integer exponents so laws with fractional powers
            # (Hall-Petch b/sqrt(d), cube-root scalings) are representable.
            exponent = random.choice(  # nosec B311
                [-2, -1, 1, 2, 3, 4, sp.Rational(1, 2), sp.Rational(-1, 2),
                 sp.Rational(3, 2), sp.Rational(-3, 2), sp.Rational(1, 3),
                 sp.Rational(-1, 3), sp.Rational(2, 3), sp.Rational(-2, 3)]
            )
            return sp.Pow(base, exponent)
        if op == sp.sqrt:
            return sp.sqrt(self._generate_candidate(depth + 1, max_depth))
        else:
            left = self._generate_candidate(depth + 1, max_depth)
            right = self._generate_candidate(depth + 1, max_depth)
            return op(left, right)

    def _generate_candidate(self, depth: int = 0, max_depth: int = 3) -> sp.Expr:
        # Wrapper preserving the exact legacy RNG stream when all nonlinear
        # flags are off. Any enabled nonlinear changes the stream by design.
        if not (self.allow_exp or self.allow_trig or self.allow_log):
            return self._generate_candidate_base(depth, max_depth)
        if depth >= max_depth or random.random() < 0.25:  # nosec B311
            return random.choice(self.building_blocks)  # nosec B311
        nonlinear_picks: list[str] = []
        if self.allow_exp:
            nonlinear_picks.append("exp")
        if self.allow_trig:
            nonlinear_picks.extend(["sin", "cos"])
        if self.allow_log:
            nonlinear_picks.append("log")
        if nonlinear_picks and random.random() < 0.25:  # nosec B311
            inner = self._random_dimensionless_inner(depth, max_depth)
            if inner is not None:
                kind = random.choice(nonlinear_picks)  # nosec B311
                if kind == "exp":
                    return sp.exp(inner)
                if kind == "sin":
                    return sp.sin(inner)
                if kind == "cos":
                    return sp.cos(inner)
                if kind == "log":
                    return sp.log(inner)
        return self._generate_candidate_base(depth, max_depth)

    def _block_dim_dict(self) -> dict[sp.Symbol, dict[str, float]]:
        """Dimensionality of every building block, skipping unregistered ones.

        Cached per block; failures are retried on later calls so variables
        registered after construction are picked up.
        """
        cache: dict[sp.Symbol, dict[str, float]] = self.__dict__.setdefault("_block_dim_cache", {})
        for block in self.building_blocks:
            if block not in cache:
                try:
                    cache[block] = _dim_dict(self.system.engine.check_expression_consistency(block))
                except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                    logger.debug("Skipping dimension-less block %s (%s)", block, e)
        return dict(cache)

    def _monomial_for_dim(
        self,
        target: dict[str, float],
        block_dims: dict[sp.Symbol, dict[str, float]],
        tries: int = 100,
    ) -> sp.Expr | None:
        """Random unit-vector monomial with exactly ``target`` dimensionality.

        Searches exponent vectors over non-ERC blocks (``a**2 * b**-1``);
        every result is dimensionally valid by construction, divisions
        included. Returns None when the budget is exhausted.
        """
        erc_set = set(self.erc_symbols)
        items = [(b, d) for b, d in block_dims.items() if b not in erc_set]
        if not items:
            return None
        # Integer exponents are searched first: they cover the bulk of
        # dimensional targets, and widening the alphabet dilutes this
        # exact-match search (measured: 13.3% -> 6.1% hit rate on a joule
        # target). Halves keep Hall-Petch reachable; thirds keep cube-root
        # scalings reachable. All arithmetic here is exact Fraction so thirds
        # compare exactly (float 1/3 would not).
        from fractions import Fraction

        def _f(d: dict[str, float]) -> dict[str, Fraction]:
            # limit_denominator maps float 1/3 (0.333...) to exact 1/3 so
            # thirds compare exactly; ints and halves are unaffected.
            return {k: Fraction(v).limit_denominator(12) for k, v in d.items()}

        def _add(a: dict[str, Fraction], b: dict[str, Fraction]) -> dict[str, Fraction]:
            out = dict(a)
            for k, v in b.items():
                out[k] = out.get(k, Fraction(0)) + v
                if out[k] == 0:
                    del out[k]
            return out

        target_f = _f(target)
        items_f = [(b, _f(d)) for b, d in items]
        integer_exps: list[Fraction] = [Fraction(-2), Fraction(-1), Fraction(1), Fraction(1), Fraction(2)]
        extended_exps: list[Fraction] = integer_exps + [
            Fraction(1, 2), Fraction(-1, 2),
            Fraction(1, 3), Fraction(-1, 3), Fraction(2, 3), Fraction(-2, 3),
        ]
        integer_budget = int(tries * 0.7)
        for attempt in range(tries):
            exps_pool = integer_exps if attempt < integer_budget else extended_exps
            picks = [random.choice(items_f) for _ in range(random.randint(1, 4))]  # nosec B311
            exps = [random.choice(exps_pool) for _ in picks]  # nosec B311
            acc: dict[str, Fraction] = {}
            for (_, dim), power in zip(picks, exps):
                acc = _add(acc, {k: v * power for k, v in dim.items() if v * power != 0})
            if acc == target_f:
                expr: sp.Expr = sp.Integer(1)
                for (block, _), power in zip(picks, exps):
                    expr = expr * sp.Pow(block, sp.Rational(power.numerator, power.denominator))
                return expr
        return None

    def _generate_guided_candidate(self, depth: int = 0, max_depth: int = 3) -> sp.Expr:
        """Dimensionally valid-by-construction candidate for the target unit.

        Strategies: exact target monomial (covers ``m*v**2``, ``I*R``,
        ``G*m1*m2/r**2`` shapes including divisions), same-dimension sums
        (``M + C*M``, keeping additive-offset reachability), sums of two
        independent target monomials (``h*omega - phi``, ``P + rho*v**2``),
        and — with ``allow_exp``/``allow_trig``/``allow_log`` — dimensionless
        nonlinear factors alone or multiplying a target monomial
        (``A*exp(-E/(R*T))`` shape). Falls back to the legacy generator when
        monomial search misses.
        """
        target = _dim_dict(self.target_unit)
        block_dims = self._block_dim_dict()
        scale = random.choice(self.erc_symbols) if self.erc_symbols else sp.Integer(1)  # nosec B311
        roll = random.random()  # nosec B311
        if roll < 0.45:
            mono = self._monomial_for_dim(target, block_dims)
            if mono is not None:
                return mono
        elif roll < 0.65:
            mono = self._monomial_for_dim(target, block_dims)
            if mono is not None:
                return mono + scale * mono
        elif roll < 0.80:
            first = self._monomial_for_dim(target, block_dims, tries=60)
            second = self._monomial_for_dim(target, block_dims, tries=60)
            if first is not None and second is not None:
                return first + scale * second
        elif self.allow_exp or self.allow_trig or self.allow_log:
            dimless = self._monomial_for_dim({}, block_dims)
            if dimless is not None and dimless != 1:
                inner = dimless
                if not isinstance(inner, sp.Expr):
                    inner = sp.sympify(inner)
                mono = self._monomial_for_dim(target, block_dims, tries=30)
                kinds: list[str] = []
                if self.allow_exp:
                    kinds.append("exp")
                if self.allow_trig:
                    kinds.extend(["sin", "cos"])
                if self.allow_log:
                    kinds.append("log")
                kind = random.choice(kinds) if kinds else "exp"  # nosec B311
                factor = (
                    sp.exp(inner)
                    if kind == "exp"
                    else sp.sin(inner)
                    if kind == "sin"
                    else sp.cos(inner)
                    if kind == "cos"
                    else sp.log(inner)
                )
                if target == {} or mono is None:
                    return factor
                return mono * factor
        return self._generate_candidate(depth, max_depth)

    def _random_dimensionless_inner(self, depth: int, max_depth: int) -> sp.Expr | None:
        """Builds a dimensionless sub-expression for exp()/sin()/cos()/log().

        Uses the nonlinearity-free base generator so transcendentals never
        nest, and verifies the dimensional gatekeeper rule (transcendental
        arguments must be dimensionless) before accepting.
        """
        dimensionless = self.system.engine.registry.dimensionless.dimensionality
        for _ in range(8):
            inner = self._generate_candidate_base(depth + 1, max_depth)
            try:
                if self.system.engine.check_expression_consistency(inner).dimensionality == dimensionless:
                    return inner
            except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                logger.debug("Discarding dimensioned exp() argument (%s)", e)
                continue
        return None

    @staticmethod
    def _ercs_inside_exp(candidate: sp.Expr, erc_symbols: list[sp.Symbol]) -> list[sp.Symbol]:
        """Ephemeral constants appearing inside exp() arguments.

        These are the inner scales (e.g. the 19.5 in ``exp(19.5*x)``) that
        the outer linear least-squares fit cannot learn, since it only fits
        one coefficient per additive term with ERCs fixed at 1.0.
        """
        erc_set = set(erc_symbols)
        found: set[sp.Symbol] = set()
        for node in candidate.atoms(sp.exp):
            found.update(s for s in node.args[0].free_symbols if s in erc_set)
        return sorted(found, key=str)

    def _lambdify_cached(self, syms: list[sp.Symbol], expr: sp.Expr):
        """Memoized ``sp.lambdify``: elites are re-scored every generation."""
        cache: dict = self.__dict__.setdefault("_lambda_cache", {})
        if len(cache) > 2000:
            cache.clear()
        key = (str(expr), tuple(str(s) for s in syms))
        fn = cache.get(key)
        if fn is None:
            fn = sp.lambdify(syms, expr, modules=["numpy"])
            cache[key] = fn
        return fn

    def _linear_outer_fit(
        self,
        candidate: sp.Expr,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        erc_values: dict[str, float],
    ) -> tuple[sp.Expr, float]:
        """Linear least-squares fit of outer coefficients for fixed ERC values.

        With all values at 1.0 this is exactly the legacy calibration path.
        """
        try:
            var_names = sorted(x_data.keys())
            syms = [sp.Symbol(k) for k in var_names]
            # Substitute physical constants (and ephemeral constants at their
            # current values) into candidate before lambdify
            subs = {sp.Symbol(k): v for k, v in self.system.constants_values.items()}
            subs.update({s: float(erc_values.get(str(s), 1.0)) for s in self.erc_symbols})
            args = [x_data[k] for k in var_names]

            def evaluate(expr: sp.Expr) -> np.ndarray:
                f = self._lambdify_cached(syms, expr.subs(subs))
                # Fitting the outer coefficients evaluates candidate parts on the
                # data, and an exp/inner-scale term routinely overflows. The
                # non-finite design matrix below rejects the candidate outright,
                # so the warning is expected rather than a signal.
                with quiet_float_warnings():
                    y_pred = f(*args)
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.full_like(y_data, float(y_pred))
                result: np.ndarray = y_pred
                return result

            base_const, term_tuple = candidate.as_coeff_add()
            parts: list[sp.Expr] = ([sp.sympify(base_const)] if base_const != 0 else []) + list(term_tuple)
            if not parts:
                return candidate, float('inf')
            design = np.column_stack([evaluate(part) for part in parts])
            if not np.all(np.isfinite(design)):
                return candidate, float('inf')

            coeffs, _, _, _ = np.linalg.lstsq(design, y_data, rcond=None)
            scale = float(np.sqrt(np.mean(y_data ** 2))) or 1.0
            tol = 1e-9 * (1.0 + scale)
            # Largest coefficients first so at least one term always survives pruning
            order = sorted(range(len(parts)), key=lambda i: abs(float(coeffs[i])), reverse=True)
            fitted: list[tuple[sp.Expr, sp.Expr, int]] = []
            for i in order:
                if abs(float(coeffs[i])) < tol and fitted:
                    continue
                fitted.append((_snap_coefficient(float(coeffs[i])), parts[i], i))

            rebuilt = [part if coeff == 1 else sp.Mul(coeff, part) for coeff, part, _ in fitted]
            calibrated_expr = sp.Add(*rebuilt)
            # Bake ephemeral constants back to their fitted values so the
            # returned expression is self-contained. Exactly-1.0 values keep
            # the legacy Integer(1) substitution.
            bake = {
                s: (sp.Integer(1) if erc_values.get(str(s), 1.0) == 1.0
                    else sp.Float(float(f"{erc_values[str(s)]:.6g}")))
                for s in self.erc_symbols
            }
            calibrated_expr = calibrated_expr.subs(bake)
            prediction = sum(float(coeff) * design[:, i] for coeff, _, i in fitted)
            mse = float(np.mean((prediction - y_data) ** 2))
            return calibrated_expr, mse
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as e:
            logger.warning("Parameter optimization failed, returning inf: %s", e)
            return candidate, float('inf')

    def _refine_inner_ercs(
        self,
        candidate: sp.Expr,
        inner_syms: list[sp.Symbol],
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        erc_values: dict[str, float],
        best_mse: float,
    ) -> tuple[dict[str, float], float]:
        """Coordinate-wise log-uniform hill-climb of inner ERC values.

        Each proposal re-runs the linear outer fit, so outer coefficients
        stay optimal for every tried inner scale. Standard library + numpy
        only; early-stops after 10 non-improving proposals.
        """
        theta = dict(erc_values)
        stalled = 0
        for _ in range(self.inner_steps):
            target = random.choice(inner_syms)  # nosec B311
            old = theta[str(target)]
            if old == 0.0 or not np.isfinite(old):
                continue
            # Two-scale proposals: wide jumps for range-finding, narrow
            # steps for fine-tuning near the optimum.
            width = random.choice([0.15, 0.7])  # nosec B311
            trial = dict(theta)
            trial[str(target)] = old * 10 ** random.uniform(-width, width)  # nosec B311
            _, mse = self._linear_outer_fit(candidate, x_data, y_data, trial)
            if np.isfinite(mse) and mse < best_mse * (1.0 - 1e-9):
                theta, best_mse, stalled = trial, mse, 0
            else:
                stalled += 1
                if stalled >= 10:
                    break
        return theta, best_mse

    def optimize_parameters(
        self,
        candidate: sp.Expr,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray
    ) -> tuple[sp.Expr, float]:
        """
        Calibrates continuous coefficients against empirical data.

        Decomposes the candidate into additive terms (ephemeral constants such
        as ``C0`` evaluate to 1.0 and become constant features) and solves a
        linear least-squares problem for one coefficient per term. Fitted
        values snap to exact physical fractions when close, and negligible
        terms are pruned.
        When ``inner_fit`` is enabled, ERCs appearing inside ``exp()`` are
        additionally hill-climbed (outer fit re-run per proposal) so inner
        scales like the 19.5 in ``exp(19.5*x)`` can be learned.
        """
        erc_values = {str(s): 1.0 for s in self.erc_symbols}
        calibrated_expr, mse = self._linear_outer_fit(candidate, x_data, y_data, erc_values)
        if not self.inner_fit or self.inner_steps <= 0 or not np.isfinite(mse):
            return calibrated_expr, mse
        inner_syms = self._ercs_inside_exp(candidate, self.erc_symbols)
        if not inner_syms:
            return calibrated_expr, mse
        theta, refined_mse = self._refine_inner_ercs(
            candidate, inner_syms, x_data, y_data, erc_values, mse
        )
        if np.isfinite(refined_mse) and refined_mse < mse:
            calibrated_expr, _ = self._linear_outer_fit(candidate, x_data, y_data, theta)
            return calibrated_expr, refined_mse
        return calibrated_expr, mse

    def fitness(self, candidate: sp.Expr, x_data: dict[str, np.ndarray], y_data: np.ndarray) -> float:
        # 1. Strict Dimensional Gatekeeping
        try:
            unit = self.system.engine.check_expression_consistency(candidate)
            if unit.dimensionality != self.target_unit.dimensionality:
                return float('inf')
        except (TypeError, ValueError, NotImplementedError, ArithmeticError) as e:
            logger.warning("Dimensional gatekeeping failed, returning inf: %s", e)
            return float('inf')

        # 2. Continuous Parameter Optimization + Occam's Complexity Penalty
        try:
            calibrated_expr, mse = self.optimize_parameters(candidate, x_data, y_data)
            if np.isinf(mse) or np.isnan(mse):
                return float('inf')
            complexity = len(list(sp.preorder_traversal(calibrated_expr)))
            return mse + 1e-4 * complexity
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.warning("Parameter optimization failed, returning inf: %s", e)
            return float('inf')

    def mutate(self, parent: sp.Expr) -> sp.Expr:
        roll = random.random()  # nosec B311
        if roll < 0.5:
            # Dimension-preserving mutation: scale by dimensionless factors
            scale = random.choice([0.5, 2.0, sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2, 3), 1.5])  # nosec B311
            return parent * scale
        if roll < 0.7 and self.erc_symbols:
            # Constant-shift mutation: append an ephemeral constant term so
            # additive offsets (e.g. the `+3` in `2*x**2 + 3`) are reachable
            return parent + random.choice(self.erc_symbols)  # nosec B311
        return self._generate_candidate(depth=1, max_depth=2)

    def run_discovery(
        self,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        generations: int = 30,
        pop_size: int = 100,
        seed: int | None = 42
    ) -> tuple[sp.Expr | None, float, list[dict[str, Any]]]:
        """Runs the evolutionary search loop and returns (best_candidate, best_mse, history)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        population: list[sp.Expr] = []
        target_dim = self.target_unit.dimensionality
        attempts = 0

        # Seed population with physics-informed template priors first
        for prior_expr in self.template_exprs:
            if len(population) >= pop_size:
                break
            if all(prior_expr != existing for existing in population):
                population.append(prior_expr)

        # Seed population with dimensionally consistent candidates
        seed_fn = self._generate_guided_candidate if self.guided else self._generate_candidate
        while len(population) < pop_size and attempts < pop_size * 20:
            cand = seed_fn(depth=0, max_depth=3)
            try:
                if self.system.engine.check_expression_consistency(cand).dimensionality == target_dim:
                    population.append(cand)
            except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                logger.warning("Population seeding failed (%s)", e)
            attempts += 1

        # Fallback filler
        while len(population) < pop_size:
            population.append(self._generate_candidate(depth=0, max_depth=2))

        best_candidate: sp.Expr | None = None
        best_score = float('inf')
        history: list[dict[str, Any]] = []
        improvements = 0
        generations_without_improvement = 0

        for gen in range(generations):
            scored = []
            for p in population:
                score = self.fitness(p, x_data, y_data)
                scored.append((score, p))
            scored.sort(key=lambda item: item[0])

            gen_best_score, gen_best_expr = scored[0]
            improved = gen_best_score < best_score
            if improved:
                best_score = gen_best_score
                best_candidate, _ = self.optimize_parameters(gen_best_expr, x_data, y_data)
                improvements += 1
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            history.append({
                'generation': gen,
                'best_score': best_score,
                'best_expr': best_candidate,
                'improved': improved,
                'generations_without_improvement': generations_without_improvement,
            })

            if best_score < 1e-25:
                break

            # Selection: top 20% elites
            elites = [p for _, p in scored[:max(1, int(pop_size * 0.2))]]
            next_pop = list(elites)
            while len(next_pop) < pop_size:
                parent = random.choice(elites)  # nosec B311
                next_pop.append(self.mutate(parent))
            population = next_pop

        # Stagnation signal: the first generation normally improves on the
        # seeded population, so ``improvements <= 1`` means the search added
        # nothing beyond its own initialisation. That is only a problem when
        # the result is also a poor fit -- an immediate exact fit is
        # convergence, not stagnation -- so the verdict is gated on a
        # scale-free residual.
        y_arr = np.asarray(y_data, dtype=float)
        y_variance = float(np.var(y_arr))
        if y_variance > 0.0:
            normalized_score = float(best_score) / y_variance
        elif np.isfinite(best_score):
            normalized_score = 0.0  # constant target with a finite residual
        else:
            normalized_score = float('inf')
        added_nothing = improvements <= 1
        self.stagnation_report = {
            'generations_run': len(history),
            'generations_requested': generations,
            'improvements': improvements,
            'generations_without_improvement': generations_without_improvement,
            'added_nothing_over_initialization': added_nothing,
            'final_normalized_score': normalized_score,
            'stagnated': added_nothing and normalized_score > 1e-3,
            'converged_early': len(history) < generations,
        }

        return best_candidate, best_score, history


# Alias for backwards compatibility
EvolutionaryDiscoveryEngine = MultiVariableSRE
