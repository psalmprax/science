"""Bayes factor correctness, checked against an oracle outside the implementation.

The oracle is the Gaussian-Gaussian conjugate marginal likelihood, which has a
closed form derived from the Gaussian product identity:

    y_i ~ N(mu, sigma^2) iid,  mu ~ N(mu0, tau^2)
    log p(D) = -(n/2)ln(2 pi sigma^2) - S/(2 sigma^2)
               + (1/2)ln(2 pi sigma^2/n) - (1/2)ln(2 pi v) - (ybar - mu0)^2/(2v)

with v = sigma^2/n + tau^2 and S = sum_i (y_i - ybar)^2. Nothing in
``apedp.discovery`` appears in that expression, so a check against it can fail
for a real reason. A check of the form "sums of likelihoods over sums of
likelihoods" passes for any implementation consistent with itself -- which is
exactly what the previous version of this machinery was.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from apedp.discovery import BayesianRefinementEngine, ModelEvidence

MU0, TAU, SIGMA = 1.0, 0.5, 2.0
Y = np.array([1.3, 0.7, 2.1, 1.9, 0.4, 1.1])


def conjugate_log_evidence(
    y: np.ndarray, sigma: float, mu0: float, tau: float
) -> float:
    """Closed-form ``log p(D)`` for a Gaussian likelihood with a Gaussian prior."""
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    ybar = float(np.mean(y))
    s_total = float(np.sum((y - ybar) ** 2))
    v = sigma**2 / n + tau**2
    return float(
        -0.5 * n * np.log(2.0 * np.pi * sigma**2)
        - s_total / (2.0 * sigma**2)
        + 0.5 * np.log(2.0 * np.pi * sigma**2 / n)
        - 0.5 * np.log(2.0 * np.pi * v)
        - 0.5 * (ybar - mu0) ** 2 / v
    )


def gaussian_log_likelihood(y: np.ndarray, sigma: float) -> Callable[[np.ndarray], float]:
    def log_likelihood(mu: np.ndarray) -> float:
        return float(
            -0.5 * np.sum((y - float(mu[0])) ** 2) / sigma**2
            - 0.5 * y.size * np.log(2.0 * np.pi * sigma**2)
        )

    return log_likelihood


def gaussian_logpdf(centre: float, width: float) -> Callable[[np.ndarray], float]:
    def logpdf(mu: np.ndarray) -> float:
        return float(
            -0.5 * (float(mu[0]) - centre) ** 2 / width**2
            - 0.5 * np.log(2.0 * np.pi * width**2)
        )

    return logpdf


def conjugate_mode(y: np.ndarray, sigma: float, mu0: float, tau: float) -> float:
    """Posterior mode of the conjugate model (equal to its mean here)."""
    n = int(np.asarray(y).size)
    ybar = float(np.mean(y))
    return (n * ybar / sigma**2 + mu0 / tau**2) / (n / sigma**2 + 1 / tau**2)


@pytest.mark.parametrize(
    "centre, width, grid_points",
    [(1.0, 0.5, 401), (1.0, 0.5, 2001), (-1.5, 0.5, 2001), (1.0, 2.0, 2001)],
)
def test_quadrature_matches_the_closed_form(centre: float, width: float, grid_points: int):
    evidence = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(centre, width),
        bounds=np.array([[centre - 9 * width, centre + 9 * width]]),
        grid_points=grid_points,
    )
    expected = conjugate_log_evidence(Y, SIGMA, centre, width)
    assert evidence.method == "quadrature"
    assert abs(evidence.log_evidence - expected) < 1e-9


def test_quadrature_answer_is_resolution_independent():
    """A converged integral must not move when the grid is refined."""
    coarse = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(MU0, TAU),
        bounds=np.array([[MU0 - 9 * TAU, MU0 + 9 * TAU]]),
        grid_points=201,
    )
    fine = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(MU0, TAU),
        bounds=np.array([[MU0 - 9 * TAU, MU0 + 9 * TAU]]),
        grid_points=4001,
    )
    assert abs(coarse.log_evidence - fine.log_evidence) < 1e-6


def test_laplace_matches_the_closed_form_for_a_gaussian_posterior():
    """For this conjugate model the Laplace approximation is exact."""
    mode = conjugate_mode(Y, SIGMA, MU0, TAU)
    evidence = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(MU0, TAU),
        map_params=np.array([mode]),
    )
    expected = conjugate_log_evidence(Y, SIGMA, MU0, TAU)
    assert evidence.method == "laplace"
    assert abs(evidence.log_evidence - expected) < 1e-6


def test_two_dimensional_quadrature_matches_the_separable_closed_form():
    """Two independent parameters: the integral factorizes, so the oracle does too."""
    m1, s1, m2, s2 = 0.5, 1.0, -1.0, 0.7
    prior_width = 2.0

    def log_likelihood(p: np.ndarray) -> float:
        return float(
            -0.5 * ((float(p[0]) - m1) / s1) ** 2
            - np.log(s1)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * ((float(p[1]) - m2) / s2) ** 2
            - np.log(s2)
            - 0.5 * np.log(2.0 * np.pi)
        )

    def prior_logpdf(p: np.ndarray) -> float:
        return float(
            -0.5 * (float(p[0]) / prior_width) ** 2
            - np.log(prior_width)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * (float(p[1]) / prior_width) ** 2
            - np.log(prior_width)
            - 0.5 * np.log(2.0 * np.pi)
        )

    evidence = BayesianRefinementEngine.marginal_likelihood(
        log_likelihood, prior_logpdf, bounds=np.array([[-8.0, 8.0], [-8.0, 8.0]]), grid_points=801
    )

    def one_dimensional(mean: float, sd: float) -> float:
        combined = sd**2 + prior_width**2
        return float(-0.5 * np.log(2.0 * np.pi * combined) - 0.5 * mean**2 / combined)

    expected = one_dimensional(m1, s1) + one_dimensional(m2, s2)
    assert abs(evidence.log_evidence - expected) < 1e-9
    assert evidence.n_params == 2


def test_bayes_factor_is_the_ratio_of_marginal_likelihoods():
    matched = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(MU0, TAU),
        bounds=np.array([[MU0 - 9 * TAU, MU0 + 9 * TAU]]),
        grid_points=2001,
    )
    mismatch_centre = -1.5
    mismatch = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(mismatch_centre, TAU),
        bounds=np.array([[mismatch_centre - 9 * TAU, mismatch_centre + 9 * TAU]]),
        grid_points=2001,
    )
    expected = float(
        np.exp(
            conjugate_log_evidence(Y, SIGMA, MU0, TAU)
            - conjugate_log_evidence(Y, SIGMA, mismatch_centre, TAU)
        )
    )
    bf, verdict, log_bf = BayesianRefinementEngine.compute_bayes_factor(matched, mismatch)
    assert bf == pytest.approx(expected, rel=1e-6)
    assert log_bf == pytest.approx(np.log10(expected), rel=1e-9)
    assert "Model 1" in verdict
    # The verdict must not claim evidential strength without naming its method.
    assert "quadrature" in verdict


def test_prior_width_drives_the_bayes_factor_where_a_likelihood_ratio_cannot():
    """Two models with identical likelihoods but different priors must be separated.

    This is the property that distinguishes a Bayes factor from a ratio of
    summed likelihoods: the likelihood ratio at a shared parameter point is 1 by
    construction, while the marginal likelihoods differ by the prior volume each
    model spreads over (the Occam penalty).
    """
    narrow = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(MU0, TAU),
        bounds=np.array([[MU0 - 9 * TAU, MU0 + 9 * TAU]]),
        grid_points=2001,
    )
    wide_width = 5.0
    wide = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(Y, SIGMA),
        gaussian_logpdf(MU0, wide_width),
        bounds=np.array([[MU0 - 9 * wide_width, MU0 + 9 * wide_width]]),
        grid_points=4001,
    )
    expected = float(
        np.exp(
            conjugate_log_evidence(Y, SIGMA, MU0, TAU)
            - conjugate_log_evidence(Y, SIGMA, MU0, wide_width)
        )
    )
    bf, _, _ = BayesianRefinementEngine.compute_bayes_factor(narrow, wide)
    assert bf == pytest.approx(expected, rel=1e-5)
    # A wider prior spreads the same likelihood mass thinner: Model 1 wins.
    assert bf > 1.0

    # The statistic that cannot see the prior reports no difference at all.
    mode = conjugate_mode(Y, SIGMA, MU0, TAU)
    shared_point = np.array([float(np.exp(gaussian_log_likelihood(Y, SIGMA)(np.array([mode]))))])
    ratio, description, log_ratio = BayesianRefinementEngine.compute_likelihood_ratio(
        shared_point, shared_point
    )
    assert ratio == pytest.approx(1.0)
    assert log_ratio == pytest.approx(0.0)
    assert "NOT a Bayes factor" in description
    assert "no prior" in description


def test_evidence_from_bic_is_penalised_and_labelled():
    evidence = BayesianRefinementEngine.evidence_from_bic(-30.0, num_params=2, num_samples=100)
    assert evidence.method == "bic"
    assert evidence.n_params == 2
    assert evidence.log_evidence == pytest.approx(-30.0 - 0.5 * 2 * np.log(100.0))
    _, verdict, _ = BayesianRefinementEngine.compute_bayes_factor(
        evidence, BayesianRefinementEngine.evidence_from_bic(-40.0, 2, 100)
    )
    # An asymptotic proxy must announce itself in the verdict.
    assert "bic" in verdict


def test_matching_prior_beats_mismatched_prior_for_data_from_that_prior():
    rng = np.random.default_rng(7)
    y = rng.normal(MU0, SIGMA, 12)
    matched = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(y, SIGMA),
        gaussian_logpdf(MU0, TAU),
        bounds=np.array([[MU0 - 9 * TAU, MU0 + 9 * TAU]]),
        grid_points=2001,
    )
    far = BayesianRefinementEngine.marginal_likelihood(
        gaussian_log_likelihood(y, SIGMA),
        gaussian_logpdf(MU0 + 6 * TAU, TAU),
        bounds=np.array([[MU0 + 6 * TAU - 9 * TAU, MU0 + 6 * TAU + 9 * TAU]]),
        grid_points=2001,
    )
    bf, _, _ = BayesianRefinementEngine.compute_bayes_factor(matched, far)
    assert bf > 10.0


@pytest.mark.parametrize(
    "call, needle",
    [
        (
            lambda: BayesianRefinementEngine.marginal_likelihood(
                lambda p: 0.0, lambda p: 0.0, bounds=np.array([[0.0, 1.0]] * 3)
            ),
            "1-2 parameters",
        ),
        (
            lambda: BayesianRefinementEngine.marginal_likelihood(lambda p: 0.0, lambda p: 0.0),
            "marginal likelihood",
        ),
        (
            lambda: BayesianRefinementEngine.marginal_likelihood(
                lambda p: 0.0, lambda p: 0.0, bounds=np.array([[1.0, 0.0]])
            ),
            "upper > lower",
        ),
        (
            lambda: BayesianRefinementEngine.compute_likelihood_ratio(
                np.array([-1.0]), np.array([0.5])
            ),
            "non-negative",
        ),
        (
            lambda: BayesianRefinementEngine.evidence_from_bic(-1.0, 1, 0),
            "num_samples",
        ),
        (
            lambda: BayesianRefinementEngine.compute_bayes_factor(
                ModelEvidence(float("nan"), "quadrature", 1), ModelEvidence(0.0, "quadrature", 1)
            ),
            "not finite",
        ),
    ],
)
def test_unmeasurable_evidence_raises(call, needle):
    with pytest.raises(ValueError, match=needle):
        call()


def test_laplace_refuses_a_point_that_is_not_a_mode():
    """A Laplace approximation about a non-stationary point is meaningless."""
    with pytest.raises(ValueError, match="mode"):
        BayesianRefinementEngine.marginal_likelihood(
            lambda p: float(p[0]), gaussian_logpdf(0.0, 1.0), map_params=np.array([0.0])
        )


def test_laplace_refuses_a_flat_log_posterior():
    """A zero-curvature point lands exactly on det(-H) = 0: no Gaussian there."""
    with pytest.raises(ValueError, match="negative-definite"):
        BayesianRefinementEngine.marginal_likelihood(
            lambda p: 0.5 * float(p[0]) ** 2,
            lambda p: -0.5 * float(p[0]) ** 2,
            map_params=np.array([0.0]),
        )


def test_laplace_refuses_a_saddle_point():
    """A saddle is negative-curvature along each axis but indefinite overall.

    ``f = -0.5(a^2 + b^2) + 2ab`` decreases in every single-axis step from the
    origin, so a per-axis maximum test accepts it; only the sign of the Hessian
    determinant rejects it. Testing the sign (rather than its magnitude) is the
    property under test here.
    """

    def saddle(p: np.ndarray) -> float:
        a, b = float(p[0]), float(p[1])
        return -0.5 * (a**2 + b**2) + 2.0 * a * b

    with pytest.raises(ValueError, match="negative-definite"):
        BayesianRefinementEngine.marginal_likelihood(
            saddle, lambda p: 0.0, map_params=np.array([0.0, 0.0])
        )


def test_scenario_12_compares_models_with_a_real_bayes_factor():
    from apedp.scenarios import APEDPScenarioSuite

    result = APEDPScenarioSuite().scenario_12_bayesian_model_comparison()
    assert result["bayes_factor"] > 1.0
    assert result["method"] == "quadrature vs quadrature"
    assert result["log_bf"] > 0.0
    # Both evidences are published, so the ratio can be rechecked by hand.
    assert result["log_evidence_model_1"] > result["log_evidence_model_2"]
