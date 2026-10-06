"""Neural surrogate modeling, permutation feature relevance, and surrogate distillation for APEDP.

Extracts neural discovery components, eliminating scattered environment-variable lookups
by supporting explicit engine injection ('sklearn' or 'jax') via centralized NumericalPolicy.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import sympy as sp

from apedp.discovery_types import (
    NeuralRelevanceResult,
    SurrogateDistillationResult,
)
from apedp.numerical_policy import (
    DEFAULT_POLICY,
    DiscoveryNumericalPolicy,
    get_neural_discovery_backend,
)

logger = logging.getLogger("apedp.neural_surrogate")

_PROBE_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    ArithmeticError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)


def _mlp_pipeline(seed: int):
    """Standardized MLP regressor pipeline with mandatory scaling."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=5000,
            random_state=seed,
        ),
    )


def _mlp_convergence(model: Any) -> dict[str, Any]:
    """Reports optimizer convergence status."""
    mlp = model.named_steps["mlpregressor"]
    n_iter = int(getattr(mlp, "n_iter_", 0))
    converged = bool(n_iter < int(mlp.max_iter))
    return {"n_iter": n_iter, "converged": converged}


def neural_relevance(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    seed: int = 42,
    *,
    engine: str | None = None,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> NeuralRelevanceResult:
    """Surrogate model + permutation importance per variable.

    Supports explicit backend injection (engine='jax' or engine='sklearn').
    """
    try:
        backend = get_neural_discovery_backend(override=engine)

        if backend == "jax":
            try:
                from apedp.neural_discovery.integration import (
                    neural_relevance as jax_neural_relevance,
                )
                raw = jax_neural_relevance(x_data=x_data, y_data=y_data, seed=seed)
                return NeuralRelevanceResult(
                    relevant=raw.get("relevant", []),
                    surrogate_r2=raw.get("surrogate_r2", 0.0),
                    converged=raw.get("converged", True),
                    n_iter=raw.get("n_iter", 0),
                    importances=raw.get("relevance", {}),
                    note=raw.get("note"),
                )
            except _PROBE_ERRORS as e:
                logger.warning("JAX neural relevance engine failed, falling back to sklearn: %s", e)

        # Scikit-learn fallback implementation
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < policy.MIN_SAMPLE_SIZE_SURROGATE:
            return NeuralRelevanceResult(note="insufficient data for MLP")

        X = np.column_stack([np.asarray(x_data[k], float).ravel() for k in names])
        if not bool(np.all(np.isfinite(X))) or not bool(np.all(np.isfinite(y))):
            return NeuralRelevanceResult(note="non-finite data")

        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
        model = _mlp_pipeline(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, ytr)

        conv = _mlp_convergence(model)
        base = float(model.score(Xte, yte))

        if base < policy.CANDIDATE_VIABILITY_R2:
            note = f"held-out surrogate too weak (R^2={base:.3f})" + (
                "" if conv["converged"] else "; optimizer did not converge"
            )
            return NeuralRelevanceResult(
                surrogate_r2=round(base, 4),
                converged=conv["converged"],
                n_iter=conv["n_iter"],
                note=note,
            )

        rng = np.random.default_rng(seed)
        importance: dict[str, float] = {}
        for j, name in enumerate(names):
            drops: list[float] = []
            for _ in range(5):
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - float(model.score(Xp, yte)))
            importance[name] = round(float(np.mean(drops)), 4)

        floor = max(0.01, 0.05 * max(importance.values(), default=0.0))
        relevant = sorted(
            [k for k, v in importance.items() if v > floor],
            key=lambda k: -importance[k],
        )

        note = "held-out permutation importance; deterministic seed" + (
            "" if conv["converged"] else "; optimizer did not converge"
        )
        return NeuralRelevanceResult(
            relevant=relevant,
            surrogate_r2=round(base, 4),
            converged=conv["converged"],
            n_iter=conv["n_iter"],
            importances=importance,
            note=note,
        )
    except _PROBE_ERRORS as e:
        logger.warning("neural_relevance encountered recoverable error: %s", e, exc_info=True)
        return NeuralRelevanceResult(note=f"neural probe failed: {e}")


def grammar_terminals_for(
    target_unit: str,
    variables: dict[str, Any],
    engine: Any = None,
) -> dict[str, Any]:
    """Unit-ontology grammar: determines which variables may legally appear under log/exp/sqrt."""
    try:
        names = list(variables.keys())
        dimensionless: list[str] = []
        units_checked = engine is not None

        if engine is not None:
            try:
                for k in names:
                    unit = engine.check_expression_consistency(sp.Symbol(k))
                    if unit.dimensionality == engine.registry("dimensionless").dimensionality:
                        dimensionless.append(k)
            except _PROBE_ERRORS as exc:
                logger.debug("unit-consistency probe failed: %s", exc)
                dimensionless = []
                units_checked = False

        note = (
            "unit-checked"
            if units_checked
            else "unit check unavailable: names unchecked"
            if engine is not None
            else "no engine: names only, unchecked"
        )
        return {
            "log_exp_allowed": sorted(dimensionless),
            "note": note,
        }
    except _PROBE_ERRORS as e:
        logger.warning("grammar_terminals_for encountered error: %s", e, exc_info=True)
        return {"log_exp_allowed": [], "note": f"grammar probe failed: {e}"}


def distill_via_surrogate(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    seed: int = 42,
    n_dense: int = 200,
    *,
    engine: str | None = None,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> SurrogateDistillationResult | dict[str, Any]:
    """Fit a neural surrogate, sample on near-manifold grid, and distill into symbolic terms."""
    from apedp.feature_selection import sparse_select_terms

    try:
        backend = get_neural_discovery_backend(override=engine)

        if backend == "jax":
            try:
                from apedp.neural_discovery.integration import distill_via_surrogate as jax_distill
                raw = jax_distill(x_data=x_data, y_data=y_data, seed=seed, n_dense=n_dense)
                return SurrogateDistillationResult(
                    terms=raw.get("terms", []),
                    dense_r2=raw.get("surrogate_r2", 0.0),
                    note=raw.get("note"),
                )
            except _PROBE_ERRORS as e:
                logger.warning("JAX distill engine failed, falling back to sklearn: %s", e)

        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        X = np.column_stack([np.asarray(x_data[k], float).ravel() for k in names])

        if len(y) < policy.MIN_SAMPLE_SIZE_SURROGATE or not bool(np.all(np.isfinite(X))):
            return SurrogateDistillationResult(note="insufficient/non-finite data")

        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _mlp_pipeline(seed)
            model.fit(Xtr, ytr)

        conv = _mlp_convergence(model)
        r2 = float(model.score(Xte, yte))

        if r2 < policy.CANDIDATE_VIABILITY_R2:
            note = f"held-out surrogate too weak (R^2={r2:.3f})" + (
                "" if conv["converged"] else "; optimizer did not converge"
            )
            return SurrogateDistillationResult(
                dense_r2=round(r2, 4),
                note=note,
            )

        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(Xtr), n_dense)
        spread = np.std(Xtr, axis=0) + policy.VARIANCE_EPSILON
        G = Xtr[pick] + rng.normal(0, 0.05, (n_dense, Xtr.shape[1])) * spread
        G = np.clip(G, Xtr.min(axis=0), Xtr.max(axis=0))
        grid = {k: G[:, j] for j, k in enumerate(names)}
        ys = model.predict(G)

        out_sparse = sparse_select_terms(grid, ys, policy=policy)
        return SurrogateDistillationResult(
            terms=out_sparse.terms,
            surrogate_r2=round(r2, 4),
            converged=conv["converged"],
            n_iter=conv["n_iter"],
            dense_r2=round(r2, 4),
            dropped=out_sparse.dropped,
            dropped_collinear=out_sparse.dropped_collinear,
            ambiguity_groups=out_sparse.ambiguity_groups,
            stability_rate=out_sparse.stability_rate,
            note="terms describe the surrogate, not raw data directly",
        )
    except _PROBE_ERRORS as e:
        logger.warning("distill_via_surrogate encountered error: %s", e, exc_info=True)
        return SurrogateDistillationResult(note=f"distill failed: {e}")
