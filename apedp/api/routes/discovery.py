# Databricks notebook source
"""Discovery endpoints — symbolic regression and Bayesian model selection."""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import BayesianRefinementEngine, MultiVariableSRE, check_data_adequacy

logger = logging.getLogger(__name__)
router = APIRouter()


def _predict(expr, platform, x_data: dict) -> np.ndarray:
    """Evaluate a candidate expression on data arrays (constants at registry values)."""
    import sympy as sp

    var_names = sorted(x_data.keys())
    syms = [sp.Symbol(k) for k in var_names]
    subs = {sp.Symbol(k): v for k, v in platform.constants_values.items()}
    f = sp.lambdify(syms, expr.subs(subs), modules=["numpy"])
    args = [np.asarray(x_data[k], dtype=float) for k in var_names]
    y_pred = f(*args)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.full_like(next(iter(args)), float(y_pred), dtype=float)
    return np.asarray(y_pred, dtype=float)


class RegressionRequest(BaseModel):
    variables: list[str] = Field(..., description="Independent variable names")
    variable_units: dict[str, str] = Field(
        default_factory=dict,
        description="Variable name -> unit string. Required for dimensionful targets unless heuristic units are explicitly enabled.",
    )
    allow_heuristic_units: bool = Field(
        default=False,
        description="Allow unlisted variables to default to dimensionless for exploratory use.",
    )
    target_unit: str = Field(default="dimensionless", description="Expected unit of the target")
    data: dict[str, list[float]] = Field(..., description="Variable -> values mapping")
    output_values: list[float] = Field(..., description="Target output values")
    generations: int = Field(default=30, ge=1, le=200, description="Evolutionary generations")
    pop_size: int = Field(default=100, ge=10, le=1000, description="Evolutionary population size")
    seed: int | None = Field(default=42, description="Random seed for reproducibility")
    allow_exp: bool = Field(default=True, description="Allow exp() with dimensionless args.")
    allow_trig: bool = Field(default=True, description="Allow sin()/cos() with dimensionless args.")
    allow_log: bool = Field(default=True, description="Allow log() with dimensionless args.")
    min_r_squared: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Refuse candidates below this R²."
    )


class RegressionResponse(BaseModel):
    best_expression: str
    best_score: float
    score_semantics: str = "penalized MSE (lower is better); not R²"
    generations_run: int
    r_squared: float = float("nan")
    bic: float = float("inf")
    aic: float = float("inf")
    degenerate: bool = False
    degenerate_reason: str = ""
    data_adequacy: dict = Field(default_factory=dict)
    evidence_level: str = "in_sample_exploratory"
    scientific_claim: str = "consistency_screening_only"


class BayesRequest(BaseModel):
    """Two ways to compare models, deliberately kept distinct.

    ``model_*_log_evidence`` is a pair of *log marginal likelihoods* and yields a
    real Bayes factor. Likelihood arrays do not: ``p(D|M) = \u222b p(D|\u03b8)p(\u03b8)d\u03b8``
    needs a prior and an integral, so those fields yield a summed-likelihood
    ratio, which the response labels as such rather than dressing it as evidence.
    """

    model_1_likelihoods: list[float] | None = None
    model_2_likelihoods: list[float] | None = None
    model_1_log_evidence: float | None = None
    model_2_log_evidence: float | None = None
    evidence_method: str = "unspecified"


@router.post("/discover/regress", response_model=RegressionResponse)
async def symbolic_regression(req: RegressionRequest) -> RegressionResponse:
    try:
        x_data = {k: np.asarray(v, dtype=float) for k, v in req.data.items()}
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Non-numeric data: {e}") from e
    y_data = np.asarray(req.output_values, dtype=float)
    adequacy = check_data_adequacy(
        {k: x_data[k] for k in req.variables if k in x_data}, y_data
    )
    if adequacy["verdict"] == "INSUFFICIENT":
        raise HTTPException(status_code=422, detail={
            "message": "Insufficient data for discovery",
            "adequacy": adequacy,
        })
    for name, values in x_data.items():
        if len(values) != len(y_data):
            raise HTTPException(
                status_code=422,
                detail=f"Length mismatch: '{name}' has {len(values)} values, target has {len(y_data)}",
            )
    missing = [v for v in req.variables if v not in x_data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing data for variables: {missing}")
    undeclared_units = [v for v in req.variables if v not in req.variable_units]
    if (
        undeclared_units
        and req.target_unit != "dimensionless"
        and not req.allow_heuristic_units
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Explicit variable_units are required for dimensionful discovery; "
                f"missing: {undeclared_units}. Set allow_heuristic_units=true only "
                "for exploratory work."
            ),
        )

    engine = DimensionalAssuranceEngine()
    for var in req.variables:
        try:
            engine.register_variable(var, req.variable_units.get(var, "dimensionless"))
        except (ValueError, TypeError, AttributeError) as e:
            raise HTTPException(status_code=422, detail=f"Bad unit for '{var}': {e}") from e
    try:
        engine.registry(req.target_unit)
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=422, detail=f"Bad target unit '{req.target_unit}': {e}") from e

    platform = ClosedLoopAssuranceSystem(engine)
    sre = MultiVariableSRE(
        assurance_system=platform,
        independent_vars=req.variables,
        target_unit_str=req.target_unit,
        allow_exp=req.allow_exp,
        allow_trig=req.allow_trig,
        allow_log=req.allow_log,
    )
    best_expr, best_score, history = sre.run_discovery(
        x_data={k: x_data[k] for k in req.variables},
        y_data=y_data,
        generations=req.generations,
        pop_size=req.pop_size,
        seed=req.seed,
    )
    if best_expr is None or not np.isfinite(best_score):
        raise HTTPException(status_code=422, detail="No dimensionally consistent candidate found")
    # Real selection statistics: R² / BIC / AIC + degeneracy refusal.
    # Previously only penalized MSE was returned, so a degenerate
    # constant fit could win on the simplicity term.
    try:
        scored = BayesianRefinementEngine.score_candidate(
            best_expr,
            y_data,
            _predict(best_expr, platform, {k: x_data[k] for k in req.variables}),
        )
    except (TypeError, ValueError, ArithmeticError, AttributeError):
        # Candidate scoring failed on numeric/symbolic grounds (non-castable data,
        # shape mismatch, non-finite evaluation). Report an unscoreable candidate
        # so the degeneracy check below refuses it, and record why.
        logger.warning("Candidate scoring failed for %s", best_expr, exc_info=True)
        scored = {"r_squared": float("nan"), "bic": float("inf"), "aic": float("inf"),
                  "degenerate": True, "degenerate_reason": "scoring failed"}
    degen, reason = BayesianRefinementEngine.is_degenerate(
        y_data,
        _predict(best_expr, platform, {k: x_data[k] for k in req.variables}),
        best_expr,
        min_r_squared=req.min_r_squared,
    )
    if degen:
        raise HTTPException(
            status_code=422,
            detail=f"Degenerate candidate refused (R²={scored.get('r_squared')}: {reason})",
        )
    return RegressionResponse(
        best_expression=str(best_expr),
        best_score=float(best_score),
        generations_run=len(history),
        r_squared=float(scored.get("r_squared", float("nan"))),
        bic=float(scored.get("bic", float("inf"))),
        aic=float(scored.get("aic", float("inf"))),
        degenerate=False,
        degenerate_reason=reason,
        data_adequacy=adequacy,
    )


@router.post("/discover/bayes")
async def bayesian_comparison(req: BayesRequest) -> dict:
    """Compare two models, reporting which quantity was actually computed."""
    if req.model_1_log_evidence is not None and req.model_2_log_evidence is not None:
        bf, verdict, log_bf = BayesianRefinementEngine.compute_bayes_factor_from_log_evidence(
            req.model_1_log_evidence,
            req.model_2_log_evidence,
            req.evidence_method,
            req.evidence_method,
        )
        return {
            "quantity": "bayes_factor",
            "bayes_factor": bf,
            "verdict": verdict,
            "log10_bayes_factor": log_bf,
            "evidence_method": req.evidence_method,
        }

    if req.model_1_likelihoods is None or req.model_2_likelihoods is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide model_1_likelihoods and model_2_likelihoods (summed-likelihood "
                "ratio), or model_1_log_evidence and model_2_log_evidence (Bayes factor)"
            ),
        )

    try:
        ratio, description, log_ratio = BayesianRefinementEngine.compute_likelihood_ratio(
            np.asarray(req.model_1_likelihoods, dtype=float),
            np.asarray(req.model_2_likelihoods, dtype=float),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "quantity": "likelihood_ratio",
        "likelihood_ratio": ratio,
        "verdict": description,
        "log10_likelihood_ratio": log_ratio,
    }
