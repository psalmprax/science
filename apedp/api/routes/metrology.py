"""Metrology endpoints — uncertainty propagation and calibration traceability."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apedp.metrology import (
    AnalyticalUncertaintyPropagator,
    CalibrationTraceabilityChain,
    has_singularity_risk,
)
from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)
router = APIRouter()


class UncertaintyRequest(BaseModel):
    expression: str = Field(..., description="SymPy expression, e.g. '0.5*m*v**2'")
    param_values: dict[str, float] = Field(..., description="Parameter name -> nominal value")
    param_uncertainties: dict[str, float] = Field(
        default_factory=dict, description="Parameter name -> standard uncertainty"
    )
    variable_order: list[str] | None = Field(default=None, description="Ordering for covariance input")
    method: str = Field(
        default="auto",
        description="Uncertainty method: 'gum' (first-order Taylor), 'monte_carlo' (GUM Suppl. 1 sampling), or 'auto' (monte_carlo when has_singularity_risk, else gum).",
    )
    mc_samples: int = Field(default=20000, ge=1000, le=200000, description="Monte Carlo draws.")


class UncertaintyResponse(BaseModel):
    evaluated_value: float
    combined_uncertainty: float
    expanded_uncertainty_95: float
    relative_uncertainty: float
    variance_budget: dict[str, float]
    percentage_budget: dict[str, float]
    method: str = "gum"
    warning: str | None = None


@router.post("/metrology/propagate", response_model=UncertaintyResponse)
async def propagate_uncertainty(req: UncertaintyRequest) -> UncertaintyResponse:
    try:
        expr = safe_parse_expr(req.expression)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Unparseable expression: {e}") from e
    if req.method not in ("gum", "monte_carlo", "auto"):
        raise HTTPException(status_code=422, detail="method must be 'gum', 'monte_carlo' or 'auto'")
    method = req.method
    if method == "auto":
        try:
            method = "monte_carlo" if has_singularity_risk(expr) else "gum"
        except (AttributeError, TypeError, ValueError):
            method = "gum"
    warning: str | None = None
    if method == "monte_carlo":
        try:
            mc = AnalyticalUncertaintyPropagator.propagate_uncertainty_monte_carlo(
                expr,
                param_values=dict(req.param_values),
                param_uncertainties=dict(req.param_uncertainties) or None,
                num_samples=req.mc_samples,
            )
            return UncertaintyResponse(
                evaluated_value=float(mc["mean"]),
                combined_uncertainty=float(mc["std"]),
                expanded_uncertainty_95=float(mc["p97_5"] - mc["p2_5"]) / 2.0,
                relative_uncertainty=(
                    float(mc["std"] / abs(mc["mean"])) if abs(float(mc["mean"])) > 1e-15 else float("nan")
                ),
                variance_budget={},
                percentage_budget={},
                method="monte_carlo",
            )
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=f"Monte Carlo propagation failed: {e}") from e
    try:
        report = AnalyticalUncertaintyPropagator.propagate_uncertainty(
            expr,
            param_values=dict(req.param_values),
            param_uncertainties=dict(req.param_uncertainties) or None,
            variable_order=req.variable_order,
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Propagation failed: {e}") from e
    # First-order Taylor is unreliable at singularities; warn when the
    # expression contains a pole, a branch cut, or a log/tan-style boundary.
    # The classification lives in ``has_singularity_risk`` so that it is
    # unit-testable without an HTTP round trip.
    try:
        risky = has_singularity_risk(expr)
        if risky:
            warning = (
                "First-order GUM can fail near singularities; "
                "re-run with method='monte_carlo' for GUM Supplement 1 sampling."
            )
    except (AttributeError, TypeError, ValueError):
        # The warning is advisory only: an expression the probe cannot classify
        # simply yields no warning instead of failing the request.
        logger.debug("Could not classify singularity risk for %s", req.expression, exc_info=True)
    return UncertaintyResponse(
        evaluated_value=float(report.evaluated_value),
        combined_uncertainty=float(report.combined_uncertainty),
        expanded_uncertainty_95=float(report.expanded_uncertainty_95),
        relative_uncertainty=float(report.relative_uncertainty),
        variance_budget={k: float(v) for k, v in report.budget.items()},
        percentage_budget={k: float(v) for k, v in report.percentage_budget.items()},
        method="gum",
        warning=warning,
    )


_CALIBRATION_CHAIN = CalibrationTraceabilityChain()


class CalibrationLogRequest(BaseModel):
    sensor_id: str
    standard_id: str
    procedure_name: str
    drift_compensation_factor: float = 1.0
    uncertainty_budget: float = 0.0
    timestamp: str


class VerifyCalibrationRequest(BaseModel):
    sensor_id: str


class RevokeRequest(BaseModel):
    record_id: str
    reason: str = ""


@router.post("/metrology/calibrate/log")
async def log_calibration(req: CalibrationLogRequest) -> dict:
    record_id = _CALIBRATION_CHAIN.log_calibration(
        sensor_id=req.sensor_id,
        standard_id=req.standard_id,
        procedure_name=req.procedure_name,
        drift_compensation_factor=req.drift_compensation_factor,
        uncertainty_budget=req.uncertainty_budget,
        timestamp=req.timestamp,
    )
    return {"record_id": record_id, "traceable": _CALIBRATION_CHAIN.verify_traceability(req.sensor_id)}


@router.post("/metrology/calibrate/verify")
async def verify_calibration(req: VerifyCalibrationRequest | None = None, sensor_id: str | None = None) -> dict:
    sid = req.sensor_id if req is not None else sensor_id
    if not sid:
        raise HTTPException(status_code=422, detail="sensor_id is required")
    return {"sensor_id": sid, "traceable": _CALIBRATION_CHAIN.verify_traceability(sid)}


@router.post("/metrology/calibrate/revoke")
async def revoke_calibration(req: RevokeRequest) -> dict:
    revoked = _CALIBRATION_CHAIN.revoke_calibration(req.record_id, req.reason)
    return {"record_id": req.record_id, "revoked": revoked}
