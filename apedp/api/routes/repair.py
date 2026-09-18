"""Repair endpoints — boundary anomaly stress-testing and regularization."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from apedp.repair import AgnosticClosedLoopAssuranceSystem

logger = logging.getLogger(__name__)
router = APIRouter()


class RepairRequest(BaseModel):
    formula: str = Field(..., description="SymPy-parseable formula, e.g. 'G*m1*m2/(r**2)'")
    variable_units: dict[str, str] = Field(
        default_factory=dict,
        description="Explicit symbol -> unit mapping. Required for ambiguous tokens in strict mode.",
    )
    critical_values: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Symbol -> extra finite boundary points to probe (e.g. {'v': [299792458.0]}).",
    )
    strict_units: bool = Field(
        default=False, description="Reject ambiguous bare-token unit guesses instead of defaulting."
    )


class RepairResponse(BaseModel):
    status: str
    anomalies: list[str]
    original_expression: str
    repaired_expression: str
    inferred_unit: str
    eps_symbols: list[str] = []
    distortion: dict = {}
    validity_domain: str = ""
    #: Probes that crashed or timed out, with whether each left its question
    #: unanswered. A non-zero ``unresolved_probe_count`` is why a status can be
    #: ``UNDETERMINED`` instead of ``APPROVED``.
    probe_errors: list[dict] = []
    unresolved_probe_count: int = 0
    error: str | None = None


@router.post("/repair/anomalies", response_model=RepairResponse)
async def detect_and_repair(req: RepairRequest) -> RepairResponse:
    if len(req.formula) > 2000:
        return RepairResponse(
            status="REJECTED",
            anomalies=[],
            original_expression=req.formula[:200],
            repaired_expression=req.formula[:200],
            inferred_unit="unknown",
            error="Formula exceeds max length 2000",
        )
    system = AgnosticClosedLoopAssuranceSystem()
    result = system.autonomous_stress_test_and_repair(
        req.formula,
        variable_units=req.variable_units or None,
        critical_values=req.critical_values or None,
        strict_units=req.strict_units,
    )
    if result.get("status") == "REJECTED":
        return RepairResponse(
            status="REJECTED",
            anomalies=[],
            original_expression=req.formula,
            repaired_expression=req.formula,
            inferred_unit="unknown",
            error=result.get("error"),
        )
    anomalies = [f"{sym}: {kind}" for sym, kind in result.get("anomalies", [])]
    return RepairResponse(
        status=result.get("status", "UNKNOWN"),
        anomalies=anomalies,
        probe_errors=list(result.get("probe_errors", [])),
        unresolved_probe_count=int(result.get("unresolved_probe_count", 0) or 0),
        original_expression=str(result.get("original_expression", req.formula)),
        repaired_expression=str(result.get("repaired_expression", req.formula)),
        inferred_unit=str(result.get("inferred_unit", "unknown")),
        eps_symbols=list(result.get("eps_symbols", [])),
        distortion=dict(result.get("distortion", {}) or {}),
        validity_domain=str(result.get("validity_domain", "")),
    )
