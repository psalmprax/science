"""Pipeline orchestration endpoints."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apedp.orchestration import PipelineJobManager, RedisPipelineJobManager

router = APIRouter()

# Both backends expose the same submit/status/result surface. Annotate the union
# so the variable is not pinned to whichever branch the reader (or mypy) sees first.
_JOB_MANAGER: PipelineJobManager | RedisPipelineJobManager

if os.environ.get("APEDP_QUEUE_BACKEND", "filesystem").lower() == "redis":
    redis_url = os.environ.get("APEDP_REDIS_URL")
    if not redis_url:
        raise RuntimeError("APEDP_REDIS_URL is required when APEDP_QUEUE_BACKEND=redis")
    _JOB_MANAGER = RedisPipelineJobManager(
        redis_url,
        queue_name=os.environ.get("APEDP_REDIS_QUEUE", "apedp:pipeline"),
        lease_seconds=int(os.environ.get("APEDP_REDIS_LEASE_SECONDS", "60")),
    )
else:
    _JOB_MANAGER = PipelineJobManager(
        state_dir=os.environ.get("APEDP_PIPELINE_JOB_DIR", "pipeline_jobs"),
        max_workers=int(os.environ.get("APEDP_PIPELINE_WORKERS", "2")),
    )


class PipelineSubmitRequest(BaseModel):
    csv_path: str
    target_column: str
    output_unit: str = "dimensionless"
    variable_units: dict[str, str] = Field(default_factory=dict)
    min_r_squared: float = Field(default=0.1, ge=0.0, le=1.0)
    strict_units: bool = False
    output_dir: str = "pipeline_results"
    max_bad_row_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    param_uncertainties: dict[str, float] | None = None
    source_metadata: dict[str, object] | None = None
    max_attempts: int = Field(default=1, ge=1, le=5)


class PipelineJobResponse(BaseModel):
    job_id: str
    status: str
    progress: str
    attempt: int
    max_attempts: int
    submitted_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False


@router.post("/pipeline/jobs", response_model=PipelineJobResponse, status_code=202)
async def submit_pipeline(req: PipelineSubmitRequest) -> dict:
    parameters = req.model_dump(exclude={"max_attempts"})
    try:
        return _JOB_MANAGER.submit(parameters, max_attempts=req.max_attempts)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/pipeline/jobs", response_model=list[PipelineJobResponse])
async def list_pipeline_jobs(limit: int = 100) -> list[dict]:
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    return _JOB_MANAGER.list(limit)


@router.get("/pipeline/jobs/{job_id}", response_model=PipelineJobResponse)
async def get_pipeline_job(job_id: str) -> dict:
    record = _JOB_MANAGER.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="pipeline job not found")
    return record


@router.post("/pipeline/jobs/{job_id}/cancel", response_model=PipelineJobResponse)
async def cancel_pipeline_job(job_id: str) -> dict:
    try:
        return _JOB_MANAGER.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pipeline job not found") from exc


@router.post("/pipeline/jobs/{job_id}/retry", response_model=PipelineJobResponse, status_code=202)
async def retry_pipeline_job(job_id: str) -> dict:
    try:
        return _JOB_MANAGER.retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="pipeline job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
