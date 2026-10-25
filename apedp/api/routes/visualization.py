"""Visualization API routes.

All figures are written to, and served from, a single output directory:
``$APEDP_VISUALIZATION_DIR`` if set, otherwise ``./visualization_output``.
Client-supplied paths are resolved inside that directory; absolute paths,
``..`` traversal and symlinks that escape it are rejected with HTTP 400.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from apedp import visualization

router = APIRouter(prefix="/visualize", tags=["visualization"])

_DIR_ENV = "APEDP_VISUALIZATION_DIR"
_DEFAULT_DIR = "visualization_output"


def _output_root() -> Path:
    root = Path(os.environ.get(_DIR_ENV) or Path.cwd() / _DEFAULT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _resolve_inside_root(name: str) -> str:
    """Resolve ``name`` under the output root or raise HTTP 400."""
    if not name or os.path.isabs(name) or "\x00" in name:
        raise HTTPException(status_code=400, detail="output_path must be a relative file name")
    root = _output_root()
    candidate = (root / name).resolve()
    if os.path.commonpath([str(root), str(candidate)]) != str(root) or candidate == root:
        raise HTTPException(status_code=400, detail="output_path escapes the visualization directory")
    return str(candidate)


class RegressionFitnessRequest(BaseModel):
    complexity: list[int]
    r_squared: list[float]
    output_path: str = "regression_fitness.png"


class UncertaintyRequest(BaseModel):
    values: list[float]
    uncertainties: list[float]
    labels: list[str] | None = None
    output_path: str = "uncertainty_propagation.png"


class CalibrationRequest(BaseModel):
    x_values: list[float]
    y_measured: list[float]
    y_ideal: list[float] | None = None
    output_path: str = "calibration_curve.png"


class HeatmapRequest(BaseModel):
    data: list[list[float]]
    x_labels: list[str] | None = None
    y_labels: list[str] | None = None
    title: str = "Correlation Heatmap"
    output_path: str = "heatmap.png"


class ParetoRequest(BaseModel):
    objectives_x: list[float]
    objectives_y: list[float]
    labels: list[str] | None = None
    output_path: str = "pareto_front.png"


class ViolinRequest(BaseModel):
    data: dict[str, list[float]]
    title: str = "Distribution Comparison"
    output_path: str = "violin.png"


class SearchProgressRequest(BaseModel):
    iterations: list[int]
    best_fitness: list[float]
    avg_fitness: list[float]
    diversity: list[float] | None = None
    exploration_rate: list[float] | None = None
    output_path: str = "search_progress.png"


@router.post("/regression")
async def visualize_regression(req: RegressionFitnessRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = visualization.plot_regression_fitness(req.complexity, req.r_squared, out)
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/uncertainty")
async def visualize_uncertainty(req: UncertaintyRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = visualization.plot_uncertainty_propagation(
            req.values, req.uncertainties, req.labels, out
        )
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibration")
async def visualize_calibration(req: CalibrationRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = visualization.plot_calibration_curve(
            req.x_values, req.y_measured, req.y_ideal, out
        )
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heatmap")
async def visualize_heatmap(req: HeatmapRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = visualization.plot_heatmap(
            req.data, req.x_labels, req.y_labels, req.title, out
        )
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pareto")
async def visualize_pareto(req: ParetoRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = visualization.plot_pareto_front(
            req.objectives_x, req.objectives_y, req.labels, out
        )
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/violin")
async def visualize_violin(req: ViolinRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = visualization.plot_violin_distribution(req.data, req.title, out)
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _search_visualizer(req: SearchProgressRequest) -> Any:
    viz = visualization.SearchProcessVisualizer()
    for i, (it, bf, af) in enumerate(zip(req.iterations, req.best_fitness, req.avg_fitness)):
        div = req.diversity[i] if req.diversity and i < len(req.diversity) else None
        exp = req.exploration_rate[i] if req.exploration_rate and i < len(req.exploration_rate) else None
        viz.update(it, bf, af, div, exp)
    return viz


@router.post("/search/progress")
async def visualize_search_progress(req: SearchProgressRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path)
    try:
        path = _search_visualizer(req).plot_search_progress(out)
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/dashboard")
async def visualize_search_dashboard(req: SearchProgressRequest) -> dict[str, Any]:
    out = _resolve_inside_root(req.output_path.replace(".png", ".html"))
    try:
        path = _search_visualizer(req).create_live_dashboard_html(out)
        return {"status": "success", "output_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_visualization(filename: str) -> FileResponse:
    file_path = _resolve_inside_root(filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=os.path.basename(filename))
