"""Epistemic endpoints — knowledge graph and requirements traceability."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apedp.epistemic import EpistemicGraph, LifecycleManager, RequirementsTraceabilityMatrix

logger = logging.getLogger(__name__)
router = APIRouter()


class AddNodeRequest(BaseModel):
    node_id: str
    node_type: str = Field(..., description="One of EpistemicGraph.ENTITY_TYPES")
    metadata: dict = Field(default_factory=dict)


class AddEdgeRequest(BaseModel):
    nodes: list[AddNodeRequest] = Field(default_factory=list)
    source_id: str
    target_id: str
    relation: str = Field(..., description="One of EpistemicGraph.EDGE_TYPES")


class RequirementRequest(BaseModel):
    req_id: str
    goal: str
    owner: str
    design_adr: str
    impl_artifact: str
    acceptance_test: str
    validation_evidence: str


class TournamentRequest(BaseModel):
    hypothesis_id: str
    formula_str: str
    adversarial_probe_name: str
    boundary_flaw_detected: bool
    repaired_formula_str: str


_EPISTEMIC_GRAPH = EpistemicGraph()
_LIFECYCLE_MANAGER = LifecycleManager()
_REQUIREMENTS_MATRIX = RequirementsTraceabilityMatrix()


@router.post("/epistemic/node")
async def add_node(req: AddNodeRequest) -> dict:
    try:
        _EPISTEMIC_GRAPH.add_node(req.node_id, req.node_type, req.metadata)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"node_id": req.node_id, "node_type": req.node_type, "acyclic": _EPISTEMIC_GRAPH.validate_graph_acyclicity()}


@router.post("/epistemic/edge")
async def add_edge(req: AddEdgeRequest) -> dict:
    try:
        for node in req.nodes:
            _EPISTEMIC_GRAPH.add_node(node.node_id, node.node_type, node.metadata)
        _EPISTEMIC_GRAPH.add_edge(req.source_id, req.target_id, req.relation)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "source": req.source_id,
        "target": req.target_id,
        "relation": req.relation,
        "acyclic": _EPISTEMIC_GRAPH.validate_graph_acyclicity(),
        "contradictions": len(_EPISTEMIC_GRAPH.find_contradictions()),
    }


@router.get("/epistemic/graph")
async def get_epistemic_graph() -> dict:
    return _EPISTEMIC_GRAPH.export_graph()


@router.get("/epistemic/lineage/{node_id}")
async def get_lineage(node_id: str) -> dict:
    try:
        return {"node_id": node_id, "lineage": _EPISTEMIC_GRAPH.get_lineage(node_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/epistemic/contradictions")
async def get_contradictions() -> dict:
    return {"contradictions": _EPISTEMIC_GRAPH.find_contradictions()}


@router.post("/epistemic/requirement")
async def register_requirement(req: RequirementRequest) -> dict:
    _REQUIREMENTS_MATRIX.register_requirement(
        req_id=req.req_id,
        goal=req.goal,
        owner=req.owner,
        design_adr=req.design_adr,
        impl_artifact=req.impl_artifact,
        acceptance_test=req.acceptance_test,
        validation_evidence=req.validation_evidence,
    )
    complete, missing = _REQUIREMENTS_MATRIX.verify_completeness()
    return {"req_id": req.req_id, "complete": complete, "missing": missing}


@router.get("/epistemic/requirements/verify")
async def verify_requirements() -> dict:
    complete, missing = _REQUIREMENTS_MATRIX.verify_completeness()
    return {"complete": complete, "missing": missing}


@router.post("/epistemic/tournament")
async def run_tournament(req: TournamentRequest) -> dict:
    from apedp.epistemic import AdversarialDeliberationTournament

    tournament = AdversarialDeliberationTournament(_EPISTEMIC_GRAPH, _LIFECYCLE_MANAGER)
    try:
        return tournament.run_tournament(
            hypothesis_id=req.hypothesis_id,
            formula_str=req.formula_str,
            adversarial_probe_name=req.adversarial_probe_name,
            boundary_flaw_detected=req.boundary_flaw_detected,
            repaired_formula_str=req.repaired_formula_str,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
