"""Epistemic Referee Dashboard & Visual DAG Explorer for APEDP-X / CPMD.

Provides graph analysis, multi-objective Pareto frontier identification,
and visual anti-snooping monitoring across candidate hypothesis lifecycles:

1. DAG Lineage Extraction:
   Transforms FullLineageExperimentLedger records into topological DAG structures
   (nodes, mutation edges, gate verdicts, parent-child lineages) formatted for
   Cytoscape / Vis.js / D3 rendering.

2. Multi-Objective Pareto Frontier Computation:
   Identifies non-dominated candidate hypotheses across competing scientific
   trade-offs (e.g. predictive accuracy vs formula complexity vs Deflated Sharpe Ratio).

3. Burned Confirmation Holdout Registry Monitor:
   Surfaces irreversible data-snooping boundaries (AGENTS.md §1.2), displaying
   burned holdout slices, failing candidate IDs, and referee rejection reasons.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateExperimentNode,
    FullLineageExperimentLedger,
)


@dataclass
class DAGNode:
    id: str
    label: str
    proposer_id: str
    formula: str
    version: int
    state: str
    metrics: dict[str, Any]
    parent_id: str | None


@dataclass
class DAGEdge:
    source: str
    target: str
    relation: str = "derived_from"


@dataclass
class DAGGraph:
    nodes: list[DAGNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)

    def to_cytoscape(self) -> list[dict[str, Any]]:
        """Format as Cytoscape.js elements."""
        elements = []
        for n in self.nodes:
            s_lower = n.state.lower()
            color = "#28a745" if s_lower == "confirmed_valid" else "#dc3545" if ("rejected" in s_lower or "failed" in s_lower) else "#007bff"
            elements.append({
                "data": {
                    "id": n.id,
                    "label": n.label,
                    "proposer": n.proposer_id,
                    "formula": n.formula,
                    "version": n.version,
                    "state": n.state,
                    "metrics": n.metrics,
                    "color": color,
                }
            })
        for e in self.edges:
            elements.append({
                "data": {
                    "id": f"{e.source}->{e.target}",
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                }
            })
        return elements

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "proposer_id": n.proposer_id,
                    "formula": n.formula,
                    "version": n.version,
                    "state": n.state,
                    "metrics": n.metrics,
                    "parent_id": n.parent_id,
                }
                for n in self.nodes
            ],
            "edges": [{"source": e.source, "target": e.target, "relation": e.relation} for e in self.edges],
        }


def extract_dag_lineage(ledger: FullLineageExperimentLedger) -> DAGGraph:
    """Extract full candidate hypothesis mutation DAG from experiment ledger."""
    nodes: list[DAGNode] = []
    edges: list[DAGEdge] = []

    candidates = ledger.get_all_candidates()

    for cand in candidates:
        label = f"{cand.candidate_id} (v{cand.version})"
        state_str = cand.state.value if hasattr(cand.state, "value") else str(cand.state)
        node = DAGNode(
            id=cand.candidate_id,
            label=label,
            proposer_id=cand.proposer_id,
            formula=cand.formula,
            version=cand.version,
            state=state_str,
            metrics=cand.metrics,
            parent_id=cand.parent_candidate_id,
        )
        nodes.append(node)

        if cand.parent_candidate_id:
            edges.append(DAGEdge(source=cand.parent_candidate_id, target=cand.candidate_id))

    return DAGGraph(nodes=nodes, edges=edges)


def compute_pareto_frontier(
    candidates: Sequence[CandidateExperimentNode],
    objective_keys: tuple[str, ...],
    maximize: tuple[bool, ...],
) -> list[CandidateExperimentNode]:
    """Compute non-dominated Pareto frontier across multiple numerical objectives.

    Args:
        candidates: Sequence of CandidateExperimentNode records.
        objective_keys: Tuple of metric keys to evaluate (e.g. ("dsr", "complexity")).
        maximize: Tuple of booleans, True if objective is to be maximized, False if minimized.

    Returns:
        List of Pareto-optimal (non-dominated) CandidateExperimentNode objects.
    """
    valid_candidates: list[tuple[CandidateExperimentNode, list[float]]] = []

    for c in candidates:
        vals = []
        has_all = True
        for key in objective_keys:
            if key not in c.metrics or not isinstance(c.metrics[key], (int, float)):
                has_all = False
                break
            vals.append(float(c.metrics[key]))
        if has_all:
            valid_candidates.append((c, vals))

    if not valid_candidates:
        return []

    pareto_front: list[CandidateExperimentNode] = []

    for i, (cand_a, vals_a) in enumerate(valid_candidates):
        is_dominated = False
        for j, (_cand_b, vals_b) in enumerate(valid_candidates):
            if i == j:
                continue

            # Check if B dominates A: B is at least as good in all objectives and strictly better in at least one
            b_better_or_equal = True
            b_strictly_better = False

            for k in range(len(objective_keys)):
                is_max = maximize[k]
                val_a = vals_a[k]
                val_b = vals_b[k]

                if is_max:
                    if val_b < val_a:
                        b_better_or_equal = False
                        break
                    if val_b > val_a:
                        b_strictly_better = True
                else:  # minimize
                    if val_b > val_a:
                        b_better_or_equal = False
                        break
                    if val_b < val_a:
                        b_strictly_better = True

            if b_better_or_equal and b_strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(cand_a)

    return pareto_front


def build_burned_holdouts_view(ledger: FullLineageExperimentLedger) -> list[dict[str, Any]]:
    """Build detailed audit view of all burned confirmation holdout slices."""
    burned = ledger.get_burned_holdouts()
    view = []

    for (dataset_id, slice_id), candidate_id in burned.items():
        record = {
            "dataset_id": dataset_id,
            "slice_identifier": slice_id,
            "burned_by_candidate_id": candidate_id,
            "formula": "UNKNOWN",
            "state": "UNKNOWN",
            "rejection_reason": None,
        }
        try:
            cand = ledger.get_candidate(candidate_id)
            record["formula"] = cand.formula
            record["state"] = cand.state.value if hasattr(cand.state, "value") else str(cand.state)
            record["rejection_reason"] = cand.rejection_reason
        except KeyError:
            pass

        view.append(record)

    return sorted(view, key=lambda r: (r["dataset_id"], r["slice_identifier"]))


def create_epistemic_explorer_app() -> Any:
    """Create a FastAPI application serving the epistemic referee explorer API."""
    import os

    from fastapi import FastAPI

    explorer_app = FastAPI(
        title="APEDP-X Epistemic DAG & Referee Explorer",
        description="Interactive visual API for candidate lineages, Pareto frontiers, and holdout registries.",
        version="1.0.0",
    )

    @explorer_app.get("/api/epistemic/dag")
    def get_dag() -> dict[str, Any]:
        ledger_path = Path(os.environ.get("APEDP_STORAGE_DIR", "artifacts/storage")) / "experiment_ledger.jsonl"
        if not ledger_path.is_file():
            return {"graph": {"nodes": [], "edges": []}, "cytoscape": []}
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)
        dag = extract_dag_lineage(ledger)
        return {"graph": dag.to_dict(), "cytoscape": dag.to_cytoscape()}

    @explorer_app.get("/api/epistemic/pareto")
    def get_pareto(objectives: str = "dsr,rmse", maximize: str = "true,false") -> list[dict[str, Any]]:
        ledger_path = Path(os.environ.get("APEDP_STORAGE_DIR", "artifacts/storage")) / "experiment_ledger.jsonl"
        if not ledger_path.is_file():
            return []
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)
        candidates = ledger.get_all_candidates()

        obj_keys = tuple(o.strip() for o in objectives.split(","))
        max_flags = tuple(m.strip().lower() in ("true", "1", "yes") for m in maximize.split(","))

        pareto = compute_pareto_frontier(candidates, obj_keys, max_flags)
        return [c.to_dict() for c in pareto]

    @explorer_app.get("/api/epistemic/burned_holdouts")
    def get_burned_holdouts() -> list[dict[str, Any]]:
        ledger_path = Path(os.environ.get("APEDP_STORAGE_DIR", "artifacts/storage")) / "experiment_ledger.jsonl"
        if not ledger_path.is_file():
            return []
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)
        return build_burned_holdouts_view(ledger)

    return explorer_app

