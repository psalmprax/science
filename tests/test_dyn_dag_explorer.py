"""Tests for Epistemic Referee Dashboard & Visual DAG Explorer.

Verifies:
1. extract_dag_lineage: nodes, edges, cytoscape elements, and version tracking.
2. compute_pareto_frontier: multi-objective non-dominated selection across (dsr, rmse).
3. build_burned_holdouts_view: burned holdout registry reporting with candidate and rejection rationale.
4. FastAPI endpoint integration (/api/epistemic/dag, pareto, burned_holdouts).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
)
from apedp_dyn.ui.dag_explorer import (
    build_burned_holdouts_view,
    compute_pareto_frontier,
    create_epistemic_explorer_app,
    extract_dag_lineage,
)


def test_dag_lineage_extraction_and_cytoscape():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "experiment_ledger.jsonl"
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)

        # Register Parent Candidate A
        ledger.register_candidate("cand_A", "proposer_1", "y = sin(x)")
        ledger.update_lifecycle_state("cand_A", CandidateLifecycleState.DEVELOPMENT_PASSED, metrics={"rmse": 0.45})

        # Register Child Candidate B mutated from A
        ledger.register_candidate("cand_B", "proposer_1", "y = sin(x) + cos(x)", parent_candidate_id="cand_A")
        ledger.update_lifecycle_state("cand_B", CandidateLifecycleState.CONFIRMED_VALID, metrics={"rmse": 0.12, "dsr": 0.98})

        # Extract DAG
        dag = extract_dag_lineage(ledger)
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1

        edge = dag.edges[0]
        assert edge.source == "cand_A"
        assert edge.target == "cand_B"

        # Check Cytoscape export
        cy = dag.to_cytoscape()
        assert len(cy) == 3  # 2 nodes + 1 edge
        node_b = next(el for el in cy if el["data"]["id"] == "cand_B")
        assert node_b["data"]["color"] == "#28a745"  # Confirmed valid green


def test_pareto_frontier_computation():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "experiment_ledger.jsonl"
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)

        # Cand 1: Low RMSE (0.1), High DSR (0.98) -> Pareto optimal
        ledger.register_candidate("c1", "p1", "f1")
        ledger.update_lifecycle_state("c1", CandidateLifecycleState.CONFIRMED_VALID, metrics={"rmse": 0.1, "dsr": 0.98})

        # Cand 2: High RMSE (0.5), Low DSR (0.80) -> Strictly dominated by Cand 1
        ledger.register_candidate("c2", "p1", "f2")
        ledger.update_lifecycle_state("c2", CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED, metrics={"rmse": 0.5, "dsr": 0.80})

        # Cand 3: Ultra low RMSE (0.05), Lower DSR (0.96) -> Pareto optimal (trade-off)
        ledger.register_candidate("c3", "p1", "f3")
        ledger.update_lifecycle_state("c3", CandidateLifecycleState.CONFIRMED_VALID, metrics={"rmse": 0.05, "dsr": 0.96})

        candidates = ledger.get_all_candidates()

        # Maximize DSR, Minimize RMSE
        pareto = compute_pareto_frontier(candidates, objective_keys=("dsr", "rmse"), maximize=(True, False))
        pareto_ids = {c.candidate_id for c in pareto}

        assert "c1" in pareto_ids
        assert "c3" in pareto_ids
        assert "c2" not in pareto_ids  # Dominated


def test_burned_holdouts_view_formatting():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "experiment_ledger.jsonl"
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)

        ledger.register_candidate("cand_fail", "proposer_x", "bad_model(x)")
        ledger.record_data_exposure("cand_fail", "dataset_btc", "slice_holdout_1", DatasetRole.CONFIRMATION_HOLDOUT, outcome_status="FAILED")
        ledger.update_lifecycle_state("cand_fail", CandidateLifecycleState.CONFIRMATION_REJECTED, rejection_reason="OOS RMSE exceeded bounds")

        view = build_burned_holdouts_view(ledger)
        assert len(view) == 1
        record = view[0]
        assert record["dataset_id"] == "dataset_btc"
        assert record["slice_identifier"] == "slice_holdout_1"
        assert record["burned_by_candidate_id"] == "cand_fail"
        assert record["rejection_reason"] == "OOS RMSE exceeded bounds"


def test_fastapi_epistemic_endpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "experiment_ledger.jsonl"
        ledger = FullLineageExperimentLedger(persistence_path=ledger_path)
        ledger.register_candidate("api_cand", "p1", "formula_api")
        ledger.update_lifecycle_state("api_cand", CandidateLifecycleState.CONFIRMED_VALID, metrics={"dsr": 0.99, "rmse": 0.08})

        with patch.dict("os.environ", {"APEDP_STORAGE_DIR": tmpdir}):
            client = TestClient(create_epistemic_explorer_app(), raise_server_exceptions=False)

            # Test DAG endpoint
            r_dag = client.get("/api/epistemic/dag")
            assert r_dag.status_code == 200
            data_dag = r_dag.json()
            assert "graph" in data_dag
            assert len(data_dag["graph"]["nodes"]) == 1

            # Test Pareto endpoint
            r_pareto = client.get("/api/epistemic/pareto?objectives=dsr,rmse&maximize=true,false")
            assert r_pareto.status_code == 200
            data_pareto = r_pareto.json()
            assert len(data_pareto) == 1
            assert data_pareto[0]["candidate_id"] == "api_cand"

            # Test Burned Holdouts endpoint
            r_burned = client.get("/api/epistemic/burned_holdouts")
            assert r_burned.status_code == 200
            assert isinstance(r_burned.json(), list)
