# Databricks notebook source
import csv
import os

import pytest
from fastapi.testclient import TestClient

from apedp.api import create_app


@pytest.fixture(scope="module")
def client():
    os.environ["APEDP_ALLOW_UNAUTHENTICATED"] = "1"
    return TestClient(create_app())


def test_discover_regress_dimensionless(client):
    resp = client.post(
        "/discover/regress",
        json={
            "variables": ["x"],
            "variable_units": {"x": "dimensionless"},
            "target_unit": "dimensionless",
            "data": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
            "output_values": [1.0, 4.0, 9.0, 16.0, 25.0],
            "generations": 5,
            "pop_size": 20,
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["best_expression"]
    assert body["best_score"] < float("inf")
    assert body["score_semantics"].startswith("penalized MSE")
    assert body["evidence_level"] == "in_sample_exploratory"


def test_discover_regress_length_mismatch_rejected(client):
    resp = client.post(
        "/discover/regress",
        json={
            "variables": ["x"],
            "data": {"x": [1.0, 2.0]},
            "output_values": [1.0],
            "generations": 2,
            "pop_size": 10,
        },
    )
    assert resp.status_code == 422


def test_dimensionful_discovery_requires_explicit_units(client):
    resp = client.post(
        "/discover/regress",
        json={
            "variables": ["x"],
            "target_unit": "joule",
            "data": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
            "output_values": [1.0, 4.0, 9.0, 16.0, 25.0],
            "generations": 2,
            "pop_size": 10,
        },
    )
    assert resp.status_code == 422
    assert "variable_units" in resp.json()["detail"]


def test_discover_bayes_reports_a_likelihood_ratio_as_such(client):
    """Likelihood arrays cannot produce a Bayes factor; the response must say so."""
    resp = client.post(
        "/discover/bayes",
        json={"model_1_likelihoods": [0.9, 0.85], "model_2_likelihoods": [0.3, 0.25]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quantity"] == "likelihood_ratio"
    assert "bayes_factor" not in body
    assert body["likelihood_ratio"] > 1.0
    assert "NOT a Bayes factor" in body["verdict"]


def test_discover_bayes_from_log_evidence_is_a_bayes_factor(client):
    resp = client.post(
        "/discover/bayes",
        json={"model_1_log_evidence": -12.0, "model_2_log_evidence": -20.0, "evidence_method": "quadrature"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quantity"] == "bayes_factor"
    assert body["bayes_factor"] > 1000.0
    assert "Model 1" in body["verdict"]
    assert "quadrature" in body["verdict"]


def test_discover_bayes_rejects_empty_request(client):
    resp = client.post("/discover/bayes", json={})
    assert resp.status_code == 422
    assert "log_evidence" in resp.json()["detail"]


def test_discover_bayes_rejects_non_likelihoods(client):
    resp = client.post(
        "/discover/bayes",
        json={"model_1_likelihoods": [-1.0, -2.0], "model_2_likelihoods": [0.5]},
    )
    assert resp.status_code == 422
    assert "non-negative" in resp.json()["detail"]


def test_metrology_propagate_linear(client):
    resp = client.post(
        "/metrology/propagate",
        json={
            "expression": "2*m",
            "param_values": {"m": 2.0},
            "param_uncertainties": {"m": 0.1},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["evaluated_value"] == 4.0
    assert abs(body["combined_uncertainty"] - 0.2) < 1e-12


def test_metrology_propagate_bad_expression_rejected(client):
    resp = client.post(
        "/metrology/propagate",
        json={"expression": "2*(", "param_values": {"m": 1.0}},
    )
    assert resp.status_code == 422


def test_metrology_calibrate_log_and_verify(client):
    log = client.post(
        "/metrology/calibrate/log",
        json={
            "sensor_id": "S1",
            "standard_id": "NIST-1",
            "procedure_name": "PROC",
            "drift_compensation_factor": 1.0,
            "uncertainty_budget": 0.01,
            "timestamp": "2026-01-01T00:00:00Z",
        },
    )
    assert log.status_code == 200, log.text
    assert log.json()["traceable"] is True
    record_id = log.json()["record_id"]

    # Test verify with JSON body
    verify_json = client.post("/metrology/calibrate/verify", json={"sensor_id": "S1"})
    assert verify_json.status_code == 200, verify_json.text
    assert verify_json.json()["traceable"] is True

    # Test verify with query param
    verify_qp = client.post("/metrology/calibrate/verify?sensor_id=S1")
    assert verify_qp.status_code == 200, verify_qp.text
    assert verify_qp.json()["traceable"] is True

    # Test revocation
    revoke_resp = client.post("/metrology/calibrate/revoke", json={"record_id": record_id, "reason": "superseded"})
    assert revoke_resp.status_code == 200, revoke_resp.text
    assert revoke_resp.json()["revoked"] is True

    # Verify after revocation
    verify_revoked = client.post("/metrology/calibrate/verify", json={"sensor_id": "S1"})
    assert verify_revoked.status_code == 200
    assert verify_revoked.json()["traceable"] is False


def test_repair_singular_and_clean(client):
    bad = client.post("/repair/anomalies", json={"formula": "1/x"})
    assert bad.status_code == 200, bad.text
    assert bad.json()["status"] == "REPAIRED"
    assert bad.json()["anomalies"]

    good = client.post("/repair/anomalies", json={"formula": "x+y"})
    assert good.status_code == 200, good.text
    assert good.json()["status"] == "APPROVED"
    assert good.json()["anomalies"] == []


def test_epistemic_node_and_bad_type(client):
    ok = client.post(
        "/epistemic/node", json={"node_id": "C1", "node_type": "Claim", "metadata": {}}
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["acyclic"] is True

    bad = client.post("/epistemic/node", json={"node_id": "X", "node_type": "claim"})
    assert bad.status_code == 422


def test_epistemic_edge_and_bad_relation(client):
    ok = client.post(
        "/epistemic/edge",
        json={
            "nodes": [
                {"node_id": "A", "node_type": "Hypothesis"},
                {"node_id": "B", "node_type": "Claim"},
            ],
            "source_id": "A",
            "target_id": "B",
            "relation": "DERIVED_FROM",
        },
    )
    assert ok.status_code == 200, ok.text

    bad = client.post(
        "/epistemic/edge",
        json={
            "nodes": [
                {"node_id": "A", "node_type": "Hypothesis"},
                {"node_id": "B", "node_type": "Claim"},
            ],
            "source_id": "A",
            "target_id": "B",
            "relation": "BOTH_FROM_NOETHER",
        },
    )
    assert bad.status_code == 422


def test_epistemic_requirement_and_tournament(client):
    req = client.post(
        "/epistemic/requirement",
        json={
            "req_id": "REQ-1",
            "goal": "Validate gravity model against data",
            "owner": "tester-team",
            "design_adr": "ADR-001-gravity",
            "impl_artifact": "model.py-implementation",
            "acceptance_test": "test_model_passes",
            "validation_evidence": "run-001-evidence",
        },
    )
    assert req.status_code == 200, req.text
    # The new tournament returns REPAIRED_CANDIDATE, not REPAIRED_AND_VALIDATED
    tourn = client.post(
        "/epistemic/tournament",
        json={
            "hypothesis_id": "H1",
            "formula_str": "1/x",
            "adversarial_probe_name": "zero",
            "boundary_flaw_detected": True,
            "repaired_formula_str": "1/(x+eps)",
            "variable_units": {"x": "dimensionless"},
            "strict_units": False,
        },
    )
    assert tourn.status_code == 200, tourn.text
    assert tourn.json()["outcome"] == "REPAIRED_CANDIDATE"


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from apedp.pipeline import DiscoveryPipeline

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for i in range(1, 11):
            writer.writerow([float(i), float(i**2)])
    monkeypatch.chdir(tmp_path)
    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    result = pipeline.run(str(csv_path), "y")
    assert result["best_score"] < float("inf")
    assert result["score_semantics"].startswith("penalized MSE")
    assert (tmp_path / "out" / "pipeline_result.json").exists()


def test_epistemic_graph_persistence_and_inspection(client):
    # Add nodes across separate requests
    r1 = client.post("/epistemic/node", json={"node_id": "P1", "node_type": "Hypothesis", "metadata": {}})
    assert r1.status_code == 200
    r2 = client.post("/epistemic/node", json={"node_id": "P2", "node_type": "Claim", "metadata": {}})
    assert r2.status_code == 200

    # Link previously added nodes without resending them
    r3 = client.post("/epistemic/edge", json={"nodes": [], "source_id": "P1", "target_id": "P2", "relation": "DERIVED_FROM"})
    assert r3.status_code == 200

    # Test GET routes
    g = client.get("/epistemic/graph")
    node_ids = [n["id"] for n in g.json()["nodes"]]
    assert "P1" in node_ids
    assert "P2" in node_ids

    lineage = client.get("/epistemic/lineage/P2")
    assert lineage.status_code == 200

    contr = client.get("/epistemic/contradictions")
    assert contr.status_code == 200

    req_v = client.get("/epistemic/requirements/verify")
    assert req_v.status_code == 200