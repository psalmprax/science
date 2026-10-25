"""HTTP tests for the /visualize API routes.

Each route writes a figure to ``output_path``. Tests use pytest's ``tmp_path``
so no artifacts are left in the repository.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apedp.api import create_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Localhost dev mode: create_app() refuses to start without a key unless
    # unauthenticated local development is explicitly enabled.
    monkeypatch.delenv("APEDP_API_KEY", raising=False)
    monkeypatch.setenv("APEDP_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("APEDP_VISUALIZATION_DIR", str(tmp_path / "viz"))
    return TestClient(create_app())


def _png(path: str) -> bool:
    import os

    with open(path, "rb") as fh:
        return os.path.exists(path) and fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_regression_route_writes_png(client, tmp_path):
    out = "regression.png"
    resp = client.post(
        "/visualize/regression",
        json={"complexity": [1, 2, 3], "r_squared": [0.5, 0.8, 0.9], "output_path": out},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"
    assert _png(str(tmp_path / "viz" / out))


def test_uncertainty_route_writes_png(client, tmp_path):
    out = "unc.png"
    resp = client.post(
        "/visualize/uncertainty",
        json={"values": [1.0, 2.0], "uncertainties": [0.1, 0.2], "output_path": out},
    )
    assert resp.status_code == 200, resp.text
    assert _png(str(tmp_path / "viz" / out))


def test_calibration_route_writes_png(client, tmp_path):
    out = "cal.png"
    resp = client.post(
        "/visualize/calibration",
        json={"x_values": [0.0, 1.0, 2.0], "y_measured": [0.0, 1.1, 1.9], "output_path": out},
    )
    assert resp.status_code == 200, resp.text
    assert _png(str(tmp_path / "viz" / out))


def test_pareto_route_writes_png(client, tmp_path):
    out = "pareto.png"
    resp = client.post(
        "/visualize/pareto",
        json={"objectives_x": [1.0, 2.0, 3.0], "objectives_y": [3.0, 2.0, 1.0], "output_path": out},
    )
    assert resp.status_code == 200, resp.text
    assert _png(str(tmp_path / "viz" / out))


def test_search_progress_route_writes_png(client, tmp_path):
    out = "progress.png"
    resp = client.post(
        "/visualize/search/progress",
        json={
            "iterations": [0, 1, 2],
            "best_fitness": [0.1, 0.2, 0.3],
            "avg_fitness": [0.05, 0.1, 0.2],
            "output_path": out,
        },
    )
    assert resp.status_code == 200, resp.text
    assert _png(str(tmp_path / "viz" / out))


def test_regression_route_rejects_bad_body(client):
    resp = client.post("/visualize/regression", json={"complexity": "not-a-list"})
    assert resp.status_code == 422


def test_download_missing_file_returns_404(client):
    resp = client.get("/visualize/download/does_not_exist.png")
    assert resp.status_code == 404


def test_download_existing_file_returns_bytes(client, tmp_path):
    root = tmp_path / "viz"
    root.mkdir(parents=True, exist_ok=True)
    (root / "x.txt").write_bytes(b"hello")
    resp = client.get("/visualize/download/x.txt")
    assert resp.status_code == 200
    assert resp.content == b"hello"


def test_download_rejects_traversal(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"top secret")
    root = tmp_path / "viz"
    root.mkdir(parents=True, exist_ok=True)
    resp = client.get("/visualize/download/..%2Fsecret.txt")
    assert resp.status_code in (400, 404)
    assert b"top secret" not in resp.content


def test_download_rejects_absolute_path(client, tmp_path):
    secret = tmp_path / "abs_secret.txt"
    secret.write_bytes(b"abs secret")
    resp = client.get(f"/visualize/download/{str(secret).replace('/', '%2F')}")
    assert resp.status_code in (400, 404)
    assert b"abs secret" not in resp.content


def test_download_rejects_symlink_escape(client, tmp_path):
    secret = tmp_path / "link_secret.txt"
    secret.write_bytes(b"link secret")
    root = tmp_path / "viz"
    root.mkdir(parents=True, exist_ok=True)
    (root / "link.txt").symlink_to(secret)
    resp = client.get("/visualize/download/link.txt")
    assert resp.status_code == 400
    assert b"link secret" not in resp.content


def test_plot_route_rejects_absolute_output_path(client, tmp_path):
    out = str(tmp_path / "escape.png")
    resp = client.post(
        "/visualize/regression",
        json={"complexity": [1, 2], "r_squared": [0.5, 0.8], "output_path": out},
    )
    assert resp.status_code == 400
    assert not (tmp_path / "escape.png").exists()


def test_plot_route_rejects_parent_traversal(client, tmp_path):
    resp = client.post(
        "/visualize/regression",
        json={"complexity": [1, 2], "r_squared": [0.5, 0.8], "output_path": "../escape.png"},
    )
    assert resp.status_code == 400
    assert not (tmp_path / "escape.png").exists()
