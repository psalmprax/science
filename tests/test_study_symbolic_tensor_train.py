"""The study of a tensor train read from a wide table: what it measures on top of ``proposers/tensor_train_reading.py``."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_wider_conditions as wider
import study_symbolic_tensor_train as study

from apedp_dyn.proposers import tensor_train_reading as reading

LOW, HIGH = 1.0, 5.0


def _bases(X: np.ndarray) -> list[np.ndarray]:
    return [study.basis(X[:, i], LOW, HIGH) for i in range(X.shape[1])]


def _noisy(clean: np.ndarray, rng: np.random.Generator, share: float = 0.01) -> tuple[np.ndarray, float]:
    sigma = share * float(np.sqrt(np.mean(clean ** 2)))
    return clean + sigma * rng.standard_normal(len(clean)), sigma


def test_a_core_holds_its_own_function_and_not_another() -> None:
    rng = np.random.default_rng(5)
    X = rng.uniform(LOW, HIGH, (400, 3))
    y, _ = _noisy(X[:, 0] ** 2 * np.log(X[:, 1] + 1.0) / (X[:, 2] + 0.7), rng)
    model = reading.grown(_bases(X), y, rank_max=1)[0]
    grid = np.linspace(LOW, HIGH, study.GRID)
    spaces = study.core_spaces(model, grid, LOW, HIGH)
    for space, own in zip(spaces, (grid ** 2, np.log(grid + 1.0), 1.0 / (grid + 0.7)), strict=True):
        assert study.miss(space, own) < 0.02
    assert study.miss(spaces[0], np.sin(2.9 * grid)) > 0.2


def test_the_table_is_the_one_the_wide_runner_reads() -> None:
    job = {"index": 40, "law": 0, "text": "2.0*x0*log(1 + x1) + 3.0*sin(x2)*x3", "inputs": 4, "noise": 0.05}
    t = study.table(job)
    key = hashlib.sha256(b"wider-conditions|40").digest()[:8]
    X = np.random.default_rng(int.from_bytes(key, "big")).uniform(LOW, HIGH, (wider.ROWS + wider.TEST_ROWS, 4))
    assert np.array_equal(t["X"], X[: study.ROWS]) and np.array_equal(t["X_clean"], X[wider.ROWS:])
    clean = 2.0 * X[:, 0] * np.log(1.0 + X[:, 1]) + 3.0 * np.sin(X[:, 2]) * X[:, 3]
    assert np.allclose(t["y_clean"], clean[wider.ROWS:])
    assert abs(float(np.std(t["y"] - clean[: study.ROWS])) / t["sigma"] - 1.0) < 0.1
    assert t["sigma"] == 0.05 * float(np.sqrt(np.mean(clean[: wider.ROWS] ** 2)))
