"""Tests for the shared seed and run-provenance contract."""

import json
import random

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from apedp.core import DimensionalAssuranceEngine
from apedp.neural_discovery.architectures import MLPResNet
from apedp.neural_discovery.config import NeuralDiscoveryConfig, TrainingConfig
from apedp.neural_discovery.training import NeuralDiscoveryTrainer
from apedp.reproducibility import dataset_sha256, seed_everything, write_provenance_artifact
from apedp.symbolic_regression import RegressionConfig, SymbolicRegressor
from benchmarks.known_truth_challenge import run_challenge_repeats


def test_seed_contract_is_repeatable_and_in_default_config():
    seed_everything(17)
    first = (random.random(), np.random.random())
    seed_everything(17)
    assert (random.random(), np.random.random()) == first

    config = NeuralDiscoveryConfig()
    assert config.seed == 42
    assert config.deterministic is True
    training_config = TrainingConfig.from_discovery_config(config)
    assert training_config.seed == config.seed
    assert training_config.deterministic is config.deterministic


def test_seed_contract_rejects_invalid_values():
    with pytest.raises(ValueError, match="seed must be"):
        seed_everything(-1)


def test_symbolic_regression_restarts_from_its_configured_seed():
    engine = DimensionalAssuranceEngine()
    engine.register_variable("x", "dimensionless")
    config = RegressionConfig(population_size=10, generations=2, seed=11)
    regressor = SymbolicRegressor(engine, ["x"], "dimensionless", config=config)
    x = np.linspace(-1, 1, 20)
    y = x**3

    first_expr, first_history = regressor.fit({"x": x}, y)
    second_expr, second_history = regressor.fit({"x": x}, y)

    assert str(first_expr) == str(second_expr)
    assert [step["best_score"] for step in first_history] == [
        step["best_score"] for step in second_history
    ]


def test_neural_training_persists_provenance_when_log_dir_is_set(tmp_path):
    config = NeuralDiscoveryConfig(
        hidden_dims=(4,),
        num_layers=1,
        max_epochs=1,
        batch_size=4,
        log_dir=str(tmp_path),
        seed=23,
        deterministic=True,
    )
    model = MLPResNet(hidden_dims=config.hidden_dims, num_layers=1, output_dim=1)
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    y = x[:, :1] * 2

    NeuralDiscoveryTrainer(config).train(model, (x, y), verbose=False)

    artifacts = list((tmp_path / "provenance").glob("training-*.json"))
    assert len(artifacts) == 1
    manifest = json.loads(artifacts[0].read_text())
    assert manifest["seed"] == 23
    assert manifest["deterministic"] is True
    assert manifest["dataset_sha256"]
    assert manifest["metrics"]["epochs"] == 1


def test_dataset_hash_and_provenance_artifact_are_stable(tmp_path):
    dataset = {"x": np.arange(6, dtype=np.float32).reshape(3, 2), "y": np.arange(3)}
    assert dataset_sha256(dataset) == dataset_sha256(dataset)
    assert dataset_sha256(dataset) != dataset_sha256(
        {"x": dataset["x"].reshape(2, 3), "y": dataset["y"]}
    )
    frame = pd.DataFrame({"x": np.arange(200)})
    changed_frame = frame.copy()
    changed_frame.iloc[-1, 0] = -1
    assert dataset_sha256(frame) != dataset_sha256(changed_frame)

    config = NeuralDiscoveryConfig(seed=9, deterministic=True)
    path = tmp_path / "run" / "provenance.json"
    payload = write_provenance_artifact(
        path,
        dataset=dataset,
        config=config,
        seed=config.seed,
        deterministic=config.deterministic,
        metrics={"r2": 1.0, "diagnostic": float("nan")},
        results={"formula": "y = x**2"},
    )

    loaded = json.loads(path.read_text())
    assert loaded == json.loads(json.dumps(payload))
    assert loaded["schema_version"] == "1.0"
    assert loaded["dataset_sha256"] == dataset_sha256(dataset)
    assert loaded["config"]["seed"] == 9
    assert loaded["deterministic"] is True
    assert loaded["metrics"]["r2"] == 1.0
    assert loaded["metrics"]["diagnostic"] is None
    assert "git_commit" in loaded


def test_repeated_known_truth_run_writes_provenance_sidecar(tmp_path):
    result_path = tmp_path / "repeat.json"
    results = run_challenge_repeats(
        str(result_path),
        seeds=(3,),
        adjudicated=False,
    )

    provenance = json.loads(
        result_path.with_suffix(result_path.suffix + ".provenance.json").read_text()
    )
    assert provenance["seed"] == 3
    assert provenance["config"]["seeds"] == [3]
    assert provenance["metrics"] == results["rates"]
