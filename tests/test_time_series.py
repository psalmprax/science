# Databricks notebook source
import numpy as np
import pytest

from apedp.time_series import build_lagged_supervised_dataset


def test_lagged_dataset_aligns_past_inputs_to_future_target():
    x_data, y_data, indices = build_lagged_supervised_dataset(
        {"temperature": np.arange(10, dtype=float)},
        np.arange(10, dtype=float),
        lags=[0, 2],
        horizon=3,
    )
    assert indices.tolist() == [2, 3, 4, 5, 6]
    assert x_data["temperature_lag_0"].tolist() == [2, 3, 4, 5, 6]
    assert x_data["temperature_lag_2"].tolist() == [0, 1, 2, 3, 4]
    assert y_data.tolist() == [5, 6, 7, 8, 9]


def test_lagged_dataset_excludes_incomplete_windows():
    x_data, y_data, indices = build_lagged_supervised_dataset(
        {"x": np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0])},
        np.arange(6, dtype=float),
        lags=[0, 1],
        horizon=1,
    )
    assert indices.tolist() == [1, 4]
    assert x_data["x_lag_0"].tolist() == [1.0, 4.0]
    assert x_data["x_lag_1"].tolist() == [0.0, 3.0]
    assert y_data.tolist() == [2.0, 5.0]


def test_lagged_dataset_rejects_invalid_requests():
    with pytest.raises(ValueError, match="horizon"):
        build_lagged_supervised_dataset({"x": [1, 2]}, [1, 2], lags=[0], horizon=0)
