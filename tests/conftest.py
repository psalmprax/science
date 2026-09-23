# Databricks notebook source
"""Collection guards for optional scientific-computing dependencies."""
from __future__ import annotations

import importlib.util


def pytest_ignore_collect(collection_path, config) -> bool:
    name = getattr(collection_path, "name", None)
    if callable(name):
        name = collection_path.basename
    if name in {"test_neural_discovery.py", "test_jev_backend.py"}:
        return importlib.util.find_spec("jax") is None
    return False