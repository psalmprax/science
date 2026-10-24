# Databricks notebook source
"""Collection guards for optional scientific-computing dependencies."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The jetflow-video package ships as a repo sub-tree (packages/jetflow-video),
# not as an installed dependency; make it importable for every test that
# reaches for it (same discipline as benchmarks/run_jetflow_distill_benchmark.py).
_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "packages" / "jetflow-video"
if _PACKAGE_DIR.is_dir() and str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))


def pytest_ignore_collect(collection_path, config) -> bool:
    name = getattr(collection_path, "name", None)
    if callable(name):
        name = collection_path.basename
    if name in {"test_neural_discovery.py", "test_jev_backend.py"}:
        return importlib.util.find_spec("jax") is None
    return False