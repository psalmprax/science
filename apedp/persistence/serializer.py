"""Serialization utilities for APEDP models and results."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import sympy as sp


def to_json(obj: Any) -> str:
    """Serialize any APEDP-compatible object to JSON string."""
    return json.dumps(obj, default=_json_default)


def from_json(json_str: str) -> Any:
    """Deserialize a JSON string back to a Python object."""
    return json.loads(json_str)


def serialize_model(expression: sp.Expr, variables: list[str], metadata: dict) -> str:
    """Serialize a symbolic regression model to JSON string."""
    data = {
        "expression": str(expression),
        "sympy_expr": sp.latex(expression),
        "variables": variables,
        "metadata": metadata,
    }
    return json.dumps(data)


def deserialize_model(json_str: str) -> dict:
    """Deserialize a JSON string back to model dict."""
    model: dict = json.loads(json_str)
    return model


def _json_default(obj: Any) -> Any:
    """Default serializer for non-standard types."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.number):
        return float(obj)
    if isinstance(obj, sp.Basic):
        return str(obj)
    if isinstance(obj, sp.Symbol):
        return str(obj)
    return str(obj)