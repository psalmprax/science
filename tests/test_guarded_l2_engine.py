"""The template engine with the structure guard on by default (the consequence PREREG-REFEREE-STRUCTURE fixed)."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.guarded_l2 import GuardedL2DiscoveryEngine
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_dyn.epistemic.structure_guard import FLAG

ROOT = Path(__file__).resolve().parents[1]


def _outside_the_grammar():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(1.0, 5.0, 600), rng.uniform(1.0, 5.0, 600)
    y = a * np.exp(-0.37 * b) + 0.3 * np.sin(2.0 * a)      # nothing the depth-1 grammar holds equals this
    return {"a": a, "b": b}, y, {"a": Dimension(), "b": Dimension()}


def test_the_guard_is_on_by_default_and_can_be_turned_off():
    assert inspect.signature(GuardedL2DiscoveryEngine.discover).parameters["structure_rule"].default is True
    assert inspect.signature(L2DiscoveryEngine.discover).parameters["structure_rule"].default is False
    X, y, dims = _outside_the_grammar()
    options = {"operators": ("+", "-", "*", "/"), "max_depth": 1, "max_complexity": 2}
    guarded = GuardedL2DiscoveryEngine(**options).discover(X=X, Y=y, dimensions=dims, target_dimension=Dimension(), seed=0)
    frozen = L2DiscoveryEngine(**options).discover(X=X, Y=y, dimensions=dims, target_dimension=Dimension(), seed=0)
    switched_off = GuardedL2DiscoveryEngine(**options).discover(X=X, Y=y, dimensions=dims, target_dimension=Dimension(),
                                                              seed=0, structure_rule=False)
    assert guarded.adjudication.status is not EpistemicStatus.UNIQUE_SUPPORT and guarded.adjudication.promoted is None
    assert switched_off.adjudication.status is frozen.adjudication.status          # exactly the frozen engine
    assert switched_off.adjudication.reason == frozen.adjudication.reason
    if frozen.adjudication.status is EpistemicStatus.UNIQUE_SUPPORT:              # the promotion the guard exists to stop
        assert FLAG in guarded.adjudication.diagnostic_flags


def test_a_true_law_is_still_promoted():
    rng = np.random.default_rng(0)
    m, v = rng.uniform(0.5, 5.0, 400), rng.uniform(1.0, 10.0, 400)
    result = GuardedL2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"), max_depth=2, max_complexity=5).discover(
        X={"m": m, "v": v}, Y=0.5 * m * v**2 + rng.normal(0.0, 0.05, 400),
        dimensions={"m": Dimension(m=1), "v": Dimension(l=1, t=-1)}, target_dimension=Dimension(m=1, l=2, t=-2), seed=1)
    assert result.adjudication.status is EpistemicStatus.UNIQUE_SUPPORT, result.adjudication.reason


def test_the_frozen_engine_file_was_not_edited_to_change_the_default():
    digest = json.loads((ROOT / "preregistration" / "DIGEST-REFEREE-STRUCTURE.json").read_text())["sha256"]
    path = "apedp_dyn/discovery_l2/l2_engine.py"
    assert digest[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_the_physics_page_uses_the_guarded_engine_and_the_registry_page_does_not():
    """Independent noise on the physics page; finite differences along one trajectory on the registry page."""
    physics = (ROOT / "apedp_dyn" / "ui" / "physics_endpoint.py").read_text(encoding="utf-8")
    registry = (ROOT / "apedp_dyn" / "ui" / "registry_endpoint.py").read_text(encoding="utf-8")
    assert "GuardedL2DiscoveryEngine(" in physics and " L2DiscoveryEngine(" not in physics
    assert "GuardedL2DiscoveryEngine" not in registry
