"""The learning loop: what it learns from, what it never touches, and that it can be undone (development only)."""
from __future__ import annotations

from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import learning_loop
from apedp_dyn.proposers.structure_memory import Memory
from apedp_mmlm.ledger_learning import parse_law

SEED_LAWS = ["x0*x1", "x0**2 + x1", "exp(-x0)*x1", "x0/(x1 + 1)", "sqrt(x0)*x1", "sin(x0) + x1", "x0 + x1 + x2", "log(x0)*x1"]


def _state() -> learning_loop.LearningState:
    return learning_loop.LearningState(Memory(), SEED_LAWS)


def _record(**changes: Any) -> dict[str, Any]:
    record = {"reported": True, "law": "2.3*x0/(x0 + 0.45)", "vouching": {"in_force": True, "applied": True},
              "measured_range": {"state": "inside"}}
    return {**record, **changes}


def test_it_learns_only_vouched_laws_from_tables_inside_the_measured_range() -> None:
    assert learning_loop.learnable(_record()) is None
    assert learning_loop.learnable(_record(measured_range={"state": "edge"})) is None
    assert "no law" in learning_loop.learnable(_record(reported=False, law=None))
    assert "last checks" in learning_loop.learnable(_record(vouching={"in_force": False, "applied": False}))
    assert "last checks" in learning_loop.learnable(_record(vouching={"in_force": True, "applied": False}))
    assert "outside" in learning_loop.learnable(_record(measured_range={"state": "outside"}))
    assert "unmarked" in learning_loop.learnable(_record(measured_range={}))


def test_learning_changes_both_learners_and_the_journal_says_so() -> None:
    state = _state()
    law = "2.3*x0/(x0 + 0.45) - 0.7*exp(-0.45*x1)"
    tree = parse_law(law)
    assert tree is not None
    before_bits, before_motifs = state.model.bits(tree), dict(state.memory.motifs)
    item = state.learn(law, source="test")
    assert item.key == "00000" and state.memory.laws == 1 and state.memory.motifs != before_motifs
    assert state.model.bits(tree) < before_bits                    # the law costs the model less once learned
    assert state.summary()["learned"] == 1


def test_a_retracted_law_is_unlearned_and_the_rest_kept(tmp_path) -> None:
    state = _state()
    keep = state.learn("1.3*x0*exp(-0.3*x1)")
    drop = state.learn("2.3*x0/(x0 + 0.45)")
    fresh = _state()
    fresh.learn("1.3*x0*exp(-0.3*x1)")
    assert state.retract(drop.key) and not state.retract(drop.key)
    assert state.memory.as_dict() == fresh.memory.as_dict()
    assert state.model.state()["sha256"] == fresh.model.state()["sha256"]
    assert [i.key for i in state.learned()] == [keep.key]
    # the saved journal rebuilds the same state over the same seed
    state.save(tmp_path / "journal.json")
    again = learning_loop.LearningState.replay(tmp_path / "journal.json", Memory(), SEED_LAWS)
    assert again.memory.as_dict() == state.memory.as_dict() and again.summary()["retracted"] == 1


def test_the_route_learns_what_passed_and_records_why_not_otherwise() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (1000, 1))
    y = 2.5 * X[:, 0] ** 2 + 3.0 + 0.0 * rng.standard_normal(1000)
    pool_y = y[700:]
    state = _state()
    out = learning_loop.discover_law({"x0": X[:700, 0]}, y[:700], X[700:], lambda rows: pool_y[rows], state=state,
                                     memory_seconds=2.0, source="test")
    assert out["reported"] and out["learning"]["learned"] and state.summary()["learned"] == 1
    frozen = _state()
    again = learning_loop.discover_law({"x0": X[:700, 0]}, y[:700], X[700:], lambda rows: pool_y[rows], state=frozen,
                                       memory_seconds=2.0, learn=False)
    assert not again["learning"]["learned"] and frozen.summary()["learned"] == 0
    assert again["vouching"]["applied"] and again["measured_range"]["state"] in ("inside", "edge")


def test_it_never_reaches_the_referee_or_the_checks() -> None:
    """The loop's module imports the checks and the route to call them, and nothing that writes thresholds."""
    import inspect

    source = inspect.getsource(learning_loop)
    for frozen_thing in ("canonical_spec", "discovery_checks", "vouching_v2.", "LEVEL", "POWER", "ACCURACY"):
        assert frozen_thing not in source.split('"""', 2)[2], frozen_thing
