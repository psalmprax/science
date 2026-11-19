"""The discovery split changed file layout, not behaviour.

``apedp/discovery/_core.py`` used to hold the Bayesian evidence machinery at
2,908 lines. That block now lives in ``apedp/discovery/_bayes.py`` and is
re-exported, because every documented import path in this repository goes
through the package (``from apedp.discovery import BayesianRefinementEngine``,
used by ``apedp/pipeline.py``, ``apedp/api/routes/discovery.py``,
``apedp/discovery_admission.py`` and others).

These tests pin the three things a careless move would break: the names still
resolve, the re-exported objects are the *same* objects rather than copies, and
the module that was split is actually smaller.
"""
from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

MOVED_NAMES = (
    "BayesianRefinementEngine",
    "ModelEvidence",
    "STANDARD_RATIONALS",
    "_dim_dict",
    "_snap_coefficient",
)


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_moved_names_still_import_from_the_package(name):
    package = importlib.import_module("apedp.discovery")
    assert hasattr(package, name), f"apedp.discovery.{name} disappeared when the module was split"


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_moved_names_still_import_from_core(name):
    core = importlib.import_module("apedp.discovery._core")
    assert hasattr(core, name)


def test_reexports_are_the_same_objects_not_copies():
    """A duplicated definition would silently diverge; identity proves it cannot."""
    core = importlib.import_module("apedp.discovery._core")
    bayes = importlib.import_module("apedp.discovery._bayes")
    for name in MOVED_NAMES:
        assert getattr(core, name) is getattr(bayes, name)


def test_the_engine_actually_lives_in_bayes_now():
    bayes = importlib.import_module("apedp.discovery._bayes")
    assert bayes.ModelEvidence.__module__ == "apedp.discovery._bayes"
    assert bayes.BayesianRefinementEngine.__module__ == "apedp.discovery._bayes"


def test_standard_rationals_is_the_same_table():
    bayes = importlib.import_module("apedp.discovery._bayes")
    table = dict(bayes.STANDARD_RATIONALS)
    assert table[0.5] == bayes.sp.Rational(1, 2)
    assert len(bayes.STANDARD_RATIONALS) == 11


def test_core_module_shrank_below_the_review_threshold():
    """The point of the split: _core is reviewable, and _bayes holds the maths."""
    core = importlib.import_module("apedp.discovery._core")
    bayes = importlib.import_module("apedp.discovery._bayes")
    core_lines = len(inspect.getsource(core).splitlines())
    bayes_lines = len(inspect.getsource(bayes).splitlines())
    assert core_lines < 2500, f"_core.py is back to {core_lines} lines"
    assert bayes_lines < 700, f"_bayes.py grew to {bayes_lines} lines; split it again"


def test_compute_r_squared_still_usable_through_the_package_path():
    """The one method other modules call through the package import."""
    from apedp.discovery import BayesianRefinementEngine

    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert BayesianRefinementEngine.compute_r_squared(y, y) == pytest.approx(1.0)
