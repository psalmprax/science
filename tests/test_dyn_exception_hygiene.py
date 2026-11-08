"""The blanket-handler ratchet must keep ratcheting.

Two things are checked here, and the second is what makes the first meaningful:

* the tree has not regressed against ``benchmarks/exception_hygiene_baseline.json``;
* the classifier that decides "silent" versus "annotated" actually classifies,
  proven on synthetic sources with known labels. A gate whose classifier always
  answers "annotated" would pass forever while policing nothing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_exception_hygiene.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_exception_hygiene", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `@dataclass(frozen=True)` resolves the defining
    # module through sys.modules, and an unregistered module raises inside
    # dataclasses itself rather than anywhere useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


# --------------------------------------------------------------------- classifier


def test_blanket_handler_without_rationale_is_silent():
    source = """
def f():
    try:
        g()
    except Exception:
        return None
"""
    records = audit.classify_source("m.py", source)
    assert len(records) == 1
    assert records[0].silenced is True


def test_blanket_handler_with_ble001_rationale_is_annotated():
    source = """
def f():
    try:
        g()
    except Exception:  # noqa: BLE001 - the probe must not take the caller down
        return None
"""
    records = audit.classify_source("m.py", source)
    assert [r.silenced for r in records] == [False]


def test_noqa_for_another_rule_is_not_a_rationale():
    """`noqa` alone is a suppression; only BLE001 records a deliberate catch-all."""
    source = """
def f():
    try:
        g()
    except Exception:  # noqa: S110
        return None
"""
    records = audit.classify_source("m.py", source)
    assert records[0].silenced is True
    assert "not BLE001" in records[0].reason


def test_blanket_handler_that_reraises_is_annotated():
    source = """
def f():
    try:
        g()
    except Exception:
        raise
"""
    records = audit.classify_source("m.py", source)
    assert [r.silenced for r in records] == [False]


def test_bare_except_counts_and_typed_handlers_do_not():
    source = """
def f():
    try:
        g()
    except:
        return None

def h():
    try:
        g()
    except (ValueError, TypeError):
        return None

def k():
    try:
        g()
    except Exception as exc:
        raise RuntimeError("wrapped") from exc
"""
    records = audit.classify_source("m.py", source)
    assert [r.silenced for r in records] == [True, False]


def test_tuple_containing_exception_counts_as_blanket():
    source = """
def f():
    try:
        g()
    except (ValueError, Exception):
        return None
"""
    records = audit.classify_source("m.py", source)
    assert records[0].silenced is True


# ------------------------------------------------------------------------- ratchet


def test_baseline_is_present_and_pinned():
    baseline_path = ROOT / "benchmarks" / "exception_hygiene_baseline.json"
    assert baseline_path.is_file(), (
        "benchmarks/exception_hygiene_baseline.json is the ratchet's pin; "
        "regenerate it with scripts/audit_exception_hygiene.py --write-baseline"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema"] == audit.SCHEMA
    assert baseline["totals"]["silent"] == sum(baseline["modules"].values())


def test_no_regression_against_the_pinned_baseline():
    baseline = json.loads((ROOT / "benchmarks" / "exception_hygiene_baseline.json").read_text(encoding="utf-8"))
    records, sizes = audit.scan()
    problems = audit.check(records, sizes, baseline)
    assert not problems, "blanket-handler hygiene regressed:\n" + "\n".join(problems)


def test_check_detects_a_new_silent_handler():
    """The gate must fail on the regression it exists to catch."""
    baseline = {
        "schema": audit.SCHEMA,
        "totals": {"silent": 0, "annotated": 0},
        "modules": {},
        "size_budget": {},
    }
    records = audit.classify_source("m.py", "try:\n    g()\nexcept Exception:\n    pass\n")
    problems = audit.check(records, {"m.py": 4}, baseline)
    assert any("silent blanket handler" in p for p in problems)


def test_size_budget_detects_growth():
    median = {
        "schema": audit.SCHEMA,
        "totals": {"silent": 0, "annotated": 0},
        "modules": {},
        "size_budget": {"big.py": 100},
    }
    problems = audit.check([], {"big.py": 120}, median)
    assert any("ceiling 100" in p for p in problems)


#: Modules that were split under the review threshold. Both must stay under it;
#: the size budget pins everything else that has not been split yet.
SPLIT_MODULES = {"apedp/discovery/_core.py": 2500, "apedp_dyn/ui/server.py": 2000}


@pytest.mark.parametrize(("module", "ceiling"), sorted(SPLIT_MODULES.items()))
def test_split_modules_stay_under_their_ceiling(module, ceiling):
    """The point of a split is that it stays split.

    These two are asserted directly rather than only through the size budget,
    because the budget is a top-25 list: a module that shrank far enough drops
    out of it entirely and would then be unpoliced exactly when it had the most
    room to grow back.
    """
    lines = len((ROOT / module).read_text(encoding="utf-8").splitlines())
    assert lines < ceiling, f"{module} grew back to {lines} lines (ceiling {ceiling})"
