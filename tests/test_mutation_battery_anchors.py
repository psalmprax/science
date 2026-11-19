"""Every registered mutant must still apply to the source it names.

The battery (``scripts/run_mutation_battery.py``) plants each mutant by exact
text and refuses to run when an anchor does not occur exactly once, which is
the right refusal. But the battery takes many minutes and was only ever meant
to run in CI, and CI was not running (``docs/SYSTEM_FLAWS_AUDIT.md``, E-07):
two of its 37 anchors stopped matching on 2026-10-25 and 2026-11-08, and the
committed report went on saying 37 of 37 killed. A formatter pass moved a
third.

This test is the same applicability check without the pins: it reads the
registrations and the files and takes well under a second, so anchor drift is
caught by the ordinary suite, by whoever caused it, at the commit that caused
it. It says nothing about whether a mutant is *killed*; only the battery does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mutation_battery.py"


def _battery() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_mutation_battery_for_anchor_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


MUTANTS = _battery().MUTANTS


def test_the_registry_is_not_empty_and_ids_are_unique() -> None:
    ids = [m["id"] for m in MUTANTS]
    assert len(ids) >= 37
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m["id"])
def test_the_anchor_applies_exactly_once(mutant: dict) -> None:
    source = (ROOT / mutant["file"]).read_text(encoding="utf-8")
    occurrences = source.count(mutant["find"])
    assert occurrences == 1, (
        f"{mutant['id']}: the anchor occurs {occurrences} times in {mutant['file']}. "
        "The source moved; update the mutant's find/replace to the current text "
        "(and re-run the battery), do not delete the mutant."
    )


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m["id"])
def test_the_mutation_changes_the_source(mutant: dict) -> None:
    assert mutant["find"] != mutant["replace"]
    source = (ROOT / mutant["file"]).read_text(encoding="utf-8")
    assert source.replace(mutant["find"], mutant["replace"], 1) != source
