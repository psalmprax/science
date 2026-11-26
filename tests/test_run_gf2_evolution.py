"""The construction-evolution loop: the judge checks every entry, refuses what breaks the rules, and the cap holds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_gf2_evolution as script

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the jail is Linux-only")


def test_answers_that_are_not_products_are_refused() -> None:
    assert isinstance(script.read_decomposition("x", 4), str)
    assert isinstance(script.read_decomposition([[1, 2]], 4), str)
    assert isinstance(script.read_decomposition([[1, 2, 99]], 4), str)
    assert isinstance(script.read_decomposition([[1, True, 2]], 4), str)
    assert script.read_decomposition([[1, 2, 3]], 4) == [(1, 2, 3)]


def test_the_seed_runs_in_the_jail_and_is_exact() -> None:
    task = script.ConstructionTask((5,), seconds=2.0)
    ev = task.evaluate(script.SEED)
    assert ev.ran and ev.details["counts"]["5"] is not None and 13 <= ev.details["counts"]["5"] <= 25


def test_a_wrong_decomposition_and_a_forbidden_one_count_for_nothing() -> None:
    task = script.ConstructionTask((5,), seconds=2.0)
    wrong = "def construct(n, modulus, seconds):\n    return [[1, 1, 1]]\n"
    ev = task.evaluate(wrong)
    assert not ev.ran and "do not give the multiplication" in ev.feedback
    forbidden = "import os\ndef construct(n, modulus, seconds):\n    return []\n"
    assert not task.evaluate(forbidden).ran


def test_the_cap_on_model_calls_holds() -> None:
    capped = script.CappedClient(lambda prompt: "reply", 2)
    capped("a")
    capped("b")
    with pytest.raises(RuntimeError):
        capped("c")
    assert capped.calls == 2


def test_the_loop_with_the_stand_in_writes_its_ledger(tmp_path: Path) -> None:
    assert script.main(["demo", "--out", str(tmp_path), "--field-seconds", "2", "--fields", "5"]) == 0
    record = json.loads((tmp_path / "gf2_evolution.json").read_text())
    assert record["model_calls"] == 2 and record["versions"] == 3 and record["versions_that_ran"] == 2
    assert (tmp_path / "ledger.jsonl").exists() and (tmp_path / "champion.py").exists()
