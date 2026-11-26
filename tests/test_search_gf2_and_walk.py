"""The AND-gate walk: its starts and results are exact as 2-forms and as circuits, and collect checks again."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf2_and_walk as script

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


def test_bilinear_decompositions_written_as_gates_are_exact_both_ways() -> None:
    gates = script.start(5, "schoolbook")
    assert len(gates) == 25 and script.two_forms_exact(gates, 5) and script.circuit_exact(gates, 5)
    assert script.is_bilinear(gates, 5)
    ref = script.start(12, "ref")
    assert len(ref) == 42 and script.verified(ref, 12)


def test_a_changed_gate_is_refused() -> None:
    gates = script.start(4, "schoolbook")
    a, b, c = gates[0]
    broken = [(a, b, c ^ 1), *gates[1:]]
    assert not script.two_forms_exact(broken, 4) and not script.circuit_exact(broken, 4)


def test_a_gate_mixing_both_inputs_is_exact_where_the_2_forms_are() -> None:
    # (x0 + y0)(x1 + y1) has 2-form x0^x1 + x0^y1 + y0^x1 + y0^y1; with (x0)(x1) and (y0)(y1) added the x-x and y-y
    # parts cancel and what is left is the bilinear x0 y1 + y0 x1: a circuit check must accept the identity
    n = 2
    x0, x1, y0, y1 = 1, 2, 4, 8
    gates = [(x0 | y0, x1 | y1, 2), (x0, x1, 2), (y0, y1, 2), (x0, y0, 1), (x1, y1, 3)]
    assert script.two_forms_exact(gates, n) and script.circuit_exact(gates, n) and not script.is_bilinear(gates, n)


@needs_gcc
def test_a_short_walk_from_the_schoolbook_is_exact(tmp_path: Path) -> None:
    binary = script.build(tmp_path / "walk")
    row = script.walk(binary, 5, "schoolbook", seed=2, seconds=2.0, slack=2, plateau=100, reset=1_000_000, target=0)
    assert row["exact"] and row["starting_gates"] == 25 and row["best_and_gates"] < 25 and row["flips"] > 0


def test_collect_checks_kept_gates_again(tmp_path: Path) -> None:
    good = [list(t) for t in script.start(4, "schoolbook")]
    for name, gates in (("good", good), ("bad", good[1:])):
        row = {"field": 4, "start": "schoolbook", "starting_gates": 16, "seed": 0, "slack": 2, "plateau": 100, "reset": 1,
               "seconds": 1.0, "best_and_gates": len(gates), "flips": 0, "plus_transitions": 0, "resets": 0, "exact": True,
               "best_is_bilinear": True, "progress": [], "gates": gates}
        (tmp_path / f"and_{name}.json").write_text(json.dumps(row))
    assert script.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "and_walk.json").read_text())
    assert record["by_field"]["4"]["walks_not_exact"] == 1 and record["by_field"]["4"]["least_and_gates_exactly"] == 16
