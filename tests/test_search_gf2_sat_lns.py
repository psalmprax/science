"""The SAT neighbourhood search: its solver finds products that exist, refuses what cannot be, and collect checks again."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf2_sat_lns as script

pytest.importorskip("pycryptosat")


def test_two_products_that_add_to_one_are_found_as_one() -> None:
    # (a, b, c1) + (a, b, c2) = (a, b, c1 + c2): a residual of rank one written with two products
    residual = script.tensor_of([(5, 3, 6), (5, 3, 9)], 4)
    outcome, found = script.solve(residual, 1, 4, symmetric=False, limit=30)
    assert outcome == "sat" and script.tensor_of(found, 4) == residual and len(found) == 1


def test_a_tensor_of_rank_two_has_no_single_product() -> None:
    residual = script.tensor_of([(1, 1, 1), (2, 2, 2)], 2)
    assert script.solve(residual, 1, 2, symmetric=False, limit=30)[0] == "unsat"
    assert script.solve(residual, 2, 2, symmetric=True, limit=30)[0] == "sat"


def test_a_redundant_decomposition_is_shortened_back_and_checked() -> None:
    base = script.start(6, "tower:2:0")
    a, b, c = base[0]
    split = [(a, b, c & 0b111), (a, b, c & ~0b111 & 0x3F), *base[1:]]
    assert len(split) == 16 and script.verified(split, 6)
    residual = script.tensor_of(split[:2], 6)
    outcome, found = script.solve(residual, 1, 6, symmetric=True, limit=30)
    assert outcome == "sat" and script.verified(found + split[2:], 6)


def test_collect_checks_kept_decompositions_again(tmp_path: Path) -> None:
    good = [list(t) for t in script.start(6, "tower:2:0")]
    for name, triples, best in (("good", good, 15), ("bad", good[1:], 14)):
        run = {"field": 6, "start": "x", "kind": "random", "symmetric": True, "seed": 0, "limit": 1, "starting_products": 16,
               "best": best, "trials": [{"trial": 0, "k": 2, "outcome": "sat", "seconds": 0.1}], "best_triples": triples}
        (tmp_path / f"lns_{name}.json").write_text(json.dumps(run))
    assert script.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "sat_lns.json").read_text())
    assert record["by_field"]["6"]["best_reached_exactly"] == 15 and not record["target_met"]
