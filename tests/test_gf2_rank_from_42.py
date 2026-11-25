"""The baseline with no language model on GF(2^12): the existing search from the verified 42, and what its record says."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import search_gf2_mul_rank_from_42 as baseline


def test_it_starts_from_the_42_products_and_the_twelve_outputs():
    atoms, slice_masks, degree = baseline.start()
    assert len(atoms) == 42 and len(set(atoms)) == 42 and len(slice_masks) == 12 and degree == 12
    assert baseline.TARGET == 41


def test_no_product_of_the_42_can_be_dropped_and_no_two_replaced_by_one():
    assert baseline.deflation()["products_dropped"] == 0
    row = baseline.rewrite(2)
    assert row["sets_tried"] == row["sets_of_products_in_the_family"] == 861
    assert row["every_set_tried"] and not row["found"]


def test_the_summary_says_how_far_the_family_was_exhausted_and_never_calls_the_target_met_without_a_shorter_decomposition():
    rows = [{"move": "deflation", "products_dropped": 0},
            {"move": "2 -> 1 rewrite", "k": 2, "sets_tried": 861, "every_set_tried": True, "found": False},
            {"move": "3 -> 2 rewrite", "k": 3, "sets_tried": 500, "every_set_tried": False, "found": False}]
    got = baseline.summarise(rows)
    assert got["rewrites_exhausted_for_k"] == [2] and got["sets_tried_in_all"] == 1361
    assert got["fewest_products_reached"] == 42 and not got["target_met"] and not got["a_rewrite_was_found"]
    found = baseline.summarise([*rows, {"move": "4 -> 3 rewrite", "k": 4, "sets_tried": 9, "every_set_tried": False, "found": True,
                                        "products_after": 41, "reconstruction_exact": True}])
    assert found["target_met"] and found["fewest_products_reached"] == 41
    unverified = baseline.summarise([{"move": "4 -> 3 rewrite", "k": 4, "sets_tried": 9, "every_set_tried": False, "found": True,
                                      "products_after": 41, "reconstruction_exact": False}])
    assert not unverified["target_met"] and unverified["fewest_products_reached"] == 42


def test_the_moves_are_put_together_into_one_record(tmp_path):
    assert baseline.main(["run", "--deflation", "--k", "2", "--out", str(tmp_path)]) == 0
    assert baseline.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "rank_search_from_42.json").read_text())
    assert record["development_not_a_result"] and len(record["moves"]) == 2
    assert record["summary"]["rewrites_exhausted_for_k"] == [2] and "this family" in record["what"]
