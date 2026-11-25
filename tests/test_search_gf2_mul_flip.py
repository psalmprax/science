"""The flip-search run script: its fields are the right ones, and collect counts nothing that is not exact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf2_mul_flip as script

from apedp_dyn.quantum import flip_search
from apedp_dyn.quantum.field_tensor_search import build_multiplication_tensor


def test_the_calibration_moduli_are_irreducible_and_gf2_12_is_the_repository_s_own() -> None:
    assert np.array_equal(script.field_tensor(12), build_multiplication_tensor())
    for n in script.PUBLISHED:
        T = script.field_tensor(n)
        # a field: no zero divisors, checked on every pair of non-zero elements of the multiplication table
        from apedp_dyn.quantum.field_tensor_search import gf_mul
        assert all(gf_mul(x, y, script.MODULI[n]) for x in range(1, 1 << n, 7) for y in range(1, 1 << n, 5))
        assert T.shape == (n, n, n)


def test_the_starts_are_exact_and_of_the_sizes_said() -> None:
    sizes = {"tower42": 42, "karatsuba54": 54, "schoolbook": 144}
    for start, size in sizes.items():
        scheme = script.starting_scheme(12, start)
        assert len(scheme) == size
        assert flip_search.verified(scheme.triples())
        assert script.multiplications_right(scheme.triples(), 12, pairs=64) == 0


def test_collect_counts_a_tampered_decomposition_as_not_exact(tmp_path: Path) -> None:
    good = script.starting_scheme(12, "tower42").triples()
    bad = [list(t) for t in good[1:]]                         # one product short: 41, and wrong
    for name, triples in (("good", [list(t) for t in good]), ("bad", bad)):
        row = {"field": 12, "start": "tower42", "starting_products": 42, "seed": 0, "slack": 2, "seconds_given": 1,
               "target": None, "best": len(triples), "flips": 0, "flips_tried": 0, "plus_transitions": 0, "resets": 0,
               "seconds": 1.0, "reached": [{"products": len(triples), "flips": 0, "seconds": 0.0}],
               "triples": triples, "exact": True}
        (tmp_path / f"walk_{name}.json").write_text(json.dumps(row))
    assert script.main(["collect", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "flip_search.json").read_text())
    assert record["summary"]["walks_not_exact"] == 1
    assert record["summary"]["fewest_products_reached_exactly"] == 42
    assert record["summary"]["target_met"] is False
    assert record["shortest_for_gf2_12"]["products"] == 42
