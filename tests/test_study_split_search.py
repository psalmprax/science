"""The study of the split search: what is scored, on which rows, and what the summary counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_wider_conditions as wider
import study_split_search as study
import study_table_split as shapes

from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.proposers.composed import formula_proposal

JOB = {"index": 3, "law": 0, "text": "2.0*x0*log(1 + x1) + 3.0*sin(x2)", "inputs": 3, "noise": 0.0}


class _Templates(ProposerBase):
    """A search of one input that tries a few forms with a scale and an offset, best fit first."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    @property
    def proposer_id(self) -> str:
        return "templates"

    def fit_and_propose(self, train_data, target_col=None, **kwargs):
        (name,) = [c for c in train_data.columns if c != target_col]
        x, y = train_data[name].to_numpy(dtype=float), train_data[target_col].to_numpy(dtype=float)
        out = []
        for template in ("{x}", "{x}**2", "log(1 + {x})", "sin({x})"):
            text = template.format(x=name)
            value = sp.lambdify(sp.Symbol(name), sp.sympify(text), "numpy")(x)
            (a, b), *_ = np.linalg.lstsq(np.column_stack([value, np.ones(len(x))]), y, rcond=None)
            out.append((float(np.mean((a * value + b - y) ** 2)), f"{float(a)!r}*{text} + {float(b)!r}"))
        return [formula_proposal(self.proposer_id, text, [name]) for _, text in sorted(out)]


def test_a_table_is_searched_by_its_parts_on_the_rows_a_proposer_has_and_scored_on_rows_without_noise():
    row = study.one(JOB, "second", _Templates, seconds=12.0)
    assert row["error"] is None and row["parts"] == 3 and row["widest"] == 1 and row["stand_ins"] == 0
    assert row["proposals"] and row["a_proposal_accurate"] and row["a_proposal_the_law"] and row["first_the_law"]
    assert all(set(p) >= {"formula", "r2", "accurate", "recovered"} for p in row["proposals"])
    assert study.DEV_ROWS == 700 and study.DEV_ROWS < wider.ROWS


def test_the_rows_without_noise_are_the_ones_the_tables_runner_scores_on():
    X, clean, law, names = study.clean_rows(JOB, "second")
    assert X.shape == (wider.TEST_ROWS, 3) and names == ["x0", "x1", "x2"]
    assert np.allclose(clean, sp.lambdify([sp.Symbol(n) for n in names], law, "numpy")(*X.T))
    given, _ = shapes.table(JOB, "second")
    assert not {tuple(r) for r in np.round(X, 9)} & {tuple(r) for r in np.round(given, 9)}


def test_a_proposal_is_scored_as_accurate_and_as_the_law_separately():
    X, clean, law, names = study.clean_rows(JOB, "second")
    assert study.scored("1.9*x0*log(x1 + 1.0) + 2.6*sin(x2) + 0.1", law, names, X, clean)["recovered"]
    near = study.scored("2.0*x0*log(1 + x1) + 3.0*sin(x2) + 1e-3*x2**2", law, names, X, clean)
    assert near["accurate"] and not near["recovered"]
    far = study.scored("x0 + x1", law, names, X, clean)
    assert not far["accurate"] and not far["recovered"]
    assert "scoring_error" in study.scored("x0 +* x1", law, names, X, clean)


def test_the_summary_counts_tables_and_sets_the_route_beside_them():
    rows = [{"index": 0, "inputs": 4, "noise": 0.01, "parts": 4, "proposals": [1], "a_proposal_accurate": True,
             "a_proposal_the_law": True, "first_accurate": True, "first_the_law": False, "error": None, "stand_ins": 1, "seconds": 50.0},
            {"index": 1, "inputs": 4, "noise": 0.01, "parts": 1, "proposals": [], "a_proposal_accurate": False,
             "a_proposal_the_law": False, "first_accurate": False, "first_the_law": False, "error": "x", "stand_ins": 0, "seconds": 70.0}]
    route = {0: {"reported": True, "accurate": True, "recovered": False}, 1: {"reported": False, "accurate": False, "recovered": False}}
    got = study.summarise(rows, route)
    block = got["cells"]["4 inputs, noise 0.01"]
    assert block["tables"] == 2 and block["split"] == 1 and block["proposed"] == 1 and block["a_proposal_the_law"] == 1
    assert block["first_the_law"] == 0 and block["errors"] == 1 and block["parts_stood_in_for"] == 1 and block["median_seconds"] == 60.0
    assert block["the_route_in_force"] == {"tables": 2, "reported": 1, "accurate": 1, "the_law": 0}
    assert got["all"]["tables"] == 2 and got["by_inputs"]["4"]["tables"] == 2
    assert "the_route_in_force" not in study.summarise(rows)["all"]


def test_shards_are_put_together_into_one_record(tmp_path):
    row = study.one(JOB, "second", _Templates, seconds=12.0)
    (tmp_path / "shard_0.json").write_text(json.dumps({"set": "second", "seconds": 12.0, "rows": [row]}))
    assert study.main(["collect", "--set", "second", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "split_search.json").read_text())
    assert record["development_not_a_result"] and record["tables_read"] == 1 and record["tables_of_the_set"] == 120
    assert record["settings"]["rows_given"] == 700 and record["summary"]["all"]["a_proposal_the_law"] == 1


def test_the_other_generator_draws_acceptable_laws_with_blocks_outside_the_block_search():
    # a seed of its own: the run's laws (OTHER_SEED, THIRD_SEED) are drawn by no test
    rng = np.random.default_rng(314159)
    texts = [str(study.draw_other(rng, 3)) for _ in range(200)]
    assert any("atan" in t for t in texts) and any("exp(-" in t and "/x" in t for t in texts)
    assert study.OTHER_SEED not in (314159,) and study.THIRD_SEED != shapes.SECOND_SEED
    assert set(study.SETS) == {"first", "second", "third", "other"} and set(study.INNERS) == {"pysr", "blocks"}


def test_compare_counts_tables_table_by_table():
    def record(flags, inner):
        rows = [{"index": i, "inputs": 3 + (i % 2), "a_proposal_the_law": law, "a_proposal_accurate": acc}
                for i, (law, acc) in enumerate(flags)]
        return {"set": "third", "inner": inner, "rows": rows}
    a = record([(False, True), (False, False), (True, True), (False, True)], "pysr")
    b = record([(True, True), (True, True), (True, True), (False, False)], "blocks")
    got = study.compare(a, b)
    assert got["tables"] == 4
    assert got["a_proposal_the_law"] == {**got["a_proposal_the_law"], "a": 1, "b": 3, "both": 1, "only_a": 0, "only_b": 2}
    assert got["a_proposal_accurate"]["only_a"] == 1 and got["a_proposal_accurate"]["only_b"] == 1
    assert got["a_proposal_the_law"]["sign_test_p"] == 0.5
    import pytest
    with pytest.raises(ValueError):
        study.compare(a, {**b, "set": "other"})
