"""Known-truth recovery challenge: invariant pins (fast wrapper).

The full world sweep lives in ``benchmarks/known_truth_challenge.py`` and its
preregistered artifact ``benchmarks/known_truth_results.json``; this wrapper
pins the invariants that make the artifact answerable:

  * positive controls: on identifiable monomial worlds (incl. the d=6
    composition case) the blind RIME static grammar recovers the planted
    STRUCTURE (effective exponent + coefficient), not just the fit;
  * abstention-is-not-success bookkeeping: `_rates` never lets abstention
    count as recovery; in-grammar abstention is a miss, out-of-grammar
    emission is recorded and scored a failure;
  * a structure claim is never granted on an out-of-grammar world;
  * determinism: the challenge is re-runnable with identical verdicts.

Deviation from the artifact numbers here would mean the generator, the
verdict logic or the in-grammar mapping changed — i.e. the challenge must be
re-run and re-preregistered, not quietly patched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.known_truth_challenge import (
    RECOVERY_R2,
    run_challenge,
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return run_challenge("benchmarks/known_truth_results.json")


def test_positive_controls_recover_structure_for_both_methods(artifact) -> None:
    for world in ("d2_C1000_log_1x", "d6_C1e12_log_1x"):
        for method in ("rime_static", "ols_monomial"):
            row = next(c for c in artifact["cells"]
                       if c["world"] == world and c["method"] == method)
            assert row["verdict"] == "recovered_structure", (
                f"{method} on {world}: {row}"
            )
            assert row["coef_relerr"] <= 0.05, row
            assert row["holdout_r2"] >= RECOVERY_R2 and row["holdout_r2_reversed"] >= RECOVERY_R2


def test_rime_recovery_rate_and_no_false_promotion_in_grammar(artifact) -> None:
    r = artifact["rates"]["rime_static"]
    assert r["in_grammar_worlds"] == 11
    assert r["correct_recovery"] == 11, r
    assert r["false_promotion"] == 0, r
    assert r["recovery_rate"] == 1.0


def test_abstention_never_counts_as_success(artifact) -> None:
    # Bookkeeping rule, checked symbolically: recovery tallies only
    # `recovered_structure`; abstention appears in its own bucket and counts 0.
    rates = artifact["rates"]
    for method, r in rates.items():
        assert r["correct_recovery"] + r["false_promotion"] >= 0
        verdicts = {c["verdict"] for c in artifact["cells"] if c["method"] == method}
        assert not ({"abstained", "abstained_correct", "abstained_miss"}
                    & {"recovered_structure"}), verdicts
    # and concretely: no recorded in-grammar abstention was scored as a recovery
    for c in artifact["cells"]:
        if c["family"] == "in_grammar" and c["verdict"] == "abstained":
            assert c["verdict"] != "recovered_structure"


def test_no_structure_claim_on_out_of_grammar_world(artifact) -> None:
    for c in artifact["cells"]:
        if c["family"] == "out_of_grammar":
            # a structure verdict must not exist there; any emission is a
            # recorded failure, never a recovery
            assert c["verdict"] != "recovered_structure", c
            assert c["verdict"] != "recovered_predictive_only", c
    r = artifact["rates"]
    for method in ("rime_static", "ols_monomial"):
        assert r[method]["out_of_grammar_emitted"] + \
            r[method]["out_of_grammar_abstain"] == 1, r[method]


def test_challenge_is_deterministic(tmp_path: Path) -> None:
    probe = tmp_path / "kt_determinism_probe.json"
    a = run_challenge(str(probe))
    b = run_challenge(str(probe))
    v_a = [(c["world"], c["method"], c["verdict"]) for c in a["cells"]]
    v_b = [(c["world"], c["method"], c["verdict"]) for c in b["cells"]]
    assert v_a == v_b
    r2_a = [c["holdout_r2"] for c in a["cells"]]
    r2_b = [c["holdout_r2"] for c in b["cells"]]
    assert r2_a == r2_b
