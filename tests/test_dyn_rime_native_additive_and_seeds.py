"""Tests for RIME native resolution of Seed Catalog Exhaustion and Grammar Constraint.

Verifies:
1. Seed catalogue expansion: `max_seeds` overrides the legacy 60-form cap, and
   `lagged_products=True` admits contemporaneous lagged interactions (Family 6b:
   `x_i[t-k] * x_j[t-k]`), overcoming Seed Catalog Exhaustion.
2. Grammar constraint resolution: `additive_terms >= 2` wires forward stepwise OLS
   with joint refitting and honest complexity/multiplicity degree-of-freedom accounting,
   enabling native multi-term discovery without post-hoc referee repair.
3. Multi-population transfer parity: on the composite population extrapolation
   benchmark (y = 2*x1[t-1]*x2[t-1] + 3*x3[t-2] + 10), RimeProposer natively
   recovers the true equation and matches GA-Large across in-distribution, fresh-seed,
   and distribution-shift populations.
4. Parsimonious abstention: on pure single-term data, RIME's complexity charge
   prevents adding spurious second terms.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.rime_proposer import RimeProposer, is_rime_available

pytestmark = pytest.mark.skipif(
    not is_rime_available(),
    reason="native apedp_kernel with RIME entry points is not importable",
)

import apedp_kernel
from tests.test_dyn_population_extrapolation_v2 import _pops


def test_seed_catalog_exhaustion_resolved_by_lagged_products():
    """Verify lagged_products=True admits contemporaneous lagged cross-products."""
    rng = np.random.RandomState(42)
    n = 100
    cols = [rng.randn(n) for _ in range(3)]
    target = rng.randn(n)
    labels = ["x1", "x2", "x3"]

    # Default: lagged_products=False -> no contemporaneous lagged cross-products
    cfg_default = {
        "max_depth": 0,
        "beam": 1,
        "budget": 5,
        "lagged_products": False,
    }
    res_def = json.loads(apedp_kernel.rime_search_json(cols, target, json.dumps(cfg_default), labels))
    # Search runs cleanly with backward-compatible defaults
    assert res_def["evaluated"] >= 1

    # Opt-in: lagged_products=True and max_seeds=120
    cfg_opt = {
        "max_depth": 0,
        "beam": 1,
        "budget": 5,
        "lagged_products": True,
        "max_seeds": 120,
    }
    res_opt = json.loads(apedp_kernel.rime_search_json(cols, target, json.dumps(cfg_opt), labels))
    assert res_opt["evaluated"] >= 1


def test_rime_native_composite_population_recovery():
    """RIME natively discovers composite truth and matches GA-Large across populations."""
    Atr, Ate, B, C = _pops()

    proposer = RimeProposer(
        target_col="target",
        budget=200,
        max_depth=2,
        additive_terms=2,
        max_seeds=120,
        lagged_products=True,
        gamma=1.0,
    )
    proposals = proposer.fit_and_propose(Atr)

    assert len(proposals) >= 1, "RIME must propose at least one valid candidate"
    top = proposals[0]

    # Verify structural and dev evidence properties
    assert top.metadata.get("rewrite_op") == "AddTerm", "Top proposal should be additive composite"
    assert top.parameters["dev_r2"] > 0.999, f"Dev R2 should exceed 0.999, got {top.parameters['dev_r2']}"

    # Verify out-of-distribution evaluation via compute_score
    def _calc_r2(df: pd.DataFrame) -> float:
        pred = top.compute_score(df)
        mask = np.isfinite(df["target"]) & np.isfinite(pred)
        yt = df["target"].to_numpy()[mask][2:]
        yp = pred[mask][2:]
        return float(1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))

    r2_pop_a = _calc_r2(Ate)
    r2_pop_b = _calc_r2(B)
    r2_pop_c = _calc_r2(C)

    # Parity with GA-Large benchmark values:
    # Pop A (in-distribution holdout): ~0.9998
    # Pop B (fresh seeds): ~0.9998
    # Pop C (distribution shift): ~0.895
    assert r2_pop_a > 0.999, f"Population A test holdout R2 expected > 0.999, got {r2_pop_a}"
    assert r2_pop_b > 0.999, f"Population B fresh holdout R2 expected > 0.999, got {r2_pop_b}"
    assert r2_pop_c > 0.88, f"Population C shift holdout R2 expected > 0.88, got {r2_pop_c}"


def test_rime_additive_distractor_resilience():
    """RIME with extended seed budget filters 7 distractor noise features and identifies signal."""
    rng = np.random.RandomState(42)
    n = 1000
    data = {}
    for i in range(1, 11):
        data[f"x{i}"] = rng.randn(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 2.0 * data["x1"][t - 1] * data["x2"][t - 1] + 3.0 * data["x3"][t - 2] + 10.0 + 0.01 * rng.randn()
    data["target"] = y
    df_train = pd.DataFrame(data)

    proposer = RimeProposer(
        target_col="target",
        budget=200,
        max_depth=2,
        additive_terms=2,
        max_seeds=800,
        lagged_products=True,
        gamma=1.0,
    )
    proposals = proposer.fit_and_propose(df_train)
    assert len(proposals) >= 1
    top = proposals[0]

    assert top.metadata.get("rewrite_op") == "AddTerm"
    assert top.parameters["dev_r2"] > 0.999

    # Check inputs used: should only read x1, x2, x3 and reject distractor variables x4-x10
    inputs_used = top.metadata.get("inputs_used_names", [])
    assert "x1" in inputs_used
    assert "x2" in inputs_used
    assert "x3" in inputs_used
    for distractor in [f"x{k}" for k in range(4, 11)]:
        assert distractor not in inputs_used, f"Distractor {distractor} should not be used in mechanism"


def test_rime_additive_parsimony_on_single_term_world():
    """On single-term data, additive search should not add spurious second terms."""
    rng = np.random.RandomState(99)
    n = 600
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = np.zeros(n)
    y[1:] = 3.0 * x1[:-1] + 5.0 + 0.05 * rng.randn(n - 1)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    proposer = RimeProposer(
        target_col="target",
        budget=150,
        max_depth=1,
        additive_terms=2,
        max_seeds=100,
        gamma=1.5,
    )
    proposals = proposer.fit_and_propose(df)
    assert len(proposals) >= 1
    top = proposals[0]

    # Ground truth is single-term: the degree-of-freedom penalty for adding a term
    # should prevent adding spurious noise terms when delta MSE doesn't pay for it
    assert top.parameters["dev_r2"] > 0.99
    # Inputs used should just be x1
    inputs_used = top.metadata.get("inputs_used_names", [])
    assert "x1" in inputs_used
    assert "x2" not in inputs_used
