"""Contrast resolution after a referee abstention, and the ladder that climbs only on ``widen_search``."""
from __future__ import annotations

import numpy as np

from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_referee.trial_ledger import CandidateTrialLedger

NOISE = 0.02


def _world(truth, dims, seed=0, n=120):
    rng = np.random.default_rng(seed)
    names = list(dims)
    X = {k: rng.uniform(1, 10, n) for k in names}
    dev = np.column_stack([X[k] for k in names])
    y = truth(dev) * (1 + NOISE * rng.standard_normal(n))
    pool = rng.uniform(1, 10, (4000, len(names)))
    reads: list[int] = []

    def measure(rows):
        reads.append(len(rows))
        return truth(pool[rows]) * (1 + NOISE * rng.standard_normal(len(rows)))

    return X, y, pool, measure, NOISE * float(np.mean(np.abs(y))), reads


def test_a_law_needing_a_dimensional_constant_climbs_one_rung_and_is_confirmed():
    dims = {"mass": Dimension(m=1), "span": Dimension(l=1)}
    X, y, pool, measure, sigma, _ = _world(lambda A: 9.81 * A[:, 0], dims)
    ledger = CandidateTrialLedger()
    rounds = cr.climb(X, y, dims, Dimension(m=1, l=1, t=-2), pool, measure, ledger=ledger, search_id="t1", noise_sigma=sigma)
    assert [(r["rung"], r["action"]) for r in rounds] == [("strict", "widen_search"), ("unit_coefficients", "confirmed")]
    assert rounds[-1]["law"] == "mass"
    assert ledger.trial_count == 2 and len({r.metadata["search_id"] for r in ledger.get_records()}) == 1
    assert ledger.effective_trial_count > 2  # every expression generated on both rungs is charged


def test_a_law_outside_every_rung_ends_in_widen_search_not_in_a_wrong_law():
    dims = {"u": Dimension(), "v": Dimension()}
    X, y, pool, measure, sigma, _ = _world(lambda A: 5 * np.exp(-A[:, 0] / 3), dims)
    rounds = cr.climb(X, y, dims, Dimension(), pool, measure, ledger=CandidateTrialLedger(), search_id="t2", noise_sigma=sigma)
    assert [r["rung"] for r in rounds] == ["strict", "unit_coefficients", "superposition"]
    assert [r["action"] for r in rounds] == ["widen_search"] * 3
    assert all(r["law"] is None for r in rounds)


def test_an_abstention_is_resolved_to_the_true_law_or_a_class_holding_it():
    """3 u^2 + 2 w: the referee abstains on 120 rows; contrast never names a law without the second term."""
    dims = {"u": Dimension(l=1), "w": Dimension(l=2), "z": Dimension(t=1)}
    for seed in range(4):
        X, y, pool, measure, sigma, reads = _world(lambda A: 3 * A[:, 0] ** 2 + 2 * A[:, 1], dims, seed)
        record = cr.climb(X, y, dims, Dimension(l=2), pool, measure, ledger=CandidateTrialLedger(),
                          search_id=f"t3-{seed}", noise_sigma=sigma)[-1]
        assert record["action"] in ("confirmed", "equivalent_class"), record
        assert "u**2" in record["law"] and "w" in record["law"], record
        assert record["rows_read"] == sum(reads) and record["design_sha256"]


def test_the_referees_verdict_is_reported_untouched():
    from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine

    dims = {"u": Dimension(l=1), "w": Dimension(t=1)}
    X, y, pool, measure, sigma, _ = _world(lambda A: 2.0 * A[:, 0] / A[:, 1], dims)
    result = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow")).discover(
        X=X, Y=y, dimensions=dims, target_dimension=Dimension(l=1, t=-1), target_name="y", noise_sigma=sigma, seed=0)
    before = (result.adjudication.status, result.adjudication.promoted)
    resolution = cr.resolve(result, list(dims), np.column_stack([X[k] for k in dims]), y, pool, measure)
    assert (result.adjudication.status, result.adjudication.promoted) == before
    assert resolution.referee_status == before[0].value
    # With these units u/w is the only expression of the right dimension, so nothing challenges it.
    assert resolution.action in ("confirmed", "unchallenged") and resolution.law == "u/w"
    assert cr.referee_abstained(result) is (before[1] is None)


def test_the_climb_feeds_and_uses_the_refusal_memory():
    """After two datasets of a domain needed the second rung, a third starts there."""
    from apedp_dyn.discovery_l2 import refusal_memory

    dims = {"mass": Dimension(m=1), "span": Dimension(l=1)}
    ledger = CandidateTrialLedger()
    for k, name in enumerate(("d1", "d2", "d3")):
        X, y, pool, measure, sigma, _ = _world(lambda A: 9.81 * A[:, 0], dims, seed=k)
        rounds = cr.climb(X, y, dims, Dimension(m=1, l=1, t=-2), pool, measure, ledger=ledger, search_id=f"m-{name}",
                          noise_sigma=sigma, domain="mechanics", dataset_id=name)
        expected = ["strict", "unit_coefficients"] if name != "d3" else ["unit_coefficients"]
        assert [r["rung"] for r in rounds] == expected
        assert rounds[-1]["action"] == "confirmed"
    views = refusal_memory.RefusalMemory(ledger, "mechanics")._views()
    assert all(set(v) <= refusal_memory.READABLE_FIELDS for v in views)  # still verdict fields only
    assert refusal_memory.RefusalMemory(ledger, "optics").rungs_to_try("x") == [0, 1, 2]
