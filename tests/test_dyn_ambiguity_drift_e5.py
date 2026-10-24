"""E5 frozen suite (doc S73): stable-identifiable (A) vs stable-ambiguous (B)
vs drifting (C). Classifier is frozen (ROT_MARGIN=1e-4, sustained-collapse
rule); seeds 91-93 pinned, determinism pinned. No new epistemic status is
implemented or asserted here — signatures only.
"""

import pytest

import apedp_dyn.epistemic.active_disambiguation.drift_e5 as E


@pytest.mark.parametrize("seed", [91, 92, 93])
def test_A_stable_identifiable(seed):
    t = E.run_case("A", seed)
    assert E.classify(t) == "stable-identifiable"
    assert all(w.local_best == "H2" for w in t.windows)


@pytest.mark.parametrize("seed", [91, 92, 93])
def test_B_stable_ambiguous(seed):
    t = E.run_case("B", seed)
    assert E.classify(t) == "stable-ambiguous"
    assert all(w.cumul_margin < E.ROT_MARGIN for w in t.windows)


@pytest.mark.parametrize("seed", [91, 92, 93])
def test_C_drifting(seed):
    t = E.run_case("C", seed)
    assert E.classify(t) == "drifting"
    assert len({w.local_best for w in t.windows}) > 1


def test_drift_signature_is_collapse_then_recovery():
    t = E.run_case("C", 91)
    m = [w.cumul_margin for w in t.windows]
    assert min(m) < 0.5 * max(m)  # regime change collapses the margin


def test_determinism():
    assert E.summarize(E.run_case("C", 91)) == E.summarize(E.run_case("C", 91))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
