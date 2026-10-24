"""Blind-benchmark protocol pins: hash-guard, example, scoring math."""

from apedp_dyn.sandbox.blind_benchmark import (
    BlindCase,
    ci_upper95_zero_failures,
    example_case_short_window,
    ledger_row,
    score_blind,
    verify_referee,
    wilson_upper_zero,
)


def _stub(promoted: bool):
    return lambda: {"promoted": promoted, "abstained": not promoted}


def test_guard_passes_on_frozen_tree():
    checked = verify_referee()
    assert len(checked) >= 15
    assert len(checked) == 15


def test_guard_detects_mutation_on_any_frozen_file(monkeypatch, tmp_path):
    """Guard must fail loudly on mutation of ANY frozen file, not just the original 3 referee files."""
    import json

    import pytest

    from apedp_dyn.sandbox.blind_benchmark import FROZEN_RECORD
    rec = json.loads(FROZEN_RECORD.read_text())
    tampered_rec = dict(rec)
    tampered_hashes = dict(rec["file_hashes_sha256"])
    target_file = "apedp_dyn/sandbox/grammar_space.py"
    tampered_hashes[target_file] = "0" * 64
    tampered_rec["file_hashes_sha256"] = tampered_hashes

    fake_frozen = tmp_path / "fake_frozen.json"
    fake_frozen.write_text(json.dumps(tampered_rec))

    monkeypatch.setattr("apedp_dyn.sandbox.blind_benchmark.FROZEN_RECORD", fake_frozen)
    with pytest.raises(RuntimeError, match="REFEREE OR FROZEN RECORD CHANGED"):
        verify_referee()


def test_example_case_abstains_and_excluded():
    rep = score_blind([example_case_short_window()])
    assert rep.n_blind == 0
    assert rep.fpr_blind is None and rep.car_blind is None
    row = rep.rows[0]
    assert row["promoted"] is False and row["abstained"] is True


def test_scoring_math():
    cases = [
        BlindCase("A", "d", True, _stub(True)),
        BlindCase("B", "d", True, _stub(False)),
        BlindCase("C", "d", False, _stub(True)),
        BlindCase("D", "d", False, _stub(False)),
    ]
    rep = score_blind(cases)
    assert rep.n_blind == 4 and rep.n_promoted == 2
    assert rep.fpr_blind == 0.5  # 1 FP (C) / 2 unwarranted (C, D)
    assert rep.fdr_blind == 0.5  # 1 FP (C) / 2 promoted
    assert rep.car_blind == 0.5  # 1 correct abstention (D) / 2 unwarranted
    assert rep.vrr_blind == 0.5  # 1 retained (A) / 2 warranted


def test_fpr_and_fdr_use_distinct_denominators():
    """FPR divides by negatives; FDR divides by promotions. Nobody may conflate them."""
    cases = [
        BlindCase("W1", "d", True, _stub(True)),
        BlindCase("W2", "d", True, _stub(True)),
        BlindCase("W3", "d", True, _stub(False)),
        BlindCase("N1", "d", False, _stub(True)),
        BlindCase("N2", "d", False, _stub(False)),
        BlindCase("N3", "d", False, _stub(False)),
        BlindCase("N4", "d", False, _stub(False)),
    ]
    rep = score_blind(cases)
    assert rep.n_promoted == 3
    assert rep.fpr_blind == 1 / 4  # 1 FP / 4 negatives
    assert rep.fdr_blind == 1 / 3  # 1 FP / 3 promotions
    assert rep.by_domain["d"]["fpr"] == 1 / 4
    assert rep.by_domain["d"]["fdr"] == 1 / 3


def test_ci_bound_table():
    assert ci_upper95_zero_failures(0) is None
    # NOTE: reviewer's table row for n=14 (22.2%) does not match its own
    # formula 1-0.05**(1/n), which gives 19.3%. Formula value pinned here.
    assert abs(ci_upper95_zero_failures(14) - 0.193) < 0.005
    assert abs(ci_upper95_zero_failures(100) - 0.030) < 0.005
    assert abs(ci_upper95_zero_failures(200) - 0.015) < 0.005


def test_populations_and_schema():
    cases = [
        BlindCase("V1", "d", True, _stub(True), population="valid",
                  proposer_id="P1", requested_claim_level="C4"),
        BlindCase("U1", "d", False, _stub(False), population="unsafe",
                  proposer_id="P1"),
        BlindCase("A1", "d", False, _stub(False), population="ambiguous",
                  proposer_id="P2", scientific_claim="X causes Y"),
    ]
    rep = score_blind(cases)
    assert rep.populations == {"valid": 1, "unsafe": 1, "ambiguous": 1}
    assert rep.n_promoted == 1 and rep.fpr_blind == 0.0 and rep.fdr_blind == 0.0
    assert rep.fpr_upper95 is not None and abs(rep.fpr_upper95 - ci_upper95_zero_failures(2)) < 1e-12
    assert rep.car_blind == 1.0 and rep.vrr_blind == 1.0


def test_ledger_row_matches_hand_table():
    # Machine-generated; guards against hand-copied CI numbers drifting.
    r70 = ledger_row(70)
    assert abs(r70["wilson"] - 0.0520) < 0.001
    assert abs(r70["exact"] - 0.0419) < 0.001
    assert abs(r70["rule3"] - 0.0429) < 0.001
    r100 = ledger_row(100)
    assert abs(r100["wilson"] - 0.0370) < 0.001
    assert abs(r100["exact"] - 0.0295) < 0.001
    assert r100["rule3"] == 0.03
    assert abs(wilson_upper_zero(14) - 0.2153) < 0.001
    assert wilson_upper_zero(0) is None


def _run3(promoted, abstained=False, rejected=False):
    return lambda: {"promoted": promoted, "abstained": abstained, "rejected": rejected}


def test_frr_coverage_and_domains():
    cases = [
        BlindCase("A", "dyn", True, _run3(True)),
        BlindCase("B", "dyn", True, _run3(False, abstained=True)),
        BlindCase("C", "fin", False, _run3(True)),
        BlindCase("D", "fin", False, _run3(False, rejected=True)),
    ]
    rep = score_blind(cases)
    assert rep.frr_blind == 0.5  # 1/2 warranted unpromoted
    assert rep.coverage_blind == 0.75  # A promoted, C promoted, D rejected
    # FLAW-A13: "dyn" has no negative controls (both cases warranted) -> fpr
    # is UNMEASURED (None), never a certified 0.0.
    assert rep.by_domain["dyn"] == {"n": 2.0, "fpr": None, "fdr": 0.0, "vrr": 0.5, "coverage": 0.5}
    # fin: FPR = 1 FP / 2 negatives; FDR = 1 FP / 1 promotion; no warranted
    # cases -> vrr is None (FLAW-A13), not a certified 0% retention rate.
    assert rep.by_domain["fin"] == {"n": 2.0, "fpr": 0.5, "fdr": 1.0, "vrr": None, "coverage": 1.0}


def test_zero_denominator_domains_report_none_not_certified_zero():
    """FLAW-A13 pin: zero-denominator per-domain metrics report None.

    A domain with zero negative-control cases has NO measured false-positive
    rate; reporting 0.0 falsely certifies a 0% FPR where no negative controls
    exist. None must propagate exactly like the headline metrics (which
    already report None on empty populations).
    """
    cases = [
        BlindCase("WP1", "pos-only", True, _run3(True)),
        BlindCase("UP1", "neg-only", False, _run3(False, rejected=True)),
    ]
    rep = score_blind(cases)
    dom = rep.by_domain
    # pos-only: warranted case promoted; no unwarranted -> fpr None.
    assert dom["pos-only"] == {"n": 1.0, "fpr": None, "fdr": 0.0, "vrr": 1.0, "coverage": 1.0}
    # neg-only: unwarranted case rejected; no warranted -> vrr None, and no
    # promotions at all -> fdr None as well.
    assert dom["neg-only"] == {"n": 1.0, "fpr": 0.0, "fdr": None, "vrr": None, "coverage": 1.0}
