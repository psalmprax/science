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
    assert rep.fpr_blind == 0.5  # 1 FP (C) / 2 promoted
    assert rep.car_blind == 0.5  # 1 correct abstention (D) / 2 unwarranted
    assert rep.vrr_blind == 0.5  # 1 retained (A) / 2 warranted


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
    assert rep.n_promoted == 1 and rep.fpr_blind == 0.0
    assert rep.fpr_upper95 is not None and abs(rep.fpr_upper95 - ci_upper95_zero_failures(1)) < 1e-12
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
    assert rep.by_domain["dyn"] == {"n": 2.0, "fpr": 0.0, "vrr": 0.5, "coverage": 0.5}
    assert rep.by_domain["fin"] == {"n": 2.0, "fpr": 1.0, "vrr": 0.0, "coverage": 1.0}
