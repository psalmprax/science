"""Dated predictions: fixed before the data, one outcome each, misses kept, the chain breaks if anything is edited."""
from __future__ import annotations

import json
from datetime import date

import pytest

from apedp_dyn.epistemic.prediction_ledger import LedgerError, PredictionLedger, TrackRecord

D = date(2026, 11, 20)
SHA = "ab" * 32


def _claim(ledger, claim_id="c1", **over):
    args = {"law": "y = 2*x", "predicts": "next month's y from x", "data_source": "https://example.org/series",
            "score": "relative RMSE on the new rows", "direction": "at_most", "bound": 0.1,
            "stated_on": D, "due": date(2026, 12, 20)}
    return ledger.claim(claim_id, **{**args, **over})


def test_a_claim_and_its_outcome_are_recorded_and_the_verdict_comes_from_the_bound(tmp_path):
    ledger = PredictionLedger(tmp_path / "p.jsonl")
    _claim(ledger)
    assert ledger.resolve("c1", observed=0.04, data_sha256=SHA, recorded_on=date(2026, 12, 21))["held"] is True
    _claim(ledger, "c2", direction="at_least", bound=0.5)
    assert ledger.resolve("c2", observed=0.2, data_sha256=SHA, recorded_on=date(2026, 12, 20))["held"] is False
    assert ledger.verify() == 4
    assert ledger.track_record(date(2027, 1, 1)) == TrackRecord(claims=2, held=1, missed=1, pending=0, overdue=0)


def test_an_outcome_cannot_come_before_the_due_date_or_twice_or_without_a_claim(tmp_path):
    ledger = PredictionLedger(tmp_path / "p.jsonl")
    _claim(ledger)
    with pytest.raises(LedgerError, match="due"):
        ledger.resolve("c1", observed=0.0, data_sha256=SHA, recorded_on=date(2026, 12, 19))
    with pytest.raises(LedgerError, match="no claim"):
        ledger.resolve("zz", observed=0.0, data_sha256=SHA, recorded_on=date(2027, 1, 1))
    with pytest.raises(LedgerError, match="SHA-256"):
        ledger.resolve("c1", observed=0.0, data_sha256="abc", recorded_on=date(2027, 1, 1))
    ledger.resolve("c1", observed=0.5, data_sha256=SHA, recorded_on=date(2027, 1, 1))
    with pytest.raises(LedgerError, match="already has its outcome"):
        ledger.resolve("c1", observed=0.0, data_sha256=SHA, recorded_on=date(2027, 1, 2))


def test_a_claim_cannot_be_restated_or_be_due_the_day_it_is_made_or_leave_its_terms_blank(tmp_path):
    ledger = PredictionLedger(tmp_path / "p.jsonl")
    _claim(ledger)
    with pytest.raises(LedgerError, match="cannot be restated"):
        _claim(ledger, bound=10.0)
    with pytest.raises(LedgerError, match="due after"):
        _claim(ledger, "c2", due=D)
    with pytest.raises(LedgerError, match="names its law"):
        _claim(ledger, "c3", score=" ")
    with pytest.raises(LedgerError, match="direction"):
        _claim(ledger, "c4", direction="roughly")


def test_an_unresolved_claim_past_its_date_is_overdue_not_forgotten(tmp_path):
    ledger = PredictionLedger(tmp_path / "p.jsonl")
    _claim(ledger)
    _claim(ledger, "c2", due=date(2027, 6, 1))
    assert ledger.track_record(date(2027, 1, 1)) == TrackRecord(claims=2, held=0, missed=0, pending=1, overdue=1)


@pytest.mark.parametrize("edit", ["bound", "delete", "verdict"])
def test_editing_or_removing_an_entry_breaks_the_chain(tmp_path, edit):
    path = tmp_path / "p.jsonl"
    ledger = PredictionLedger(path)
    _claim(ledger)
    _claim(ledger, "c2")
    ledger.resolve("c1", observed=0.5, data_sha256=SHA, recorded_on=date(2027, 1, 1))
    lines = path.read_text().splitlines()
    if edit == "delete":
        lines = lines[1:]
    else:
        index, key, value = (0, "bound", 9.0) if edit == "bound" else (2, "held", True)
        entry = json.loads(lines[index])
        entry[key] = value
        lines[index] = json.dumps(entry, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        ledger.verify()
    with pytest.raises(LedgerError):
        _claim(ledger, "c9")  # nothing is appended to a ledger that does not verify
