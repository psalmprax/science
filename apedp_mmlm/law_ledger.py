"""MMLM-L learning from the trial ledger: every referee-confirmed law, once, under its domain.

Status: experimental

:mod:`apedp_mmlm.ledger_learning` keeps one copy of the earlier equation model
per domain. The law model (:mod:`apedp_mmlm.law_language`) carries domains
itself, so one model is fed here: ``sync`` reads the ledger, calls ``observe``
for each new confirmation (the same test of what counts as one,
``ledger_learning.is_confirmation``) and ``retract`` for each retraction of a law
it learned. Records are remembered by candidate and timestamp, so calling
``sync`` again learns nothing twice. It reads the ledger and never writes to it.
"""
from __future__ import annotations

from typing import Any

from apedp_mmlm.equation_language import Tree
from apedp_mmlm.law_language import LawLanguageModel
from apedp_mmlm.ledger_learning import is_confirmation, parse_law


class LawLedgerLearner:
    def __init__(self, model: LawLanguageModel, default_domain: str | None = None) -> None:
        self.model = model
        self.default_domain = default_domain
        self._learned: dict[str, tuple[str | None, Tree]] = {}  # candidate_id -> (domain, tree)
        self._seen: set[str] = set()

    def sync(self, ledger: Any) -> dict[str, int]:
        """Learn every new confirmation and apply every new retraction; returns what was done."""
        done = {"learned": 0, "retracted": 0, "unparseable": 0}
        for record in ledger.get_records():
            key = f"{record.candidate_id}|{record.timestamp}"
            if key in self._seen:
                continue
            self._seen.add(key)
            target = record.metadata.get("retracts")
            if record.status == "retracted" and target in self._learned:
                domain, tree = self._learned.pop(target)
                done["retracted"] += int(self.model.retract(tree, domain))
                continue
            if not is_confirmation(record) or record.candidate_id in self._learned:
                continue
            tree = parse_law(record.formula)
            if tree is None:
                done["unparseable"] += 1
                continue
            label = record.metadata.get("domain") or self.default_domain
            domain = str(label) if label is not None else None
            self.model.observe(tree, domain)
            self._learned[record.candidate_id] = (domain, tree)
            done["learned"] += 1
        return done

    def state(self) -> dict[str, Any]:
        return {"model": self.model.state(), "laws_learned_from_ledger": len(self._learned)}
