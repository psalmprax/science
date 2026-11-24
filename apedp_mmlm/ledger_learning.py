"""MMLM Track S learning from the project's own trial ledger, one model per domain.

Status: experimental

``DomainLearner.sync(ledger)`` reads the trial ledger and feeds every law the
referee side confirmed to the self-adjusting equation model of that law's
domain. Nothing else in a record is read: not fits, errors or holdout
statistics, only what was confirmed, where, and what the law was.

What counts as confirmed (and nothing else does):

- a ``refusal_memory`` loop's confirmation round that ended ``unique_support``;
- a ``contrast_resolution`` round that ended ``confirmed``.

A proposer-validation round is a search step, not a confirmation, and a law the
model itself made plausible is never fed back unless the referee side confirmed
it on data. A record with status ``retracted`` and ``metadata["retracts"]`` set
to an earlier record's ``candidate_id`` makes the model unlearn that law.

Each domain's model starts from the same seed corpus, so a domain with no
confirmed laws behaves exactly like the seed model; it then moves only on its
own domain's confirmations. ``sync`` is idempotent.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import sympy as sp

from apedp_mmlm import equation_language as el

_ALLOWED = re.compile(r"[A-Za-z0-9_+\-*/(). ,]*")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNCTIONS = {"sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh}


def parse_law(text: str) -> el.Tree | None:
    """A ledger formula as a tree, or None if it is not a plain expression.

    Ledger text is parsed only if it consists of names, numbers and arithmetic;
    anything with attribute access, quotes, brackets or double underscores is
    refused, because parsing evaluates.
    """
    text = text.replace("^", "**").strip()
    if not text or "__" in text or not _ALLOWED.fullmatch(text) or re.search(r"[A-Za-z_)]\s*\.", text):
        return None
    names = {n for n in _NAME.findall(text) if n not in _FUNCTIONS and not re.fullmatch(r"[eE]\d*", n)}
    try:
        expr = sp.sympify(text, locals={**{n: sp.Symbol(n) for n in names}, **_FUNCTIONS})
        return el.to_tree(expr) if isinstance(expr, sp.Expr) and expr.free_symbols else None
    except (sp.SympifyError, SyntaxError, TypeError, ValueError, AttributeError):
        return None


def is_confirmation(record: Any) -> bool:
    meta = record.metadata
    if record.status == "confirmed" and meta.get("resolved_by") == "contrast":
        return True
    return bool(meta.get("stage") == "confirmation" and meta.get("status", record.status) == "unique_support")


class DomainLearner:
    """One self-adjusting equation model per domain, fed from the trial ledger."""

    def __init__(self, seed_trees: Sequence[el.Tree], default_domain: str = "unassigned") -> None:
        self._seed = list(seed_trees)
        self.default_domain = default_domain
        self._models: dict[str, el.SelfAdjustingEquationModel] = {}
        self._learned: dict[str, tuple[str, el.Tree]] = {}   # candidate_id -> (domain, tree)
        self._seen: set[str] = set()

    def model(self, domain: str) -> el.SelfAdjustingEquationModel:
        if domain not in self._models:
            self._models[domain] = el.SelfAdjustingEquationModel().fit(self._seed)
        return self._models[domain]

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
                done["retracted"] += int(self.model(domain).retract(tree))
                continue
            if not is_confirmation(record) or record.candidate_id in self._learned:
                continue
            tree = parse_law(record.formula)
            if tree is None:
                done["unparseable"] += 1
                continue
            domain = str(record.metadata.get("domain") or self.default_domain)
            self.model(domain).observe(tree)
            self._learned[record.candidate_id] = (domain, tree)
            done["learned"] += 1
        return done

    def state(self) -> dict[str, Any]:
        return {domain: model.state() for domain, model in sorted(self._models.items())}
