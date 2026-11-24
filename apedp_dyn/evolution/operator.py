"""The loop in which a model rewrites a piece of the search and a fixed judge scores each version.

Status: experimental

A *task* fixes three things: the contract (which function a candidate has to
define, and what it must do), the judge (``evaluate``: code in, score and
feedback out), and a seed version to start from. This module is the rest:

1. **The lineage.** Every version ever scored is kept, with its code, its
   parent, its score and what it failed on (``Archive``). Nothing is dropped,
   so the number of versions tried is always known.
2. **One variation** (``vary``). The model is shown the contract, the rules of
   the restricted language, the leading versions with their scores, the latest
   refusals with their reasons, the full code of one parent and what the judge
   said about it. It replies with a complete new version in a fenced block.
   What the check refuses, or the jail reports as failing everywhere, goes
   back to the model with the reason, up to ``repairs`` times. Every reply
   that held new code is scored and kept, whether it ran or not.
3. **The run** (``evolve``): variations until the step budget or the deadline,
   each from one of the leading versions.

**What the model cannot do.** It never sees a problem the task holds back, it
does not choose which version wins, and it does not install anything: the loop
writes versions to a ledger and stops. The judge is the task's own code, run
in the caller's process on text and numbers that came out of the jail.

**What is not known.** No real model has run this loop. It is tested with a
scripted stand-in that makes one mistake of each kind (refused code, code that
fails when run) and then an improvement.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from apedp_dyn.evolution import restricted

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Contract:
    """What a candidate has to be, in the words the model is given."""

    function: str               #: the function a candidate has to define
    parameters: int             #: how many positional parameters it takes
    signature: str              #: for example ``propose(X, y, names)``
    description: str            #: what it receives, what it must return, how it is scored
    notes: str = ""             #: anything that helps: how the judge reads the output, what tends to work
    subject: str = "one function of a search for natural laws"   #: what the model is told it is improving


@dataclass
class Evaluation:
    """What the judge says about one version."""

    ran: bool                               #: the code passed the check and ran on at least one problem
    score: tuple[float, ...] = field(default_factory=tuple)   #: larger is better; compared from left to right
    summary: str = ""                       #: one line for the lineage
    feedback: str = ""                      #: what this version failed on, for the model
    error: str = ""                         #: why it was refused or could not run
    details: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    contract: Contract

    def evaluate(self, code: str) -> Evaluation:
        """Check, run and score one version. Never raises for a bad candidate."""


@dataclass
class Version:
    id: int
    parent: int | None
    origin: str                 #: "seed" or "model"
    code: str
    evaluation: Evaluation
    step: int = 0
    repairs: int = 0
    thought: str = ""           #: the model's own line on what it changed

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        e = self.evaluation
        return {"id": self.id, "parent": self.parent, "origin": self.origin, "step": self.step, "repairs": self.repairs,
                "sha256": self.sha256, "ran": e.ran, "score": list(e.score), "summary": e.summary, "error": e.error,
                "thought": self.thought, "feedback": e.feedback, "details": e.details, "code": self.code}


class Archive:
    """Every version scored so far."""

    def __init__(self) -> None:
        self.versions: list[Version] = []

    def add(self, parent: int | None, origin: str, code: str, evaluation: Evaluation, *, step: int = 0, repairs: int = 0,
            thought: str = "") -> Version:
        version = Version(len(self.versions), parent, origin, code, evaluation, step, repairs, thought)
        self.versions.append(version)
        return version

    def ranked(self) -> list[Version]:
        """Versions that ran, best score first; among equals the earlier one."""
        return sorted((v for v in self.versions if v.evaluation.ran), key=lambda v: (tuple(-s for s in v.evaluation.score), v.id))

    def best(self) -> Version | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def known(self, code: str) -> Version | None:
        digest = hashlib.sha256(code.encode()).hexdigest()
        return next((v for v in self.versions if v.sha256 == digest), None)

    def lineage(self, limit: int = 6, failures: int = 3) -> str:
        """The leading versions with what they changed, and the latest ones that were refused or did not run."""
        rows = []
        for v in self.ranked()[:limit]:
            origin = "the seed" if v.parent is None else f"from version {v.parent}"
            change = f' | change: "{v.thought[:140]}"' if v.thought else ""
            rows.append(f"  version {v.id} ({origin}): {v.evaluation.summary}{change}")
        failed = [v for v in self.versions if not v.evaluation.ran]
        if failed:
            rows.append(f"  {len(failed)} versions were refused or did not run; the latest:")
            rows += [f"    version {v.id} (from version {v.parent}): {v.evaluation.error[:160]}" for v in failed[-failures:]]
        return "\n".join(rows or ["  none ran"])

    def write(self, path: Path) -> None:
        path.write_text("".join(json.dumps(v.as_dict(), sort_keys=True) + "\n" for v in self.versions), encoding="utf-8")


def extract_code(reply: str) -> str | None:
    """The code of a reply: its last fenced block, or the whole reply if it has no fence and parses as Python."""
    blocks = _FENCE.findall(reply)
    if blocks:
        return str(blocks[-1]).strip() + "\n"
    text = reply.strip()
    if text.startswith("def ") or "\ndef " in text:
        start = text.find("def ")
        return text[start:].strip() + "\n"
    return None


def thought_of(reply: str) -> str:
    match = re.search(r"^\s*THINK\s*[:=]\s*(.+)$", reply, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip()[:300] if match else ""


def _prompt(task: Task, archive: Archive, parent: Version, steps_left: int) -> str:
    c = task.contract
    notes = f"Notes:\n{c.notes}\n" if c.notes else ""
    return (
        f"You are improving {c.subject}. A fixed judge scores every version; you cannot "
        f"change the judge.\n\nThe function: {c.signature}\n{c.description}\n{notes}\n"
        f"The language: {restricted.describe()}\n\n"
        f"Versions so far, best first:\n{archive.lineage()}\n\n"
        f"The version to improve is version {parent.id}:\n```python\n{parent.code.rstrip()}\n```\n"
        f"What the judge said about it:\n{parent.evaluation.feedback or '  nothing'}\n\n"
        f"Variations left after this one: {steps_left}.\n"
        "Reply with one line starting with THINK: saying what you change and why, then the complete new version "
        f"(the function {c.function} and any helpers) in one ```python block. Change one thing at a time."
    )


def _repair_prompt(task: Task, code: str, reason: str) -> str:
    c = task.contract
    return (
        f"Your version of {c.signature} was not accepted.\nReason: {reason}\n\n"
        f"The language: {restricted.describe()}\n\nYour code:\n```python\n{code.rstrip()}\n```\n"
        "Reply with the complete corrected version in one ```python block."
    )


def vary(client: Callable[[str], str], task: Task, archive: Archive, parent: Version, *, step: int = 0,
         steps_left: int = 0, repairs: int = 2, deadline: float | None = None) -> list[Version]:
    """Ask the model for one new version of ``parent``; let it repair what does not run.

    Returns the versions added to the archive: every reply that held new code was scored and is
    kept, whether it ran or not, and the last one is the outcome. The list is empty if the model
    gave no code, could not be reached, or only repeated versions already scored.
    """
    prompt = _prompt(task, archive, parent, steps_left)
    added: list[Version] = []
    thought = ""
    for attempt in range(repairs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            reply = client(prompt)
        except Exception:  # noqa: BLE001 - a model that cannot be reached ends this variation, never the run
            break
        code = extract_code(reply)
        if code is None:
            prompt = (f"Your reply contained no code. Reply with the complete function {task.contract.function} "
                      "in one ```python block.")
            continue
        thought = thought or thought_of(reply)          # the idea of the variation, kept through its repairs
        known = archive.known(code)
        if known is not None:
            prompt = (f"That is exactly version {known.id}, which has been scored already. Reply with a version that "
                      f"differs, the complete function {task.contract.function} in one ```python block.")
            continue
        evaluation = task.evaluate(code)
        added.append(archive.add(parent.id, "model", code, evaluation, step=step, repairs=attempt, thought=thought))
        if evaluation.ran:
            break
        prompt = _repair_prompt(task, code, evaluation.error or "it did not run on any problem")
    return added


def evolve(client: Callable[[str], str], task: Task, seed_code: str, *, steps: int = 20, repairs: int = 2,
           parents: int = 3, seconds: float | None = None, archive: Archive | None = None,
           on_version: Callable[[Version], None] | None = None) -> Archive:
    """Score the seed, then let the model vary the leading versions ``steps`` times.

    Step ``k`` starts from the ``k mod parents``-th best version so far, so that the run does not
    climb from one version only. ``seconds`` bounds the whole run. ``on_version`` is called for the
    seed and for every version added (to write a ledger as the run goes).
    """
    archive = archive or Archive()
    deadline = None if seconds is None else time.monotonic() + seconds
    if not archive.versions:
        seed = archive.add(None, "seed", seed_code, task.evaluate(seed_code))
        if on_version:
            on_version(seed)
    for step in range(steps):
        if deadline is not None and time.monotonic() >= deadline:
            break
        ranked = archive.ranked()
        if not ranked:
            break
        parent = ranked[step % min(parents, len(ranked))]
        for version in vary(client, task, archive, parent, step=step + 1, steps_left=steps - step - 1, repairs=repairs,
                            deadline=deadline):
            if on_version:
                on_version(version)
    return archive


def summary(archive: Archive) -> dict[str, Any]:
    """Counts for a report: how many versions, how many ran, the best, and who wrote it."""
    best = archive.best()
    ran = [v for v in archive.versions if v.evaluation.ran]
    return {"versions": len(archive.versions), "ran": len(ran), "refused_or_failed": len(archive.versions) - len(ran),
            "by_model": sum(v.origin == "model" for v in archive.versions),
            "repaired": sum(v.repairs > 0 and v.evaluation.ran for v in archive.versions),
            "best": None if best is None else {"id": best.id, "origin": best.origin, "score": list(best.evaluation.score),
                                               "summary": best.evaluation.summary, "sha256": best.sha256},
            "seed_score": list(archive.versions[0].evaluation.score) if archive.versions else None}
