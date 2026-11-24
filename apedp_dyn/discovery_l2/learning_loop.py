"""The system learning from the laws its referee confirms: self-improvement inside fixed limits.

Status: experimental

**Not in force; development only.** Nothing calls it but its tests and
``scripts/run_learning_stream.py``. Putting it in force needs a blind test and
the owner's word (AGENTS.md 1.5).

**What learns.** Two things, from the same laws:

* the **memory of confirmed laws** (``proposers.structure_memory.Memory``): each
  learned law adds its one-variable parts and its whole shape, so the next
  problem gets candidates grown from them. In force for tables this memory is a
  file built once from recorded runs (PREREG-MEMORY: 17 laws recovered of 80
  became 23). Here it grows with every law the system confirms;
* the **law language model** (MMLM-L, ``apedp_mmlm.law_language``): each
  learned law is observed, and the model orders the first pass's exhaustive
  search. That order changes how fast the first pass finds a law, not which
  laws it can reach: it tries every candidate up to its size limit
  (RESULT-MMLM-L: the gain there came from the stop rule, not from the prior).

**What never learns.** The referee's path, its confirmation on measured rows,
the last checks, the measured range and every threshold. They decide what
counts as a confirmed law, and a loop allowed to change them could raise its
own score without finding anything (audit M-37, M-38: a scripted adversary
did exactly that to two judges).

**What it learns from.** Only a law that the route reported after confirmation
on measured rows, that the last checks in force vouched for, and whose table
was inside, or at the edge of, the measured range. Never a tentative law. Never
a law from a table outside the range. Never a law while the last checks are off.

**Undoing it.** Every learned law is journaled with its source and domain.
``retract`` removes one and rebuilds both learners from the seed and what is
left, so a law refuted later stops being used. The state is the seed plus the
journal; ``save`` writes the journal, ``LearningState.replay`` rebuilds it.

**On the ODE route** (``discover_component``) only the memory learns, beside
the route's four searches. The route has no last checks, so its reported law
is what is learned.

**Taking it apart.** ``LearningState(learners=...)`` lets one learner learn
alone, and ``learnable(beyond_range=True)`` drops the measured-range condition.
Both exist for development streams (``scripts/run_learning_families.py``) and
are not for use in force.

**The risk it carries.** A wrong law that passes the referee is learned, and
its parts are offered on later problems. The referee and the last checks
still judge every later law; what they let through is the bound. The stream
experiment counts wrong laws over time for that reason.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import default_route, refined_search, vouched_route_v2
from apedp_dyn.discovery_l2.evolved_stage import SideBySide
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.structure_memory import Memory, StructureMemoryProposer
from apedp_mmlm import law_language
from apedp_mmlm.ledger_learning import parse_law

#: measured-range marks a table may carry for its law to be learned
LEARNABLE_RANGE = ("inside", "edge")
PROPOSALS = 3            # laws the memory hands over, as in force
SECONDS = 60.0           # how long the memory may grow them, as in force


@dataclass
class Learned:
    """One law the loop learned, with where it came from."""

    key: str
    law: str
    domain: str | None
    source: str
    at: float
    retracted: bool = False


LEARNERS = ("memory", "model")


class LearningState:
    """The learners and the journal of what they learned.

    ``learners`` says which of the two learn (both by default); the other stays
    at its seed. A journal is kept either way, so the same laws can be replayed
    into a state that learns with both.
    """

    def __init__(self, memory_seed: Memory, seed_laws: Sequence[str], seed_domains: Sequence[str | None] | None = None,
                 learners: Sequence[str] = LEARNERS) -> None:
        unknown = set(learners) - set(LEARNERS)
        if unknown:
            raise ValueError(f"learners must be among {LEARNERS}, not {sorted(unknown)}")
        self.learners = tuple(learners)
        self._memory_seed = memory_seed.as_dict()
        self._seed_laws = list(seed_laws)
        self._seed_domains = list(seed_domains) if seed_domains is not None else [None] * len(self._seed_laws)
        self.journal: list[Learned] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self.memory = Memory.from_dict(self._memory_seed)
        trees, domains = [], []
        for law, domain in zip(self._seed_laws, self._seed_domains, strict=True):
            tree = parse_law(law)
            if tree is not None:
                trees.append(tree)
                domains.append(domain)
        self.model = law_language.LawLanguageModel().fit(trees, domains)
        for item in self.journal:
            if not item.retracted:
                self._apply(item.law, item.domain)

    def _apply(self, law: str, domain: str | None) -> bool:
        if "memory" in self.learners:
            self.memory.learn(law)
        tree = parse_law(law)
        if tree is not None and "model" in self.learners:
            self.model.observe(tree, domain)
        return tree is not None

    def learn(self, law: str, *, domain: str | None = None, source: str = "") -> Learned:
        """Add a law both learners learn from, and journal it."""
        item = Learned(f"{len(self.journal):05d}", str(law), domain, source, round(time.time(), 3))
        self._apply(item.law, item.domain)
        self.journal.append(item)
        return item

    def retract(self, key: str) -> bool:
        """Unlearn one journaled law: both learners are rebuilt from the seed and the laws left."""
        for item in self.journal:
            if item.key == key and not item.retracted:
                item.retracted = True
                self._rebuild()
                return True
        return False

    def prior(self, domain: str | None = None) -> Callable[[Any], float]:
        """The first pass's order: the law model's code length of a search node."""
        order: Callable[[Any], float] = law_language.search_prior(self.model, domain)
        return order

    def grower(self, seconds: float = SECONDS, proposals: int = PROPOSALS) -> ProposerBase:
        """The memory as a source of candidates, behind the fit guard, as in force. It reads this state's memory live."""
        return FitGuardProposer(StructureMemoryProposer(self.memory, max_proposals=proposals, seconds=seconds,
                                                        target_col="__target__"), target_col="__target__")

    def learned(self) -> list[Learned]:
        return [item for item in self.journal if not item.retracted]

    def summary(self) -> dict[str, Any]:
        return {"learned": len(self.learned()), "retracted": sum(item.retracted for item in self.journal),
                "motifs": len(self.memory.motifs), "shapes": len(self.memory.skeletons), "model": self.model.state()}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"journal": [asdict(item) for item in self.journal]}, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")

    @classmethod
    def replay(cls, path: Path, memory_seed: Memory, seed_laws: Sequence[str],
               seed_domains: Sequence[str | None] | None = None, learners: Sequence[str] = LEARNERS) -> LearningState:
        """The state a saved journal gives over the same seed."""
        state = cls(memory_seed, seed_laws, seed_domains, learners)
        state.journal = [Learned(**item) for item in json.loads(path.read_text(encoding="utf-8"))["journal"]]
        state._rebuild()
        return state


def learnable(record: Mapping[str, Any], *, beyond_range: bool = False) -> str | None:
    """Why a route's record may not be learned from, or ``None`` if it may.

    ``beyond_range`` drops the measured-range condition. **Development only**:
    it measures what learning from tables the last checks were never tested on
    (more than three inputs, say) would add, and must not be used in force.
    """
    if not record.get("reported") or not record.get("law"):
        return "no law was reported"
    vouching = record.get("vouching") or {}
    if not vouching.get("in_force") or not vouching.get("applied"):
        return "the last checks did not run on it"
    state = (record.get("measured_range") or {}).get("state")
    if not beyond_range and state not in LEARNABLE_RANGE:
        return f"its table is {state or 'unmarked'} with respect to the measured range"
    return None


def discover_law(
    X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    state: LearningState, proposer: ProposerBase | None = None, learn: bool = True, domain: str | None = None,
    source: str = "", noise: str = "relative", seed: int = 0, memory_seconds: float = SECONDS, beyond_range: bool = False,
) -> dict[str, Any]:
    """The refined path with the state's memory beside ``proposer`` and the state's order for the first pass,
    then the last checks in force and the measured range; the law is learned if ``learnable`` allows.

    The record is ``vouched_route_v2.checked``'s, with ``learning`` added.
    """
    grower = state.grower(memory_seconds)
    stage: ProposerBase = grower if proposer is None else SideBySide(proposer, grower)
    started = time.perf_counter()
    result = refined_search.discover_refined(X, np.asarray(Y, dtype=float), pool_X, measure, proposer=stage, noise=noise,
                                             seed=seed, structure_bits=state.prior(domain))
    law = result.law
    record = {"route": default_route.REFINED, "action": result.action, "law": law, "tentative_law": result.tentative_law,
              "tentative_reason": default_route.NOTHING_CHALLENGED if result.tentative_law else None,
              "law_as_confirmed": law or result.tentative_law, "reported": law is not None, "complete": True,
              "dropped_terms": list(result.dropped_terms), "units": dict(result.units),
              "proposer_failures": list(result.proposer_failures), "rounds": [], "refined": result,
              "main_search": result.path}
    out: dict[str, Any] = vouched_route_v2.checked(record, X, Y, pool_X)
    why_not = learnable(out, beyond_range=beyond_range)
    learned = None
    if learn and why_not is None:
        learned = state.learn(str(out["law"]), domain=domain, source=source)
    first = result.path.first_pass if result.path is not None else {}
    out["learning"] = {"learned": learned is not None, "key": None if learned is None else learned.key,
                       "why_not": why_not if learn else "learning was off for this call", "seconds": round(time.perf_counter() - started, 2),
                       "first_pass": {k: first.get(k) for k in ("evaluated", "candidates", "stopped_early") if k in first},
                       "state": {"learned": len(state.learned()), "motifs": len(state.memory.motifs)}}
    return out


def discover_component(dev: Any, pool: Any, component: int, *, state: LearningState, budget_seconds: float,
                       seed: int = 0, learn: bool = True, source: str = "", memory_seconds: float = SECONDS
                       ) -> tuple[Any, dict[str, Any]]:
    """The ODE route in force for one component, with the state's memory beside its four searches.

    Only the memory takes part: the ODE route's searches take no first-pass
    order. The route has no last checks and no measured range, so the law it
    reports (not a tentative one) is what is learned. Its reported laws were
    wrong in 3 of 165 on the blind systems (RESULT-ODE-ADDITIONS).
    """
    from apedp_dyn.ode import default_route as ode_default_route
    from apedp_dyn.ode import route_in_force

    searches = ode_default_route.search_stage(budget_seconds, seed) if budget_seconds > 0 else None
    grower = state.grower(memory_seconds)
    stage: ProposerBase = grower if searches is None else SideBySide(searches, grower)
    started = time.perf_counter()
    found, done = route_in_force.discover_component(dev, pool, component, budget_seconds=budget_seconds, seed=seed,
                                                    proposer=stage)
    learned = None
    why_not = None if found.law else "no law was reported"
    if learn and why_not is None:
        learned = state.learn(str(found.law), source=source)
    done = dict(done)
    done["learning"] = {"learned": learned is not None, "key": None if learned is None else learned.key,
                        "why_not": why_not if learn else "learning was off for this call",
                        "seconds": round(time.perf_counter() - started, 2),
                        "state": {"learned": len(state.learned()), "motifs": len(state.memory.motifs)}}
    return found, done
