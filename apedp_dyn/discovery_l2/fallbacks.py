"""What is tried, in order, for a table on which the route and the table's family have reported no law.

Status: experimental

**Not in force.** Nothing calls it but ``family_route.py``, its study script and its tests. Putting it
in force needs a blind test and the owner's word (AGENTS.md 1.5).

**Why this exists.** On 27 November 2026 four things that had shown no gain were kept on the owner's
word, with a design for each (``docs/SOLUTIONS_FOR_NO_GAIN_ITEMS_2026-11-27.md``), and the owner then
decided: build the best of them, and make the others its fallbacks. The best is the reading of a family
of tables (``across_tables.py``). This module is the rest of that sentence: the other three, standing
behind it, tried one after another and only where nothing before them led to a reported law.

**Each fallback proposes and nothing more** (AGENTS.md 1.1). It reads the rows a proposer may read and
hands over laws. Those laws go to the route in force *beside* the laws the table was already given, never
in their place, and the referee decides. The chain stops at the first fallback after which the route
reports a law.

**The order**, cheapest first, the one that costs calls last:

1. ``SolvedCases``: the laws of the solved tables most like this one (``proposers/experience.py``),
   their constants refitted here, each tried with its inputs in every order (up to three inputs).
2. ``TensorTrain``: the table read as a product of one-input functions
   (``proposers/tensor_train_reading.py``).
3. ``LanguageModel``: shapes named by a hosted model and fitted here
   (``proposers/llm_proposer.py``). Only with a client, which means only with a key and the caller's word.
   Two calls a table.

**What each is today.** They are fallbacks in the form each had on 27 November, not the redesigns of
that document: the cases are found by a hand-written signature, not a learned likeness; the tensor train
reads products only (rank one), and under-reads at 5 % noise; the language model is asked once with
twelve rows and gets no second round on what its shapes leave over. Each of those redesigns stays to be
built, in that order, if this arrangement earns its test.
"""
from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2.across_tables import Table, shape_of
from apedp_dyn.proposers import experience, llm_proposer

MAX_PERMUTED_INPUTS = 3      #: up to this many inputs a case's law is tried with its inputs in every order
CASE_CONSTANTS = 10
ANY_CONSTANTS = 60           #: a fallback's law may carry more constants than a shape read across a family


class Fallback(Protocol):
    """A source of laws for one table."""

    name: str

    def applies(self, table: Table) -> str | None:
        """``None`` if it can be tried on this table, otherwise why not."""

    def laws(self, table: Table) -> list[str]:
        """Laws in the table's own input names, best first. May be empty."""


def _renamed(expr: sp.Expr, names: Sequence[str], order: Sequence[int]) -> sp.Expr:
    """A law written in ``x0, x1, ...`` with ``x{i}`` read as the table's input ``names[order[i]]``."""
    placeholders = {sp.Symbol(f"x{i}"): sp.Symbol(f"__input{order[i]}") for i in range(len(names))}
    return expr.xreplace(placeholders).xreplace({sp.Symbol(f"__input{j}"): sp.Symbol(str(names[j])) for j in range(len(names))})


@dataclass
class SolvedCases:
    """The laws of the solved tables most like this one, refitted on it."""

    store: experience.Experience
    nearest: int = 8
    handed: int = 5
    name: str = "solved cases"

    def applies(self, table: Table) -> str | None:
        inputs = len(table.names)
        if not any(case.inputs == inputs for case in self.store.cases):
            return f"the store holds no solved table of {inputs} inputs"
        return None

    def laws(self, table: Table) -> list[str]:
        inputs = len(table.names)
        orders = (list(itertools.permutations(range(inputs))) if inputs <= MAX_PERMUTED_INPUTS else [tuple(range(inputs))])
        fitted: dict[str, tuple[float, str]] = {}
        for case in self.store.nearest(table.X, table.y, k=self.nearest):
            try:
                skeleton = llm_proposer.parse_skeleton(experience.skeleton(case.law), inputs, max_constants=CASE_CONSTANTS)
            except (llm_proposer.UnsafeExpressionError, sp.SympifyError, SyntaxError, TypeError):
                continue
            for order in orders:
                # the case's law with its inputs read in this order, fitted on this table's columns as they stand
                permuted = skeleton.xreplace({sp.Symbol(f"x{i}"): sp.Symbol(f"__p{order[i]}") for i in range(inputs)}).xreplace(
                    {sp.Symbol(f"__p{j}"): sp.Symbol(f"x{j}") for j in range(inputs)})
                key = llm_proposer.structure_key(permuted)
                try:
                    constants, error = llm_proposer.fit_constants(permuted, table.X, table.y)
                except Exception:  # noqa: BLE001, S112 - a shape that cannot be fitted here is skipped; the others are still tried
                    continue
                if not np.isfinite(error) or (key in fitted and fitted[key][0] <= error):
                    continue
                law = permuted.xreplace({sp.Symbol(name): sp.Float(float(f"{value:.8g}")) for name, value in constants.items()})
                fitted[key] = (float(error), str(_renamed(law, table.names, tuple(range(inputs)))))
        return [law for _error, law in sorted(fitted.values())[: self.handed]]


@dataclass
class TensorTrain:
    """The table read as a product of one-input functions, if it is one."""

    name: str = "tensor train"
    last: dict[str, Any] | None = None

    def applies(self, table: Table) -> str | None:
        return None if len(table.names) >= 2 else "a table of one input has no product to read"

    def laws(self, table: Table) -> list[str]:
        from apedp_dyn.proposers import tensor_train_reading

        self.last = tensor_train_reading.read_product(table.X, table.y, table.names)
        return [str(self.last["law"])] if self.last.get("law") else []


@dataclass
class LanguageModel:
    """Shapes named by a hosted model, fitted here. Tried only if a client is given."""

    client: Callable[[str], str] | None = None
    handed: int = 5
    name: str = "language model"

    def applies(self, table: Table) -> str | None:
        return None if self.client is not None else "no client was given: it needs a key and the caller's word"

    def laws(self, table: Table) -> list[str]:
        assert self.client is not None
        proposer = llm_proposer.LlmEquationProposer(self.client, target_col="__target__", rounds=1, max_proposals=self.handed)
        frame = pd.DataFrame({**table.columns(), "__target__": np.asarray(table.y, dtype=float)})
        out = []
        for proposal in proposer.fit_and_propose(frame, target_col="__target__"):
            written = sp.sympify(str(proposal.formula), locals={f"x{i}": sp.Symbol(f"x{i}") for i in range(len(table.names))})
            out.append(str(_renamed(written, table.names, tuple(range(len(table.names))))))
        return out


def try_in_order(table: Table, known: Sequence[str], fallbacks: Sequence[Fallback],
                 read: Callable[[Table, list[str]], dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Each fallback in turn until the route reports a law: the reading that reported (or ``None``) and what each did.

    ``known`` are the laws the table was already given; they are handed over again beside each fallback's own, so the
    referee meets the same rivals. ``read(table, laws)`` is the route in force on those laws. A fallback that raises is
    written down and passed over; it does not stop the chain.
    """
    trail: list[dict[str, Any]] = []
    for fallback in fallbacks:
        started = time.perf_counter()
        step: dict[str, Any] = {"fallback": fallback.name, "tried": False}
        trail.append(step)
        why_not = fallback.applies(table)
        if why_not is not None:
            step["why_not"] = why_not
            continue
        try:
            laws = [law for law in fallback.laws(table) if shape_of(law, table.names, ANY_CONSTANTS) is not None]
        except Exception as exc:  # noqa: BLE001 - a fallback that fails says so and the next one is tried
            step.update({"tried": True, "error": f"{type(exc).__name__}: {exc}"[:200], "seconds": round(time.perf_counter() - started, 2)})
            continue
        step.update({"tried": True, "proposed": len(laws)})
        if not laws:
            step["seconds"] = round(time.perf_counter() - started, 2)
            continue
        reading = read(table, [*laws, *known])
        own = {shape.key for shape in (shape_of(law, table.names, ANY_CONSTANTS) for law in laws) if shape is not None}
        reported_shape = shape_of(str(reading.get("law")), table.names, ANY_CONSTANTS) if reading.get("reported") else None
        step.update({"reported": bool(reading.get("reported")), "law": reading.get("law") if reading.get("reported") else None,
                     "the_reported_law_is_one_of_its_own": bool(reported_shape is not None and reported_shape.key in own),
                     "seconds": round(time.perf_counter() - started, 2)})
        if reading.get("reported"):
            return reading, trail
    return None, trail
