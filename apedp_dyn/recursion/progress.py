"""The test of the word "recursive": does the rate of solving fresh problems keep rising?

Status: experimental

At generation ``k`` the searcher of that generation is run, learning nothing,
on problems drawn for that measurement alone from a fixed distribution
(``curriculum.measurement``). With ``solved_k`` of them solved in ``cpu_k``
seconds of processor time,

    g_k = solved_k / (cpu_k / 3600)        fresh laws solved per CPU-hour.

The claim under test:

* **recursive**: ``g`` rises at every step, each rise larger than
  ``RISE_ERRORS`` standard errors of the difference;
* **improved, not shown recursive**: the last ``g`` is above the first by that
  margin, but some step did not rise;
* **not shown**: neither.

A system that only keeps ``g`` positive is finding laws; one whose ``g`` rose
once has improved; only one whose ``g`` keeps rising improves its own improving.
Wrong laws are reported beside every ``g``: a rate bought with wrong laws is
not a gain, and ``verdict`` refuses "recursive" if their share rose.

The standard error treats the count of solved problems as binomial and the CPU
time as known. ``RISE_ERRORS`` is a development constant of this test, not a
threshold of the referee; a preregistered run would fix it in its plan.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

RISE_ERRORS = 2.0


@dataclass(frozen=True)
class Measurement:
    """One generation's run on its fresh problems."""

    generation: int
    problems: int
    solved: int
    wrong: int
    reported: int
    cpu_seconds: float

    @property
    def g(self) -> float:
        """Fresh laws solved per CPU-hour."""
        return self.solved / (self.cpu_seconds / 3600.0) if self.cpu_seconds > 0 else 0.0

    @property
    def standard_error(self) -> float:
        if self.problems <= 0 or self.cpu_seconds <= 0:
            return 0.0
        share = min(max(self.solved / self.problems, 0.5 / self.problems), 1 - 0.5 / self.problems)
        return math.sqrt(self.problems * share * (1 - share)) / (self.cpu_seconds / 3600.0)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "g": round(self.g, 3), "standard_error": round(self.standard_error, 3)}


def rose(before: Measurement, after: Measurement, errors: float = RISE_ERRORS) -> bool:
    return after.g - before.g > errors * math.hypot(before.standard_error, after.standard_error)


def verdict(measurements: Sequence[Measurement], errors: float = RISE_ERRORS) -> dict[str, Any]:
    """What the measurements show about the claim."""
    ordered = sorted(measurements, key=lambda m: m.generation)
    if len(ordered) < 3:
        return {"verdict": "not shown", "reason": "fewer than three generations: a rate can rise once without recursion",
                "g": [round(m.g, 3) for m in ordered]}
    steps = [rose(a, b, errors) for a, b in itertools.pairwise(ordered)]
    overall = rose(ordered[0], ordered[-1], errors)
    first, last = ordered[0], ordered[-1]
    wrong_first = first.wrong / first.problems if first.problems else 0.0
    wrong_last = last.wrong / last.problems if last.problems else 0.0
    out: dict[str, Any] = {"g": [round(m.g, 3) for m in ordered], "steps_rose": steps, "overall_rose": overall,
                           "wrong_share": [round(m.wrong / m.problems, 3) if m.problems else 0.0 for m in ordered]}
    if all(steps) and wrong_last <= wrong_first:
        out["verdict"] = "recursive"
    elif all(steps):
        out["verdict"] = "not shown"
        out["reason"] = "the rate rose at every step, and so did the share of wrong laws"
    elif overall:
        out["verdict"] = "improved, not shown recursive"
    else:
        out["verdict"] = "not shown"
    return out
