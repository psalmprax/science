"""The template engine with the referee's structure guard on by default.

Status: experimental

PREREG-REFEREE-STRUCTURE-2026-11 fixed in advance what would follow if its
first two expectations held: Guard 3 (``epistemic/structure_guard.py``) becomes
the default for the template engine. They held (SRBench, run once: 3 of 113
promotions not accurate against 160 of 268 without the guard; 71 of the 72
correct held-out promotions kept; ``preregistration/RESULT-REFEREE-STRUCTURE-2026-11.md``).

The sentence named the default of ``L2DiscoveryEngine.discover``. That method
lives in ``discovery_l2/l2_engine.py``, which the same preregistration froze by
digest, so it is not edited. ``GuardedL2DiscoveryEngine`` is the engine with
the default changed, and callers are moved to it one at a time, each with its
tests run and every changed verdict examined.

**Which callers.** The guard's test is a permutation test: it is exact when the
rows are exchangeable under the true law, which holds for independent draws and
for rows whose only error is independent noise. It does not hold for targets
made by differentiating one trajectory, where neighbouring errors are shared;
there the guard would refuse true laws for the structure of the differencing
error. Such callers stay on ``L2DiscoveryEngine`` until the guard has been
measured on their kind of rows. ``docs/SYSTEM_FLAWS_AUDIT.md`` (M-10) lists who
has moved and who has not.

Passing ``structure_rule=False`` gives the frozen engine's behaviour exactly.
"""
from __future__ import annotations

from typing import Any

from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine, L2DiscoveryResult


class GuardedL2DiscoveryEngine(L2DiscoveryEngine):
    """``L2DiscoveryEngine`` whose ``discover`` applies the structure guard unless told not to."""

    def discover(self, *args: Any, structure_rule: bool = True, **kwargs: Any) -> L2DiscoveryResult:
        return super().discover(*args, structure_rule=structure_rule, **kwargs)
