"""Recursive self-improvement on CPUs, inside a judge that never changes.

Status: experimental

**Not in force; development only.** Nothing in the routes calls this package.
Its runs write proposals; adopting one needs the owner's word (AGENTS.md 1.5).

The recursion has three levels:

* **Level 0, finding laws:** the refined route with its referee and last checks,
  as in force (``learning_loop.discover_law``);
* **Level 1, improving the finders** (``levels.level1``): the searcher's
  settings, its library of building blocks and, when the owner provides a
  coding model, the code of a candidate generator change from one generation to
  the next. A change is kept only if it solves more fresh problems for the CPU
  it spends than the searcher it would replace, on problems drawn for that
  comparison alone;
* **Level 2, improving the improver** (``levels.level2``): how Level 1 changes
  the searcher (how many variants it tries, how far it moves each setting, how
  many problems decide, which problems it studies) is itself varied, and each
  way is scored by the laws its Level 1 run gained per CPU-hour.

The tools it adds, all of which run on CPUs:

* ``library``: building blocks learned from solved laws by compression
  (DreamCoder's idea, in a small form), and a proposer that builds candidates
  from them;
* ``curriculum``: invented laws drawn at the level of difficulty the system
  solves about half the time;
* ``egraph``: equality saturation, to show two expressions equal by rewrites;
* ``smt``: proofs with the z3 solver that a law has no pole on a box, or that
  two rational expressions are equal there;
* ``mdl``: description length in bits, the score the library is chosen by.

**The test of the word "recursive"** (``progress``): let ``g_k`` be the fresh
laws solved per CPU-hour at generation ``k``, on problems drawn for that
measurement and never used to choose anything. The system is recursively
self-improving only if ``g_k`` rises from each generation to the next, not
merely if it stays positive.

**What never changes inside the loop** (``guard``): the referee, its
thresholds, the last checks, the measured range, the recovery scorer and the
benchmark's fingerprints are sealed by their SHA-256 when a run starts and
checked at every generation. A searcher's settings may not name a threshold.
The held-back invented laws are never opened. A loop that could change its
judge would get better at passing it, not at finding laws (audit M-37, M-38).
"""
