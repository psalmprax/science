# PREREG-MAIN-SEARCH-2026-11: the main search path as a whole, and the in-house GA/GP

**Status:** frozen before the run by `preregistration/DIGEST-MAIN-SEARCH.json`. Runs once, on GitHub
Actions (`.github/workflows/srbench-main-search.yml`), started by the first branch named
`srbench-main-search-run-*` whose shards start.

## What is tested

Three things the earlier preregistrations left open (audit M-10 and M-11):

1. **The path as a whole.** `apedp_dyn/discovery_l2/main_search.py` is the route for discovering a
   law from tabular data since PySR was made the main search: an exhaustive first pass with the
   confirmed early stop, PySR seeded by it, contrast confirmation, and an adequacy check. Each stage
   has its own result; the path has none.
2. **The fix for M-10.** The referee's sole-survivor rule promoted a wrong law in 62.7 % of its
   promotions on this benchmark. The path does not use that rule. The question is whether what the
   path reports can be relied on.
3. **GA/GP.** The in-house evolutionary search (`apedp.evolutionary_search.evolve`, native kernel)
   was never run on this benchmark, so there was no number to keep or retire it on.

## Data and protocol

SRBench's 133 ground-truth problems (119 Feynman, 14 Strogatz), with the splits and the noise of
PREREG-SRBENCH-BASELINE (its harness builds them, so the rows are the same), at noise 0 and 1 %.
266 runs per engine.

- **`main_search`.** 70 % of the training rows are development rows; the other 30 % are the pool
  whose targets only the confirmation step may read. PySR has 240 s, as in PREREG-SRBENCH-EXTERNAL.
  The first pass uses the size order, not the learned prior (the prior was trained on these laws).
  A run *reports a law* when the outcome is `confirmed`, `equivalent_class` or `unchallenged`;
  `widen_search` reports none.
- **`gagp`.** All training rows; 8 islands, 1,500 generations, every function family on; strict
  native kernel. Its pick is scored as any engine's model is.

A reported law (or GA/GP's pick) is scored on the test rows as in the baseline: *accurate* if test
R² is at least 0.999; *symbolically recovered* by the baseline's check. A run is killed after 480 s.

## Expectations, fixed now

1. **What the path reports predicts:** of the runs in which the path reports a law, at most 5 % are
   not accurate (pooled over both noise levels and both families). This is the M-10 criterion.
2. **What it reports is mostly the law:** at least 60 % of the reported laws are symbolically
   recovered (the referee's promotions: 37.3 %).
3. **Little is lost against PySR alone:** on the Feynman problems the path's symbolically recovered
   laws are at least 62 at noise 0 and at least 58 at 1 % (85 % of the 73 and 68 PySR recovered alone
   in PREREG-SRBENCH-EXTERNAL).
4. **GA/GP is behind PySR:** on the Feynman problems GA/GP recovers fewer laws than PySR alone did,
   at both noise levels (fewer than 73 and fewer than 68).
5. Reported without a criterion: refusals, which stage found each law, the Strogatz problems,
   timings, and GA/GP's accuracy solutions.

A miss on 1 to 4 is reported as a miss. If 4 holds, GA/GP is retired as a law-discovery engine; if it
does not, it is kept and the decision goes back to the owner.

## Disclosed before freezing

- **What was seen while building the path.** A smoke run of the real PySR through the path on one
  toy law (60 s, 1 % noise): PySR did not return the law; the path reported a long accurate
  approximation as `equivalent_class`. On noise-free toy data PySR returned the exact law.
- **Changes made after that, on toy laws and four benchmark problems** (`feynman_I_12_1`,
  `feynman_I_12_2`, `feynman_I_6_2a`, `strogatz_bacres1`, with a stand-in for PySR because PySR
  cannot run in the authoring environment): the simplest member of an equivalent class is reported;
  noise-free data is handled by confirming fits that are exact to numerical precision directly on
  pool rows; constants inside searched expressions are refitted; and the adequacy check gained a
  second condition. That last one came from `strogatz_bacres1`: the first pass stopped on
  `A - B*cos(1/x1)` (test R² 0.95) and the residual check passed it. The law now also has to
  predict at least as well as a nearest-neighbour smoother, which refuses that candidate (its error
  is 3.2 times the smoother's). These four problems are not removed or treated differently.
- **The comparison in expectation 3 is not like for like.** PySR alone had all the training rows; in
  the path it has 70 % of them, and the path can refuse. The 85 % is my estimate of what that costs.
- **Expectation 1 depends on the adequacy check, which has no result of its own.** In dimensions
  above about four the smoother is poor and the check is weak; inaccurate laws may get through
  there.
- **GA/GP's configuration** was sized by timing on one toy law (172 s on two cores) so that it has
  about the same time as PySR. On that toy law and on `feynman_I_12_1` it returned accurate, long
  expressions that were not the law. It has not been tuned for this benchmark; neither has PySR.
- The workflow first runs both engines on three problems with a 30 s budget (`smoke`). If the smoke
  fails, nothing preregistered has run; the fault is fixed, the digest re-frozen with the reason
  recorded here, and a new run branch started.

## Re-frozen once before any run (2026-11-19)

The first run branch (`srbench-main-search-run-1`, commit `7cf474c`) stopped at the smoke job; no
shard started and no preregistered number exists from it. The smoke found two faults in the harness
side, both fixed before this freeze:

- PySR reads a seed expression as Julia, where a power is `^`; the first pass's `**` made it fail to
  parse on `feynman_I_12_2`. Seeds are now written with `^`, and if PySR still rejects a seed the
  search runs without it and says so in its metadata.
- GA/GP's pick can contain `erf`, a Bessel function or `sign`, which the scoring could not evaluate
  on arrays (`feynman_I_12_1`). They are now evaluated through SciPy.

What the smoke showed on its three problems, for the record (30 s budgets, not part of the result):
the path reported the exact law on `feynman_I_12_1` from the first pass; on `strogatz_bacres1` it
reported an accurate law that was not the symbolic one, as `equivalent_class`; GA/GP recovered
`feynman_I_12_2` at noise 0 and was neither accurate nor symbolic on `strogatz_bacres1`. Nothing in
the protocol or the expectations was changed after seeing these.

## Deviation rule

The run is not repeated. If a shard fails for a reason of infrastructure (a runner lost, a download
that fails), that shard alone is re-run and the fact is recorded with its date. Any change to a
frozen file after shards have started makes the result a development result.
