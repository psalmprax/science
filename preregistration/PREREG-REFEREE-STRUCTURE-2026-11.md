# PREREG-REFEREE-STRUCTURE-2026-11: does Guard 3 stop the referee promoting laws that do not fit?

**Status:** frozen before the run by `preregistration/DIGEST-REFEREE-STRUCTURE.json`. Runs once, on
GitHub Actions (`.github/workflows/srbench-referee-structure.yml`), started by the first branch named
`referee-structure-run-*` whose shards start.

## What is tested

PREREG-SRBENCH-BASELINE-2026-11 ran the template engine with the referee on SRBench. The referee
promoted a law in 268 of 532 runs. 100 of those were the benchmark's law, 168 were not, and 160 were
not accurate on the test rows (audit M-10). `UNIQUE_SUPPORT` is a verdict between the candidates the
search built; when the law is outside the grammar it certifies the least bad one.

`apedp_dyn/epistemic/structure_guard.py` adds one requirement to a promotion: the promoted law must
leave no structure in its holdout residuals over the input space. The statistic is the leave-one-out
R² of each residual predicted from the residuals of its nearest neighbours in the standardised
inputs, at 4, 8, 16, 32 and 64 neighbours; each is standardised and the largest is taken. It is
compared with the same quantity on 999 seeded permutations of the residuals, at the canonical
significance level (0.01), so a true law is refused at about that rate whatever the neighbourhood
sizes are. Residuals within `ABSOLUTE_EPS` (10⁻¹²) of the holdout observation RMS pass as an exact
fit up to rounding. Missing evidence fails. The guard is applied to
the verdict `adjudicate` returned, so it covers every route to a promotion; `adjudicate.py` is not
changed. It is on only when `L2DiscoveryEngine.discover` is called with `structure_rule=True`.

The question is whether, with the guard on, the referee stops promoting laws that do not fit and
keeps promoting the ones that are right.

## Data and protocol

Exactly the L2 arm of PREREG-SRBENCH-BASELINE-2026-11: SRBench's 133 problems (119 Feynman, 14
Strogatz), noise 0, 0.1 %, 1 % and 10 %, the same training and test rows, the same seed, the same
engine settings, 120 s per run, the same scoring. 532 runs. The one difference is
`structure_rule=True`. PySR is not involved.

A promotion is a run whose verdict is `UNIQUE_SUPPORT`. It is *correct* when the promoted structure
is the benchmark's law by the baseline's symbolic criterion, *false* otherwise, and *not accurate*
when the run's test R² is below 0.999, as in the baseline.

The comparison is with the recorded baseline, run for run.

**Development and held-out problems.** Thirty problems (those whose name hashes to 0 modulo 5 under
the tag in `scripts/run_srbench_referee_structure.py`, the same thirty as PREREG-REFINED-SEARCH) were
used while building the guard. The other 103 were not run with it. The expectations are stated on
the 103: 412 runs, in which the baseline promoted 201 laws, 72 correct and 129 false, 121 of them
not accurate, with 56 timeouts.

## Expectations, fixed now

1. **What it promotes fits.** Of the laws promoted on the held-out problems, at most 5 % are not
   accurate (the baseline: 121 of 201, 60 %).
2. **Correct laws are kept.** Of the 72 held-out runs in which the baseline promoted the correct law,
   at least 65 promote the correct law again.
3. **What it promotes is the law.** Of the laws promoted on the held-out problems, at most 10 % are
   false by the symbolic criterion (the baseline: 129 of 201, 64 %).
4. Reported without a criterion: each noise level separately, the 30 development problems,
   timeouts, how many refusals carry the guard's flag, the test R² of every law still wrongly
   promoted, and time.

A miss is reported as a miss.

## What follows from the result, fixed now

- If expectations 1 and 2 hold, `structure_rule=True` becomes the default of
  `L2DiscoveryEngine.discover`. That is a change of the engine's behaviour everywhere it is used; it
  is made in its own commit with the repository's regression suite, and any test whose verdict
  changes is examined case by case, not edited to pass.
- If expectation 1 holds and 2 does not, the guard stays off by default and the lost laws are
  examined.
- If expectation 1 does not hold, the guard stays off by default and audit M-10 stays open for the
  default referee.

`adjudicate()` itself gets no default guard in any case: callers that do not hand it holdout inputs
could not be tested and would be refused wholesale.

## Disclosed before freezing

- **Development changed the design three times.** A first version sat inside `adjudicate` and
  guarded two of its three promotion exits; at 1 % noise 3 of the 10 wrong development promotions
  left through the third. The guard was then moved to act on the verdict. At zero noise it refused
  4 of 7 correct development laws, because the right law's rounding error follows the size of the
  prediction; the rounding allowance was added. And with eight neighbours alone it passed one wrong
  development law at 10 % noise (`feynman_I_15_3x`, test R² 0.990, p = 0.17), which larger
  neighbourhoods see (p ≤ 0.005 at 16 and above); the statistic became the largest over five sizes,
  calibrated by the same permutations. All three were found on development problems and all three
  are tested (`tests/test_referee_structure_guard.py`). A design changed three times on thirty
  problems is fitted to them to some degree; that is what the 103 are for.
- **Development result with the code as frozen**, 30 problems, run locally on two cores:

  | Noise | Baseline: correct / false promotions | With the guard: correct / false | Not accurate | Refused by the guard | Timeouts: baseline / here |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | 0 | 7 / 10 | 7 / 0 | 0 | 10 | 3 / 3 |
  | 1 % | 7 / 10 | 8 / 0 | 0 | 9 | 3 / 3 |
  | 10 % | 7 / 9 | 8 / 0 | 0 | 9 | 3 / 2 |

  Over the three levels the baseline promoted 21 correct and 29 false laws on these problems; with
  the guard 23 correct and 0 false, none inaccurate, and all 21 of the baseline's correct promotions
  are kept. The two extra correct promotions are one problem (`feynman_III_15_27`, about 100 s)
  that finished inside the time limit here at two levels and timed out in the baseline; it timed
  out here too at noise 0. One of the baseline's false promotions at 1 % (`strogatz_shearflow2`)
  timed out here before a verdict, on a loaded machine; an earlier development run had refused it.
  The 0.1 % level was not run in development. Runs that end
  without a promotion before the guard are the same with any version of it, so only the 52 runs
  with a promotion before the guard were repeated after its last change; the other 38 are from the
  run before that change.

- **What was and was not tuned.** 999 permutations, at most 1,500 rows and the level (the canonical
  0.01) were chosen before any benchmark row was read and were not changed. The rounding tolerance
  is the referee's existing `ABSOLUTE_EPS`. The neighbourhood sizes were changed once, as described
  above: from eight alone to the five sizes 4 to 64, without trying other sets.
- **The baseline was known** when the numbers above were chosen, including its held-out rows.
  Eight of the baseline's false held-out promotions are accurate (test R² at least 0.999): the
  guard may or may not see them, and they count against expectation 3 and not against 1.
- **What the guard cannot see.** A wrong law whose error is small beside the noise leaves residuals
  that look like noise. At 10 % noise such laws can still be promoted; they are counted.
- **Where the test's level is not guaranteed.** The permutation test assumes the rows are
  exchangeable. SRBench's rows are independent draws, which is that case. Rows taken along one
  trajectory are not, and nothing here measures the guard on them.
- **Timeouts depend on the machine.** The baseline timed out in 56 held-out runs at 120 s on GitHub's
  runners. A correct law the baseline promoted that times out here counts as not kept.
- **Not covered:** the multi-term engine, and every other caller of `adjudicate`.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
