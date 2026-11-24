# PREREG-VOUCHING-V2-2026-11: does the second version of the last checks stop wrong laws without costing right ones, on laws nobody has run?

**Status:** frozen before the run by `preregistration/DIGEST-VOUCHING-V2.json`. Runs once, on GitHub
Actions (`.github/workflows/vouching-v2-blind.yml`), started by the first branch named
`vouching-v2-run-*` whose shards start.

## What is tested

The table route's last checks (`apedp_dyn/discovery_l2/vouching.py`) passed a blind test on
counterfeits (PREREG-COUNTERFEIT-2026-11) and then failed one behind a real search
(PREREG-VOUCHED-SRBENCH-2026-11): on the physics benchmark they withheld 43 of the 193 laws the
route recovered and stopped nothing. On 2026-11-23 the owner took them out of the route in force,
with the condition that a repaired version return only after it has been run on both benchmarks
and on laws drawn after it was written. This is that run.

`apedp_dyn/discovery_l2/vouching_v2.py` is the repair. It differs from the first version in three
ways, each aimed at a fault found on a recorded run:

- **Errors are planted where the law's constants cannot follow them** (audit M-33). Each shape
  of error tried is first made orthogonal to the law's gradient in its constants. A shape the
  constants can follow almost entirely is dropped, since refitting would absorb it.
- **The chance of seeing an error is computed for the test that looks for it.** An F test asks
  whether polynomials of the inputs up to the fourth degree explain the residuals, and the power
  rule takes that test's power from the noncentral F distribution. The power does not depend on
  how smooth the shape is.
- **The rivals are not used as shapes of error** (audit M-35). The searches proposed them on
  the same rows, so a test in their direction takes the rows' noise for an error. They are
  judged by the class rule alone, as before.

The neighbour test with its sign-flip null, the class rule and the fit-error rule are the first
version's. Two tests, each at half the canonical level 0.01. It does not repair M-34: a law
whose error sits on a few rows next to a pole, with no rival pointing at it, can pass.

## Development, done before freezing, and what it showed

All of it on records whose answers were known and that were read before the module was written.
`scripts/replay_vouching_v2.py` gives each version the same rows and candidates the route had;
nothing is searched again.

| Record | | The route | First version | Second version |
| :--- | :--- | ---: | ---: | ---: |
| PREREG-VOUCHED-SRBENCH, 229 reports behind four searches | laws kept, of 193 | | 150 | **192** |
| | inaccurate reported | 0 | 0 | 0 |
| PREREG-COUNTERFEIT, 229 reports on handed candidates | wrong reported | 16 | 2 | **0** |
| | laws kept, of 146 | | 139 | 140 |
| | the law named | 146 | 146 | 150 |

The red-team campaigns' worlds are replayed too (`benchmarks/vouching_v2_dev_campaign1.json`,
`_campaign2.json`), 1,500 records of each, 150 for each arena and stage, drawn again:

| Campaign | | The route | First version | Second version |
| :--- | :--- | ---: | ---: | ---: |
| First | wrong reported | 472 | 1 | 1 |
| | laws kept, of 524 | | 494 | 493 |
| Second (its climb aimed at the first version) | wrong reported | 387 | 136 | 168 |
| | laws kept, of 769 | | 742 | 758 |
| Second, noise up to 5 % (1,322 records) | wrong reported | 261 | 107 | 98 |
| Second, noise above 5 % (178 records) | wrong reported | 126 | 29 | **70** |

**Above 5 % noise the second version is weaker than the first.** Nearly all of the difference is
on the climb in the noise-level arena, at 7 % and 10 % noise on 1,000 rows. There the first version
withheld the law because an error the size of a rival's difference would not have been seen. The
second version computes its power for errors of polynomial shape, sees enough power, and passes
it. This test does not go above 5 % noise. The weakness is recorded as audit M-36, open whatever
the result.

**Three changes were made while replaying, so the replay shows what was built:**

1. A test of the residuals in the rivals' directions was taken out. It refused 8 of 90 right laws
   at 1 % noise on the physics benchmark. Refitting each rival on half the rows still refused
   7, while the same test on fresh noise refused none: the searches' choice of form carries the
   noise.
2. Polynomial shapes were raised from the third degree to the fourth. On the counterfeit
   benchmark this stopped the one wrong law the third degree let through: a thirteen-term cubic.
3. A third test, the largest standardized residual against Gaussian draws, was dropped. On the
   campaigns' worlds with noise of another shape it refused most right laws (28 of 157 kept where
   the first version kept 157), and it changed nothing on either benchmark.

## Data and protocol

**Laws.** `benchmarks/vouching_v2_blind_laws.json`, drawn by `scripts/run_vouching_v2_blind.py
laws` after the module's last change, in two parts. Every structure is left out that occurs in
the invented-laws benchmark (by fingerprint; no held-back law is read), among the red-team
campaigns' 300 laws, the memory's 120 laws or the counterfeit benchmark's 60, or in the other part.

- **handed**: 60 laws (7 with one input, 30 with two, 23 with three) drawn by the counterfeit
  benchmark's generator with seed 20261203 (90 drawn, the first 60 used nowhere else kept), and
  their 360 cases. This is PREREG-COUNTERFEIT's design unchanged. Each case hands the route the
  law's counterfeits as candidates: the law beside one of each band of distance, or only wrong
  ones. Noise is 0, 1 % and 5 %, on 1,000 uniform rows. No search engine runs.
- **searched**: 40 laws of the invented-law grammar (6 with one input, 20 with two, 14 with
  three), seed 20261204. Each at noise 0, 1 % and 5 %, 120 runs, behind the search stage in force
  for tables (`stage_in_force.search_stage`: four searches side by side, 240 seconds, and the
  memory of reported laws), with PREREG-MEMORY's rows, split and scoring: 1,000 uniform rows, 70 %
  for development, 30 % as the pool, and 500 more without noise for scoring.

**Reports.** In every case and run the route searches, or is handed its candidates, once. The
route's report is then taken three ways: as it is (`route`), after the first version of the
checks (`first`) and after the second (`second`), with the same rows, pool inputs and
inseparable candidates. In the handed part the best fit and the best fit confirmed on fresh rows
are scored as well, as in PREREG-COUNTERFEIT.

**Scoring** is each part's earlier run's. Handed: right if it is the law, wrong if it is not and
below R² 0.999 on fresh rows. The score is (right − 19 × wrong) / 180. Searched: recovered if it
is the law after refitting constants, inaccurate if its test R² is below 0.999. A case or run that
crashes or exceeds its time limit (600 s handed, 1,500 s searched) reports nothing. Checks that
crash vouch for nothing.

## Expectations, fixed now

1. **What the second version passes can be relied on, handed:** of the laws `second` reports, at
   most 5 % are wrong at noise 0, at 1 % and at 5 %, each taken alone (`REFEREE_MAX_FPR`).
2. **It is worth having, handed:** `second` scores above `route`.
3. **It does not buy that with right laws, handed:** of the cases in which `route` reports the law,
   `second` reports it in at least 95 % at noise up to 1 % (one minus `REFEREE_MAX_FPR`) and in at
   least 75 % at 5 % noise (`REFEREE_MIN_VRR`).
4. **Nor behind a real search:** of the runs in which `route` recovers the law, `second` recovers it
   in at least 95 % at noise up to 1 % and in at least 75 % at 5 %.
5. **What it reports behind a search can be relied on:** of the laws `second` reports in the 120
   runs, at most 5 % are inaccurate.
6. Reported without a criterion: the first version beside the second throughout; every method by
   noise level; the reasons for every right law withheld; errors, timeouts and time.

The 95 % at noise up to 1 % is the lesson of PREREG-VOUCHED-SRBENCH. Where the route's reports
already hold, the checks may cost no more than a referee is allowed to get wrong. At 5 % noise,
where they hold back wrong laws, the canonical share a referee must retain applies.

No forecast file is filed: the system's forecaster predicts a route's recoveries, and has nothing
to say about checks applied after it.

**The author's own expectations**, recorded so that they can be wrong, and no criterion. The first
holds, about four chances in five: at 5 % noise about fifty reports, so three wrong ones break it,
and on the second campaign's worlds up to 5 % noise the second version still passed 98 wrong
laws. The second holds. The third holds, three chances in four, the 5 % part being the risk.
**The fourth is the open one, two chances in three**: on the physics benchmark the replay kept
192 of 193, but these laws are of the grammar the red team drew from, with sharper corners than
textbook formulas. The fifth holds.

A miss is reported as a miss.

## What follows from the result, fixed now

- **If all five hold,** the second version becomes the last step of the route in force for
  tables, in a new module around the frozen ones, with its cost and benefit here stated wherever
  the route is described, and M-34 and M-36 stated as open: it was tested up to 5 % noise
  only, and above that it is known to be weaker than the first version.
- **If 3 or 4 misses and the others hold,** it is not put in force. It stays available on
  request, described with the share of right laws it cost.
- **If 1, 2 or 5 misses,** it is not adopted. Every wrong or inaccurate law it passed is listed,
  and no variant is called measured until it has been run on laws drawn after it was written.
- Whatever happens, the first version's numbers here are recorded beside the second's.

## Disclosed before freezing

- **Both benchmarks and both campaigns are development material for this module.** It was
  written and changed while being replayed on them. Nothing here is a blind test of anything
  but the 100 laws above.
- **The searched part uses the search stage in force**, memory included, measured on tables in
  PREREG-PORTFOLIO-SEARCH and PREREG-MEMORY. Its laws come from the grammar the memory's way of
  joining parts was written for.
- **5 % noise behind a search has not been run before.** The route's own reports there are not
  measured to hold.
- **The laws file is the run's definition.** Its generator fits counterfeits by least squares and
  rounds their constants, and on one of GitHub's runners an earlier generator of this kind did not
  give its file byte for byte. The smoke job compares the two and reports the comparison; the run
  reads the file.
- **M-34 is not repaired** and a law with a pole in the domain can pass.
- **M-36 is not repaired**: above 5 % noise the second version lets through more wrong laws
  than the first (above). Above 5 % noise it is still better than the route in force, which has
  no last checks: 70 wrong against 126.
- **Unequal cost.** The second version takes about a third of a second for a law on 700 rows, as
  the first does.
- **The smoke job's law** is in no set.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
