# PREREG-PORTFOLIO-SEARCH-2026-11: four different searches side by side as the refined path's search stage

**Status:** frozen before the run by `preregistration/DIGEST-PORTFOLIO-SEARCH.json`. Runs once, on
GitHub Actions (`.github/workflows/srbench-portfolio.yml`), started by the first branch named
`srbench-portfolio-run-*` whose shards start.

## What is tested

PREREG-REFINED-SEARCH-2026-11 measured the refined path on SRBench with one PySR search, serial,
240 seconds on one core of a four-core runner: 212 laws reported in 266 runs, none inaccurate, 174
recovered. Here the search stage is four searches at the same time, each in its own process on its
own core, their candidates merged before confirmation (`apedp_dyn/proposers/portfolio.py`). Nothing
else changes. The four (`apedp_dyn/proposers/deep_pass.py`, `mixed_members`):

1. the refined run's own PySR search, with its seed;
2. the exhaustive first pass at a larger size limit (8 nodes for one variable, 7 for two or
   three, nothing for more), no Julia;
3. a PySR search with algebraic operators only (`sqrt`, `square`, and the four arithmetic ones);
4. the default PySR search with the next seed.

The question is whether three more cores, used this way, recover laws that one search does not, on
problems this was not built on, without reporting anything wrong and without losing what one
search finds.

## Data and protocol

Exactly as PREREG-REFINED-SEARCH-2026-11: SRBench's 133 problems, noise 0 and 1 %, the same
training and test rows, the same development rows and pool, the same units, 240 s per search, the
same scoring. 266 runs. The time limit per run is 720 s (the refined run: 600 s), because four
runtimes have to start. A reported law is one the path confirmed against rivals (`confirmed` or
`equivalent_class`); a tentative law counts as not reported and not recovered.

**Two readings of "recovered".** The benchmark's criterion refits a law's constants to the clean
law on the test rows and accepts a relative error below 1e-8. A form that only follows the law on
the sampled range can pass it narrowly (audit M-23). *Recovered by form* means recovered by that
criterion with an error of at most 1e-10 (`STRICT_REL_RMSE` in the runner). The expectations below
are judged by form. The benchmark's own count is reported beside each, and every narrow pass is
listed by name.

The comparison is with the recorded refined run, run for run. That run has no narrow pass, so its
numbers are the same under both readings.

**Development and held-out problems.** The thirty development problems of the refined path (those
whose name hashes to 0 modulo 5 under the tag in `scripts/run_srbench_refined.py`) were used to
build this. The other 103 were not run through any part of it. The expectations are stated on the
103.

## Expectations, fixed now

1. **What it reports can be relied on:** of the laws reported in all 266 runs, at most 2 % are
   not accurate.
2. **It recovers more without noise, on problems it was not built on:** on the 103 held-out
   problems at noise 0 it recovers at least 63 by form (the refined run: 61).
3. **It is not worse with noise:** on the 103 at 1 % noise it recovers at least 59 by form (the
   refined run: 61).
4. **It keeps what was found:** of the 122 held-out runs the refined run recovered, at least 116
   are recovered again by form.
5. Reported without a criterion: the benchmark's own counts, narrow passes, the 30 development
   problems, searches that failed and in how many runs, how many distinct candidates the four
   proposed, which members' candidates were merged, the path's actions, timeouts and time.

A miss is reported as a miss.

## What follows from the result, fixed now

- If expectations 1, 2 and 3 hold, this portfolio becomes the search stage of the default route
  wherever four cores are available, in a new module around the frozen ones.
- If expectation 1 misses, it is not adopted, whatever it recovers.
- If expectation 2 misses, the result is recorded as: four different searches recover no more
  than one on new problems. The portfolio stays available and is not the default, and more cores
  are not offered as a way to recover more laws from tables until a different member is proposed.
- If expectation 2 holds and expectation 3 or 4 misses, the gain without noise is paid for with
  noise or with lost laws. It is documented as an option for clean data and is not the default.

## Disclosed before freezing

- **Two development rounds on the same 30 problems (60 runs each, four-core runners).** Round
  one, four seeds of the same search: 27 and 25 recovered, exactly the runs one search recovers.
  That portfolio was dropped. Round two, the four members above: 29 and 25 by the benchmark's
  criterion, 28 and 25 by form, against 27 and 25. Three runs gained (`feynman_test_5` and
  `feynman_I_41_16` without noise, `feynman_II_11_3` at 1 %), one lost (`feynman_I_6_2a` at 1 %,
  where the path could not settle between more rivals). 54 laws reported, none inaccurate. No
  search failed. The slowest run took 511 s.
- **One of the three gains is not the law.** For `feynman_I_41_16` (Planck's law) the reported
  form is a cubic in the dimensionless product plus a cosine of a cosine; refitted, its error is
  9.2e-9, under the benchmark's tolerance of 1e-8. That is why the expectations are judged by
  form. The rule (an error of at most 1e-10) was written after seeing this case and after looking
  at the record: real recoveries there sit below 1e-12, the two narrow passes found (both in
  PREREG-LLM-CONTROLLER's run, both on held-out problems, `feynman_I_10_7` and
  `feynman_III_4_33`) above 1e-9. Their formulas were read for that audit, after the four members
  were fixed.
- **The net gain in development is small:** one run of 30 without noise by form, none at 1 %.
  Expectation 2 asks for two more of 103. It can miss.
- **How the members were chosen.** The deeper pass after a local experiment on the 18 development
  runs the first passes miss: alone it recovered 3 (PySR: 10), one of them a law no PySR run here
  had recovered, and at 1 % noise it returned four laws that fit and are not the law. The
  algebraic search and the second seed were chosen without an experiment of their own. What each
  member contributes is not separated by this run, beyond which member's candidates were merged.
- **Four times the compute.** Four cores for the same wall time are compared with one core. The
  comparison says what the extra cores buy, not that the method is better at equal cost. The
  first member has the refined run's seed, but shares the machine with three other processes and
  may complete fewer iterations in its 240 s than it did alone.
- **The held-out problems are the comparison set of three earlier recorded runs** (the main
  search, the refined path, the language-model controller). Their aggregate results on the 103
  (how many missed, and with which action) were looked at while planning this. No held-out
  problem's own result was used to choose a member or a setting.
- Units are information the search uses, as in PREREG-REFINED-SEARCH; the published methods that
  use units on these problems are not run here.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
