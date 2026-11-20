# PREREG-ODE-REFINED-2026-11: ODE systems through the refined path, with four searches side by side

**Status:** frozen before the run by `preregistration/DIGEST-ODE-REFINED.json`. Runs once, on
GitHub Actions (`.github/workflows/odebench-refined.yml`), started by the first branch named
`odebench-refined-run-*` whose shards start.

## What is tested

PREREG-ODE-MAIN-SEARCH-2026-11 measured the main search on ODEBench, one component of
`dx/dt = f(x)` at a time: 80 and 69 of 117 components recovered at noise 0 and 1 %, and 9 of 184
reported laws wrong. Two changes are tested together here, around that frozen route and without
editing it:

1. each component goes through the refined path (`apedp_dyn/ode/refined_ode.py`): a sparse
   polynomial fit is one more candidate, the law reported is the form the path confirmed with
   every unnoticed term dropped, and a law that nothing challenged is returned as tentative, not
   reported;
2. the search stage is four PySR searches with different seeds, each in its own process on its
   own core, merged before confirmation (`apedp_dyn/proposers/portfolio.py`). The first has the
   recorded run's seed.

## Data and protocol

Exactly as PREREG-ODE-MAIN-SEARCH-2026-11: ODEBench's 63 systems (117 components), noise 0 and
1 %, the same trajectories, noise realisations and window rows (the worlds are rebuilt from that
run's tag), development rows from two trajectories and the pool from a third, 240 s per search,
the same scoring. 234 runs. The time limit per run is that run's 480 s plus 120 s, because four
Julia runtimes have to start. A reported law is one the path confirmed against rivals
(`confirmed` or `equivalent_class`); a tentative law counts as not reported and not recovered. A
law is *wrong* as in that preregistration: reported, not accurate on the test trajectory, and not
the law.

The comparison is with the recorded result of PREREG-ODE-MAIN-SEARCH's `main_search` engine, run
for run.

**Development and held-out systems.** Fifteen systems (31 components; those whose id hashes to 0
modulo 4 under the tag in `scripts/run_odebench_refined.py`) were used while building this. The
other 48 systems (86 components) were not run through any part of it. The expectations that
compare with the recorded route are stated on the 86.

## Expectations, fixed now

1. **What it reports can be relied on:** of the laws reported in all 234 runs, at most 5 % are
   wrong.
2. **It recovers more, on systems it was not built on:** of the 86 held-out components it
   recovers at least 60 without noise and at least 49 at 1 % noise (the recorded route: 56 and
   48).
3. **It keeps what was found:** of the 104 held-out runs the recorded route recovered (56 and
   48), at least 98 are recovered again.
4. Reported without a criterion: the 31 development components, laws returned as tentative, terms
   dropped, how many searches failed and in how many runs, how many distinct candidates the four
   searches proposed, systems with every component recovered, the path's actions, and time.

A miss is reported as a miss.

## What follows from the result, fixed now

- If expectations 1 and 2 both hold, `refined_ode` with four searches is the ODE route the
  documentation gives first and new callers are pointed to; the recorded route stays selectable.
- If expectation 1 misses, it is not adopted, whatever it recovers.
- If expectation 1 holds and expectation 2 misses, it is recorded as no better than the recorded
  route on new systems, and the four searches are not made the default anywhere on this evidence.

## Disclosed before freezing

- **Development, three passes over the same 15 systems.** First passes alone, locally: 24 of 31
  at noise 0, all correct; at 1 % noise 16 the law and 4 not accurate, which is why a law found
  without a search stage is tentative. One PySR search, in CI: 25 and 22 recovered, against 24 and
  21 for the recorded route; that gain was judged too small to predict. Four searches, in CI: 28
  and 23 recovered; all 45 runs the recorded route recovered were recovered again; 57 laws
  reported, 2 not accurate, 1 wrong (the recorded route on the same components: 51, 2, 1); no
  search failed; 62 of 62 runs finished, the slowest in 445 s.
- **The number of searches was chosen on those systems**, after one search had shown one more
  component at each noise level. Expectation 2 asks for 4 more of 86 without noise and 1 more at
  1 %; development showed 4 and 2 more of 31. The margins are smaller than development's on
  purpose.
- **The same idea gained nothing on tables.** On SRBench's 30 development problems, four seeds of
  the same search recovered exactly the runs one search recovers (27 and 25 of 30). Why seeds help
  on window rows from trajectories and not there is not understood, so expectation 2 can miss.
- **Expectation 1 is close to the recorded result.** The recorded route's wrong laws were 9 of
  184 (4.9 %), and 8 of 133 on the 86 held-out components. Development did not remove a wrong
  law; it added correct ones. This expectation is the standing design criterion, not a comfortable
  prediction.
- **The recorded result is the baseline and was known** when the numbers above were chosen.
- **Two things change at once.** Which of the two does what is not separated by this run; the
  development passes above are the only evidence on that.
- Seven ODEBench systems are the Strogatz systems that SRBench also carries. The refined path was
  built on SRBench's development problems, one of which is a Strogatz problem
  (`strogatz_shearflow2`).
- Rows from one trajectory are a weak test of adequacy (audit M-14). Nothing here integrates a
  candidate system forward.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
