# PREREG-VOUCHED-SRBENCH-2026-11: what do the table route's last checks cost on the physics benchmark, behind a real search?

**Status:** frozen before the run by `preregistration/DIGEST-VOUCHED-SRBENCH.json`. Runs once, on
GitHub Actions (`.github/workflows/srbench-vouched.yml`), started by the first branch named
`srbench-vouched-run-*` whose shards start.

## What is tested

The checks of `apedp_dyn/discovery_l2/vouching.py` became the last step of the route in force for
tables on 2026-11-22 (`discovery_l2/vouched_route.py`), on a run in which the candidates were
handed over and no search engine ran (PREREG-COUNTERFEIT-2026-11). That run fixed what has to be
said with them: *not measured behind a search engine or on the physics benchmark, where what they
cost in recovered laws is unknown until a run says.* This is that run.

The question is cost, not benefit. On this benchmark the route's reports already hold: in the
recorded run of the route in force (PREREG-PORTFOLIO-SEARCH-2026-11) 1 of 224 reported laws was
not accurate. The checks can add little there. What they can do is withhold laws that were
right: 208 of that run's 266 decisions came out of a class the path could not separate, and the
checks withhold a law whenever members of such a class pass their structure test and disagree,
or whenever an error large enough to make the law wrong could not have been seen.

## Data and protocol

Exactly PREREG-PORTFOLIO-SEARCH-2026-11's: SRBench's 133 ground-truth laws (119 Feynman, 14
Strogatz) at noise 0 and 1 % of the target's root mean square, 266 runs; 1,000 training rows of
which 70 % are development rows and 30 % the pool the path may measure, 500 test rows without
noise; the problems' physical units; four different searches side by side, 240 seconds each; a
run that crashes or exceeds its time limit counts as reporting nothing. A law is *recovered* if
it is the law after refitting constants, and *not accurate* if its test R² is below 0.999.

**Each run searches once and yields two reports that differ in nothing but the checks:**

- `route`: what `refined_search.discover_refined` reports, as in the recorded run;
- `vouched`: that report after `vouched_route.checked`, given the law, the candidates the path
  named as inseparable from it, the development rows and the pool's inputs. This is what the
  route in force for tables does.

So a law lost or gained between the two is the checks' doing. The `route` arm is a second run of
a recorded route; its totals may differ from the record (189 recovered, 224 reported, 1 not
accurate) by what a search changes from one run to the next, and that difference is reported.

## Expectations, fixed now

1. **What the checked route reports can be relied on:** of the laws `vouched` reports in the
   266 runs, at most 5 % are not accurate (the canonical `REFEREE_MAX_FPR`).
2. **The checks keep what a referee must keep:** of the runs in which `route` recovers the law,
   `vouched` recovers it in at least 75 % (the canonical `REFEREE_MIN_VRR`).
3. **They cost little where the route already holds:** that share is at least 95 %, so that the
   checks withhold a right law no more often than a referee may report a wrong one
   (1 − `REFEREE_MAX_FPR`).
4. Reported without a criterion: both arms by noise level, by family and on the 103 problems
   never used in development; every recovered law the checks lost, with the rule that withheld
   it; laws gained by passing a report to another member of its class; inaccurate laws
   withheld; errors, timeouts and time.

**What the earlier run would predict, taken at its word.** In PREREG-COUNTERFEIT's cases at
noise up to 1 % the route reported the law 111 times and the checks kept all 111; of 30
approximations they turned 4 into the law and withheld 5. On that showing the share kept here is
above 97 % and the third expectation holds. No forecast file is filed: the system's forecaster
has these problems' recorded outcomes, and nothing on record says what the checks do behind a
search.

**The author's own expectations**, recorded so that they can be wrong, and no criterion.

- The first holds. The checks can only withhold a report or pass it to another member of its
  class, and the route's own share was 1 in 224.
- The second holds, nine chances in ten.
- **The third is open: about three chances in four.** Without noise a right law fits exactly
  and passes untouched, so every loss will be at 1 % noise. There the route recovered 88 laws in
  the recorded run; I expect the checks to lose between two and ten of them, mostly laws whose
  spread is small beside their size, where 1 % noise hides an error of the size in question, and
  some to classes with many members. Nine lost of about 190 is the line.
- They gain one to three laws by passing a report to a member of its class that is the law.

A miss is reported as a miss.

## What follows from the result, fixed now

- **If 1, 2 and 3 hold,** the checks stay the last step of the table route, and the sentence
  that goes with them changes from "not measured behind a search engine" to this run's numbers.
- **If 2 misses,** the checks are taken out of the route in force: what asks for the table
  route gets the route without them again, through a new module, and `vouched_route` stays for
  whoever calls it. Every law they lost is listed with the rule that withheld it, and no variant
  goes back in force until it has been run on this benchmark and on laws drawn after it was
  written.
- **If 2 holds and 3 misses,** the checks cost more here than a referee is allowed to get wrong,
  on a benchmark where they have little to add. They stay in force until the repository's owner
  decides, the cost is stated first wherever the route is described, and the owner is asked.
- **If 1 misses** while the `route` arm's own share is within the limit, the checks made the
  reports less reliable here, and they are taken out of the route in force as under the second
  point.
  If the `route` arm's share is over the limit too, that is the route's failure on a second run
  of a recorded result and is recorded as such.
- The earlier run's prediction and the author's are compared with the result whatever happens.

## Disclosed before freezing

- **These problems are not new to the route.** It was developed on 30 of them and has been run
  on all 133 in three recorded runs. Nothing here is a blind test of the route. For the checks
  it is the first run on this benchmark: no setting of theirs was chosen on it, and they have
  been run on none of its problems.
- **The checks are as they were frozen** by PREREG-COUNTERFEIT's digest, with the weakness that
  run showed (audit M-33): their power rule is too confident for an error of another shape than
  those it plants. A repair exists as a sketch and is not in this run.
- **No benefit can be shown here.** With about one inaccurate law in two hundred reports, a
  drop in wrong laws is not measurable. The first expectation can only fail.
- **Two noise levels, and neither above 1 %.** Where the checks earn their keep, at 5 % noise,
  this benchmark has no runs.
- **The searches are not reproducible to the law.** Each has a time budget, so two runs of the
  same search can return different candidates. Both arms read the same search within a run, so
  the comparison between them is exact; the comparison with the recorded run is not.
- **Tables with more than seven inputs** would give the checks fewer rows than they were
  measured with. With 700 development rows no problem here is affected.
- **Unequal cost.** The checks add about a second to a run of about five minutes.
- **The smoke job** runs two of the 30 development problems with a short search.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
