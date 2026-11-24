# PREREG-REFINED-SEARCH-2026-11: the main search path with units, a polynomial library and a stricter report

**Status:** frozen before the run by `preregistration/DIGEST-REFINED-SEARCH.json`. Runs once, on
GitHub Actions (`.github/workflows/srbench-refined.yml`), started by the first branch named
`srbench-refined-run-*` whose shards start.

## What is tested

PREREG-MAIN-SEARCH-2026-11 measured the main search path on SRBench: 152 of 266 runs recovered, 4
of 186 reported laws not accurate, 78 runs refused. That run and the ODE run after it showed four
things the path does not do. `apedp_dyn/discovery_l2/refined_search.py` does them around the frozen
path, without editing it:

1. the search stage works on the dimensionless products of the inputs when the problem carries
   units (`proposers/composed.UnitsReducingProposer`);
2. a sparse polynomial fit is offered as one more candidate (`PolynomialLibraryProposer`);
3. the law reported is the form the path confirmed, with every term dropped that the data do not
   notice (audit M-13, M-15);
4. the search stage runs on every problem, and a law that nothing challenged is returned as
   tentative, not reported (M-14).

## Data and protocol

Exactly as PREREG-MAIN-SEARCH-2026-11: SRBench's 133 problems, noise 0 and 1 %, the same training
and test rows, the same 70 % development rows and 30 % pool, PySR with 240 s, the same scoring. 266
runs. The units are those in `benchmarks/srbench/manifest.json` (from the Feynman dataset's own
table; the Strogatz problems have none). A reported law is one the path confirmed against rivals
(`confirmed` or `equivalent_class`); a tentative law counts as not reported and not recovered.

The comparison is with the recorded result of PREREG-MAIN-SEARCH, run for run.

**Development and held-out problems.** Thirty problems (those whose name hashes to 0 modulo 5 under
the tag in `scripts/run_srbench_refined.py`) were used while building this; the other 103 were not
run through any part of it. The expectations below that compare with the main path are stated on
the 103.

## Expectations, fixed now

1. **What it reports can be relied on:** of the laws reported in all 266 runs, at most 5 % are not
   accurate.
2. **It recovers more, on problems it was not built on:** on the 103 held-out problems it recovers
   at least 62 without noise and at least 58 at 1 % noise (the main path: 55 and 52).
3. **The known failure is gone:** on `strogatz_lv1`, where the main path reported a law with test
   R² 0.35 at both noise levels, no inaccurate law is reported at either.
4. Reported without a criterion: the 30 development problems, how many problems the units reduced
   and to how many variables, laws returned as tentative, terms dropped, search stages that
   failed, the Feynman totals beside the main path's 69 and 70, and time.

A miss is reported as a miss.

## Disclosed before freezing

- **Development.** The 30 development problems were run with the first passes only (no PySR),
  with and without this module, locally. At the time of freezing 23 problem-and-noise pairs had
  finished both ways: 12 recovered without and 19 with, none lost, no law returned that was
  neither recovered nor accurate. PySR was in the loop for three development runs in CI (all three
  recovered). The two problems that timed out in PREREG-MAIN-SEARCH (`feynman_II_11_20`,
  `feynman_II_11_3`) are development problems.
- **Expectation 3 is a check on a known case**, not a blind prediction: rule 4 above was written
  because of `strogatz_lv1`. That problem is not among the 30, and it has not been run through
  this module.
- **The recorded main-path result is the baseline**, and it was known when the numbers in
  expectation 2 were chosen (55 and 52 on the same 103 problems). The margins are small on
  purpose; the development gain was larger, and development is where gains are overstated.
- **The comparison is not only units.** Four things change at once, and the search stage now runs
  on every problem (the main path skipped it in 60 runs its first pass had settled). This run
  measures the module as a whole. Which part does what is not separated.
- **Units are information the main path did not have.** A method that uses them is not being
  compared like for like with one that does not; the published methods that use units on these
  problems are the fair comparison, and none is run here.
- A law with a hidden dimensional constant is not of the form the units wrapper searches. The
  Feynman problems give every constant as a variable, so the benchmark does not test that case.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
