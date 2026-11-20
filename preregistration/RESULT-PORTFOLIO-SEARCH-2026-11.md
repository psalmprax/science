# Result: PREREG-PORTFOLIO-SEARCH-2026-11

**Run:** once, 2026-11-20 12:32–14:53 UTC, GitHub Actions run `44736656957` on the branch
`srbench-portfolio-run-1` at commit `6f3b8bd`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-PORTFOLIO-SEARCH.json`). The smoke job, 32
shards and the collect job all succeeded on the first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-PORTFOLIO-SEARCH-2026-11/`](RESULT-PORTFOLIO-SEARCH-2026-11/):
`results.json`, `runs.jsonl` (266 runs) and the 32 shard files. The collect job printed them to its
log with their checksums; they were rebuilt from the log and all 34 match the SHA-256 the job
computed.

## Outcome

**All four expectations held, so the portfolio becomes the default route's search stage wherever
four cores are available.** On the 103 problems it was not built on, four different searches side
by side recovered 73 laws without noise and 63 at 1 %, against 61 and 61 for one search. It kept 118
of the single search's 122 recoveries there, and 1 of its 224 reported laws is not accurate.

SRBench's 133 problems × 2 noise levels (0 and 1 %). One run ended in an error (below); none timed
out; no search failed.

| | Runs | Law reported | Reported and not accurate | Recovered (by form) | Refused (`widen_search`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All, both noise levels | 266 | 224 | **1** | **189** | 41 |
| Noise 0 | 133 | 109 | 0 | 101 | 23 |
| Noise 1 % | 133 | 115 | 1 | 88 | 18 |
| Feynman, noise 0 | 119 | 97 | 0 | 89 | 21 |
| Feynman, noise 1 % | 119 | 102 | 1 | 78 | 17 |
| Strogatz, both | 28 | 25 | 0 | 22 | 3 |

**No recovery passes the tolerance narrowly:** all 189 have a refit error below 1e-10, so the
benchmark's own count and the count by form are the same (189). 95 % interval (Clopper-Pearson) for
not accurate among reported: 0–2.5 %.

Beside the earlier results on the same problems:

| | Recovered, all 266 runs | Feynman, noise 0 (of 119) | Feynman, noise 1 % | Strogatz (of 28) | Reported and not accurate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Four searches** (this run) | **189** | **89** | **78** | **22** | 1 of 224 |
| Refined path, one search (PREREG-REFINED-SEARCH) | 174 | 78 | 77 | 19 | 0 of 212 |
| Main search path (PREREG-MAIN-SEARCH) | 152 | 69 | 70 | 13 | 4 of 186 |

The 103 held-out problems:

| | Noise 0 | Noise 1 % |
| :--- | ---: | ---: |
| Four searches | **73** | **63** |
| Refined path, one search, same problems | 61 | 61 |
| Gained / lost, run for run | 13 / 1 | 5 / 3 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws reported in all 266 runs, at most 2 % not accurate | 1 of 224, 0.45 % | **held** |
| 2 | On the 103 held-out problems at noise 0, at least 63 recovered by form | 73 | **held** |
| 3 | On the 103 at 1 % noise, at least 59 recovered by form | 63 | **held** |
| 4 | Of the 122 held-out runs one search recovered, at least 116 recovered again by form | 118 | **held** |

**What follows, as fixed before the run:** expectations 1, 2 and 3 held, so this portfolio becomes
the search stage of the default route wherever four cores are available, in a new module around the
frozen ones. That module is `apedp_dyn/discovery_l2/search_stage.py` (added with this result).

## Reported without a criterion

- **The 30 development problems:** 28 and 25 recovered, as in development; 55 laws reported, none
  inaccurate.
- **The one law reported and not accurate:** `feynman_I_30_5` at 1 % noise (the law is
  `asin(lambda/(n*d))`): a long rational form in sines and cosines that is not the law and fits
  the test rows to R² 0.993, under the 0.999 line.
- **Searches that failed:** none, in 213 runs where the search stage ran (it is skipped when the
  units leave no product to search over).
- **Distinct candidates from the four searches:** median 25 per run; ten go on to confirmation.
- **Which member found each law:** not separable from the records. A candidate is recorded only as
  coming from the search stage. Every run's first merged slot came from the first member, the
  single run's own search, as the round-robin merge places it.
- **The path's actions:** `confirmed` 16, `equivalent_class` 208, `widen_search` 41 (one search: 54
  refusals). Terms dropped from the confirmed form in 13 runs. No tentative laws.
- **Time:** median 284 s per run, the slowest 516 s (limit 720 s); the search stage's wall time had
  a median of 266 s for a 240 s budget.

## After the run (not preregistered)

**One run ended in an error, and it was a held-out recovery of the single search.**
`feynman_III_15_12` without noise ended with `LinAlgError: SVD did not converge in Linear Least
Squares` while one candidate was being fitted, the failure found the same day in the ODE run (audit
M-24). It is the one held-out loss at noise 0. A guard that drops an unfittable candidate would have
given that run back to the search; it is not written.

**Where the gain is.** Without noise, 13 held-out laws gained and 1 lost; at 1 % noise, 5 gained and
3 lost. Among the gains is `feynman_I_10_7` (the relativistic mass), which the language-model
controller's run had passed only narrowly with an approximation; here it is recovered by form.
Refusals fell from 54 to 41: the path more often had a candidate that settled the comparison.

**What this does and does not say.** Four cores used as four different searches recover more
laws, on problems they were not built on, without making the report less reliable. The comparison
is four cores against one for the same wall time; it is not a claim that the method is better at
equal compute. Which of the four members does the work is not separated. The ODE run the same day
measured four seeds of one search and was not adopted; that is a different portfolio on different
rows, and this result does not carry over to it.
