# Result: PREREG-ODE-REFINED-2026-11

**Run:** once, 2026-11-20 11:23–13:04 UTC, GitHub Actions run `44727147894` on the branch
`odebench-refined-run-1` at commit `0809693`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-ODE-REFINED.json`). The smoke job, 32
shards and the collect job all succeeded on the first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-ODE-REFINED-2026-11/`](RESULT-ODE-REFINED-2026-11/): `results.json`,
`runs.jsonl` (234 runs) and the 32 shard files. The collect job printed them to its log with their
checksums; they were rebuilt from the log and all 34 match the SHA-256 the job computed.

## Outcome

**Not adopted.** The route recovers more than the recorded ODE route and keeps nearly all of what
that route found, but 10 of the 195 laws it reports are wrong, 5.1 %, and the limit fixed in
advance was 5 %. The preregistration says that if this expectation misses the route is not adopted,
whatever it recovers.

ODEBench's 63 systems, 117 components × 2 noise levels (0 and 1 %). Two runs ended in an error (a
least-squares fit that did not converge, below); none timed out; no search failed.

| | Runs | Law reported | Reported and wrong | Recovered (the law) | Refused (`widen_search`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All, both noise levels | 234 | 195 | **10** | **165** | 37 |
| Noise 0 | 117 | 99 | 1 | 92 | 17 |
| Noise 1 % | 117 | 96 | 9 | 73 | 20 |

The recorded ODE route (PREREG-ODE-MAIN-SEARCH, `main_search`), same runs:

| | Law reported | Reported and wrong | Recovered, noise 0 | Recovered, noise 1 % | Systems with every component recovered |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **This run** | 195 | 10 (5.1 %) | **92** | **73** | 43 and 29 of 63 |
| Recorded route | 184 | 9 (4.9 %) | 80 | 69 | 35 and 27 of 63 |

95 % intervals (Clopper-Pearson) for wrong among reported: 2.5–9.2 % here, 2.3–9.1 % for the
recorded route. The two rates cannot be told apart; the limit was a fixed number and it was passed.

The 86 held-out components (48 systems), which no part of this was run on before:

| | Noise 0 | Noise 1 % |
| :--- | ---: | ---: |
| This run | **63** | **50** |
| Recorded route, same components | 56 | 48 |
| Gained / lost, run for run | 9 / 2 | 4 / 2 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws reported in all 234 runs, at most 5 % are wrong | 10 of 195, 5.1 % | **missed, by one law** |
| 2 | Of the 86 held-out components, at least 60 recovered without noise and at least 49 at 1 % | 63 and 50 | **held** |
| 3 | Of the 104 held-out runs the recorded route recovered, at least 98 recovered again | 100 | **held** |

Nine wrong laws would have been 4.6 %. Expectation 1 is a miss.

**What follows, as fixed before the run:** expectation 1 missed, so `refined_ode` with four
searches is not the ODE route the documentation gives first, and it is not made a default
anywhere. The recorded route stays where it is.

## Reported without a criterion

- **The 31 development components:** 29 and 23 recovered (development had 28 and 23); 57 laws
  reported, 2 not accurate, 1 wrong.
- **Tentative laws:** none. Every run had a search stage.
- **Terms dropped from the confirmed form:** in 9 runs.
- **Searches that failed:** none, in 232 runs with a search stage.
- **Distinct candidates from the four searches:** median 32 per run (8 from each in 214 runs); ten
  go on to confirmation.
- **The path's actions:** `confirmed` 6, `equivalent_class` 189, `widen_search` 37. Which stage
  found the reported law: the search 183, the first pass 12.
- **Time:** median 274 s per run, the slowest 462 s (limit 600 s); the search stage's wall time
  had a median of 266 s for a 240 s budget.

## After the run (not preregistered)

**By form.** No recovery passes the tolerance narrowly: every one of the 165 has a refit error
below 1e-10 (audit M-23). The counts above are the same under both readings.

**The wrong laws are the recorded route's wrong laws.** Eight of its nine are wrong again, on the
same components, with nearly the same forms: seven at 1 % noise, most of them a sine or a power
standing in for the law's own terms (test R² 0.990 to 0.999, under the 0.999 line), and
`dx/dt = -x**3` without noise, reported as `-x**3 - 4.4e-9*exp(exp(x))`, exact on the rows and
unbounded beyond them. The ninth (`48, 1` at 1 %) is now recovered. Two are new, both at 1 % noise
on components the recorded route had refused: `49, 1` and `53, 1`. So the refined report and three
more searches did not remove a wrong law; they added recoveries and two wrong laws.

**The pruning did not do what it was written for on these rows** (audit M-15). The `exp(exp(x))`
term above survived: the path dropped a constant and kept it. One held-out law was lost the same
way (`61, 1` without noise: `x0*x2 - 10*x1` plus two terms with coefficients of 1e-16 and 1e-12,
scored as not the law because of them). The pruning rule is the same one that reported nothing
inaccurate on SRBench; on window rows it leaves terms that are small on the rows and not small
elsewhere.

**Two runs ended in an error, and both were recoveries of the recorded route** (`12, 0` without
noise, `54, 0` at 1 %). `numpy.linalg.lstsq` did not converge while one candidate was being fitted
in `contrast_confirmation.fit_candidate`, and the exception ended the run. With four searches
there are more candidates, and one that cannot be fitted is more likely. A candidate that cannot be
fitted should be dropped, not end the run (audit M-24). These two account for half of the four
held-out recoveries lost.

**Where the gain is.** Without noise: 14 components gained and 2 lost over all 117; at 1 %: 6 and
2. Systems with every component recovered: 43 against 35 without noise, 29 against 27 at 1 %.
Laws reported that are accurate and are not the law: 20 against 26.

**What this says.** Four searches side by side recover more ODE components on systems they were
not built on (seven more of 86 without noise, two more at 1 %), which is what development showed.
They do not make the report more reliable, and reliability is the criterion the route had to meet.
The two things to repair before another attempt are named above: the pruning on window rows and
the fit that ends a run. Neither is repaired here.
