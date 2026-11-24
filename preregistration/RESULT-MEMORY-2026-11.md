# Result: PREREG-MEMORY-2026-11

**Run:** once, 2026-11-22 01:18–03:21 UTC, GitHub Actions run `44970829582` on the branch
`memory-laws-run-1` at commit `5f9bd04`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-MEMORY.json`). The smoke job, 20
shards and the collect job all succeeded on the first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-MEMORY-2026-11/`](RESULT-MEMORY-2026-11/): `results.json` and
`runs.jsonl.gz` (160 rows: 40 laws, two noise levels, two arms). The collect job printed them to
its log with their checksums; they were rebuilt from the log and both match the SHA-256 the job
computed (`SHA256SUMS`).

## Outcome

**Both expectations held.** With the memory beside the four searches the route recovers 23 of
80 where it recovers 17 without it: six laws gained, none lost, and no wrong law in either arm.
The gain is almost all without noise.

No run ended in an error and none timed out.

| Arm | Recovered, no noise | Recovered, 1 % | Recovered of 80 | Law reported | Reported and wrong | Accurate, not the law |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default` (four searches) | 12 | 5 | 17 | 55 | 0 | 38 |
| `memory` (the same candidates and the memory's) | **17** | **6** | **23** | 57 | 0 | 34 |

95 % intervals (Clopper-Pearson) for wrong among reported: 0 to 6.5 % and 0 to 6.3 %. No wrong
law was seen; with 55 and 57 reports a rate above the limit is not excluded.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws each arm reports in its 80 runs, at most 5 % are wrong | 0 of 55 and 0 of 57 | **held** |
| 2 | The `memory` arm recovers at least as many laws as `default` at each noise level, and more in total | 17 and 12; 6 and 5; 23 and 17 | **held** |

## What follows, as fixed before the run

Expectations 1 and 2 held, so **the memory is put beside the default search stage for tables**
(`apedp_dyn/discovery_l2/stage_in_force.py`, a new module around the frozen ones), and this
result is stated wherever that stage is described.

What the result does not license was disclosed before the run and stands: the way the memory
joins parts was written knowing this grammar, the memory holds laws of the same family, and
laws with several variables inside one function, which most of the physics benchmarks' laws
are, cannot be grown from one-variable parts.

These 40 laws have now been run on and are development material from here on.

## Reported without a criterion

- **The memory's laws were the law itself in 26 of the 80 runs** by the runner's check: 15
  without noise, 11 at 1 %.
- **Without noise the route reported all 15.** The default arm recovered 11 of the same 15, so
  four of the gains are laws the searches had missed and the memory supplied. A fifth gain
  without noise (`memory_4033`) is a law the runner's check did not find among the memory's
  proposals, although both arms were handed the same searched candidates.
- **At 1 % noise the route reported 6 of the 11.** In the other five the memory had handed it
  the law, and it reported another candidate: each time the path could not tell its candidates
  apart on the rows it may measure (`equivalent_class`) and named the one with the shortest
  description, which was accurate on the scoring rows (R² 0.99997 or better) and not the law.
  For `-3.7*x0 + 3.7/x0` it reported `-1.016*x0*log(x0) - 5.944*log(x0)`. The memory arm gains
  one law at this noise level.
- **At 1 % noise most of what the route reports is not the law, in both arms.** Of 36 laws
  reported by each arm, 5 and 6 are the law and 31 and 30 are accurate without being it. None
  is wrong by the benchmark's criterion, which asks for R² of 0.999 on rows without noise.
- **Run for run:** gained 5 and 1 (no noise, 1 %), lost none. The six: `memory_4004`,
  `memory_4010`, `memory_4013`, `memory_4031`, `memory_4033` without noise, `memory_4021` at 1 %.
- **By number of inputs** (runs; recovered by `default`, by `memory`; the law among the
  memory's proposals): one input, 18 runs: 4, 6; 8. Two inputs, 38 runs: 11, 13; 13. Three
  inputs, 24 runs: 2, 4; 5.
- **The path's actions:** `default`: `equivalent_class` 48, `widen_search` 25, `confirmed` 7.
  `memory`: `equivalent_class` 50, `widen_search` 23, `confirmed` 7.
- **The memory's growing** handed over three laws in every run; it stopped at its size limit in
  60 runs, on an exact fit in 10 and out of time (60 s) in 10.
- **Time:** the searches and the `default` arm, median 282 s a run; the `memory` arm on the
  recorded candidates, median 63 s.

## The system's own forecast, scored

Filed before the run for the `default` arm (`FORECAST-MEMORY-2026-11.json`); scored by
`scripts/forecast_runs.py score` into `benchmarks/outcome_forecast_scored.json`.

| | Forecast (80 % interval) | What happened | Knowing only the earlier runs' averages |
| :--- | ---: | ---: | ---: |
| Recovered | 50 (37 to 62) | **17** | 57 |
| Law reported | 64 (52 to 76) | 55 | 67 |
| Reported and wrong | 0.4 (0 to 12) | 0 | 0.6 |

**The forecast of recoveries missed, far outside its interval.** It had been learned from runs
on the physics benchmarks and had seen no run on invented laws; the preregistration said so,
and its author expected it to be too high. Run by run it still ranked the laws better than
chance (0.68 on a scale where 0.5 is chance), and it was closer than the earlier runs' average.

## The author's expectations, recorded before the run

- "The forecast is too high; about 26 of 80 for `default`": 17. The forecast was too high, and
  so was the author's own number.
- "The memory hands over the law in about 12 of 80 runs, about half of them laws the searches
  find anyway": 26 runs, of which the default arm recovered 16.
- "The memory arm gains 4 to 8 laws and loses one or two": gained 6, lost none.
- "Expectation 2 holds, with perhaps one chance in three that it does not": held.
