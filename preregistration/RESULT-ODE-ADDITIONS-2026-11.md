# Result: PREREG-ODE-ADDITIONS-2026-11

**Run:** once, 2026-11-22 01:26–04:18 UTC, GitHub Actions run `44971460220` on the branch
`ode-additions-run-1` at commit `27590f8`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-ODE-ADDITIONS.json`). The smoke job, 30
shards and the collect job all succeeded on the first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-ODE-ADDITIONS-2026-11/`](RESULT-ODE-ADDITIONS-2026-11/):
`results.json` and `runs.jsonl.gz` (1,932 rows: 276 units, seven rows each). The collect job
printed them to its log with their checksums; they were rebuilt from the log and both match the
SHA-256 the job computed (`SHA256SUMS`).

## Outcome

**All four expectations held, and the gains are small.** On 46 systems none of this was built
on, the two additions recover 2 components more than the route in force (118 against 116 of
276) and the follow-up experiment 5 more (123). The share of reported laws that are wrong rose
from 2.4 % to 3.2 %, inside the 5 % fixed in advance. On the development systems the same two
steps had been worth 26 and a further 41 components of 620.

No unit ended in an error and none timed out.

| Arm | Recovered, no noise | Recovered, 1 % | Recovered of 276 | Law reported | Reported and wrong |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `default` (the route in force) | 94 | 22 | 116 | 164 | 4 (2.4 %) |
| `second_engine` | 94 | 23 | 117 | 165 | 3 (1.8 %) |
| `precision` | 94 | 23 | 117 | 186 | 6 (3.2 %) |
| `both` | 94 | 24 | 118 | 188 | 6 (3.2 %) |
| `experiment` | **99** | 24 | **123** | 188 | 6 (3.2 %) |
| `experiment_random` | 97 | 24 | 121 | 188 | 6 (3.2 %) |
| sparse reference (`sparse_poly3`) | 95 | 6 | 101 | 276 | 127 (46 %) |

95 % intervals (Clopper-Pearson) for wrong among reported: 0.7–6.1 % for `default`, 1.2–6.8 %
for `both` and for `experiment`. The limit was a fixed number and it was not passed; the
intervals do not exclude a rate above it.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws `both` reports, and of those `experiment` reports, at most 5 % are wrong | 6 of 188 each, 3.2 % | **held** |
| 2 | `both` recovers at least as many components as `default` at each noise level | 94 and 94; 24 and 22 | **held**, by a tie and by two |
| 3 | `experiment` recovers at least as many as `both` at each noise level, more in total, and reports no more wrong laws | 99 and 94; 24 and 24; 123 and 118; 6 and 6 | **held** |
| 4 | `experiment` recovers more components in total than `experiment_random` | 123 and 121 | **held, by two components** |

The preregistration disclosed that the machinery itself varies by about one law between
machines and that a difference of one component between arms is not a finding. Expectation 4
holds by two: three components the chosen restarts recovered and the random ones did not, one
the other way round. It is recorded as held; it does not show that choosing where to measure
matters.

## What each addition did, read from the rows

This section is analysis after the run. No criterion was fixed for it.

**The second engine** changed what is reported in nine runs, eight of them at 1 % noise: one
approximation became the law, one wrong law was no longer reported, two runs that had reported
nothing reported an approximation, and five approximations were replaced by other
approximations.

**Adequacy at the rows' precision reinstated 23 laws that the path's last check had refused, and
one of them is the law.** Nineteen are accurate without being the law and three are wrong
(GlycolyticOscillation component 1 without noise, Invented11 component 0 and MarhlCalcium
component 2 at 1 %). The route in force had reported nothing in all 23 runs. On the development
systems the same check had refused the true law in 20 of the 28 runs it refused, which is what
the addition was built on; on these systems what it refused was, 22 times in 23, rightly
refused.

**The experiment** ran in the 184 runs that had a reported law of more than one term, with 16
restarts each; all 16 stayed bounded in 157 of them (chosen) and 148 (random). It shortened 9
laws with chosen restarts and 3 with random ones. Of the 9: five became the law, one stayed an
approximation and three wrong laws stayed wrong. All five gains are without noise; at 1 % it
changed the outcome of no run. Two of the five were laws the reinstatement had put back.

**What the route reports, by kind:**

| | The law | Accurate, not the law | Wrong | Nothing |
| :--- | ---: | ---: | ---: | ---: |
| `default`, no noise | 94 | 6 | 1 | 37 |
| `default`, 1 % | 22 | 38 | 3 | 75 |
| `both`, no noise | 94 | 10 | 2 | 32 |
| `both`, 1 % | 24 | 54 | 4 | 56 |
| `experiment`, no noise | 99 | 5 | 2 | 32 |
| `experiment`, 1 % | 24 | 54 | 4 | 56 |

At 1 % noise two of every three laws the route reports are approximations that are accurate on
the test trajectory and are not the law, in every arm.

**The second engine and the experiment without the reinstatement.** In 253 of the 276 units the
`second_engine` arm and the `both` arm report the same law, so the experiment was run on that
very law; in the other 23 `both` reports only through reinstatement. The outcome of the second
engine followed by the experiment, without reinstatement, can therefore be read off the rows
unit by unit: 97 and 23 recovered (120), 165 laws reported, 3 wrong (1.8 %). It was not an arm
of this run.

## What follows

As fixed before the run:

- Expectations 1 and 3 held, so **the experiment becomes the default follow-up wherever the
  caller can restart the system**, with chosen restarts. Expectation 4 held by two components,
  and the choosing is not claimed to matter.
- Expectations 1 and 2 held, so **the second engine becomes part of the default ODE route**
  (`apedp_dyn/ode/route_in_force.py`, a new module around the frozen ones).

**Deviation, on the cautious side, recorded 2026-11-22.** The preregistration fixed that if 1
and 2 held, adequacy at the rows' precision would become part of the default route as well. It
is **not** made part of it. The expectations were written for an arm as a whole and both hold;
the rows show that what this addition adds on its own is 23 reported laws of which one is the
law and three are wrong. A route that takes it on reports 23 more laws to gain three
recoveries (one directly, two after the experiment) and three wrong laws. The addition stays
available (`at_precision=True`) and is off by default until a run fixed in advance measures
what it adds, not what the arm around it does. The owner may overrule this.

The first blind set's result stands as recorded. These 46 systems have now been run on and are
development material from here on.

## The system's own forecast, scored

Filed before the run for the `default` arm (`FORECAST-ODE-ADDITIONS-2026-11.json`); scored by
`scripts/forecast_runs.py score` into `benchmarks/outcome_forecast_scored.json`.

| | Forecast (80 % interval) | What happened | Knowing only the earlier runs' averages |
| :--- | ---: | ---: | ---: |
| Recovered | 90 (51 to 129) | 116 | 182 |
| Law reported | 138 (98 to 177) | 164 | 215 |
| Reported and wrong | 25 (0 to 63) | 4 | 2 |

All three totals fall inside their intervals, which are wide. On recoveries and on reports the
forecast is closer than the earlier runs' averages. On wrong laws it is six times too high and
the plain average is closer. Run by run, its chances of a recovery score no better than a
single number for every run would have (Brier 0.244; 0.244 for the rate seen afterwards, 0.301
for the earlier average), so it placed the total and did not tell the runs apart (0.60 on a
scale where 0.5 is chance); it did tell apart the four runs that ended in a wrong law (0.89).

## The author's expectations, recorded before the run

- "`default` recovers about 145 and reports fewer than 5 wrong laws": 116 and 4. Too high by 29.
- "2 holds at 1 % noise by a clear margin and without noise by one or two components or a tie":
  by two at 1 %, which is not a clear margin, and by a tie without noise.
- "3 holds: about 15 more recoveries, mostly on the invented systems": 5 more, three of them on
  the rational systems.
- "4 is the weakest; two chances in three": held, by two.

## Reported without a criterion

- **By part, recovered / reported / wrong.** Invented systems (180 runs): `default` 95 / 126 /
  1, `both` 96 / 141 / 2, `experiment` 98 / 141 / 2, `experiment_random` 99 / 141 / 2, sparse
  88 / 180 / 83. Rational systems (96 runs): `default` 21 / 38 / 3, `both` 22 / 47 / 4,
  `experiment` 25 / 47 / 4, `experiment_random` 22 / 47 / 4, sparse 13 / 96 / 44.
- **The sparse reference** recovers 95 components without noise, one more than the route in
  force, and 6 at 1 %. It reports a law for every component and 46 % of them are wrong.
- **Run for run:** `both` against `default`, gained 0 and 2 (no noise, 1 %), lost none;
  `experiment` against `both`, gained 5 and 0, lost none; `experiment` against
  `experiment_random`, gained 3 and lost 1 without noise, none either way at 1 %.
- **Wrong laws of `default`:** GlycolyticOscillation component 1 (1 %), Invented00 component 0
  (1 %), MarhlCalcium components 0 (no noise) and 1 (1 %). **Of `both` and `experiment`:**
  Invented00 component 0 (1 %), MarhlCalcium components 0 (no noise) and 1 (1 %), and the three
  reinstated ones named above.
- **The path's actions:** `default`: `widen_search` 112, `equivalent_class` 106, `confirmed`
  58. `both`: `equivalent_class` 179, `widen_search` 88, `confirmed` 9.
- **Time:** the searches and the `default` arm, median 283 s a unit, longest 464 s; each further
  arm about 12 s; an experiment under a second in the median and 476 s at the longest. 26.8
  hours of computing, summed over the rows.
