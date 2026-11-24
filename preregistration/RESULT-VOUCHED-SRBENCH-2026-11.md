# Result: PREREG-VOUCHED-SRBENCH-2026-11

**Run:** once, 2026-11-22 23:05 to 2026-11-23 01:54 UTC, GitHub Actions run `45081278316` on the
branch `srbench-vouched-run-1` at commit `ae7c026`, which carries the frozen preregistration and
harness (digests repeated in `results.json`, and equal to `DIGEST-VOUCHED-SRBENCH.json`). The
smoke job, 32 shards (23:14 to 01:52) and the collect job all succeeded on the first attempt; no
shard was re-run.

Outputs, unedited, in [`RESULT-VOUCHED-SRBENCH-2026-11/`](RESULT-VOUCHED-SRBENCH-2026-11/):
`results.json` and `runs.jsonl.gz` (266 rows, each with both reports). The collect job printed
them to its log with their checksums; they were rebuilt from the log and both match the SHA-256
the job computed (`SHA256SUMS`).

## Outcome

**The first two expectations held and the third missed, by a wide margin.** On the physics
benchmark, behind the four searches, the last checks stopped no inaccurate law (there was none to
stop) and withheld 43 of the 193 laws the route recovered: **150 kept, 78 %**, where 95 % was the
line. Every loss was at 1 % noise, where the checks kept 47 of 90.

| Arm | Reported | Recovered | Not accurate | Accurate, not the law |
| :--- | ---: | ---: | ---: | ---: |
| `route` (the route without the checks) | 229 | 193 | 0 | 36 |
| `vouched` (the route in force) | 182 | **150** | 0 | 32 |

| | Noise 0 | Noise 1 % | Feynman | Strogatz | 206 runs never used in development |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `route`, recovered | 103 | 90 | 168 | 25 | 141 |
| `vouched`, recovered | 103 | **47** | 133 | 17 | 109 |

One run ended in an error (`feynman_III_8_54` without noise: a least-squares fit that did not
converge, audit M-24) and one exceeded its time limit (`feynman_III_9_52` without noise); both
count as nothing reported in both arms.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws `vouched` reports, at most 5 % are not accurate | 0 of 182 (interval 0 to 2.0 %) | **held** |
| 2 | Of the runs in which `route` recovers the law, `vouched` recovers it in at least 75 % | 150 of 193, 77.7 % (interval 71 to 83 %) | **held** |
| 3 | That share is at least 95 % | 77.7 % | **missed** |

## What follows, as fixed before the run

Expectation 2 held and 3 missed, so: **the checks cost more here than a referee is allowed to get
wrong, on a benchmark where they had nothing to add. They stay in force until the repository's
owner decides, the cost is stated first wherever the route is described, and the owner is
asked.** The owner is asked with this result.

Because the interval for expectation 2 reaches below 75 %, the second expectation's pass is
narrow: another run of the same searches could fall on the other side of the line.

## Why the laws were lost

Found after the run by calling the checks again on each lost law's rows (development; the
frozen code, the run's own rows):

- **42 of the 43 were withheld by the power rule**, and in every one of the 42 the shapes the rule
  could not see were the differences between the law and its rivals: the other candidates the
  path could not tell apart from it. Every polynomial shape planted in those 42 was seen at
  least four times in five. The 43rd was withheld because members of its class disagreed.
- **Those differences sit on a few rows.** Over the rivals of the lost laws, a median of 56 % of a
  rival's squared difference from the law falls on 1 % of the rows (seven of 700), and for more
  than half of them more than half does: rivals that part from the law sharply on a few rows, as
  a pole or a steep corner would. The path keeps them because, on the thirty rows it may measure,
  they cannot be told apart from the law.
- **The structure test cannot see an error on seven rows.** It asks whether each residual is
  predicted by its neighbours' residuals; a spike is not. So the rule concludes, correctly as far
  as its own test goes, that an error of that shape would have been missed, and withholds the
  law. This is the blindness the second red-team campaign found (audit M-34), with the opposite
  effect: there it let wrong laws through, here it costs right ones. Recorded as audit M-35.
- **It is not the size of the error against the noise:** the error that makes a law wrong is
  about twice the noise for lost and kept laws alike (medians 2.3 and 2.1).
- **The counterfeit benchmark could not have shown it.** Its laws had one to three inputs and its
  candidates were handed over; a law's rivals there were its counterfeits, which differ from it
  smoothly. Here the rivals come from four searches.
- By number of inputs, of the laws recovered at 1 % noise: kept 13 of 24 with two inputs, 17 of
  26 with three, 11 of 21 with four, 5 of 12 with five, 1 of 7 with six or more.

## Reported without a criterion

- **The checks were applied to all 229 reported laws**: 180 passed unchanged, 2 were passed to
  another member of their class, 47 were withheld (46 by the power rule, 1 by the class rule).
  They gained no law.
- **The `route` arm against the recorded run of the same route** (PREREG-PORTFOLIO-SEARCH): 193
  recovered against 189, 229 reported against 224, 0 not accurate against 1. The searches run to
  a time budget, so a second run differs; both arms here read the same search.
- **The earlier run's prediction** (above 97 % kept, from PREREG-COUNTERFEIT's cases at noise up
  to 1 %) was far off: that benchmark's rivals were not of this kind.
- **The author's expectations**: right that the first two would hold and that every loss would
  be at 1 % noise; wrong on the size, two to ten lost laws expected and 43 lost; wrong on the
  cause, laws whose spread is small beside their size were expected and rivals with poles were
  found. No law was gained, where one to three were expected.
- **Time.** A median of 290 seconds a run; the checks add about a second.

## What this result does not say

It says nothing about the checks where they were shown to help: at 5 % noise, on candidates
handed over. It says that on the physics benchmark at 1 % noise, behind a real search, they
withhold nearly half of the laws the route recovers, for a reason that can be named, and stop
nothing in exchange. A repair (testing a rival-shaped error against the rival's own fit instead
of the neighbour statistic) would have to be written and run on this benchmark and on laws
drawn after it.
