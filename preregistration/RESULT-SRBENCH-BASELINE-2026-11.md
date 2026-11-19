# Result: PREREG-SRBENCH-BASELINE-2026-11

**Run:** once, 2026-11-19 06:54–09:40 UTC, GitHub Actions run `44475539331` on the branch
`srbench-baseline-run-1` at commit `c6e1c1e`, which froze the preregistration, the harness, the
fetch script and the manifest (the harness checks their digests before running; they are repeated in
`results.json`). 32 shards, one collect job, all successful; no shard was re-run.

Outputs, unedited, in [`RESULT-SRBENCH-BASELINE-2026-11/`](RESULT-SRBENCH-BASELINE-2026-11/):
`results.json`, `runs.jsonl` (2,128 runs) and the 32 shard files. The artifact store is not reachable
from the authoring environment, so the files were printed to a job log by a throwaway workflow
(run `44505819913`, branch `srbench-fetch-results-1`, not merged) and rebuilt from it; all
34 files match the SHA-256 the job computed before printing.

## Outcome

133 problems (119 Feynman, 14 Strogatz) × 4 noise levels × 4 engines, 120 s per run.

Symbolic recovery, pooled over the four noise levels (532 runs per engine):

| Engine | Recovered | 95 % interval | Feynman (of 476) | Strogatz (of 56) | Accuracy solutions | Timeouts | Median seconds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | **108** (20.3 %) | 17.0–24.0 % | 96 | 12 | 121 | 68 | 23.4 |
| Extended hierarchical | 36 (6.8 %) | 4.8–9.2 % | 32 | 4 | 36 | 0 | 1.0 |
| RIME | 24 (4.5 %) | 2.9–6.6 % | 20 | 4 | 24 | 0 | 1.3 |
| RIME-AD | 11 (2.1 %) | 1.0–3.7 % | 5 | 6 | 46 | 328 | 121.1 |

Symbolic recovery by noise level (of 133 problems each):

| Engine | 0 | 0.1 % | 1 % | 10 % |
| :--- | ---: | ---: | ---: | ---: |
| L2 | 27 | 27 | 27 | 27 |
| Extended hierarchical | 9 | 9 | 9 | 9 |
| RIME | 6 | 6 | 6 | 6 |
| RIME-AD | 7 | 2 | 1 | 1 |

By family and noise, symbolic recovery (Feynman of 119 / Strogatz of 14):

| Engine | 0 | 0.1 % | 1 % | 10 % |
| :--- | ---: | ---: | ---: | ---: |
| L2 | 24 / 3 | 24 / 3 | 24 / 3 | 24 / 3 |
| Extended hierarchical | 8 / 1 | 8 / 1 | 8 / 1 | 8 / 1 |
| RIME | 5 / 1 | 5 / 1 | 5 / 1 | 5 / 1 |
| RIME-AD | 5 / 2 | 0 / 2 | 0 / 1 | 0 / 1 |

L2's referee, per noise level (the 17 timeouts per level are not counted in any column):

| | 0 | 0.1 % | 1 % | 10 % | Pooled |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Promoted a recovered law | 25 | 25 | 25 | 25 | 100 |
| Promoted anything else (false promotion) | 42 | 42 | 42 | 42 | **168** |
| No promotion | 49 | 49 | 49 | 49 | 196 |

## Against the expectations

1. **The referee rarely promotes a wrong law (false promotions at most 5 % of L2's promotions):
   missed, by a wide margin.** 168 of 268 promotions were of something other than the recovered law:
   **62.7 %**. Feynman 156 of 252 (61.9 %); Strogatz 12 of 16 (75 %).
2. Recovery and accuracy rates are reported above without a criterion, as preregistered.

## What this means

- On an outside benchmark the best in-house engine recovers one law in five. The other three recover
  between 2 % and 7 %.
- A promotion by the referee on this benchmark is more often wrong than right. The preregistration
  disclosed one wrong promotion seen in the four-problem smoke test; the full run shows it is the rule
  on problems whose law the search cannot express, not an exception.
- The 5 % expectation was set from the blind benchmark (7 false discoveries in 288 runs), whose laws
  were drawn from forms L2's search can build. That number does not carry over to laws outside the
  search space.

## After the run (not preregistered)

Everything in this section was computed from `runs.jsonl` after seeing the result.

- **Why the referee promoted.** In 116 of the 168 false promotions the stated reason is that no
  structurally distinct rival survived; in the other 52 the nearest rival was separated by more than
  the threshold. `UNIQUE_SUPPORT` is a statement about the candidates the search produced. It is not
  conditioned on the promoted law fitting well in absolute terms.
- **How badly the promoted laws fit.** Test R² is recorded for 160 of the 168. 36 are below 0.5, 60
  between 0.5 and 0.9, 35 between 0.9 and 0.99, 21 between 0.99 and 0.999 and 8 at or above 0.999. The
  median is 0.81. Examples at zero noise: `feynman_III_15_12` promoted `x0` (R² 0.22);
  `feynman_III_10_19`, whose law is `mom*sqrt(Bx**2+By**2+Bz**2)`, promoted `x0*(x2 + x3)` (R² 0.84).
- **Four of the 168 are a scoring artefact.** `strogatz_vdp1` (`10*(y - (x**3 - x)/3)`) promoted
  `x0**3 + x0 + x1` at every noise level. That is the right structure with free coefficients, and the
  fitted model is scored as recovered; the promoted structure is compared without its coefficients
  and is scored as different. Without those four the share is 164 of 264 (62.1 %). The expectation is
  missed either way, and the preregistered figure stands.
- **Four more are close approximations, not the law.** `feynman_I_48_2`
  (`m*c**2/sqrt(1-v**2/c**2)`) promoted `x0*x2**2` at each noise level with R² 0.9993; it counts as an
  accuracy solution and is not the law.
- **The same problems fail at every noise level.** 44 problems were falsely promoted at least once;
  40 of them at all four levels. The identical counts across noise levels for L2, RIME and the
  extended hierarchical engine are not a sign that noise was not applied: L2's accuracy solutions
  vary (31, 30, 31, 29) and RIME-AD's recovery falls from 7 to 1. These engines choose among a fixed
  set of structures, and on this benchmark the choice does not change up to 10 % noise.
- **Timeouts.** L2 timed out on 17 problems at every level (13 Feynman, 4 Strogatz). RIME-AD timed
  out on 82 Feynman problems at every level, which is 62 % of all its runs.

## Consequences recorded elsewhere

- Audit entry M-10 in `docs/SYSTEM_FLAWS_AUDIT.md`: the referee promotes a sole survivor without an
  absolute-fit condition. No referee code was changed in response; a change to the referee needs its
  own preregistered test (contrast confirmation's self-consistency check, which returns `no_survivor`
  when the survivor does not match its own development error, is the candidate).
- PREREG-SRBENCH-EXTERNAL-2026-11 compares outside engines with the best baseline engine at noise 0
  and 1 %: that reference is L2 with 24 of 119 Feynman problems at both levels.
