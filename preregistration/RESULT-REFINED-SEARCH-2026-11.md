# Result: PREREG-REFINED-SEARCH-2026-11

**Run:** once, 2026-11-20 00:54–02:47 UTC, GitHub Actions run `44649600596` on the branch
`srbench-refined-run-1` at commit `7626330`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-REFINED-SEARCH.json`). The smoke job, 32
shards and the collect job all succeeded on the first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-REFINED-SEARCH-2026-11/`](RESULT-REFINED-SEARCH-2026-11/):
`results.json`, `runs.jsonl` (266 runs) and the 32 shard files. The collect job printed them to its
log with their checksums; they were rebuilt from the log and all 34 match the SHA-256 the job
computed.

## Outcome

133 SRBench problems × 2 noise levels (0 and 1 %). No run ended in an error or a timeout, and no
search stage failed.

| | Runs | Law reported | Reported and not accurate | Recovered (the law) | Refused (`widen_search`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All, both noise levels | 266 | 212 | **0** | **174** | 54 |
| Noise 0 | 133 | 99 | 0 | 88 | 34 |
| Noise 1 % | 133 | 113 | 0 | 86 | 20 |
| Feynman, noise 0 | 119 | 89 | 0 | 78 | 30 |
| Feynman, noise 1 % | 119 | 102 | 0 | 77 | 17 |
| Strogatz, both | 28 | 21 | 0 | 19 | 7 |

95 % intervals (Clopper-Pearson): not accurate among reported 0–1.7 %; the law among reported
76.2–87.0 % (174 of 212, 82.1 %).

Beside the earlier results on the same problems:

| | Recovered, all 266 runs | Feynman, noise 0 (of 119) | Feynman, noise 1 % | Strogatz (of 28) | Reported and not accurate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Refined path** (this run) | **174** | **78** | **77** | **19** | **0 of 212** |
| Main search path (PREREG-MAIN-SEARCH) | 152 | 69 | 70 | 13 | 4 of 186 |
| PySR alone (PREREG-SRBENCH-EXTERNAL) | | 73 | 68 | | |

The 103 held-out problems, which no part of the module was run on before this:

| | Noise 0 | Noise 1 % |
| :--- | ---: | ---: |
| Refined path | **61** | **61** |
| Main search path, same problems | 55 | 52 |
| Gained / lost against the main path, run for run | 11 / 5 | 15 / 6 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws reported in all 266 runs, at most 5 % are not accurate | 0 of 212 | **held** |
| 2 | On the 103 held-out problems, at least 62 recovered without noise and at least 58 at 1 % | 61 and 61 | **missed at noise 0, by one law; held at 1 %** |
| 3 | On `strogatz_lv1`, no inaccurate law is reported at either noise level | recovered at both (test R² 1.0 and 0.99998) | **held** |

Expectation 2 is a miss. The path recovered six more held-out laws than the main path without noise
and nine more at 1 %; the number fixed in advance for noise 0 was seven more.

## Reported without a criterion

- **Development problems (30).** 27 recovered at noise 0 and 25 at 1 % (the main path: 21 and 24).
  The gain there is larger than on the held-out problems (six of 30 against six of 103 at noise
  0), which is what building on them does.
- **Units.** The units reduced the problem in 226 of 266 runs: 113 of the 119 Feynman problems, by a
  median of three variables. In 26 of them the units fix the law up to a constant and nothing is
  left to search over. The Strogatz problems have no units.
- **Tentative laws.** None: the search stage ran on every problem, so no law was left unchallenged.
  In the main path's run 24 reports were unchallenged, two of them the inaccurate `strogatz_lv1`
  laws.
- **Terms dropped.** In 13 runs the confirmed form lost at least one term the data did not notice.
- **Which stage found the law.** Of the 212 reported laws, 206 came from the search stage and 6
  from the first pass.
- **By action.** `confirmed` 15 (14 the law); `equivalent_class` 197 (160 the law). All 212 are
  accurate; the 38 that are accurate and not the law are forms the rows cannot tell from it.
- **Time.** Median 274 s per run, the same as the main path (274 s); the longest run took 472 s.

## After the run (not preregistered)

Everything below reads `runs.jsonl` after the fact. It explains the numbers above and does not
replace them.

**What the eleven lost held-out runs are.** Six are refusals where the main path had reported the
law: `feynman_I_18_4`, `feynman_II_6_15a` and `feynman_I_43_16` at noise 0; `feynman_I_8_14`,
`feynman_I_18_12` and `feynman_I_43_16` at 1 %. Five are accurate laws (test R² at least 0.99998)
in a form the symbolic check does not accept: `feynman_II_35_18` and `feynman_test_7` at noise 0;
`feynman_II_11_27`, `feynman_I_16_6` and `feynman_test_7` at 1 %. Whether the refusals come from
searching over dimensionless products, which changes what the search finds easy, or from run-to-run
variation in PySR was not examined; the comparison is between two single runs.

**What changed at once.** Units, a polynomial library, the confirmed-and-pruned report, and the
search stage running on every problem. This run measures them together. The Strogatz gain (19
against 13) cannot come from units, since those problems have none.

**What it does not show.** That the path is better than published methods that use units: none was
run. And the units are information the main path and PySR alone were not given, so the 78 and 77
are not a like-for-like comparison with 69 and 70 or 73 and 68.

**Consequence.** The refined path is available as
`apedp_dyn/discovery_l2/refined_search.discover_refined`. The default route is unchanged: it is set
in `apedp_dyn/discovery_l2/route.py`, which is frozen by PREREG-LLM-PROPOSER's digest, and a
preregistration that missed one of its expectations is not the occasion to move a default by
itself. The evidence for moving it is in the tables above.
