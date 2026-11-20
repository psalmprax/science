# Result: PREREG-LLM-CONTROLLER-2026-11

**Run:** once, 2026-11-20 01:08–03:53 UTC, GitHub Actions run `44651305642` on the branch
`srbench-controlled-run-1` at commit `d0d4953`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-LLM-CONTROLLER.json`). The smoke job, 32
shards and the collect job all succeeded on the first attempt; no shard was re-run. Most of the
elapsed time was spent queued behind other runs.

Outputs, unedited, in [`RESULT-LLM-CONTROLLER-2026-11/`](RESULT-LLM-CONTROLLER-2026-11/):
`results.json`, `runs.jsonl` (266 runs) and the 32 shard files. The collect job printed them to its
log with their checksums; they were rebuilt from the log and all 34 match the SHA-256 the job
computed.

The model is `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA's API, with reasoning on. It is shown
twelve rows of numbers and the variable count, and answers with a configuration for the search:
which unary operators to allow, extra features, and guesses to fit. It writes no law that is taken
on its word; everything goes through the same confirmation as in the refined path.

## Outcome

133 SRBench problems × 2 noise levels (0 and 1 %). No run ended in an error or a timeout.

| | Runs | Law reported | Reported and not accurate | Recovered (the law) | Refused (`widen_search`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All, both noise levels | 266 | 207 | **1** | **173** | 59 |
| Noise 0 | 133 | 103 | 1 | 92 | 30 |
| Noise 1 % | 133 | 104 | 0 | 81 | 29 |
| Feynman, noise 0 | 119 | 92 | 1 | 81 | 27 |
| Feynman, noise 1 % | 119 | 92 | 0 | 71 | 27 |
| Strogatz, both | 28 | 23 | 0 | 21 | 5 |

Beside the refined path without the model (PREREG-REFINED-SEARCH), same problems and budget:

| | Recovered, all 266 | Noise 0 | Noise 1 % | Feynman 0 / 1 % | Strogatz | Not accurate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Model configures the search** (this run) | 173 | 92 | 81 | 81 / 71 | 21 | 1 of 207 |
| Refined path | 174 | 88 | 86 | 78 / 77 | 19 | 0 of 212 |

The 103 held-out problems:

| | Noise 0 | Noise 1 % | Both |
| :--- | ---: | ---: | ---: |
| Model configures the search | 66 | 57 | **123** |
| Refined path | 61 | 61 | 122 |
| Gained / lost against the refined path, run for run | 11 / 6 | 4 / 8 | 15 / 14 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws reported, at most 5 % are not accurate | 1 of 207, 0.5 % (interval 0.01–2.7 %) | **held** |
| 2 | Held-out laws recovered (206 runs) within 6 of the refined run's 122 | 123 | **held** |
| 3 | At least 90 % of the model's replies are readable | 214 of 214 | **held** |

Expectation 2 is informative: the model failed in none of the runs in which it was asked.

## Reported without a criterion

- **When the model was asked.** In 214 runs. In the other 52 (26 Feynman problems at both noise
  levels) the units fix the law up to a constant and there is nothing to configure; 50 of those 52
  are recovered, in both runs.
- **On the 214 runs where it was asked:** 123 recovered with the model, 124 without.
- **What it did.** It narrowed the unary operators in 162 runs, added 785 features (3.7 per run)
  and gave 625 guesses that could be fitted; 9 items were rejected as unreadable or unsafe. One call
  per run in 183 runs; up to five where the API had to be retried.
- **Development problems (30).** 26 and 24 recovered (the refined path: 27 and 25).
- **Time.** Median 308 s per run against 274 s; the longest took 532 s.

## After the run (not preregistered)

Everything below reads `runs.jsonl` after the fact. It explains the numbers above and does not
replace them.

**The one inaccurate law is the model's doing.** `feynman_III_8_54` at noise 0 is
`sin(E t/ħ)²`. The model allowed `sqrt`, `exp`, `log` and `square`, and left out the sine. The
search then returned a formula in powers of a logarithm (test R² 0.91), and contrast confirmation
and the adequacy check let it through: against a nearest-neighbour smoother its error was 47 %,
and no structure was found in its residuals. The refined path recovered this law. It is the first
inaccurate report in 419 laws reported by the refined path in its two runs, and it shows the cost
of narrowing: a search that cannot express the law cannot be corrected by the stages after it.

**The same total, and different laws.** The two runs agree on 156 of the laws they recover. 17 are
recovered only with the model and 18 only without, so 191 of 266 runs are recovered by one or the
other (137 of the 206 held-out runs, against 122 and 123). That is not a result of this experiment
and not what a merged search would achieve, since confirmation would still have to pick the right
candidate. It does say that what limits recovery here is which candidates a single search
produces, and that several differently configured searches produce different ones.

**With noise the model's version is behind (81 against 86), without it ahead (92 against 88).**
Two single runs do not separate a real difference from PySR's run-to-run variation, which has not
been measured.

**What it means for the question asked.** A language model that configures the search is as good
as no language model, to within a law, on this benchmark, and it costs 34 seconds and an API call
per run. Together with PREREG-LLM-PROPOSER (the model writing the laws: 78 recovered against 152)
the finding so far is that this model adds nothing to the path by itself. A stronger model was not
tried.
