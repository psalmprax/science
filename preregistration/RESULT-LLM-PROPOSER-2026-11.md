# Result: PREREG-LLM-PROPOSER-2026-11

**Run:** once, 2026-11-19 22:57 – 2026-11-20 00:21 UTC, GitHub Actions run `44633332462` on the
branch `llm-proposer-run-1` at commit `1d53ae2`, which froze the preregistration, the harness and
the model (`nvidia/nemotron-3-ultra-550b-a55b`, reasoning on with a budget of 2 000 tokens). The
smoke job, six shards and the collect job succeeded on the first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-LLM-PROPOSER-2026-11/`](RESULT-LLM-PROPOSER-2026-11/): `results.json`,
`runs.jsonl` (266 runs) and the six shard files, rebuilt from the collect job's log; all 8 match the
SHA-256 the job computed.

## Outcome

SRBench's 133 problems at noise 0 and 1 %, the language model in the place PySR has in the main
search path, given numbers only. Laws are scored in the form the path confirmed.

| | Runs | Law reported | Not accurate | Recovered | Found by the model and recovered | Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| All, both noise levels | 266 | 92 | **2 (2.2 %)** | 78 | **12** | 7 |
| Noise 0 | 133 | 45 | 1 | 43 | 10 | 5 |
| Noise 1 % | 133 | 47 | 1 | 35 | 2 | 2 |
| Feynman, noise 0 | 119 | 40 | 0 | **40** | 9 | 3 |
| Feynman, noise 1 % | 119 | 43 | 0 | **32** | 2 | 1 |

The same path with PySR as the search stage (PREREG-MAIN-SEARCH): 186 reported, 4 not accurate,
152 recovered; Feynman 69 and 70.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | At most 5 % of the reported laws are not accurate | 2 of 92, 2.2 % (interval 0.3 to 7.6 %) | **held** |
| 2 | In at least 10 runs the reported law was found by the model and is recovered | 12 | **held** |
| 3 | Fewer Feynman laws than PySR alone (fewer than 73 and 68) | 40 and 32 | **held** |

Scored on the search's own text instead of the confirmed form the counts are the same (2 not
accurate, 78 recovered).

## Reported without a criterion

- **How much the model adds.** The first pass settled 59 runs without the model being asked (57
  reported a law). The model was asked in 200 runs, with 881 requests and 1 397 proposals. In 35 of
  those 200 a law was reported, 19 of them the model's (12 recovered, all 19 accurate); the path
  refused the other 165.
- **What it proposes.** Almost all of its recovered laws are products and ratios of the variables
  with free exponents (`c0*x0**c1*x1**c2/...`). That form fits a monomial law exactly without
  noise and gives exponents like 0.998 with it, which is accurate and is not the law's form: 10 of
  the 12 recoveries are at noise 0.
- **Against PySR in the same place.** 77 runs are recovered with either search stage, 75 only with
  PySR, 1 only with the model (`feynman_II_34_29b` at noise 0, the case in which PySR's text was
  wrong and its confirmed fit right).
- **Replies.** 940 reply lines could not be used (not an allowed expression) against 1 926
  skeletons tried. In 65 proposals the form matched what the model had offered before seeing any
  data. Every final answer came back as an answer (none ran out of tokens while reasoning).
- **Errors.** Two runs were lost to the service's rate limit after retries (HTTP 429). Five ended
  in `ValueError: array must not contain infs or NaNs`: a proposed formula was not finite on the
  data and a fit inside the run raised instead of skipping it.
- **Time.** Median 98 s per run, maximum 227 s.

## After the run (not preregistered)

- **The two inaccurate reports are the same case as in PREREG-MAIN-SEARCH** and have nothing to do
  with the model: `strogatz_lv1` at both noise levels, an unchallenged first-pass law with test R²
  0.35 (audit M-14).
- **The five crashes are a defect of the harness, not of the model's answers** (recorded as audit
  M-17): `strogatz_bacres1` at both noise levels, `strogatz_bacres2`, `feynman_II_13_17` and
  `feynman_II_8_7`. They count as errors here, as the preregistration requires, and they lower the
  model's numbers slightly. `proposers.composed.CompositeProposer` now contains a failing search
  stage; the frozen runner did not.
- **Reading.** With numbers only and fifteen proposals a problem, a strong general model is a
  useful second opinion on monomial laws and no replacement for a search: it adds 9 Feynman laws
  to the first pass's 31 without noise, where PySR adds about 38. The published systems that do
  well with language models use them to steer a search, not to write the equations; that design
  is not what this run tested.

## What changes

- The language-model proposer has a result: it works, what the path reports with it is as
  reliable as with PySR (the two failures are the path's own), and it recovers about half as many
  laws. It stays an optional proposer, not the default.
- The API key used for this run was supplied by the owner and lives only in the repository
  secret; it should be rotated now that the run is recorded.
