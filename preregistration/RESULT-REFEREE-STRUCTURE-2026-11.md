# Result: PREREG-REFEREE-STRUCTURE-2026-11

**Run:** once, 2026-11-20 03:00–04:05 UTC, GitHub Actions run `44664709007` on the branch
`referee-structure-run-1` at commit `990f7f3`, which carries the frozen preregistration, the guard
and the harness (digests repeated in `results.json`, and equal to
`DIGEST-REFEREE-STRUCTURE.json`). The smoke job, 32 shards and the collect job all succeeded on the
first attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-REFEREE-STRUCTURE-2026-11/`](RESULT-REFEREE-STRUCTURE-2026-11/):
`results.json`, `runs.jsonl` (532 runs) and the 32 shard files. The collect job printed them to its
log with their checksums; they were rebuilt from the log and all 34 match the SHA-256 the job
computed.

## Outcome

The template engine with the referee on SRBench's 133 problems at four noise levels, exactly as the
baseline's L2 arm, with Guard 3 on. A promotion is a verdict of `UNIQUE_SUPPORT`.

The 103 held-out problems (412 runs), which the guard was not built on:

| | Promotions | Correct | False (symbolic) | Not accurate | Refused by the guard | Timeouts |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **With the guard** | 81 | **71** | 10 | **3** | 126 | 68 |
| Baseline, same runs | 201 | 72 | 129 | 121 | – | 56 |

By noise level, with the guard (held-out, 103 runs each):

| Noise | Promotions | Correct | False | Not accurate |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 20 | 18 | 2 | 0 |
| 0.1 % | 20 | 18 | 2 | 0 |
| 1 % | 19 | 17 | 2 | 0 |
| 10 % | 22 | 18 | 4 | 3 |

All 532 runs: 113 promotions, 103 correct, 10 false, 3 not accurate (the baseline: 268, 100, 168 and
160).

## Expectations

| | Expectation (held-out problems) | Result | |
| :--- | :--- | :--- | :--- |
| 1 | At most 5 % of promotions are not accurate | 3 of 81, 3.7 % (the baseline: 121 of 201, 60 %) | **held** |
| 2 | Of the 72 runs where the baseline promoted the correct law, at least 65 do so again | 71 | **held** |
| 3 | At most 10 % of promotions are false by the symbolic criterion | 10 of 81, 12.3 % (the baseline: 129 of 201, 64 %) | **missed** |

On expectation 1 the count is inside the criterion and the interval is not: with 81 promotions the
data are compatible with a true rate between 0.8 and 10.4 %.

## Reported without a criterion

- **What became of the baseline's 129 false held-out promotions:** 119 are refused by the guard, 6
  are still promoted and 4 timed out.
- **The one correct law not kept:** `feynman_II_8_7` at 1 % noise, refused with a neighbour R² of
  0.03 at 16 neighbours. A refusal of a true law, at about the rate the test's level allows (1 in
  72).
- **Development problems (120 runs):** 32 promotions, all 32 correct; 39 refused by the guard
  (the baseline: 28 correct and 39 false).
- **Timeouts:** 68 held-out runs against 56 in the baseline, at the same 120 s limit on a different
  day's machines. 28 runs timed out here that had not; none of them was a correct promotion in the
  baseline. Median time per run 18.5 s against 23.4 s.
- **Every law still wrongly promoted on the held-out problems, with its test R²:**

  | Problem | Noise | Promoted structure | Test R² after refitting its terms |
  | :--- | :--- | :--- | ---: |
  | `strogatz_glider1` | 0, 0.1 %, 1 % | `x0**2 + sin(x1)` | 1.000 |
  | `strogatz_vdp1` | 0, 0.1 %, 1 % | `x0**3 + x0 + x1` | 1.000 |
  | `feynman_I_48_2` | 10 % | `x0*x2**2` | 0.9993 |
  | `feynman_I_30_5` | 10 % | `x0/(x1*x2)` | 0.9984 |
  | `feynman_I_39_11` | 10 % | `x1*x2/x0**(3/2)` | 0.9982 |
  | `feynman_II_11_28` | 10 % | `exp(x0*x1)` | 0.9898 |

## After the run (not preregistered)

Everything below reads `runs.jsonl` after the fact. It explains the numbers above and does not
replace them.

**Why expectation 3 was missed.** Six of the ten false promotions are two Strogatz problems at
three noise levels. In both the promoted structure has exactly the law's terms (`-0.05 x² - sin y`;
`10 y - (10/3) x³ + (10/3) x`), and with one coefficient fitted per term it predicts the test rows to
R² 1.000. The symbolic criterion scores the structure as the template engine wrote it, with a
single common coefficient, and that is not the law. These are the "false and accurate" promotions
the preregistration said the guard might not see; it could not have refused them, because their
residuals carry no structure. The other four are at 10 % noise, where a wrong law whose error is
small beside the noise leaves residuals that look like noise: three with test R² between 0.990 and
0.998, one above 0.999. Without the Strogatz six the rate would have been 4 of 75; that is an
observation, not a re-scoring.

**What the guard did not cost.** 71 of 72 correct held-out promotions are kept, and on all 133
problems the correct promotions went from 100 to 103 (three more runs finished inside the time
limit).

**What it does not show.** The rows of SRBench are independent draws, which is the case a
permutation test is exact for. Rows taken along one trajectory are not exchangeable, and the
template engine is also used on such rows (the UI's registry page). Nothing here measures the
guard there.

## What follows

The preregistration fixed it: expectations 1 and 2 hold, so Guard 3 becomes the default for the
template engine. One thing in that sentence cannot be done as written. It names the default of
`L2DiscoveryEngine.discover`, and `apedp_dyn/discovery_l2/l2_engine.py` is frozen by this
preregistration's own digest; editing it would make the recorded run unrepeatable from the tree.
The default is therefore changed in a subclass that the repository's callers are moved to, one
caller at a time, with any verdict that changes examined and not edited to pass, and with the
callers that work on trajectory rows left on the unguarded engine until the guard has been
measured there. That work is recorded in the audit (M-10) as it is done.
