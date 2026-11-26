# Result: PREREG-MMLM-QI-2026-11

**Run:** once, 2026-11-26 20:35 – 23:03 UTC, GitHub Actions run `45891876673` on the branch `mmlm-qi-run-1`
at commit `c63e675`, which froze the preregistration (`preregistration/DIGEST-MMLM-QI.json`). All 25 arm jobs,
the baselines job and the collect job succeeded on the first attempt; no arm is missing. Every arm record
carries commit `c63e6752` and the six frozen digests, identical to the digest file.

Outputs, unedited, in [`RESULT-MMLM-QI-2026-11/`](RESULT-MMLM-QI-2026-11/): `results.json`, the collect job's
record (SHA-256 49fb2348...3f72fa, byte for byte the record printed in the job's log), with every arm's record,
the training curves, the Kneser-Ney baselines and the expectations as the runner computed them.

## Outcome

WikiText-2 as bytes (the tokenised copy, with `<unk>` for rare words), test split, bits per byte; lower is
better. Every run trained for 2,700 seconds after a sweep of seven rates at 150 seconds each, on one GitHub
runner with four threads.

| Run | Parameters | Rate kept | Bytes seen | Scoring (bytes/s) | Processor | Test |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| `pure-complex-16` | 131,104 | 0.03 | 506.7 M | 561,954 | AMD EPYC 9V45 | 3.1115 |
| `pure-complex-16@1` | 131,104 | 0.03 | 306.6 M | 301,832 | AMD EPYC 9V74 | 3.0528 |
| `pure-complex-16@2` | 131,104 | 3 | 202.8 M | 238,375 | Intel Xeon Platinum 8370C | 3.1251 |
| `pure-complex-32` | 524,352 | 3 | 208.6 M | 250,348 | AMD EPYC 9V74 | 2.7211 |
| `pure-complex-32@1` | 524,352 | 0.03 | 154.2 M | 168,999 | AMD EPYC 7763 | 2.8264 |
| `pure-complex-32@2` | 524,352 | 3 | 217.9 M | 362,162 | Intel Xeon 6973P-C | 2.7465 |
| `pure-complex-64` | 2,097,280 | 3 | 92.6 M | 180,563 | Intel Xeon 6975P-C | **2.6065** |
| `pure-real-16` | 65,552 | 0.01 | 400.6 M | 395,409 | AMD EPYC 9V74 | 3.3713 |
| `pure-real-32` | 262,176 | 0.03 | 222.1 M | 338,754 | Intel Xeon Platinum 8573C | 3.3837 |
| `pure-real-64` | 1,048,640 | 0.03 | 133.7 M | 183,987 | AMD EPYC 9V74 | 3.4085 |
| `pure-real-23` | 135,447 | 0.01 | 409.6 M | 585,385 | Intel Xeon 6975P-C | 3.1271 |
| `pure-real-45` | 518,445 | 0.01 | 215.6 M | 256,420 | AMD EPYC 9V74 | 3.1748 |
| `pure-real-90` | 2,073,690 | 0.03 | 77.6 M | 119,971 | AMD EPYC 9V74 | 3.4585 |
| `mixed-complex-16-2` | 262,176 | 3 | 88.7 M | 77,297 | Intel Xeon Platinum 8573C | 3.1018 |
| `mixed-complex-32-2` | 1,048,640 | 1 | 43.6 M | 36,326 | AMD EPYC 7763 | 2.8601 |
| `mixed-real-16-2` | 131,088 | 3 | 104.7 M | 73,266 | Intel Xeon Platinum 8370C | 3.1315 |
| `mixed-real-32-2` | 524,320 | 3 | 79.0 M | 57,494 | AMD EPYC 7763 | 2.8842 |
| `pure-complex-23` | 270,894 | 0.1 | 276.2 M | 321,364 | Intel Xeon Platinum 8573C | 2.8766 |
| `pure-complex-45` | 1,036,890 | 0.03 | 143.5 M | 187,119 | AMD EPYC 9V74 | 2.7239 |
| `hmm-256` | 131,328 | 0.3 | 252.3 M | 234,827 | AMD EPYC 7763 | 2.9713 |
| `hmm-256@1` | 131,328 | 0.1 | 250.7 M | 217,102 | AMD EPYC 7763 | 2.7765 |
| `hmm-256@2` | 131,328 | 0.1 | 332.4 M | 307,199 | AMD EPYC 9V74 | 2.7275 |
| `hmm-1024` | 1,311,744 | 0.3 | 57.6 M | 65,600 | AMD EPYC 9V45 | 2.4217 |
| `hmm-1024@1` | 1,311,744 | 0.3 | 56.8 M | 63,614 | AMD EPYC 9V45 | 2.4038 |
| `hmm-1024@2` | 1,311,744 | 0.3 | 61.8 M | 79,890 | Intel Xeon 6973P-C | 2.4214 |
| Kneser-Ney, 0 to 4 bytes of history | | | | | | 4.6242, 3.3705, 2.7010, 2.1589, **1.8225** |

Mixed byte by byte with Kneser-Ney of 4 bytes, every run's weight chosen on the selection slice from the grid
0, 0.05, ..., 1 was **0**: each hybrid is Kneser-Ney alone, 1.8225 bits per byte.

References, not rerun (RESULT-MMLM-CM-2026-11): MMLM-CM 1.45485, PPMd order 16 primed 1.51287, MMLM-CM's LSTM
expert 1.77240, its order-4 counts 1.94505. The best quantum-inspired arm is behind each: by 1.1517, 1.0936,
0.8341 and 0.6615 bits per byte.

## Expectations

As the runner computed them (`results.json`, `expectations`):

1. **Learns more than three-byte statistics — held.** The best of the 15 quantum-inspired arms on the
   selection slice is `pure-complex-64` (2.6238 there): 2.6065 on the test split, against Kneser-Ney with
   2 bytes, 2.7010.
2. **Beats a hidden Markov model with D^2 states by 0.05 — missed, at both sizes.** Three-seed means:
   `hmm-256` 2.8251 against `pure-complex-16` 3.0965 (the automaton is 0.2714 *better*); `hmm-1024` 2.4156
   against `pure-complex-32` 2.7647 (0.3490 better). The larger seed spread is 0.2438 at D = 16 and 0.1053 at
   D = 32; both margins are larger than it (at D = 16 by only 0.028), so neither is within the seeds' noise.
3. **Complex beats real of about the same parameters by 0.02 at D = 16, 32 and 64 — missed**: 0.0156 at
   D = 16, below 0.02 (the margin is met at 32, 0.4537, and at 64, 0.8520). The shortfall at D = 16 (0.0044)
   is far inside `pure-complex-16`'s own seed spread (0.0723). The real model of the same D minus the complex
   one: 0.2598, 0.6626, 0.8020. One seed per arm.
4. **A mixed state beats a pure one of the same size by 0.02 — missed, at both sizes.** The pure model of
   about the same parameters is better: by 0.2252 (`pure-complex-23` against `mixed-complex-16-2`) and 0.1362
   (`pure-complex-45` against `mixed-complex-32-2`). Same D, pure minus mixed: 0.0097 and -0.1390; real mixed
   minus complex mixed: 0.0297 and 0.0241. One seed per arm.
5. **Better with a larger state — held.** `pure-complex-16`, `-32`, `-64`: 3.1115, 2.7211, 2.6065 (one seed).
6. **Beats the strongest Kneser-Ney — missed.** 2.6065 against 1.8225.
7. **Mixed with counting, adds what an automaton of its size does not — missed.** The weight on the model is
   0 for every run, so each hybrid equals counting alone: the gain over counting is 0.0000 at D = 16 and 32,
   and so is the gain over the hidden Markov model's hybrid (seed spread 0).

## Proposal (the owner decides; nothing changes in force, AGENTS.md 1.5)

The plan's table, read in order: 7 failed, so the first line that applies is the fourth. **Proposal: do not
develop quantum-inspired language models in MMLM.** Mixed with counting they gained nothing measurable over
counting alone: no weight of 0.05 or more was chosen for any run (weights between 0 and 0.05 were not on the
grid). Alone they lose to a hidden Markov model with D^2 states (2.5 times the parameters at D = 32) at both
sizes tested. Expectation 6 failed as well, so no claim is made that these models compete with counting.

What else the run shows, under this optimiser and this budget (the plan's Limits), one seed per arm except
where three are named:

- The complex pure models beat the real ones of about their parameters by 0.45 and 0.85 bits per byte at
  D = 32 and 64, but the real models got *worse* as they grew (`pure-real-23`, `-45`, `-90`: 3.1271, 3.1748,
  3.4585; `pure-real-16` 3.3713 against `-64` 3.4085), so part of that gap may be how the real models trained
  rather than what complex amplitudes add.
- A larger state helped the pure complex models, the mixed ones and the hidden Markov models, not the pure
  real ones.
- At equal parameters and equal wall-clock a mixed state did worse than a pure one; the mixed models also
  saw about a third of the bytes (88.7 M against 276.2 M, 43.6 M against 143.5 M).
- The strongest model trained here, quantum-inspired or not, was the 1,024-state hidden Markov model (2.40 to
  2.42), still about 0.6 bits per byte behind four bytes of counting.

## Deviations and disclosures

No deviation from the protocol: every rate kept is its sweep's best, the parameters kept are the best point
of each training curve, sweeps were 150 seconds and training 2,700 seconds with nine evaluations. Two things
the reading should know:

- Rate 3 diverged in the sweep of `hmm-256` (seeds 0 and 2) and was excluded, as the protocol requires; the
  second rehearsal had no such case.
- Seven of the 25 runs kept rate 3, the top of the grid: `pure-complex-16@2`, `pure-complex-32` (seeds 0 and
  2), `pure-complex-64` (the best quantum-inspired arm), `mixed-complex-16-2`, `mixed-real-16-2` and
  `mixed-real-32-2`. Before freezing, the review treated a rate at the top of the grid as a sign of
  under-tuning when it happened to the hidden Markov model (its finding 1, which added rates 1 and 3). These
  arms may be under-tuned in the same way, which bears on their losses to the automaton and to counting.
- Runners differed in processor (recorded per run above); at equal wall-clock this changes how many bytes a
  run saw, which the plan disclosed before the run.

## Independent check (AGENTS.md 1.6)

A reviewer agent that took no part in the work checked this record cold against `results.json`, the frozen
plan and the runner. It found every number right (the record byte-identical to the job's log; all 27 jobs on
the frozen commit with matching digests; parameters, rates, bytes seen, test scores, Kneser-Ney values, the
best arm on the selection slice, every expectation's arithmetic; the runner matching the plan's words; the
proposal the right row). Its findings on the first draft, and what was done:

1. The data was called "WikiText-2 (raw)"; it is the tokenised copy with `<unk>`. *Done:* corrected.
2. Expectation 3 was written as missed at one D and held at two; the plan reads it as one test. *Done:* it is
   reported as missed, with the D = 16 shortfall set against that arm's seed spread.
3. The remarks beyond the numbers went too far (complex amplitudes "help"; a larger state "helps"; a mixed
   state "does not help") without the real models' worsening with size, the arms each claim covers, the
   bytes the mixed models saw, and one seed. *Done:* qualified as above.
4. The proposal's wording: "gain nothing" holds only on the weight grid, and the automaton's 2.5 times the
   parameters at D = 32 was left out. *Done.*
5. Items the plan says are reported were missing: the gaps to the other three references, bytes seen for most
   runs, scoring speed, processor per run (and the exact processor names). *Done:* the table and the
   references line.
6. Two facts bearing on the reading were not stated: rate 3 diverging for `hmm-256` (seeds 0 and 2) and seven
   runs keeping rate 3, the top of the grid. *Done:* under "Deviations and disclosures".
7. Expectation 2's margins were said to be outside the spread "because they point the other way"; they are
   outside because each is larger than the spread (at D = 16 by only 0.028). *Done.*
