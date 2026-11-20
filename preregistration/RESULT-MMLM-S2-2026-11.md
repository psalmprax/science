# Result: PREREG-MMLM-S2-2026-11

**Run:** once, 2026-11-19, at commit `384caf9`, which froze the preregistration, the script, the model,
the ledger learner and both searches (`preregistration/DIGEST-MMLM-S2.json`; the script checks it
before running). Python 3.13.18, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0. One CPU core.

Outputs, unedited, in [`RESULT-MMLM-S2-2026-11/`](RESULT-MMLM-S2-2026-11/): `results.json`, `runs.json`
(every search run and every law's cost) and `run.log`.

## Outcome

**Part 1, prior-ordered search with an early stop** (128 runs; plain search = every candidate, no prior):

| Laws, noise | Runs | Recovered: guided / plain | Stopped early | Wrong early stops | Median share of candidates fitted | Seconds: guided / plain |
| :--- | ---: | :--- | ---: | ---: | ---: | :--- |
| Other fields, 1 % | 24 | 22 / 23 | 23 | 2 | 0.64 % | 8.2 / 95.2 |
| Other fields, 5 % | 24 | 22 / 22 | 24 | 2 | 0.59 % | 2.1 / 94.6 |
| Invented, 1 % | 40 | 40 / 38 | 40 | 0 | 0.07 % | 0.6 / 70.2 |
| Invented, 5 % | 40 | 37 / 40 | 40 | 3 | 0.07 % | 0.6 / 70.1 |
| **All** | 128 | **121 / 123** | 127 | **7** | 0.3 % | **11.5 / 330.1** |

Ordering the candidates by the prior took 7.6 s once, for all 128 runs.

**Part 2, Feynman to Strogatz** (14 laws, 110 tokens, 7.9 tokens per law): bits per token: uniform
4.70, unigram 3.92, model 4.04.

**Part 3, two domains through the ledger** (mean bits per law, second half of each domain's laws):

| Domain | Frozen | Pooled learner | Per-domain learner |
| :--- | ---: | ---: | ---: |
| invented (60 laws) | 19.19 | 11.74 | 11.59 |
| dynamics (13 of 14 parsed) | 30.99 | 29.58 | 28.72 |

Final weights: the invented domain settled on the automaton (0.92), the dynamics domain on the
unigram (0.96).

## Against the expectations

1. **No laws lost (at most 6 fewer than plain): held.** 2 fewer over 128 runs.
2. **Wrong early stops at most 5 % (at most 6): missed by one.** 7 of 128 (5.5 %).
3. **Real work saved: held.** Median 0.3 % of the candidates fitted; 3.5 % of the plain search's time
   (5.8 % counting the one-time ordering).
4. **Transfer to the Strogatz laws (below both unigram and uniform): missed.** Below the uniform code,
   but slightly above a unigram fitted on the same Feynman formulas.
5. **Per-domain not worse than pooled: held** (11.59 against 11.74).

## What the misses are (analysis after the run)

**The wrong stops are laws the stop test cannot see apart at this noise.** `entropy_term` (x·log x) was
stopped at x·√x at both noise levels: on [1, 5] the two differ by less than the residual test can
detect in 300 rows. `compound_growth` was stopped at a law missing its prefactor's partner twice, and
three invented two-term laws at 5 % noise were stopped at their dominant term `x0`. In every one of
the seven the plain search recovered the law. That is the stop rule's stated failure direction, at a
measured rate of 5.5 %, and it is the price of the roughly thirty-fold saving. Where a wrong law is
costly, the guided search should be followed by the plain one or by contrast confirmation.

**The model's knowledge of structure did not carry to a different, longer kind of law.** Trained on
Feynman formulas, it prices the Strogatz right-hand sides no better than counting token frequencies
does. In the dynamics domain the self-adjusting mixture found that out by itself: its weights moved
to the unigram (0.96). So the earlier transfer result (short laws from other fields) does not extend
to long ODE right-hand sides; the prior should not be expected to help there until it has seen laws
of that kind.

## What follows

- The prior now saves real work on short laws: 3.5 % of the time for 2 fewer laws in 128, at a 5.5 %
  rate of stopping on a near-miss.
- Learning by domain through the ledger works and does not hurt, and the weights adapt differently per
  domain. The gain over a pooled model is small (1 % on one domain, 3 % on the other).
- Not addressed by this run: data with no law-like style at all (market data). There is no corpus of
  confirmed market laws to train or test a model of their language on; the ledger feed is the
  mechanism by which one would accumulate.
