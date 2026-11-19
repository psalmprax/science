# Result: PREREG-CONTRAST-2026-11

**Run:** once, 2026-11-19, at commit `1ccfd14`, which froze the preregistration, the script and
`contrast_confirmation.py` (`preregistration/DIGEST-CONTRAST.json`; the script checks it before
running). Python 3.13.18, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0. One CPU core, a few minutes.

Outputs, unedited, in [`RESULT-CONTRAST-2026-11/`](RESULT-CONTRAST-2026-11/): `results.json`,
`runs.jsonl` (every run's candidates' verdicts in each arm) and `run.log`.

## Outcome

80 runs (40 fresh two-term laws × noise 1 % and 5 %). The true law was among L2's twelve candidates in
all 80.

| | Contrast (arm A) | Random rows (arm B) | Referee on 120 rows |
| :--- | ---: | ---: | ---: |
| True law confirmed uniquely | **63** (79 %, 95 % interval 68–87 %) | 54 (68 %) | 28 promoted |
| True law in the returned class | 78 | 77 | |
| A wrong law named alone | **0** (interval 0–4.5 %) | 0 | 5 promoted |
| `no_survivor` although the true law was there | 2 | 3 | |
| Median rows read | 60 | 222.5 | 0 |

Truth removed from the candidates (arm C): `no_survivor` in 61 of 80; a wrong law named alone in 19.

## Against the expectations

1. **Rarely names a wrong law (at most 5 %): held.** 0 of 80.
2. **Choosing the rows is what helps (paired sign test, p < 0.01): held.** 9 runs confirmed uniquely
   by contrast and not by random rows, 0 the other way; p = 0.002. The gain is at 5 % noise (31 against
   23); at 1 % the arms are level (32 against 31). The control read more rows, not fewer (its
   comparisons do not share rows), so the advantage is not from more data.
3. **Rarely refutes a true law (at most 5 %): held.** 2 of 80 (2.5 %; the interval reaches 8.7 %, so
   this run does not rule out a rate above 5 %).
4. **Notices a missing law (`no_survivor` at least 80 %, wrong law alone at most 5 %): missed on both
   counts.** 61 of 80 (76 %) and 19 of 80 (24 %).

## What the miss is (analysis after the run, not preregistered)

In 14 of the 19, arm A on the same run had returned an equivalence class: the true law and a rival
could not be told apart on this domain. Delete the true law and that rival is left standing with
nothing to contradict it, and it is reported `unique`. The verdict means "the only candidate left that
the data here cannot refute", and when the true law is absent that candidate can be a wrong law that
is indistinguishable from it on this domain. In the other 5 the survivor was distinguishable from the
truth in arm A and still passed the refutation test once the truth was gone.

So `no_survivor` is good evidence that the true law is missing (61 of 80 when it was, 2 of 80 when it
was not), but `unique` is not evidence that it is present. The procedure cannot be used on its own to
decide when a search space is large enough.

## What follows

- Contrast confirmation does what it was built for: on fresh laws it confirmed the true law in 63 of
  80 runs where the referee promoted it in 28, named no wrong law, and the referee promoted a wrong
  law in 5 of the same runs.
- It needs a pool to measure from and a known noise model, and its `unique` is relative to the
  candidates it was given.
- Next: wire it in as the step after a `MULTI_HYPOTHESIS` verdict when a pool exists, and fix the
  ladder problem with `no_survivor` only (climb when refuted, never stop because of `unique`).
