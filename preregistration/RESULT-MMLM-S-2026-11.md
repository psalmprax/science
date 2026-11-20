# Result: PREREG-MMLM-S-2026-11

**Run:** once, 2026-11-19, at commit `bb2dffe`, which froze the preregistration, the script, the
model (`apedp_mmlm/equation_language.py`) and the search (`preregistration/DIGEST-MMLM-S.json`; the
script checks it before running). Python 3.13.18, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0. One CPU core.

Outputs, unedited, in [`RESULT-MMLM-S-2026-11/`](RESULT-MMLM-S-2026-11/): `results.json`, `runs.json`
(every law's positions, selections and per-law costs) and `run.log`.

## Outcome

**Part A, the model as a prior** (fitted on SRBench's 133 formulas, not updated):

| Laws | Reachable | Found earlier / later than size order | Median position: size order → model |
| :--- | ---: | :--- | :--- |
| Other fields, all 24 | 24 | 23 / 1 (p = 1.5 × 10⁻⁶) | 6,111.5 → 303.5 |
| Other fields, 19 whose shape is not in training | 19 | 18 / 1 (p = 3.8 × 10⁻⁵) | 5,921 → 236 |
| Invented at random (no criterion) | 76 of 120 | 60 / 10 (p = 4 × 10⁻¹⁰) | 110 → 23 |

Selection from 300 rows at 1 % noise, other-field laws: the search recovered 24 of 24 without the model
and 24 of 24 with it.

**Part B, learning from a stream** (mean bits per law, before the adaptive copy learns it):

| Stream | Half | Frozen | Self-adjusting |
| :--- | :--- | ---: | ---: |
| 120 invented laws | first | 19.63 | 13.48 |
| | second | 17.55 | **9.28** (53 % of frozen) |
| 24 other-field laws | first | 14.81 | 14.66 |
| | second | 17.03 | 15.31 (90 %) |

Final weights after the invented stream: automaton 0.84, tree 0.12, unigram 0.03, bayesian 0.02.

## Against the expectations

1. **A1, order on other-field laws (sign test p < 0.01 and median at most half): held**, on all 24 and
   on the 19 whose shape is not in training. The model's median position is about one twentieth of
   size order's.
2. **A2, selection from data not made worse: held**, but it shows nothing more than that: the search
   already recovered every law without the model, so this test could not have shown a gain.
3. **B1, the adaptive copy at most 90 % of the frozen copy's cost on the second half: held** (53 %).

## What this does and does not show (analysis after the run)

- The prior transfers beyond the formulas it was trained on, including to laws drawn at random, where
  it was not expected to help.
- "Position among the candidates" is a count, not time saved: the search still evaluates every
  candidate. Turning the order into saved work needs a search that stops early, with a rule for when,
  and that rule's error rate measured.
- The other-field laws are short (six nodes at most) and I chose them. The invented set was not
  chosen, but its laws are simple powers and sums.
- On the other-field stream the adaptive copy's gain is small (10 % in the second half over 12 laws);
  the large gain is on the invented stream, whose 120 laws share one generator's style. It learns a
  style it is shown many examples of; 24 mixed laws are too few to show much.
- Nothing here tests the model on data without a physical-law style, such as market data.
