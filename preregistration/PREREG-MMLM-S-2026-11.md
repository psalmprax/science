# Preregistration: does MMLM's equation model help discovery, and does it learn?

**Document ID:** PREREG-MMLM-S-2026-11
**Frozen:** at the commit that adds this file; `preregistration/DIGEST-MMLM-S.json` holds the SHA-256 of
this document, `scripts/run_prereg_mmlm_track_s.py`, `apedp_mmlm/equation_language.py`,
`apedp_dyn/discovery_l2/exhaustive_search.py`, the two scripts imported and the SRBench manifest. The
script refuses to run if any changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Why

A development study found that a language model of equations (MMLM Track S), cross-validated on
SRBench's 133 formulas, put a held-out law at a median position of 112 in the search where size order
put it at 6,268. That was trained and tested on the same kind of formula, with the tokenisation and
the comparison chosen after seeing the data. Since then the model has been made self-adjusting
(`SelfAdjustingEquationModel`: a mixture of four engines that re-weights itself by how well each
predicted every confirmed law, then learns the law). Neither claim has been tested on anything fresh.

## Part A: the model as a prior, on laws from elsewhere

The model is fitted once on all 133 SRBench formulas and not updated during part A.

- **Other-field laws.** 24 laws from chemistry, biology, engineering, finance and geometry, listed in
  the script (`OTHER_FIELD_LAWS`), written before the model was run on any of them. All 24 are among
  the search's candidates (checked without the model). Five have the same token shape as some training
  formula (`beer_lambert`, `compound_growth`, `cylinder_volume`, `drag_force`, `radioactive_decay`);
  results are reported for all 24 and for the 19 without them. They are my selection, made knowing the
  model favours products and ratios, which is why the second set exists.
- **Invented laws.** The blind benchmark's random generator with a seed never used (20261122): the 76
  of its 120 laws the search can reach. Nobody chose these.

Measured: (1) the law's position among the candidates in the search's own order and in order of the
model's code length; (2) for the other-field laws, with 300 rows of data at 1 % noise, whether
`exhaustive_search.search` selects the true law with the model as its structure cost and without it,
scored by the baseline's symbolic-recovery check.

## Part B: does feeding it laws make it better?

Two copies of the SRBench-fitted model read a stream of 120 invented laws (seed 20261121, never used).
One is frozen. The other is told each law after coding it (`observe`): it adjusts its weights and
learns the law. Each law's cost in bits is recorded for both before the adaptive copy learns it. The
same is done for the 24 other-field laws as a second stream.

## Expectations, fixed now

1. **A1, order on other-field laws:** the model places the law earlier than size order more often than
   later (one-sided sign test, p < 0.01), and the median position under the model is at most half the
   median in size order. Required on all 24 and on the 19 whose shape is not in training.
2. **A2, selection from data:** with the model, the search recovers at least as many other-field laws
   as without it, less one (it must not make selection worse).
3. **B1, learning:** on the second half of the invented stream (laws 61-120), the adaptive copy's mean
   cost is at most 90 % of the frozen copy's.
4. Reported without a criterion: order on the invented laws (the model was trained on physics and these
   are random, so it may not help or may hurt), the other-field stream, and the final weights.

A miss on 1, 2 or 3 is reported as a miss.

## Deviation rule

One run, locally. A crash is fixed by a dated deviation committed here before re-running. Changing the
laws, the model or an expectation after the run is a new preregistration.
