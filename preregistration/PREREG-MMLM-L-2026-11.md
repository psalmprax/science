# PREREG-MMLM-L-2026-11: the scientific model of MMLM on laws it has not seen

**Status:** frozen before the run by `preregistration/DIGEST-MMLM-L.json`. Runs once:
`python scripts/run_prereg_mmlm_law.py --out preregistration/RESULT-MMLM-L-2026-11`.

## What is tested

`apedp_mmlm/law_language.py` (MMLM-L) is a language model over laws: counting experts that each see one
view of the partial expression tree, mixed with weights fitted on held-out laws, with a pooled and a
per-domain version of four views, and a canonical token order that does not depend on variable names.
It replaces `SelfAdjustingEquationModel` (the "earlier model") as the prior for law discovery if it is
better where it matters:

1. as a language model, on laws from a source that was never used to build either model;
2. as a search prior, with an early stop that is safe (`apedp_dyn/discovery_l2/confirmed_stop.py`).

## Data

- **Training:** SRBench's 133 ground-truth laws (`benchmarks/srbench/manifest.json`), each with a
  domain: the Feynman volume (`feynman_I`, `feynman_II`, `feynman_III`, `feynman_test`) or `dynamics`
  for the 14 Strogatz systems.
- **Fresh laws:** ODEBench (the benchmark of the ODEFormer paper, 63 dynamical systems), one law per
  component, parameters replaced by the first published values. The file is pinned by SHA-256 in
  `apedp_mmlm/law_corpora.py`. Components whose canonical tokens equal a training law's, or an earlier
  component's, are removed; the counts are reported. They are scored in the domain `dynamics`.
- **Search laws:** 40 laws from the blind benchmark's generator with a seed not used before
  (20261125; at most three variables, within the enumerated space), and every fresh ODEBench law with
  one to three variables that the enumeration can express. Each at 1 % and 5 % relative noise, 300
  rows, inputs uniform on [1, 5]. For an ODEBench law the right-hand side is regressed on sampled
  states, not on a simulated trajectory.

## Protocol

**Part 1, language.** Both models and a unigram are fitted on the 133 training laws and then score
the fresh laws in file order, in two modes: *frozen* (nothing is learned) and *learning* (each law is
scored, then learned, as a stream of confirmed laws would be). The same is repeated with the Feynman
laws only as training, so that `dynamics` is a domain with no laws at the start.

**Part 2, search.** For every search law and noise level: the plain exhaustive search; the earlier
prior and the new prior, each with the first-stop rule of PREREG-MMLM-S2 and with the confirmed
stop. A law counts as recovered by the benchmark's symbolic-recovery check on 200 unseen rows. A
wrong early stop is a run that stopped early on a law that is not the true one.

## Expectations, fixed now

1. **Better language model on fresh laws:** with all 133 laws as training, the law model's mean bits
   per fresh law are below the earlier model's, frozen, and it is lower on more laws than it is higher
   (paired sign test, p < 0.05).
2. **Learns faster:** in learning mode the law model's mean bits per fresh law are below both the
   earlier model's and the unigram's, and the sign test against the earlier model has p < 0.05.
3. **Not a worse search order:** over the search laws, the median position of the true law under the
   law prior is not higher than under the earlier prior.
4. **The confirmed stop is safe:** with the law prior, wrong early stops are at most 5 % of the runs.
5. **No laws lost:** with the law prior and the confirmed stop, laws recovered are at least the plain
   search's less 5 % of the runs.
6. **Work is still saved:** with the law prior and the confirmed stop, the median share of candidates
   fitted is at most 10 %, and total search time is at most 25 % of the plain search's (the one-time
   cost of ordering the candidates is not included, as in PREREG-MMLM-S2).
7. Reported without a criterion: everything for the Feynman-only training (see below), the other
   three search arms, and per-set numbers.

A miss on 1 to 6 is reported as a miss.

## Disclosed before freezing

- All development used laws already seen: the 133 SRBench laws, the 24 other-field laws and the 40
  invented laws of the earlier preregistrations. On those: 10-fold cross-validated cost 25.8 bits per
  law for the law model against 31.5 for the earlier model; with the confirmed stop the search test of
  PREREG-MMLM-S2 would have had 3 wrong stops in 128 runs (2 with the earlier prior) against 7, at a
  median 3.2 % of candidates fitted.
- **A weakness seen in development, not fixed.** Trained on the Feynman laws only, the law model
  prices the 14 Strogatz laws at 32.5 bits per law frozen, against 30.8 for a unigram. It overtakes
  once it learns from the stream (25.9 against 30.4). The Feynman-only arm here is expected to show
  the same on ODEBench: no better than a unigram when frozen, better when learning. No criterion is
  set on it.
- The search order with the law prior was better than the earlier prior's on the other-field laws
  (median position 220.5 against 303.5) and equal on the invented laws (7 against 7). That is why
  expectation 3 says "not higher" and not "lower".
- I have not looked at the ODEBench equations beyond the first entry of the file and the counts of
  operators and functions used (needed to write the parser). I know the benchmark's contents in
  general terms from the literature.
- One law of the 133 still reads differently under renaming (`strogatz_barmag2`): the algebra system
  writes `sin(a - b)` or `-sin(b - a)` depending on the names. Not fixed.

## Deviation rule

The run is not repeated. A crash is fixed only if the fix cannot change a number (for example an
output path), the fix is recorded with its date in the result, and the digest is re-frozen with the
old digest kept in the log. Any other change makes the result a development result.
