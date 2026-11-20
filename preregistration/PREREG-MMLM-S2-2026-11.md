# Preregistration: does MMLM's prior save search work, transfer to unchosen laws, and learn by domain?

**Document ID:** PREREG-MMLM-S2-2026-11
**Frozen:** at the commit that adds this file; `preregistration/DIGEST-MMLM-S2.json` holds the SHA-256 of
this document, `scripts/run_prereg_mmlm_track_s2.py`, the script it imports, the model, the ledger
learner, the guided and exhaustive searches and the SRBench manifest. The script refuses to run if any
changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Why

PREREG-MMLM-S-2026-11 held, and its report listed what it did not show. This run takes three of those:
the search order was a count and saved no work; the laws were short and chosen by me; and learning was
shown only on one style of law fed directly, not from the ledger by domain.

## Part 1: the prior-ordered search with an early stop

`guided_search.GuidedSearch` fits candidates most-plausible first and stops when the best candidate's
residuals cannot be predicted from their nearest neighbours' residuals (leave-one-out R² ≤ 0, k = 8).
Its stop rule was written and checked on SRBench's 30 reachable formulas only. Here it meets:

- the 24 other-field laws of the first preregistration (the model was never fitted or tuned on them);
- the first 40 search-reachable laws from the random generator with a seed never used (20261123).

Each at 1 % and 5 % multiplicative noise, 300 rows: 128 runs. Both the guided search and the plain
exhaustive search (no prior, every candidate) run on the same data; recovery is the baseline's
symbolic-recovery check. The model is fitted on SRBench's 133 formulas and not updated.

## Part 2: laws nobody chose, longer than the search can reach

The model is fitted on the 119 Feynman formulas only and prices the 14 Strogatz ODE right-hand sides:
another field, not selected by me, and long (none is within the search's reach). Measured in bits per
token against a uniform code and a unigram code fitted on the same Feynman formulas.

## Part 3: learning from the ledger, by domain

Seed: the Feynman formulas. Two domains are interleaved through a real `CandidateTrialLedger` as
confirmed records: "dynamics" (the Strogatz laws) and "invented" (60 laws, generator seed 20261124,
never used). After each record `DomainLearner.sync(ledger)` runs. Each law's cost is recorded before it
is learned, for the per-domain models, for one pooled self-adjusting model fed everything, and for a
frozen model. One Strogatz law does not survive the ledger's text form (13 of 14 parse; checked without
the model); it is counted as unparsed.

## Expectations, fixed now

1. **No laws lost:** over the 128 runs, the guided search recovers at least as many laws as the plain
   search, less 5 % of the runs (at most 6 fewer).
2. **Few wrong stops:** runs that stop early on a law that is not the true one are at most 5 % (at
   most 6).
3. **Real work saved:** the median share of candidates fitted is at most 5 %, and the guided search's
   total time over the 128 runs is at most 20 % of the plain search's. The one-time cost of ordering
   the candidates is reported beside it and not included.
4. **Transfer:** on the Strogatz laws the model's bits per token are below both the unigram's and the
   uniform code's.
5. **Domains do not hurt:** on the second half of the invented domain's laws, the per-domain model's
   mean cost is at most 1.02 times the pooled model's.
6. Reported without a criterion: everything for the dynamics domain (13 laws is too few to demand a
   result), and per-set, per-noise search numbers.

A miss on 1 to 5 is reported as a miss.

## Deviation rule

One run, locally. A crash is fixed by a dated deviation committed here before re-running. Changing a
rule, a law set or an expectation after the run is a new preregistration.
