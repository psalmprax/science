# PREREG-MMLM-ORDER-2026-11: MMLM's prior in the default route, against the size prior

**Status:** draft, not frozen. Written on 27 November 2026 on the owner's word "go ahead to test all" (02:32
Berlin time). That word followed the owner's question whether MMLM is a reliable model for APEDP-X; the answer
given was that it may help order a search on familiar ground, is not a discoverer, and is not used by the route
in force. This test checks that answer in the route in force. It is frozen by
`preregistration/DIGEST-MMLM-ORDER.json` after an independent review and a rehearsal, and run once on a branch
`mmlm-order-run-1`.

## What is compared

Both arms run the route in force, `apedp_dyn.discovery_l2.refined_search.discover_refined`, on the same
development rows, the same pool and the same seeds. They differ only in the prior of its guided search.

| Arm | Prior | What the prior does in the guided search |
| :--- | :--- | :--- |
| `default` | the size prior (`main_search.size_prior`), as in force | orders candidates by size; their structural code length |
| `mmlm` | MMLM's law model (`apedp_mmlm.law_language.LawLanguageModel`), fitted on SRBench's 133 ground-truth laws, pooled over domains, through `law_language.search_prior` | orders candidates most-plausible first, so the confirmed stop may come sooner; their structural code length in the description length, so it can also change which candidate wins |

- **The same searched candidates.** The default search stage (`search_stage.default_search_stage`, 240
  seconds) runs once per unit, outside both arms, in a pass of the route whose result is not used. Its
  candidates are recorded (`extended_route.Recording`) and replayed into both arms
  (`extended_route.replayed`). So nothing a search happened to find differs between the arms.
- **Prepared once.** As in use (`main_search.discover`'s `guided`), each arm's guided search is built once
  per process with every candidate priced, before the units run. The pricing is timed apart and reported. It is
  not route time.
- **Which arm runs first** alternates by a draw from the unit's name, so neither arm always pays the warm-up.

## Laws

- `invented`: 50 laws drawn at run time from the invented-law grammar (`scripts/build_invented_laws.py`) with
  seed 2026112731. No other test draws from this seed; the rehearsal draws from 2026112732. Left out are:
  - every structure of the invented-laws benchmark, by fingerprint only, so no held-back law is read;
  - the eighty laws the search proposers were developed on;
  - the memory test's forty (`benchmarks/memory_blind_laws.json`).
- `other_fields`: the 24 laws from chemistry, biology, engineering, finance and geometry of PREREG-MMLM-S
  (`run_prereg_mmlm_track_s.OTHER_FIELD_LAWS`). None is among MMLM's training laws.

Rows, noise, split and scoring are the SRBench runner's (`scripts/run_srbench_baseline.py`):
- inputs drawn on (1, 5); noise 0 and 1 %; 1,000 rows;
- 70 % of the training rows for development and 30 % as the pool;
- scoring on 500 noise-free rows.

That gives 74 laws × 2 noise levels = 148 units, each with both arms, in 20 shards. A unit has 1,800 seconds
for the search pass and both arms. A unit that times out, or an arm that raises, recovers nothing and reports
nothing.

## Expectations, fixed now

A law is **recovered** when the reported law is symbolically the truth (`symbolic`). It is **reported** when the
route outputs a law. It is **wrong** when it is reported but neither recovered nor accurate (test R² at least the
runner's `R2_ACCURACY`).

1. **No laws lost.** On each set (`invented`, `other_fields`; both noise levels together), `mmlm` recovers at
   least as many laws as `default`.
2. **Faster.** Over the units where both arms finished, `mmlm`'s median route time is at most 0.8 times
   `default`'s. The route time excludes the search pass, which is shared, and the pricing, which is done once.
3. **No more wrong laws.** Over all units, `mmlm`'s share of wrong laws among those it reports must meet three
   conditions:
   - at most `REPORTED_WRONG_SHARE_MAX` (5 %);
   - its one-sided exact upper bound at `REPORTED_WRONG_SHARE_BOUND_CONFIDENCE` (95 %) at most
     `REPORTED_WRONG_SHARE_BOUND_MAX` (10 %); all three are from `apedp_dyn.conf.discovery_checks`, defined
     for exactly this, the share of wrong laws a method reports in a test;
   - its count of wrong laws at most `default`'s.

The 0.8 of expectation 2 is this test's own number, fixed here and frozen with it. It is not a check of the
referee.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

- **All three hold:** propose MMLM's prior as the route's prior in place of the size prior. That would change
  the route in force, so it needs the owner's explicit word and goes through the canonical-spec steps if the
  owner agrees.
- **Any one fails:** list MMLM as the route's prior as retired (deprecated, kept for replication; nothing in
  force changes), under the owner's standing word of 27 November 00:41 ("retire anything that does not meet
  our expectation"). MMLM's other experimental uses are not touched by this test.

Run-for-run gains and losses (a law recovered by one arm and not the other), the timeouts and errors, and the
pricing time are reported beside the expectations.

## Limits

- **Familiar ground is not tested.** MMLM was fitted on SRBench's laws, and there is no clean held-out set of
  physics laws left: the eight held-back real tables are used up, and the forty held-back invented laws are
  reserved. Both sets here are new to MMLM. A pass would say it helps beyond its training ground. A failure
  would not show it cannot help on laws like those it was fitted on.
- **One search per unit.** The search stage runs once, and both arms see its candidates. A different search
  run could give the route other candidates. That variation is the same for both arms.
- **CPU timings on shared runners.** Each unit runs both arms on the same runner, so expectation 2 is read on
  paired times. The runners themselves differ.

## Disclosed before freezing

- The 24 other-field laws were read once before, by PREREG-MMLM-S, to test the sibling equation model
  (`apedp_mmlm/equation_language.py`, Track S), which put the law at a median position of 303.5 against 6,111.5
  in size order. The law model tested here (MMLM-L) was not built or tested on them. Its own preregistered test
  (PREREG-MMLM-L) found it orders candidates no better than that earlier model (median position 20 against
  24.5 on ODEBench laws).

- The route's `structure_bits` is a code length as well as an order. This plan says so, and the arm is named
  "MMLM's prior" rather than "MMLM's order".
- Smoke checks (local, no search engine: the proxy blocks the Julia download for PySR, so the search stage
  added nothing): on two smoke laws at 1 % noise both arms recovered both. Route times were 0.13 and 1.04 s for
  the ratio law (`mmlm` first) and 0.64 and 0.33 s for the power law. They are not a reading of anything.
- Seeds 20261203 and 20261204, first written here, are drawn by `scripts/run_vouching_v2_blind.py`. They were
  replaced before anything ran.
