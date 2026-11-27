# PREREG-MMLM-ORDER-2026-11: MMLM's prior against the size prior, in the table route in force

**Status:** draft, not frozen; revised after the independent review of 27 November (below). Written on 27 November
2026, on the owner's word "go ahead to test all" (02:32 Berlin time).

That word followed the owner's question whether MMLM is a reliable model for APEDP-X. The answer given was that it
may help order a search *on familiar ground*, is not a discoverer, and is not used by the route in force. This test
checks that answer where it should hold: on laws whose shapes MMLM has met before, with a guard set where it cannot
help.

The plan is frozen by `preregistration/DIGEST-MMLM-ORDER.json` after a check of the revision and a rehearsal, and
runs once on a branch `mmlm-order-run-1`.

## What is compared

The route is the table route in force, `vouched_route_v2`, in three parts:
- the search stage in force (`stage_in_force.search_stage`: the search portfolio, with the memory of confirmed laws
  beside it);
- the refined path (`refined_search.discover_refined` → `main_search.discover`);
- the second version of the last checks (`vouched_route_v2.checked`).

The two arms differ only in the prior the route's guided search is given (`structure_bits`).

| Arm | Prior |
| :--- | :--- |
| `default` | the size prior (`main_search.size_prior`), as in force |
| `mmlm` | MMLM-L (`apedp_mmlm.law_language.LawLanguageModel`), fitted on SRBench's 133 ground-truth laws, pooled over domains, through `law_language.search_prior` |

**What the prior does in the route.** It reaches these places:

- **First-pass order and code length.** In the first pass it orders the candidates, most plausible first, so the
  confirmed stop may come sooner (`guided_search`, `confirmed_stop`). It is also their structural code length in
  the description length.
- **The full search after the first pass.** Where no candidate stops the first pass early, it prices every single
  term and every sum of two terms there (`exhaustive_search`).
- **Inputs to confirmation.** Through the first-pass law, it decides whether the first pass "settles" the table.
  It also sets the order of candidates given to confirmation, which decides ties among equivalent laws and which
  candidates fit under the limit of twelve.
- **Its weight.** MMLM's code lengths run 4 to 51 bits (median about 35), against 1 to 6 for the size prior.

Two parts of the route are left alone:

- the deep pass of the search stage keeps its own size prior;
- the search stage, which is shared (below).

**The same candidates, and neither arm favoured.** For each unit (one law at one noise level), these steps run:

1. Both arms' first-pass laws are computed.
2. The search stage in force runs once, in its own process, seeded with both first-pass laws.
3. Its candidates are recorded (`extended_route.Recording`) and replayed to each arm (`extended_route.replayed`).
4. Each arm runs in its own process, forked from the same state, so neither warms the other, and one arm's failure
   or time-out cannot erase the other's row.
5. Each arm builds its guided search on every call, as the route does, so MMLM's pricing is in its time.

A unit whose search stage fails (a member raises, or nothing is proposed) is marked `search_failed`. That unit is
left out of every comparison.

## Laws

There are four sets. The first three are **familiar ground**: MMLM, or the model it grew from, has met these shapes
in development. That is where its claimed help should show. The fourth is a **guard**.

| Set | Laws | What they are | Why familiar |
| :--- | :--- | :--- | :--- |
| `other_fields` | 23 | The other-field laws of PREREG-MMLM-S (`run_prereg_mmlm_track_s.OTHER_FIELD_LAWS`), without `reynolds_number`, which is the same law as `ideal_gas_pressure` (`x0*x1/x2`). | MMLM-L was developed on them (PREREG-MMLM-L, lines 66 to 67 and 76 to 77). Five have a shape in its training laws: `beer_lambert`, `compound_growth` and `radioactive_decay` (as `feynman_I_40_1`), `drag_force` (`feynman_II_27_16`), `cylinder_volume` (three Feynman laws). Results are reported with and without those five. |
| `odebench` | 12 | ODEBench's right-hand sides with at most three inputs that the first pass reaches (`a*f+b` within six nodes), one per shape, none sharing a shape with the set above. | MMLM-L's preregistered test read ODEBench once. |
| `power_laws` | 30 | The blind benchmark's generator (`run_blind_discovery_benchmark.invented_laws`: powers and products of powers, as in PREREG-MMLM-L) with seed 2026112733, which no other test draws: the first 30 reachable shapes, at most three inputs, one per shape, none sharing a shape with the sets above. | The generator has about fifty reachable shapes, and MMLM-L was developed on forty of its laws. |
| `guard` | 30 | The invented-law grammar (`scripts/build_invented_laws.py`) with seed 2026112731, which no other test draws. Left out are the invented-laws benchmark's structures (by fingerprint), the eighty laws the search proposers were developed on, and the memory test's forty. | Mostly out of the first pass's reach, so the order cannot help: a test that MMLM's prior does no harm there. |

Rows, noise, split and scoring follow the SRBench runner:
- inputs on (1, 5);
- noise 0 and 1 %;
- 1,000 rows, 70 % for development and 30 % as the pool;
- 500 noise-free rows for scoring.

That makes 95 laws and 190 units. The search budget is 240 seconds; the search's process has 1,440 seconds and each
arm's process 900. Laws are dealt to 20 shards in turn, both noise levels of a law on the same shard.

A law is **recovered** when the law reported after the last checks is symbolically the truth. It is **reported**
when the route outputs a law after the last checks (`RefinedResult.law`, confirmed or of the equivalent class, and
then vouched). Tentative laws are not reported. A law is **wrong** when it is reported but neither recovered nor
accurate: test R² at least `DISCOVERY_ACCURACY_R2` from `apedp_dyn.conf.discovery_checks` (0.999), the runner's
`R2_ACCURACY`.

## Expectations, fixed now

All are read over the units where both arms finished.

1. **A gain on familiar ground.** Over `other_fields`, `odebench` and `power_laws`, MMLM's arm recovers a law the
   default's does not in more units than the reverse. A one-sided exact sign test on those discordant units must
   give p ≤ 0.05.
2. **No harm on the guard.** On `guard`, the units where the default recovers a law and MMLM's arm does not are no
   more than the reverse.
3. **No more wrong laws.** Over all units, MMLM's arm reports no more wrong laws than the default's. The share of
   wrong laws and its bound (`report_share.verdict`: `REPORTED_WRONG_SHARE_MAX` 5 % and
   `REPORTED_WRONG_SHARE_BOUND_MAX` 10 % at `REPORTED_WRONG_SHARE_BOUND_CONFIDENCE` 95 %) are reported for both
   arms. They are not read: they measure the route, not the prior.
4. **Not slower end to end.** Paired by unit, MMLM's median end-to-end time is at most 1.1 times the default's.
   End-to-end time is the shared search plus the arm's route time, pricing included.

The test is **not read** in three cases:

- a unit is missing;
- more than a fifth of the units have a failed search;
- a loaded repository file differs from the checkout.

The p ≤ 0.05, the 1.1 and the fifth are this test's own numbers, frozen with it. They are not checks of the referee.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

- **All four hold:** MMLM's prior helps on familiar ground, and the test's claim stands there. That is not enough
  to put it in the route, whose laws are not familiar ground. The proposal is a blind test on reachable laws from
  a new source before anything goes in force; no such source is in the repository today. MMLM stays experimental
  and the route is unchanged.
- **Any one fails:** MMLM is listed as retired as the route's prior (deprecated, kept for replication; nothing in
  force changes), under the owner's standing word of 27 November 00:41 ("retire anything that does not meet our
  expectation"). This test does not touch MMLM's other experimental uses.
- **Not read:** a rerun is put to the owner.

The following are reported beside, and not read:
- the run-for-run gains and losses by set;
- the `other_fields` results with and without the five training shapes;
- the units where the two arms' first-pass laws differ;
- the timeouts, errors and failed searches;
- each arm's route time.

## Limits

- **No blind reachable set.** The familiar sets are familiar by construction: this test can retire the claim, not
  establish it for new fields. A pass leads only to a blind test.
- **The earlier evidence points one way.** PREREG-MMLM-L found that MMLM-L's prior orders candidates no better than
  the earlier model on ODEBench laws (median position 20 against 24.5). Its search gain came from the confirmed
  stop, not the prior.
- **The route as code says it is in force.** `vouched_route_v2` was put in force by an agent on a preregistered
  consequence, without the owner's word, hours after the owner had taken the first version out. The module says
  so and offers `APEDP_TABLE_VOUCHING=off`. This test uses the route as the code has it. If the owner takes the last
  checks out, the reported laws here are the refined path's after checks that are no longer in force; the
  run-for-run comparison before the checks is in the rows (`route_law`).
- **CPU timings on shared runners.** Each unit runs both arms on one runner, so expectation 4 is read on paired
  times; the runners themselves differ. Each row records the runner's cores and the search stage's description.

## Disclosed before freezing

- **The independent review of 27 November** found thirteen problems in the first draft. Each was addressed:
  - Speed was the draft's main expectation, but the timed part was a few percent of a discovery, pricing was
    repeated in the full search, and "as in use" was false. Speed is now a not-slower guard, end to end, with
    pricing per call as the route does it.
  - The draft said MMLM had not been built or tested on the other-field laws. MMLM-L was developed on them, and
    five share shapes with its training laws. The sets are now labelled familiar, the duplicate is removed, and the
    training shapes are reported apart.
  - One arm's error erased the other's row, and a failed search counted as a normal unit. Each arm now runs in its
    own process, and a failed search is marked.
  - The draft's route was the refined path without the memory or the last checks. It is now the table route in
    force.
  - The shared search was seeded by the default arm's first-pass law only. It is now seeded with both.
  - What `structure_bits` does was described incompletely. It is described in full above.
  - Collect checked nothing and lost the pricing files. It now checks units, digests and loaded files.
  - The frozen list missed files the result depends on. It is extended, and each shard records the digest of every
    repository file it loaded.
  - The draft's expectations 1 and 3 could pass on ties, and its limits measured the route. Expectation 1 now
    asks for a gain, and the limits are reported, not read.
  - The order of the arms was unbalanced. Each arm now has its own fork.
  - One rehearsal law shared a shape with a final law. The rehearsal now draws only shapes the final sets leave out.
  - Shards split the noise levels, Julia was installed inside a unit, and a time-out left processes running. Laws
    are now dealt by law, Julia starts once per shard, and process groups are killed on a time-out.
  - The wording is corrected.
- **The 24 other-field laws, the twelve ODEBench laws and the generator's shapes have all been read before.** That
  is why they are called familiar here.
- **First rehearsal** (`mmlm-order-smoke-1`, run 45932078529, the first draft's runner, 15 guard-like laws, 60
  seconds of search): both arms recovered 4 of 30 runs, with no run-for-run difference. MMLM's arm was slower, with
  a median of 5.09 against 4.48 seconds. It reported 11 laws in each arm, none wrong. These are development numbers
  on other laws. A second rehearsal runs the revised runner before freezing.
- Seeds 20261203 and 20261204, first written into the draft, are drawn by `scripts/run_vouching_v2_blind.py`. They
  were replaced before anything ran.
- Smoke checks (local, no search engine: the proxy blocks Julia's download): both arms recovered both smoke laws. A
  local run where PySR could not start was correctly marked `search_failed` by the revised runner. The memory still
  proposed three candidates there, which the first draft would have counted as a normal unit.
