# RESULT-MMLM-ORDER-2026-11: MMLM's prior against the size prior, in the table route in force

**Status:** recorded on 27 November 2026, after an independent check of every number (below). The preregistered run
of `preregistration/PREREG-MMLM-ORDER-2026-11.md` (frozen by `DIGEST-MMLM-ORDER.json`) ran once.
- Run 45953548428, branch `mmlm-order-run-1`, the freeze commit 04277d3. It is the only preregistered run.
- The collection's record is kept unedited in
  [`RESULT-MMLM-ORDER-2026-11/results.json`](RESULT-MMLM-ORDER-2026-11/results.json).
- `RESULT-MMLM-ORDER-2026-11/shards/` holds every shard's rows and its map of loaded files, byte for byte as each
  shard printed them in its log.

## What ran

All 190 units ran: 95 laws at noise 0 and 1 %, both arms each, in 20 shards. Every unit and arm has exactly one row.
- No search failed and no arm raised or timed out.
- Every row ran the route in force: the second version of the last checks, the memory beside the searches, four
  cores ("four searches side by side, 240 s; and laws grown from 28 remembered motifs").
- The shards loaded 263 repository files. Of those, 261 are in the map frozen with the plan, and none differs from
  it. The other two are the runner and `apedp_mmlm/law_corpora.py`, both frozen by name and unchanged. The map's
  one file the shards did not load is `report_share.py`, which only the collection uses.

The two arms' first-pass laws differed in 13 units, each of which got one search per arm. The other 177 units
shared one search, so both arms were given identical candidates.

## The expectations, as `summarise` read them

| Expectation | Result | Held |
| :--- | :--- | :--- |
| 1. A gain on familiar ground (other-field, ODEBench and power laws): more laws recovered run for run, sign test p ≤ 0.05 | 0 gained, 0 lost; p = 1 | **no** |
| 2. No harm on the guard: losses no more than gains | 1 gained, 0 lost | yes |
| 3. No more wrong laws than the default | 0 and 0 | yes |
| 4. Not slower end to end: median at most 1.1 times the default's | 321.7 against 320.0 seconds, 190 units | yes |

The test is read. Expectation 1 does not hold, so the outcome is the plan's second: **MMLM is listed as retired as
the route's prior** (deprecated, kept for replication; nothing in force changes). It was never the route's prior,
so the route is unchanged. MMLM's other experimental uses are not touched.

## By set

| Set | Units | Recovered (default / MMLM) | Reported | Wrong | First pass: median candidates evaluated (default / MMLM) | First pass stopped early (default / MMLM) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| other_fields | 46 | 45 / 45 | 45 / 45 | 0 / 0 | 29,593 / 16,770 | 17 / 25 |
| odebench | 24 | 22 / 22 | 22 / 22 | 0 / 0 | 9,560 / 430 | 14 / 16 |
| power_laws | 60 | 58 / 58 | 58 / 58 | 0 / 0 | 20,717 / 4,575 | 26 / 35 |
| guard | 60 | 22 / 23 | 37 / 38 | 0 / 0 | 29,593 / 29,593 | 5 / 4 |
| all | 190 | 147 / 148 | 162 / 163 | 0 / 0 | 29,593 / 14,563.5 | 62 / 80 |

The medians pool laws with different numbers of inputs.

- **Other-field laws.** The five with a shape among MMLM's training laws were recovered 10 times each way; the other
  18, 35 times each way.
- **Wrong laws among the reports** (`report_share`, reported, not read): 0 of 162 for the default and 0 of 163 for
  MMLM, upper bounds 1.83 % and 1.82 %. All 15 reported laws per arm that are not the truth have a test R² of at
  least 0.999.
- **The one unit where the arms differ** is on the guard: `guard_05` at 1 % noise
  (`0.3*sqrt(x0 + 0.7)/(x0**2 + 3.7)`). It shared one search, so both arms had identical candidates. Before the
  last checks the default had no law, and MMLM's arm had an accurate law that was not the truth. The last checks
  replaced that law with another member of its class, which is the truth.
- **Before the last checks** (`route_law`) there is no run-for-run difference in any set: 148 recovered each way,
  127 on familiar ground. Each arm reports 167 and 168 laws there, 2 of them wrong each (`guard_15` and
  `ode_odebench_05_0`, both at 1 %), with upper bounds of 3.72 % and 3.70 %. With the checks out, the outcome
  would be the same: expectation 1 fails at 0 and 0, while 2 and 3 hold.
- **The last checks' effect.** They withheld two true laws per arm on familiar ground (`power_06` and
  `ode_odebench_57_0` at 1 %), which is the drop from 127 to 125. They turned `guard_17` at 1 % into the truth in
  both arms.
- **The 13 units with a search per arm:** `field_compound_growth` (1 %), `ode_odebench_05_0` (1 %), and on the
  guard 01, 04 and 13 at both noise levels, and 12, 21, 24, 25 and 29 at 1 %.
  - In two of them only one arm reported a law: `guard_04` at 1 % (the default) and `guard_12` at 1 % (MMLM),
    neither the truth.
  - In four of them MMLM's replayed candidates held the default's first-pass law, coefficients and all:
    `ode_odebench_05_0`, `field_compound_growth`, `guard_24` and `guard_29`, all at 1 %. The reverse never
    happened.
- **Route time,** without the search: median 7.31 seconds for MMLM's arm against 5.84 for the default's. MMLM's
  arm was longer in 169 of 190 paired units, by a median of 1.92 seconds.

## What it means

- **On familiar ground MMLM's first pass does less work, and the route reports the same laws either way.**
  - On the familiar sets its first pass evaluated fewer candidates in 66 of 130 units, more in 9 and the same in
    55.
  - It stopped early in 19 units where the default did not, and never the reverse. But it stopped on the same
    first-pass law as the default in 128 of 130.
  - The first pass found the route's final law in 6 units for MMLM against 5 for the default.
  - In 128 of the 130 familiar units both arms were given identical candidates. Those come from a search stage
    whose deep pass uses the size prior at greater depth, and confirmation picked the same law from them.
- **This answers the owner's question in the route's own terms.** The claim made then was that MMLM "may help order
  a search on familiar ground". This run shows less first-pass work there, reported beside and not read, and no
  change in what the route reports, which is what was read. MMLM is retired as the route's prior. The result agrees
  with PREREG-MMLM-L, which found that MMLM-L's prior ranked candidates no better than the earlier model's, and
  that the search's gain came from the confirmed stop.
- **The route was slower with MMLM by about two seconds a unit,** against a search of about five minutes (median
  313.5 seconds). The rows do not measure where those seconds go.

## Limits

- **Little room for a gain.** The route recovers almost everything reachable either way: 125 of 130 on familiar
  ground. 177 of the 190 units had identical candidates, and only 2 of the 13 units with a search per arm are on
  familiar ground. The plan's Power paragraph said before the run that expectation 1 had little room.
- **All three familiar sets had been read before.** They are familiar by design. A pass would only have led to a
  blind test, so a failure there is the stronger reading.
- **The other arm's first-pass law can come back.** The plan said each arm getting the other's first-pass law "never
  happens in use". In use, though, the stage's deep pass, which uses the size prior, gives the size prior's law to
  any route, MMLM's included. It did so in 4 of the 13 units with a search per arm, and in the 177 shared units it
  gave the same candidates to both arms. That narrows what the comparison isolates: it is the prior's effect inside
  the route as it is, not the prior on its own.
- **Searches run per arm differ by more than the prior.** In those 13 units PySR ran twice under a wall-clock
  budget, so a difference there cannot be put down to the prior alone.
- **The route as the code has it in force.** The second version of the last checks was put in force by an agent,
  without the owner's word. The one unit where the arms differ arises from those checks. Before them there is no
  difference, so the outcome stands either way (above).
- **One run.** The guard's "no harm" rests on a single unit.
- **CPU timings on shared runners.** Each unit's two arms ran on the same runner, but the runners differ from one
  another.
- **The runner's search-process limit and description differ from the plan's, with no effect here.** The runner gave
  the search's process `2 * SEARCH_LIMIT_S + budget` (2,640 seconds), where the plan says 1,440. Neither was
  approached: the longest unit's two searches took 706 seconds together. The runner's docstring still describes
  one shared search seeded with both laws, while the code does what the plan says.

## Independent check (AGENTS.md 1.6)

A reviewer that took no part in the run worked from the rows with its own scripts.
- **Provenance.** It confirmed the run's provenance. The shard rows, the loaded maps and `results.json` are
  byte-identical to the logs, and the 45 frozen files and the 262-file map match the checkout.
- **The laws.** It confirmed that the 95 laws are the plan's construction, with the training flags on exactly the
  five named laws and every familiar shape distinct and reachable. Each law's two noise levels ran on the shard it
  was dealt to.
- **The numbers and the outcome.** Every number recomputes exactly, and the outcome is the plan's consequence.

It found six points. None changes the outcome. Each is now in the text above:

- **The cross-replay count was false.** `other_first_pass_replayed_units: 0` came from a string comparison that
  cannot match, because first-pass laws are stored as sympy prints and replayed formulas in the stage's own text.
  Checked by value, it is 4 of 13. The count in `results.json` is wrong and is left as recorded; the right one is
  here.
- **The loaded-files sentence was inaccurate, and the check is one-sided.** The collection's check
  (`frozen_loaded.get(f, h) != h`) lets through a loaded file missing from the frozen map, and never notices a
  frozen file that was not loaded. Here no loaded content differs.
- **The only discordant unit comes from the last checks, and the plan's limit on them was missing.** Both are now
  stated, with the comparison before the checks.
- **"What it means" claimed more than the rows show.** The first pass's work is now described, not read as a
  result. The earlier evidence is set beside it, the cause of the extra route time is not asserted, and the claim
  is quoted as the plan wrote it.
- **The limits left things out.** The missing limits are added.
- **Plan and runner disagree on the search-process limit, and the docstring is stale.** Both are disclosed here.

The frozen files are not edited.
