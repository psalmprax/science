# What gets added: every solution tested on 26 November, with its hybrids, by category

The owner asked on 26 November (20:32) for every suggested solution to be tested with hybrid approaches
where possible, and for the results to be sorted "so that we know what will be added finally into the
system". This register covers what was suggested and tested that day: the four unconventional approaches
to GF(2^12) multiplication, the patent-derived count of independent trials, the loops of the non-neural
program search, the quantum-inspired language model, and the video approaches. Each entry names its
evidence in the repository. Nothing here is in force: an entry in category A goes in only on the owner's
word, after its preregistered test and independent review (AGENTS.md 1.5 and 1.6).

Categories:

- **A. To add, on your word** — it passed its tests, and the steps before it can go in are named.
- **B. Keep as a tool or an option** — useful for a purpose, not a default.
- **C. Do not add** — tested; no gain, or unsafe.
- **D. Waiting for results** — its decisive run has not finished; the category it moves to is fixed in
  advance by its preregistration.
- **F. Kept, with a design** (added 27 November) — showed no gain, put up for retirement, and kept on the
  owner's word with a redesign that is not built.

## A. To add, on your word

| Solution | Hybrids tried | Evidence | Before it goes in |
| :--- | :--- | :--- | :--- |
| **Counting independent trials by clusters** for the deflated Sharpe ratio (after US20190294990A1 and López de Prado and Lewis) | four: cluster series with an eigenvalue count across clusters; the trial's own Sharpe ratio with the cluster count; both counts; either count | Simulated searches: false alarms at most 1 in 300, as the referee's count; power far higher where strategies come in families (e.g. 85 % against 19 %). Real prices (six series, 56 strategies): false alarms 0.17 % as the referee's count; planted edges found 52 to 62 % of the time against 0 %. No hybrid beat it clearly; "own Sharpe ratio with the cluster count" is unsafe (11 to 35 % false alarms on real prices). `benchmarks/referee_trials/independent_trials_dev.json`, `independent_trials_real_dev.json`, `independent_trials_hybrid_dev.json` | a preregistered blind test, an independent review, and your words "make it the default"; it changes the referee, so the trustees' signing rules apply to any change of the specification |

## B. Keep as a tool or an option

| Solution | Hybrids tried | Evidence | Use |
| :--- | :--- | :--- | :--- |
| **JetFlow's Jet2 step, fixed** (`packages/jetflow-video/jetflow_video/stepper.py`; `RESULT-VIDEO-FLOW-2026-11`) | on MeanFlow's velocity (6c) | As JetFlow's sampler on its own network it beat Euler at four calls on real footage and on the collision world (57.97 against 67.96; 1.218 against 1.392); not on the UNet anywhere; on MeanFlow's velocity only on real footage and only because of one bad seed | the plan's proposal: keep only this part of JetFlow; a sampler arm in the physics plan's step 3 |
| **Flip walk that keeps a Frobenius symmetry** (`scripts/gf2_orbit_walk.c`) | then the plain walk; then SAT | Reaches the best known 24 for GF(2^8) and 30 for GF(2^9) from the schoolbook; leaves the 42s for GF(2^12) unchanged (209 core-hours). Symmetric then plain is the best or level arm at equal time on GF(2^8) to GF(2^10), a small gain. `orbit_walk.json`, `hybrid_walks_local.json` | the searcher to start with on small fields and new targets, followed by the plain walk |
| **SAT on neighbourhoods** (`scripts/search_gf2_sat_lns.py`) | after either walk | Around the 42s, none of 290,952 neighbourhoods of 5 to 9 products can be one shorter; from the GF(2^7) schoolbook it shortens 49 to 28. After a walk, nothing in 20 seconds. `sat_lns.json` | a check that a decomposition is locally shortest, not a searcher |
| **Complete searches of symmetric decompositions** (`scripts/gf2_invariant_search.*`) | mixed subfields | None at 41 or 42 invariant under a Frobenius power over GF(2); over GF(4) 14 products exist, 13 do not among 56 thousand million invariant choices | negative certificates for the record |
| **The plain C flip walk** (`scripts/gf2_flip_walk.c`) | look-ahead; learned scorer; settings loop; then SAT | 42 stays 42 in 274 core-hours; the baseline everything else is measured against | the calibration baseline |

## C. Retired: did not meet their expectations

Retired on your word of 27 November (00:41, "retire anything that does not meet our expectation"): marked
`deprecated` in `docs/DOCUMENTATION.md`, run by nothing, kept only so their records can be replicated (frozen
files are never edited or moved). Nothing in force used any of them.

| Solution | Hybrids tried | Why not |
| :--- | :--- | :--- |
| **A language model evolving the GF(2^n) builder** (AlphaEvolve-style, `scripts/run_gf2_evolution.py`) | — | 118 calls; 75 of its 77 versions failed to run and none was repaired, so the best is the starting builder. Revisit only after finding why they failed (the run's ledger, in its artifact) and with a stronger coding model; the count of calls is said before any larger run |
| **Look-ahead and a scorer learned from the walk's own descents** (the non-neural program search's loops, `scripts/gf2_guided_walk.c`) | with merge-greedy choice; a settings loop one level up | Held out, only the last move before a new best is predictable, and it is a merge by construction; the learned weights are near zero; no gain at equal time; the settings loop settled on the default. `guided_walk_local.json`, `guided_walk_settings_local.json` |
| **AND gates instead of products** (`scripts/gf2_and_walk.c`) | — | Reaches the known counts up to GF(2^6), stops at 27 for GF(2^7) (22 known), 42 for GF(2^12); and it counts AND gates, not the bilinear products of the target. `and_walk.json` |
| **A walk over GF(4) toward 13 products** (`scripts/gf4_flip_walk.c`) | — | GF(4^5) reaches the published 11; GF(4^6) stays at 14 in 178 core-hours. `gf4_flip_walk.json` |
| **The 719 OpenAI mathematics manuscripts** | — | none applies directly; one lesson on process taken |
| **A quantum-inspired byte model** (MMLM-QI; `RESULT-MMLM-QI-2026-11`; the plan's proposal, for your word) | mixed byte by byte with Kneser-Ney counting | Preregistered run, 45 minutes per model: the best, `pure-complex-64`, 2.61 bits per byte, against 1.82 for four bytes of counting and 2.40 to 2.42 for a 1,024-state hidden Markov model; it lost to the hidden Markov model of its state's size at both sizes (by 0.27 and 0.35 on three seeds), and mixed with counting it got weight 0 in every run, so the hybrid adds nothing. Complex amplitudes beat real ones by 0.45 and 0.85 at the larger sizes (the real models trained worse as they grew); a mixed state did not help |
| **JetFlow as the way to train video** (flow matching on JetFlow's network with its own sampler; `RESULT-VIDEO-FLOW-2026-11`) | its sampler on the UNet (6d) | Final run, 20 minutes of training, three seeds: lost to plain flow matching at four calls on real footage (57.97 against 38.72) and on fluid, won only on collision; its network lost to the UNet with the same sampler everywhere; its sampler on the UNet beat Euler nowhere. Retired on your word; its network lives on in the hybrid below |
| **MeanFlow on the UNet** (`RESULT-VIDEO-FLOW-2026-11`) | the Jet2 step on its velocity (6c) | Beat flow matching at one call on both toy worlds, not on real footage (144.21 against 97.78, no seed lower); one of its three seeds scored far worse there. Retired on your word as a model on its own; the MeanFlow objective lives on in the hybrid |
| **MeanFlow on JetFlow's network** (the hybrid `meanflow-jetflow`; `RESULT-VIDEO-HYBRID-2026-11`, the larger preregistered test) | against both parents and plain flow matching, 8 frames of 32 x 32, 45 minutes, real footage split by group | All 36 jobs ran. At one call it kept a lower FD on real footage (460 against 553 for MeanFlow on the UNet and 574 for JetFlow's best sampler, on two of three seeds), and on collision; not on fluid. Its real-footage clips were 3.1 times as rough as plain flow matching's (the flicker guard failed), and at four calls it was only 2.5 % better than flow matching (H3 failed). The plan's table gave no proposal; listed as retired under your standing word of 00:41. The roughness goes with JetFlow's network as much as with the hybrid |
| **MMLM as the table route's prior** (MMLM-L in place of the size prior; `RESULT-MMLM-ORDER-2026-11`, on your question whether MMLM is reliable for APEDP-X) | the route in force with its search and last checks; a guard set where the order cannot help | All 190 units ran. On familiar ground (other-field, reachable ODEBench and power laws) its first pass evaluated a median of 5,847 candidates against 26,504 and stopped early more often, but both arms recovered the same laws, 125 of 130, unit for unit (no gain: expectation 1 failed). No wrong law in either arm; not slower end to end. Retired as the route's prior, as the plan fixed; MMLM's other experimental uses are untouched |

## D. Waiting for results

| Solution | Hybrids in the test | Rehearsal so far | Where its result goes |
| :--- | :--- | :--- | :--- |
| *(nothing waiting: both results are recorded, 27 November)* | | | |

## E. Planned, not built

| Solution | Hybrids in the plan | Where it stands | First gate |
| :--- | :--- | :--- | :--- |
| **Physics-first video** (`docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md`): a physics adapter on an open pretrained model, scored by a judge built on the referee | LoRA and tensor-train adapters at equal size, a tenth-size tensor train, a LoRA/tensor-train hybrid; a LoRA on ordinary footage as the control | plan written on your word (26 Nov); the judge is built first, on CPU | gate J (the judge on a blind set), then gate 1 (a 10-point physics gain without worse pictures); numbers proposed, for your word |

## F. Put up for retirement, kept on your word, with a design for each (27 November)

On 27 November four things that had shown no gain were put to you for retirement. Your word (10:23): do
not retire them, find a solution for each. The designs are in
`docs/SOLUTIONS_FOR_NO_GAIN_ITEMS_2026-11-27.md`. None is in force; each has a first look that can stop
it and a rule that retires it.

**Built on your word of 18:01 ("build the best of the above and make the others fallback").** The best is
the first row, learning across tables: `apedp_dyn/discovery_l2/across_tables.py` and `family_route.py`.
The other three stand behind it as fallbacks (`apedp_dyn/discovery_l2/fallbacks.py`), tried in this order
and only for a table that still has no law: solved cases, the tensor-train reading, a language model
(only with a key; no call has been made). They are the fallbacks as they stood on 27 November, not the
redesigns in the table below. Experimental, not in force: the route in force is unchanged. What of the
designs is not built is listed in section 9 of the design document.

**Its first development run showed a flaw in the build, which is repaired**
(`benchmarks/family_stream/dev_run1.json`; 16 families, 240 tables). The route table by table reported
85 laws, the family route 104, with the fallbacks 116, none inaccurate; but a table put into a family it
does not belong to reported 98. What the referee reports on a table depends on the list of laws before
it, and the build read again only the tables without a law and counted whatever was then reported. Now a
re-reading counts only if the law it reports was handed in, and handed laws take turns with the table's own.

**The second run, with the repaired route** (`benchmarks/family_stream/dev_run2_summary.json`; 16 new
families, 240 tables): the route table by table 118 laws; the family route 124; the control 120; with the
fallbacks 129; none inaccurate. So siblings' shapes add little (6 tables against 2). The reason: of the
122 tables without a law, the referee already had an accurate law among its own candidates on 91.
Candidates are not what those tables lack. The wider law: the family's own construction 12 of 48 family
readings, a direct search of the pooled table at equal search time 22, both 24, all accurate. Those 24
laws cover 120 tables, 42 of them without a law of their own (26 at 10 % noise); that last step, a table
taking its law from its family's, is not built or measured. The tensor-train fallback added 4 laws, three
the law itself, all products.

| Thing | What was measured | The design | First look | Retired if |
| :--- | :--- | :--- | :--- | :--- |
| **The learning loop** (`discovery_l2/learning_loop.py`) | no gain on invented laws or the two formula benchmarks; 8 gained and 2 lost on ODE systems (p = 0.055), where parts recur | learn across tables: a confirmed shape tried on its siblings, siblings searched together with constants of their own, the constants read as a table and the wider law refereed on all rows; a remembered part becomes an operator only if it beats a random part of the same size | six families at 5 % and 10 % noise, searched singly and together (CPU) | the joint search beats neither the single searches nor tables grouped at random |
| **A language model steering the search** (`proposers/llm_controller.py`) | 173 of 266 against 174; the two runs together 191 | a second reader: its plan added as one more member, never in place of one; a computed dossier instead of twelve rows; rounds on what is left over; a harness learned on invented laws and frozen | at most 120 calls on tables the route has missed: can it name the shape there? | its member does not beat a member planned at random at equal CPU |
| **A language model shown solved cases** (`proposers/experience.py`) | solved on the list for 13 (model), 10 (cases, no model), 7 (model shown the cases) of 60; the two lists joined 19 | the cases as a proposer of their own; a likeness of tables that is learned; a model sees a case only as evidence fitted to the table, and its list is joined | nearest cases against random cases on the 60 problems (CPU, hours) | nearest are no better than random |
| **The tensor-train equation model** (`apedp_mmlm`) | 376 bits a held-out law, 40.8 repaired, against 33.7 for counting; and the best prior bought no law | A: built from the counted automaton, so it cannot cost more bits (an hour; buys no law). B: a tensor train fitted to the real rows of a wide table and read as a formula | **done** (`benchmarks/symbolic_tensor_train/dev_probe_1.json`): at 1 % noise at the table's noise on 14 of 20 wide tables in about 8 s, one law of four inputs read exactly; not ahead of the split search, and behind it at 5 % noise (6 accurate against 13) | the stage with it reports no more accurate wide laws than the stage in force, or the split search does all it does |

## Waiting for your word, apart from the above

- Section F: decided on 27 November at 18:01 (the families built as the primary, the other three as
  fallbacks). Built, repaired and measured twice in development. Not proposed for a preregistered test
  as it stands: against its control it adds too little. What waits for your word: whether to build and
  measure the one part that looked worth it, a table taking its law from the law its family's pooled
  table was given (a change in what counts as reported for a table, so yours to allow); and whether to
  look at why the referee declines accurate laws it already has (91 of 122 tables).
- Gate J part 1 of the physics judge: the reading of its blind test, each scene kind on its own or pooled
  (`preregistration/PREREG-PHYSICS-J1-2026-11.md`); stage 2 and the run wait for it.
- Earlier, unchanged: PREREG-RANKED-MEMORY-2026-11; the search-budget and kept-processes proposals; branch
  protection on `master`; the trustees' signatures; the dashboard's remint; the two finance audits ("run it").
