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

## A. To add, on your word

| Solution | Hybrids tried | Evidence | Before it goes in |
| :--- | :--- | :--- | :--- |
| **Counting independent trials by clusters** for the deflated Sharpe ratio (after US20190294990A1 and López de Prado and Lewis) | four: cluster series with an eigenvalue count across clusters; the trial's own Sharpe ratio with the cluster count; both counts; either count | Simulated searches: false alarms at most 1 in 300, as the referee's count; power far higher where strategies come in families (e.g. 85 % against 19 %). Real prices (six series, 56 strategies): false alarms 0.17 % as the referee's count; planted edges found 52 to 62 % of the time against 0 %. No hybrid beat it clearly; "own Sharpe ratio with the cluster count" is unsafe (11 to 35 % false alarms on real prices). `benchmarks/referee_trials/independent_trials_dev.json`, `independent_trials_real_dev.json`, `independent_trials_hybrid_dev.json` | a preregistered blind test, an independent review, and your words "make it the default"; it changes the referee, so the trustees' signing rules apply to any change of the specification |

## B. Keep as a tool or an option

| Solution | Hybrids tried | Evidence | Use |
| :--- | :--- | :--- | :--- |
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

## D. Waiting for results

| Solution | Hybrids in the test | Rehearsal so far | Where its result goes |
| :--- | :--- | :--- | :--- |
| **JetFlow against plain flow matching and MeanFlow** (`PREREG-VIDEO-FLOW-2026-11`) | MeanFlow on JetFlow's network; JetFlow's sampler on the UNet; the Jet2 step on MeanFlow's velocity | JetFlow's sampler lost to plain flow matching at four calls on all three data sets (2 minutes of training). Its Jet2 step was found to have its curvature reversed since 6 November; fixed on your word, so the run scores the fixed step (no recorded result had used the reversed one) | the plan's consequences: JetFlow kept only if it (or its corrected step) beats flow matching on real footage and a toy world; a hybrid only if it beats what it would replace |

## E. Planned, not built

| Solution | Hybrids in the plan | Where it stands | First gate |
| :--- | :--- | :--- | :--- |
| **Physics-first video** (`docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md`): a physics adapter on an open pretrained model, scored by a judge built on the referee | LoRA and tensor-train adapters at equal size, a tenth-size tensor train, a LoRA/tensor-train hybrid; a LoRA on ordinary footage as the control | plan written on your word (26 Nov); the judge is built first, on CPU | gate J (the judge on a blind set), then gate 1 (a 10-point physics gain without worse pictures); numbers proposed, for your word |

## Waiting for your word, apart from the above

- Earlier, unchanged: PREREG-RANKED-MEMORY-2026-11; the search-budget and kept-processes proposals; branch
  protection on `master`; the trustees' signatures; the dashboard's remint; the two finance audits ("run it").
