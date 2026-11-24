# Result: PREREG-MAIN-SEARCH-2026-11

**Run:** once, 2026-11-19 18:39–21:44 UTC, GitHub Actions run `44590126072` on the branch
`srbench-main-search-run-2` at commit `d939f98`, which carries the re-frozen preregistration and
harness (digests repeated in `results.json`). The kernel build, the smoke job, 32 shards and the
collect job all succeeded on the first attempt; no shard was re-run. (The branch
`srbench-main-search-run-1` stopped at its smoke job before any shard started; the preregistration
records that and the re-freeze.)

Outputs, unedited, in [`RESULT-MAIN-SEARCH-2026-11/`](RESULT-MAIN-SEARCH-2026-11/): `results.json`,
`runs.jsonl` (532 runs) and the 32 shard files. The collect job printed them to its log with their
checksums; they were rebuilt from the log and all 34 match the SHA-256 the job computed.

## Outcome

133 SRBench problems × 2 noise levels (0 and 1 %) × 2 engines.

### The main search path

| | Runs | Law reported | Reported and accurate | Reported and not accurate | Reported and the law | Refused (`widen_search`) | Timeouts |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All, both noise levels | 266 | 186 | 182 | **4 (2.2 %)** | 152 (81.7 %) | 78 | 2 |
| Noise 0 | 133 | 81 | 79 | 2 | 76 | 50 | 2 |
| Noise 1 % | 133 | 105 | 103 | 2 | 76 | 28 | 0 |
| Feynman, noise 0 | 119 | 73 | 72 | 1 | 69 | 44 | 2 |
| Feynman, noise 1 % | 119 | 95 | 94 | 1 | 70 | 24 | 0 |
| Strogatz, both | 28 | 18 | 16 | 2 | 13 | 10 | 0 |

95 % intervals (Clopper-Pearson): not accurate among reported 0.6–5.4 %; the law among reported
75.4–87.0 %.

Symbolic recovery on the Feynman problems (of 119), beside the earlier single-engine results:

| Engine | Noise 0 | Noise 1 % |
| :--- | ---: | ---: |
| PySR alone (PREREG-SRBENCH-EXTERNAL, all training rows) | 73 | 68 |
| **Main search path** (70 % of the rows; the rest is the pool) | **69** | **70** |
| GA/GP, native kernel (this run) | 53 | 7 |
| L2 (baseline run) | 24 | 24 |

### GA/GP

| | Runs | Recovered | Accuracy solutions | Errors | Timeouts | Median seconds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feynman, noise 0 | 119 | 53 | 77 | 3 | 1 | 216 |
| Feynman, noise 1 % | 119 | 7 | 45 | 4 | 0 | 233 |
| Strogatz, noise 0 | 14 | 3 | 5 | 1 | 0 | 92 |
| Strogatz, noise 1 % | 14 | 0 | 4 | 0 | 0 | 101 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the runs in which the path reports a law, at most 5 % are not accurate | 4 of 186, 2.2 % | **held** |
| 2 | At least 60 % of the reported laws are symbolically recovered | 152 of 186, 81.7 % | **held** |
| 3 | Feynman: at least 62 recovered at noise 0 and at least 58 at 1 % | 69 and 70 | **held** |
| 4 | GA/GP recovers fewer Feynman laws than PySR alone (fewer than 73 and 68) | 53 and 7 | **held** |

On expectation 1 the count is inside the criterion and the interval is not quite: with 186 reports
the data are compatible with a true rate of up to 5.4 %.

For comparison, the referee's sole-survivor rule on the same benchmark (audit M-10): 168 of 268
promotions were not the law (62.7 %), with a median test R² of 0.81 among them.

## Reported without a criterion

- **Refusals.** 78 of 264 completed runs ended in `widen_search`: 50 at noise 0 and 28 at 1 %. In 14
  a law had survived confirmation and was turned down by the adequacy check; in the other 64 no
  candidate survived. The search stage had run in 76 of the 78.
- **Which stage found the law.** Of the 186 reported laws, 61 came from the first pass and 125 from
  PySR. The first pass settled 60 runs without PySR being started (58 of them reported a law).
- **By action.** `confirmed` 39 (38 the law, 39 accurate); `equivalent_class` 123 (92 the law, 121
  accurate); `unchallenged` 24 (22 the law, 22 accurate). On noisy data nearly every report is an
  equivalence class: several formulas cannot be told apart on the rows available and the simplest is
  reported, which is why 27 accurate reports at 1 % are not the law's exact form.
- **Across noise levels.** 67 problems are recovered at both levels, 9 only without noise, 9 only
  with it.
- **Time.** Median 274 s per run, of which PySR's budget is 240 s; two runs reached the 480 s limit
  (`feynman_II_11_20` and `feynman_II_11_3`, both at noise 0).
- **GA/GP.** 131 accuracy solutions in 266 runs against 63 recoveries: it fits well and returns the
  law's form much less often, and at 1 % noise almost never (7 of 133). Eight runs ended in an error
  while its pick was being scored (complex or relational values in the returned expression).

## After the run (not preregistered)

Everything in this section was looked at after the results were known.

**The four reports that were not accurate.**

| Problem | Noise | Action | Found by | Test R² | What happened |
| :--- | ---: | :--- | :--- | ---: | :--- |
| `strogatz_lv1` | 0 | unchallenged | first pass | 0.352 | The first pass stopped on `sin(exp(x0*x1))`, nothing challenged it, and the adequacy check passed it (residual structure −0.01, error 0.83 of the smoother's). |
| `strogatz_lv1` | 1 % | unchallenged | first pass | 0.353 | The same. |
| `feynman_II_34_29b` | 0 | equivalent class | PySR | 0.800 | The reported text is `x0*(x2 + x3*x4/x1)**2`. What was confirmed exact (relative error 7e-16 on the pool) is its refit with a free coefficient per additive term, which keeps the cross term alone. The text is wrong; the confirmed fit is not. |
| `feynman_test_9` | 1 % | equivalent class | PySR | 0.994 | The same kind: a 46-term expansion whose refit is accurate (R² 0.99998) and whose text is not. |

Two different defects:

1. **The path reports the search's text, not the fit it confirmed.** Confirmation and the adequacy
   check are applied to the candidate refitted with one coefficient per additive term
   (`contrast_resolution.fitted_candidate`); `MainSearchResult.law` is the text before that refit.
   Rescoring all 186 reported laws in the confirmed form gives 184 accurate (not 182) and 150
   recovered (not 152: three two-term laws at 1 % noise lose the symbolic check and one ten-term
   law gains it). In that form the only inaccurate reports are the two on `strogatz_lv1`. The
   frozen path is unchanged; `apedp_dyn/discovery_l2/route.py` now returns `law_as_confirmed`
   beside `law`, and the ODE and series routes report the confirmed form.
2. **The adequacy check passed a poor law on `strogatz_lv1`.** The preregistration disclosed that
   this check had no result of its own. It refused 14 laws in this run and let this one through at
   both noise levels: on a set of trajectory rows a nearest-neighbour smoother is itself poor (the
   law's error was 83 % of the smoother's), so "at least as good as the smoother" asks little.
   Both failures are `unchallenged` reports from the first pass. Of the 24 `unchallenged` reports
   2 were wrong (8 %); of the 162 `confirmed` or `equivalent_class` reports, 2 were not accurate as
   text and none in the confirmed form. This is one problem at two noise levels, so it supports a
   suspicion, not a rate.

**What the path costs against PySR alone.** On the Feynman problems it recovers 69 and 70 where
PySR alone recovered 73 and 68, with 30 % fewer rows for the search. The difference is within what
two runs of a stochastic search would show, and this run was not designed to resolve it.

**Where the remaining laws are lost.** At noise 0 the path refuses 50 of 133 problems. It reports
nothing wrong there apart from the cases above, and it reports nothing at all on more than a third
of the benchmark; these are mostly laws PySR does not reach in 240 s.

## Consequences, as preregistered

- **Expectations 1 to 3 held, so the default route changes.** `DEFAULT_ROUTE` in
  `apedp_dyn/discovery_l2/route.py` is now `main_search` (it was `l2`).
- **Expectation 4 held, so GA/GP is retired as a law-discovery engine.** The module
  (`apedp/evolutionary_search.py`) is in this preregistration's digest and is not edited; the
  retirement is recorded in `docs/DOCUMENTATION.md`, the audit (M-11) and the changelog. It stays
  importable for replication.
- **M-10** (the referee's sole-survivor rule) is addressed for the route that is now the default.
  The rule itself is unchanged in the referee and is still what the template engine uses.

## Note of 2026-11-21 (not part of the recorded result)

Every recovery in this run was judged again on a range twice as wide in each variable
(`scripts/check_recoveries_out_of_range.py`, audit M-23). All 152 recoveries of the main search
hold there. One of the comparison engine's 63 does not: for `feynman_I_25_13` at 1 % noise it
returned a form that equals the law only where the benchmark samples. The counts above are left as
recorded.
