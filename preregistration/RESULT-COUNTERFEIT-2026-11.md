# Result: PREREG-COUNTERFEIT-2026-11

**Run:** once, 2026-11-22 21:00–21:13 UTC, GitHub Actions run `45069580150` on the branch
`counterfeit-run-1` at commit `2fc02ee`, which carries the frozen preregistration and harness
(digests repeated in `results.json`, and equal to `DIGEST-COUNTERFEIT.json`). The smoke job, 12
shards (21:06 to 21:12) and the collect job all succeeded on the first attempt; no shard was
re-run.

Outputs, unedited, in [`RESULT-COUNTERFEIT-2026-11/`](RESULT-COUNTERFEIT-2026-11/): `results.json`
and `runs.jsonl.gz` (360 rows: 60 laws, three noise levels, the law offered or withheld; four
methods in each row). The collect job printed them to its log with their checksums; they were
rebuilt from the log and both match the SHA-256 the job computed (`SHA256SUMS`).

## Outcome

**All five expectations held.** Handed counterfeits of a law, the route in force reported no
wrong law at noise up to 1 % and sixteen at 5 %. The checks built from the red team's findings
stopped fourteen of the sixteen and kept 139 of the route's 146 right laws. Reporting the best
fit was wrong in all 180 cases where no candidate was right, and confirming it on fresh rows
stopped seven of those.

No case ended in an error, none timed out, and the checks failed on none.

| Method | The law | Wrong | Accurate, not the law | Nothing | Score |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `route` (the route in force) | 146 | 16 | 67 | 131 | −0.88 |
| `vouched` (the same after the checks) | **146** | **2** | 43 | 169 | **0.60** |
| `best_fit` | 172 | 180 | 8 | 0 | −18.04 |
| `validated` (best fit confirmed on fresh rows) | 171 | 173 | 8 | 8 | −17.31 |

By noise level, of the laws each reported:

| Noise | `route`: the law, wrong, of reported | Score | `vouched`: the law, wrong, of reported | Score |
| :--- | ---: | ---: | ---: | ---: |
| none | 61, 0, of 62 | 1.02 | 61, 0, of 62 | 1.02 |
| 1 % | 50, 0, of 79 | 0.83 | 54, 0, of 74 | 0.90 |
| 5 % | 35, **16**, of 88 | −4.48 | 31, **2**, of 55 | −0.12 |

A score above 1 comes from a law found where it had been withheld: the route's own first pass
found the law in three such cases, one at each noise level.

95 % intervals (Clopper-Pearson) for wrong among reported: the route at noise up to 1 %, 0 of
141, 0 to 2.6 %; the route at 5 %, 16 of 88, 11 to 28 %; `vouched` at 5 %, 2 of 55, 0.4 to
12.5 %. **The third expectation held on the count. With 55 reports a rate above the limit is
not excluded at 5 % noise**, and at that level the checked route's score is still below zero:
two wrong laws cost 38 and it named the law 31 times.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws the route reports at noise up to 1 %, at most 5 % are wrong | 0 of 141 | **held** |
| 2 | The route scores above `best_fit` and above `validated` | −0.88 against −18.04 and −17.31 | **held** |
| 3 | Of the laws `vouched` reports, at most 5 % are wrong at each noise level | 0 of 62; 0 of 74; 2 of 55 (3.6 %) | **held** |
| 4 | `vouched` scores above `route` | 0.60 against −0.88 | **held** |
| 5 | Of the cases in which the route reports the law, `vouched` still reports it in at least 75 % | 139 of 146 (95 %; at 5 % noise 28 of 35) | **held** |

## What follows, as fixed before the run

Expectations 3, 4 and 5 held, so **the checks become the last step of the route in force for
tables** (`apedp_dyn/discovery_l2/vouched_route.py`, a new module around the frozen ones;
`APEDP_TABLE_VOUCHING=off` turns them off). A law that fails them is shown as not vouched for,
with the reason, and is not reported.

Stated wherever that route is described, as fixed before the run: measured on candidates handed
over, uniform rows and Gaussian noise up to 5 %; not measured behind a search engine or on the
physics benchmark, where what the checks cost in recovered laws is unknown until a run says.

Expectations 1 and 2 are recorded as they fell: both held.

These 60 laws have now been run on and are development material from here on.

## The forecast, scored

`benchmarks/counterfeit_forecast_scored.json` (`scripts/score_counterfeit_forecast.py`).

- **Nine of twelve totals fell inside their 80 % intervals. The three outside are all the
  route's**, which did better than forecast: 16 wrong laws where 31 were expected (24 to 37),
  146 right where 156 were (149 to 163), a score of −0.88 where −2.36 was (−3.05 to −1.66).
  For `vouched` it said 151 right, 1.3 wrong and 0.70; there were 146, 2 and 0.60, all inside.
- Where the miss came from: at 5 % noise the route reported 3 wrong laws in the 60 cases with
  the law offered, where the campaign's worlds gave 10, and 13 in the 60 without it, where they
  gave 20. The campaign's worlds drew counterfeits at every distance, including those only just
  beyond the accuracy line; this benchmark leaves those out, which was disclosed and which the
  forecast said its intervals do not cover.
- It gave the third expectation 81 % and the others more than 99.9 %. All held: Brier score
  0.007, where even odds score 0.25. Five forecasts that all came true say little about
  calibration.

**The author's own expectations.** Right on which expectations would hold. **Wrong on the
route's wrong laws: about 38 were expected, more than the forecast's 31, and there were 16.**
The reasoning was that withheld cases hold no accurate candidate, so reports that were harmless
in the campaign would be wrong here. The route's reports in those cases were mostly its own
stages' (the first pass and the polynomial library), and 36 of 52 were accurate. `vouched` was expected to report about 45 laws at 5 %
noise and reported 55; three in four right laws were expected to be kept there and 28 of 35
were.

## Reported without a criterion

- **Where the route's sixteen wrong laws came from.** All at 5 % noise: 13 with the law
  withheld, 3 with it offered. **Twelve were forms of the route's own exhaustive first pass, one
  a polynomial of its own library, and three were handed counterfeits.** Fourteen came out of a
  class the path could not separate (`equivalent_class`), where it names the shortest member; in
  two of the three offered cases the law was in that class and a shorter form was named, and in
  the third the law was no longer in it. So at this noise the benchmark measured the checks
  mostly against the route's own approximations, not against the counterfeits it was handed.
- **What the checks did with the route's 229 reports.** 191 passed. Of the 146 right laws 139
  passed and 7 were withheld, all at 5 % noise; 7 further reports were passed on to another
  member of their class that is the law (4 at 1 % noise, 3 at 5 %), so `vouched` also names the
  law 146 times. Of the 67 accurate approximations 43 passed, 7 became the law and 17 were
  withheld. Of the 16 wrong laws 14 were withheld and 2 passed.
- **Which rule withheld what** (38 reports):

  | Rule | Wrong | Accurate, not the law | The law |
  | :--- | ---: | ---: | ---: |
  | Structure left in the residuals | 7 | 12 | 0 |
  | Members of the class pass and disagree | 4 | 1 | 3 |
  | An error that size would have been missed | 3 | 4 | 4 |

  The structure test refused none of the 146 right laws. The seven right laws that were lost
  were lost to the other two rules: the data could not tell the law from a rival that differs
  from it, or could not have shown an error of the size in question.
- **Reports passed to another member of the class:** 11, seven into the law and four from one
  accurate approximation to another.
- **The two wrong laws the checks passed** are both the route's own, both at 5 % noise with
  the law withheld, and both wrong by a small margin: a thirteen-term cubic of the polynomial
  library (law 6029; test R² 0.9982, the line being 0.999) and a four-term form of the first
  pass (law 6045; R² 0.9989). For both the power rule had put the chance of seeing an error of
  the size that makes a law wrong above its minimum (97 % and 91 %, the minimum being 80 %).
- **The plain methods.** `best_fit` reported a wrong law in all 180 cases with the law
  withheld; confirming on thirty fresh rows stopped 7 of the 180. With the law offered they
  named it in 172 and 171 of 180 and an accurate counterfeit in 8, never a wrong one (the
  confirmed method was silent once).
- **With the law withheld the route reported nothing in 128 of 180 cases** (58, 41 and 29 by
  noise level), an accurate approximation of its own in 36, the law in 3 and a wrong law in 13.
- **Time.** A median of 17 seconds a case for the route, 6,264 seconds of computing in all; the
  shards took between two and a half and five and a half minutes.

## Looked at after the run (development, labelled as such)

Why the two wrong laws passed, from refitting each form a hundred times to fresh noisy rows of
its law (same inputs, same noise level; not part of the preregistered run):

- **Law 6029, the cubic.** The form is wrong at best: fitted to the law without noise it reaches
  R² 0.9983. On fresh rows the structure test refuses it 95 times in 100. The run's draw was one
  of the other five (p 0.047, the level being 0.01). The rule allows a law through when such an
  error would be seen four times in five; here it would be seen nineteen times in twenty and was
  not. This is the miss rate the rule permits.
- **Law 6045, the first-pass form.** Fitted without noise it reaches R² 0.99898, a hair under
  the line. On fresh rows the structure test refuses it 63 times in 100, **where the power rule
  had put that chance at 91 %**: the shapes of error it plants (differences from rivals,
  low-order polynomials) are easier to see than this form's real error. For an error of another
  shape than those planted, the rule's estimate is too high (audit M-33).

## What this result does not say

As disclosed before the run, and unchanged: the counterfeits are of the red team's making and
the laws of its grammar; no search engine ran; nothing has searched for data that defeats the
checks; no candidate was only just wrong (the far band starts at 0.05); a law is judged where
the data were; one draw of rows per case.

What the run adds to that list: at 5 % noise the checks pass a wrong law about one time in
eight that the route offers one (2 of 16), the interval for their wrong share there reaches
12.5 %, and what they cost behind a real search, on the physics benchmark's laws, has not been
run.
