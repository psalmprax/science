# PREREG-LLM-CONTROLLER-2026-11: a language model that configures the search

**Status:** frozen before the run by `preregistration/DIGEST-LLM-CONTROLLER.json`. Runs once, on
GitHub Actions (`.github/workflows/srbench-controlled.yml`), started by the first branch named
`srbench-controlled-run-*` whose shards start.

## What is tested

PREREG-LLM-PROPOSER-2026-11 put a language model in the search's place and it recovered about half
of what PySR recovers there. The other design, used by the published systems that do well, has the
model shape the search and a conventional search do the work. `apedp_dyn/proposers/llm_controller.py`
is that design: once per problem the model chooses which unary operators PySR may use, up to four
derived inputs to add, and up to three guesses with free constants; every item is checked before it
is used.

The model sees numbers only: anonymous variables (after the units reduction, the dimensionless
products), their ranges, twelve rows and the first pass's best expression. No names, no units, no
description.

**Model:** `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA's hosted API, temperature 0, reasoning on
with a budget of 2 000 tokens, one request per run.

## Protocol

Exactly PREREG-REFINED-SEARCH-2026-11 (same problems, noise levels, rows, pool, PySR budget of
240 s, scoring, and the same split into 30 development and 103 held-out problems), with one
difference: inside the units wrapper the search stage is the controlled search instead of plain
PySR. 266 runs. The comparison is with that preregistration's result, run for run.

## Expectations, fixed now

1. **What it reports can be relied on:** of the laws reported in all 266 runs, at most 5 % are not
   accurate.
2. **With numbers only, the controller makes no material difference:** on the 206 held-out runs the
   number recovered is within 6 of the refined path's number on the same runs, in either direction.
3. **The model's reply is usable:** in at least 90 % of the runs in which the model answered, the
   reply could be read as a configuration.
4. Reported without a criterion: runs in which the model could not be reached, how often it
   narrowed the operators, how many derived inputs and guesses it gave and how many were rejected,
   problems recovered with the controller and not without and the reverse, and time.

Expectation 2 is a prediction of no effect. If the controller recovers more than 6 additional runs,
that is a miss of the expectation and a finding in its favour; both will be said.

## Disclosed before freezing

- **PREREG-REFINED-SEARCH-2026-11's result is not known.** Its run was in progress when this was
  frozen. The comparison number is whatever that run records.
- One smoke run with the real model and PySR (30 s) on three toy laws: every reply was readable; on
  two laws the model narrowed the operators or added products and ratios and the law was found; on
  the third it added an input copied from the first pass's wrong guess and the path returned an
  approximation as an equivalence class. Three laws; no benchmark problem was run through the
  controller.
- The prompt and the limits (four derived inputs, three guesses) were written once and not tuned.
- **The key.** The run needs the repository secret to be valid for its whole length. A run in
  which the model cannot be reached falls back to the default search by design and is recorded as
  such. If that happens in more than a fifth of the runs, expectation 2 is declared uninformative
  for this run instead of held.
- The service may change the model behind a name; the model name and every configuration it
  returned are in the record.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.
