# PREREG-LLM-PROPOSER-2026-11: a language model as the search stage, given numbers only

**Status:** frozen before the run by `preregistration/DIGEST-LLM-PROPOSER.json`. Runs once, on GitHub Actions
(`.github/workflows/llm-proposer.yml`), started by the first branch named `llm-proposer-run-*` whose
shards start.

## What is tested

`apedp_dyn/proposers/llm_proposer.py` asks a language model for expression skeletons, fits their
constants, and returns the best as candidates. It had never been run with a real model. Here the
model stands where PySR stands in the main search path (`apedp_dyn/discovery_l2/main_search.py`):
it is asked only when the exhaustive first pass has not settled the problem, and what it proposes
goes through contrast confirmation and the adequacy check like any other candidate.

The model is given numbers only. Variables are called `x0, x1, ...`, no description of the problem
is passed, and it sees twelve rows of data and its own earlier attempts with their errors. It cannot
answer by recalling a named law from its name; it can still guess common forms.

**Model:** `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA's hosted API, temperature 0, reasoning on
with a budget of 2 000 tokens (`chat_template_kwargs.enable_thinking`, `reasoning_budget`), at most
4 096 tokens per reply. Chosen as recorded under "Disclosed before freezing".

## Data and protocol

SRBench's 133 ground-truth problems at noise 0 and 1 %, with the same splits, the same 70 %
development rows and the same 30 % pool as PREREG-MAIN-SEARCH-2026-11. 266 runs. Three rounds of up
to five proposals, at most eight candidates returned. A reported law is scored on the test rows as
in that preregistration: accurate if test R² is at least 0.999, symbolically recovered by the
baseline's check.

## Expectations, fixed now

A reported law is scored in the form the path confirmed (one refitted coefficient per additive
term; audit M-13, found after PREREG-MAIN-SEARCH's run). The counts for the search's own text are
reported beside them, because that is how PREREG-MAIN-SEARCH was scored.

1. **What the path reports predicts:** of the runs in which a law is reported, at most 5 % are not
   accurate. The same criterion as PREREG-MAIN-SEARCH-2026-11, with a different search stage.
2. **The model contributes laws:** in at least 10 of the 266 runs the reported law was found by the
   model (not by the first pass) and is symbolically recovered.
3. **It is not a substitute for PySR:** on the Feynman problems it recovers fewer laws than PySR
   alone did (fewer than 73 at noise 0 and fewer than 68 at 1 %).
4. Reported without a criterion: how often the model's replies could not be used, how often a
   proposal matched what the model said before seeing data, the number of requests, and a comparison
   with the PySR path's result.

A miss on 1 to 3 is reported as a miss. Expectation 3 is a prediction, not a hope: if the model
does as well as PySR that is a finding, and it is reported as a missed expectation.

## Disclosed before freezing

- **PREREG-MAIN-SEARCH-2026-11's result is known** (recorded 2026-11-19): with PySR as the search
  stage the path reported 186 laws in 266 runs, 4 not accurate, and recovered 69 and 70 of the 119
  Feynman laws. The three expectations above were written before that result and are unchanged.
- **Five smoke runs reached the API before this freeze, all on toy laws.** (1) None of the model
  names I had expected is in the catalogue (32 models listed). (2) The first listed model that
  answered gave no usable proposal on three toy laws, with nothing recorded about why. (3) Of nine
  chat models, five returned empty replies and four timed out. (4) With the reason recorded: the
  three models that answer in time are reasoning models, and each used its whole 4 096-token budget
  working through the rows by hand and never wrote an answer. (5) With reasoning switched off, or on
  with a budget of 2 000 tokens, by the parameter NVIDIA documents for these models, the two
  Nemotron models answer in both settings; `minimaxai/minimax-m3.1-flash` answers with reasoning
  off and rejects the budget parameter. Between the runs the reply parser was made tolerant of code
  fences, labels, subscripts and prose, and the client learned to read answers from reasoning
  models and to record why a reply was empty.
- **Model choice.** The rule was written before the fifth run was read: the model answers without
  failing; the text is an answer and not its working; it yields proposals; its proposals contain
  the toy law's form, or failing that the proposal with the lowest error decides; ties go to the
  faster. The toy law was `x0*exp(-0.37*x1)` with 1 % noise. No model proposed its form. Best
  relative error of a proposal, and seconds per request: `nemotron-3-ultra` with reasoning 0.013
  (a rational form, at the noise level) and 23.5 s; `nemotron-3.5-lightning` with reasoning 0.062
  and 15.3 s; `nemotron-3-ultra` without reasoning 0.083 and 3.5 s; `nemotron-3.5-lightning`
  without 0.164 and 2.9 s; `minimax-m3.1-flash` without 0.39 and 36 s. By the rule:
  `nemotron-3-ultra` with reasoning. One toy law decided this; it is a way of choosing, not a
  comparison of the models.
- The model is asked only when the first pass has not settled the problem, as PySR is in
  PREREG-MAIN-SEARCH (204 of 266 runs there). At about 24 s a request and four requests a problem,
  the run is spread over six shards.
- No benchmark problem was run through a model before this freeze.
- The free tier is rate limited. Requests are spaced and retried; a run whose requests all fail is
  recorded as an error, not retried by hand.
- The model's answers are not reproducible in the way the other engines' are: the service may
  change the model behind a name. The model name and every proposal it made are in the record.

## Deviation rule

The run is not repeated. If a shard fails for a reason of infrastructure (a runner lost, the
service unavailable for the whole shard), that shard alone is re-run and the fact is recorded with
its date. Any change to a frozen file after shards have started makes the result a development
result.
