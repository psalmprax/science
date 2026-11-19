# PREREG-LLM-PROPOSER-2026-11: a language model as the search stage, given numbers only

**Status:** DRAFT until the model is fixed below and `preregistration/DIGEST-LLM-PROPOSER.json` is
written; the run refuses to start before that. Runs once, on GitHub Actions
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

**Model:** MODEL_TO_BE_FIXED (NVIDIA hosted API, temperature 0). Chosen as recorded under
"Disclosed before freezing".

## Data and protocol

SRBench's 133 ground-truth problems at noise 0 and 1 %, with the same splits, the same 70 %
development rows and the same 30 % pool as PREREG-MAIN-SEARCH-2026-11. 266 runs. Three rounds of up
to five proposals, at most eight candidates returned. A reported law is scored on the test rows as
in that preregistration: accurate if test R² is at least 0.999, symbolically recovered by the
baseline's check.

## Expectations, fixed now

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

- **Model choice.** MODEL_CHOICE_TO_BE_RECORDED
- Two smoke runs before this one reached the API. The first found none of my preferred model names
  in the catalogue. The second took the first listed model that answered and got no usable proposal
  on three toy laws, with nothing recorded about why; the reply parser was then made tolerant of
  code fences, labels, subscripts and prose, and the proposer now keeps what its last call saw.
  Toy laws only; no benchmark problem was run through a model before this freeze.
- The free tier is rate limited. Requests are spaced and retried; a run whose requests all fail is
  recorded as an error, not retried by hand.
- The model's answers are not reproducible in the way the other engines' are: the service may
  change the model behind a name. The model name and every proposal it made are in the record.

## Deviation rule

The run is not repeated. If a shard fails for a reason of infrastructure (a runner lost, the
service unavailable for the whole shard), that shard alone is re-run and the fact is recorded with
its date. Any change to a frozen file after shards have started makes the result a development
result.
