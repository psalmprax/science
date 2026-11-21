# A search that a model may rewrite, under a judge it cannot touch

Status: built on 2026-11-21, tested with a scripted stand-in for the model. **No real model has
driven it and nothing it could produce is installed.** Code: `apedp_dyn/evolution/`,
`apedp_dyn/discovery_l2/evolved_stage.py`, `scripts/run_search_evolution.py`.

## What it is

The second step after `proposers/agentic_variation.py`, and the closer one to Agentic Variation
Operators (Chen et al., 2026, arXiv:2603.24517). There a model varies candidate laws for one
problem. Here it varies a piece of the search's own code, and every version is scored on many
problems by a judge that stays fixed.

The piece is one function, `propose(X, y, names)`: it is given development rows and returns up to
four candidate laws as text. The polynomial library the refined path already uses is such a
function, written by hand. The judge asks, on problems whose law is known, whether the law is
among the candidates.

## The four parts

| Part | What it does |
| :--- | :--- |
| `restricted.py` | Reads a candidate's syntax tree and refuses it unless it stays inside an allowlist: no import, no class, no name or attribute that starts with an underscore, no attribute outside a fixed list of computing ones. Nothing is compiled or run by the check |
| `jail.py`, `_jail_child.py` | Runs a checked candidate in a fresh process that is given none of the caller's environment, cannot write or open a file, cannot open a socket, cannot start a process, and is capped in memory and CPU time. Each problem is a fresh start, so nothing carries over |
| `operator.py` | The loop. The model is shown the contract, the rules of the language, the leading versions with their scores, the latest refusals, one parent's code and what the judge said about it. It replies with a complete new version. A refusal or a failure goes back to it for repair. Every version scored is kept |
| `evolved_proposer.py` | The judge for `propose`, and a proposer that runs one version in the jail for the path. A candidate's output is text; each string goes through the whitelist parser of `llm_proposer` after a check that it cannot stall that parser |

## What holds, and how each is tested

| Claim | Test in `tests/test_evolution.py` |
| :--- | :--- |
| Code outside the subset never runs | 45 refusals, from `import os` to an attribute assignment on numpy |
| The process stops what the check missed | The same process given code the check would refuse, with the whole of numpy's module to climb out of: it cannot write, read, open a socket, start a process, or see a variable of the caller |
| A candidate cannot outlast its time | An endless loop, one that catches `Exception`, and one stuck inside a single operation where no exception can reach it |
| The judge reads text and nothing else | Hostile strings (`9**9**9**9`, `__import__('os')`) are refused before the parser sees them |
| A reply is only ever parsed in the caller's process | A reply whose code would mark the caller if any of it ran there |
| The model never sees a validation law | The validation laws are searched for in every prompt of a full run |
| Nothing is installed | A run leaves a ledger, a summary and the champion's code as a text file |

## What it does not do

- **It is not a container.** The jail is a Linux process with resource limits. The kernel is the
  same. A run with a model that might be hostile, or on a machine that holds anything worth
  stealing, should put a container or a virtual machine without network around the whole run.
- **No scipy in the jail.** A candidate has numpy and `math`. Constants inside a function have to
  be found by a grid or a hand-written iteration.
- **The judge's score is not a measurement.** It says whether a law is proposed, on laws drawn
  from one grammar, and the champion is chosen by it. A version can get better at that grammar
  without getting better at laws.
- **Each candidate takes a search's place.** The path confirms at most ten candidates of a stage.
  A generator that proposes badly can lose a law the searches had found.

## The rule for adoption

1. The loop writes versions to a ledger and stops. It installs nothing.
2. A champion is run beside the default searches on the twenty development laws
   (`run_search_evolution.py score`), against the seed and against the default stage alone.
3. If that is worth testing, a preregistration is written and frozen, and the champion is tested
   once on laws no version was scored on: the forty held-back invented laws, and laws from another
   source, because the invented ones share the training laws' grammar.
4. Only then, by a reviewed commit, does a version join the route. It is still run in the jail.

## Running it

Without a model (this is all that has been run):

    python scripts/run_search_evolution.py baseline --out out/baseline
    python scripts/run_search_evolution.py demo --out out/demo

With a model that serves chat completions in OpenAI's format:

    export CODING_MODEL=...            # the model's name
    export CODING_MODEL_API_KEY=...    # read from the environment, written nowhere
    export CODING_MODEL_BASE_URL=...   # unless it is NVIDIA's service
    python scripts/run_search_evolution.py evolve --out out/run --steps 30

On GitHub: add the repository secret `CODING_MODEL_API_KEY` and the repository variables
`CODING_MODEL` and `CODING_MODEL_BASE_URL`, then push a branch named `search-evolution-run-1`.
Only the job that talks to the model is given the key.

## What the scripted stand-in showed (2026-11-21, not a measurement)

Sixteen training and sixteen validation laws. The seed proposes none of them. The stand-in's
first reply is refused by the check (an import), its repair fails when run, its second repair
runs and proposes 1 of 16 training laws and 4 of 16 validation laws. Its next version never
returns and is stopped on every problem; the one after returns strings the judge refuses to read.
This shows the machinery, and that a hand-written grid over constants does not solve the task.
