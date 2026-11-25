# Towards a research intelligence: what is on the list, and what waits for its gate

On 24 November 2026 the owner asked whether two outside papers were correct and, if so, to add
them to the list: an assessment ("not on a demonstrated path to superintelligence", five
milestones) and a seven-stage programme with a proposed package, five first deliverables and a
first experiment on GF(2^12) multiplication.

**The assessment is correct. The programme is correct in its order and its gates, and proposes to
build more than its own gates allow.** This page records which parts are on the list, each with
the measurement that decides it, and which parts wait.

The rule taken from the papers, and already this repository's rule: a stage is started when the
gate before it has been passed in a preregistered test, not when its code can be written.

## What the papers say that this system's own measurements confirm

| Claim | What has been measured here |
|---|---|
| The referee is not an intelligence engine | It checks; what proposes are the searches |
| No demonstrated advantage from a language model | Writing the laws: 78 recovered against 152 with the search. Steering the search: 173 against 174. Choosing the moves in MMLM-POC-001: no move kept in 180. Ranking as the MMLM's expert: the true law at a median place of 2 against 49, in development, not yet as laws recovered |
| No transfer between domains | Not tested |
| No self-improvement | The recursion: 1, 3, 0, 3, 1 solved of 24 over five generations. The learning loop: no gain beside the searches on any family |
| The system must not redefine the rules it is judged by | The recursion's judge seals the referee, the thresholds, the last checks and, since 24 November, the code that decides adoption |
| A verdict needs evidence the proposer cannot make up | Open: Lean evidence is a record a caller can construct (outside audit, 24 November) |

## On the list now

Each is an experiment with a pass mark, or a small piece one of them needs.

1. **The four-arm experiment on GF(2^12) multiplication** (the programme's first experiment, and
   its stage 1 gate). Random search, the hand-written heuristic, a real language model with the
   same tools, and the language model beside the search; the same starting products, the same
   budget, the checker's verdict. Three arms exist in development
   (`scripts/run_mmlm_poc_001_llm.py`). Missing: the hybrid arm; a preregistration.
   **25 November: the reference is not what it was taken to be.** The "published 51" is the
   Toffoli count of one circuit, and the literature's bound on the number of products is 42, which
   is now in the repository and checked exactly
   (`docs/GF2_12_MULTIPLICATION_RECORD_2026-11-25.md`). Beating 51 is therefore not a discovery.
   **Decided by the owner on 25 November** (`docs/OWNER_DECISIONS_2026-11-25_AFTERNOON.md`, 4):
   the target is 41 products or fewer, starting from the verified 42; the first gate (reproduce
   the reference exactly) is cleared with the 42. First a baseline with no language model: what
   the existing search reaches on GitHub's machines, which any model then has to beat.
   **Pass mark, proposed (fixed when the plan is frozen):** the language-model arm ends with fewer
   wrong entries than the heuristic on more starts than the other way round, by a sign test fixed
   beforehand, at equal cost.
2. **The same gate for laws.** The language model as the MMLM's expert on fresh, harder laws,
   counted as laws recovered through the referee, against the search without it at equal time.
   Offered on 24 November; preregistered before it is run.
3. **A prediction before each move.** In both experiments the model says what it expects its move
   to do (for example "two fewer wrong entries"), and what would show it wrong. Scored: how often
   the stated effect happened. This is the papers' "hypothesis with a disproof", in the one form
   that can be counted.
4. **One cost record for every arm:** wall time, CPU including child processes, candidate
   evaluations, checker calls, and the service's own count of tokens. Already recorded piecemeal
   (`levels._cpu` counts children; the language-model runs record the service's usage). To do:
   one record shape used by every comparison, so that "equal budget" can be read off.
5. **A fast hosted model in the generator loop** (`scripts/run_fast_generator_loop.py`): which
   model answers soonest with code that runs, where the loop's time goes, and the champion
   against the seed on laws drawn afterwards. Development; first run capped at 300 calls.
6. **Signed evidence receipts** (the papers' evidence service, and the audit's Lean finding).
   A proposal only: the checker, not the proposer, writes a receipt binding the candidate, the
   protocol, the checker's version, the input and the output, and signs it with a key the
   research process cannot read. It changes the referee's first stage, so it waits for the
   owner's word (AGENTS.md 1.5).

## Listed, and waiting for a gate

| Stage of the programme | Opens when | Why not now |
|---|---|---|
| 2. Memory of experience (episodic, semantic, procedural, failure) | Item 1 or 2 passes | A memory and a learning loop exist and were measured: the memory gained 6 laws and lost none in its blind test, inside its own grammar; the learning loop gained nothing beside the searches. Four kinds of store would be built on a result that is not there |
| 3. Transfer between domains | Stage 2 shows a gain inside one domain | Nothing has been shown to carry even within a domain |
| 4. Changing a model's weights (fine-tuning, LoRA, weight-space programming) | Hardware and a model whose weights can be changed; item 1 or 2 passes | The MMLM's expert on a GPU still waits for a Colab run |
| 5. Controlled self-improvement | A change adopted by the recursion's judge holds on fresh problems, twice | The recursion exists with a sealed judge and has shown no rise. It is running again with the searches on |
| 6. Autonomous research missions | Stages 1 to 5 | |
| The package `apedp_dyn/intelligence/` (28 files), the mission controller, the knowledge store | With the stage each part belongs to | Plumbing with no experiment attached. The programme's own advice: evidence of increasing intelligence, not an architecture that looks increasingly intelligent |

## What is not taken over

- The code samples: they are sketches, and one of them calls any improvement above zero
  "positive". Comparisons here are paired, with a test fixed beforehand.
- "Verified discovery advantage" as one number: the paired counts it is made of (gained, lost,
  could it be chance) say more and are already what every comparison here reports.
- A date in the programme (no new architecture before a competition deadline of 20 January 2027)
  is the owner's to confirm; nothing here depends on it.
