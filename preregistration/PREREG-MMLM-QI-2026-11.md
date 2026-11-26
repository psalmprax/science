# PREREG-MMLM-QI-2026-11: trained quantum-inspired byte models against a classical automaton of their size

**Status:** draft, not frozen. It will be frozen by `preregistration/DIGEST-MMLM-QI.json`, which pins this
file, `scripts/run_prereg_mmlm_qi.py`, `apedp_mmlm/quantum_inspired.py`, `apedp_mmlm/text_corpus.py`,
`.github/workflows/mmlm-qi.yml` and `tests/test_quantum_inspired_lm.py`, after the independent review below. Runs once, on GitHub, by pushing the
frozen commit to a branch `mmlm-qi-run-1`; the runner checks the digest before it reads anything.

## Why

On 26 November the owner asked whether the proposal "MMLM-QI-001, a quantum-inspired non-neural language
model" can help MMLM, and on hearing the answer said "yes set it up". MMLM already has a byte model on
WikiText-2 (MMLM-CM, 1.455 bits per byte on the test split, RESULT-MMLM-CM-2026-11) and a tensor-train
engine for equations whose cores were never trained (`apedp_mmlm/tensor_train_equations.py`: 40.8 bits per
held-out equation, the unigram's cost, behind a weighted automaton at 33.7). This tests what the proposal
adds that MMLM has not tried: models on a quantum state whose operators are *trained*, against the
classical model they have to beat.

## What is tested

`apedp_mmlm/quantum_inspired.py`. A byte model on a state of size D: pure (a unit vector psi) or mixed
(a density matrix rho with r = 2 operators per byte), complex or real. Byte x comes with probability
Tr(K_x rho K_x^dagger) (|K_x psi|^2 for a pure state) and the state becomes K_x rho K_x^dagger over that
probability. The operators are the row blocks of Q from a QR decomposition of a free matrix, so
sum over x of K_x^dagger K_x = I holds exactly and the probabilities sum to one for every state. A pure
state is the same model as a left-canonical uniform matrix product state read left to right as a Born
machine (the proposal's candidate B); the mixed state is its open-system generalisation (candidate C).

Because rho evolves linearly, the model is a weighted automaton with D^2 states. The classical control
is therefore a hidden Markov model with N = D^2 states (a nonnegative weighted automaton; a general one
with negative entries cannot be trained as a distribution in this way). For D = 16 its parameter count
(131,328) is about the complex model's (131,104); for D = 32 it has 1.31 million against 0.52 million.

## Arms (25)

| Arm | What | Parameters |
| :--- | :--- | ---: |
| `pure-complex-16`, `-32`, `-64` | pure state, complex | 131,104 / 524,352 / 2,097,280 |
| `pure-real-16`, `-32`, `-64` | pure state, real, same D | 65,552 / 262,176 / 1,048,640 |
| `pure-real-23`, `-45`, `-90` | pure state, real, about the parameters of complex 16, 32, 64 | 135,447 / 518,445 / 2,073,690 |
| `mixed-complex-16-2`, `-32-2` | mixed state, complex, two operators per byte | 262,176 / 1,048,640 |
| `mixed-real-16-2`, `-32-2` | mixed state, real | 131,088 / 524,320 |
| `hmm-256`, `hmm-1024` | hidden Markov model with 16^2 and 32^2 states | 131,328 / 1,311,744 |
| `pure-complex-23`, `-45` | pure state, complex, about the parameters of the two complex mixed models | 270,894 / 1,036,890 |
| `...@1`, `...@2` of `pure-complex-16`, `-32`, `hmm-256`, `hmm-1024` | expectation 2's four arms with seeds 1 and 2 as well; it is read on the three-seed means | |

## Data

WikiText-2 as bytes, the copy in the PyTorch examples repository (`apedp_mmlm/text_corpus.py`, files pinned by
SHA-256; it is the tokenised version, with `<unk>` for rare words, not the raw one). Training split (10.8 MB):
every model trains on it. Validation split: its first 300,000 bytes choose the learning rate and the
parameters kept. **Test split (1.26 MB): scored only by this run.** It was read once before, by
RESULT-MMLM-CM-2026-11's run; nothing in this preregistration's development has read it (development used
the training split, the first 300,000 bytes of `valid_a` and `valid_b`).

## Protocol

Each arm runs alone on one GitHub runner (ubuntu-latest, four threads):

1. **Rate sweep.** Seven learning rates, 0.003, 0.01, 0.03, 0.1, 0.3, 1 and 3, the same for every arm; at
   each, a fresh model trains for 150 seconds and is scored on the selection slice. The rate kept is the
   best; a rate whose score is not finite (it diverged) is never kept. A byte's probability is never taken
   below 1e-30 (no log of zero, no undefined state).
2. **Training.** A fresh model (same seed) at that rate trains for 2,700 seconds (45 minutes): Adam,
   batch 64 streams, segments of 128 bytes with the state carried between segments (truncated
   back-propagation), gradient norm clipped at 1. Every 300 seconds and at the end it is scored on the
   selection slice; the parameters kept are the best of those.
3. **Scoring.** The test split cut into 32 contiguous pieces, each scored from the model's initial state;
   bits per byte over all of them.

Equal wall-clock is the comparison: a model that costs more per byte sees fewer bytes, and the bytes
seen are reported. `baselines`: interpolated Kneser-Ney with 0 to 4 bytes of history, fitted on the
training split, scored on the test split as one stream. References, not rerun, from
RESULT-MMLM-CM-2026-11: MMLM-CM 1.455, PPMd order 16 primed with the training text 1.513, MMLM-CM's LSTM
expert alone 1.772, its order-4 counts alone 1.945.

## Expectations, fixed now

All in bits per byte on the test split. "The best quantum-inspired arm" is, of the 15 non-HMM arms with
seed 0, the one best on the selection slice (not on the test split).

1. **It learns more than three-byte statistics:** the best quantum-inspired arm costs less than
   Kneser-Ney with 2 bytes of history.
2. **Quantum structure beats a classical automaton of its state's size:** over seeds 0, 1 and 2,
   `pure-complex-16`'s mean costs at least 0.05 less than `hmm-256`'s, and `pure-complex-32`'s at least
   0.05 less than `hmm-1024`'s. (The automaton has D^2 states; at D = 32 it has 2.5 times the parameters.)
3. **Complex amplitudes earn their place:** `pure-complex-D` costs at least 0.02 less than the real model
   with about its parameters (`pure-real-23`, `-45`, `-90`) for each of D = 16, 32, 64. The real model of
   the same D is reported beside it.
4. **A mixed state earns its place at equal size:** `mixed-complex-16-2` costs at least 0.02 less than
   `pure-complex-23`, and `mixed-complex-32-2` at least 0.02 less than `pure-complex-45` (each pair about
   the same number of parameters). The pure model of the same D, and the real mixed models, are reported
   beside them.
5. **A larger state is better:** `pure-complex-16` > `pure-complex-32` > `pure-complex-64`.
6. **Competitive with counting:** the best quantum-inspired arm costs less than Kneser-Ney with 4 bytes
   of history.
7. **A hybrid with counting gains from the quantum structure:** each arm is also mixed byte by byte with
   Kneser-Ney with 4 bytes of history (weight on the model from 0, 0.05, ..., 1, chosen on the selection
   slice; both scored on the same bytes of the test split). For D = 16 and 32, `pure-complex-D` mixed so
   costs at least 0.02 less than Kneser-Ney alone **and** at least 0.02 less than `hmm-D^2` mixed the same
   way (both on the three-seed means). The second half asks whether what the mixture gains comes from
   the quantum-inspired model or would come from any second model of its size.

Reported without a criterion: the gap from the best arm to each reference; bytes seen; scoring speed;
the rates chosen; each runner's processor; and, for 2 and 7, the larger seed spread at each D (a margin
no larger than it is reported as within the noise of the seeds, D by D). 3 and 4 rest on one seed per arm.

A miss is reported as a miss. An arm that fails to run is reported as missing; each expectation that
needs it is then marked not read, and the others are read.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

Read in this order; the first line that applies is the proposal. 1, 4 and 5 propose nothing on their own;
they say what the model is like.

| 7 (hybrid) | 2 (against the automaton) | 3 (complex) | Proposal |
| :--- | :--- | :--- | :--- |
| holds | holds | holds | develop the complex quantum-inspired model as one more expert in MMLM-CM's mixture, under its own preregistration |
| holds | holds | fails | the same with the real model: the gain is the structure, not complex numbers (under this optimiser and budget) |
| holds | fails | either | an expert is still worth a preregistration of its own, but as a hybrid only: it adds to counting more than an automaton of its size does, though alone it does not beat that automaton |
| fails | any | any | do not develop quantum-inspired language models in MMLM: mixed with counting, they add nothing a hidden Markov model with D^2 states (2.5 times the parameters at D = 32) does not |

6 is read on its own: if it fails, no claim is made that these models compete with counting, whatever the
table says.

## Disclosed before freezing

- **Development numbers (local, two cores, 100-second runs, selection on `valid_a`'s first 100,000
  bytes).** After 100 s: `pure-complex-16` 3.27 to 3.34 (batch 32 to 128, rates 0.01 and 0.03), 3.85 at
  rate 0.003; `hmm-256` 4.28 at rate 0.01, 3.68 at 0.03, 3.27 at 0.1; `pure-real-16` 4.23 at 0.03.
  Training speeds: `pure-complex-32` 32,000 bytes per second, `mixed-complex-16-2` 19,000, `hmm-1024`
  11,000 (batch 32). Kneser-Ney on `valid_b`: 4.62, 3.39, 2.72, 2.18, 1.85 with 0 to 4 bytes of history.
- **A change made while the rehearsal ran (26 Nov, 20:05).** The owner passed on another assistant's
  write-up of a tensor-network language model and its prototype (character-level TinyStories, CPU, one
  run per model: a 4-character n-gram 1.88 bits per character, a small LSTM 1.50, a hidden Markov model
  with 128 states 2.25, a Born machine with bond dimension 32 2.83, a locally purified state with bond
  dimension 32 2.49; numbers not checked here). The mixed-state model here is a uniform locally purified
  state in its trace-preserving form, with the number of operators per byte as its purification
  dimension. That write-up compared it with the pure state at the same bond dimension, where the mixed
  one has twice the parameters. So expectation 4 now compares at equal parameters, and two arms were added
  for it (`pure-complex-23`, `-45`); the comparison at the same D is still reported. The two new arms are
  not in the rehearsal; they are the rehearsed pure model at other sizes.
- **Choices made on those numbers:** the rate grid (0.3 added after the hidden Markov model chose the
  highest rate tried, 0.1; then 1 and 3 after the rehearsal, where both hidden Markov models chose 0.3 and
  `hmm-1024` was still falling steeply there: 4.04 at 0.1, 2.76 at 0.3); batch 64 and segments of 128;
  the D values and the real sizes with matched parameters.
- **Rehearsal** on the development split (selection on `valid_a`'s first 300,000 bytes, scored on
  `valid_b`), the 17 arms then planned, sweep 60 s at each rate and 10 minutes of training (run 45871761319,
  `benchmarks/mmlm/mmlm_qi_rehearsal.json`). Bits per byte: `pure-complex-16` 3.08 (second seed 3.06),
  `-32` 2.84, `-64` 2.77; `pure-real-16` 3.82, `-32` 3.76, `-64` 3.52, `-23` 3.45, `-45` 3.72, `-90` 3.73;
  `mixed-complex-16-2` 3.13, `-32-2` 2.95; `mixed-real-16-2` 3.22, `-32-2` 2.93; `hmm-256` 3.05 (second
  seed 2.82), `hmm-1024` 2.50. Kneser-Ney with 0 to 4 bytes of history 4.62, 3.39, 2.72, 2.18, 1.85. Rates
  chosen: 0.01 to 0.3; the hidden Markov model chose 0.3 (the top of the grid) and its two seeds chose 0.3
  and 0.1 (its seed spread was 0.23). Read as the plan reads: 1 missed (2.77 against 2.72),
  2 missed (`hmm-1024` better by 0.34), 3 held, 4 (as it then read, at the same D) missed, 5 held, 6
  missed; the seed spread covers the margin of 2 at D = 16 (0.03 against 0.23) but not at D = 32 (0.34).
  Every arm finished within 11 minutes of training and scoring; the final run's 45 minutes plus a
  17.5-minute sweep (seven rates) and its scoring fit the 120-minute step limit.
- **Changes after the rehearsal, before freezing.** Expectation 4 compares at equal parameters (above).
  Expectation 7, the hybrid with counting, was added on the owner's word of 26 November (20:32: "test
  hybrid approach in all the suggested solutions"). Then, on the independent review (below): rates 1 and
  3; three seeds for expectation 2's arms, read on means; the best arm chosen on the selection slice; a
  floor on a byte's probability; each expectation read on its own when an arm fails; the table of
  proposals; each runner's processor recorded; the test file frozen and its training test given 10
  seconds. A second rehearsal of all 25 arms runs this code before freezing (`mmlm-qi-smoke-2`).

## Limits

- One seed per arm, except expectation 2's four arms (three each). A seed spread mixes the start, the
  rate chosen and the runner's speed (the data order is the same for every seed).
- Equal wall-clock is not equal compute: in the rehearsal two runs of the same model differed by up to
  36 % in bytes seen (`hmm-256` 59.2 and 80.6 million in 600 s). Each runner's processor and core count
  are now recorded, and bytes seen are reported.
- The real-valued arms may be under-optimised: in the rehearsal's sweep no rate took `pure-real-16` or
  `pure-real-23` below the cost of Kneser-Ney with no history, and `pure-real-45` and `-90` came out worse
  than `-23`. Expectation 3 is therefore about complex against real under this optimiser and budget.
- 45 minutes of training is short for the larger models; the conclusion is about these budgets.
- Scoring in 32 pieces from the initial state is slightly pessimistic for every model, against the
  references' single stream; a piece is 39,000 bytes long.

## Review (AGENTS.md 1.6)

Before freezing, a separate reviewer that did not take part checks this plan against the runner: the
arms, the denominators, the margins and the consequences against what `expectations` in
`scripts/run_prereg_mmlm_qi.py` computes. Its findings, and what was done about each, are written here.
Before the result is recorded, a reviewer checks every number of the record against the raw outputs.
The rule on traps concerns blind tests of a check; this compares models on a fixed public benchmark and
tests no check, so it has no trap arm, and says so.

**Findings of the review before freezing (26 November), and what was done:**

1. The rate grid held the hidden Markov model at its top (both chose 0.3; `hmm-1024` still falling) and so
   under-tuned the control. *Done:* rates 1 and 3 added; a diverging rate is never kept.
2. The proposals contradicted each other (2 fails with 7 holding; 7 failing with 2 and 3 holding) and 1, 4
   and 5 had none. *Done:* one table read in order; 1, 4 and 5 propose nothing; the automaton's size stated.
3. Expectation 2's margin (0.05) was inside one seed's spread (0.23). *Done:* its four arms run with three
   seeds and it is read on the means; spreads reported per D, for 7 too.
4. The real-valued arms looked under-optimised (no sweep rate took two of them below the no-history
   counting model). *Done:* disclosed; expectation 3 limited to this optimiser and budget.
5. The plan said a failed arm leaves only its expectations unread; the runner read none. *Done:* each
   expectation is read on its own.
6. The code after the rehearsal had not run on GitHub. *Done:* a second rehearsal before freezing.
7. Equal wall-clock is not equal compute (36 % between two runs of one model). *Done:* the processor and
   core count recorded; stated under Limits.
8. The best arm was picked on the test split. *Done:* picked on the selection slice.
9. Two sentences of the rehearsal bullet were wrong (the spread covers 2's margin only at D = 16; a causal
   claim the record cannot check). *Done:* corrected.
10. A zero probability would give a log of zero and an undefined state. *Done:* a floor of 1e-30.
11. The data is the tokenised WikiText-2, not the raw. *Done:* the wording.
12. The test file decides whether an arm runs but was not frozen, and its 6-second training test could
    fail on a slow runner. *Done:* frozen; 10 seconds.

The reviewer checked and found no fault in: the arms and parameter counts, the protocol against the
runner and workflow, the arithmetic of every expectation, the rehearsal's numbers, and that nothing here
changes what is in force.
