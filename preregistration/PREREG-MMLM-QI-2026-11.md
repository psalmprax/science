# PREREG-MMLM-QI-2026-11: trained quantum-inspired byte models against a classical automaton of their size

**Status:** draft, not frozen. It will be frozen by `preregistration/DIGEST-MMLM-QI.json`, which pins this
file, `scripts/run_prereg_mmlm_qi.py`, `apedp_mmlm/quantum_inspired.py`, `apedp_mmlm/text_corpus.py` and
`.github/workflows/mmlm-qi.yml`, after the independent review below. Runs once, on GitHub, by pushing the
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

## Arms (19)

| Arm | What | Parameters |
| :--- | :--- | ---: |
| `pure-complex-16`, `-32`, `-64` | pure state, complex | 131,104 / 524,352 / 2,097,280 |
| `pure-real-16`, `-32`, `-64` | pure state, real, same D | 65,552 / 262,176 / 1,048,640 |
| `pure-real-23`, `-45`, `-90` | pure state, real, about the parameters of complex 16, 32, 64 | 135,447 / 518,445 / 2,073,690 |
| `mixed-complex-16-2`, `-32-2` | mixed state, complex, two operators per byte | 262,176 / 1,048,640 |
| `mixed-real-16-2`, `-32-2` | mixed state, real | 131,088 / 524,320 |
| `hmm-256`, `hmm-1024` | hidden Markov model with 16^2 and 32^2 states | 131,328 / 1,311,744 |
| `pure-complex-23`, `-45` | pure state, complex, about the parameters of the two complex mixed models | 270,894 / 1,036,890 |
| `pure-complex-16@1`, `hmm-256@1` | the same two with a second seed, for the noise | |

## Data

WikiText-2, raw, bytes (`apedp_mmlm/text_corpus.py`, files pinned by SHA-256). Training split (10.8 MB):
every model trains on it. Validation split: its first 300,000 bytes choose the learning rate and the
parameters kept. **Test split (1.26 MB): scored only by this run.** It was read once before, by
RESULT-MMLM-CM-2026-11's run; nothing in this preregistration's development has read it (development used
the training split, the first 300,000 bytes of `valid_a` and `valid_b`).

## Protocol

Each arm runs alone on one GitHub runner (ubuntu-latest, four threads):

1. **Rate sweep.** Five learning rates, 0.003, 0.01, 0.03, 0.1 and 0.3, the same for every arm; at each, a
   fresh model trains for 150 seconds and is scored on the selection slice. The rate kept is the best.
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

All in bits per byte on the test split; "the best quantum-inspired arm" is the lowest of the 15 non-HMM
arms with seed 0.

1. **It learns more than three-byte statistics:** the best quantum-inspired arm costs less than
   Kneser-Ney with 2 bytes of history.
2. **Quantum structure beats a classical automaton of its size:** `pure-complex-16` costs at least 0.05
   less than `hmm-256`, and `pure-complex-32` at least 0.05 less than `hmm-1024`.
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
   way. The second half asks whether what the mixture gains comes from the quantum-inspired model or
   would come from any second model of its size.

Reported without a criterion: the gap from the best arm to each reference; bytes seen; scoring speed;
the rates chosen; and the seed spread of `pure-complex-16` and `hmm-256`. Where a margin in 2, 3 or 4 is
no larger than the larger seed spread, the record says that margin is within the noise of one seed.

A miss is reported as a miss. An arm that fails to run is reported as missing; the expectations that
need it are then not read.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

- **2 and 3 hold:** propose developing the quantum-inspired model further for MMLM, first as one more
  expert in MMLM-CM's mixture, under its own preregistration.
- **2 holds, 3 fails:** the gain is the linear-algebra structure, not complex numbers; propose a real
  Kraus model or a weighted automaton as that expert instead, and not complex amplitudes.
- **2 fails:** propose not developing quantum-inspired language models in MMLM; a classical automaton of
  the same size does as well.
- **6 fails:** no claim that these models compete with counting, whatever 2 to 5 say.
- **7 holds:** propose the quantum-inspired model as one more expert in MMLM-CM's mixture (a hybrid, not
  a replacement), under its own preregistration, whatever 6 says. **7 fails:** it adds nothing to counting
  that a hidden Markov model of its size would not.

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
  highest rate tried, 0.1, so that neither family is held at the edge of the grid); batch 64 and segments
  of 128; the D values and the real sizes with matched parameters.
- **Rehearsal** on the development split (selection on `valid_a`'s first 300,000 bytes, scored on
  `valid_b`), the 17 arms then planned, sweep 60 s at each rate and 10 minutes of training (run 45871761319,
  `benchmarks/mmlm/mmlm_qi_rehearsal.json`). Bits per byte: `pure-complex-16` 3.08 (second seed 3.06),
  `-32` 2.84, `-64` 2.77; `pure-real-16` 3.82, `-32` 3.76, `-64` 3.52, `-23` 3.45, `-45` 3.72, `-90` 3.73;
  `mixed-complex-16-2` 3.13, `-32-2` 2.95; `mixed-real-16-2` 3.22, `-32-2` 2.93; `hmm-256` 3.05 (second
  seed 2.82), `hmm-1024` 2.50. Kneser-Ney with 0 to 4 bytes of history 4.62, 3.39, 2.72, 2.18, 1.85. Rates
  chosen: 0.01 to 0.3; the hidden Markov model chose 0.3 (the top of the grid) and its two seeds chose 0.3
  and 0.1, which is most of its seed spread of 0.23. Read as the plan reads: 1 missed (2.77 against 2.72),
  2 missed (`hmm-1024` better by 0.34), 3 held, 4 (as it then read, at the same D) missed, 5 held, 6
  missed; the seed spread covers the margins of 2 and 4. Every arm finished within 11 minutes of training
  and scoring; the final run's 45 minutes plus a 12.5-minute sweep fit the 120-minute step limit.
- **Changes after the rehearsal, before freezing.** Expectation 4 compares at equal parameters (above).
  Expectation 7, the hybrid with counting, was added on the owner's word of 26 November (20:32: "test
  hybrid approach in all the suggested solutions"); its computation was checked locally on a short run
  (an untrained model gets weight 0 and the mixture equals Kneser-Ney), not in the rehearsal.

## Limits

- One seed per arm, except the two repeated ones.
- Runner speeds vary between jobs; bytes seen are reported.
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

*Findings: to be written here.*
