# PREREG-MMLM-CM-2026-11: the generic model of MMLM on text it has not seen

**Status:** frozen before the run by `preregistration/DIGEST-MMLM-CM.json`, which also pins the
trained network's weights. Runs once:
`python scripts/run_prereg_mmlm_cm.py --out preregistration/RESULT-MMLM-CM-2026-11`.

## What is tested

`apedp_mmlm/context_mixing.py` (MMLM-CM) is a byte-level language model: counting experts over the
training text, the same contexts counted over the text read so far, two long-range match experts and
one small LSTM trained on CPU (`apedp_mmlm/neural_expert.py`), combined by a mixer whose weights are
fitted on held-out text. The question is whether this is a competitive general language model for
its size, whether the mixture is more than its parts, and whether each part earns its place.

## Data

WikiText-2, raw, bytes (`apedp_mmlm/text_corpus.py`, files pinned by SHA-256).

- **Training split (10.8 MB):** the counts and the network.
- **Validation split (1.1 MB):** what a count state is worth, and the mixing weights.
- **Test split (1.3 MB):** read only by this run. Every number below is bits per byte on it.

## Protocol

The model is fitted as above and scores the test split from its first byte. Six arms differ only in
which experts the mixer is given: all; without the network; without the read-so-far counts; without
the match experts; without the word experts; training-text counts only. Each expert is also scored
alone (its calibrated prediction, no mixing).

Reference compressors (gzip, bz2, lzma with a 64 MB dictionary, zstd with a 128 MB window, PPMd
variant I at orders 6, 12 and 16) are scored two ways: on the test split alone, and *primed*, as the
extra bytes needed for the test split after the training split. Primed is the fair comparison,
because the model has read the training split too. The compressors adapt as they read; so does the
model, through its read-so-far and match experts.

## Expectations, fixed now

1. **Beats every reference compressor:** the model with all experts costs fewer bits per byte than
   each compressor primed with the training text.
2. **More than its parts:** it costs fewer bits per byte than each of its experts alone, including
   the network.
3. **The network earns its place:** leaving it out costs at least 0.005 bits per byte more.
4. **So does every other group:** leaving out the read-so-far counts, the match experts or the word
   experts each costs more than the full model.
5. Reported without a criterion: the absolute level, the training-text-counts-only arm and the
   compressors' unprimed numbers.

A miss on 1 to 4 is reported as a miss.

## Disclosed before freezing

- **Development numbers.** With the validation split cut in two (first half to fit, second to
  measure): 1.476 bits per byte with all experts; 1.498 without the network; 1.497 without the
  read-so-far counts; 1.503 without the match experts; 1.497 without the word experts; 1.582 with
  training-text counts only; the network alone 1.789; the best primed compressor (PPMd order 16)
  1.536 (`benchmarks/mmlm_cm_dev_study.json`).
- **Choices made on that second half:** adding order 8 and the read-so-far counts; selecting mixing
  weights by the network's confidence (1.481 to 1.476); not selecting them by the preceding byte
  (1.483, worse than 1.481). The PPMd orders were chosen to favour the baseline: order 16 was its best.
- **One difference from development.** Here the whole validation split fits the calibration and the
  mixer, not half of it. More fitting text is expected to help slightly; it was not measured.
- **The network.** A single-layer LSTM, 384 hidden units, 806,144 weights, trained for 40 minutes on
  two CPU cores (7,411 steps, 4.4 passes over the training split), once, with seed 0. Its weights are
  committed and pinned. It was not tuned: this is the first and only training run.
- **The test split.** MMLM-CM has never been run on it. An earlier unfinished draft script of mine
  fetched the three files and may have scored the old tracks on the first 8,000 tokens of the test
  split; nothing it produced was kept or used.
- **What this cannot show.** A trained transformer with far more data and compute is well ahead of
  any model of this size on text. The claim tested is narrower: the best byte-level model that can be
  fitted here in an hour on two CPU cores, against the strongest classical compressors.

## Deviation rule

The run is not repeated. A crash is fixed only if the fix cannot change a number (for example memory
use or an output path), the fix is recorded with its date in the result, and the digest is re-frozen
with the old digest kept in the log. Any other change makes the result a development result.
