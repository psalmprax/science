# Result: PREREG-MMLM-CM-2026-11

**Run:** once, 2026-11-19, at commit `f6ac90d`, which froze the preregistration, the script, the model,
the corpus loader and the network's weights (`preregistration/DIGEST-MMLM-CM.json`; the script checks
it before running). Python 3.13.18, numpy 2.5.3, torch 2.9.1, pyppmd 1.3.1, zstandard 0.25.0. Two CPU
cores; 115 s to fit the counts and compute every feature, then 30 to 100 s per mixer. No deviation.

Outputs, unedited, in [`RESULT-MMLM-CM-2026-11/`](RESULT-MMLM-CM-2026-11/): `results.json` and `run.log`.

## Outcome

Bits per byte on the WikiText-2 test split (1,256,449 bytes, never read before this run). Lower is
better.

| Model | Bits per byte |
| :--- | ---: |
| **MMLM-CM, all experts** | **1.455** |
| without the neural expert | 1.478 |
| without the word experts | 1.476 |
| without the read-so-far counts | 1.475 |
| without the match experts | 1.480 |
| training-text counts only | 1.549 |
| the neural expert alone | 1.772 |
| the best counting expert alone (order 4) | 1.945 |

| Reference compressor | Alone | Primed with the training text |
| :--- | ---: | ---: |
| PPMd (variant I), order 16 | 1.737 | **1.513** |
| PPMd, order 12 | 1.738 | 1.513 |
| PPMd, order 6 | 1.756 | 1.587 |
| lzma -9e, 64 MB dictionary | 2.143 | 1.857 |
| zstd -22, 128 MB window | 2.160 | 1.870 |
| bz2 -9 | 2.017 | 2.018 |
| gzip -9 | 2.620 | 2.614 |

## Against the expectations

1. **Beats every reference compressor: held.** 1.455 against 1.513 for the best of them (PPMd order
   16, primed), 3.8 % fewer bits.
2. **More than its parts: held.** 1.455 against 1.772 for its best expert, the network alone.
3. **The network earns its place: held.** Leaving it out costs 0.023 bits per byte (limit 0.005).
4. **So does every other group: held.** Leaving out the read-so-far counts costs 0.020, the match
   experts 0.025, the word experts 0.021.

All four held.

## What this means

- MMLM now has a general language model that works: on text it has not seen it is better than every
  classical compressor run here, each given the same training text. The old tracks A to C are
  superseded by it; they were at 4.2 bits per byte on other text and behind gzip.
- The mixture is the point. The network alone is at 1.77 and the best counting expert at 1.95; mixed,
  they reach 1.455. No single part is close.
- The network is the best single expert and still only the fourth part of the gain: it adds 0.023 on
  top of everything else, about the same as each other group. The counting experts already cover
  most of what a small network learns in 40 minutes.
- This is not state of the art for language modelling. The reference set is classical compressors.
  Large neural models trained with far more data and compute are well ahead of 1.455 bits per byte
  on text like this; none was run here.

## After the run (not preregistered)

- **No sign of overfitting to the tuning text.** The full model costs 1.459 on the validation split
  it was fitted on and 1.455 on the test split.
- **Development predicted the test.** On the development half the full model was at 1.476 and PPMd
  order 16 at 1.536; the ratio (0.961) is the same on the test split (0.962).
- **Cost.** The network: 806,144 weights, 40 minutes of training on two CPU cores. Everything else:
  under two minutes to fit on 10.8 MB, plus about a minute and a half for the mixer. Scoring is
  dominated by the network (200,000 bytes took 2.4 s on two cores when timed in development).

## Correction (2026-11-19, after the run)

The preregistration, this result and the corpus loader call the text "WikiText-2, raw". It is not the
raw version. The files (from the PyTorch examples repository, pinned by SHA-256, unchanged) are the
standard tokenised WikiText-2, in which rare words are written `<unk>` (14,334 training lines contain
it) and punctuation is spaced. No number changes, and the comparison holds, because the model and
every compressor were scored on the same bytes. What changes is how the absolute level reads:
tokenised text with `<unk>` is easier to predict than raw text, so 1.455 bits per byte must not be
compared with byte-level results published on raw text. The frozen files keep the wrong word, since
editing them would break the digest; this note and the documentation carry the correction.
