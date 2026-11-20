# Preregistration: PySR, PhySO and the exhaustive search on the baseline's SRBench protocol

**Document ID:** PREREG-SRBENCH-EXTERNAL-2026-11
**Frozen:** at the commit that adds this file. `preregistration/DIGEST-SRBENCH-EXTERNAL.json` holds the
SHA-256 of this document, `scripts/run_srbench_external.py`, `scripts/run_srbench_baseline.py`,
`scripts/fetch_srbench_ground_truth.py` and `benchmarks/srbench/manifest.json`; the harness refuses to
run if any changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.
**Frozen before any result of PREREG-SRBENCH-BASELINE-2026-11 was seen** (that run was still in
progress, 8 of 32 shards finished, none read).

## Why

The baseline measures the repository's four existing engines. This run measures three more on the
same problems, rows and scoring, so that the question "is an outside engine better than ours, and
is the new exhaustive search better than both" has a number.

## Protocol

Everything is imported from the baseline harness unchanged: the 133 problems, the 1,000/500 split
seeded by problem name, renamed variables, noise on the training target only, one isolated process
per run, symbolic recovery (constants except exponents refitted to the clean truth, no term refitted
away) and accuracy (test R² ≥ 0.999). What differs:

| | |
| :--- | :--- |
| Engines | **PySR** 2.7.2 (`+ - * /`, `sqrt exp log sin cos square`; serial, deterministic, seed 0, `model_selection="best"`); **PhySO** 1.1.11 (`config0`, units supplied, two free constants, the same function set); **exhaustive_mdl** (`apedp_dyn/discovery_l2/exhaustive_search.py`, 6 nodes up to five variables, 5 beyond) |
| Time | PySR and PhySO search for 240 s; every run's process is killed at 330 s. The baseline's engines had 120 s; they finish in seconds to tens of seconds or time out, and the difference is stated wherever the two runs are compared |
| Noise | 0 and 1 % (two of the baseline's four levels, to keep the run inside a day) |
| Runs | 133 × 2 × 3 = 798 |

## Expectations, fixed now

1. **An outside engine beats every baseline engine:** at each of the two noise levels, the better of
   PySR and PhySO has a higher symbolic recovery rate on the Feynman problems than the best of L2,
   RIME, RIME-AD and the extended hierarchical search at that level.
2. **The exhaustive search beats the baseline engines too:** the same comparison with
   `exhaustive_mdl` in place of the outside engines.
3. Whether `exhaustive_mdl` beats PySR or PhySO is reported without a criterion.

A miss on 1 or 2 is reported as that.

## Disclosed before freezing

- Each adapter was run on a toy law in no benchmark (y = x0·x1 + x0), on GitHub Actions for PySR and
  PhySO; all three recover it.
- PhySO was also run once on `feynman_I_12_1` (5 epochs) while writing its adapter.
- `exhaustive_mdl` was run on the four problems the baseline's preregistration already disclosed
  (`feynman_I_12_1`, `feynman_II_6_15a`, `feynman_I_6_2a`, `strogatz_bacres1`, 1 % noise): it recovered
  the first and third. Its pairing rule (leaders summed with every simple term) and its floor on
  rounding-level residuals were written while looking at three synthetic laws, not at benchmark
  problems.
- PhySO's time stop is this harness's, not PhySO's: its authors run it for far longer, so its number
  here is PhySO at four minutes, not PhySO.

## Deviation rule

One run, on GitHub Actions (`.github/workflows/srbench-external.yml`, branch
`srbench-external-run-1`). An infrastructure failure is fixed by a dated deviation committed here
before re-running. Changing an engine's settings, the scoring or an expectation after the run is a
new preregistration.
