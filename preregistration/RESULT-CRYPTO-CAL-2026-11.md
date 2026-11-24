# Result: PREREG-CRYPTO-CAL-2026-11

**Run:** once, 2026-11-17 02:11:02–02:11:55 UTC, at commit `2cf7789`, which froze the
preregistration and the script (`preregistration/DIGEST-CAL.json`; the script checks it before
running). Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5.

**Data:** `panel_development.csv.gz` from the `market-data` branch (SHA-256
`96b5ee0b88c4dd5c004565f2582c8683b440513ad5dc9f8d511a0fb49fa4644e`), 2021-01-01 to 2025-12-31:
52,887 asset-day pairs on 1,765 days in the point-in-time top-30 universe. The holdout file was not
given to the script.

Outputs, unedited, in [`RESULT-CRYPTO-CAL-2026-11/`](RESULT-CRYPTO-CAL-2026-11/): `results.json`,
`candidates.jsonl` (every candidate's measured IC and verdict), `promotion_journal.jsonl` (every
stage record) and `run.log`.

## Outcome

| Family | n | Passed stage 2 | Passed stage 3 | Measured development IC |
| :--- | ---: | ---: | ---: | :--- |
| Placebo | 500 | **0** (95% interval 0–0.74%) | 0 | mean 0.0003, sd 0.0046, largest in absolute value 0.0178 |
| Injected, ρ = 0.010 | 100 | 1 | 0 | mean 0.0095 |
| Injected, ρ = 0.015 | 100 | 11 | 11 | mean 0.0149 |
| Injected, ρ = 0.020 | 100 | 55 | 50 | mean 0.0204 |
| Injected, ρ = 0.025 | 100 | 86 | 86 | mean 0.0249 |
| Injected, ρ = 0.030 | 100 | **97** | 97 | mean 0.0292 |
| Injected, ρ = 0.050 | 100 | **100** | 100 | mean 0.0488 |
| Reversal (one day) | 1 | yes (IC 0.0287) | **no**: literature R² 1.00 | |
| Momentum (twenty days) | 1 | no (IC −0.0028) | not reached | |

**All three preregistered expectations held:**

1. Placebo false passes at stage 2: 0 of 500, against a criterion of at most 1%. The placebo ICs'
   spread (sd 0.0046) is the 0.004 the preregistration predicted from the sample size, so the
   0.02 bar sits about four standard deviations out on this data, fat tails and cross-sectional
   dependence included.
2. Power: 1% at ρ = 0.01 (criterion at most 10%), 97% at ρ = 0.03 and 100% at ρ = 0.05 (criterion
   at least 90%). The curve crosses one half at an effect of about 0.02, which is where the canonical
   threshold puts it.
3. Literature: one-day reversal is a real effect in this panel (IC 0.0287, every fold positive)
   and passed stage 2; stage 3 refused it as the literature factor it is. Momentum did not pass
   stage 2, so its half of this expectation is untested.

Six candidates passed stage 2 and were refused at stage 3, all with an IC within 0.0005 of the
threshold: five injected at ρ = 0.02 and one at ρ = 0.01, each refused because its Diebold–Mariano
p-value against a zero forecast was 0.010 to 0.013, just above 0.01. At the boundary the two tests
disagree by a hair, in the conservative direction.

## What this does and does not show

- It measures stages 2 and 3 only, on development data. Stage 4 cannot be calibrated this way: the
  experiment ledger lets one candidate confirm on a given holdout window, by design.
- The injected signals are additive and stationary by construction. A real effect that comes and
  goes with the market regime would fail the five-fold stability requirement more often than these.
- A placebo false-pass rate of 0 of 500 bounds the rate below 0.74% at 95%; it does not show it is
  zero.
- Every stage-1 record is reported, not proved, as in the parent preregistration.
