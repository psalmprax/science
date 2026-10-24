# Statistical Equations Study (APEDP) — resumable

**Status:** measurement complete (demography + epidemics). Resume entry
points at bottom.
**Language rule:** fit/extrapolation claims only. "Discovered" means
"machine-generated candidate," never "new law." Percentages below are
error reductions on stated windows, not superiority claims.

## 1. The quadratic candidate (system-generated, seed 7, train ≤1940)

```
N(t) = 0.00614670868347349·t² − 0.0325651260504243·t + 4.3178921568618
```

Working form: `N(t) = 0.00615t² − 0.0326t + 4.32`.
t = year − 1790 (dimensionless count); N = US population, millions.
Intercept ≈ 1790 population (3.9M). No saturation mechanism.

## 2. US benchmark (decennial census; 2020 actual 331.449M)

Train ≤1940 → extrapolate 1950–2020.

| Model | 2020 pred | MAE | MSE | RMSE |
|-------|-----------|-----|-----|------|
| Logistic (fitted K=186M — ours, not Pearl-Reed's ~197M) | 180.4 | 74.4 | 7821.8 | 88.4 |
| Gompertz | 255.5 | 36.3 | 1940.7 | 44.1 |
| Quadratic (above) | 322.0 | 7.7 | 73.0 | 8.5 |

Error reduction: −89.6% vs logistic, −78.8% vs Gompertz (MAE basis).
Metric rationale: MAE = typical error (population units); NMAE = cross-era
comparison; max-error = worst case; MSE/RMSE only when punishing large
misses is the goal (they exaggerate single-point catastrophes: 107× here).

## 3. Relativity cross-check (same session, Bertozzi 1964, 5 runs)

Classical `v=√(2K/m)` predicts superluminal electrons at all five
energies (1.40c–7.66c); measured ≤c always; relativistic form matches to
~3%. Candidate `v²/2c²` saturates 0.376→0.500 while γ−1 runs 0.98→29.35:
regime tag confirmed. D-T Q-value recomputed: 17.589 vs 17.6.

## 4. Rolling-origin, USA (6 windows; ranks 1,2,2,2,2,1)

Best 2/6, second 4/6, worst 0/6 (corrected from an early "4/6" miscount).
Worst-case NMAE: quad 13.5%, logistic 30.8%, Gompertz 32.8%.
Gompertz fit-fails 1/6. Full per-window MAE/RMSE/NMAE/max tables measured
in-session; scale analysis `E_kin/E_rest = v²/2c² = 5e-07 → small-correction`
(`apedp/scientific_layers.py:scale_ratio`, tested).

## 5. Cross-country pilot (9 × 3 windows, OWID UN-era)

No universal winner. France: logistic 0.3% NMAE. UK: exponential takes
later windows (steady segments, not an inherent-growth claim). Nigeria:
quad wins; Gompertz fails 2/3. Japan: quad takes stagnation window.
China: quad early → logistic later. Brazil: logistic sweep. USA/Germany/
India: mixed.

## 6. Full sweep (785 country-windows, 265 countries; harness
`/tmp/opencode/pop_sweep.py`, source `/tmp/opencode/pop_owid.csv`)

Wins (NMAE rank-1): quadratic 315 (40%), logistic 245 (31%),
exponential 150 (19%), Gompertz 75 (10%). Fit failures: Gompertz 35.3%
(277/785); others 0%. Per-row MAE/RMSE/NMAE/max/bias/endpoints recorded.

## 7. Epidemics (OWID COVID cumulative cases; `/tmp/opencode/covid_sweep.py`)

Same 4-model protocol, day-windows per country (8 countries). Verdict:
single-sigmoid families largely fail on multi-wave curves (double-digit
NMAEs; exponential blowups to 2418%; Gompertz fit-fails recur).
Quadratic most robust (no failures, no catastrophes; e.g. Brazil w3 2.9%)
but errors stay large. Negative result recorded as such: the protocol
certifies demography extrapolatable and multi-wave epidemics largely not.

## 8. Adopted interpretation

"A comparatively stable candidate historical extrapolator over the
tested windows; not established as a general population law."
Complementary-pair principle: approximations should ship with regime
gauges (#2 computes, #1 certifies). Fit, extrapolation, constraints,
interpretability, status scored separately — never collapsed.

## 9. Methodological principle (adopted)

Model selection is treated as an empirical hypothesis-testing problem
rather than a ranking exercise: an observed lower error is not
interpreted as evidence of superiority unless the difference is shown
to be robust under independent or repeated out-of-sample evaluation.
Throughout, three quantities are kept separate: win frequency (lowest
observed error), effect size (how much lower), statistical uncertainty
(distinguishable from sampling variation or not).

Precedent: a 5.3% MAE advantage (Nu candidate vs Dittus-Boelter) failed
to reproduce across fresh samples (p = 0.210/0.122/0.060, winner flips
by seed) and the superiority claim was withdrawn. The retraction is
recorded as validation-layer success, not candidate success.

## 10. Resume entry points

- `/tmp/opencode/pop_sweep.py`, `/tmp/opencode/covid_sweep.py` (harnesses)
- `/tmp/opencode/pop_owid.csv` (58k rows), `/tmp/opencode/covid.csv` (93.8M)
- `tests/test_scientific_layers.py` (scale/taxonomy/domain/report tests)
- Open: per-regime discovery-engine recurrence; digitized θ(r)/T00(r)
  warp profiles; Natario-vs-continuity template test.
