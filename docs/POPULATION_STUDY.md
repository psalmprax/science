# Population Extrapolation Study (APEDP) — resumable

**Status:** measurement complete through full-country sweep. Resume entry
points at bottom.
**Language rule for this file:** fit/extrapolation claims only; no law-claims.
Saturation mechanism: none established for polynomial forms.

## 1. US benchmark (corrects earlier "order of magnitude" wording)

Data: US decennial census, millions; 2020 actual 331.449M (Census Bureau).
Train ≤1940, extrapolate 1950–2020, t = year−1790.

| Model | 2020 pred | Extrap MAE | vs logistic | vs Gompertz |
|-------|-----------|------------|-------------|-------------|
| Logistic (fitted K=186M — ours, not Pearl-Reed's ~197M) | 180.4 | 74.4 | — | — |
| Gompertz | 255.5 | 36.3 | −51.2% | — |
| Discovered quadratic `0.00615t²−0.0326t+4.32` | 322.0 | 7.7 | −89.6% | −78.8% |

Bertozzi 1964 cross-check (relativity side-experiment, same session):
classical predicts superluminal electrons at all 5 runs (1.40c–7.66c);
relativistic form matches to ~3%. D-T Q-value recomputed: 17.589 vs 17.6.

## 2. Rolling-origin, USA (6 windows; MAE | NMAE% | rank)

Quadratic ranks 1,2,2,2,2,1 — best 2/6, second 4/6, worst 0/6 (corrected
from an earlier "4/6" miscount). Worst-case NMAE: quad 13.5%, logistic
30.8%, Gompertz 32.8%. Gompertz fit-fails 1/6 windows.

## 3. Cross-country pilot (9 countries × 3 windows, OWID UN-era data)

No universal winner. France: logistic at 0.3% NMAE (saturation-shaped
trajectory fits saturation assumptions). UK: exponential takes later
windows (steady-growth segments — not a claim about inherent growth).
Nigeria: quadratic wins; Gompertz fails 2/3 fits. Japan: quadratic takes
the stagnation window. China: quadratic early → logistic later
(structural transition). Brazil: logistic sweep. USA/Germany/India: mixed.

## 4. Full sweep (785 country-windows, 265 countries)

Harness: `/tmp/opencode/pop_sweep.py` (repo-external scratch; promote if
rerun needed). Models: logistic, Gompertz, quadratic, exponential.
Windows: train ≤1980→test 1981–2000; ≤1990→1991–2010; ≤2000→2001–2023.

- Wins by NMAE rank-1: quadratic 315 (40%), logistic 245 (31%),
  exponential 150 (19%), Gompertz 75 (10%). Quadratic leads all 3 windows.
- Fit failures: Gompertz 35.3% (277/785); others 0.0%.
- Per-row records: MAE, RMSE, NMAE, max-abs, max-pct, bias, endpoint
  pred-vs-actual, n-train, n-test (available in harness output).

## 5. Standing interpretation (adopted)

"A comparatively stable candidate historical extrapolator over the
tested windows; not established as a general population law."
Regime determines structure; the protocol identifies which. Fit,
extrapolation, constraints, interpretability, and status are separate
scores — never collapsed into one winner.

## 6. Caveats

- Census transcription is secondary (primary paywalled); deciding signals
  are robust to it, cite accordingly.
- NMAE comparability across eras is imperfect; absolute errors recorded.
- Quadratic has no saturation mechanism; unbounded long-horizon growth is
  a scope limit, not a domain violation.
- Structural-recurrence test open: whether the engine independently
  rediscovers per-regime forms (fixed-form fitting done here instead).

## 7. Resume entry points

- `/tmp/opencode/pop_sweep.py` — full harness (records carry all metrics)
- `/tmp/opencode/pop_owid.csv` — OWID source (58k rows, re-downloadable)
- Open: per-regime discovery runs; digitized θ(r)/T00(r) warp profiles;
  Natario-vs-continuity template test.
