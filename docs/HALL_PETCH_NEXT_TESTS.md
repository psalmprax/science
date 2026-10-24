# Hall-Petch Next Tests — Research Protocol

Status: L3 established (held-out extrapolation validation). L4–L8 open.
Source data: `NanocrystallineMicrocompressionBenchmark.get_dataset()`
(`apedp/ingestion/external_benchmarks.py:246`), n=11 Cu micropillar,
3nm–50um, turnover 12nm→3nm. Prior findings: `docs/SEMF_AME2020_FINDINGS.md`,
session logs 2026-09-22.

## Established (L3)

- Train: d>=20nm (7 pts). Locked holdout: 12/8/5/3nm (4 pts).
- Full RMSE: classical 239.21 vs regul 40.34 MPa. Holdout RMSE:
  classical 1068.77 vs regul 137.77 MPa (7.76x / 87.1% lower).
- Per-point holdout errs (class/regul MPa): 12nm +292/−64,
  8nm +592/−8, 5nm +1080/+114, 3nm +1723/+242.
- Full params (lit-seeded single start): regul
  s0=97.40MPa, k=0.25685 MPa·m^0.5, dc=22.46nm, q=0.966.
  Train-only params: s0=59.52MPa, k=0.38233, dc=66.61nm, q=0.6522.
  Both q>0.5 → finite small-d limit = s0 vs classical →INF
  (under fitted forms, not an experimental fact at d→0).
- BIC 429.2 vs 394.9 (Δ≈34.4, conditional on Gaussian iid + small n).
- 2σ coverage regul 1.000 vs class 0.091 (empirical coverage only).
- Gate probe on fitted point: contract coverage 0.727 REFUTED —
  statistical validity ≠ physical validity. `CONTRACT_VALIDATED`
  NOT claimed for fitted params (literature record HP-EXP-001 passes
  per `tests/test_external_benchmarks.py:143`).
- Provenance: `PHYSICAL MODEL sigma=s0+(k/sqrt(d))/(1+(dc/d)^q)` SI;
  `max(d,1e-11)` safeguard + `min(sigma,1.5GPa)` cap never active
  (min data 3e-9 m, max regul pred <1.0 GPa).
- Single literature-informed start `p0=[90e6,0.27e6,25e-9,0.95]`
  (+bounds) was used → independent rediscovery NOT established.
- Claim matrix: Prediction PASS / Physics NOT ESTABLISHED.

## Caveat (applies to all three tests below)

Coarse-grain training data (d>=20nm) alone may not contain enough
information to identify a turnover scale. A correct result may therefore
be `REFUSED` / "turnover not identifiable" rather than a forced discovery.
Absence of turnover evidence from classical-regime data is itself informative.

## Test 1 — Random-start fitting (robustness to initialization)

- Data: train d>=20nm only (7 pts). Holdout locked (12/8/5/3nm).
- Form: supplied regul structure (literature-assisted track).
- Starts: N=50 random initializations, log-uniform physically valid
  bounds: s0∈[10,1000]MPa, k∈[0.01,2.0] MPa·m^0.5,
  dc∈[1,200]nm, q∈[0.1,3.0]. Fixed seeds for reproducibility.
- Fit: `scipy.optimize.curve_fit`, same bounds, maxfev 50000.
- Record per seed: converged flag, (s0,k,dc,q), train RMSE, test RMSE.
- Stability criteria: distributions (median/IQR) of dc, q; fraction with
  q>0.5; fraction within lit-neighborhood
  (s0 90–100MPa, k 0.25–0.27, dc 20–30nm, q 0.9–1.1) vs train-only
  neighborhood (s0~60, dc~67nm, q~0.65); test-RMSE distribution.
- Pass interpretation: parameter region is an attractor of
  data-constrained optimization, not just the lit-seeded start.

## Test 2 — Leave-one-out / bootstrap (small-n stability)

- LOO-A (full-data stability): 11 fits, leave one of 11 out, refit regul
  (same lit-seeded protocol for comparability + optionally best random
  protocol from Test 1). Report per-omission (dc, q, full RMSE).
- LOO-B (holdout stability): 7 fits, leave one of train-7 out, refit on 6,
  evaluate on locked 4-pt holdout. Report test-RMSE distribution.
- Perturbation bootstrap: resample each sigma within its stated
  measurement uncertainty (Gaussian, per-point unc), 50 reps, refit,
  report dc/q/test-RMSE spreads.
- Question answered: did APEDP discover turnover structure, or did four
  parameters exploit a handful of points? Stable q>0.5 + bounded dc
  spread + holdout RMSE consistently << classical (~1069 MPa) supports
  structure; wild swings support caution.

## Test 3 — Uninformed structural discovery (Track C)

- Inputs ONLY: {d (meter), sigma (pascal)}, SI units, positivity,
  locked holdout (12/8/5/3nm). DO NOT supply the turnover equation,
  its parameters, or lit values.
- Grammar: generic operators (+,−,×,÷,sqrt,log,exp,powers) +
  complexity penalty (BIC) + `check_data_adequacy` gate +
  asymptotic/domain checks.
- Baseline implementation: `apedp.discovery.bic_forward_select` over
  generic columns (d, 1/d, sqrt(d), 1/sqrt(d), log(d), d^2, ...)
  on train d>=20nm; evaluate winner on locked holdout vs classical.
- Success: system independently finds a turnover-type structure
  (e.g. equivalent to `s0+(k/sqrt(d))/(1+(dc/d)^q)` or any form with
  a characteristic scale + finite small-d limit) that survives the
  locked holdout. Alternative turnover forms also count.
- Valid negative: `REFUSED` / "no feature improves on intercept-only BIC"
  / "turnover not identifiable from classical-regime data" — record as
  such, do not force a fit.

## Repro (single-start reference)

```bash
PYTHONPATH=. python3 -c "
from scipy.optimize import curve_fit
from apedp.ingestion import NanocrystallineMicrocompressionBenchmark as HP
import numpy as np
d,s,u,_=HP.get_dataset(); d=d.ravel()
def regul(dv,s0,k,dc,q): return np.minimum(s0+(k/np.sqrt(np.maximum(dv,1e-11)+1e-10))/(1.0+(dc/np.maximum(dv,1e-11))**q),1.5e9)
tr=slice(0,7)
print(curve_fit(regul,d[tr],s[tr],p0=[90e6,0.27e6,25e-9,0.95],bounds=([0,0,1e-9,0.1],[3e9,3e6,200e-9,3.0]),maxfev=50000)[0])
"
```

Evidence ladder: current L3 (held-out extrapolation). Tests 1–2 target L5
(discovery robustness); Test 3 targets true structural discovery. L6–L8
(cross-material, mechanistic, experimental) remain open.

## Results (measured 2026-09-22, system python3 + scipy 1.17.1)

### Test 1 — Random-start (N=50, log-uniform uninformed starts, train d>=20nm)

- Converged 50/50. Medians: s0=59.5MPa, k=0.382, dc=66.6nm, q=0.652
  (IQRs tight; one outlier dc=1.9nm, testRMSE 633MPa).
- frac q>0.5 = 0.98. TestRMSE med 137.8, IQR [137.8,137.9],
  min 137.8 max 633.2 MPa. 100% < classical (1068.8), 98% < 200MPa.
- 0% in LIT-neighborhood (90–100/0.25–0.27/20–30nm/0.9–1.1);
  98% in train-only basin (~60/0.38/67nm/0.65).
- Reading: parameter region is an attractor of data-constrained
  optimization given the form (not just the lit start), but the attractor
  is the train-only basin, NOT the lit basin. Best-test = lit-seeded point.

### Test 2 — LOO / bootstrap

- LOO-A (leave-1-of-11): dc 13.6–28.8nm, q 0.841–1.147, 11/11 q>0.5.
  Most sensitive: omit 200nm → dc 13.6/q 1.147; omit 3nm → dc 28.8/q 0.841.
- LOO-B (leave-1-of-train7, locked 4-pt test): dc 39.3–80.2nm,
  q 0.624–0.970, testRMSE 74.4–248.2MPa — all << classical 1068.8.
  Worst = omit boundary 20nm (248.2); best = omit 50nm (74.4).
- Bootstrap (50 reps, per-point Gaussian unc): dc med 20.8
  IQR [17.5,25.7] range [12.5,48.9]; q med 1.000 IQR [0.935,1.087],
  100% >0.5.
- Reading: qualitative turnover (q>0.5) survives every perturbation;
  exact dc is unstable (13–80nm across splits). Structure robust,
  scale not pinned — small-n caveat retained.

### Test 3 — Uninformed grammar (bic_forward_select, NO turnover form)

- Adequacy gate on coarse train: OK (range ratio 2500, n=7).
- Winner (5 terms + intercept, BIC 232.95): log(d), d, d^2,
  1/sqrt(d), sqrt(d) with coefficients up to 1e16 (massive
  cancellations, 6 params on 7 pts).
- Holdout RMSE 78.5MPa (beats regul 137.8 numerically), per-point
  916/901/850/744 vs true 980/920/780/620.
- Reading: statistical win, PHYSICAL FAIL (L1). Absurd magnitudes,
  no asymptote reasoning, uninterpretable soup — Track C with naive
  grammar does NOT discover turnover structure. Correct APEDP response
  is REFUSE-at-physics / demand admissibility (dimensions, magnitudes,
  asymptote audit), exactly the statistical≠physical separation.
  Coarse-train-only identifiability of a turnover scale is not
  demonstrated without the structural prior.

## Synthesis — refusal as a feature (review 2026-09-22)

Separated discovery modes and their information supply:

| Discovery mode       | Information supplied                                    | Outcome                       |
| -------------------- | ------------------------------------------------------- | ----------------------------- |
| Classical Hall–Petch | Classical structure                                     | Poor nanoscale extrapolation  |
| Regularized model    | Turnover structure + literature-informed initialization | Strong holdout prediction     |
| Uninformed discovery | Coarse data + generic grammar only                      | Refusal / insufficient identifiability |

Conclusion is NOT "APEDP failed to discover the equation." It is:
given only seven coarse-regime observations, the tested
structure-uninformed pipeline does not have sufficient evidence to
identify a physically valid turnover structure. Wording caveat: this
binds the claim to the tested procedure (seven points insufficient for
*this* search procedure), not to seven points being fundamentally
insufficient under all possible algorithms.

Core principle preserved:

```text
NO EVIDENCE ≠ INVENT A LAW
```

APEDP outcomes are tri-state: DISCOVER (evidence + assurance survive),
REFUSE (insufficient evidence / failed physical validation), CONTRADICT
(candidate explicitly contradicted by independent observations). Forcing
every dataset to yield an equation is an anti-goal; the 6-parameter
soup on 7 points (BIC 232.95, coef to 1e16) is the cautionary example —
impressive RMSE, not physics.

Deeper point demonstrated: prediction can be enabled by prior structure
that discovery cannot infer from the available regime. The supplied
`(dc/d)^q` form tells the system a characteristic suppression scale may
exist; the uninformed system was never given that, and the holdout
(12/8/5/3nm) stayed locked so structure selection could not see turnover.

User-reported companion results (not from this session's logs; recorded
as reported): generic symbolic runs showed severe extrapolation
instability (holdout RMSE range 680–31,243 MPa across seeds — evidence
of structural non-identifiability, stronger than "it failed"); the
family selector refused model-family selection on 7 observations vs its
configured minimum-8 safeguard. The minimum-8 rule is an algorithmic
safeguard, not a mathematical law — report it as such, alongside the
independent symbolic-run failure. Also noted: runs using
`grain_nm = d/1e-9` with dimensionless registration are
structure-uninformed *normalized* symbolic discovery, not
physics-aware dimensional discovery — label accordingly.

## Next program — three levels of prior

- Experiment A (no prior): {d, sigma} only → expect REFUSE (demonstrated).
- Experiment B (dimensional prior): d=length, sigma=pressure, valid
  operators, dimensional algebra, positivity/domain — NO turnover
  structure. Open question whether a valid turnover candidate emerges.
- Experiment C (structural prior): grammar allowing characteristic
  scales `1+(dc/d)^q`, but NOT the complete equation. Test for
  structural equivalence with the turnover form.

```text
NONE ────────────────→ REFUSE
DIMENSIONAL ─────────→ ?
STRUCTURAL ──────────→ ?
LITERATURE MODEL ────→ strong prediction
```

This quantifies how much prior information is required before the
hidden regime is recoverable — more scientific than maximizing accuracy.

## Paper paragraph (adopted)

Hall–Petch turnover benchmark. A classical Hall–Petch model and a
regularized turnover model were evaluated against a locked nanoscale
holdout. The regularized structure achieved substantially lower
extrapolation error. Random initialization, leave-one-out analysis, and
bootstrap resampling supported the robustness of the qualitative
finite-turnover behavior, although the characteristic transition scale
dc remained weakly identified. In a separate structure-uninformed
experiment, the discovery system was given only the seven coarse-regime
observations and a generic symbolic grammar. The family-selection stage
refused model selection because of its configured minimum sample
requirement, while independent symbolic runs failed to produce a stable
physically plausible turnover candidate. APEDP therefore refused to
infer a turnover law from the coarse data alone. This demonstrates the
distinction between prediction under a supplied structural prior and
structure discovery from insufficient evidence.

Architectural lesson: APEDP is an equation discovery + evidence
adjudication system (generator proposes; assurance checks; validation
tests; epistemic layer decides sufficiency). Sometimes the correct
output is REFUSE — NOT IDENTIFIABLE. That is what distinguishes it from
ordinary symbolic regression.
