# Publication Data Discovery Findings

Date: 2026-09-22

## Scope

These experiments test whether APEDP can generate equations from publication-backed or literature-derived datasets. They are empirical discovery demonstrations, not proofs of new physical laws.

Every result below uses a train/test or train/holdout split. A candidate is only called better when its locked test error beats the stated baseline. Novelty remains corpus-relative.

## Summary

| Dataset | Generated result | Test outcome | Scientific interpretation |
|---|---|---|---|
| Bertozzi 1964 electron energy | Novel-looking nonlinear candidates | Best candidate improved over classical but lost to relativistic physics | Hypothesis generation; no new law |
| Brown lifter | `lift = 18.30445 * I_mA` | Stable across five seeds; holdout MSE `0.000396` | Strong empirical scaling rediscovery; classified as known/structural |
| NACA 0012 transonic Cp | `Cp = -5.0461 M^2 + 5.6633 M - 2.1693` | Test MSE `0.00930` vs classical `0.02488` | Better local empirical approximation; not a universal aerodynamic law |
| Nanocrystalline copper | Constant candidate after evolutionary stagnation | Failed to infer inverse-Hall–Petch turnover | Correct negative result: turnover was absent from training regime |
| AME2020 nuclear masses | Refit SEMF coefficients | Held-out RMSE `0.06445 MeV/nucleon` | Known-model parameter refit, not novel equation discovery |

## 1. Bertozzi 1964

**Source:** W. Bertozzi, *American Journal of Physics* 32 (1964), 551–555; values stored through a secondary table.

**Dataset:** Five kinetic-energy/electron-velocity observations.

**Protocol:** Four points for discovery; final point locked as holdout; three evolutionary seeds.

**Known baselines:**

- Classical: `v/c = sqrt(2*K/m)`
- Relativistic: `v/c = sqrt(1 - 1/(1 + K/m)^2)` with `m = 0.511 MeV`

**Generated candidates:**

- Seed 1/3: `-0.0205*K^2 + 0.1332*K + 0.8030`; holdout error `2.8128`
- Seed 2: `1.0266 - 0.03415*exp(K**(-2/3))`; holdout error `0.01365`

**Holdout comparison:**

- Classical error: `6.6621`
- Relativistic error: `0.00054`
- Best generated candidate: `0.01365`

**Finding:** APEDP generated apparently novel formulas, but none beat the known relativistic law. The polynomial candidates fit training data and extrapolated catastrophically, demonstrating why locked holdouts are mandatory.

## 2. Brown Lifter

**Source:** Ianconescu et al., *Journal of Electrostatics* 69 (2011), 512–521; arXiv:1011.1393.

**Dataset:** Ten calculated-table rows with voltage, current, and lift.

**Protocol:** Seven rows for discovery; final three rows locked as holdout; five seeds.

**Generated candidate:**

```text
lift_g = 18.30445 * I_mA
```

All five seeds produced the same expression.

**Results:**

- Candidate holdout MSE: `0.000396`
- Fitted `V*I` baseline holdout MSE: `12.5235`
- Constant baseline holdout MSE: `63.8970`

**Finding:** The candidate generalized strongly, but it was not novel according to the current corpus audit. The table is partly calculated and limited in geometry, so the result is a scaling-law rediscovery rather than a new physical law.

## 3. NACA 0012 Transonic Pressure Data

**Source:** NASA Langley / AGARD-AR-138 curated measurements at chord station `x/c = 0.25`.

**Dataset:** Eleven Mach-number/pressure-coefficient observations with measurement uncertainties.

Dataset hash:

```text
6c704e8311ea6e6bdfe354903b273f61b10b5adbd72286740b4de085f87d5e4e
```

**Protocol:** Eight points for training; final three points locked as test; uncertainty-weighted training score; three seeds.

**Generated equation:**

```text
Cp = -5.0461*M**2 + 5.6633*M - 2.1693
```

All three seeds converged to essentially the same expression.

**Test results:**

- Generated quadratic test MSE: `0.00930`
- Fitted classical Prandtl–Glauert test MSE: `0.02488`
- Improvement: approximately `62.6%` lower test MSE

**Finding:** APEDP generated a stable empirical equation that outperformed the classical baseline on the locked test points. It remains a local approximation for this airfoil condition and does not establish a universal transonic law.

## 4. Nanocrystalline Copper Microcompression

**Source:** Greer/Uchic experimental microcompression literature compilation, covering electrodeposited nanocrystalline copper.

**Dataset:** Eleven grain-size/yield-strength observations from `3 nm` to `50 um`, including measurement uncertainties.

**Protocol:** Eight larger-grain points for training; three smaller-grain points locked as holdout.

**Baselines:**

- Classical Hall–Petch
- Quadratic model in log grain size
- Constant training mean

**Holdout MSE:**

- Classical Hall–Petch: `888,785`
- Quadratic log-size model: `185,263`
- Constant: `75,764`

The evolutionary search returned the constant `526.875` for completed seeds and marked the search as stagnated.

**Finding:** The training partition contained only a monotonic Hall–Petch branch. The inverse-Hall–Petch turnover appeared only in the holdout. APEDP correctly reports this as underdetermined rather than pretending to discover a turnover equation from absent evidence.

The regime diagnostic returns:

```text
UNDERDETERMINED: no observed slope reversal supports a new regime outside the measured domain
```

When the full dataset is used, the observed slope reversal is detected and the regime is classified as covered.

## 5. AME2020 Nuclear Masses

**Source:** Huang/Wang et al., *Chinese Physics C* 45 (2021), AME2020 mass evaluation.

**Dataset:** `2,326` experimental nuclei after parsing; `1,230` extrapolated rows excluded.

**Protocol:** `1,812` nuclei for training and `454` held out; nuclei restricted to `A >= 16`.

The refit used the known Semi-Empirical Mass Formula structure:

```text
B = av*A - as*A**(2/3) - ac*Z**2/A**(1/3)
    - aa*(A - 2*Z)**2/A + pair_scale*delta
```

Fitted coefficients:

```text
av         = 15.4012
as         = 16.5160
ac         = 0.69416
aa         = 22.5972
pair_scale = 0.93850
```

**Held-out RMSE:**

- Literature coefficients: `0.11836 MeV/nucleon`
- Refit SEMF: `0.06445 MeV/nucleon`
- SEMF plus simple shell features: `0.06868 MeV/nucleon`

**Finding:** Parameter fitting improves prediction, but this is a refit of a known equation family, not new equation discovery. The shell extension did not improve this random split.

## Overall Conclusions

1. APEDP can generate stable empirical equations that beat simple baselines on locked test data.
2. It can also correctly refuse or downgrade a search when the training regime does not contain the phenomenon being claimed.
3. Novelty audits are corpus-relative and sometimes classify useful empirical forms as apparently novel or map simple scalings to broad structural references.
4. Publication data quality matters: calculated tables, curated tables, and primary experimental measurements must not be treated as equivalent evidence.
5. The strongest remaining requirement is independent multi-condition validation: different runs, geometries, laboratories, Reynolds numbers, materials, or physical regimes.
6. None of these experiments establishes a new physical law. They demonstrate equation generation, refitting, extrapolation failure detection, and evidence-aware refusal.
