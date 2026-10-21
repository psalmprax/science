# RIME-AD Five-Gate Scientific Validation Programme

---

### Gate 1 — Exact Structural Recovery

| Case | Outcome |
| :--- | :---: |
| Landauer staircase | **PASS** |
| Schwinger tunneling | **PASS** |
| Planck limitation correctly held | **PASS** |
| Brus confinement | **PASS** |
| Detuned Rabi resonance | **PASS** |
| Resonant Rabi dynamics | **PASS** |
| Landau-Zener transition | **PASS** |
| BCS energy gap | **PASS** |
| IQHE staircase | **PASS** |

### Gate 2 — Noise and Identifiability

| Landauer 2% noise quantized (step `0.9893`) | **PASS** |
| Landauer 5% noise quantized (step `0.9380`) | **PASS** |
| Schwinger 2% noise: no false exact claim (traj R² `-0.879`) | **PASS** |

### Gate 3 — Adversarial Controls

| Control | Abstained |
| :--- | :---: |
| Sine wave not called quantized | **PASS** |
| Sine wave yields no exact implicit relation | **PASS** |
| White noise yields no exact ODE | **PASS** |
| Null world yields no exact relation | **PASS** |

### Gate 4 — Matched-Compute Ablation (RIME vs Native GA on Identical Holdout Splits)

| Task / Representation | Holdout R² | Execution Time | Discovered Formula / Search Detail |
| :--- | :---: | :---: | :--- |
| Rabi: RIME shared DAG | `1.000000` | `22.92 ms` | `-1.0000 * (cos(t)^2) + 1.0000` |
| Rabi: Native GA | `0.999992` | `37.36 ms` | `18.8415 * besselj(sin(besselj(sin(t)))) + -15.6500` (3798 evals) |
| Brus: RIME single-term DAG | `-34.567941` | `14.78 ms` | `2.0654 * (1 / (R^2)) + 1.7003` (collapses on holdout) |
| Brus: RIME multi-rational | `1.000000` | `< 1.0 ms` | `1.7400 - 0.2800*(1/R) + 2.4500*(1/R^2)` (exact holdout recovery) |
| Brus: Native GA single tree | `-17.539540` | `4.64 ms` | `1.7533 * tan(((1.1112 / t) / t)) + 1.7062` (1845 evals, collapses on holdout) |

### Gate 5 — Sealed Confirmation (fresh seed 12345, frozen configs)

| Landauer 2% still quantized | **PASS** |
| Schwinger still exact | **PASS** |
| Planck still not exact | **PASS** |

---

**Overall: ALL GATES PASSED**