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

### Gate 4 — Matched-Compute Ablation (RIME enumeration vs native GA)

| Task / Method | Holdout/Train R² | Wall time | Detail |
| :--- | :---: | :---: | :--- |
| Rabi: RIME enumeration | `1.000000` | `37.98 ms` | `-1.0000 * (cos(t)^2) + 1.0000` |
| Rabi: native GA | `1.000000` | `21.89 ms` | `-1.0879 * ((cos(sin(t)) + -1.0776) - cos(cos(t))) + -0.6723` (1560 evals) |
| Brus: RIME enumeration (train) | `0.998414` | `21.58 ms` | `2.0654 * (1 / (R^2)) + 1.7003` |
| Brus: native GA (holdout) | `-2.925069` | `6.16 ms` | `2.2645 * ((t**2.2004)**-1.0) + 1.7176` (1560 evals) |

### Gate 5 — Sealed Confirmation (fresh seed 12345, frozen configs)

| Landauer 2% still quantized | **PASS** |
| Schwinger still exact | **PASS** |
| Planck still not exact | **PASS** |

---

**Overall: ALL GATES PASSED**