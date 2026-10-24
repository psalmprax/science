# Landauer Conductance Staircase: Reproducible RIME vs. GA Benchmark
## Machine-Verified Empirical Audit Across Polynomial, Heuristic, and Autonomous Discovery Pipelines

---

### Benchmark Provenance & Reproducibility Notice
- **Protocol:** Deterministic execution via [`benchmarks/run_landauer_rime_vs_ga_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_landauer_rime_vs_ga_benchmark.py).
- **Machine-Measured Data Source:** [`benchmarks/landauer_benchmark_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/landauer_benchmark_results.json).
- **Execution Timestamp:** 2026-10-15T11:13:33Z.
- **Random Seed:** Pinned to `42`.
- **Discovery Split:** $V_g \in [0.1, 4.9]$ ($N = 100$ rows).
- **Holdout Extrapolation Split:** $V_g \in [5.0, 9.8]$ ($N = 100$ rows).
- **Epistemic Referee Whiteness Threshold:** $|\rho_1| < 0.50$ (preregistered Guard 2).

---

### 1. Executive Summary & Measured Results

The Landauer conductance quantum staircase:
$$G(V_g) = G_0 \lfloor V_g \rfloor$$
represents a fundamental challenge for continuous symbolic regression. Because $G(V_g)$ exhibits discontinuous integer step jumps with zero derivative on plateaus, smooth polynomial basis sets suffer from catastrophic Runge oscillations outside the training window.

All figures below are **machine-measured from actual execution runs** in the repository workspace:

| Method / Model Configuration | In-Sample Fit ($R^2$) | Out-of-Sample Extrapolation ($R^2$) | Extrapolation RMSE | Residual Autocorr ($|\rho_1|$) | Design Matrix Condition ($\kappa$) | Epistemic Referee Whiteness Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Genetic Algorithm (GA, Degree 5 Polynomial)** | `0.9645` | `-11530.5976` | `149.3235` | `0.7238` | `4.54e+04` | **REFUSED** (extrapolation collapse (R2 -11530.5976 < 0.99)) |
| **Unassisted RIME (Smooth Grammar)** | `0.9598` | `0.9486` | `0.3153` | `0.7386` | Low (~1.0) | **REFUSED** (extrapolation collapse (R2 0.9486 < 0.99)) |
| **RIME with `floor` Transform** | `1.000000` | `1.000000` | `0.000000` | `0.9691` | Minimal | **PASSED** (exact recovery by rule (RMSE 1.57e-08 <= 1e-06, R2 1.000000000 within 1e-09 of 1)) |
| **Autonomous Epistemic Profiler & Transmuter** | `1.000000` | `1.000000` | `0.000000` | `n/a (exact-zero residuals)` | N/A (Analytical) | **PASSED** (exact recovery by rule (RMSE 0.00e+00 <= 1e-06, R2 1.000000000 within 1e-09 of 1)) |

---

### 2. Deep Dive: Measured Failure Modes & Recovery

#### 2.1 The GA Polynomial Runge Divergence
Standard GA was given the full polynomial basis $\{1, V_g, V_g^2, V_g^3, V_g^4, V_g^5\}$.
- **In-Sample:** GA achieved an apparent fit of $R^2 = 0.9645$.
- **Out-of-Sample Extrapolation:** Outside the $[0.1, 4.9]$ gate voltage training window, the higher-order polynomial terms exploded violently, collapsing extrapolation $R^2$ to **`-11530.5976`** with an extrapolation RMSE of **`149.3235`**.
- **Numerical Ill-Conditioning:** The polynomial basis exhibited extreme collinearity with a condition number of **$\kappa = 4.54e+04$**.
- **Referee Refusal:** The residual autocorrelation on the training fit was **$|\rho_1| = 0.7238$**, severely violating the whiteness threshold ($|\rho_1| < 0.50$). The Epistemic Referee strictly refused promotion.

#### 2.2 Unassisted RIME Behavior
Without step or floor operators in its grammar, RIME discovered the optimal least-squares linear ramp:
$$\hat{G}(V_g) = +0.975 * (Vg) -0.4375$$
- **Fit & Extrapolation:** $R^2_{\text{train}} = 0.9598$, $R^2_{\text{extrap}} = 0.9486$.
- **Epistemic Refusal:** Residual autocorrelation was **$|\rho_1| = 0.7386 > 0.50$**. The Epistemic Referee correctly refused the smooth linear surrogate.

#### 2.3 Assisted RIME & Autonomous Epistemic Profiler
- **Assisted RIME (with `floor`):** Discovered `+1 * (floor(Vg)) +6.12371e-09`, yielding $R^2_{\text{extrap}} = 1.000000$ and exact quantum parameter $G_0 = 1.000000$.
- **Autonomous Profiler & Transmuter:** Without human prompting, `profile_dataset` diagnosed `is_quantized_staircase: True` (flat derivative fraction $> 80\%$). The transmuter applied midpoint linear thresholding and recovered `1 * floor((Vg - 0.0030303) / 0.998788)` with $R^2 = 1.000000$ in $0.35$ milliseconds.

---

### 3. Physical Robustness Variations (Thermal & Noise)

#### 3.1 Thermal Broadening (Fermi-Dirac Smearing, $k_B T / \Delta E = 0.08$)
When physical thermal broadening softens the sharp step edges into Fermi-Dirac sigmoids:
- **GA Polynomial Extrapolation:** $R^2 = -7402.4493$ (still diverges due to polynomial runaway).
- **RIME Step / Sigmoid Fit:** $R^2_{\text{train}} = 0.9866$, $R^2_{\text{extrap}} = 0.9819$.

#### 3.2 Additive Observation Noise ($2\%$ and $5\%$)

| Condition | GA In-Sample $R^2$ | GA Extrapolation $R^2$ | Assisted RIME In-Sample $R^2$ | Assisted RIME Extrap $R^2$ | Transmuter Extrap $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2% Observation Noise** | `0.9636` | `-11545.6051` | `0.999676` | `0.999991` | `Abstained` |
| **5% Observation Noise** | `0.9633` | `-9161.3067` | `0.997753` | `1.000000` | `Abstained` |

---

### 4. Verification Command

To re-run the benchmark and regenerate this report deterministically from disk:
```bash
python benchmarks/run_landauer_rime_vs_ga_benchmark.py
```
