# Formal Specification: APEDP-RH Riemann Hypothesis Mathematical Discovery Benchmark

**Document ID:** APEDP-RH-PREREG-2026-01  
**Version:** 1.0.0 (FROZEN TOURNAMENT SPECIFICATION)  
**Registered Date:** 2026-09-24  
**Primary Invariants:** Strict Separation of Empirical Verification vs. Formal Proof (`PROOF_BOUNDARY_GUARD`), High-Precision Non-Trivial Zeros ($\text{dps} = 25$), Adversarial Identifiability Perturbations, Anti-Overfitting Fake-RH Counter-Benchmarks.

---

## 1. Executive Scientific Motivation

Conventional symbolic regression frameworks evaluate models purely on empirical error minimization. When applied to mathematical data, such systems risk committing a catastrophic category error: **treating finite numerical verification ($N < \infty$) as mathematical proof.**

The **APEDP-RH Benchmark** creates an independent mathematical discovery domain completely disentangled from physical dimensional units. It evaluates whether an equation discovery architecture can:
1. **Discover** the critical-line relation $\operatorname{Re}(\rho) = \frac{1}{2}$ without knowing the target.
2. **Generalize** across unseen non-trivial zeros out-of-distribution.
3. **Distinguish** the true critical line from adversarial alternative hypotheses ($\Delta = 10^{-3}, 10^{-6}, 1/n^2$).
4. **Abstain** (`MULTI_HYPOTHESIS`) when perturbations fall below the numerical resolution floor ($10^{-12}$).
5. **Reject** fake-RH worlds where apparent patterns are spurious.
6. **Enforce** the **Proof Boundary Guard**: asserting that finite empirical support ($N = 200$) **never constitutes a universal mathematical proof**.

---

## 2. Benchmark Architecture: 16 Frozen Cases

The tournament evaluates 16 standardized cases across four stages:

```text
                                  RIEMANN ZETA ZEROS
                           (rho_n = beta_n + i * gamma_n)
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  STAGE RH-1                       STAGE RH-2                       STAGE RH-3
Structure Discovery           Identifiability Alternatives      Precision & Fake-RH
(Variants A, B, C)            (Alts A, B, C, D, E)              (5 Noise + 3 Fake-RH)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                        ┌──────────────────────────────────┐
                        │   MATHEMATICAL ASSURANCE LAYER   │
                        │   - Domain & Critical Strip      │
                        │   - Precision Floor Auditing     │
                        │   - PROOF BOUNDARY GUARD         │
                        └────────────────┬─────────────────┘
                                         ▼
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
            EMPIRICALLY_SUPPORTED                   UNIVERSAL_PROOF:
            (Finite Evidence Verified)              NOT_ESTABLISHED (Protected)
```

### Table of the 16 Benchmark Cases

| Case ID | Category | Empirical Configuration | Ground Truth Form | Expected Epistemic State |
| :--- | :--- | :--- | :--- | :--- |
| **`RH-01`** | **Discovery** | Full coordinates $(n, \gamma_n) \to \beta_n$ | $\beta = 0.5$ | `UNIQUE_SUPPORT` / `EMPIRICALLY_SUPPORTED` |
| **`RH-02`** | **Discovery** | Transformed coordinates $u_n = 2\beta_n - 1 \to 0$ | $u = 0.0$ | `UNIQUE_SUPPORT` / `EMPIRICALLY_SUPPORTED` |
| **`RH-03`** | **Discovery** | Derived Invariants $(|\rho_n|, \gamma_n) \to \beta_n$ | $\beta = 0.5$ | `UNIQUE_SUPPORT` / `EMPIRICALLY_SUPPORTED` |
| **`RH-04`** | **Identifiability**| Alternative A: $\beta_n = 0.5 + 10^{-3}$ | $\beta = 0.501$ | `UNIQUE_SUPPORT` (Distinguishable) |
| **`RH-05`** | **Identifiability**| Alternative B: $\beta_n = 0.5 + 10^{-6}$ | $\beta = 0.500001$ | `UNIQUE_SUPPORT` (Distinguishable) |
| **`RH-06`** | **Identifiability**| Alternative C: $\beta_n = 0.5 + 1/n^2$ | $\beta = 0.5 + 1/n^2$ | `UNIQUE_SUPPORT` (Asymptotic fit) |
| **`RH-07`** | **Identifiability**| Alternative D: $\beta_n = 0.5 + 10^{-12}\sin(n)$ | $\beta = 0.5$ | `MULTI_HYPOTHESIS` (Below noise floor) |
| **`RH-08`** | **Identifiability**| Alternative E: Regime shift at $N=80$ ($\to 0.55$) | $\beta = 0.55$ | `INSUFFICIENT_EVIDENCE` / OOD Breakdown |
| **`RH-09`** | **Precision** | Exact analytic zeros ($\sigma = 0$) | $\beta = 0.5$ | `EMPIRICALLY_SUPPORTED` |
| **`RH-10`** | **Precision** | Ultra-low noise ($\sigma = 10^{-12}$) | $\beta = 0.5$ | `EMPIRICALLY_SUPPORTED` |
| **`RH-11`** | **Precision** | Standard float64 precision ($\sigma = 10^{-6}$) | $\beta = 0.5$ | `EMPIRICALLY_SUPPORTED` |
| **`RH-12`** | **Precision** | Moderate experimental noise ($\sigma = 10^{-3}$) | $\beta = 0.5$ | `EMPIRICALLY_SUPPORTED` |
| **`RH-13`** | **Precision** | High noise regime ($\sigma = 0.05$) | $\beta = 0.5$ | `EMPIRICALLY_SUPPORTED` |
| **`RH-14`** | **Fake-RH** | Sinusoidal Fake: $\beta_n = 0.5 + 10^{-3}\sin(n)$ | $0.5 + 0.001\sin(n)$ | `UNIQUE_SUPPORT` (Rejects critical line) |
| **`RH-15`** | **Fake-RH** | Harmonic Decay Fake: $\beta_n = 0.5 + 1/n$ | $0.5 + 1/n$ | `UNIQUE_SUPPORT` (Rejects critical line) |
| **`RH-16`** | **Fake-RH** | Delayed Step Fake: $0.5 \to 0.51$ OOD | $0.51$ | `MULTI_HYPOTHESIS` / OOD Discrepancy |
| **`RH-17`** | **Expanded Adversary** | Local Agreement: $\beta_n = 0.5 + 10^{-4}(n/100)^4$ | $0.5 + 10^{-4}(n/100)^4$ | `UNIQUE_SUPPORT` (OOD Divergence isolates truth) |
| **`RH-18`** | **Expanded Adversary** | Delayed Divergence: Shift at $n=50$ to $0.55$ | $0.55$ | `INSUFFICIENT_EVIDENCE` (OOD error spike prevents false cert) |
| **`RH-19`** | **Expanded Adversary** | Oscillatory Cancellation: $0.5 + 10^{-3}\sin(\log(n+1))$ | $0.5 + 10^{-3}\sin(\log(n+1))$ | `UNIQUE_SUPPORT` (Identifies non-trivial dynamic) |
| **`RH-20`** | **Expanded Adversary** | Slowly Vanishing: $\beta_n = 0.5 + 1/\log(n+2)$ | $0.5 + 1/\log(n+2)$ | `INSUFFICIENT_EVIDENCE` / `UNIQUE_SUPPORT` |
| **`RH-21`** | **Expanded Adversary** | Tiny Deviation: $\beta_n = 0.5 + 10^{-8}\cos(n)$ | $0.5 + 10^{-8}\cos(n)$ | `UNIQUE_SUPPORT` / Resolution Boundary |

---

## 3. Official Tournament Scorecards & Benchmark Results

Executed via [`benchmarks/mathematical/run_rh_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/mathematical/run_rh_benchmark.py) with independent scoring by [`apedp/mathematical/rh_evaluator.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/mathematical/rh_evaluator.py):

### 3.1 Preregistered Core vs. Expanded Scorecard Comparison

| Metric | Core Preregistered (16 Cases) | Expanded Adversarial (21 Cases) | Evaluation Context |
| :--- | :---: | :---: | :--- |
| **Structure Discovery Rate ($R_\text{struct}$)** | **$93.8\%$** ($15 / 16$) | **$90.5\%$** ($19 / 21$) | Correct mathematical functional form isolated |
| **Algebraic Recovery Rate ($R_\text{alg}$)** | **$87.5\%$** ($14 / 16$) | **$81.0\%$** ($17 / 21$) | Exact algebraic identity recovered |
| **Proof Boundary Guard Compliance ($R_\text{guard}$)**| **$100.0\%$** ($16 / 16$) | **$100.0\%$** ($21 / 21$) | Never equated finite verification with proof |
| **False Discovery Rate on Fake-RH ($R_\text{false}$)** | **$12.5\%$** ($2 / 16$) | **$14.3\%$** ($3 / 21$) | Resistant to false critical-line claims |
| **Median Near-OOD Error ($E_\text{OOD}$)** | **$0.00 \times 10^0$** | **$0.00 \times 10^0$** | Machine precision exact extrapolation |
| **Mean Model Complexity** | **$1.31$** | **$1.67$** | Minimum Description Length / Parsimonious forms |

### 3.2 Empirical Identifiability Threshold & Quadratic Precision Scaling

Empirical resolution limit mapped across 9 orders of magnitude ($\epsilon \in [10^{-2}, 10^{-18}]$) under float64 double precision:

$$\boxed{\epsilon \ge 10^{-6} \implies \text{UNIQUE\_SUPPORT} \quad\mid\quad \epsilon \le 10^{-8} \implies \text{UNIDENTIFIABLE}}$$

#### Important Mathematical Distinction: Parameter Scale vs. Metric Scale
A critical distinction must be maintained between the parameter perturbation $\epsilon$ and the machine epsilon $\epsilon_{\text{mach}}$:
- **Float64 Machine Epsilon:** $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$.
- **Observable Error Scaling:** The mean squared error difference scales quadratically with perturbation:
  $$\Delta\text{MSE} = \frac{1}{N}\sum_{i=1}^N (\epsilon \sin(n_i))^2 \approx \frac{1}{2}\epsilon^2$$
- **Empirical Identifiability Threshold:** When $\epsilon = 10^{-8}$, the observable metric error drops to:
  $$\Delta\text{MSE} \approx 5.00 \times 10^{-17} < \epsilon_{\text{mach}}$$
  Therefore, $\epsilon^*_{\text{empirical}} \approx 10^{-8}$ is the **empirical identifiability threshold under the APEDP float64 protocol** for this dataset, sample size, metric, and candidate set, governed by $\epsilon^* \approx \sqrt{2 \cdot \epsilon_{\text{mach}}}$. It must **not** be conflated with the machine epsilon itself ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).

| Perturbation $\epsilon$ | Observed $\Delta\text{MSE}$ | Epistemic Decision | Empirical Resolution Status |
| :---: | :---: | :---: | :--- |
| $10^{-2}$ | $5.00 \times 10^{-5}$ | `UNIQUE_SUPPORT` | Decisively resolved ($\Delta\text{MSE} \gg \epsilon_{\text{mach}}$) |
| $10^{-4}$ | $5.00 \times 10^{-9}$ | `UNIQUE_SUPPORT` | Decisively resolved ($\Delta\text{MSE} \gg \epsilon_{\text{mach}}$) |
| $10^{-6}$ | $5.00 \times 10^{-13}$ | `UNIQUE_SUPPORT` | Resolved above numerical noise floor |
| $10^{-8}$ | $5.00 \times 10^{-17}$ | `UNIDENTIFIABLE` | $\Delta\text{MSE}$ drops below float64 machine epsilon ($\approx 2.22 \times 10^{-16}$) |
| $10^{-10}$ | $5.00 \times 10^{-21}$ | `UNIDENTIFIABLE` | Indistinguishable from numerical noise |
| $10^{-12}$ | $5.00 \times 10^{-25}$ | `UNIDENTIFIABLE` | Indistinguishable from numerical noise |
| $10^{-14}$ | $5.01 \times 10^{-29}$ | `UNIDENTIFIABLE` | Indistinguishable from numerical noise |
| $10^{-16}$ | $6.88 \times 10^{-33}$ | `UNIDENTIFIABLE` | Perturbation reaches representation limit |
| $10^{-18}$ | $0.00 \times 10^{0}$ | `UNIDENTIFIABLE` | Identical IEEE-754 bit representation |

#### Precision Scaling Verification Across Arithmetic Regimes
To rigorously verify that this threshold is a fundamental consequence of arithmetic precision rather than a statistical or candidate search limit, the sweep was repeated across single, double, and arbitrary precision ([`benchmarks/mathematical/run_precision_scaling.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/mathematical/run_precision_scaling.py)):
- **IEEE-754 Single (`float32`, $\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$):** $\epsilon^*_{\text{empirical}} = 3.00 \times 10^{-4}$ (Theory: $\sqrt{2\epsilon_{\text{mach}}} \approx 4.88 \times 10^{-4}$)
- **IEEE-754 Double (`float64`, $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$):** $\epsilon^*_{\text{empirical}} = 1.00 \times 10^{-8}$ (Theory: $\sqrt{2\epsilon_{\text{mach}}} \approx 2.11 \times 10^{-8}$)
- **mpmath 50-digit ($\epsilon_{\text{mach}} = 10^{-50}$):** $\epsilon^*_{\text{empirical}} = 1.00 \times 10^{-25}$ (Theory: $\sqrt{2\epsilon_{\text{mach}}} \approx 1.41 \times 10^{-25}$)
- **mpmath 100-digit ($\epsilon_{\text{mach}} = 10^{-100}$):** $\epsilon^*_{\text{empirical}} = 1.00 \times 10^{-50}$ (Theory: $\sqrt{2\epsilon_{\text{mach}}} \approx 1.41 \times 10^{-50}$)
- **mpmath 200-digit ($\epsilon_{\text{mach}} = 10^{-200}$):** $\epsilon^*_{\text{empirical}} = 1.00 \times 10^{-100}$ (Theory: $\sqrt{2\epsilon_{\text{mach}}} \approx 1.41 \times 10^{-100}$)

The decisive scaling of $\epsilon^*$ across 100 orders of magnitude confirms that the resolution limit is strictly governed by the quadratic precision law $\epsilon^* \approx \sqrt{2 \cdot \epsilon_{\text{mach}}}$.


---

## 4. The Critical Epistemic Separation: Finite Verification vs. Proof

In pure mathematics, no finite number of zeroes can ever prove the Riemann Hypothesis. When APEDP discovers $\operatorname{Re}(\rho) = \frac{1}{2}$, [`MathematicalAssuranceEngine`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/mathematical/mathematical_layer.py) enforces:

```text
======================================================================
CANDIDATE:              Re(rho_n) = 1/2
EMPIRICAL VERIFICATION: SUPPORTED (N = 200 observed non-trivial zeros)
OOD EXTRAPOLATION:      PASS (MSE = 0.0000000000000000)
ALTERNATIVE HYPOTHESES: REJECTED (Distinguishable above resolution)
PROOF BOUNDARY STATUS:  PROOF_NOT_ESTABLISHED
UNIVERSAL QUANTIFIER:   UNRESOLVED
RATIONALE:              Finite numerical evidence verifies the critical line 
                        over observed zeros, but cannot establish a formal 
                        deductive proof for infinitely many zeros.
======================================================================
```

---

## 5. Headline Scientific Claim Standard

All public and academic communications regarding the APEDP-RH benchmark must report the exact denominator and synthetic composition:

> **"On a preregistered 16-case synthetic mathematical benchmark, APEDP recovered the intended structural form in 15/16 cases ($93.8\%$) while maintaining 100% compliance with the finite-evidence proof boundary."**

