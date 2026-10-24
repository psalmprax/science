# Preregistration: APEDP 4.0 Autonomous Equation & Model Synthesis Protocol

**Document ID:** PREREG-APEDP-V4-SYNTHESIS-2026-09-25  
**Version:** 4.0.0-PROPOSED  
**Principal Investigator:** APEDP Core Scientific Team  
**Date:** September 25, 2026  
**Status:** PREREGISTERED & METHODOLOGICALLY FROZEN  

---

## 1. Executive Summary & Scientific Motivation

### 1.1 Beyond Symbolic Regression: The Synthesis Paradigm
Conventional symbolic regression operates under an **identification hypothesis**:
$$\text{Data } D \longrightarrow \text{Select existing law from known hypothesis family } f(x)$$

APEDP 4.0 departs from library lookup and template fitting to investigate **autonomous equation synthesis**:
$$\text{Data } D + \text{Physical Constraints } C + \text{Domain Knowledge } K \longrightarrow \text{Synthesize novel mathematical structure } f_{\text{novel}}(x)$$

The foundational research question is:
> **Can APEDP 4.0 generate a mathematical model whose specific structural form was never supplied in any grammar, template library, or prompt prior, recover it from constrained observations under noise, and demonstrate superior OOD generalization and epistemic identifiability compared to standard regression baselines?**

---

## 2. Experimental Setup: The Blind Synthesis Benchmark

### 2.1 The Target Law: Multi-Regime Rational-Exponential Transition
The target equation represents a multi-scale physical envelope common in non-equilibrium thermodynamics, turbulent energy dissipation, and boundary-layer kinetics:

$$y^*(x) = \frac{A x^\alpha}{1 + B x^\beta} \exp\left(-C x^\gamma\right)$$

**Nominal Ground-Truth Parameterization:**
- $A = 2.45$, $\alpha = 1.50$ (sublinear power-law onset)
- $B = 0.85$, $\beta = 2.00$ (saturation transition)
- $C = 0.30$, $\gamma = 1.00$ (exponential far-field dissipation)

### 2.2 Physical Regimes & Challenge Characteristics
1. **Regime I ($x \in [0.1, 1.0]$):** Power-law growth dominated by $x^\alpha$.
2. **Regime II ($x \in [1.0, 3.5]$):** Rational saturation turnover where denominator $1 + B x^\beta$ arrests growth.
3. **Regime III ($x \in [3.5, 8.0]$):** Exponential decay dominated by $\exp(-C x)$.
4. **Sampling Trap:** Deliberate observation gap in $x \in [2.2, 3.0]$ to test for avoided-pole vulnerabilities.
5. **Noise:** Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.02 \times \text{std}(y)$.

---

## 3. Input Specification Provided to Systems

No competitor or APEDP engine is given the structural skeleton $\frac{x^\alpha}{1 + x^\beta} e^{-x}$. 

Instead, systems are provided with:
1. **Observation Data:** $N = 150$ noisy observations in $[0.1, 5.0]$ with an unobserved gap in $[2.2, 3.0]$.
2. **Dimensional Constraint:**
   - Input $x$: $[L]$ (length, meters)
   - Target $y$: $[M \cdot L^{-1} \cdot T^{-2}]$ (pressure / stress, Pascals)
3. **Physical Boundary Conditions:**
   - Regular origin: $\lim_{x \to 0^+} y(x) = 0$
   - Far-field extinction: $\lim_{x \to \infty} y(x) = 0$
   - Positivity: $y(x) \ge 0$ for all $x \in (0, \infty)$
4. **Evaluation Split:**
   - **Train:** $N_{\text{train}} = 100$ points in $[0.1, 5.0]$ (excluding gap).
   - **Near-OOD:** $N_{\text{near}} = 100$ points in $[5.0, 8.0]$.
   - **Far-OOD:** $N_{\text{far}} = 100$ points in $[8.0, 15.0]$.

---

## 4. Evaluated Competitor Paradigms

| Paradigm | Configuration | Strategy |
| :--- | :--- | :--- |
| **gplearn** | Standard Genetic Programming | Unconstrained tree evolution |
| **Wrapped AI Feynman** | AI Feynman + Stage A/B/C | Dimensional filtering + polynomial/transcendental brute-force |
| **LLM-SR** | ICLR 2025 Few-Shot Prompting | Prompted symbolic exploration |
| **Neural AutoML** | Multilayer Perceptron (MLP) | Flexible uninterpretable black-box baseline |
| **APEDP 4.0 Synthesis Engine** | Rust Kernel + Directed Mutation + Epistemic Stage C | High-throughput structural mutation ($10^5$ forms) + VarPro + Active Singularity Audit |

---

## 5. Quantitative Success Criteria & Evaluation Metric

A system achieves **Decisive Scientific Discovery** on this benchmark if and only if it satisfies all 5 criteria simultaneously:

1. **Structural Equivalence ($\ge 95\%$ Match):**
   - The recovered equation simplifies algebraically to $A x^\alpha (1 + B x^\beta)^{-1} e^{-C x^\gamma}$ with $|\alpha - 1.5| < 0.05$, $|\beta - 2.0| < 0.05$, and $|\gamma - 1.0| < 0.05$.
2. **Near-OOD Superiority:**
   - $\text{MSE}_{\text{near}}(\text{APEDP}) < 0.10 \times \text{MSE}_{\text{near}}(\text{Neural AutoML})$.
3. **Far-OOD Extrapolation Stability:**
   - $\text{MSE}_{\text{far}}(\text{APEDP}) < 1.0 \times 10^{-3}$ (extinction obeyed; no divergence to $\pm \infty$).
4. **Epistemic Robustness (Zero Pathologies):**
   - Condition number $\kappa < 10^4$ (well-conditioned identifiability).
   - Active Singularity Prober detects 0 unobserved poles in the gap $[2.2, 3.0]$.
5. **Formal Adjudication:**
   - Stage C classifies candidate as **$C_1$ (Decisive Unique Support)**.

---

## 6. Execution Roadmap

1. **Synthetic Generator (`benchmarks/synthetic_synthesis_generator.py`):**
   - Generate frozen datasets (`train.csv`, `near_ood.csv`, `far_ood.csv`, `spec.json`).
2. **Synthesis Engine Prototype (`apedp/model_synthesis.py`):**
   - Implement structural mutation operators that compose rational envelopes and exponential damping over elementary seeds.
3. **Batch Execution & Comparison:**
   - Run gplearn, Wrapped AI Feynman, Neural AutoML, and APEDP 4.0.
4. **Scientific Report Publication (`docs/APEDP_V4_SYNTHESIS_BENCHMARK_REPORT.md`):**
   - Report confusion matrices, OOD trajectories, and epistemic audit logs.
