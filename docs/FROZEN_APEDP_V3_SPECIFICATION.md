# Frozen Specification: APEDP 3.0 Epistemically Aware Discovery Architecture

**Document ID:** APEDP-V3-FROZEN-2026-09-25  
**Version:** 3.0.0-FROZEN (IMMUTABLE TOURNAMENT & BENCHMARK SPECIFICATION)  
**Freeze Date:** 2026-09-25  
**Audit Status:** AUDITED & CRYPTOGRAPHICALLY LOCKED  

---

## 1. Architectural Pipeline & Adjudication Sequence

APEDP 3.0 replaces naive error minimization with a five-stage epistemic adjudication pipeline:

$$\boxed{
\text{Candidate Generation}
\longrightarrow
\text{Stage A (Math/Dim Audit)}
\longrightarrow
\text{Stage B (Identifiability & Stability)}
\longrightarrow
\text{Stage C (Triad Adjudication)}
\longrightarrow
\text{Mathematical Assurance Guard}
}$$

1. **Candidate Generation Layer:**
   - Multi-branch generator: Canonical physics grammar + Evolutionary search + Variable Projection (VarPro) + Dimensional constant synthesizer.
   - Dual-budget synergy ensuring candidate pool contains both parsimonious invariants and flexible reference forms.

2. **Stage A: Mathematical & Dimensional Audit:**
   - Strict dimensional homogeneity verification (`dim_status == "VALID"`).
   - Domain sanity and singularity avoidance.
   - Rejects unphysical or dimensionally mismatched candidates upfront.

3. **Stage B: Parameter Identifiability & Stability Gating:**
   - Design matrix conditioning: $\kappa(X^T X) \le 10^6$.
   - Paired bootstrap stability: $S(f) \ge 0.15$ across resampled sub-cohorts.
   - Predictability baseline: holdout $R^2 \ge 0.0$.

4. **Stage C: Triad Adjudication Engine:**
   - **MDL / BIC Parsimony Pre-Pruning:** Complex/hybrid candidates ($\Delta k \ge 20$) are pruned upfront unless $\Delta\text{BIC} < -6.0$.
   - **Noise-Floor Equivalence Gate:** If $|\Delta\text{MSE}| \le 0.0015$ or paired bootstrap $t$-statistic $|d| \le 1.96 \cdot \text{SE}(d)$, candidates are classified as observationally indistinguishable (`MULTI_HYPOTHESIS`).
   - **Active Singularity & Pole Stress Testing:** Denominator roots, branch points, and logarithmic cuts are analytically probed. Unsampled gap poles (`AVOIDED_POLE_IN_GAP`) and unobserved asymptotic boundaries (`UNOBSERVED_CRITICAL_REGIME`) competing against smooth polynomials trigger immediate demotion to `INSUFFICIENT_EVIDENCE`.

5. **Mathematical Assurance & Proof Boundary Guard:**
   - Universal mathematical conjectures (e.g. Riemann Hypothesis critical line) evaluated over finite empirical zeros ($N < \infty$) are strictly certified as `EMPIRICALLY_SUPPORTED`.
   - The universal quantifier remains protected (`PROOF_NOT_ESTABLISHED`).

---

## 2. Frozen Parameter Register

| Parameter | Frozen Value | Architectural Scope | Rationale |
| :--- | :---: | :--- | :--- |
| `holdout_margin_ratio` ($\tau$) | **$1.15$** | Stage C Relative Winner Gate | Requires 15% clear predictive superiority |
| `max_condition_number` ($\kappa_{\max}$) | **$1.0 \times 10^6$** | Stage B Identifiability | Rejects collinear, non-identifiable parameters |
| `min_bootstrap_stability` ($S_{\min}$) | **$0.15$** | Stage B Structural Stability | Rejects spurious fits sensitive to perturbation |
| `min_holdout_r2` | **$0.0$** | Stage B Predictive Baseline | Candidates must beat horizontal mean baseline |
| `noise_floor_delta_mse` | **$0.0015$** | Stage C Observational Equivalence | Rejects illusory precision in narrow regimes |
| `bootstrap_se_multiplier` | **$1.96$** | Stage C Paired Significance | $95\%$ two-tailed observational equivalence |
| `bic_parsimony_penalty` | **$-6.0$** | Stage C MDL Pre-Pruning | Bayes factor $> 20$ required for parameter explosion |
| `singularity_boundary_ratio`| **$0.10$** | Singularity Probing | Critical regime defined within $10\%$ domain span |
| `proof_boundary_guard` | **ACTIVE (Enforced)** | Mathematical Assurance | Prohibits claiming formal proof from finite data |

---

## 3. Mathematical Identifiability: The Quadratic Precision Scaling Law

In evaluating small perturbations $\beta_n = 0.5 + \epsilon \sin(n)$:
The observable difference in mean squared error scales quadratically:

$$\Delta\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\epsilon \sin(n_i))^2 \approx \frac{1}{2} \epsilon^2$$

Consequently, the empirical identifiability threshold $\epsilon^*_{\text{empirical}}$ under arithmetic precision with machine epsilon $\epsilon_{\text{mach}}$ obeys:

$$\boxed{\epsilon^*_{\text{empirical}} \approx \sqrt{2 \cdot \epsilon_{\text{mach}}}}$$

### Empirical Multi-Precision Verification Table
Executed across 5 distinct arithmetic precision environments ([`benchmarks/mathematical/run_precision_scaling.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/mathematical/run_precision_scaling.py)):

| Arithmetic Regime | Decimal Digits | Machine Epsilon ($\epsilon_{\text{mach}}$) | Theoretical Threshold ($\sqrt{2\epsilon_{\text{mach}}}$) | Empirical Threshold ($\epsilon^*_{\text{empirical}}$) | Scaling Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IEEE-754 Single (`float32`)** | $7$ | $1.19 \times 10^{-7}$ | $4.88 \times 10^{-4}$ | **$3.00 \times 10^{-4}$** | Tracks $\sqrt{2\epsilon_{\text{mach}}}$ ($0.61\times$) |
| **IEEE-754 Double (`float64`)** | $16$ | $2.22 \times 10^{-16}$ | $2.11 \times 10^{-8}$ | **$1.00 \times 10^{-8}$** | Tracks $\sqrt{2\epsilon_{\text{mach}}}$ ($0.47\times$) |
| **mpmath Arbitrary (50-digit)** | $50$ | $1.00 \times 10^{-50}$ | $1.41 \times 10^{-25}$ | **$1.00 \times 10^{-25}$** | Tracks $\sqrt{2\epsilon_{\text{mach}}}$ ($0.71\times$) |
| **mpmath Arbitrary (100-digit)** | $100$ | $1.00 \times 10^{-100}$ | $1.41 \times 10^{-50}$ | **$1.00 \times 10^{-50}$** | Tracks $\sqrt{2\epsilon_{\text{mach}}}$ ($0.71\times$) |
| **mpmath Arbitrary (200-digit)** | $200$ | $1.00 \times 10^{-200}$ | $1.41 \times 10^{-100}$ | **$1.00 \times 10^{-100}$** | Tracks $\sqrt{2\epsilon_{\text{mach}}}$ ($0.71\times$) |

**Conclusion:** The empirical transition tracks the predicted $\sqrt{2\epsilon_{\text{mach}}}$ scaling across five arithmetic regimes spanning 100 orders of magnitude ($10^{-4} \to 10^{-8} \to 10^{-25} \to 10^{-50} \to 10^{-100}$). This demonstrates that $\epsilon^*_{\text{empirical}} \approx 10^{-8}$ under float64 is the mathematical consequence of quadratic loss under IEEE-754 double precision arithmetic, rather than an algorithmic or statistical candidate search failure.

---

## 4. Frozen Corpus & Benchmark Integrity

### 4.1 Physical Benchmark Corpus (30 Problems, Seed 42)
- **Class 1 (Exact Isolated Laws, $\mathcal{B}_1$):** `P01`–`P06`
- **Class 2 (Observational Equivalence, $\mathcal{B}_2$):** `P07`–`P12`
- **Class 3 (Measurement Noise & Symmetries, $\mathcal{B}_3$):** `P13`–`P18`
- **Class 4 (Singularities & Critical Regimes, $\mathcal{B}_4$):** `P19`–`P24`
- **Class 5 (Dimensionality & Continuum Hybrids, $\mathcal{B}_5$):** `P25`–`P30`
- **Unique Physical Laws Subset ($\mathcal{E}_1$):** 17 Problems (`P01`–`P06`, `P13`–`P18`, `P22`, `P26`, `P27`, `P29`, `P30`)

### 4.2 Mathematical Benchmark Corpus (APEDP-RH, 21 Cases)
- **Discovery Stage (`RH-01` to `RH-03`):** Unbiased reconstruction of critical line $\operatorname{Re}(\rho) = 1/2$.
- **Identifiability Stage (`RH-04` to `RH-08`):** Competing alternative hypotheses ($\Delta = 10^{-3}, 10^{-6}, 1/n^2$, delayed shifts).
- **Precision Floor Stage (`RH-09` to `RH-13`):** Controlled measurement noise ($\sigma \in [0, 0.05]$).
- **Fake-RH Counter-Benchmarks (`RH-14` to `RH-16`):** Anti-overfitting controls with spurious patterns.
- **Expanded Adversarial Suite (`RH-17` to `RH-21`):** Local agreement, delayed divergence, oscillatory cancellation, slow vanishing, and tiny deviation.

---

## 5. Cryptographic Module Checksums (SHA-256)

```text
File Path                                           SHA-256 Checksum
──────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────
apedp/decision/stage_c.py                           aa82d885cbcfb8bfec4842a7b85dd00a8a78ddd9535c65661f27705db37ecd37
apedp/singularity_probing.py                        64fafeac9561b344539cdee682bb7e1b7d16e509f98d1fd9ee3213a8e2790f62
apedp/mathematical/mathematical_layer.py            b3b34907201bc5d4519b58d5ef1b102b2859b4c4536465cdf1a80b61ff52fd2d
apedp/mathematical/rh_precision.py                  afc7d4013d91d424ff7f1aaf38ec90dd73bc31d2b85248ec1021b44df55ac3c4
apedp/mathematical/rh_adversarial.py                6977647ac4bc1a6245e09e0788ca5b47a9eab8b2ed7c07c0a3f1501be0fb4d06
apedp/mathematical/precision_scaling_experiment.py  d79a7a5037196eeeda126a8cf3a4a7edbfd551d464b4cb9acb70051b154fad1a
benchmarks/abstention_wrapper.py                    496e634625c954dc8541344fe7597c824b1d50efa2299f10e7b67c770057cce8
benchmarks/independent_evaluator.py                 6c50dd7e1b487d792622c02f193f72fbafe815a79d9a98625773a6cbac9bf71b
benchmarks/run_wrapped_tournament.py                033436c95c9842b5f6ed698fe904e3b8e39de29d12f1a0196c5f8ea3bc92bd17
benchmarks/mathematical/run_rh_benchmark.py         ef79c767f0c866f30c46f5623be892e435431cc8222c2d2e54798c2e6f5c806d
──────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────
```
