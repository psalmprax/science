# Formal Pre-Registration: Phase 4 Controlled Scientific Benchmark Protocol

**Status:** PRE-REGISTERED (FROZEN PRIOR TO EXTERNAL BENCHMARK EXECUTION)  
**Protocol Version:** 1.0.0  
**Target Architecture:** APEDP 3.0 (Autonomous Physical Equation Discovery Pipeline)  
**Registered Date:** 2026-09-24  
**Principal Invariants:** Hard Stage A Gates (`INV-1` to `INV-10`), Epistemic State Machine, Neural Adversarial Reference Ceiling Check.

---

## 1. Executive Summary & Epistemic Positioning

Scientific equation discovery must not be reduced to unconstrained curve fitting or genetic expression optimization. Conventional symbolic regression systems frequently fall into the **False Confidence Trap**: selecting an overfitted or dimensionally nonsensical equation because it minimized mean squared error on sample data points or sat on an arbitrary complexity-fit heuristic frontier.

APEDP 3.0 introduces a paradigm shift from *symbolic regression* to **Scientific Model Discovery and Epistemic Inference**. Rather than forcing a single winner on incomplete data, APEDP:
1. Filters candidates through strict mathematical and dimensional admissibility gates (`VALID`, `INVALID`, `UNRESOLVED`).
2. Evaluates parameter identifiability (condition number $\kappa \le 10^6$) and bootstrap stability ($S(f) \ge 0.15$).
3. Formally characterizes uncertainty across four epistemic states (`UNIQUE_SUPPORT`, `MULTI_HYPOTHESIS`, `INSUFFICIENT_EVIDENCE`, `NO_VALID_CANDIDATE`).
4. Employs a **Neural Adversarial Reference** to benchmark the flexible empirical noise floor and detect missing physical structure.
5. Deploys **Active Hypothesis Discrimination (AHD)** to compute the optimal experimental measurements ($x^*$) required to resolve ambiguity.

This pre-registration protocol establishes a controlled, reproducible head-to-head evaluation of APEDP against five leading external competitor paradigms. All evaluation criteria, model budgets, candidate pools, loss functions, and statistical tests are locked in advance to prevent post-hoc tuning or selective reporting.

---

## 2. Competitor Lineup & Paradigms

APEDP will be evaluated in head-to-head comparison against five representative paradigms:

| Competitor Paradigm | Representative Implementation | Core Search Mechanism | Primary Selection Criterion |
|---|---|---|---|
| **Pareto Genetic SR** | **PySR** (v0.16+, Julia backend) | Multi-population evolutionary GP with simulated annealing | Pareto frontier over MSE vs AST complexity |
| **Physics-Guided Decomposition** | **AI Feynman 2.0** | Dimensional analysis, symmetry detection, separability decomposition | Recursive modular decomposition + BIC |
| **Classical GP Baseline** | **gplearn** (v0.4+) | Standard Genetic Programming tree search | Penalized fitness (MSE + parsimony coefficient) |
| **LLM-Guided SR** | **LLM-SR** (GPT-4o / Claude 3.5 Sonnet / Gemini 1.5 Pro) | Few-shot prompting, hypothesis proposal, iterative code refinement | In-context log-likelihood + holdout MSE |
| **Empirical Neural AutoML** | **Flax/JAX Deep MLP / ResNet** | Gradient descent / AdamW with dropout and weight decay | Holdout cross-validation loss |

---

## 3. Resource Equalization & Compute Bounds

To guarantee methodological validity, computational resources are governed by two distinct constraints. We explicitly note that these represent **two different fairness axes**, and we do **not** claim that the paradigms receive computationally equal resources:
1. **Search-Budget Fairness ($N_{\text{evaluations}}$):** Cap on the number of discrete candidate model formulations generated and evaluated against data. The differing budget caps (e.g. 500 for LLM-SR vs 10,000 for Symbolic SR) reflect the **natural search mechanism of each paradigm**, rather than an assertion of equal search opportunity.
2. **Computational-Budget Fairness ($T_{\text{wall}} \le 300\text{s}$):** Hard wall-clock limit per problem per method on identical host hardware.

### 3.1 Paradigm-Specific Candidate Evaluation & Budget Definitions

| Paradigm | Definition of 1 Candidate Evaluation | Search Budget ($N_\text{eval}$) | Compute Bounds | Telemetry Metrics Recorded |
|---|---|---|---|---|
| **Symbolic SR (APEDP, PySR, gplearn)** | Generation of 1 discrete syntactic AST expression + non-linear parameter estimation (e.g. BFGS/LM) + holdout residual evaluation. | $N_{\text{eval}} \le 10,000$ | $T_{\text{wall}} \le 300\text{s}$, single CPU thread | AST depth, node count, generated/evaluated/surviving counts |
| **Physics-Guided (AI Feynman 2.0)** | One dimensional/symmetry transformation check, separability test, or sub-expression polynomial fit. | $N_{\text{eval}} \le 10,000$ | $T_{\text{wall}} \le 300\text{s}$, 4 CPU cores | Module execution traces, sub-fits evaluated |
| **LLM-Guided SR (LLM-SR)** | One generated Python/SymPy mathematical candidate function from prompt + parameter tuning + holdout loss evaluation. | $N_{\text{eval}} \le 500$ (50 rounds $\times$ 10 candidates) | API timeout 300s, max 100k tokens | Prompt tokens, completion tokens, API calls, latency |
| **Neural Reference Budget (Neural AutoML / APEDP Reference)** | Architectural trial (training 1 MLP/ResNet configuration over gradient descent steps to convergence). | $N_{\text{NN,trials}} \in [1, 5]$ distinct hyperparameter trials | $T_{\text{wall}} \le 60\text{s}$, GPU/CPU execution | Number of trials $N_{\text{NN,trials}}$, parameter count, epochs/steps, FLOPs estimate, learning rate |

### 3.2 Telemetry Reporting Mandate
To prevent false equivalences, results will **not** normalize resource metrics into a single "efficiency score." The benchmark publication will report the complete telemetry profile for every competitor:
- Search budget consumed: $N_{\text{eval}}$ (generated, evaluated, and surviving candidates).
- Neural Reference trials: $N_{\text{NN,trials}}$ with parameter counts and training steps.
- Physical execution time: Wall-clock seconds ($T_{\text{wall}}$).
- Core execution time: CPU core-seconds and accelerator (GPU) seconds.
- Floating-point operations: Estimated FLOPs where practical.
- LLM API telemetry: Total prompt tokens, completion tokens, and API call count.
- Deterministic reproducibility: Exact random seeds ($s \in \{42, 101, 2024\}$) and hardware environment specification.

### 3.3 Expression Complexity Bounds
- Maximum AST depth: **10**.
- Maximum total nodes (tokens): **30**.
- Allowed operator set: $\{+, -, \times, /, \text{pow}, \exp, \log, \sin, \cos, \sqrt{\cdot}\}$.

### 3.4 Data Splits
Every dataset is partitioned deterministically using fixed random seeds ($s \in \{42, 101, 2024\}$):
- **Training Split:** 60% of points within domain $\mathcal{D}_0$.
- **Holdout Split:** 20% of points within domain $\mathcal{D}_0$ (interpolated).
- **Near-OOD Split:** 10% of points in extrapolation domain $\mathcal{D}_1 = [x_{\max}, 1.5 x_{\max}]$.
- **Far-OOD Split:** 10% of points in extrapolation domain $\mathcal{D}_2 = [1.5 x_{\max}, 3.0 x_{\max}]$.

---

## 4. Frozen Evaluation Pipeline & Decision Invariants

APEDP and competitors will be audited under the exact, frozen APEDP 3.0 multi-stage decision pipeline:

```
                          DATASET (Train, Holdout, Near-OOD, Far-OOD)
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   STAGE A: Mathematical & Dimensions  │
                        │   - Denominator roots D(x) = 0        │
                        │   - Units consistency                 │
                        │   - Transcendental arguments          │
                        │   OUTCOME: VALID / INVALID / HOLD     │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼ (Admissible Only)
                        ┌───────────────────────────────────────┐
                        │   STAGE B: Viability & Identifiability│
                        │   - Finite holdout MSE & R² ≥ 0.0     │
                        │   - Design matrix cond(J) ≤ 10⁶       │
                        │   - Bootstrap stability S(f) ≥ 0.15   │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼ (Viable Only)
                        ┌───────────────────────────────────────┐
                        │   STAGE C: Epistemic Adjudication     │
                        │   - Holdout margin ratio: 1.15        │
                        │   - Neural Adversarial Ceiling check  │
                        │   - Occam parsimony vs ambiguity      │
                        │   - Cross-class trade-off analysis    │
                        └───────────────────┬───────────────────┘
                                            │
                  ┌─────────────────────────┼─────────────────────────┐
                  ▼                         ▼                         ▼
           UNIQUE_SUPPORT            MULTI_HYPOTHESIS       INSUFFICIENT_EVIDENCE
        (Decisive Discovery)        (Active Discrim.)       (Pathology / Artifact)
```

### Frozen Decision Rules:
1. **Rule C-1 (Occam Selection):** If exactly one admissible, viable candidate achieves holdout MSE $\le 1.15 \times \min(\text{MSE})$ and matches the Neural Adversarial Reference ceiling, emit `UNIQUE_SUPPORT`.
2. **Rule C-2 (Multi-Hypothesis Preservation):** If multiple admissible candidates lie within the $1.15 \times \min(\text{MSE})$ margin, or if in-sample and holdout/OOD rankings disagree, emit `MULTI_HYPOTHESIS`. Never force an unevidenced winner.
3. **Rule C-3 (False Confidence Halt):** If all candidates fail parameter condition bounds ($\kappa > 10^6$) or bootstrap stability ($S(f) < 0.15$), emit `INSUFFICIENT_EVIDENCE`.
4. **Rule C-4 (Missing Structure Flag):** If the best symbolic equation exhibits holdout error $> 1.5 \times \text{MSE}_\text{neural}$, flag `MISSING_STRUCTURE` and admit the Hybrid model ($f_\text{sym} + \Delta_\text{NN}$).

---

## 5. Benchmark Corpus (Balanced 30 Problems: 6 Problems $\times$ 5 Classes)

To guarantee balanced representation without any domain dominating the statistical outcomes, the corpus contains exactly **6 problems per class** ($6 \times 5 = 30$). Every competitor will report both the overall profile $\mathbf{M}_\text{overall}$ and per-class profiles $\mathbf{M}_c$ for $c \in \{1, \dots, 5\}$.

### Class 1: Canonical Physical Laws (6 Problems)
- P1.1: Kepler's Third Law ($T^2 = k a^3$)
- P1.2: Newton's Law of Universal Gravitation ($F = G m_1 m_2 / r^2$)
- P1.3: Stefan-Boltzmann Radiation ($P = \sigma \epsilon A T^4$)
- P1.4: Ideal Gas Law ($P = n R T / V$)
- P1.5: Relativistic Energy-Momentum ($E = \sqrt{p^2 c^2 + m^2 c^4}$)
- P1.6: Simple Harmonic Oscillator Period ($T = 2\pi \sqrt{m/k}$)

### Class 2: Observationally Equivalent & Narrow-Regime Problems (6 Problems)
- P2.1: Isobaric Heat Capacity (Shomate polynomial vs Quadratic $C_p$ over $T \in [300, 500]\text{K}$)
- P2.2: Low-Strain Hookean Elasticity vs Nonlinear Neo-Hookean ($F = k x$ vs $F = \mu (1+x - (1+x)^{-2})$)
- P2.3: Michaelis-Menten Kinetics vs Lineweaver-Burk in substrate saturation
- P2.4: Arrhenius Kinetics vs Biexponential Relaxation in narrow temperature span
- P2.5: Atmospheric Barometric Density vs Linearized Exponential in planetary boundary layer
- P2.6: Damped Wave Amplitude vs Linearized Attenuation in near-field

### Class 3: Asymptotic & Extrapolation Regimes (6 Problems)
- P3.1: Relativistic Velocity Addition ($v = (u+w)/(1+uw/c^2)$ as $u, w \to c$)
- P3.2: Yukawa Screened Potential ($V(r) = -g^2 e^{-\mu r} / r$ as $r \to \infty$)
- P3.3: Damped Nonlinear Oscillator Asymptotics ($x(t) \to 0$ without secular drift)
- P3.4: High-Temperature Blackbody Radiation (Rayleigh-Jeans limit vs Planck distribution)
- P3.5: Fermi-Dirac Distribution High-Energy Tail (approaching Maxwell-Boltzmann)
- P3.6: High-Reynolds Number Drag Transition ($C_D \to \text{const}$ asymptotic plateau)

### Class 4: Singular & Pathological Regimes (6 Problems)
- P4.1: Resonance Pole avoided on grid points ($f(x) = \frac{A}{(x - x_0)^2 + \Gamma^2/4}$)
- P4.2: Critical Opalescence / Susceptibility Divergence ($\chi \propto (T - T_c)^{-\gamma}$)
- P4.3: Boundary Layer Singularity in viscous fluid flow ($f(x) = 1/\sqrt{x}$)
- P4.4: High-order polynomial Runge Phenomenon near boundaries
- P4.5: Dimensional Incompatibility Trap (summing incompatible physical units with tiny numerical error)
- P4.6: Multi-Parameter Collinear Singularity ($\kappa(J) > 10^8$ identifiability trap)

### Class 5: Multivariable Non-Separable & Hybrid Regimes (6 Problems)
- P5.1: Aerodynamic Drag with Mach-dependent Compressibility ($C_D(M, \alpha, \text{Re})$)
- P5.2: Turbulent Boundary Layer Dissipation Rate ($\epsilon = C_\mu k^{3/2} / L$)
- P5.3: Two-Compartment Pharmacokinetic Drug Clearance ($C(t) = A e^{-\alpha t} + B e^{-\beta t}$)
- P5.4: Non-Hermitian Quantum Damped Wavepacket Dispersion
- P5.5: Plasma Hall-MHD Coupled Drift Wave Velocity
- P5.6: Non-Newtonian Shear-Thinning Fluid Viscosity (Carreau-Yasuda model)

---

## 6. Pre-Declared Performance Metrics & Statistical Hypotheses

### 6.1 Multi-Dimensional Evaluation Profile Mandate
Results will **NOT** be collapsed into an aggregated scalar ranking (e.g. "APEDP = 9.7 vs PySR = 9.1"), as single scores obscure fundamental scientific trade-offs between computational throughput, model parsimony, and epistemic honesty.

Instead, every system will be characterized across an explicit 7-dimensional evaluation vector:
$$\mathbf{M} = \Big( R_\text{rec},\; R_\text{false},\; P_\text{epi},\; E_\text{OOD},\; R_\text{dim},\; \eta_\text{AHD},\; C_\text{compute} \Big)$$

Both overall results $\mathbf{M}_\text{overall}$ and per-class breakdowns $\mathbf{M}_c$ for each class $c \in \{1,\dots,5\}$ will be published.

### 6.2 Primary Metric Definitions
1. **Symbolic Recovery Rate ($R_\text{rec}$):** Fraction of problems where the algorithm discovers an expression algebraically isomorphic (`SymbolicEquivalenceEngine`) to the ground truth law.
2. **False Discovery Rate ($R_\text{false}$):** Fraction of problems where the algorithm issues `UNIQUE_SUPPORT` (or an unhedged top model recommendation) for an overfitted, dimensionally invalid, or unidentifiable equation.
3. **Epistemic Calibration Precision ($P_\text{epi}$):** Formally evaluated across six distinct ground-truth epistemic problem categories to prevent metric gaming:
   $$P_\text{epi} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\Big( \text{System\_Epistemic\_State}_k == \text{Ground\_Truth\_Epistemic\_Class}_k \Big)$$
   where the ground truth classes are defined as:
   - **$\mathcal{C}_1$ (Uniquely Identifiable):** Generating law is uniquely determined by data $\implies$ Target: `UNIQUE_SUPPORT(true_model)`.
   - **$\mathcal{C}_2$ (Observationally Non-Identifiable):** Multiple distinct admissible laws achieve identical holdout error within empirical noise $\implies$ Target: `MULTI_HYPOTHESIS(candidates)`. Premature selection of one is penalized as an epistemic failure.
   - **$\mathcal{C}_3$ (Insufficient Data / Ill-Conditioned / Unstable):** Parameter collinearity ($\kappa > 10^6$) or bootstrap instability ($S(f) < 0.15$) $\implies$ Target: `INSUFFICIENT_EVIDENCE(winner=None)`.
   - **$\mathcal{C}_4$ (Syntactic / Structural Equivalence):** Different algebraic representations of the same underlying law $\implies$ Target: Canonicalized single equivalence family.
   - **$\mathcal{C}_5$ (Missing Nonlinear Structure):** Pure symbolic models fail empirical noise floor $\implies$ Target: `MISSING_STRUCTURE` flag + Hybrid model admission.
   - **$\mathcal{C}_6$ (Falsification & Belief Revision):** New experimental regime contradicts previously supported model $\implies$ Target: Belief update (revocation of $H_1$, certification of $H_2$).

### 6.3 Epistemic Confusion Matrix Mandate
To expose the qualitative nature of epistemic errors (e.g. converting ambiguity into false certainty vs. overly conservative uncertainty), every competitor will be evaluated via a complete $6 \times 6$ Epistemic Confusion Matrix:

| Ground Truth Category | UNIQUE | MULTI | INSUFFICIENT | EQUIVALENT | MISSING_STRUCT | REVISED |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\mathcal{C}_1$: Unique** | True Positive | Over-cautious | Failure to certify | Family error | Misclassified | Unrevised |
| **$\mathcal{C}_2$: Ambiguous** | **False Certainty** | True Ambiguity | Over-cautious | Family error | Misclassified | Unrevised |
| **$\mathcal{C}_3$: Insufficient** | **False Confidence** | Misdiagnosed | True Rejection | Family error | Misclassified | Unrevised |
| **$\mathcal{C}_4$: Equivalent** | Duplicate | Split | Misdiagnosed | True Invariant | Misclassified | Unrevised |
| **$\mathcal{C}_5$: Missing Struct** | Overfitted Sym | Misdiagnosed | Misdiagnosed | Family error | True Detection | Unrevised |
| **$\mathcal{C}_6$: Falsification** | Dogmatic Stance | Hesitant | Misdiagnosed | Family error | Misclassified | True Revision |

4. **Out-of-Distribution Extrapolation Error ($E_\text{OOD}$):** Median Relative Root Mean Squared Error (RRMSE) evaluated on near-OOD ($\mathcal{D}_1$) and far-OOD ($\mathcal{D}_2$) test sets.
5. **Dimensional Validity Rate ($R_\text{dim}$):** Percentage of selected equations satisfying dimensional homogeneity without unscaled transcendental arguments.
6. **Active Query Efficiency ($\eta_\text{AHD}$):** Number of targeted experimental measurement queries required to collapse `MULTI_HYPOTHESIS` ambiguity to `UNIQUE_SUPPORT` (measured against random and uniform grid search baselines).
7. **Computational Resource Cost ($C_\text{compute}$):** Wall-clock execution time $T_\text{wall}$, total candidates sampled $N_\text{eval}$, and FLOPs/core-seconds consumed.

### 6.4 Pre-Registered Statistical Hypotheses & Tests
- **Hypothesis H1 (False Discovery Reduction):** APEDP will achieve a statistically significant reduction in $R_\text{false}$ compared to PySR, gplearn, and LLM-SR ($p < 0.01$, one-tailed Wilcoxon signed-rank test), driven by Stage A math/dim checks and Stage B identifiability filters.
- **Hypothesis H2 (Epistemic Calibration):** In Class 2 ($\mathcal{C}_2$) benchmarks, APEDP will emit `MULTI_HYPOTHESIS` in $\ge 90\%$ of cases, whereas competitor baselines will prematurely declare a single winner in $> 80\%$ of cases ($p < 0.01$).
- **Hypothesis H3 (Missing Structure Detection):** In Class 5 ($\mathcal{C}_5$) benchmarks, APEDP's Neural Adversarial Reference will successfully flag `MISSING_STRUCTURE` and trigger a hybrid model that achieves lower holdout error than any pure symbolic candidate ($p < 0.05$).
- **Hypothesis H4 (Active Hypothesis Discrimination):** AHD-v1/v2 will identify discriminating points $x^*$ that achieve disambiguation in fewer than 3 measurement iterations in $\ge 80\%$ of ambiguous cases ($\eta_\text{AHD} \le 3$).
- **Multiple Comparisons Correction:** All family-wise hypothesis tests will apply the **Bonferroni-Holm step-down procedure** with baseline $\alpha = 0.05$.

---

## 7. Open Science Commitments & Anti-P-Hacking Safeguards

1. **Protocol Immutability & Parameter Freeze:**
   - This document is frozen and version-controlled prior to tournament execution.
   - All internal APEDP decision thresholds ($\tau=0.08, \text{margin}=1.15, \kappa=10^6, S(f)=0.15$), Stage A rules, Stage B filters, and Stage C adjudication logic are **strictly locked**.
   - No post-hoc tuning of APEDP against the 12 internal cases or competitor benchmark results is permitted.
2. **Complete Reporting Mandate:** Every tournament run across all 30 problems, including failures, timeouts, and unresolvable ambiguities, will be published in full in the open benchmark ledger.
3. **Reproducibility Guarantee:** All synthetic data generators, seeds, and execution scripts will be preserved in `benchmarks/generate_corpus.py` with reproducible Conda/Docker container definitions and cryptographic SHA-256 checksums.

---

## 8. Blind Independent Evaluation Protocol

To prevent any possibility of information leakage or prompt contamination, the tournament enforces strict physical data isolation:

```
                            BENCHMARK CORPUS
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
          PUBLIC PARTITION                    HIDDEN PARTITION
         (Exposed to models)              (Isolated from models)
         benchmark_corpus/public/         benchmark_corpus/hidden/
                 │                                   │
                 ├─ train.csv                        └─ ground_truth.json
                 ├─ validation.csv                      - True analytical formula
                 ├─ near_ood.csv                        - True physical parameters
                 ├─ far_ood.csv                         - Oracle epistemic category
                 └─ problem_spec.json                   - Boundary singularities
                    (variable names, bounds, units)     - Admissible equivalence set
                 │                                   │
                 ▼                                   │
         ┌───────────────┐                           │
         │  Competitor   │                           │
         │    Runners    │                           │
         │ (APEDP, PySR, │                           │
         │  Feynman, ...)│                           │
         └───────┬───────┘                           │
                 │ (Candidate outputs & verdicts)    │
                 ▼                                   ▼
         ┌──────────────────────────────────────────────────┐
         │          INDEPENDENT EVALUATOR HARNESS           │
         │      (joins candidate outputs with ground truth) │
         └────────────────────────┬─────────────────────────┘
                                  │
                                  ▼
         ┌──────────────────────────────────────────────────┐
         │      OUTPUT: Evaluation Profile M & Confusion    │
         │      Matrix (Statistical Wilcoxon / Holm tests)  │
         └──────────────────────────────────────────────────┘
```

1. **Physical Isolation:** Competitor execution pipelines are passed paths strictly to `benchmark_corpus/public/P{id}/`. The process environment and filesystem permissions prevent access to `benchmark_corpus/hidden/`.
2. **Post-Hoc Blind Join:** The independent evaluator (`benchmarks/independent_evaluator.py`) reads the serialized candidate predictions and epistemic verdicts from competitor output directories, joins them with `benchmark_corpus/hidden/P{id}/ground_truth.json`, and computes all formal metrics objectively.
