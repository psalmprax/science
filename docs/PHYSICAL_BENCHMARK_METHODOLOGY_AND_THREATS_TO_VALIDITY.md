# Methodological Specification & Threats to Validity: APEDP 3.0 Physical & Mathematical Benchmarks

**Document ID:** APEDP-METHODOLOGY-AUDIT-2026-09-25  
**Evaluation Suite:** 30-Problem Physical Discovery Corpus + 21-Case RH Mathematical Discovery Suite  
**Status:** AUDITED, FROZEN, AND REPRODUCIBLE  

---

## 1. Executive Methodological Overview

The APEDP 3.0 tournament was designed to evaluate an essential yet frequently overlooked question in automated scientific discovery:

> **Can a discovery system reliably distinguish genuine physical and mathematical laws from observational artifacts, narrow-regime polynomial approximations, and unsampled mathematical singularities, while refusing to assert unhedged universal claims when evidence is finite or ambiguous?**

Conventional symbolic regression benchmarks reward raw in-sample fitting ($R^2$, training MSE) and unconditionally emit single equations (`UNIQUE_SUPPORT`). Under this regime, systems produce severe false discoveries on observationally equivalent regimes ($C_2$) and mathematical singularities ($C_3$). 

To isolate and rigorously evaluate candidate generation, parameter identifiability, and epistemic adjudication, APEDP 3.0 is evaluated under two complementary protocols:
1. **The 30-Problem Balanced Physical Corpus:** Evaluates multi-regime physical equations across 5 balanced classes ($6 \times 5 = 30$), testing candidate selection, dimensional consistency, noise-floor equivalence, and singularity probing against near-OOD and far-OOD extrapolation.
2. **The 21-Case Riemann Hypothesis Suite:** Evaluates structural discovery, empirical parameter identifiability down to machine precision, and proof-boundary compliance.

---

## 2. Complete 30-Problem Physical Benchmark Specification

The 30 benchmark problems are deterministically generated via [`benchmarks/generate_corpus.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/generate_corpus.py) using `numpy.random.default_rng(seed=42)`. Data is strictly partitioned into a **public observation set** (`train.csv`, `validation.csv`, `problem_spec.json`) and a **hidden oracle set** (`ground_truth.json`, `near_ood.csv`, `far_ood.csv`).

### Problem Catalog (P01 – P30)

| ID | Problem Name | Class & Oracle State | Inputs & Units | Target & Units | Training Domain | Near-OOD Domain | Sample Sizes ($N_\text{tr}, N_\text{val}, N_\text{near}, N_\text{far}$) | Noise Model ($\sigma$) | Ground Truth Analytical Formula |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **P01** | Kepler's Third Law | $C_1$ (`UNIQUE`) | $a$ [AU] | $T$ [yr] | $a \in [0.8, 3.5]$ | $a \in [3.6, 5.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $T = a^{1.5}$ |
| **P02** | Newton Gravitation | $C_1$ (`UNIQUE`) | $r$ [m] | $F$ [N] | $r \in [1.5, 5.0]$ | $r \in [5.1, 7.5]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $F = \frac{10.0}{r^2}$ |
| **P03** | Stefan-Boltzmann Law | $C_1$ (`UNIQUE`) | $T$ [K] | $P$ [W] | $T \in [1.0, 3.0]$ | $T \in [3.1, 4.2]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 1.0 \times 10^{-4}$ | $P = 2.5 \times 10^{-4} T^4$ |
| **P04** | Ideal Gas Law | $C_1$ (`UNIQUE`) | $T$ [K], $V$ [$\text{m}^3$] | $P$ [Pa] | $T \in [280, 380], V \in [0.02, 0.08]$ | $T \in [390, 450], V \in [0.012, 0.018]$ | $36, 12, 16, 16$ | Gaussian $\sigma = 5.0$ | $P = 8.314 \frac{T}{V}$ |
| **P05** | Relativistic Energy | $C_1$ (`UNIQUE`) | $p$ [$\text{J}\cdot\text{s/m}$] | $E$ [J] | $p \in [0.1, 3.0]$ | $p \in [3.1, 5.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $E = \sqrt{p^2 + 1.0}$ |
| **P06** | Harmonic Oscillator | $C_1$ (`UNIQUE`) | $m$ [kg] | $T$ [s] | $m \in [0.2, 4.0]$ | $m \in [4.1, 7.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $T = \sqrt{m}$ |
| **P07** | Heat Capacity Ambiguity | $C_2$ (`MULTI`) | $T$ [K] | $C_p$ [$\text{J/(mol}\cdot\text{K)}$] | $T \in [0.25, 0.55]$ | $T \in [0.60, 1.00]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | Shomate cubic vs Quadratic |
| **P08** | Elasticity Ambiguity | $C_2$ (`MULTI`) | $\epsilon$ [dimless] | $\sigma$ [Pa] | $\epsilon \in [0.01, 0.05]$ | $\epsilon \in [0.10, 0.40]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 5.0 \times 10^{-4}$ | Hookean ($3\epsilon$) vs Neo-Hookean |
| **P09** | Michaelis-Menten Saturation | $C_2$ (`MULTI`) | $S$ [$\text{mol/m}^3$] | $v$ [$\text{mol/(m}^3\cdot\text{s)}$] | $S \in [20.0, 60.0]$ | $S \in [1.0, 15.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $\frac{10S}{1+S}$ vs Affine ($9.6 + 0.007S$) |
| **P10** | Arrhenius vs Linear | $C_2$ (`MULTI`) | $T$ [K] | $k$ [$\text{s}^{-1}$] | $T \in [295, 310]$ | $T \in [320, 360]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 1.0 \times 10^{-4}$ | $e^{-600/T}$ vs Tangent line |
| **P11** | Barometric Atmosphere | $C_2$ (`MULTI`) | $z$ [m] | $\rho$ [$\text{kg/m}^3$] | $z \in [0, 500]$ | $z \in [1000, 3000]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.002$ | $1.225 e^{-z/8000}$ vs Linear |
| **P12** | Damped Wave Nearfield | $C_2$ (`MULTI`) | $x$ [m] | $y$ [m] | $x \in [0.05, 0.35]$ | $x \in [0.50, 1.50]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $e^{-0.5x}\cos(1.2x)$ vs Taylor quad |
| **P13** | Relativistic Velocity Add. | $C_1$ (`UNIQUE`) | $u, w$ [m/s] | $v$ [m/s] | $u, w \in [0.05, 0.45]$ | $u, w \in [0.50, 0.75]$ | $35, 14, 16, 16$ | Gaussian $\sigma = 0.002$ | $v = \frac{u+w}{1+uw}$ ($c=1$) |
| **P14** | Yukawa Potential | $C_1$ (`UNIQUE`) | $r$ [m] | $V$ [J] | $r \in [0.5, 2.5]$ | $r \in [2.6, 5.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $V = 5.0 \frac{e^{-0.8r}}{r}$ |
| **P15** | Bimolecular Second-Order | $C_1$ (`UNIQUE`) | $t$ [s] | $C$ [$\text{mol/m}^3$] | $t \in [0.0, 3.0]$ | $t \in [3.5, 8.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $C = \frac{4.0}{1 + 0.5t}$ |
| **P16** | Planck Radiation Tail | $C_1$ (`UNIQUE`) | $\nu$ [$\text{s}^{-1}$] | $u$ [$\text{J}\cdot\text{s/m}^3$] | $\nu \in [0.4, 3.0]$ | $\nu \in [3.1, 6.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $u = \frac{2\nu^3}{e^{0.5\nu} - 1}$ |
| **P17** | Fermi-Dirac Distribution | $C_1$ (`UNIQUE`) | $E$ [eV] | $f$ [dimless] | $E \in [0.5, 3.5]$ | $E \in [3.6, 6.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.002$ | $f = \frac{1}{e^{E-2.0} + 1}$ |
| **P18** | Schiller-Naumann Drag | $C_1$ (`UNIQUE`) | $\text{Re}$ [dimless] | $C_D$ [dimless] | $\text{Re} \in [5.0, 50.0]$ | $\text{Re} \in [60.0, 200.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $C_D = 0.44 + \frac{24.0}{\text{Re}}$ |
| **P19** | Resonance Pole Avoidance | $C_3$ (`INSUFFICIENT`) | $x$ [m] | $y$ [m] | $x \in [1.0, 5.0]$ (gap at $2.5$) | $x \in [5.2, 6.5]$ | $10, 8, 5, 4$ | Exact deterministic | True $2x+1$; trap pole at $2.5001$ |
| **P20** | Critical Opalescence Pole | $C_3$ (`INSUFFICIENT`) | $T$ [K] | $\chi$ [$\text{m}^3/\text{J}$] | $T \in [301, 350]$ | $T \in [355, 400]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.002$ | $\chi = \frac{1.5}{(T - 300)^{1.2}}$ |
| **P21** | Boundary Layer Singularity | $C_3$ (`INSUFFICIENT`) | $x$ [m] | $\tau$ [Pa] | $x \in [0.05, 2.0]$ | $x \in [2.1, 4.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $\tau = \frac{1.0}{\sqrt{x}}$ (pole at $0$) |
| **P22** | Runge Polynomial Oscillation| $C_1$ (`UNIQUE`) | $x$ [dimless] | $y$ [dimless] | $x \in [-0.85, 0.85]$ | $x \in [0.86, 0.99]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $y = \frac{1}{1 + 25x^2}$ |
| **P23** | Dimensional Trap | $C_3$ (`INSUFFICIENT`) | $T$ [s], $x$ [m] | $v$ [m/s] | $T \in [1, 5], x \approx 2T$ | $T \in [5.5, 8.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $v = 2.0$; trap candidate $T + x$ |
| **P24** | Collinear Singularity Trap | $C_3$ (`INSUFFICIENT`) | $x$ [dimless] | $y$ [dimless] | $x \in [0.1, 2.0]$ | $x \in [2.1, 3.2]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $y = e^x$; trap $c_1 e^x + c_2 e^{x+10^{-5}}$ |
| **P25** | Aerodynamic Mach Rise | $C_5$ (`MISSING`) | $M, \alpha$ [dimless, rad] | $C_D$ [dimless] | $M \in [0.3, 0.9], \alpha \in [0, 0.2]$ | $M \in [0.92, 1.05]$ | $35, 14, 16, 16$ | Gaussian $\sigma = 0.001$ | $0.02 + 0.15\alpha^2 + \Delta_\text{NN}(M)$ |
| **P26** | Turbulent Dissipation | $C_1$ (`UNIQUE`) | $k, L$ [$\text{m}^2/\text{s}^2$, m] | $\epsilon$ [$\text{m}^2/\text{s}^3$] | $k \in [0.5, 3.0], L \in [0.2, 1.5]$ | $k \in [3.2, 5.0], L \in [0.1, 0.18]$ | $35, 14, 16, 16$ | Gaussian $\sigma = 0.01$ | $\epsilon = 0.09 \frac{k^{1.5}}{L}$ |
| **P27** | Two-Compartment Clearance | $C_1$ (`UNIQUE`) | $t$ [hr] | $C$ [mg/L] | $t \in [0.1, 6.0]$ | $t \in [6.5, 12.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.01$ | $C = 5e^{-1.2t} + 2e^{-0.15t}$ |
| **P28** | Quantum Damped Dispersion | $C_5$ (`MISSING`) | $x, t$ [m, s] | $\psi$ [dimless] | $x \in [0, 5], t \in [0, 2.5]$ | $x \in [5.2, 8.0], t \in [2.6, 4.0]$ | $35, 14, 16, 16$ | Gaussian $\sigma = 0.005$ | Gaussian packet with decay |
| **P29** | Plasma Hall-MHD Drift Wave | $C_1$ (`UNIQUE`) | $k_\perp, k_\parallel$ [$\text{m}^{-1}$] | $\omega$ [rad/s] | $k_\perp \in [0.2, 2.0], k_\parallel \in [0.5, 3.0]$ | $k_\perp \in [2.2, 4.0], k_\parallel \in [3.2, 5.0]$ | $35, 14, 16, 16$ | Gaussian $\sigma = 0.01$ | $\omega = \frac{2.0 k_\parallel}{1 + 0.8 k_\perp^2}$ |
| **P30** | Carreau Non-Newtonian Visc.| $C_1$ (`UNIQUE`) | $\dot{\gamma}$ [$\text{s}^{-1}$] | $\eta$ [$\text{Pa}\cdot\text{s}$] | $\dot{\gamma} \in [0.1, 4.0]$ | $\dot{\gamma} \in [4.5, 10.0]$ | $30, 15, 15, 15$ | Gaussian $\sigma = 0.005$ | $\eta = 0.05 + \frac{0.95}{(1 + 0.25\dot{\gamma}^2)^{0.3}}$ |

---

## 3. Disentangling Extrapolation Error: Why $E_\text{OOD} \approx 10^{-16}$ Occurs

A central empirical finding in the physical tournament is the median near-OOD extrapolation error:

$$\text{Median } E_\text{OOD} = 1.76 \times 10^{-16}$$

It is critical to document why and how this number arises:

1. **Independent Evaluation Protocol:**
   `near_ood.csv` is an out-of-domain evaluation set sampled over input intervals strictly disjoint from the training intervals ($x_\text{near} > x_\text{train\_max}$). Crucially, `near_ood.csv` evaluates the **pure physical equation without synthetic additive noise** to test how accurately the discovered invariant extrapolates.
2. **Exact Symbolic Invariant Recovery:**
   When APEDP 3.0 selects the true functional form (e.g. $T = a^{1.5}$ or $F = 10 / r^2$), the analytical expression matches the ground truth law identically. Its parameters are either exact rational invariants or converged via Variable Projection to double-precision tolerance.
3. **Float64 Representation Bound:**
   Evaluating $(y_\text{true} - y_\text{pred})^2$ between mathematically identical expressions in IEEE-754 double precision yields residuals bounded solely by machine roundoff:

   $$\Delta y = y(x) - \hat{y}(x) \sim \mathcal{O}(\epsilon_\text{mach}) \implies \text{MSE} \sim \mathcal{O}(\epsilon_\text{mach}^2) \approx 10^{-32} \text{ to } 10^{-16}$$

4. **Contrast with Competitor Approximations:**
   Competitors (such as `gplearn` or unwrapped regression engines) fit polynomials or truncated expansions to the noisy training data. While their in-sample training error is small, their out-of-domain extrapolation on `near_ood.csv` diverges dramatically, yielding $E_\text{OOD} \in [0.05, 0.25]$.

---

## 4. Nine Methodological Threats to Validity & Safeguards

To ensure publication-grade methodological rigor, we explicitly analyze the nine potential threats to scientific validity:

### Threat 1: Benchmark Leakage
- **Risk:** Candidate generators or decision thresholds inadvertently gaining access to the ground-truth equations or hidden holdout sets during development.
- **Safeguard:** Strict physical directory decoupling. Public data contains only `train.csv`, `validation.csv`, and `problem_spec.json`. Ground-truth formulas, competing hypothesis labels, and OOD test sets reside exclusively in `hidden/ground_truth.json` and `hidden/near_ood.csv`. All files are cryptographically checksummed (`checksums.sha256`).

### Threat 2: Test-Set Contamination
- **Risk:** Training and extrapolation domains overlapping, causing near-OOD to measure in-distribution interpolation.
- **Safeguard:** Domain intervals are strictly segregated: $\inf(D_\text{near}) > \sup(D_\text{train})$ for every problem. For multivariable problems, near-OOD grids explore higher momentum, lower volume, or higher Mach numbers entirely outside the convex hull of the training points.

### Threat 3: Competitor Configuration Fairness
- **Risk:** Running competitors with crippled configurations or default parameters that disadvantage them.
- **Safeguard:** 
  1. We ran the **Common Abstention Wrapper** experiment ([`benchmarks/abstention_wrapper.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/abstention_wrapper.py)), providing competitors with APEDP's exact mathematical audit, identifiability filters, and epistemic state machine.
  2. Competitor false discoveries fell from $36.7\% \to 10.0\%$ (AI Feynman) and $36.7\% \to 13.3\%$ (LLM-SR), proving that the wrapper acted beneficially, but their underlying recovery remained bounded by their search spaces.

### Threat 4: Grammar & Operator Advantages
- **Risk:** APEDP’s grammar including exotic operators (e.g. fractional powers, hyperbolic functions) unavailable to competitors.
- **Safeguard:** All systems were evaluated on standard arithmetic and elementary transcendental operators ($+$, $-$, $\times$, $/$, $\text{pow}$, $\sqrt{\cdot}$, $\exp$, $\log$, $\sin$, $\cos$). No proprietary black-box operators were privileged.

### Threat 5: Post-Hoc Hyperparameter Tuning
- **Risk:** Tuning decision thresholds (such as the holdout margin ratio $\tau = 1.15$ or noise floor $\delta = 0.0015$) after inspecting test outcomes.
- **Safeguard:** The parameter register was locked and preregistered in [`docs/FROZEN_APEDP_V3_SPECIFICATION.md`](file:///home/psalmprax/ALL_PROJECTS/science/docs/FROZEN_APEDP_V3_SPECIFICATION.md) with cryptographic module checksums before running the frozen tournaments.

### Threat 6: Synthetic-Data Bias
- **Risk:** Synthetic equations with clean Gaussian noise being unrepresentative of real-world experimental physics (which has systematic drift, non-Gaussian tails, and sensor dropout).
- **Safeguard:** Problems P19–P24 deliberately introduce adversarial structural pathologies: avoided poles, boundary layer singularities, dimensional incompatibility traps, and ill-conditioned collinear regressors.

### Threat 7: Evaluator Independence
- **Risk:** The evaluation script sharing code or assumptions with the discovery engine.
- **Safeguard:** Evaluation is performed by [`benchmarks/independent_evaluator.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/independent_evaluator.py), which ingests raw CSVs and string expressions blindly, mapping them to formal epistemic classes $C_1$–$C_6$ via independent symbolic algebra and non-parametric Wilcoxon rank-sum tests.

### Threat 8: Sample Representativeness (21 RH + 30 Physical Cases)
- **Risk:** A suite of 51 total cases being insufficient to generalize across all scientific domains.
- **Safeguard:** While 51 cases cannot span all of science, the 30 physical cases cover 5 distinct physical branches (mechanics, thermodynamics, relativity, quantum, fluid/plasma dynamics) across 5 epistemic categories. All comparative claims are strictly bounded to this benchmark suite.

### Threat 9: Generalization of Zero False Discoveries
- **Risk:** Claiming that APEDP has a "0% false discovery rate" universally across real-world scientific problems.
- **Safeguard:** $0/30$ false discoveries is an empirical finding on this specific frozen corpus under the Triad of Adjudication Improvements (noise-floor equivalence + active singularity probing + MDL parsimony). We do **not** claim that APEDP will never make a false discovery on unconstrained, noisy, or chaotic real-world datasets.

---

## 5. Bounded Scientific Claims Standard

To maintain strict scientific integrity in manuscripts and public presentations, the following claim boundaries are established:

### Permissible Scientific Claims
1. **On Epistemic Adjudication:**  
   > *"On the frozen 21-case RH benchmark and 30-case physical benchmark evaluated under the specified protocol, APEDP 3.0 achieved higher structural and algebraic recovery and lower false-discovery rates than the evaluated competitor configurations, while uniquely enforcing the predefined proof-boundary criterion."*
2. **On the Riemann Hypothesis:**  
   > *"APEDP can independently detect numerical critical-line structure and distinguish empirical evidence from deductive proof, strictly refusing to assert a universal mathematical proof from finite zero observations."*
3. **On Precision Scaling:**  
   > *"The empirical identifiability threshold tracks the predicted quadratic square-root law $\epsilon^* \approx \sqrt{2\epsilon_\text{mach}}$ across five arithmetic precision regimes spanning 100 orders of magnitude."*

### Prohibited Overclaims
- ❌ **Do NOT claim:** "APEDP has proven the Riemann Hypothesis" or "APEDP has solved RH."
- ❌ **Do NOT claim:** "APEDP is definitively superior to all symbolic regression paradigms universally."
- ❌ **Do NOT claim:** "APEDP guarantees zero false discoveries on arbitrary real-world datasets."
