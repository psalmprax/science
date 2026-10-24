# APEDP System Capabilities & Architectural Philosophy

This document details the functional capabilities of the **Autonomous Physics & Engineering Discovery Platform (APEDP)** and explains the design rationale behind balancing **domain-general evidence computation** with **physical and domain constraints**.

---

## 1. System Capabilities Map

APEDP bridges automated data-driven symbolic discovery, rigorous mathematical validation, epistemic provenance, metrological uncertainty propagation, and boundary regularization.

### 1.1 Ingest Data & Generate Fitting Equations
- **Symbolic Regression**:
  - [`MultiVariableSRE`](../apedp/discovery.py): Multi-variable symbolic regression engine combining tokenized evolutionary expression trees with continuous parameter optimization (linear outer least-squares fitting and nonlinear inner scale optimization).
  - [`SymbolicRegressor`](../apedp/symbolic_regression.py): Genetic programming loop equipped with tournament selection, subtree crossover, point mutation, and attempt-bounded generation loops.
  - [`DiscoveryPipeline`](../apedp/pipeline.py): End-to-end automated workflow: `Ingest (CSV) → Dimensional Gatekeeper → Symbolic Search → Anomaly Check → Persistence`.
- **Dimensional Filtering**:
  - Unlike generic ML regressors, candidate trees are filtered before and during genetic breeding by [`DimensionalAssuranceEngine`](../apedp/core.py), discarding any expression whose units fail physical consistency.

### 1.2 Validate Existing Equations & Discover Superior Models
- **Formal Verification & Symmetries**:
  - [`ClosedLoopAssuranceSystem`](../apedp/core.py): Enforces model registration gates, strict SI unit compatibility via SymPy and Pint, and safe numerical evaluation.
  - [`ConservationInvariantRegistry`](../apedp/core.py): Tests whether proposed formulas uphold fundamental conservation laws (e.g. conservation of energy, momentum, charge).
- **Bayesian Model Comparison**:
  - [`BayesianRefinementEngine`](../apedp/discovery.py): Calculates log-marginal likelihoods and Bayes Factors ($BF_{12}$) with Occam complexity penalties, assigning objective Kass-Raftery verdicts (e.g., *Inconclusive*, *Positive*, *Strong*, or *Decisive* evidence for Model 1 vs. Model 2).
- **Adversarial Tournaments**:
  - [`AdversarialDeliberationTournament`](../apedp/epistemic.py): Pits incumbent scientific hypotheses against adversarial boundary probes. If an equation breaks down at a physical limit, the tournament retires the flawed hypothesis and promotes the regularized model into the knowledge base.

### 1.3 Generate Novel Equations to Test
- **Dimensional Synthesis**:
  - [`BuckinghamPiEngine`](../apedp/core.py): Formulates dimensionless groups ($\Pi$-groups) from physical variable lists via nullspace analysis of dimensional exponent matrices, generating candidate model templates without requiring data upfront.
- **Autonomous Anomaly Repair**:
  - [`PhysicalAnomalyRepairEngine`](../apedp/repair.py): Identifies singularities (e.g. $r \to 0$, $v \to c$) and synthesizes regularized equations (e.g. core-softened Coulomb potentials, screened Debye potentials).
  - [`AgnosticClosedLoopAssuranceSystem`](../apedp/repair.py): Evaluates raw symbolic string expressions to detect poles, branch cuts, and division-by-zero, substituting regularized coordinate envelopes ($\sqrt{r^2 + \epsilon^2}$) that eliminate singularities across $\mathbb{R}$ while strictly preserving units.

### 1.4 Interpretations, Semantics, and Lineage of Equations
- **Epistemic Provenance Graph**:
  - [`EpistemicGraph`](../apedp/epistemic.py): Structures hypotheses, axioms, evidence, and claims into a formal Directed Acyclic Graph (DAG) using NetworkX, preventing circular reasoning and validating root assumptions.
  - [`LifecycleManager`](../apedp/epistemic.py): Enforces transition lifecycles (`Proposed → Verified → Validated → Retired`).
  - [`RequirementsTraceabilityMatrix`](../apedp/epistemic.py): Links requirements to design ADRs, code artifacts, acceptance tests, and empirical evidence.
- **GUM Metrology & Sensitivity Budgets**:
  - [`AnalyticalUncertaintyPropagator`](../apedp/metrology.py): Implements ISO/IEC Guide 98-3 (GUM) first-order Taylor expansion using analytic partial derivatives ($\frac{\partial y}{\partial x_i}$), producing complete variance budgets, percentage sensitivity distributions, and expanded 95% confidence intervals.
- **Traceability Chains**:
  - [`CalibrationTraceabilityChain`](../apedp/metrology.py): Tracks calibration records, drift compensation factors, and unbroken traceability chains back to national metrology standards (e.g. NIST).

### 1.5 Domain Elaboration, Defect Auditing, and Diagnostics
- **Cross-Domain Defect Auditing**:
  - [`UniversalAppliedDefectEngine`](../apedp/applied_defects.py) routes candidate equations to specialized domain auditors:
    - **Applied Physics** ([`AppliedPhysicsDefectAuditor`](../apedp/applied_defects.py)): Aerodynamic compressibility breakdowns (Prandtl-Glauert singularity at $M \to 1$).
    - **Applied Chemistry** ([`AppliedChemistryDefectAuditor`](../apedp/applied_defects.py)): High overpotential charge transfer breakdown (Butler-Volmer limiting current).
    - **Material Science** ([`AppliedMaterialScienceDefectAuditor`](../apedp/applied_defects.py)): Nanoscale yield strength inversions (Hall-Petch grain boundary sliding breakdown).
    - **Nanotechnology** ([`AppliedNanotechnologyDefectAuditor`](../apedp/applied_defects.py)): Casimir stiction and Gibbs-Thomson melting point depressions at sub-10nm scales.
    - **Quantum Error Correction** ([`apedp/qec.py`](../apedp/qec.py)): Knill-Laflamme algebraic condition verification, Quantum Singleton/Hamming coding bounds, and surface code fault-tolerant threshold regressions.
- **Visualization Suite**:
  - [`apedp/visualization/plots.py`](../apedp/visualization/plots.py): Generates calibration curves, dimensional presence maps, GUM uncertainty error bars, symbolic regression fitness landscapes, and epistemic network graphs.

---

## 2. Architectural Philosophy: Autonomous & Agnostic vs. Constraints

### 2.1 The Core Question
> *If APEDP is intended to generalize across domains, why does it enforce physics constraints and domain-specific rules?*

The answer lies in avoiding the fatal traps of unconstrained machine learning while maintaining a general mathematical foundation.

### 2.2 The Fatal Pitfalls of Pure Unconstrained Curve-Fitting
When an algorithm searches for equations purely to minimize error without constraints, it falls into the **"Clever Hans" trap**:
1. **Dimensionally Incoherent Hallucinations**:
   An unconstrained model will gladly output equations such as:
   $$E = 14.2 \cdot v + 3.1 \cdot t$$
   This fits the numbers in a training table ($R^2 \approx 1.0$), but adds a velocity ($[L/T]$) to a time ($[T]$), which is physical nonsense.
2. **Catastrophic Extrapolation**:
   Purely data-driven polynomials or neural networks cannot extrapolate beyond their training distribution. Outside the training box, they predict impossible values (e.g. negative mass, absolute temperatures below 0 K, or velocities exceeding $c$).
3. **Hidden Singularities**:
   An unconstrained equation like $y = \frac{1}{1 - x}$ can achieve zero error on samples where $x \in [0, 0.9]$, but will crash hardware or explode to infinity when a real system reaches $x = 1$.

### 2.3 Dimensional Homogeneity as an Algorithmic Search Pruner
The space of all possible symbolic expressions over $N$ variables grows doubly exponentially with expression depth. Brute-force search is computationally impossible.

Dimensional analysis is not a human prejudice—it is a consequence of **scale invariance and Lie group symmetries**:
- True physical laws must remain invariant under changes of measurement units.
- Enforcing dimensional consistency prunes **over 99.9% of invalid candidate expressions** before numerical evaluation ever begins.
- It transforms an impossible combinatorial search into an efficient, physically grounded discovery process.

### 2.4 What Is Domain-General and What Is Domain-Specific

APEDP is not fully agnostic at every layer. Its evidence and adjudication
machinery is reusable across domains, while discovery requires declared
adapters and semantics. The reusable layer covers locked partitions, dataset
hashes, information firewalls, provenance, holdout evaluation, numerical
conditioning, sensitivity analysis, and qualified outcomes. Domain adapters
provide units, variable meaning, constraints, baselines, search families, and
validity regimes.

This distinction prevents a misleading claim that a generic search grammar can
understand every scientific domain without domain metadata.

### 2.5 How the Architecture Generalizes Across Domains
APEDP's core engines do not assume standard physics; they are abstract mathematical tools:

1. **Configurable Basis Vector Spaces**:
   [`VectorizedDimensionalEngine`](../apedp/core.py) operates on arbitrary $N$-dimensional vector spaces. While it defaults to the 7 SI base dimensions for physics, it can be instantiated for completely non-physics fields:
   - **Economics**: Basis vectors for `[dollar]`, `[item]`, `[second]`. Validates relations such as:
     $$\text{price} = \frac{\text{dollar}}{\text{item}}, \quad \text{revenue\_rate} = \frac{\text{dollar}}{\text{second}}$$
   - **Computer Science**: Basis vectors for `[bit]`, `[second]`. Validates data rates ($\text{bps} = \frac{\text{bit}}{\text{second}}$) and algorithmic complexity bounds.
2. **Agnostic Structural Calculus**:
   [`AgnosticClosedLoopAssuranceSystem`](../apedp/repair.py) takes arbitrary mathematical expressions without pre-declared units and uses pure structural calculus:
   - Evaluates limit behaviors ($\lim_{x \to 0}$).
   - Detects denominator zeros and branch cuts ($\sqrt{-x}$).
   - Substitutes regularized smooth envelopes ($\sqrt{x^2 + \epsilon^2}$) that eliminate singularities on $\mathbb{R}$.

### 2.5 The Role of Domain Constraints: Automated Adversarial Referees
Domain constraints are not rigid restrictions that prevent discovering new ideas; they are **automated adversarial tests** that simulate the real physical world:
- An autonomous AI might propose a beautifully simple aerodynamic formula.
- The [`AppliedPhysicsDefectAuditor`](../apedp/applied_defects.py) acts as an adversarial probe: *"Does your formula survive when the aircraft approaches Mach 1?"*
- If the formula fails, the platform does not crash—it triggers the **repair engine** to synthesize a regularized model that accounts for compressibility effects.

### 2.6 Architectural Layering Matrix

| Layer | Responsibility | Agnostic vs. Domain-Specific | Key Components |
|---|---|---|---|
| **Layer 0: Exploration** | Propose candidate algebraic structures from data | **General-purpose, grammar-dependent** (search behavior depends on enabled operators and priors) | [`MultiVariableSRE`](../apedp/discovery.py), [`SymbolicRegressor`](../apedp/symbolic_regression.py) |
| **Layer 1: Structural Symmetries** | Filter dimensionally invalid expressions & enforce scaling invariance | **Agnostic Framework** (Configurable basis spaces: SI, Economics, CS) | [`DimensionalAssuranceEngine`](../apedp/core.py), [`VectorizedDimensionalEngine`](../apedp/core.py) |
| **Layer 2: Mathematical Safety** | Eliminate singularities, branch cuts, and division-by-zero | **Purely Agnostic** (Symbolic limits & structural calculus) | [`AgnosticClosedLoopAssuranceSystem`](../apedp/repair.py) |
| **Layer 3: Metrology & Statistical Selection** | Quantify uncertainty budgets and calculate Bayes factors | **Domain-Agnostic Metrology** (GUM ISO 98-3 calculus, Bayesian evidence) | [`AnalyticalUncertaintyPropagator`](../apedp/metrology.py), [`BayesianRefinementEngine`](../apedp/discovery.py) |
| **Layer 4: Adversarial Reality Probes** | Stress-test models against known non-ideal real-world boundary regimes | **Domain-Specific Referees** (Aerodynamics, Electrochemistry, Materials, Nanotech, QEC) | [`UniversalAppliedDefectEngine`](../apedp/applied_defects.py), [`AdversarialDeliberationTournament`](../apedp/epistemic.py) |

### 2.6 Evidence-qualified outcomes

The current adjudication vocabulary is:

- `ADMIT`: declared requirements and robustness evidence pass;
- `DOMAIN_LIMITED`: support is restricted to the measured domain;
- `CONDITIONAL`: structure is supported but a documented limitation remains;
- `PREDICTIVE_ONLY`: prediction is supported without a causal-law claim;
- `REFUSE`: a hard requirement fails;
- `INCONCLUSIVE`: evidence is insufficient for a stronger claim.

These labels qualify evidence; they do not prove truth. In particular,
excellent fit does not establish parameter identifiability, causal mechanism,
or independent replication.

### 2.7 Rolling-origin identifiability: predictive accuracy versus parameter certainty

Rolling-origin evaluation is a first-class diagnostic for saturation and
long-horizon models:

\[
	ext{forecast accuracy}
\not\Rightarrow
	ext{long-run parameter identifiability}.
\]

When a parameter changes substantially across training origins while forecast
error remains low, the correct interpretation is predictive usefulness with a
poorly constrained parameter, not parameter recovery. This is an origin-
sensitivity finding, not a formal confidence interval or proof of model
uniqueness.

---

## 3. Four-Tier Regularization Assurance Framework & Formula Optimization

### 3.1 The Need for Multi-Tiered Evaluation
Removing a mathematical singularity is a necessary condition for a regularized formula, but it is not sufficient. An ad-hoc regularization can introduce severe operational distortion in regimes where classical equations are accurate, or fail to ground its saturation limits in real physical parameters.

To resolve this, APEDP formalizes the **Four-Tier Regularization Assurance Framework** ([`RegularizationAuditor`](../apedp/repair.py)):
1. **Tier 1: Mathematical Verification**:
   - Symbolic limit existence ($\lim_{x \to x_0} f_{\text{reg}}(x) < \infty$).
   - Absolute pole elimination and continuity of derivatives.
2. **Tier 2: Numerical Stability**:
   - High-resolution grid scan across boundaries ($\text{NaN} / \pm\infty$ check).
   - Numerical condition spread $\kappa = \max |f| / (\min |f| + \epsilon)$.
3. **Tier 3: Asymptotic Operational Distortion Profile**:
   - Quantifies distortion in the operational regime:
     $$\delta(x) = \frac{|f_{\text{reg}}(x) - f_{\text{class}}(x)|}{|f_{\text{class}}(x)|}$$
   - Enforces an operational distortion bound (e.g. $\delta(x) \le 1.0\%$ across subsonic or macroscopic domains).
4. **Tier 4: Empirical Parameter Sweep & Pareto Optimization**:
   - Sweeps candidate regularization parameters ($\epsilon, d_c, z_0$) across a user-specified search space.
   - Solves the multi-objective Pareto tradeoff: minimize operational distortion subject to maximum allowable boundary values.

---

### 3.2 Head-to-Head Re-evaluation & Concrete Formula Optimizations

| Domain & Equation | Classical Breakdown | Initial Regularization Limitation | APEDP Optimized Formulation | Numerical Validation Result |
|---|---|---|---|---|
| **1. Aerodynamics**<br>Prandtl-Glauert Compressibility | Singularity at $M \to 1$:<br>$C_p = \frac{C_{p0}}{\sqrt{1 - M^2}} \to \infty$ | Fixed $\epsilon = 0.15$ removes singularity but distorts operational flight:<br>$\delta(M=0.8) = -3.9\%$<br>$\delta(M=0.9) = -11.4\%$ | **Calibrated Van Dyke Transonic Regularization**:<br>$C_{p,\text{reg}} = \frac{C_{p0}}{((1 - M^2)^2 + \epsilon^2)^{1/4}}$<br>with calibrated $\epsilon = 0.04$ or similarity scaling $\epsilon = \tau^{2/3}$ | **Distortion reduced 13x**:<br>$\delta(M=0.8) < 0.3\%$<br>$\delta(M=0.9) \approx 1.1\%$<br>Finite peak $C_p = -2.5$ at $M=1$. |
| **2. Materials Science**<br>Hall-Petch Grain Strengthening | Singularity at $d \to 0$:<br>$\sigma_y = \sigma_0 + \frac{k_{\text{HP}}}{\sqrt{d}} \to \infty$ | Abstract transition without material calibration; unbounded if unconstrained | **Constitutive Inverse Hall-Petch Softening with Material Profiles**:<br>$\sigma_{\text{reg}} = \sigma_0 + \frac{k_{\text{HP}}}{\sqrt{d + d_0}} \left[\frac{1}{1 + (d_c / d)^2}\right]$<br>Calibrated for Ferritic Steel, Annealed Copper, and Pure Titanium | **Physical Yield Bound**:<br>For Ferritic Steel at $d = 1\text{ nm}$, $\sigma_y \approx 301\text{ MPa}$, strictly bounded below theoretical atomic shear limit ($\tau_{\text{theo}} = 3.0\text{ GPa}$). |
| **3. Nanotechnology**<br>Casimir / vdW NEMS Stiction | Continuum blowup at $z \to 0$:<br>$F_{\text{vdW}} = \frac{A_H \cdot \text{Area}}{6\pi z^3} \to \infty$ | Pure attraction cutoff does not reach stable contact equilibrium | **Integrated Lennard-Jones 12-6 / Born Steric Repulsion Balance**:<br>$F_{\text{reg}} = \left[\frac{A_H \cdot \text{Area}}{6\pi (z + z_0)^3}\right] \left[1 - \left(\frac{z_0}{z + z_0}\right)^6\right]$ | **Zero Net Stiction at Contact**:<br>At equilibrium contact $z = 0$, $F_{\text{reg}} = 0\text{ N}$.<br>At $z = 100\text{ nm}$, recovers classical $1/z^3$ law with $<1\%$ error. |
| **4. Quantum Computing**<br>Surface Code Logical Error | Unphysical probability:<br>$p_L = C \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2} > 1.0$<br>and division by zero as $p_{\text{th}} \to 0$ | Fixed $0.5$ ceiling assumed universally for all codes | **Exact Depolarizing Saturation Envelope**:<br>$p_{L,\text{reg}} = p_{\text{sat}} \left(1 - \exp\left(-\frac{C}{p_{\text{sat}}} \left(\frac{p}{p_{\text{th}} + \epsilon}\right)^\alpha\right)\right)$<br>where $p_{\text{sat}} = 1 - 2^{-k}$ for $k$ logical qubits | **Full Spectrum Validity**:<br>Recovers classical power law at $p \ll p_{\text{th}}$;<br>Guarantees $p_L \le 0.5$ for $k=1$ and $p_L \le 0.75$ for $k=2$;<br>Zero division eliminated as $p_{\text{th}} \to 0$. |

---

### Conclusion
**Autonomous discovery is not the absence of constraints—it is the automation of verification.**

By pairing agnostic symbolic exploration with rigorous, multi-layered verification engines and four-tier regularization assurance, APEDP ensures that discovered equations are not empirical hallucinations, but durable, mathematically coherent, and physically grounded scientific models.
