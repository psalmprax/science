# Pre-Registration Protocol: APEDP-Bench 250 (Scaled Multi-Domain Epistemic Benchmark)

**Document ID:** APEDP-PREREG-2026-05  
**Version:** 1.0.0 (FROZEN PRIOR TO SYNTHETIC DATA GENERATION)  
**Registered Date:** 2026-09-24  
**Primary Invariants:** Frozen Stage A/B/C Gates (`INV-1` through `INV-10`), Dual-Budget Fairness Axis ($N_\text{eval}$ vs. $T_\text{wall}$), 11-Dimensional Profile Metric $\mathbf{M}$, Zero Ground-Truth Contamination.

---

## 1. Executive Scientific Motivation

The initial 30-problem preregistered benchmark established that:
1. Conventional symbolic regression baselines incur severe false discovery rates ($36.7\%$) on observationally ambiguous and singular systems.
2. Wrapping competitors with APEDP's Scientific Decision Layer reduces false discoveries to $\le 13.3\%$, disentangling search generator recall from epistemic precision.
3. APEDP 3.0 achieved $100\%$ recovery on ground-truth unique laws ($17/17$) and $83.3\%$ overall recovery with $10^{-16}$ OOD median extrapolation error.

To prove that these results are not small-sample artifacts or narrow physics domain biases, this pre-registration protocol formally specifies **APEDP-Bench 250**: a scaled benchmark of **250 balanced scientific discovery problems** spanning 5 distinct empirical domains, 5 epistemic classes, and 5-seed noise replications ($3,750$ evaluation instances).

---

## 2. Benchmark Corpus Taxonomy (250 Problems)

### 2.1 Domain Partition (50 Problems per Domain)
1. **Classical Mechanics, Gravitation & Orbital Astrophysics ($\mathcal{D}_1$):**
   - Central potentials, Keplerian orbits, relativistic perihelion precession, non-inertial Coriolis/centrifugal mechanics, coupled pendulums, three-body Lagrange points.
2. **Thermodynamics, Statistical Mechanics & Chemical Kinetics ($\mathcal{D}_2$):**
   - Equation of state manifolds (van der Waals, Redlich-Kwong, Dieterici), heat capacities (Einstein/Debye solids, Shomate polynomials), Arrhenius/Eyring kinetics, Michaelis-Menten enzymes, Langmuir/BET adsorption isotherms.
3. **Electrodynamics, Wave Mechanics & Optics ($\mathcal{D}_3$):**
   - Liénard-Wiechert potentials, skin depth, Fresnel reflection/transmission coefficients, wave packet dispersion, Drude-Lorentz optical conductivity, diffraction gratings.
4. **Fluid Dynamics, Hydrodynamics & Transport Phenomena ($\mathcal{D}_4$):**
   - Poiseuille/Couette viscous flow, boundary layer profiles (Blasius), drag coefficients (Stokes, Schiller-Naumann), Rayleigh-Bénard thermal convection, turbulent dissipation scales (Kolmogorov).
5. **Condensed Matter, Non-Newtonian Rheology & Biophysics ($\mathcal{D}_5$):**
   - Carreau/Bingham non-Newtonian viscosity, Hall-Petch grain boundary strengthening, Bethe-Bloch ion stopping power, Neo-Hookean/Mooney-Rivlin elasticity, Hodgkin-Huxley gating variables.

### 2.2 Epistemic Difficulty Classes ($5 \times 50 = 250$ Problems)
Each domain $\mathcal{D}_k$ is populated with exactly 10 problems per difficulty class:
- **Class 1 (Canonical Separable Physics):** Exact analytic, dimensionally consistent, identifiably isolated law. Expected State: `UNIQUE_SUPPORT` ($C_1$).
- **Class 2 (Observational Equivalence & Narrow Baselines):** Indistinguishable mathematical formulations over the observation window (e.g. Taylor series vs transcendental, linear vs saturating). Expected State: `MULTI_HYPOTHESIS` ($C_2$).
- **Class 3 (Asymptotic, Coordinate-Coupled & Extrapolation Traps):** Regimes where in-sample MSE strongly favors naive unconstrained polynomials that diverge catastrophically out-of-domain. Expected State: `UNIQUE_SUPPORT` ($C_1$) or `MULTI_HYPOTHESIS` ($C_2$).
- **Class 4 (Pathological Singularities, Poles & Avoided Traps):** Data collected near poles, non-analytic cusps, or division-by-zero boundaries. Expected State: `INSUFFICIENT_EVIDENCE` ($C_3$) or rejection.
- **Class 5 (Multivariable, Non-Separable & Residual Continuum):** Complex physics where simple low-dimensional closed-form formulas do not exist, requiring hybrid symbolic-neural discovery. Expected State: `MISSING_STRUCTURE` ($C_5$) or `HYBRID_PARADOX`.

---

## 3. Replication & Noise Sensitivity Protocol

To test the robustness and boundary stability of the epistemic framework, each of the 250 problems is instantiated under **15 distinct evaluation configurations**:
1. **Multi-Seed Replication:** 5 deterministic pseudorandom seeds:
   - $S \in \{42, 137, 2026, 8191, 104729\}$
2. **Noise Regimes:**
   - **Clean:** $\sigma_\text{noise} = 0.0$ (analytic ground truth verification)
   - **Low Noise:** $\sigma_\text{noise} = 0.01$ ($1\%$ Gaussian observation uncertainty)
   - **High Noise:** $\sigma_\text{noise} = 0.05$ ($5\%$ Gaussian observation uncertainty)
3. **Data Splits per Problem Instance:**
   - Training ($N = 50$, randomized in domain bounds)
   - Validation ($N = 25$, independent draw)
   - Near-OOD ($N = 25$, extrapolated $1.25\times$ to $1.5\times$ beyond training boundary)
   - Far-OOD ($N = 25$, extrapolated $2.0\times$ to $5.0\times$ beyond boundary)

---

## 4. Pre-Registered Hypotheses for APEDP-Bench 250

1. **Hypothesis H1 (Dimensional Generalization):**
   APEDP's Dimensional Constant Synthesis layer will maintain $100.0\%$ dimensional validity without generating spurious physical false positives across higher-dimensional variable spaces ($D \ge 4$).

2. **Hypothesis H2 (Sensitivity–Abstention Invariant):**
   The Stage C decision machine will achieve $\ge 85\%$ unique law recovery on Class 1 across all 5 scientific domains, while maintaining $\ge 80\%$ abstention rate (`MULTI_HYPOTHESIS` or `INSUFFICIENT_EVIDENCE`) on Class 2 and Class 4 problems, confirming that parsimony and ambiguity guards are scale-invariant.

3. **Hypothesis H3 (Common Abstention Portability):**
   When external search paradigms (AI Feynman, LLM-SR, gplearn) are wrapped with the APEDP decision layer on the 250-problem suite, competitor false discovery rates will be capped at $\le 15\%$, while APEDP will retain a statistically significant OOD extrapolation advantage ($p < 10^{-6}$, Holm-corrected) due to its hybrid neural-adversarial reference.

---

## 5. Pre-Registered Metric Specification

The tournament evaluation must compute the full **11-Dimensional Profile Vector**:
$$\mathbf{M} = \Big( R_\text{exact},\; R_\text{algebraic},\; R_\text{structural},\; R_\text{numerical},\; R_\text{physical},\; R_\text{rec},\; R_\text{false},\; P_\text{epi},\; E_\text{OOD},\; R_\text{dim},\; T_\text{wall},\; N_\text{eval} \Big)$$

- **No Single Scalar Aggregation:** Results must be presented as vector profiles and 6×6 Epistemic Confusion Matrices.
- **Blinded Scoring:** Candidate solutions must be scored strictly through [`benchmarks/independent_evaluator.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/independent_evaluator.py) without allowing search algorithms access to the hidden test splits or ground truth formulas.
