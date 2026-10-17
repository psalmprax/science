# Sparrow Quantum Deterministic Photonics Blueprint & Epistemic Audit

## 1. Overview & Provenance

This document records the theoretical formulation, physical implementation, and epistemic referee audit of the deterministic photonics quantum computing paradigm pioneered by **Sparrow Quantum** and the **Hy-Q Center for Hybrid Quantum Networks (Niels Bohr Institute, University of Copenhagen)**.

* **Presentation:** *"Explained: How Deterministic Photonics Gets Us to Fault-Tolerant Quantum Computing"* (Speaker: Ming-Lai Chan, Senior Quantum Photonics Engineer, Sparrow Quantum). YouTube ID: `H2Xmsyhewtg`.
* **Foundational Blueprint:** *“Practical blueprint for low-depth photonic quantum computing with quantum dots”*, Ming Lai Chan, Aliki Anna Capatos, Peter Lodahl, Anders S. Sørensen, Stefano Paesani (*npj Quantum Information*, 2025 / `arXiv:2507.16152`).
* **Repository Implementation:** [`apedp_dyn/quantum/sparrow_photonics.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/quantum/sparrow_photonics.py).
* **Test & Verification Suite:** [`tests/test_dyn_sparrow_deterministic_photonics.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_sparrow_deterministic_photonics.py).

---

## 2. Theoretical Breakdown: Probabilistic vs. Deterministic Photonics

### 2.1 The Probabilistic Multiplexing Bottleneck
Standard linear optics quantum computing (LOQC / FBQC platforms such as PsiQuantum) relies on **Spontaneous Parametric Down-Conversion (SPDC)** in nonlinear crystals (e.g. KTP, BBO, or silicon waveguides). Because photon pair generation is a spontaneous Poissonian process, pump power must be suppressed ($p_{\text{pair}} \ll 0.1$) to prevent higher-order multi-pair noise.

Consequently, generating an $n$-photon entangled cluster state scales exponentially:
$$P_{\text{SPDC}}(n) \approx (p_{\text{pair}} \cdot \eta_{\text{sys}})^n$$

To counteract this, SPDC architectures require massive spatial multiplexing trees:
$$N_{\text{modes}} \propto \left(\frac{1}{p_{\text{pair}}}\right)^n$$
At $n=12$, this requires millions of optical modes and thousands of high-speed active spatial optical switches. Every switch introduces insertion loss and optical depth. In photonics, optical loss is the fatal noise channel because lost photons destroy entanglement non-recoverably once loss exceeds the fault-tolerant threshold ($\sim 8\% - 10\%$).

### 2.2 Deterministic On-Demand Quantum Dot Emitters
Sparrow Quantum eliminates the multiplexing tree by using a self-assembled **In(Ga)As quantum dot (QD)** embedded in a slow-light photonic crystal waveguide:
* Coherent optical and microwave $\pi$-pulses trigger on-demand emission of indistinguishable single photons.
* Emission efficiency scales directly with outcoupling efficiency:
$$P_{\text{QD}}(n) \approx \eta_{\text{out}}^n$$
At $\eta_{\text{out}} = 93\%$, $P_{\text{QD}}(12) \approx 0.419$ (41.9%), compared to $P_{\text{SPDC}}(12) \approx 9.2 \times 10^{-19}$. The quantum dot source achieves an advantage of over **$4.5 \times 10^{17}\times$**.

---

## 3. Excitation-Based Feedback (EBF) & 100% Passive Fusions

### 3.1 Time-Bin Qubit Encoding
Rather than separating photons across millions of spatial fibers, photons are emitted sequentially in time down a single spatial waveguide mode. A single electron or hole spin trapped in the quantum dot precesses in an in-plane magnetic field ($B_{\text{ext}} \approx 2.5\ \text{T}$):
* A photon is emitted in the **early time bin** ($|E\rangle$) or **late time bin** ($|L\rangle$), spaced by $\tau_{\text{TB}} \approx 0.5\ \text{ns}$.
* Entanglement is mediated by the QD spin: the emission sequence automatically weaves a 1D time-bin cluster state:
$$|\Phi_{1D}\rangle = \bigotimes_k \left(|0\rangle_k |E\rangle_k + |1\rangle_k |L\rangle_k\right)$$

### 3.2 The EBF Mechanism (Eq. 7)
In standard Fusion-Based Quantum Computing (FBQC), when a photon loss event occurs on an entangled resource state, the measurement network must dynamically adapt by applying a single-qubit measurement rather than an entangling two-qubit Bell-state measurement. Previously, this required fast reconfigurable Mach-Zehnder interferometers (active phase shifters) in the optical circuit.

Sparrow Quantum invented **Excitation-Based Feedback (EBF)**:
Instead of shifting optical paths, the control computer adjusts the timing of the laser excitation pulse generating the photon by an intentional time delay $\tau_{\text{EBF}}$.

To preserve the quantum dot state and avoid inter-pulse crosstalk, $\tau_{\text{EBF}}$ must satisfy the fundamental timing inequality:
$$\tau_{\text{TB}} < \tau_{\text{EBF}} < \tau_{\text{int}} - \tau_\pi$$
where:
* $\tau_{\text{TB}} = 0.5\ \text{ns}$ (time-bin separation)
* $\tau_{\text{int}} = 11.83\ \text{ns}$ (laser repetition interval)
* $\tau_\pi = 4.0\ \text{ns}$ (duration of the spin-flip $\pi$-pulse)

**Consequence:**
* When $\tau_{\text{offset}} = 0$, both photons arrive synchronously at the 50:50 beamsplitter and undergo two-photon Hong-Ou-Mandel (HOM) interference, executing an entangling Type-II Bell state fusion ($P_{\text{succ}} = 50\% \cdot \eta^2$).
* When $\tau_{\text{offset}} = \tau_{\text{EBF}} = 5.0\ \text{ns}$, the two photons arrive at different times and cannot interfere ($V_{\text{HOM}} = 0$). The beamsplitter acts as a simple passive $50:50$ power distributor, routing each photon independently to a photodetector. This realizes two deterministic single-qubit $Z$-measurements ($P_{\text{succ}} = \eta^2$) with **100% passive linear optics**.

**Impact:** Reduces active phase shifters per logical qubit by $9L^2 - 4L + 1 = 2,241$ elements at $L=16$, and reduces optical depth to just 4 active phase shifters and 5–7 beamsplitters per photon.

---

## 4. Semi-Fluid Foliated Floquet Colour Code (sFFCC) Architecture

The synchronous foliated Floquet colour code integrates time-bin resource states directly into a 3D topological error-correcting lattice:

### 4.1 Resource Scaling Formulas (Code Distance $L$)
* **Quantum Dot Emitters:**
  $$N_{\text{QD}} = 6 L^2$$
  At $L=16$: $N_{\text{QD}} = 1,536$ emitters per logical qubit.
* **Photodetectors (SNSPDs):**
  $$N_{\text{det}} = 18 L^2$$
  At $L=16$: $N_{\text{det}} = 4,608$ detectors.
* **Active Optical Elements (with EBF):**
  $$N_{\text{active}} = 24 L^2 = 6,144$$
* **Encoded Fusions per Cycle:**
  $$N_{\text{fusions}} = 18 L^3 = 73,728$$
* **Logical Clock Cycle Time:**
  $$\tau_{\text{cycle}} = 6 \cdot L \cdot \tau_{\text{physical}} = 12 \cdot L \cdot \tau_{\text{rep}}$$
  For $\tau_{\text{rep}} = 13.8\ \text{ns}$ and $L=16$:
  $$\tau_{\text{cycle}} = 12 \times 16 \times 13.8\ \text{ns} = 2.6496\ \mu\text{s}$$
  Logical clock frequency: **$377.4\ \text{kHz}$** ($\sim 1,000\times$ faster than trapped-ion or neutral-atom processors).

### 4.2 Sub-Threshold Logical Error Rate Scaling
From empirical fits to numerical lattice simulations under combined noise (Chan et al., Fig. 10(a)):
$$\log_{10} p_L = \log_{10} p_0 - \Lambda \cdot L$$
* Intercept: $p_0 \approx 0.122$
* Near-term target ($\sum_i p_i / p_{\text{th}, i} = 0.62$): $\Lambda = 0.106 \implies L \approx 48$ for $p_L \le 10^{-6}$.
* Further-term target ($\sum_i p_i / p_{\text{th}, i} = 0.46$): $\Lambda = 0.318 \implies L \approx 16$ for $p_L \le 10^{-6}$, $L \approx 26$ for $10^{-9}$, $L \approx 35$ for $10^{-12}$.

---

## 5. Epistemic Referee Audit: Table 2 Fault-Tolerance Thresholds

The module [`SparrowFaultToleranceReferee`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/quantum/sparrow_photonics.py#L241-L314) enforces an 8-channel epistemic audit:

| Noise Channel | Physical Error Mechanism | Paper Threshold (Table 2) | Operating Target | Status |
|---|---|---|---|---|
| **Loss Rate** | Waveguide outcoupling, fiber coupling, detector QE | $\eta > 92.0\%$ ($p_{\text{loss}} \le 8.0\%$) | $\eta = 93.0\%$ | `PASS` |
| **Branching Ratio** | Imperfect optical cyclicity ($p_{\text{branch}} = 1/(C+1)$) | $C > 574$ ($p_{\text{branch}} \le 0.174\%$) | $C = 650$ | `PASS` |
| **Emitter-Emitter HOM** | Spectral wandering / diffusion mismatch | $\mathcal{V}_{\text{HOM}}^{ee} > 96.0\%$ | $97.0\%$ | `PASS` |
| **Single-Emitter HOM** | Phonon sideband and pure dephasing | $\mathcal{V}_{\text{HOM}}^{se} > 99.4\%$ | $99.6\%$ | `PASS` |
| **Laser Spin-Flips** | Off-resonant Raman rotation errors | $\bar{\kappa} < 1.2 \times 10^{-3}$ | $8.0 \times 10^{-4}$ | `PASS` |
| **Spin Coherence** | Transverse spin relaxation $T_2$ | $T_2 > 417 \tau_{\text{rep}} \approx 5.75\ \mu\text{s}$ | $113\ \mu\text{s}$ | `PASS` |
| **Overhauser Dephasing**| Hyperfine nuclear spin fluctuations $T_2^*$ | $T_2^* > 56 \tau_{\text{round}} \approx 1.55\ \mu\text{s}$ | $2.0\ \mu\text{s}$ | `PASS` |
| **Emitter Blinking** | Charge state trapping / dark periods | $P_A / P_D > 15.4$ | $50.0$ | `PASS` |

---

## 6. Verification Records

The complete implementation passes all unit, integration, and integrity tests:
```bash
pytest tests/test_dyn_sparrow_deterministic_photonics.py -v
```
* `test_deterministic_vs_probabilistic_scaling`: **PASSED** ($10^{17}\times$ scaling advantage confirmed).
* `test_passive_fusion_engine_and_ebf_timing`: **PASSED** (Eq. 7 timing inequality verified; passive $Z$-measurement conversion validated).
* `test_sffcc_lattice_hardware_scaling`: **PASSED** (1,536 emitters, 2,241 saved elements, $2.65\ \mu\text{s}$ cycle time, $P_L < 10^{-6}$ verified).
* `test_sparrow_fault_tolerance_referee`: **PASSED** (compliant hardware approved; 12% loss rejected with root-cause diagnostic).
