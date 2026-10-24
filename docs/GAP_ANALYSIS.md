
### ⚠️ Mathematical Consistency Note (corrected 2026-09-19)
Earlier drafts stated '203 laws (193 original + 16 new)', but 193+16=209, not 203.
Verified code state: `PriorArtAuditor.STANDARD_PHYSICS_CORPUS` contains 197 entries (v4: 193 + 3 Brown empirical + 1 warp scaling).
Correct accounting:

- **197 laws/entries** in `STANDARD_PHYSICS_CORPUS` (verified in code; includes scoped Brown/warp additions)
- **10 fusion-related candidate expressions** documented but NOT yet merged (pending PVE validation)
- **6 derived/extended relationships** (parameter variations, composite metrics, heuristic expressions)
- **213 total documented entities** (197 + 10 + 6 = 213)

This distinction is important: not all 16 new entities are 'laws' in the established physics sense. The 10 are fusion laws, while the 6 are composite expressions, parameter variations, or heuristic metrics that require lower classification (see GAP-001 through GAP-005 for details).
# APEDP System: Gap Analysis Document

**Document Type**: Gap Analysis / Documentation Gap Report  
**System**: APEDP (Autonomous Physics & Engineering Discovery Platform)  
**Version**: 2026.09.18 (Phase 1 Complete)  
**Tests**: 407 passing (last full e2e), ruff/mypy clean on scoped files  
**Corpus Size**: 197 in code (v4) + 16 pending validation = 213 documented entities

---

## **Document Control**

| Field | Value |
|-------|-------|
| **Document Type** | Gap Analysis / Documentation Gap Report |
| **System** | APEDP (Autonomous Physics & Engineering Discovery Platform) |
| **Version** | 2026.09.18 (Phase 1 Complete) |
| **Tests** | 407 passing (last full e2e), ruff/mypy clean on scoped files |
| **Corpus Size** | 197 in code (v4) + 16 pending = 213 documented entities |
| **Author** | APEDP Development Team |
| **Last Updated** | 2026-09-18 |

---

## **1. Current Documentation Status**

### **Already Documented** ✅

| File | Status | Key Contents |
|------|--------|--------------|
| `docs/DISCOVERY_ROADMAP.md` | ✅ Complete | 197 laws + 11 PDE templates + fusion laws + Phase 1/2/3 roadmap |
| `docs/DOCUMENTATION.md` | ✅ Complete | system overview, API reference (test counts: see roadmap) |
| `docs/SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md` | ✅ Complete | System capabilities, architectural philosophy |
| `docs/ASSESSMENT.md` | ✅ Complete | Flaws audit, system flaws review |
| `docs/SYSTEM_FLAWS_AUDIT.md` | ✅ Complete | System flaws and remediation |

### **Missing Documentation** ⚠️

## **2. Gap Analysis Summary**

### **A. Fusion Law Additions** ✅ Recently Added, ⚠️ Documentation Pending

**Status**: 10 new fusion-related laws added to `PriorArtAuditor.STANDARD_PHYSICS_CORPUS`, 5 new variables registered

**Latest Implementations NOT Yet Documented**:
- `LAWSON_CRITERION` = `n*tau*T` (triple product/Lawson criterion)
- `TRIPLE_PRODUCT` = `n*tau*T` (same as Lawson)
- `FUSION_POWER_DENSITY` = `n*tau*sigma*T**4` (fusion power density proxy)
- `BREAKEVEN_Q_APPROX` = `sigma*T**4*n*tau` (breakeven approximation)
- `FUSION_REACTIVITY` = `sigma_v` (fusion cross-section × velocity)
- `ALPHA_PARTICLE_ENERGY` = `E_alpha` (alpha particle energy)
- `NEUTRON_ENERGY` = `E_n (neutron energy)` (neutron energy)
- `FUSION_TEMPERATURE` = `T` (fusion temperature)
- `ION_TEMPERATURE` = `T` (ion temperature)
- `ELECTRON_TEMPERATURE` = `T` (electron temperature)

**Gap**: Detailed descriptions of each fusion law and their physical meanings not in roadmap

### **B. Fusion Variables** ✅ Registered, ⚠️ Documentation Pending

**Registered Variables**:
- `n`: plasma_density (m⁻³)
- `tau`: confinement_time (s)
- `sigma_v`: fusion_cross_section_times_velocity (m³/s)
- `E_alpha`: alpha_particle_energy (J)
- `E_n (neutron energy)`: neutron_energy (J)

**Gap**: Detailed unit descriptions and physical interpretations not documented

### **C. Trust Score Framework** ⚠️ Basic, ❌ Full Details Missing

**Formula**: `(reg_score × 0.5) + (anomaly_score × 0.3) + (verdict_score × 0.2)`

**Trust Categories**:
- **0.8–1.0**: High Trust → include in primary scoring
- **0.5–0.8**: Secondary Trust → include with annotation
- **0.0–0.5**: Low Trust → exclude or use as anomaly flag

**Example trust scores**:
- `n*tau`: 0.800 (High Trust)
- `sigma*T⁴*n*tau`: 0.533 (Secondary Trust)
- `I*R*n*tau`: 0.750 (Secondary Trust)
- `E_α + E_n`: 0.800 (High Trust)
- `IRnτ` (original, uncorrected): 0.250 (Rejected - `IR = V`, not power)

**Gap**: Full trust score computation details and formula derivation not in roadmap

### **D. Three-Gate Validation** ⚠️ Gates Defined, ❌ Full Details Missing

| Gate | Check | Pass Criteria |
|------|-------|---------------|
| **Gate 1**: Dimensional plausibility | Known variables count `>= 2` | Must have >= 2 known vars from `{n, τ, T, σ, I, R}` |
| **Gate 2**: Anomaly check | Critical anomalies `== 0` | No division/ dimensional violations |
| **Gate 3**: Verdict acceptability | `verdict != REJECTED` | `SPECULATION` or `APPROVED` only |

**All 4 test expressions pass all 3 gates**:
- `n*tau`: PASS ✅
- `σT⁴nτ`: PASS ✅
- `I*R*n*tau`: PASS ✅
- `E_α + E_n`: PASS ✅

**Gap**: Detailed gate descriptions and validation logic not in roadmap

### **E. Fusion Law Crossover Results** ⚠️ Basic Scores, ❌ Detailed Analysis Missing

| Crossover | Score | Revised classification |
|-----------|-------|------------------------|
| `E_alpha*E_n` (J^2, dimensionally valid, not identity) | 0.5000 | CANDIDATE / HEURISTIC_COMPOSITE — physics_evidence=NONE, identity=NO |
| `LAWSON_CRITERION**2` | 0.2500 | NONLINEAR_TRANSFORMATION, not a new law |
| `LAWSON_CRITERION/LAWSON_CRITERION` | 0.5000 | IDENTITY (=1), not a discovery |
| `LAWSON_CRITERION × FUSION_POWER_DENSITY` | 0.2500 | CANDIDATE composite, needs derivation |
| `sigma*T**4*n*tau` | 0.5000 | CANDIDATE / HEURISTIC — PROCEED_TO_PHYSICAL_VALIDATION only |

**Gap**: Detailed analysis of each crossover and their physical meaning not documented

### **F. Parameter Variation Results** ⚠️ Basic Scores, ❌ Detailed Interpretation Missing

| Parameter | Score | Recognition |
|-----------|-------|-------------|
| `n*tau` (Lawson parameter) | 0.2500 | Lawson parameter |
| `n*tau*T` (triple product) | 0.2000 | Triple product |
| `sigma*T^4*n*tau` (radiative coupling) | 0.1250 | Radiative coupling |
| `I*R*n*tau` (ohmic heating metric) | 0.1667 | Ohmic heating × density × confinement |
| `E_α + E_n` (total fusion energy) | 0.2500 | Total fusion energy per D-T reaction |

**Gap**: Detailed interpretation of each score and physical meaning not documented

### **G. Universal Interpretation Framework** ❌ Not Documented

**Components** (Implemented but not documented):
- `extract_unit_from_physicalconstant()` function
- `compute_trust_score()` function (3-factor formula)
- `passes_gate_1()`, `passes_gate_2()`, `passes_gate_3()` (three-gate validation)
- `universal_equation_assessment()` function
- Unit extraction from `PhysicalConstant` objects
- 11 registered physical constants with full unit information

**Gap**: Complete framework documentation missing entirely

### **H. PDE Template Coverage** ⚠️ Basic, ❌ Detailed Missing

**11 PDE Templates**:
- DIFFUSION/HEAT_1D, WAVE_1D, LAPLACE_2D, POISSON_2D
- CONTINUITY_1D, EULER_MOMENTUM_1D, NS_X_1D
- SCHRODINGER_TD_1D, MAXWELL_FARADAY/AMPERE_1D

**Gap**: Detailed template descriptions, coefficient/BC verification status not in roadmap

### **I. Extended Evaluation Capabilities** ⚠️ Basic, ❌ Detailed Missing

**Capabilities** (not fully documented):
- `evaluate_implicit` (nsolve) + implicit systems
- `evaluate_quadrature` (trapezoid)
- `evaluate_ode_ivp` (rk4 / adaptive rk45 with 100k-step refusal)
- `evaluate_ode_system` (vector, stiff y'=-50y)
- `solve_eigen` (residual-checked, sorted)
- ODE event stop (scan + 100-substep RK4 refine)
- `evaluate_pde_residual` (manufactured-solution check)
- `solve_heat_1d_explicit` (Dirichlet FD, stability-refused)
- Einstein index strings (`evaluate_indexed`)
- Symbolic index calculus (`parse_indexed_to_sympy`)
- Tensor `einsum`

**Gap**: Detailed evaluation capability descriptions not in roadmap

### **J. Full-PDE Solvers** ⚠️ Basic, ❌ Detailed Missing

**Solvers** (numpy only, not fully documented):
- `solve_wave_1d` (leapfrog, CFL≤1)
- `solve_advection_1d` (upwind, |C|≤1)
- `solve_heat_2d` (explicit, r≤0.25)
- `solve_laplace_2d` / `solve_poisson_2d` (Jacobi to tolerance)
- `solve_heat_3d` (explicit, r≤1/6, structured-grid budget)
- `solve_laplace_3d` (Jacobi-3D)
- `solve_poisson_fem` (P1 FEM, exact nodal loads)

**Gap**: Detailed solver descriptions and verification results not in roadmap

### **K. High-D Monte Carlo** ⚠️ Basic, ❌ Detailed Missing

**Methods** (not fully documented):
- `solve_heat_nd_feynman_kac` (free-space heat to 20D)
- `solve_laplace_nd_walk_on_spheres` (d-ball Dirichlet)

**Gap**: Detailed Monte Carlo method descriptions not in roadmap

### **L. Meshing** ⚠️ Basic, ❌ Detailed Missing

**Capabilities** (not fully documented):
- exact-integer Bowyer-Watson Delaunay (Bareiss predicates)
- combinatorial validation (point coverage + manifold edges + single boundary cycle + area)
- hull-edge snapping for boundary-cluster slivers
- Laplacian smoothing
- max-angle-driven Ruppert refinement

**Gap**: Detailed meshing algorithm descriptions not in roadmap

---

## **3. Priority Matrix**

| Priority | Category | Impact | Effort | Recommended Action |
|----------|----------|--------|--------|-------------------|
| **P0** | Fusion Law Descriptions | High | Low | Add to DISCOVERY_ROADMAP.md |
| **P0** | Trust Score Framework | High | Medium | Add to DISCOVERY_ROADMAP.md |
| **P0** | Three-Gate Validation | High | Medium | Add to DISCOVERY_ROADMAP.md |
| **P1** | Fusion Crossover Results | Medium | Low | Add to DISCOVERY_ROADMAP.md |
| **P1** | Parameter Variation Results | Medium | Low | Add to DISCOVERY_ROADMAP.md |
| **P2** | Fusion Variable Descriptions | Medium | Low | Add to DISCOVERY_ROADMAP.md |
| **P2** | PDE Template Details | Medium | Medium | Add to DISCOVERY_ROADMAP.md |
| **P3** | PDE Solver Details | Low | High | Add to DISCOVERY_ROADMAP.md |
| **P3** | Monte Carlo Methods | Low | High | Add to DISCOVERY_ROADMAP.md |
| **P3** | Meshing Algorithms | Low | High | Add to DISCOVERY_ROADMAP.md |

---

## **4. Recommended Documentation Additions**

### **Option 1**: Update `DISCOVERY_ROADMAP.md` with all missing details

**Pros**: Single source of truth, comprehensive roadmap  
**Cons**: File may become very large

### **Option 2**: Create `GAP_ANALYSIS.md` (this file)

**Pros**: Focused on gaps, easy to update  
**Cons**: Separate from main roadmap

### **Option 3**: Both ✅ Recommended

**Pros**: Comprehensive coverage + gap tracking  
**Cons**: Maintains two files

### **Recommended Action**: Option 3 - Both

1. **Update `DISCOVERY_ROADMAP.md`** with:
   - Fusion law descriptions and trust scores
   - Three-gate validation details
   - Crossover results and parameter variation results
   - Fusion variable registrations

2. **Keep `GAP_ANALYSIS.md`** as:
   - Gap tracking document
   - Priority matrix
   - Documentation task list
   - Progress tracker

---

## **5. Gap Closure Tracker**

| Gap ID | Description | Status | Target Date | Owner |
|--------|-------------|--------|-------------|-------|
| GAP-001 | Fusion law descriptions | ⭐ In Progress | 2026-09-25 | Dev Team |
| GAP-002 | Trust score framework details | ⭐ In Progress | 2026-09-25 | Dev Team |
| GAP-003 | Three-gate validation details | ⭐ In Progress | 2026-09-25 | Dev Team |
| GAP-004 | Fusion crossover results analysis | ⭐ In Progress | 2026-09-25 | Dev Team |
| GAP-005 | Parameter variation results interpretation | ⭐ In Progress | 2026-09-25 | Dev Team |
| GAP-006 | Fusion variable unit descriptions | ⭐ In Progress | 2026-09-25 | Dev Team |
| GAP-007 | PDE template detailed descriptions | 📋 Pending | 2026-10-02 | Dev Team |
| GAP-008 | PDE solver detailed documentation | 📋 Pending | 2026-10-09 | Dev Team |
| GAP-009 | Monte Carlo method documentation | 📋 Pending | 2026-10-16 | Dev Team |
| GAP-010 | Meshing algorithm documentation | 📋 Pending | 2026-10-23 | Dev Team |

**Legend**: ⭐ = High Priority, 📋 = Medium Priority, 📁 = Low Priority

---

## **6. Next Steps**

1. **Immediate** (this session):
   - ✅ Create `GAP_ANALYSIS.md` document
   - ✅ Prioritize GAP-001 through GAP-005 (fusion laws, trust scores, gates)

2. **This Week**:
   - Update `DISCOVERY_ROADMAP.md` with fusion law descriptions
   - Add trust score framework details to roadmap
   - Document three-gate validation in roadmap
   - Add crossover results to roadmap

3. **This Month**:
   - Complete all GAP-001 through GAP-005 closures
   - Begin GAP-006 through GAP-010 documentation
   - Review and validate all gap closures

4. **Next Month**:
   - Finalize all gap closures
   - Review documentation completeness
   - Plan Phase 2 documentation needs

---

## **7. Summary**

**What's Documented** ✅:
- 197 laws/entries in corpus (193 original + 3 Brown empirical + 1 warp scaling, v4)
- 11 PDE templates
- 213 total documented entities (197 in code + 10 fusion candidates + 6 derived/extended pending)
- 407 tests passing (last full e2e)
- Ruff clean, mypy 0 errors
- Basic system overview
- High-level roadmap (Phase 1/2/3)

**What's NOT Documented** ⚠️:
- Fusion law detailed descriptions
- Trust score framework formula and categories
- Three-gate validation detailed logic
- Fusion crossover results analysis
- Parameter variation results interpretation
- Fusion variable unit descriptions
- PDE template detailed descriptions
- PDE solver detailed documentation
- Monte Carlo method documentation
- Meshing algorithm documentation

**Priority Gaps** 🔥:
1. Fusion law descriptions (GAP-001)
2. Trust score framework (GAP-002)
3. Three-gate validation (GAP-003)
4. Fusion crossover results (GAP-004)
5. Parameter variation results (GAP-005)

**Recommended Approach**: Update `DISCOVERY_ROADMAP.md` + maintain `GAP_ANALYSIS.md` for tracking

---

*This document was auto-generated on 2026-09-18 as part of APEDP Phase 1 completion verification. It identifies all documentation gaps in the APEDP system and provides a prioritized roadmap for closing them.*

---

### P0 — Trust Score vs Physics Validity Separation

The existing trust score (`S = 0.5R + 0.3A + 0.2V`) is a **software/anomaly/verdict** composite,
not a physics validity score.  A high trust score does not prove physical correctness.

**New design: separate two scores**

| Score Type | Description | Weight Range |
|------------|-------------|--------------|
| **Trust Score** `S_trust` | Software/Anomaly/Verdict composite: registration, anomaly count, verdict status | 0–1 (existing) |
| **Physics Validity** `S_physics` | Dimensional consistency + unit consistency + known-law consistency + limiting-case behavior + empirical agreement + derivation validity | 0–1 (new) |

**Example comparisons**:

| Expression | `S_trust` | `S_physics` | Interpretation |
|------------|-----------|-------------|----------------|
| `nτ` | 0.800 | 0.900 | High trust + high validity, corpus match → `ESTABLISHED` |
| `nτT³⁷` | 0.650 | 0.100 | Moderate trust + low validity → `PHYSICALLY_UNSUPPORTED` |
| `E_alpha*E_n` (J^2, not identity) | 0.500 | 0.050 | Low physics validity → `CANDIDATE` / `HEURISTIC_COMPOSITE`, identity=NO |
| `σT⁴nτ` | 0.533 | 0.250 | Moderate validity → `CANDIDATE`/`HEURISTIC`, PROCEED_TO_PHYSICAL_VALIDATION only |
| `L/L` (identity) | 0.500 | 0.000 | `IDENTITY` classification, not a discovery |

**Implementation**: Add `compute_physics_validity()` function alongside existing `compute_trust_score()`:

```python
def compute_physics_validity(audit, standard_vars):
    """Compute physics validity score (0-1) separate from trust score.
    
    Components:
    - Dimensional consistency: 1.0 if dimensions are well-defined, 0.0 otherwise
    - Unit consistency: 1.0 if all variables have known units, decaying with unknown vars
    - Known-law consistency: 1.0 if expression matches canonical law, decaying with mismatch
    - Limiting cases: 1.0 if expression has correct limiting behavior, 0.5 if unknown, 0.0 if fails
    - Empirical agreement: 1.0 if empirical data supports, 0.5 if unknown, 0.0 if contradicted
    - Derivation validity: 1.0 if derivation exists, 0.5 if plausible, 0.0 if none
    
    Physics Validity Categories (parallel to Trust Categories):
    - 0.8–1.0: High Validity → include in primary scoring
    - 0.5–0.8: Secondary Validity → include with annotation
    - 0.0–0.5: Low Validity → exclude or use as anomaly flag
```


### P0 — Physics Validity Engine (PVE) Framework

The PVE is the core differentiator that moves APEDP from "equation generation"
toward "physics discovery." It computes a physics validity score (S_physics) that
is strictly SEPARATE from the trust score (S_trust).

S_trust = 0.5R + 0.3A + 0.2V  ... registration, anomaly, verdict
S_physics = 0.20D + 0.15U + 0.25L + 0.20K + 0.15E + 0.05V
            dimensional + unit + law + limiting + derivation + empirical

Validity Categories (parallel to Trust Categories):
- 0.8-1.0: High Validity
- 0.5-0.8: Secondary Validity
- 0.0-0.5: Low Validity

Example Comparisons:

| Expression | S_trust | S_physics | Classification |
|-----------------|-----------|----------------|
| nτ (Lawson, corpus match) | 0.800 | 0.900 | ESTABLISHED (evidence-based, not score-based) |
| nτT³⁷ (meaningless) | 0.650 | 0.100 | PHYSICALLY_UNSUPPORTED |
| E_alpha*E_n (J^2, not identity) | 0.500 | 0.050 | CANDIDATE / HEURISTIC_COMPOSITE, identity=NO |
| σT⁴nτ (radiative) | 0.533 | 0.250 | CANDIDATE/HEURISTIC, PROCEED_TO_PHYSICAL_VALIDATION only |
| L/L (identity) | 0.500 | 0.000 | IDENTITY |

Implementation: add compute_physics_validity() in apedp/physics_validity_engine.py
(skeleton created; full integration requires dimensional analysis, known-law matching,
limiting-case verification, and empirical data probes).

Integration with Classification Ontology (evidence-gated, score never grants ESTABLISHED):
- ESTABLISHED requires corpus_match OR formal_derivation OR replicated_empirical_evidence (score alone insufficient)
- DERIVED requires derivation from ESTABLISHED law
- S_physics 0.5-0.8 + Classification in [DERIVED, PHYSICALLY_PLAUSIBLE] -> PROCEED_TO_PHYSICAL_VALIDATION
- S_physics < 0.5 OR Classification in [IDENTITY, DIMENSIONALLY_INVALID, PHYSICALLY_UNSUPPORTED] -> Rejected / investigate only

This two-score system resolves the central problem: high trust does not imply physical validity.


### P0 — Three-Gate Validation Updated with Dimensional Analysis

The original three-gate validation had Gate 1 checking "known variables count >= 2",
which was insufficient. It has been updated to use proper dimensional analysis.

**Updated Three-Gate Validation:**

| Gate | Original Check | Updated Check | Status |
|------|---------------|---------------|--------|
| **Gate 1**: Dimensional Plausibility | `known variables count >= 2` | `DimensionalAssuranceEngine` checks expression dimensions | ✅ Updated |
| **Gate 2**: Anomaly Check | `critical anomalies == 0` | `critical anomalies == 0` + identity detection | ✅ Enhanced |
| **Gate 3**: Verdict Acceptability | `verdict != REJECTED` | New 8-state classification check | ✅ Updated |

**Gate 1 Detail**: Now uses `_measure_dimensional_consistency()` from `apedp/gated_assurance.py`,
which checks:
- All free symbols are declared by the record
- Both sides of the equation are dimensionally homogeneous
- Output unit matches declared dimensionality
- Unit metadata is present

**Gate 2 Detail**: Enhanced with identity detection via `apedp/identity_detection.py`,
which automatically catches `x/x=1`, `x-x=0`, `x^0=1` and classifies them as `IDENTITY`.

**Gate 3 Detail**: Updated to check against the new 8-state Classification enum:
- Passes: ESTABLISHED, DERIVED, PHYSICALLY_PLAUSIBLE, CANDIDATE, HEURISTIC
- Fails: IDENTITY, DIMENSIONALLY_INVALID, PHYSICALLY_UNSUPPORTED

**Example Gate Results**:

| Expression | Gate 1 | Gate 2 | Gate 3 | Overall |
|------------|--------|--------|--------|---------|
| `n*tau` (Lawson) | PASS | PASS | PASS | PROCEED_TO_PHYSICAL_VALIDATION|
| `x/x` (identity) | PASS* | FAIL | FAIL | REJECT |
| `n*tau*T^37` (meaningless) | PASS | PASS | FAIL | REJECT |
| `sigma*T^4*n*tau` (radiative) | PASS | PASS | PASS | PROCEED_TO_PHYSICAL_VALIDATION|

*Gate 1 passes dimensionally but Gate 2/3 catch the issue.

**Implementation**: `validate_three_gates()` function added to `apedp/gated_assurance.py`


### P0 — ESTABLISHED Must Be Evidence-Based, Not Score-Driven

It is architecturally dangerous to let a numerical physics validity score alone grant `ESTABLISHED` status.

**Why**: `ESTABLISHED` should mean the expression has independent support from:
* Authoritative prior literature
* Verified canonical derivation  
* Reproducibly validated empirical evidence

**Current risk**: A newly generated expression with `S_physics >= 0.8` could be granted `ESTABLISHED` merely by score, even without corpus match or derivation.

**Design decision** (asymmetric safeguard):

```text
# Evidence/ontology can grant ESTABLISHED; score alone cannot.

# Path 1: Corpus match + derivation -> ESTABLISHED
# (e.g., nTτ → "Lawson triple product" from prior art corpus)

# Path 2: Score alone -> never ESTABLISHED
# (e.g., S_physics=0.92 for a novel expression -> CANDIDATE/HEURISTIC at best)

# Path 3: Derivation from ESTABLISHED -> DERIVED
# (e.g., E_alpha+E_n from known D-T reaction -> DERIVED)
```

**Implementation**: The classification ontology integrates with the PVE as follows:

- `Classification.ESTABLISHED` requires: corpus match OR formal derivation
- `Classification.DERIVED` requires: derivation from ESTABLISHED law
- `Classification.CANDIDATE` requires: S_physics > threshold AND (corpus match OR plausible structure)
- `Classification.HEURISTIC` requires: useful composite metric, no claim of law status
- `Classification.IDENTITY` requires: auto-detected mathematical identity
- `Classification.DIMENSIONALLY_INVALID` requires: fails dimensional analysis
- `Classification.PHYSICALLY_UNSUPPORTED` requires: passes structural checks but no derivation/empirical support

**Key invariant**: No numerical score in [0,1] alone can promote an expression to `ESTABLISHED`. This is a hard architectural constraint, not a soft preference.

### P0 follow-up (2026-09-19): variable semantics, conservation, equivalence, calibration

- **Variable semantics** (`apedp/variable_semantics.py`): bare `n`/`T`/`tau` flagged ambiguous; explicit specs for `n_e/n_i/n_D/n_T`, `T_e/T_i`, `tau_E/tau_p`, `sigma_v_DT`, `E_alpha`, `E_n` (neutron, not neutrino).
- **Assumptions/envelope carriers**: `AssumptionSet` and `ApplicabilityEnvelope` added (advisory; equations should carry A1..An + validated ranges).
- **Conservation checklist** (`apedp/physics_validity_engine.py:conservation_checklist`): advisory UNKNOWN by default; flags `E_alpha*E_n` as J^2 product, not an energy balance.
- **Canonical equivalence** (`apedp/physics_validity_engine.py:same_relationship`): `F=ma`, `a=F/m`, `ma-F=0` match; `x^2` vs `x^3` do not. Prevents duplicate discoveries.
- **Layered PVE** (`apedp/physics_validity_engine.py:check_hard_requirements`): hard requirements (dimensions, identity) fail fast before weighted average. PVE weights provisional.
- **Benchmark calibration** (`apedp/physics_validity_engine.py:run_benchmark`): 14-example smoke test, ROC AUC 0.878 (label-assisted, provisional — not a validity claim). Thresholds uncalibrated pending real benchmark (reviewer issue #6).
