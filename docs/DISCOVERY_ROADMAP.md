# APEDP Discovery Roadmap: Certification + Data-Free Conjecture

Status: implemented (commit after this edit). 301 tests green, ruff + mypy clean.

## Coverage ladder (all equations)

- **L0 algebraic:** 193 named laws in `PriorArtAuditor` (v3+), all parse-gated: mechanics, EM/electronics, thermo/fluids/heat, waves/optics/QM/relativity, chemistry/materials, vibrations/acoustics, signals, cycles/combustion, mass transfer, electrochem/semiconductors, structures/geo/astro, turbomachinery, bio/misc rates.
- **L1 PDE templates** (`apedp/pde_templates.py`, `safe_parse_pde`): DIFFUSION/HEAT_1D, WAVE_1D, LAPLACE_2D, POISSON_2D, CONTINUITY_1D, EULER_MOMENTUM_1D, NS_X_1D, SCHRODINGER_TD_1D, MAXWELL_FARADAY/AMPERE_1D. Class-level shape match only (coefficients/BCs NOT verified).
- **L2 black-box:** no symbolic form needed — falsifier grids judge numerically.
- **OUT_OF_SCOPE** (`NoveltyStatus.OUT_OF_SCOPE`): PDE with no template, integral forms beyond the allowlist, or unparseable input. Returned instead of a forced known/novel verdict.

## 1. Where we stand (measured)

- Prior-art corpus: **106 named laws** (`epistemic.py` v3, all parseable): 10 structural families + mechanics, EM/electronics, thermo/fluids/heat, waves/optics/QM/relativity, chemistry/materials, engineering/control.
- Feynman curated: 5/120 in `ingestion/external_benchmarks.py:83-139`, each with units + ranges + `generate_data(noise)`.
- Scenarios: 22 in `scenarios.py:49-70`. Benchmarks: 25-case + 100-case (`benchmarks.py`), 6 adversarial traps, 5 gated contracts, NACA 11 pts + nano 11 pts (only real measured data), `anscombe.json` 44 rows, 289 pytest tests green.
- Repair demo: `REPAIRED` → `1/x`, `sqrt/log/sin(x)/x`, Lorentz `gamma`, rel-KE; `APPROVED` → `1/(x²+1)`, `exp`, `x+y`, `x²`. `1/(1-x)` needs `critical_values={'x':[1.0]}` (default probes `[0]` + auto `M=±1`, `v→c`).
- Discovery demo: Ohm → `I*R R²=1.0`, kinetic → `m*v²/2 R²=1.0`, Stefan-Boltzmann → `0.9999·T⁴σ R²=1.0`; constant `y=5` correctly `REFUSED`.
- Verdict: all of the above are **rediscoveries** (law-generated data fitted back), not new laws. `PriorArtAuditor.audit_novelty('I*R')` returns `APPARENTLY_NOVEL_WITHIN_CORPUS` over a 10-model corpus with an explicit not-proof disclaimer.

## 2. Track A — certify real discoveries

Goal: `PERSISTED` means prior-art searched + predicted on locked data + survived falsification.

### A1. Prior art 10 → literature scale (`epistemic.py`)
Expand `STANDARD_PHYSICS_CORPUS` with full Feynman-120 + `harvest_all_disciplines()` + 1 versioned textbook CSV (hashed). Add structural matcher next to `check_symbolic_equivalence`: sympy canonical + param-rename isomorphism, plus Pi-group signature via `BuckinghamPiEngine.compute_pi_groups_symbolic`. Keep `NoveltyAuditResult` shape; report real `searched_corpus_size`. Accept: `I*R` matches `V=IR`; `x²` does not match `σT⁴`.

### A2. Prospective loop (`pipeline.py`, `epistemic.py:DataPartitionVault`)
New `run_prospective(train_csv, holdout_csv)`: discover on train only → freeze `ScientificProvenanceChain` hash → single locked-holdout eval → refuse on R²/distortion regression. Discover must not import the holdout. Accept: a train-fit/holdout-refused test exists.

### A3. Operator ceiling (`discovery.py:652-704`, `safety.py:_SAFE_FUNCS`)
`Fraction`-based `±1/3, ±2/3` (halves stay float-exact; thirds must not use `acc == target` float compare), add `erf` (`erfc` if needed), dimensionless-args gate unchanged. Accept: pendulum/Hall-Petch + 1 erf-law rediscovered; seeded legacy tests unchanged or re-baselined.

### A4. Real-data + scale
`ingest_csv`: per-column units + uncertainty passthrough to `propagate_uncertainty_monte_carlo`. Memoize `lambdify` per elite per generation (`discovery.py:851-919`). Accept: NACA + nano end-to-end via pipeline; same winners, lower wall-time.

## 3. Track B — data-free composer (conjecture engine)

Goal: ranked testable conjectures tagged `THEORETICAL_SPECIFICATION`, feedable as `template_priors` into Track A. No R², never higher maturity without data.

- Generators (corpus in, candidates out): crossover via `_monomial_for_dim`; generalize (const→field, `x`→`x/(1+x/K)`); correspondence mutant (must reduce to parent via `limit_with_timeout`, e.g. `v/c→0` → Newton).
- Data-free tests (existing probes only): homogeneity + Pi-groups; limit lattice recovers parents; `ConservationInvariantRegistry` clean; repair audit recorded; `IndependentAdversarialFalsifier` boundary/numerical axes on declared domains.
- Score by simplicity + symmetry + correspondence depth. Output `list[str]` accepted as `template_priors` with `ModelEvidenceRecord` + provenance.
- Accept: 3+ conjectures passing B-tests, corpus-novel per A1, runnable through A2 once data arrives.

## 4. Order, risks, verification

1. A1 → B1/B2 (composer without big corpus "discovers" Ohm weekly). 2. A2 (makes both honest). 3. A3 → A4.
- Risks: thirds float compare → `Fraction`; composer explosion → cap + Pi-signature dedupe; holdout leakage → vault enforcement + test.
- Verification: 289 existing tests green + new: structural-isomorphism, train/holdout refusal, thirds-law, composer-correspondence. `ruff check apedp/`, `mypy apedp/ --ignore-missing-imports` clean.
