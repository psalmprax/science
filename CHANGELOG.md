# Changelog: APEDP & APEDP-X Discovery Framework

All notable changes, bug fixes, audit remediations, and architectural milestones are documented in this file.

---

## [Unreleased] — Governance re-certification ceremony UNBLOCKED (structural gap addressed, 2026-10-22)

The one measured structural gap — the governance manifest desync that could
only be described as "trustee-blocked" with no formal procedure to resolve —
now has its mechanism:

- `scripts/run_governance_remint_ceremony.py` (audit-only, fail-closed):
  `--status` reports the desync and the outstanding quorum exactly;
  `--request` mints the certification request anchored to
  `canonical_spec.py`'s import-time pin (refuses on pin disagreement);
  `--collect` records a trustee signature ONLY after verifying it against
  the manifest's REGISTERED public key (forgeries and unregistered signers
  are refused; keys come from the environment, never from the repo, and are
  never generated or stored here); `--mint` re-verifies every collected
  signature from scratch, extends the transparency log with
  `PREREGISTRATION_RECERTIFICATION` entries through the production hashing
  API, verifies the draft with the production verifier against the disk
  spec, and writes the manifest only on full quorum.
- Boundaries preserved per AGENTS §1.2/§1.3: bypassing, local re-minting,
  fabrication, and spec-file mutation are all structurally impossible in
  the tool; the live red remains red until real external trustees run the
  ceremony.
- Pinned in `tests/test_dyn_governance_remint_ceremony.py` (5):
  sandbox end-to-end re-certification (2-of-3, roster/spec_id carried
  through, log chain verified), forgery/unregistered/stale refusals,
  below-quorum refusal, pin-drift fail-closed, and the live-repo guard
  asserting the repository cannot be minted without real trustees.
- FLAW-01 status in `docs/SYSTEM_FLAWS_AUDIT.md` updated: mechanism
  unblocked; only the external signature ceremony is outstanding.

## [Unreleased] — Margin theorem ADJUDICATED into the certified boundary (2026-10-22)

Promotion-checklist item 2 of the CBF proximity-backlog gate is now closed
at theorem level:

- Deciding layer: `benchmarks/run_margin_theorem_referee_decision.py`
  (audit-only, fail-closed) re-reproduces the packet's 6/6 machine checks
  fresh at HEAD, verifies all evidence pointers against disk, and applies
  the preregistered criteria C1–C5 mechanically.
- **Verdict: ACCEPTED_INTO_CERTIFIED_BOUNDARY** (5/5 criteria, 0
  refusals, evidence fingerprint fdfda1611e452495…) recorded durable in
  `benchmarks/margin_theorem_referee_decision.json`; the packet's decision
  slot is stamped from the record on every rebuild (merge hook in the
  packet builder; the identity fingerprint excludes the slot itself).
- Scope: the verdict grants the THEOREM's acceptance only. Checklist
  items 1 (N ≥ 256 production requirement) and 3 (scipy gate / Rust port)
  remain open; the restricted certified path stays `experimental`, wired
  to nothing; the full all-pairs fail-closed re-check remains the
  revocation trigger in every release.
- Durability: the record is never silently overwritten — re-adjudication
  requires `--re-adjudicate --reason` and preserves the prior verdict.

## [Unreleased] — SRE gate instrumented: tree-vs-arena baseline recorded, port DEFERRED (2026-10-22)

The Rust-migration decision record's rank-1 item ("port `genetic.rs` trees
onto the rime arena `Graph`, Rust→Rust") was supposed to be evidence-gated
but had no tree-vs-arena measurement. Instrumented:

- `tests/sre_tree_vs_arena_profile.rs`: deterministic GP workload (169
  candidates from the production `GeneticOps::random_tree` builder at the
  SRE arithmetic flags, 60k rows ×2 vars), semantics-identical
  tree→arena conversion (IEEE-exact Sub→Add∘Neg, parity-gated bit-exact
  incl. NaN on the covered subset), throughput on the identical candidate
  set, artifact written to `benchmarks/sre_tree_vs_arena_baseline.json`.
- **Measured (gate: TREE-FAVORED)**: arena 0.51× of the tree path (403 ms
  vs 784 ms) on unshared GP trees; the semantics-identical conversion
  covers only 10.1% of the pool (tree Div/Min/Max/powf/unbounded
  transcendentals have no drift-free arena counterpart — tree `a−b` maps
  exactly as `Add∘Neg`, which is what parity is pinned on). The 222×
  figure is Python-vs-Rust, NOT tree-vs-arena.
- Decision record §4 rank 1 revised honestly: **evaluation-speed case for
  the Rust→Rust port is measured-NEGATIVE**; revisit only with a measured
  structural-sharing/hash-consing upside. `MultiVariableSRE` keeps
  delegating via FFI to the native tree engine.

## [Unreleased] — Blocker-sweep repairs: dashboard PDE-transfer regression + stale E5/E4-T1 pins re-minted (2026-10-22)

Release-blocker sweep on current HEAD (bisection-verified causes):

- **`pde_transfer.py` FLAW-05 regression repaired at the cause**: the soundness
  guard had been inserted as early `return`s that bypassed the
  `FrozenTransferResult` construction (bare-float path — callers expecting
  `.r2` got AttributeError; the dataclass return was dead code). The guard's
  semantics now live in the `r2` VALUE, never the return TYPE.
  `tests/test_dyn_pde_transfer.py` 4/4.
- **Dashboard contract fixture re-minted**: after the lawful pool widening
  (Gate A dev-only per the §1.2 anti-snooping rule + symmetric AST
  enumeration in `0eb7a04`), the duffing fixture at noise=0.05 honestly
  abstains (`multi_hypothesis`) and must not be test-forced — noise=0.01
  restores the Δ/τ≈1.3 unique path the contract verifies.
- **E5 / E4-T1 diagnostics re-derived and re-minted (ERR-INC-74-79, disclosed
  in `docs/TRADING_EQUATION_FORENSICS.md` §80c)**: the §74–79 signatures had
  been established under the buggy backward-windowed d5 target (F1, fixed in
  `68a3dadb`) and pre-A01 leaky train blocks (fixed in `9a73170`). Corrected
  harness re-derivation (deterministic, verified twice): market/null E5
  parity replaces the withdrawn **drifting contrast; coupling table splits
  3 stable/3 drifting; §78 deltas re-pinned (M2≈M3 inverted on ETH/AVAX,
  spread capped .08); §79 gates 10/6/8/1/0/0 with ETH as the one pinned
  marginal permutation survivor (.0498); §80 incremental battery re-minted
  again under the fully corrected harness (permutation NULL everywhere,
  min ETH .0846 > α; BTC HA sign-flip 9/11 pinned)
  (`tests/test_dyn_e5_market.py`, `test_dyn_e5_r1.py`, `test_dyn_ensemble.py`,
  `test_dyn_gated_ensemble.py`, `test_dyn_incremental.py`).
- **Quantum-vacuum subthreshold re-pin under the F14 referee**: the old
  abstention was a manufactured refusal (train-loss anchor; Δ was negative) —
  under the holdout-anchored pair the saturated residuals separate at
  Δ/τ≈1.09 with a bootstrap-confirmed weak-margin promotion carrying the
  RESIDUAL_STRUCTURE_UNRESOLVED diagnostic (`tests/test_dyn_quantum_sandbox.py`).
- **Governance remaining OPEN (documented, trustee-blocked)**: the manifest
  still seals the pre-v1.8 spec hash (FLAW-01; quorum signatures not
  producible locally). Typecheck scope is the documented 2-file pyproject
  Phase-1 boundary (green).

## [Unreleased] — Gate-limitation closure: spatial index + model-checked proximity lock (2026-10-22)

The gate decision's three recorded limitations are each acted on at the
measured cause (`docs/cbf_proximity_graph_backlog_gate.md` completion
addendum):

- **"Spatial index needed to beat the dense [N,N,2] floor" — measured.**
  `apedp_dyn/robotics/proximity_margin.py` gains `indexed_proximity_pairs`
  (scipy cKDTree, candidate radius r_safe + h* + slack: a SUPERSET of the
  constrained band by construction, so every EXCLUDED pair is provably
  free — soundness pinned, including the exact boundary-equality semantics
  where scipy's `at most r` admits the pair the dense h < h* filter
  excludes) and `indexed_barrier_components` (exact sparse mirror of the
  dense formulas, array-equal pinned). Measured: the index beats the dense
  build floor 3.3–3.6× at N=256 and 3.1–5.8× at N=512 (at N=128 the tree
  overhead has not paid off — recorded as gate information). The COMPOSED
  restricted path (index + restricted alternating projections + FULL
  all-pairs fail-closed re-check) measures 3.31× over the dense solve at
  N=256 (64.0 → 19.3 ms, constraint ratio 0.054) and 3.6× at N=512
  (258 → 71.6 ms, ratio 0.028); at N=128 only 1.31× — the certified
  contract's own all-pairs re-check dominates there, and that re-check is
  the fail-closed honesty floor, deliberately never removed.
- **"Pitch-only, wired to no pipeline" → evidence class completed.**
  `model_check_unicycle_horizon_proximity` realizes the proximity-LOCKED
  commands through the production arc stepper on adversarial horizons
  (head-on top-speed closing; multi-agent closing grids, 2 seeds):
  min h > 1e-3 throughout (measured evidence, explicitly not a formal
  claim). Promotion checklist added to the gate doc (production N ≥ 256
  requirement + referee-adjudication acceptance of the margin theorem
  into the certified boundary + scipy-or-kernel-port decision + the
  all-pairs re-check stays in every release, forever).
- **"Dev-box medians" → variance quantified** over two full reruns of the
  same seeded harness: certified 5 ms gate crossing stable (N=256 twice);
  QP-time noise 5–6% at N ≥ 256, ~31% at N=128 (scheduling dominated); the
  heuristic 50 ms boundary sits exactly at its edge (N=512: 47.5 vs 50 ms
  flipped between runs) — recorded verbatim as the boundary sensitivity;
  thresholds remain decision thresholds, not constants.
- Pins: `tests/test_dyn_proximity_margin_pins.py` 10/10 green
  (index superset + provably-free exclusions, boundary-equality pair,
  array-exact sparse-vs-dense, collision-free horizons, fail-closed
  max_iter=0 refusal, midpoint saturation).

## [Unreleased] — Gate 1 executed: vectorized pair sweep + margin pitch (2026-10-22)

- **Heuristic pair-loop VECTORIZED** (`multi_agent_step` now uses ONE
  `pairwise_repulsive_forces` sweep): EXACT per-pair mirror of the scalar
  rule — degenerate [k_rep, 0] fallback, r_warning zeros filter (the mask
  IS the proximity filter a graph would replicate), head-on tangential
  term at anti > 0.7. Pinned branch-exact per pair AND against an inline
  legacy reference at whole-step level across N = 2/5/8/33
  (`tests/test_dyn_multi_agent_vectorized_pins.py`, 9 tests green;
  pre-existing robotics suites 14/14 green). Re-profiled with the same
  seeded harness: exponent 1.91 → 1.06, 5 ms budget crossing N=32 →
  N=128, 50 ms no longer crossed in-sweep, 26× faster at N=256 — a
  proximity graph is NOT needed for this path (the gate doc carries the
  pre/post tables; certified-path gate numbers unchanged).
- **Non-neighbor barrier MARGIN argument pitched + machine-checked**
  (`apedp_dyn/robotics/proximity_margin.py`, experimental/evidence-only,
  wired to no pipeline): pairs at h ≥ h* = 2·v_max/γ hold hdot ≥ −γ·h
  CONSTRUCTIVELY (box-bounded worst case −2·v_max saturates the law
  EXACTLY at h = h*, never violates); `solve_cbf_qp_proximity` solves the
  alternating-projection QP over the h < h* restricted set and releases
  ONLY through the FULL all-pairs fail-closed re-check (the margin
  algebra re-verified numerically at release); FREE/CONSTRAINED
  bookkeeping re-decided per step from the h matrix the barriers build
  already computes. Measured on the A3-guarded grid: constraint ratio
  0.19 at N=64. This is the soundness prerequisite the certified-path
  N≥256 gate demanded — STILL experimental: no production wiring, and a
  production proximity structure additionally needs a spatial index (the
  dense [N,N,2] floor stands). Pins: `tests/test_dyn_proximity_margin_pins.py`
  (margin formula, midpoint saturation, restricted-solve full re-check on
  randomized fixtures, provably-violated max_iter=0 refusal, constraint
  bookkeeping).
- **Hygiene**: two pre-existing ruff flags in `cbf_certificate.py`
  surfaced by package-level linting fixed at the cause (genuinely-unused
  `import torch` removed after grep confirmation; unused unpack renamed).

## [Unreleased] — CBF proximity-graph backlog gate decision (2026-10-22)

- **O(N²) profile recorded** (`benchmarks/run_cbf_scaling_profile.py`,
  results in `benchmarks/cbf_scaling_profile_results.json`, gate record in
  `docs/cbf_proximity_graph_backlog_gate.md`; status `audit-only`):
  deterministic A3-guarded fixtures; the certified QP's factory-lemma
  1-pass FLOOR isolated from adversarial-iteration growth — the all-pairs
  barriers build + constraint sweep alone costs 9.9 ms at N=256 (43 ms at
  N=512), with uniformly-random commands adding ≈8× more at N≥256; the
  heuristic transport stepper costs ≈6 µs per Python pair invocation and
  crosses the 5 ms real-time budget at N=32 (50 ms at N=128).
- **Gate decision (criteria preregistered in the harness BEFORE
  measuring)**: the heuristic path activates NOW at N≥32 — its first fix
  is VECTORIZED pair-force computation (behavior-identical, no certificate
  soundness question; a k-NN graph adds nothing a distance mask does
  not); the certified path's proximity restructuring activates only for
  production N ≥ 256 real-time certified control AND a soundness proof
  that non-neighbor pairs cannot violate barriers within the assumption
  box (pair-ratio counterfactuals 0.38→0.011 at k=4 recorded as
  computational estimates only — pruning AFTER the dense barriers build
  keeps the O(N²) floor, so a production structure must use a spatial
  index instead). Certified code, pins and battery mutants untouched —
  measurement + gate record only.

## [Unreleased] — Graph census verification and gap closure (2026-10-22)

Response to the external master graph census. Verified ground truth
BEFORE acting; then closed the actionable slots and corrected the record
(`docs/graph_census_2026_10_22.md`):

- **Three "implemented but untested" graph surfaces are now pinned**
  (`tests/test_dyn_graph_census_pins.py`, 5 tests green; census claim of
  zero coverage verified by search first): W3C PROV-lite
  `ProvenanceGraph` (nodes + links verbatim, no shared-mutable-default
  leak), `plot_epistemic_graph` (real deterministic PNG render, magic
  bytes verified), and the `ak.rime_rewrite_json` PyO3 FFI binding for the
  frozen structural rewrites — valid `WrapNonlinearity` appends the
  wrapped node and makes it the output per `rewrite.rs` `apply` semantics,
  a bad operand index is REJECTED (no fabricated dangling references),
  malformed graph/op JSON fails closed.
- **Two census rows corrected as stale/clarified:** the Level-1 state
  dependency graph EXISTS as the DE arm's screening + dense escape hatch
  (never a causal adjacency verdict — kernel `coupled_system.rs` stays
  flat SINDy as the escape hatch); the AD row clarified — exact-rule
  SYMBOLIC service in `de_arm/ad_service.py`, kernel rightly keeps forward
  Taylor Jets, torch stays the reverse-mode runtime.
- **Genuinely unimplemented classes** (robotics proximity graphs, mesh
  PDE discovery, GP-tree→DAG sharing, 2D/3D tensor networks, kernel
  reverse-mode tape) recorded as rationale'd backlog categories, not
  promises — each with the deprioritization reason and its gate.
- **Honest pin-failure note:** the census FFI pin initially expected a
  named-field serde shape for the `Tanh(usize)` variant; the test
  assertion was corrected to the true newtype-variant serialization — the
  FFI matched `rewrite.rs` exactly (cause fixed at the failing assertion,
  not at the pinned boundary).

## [Unreleased] — DE-arm frozen benchmark vs dense flat library (2026-10-22)

- **Frozen benchmark** (`benchmarks/run_de_arm_vs_dense_benchmarks.py`,
  results in `benchmarks/de_arm_vs_dense_benchmark_results.json`, doc in
  `docs/de_arm_vs_dense_benchmarks.md`; status `audit-only`): deterministic
  RK4 fixtures of Lotka-Volterra, FitzHugh-Nagumo and Lorenz (noise-free,
  seeded). Dense kernel and the arm's screened mirror both structurally
  exact on all three systems at the default screen depth (top_k=2, 0 false
  exclusions). Library-reduction hypothesis MEASURED, not assumed:
  ≈10% (Lorenz, 10→9 columns), 25% (FHN), 0% (LV) at D≤3 — matching the
  review's correction that the 90% claim is a hypothesis for higher
  dimensions/richer grammars. The aggressive screen (top_k=1) drops 2 true
  parents on Lorenz with visible R² collapse (0.63, 0.04) — demonstrating
  the dense escape hatch is load-bearing; screen exclusions are reported
  verbatim, never hidden.
- **Fixture-integrity finding (recorded as evidence, not patched away)**:
  a first Lorenz fixture with a narrow slow-manifold window yielded
  identically-zero-residual but NON-physical polynomial fits in BOTH
  engines (R²=1.0 with wrong structure) — noise-free near-collinear data
  makes many column combinations exactly degenerate. The benchmark now
  guards window sweep on BOTH the full window and its train partition.
- **Representational fix** (`de_arm/dependency_screen.py`): the reduced
  mirror library gains kept·kept parent products — without them the
  screened library could not express parent×parent monomials the dense
  engine demonstrably recovers (Lorenz ẏ = 28x − y − xz needs x·z).
  Excluded parents still never enter the screened library (pin holds).
- **Hygiene**: corrupted docstring token in `de_arm/typed_model.py`
  repaired; ruff clean; DE-arm pin suite re-verified after each change.

## [Unreleased] — DE-discovery arm, separate arm boundary (2026-10-21)

Per the RIME–APEDP-X review thread (build the differential-equation
discovery capability as a SEPARATE arm reusing shared infrastructure,
never as new graph nodes inside the RIME core; reviewers' corrections
applied literally — no claimed 90% library reduction, no causal-topology
oracle claim, dense escape always present), phase 1 lands behind a clean
module boundary (`apedp_dyn/de_arm/`, all `experimental`):

- **Typed mathematical semantics** (`de_arm/typed_model.py`): every arm
  candidate declares its kind — STATIC_EXPRESSION / ODE_SYSTEM (states +
  independent variable + dt + initial conditions) / PDE_SYSTEM (fields +
  spatial axes with positive dx + declared boundary treatment). Incomplete
  declarations are fail-closed refusals, so structural checks run BEFORE a
  search, not after it.
- **AD as shared infrastructure** (`de_arm/ad_service.py`): a symbolic
  derivative service over a closed typed expression vocabulary with exact
  product/quotient/chain rules and declared domain refusals; the module
  documents both honest limits — it differentiates the implemented
  candidate computation (a perfect derivative of a wrong hypothesis stays
  wrong) and measured data keeps its existing finite-difference stencils
  for residuals. Fornberg remains valid on irregular grids (correction C:
  no blanket first-order degradation claim).
- **Level-1 dependency screening with a MANDATORY escape hatch**
  (`de_arm/dependency_screen.py`): per-state predictive-increment ranking
  of candidate parents (lag-embedded relative-RSS gains on the training
  partition; deterministic, dependency-free) prunes each state's candidate
  library, but the DENSE kernel result always ships beside the screened
  mirror fits in one referee-facing `OdeArmOutcome` — a parent the screen
  excluded can never silently vanish, and the screen's ranking travels
  with an explicit not-a-causal-verdict disclaimer. The screened fit is a
  documented python ridge+prune mirror whose provenance never claims
  kernel parity.
- **Pipeline/lifecycle integration**: the dispatch surface's coupled-ODE
  route now hands the arm's evidence artifact to the referee path; the arm
  contains no acceptance/promotion/verdict field anywhere (pinned), and
  the referee's frozen protocol stays the sole authority.

Pins: `tests/test_dyn_de_arm_p1_pins.py` (10: typing refusals, AD
exactness incl. the product-rule mutation boundary, screening semantics
with the escape hatch, dispatch integration, referee independence). The
arm's mutants join the mutation battery register (MUT-G1..G3).

---

## [Unreleased] — EHI re-foundation per external review (2026-10-21)

The external architecture critique's four build items and the fifth pillar
landed, each replacing a math shortcut the review named (all `experimental`;
pins `tests/test_dyn_ehi_phase5_pins.py`, 16 mixed behavior+mutation pins;
none of the touched layers decides acceptance — the referee keeps authority):

**Dispatch wiring (follow-up, same review thread):** the sieve + selector now
route proposals AUTOMATICALLY (`apedp_dyn/orchestration/dispatch.py`):
decisive agreement on both layers ⇒ the one engine runs (L2 open discovery /
ODE diff-Padé / frozen-operator PDE / Hawkes MLE); selector ambiguity keeps
every sieve-admitted family live (MULTIPLE_ROUTES, masks honored even with
wired runners); decisive disagreement ⇒ CONFLICT abstention, never an
override; unresolved structure ⇒ UNROUTABLE with open questions. Per-family
failures are recorded in `errors`, never laundered; products carry
referee-facing material only. Pins `tests/test_dyn_discovery_dispatch_wiring.py`
(9). **Two broken seams fixed on the way:** `discover_ode` called the kernel
with raw lists + an int while the seam takes three JSON strings (matching
every other caller — no test had exercised it end-to-end); and a decisive
route is now refused when the winner's DEFINING diagnostic never fired
(pure noise on a time axis can no longer fabricate a PDE claim via
competitive misfit).

- **Model-class selection replaces deterministic routing**
  (`apedp_dyn/orchestration/model_class_selector.py`): P(M|D) over
  documented per-class diagnostic profiles with ambiguity-preserving
  outcomes (`MULTIPLE_CLASSES_PLAUSIBLE`, `INSUFFICIENT_METADATA`); the
  declaration-narrowed class table keeps point processes as point
  processes. **Bug fixed by the pin probe:** the sieve's empty-string axis
  sentinel was treated as a found axis, stamping `time_axis=1.0` onto pure
  isotherms (routing to `ode_kinetic` at bf 1.0); and the class-firing
  penalty now keys on NON-shared diagnostics (`time_axis≈1` no longer rides
  for every dynamic class at PDE/ODE ties) with a documented 2-nat cost.
  A documented aggregation-validity criterion (mean-field amplitude vs
  field spread) keeps a traveling wave's negligible ripple from being
  mis-badged as a kinetic ODE signal while still allowing genuinely kinetic
  2D fields to contest ODE against a firing PDE diagnostic.
- **Constraint STATES replace automatic enforcement**
  (`apedp/dimensional_invariant_engine.py::classify_constraints`): boundary
  conditions are hypotheses with states `KNOWN` / `EMPIRICAL` /
  `UNRESOLVED` / `REJECTED`; training-partition evidence carries
  `is_physical_claim=False`; subsection cases (offset-Langmuir → origin
  REJECTED; sub-critical window → saturation UNRESOLVED, not rejected);
  declared independent laws alone express `KNOWN`.
- **Expected information gain replaces unbounded-gap maximization**
  (`apedp_dyn/epistemic/active_experimentalist.py`): I(H;Y|x) by exact
  Gaussian-mixture quadrature, argmax under feasible ranges / cost / noise,
  field-complete `ExperimentOrder` requests and fail-closed refusal of
  zero-noise models and no-honest-experiment conditions.
- **Fifth pillar: research controller**
  (`apedp_dyn/epistemic/research_controller.py`): belief/governance state
  separation (referee constants fingerprints re-verified before every
  planned action; in-session tamper raises), deterministic planner over
  {inspect, test-constraints, information experiment, improve data quality,
  ABSTAIN}, five-part evidence ledger, controlled exits
  (`ABSTAINED` / `BLOCKED_ON_PROBLEM_SPECIFICATION` /
  `EVIDENCE_INTEGRATED_AWAITING_REFEREE`; one acquisition per session —
  incorporation requires a referee re-adjudication, never self-adjudication),
  and in-run protocol revisions returned as versioned
  `PROPOSED_NOT_APPLIED` artifacts.

---

## [Unreleased] — Level-5 autonomy pillars (2026-10-21)

Four autonomous-scientist pillars landed at the roadmap's named module
paths (all `experimental`; each with a dedicated pin file, 23 pins total,
and the honest disclosure that none of them touches acceptance authority):

- **Pillar 1** `apedp_dyn/orchestration/modality_sieve.py`: autonomous
  modality & topology sieve — structural evidence detectors (isochronous
  monotone time axis with a declared-only irregular-axis contract, bounded
  relative curvature, log-axis 2D spectral dispersion, Fano burstiness)
  classify raw data into static-manifold / ODE-kinetic / PDE-field /
  Hawkes-event modalities and MASK ill-posed engine families before any
  proposer runs; fail-closed UNRESOLVED on conflicting evidence.
  Pins: `tests/test_dyn_modality_sieve.py` (6, incl. the isotherm-vs-
  kinetics discriminator and the Poisson-refusal).
- **Pillar 2** `apedp/dimensional_invariant_engine.py`: autonomous
  dimensional & boundary synthesis — unit→exponent conversion (pint SI
  bases, integral-exponent fail-closed) feeding the audit-proven
  Buckingham-Π engine (exact integer nullspace in
  `apedp_dyn/conf.dimensionless`), log-band empirical limit estimators
  (origin pin q(0)=0 AND independent finite-plateau detection steering the
  rational saturation family; fail-closed on ambiguous curves), and the
  interpolated half-max characteristic scale with the {s*/3, s*, s*·3}
  search bank. Pins: `tests/test_dyn_dimensional_invariant_engine.py`
  (5, incl. pint-verified dimensionless Π and both Langmuir constraints).
- **Pillar 3** `apedp_dyn/epistemic/active_experimentalist.py`:
  closed-loop active experimental disambiguation — x* = argmax pairwise
  divergence − k·σ with k imported from the frozen E4 policy
  (`DEFAULT_DETECTABILITY_K`, no new threshold), deterministic clustered
  target-range acquisition orders, fail-closed out-of-band/NaN point
  rejection (instruments that lie are counted, never clamped), and residual
  referee handoff — no promotion decision inside the layer.
  Pins: `tests/test_dyn_active_experimentalist.py` (5).
- **Pillar 4** `apedp_dyn/epistemic/enclave_referee.py`: confidential
  adjudication — Stage-C pair verdicts computed in a fresh OS subprocess
  over permission-sealed inputs (aggregates only, no shared sys.modules),
  an Ed25519-signed attestation binding verdict + request SHA-256 +
  aggregates, tamper-fail verification, and UNRESOLVED on any
  infrastructure failure (timeout/crash/missing output). Honest
  "process-isolated" executed_by claim — TEE only by explicit runtime
  declaration (`APEDX_TEE_ENCLAVE`), never auto-upgraded.
  Pins: `tests/test_dyn_enclave_referee.py` (7, incl. reproducibility,
  forged-signature tampering, hash-binding failure and fail-closed
  timeout).

Verification: pillar pins 23/23 (`EXIT_STATUS=0`); AGENTS.md dynamic
regression suite green after the DOCUMENTATION.md registry indexing; the
new modules are registered `experimental` in the §18.1 registry with
docstring `Status:` lines matching the tags.

---

## [Unreleased] — External 19-flaw audit remediation (2026-10-21, same day)

An independent adversarial audit catalogued 19 flaws (Flaw severity matrix in
`docs/SYSTEM_FLAWS_AUDIT.md`, 2026-10-21 round). 18 fixed by execution-verified
edits; FLAW-01 remains OPEN by protocol:

- **FLAW-01 (Critical, OPEN — trustee-blocked):** the Ed25519 governance
  manifest still seals the deprecated spec SHA `a6d026…`. Re-signing requires
  trustee private keys held outside the repository; the distributor must not
  bypass the manifest. (Spec centralization for FLAW-13 below follows the
  written §1.3 workflow and does not mint governance claims.)
- **FLAW-02:** `apedp_dyn/ode/proposer.py` scored the identity instead of the
  discovered ODE RHS — every ODE proposal evaluation was `y = x_i`. The
  scorer now evaluates `sum(coeff · basis(x))` over the kernel term encoding
  (measured: RHS-vs-true-derivative RMSE 0.0000 on a probe where the identity
  was 1.67).
- **FLAW-03:** S3 lock takeover was a TOCTOU race; now conditional
  (`IfNoneMatch="*"`, 412 → contested) with a read-back token verification
  fallback that serializes non-conditional backends. v2 (mutation-pin run):
  the parameter-rejection fallback now ISSUES the unconditional put before
  the read-back — the first draft skipped it, so the read-back could never
  see our token and every acquire on old-botocore backends deadlocked to
  timeout.
- **FLAW-04:** uniform-grid stencils in `rime/diff_pade.rs` now gate on actual
  grid uniformity (1e-4) and fall back to Fornberg arbitrary-grid weights.
  v2 (mutation-pin run): the first Fornberg translation was structurally wrong
  (weights stopped annihilating constants; ~1e3x error at stencil edges) —
  `fornberg_weights` now mirrors the proven Python
  `_fornberg_centered_weights` recurrence, pinned by
  `test_nonuniform_grid_derivatives_use_fornberg_fallback` (quadratics
  differentiated exactly on a strongly non-uniform grid).
- **FLAW-05:** the constant-field `R² = 1.0` tautology is removed in
  `rime/pde.rs`, `pde_discovery.py` and `pde_transfer.py` (perfect resorbed
  constant may claim 1.0; real residual energy on a flat target scores 0.0);
  a failed linear `solve` no longer unhides the STRidge significance rescue
  (se = 0 ⇒ rescue disabled, pure relative pruning).
- **FLAW-06:** pole evaluations in the diff-Padé rollout set
  `pole_singularity_hit` and disqualify `is_exact_recovery` instead of
  silently integrating through singularities; Koopman ss_tot is centered and
  zero-variance targets score 0.0. The pole-flag report contract is now
  mutation-pinned on BOTH rollout branches: Bessel-J0 (forces the 2nd-order
  branch, S(0)=0) and the parabola with `max_diff_order=1` (forces the
  1st-order branch, P(0)=0).
- **FLAW-07:** C6 apex evidence set now includes `EQUIVALENCE_RIVALS_REJECTED`
  and `INTERVENTIONAL_INVARIANCE` (tier monotonicity restored).
- **FLAW-08:** re-anchored referees run the block-bootstrap tiebreak on
  provenance-recovered paired residuals; unpairable pairs stay on the
  disclosed point-estimate contract (INCOMPLETE).
- **FLAW-09:** the Hawkes eval-time binary collapse returns the collapsed
  binary vector instead of silently switching to an unbounded z-score; the
  pinned fit-time continuous declaration is untouched.
- **FLAW-10:** core-mode Lean theorems relying on injected axioms are labeled
  `syntactic_schema_check::…` — never presented as independent FORMAL_PROOF.
- **FLAW-11:** opposed one-sided limit probes on an asymptote raise rather
  than averaging a fake 0.0 (previously `limit(1/x, x→0)` timed-path
  returned 0.0); the wall-clock regression retains its budget contract.
- **FLAW-12:** the implicit-RIME exact-RMSE ceiling measures scale over ACTIVE
  non-constant library columns only (`test_near_zero_scale_misfit_not_certified`
  pins the near-zero-scale refusal with its O(1e5) exact positive control;
  MUT-E4 kills the re-included constant column).
- **FLAW-13 (spec centralization per AGENTS §1.3):** preregistered spec gains
  `ode_referee_min_mean_r2=0.95` / `ode_referee_min_worst_r2=0.85`
  (`canonical_preregistered_spec_v1.json` re-pinned:
  `a11fd957…` →
  `5bb42f0341b7a024d21280ee69922bd7dbfebd33cf55506253e3f5248411ec04`;
  rationale: removing hardcoded referee literals per the
  specification-centralization rule; Ed25519 manifest re-certification remains
  trustee-blocked as recorded under FLAW-01); `OdeReferee` defaults bind to the
  spec, and mean holdout R² < 0 earns the new honest `UNSTABLE_DIVERGENT`
  verdict instead of `PREDICTIVE_WRONG_STRUCTURE`.
- **FLAW-14:** one-sided upwind stencils with zero-flux Neumann boundaries
  replace toroidal `np.roll` in the audio spectral advection.
- **FLAW-15:** the Lie-symmetry nullspace gate is normalized
  (`|λ| ≤ tol · λ_max`).
  v2 (mutation-pin run, 2026-10-21): the v1 relative gate left the EIGENSOLVE
  itself scale-blind — on O(1e6)-amplitude ODEs the multi-stratum Gram matrix
  corrupted a structural zero into eigenvalue ~1.0 (basis residual ~7e5, 9
  vs 4 genuine directions reported); `discover_point_symmetries` now solves
  the CORRELATION matrix (kernel-preserving congruence, generators map back
  via D⁻¹v) and `jacobi_eigen` pre-normalizes to unit spectral scale — the
  scaled system now reports the same symmetry verdict as the unscaled one
  (`test_scale_dominant_ode_keeps_genuine_symmetries`). An
  absolute-tolerance-gate mutant was registered and GREW no natural kill
  after this fix (the defect expression it targeted is dissolved by the
  corrected eigensolve); it was removed from the battery with its rationale
  recorded in `scripts/run_mutation_battery.py` rather than shipped as a
  silent survivor.
- **FLAW-16:** Hamiltonian archetype aligned to the pinned
  `CONTINUOUS_FACTOR` contract.
- **FLAW-17:** CBF-QP closes with a final norm-ball projection before the
  fail-closed certificate checks. v2 (mutation-pin run): the plain re-scale
  is provably insufficient (scaling the corrected endpoint shrinks the
  deflection that repaired its pair and re-breaks the barrier); the close-out
  now distributes each remaining deficit BETWEEN ITS ENDPOINTS, each share
  bounded by the endpoint's direction-aware norm-ball headroom, |v| ≤ v_max
  exactly, and runs only when the loop exhausted with pending corrections —
  `max_iter=0` still refuses fail-closed.
- **FLAW-18:** ledger appends are flock-guarded (`LOCK_EX`/`LOCK_UN`, graceful
  degradation without fcntl) and remote auto-sync is amortized in batches of
  16 pending appends with a flush before snapshots.
- **FLAW-19:** dead `np.gradient` removed from the SIR proposer; formula uses
  `i_col` physical notation instead of the internal `I_arr` identifier.
- **FLAW-STRAITA (same-day follow-through; exact-RMSE gate audit):**
  `rime/implicit.rs` still carried two COMPENSATING anchored gauges in
  raw-coefficient space — a raw `max_c = max|c_j|` rescale applied after the
  strata-coherent `c_j = v_j/norm_j` mapping, and a raw `|c_j| > 1e-6`
  active-count threshold that the first gauge had been accidentally inflating
  above threshold. At O(1e6)-scale strata the pair degenerated the genuinely
  pure 1e6 circle to the constant fallback `-1 = 0` (rmse 1.0, refusal) —
  probed before the fix; removing only the first gauge exposed the second
  (`test_large_scale_exact_relation_certifiable` failed at rmse 3.6e-17).
  Fixed with coherent contribution-space gauges: no normalization after
  `v/norm` (the Jacobi eigenvector is already 2-normalized), and membership
  by contribution ratio `|c_j|·norm_j > 1e-6·max_j(|c_j|·norm_j)`. Post-fix
  probes (rebuilt extension): pure 1e6 circle certifies rmse 4.1e-17,
  near-zero-scale pure circle 1.6e-18, unit circle 5.7e-16 unchanged;
  MUT-E4's kill restored on the natural near-miss fixture. Pins:
  `test_flaw_straita_implicit_rs_raw_gauges_removed` +
  `test_flaw_straita_extreme_scale_circle_certifies_not_degenerate`.
  Verification: `cargo test --lib` 193/193, mutation battery ALL KILLED,
  `scripts/check.sh` full tier green.
- **FLAW-A01 (Set A follow-through; 4-module lookahead leak):**
  `ensemble_gated.py`, `ensemble_m.py`, `incremental_info.py` and
  `e5_market.py` built train blocks with the raw `dates < origin` mask while
  the `_d5` target forward-shifts the 5-day rolling mean (`shift(-5)`) —
  blocks within 5 days of the origin boundary read forward out-of-sample
  rows. Fixed by the shared helper `trading_e4t1_v2::_train_blocks_no_leak`
  (mirrors `run_origin`'s corrected mask `tr[tr + 5 < origin_idx]`) applied at
  every train-block call site in the four modules. Pins: behavioral pin
  (pre-fix leak demonstrated inline) + four-consumer text pin in
  `tests/test_dyn_external_audit_2026_10_21_pins.py`.
- **FLAW-A09 (Set A follow-through; sibling lag-buffer leak):** the
  distributed-lag scorer wrote `lag_state["buffers"][base_col]` on the first
  lag term of a call, so a second lag term on the same base column read that
  same call's own tail as prior history (future data prepended to the head).
  Buffer is now snapshotted pre-call and written once per (call, base
  column). Pin:
  `test_distributed_lag_multi_lag_same_column_streaming_safe` (chunked ==
  single-pass with two lag terms sharing one column).
- **FLAW-A13 (Set A follow-through; zero-denominator masking):** blind
  benchmark per-domain metrics reported `0.0` on zero denominators — falsely
  certifying a 0% FPR where no negative controls exist. Per-domain
  `fpr`/`fdr`/`vrr` now report `None` on empty denominators, mirroring the
  headline convention; pins updated to the honest contract and a dedicated
  zero-denominator pin added (`tests/test_dyn_blind.py`).
- **Frozen-record re-mint v1_8 → v1_9 (per freeze protocol):** the FLAW-07 and
  FLAW-08 referee edits legitimate the guard-file refresh;
  `apedp_dyn/sandbox/APEDPX_FROZEN_v1_9.json` supersedes v1_8 with
  `remint_log` rationale for `adjudicate.py` + `claim_ladder.py` (13/15 other
  guard files byte-identical); thresholds, env pins, and blind-case
  expectations unchanged. Post-mint: blind + round-2 suites 56 passed.
- **Mutation-pin coverage for the 2026-10-21 remediation:**
  `tests/test_dyn_external_audit_2026_10_21_pins.py` (23 tests) pins
  FLAW-02/03/05/07/08/09/10/11/13/14/16/17/18/19; each pin carries an inline
  old-semantics mirror so reverting the fix flips the pin. Four
  reintroduction mutants (MUT-D1..D4: S3 conditional take-over, fallback-put
  deadlock, CBF budget-aware deficit split, re-anchor residual recovery)
  registered in `scripts/run_mutation_battery.py` and wired to the pin suite —
  battery re-run: ALL_KILLED, 9/9 mutants
  (`benchmarks/mutation_battery_report.json`), byte-identical restoration
  verified per mutant.
- Scope note: Rust kernel edits (`diff_pade.rs`, `koopman.rs`, `implicit.rs`,
  `lie_symmetry.rs`, `pde.rs`) are outside the Python freeze record;
  `cargo test` 186/187→186/186 covers them (see FLAW-04/05/06/12/15).

Verification: Rust `cargo test` 186/186; affected Python suites re-run green
(`test_dyn_pde_ode_packages`, `test_dyn_claim_ladder`, `test_evidence_tiers`,
`test_dyn_lean_universal_bridge`, `test_flaw_fixes`, hawkes pins, STEM
contracts, CBF certificate, ledger benchmarks, audio transport, remote
storage, correspondence epistemic bootstrap/caller/incremental/reporting).

---

## [Unreleased] — Micro-sample referee remediation (2026-10-21)

### Two referee defects found by execution, fixed with regression pins
- Context: an executed double-check of an external APEDP-X reconciliation
  (Langmuir 1918 micro-sample adjudication) reproduced 18/20 claims digit-for-
  digit (RimeProposer `min_rows=30` safeguard; `+37.7147*(sat_norm(p)) +3.97006`
  with dev_r2=0.9935; 8-scale LOOCV RMSE 0.404810 vs 0.408720; Wilcoxon W=22.0
  p=0.3652) and falsified two: the negative-control "5 proposals" figure is
  4 on this host/kernel, and the "10.2×" timing ratio sits in no measured
  protocol scope (scalar-dot 153×, full fold 1.5×).
- **Defect 1 (soundness):** the external `sound_continuous_monotonicity_proof`
  filtered positive roots by an absolute 1e-9 tolerance, certifying dN/dp =
  p·(p − 5e-10) — genuinely negative on (0, 5e-10) — as monotonic. Fixed in
  `benchmarks/run_rime_microsample_adjudication.py::sound_monotonicity_v2`:
  exact root isolation with NO tolerance filter, interval partitions with
  40-digit probes, and an UNRESOLVED fallback for non-algebraic expressions.
  Regression world: `tests/test_dyn_rime_microsample_adjudication.py::
  test_counterexample_with_micro_root_is_violated` (external engine's verdict
  was WRONG, fixed engine returns VIOLATED).
- **Defect 2 (decision semantics):** the external negative-control check
  `all(p.parameters.get("scale", 0) < 0 for p in proposals)` is vacuously
  True for an empty packet, so a proposer abstention could be reported as a
  dynamics-inversion rejection. Fixed in
  `classify_referee_decision`: ABSTAIN (empty packet, asserted by test) /
  REJECT (gate failure) / PROMOTE are distinct outcomes and never conflated.
- **Claim converted from prose to measurement:** the Michaelis-Menten N=12
  "PROMOTED (12/12 folds)" bullet existed only in narrative; the new harness
  re-executes it (LOOCV RMSE 0.1740, 12/12 admissible under the exact family
  certificate q′(p) = w·s/(s+p)² > 0 ratified by symbolic identity check).
  Artifact: `benchmarks/rime_microsample_adjudication_results.json`.
- **Fold-level proposer recovery frozen (2026-10-21 followup):**
  `run_fold_level_rime_recovery` executes RimeProposer inside each of the 11
  Langmuir LOOCV folds and records fold-level recovery/abstention in the
  artifact, in two disclosed modes: bank-restricted (Python-built sat_m
  columns; RMSE 0.482436, selections {0.8×1, 0.7×1, 0.6×2, 0.66×7}, zero
  abstentions) and unrestricted-grammar (raw p + declared repertoire,
  kernel-derived scales; the engine re-selects a saturation family in 11/11
  folds at measured cost 0.629997). A disclosed_limit field pins that the
  two modes are not the same experiment; per-fold partial-gate decisions
  (capacity_positive subset) are recorded with the graph-structural gap
  disclosed as not-yet-wired. Three new pins (14 total in
  tests/test_dyn_rime_microsample_adjudication.py).
- **Structure-aware referee patch (defect-3 closure):** the external referee
  promoted a physically unstructured counterfeit quadratic (w=5, |b|=0.5,
  positive data correlation) while rejecting the tournament's own verified
  sat_norm winner via a |b|<=1.0 tolerance. `adjudicate_structure_aware` in
  the artifact reads structure from the declared formula family, applies the
  exact family certificate (monotone for ANY intercept), gates on {w>0,
  q_inf=w+b>0, exact monotonicity}, and REPORTS q(0)=b with an explicit
  origin_strict_satisfied flag instead of a tolerance gate. Measured:
  counterfeit -> REJECTED; discovery (w=37.7147, b=3.97006) -> PROMOTED with
  q(0) deviation disclosed; negative-scale saturation -> inversion rejection;
  b=0 baseline -> origin-strict PROMOTED; the 11 bank-restricted fold
  proposals re-adjudicate 11/11 PROMOTED as verified structures with 0/11
  origin-strict. Five new pins (19 total).
- **Graph-equation structural validation wired into the fold referee
  (2026-10-21 followup):** `validate_kernel_graph_structure` enforces the
  reviewer's structural check from kernel metadata — graph_id provenance,
  canonical equation as the authoritative token (declared-column mode or
  kernel-transform mode resolved against `inputs_used_idx` /
  `declared_nominal_dependencies`; the kernel emits empty
  `inputs_used_names` for transform columns, spec indices are the signal),
  independent feature reconstruction, and executable affine-readout
  verification (compute_score == w*feature+b on the dev rows within 1e-9;
  measured max deviation 0.0 machine-exact). The 11 Langmuir folds re-run
  through the kernel with this referee: 11/11 PROMOTED with 11/11
  structure-validated and per-fold graph ids/equations/gates frozen.
  A tampered-graph test proves the validator rejects when the executable
  readout deviates while metadata checks pass. Three new pins (22 total).
- **Michaelis-Menten pathway frozen (2026-10-21 followup):**
  `run_michaelis_menten_graph_validated` executes the same graph-validated
  referee path for the MM system: the actual RimeProposer runs inside each
  of the 12 LOOCV folds over the MM saturating bank, every proposal is
  metadata-validated with a machine-exact affine-readout check (max dev
  0.0), structure-aware gates apply with declared s_m. Measured: 12/12
  PROMOTED, 12/12 structure-validated, zero abstentions, actual-RIME LOOCV
  RMSE 0.1316 (beats the b≡0 baseline 0.1740 on this dataset); readout w
  recovers Vmax=10.0 within 5%; per-fold origin deviations disclosed.
  The full-data tournament reproduces the external '6 proposals PROMOTED'
  row with measured semantics (6/6 admissible, saturation top). Two new
  pins (24 total).
- Also independently re-measured: `cargo test --lib` = 186 passed / 0 failed
  (the "186/186" claim reproduces exactly as the lib-only scope; full
  `cargo test` = 204 passed / 2 ignored). The dynamic pytest scope
  (`pytest tests/test_dyn_*.py`, 1664 collected) measured **1639 passed /
  24 failed / 1 skipped in 25 min (exit 1)** — the external "125 of 125"
  claim does not reproduce in this scope; all 24 failures are pre-existing
  tree-state failures (dashboard promoted-binding ×4, e5_market/e5_r1 ×4,
  ensembles ×6, governance manifest, eseds scorecard, ha_coupling,
  policy_grammar, quantum-sandbox honesty ×2 minus one duplicate, rust
  advanced-ops, stem Hamiltonian, ui sandbox/studio), each reproduced
  identically in a 74 s isolated rerun of the 24 node IDs on the committed
  tree — an additive-only commit was present during that rerun, so the new
  files cannot be their cause.

### Tutorial notebook refreshed to the current platform record (2026-10-21)

`notebooks/apedp_scientific_discovery_tutorial.ipynb` (guard: `tests/
test_dyn_notebook_adapter.py::test_tutorial_notebook_structure_and_execution
— every cell executes in one sequential namespace) was updated to the
current frozen record:

- **Badge launch URLs fixed**: the Colab / Kaggle badges pointed at
  `owner/science` and `blob/main/` — neither exists. They now target
  `psalmprax/science` at `blob/master/` (the only launchable default branch,
  verified by rendering the GitHub view).
- **Step 5 demo-vs-production marker**: the tutorial's synthetic holdout
  demo (`cand_alpha_01`/`cand_beta_02` on `holdout_slice_Q3`) now recreates
  its demo ledger when a previous run already burned the slice at the
  default path (`[demo-reset]` notice). Production holdout ledgers use other
  filenames and are never touched (AGENTS.md §1.2 discipline unaffected).
- **Step 7 — micro-sample symbolic discovery**: live `RimeProposer` search
  on Langmuir 1918 (N=11) plus the structure-aware, graph-validated referee
  (`adjudicate_structure_aware`, `certify_origin_constrained_rime`,
  `sound_monotonicity_v2`); cells reproduce the frozen semantics (saturation
  family UNIQUE_WINNER, PROMOTED with `origin_strict_satisfied` reported as
  False, no `|b|` tolerance gating).
- **Step 8 — companion pathways**: SIR epidemiology discovery via
  `CompartmentalSirProposer` (fitted β=0.2996/γ=0.1199/R0=2.499 against
  true 0.30/0.12/2.50 with exact mass conservation) and the frozen
  Michaelis-Menten pathway re-executed through the graph-validated referee
  (12/12 folds PROMOTED, actual-RIME LOOCV RMSE 0.1316, tournament PROMOTED
  6/6) — matching `benchmarks/rime_microsample_adjudication_results.json`
  digit-for-digit.

---

## [Unreleased] — Repair Python 3.12 blind-freeze CI failure (2026-10-20)

### Lorenz public benchmark: prereg frozen + confirmation harness built (2026-10-20)
- Context: an external evaluation transcript (bare-BIC Lorenz trial) claimed a
  successful "validation" while its own stress test showed the spurious x²y
  term winning training-BIC; its operational model mixed ground-truth constants
  into the "discovered" physics and validated by 3D-plot eyeballing (explicit
  Euler diverges from RK45 truth at t≈0.26–0.34 even with exact coefficients,
  so the plot instrument is uninformative). This prereg turns that exchange
  into a properly measured registered protocol.
- `docs/PREREGISTRATION_LORENZ_PUBLIC_BENCHMARK_V1.md` — FROZEN: catalog Lorenz
  world σ=10/ρ=28/β=8/3, seeds {101,202,303} × noise {0.5–3%} (12 registered
  runs/path), SG(21,3)+np.gradient estimator declared, 70/30 dev/sealed-holdout
  split with full-lineage burn; four registered paths — PATH-R (referee-gated
  forward selection, dev-BIC ordered), PATH-B (greedy forward training-BIC),
  PATH-B′ (curated 4-candidate transcript ranking), PATH-C (user-instructed
  second negative control: the true-constants chimera + magnitude pruning +
  Euler validation — a plot-validated anti-pattern registered specifically to
  test the instruments); hypotheses H1–H6 with margin provenance labeled
  [Pilot]/[Design]. H4's drafted 0.5-SE floor was REFUTED by the 12-run pilot
  (SG derivative bias is systematic and WORST at low noise: 1.4–2.0 SE at
  0–0.5%) and corrected to ≤ 2.5 SE PRE-freeze, never post-hoc. Pilot
  observations: the referee never promoted the BIC winner (0/12); greedy BIC
  admits 3rd-order spurious terms 12/12 (stronger than the transcript's
  curated ranking); referee settings frozen at n_boot=1000/seed=7.
- `benchmarks/run_lorenz_public_benchmark.py` — the frozen harness (audit-only):
  implements all four paths over the registered grid, computes the registered
  instruments verbatim, burns the sealed confirmation windows in the full-
  lineage ledger (`lorenz_public_benchmark_ledger.jsonl`), writes
  `benchmarks/lorenz_public_benchmark_results.json` with git-head + argv +
  timestamp provenance, and a `--smoke` wiring mode (single seed, 100 boot)
  explicitly labeled non-confirmation. The same-integrator divergence-horizon
  control makes integrator error cancel by construction (the H6 instrument).
- Pins: `tests/test_dyn_lorenz_public_benchmark.py` (3 — payload schema, H1
  wiring margin, H6 chimera-catch margin; smoke settings). Smoke verified
  green (3/3) plus harness exit-0; results artifacts commit separately after
  the full confirmation runs per the prereg's decision rule. Expected honest
  outcome risk disclosed up front: PATH-R's smoke wiring missed the true
  structure (the spurious x² proxy promoted first); H3's [Design] margin may
  come back NOT CONFIRMED — that is a registered outcome, never a post-hoc
  harness redesign.

### Frozen referee record refresh
- The Actions regression failure was `verify_referee()` rejecting
  `apedp_dyn/sandbox/multi_grammar_sandbox.py`: its current SHA-256 no longer
  matched the v1.7 pin after the Python 3.12 CI regression fix added the file.
  Preserved the integrity check and minted `APEDPX_FROZEN_v1_8.json` with the
  current source hash; `blind_benchmark.py` now consumes v1.8. No thresholds or
  runtime behavior changed in this record-only refresh.

## [Unreleased] — register limitations addressed: F14b wiring, F16 CBF certificate, LEAN reproduction (2026-10-20)

### Full-suite failure triage and first-wave fixes (2026-10-20)
- Reproduced the full red state in one non-deselected run:
  `pytest tests/ -q` → **97 failed / 3,321 passed / 4 skipped / 68:35 (rc=1)**.
  Triage below; first wave of true-root-cause fixes landed this commit:
  1. **Facade-observer regression → 62 tests red** (`tests/test_facade_contracts.py`):
     the shared detector scanned the stale `build/lib` packaging artifacts
     (`tools/coverage_audit.py` had `build`/`dist` missing from `SKIP_DIRS`; the
     built node_modules are byte-for-byte shims of the real facades), then
     importlib can not import `build.lib.apedp.*`. Fix: `build`/`dist` pruned
     from the audit's own source-tree scans (same rationale as the
     `.kilo/worktrees` exclusion: not the tree under test). `tests/
     test_facade_contracts.py` now 65/65 green. Facade-pin tools regression
     green: `test_coverage_audit`(23) + `test_affected_tests`(22) +
     `test_check_tools_fail_loudly`(10).
  2. **Trading E4T1 data-plumbing import never wired (7 red)**:
     `test_dyn_trading_e4t1.py` reads snapshots via `T1.load_frames()` (v2's
     docstring pins the same panels as v1); was written against `T` (v2) that
     never re-exported the loaders. True fix: switch `_panels()` in the test's
     setup path to v1 (data-identical), keep T for hypotheses and T1 for v1's
     protocol. `tests/test_dyn_trading_e4t1.py` now 9/9 (was 2/9).
  3. **RIME_SUITE_TOTAL 212 → 213** (`tests/test_dyn_reporting_integrity.py`):
     every file the machine-verification counter scans was re-collected;
     actual = 213. Per that test's own failure message the constant is
     re-derived, never hand-edited from prose.
- **Remaining 27 failures, triaged but NOT silently re-pinned** (each needs a
  data- or referee-recalibration decision, which is exactly why they were not
  jogged): (a) physics/adjudication flips on borderline regimes — dashboard
  (4, promoted binding now MULTI), quantum_vacuum/sandbox/ui_sandbox creep
  abstains (3, expected MULTI got UNIQUE): opposite-direction verdict drift
  through the shared discovery/adjudicate path; (b) market-panel numeric pins
  drift (16: e5_r1 3, ensemble 3, gated_ensemble 3, e5_market 1, policy
  grammar index 1, ha_coupling 1, eseds scorecard 1, ui_studio series
  reconstruction 1, system classes/gate counts/R²): all read the Oct-2
  `crypto_top15_orderflow.csv` snapshot — needs one joint re-derivation
  session over a declared data version; (c) contract-evolution pins:
  `stem_grammar` strategic-arch (1), `external_60` near-OOD-floor (1),
  scale-invariance 1e9 edge (1), `tier_b` Planck R² 0.94 vs 0.95 floor (1),
  `rust` native GA advanced-ops (1); (d) governance-manifest SHA (1): needs
  renewed trustee certification (external).

### Local credential remediation: PAT stripped from the remote URL, credential helper installed (2026-10-20)
- `remote.origin.url` rewritten from the token-embedded URL to the bare
  `https://github.com/psalmprax/science.git`; repo-local
  `credential.helper = store` installed. Public read path verified live
  (`git ls-remote origin HEAD` rc=0); the next push will prompt once and the
  helper persists the credential (plaintext `~/.git-credentials` — prefer
  `gh auth login` or `git-credential-libsecret` to avoid that if available).
- Workspace sweep: the embedded-token URL pattern matches NO tracked file
  and only one vendored third-party fixture (`.venv/.../LibGit2/test/
  libgit2-tests.jl`) — the leak is local-config-only. **The exposed PAT is
  still live until deleted on GitHub** (Settings → Developer settings →
  Personal access tokens): rotation there is the user's step — an agent
  session cannot do it without account credentials.

### REG-F14 — restart-proof sealed-slice consumption + fail-closed ledger writes + canonical-spec shrink import (2026-10-20)
- `apedp_dyn/epistemic/caller_data_protocol.py`:
  1. **Durable lead state (I2 across restarts):** `CallerSplitProtocol` now
     persists issued leads and sealed-consumption provenance to an atomic
     JSON state file (`persistence_path` argument, or derived from the
     attached ledger's persistence directory, e.g.
     `caller_lead_protocol_state.json` next to `experiment_ledger.jsonl`).
     A restarted process reloads the store: an open lead cannot be re-issued,
     a consumed slice cannot be re-consumed (`ConfirmationAlreadyConsumedError`
     from the durable record), the slice is claimed spent BEFORE evaluation
     (a mid-evaluation crash leaves it spent, never resurrectable), and a
     corrupt store fails CLOSED (`CallerProtocolStateError`) — the protocol
     never silently falls back to memory-only operation. Without a resolvable
     state path (in-memory unit pins), in-memory-only behavior is retained as
     a documented unit-use limitation.
  2. **Fail-closed I3:** the silent `except Exception: pass` around ledger
     exposure writes in `SealedConfirmationSlice.consume` is REMOVED —
     ledger write failures now propagate. Additionally, with a ledger
     attached, `consume` refuses an UNREGISTERED candidate BEFORE spending
     the sealed slice (`CallerComplianceError`), so an evaluation that could
     not be written up afterwards can never happen.
  3. **No hardcoded threshold literal:** the mode-(b) `dsr_shrink`
     disclosure constant is now imported from the frozen canonical spec
     (`apedp_dyn.conf.canonical_spec.DSR_THRESHOLD`, hash-verified at
     import). The spec JSON itself was NOT modified — it is quorum-signed
     under `specifications/governance_manifest.json` (external trustees);
     changing a preregistered constant requires a re-signed revision, not an
     edit. Values are unchanged (0.95).
- Pins updated: `tests/test_dyn_caller_data_protocol.py` (7 → 14: end-to-end
  restart durability, open-lead restart refusal, aborted-evaluation spent
  claim, corrupt-store fail-closed (3 shapes), unregistered-candidate
  pre-spend refusal, ledger-failure propagation, canonical-spec import pin);
  `tests/test_dyn_caller_compliance_wiring.py` updated for I3 (mode-(a)
  register-first discipline). Two wiring-census rows in
  `docs/DOCUMENTATION.md` + the register updated accordingly.
- **Pre-existing (NOT introduced here), disclosed:**
  `tests/test_dyn_governance.py::test_canonical_repository_governance_manifest_valid`
  fails at the committed tip independent of this work — the governance
  manifest pins spec SHA `a6d026…` while the spec on disk hashes `a11fd957…`
  (the same value `canonical_spec.py` import-check passes against). Fixing
  it requires trustee re-certification of the manifest, not a code edit.

### Frozen-record re-mint v1_7 → v1_8 (rationale, per freeze protocol)
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_8.json` (supersedes v1_7;
  `blind_benchmark.py` now consumes v1_8 via `FROZEN_RECORD`). Only guard-file
  hashes refreshed; thresholds, env pins, and blind-case expectations
  unchanged from v1_7.
- Remint rationale (2 entries, both verifiable in-tree):
  1. `apedp_dyn/sandbox/multi_grammar_sandbox.py` — legalizes drift introduced
     by commit `3419425` (purge forward-overlapping development targets) that
     landed AFTER the v1_7 mint without its own re-mint (v1_7's hash for that
     file was stale). File is worktree==HEAD at mint; harness fix, not a
     referee edit.
  2. `apedp_dyn/sandbox/multi_asset_signals_sandbox.py` — additive REG-F14b
     per-caller compliance footer (`publish_caller_compliance_record`
     delegating to `caller_registry`); no method, threshold, or behavior
     change; mandated by the freeze protocol before blind scoring proceeds.
- Post-mint verification: `pytest tests/test_dyn_blind.py
  tests/test_dyn_blind_audit_round2_remediation.py` → 55 passed,
  verify_referee() satisfied.

### REG-F14/F14b — per-caller wiring over the production-caller census (mode b)
- `apedp_dyn/epistemic/caller_registry.py`: auditable census of six production
  data-consuming verdict publishers (dyn_sweep, multi_asset_signals,
  real_market_l2_sweep, blind_benchmark, calibrated_benchmark_suite,
  canonical_ledger); each caller module wired with a thin
  `publish_caller_compliance_record` publisher; records validate via the F14b
  either-or validator at publish time; `record_selection_exposure` records the
  selection-stage `TRAINING_DEVELOPMENT` exposures that make the derived
  multiplicity live. Pins: `tests/test_dyn_caller_compliance_wiring.py` (8).
  Honest boundary: every census caller is wired in mode (b) — no production
  pipeline operationally holds a sealed confirmation split; mode (a) stays
  available per caller and is claimed for no one.

### REG-F16 — formal CBF-QP certificate + model-check artifact (+ scaled verification)
- `apedp_dyn/robotics/cbf_certificate.py` + `tests/test_dyn_cbf_certificate.py`
  (8 pins) + `benchmarks/cbf_certificate_report.json` (battery exit 0, 64
  certified releases verified, worst slack −3.3e-16; unicycle realization
  model-checked through the PRODUCTION arc stepper over seeded adversarial
  head-on sweeps — min barrier 0.3588, 0 collisions). FORMAL scope is at the
  certified velocity layer under an explicit assumption object (refuses
  γ·dt > 0.5); the unicycle realization is disclosed as model-checked
  evidence, NOT formal. `batch_multi_agent_cbf_step` is untouched and its
  not-a-certified-CBF disclosure pin intact; `certified_cbf_guard` is additive.
- NEW scaled-verification layer in the same battery: the same fail-closed
  formal checks re-verified on deterministic jittered-lattice grids at
  16/32/64/128 agents (1440 additional releases, worst slack −9.91e-10 ≥ −tol;
  Euler invariance h > 0 at every step at every scale; realized guard
  collision-free at every scale under the physically-consistent
  realized-velocity-feedback nominal discipline — 128-agent realized min
  barrier 0.3919) plus recorded per-step WALL-CLOCK guard and guard+arc-stepper
  latencies (measurement of the environment that ran the battery, explicitly
  not a guarantee; mean guard+step 0.60/0.77/1.31/3.39 ms at 16/32/64/128
  agents). A separate DISCLOSED scope probe records verbatim that fresh
  full-range random nominals redrawn every step at 128 agents are OUTSIDE the
  realized layer's evidence (deterministic breach −0.1102, 17 collision steps)
  — the realized layer stays scenario-scoped evidence.

### REG-F14/F14b — mode (a) operational wiring landed (blind_benchmark first)
- `apedp_dyn/epistemic/caller_registry.py`: mode-(a) registry helpers
  `issue_sealed_confirmation_lead` (census- and dataset-family-checked) and
  `publish_sealed_compliance_record` (refuses publication while the sealed
  confirmation slice is unconsumed, with foreign provenance, or for a foreign
  head; demands one-shot provenance and `archived_verdict_source =
  "confirmation"`).
- `apedp_dyn/sandbox/blind_benchmark.py`: first census caller wired for mode
  (a) over its OWN real slice axis — `open_sealed_confirmation_lead`
  (selection = worked-example/selection case ids, confirmation = the SEALED
  blind case ids), `evaluate_confirmation_cases` (frozen-referee-checked,
  closed confirmation family, blind-metrics verdict),
  `publish_sealed_compliance_record`.
- Pins: `tests/test_dyn_caller_compliance_wiring.py` 8 → 12 (sealed consume
  exactly once; second candidate refused; publication refused while
  unconsumed / with foreign provenance; sealed record validates).
  Honest boundary: the five remaining callers remain mode (b) by default and
  the pinned mode-(a) demonstration is the wiring path — a live production
  promotion through the sealed slice is pending, disclosed as such.

### REG-LEAN — pinned-toolchain reproduction of the 323-record sweep
- `scripts/run_lean_pinned_reproduction.py` +
  `benchmarks/lean_pinned_toolchain_reproduction.json`: ALL 323/323 records
  reproduced (proof_claim_status + kernel_exit_code per candidate) under the
  independently provisioned pinned toolchain `leanprover--lean4---v4.34.1`
  (commit `5045d0056413266e57c625dcd7c365b10e377c52`) vs the reference's
  v4.35.0; counts 323/315/8/0. Reference artifact + committed `.lean`
  evidence restored byte-identically (HEAD-committed bytes take precedence
  over regeneration — regeneration silently un-remediates the u15 honest
  disclosure; caught and fixed in the runner).
- **Root-cause fix:** `apedp_dyn/epistemic/lean_universal_bridge.py` compile
  subprocess timeouts raised 15/25 s → 240 s (docker paths unchanged except
  none needed): under 8-way parallel contention the two heaviest fluid-cell
  files (NavierStokes / forced-ODE) exceeded the old timeout and were reported
  P1 (exit −1) while compiling clean standalone, so the reference artifact
  could not be reproduced as-is. Exit codes now reflect verification, not
  wall-clock contention. The register's external (outside-this-environment)
  sign-off remains outstanding and is NOT claimed.

### ERR-INC-80 — §80 pins re-derived under the forward-overlap-purged harness (disclosed)
- Commit `3419425` purged the 5-day-ahead window from the shared development
  targets (`trading_e4t1_v2.run_origin`) — a legitimate leakage fix that
  shifted the frozen §80 headline magnitudes, leaving 4 of 5 pins in
  `tests/test_dyn_incremental.py` failing on the tree.
- Same seed (8080), same artifacts, no threshold moved: the pins were
  RE-DERIVED (deterministic, verified) under the purged harness and
  SHARPENED (exact per-asset values incl. permutation p, corr(ŷP,ŷA),
  BTC win row; pre-registered headline rule perm_p >= 0.05 kept and made
  load-bearing again). The headline decision (Case 2/4, permutation-null
  everywhere — min .0896, ETH) is unchanged; only the magnitudes moved
  because the leak they were computed under was purged.
- Full disclosure table in `docs/TRADING_EQUATION_FORENSICS.md` (§80b).

### REG-MUT — targeted mutation battery, CI-gated (CLOSED-WITH-EVIDENCE)
- `scripts/run_mutation_battery.py`: dependency-free battery for the register's
  three targets (Finding-14 adjudicate anchor ×2 mutants, F8
  frozen-library-term-resolution fail-closed + diagnostic-detail ×2, hawkes
  standardization ×1), each pinned via targeted pytest runs in fresh
  interpreters with byte-identical restoration verified and pristine sanity
  pins. Exit 0 requires ALL mutants killed; a survivor is a REG-MUT failure.
- **Honest first run:** MUT-C1 (eval-frame-leak standardization) SURVIVED —
  the inherited dynamic pin scored a fit/eval-identical frame, so it could
  not distinguish the property. The pin was SHARPENED
  (`tests/test_dyn_hawkes_standardization_pins.py`: degenerate-fit continuous
  fallback + distribution-shifted eval frame; Finding-10 property now
  load-bearing), never weakened; re-run: ALL_KILLED (5/5, pinned report at
  `benchmarks/mutation_battery_report.json`).
- CI gate: new `mutation-battery` job in `.github/workflows/test.yml`
  (survived mutant fails CI; report uploaded, never hidden) and
  `scripts/check.sh` full tier. Register row updated with status + honesty
  trail (`docs/SYSTEM_FLAWS_AUDIT.md` register status updates).

### JetFlow v0.3 collision-world confirmation — honest H1 boundary result
- `benchmarks/run_jetflow_video_v03_confirmation.py` ran the frozen
  confirmation seeds {407, 508, 509} (exploratory 404 excluded):
  **H1-collision CONFIRMED 3/3** — the fluid-domain JetFlow advantage does NOT
  transfer (diffusion wins SSIM + temporal on every seed): the published
  advantage is DOMAIN-SCOPED to continuum transport. H2 (PSNR floor), H3
  (train-ratio ceiling) CONFIRMED; H4 CONFIRMED after **ERR-H4-UNITS**, a
  disclosed units erratum: the prereg text mis-stated the passes metric as
  amortized-over-n_gen; verdicts recomputed from the SAME committed per-seed
  artifacts — no re-run, no threshold moved, no seed changed.
- Report: `docs/JETFLOW_VIDEO_V03_CONFIRMATION_REPORT.md`; verdicts:
  `benchmarks/jetflow_video_v03_confirmation_results.json` (+ per-seed
  artifacts under `benchmarks/jetflow_video_v03_confirmation/`).

### Caller-layer three-role data protocol (REG-F14/F14b reference implementation)
- `docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md` +
  `apedp_dyn/epistemic/caller_data_protocol.py` (`experimental`, additive):
  disjoint SELECTION/CONFIRMATION/PROSPECTIVE roles at issue time; sealed
  one-shot confirmation consumption (REG-F14 letter); confirmation exposure
  records integrate the existing anti-snooping burn ledger; archived verdict
  source must be `confirmation` with selection superseded; F14b either-or
  compliance (sealed split OR ledger-derived DSR-adjusted multiplicity —
  computed, never hand-written; refuses deflation; reuse events inflate).
- Acceptance pins `tests/test_dyn_caller_data_protocol.py` (7) include the
  register's end-to-end letter: pool on slice A, head confirmed exactly once
  on sealed slice B, verdict provenance from B only, A∩B=∅, second consume
  refused. Register status: OPEN — per-caller wiring of every production
  caller (mode a/b) is still pending and remains the acceptance gate.

## [Unreleased] — E4 divergence-point selection, second JetFlow world, verified-knowledge-module library (2026-10-19)

### Gap-1 generalization: cost-aware divergence-point selection (`experimental`)
- `apedp_dyn/epistemic/active_disambiguation/divergence_selection.py`: extends the fixed
  E4 measurement menu to proposing the query point itself on a continuous domain, ranked
  by EXPECTED COST TO RESOLUTION (`ceil((k·σ/gap)²)·cost(x)`) under an explicit budget —
  not raw gap maximization. Named abstention causes (`INSUFFICIENT_DIVERGENCE`,
  `INSUFFICIENT_BUDGET`), preserved P2/P3 no-truth-leakage (selection sees only
  hypotheses + observation/cost model), P6 re-adjudication through `adjudicate()`,
  and P8 determinism (frozen grid + deterministic refinement; zero randomness in
  selection). `policy.py` refactor made the shared `DEFAULT_DETECTABILITY_K` explicit
  (behavior-identical; 20 E4 regression tests pass). Pins:
  `tests/test_dyn_divergence_selection.py` (8 tests: analytic optimum near x*=−0.2
  for the constructed pair, cost-awareness, named abstentions, no-truth-leakage,
  determinism, budgeted closed-loop resolution to UNIQUE_SUPPORT).

### Second JetFlow problem family: collision/occlusion world (`experimental`) — honest negative
- `apedp_dyn/video/collision_domain.py`: layered 2D rigid-body scenes (same-layer elastic
  collisions + wall bounces; cross-layer pass-through with genuine parallax occlusion —
  real disappearance/reappearance, pinned). Byte-reproducible by construction;
  manifest split-exclusivity via the shared `held_out_split` instrument.
- `benchmarks/run_jetflow_vs_diffusion_matched.py`: `--domain` parameter added
  (default `fluid` preserves the original exploratory path); instrumentation converted
  to grad-free `_to_np` (fixes a latent `requires_grad` crash in the student path).
- **Exploratory seed-404 result is a disclosed NEGATIVE result for advantage transfer:**
  on the collision world the matched diffusion baseline wins every instrument
  (SSIM 0.065 vs 0.017; PSNR 20.8 vs 13.7 dB; temporal 0.078 vs 0.234; SWD 0.167 vs 0.476)
  and trains faster (8.2 s vs 11.1 s @300 steps). Conclusion per prereg: the fluid-domain
  advantage is DOMAIN-SCOPED to continuum transport; no unscoped generalization claim.
- New prereg frozen: `docs/PREREGISTRATION_JETFLOW_VIDEO_V03_COLLISION.md`
  (confirmation seeds {407, 508, 509}; H1-collision replicates the ordering on ≥2/3 seeds;
  v0.2 fluid protocol untouched per the deviation rule). Boundary cross-reference added to
  `docs/jetflow_vs_diffusion_matched_benchmark.md`. Pins:
  `tests/test_dyn_collision_domain.py` (8 tests).

### Gap-3 realization: Verified Knowledge Module Library (`experimental`)
- `apedp_dyn/epistemic/verified_knowledge_module.py`: read-only additive layer over the
  existing ledgers (they remain the single source of truth). Entry gates: experiment-ledger
  `CONFIRMED_VALID` + promotion Stage 4 (out-of-sample transfer); economic stages 5–6 are
  deliberately NOT required (domain-general library; disclosure shipped on every record).
  Metadata contract: preconditions required; finite ordered validity domains;
  uncertainty requires value + estimation method (no naked floats); counterexamples
  REQUIRED (a module without a documented failure is not verified knowledge); dependency
  closure at registration and in composition; composition INTERSECTS validity domains and
  refuses disjoint/out-of-chain reuse. Pins: `tests/test_dyn_verified_knowledge_module.py`
  (12 tests).

## [Unreleased] — Round-3 external assessment record (2026-10-19)

- Registered the L2 holdout R² base floor, F-test significance level, and fitted
  predictor degrees of freedom in the canonical preregistered specification;
  pruning now consumes those pinned values rather than hardcoding the 95% gate.
- Post-round-3 external review: **6.5/10 overall** (up from ~6/10), credited to
  reliability and audit discipline, while explicitly noting the remediation
  does not by itself establish stronger scientific discovery capability or a
  demonstrated advantage over competing methods. Scorecard and the
  reviewer-stated criteria for reaching 8/10 (REG requirement closure with
  per-item acceptance evidence, independent review of mathematical
  certificates and critical code paths, reproducible external evaluation of
  discovery validity) and 9/10 (measurable advantage over established
  alternatives across multiple independent problem families) are recorded in
  `docs/SYSTEM_FLAWS_AUDIT.md` ("Round-3 external assessment record") as
  forward requirements — not claimed progress.

## [Unreleased] — Remediation checkpoint frozen at `47594e7`; open-requirements register (2026-10-19)

- `47594e7` is tagged `remediation-checkpoint-47594e7` as the frozen
  remediation milestone. Per the external review's final recommendation, F14
  (three-role data protocol), F16 (formal CBF-QP certificate), mutation
  testing of referee/leakage paths, and an independent Lean toolchain run are
  recorded as **explicit unresolved requirements** with exact acceptance
  criteria in `docs/SYSTEM_FLAWS_AUDIT.md` ("Outstanding requirements
  register", IDs REG-F14, REG-F14b, REG-F16, REG-MUT, REG-LEAN). None may be
  marked complete silently.
- F17 scale-aware gate — reviewer acceptance items closed: the polynomial
  library's constant `1` column had pinned the term-scale denominator at O(1),
  defeating scale-awareness below O(1) data (a mis-fitting linear candidate at
  near-zero scale was certified permissively). The term scale is now measured
  over NON-constant columns; near-zero-scale exact laws certify scale-
  consistently while misfits refuse. Degenerate inputs pinned: all-zero and
  constant data (no NaN via the 1e-12 floor), <10 samples and NaN inputs
  rejected explicitly at the binding boundary. Rust lib 186/186. New pins:
  `TestF17ExtremeScaleRobustness` (6 tests), `TestF8ErrorDiagnosticDetail`.
  Round-2 suite total: 46 tests.

## [Unreleased] — External-review round (cont.): F17 scale-aware gate, F8 fail-closed mirror, Lean chain pin (2026-10-19)

Continued from the review-response round above (same external report, remaining "additional scrutiny" items answered in code rather than in prose):

### F17 — scale-aware exact-recovery gate (Rust `rime/implicit.rs`, rebuilt)
- The exact-recovery gate is no longer an absolute `rmse < 1e-5`. It is measured
  RELATIVE to the RMS magnitude of the relation's own residual terms, so
  O(1e5)-scale genuinely exact relations (the audit's QEC-scale class) are
  certifiable while small-scale near-misses are refused at the same relative
  tolerance (1e-5). Verified through the rebuilt pyo3 binding: O(1) circle
  exact (rmse 7e-16), 1e5-scale exact recovered (rmse 9e-13), small-scale
  1e-4-perturbed relation REFUSED. The "0 = 0" tautology guard stays
  scale-independent via the active-term count. Rust lib: 186/186. Pinned by
  `TestF17ScaleAwareExactGate` (4 tests).

### F8 — frozen-library mirror drift fails closed
- `_frozen_residual_r2` no longer silently skips dev-discovered active terms
  that the Python cold-end mirror cannot resolve — a dropped column corrupted
  the frozen R^2 toward flattering values (the original F8 defect class).
  Unresolved terms now raise naming the drifted term(s). Pinned by
  `TestF8FrozenLibraryTermAudit`: fail-closed raise, term-classification pin
  (`u^3` is a cubic STATE nonlinearity, `u_xxx` the third-order spatial
  derivative — distinct series, per the reviewer's classification remark), and
  a mirror-coverage pin that every Rust 1D catalog term (including `u^3` and
  `u_xxx`) resolves in the Python mirror.

### Lean — complete dependency-chain integrity pin
- The reviewer's request to "compile the complete dependency chain and check
  for unintended axioms or unproved assumptions" is pinned mechanically: for
  every P2-verified record in the regenerated provenance (315 records), the
  emitted Lean source in the tracked `apedp_dyn/lean_formal/` directory must
  be `sorry`-free, and every `sorry`-carrying file must belong to a
  P0-refuted record — the sorry channel can no longer certify under any
  label drift. (`TestLeanProofChainIntegrity`, 2 tests over the full family.)


Follow-through on the external remediation review (rating round). All items below are
pinned by `tests/test_dyn_blind_audit_round2_remediation.py`.

### F14 — holdout selection-reuse disclosure
- The reviewer's composition critique is accepted: the pair verdict selects the
  promotion head on the SAME holdout slice that later certifies it, so that slice
  is a selection set, not an untouched confirmation set. The referee cannot mint
  new confirmation data, so every pair-path `UNIQUE_SUPPORT` verdict now carries
  the explicit diagnostic flag
  `selection_reused_confirmation_holdout(n_candidates=<N>)` (the pool size that
  was compared on the slice). Callers MUST charge that selection to the DSR trial
  ledger (`search_id` + `selection_pool_size` — the standing multiple-testing
  correction) or hold out an independent sealed confirmation split; a FULL
  contract claim without that separation is misstated and callers/auditors can
  now detect it mechanically from the verdict record. The former
  `test_tau_shrunk...` fixture documents the two-directional τ rule (Δ ≤ τ ->
  `MULTI_HYPOTHESIS`; Δ > τ -> `UNIQUE_SUPPORT`), establishing that uniqueness
  comes from the prespecified τ, never from the (construction-guaranteed)
  sign of the separation.
- Re-minted: `adjudicate.py` SHA pin (`f56987b9…`) in `frozen_spec.py` and
  `APEDPX_FROZEN_v1_7.json`.

### F4 — multiplicity coverage pins
- Pinned: the ledger charges every recorded trial including repeated
  evaluations of the same structure under different seeds/hyperparameter
  settings (6 records under one `search_id` -> 6 trials), and charges the full
  declared pool beyond recorded survivors (40 declared, 2 recorded -> 40
  trials, `hidden_trials=38`), with the DSR verdict decreasing under the
  hidden multiplicity at matched variance.

### F5 — annualization numerical pin
- The fixture (annualized SR 2.5, 500 trials): `annualization_factor=1.0`
  manufactures significance (p < 1e-6, `is_significant=True`); the
  F5-corrected call passes `annualization_factor=252.0`, yielding SE on the
  annualized scale (≈1.0) and `p ≈ 0.094` (not significant). Both directions
  pinned numerically.

### F16 — safety-envelope battery (not a CBF certificate)
- The reviewer is correct that the pull-away deflection is a heuristic, not a
  forward-invariance guarantee. The stepper's docstring already states it is
  "not a certified quadratic-programming CBF-QP with formal
  forward-invariance certificates"; this disclosure is now itself pinned by a
  test so it cannot silently regress into an overclaim.
- New behavioral envelope pins (all on the portable CPU path):
  finiteness throughout; measured inter-agent separation floor > 0.6 (with
  `r_safe=0.5`) across a collinear head-on encounter, a high-relative-velocity
  approach (2.5u/s closing speed), a dt sweep over [0.05, 0.10, 0.20], and
  simultaneous multi-pair engagement of 4 agents; exact agent-index
  permutation symmetry (max deviation 0.0); finite escape dynamics from
  degenerate coincident starts; and the F16 regression class itself —
  forward progress > 0.4 into a symmetric head-on encounter where the
  pre-fix radial-only deflection deadlocked.
- A formal `dot(h) + alpha(h) >= 0` CBF-QP synthesis remains future work
  (tracked here, not smuggled into the heuristic's claim).

### Honest-scope note
- Reviewer items NOT closed by code here and deliberately not claimed:
  (a) F17's `rmse < 1e-5` gate is still an absolute threshold — a scale-aware
  RMSE normalization is a kernel change deferred with its own acceptance
  criteria; (b) F8's mathematical compatibility of the non-uniform fitting
  and scoring libraries needs a dedicated term-audit, not a test tweak; and
  (c) "full test suite" in prior notes meant the AGENTS.md dynamic battery
  (a named collection), never the entire repository — wording now matches.

## [Unreleased] — Blind-audit remediation round 2: F14–F17 + freeze re-mint (2026-10-19)

### Frozen-record re-mint (rationale, per freeze protocol)
- `apedp_dyn/epistemic/adjudicate.py` (F14 remediation below),
  `apedp_dyn/sandbox/grammar_space.py` (F1 causal-target remediation below),
  `apedp_dyn/sandbox/trading_signals_sandbox.py` (F5 annualization wiring), and
  `apedp_dyn/ui/server.py` (F7 state-series dictionary reconstruction) are
  frozen guard files; all four required edits in this round. Both freeze
  surfaces were re-minted: `frozen_spec.py`'s pinned SHA-256 registry (for the
  two files it registers) and a new blind-protocol record
  `apedp_dyn/sandbox/APEDPX_FROZEN_v1_7.json` (supersedes v1_6;
  `blind_benchmark.py` now consumes v1_7 via `FROZEN_RECORD`): each remint
  rationale is logged in the record's `remint_log`. Referee threshold semantics
  and grammar-space primitive sets are untouched; verification is covered by
  `tests/test_audit_open_flaws_remediated.py::
test_all_frozen_records_verified_against_pinned_hashes`, `tests/test_dyn_blind.py`
  (`verify_referee()`), and `tests/test_documentation_and_spec_integrity.py`.

### Critical fixes (new audit Findings 14–17)
- **F14 — `apedp_dyn/epistemic/adjudicate.py`:** the pair evaluation was anchored
  on the train-loss ordering. An overfitted train-loss winner made
  `delta = rival_ho − best_ho` negative, trivially tripping `delta <= tau` for
  any `tau > 0` and denying `UNIQUE_SUPPORT` to genuine generalizing candidates
  (misclassified `MULTI_HYPOTHESIS`). The pair anchor is now the holdout-lowest
  RMSE candidate when it is gate-4 certified; `delta` is therefore always a
  holdout separation and the `delta <= tau` check verifies the holdout criterion
  directly before any `MULTI_HYPOTHESIS` verdict. A re-anchored pass falls back
  to the honest point-estimate contract for the bootstrap tiebreaker (residual
  series correspondence cannot be attributed across the re-anchor).
- **F15 — `apedp_dyn/proposers/ncm_structural_proposer.py`:** all four archetype
  branches (BOUNDED_RATIONAL_SATURATION, CONSERVED_DAMPED_OSCILLATOR,
  DIFFUSION_ADVECTION_CONTINUUM, CYCLIC_HARMONIC_REGIME) now emit real
  `compute_score` closures. Each closure builds its design matrix from the
  candidate frame the same way the fit did (frame-local gradients / lens, not
  fit-time arrays), falls back to an all-zero vector only for missing columns,
  and reproduces the fit-time prediction in-sample.
- **F16 — `apedp_dyn/cuda/batch_physics_steppers.py`:** `batch_multi_agent_cbf_step`
  now applies the reference CPU implementation's anti-parallel tangential
  breakaway force (`|anti-parallel| > 0.7` → perpendicular breakaway scaled by
  `anti_par * k_rep * (1/d − 1/r_w)/d^*2`), eliminating permanent deadlock on
  collinear trajectories.
- **F17 — `rust/apedp-kernel/src/rime/implicit.rs`:** `is_exact_recovery` now
  requires `active_count >= 2` in addition to `rmse < 1e-5`; the all-zero
  pruned vector (algebraic residual identically 0, the "0 = 0" tautology) can no
  longer be certified as an exact physical recovery.
- **F10 — `apedp_dyn/proposers/hawkes_orderflow_proposer.py`:** the
  standardized-intensity scorer standardized against the *evaluated frame's*
  mean/std, which over a holdout folds the whole future distribution of the
  point process into every `t`. Scoring standardization now couples only to the
  FIT-time intensity distribution constants (`fit_intens_mean`/
  `fit_intens_std`, the same frozen coupling already used by the surge
  threshold quantile), so the emitted signal is strictly causal at evaluation
  time.
- **F12 — `apedp_dyn/epistemic/lean_universal_bridge.py`:** the prover axiom
  `exp_damp_le_one (s : R) : exp (-s) ≤ (1 : R)` was a false universal
  (`exp(-s) > 1` for `s < 0`). The axiom and the synthesized squeeze/exponential
  damping theorem both now carry the non-negativity hypothesis `(hs : (0 : R) ≤ s)`;
  the bound is only claimed in-damped-regime.

### Lean-proof inventory reconciliation
- The unresolved `test_all_system_equations_lean_proofs.py` failures were two
  distinct problems, both fixed:
  1. The five per-sweep provenance inputs (`fluid_ode_dynamical_sweep_`,
     `openai_math_advanced_hypotheses_sweep_`, `ga_depth2_sweep_`,
     `ga_depth3_sweep_`, `expanded_space_discovery_`) and the cross-domain
     verification record existed only in a temporary IDE artifact folder, not
     in the repository. They are now tracked under `artifacts/` (AGENTS.md
     §1.4: documentation/records must be verified in the repo workspace).
  2. `lean_universal_bridge.py::_liberal_parse_formula` returned non-Expr
     sympy parses (tuple/list forms from bracketed heaviside-indicator
     formulas), which crashed `_ast_polynomial_total_degree` during the full
     sweep. It now accepts only single sympy expressions (trimming otherwise,
     failing closed to None), and the degree helper rejects non-Expr input.
  After regeneration, the reproducible record is 323 evaluated / 315 P2 / 8 P0
  / 0 P1: the 6 additional P0s are grammar-space candidates whose declared
  saturation bounds are not provable over their formulas (unbounded degree-3
  products like `(LeadSpill * DispNorm) * EntropyCert`, and heaviside-indicator
  products like `[OFI * I(Lead > 0)] * VolRatio` that fail closed on AST
  refutation) — the former 321/2 record overclaimed those P2s; the traps
  (`PHYS-ENERGY-LEAK-TRAP`, `FIN-CUBIC-RUNAWAY-POLYNOMIAL` → P0) and all spot
  checks (`STRUCT-MASTER-SIZING-PHI`, `GA-D3-CHAMPION-U90_H60M`,
  `TRADING-SCORE-N90` → P2) are unchanged. The test contract pins the
  regenerated record; no preregistered threshold changed.

### Verification notes
- `tests/test_dyn_grammar_space.py`: the frozen d5-target pin was stale — it
  still pinned the backward window `s.shift(-1).rolling(5)` (80% leakage into
  the "future" target); it now pins the shipped forward-only
  `s.rolling(5).mean().shift(-5)` (aggregate over s[t+1..t+5]) plus NaN-tail and
  finiteness invariants.
- F2 (basis screening holdout leak) and F3 (mismatched SS_tot mask) were
  verified already remediated on disk in
  `apedp_dyn/discovery_l2/multiterm_engine.py` (full-series causal evaluation;
  masked-partition `ss_tot` on `y_ho_m`), pinned by
  `tests/test_audit_open_flaws_remediated.py`.
- **F8 — `apedp_dyn/epistemic/pde_discovery.py`:** Added missing spatial stencil
  term `u_xxx` and polynomial term `u^3` to the non-uniform PDE holdout evaluation
  library, aligning with 1D/2D discovery terms.
- **F9 — `apedp_dyn/proposers/distributed_lag_interaction_proposer.py`:** Replaced
  single `lag_state["buffer"]` with per-column buffer dictionary `lag_state["buffers"]`
  so multi-column lag features do not overwrite each other during streaming evaluation.
- **F10 — `apedp_dyn/proposers/hawkes_orderflow_proposer.py`:** Eliminated full-frame
  mean volume and price-change leakage in arrival weight normalization by using
  fit-time reference scales or causal expanding means.
- **F11 — `apedp_dyn/molecular/se3_flow.py`:** Symmetrized bond pair exclusions in
  `_non_bonded_pairs` so reversed bond tuples `(j, i)` are correctly excluded from
  non-bonded core repulsion.

## [Unreleased] — forensic audit remediation, §4.1 freeze re-mint

- `apedp_dyn/sandbox/multi_asset_signals_sandbox.py` is pinned by
  `APEDPX_FROZEN_v1_4.json` (blind-protocol referee freeze). The §4.1 remediation
  (purge `.bfill()` lookahead from the sandbox fit path → causal forward-fill)
  required editing that pinned file. Freeze protocol: re-mint the recorded
  SHA-256 with the rationale logged in the record's `remint_log`; referee logic
  itself is untouched and re-verified by `test_dyn_blind.py`.
- `test_no_lookahead_fill_in_proposer_fit_paths` now scans all of
  `apedp_dyn/` and flags backward fill only (`bfill`/`backfill`), the actual
  lookahead defect class; forward fill is the prescribed causal remediation.

## [Unreleased] — External-audit remediation: denominators, lookahead, honest referees, path & docs integrity (2026-10-19)

Every item below was reproduced against the pre-fix code and is pinned by a regression test that fails before the fix. No preregistered threshold changed; the canonical spec hash is untouched.

### Statistical/dimension fixes
- `apedp_dyn/sandbox/blind_benchmark.py`: `fpr_blind` now divides false promotions by UNWARRANTED negatives (its documented meaning); the previously conflated promoted-denominator quantity is reported separately as `fdr_blind` (fp / promoted), and `by_domain` rows carry both `fpr` and `fdr`. The binomial `fpr_upper95` bound uses the negatives denominator. `tests/test_dyn_blind.py` pins the distinction (fin sweep: FPR 0.5 vs FDR 1.0) in `test_fpr_and_fdr_use_distinct_denominators`.
- `apedp_dyn/epistemic/novelty_adjudication.py`: HAC long-run variance uses the canonical Newey-West divisor `n` for every autocovariance term (previously `n-k` for `k>0`, which is not PSD); the 1e-12 variance floor is retained and disclosed. Adversarial check over 2000 random 6-column draws: non-negative variance in every draw (min 0.00139).
- `apedp_dyn/epistemic/trial_ledger.py`: selection effects are now charged to the DSR, not merely disclosed. `effective_trial_count` groups records by `search_id` and counts `max(recorded_candidates, selection_pool_size)` per group, so a proposer that searched K candidates and emitted only its best is charged K trials; `dsr_with_selection_adjustment()` reports the adjusted `n_trials`, `naive_trial_count`, and `hidden_trials`. Malformed pool sizes and empty-ledger calls fail closed. `InvariantTransferProposer` now stamps a shared per-call `search_id` alongside `selection_pool_size`/`selection_rank`. Regression tests pin the 500-of-1 charge, shared-pool deduplication, and both fail-closed paths.
- `apedp_dyn/video/eval_gate.py`, `rust/apedp-kernel/src/video.rs`, `apedp/rust_bridge.py`: SSIM stabilizers are the canonical squared constants `C1=(0.01)^2=1e-4`, `C2=(0.03)^2=9e-4`; all call sites, kernel unit tests, and Python/Rust parity tests agree. Legacy disclosure: benchmark numbers recorded before 2026-10-19 (including the 0.514 SSIM row in this file) used the unsquared 0.01/0.03 values; historical measurements are disclosed, not rewritten.

### Lookahead / faithfulness fixes
- `apedp_dyn/proposers/regime_switching_proposer.py`: removed backward-fill in both the fit path and the scorer. Volatility warm-up rows (first `min_periods-1 = 4`) are no longer fabricated from future observations; they are excluded from fitting (`metadata.samples_metric_missing`) and score as NaN. Regression test pins the 4-row warm-up and exact formula coefficients.
- `apedp_dyn/proposers/distributed_lag_interaction_proposer.py`: column roles come from an explicit `column_spec` recipe map, not substring parsing (`_X_`/`_lag` collisions). Coefficients are mapped back to raw scale (`raw_w = coef/x_std`, `bias = intercept - sum(coef*x_mean/x_std)`) so the emitted formula scores the fitted model identically: fixed R^2 0.995982 vs -1249.5174 under the legacy mismatch.
- `apedp_dyn/ode/referee.py`: rewritten to adjudicate from a real RK4 holdout rollout (building RHS terms from the reported monomials, with divergence flagged as `UNSTABLE_FORWARD_SIMULATION`). The discovery-side in-sample `mean_system_r2` is never used as evidence; missing equations or fewer than two holdout rows fail closed as `INSUFFICIENT_EVIDENCE`; unknown state references raise. New tests pin wrong-structure rejection despite in-sample 0.999, unstable rollout, and the fail-closed path.
- `apedp_dyn/proposers/hawkes_orderflow_proposer.py`: `TAIL_RISK` is declared only when the binary surge threshold actually binds on the fit data; a degenerate threshold now emits `CONTINUOUS_FACTOR` with standardized intensities (`metadata.score_mode` discloses the choice). `rime_ad_proposer.py`: an evaluator failure now returns NaN instead of fabricating a function via substring matching. `molecular/se3_flow.py` clamps near-coincident atom distances (`_MIN_PAIR_DISTANCE = 1e-8`) in energy and gradient; `audio/spectrogram_transport.py` returns NaN R^2 for zero-variance targets instead of dividing by zero.

### Rust kernel hardenings (rebuilt, 180/180 lib tests)
- `flow_head.rs::erf_hp`: inputs whose square overflows (`|x| > ~1.34e154`, `+/-inf`) return the saturated `+/-1` instead of NaN from `-inf + inf`; NaN propagates. New saturation test.
- `ode.rs`: stages 3-7 now perform the same `h < min_h` underflow check as stage 2, so a divergent system terminates with an explicit step-control failure instead of burning `max_steps`. New regression test.
- `genetic.rs::score_candidate_affine`: two-pass variance (outputs are already materialized) removes the catastrophic-cancellation misclassification of near-constant predictors with large offsets; a pure constant can no longer receive a spurious slope. New regression test on `Const(123456.789)`.

### Boundary & documentation integrity
- `CandidateProposal.declared_scale_invariant` is documented as declarative and unenforced (ARCHITECTURE.md principle 5; `proposer_interface.py` field note), and `tests/test_audit_open_flaws_remediated.py` now fails if any production module reads it in control flow. `InvariantTransferProposer` discloses search size on every candidate (`degree_trials`, `search_id`, `selection_pool_size`, `selection_rank`, `dev_side_gates`), making the dev-side order-statistic bias explicit and machine-chargeable by the trial ledger.
- ARCHITECTURE.md §3.2 now describes the actual burn mechanism (`record_data_exposure(..., CONFIRMATION_HOLDOUT, outcome_status="FAILED")` plus terminal lifecycle states), §4.1 inventories all frozen generations (v1/v1_1/v1_2/v1_3), and §4.2 records the path policy now implemented by `apedp_dyn/conf/paths.py` (`APEDP_REPO_ROOT`, `APEDP_LEAN_ROOT`, `APEDP_ARTIFACT_DIR`, repo-relative fallbacks). All `/home/<user>/...` literals were removed from `apedp_dyn/`, `scripts/`, and tests; the Lean bridge resolves its output root through `APEDP_LEAN_ROOT`. The live documentation index (`docs/DOCUMENTATION.md`) converted all 272 `file:///home/...` links to repository-relative links, and `test_documentation_links_exist_on_disk` now validates every relative link (301 checked) and rejects absolute `file://` links outright; archived historical documents retain their original provenance URIs by design.
- `docs/DOCUMENTATION.md`: the entire `epistemic/active_disambiguation` package (10 modules) and all remaining `discovery_l2` modules are now indexed, plus `conf/paths.py`. `tests/test_documentation_and_spec_integrity.py::test_orphan_module_detection` now scans the whole `apedp_dyn` tree recursively against a pinned debt ledger that may only shrink. Core packages (gates, ode, pde, market, video, estimation, model, harness, fitting, simulation, validation, experiments), the frozen root sweeps, the 18-problem benchmark catalog, and all 67 historical sandbox harnesses are now indexed: the debt ledger is EMPTY (125 -> 67 -> 0) and there is no pin left to mask a regression. Guard-set/referee-side sandboxes are tagged `production`, UI-exposed ones `experimental`, and forensic/CPMD harnesses `audit-only`.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_3.json`: guard set re-minted (supersedes v1_2) after the frozen-set `ui/server.py` drag-route change; `verify_referee()` passes and the record carries its post-mint test outcome.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_4.json`: guard set re-minted (supersedes v1_3) after the live trial-ledger/promotion-lifecycle wiring in `trading_signals_sandbox.py`; the only changed guard file is that sandbox, no blind case imports it, and the blind suite re-passes at 9/9 with identical metrics. `verify_referee()` passes; v1/v1_1/v1_2/v1_3 remain on disk as historical records.

### Live wiring of the trial ledger and the promotion lifecycle
- `apedp_dyn/sandbox/trading_signals_sandbox.py` (a live UI-driven pipeline): the GA search now records EVERY fitness evaluation in a `CandidateTrialLedger` (in-sample Sharpe per evaluated chromosome, shared `search_id`), and the holdout Sharpe is deflated by that real multiplicity via `dsr_with_selection_adjustment` before it enters the evidence bundle (`metrics.dsr_selection_adjusted`, `trial_multiplicity` block with recorded/effective/hidden trials and expected-max-SR). The six-stage `PromotionTracker` now runs on the live candidate and honestly refuses at stage 1 for GA strategies (no Lean-verified law, no dimensional analysis), so `promotion_lifecycle.is_halted` is true and economic validity is never reached on unverified mathematical consistency. Measured example (20 pop x 5 generations, seed 7): 95 charged trials, holdout Sharpe -0.13, expected max SR 1.93 under multiplicity. New regression test pins the charge, the deflation ordering, and the stage-1 refusal.
- `apedp_dyn/epistemic/trial_ledger.py`: selection-adjusted multiplicity accounting (`effective_trial_count`, `dsr_with_selection_adjustment`) with fail-closed handling of malformed `selection_pool_size` values and empty ledgers.

### Recurrence guards & test-environment honesty
- `tests/test_audit_open_flaws_remediated.py` adds two static guards over runtime code: no back/forward-fill in proposer fit paths (`test_no_lookahead_fill_in_proposer_fit_paths`) and no concrete host home-directory literals outside container paths (`test_no_host_specific_home_paths_in_runtime_code`, AST-aware so docstrings are ignored).
- `tests/test_dyn_lean_universal_bridge.py`: the Docker-Mathlib test now skips with an explicit reason when the required image is not present locally (a 25 s execution budget cannot include a multi-GB pull). The local Lean 4 toolchain was restored (v4.35.0) after removing failed elan download artifacts; the suite is 10 passed / 1 environment skip.
- `tests/test_dyn_ui_studio.py`: the W2_2019 endpoint pin was synced to the audited sole-survivor contract from `tests/test_dyn_real_market_l2_sweep.py` (unique support plus holdout/whiteness/complexity guards) instead of the retired multi-hypothesis outcome.

### CI coverage and local gates (rating remediation)
- `.github/workflows/test.yml`: CI now triggers on EVERY branch (`branches: ['**']`) and on PRs targeting `stage`. Work happened on `stage` for an entire session with no CI coverage because that branch was absent from the trigger list, which is how a broken working tree survived unnoticed.
- `.pre-commit-config.yaml` (new): a local gate for UNCOMMITTED work — ruff on changed files, `verify_referee()`, the import smoke tier, and the documentation/audit guards. CI only ever checks out committed files, so this is the only thing that can catch a broken tree before it is committed.
- `scripts/check.sh` (new): `--quick` runs the fast tier; the default runs the complete dynamic regression battery.

### Lookahead remediation completed in the multi-asset panel
- `apedp_dyn/sandbox/multi_asset_signals_sandbox.py`: the 20-day rolling volatility is computed causally (`ffill`, was `bfill`, which back-filled the first 4 rows from future observations). `ffill` cannot fill leading NaNs, so the undefined warm-up rows are now EXCLUDED from every aligned array via `_drop_volatility_warmup` (returns, prices, features, cross-median) instead of poisoning the panel with NaN or being fabricated. Applied at all three panel constructions (30-asset two-tier analysis, epistemic adjudication, 90-asset policy-grammar audit).
- Honest consequence, re-pinned: with causal volatility the CAN-G gate now DETECTS the cross-asset ranking trap the fabricated volatility used to mask (naive pooled Sharpe -0.095 vs two-tier 1.116; within-class crypto 1.63 vs equity 0.20). The candidate still promotes at C3_STRUCTURAL because stratified allocation is what makes it admissible. `tests/test_dyn_trading_sandboxes_epistemic.py` asserts the trap detection, its reason string, and the stratified-vs-naive ordering.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_5.json`: guard set re-minted (supersedes v1_4) for that file; blind outcomes identical (no blind case imports it), `verify_referee()` passes.

### Discovery-path multiplicity wiring
- `apedp_dyn/sandbox/real_market_l2_sweep.py`: every candidate the L2 engine scores is now recorded in a `CandidateTrialLedger` (shared `search_id`, true evaluated count, per-candidate rank and in-sample loss), and `WindowDiscoveryRecord.trial_multiplicity` exposes the pool. Measured on W2_2019: 3195 evaluated, and the promoted `vol - 1/ma_dev**(3/2)` ranks 217th by in-sample loss — the promotion is holdout-adjudicated, not best-fit, which is exactly the selection bias the disclosure exposes. `CandidateTrialLedger` gained an explicitly named `selection_statistic` field (Sharpe for strategy searches where DSR applies; in-sample loss for equation searches where it does not) and tolerant loading of older ledger files.
- `apedp_dyn/epistemic/pde_discovery.py`: `discover_pde_nonuniform` was verified to run discovery on the develop partition only and to score the frozen operator on holdout rows (the F841 findings that prompted the review were stale). Locked by `tests/test_dyn_coupled_and_2d_pde.py::test_nonuniform_discovery_never_fits_holdout_rows`, which plants a different decay rate inside the holdout and asserts that corrupting the holdout rows cannot change the discovered formula or coefficients while the holdout R^2 does move.
- `apedp_dyn/ode/referee.py` pin reconciliation: a TERM referencing a state absent from the holdout raises (internally inconsistent result); a discovered STATE SET that differs from the holdout is an epistemic refusal (INSUFFICIENT_EVIDENCE). Both paths are now pinned separately in `tests/test_dyn_pde_ode_packages.py`.
- Latent bug fixed: `apedp_dyn/robotics/multi_agent_transport.py` referenced an undefined `_TWO_PI` in the support-ring branch, reachable only with >= 3 agents, so every existing test passed while that branch would have raised NameError. Constant defined, `np.array` default hoisted, and a 3-agent regression test added.

### Open verification item (flagged, not resolved)
- The JetFlow-Video matched-comparison numbers recorded in DOCUMENTATION.md (jet_tt_2step 10.74 dB PSNR / 0.514 SSIM / 0.275 temporal smoothness vs diffusion baselines) were produced while the incorrect 4-probe Jet2 acceleration was in the working tree. The reverted (fused-probe) formulation is the mathematically correct one, so those benchmark artifacts should be re-generated before being cited; training was NOT re-run in this session and the numbers are therefore flagged as unverified against the corrected stepper.

### Deep forensic audit remediation (L2 / proposers / Rust kernels / UI server)
- `rust/apedp-kernel/src/video.rs`: `ssim_frame` with `win == 0` passed the unsigned size guard (`h < 0` is never true) and then incremented the window loops by zero forever — a 100%-CPU hang reachable from Python. Now rejected explicitly. `compute_optical_flow_divergence` clamped the neighbour index at borders, so every border derivative spanned one cell but was divided by `2*dx` (halved); borders now use one-sided differences. New tests pin both.
- `rust/apedp-kernel/src/flow_head.rs`: `loss_and_v_grad` on a degenerate plane (H<2 or W<2) reached `adv_loss /= n_all` with `n_all == 0.0` and silently returned NaN; it now returns an explicit error.
- `rust/apedp-kernel/src/rime/high_dim_pde.rs`: the viscosity/diffusion coefficients were validated for finiteness only; a negative `nu` is the backward heat equation (unbounded high-frequency growth) and is now refused.
- `apedp_dyn/discovery_l2/ast_generator.py`: binary-operator pair enumeration was asymmetric (n1 from the new frontier only), so `n_old/n_new` was unreachable and the commutative "the reverse pair will create it" guard dropped BOTH orderings; self-products were excluded outright, making squares impossible without `pow`. Pairs are now enumerated symmetrically with both orderings for the non-commutative operators. Subtraction orderings remain deduplicated **by design**: the basis fingerprint orients sign, so `theta*f` and `-theta*(-f)` are one basis function under a free coefficient. `collect_variable_names` now recurses into `RollingOpNode` (was `frozenset()`, which misrouted univariate rolling terms into the multivariate interaction quota). 5 new tests.
- `apedp_dyn/discovery_l2/l2_engine.py` and `multiterm_engine.py`: Gate A no longer evaluates the holdout partition (AGENTS.md §1.2 violation — the proposer search was discarding candidates on unseen data). The multi-term basis now evaluates terms on the full causal series (rolling state carries across the split instead of re-warming inside the holdout), tolerates warm-up NaN instead of rejecting every rolling term outright, and fits on the finite overlap with the excluded row counts disclosed in `gate4_status`. 2 new tests (rolling term reaches the pool; Gate A never touches holdout).
- `apedp_dyn/proposers/distributed_lag_interaction_proposer.py`: ElasticNetCV now uses `TimeSeriesSplit` (plain KFold trained each fold on later rows, leaking the future into hyperparameter selection); the unknowable lag warm-up rows are excluded from the fit instead of being fabricated as zeros; the scorer carries an explicit state buffer so streaming chunks shorter than the lag no longer lose 100% of the lag signal (chunked scoring now equals single-pass scoring to 1e-12). `declared_scale_invariant` corrected to False (the model has an intercept and raw-scale weights).
- `apedp_dyn/proposers/regime_switching_proposer.py`: `declared_scale_invariant` corrected to False — the regime split branches on a threshold estimated on the training scale, so rescaling moves observations across it.
- `apedp_dyn/proposers/rime_ad_proposer.py`: the differential-Padé scorer is pole-aware (a vanishing denominator returned 0.0 and poisoned the whole trajectory; it now yields NaN and ends the integration, scoring the unreachable suffix NaN) and bounded (the step count was unbounded — up to ~200k Python iterations per unit distance; capped at 20k with the suffix marked unreachable).
- `apedp_dyn/ui/server.py`: `/api/discover/physics` term reconstruction resolves state series through an explicit per-system dictionary (a term naming an undefined state raised NameError -> HTTP 500) and matches explicit cubic forms instead of the substring test `"3" in t_name`; `/api/benchmarks/trading-np93` no longer overwrites `false_promotion_rate`/`passed_contract` with hardcoded 0.0/True on every response — it serves the frozen measurement and derives the verdict from the recorded counts. 3 new tests (measured verdict, all systems 200, upload wired).
- Audit items NOT reproducible in this tree, verified rather than assumed: `/api/sandbox/upload` already routes to the `custom_upload` branch (200 with a real adjudication), and the audit's `spatial_divergence_2d` / `rime_high_dim_pde_step` symbols do not exist — the corresponding real symbols (`compute_optical_flow_divergence`, `eval_jet2_neural_pde_operator`) were audited instead and the defects they do carry were fixed.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_6.json`: guard set re-minted (supersedes v1_5) for `ui/server.py`; blind outcomes identical (no blind case imports it), `verify_referee()` passes. Rust kernel rebuilt (184/184 lib tests).

### Data-integrity: vendored lynx fixture
- `apedp_dyn/sandbox/lynx_hare.py` no longer reads `/tmp/lynx.csv` (an untracked, host-dependent fixture whose contents changed mid-session and made the lynx-cycle sandbox unrunnable). The canonical MacLulich series (R `datasets::lynx`, 1821-1934, 114 annual counts) is vendored at `data/market/lynx_maclulich.csv` with provenance recorded in `data/market/PROVENANCE.json`; `load_lynx()` defaults to it and still accepts an explicit path. `tests/test_dyn_ui_sandbox.py::test_lynx_real_data` now runs against tracked real data (18/18 in that suite).

### Transport-stability reconciliation (audio) and stepper revert
- `apedp_dyn/audio/spectrogram_transport.py::predict_next_frame` integrates the frequency-axis advection `S_t = -c S_f` with a CFL-enforced upwind scheme instead of routing it through the generic attribute-transport stepper: centered Euler/Jet2 differencing of pure advection is unconditionally unstable (von Neumann `|G|^2 = 1 + nu^2 sin^2(k df) > 1`) and accumulates negative spectral magnitudes. `order` is retained for interface compatibility, so `tests/test_dyn_audio_field_transport.py::test_jet2_order_beats_euler_on_accelerating_chirp` was vacuous (both orders identical by construction) and is re-pinned as `test_accelerated_chirp_transport_is_stable_and_order_invariant`: identical results across orders, mean R^2 0.9885, min R^2 0.977, non-negative predictions. The DOCUMENTATION.md row no longer claims a per-frame Jet2 advantage for this path; the stepper-level second-order advantage is pinned in `test_accelerated_ridge_transport_order2_error` (relative error 1.87e-3 -> 8.7e-5).
- `apedp_dyn/video/jet_flow_stepper.py`: reverted an uncommitted 4-probe "fix" of the Jet2 acceleration. It split the material derivative into separate temporal/spatial probes but omitted the `|v|` scaling of the convective term (the kernel composes `(v_t+ - v_t-)/dt_span + (v_sp+ - v_sp-)/two_h`), so the acceleration was wrong: robotics trajectory accuracy regressed (5/6 failing, full-circle error 0.023 vs the 5e-3 contract) and the audio order-2 error rose to 6.9e-4. The HEAD formulation passes the fused characteristic probe `v(z + h v, t + h)` as the temporal term, which already equals the material derivative `dv/dt = dv_t/dt + (v.grad)v`, and keeps `compute_acceleration` and the kernel path in parity. Robotics back to 6/6, molecular 6/6, audio order-2 error 8.7e-5.

### Numeric-exactness repair: periodic-safe variance
- `apedp_dyn/market/lob_transport.py::snapshot_variance`: the wrap-safe rewrite used the mean-resultant approximation `2*(L/2pi)^2*(1-|R|)`, which is only first-order accurate for a finite-width packet and inflated the diffusion constant recovered from `d var_x/dt` by ~5% (the 1e-6 recovery contract in `tests/test_dyn_lob_transport.py::test_statistical_observables_match_parameters` failed). Replaced with the exact mass-weighted variance of the wrapped angular deviation from the mean angle, which is wrap-safe, exact for any packet width, and NaN for zero-mass fields. Suite back to 5/5.

### Quadratic-drag discovery (from the same working tree)
- `discovery_l2/ast_generator.py` bounded canonicalization (`collect_variable_names`), `discovery_l2/multiterm_engine.py` structure-aware quota screen, and `ui/server.py` route defaults (3, 2, 4) + `max_basis_terms=40`; generated powers are now canonicalized (`x*x -> x^2`) instead of cancelled by the quota screen. End-to-end drag sweep recovers `-0.5*v*|v|`-family structure where the pre-fix screen discarded it; 7 new regression tests.

## [Unreleased] — Packaging & platform hardening: standalone wheels, Databricks, kernel-less imports (2026-10-19)

### Standalone-install contract (wheels)
- The six domain wrappers (`packages/apedp-{audio,continuum,distributed,molecular,quantum,robotics}`) now declare `"apedp>=1.0.0"` in both `pyproject.toml` and `setup.py` (12 files). They re-export from `apedp_dyn` at import time, so installing a wrapper alone previously succeeded silently and then failed with `ModuleNotFoundError` at first use. Verified: the rebuilt `apedp-quantum` wheel metadata now carries `Requires-Dist: apedp>=1.0.0`, and a wrapper-only install in a fresh venv fails fast with `ERROR: No matching distribution found for apedp>=1.0.0` instead of installing broken.
- Root wheel (`apedp`) now excludes `apedp_kernel*` from `[tool.setuptools.packages.find]`. The dev tree's `apedp_kernel/` symlink (into the venv's compiled site-packages) was swept into the pure-Python wheel as a wrapper-only package, and a stale `build/lib/apedp_kernel` staging directory from an earlier build leaked it into every subsequent wheel even after the finder was fixed — producing `ModuleNotFoundError: No module named 'apedp_kernel.apedp_kernel'` on import in kernel-less environments. With the exclude and a cleaned `build/`, the rebuilt wheel contains no `apedp_kernel/`; the extension continues to ship exclusively from its maturin build (`rust/apedp-kernel`).
- New `python-packages` CI job in `.github/workflows/wheels.yml`: builds all seven packages from `packages/*` and asserts each wrapper's `METADATA` contains `Requires-Dist: apedp` and that jetflow-video's does not (it is independently guarded as fully standalone). `packages/**` added to the PR trigger paths.
- Regression tests in `tests/test_dyn_all_subpackages.py`: `test_domain_wrappers_declare_apedp_core_dependency` (both metadata files, per wrapper) and `test_jetflow_video_stays_self_contained` (no `apedp` imports anywhere in its source).

### Databricks as a first-class runtime
- `apedp_dyn/notebook_adapter.py`: `PlatformType.DATABRICKS` detected via `DATABRICKS_RUNTIME_VERSION` / `DATABRICKS_CLUSTER_ID` (before the CI env markers, so Databricks job compute is not misclassified); `is_databricks()` exported through `apedp_dyn/env.py`. Storage routing: temp files under `/local_disk0/apedp_temp` (node-local NVMe-class scratch), persistent storage at `/dbfs/FileStore/apedp_storage` when a DBFS mount is present, otherwise the driver-local directory with a high-visibility EPISTEMIC RISK warning (AGENTS.md §1.2: driver-local storage dies with the cluster and erases burned holdouts). Worker governor: one worker on GPU clusters, `min(cpu_count, 16)` on CPU clusters. Four new tests (detection by both env markers, DBFS-persistent and ephemeral-fallback paths with warning); `tests/test_dyn_notebook_adapter.py` is now 30/30.

### Kernel-less imports become honest
- New `apedp_dyn/kernel_shim.py`: a `missing_kernel` sentinel exposing the same contract as the absent extension. The five modules with bare module-level `import apedp_kernel` (`ode/dispatch.py`, `epistemic/coupled_discovery.py`, `epistemic/pde_discovery.py`, `epistemic/high_dim_pde.py`, `proposers/rime_ad_proposer.py` — the last two in the eager import chain via `apedp_dyn/ode/__init__.py` and `apedp_dyn/pde/__init__.py`) now guard the import and fall back to the sentinel, instead of exploding `import apedp_dyn` with a bare `ModuleNotFoundError` on any machine without the native extension (masked in development, where the kernel is always present, and contradicting the documented pure-Python fallback story).
- Behavior: `import apedp_dyn` always succeeds; probing stays attribute-safe (`hasattr(ak, "__file__")` is False, `repr()` works); calls into kernel-only engines raise an actionable `RuntimeError` naming the prebuilt-wheel / `maturin develop --release` install step. There are no pure-Python implementations for those engines, so refusing the call — not guessing a result — is the honest failure mode.
- Verification (fresh venv, kernel absent): `import apedp, apedp_dyn, apedp_quantum` succeeds; the DMRG reference engine runs (`dmrg_ground_state` n_sites=4 energy -1.616025 vs exact diagonalization); `discover_ode` raises the documented RuntimeError. Regression tests added to `tests/test_dyn_all_subpackages.py` (sentinel probe-safety + subprocess import-chain test that blocks `apedp_kernel` via `sys.modules` and asserts the actionable call-time error).
- `docs/DOCUMENTATION.md` module registry: Databricks behavior added to the notebook-adapter row (30/30) and a new row indexes `apedp_dyn/kernel_shim.py` with this verification evidence.

### Test-suite reporting (machine-verified by tests/test_dyn_reporting_integrity.py)

- The Python RIME/repair/referee/STEM suite total is **212 tests across 43 files**, counted by importing each file in tests/test_dyn_reporting_integrity.py and assert-verified in CI. Raised 211 -> 212 after `test_dyn_trading_universe_sweep.py` gained a test function post-mint (constant bumped by re-running the assertion test, per its own rule). Earlier prose quoted a 74-test subset of 11 files; that figure is a scoped subset, not the suite total, and is superseded by the machine-verified number.
- Named exclusions from the wider `tests/` directory (not silently omitted):
  - `tests/test_all_system_equations_lean_proofs.py::test_spot_check_lean_kernel_compilation` — environmental: the Lean kernel compile exceeds the test's 15s budget (pre-existing).
  - `tests/test_dyn_lean_universal_bridge.py` — same environmental Lean cause.
  - `outputs/nano_ds10_freeze.json` — PIN RECONCILED: previously reported as drifted at d37d0f1c...; pin in `apedp_dyn/epistemic/frozen_spec.py:40` was reconciled to `b809ddcba1eefff2f7d3f21cc21247467c60dcda2f482ba0df1b528d6a28376f`, verified by `test_all_frozen_records_verified_against_pinned_hashes`.

### Documented Residual Design Boundaries & Preregistration Rationale
- **Standalone Verification Harnesses (`tests/test_dyn_rime_scope_refusal_mv_v2.py`)**: Collects 10 dynamic verification cases for multivariate scope refusal v2. Intentionally maintained as an independent dynamic test suite rather than appended to the 43-file / 212-test frozen RIME baseline to avoid mutating the historical suite accounting constant.
- **Stage 6 Parameter Defaults (`apedp_dyn/epistemic/promotion_lifecycle.py:177-178`)**: Default literals (`prospective_hurdle = 0.70`, `max_dd_ceiling = 0.25`) in `verify_stage_6()` reflect the standard preregistered real-market promotion barrier; preserved to maintain contract continuity across callers.
- **Trial Multiplicity Minimum Variance Regularizer (`apedp_dyn/epistemic/trial_ledger.py:122,132`)**: The `min_var = 1e-4` floor in `compute_expected_max_sharpe()` ensures that degenerately zero-variance trials cannot evade Bailey & López de Prado multiple-testing haircut penalties.
- **Joint Literature Span Projection (`apedp_dyn/epistemic/novelty_adjudication.py:194-203`)**: Unadjusted sample $R^2$ is intentionally retained for the linear span projection of literature baselines. This provides a strictly conservative bound against premature structural novelty attributions by ensuring the baseline space spans maximal variance.
- **NCM Naming & Scope Boundary (`apedp_dyn/proposers/ncm_structural_proposer.py`)**: "Next-Concept Modeling" is a naming inspiration only. The source work (Kumbong et al., arXiv:2610.14162, cs.CL) is an NLP text-generation method (multiscale residual VQ-VAE over text tokens; a transformer emitting one code level at a time). No part of that architecture — codebooks, hierarchical quantization, learned latents, transformer, parallel generation — is implemented here, and its 10x-31x speedups are against diffusion language models on text benchmarks, applicable to no RIME/kernel search path. This module is a classical two-tier archetype-fitting baseline (Ridge L2 + hard coefficient thresholding) verified by parameter-recovery tests. Earlier claims of a 10x-30x search speedup, L1/L2 penalized pruning, qualitative-invariant evaluation, and `production` status were fabrications; status remains `experimental` per AGENTS.md §3, and the docstring now states the actual architecture and scope.

## [Unreleased] — Audit hardening: fail-closed referee defaults, neutral floor rules, status-tag enforcement (2026-10-18)

### Fabrication fixes
- `apedp_dyn/epistemic/agnostic_adapter.py`: removed the fabricated metrics fallback (returned `in_sample_r2=0.95, holdout_rmse=0.02` when a candidate had no callable and no recorded metrics). Unrecorded declared invariants with neither evaluator nor callable now evaluate as unverified (fail-closed) instead of defaulting to pass, and unhandled invariant kinds are unverified instead of passing. `no_model_callable`/`execution_error` now report `in_sample_r2=-999, holdout_rmse=999` per the module's failure convention.
- `apedp_dyn/sandbox/agnostic_domain_sandbox.py`: the `volume_density_conservation` invariant now carries a real computed evaluator (`volume_eval_valid`, spatial flatness of predicted density) instead of relying on the fabricated default pass; Arm C VRR 4/4 and FPR 0/4 are preserved with the real evaluator. Regression tests added in `tests/test_dyn_agnostic_adapter.py` (8 tests).

### Boundary fixes
- `apedp_dyn/grammar/floor_rules.py` (new): `fit_floor_rule` and `floor_rule_candidate` moved verbatim out of `apedp_dyn/epistemic/prior_guided_search.py`; both the prior-guided search adapter (which keeps `_fit_floor_rule`/`_floor_rule_candidate` aliases for compatibility) and `apedp_dyn/proposers/rime_ad_proposer.py` now import them from the neutral grammar layer, so a proposer no longer reaches into referee-package private helpers (AGENTS.md §1.1). The remaining epistemic imports in the RIME-AD proposer (`profile_dataset`, `staircase_prior`) are public dev-only diagnostics whose modules document the §1.1 boundary; they are read-only, never touch holdout rows, and never promote.
- Lean bridge status demoted `production` -> `experimental` in the module registry with an explicit `Status:` line (`apedp_dyn/epistemic/lean_universal_bridge.py`) and the environment gate recorded in its capability row: end-to-end verification requires an external Lean 4 toolchain. The JetFlow-vs-diffusion matched-benchmark registry row was corrected `audit-only` -> `experimental` to match the harness's own declared status.

### Enforcement
- `tests/test_documentation_and_spec_integrity.py::test_registry_status_tags_match_code_status_labels` (new): machine-checks that every registry status tag is one of the four canonical AGENTS.md §3 tags and matches the linked module's docstring `Status:` line wherever present, so documentation can no longer certify a stability level the code does not declare. Registry preamble in `docs/DOCUMENTATION.md` documents the convention.

### Test-hygiene fixes (flaw sweep follow-through)
- `tests/test_jetflow_video_sampler_direction.py`: the two stubs in `_tiny_model`/`_Probe.attach` were stale against the current model call contract (`compute_flow_guided_velocity` now passes `plucker_rays=`, `steer_velocity_field` now receives `depth_engine=`). All five sampler-direction contract tests were crashing on `TypeError` rather than testing anything; with the stubs updated they execute and pass, so the t:1->0 direction fix is now actually verified (starts at t=1 from fresh noise, |z| grows by e under v=-z) instead of merely asserted.
- `benchmarks/run_jetflow_cpu_export_benchmark.py`: `torch.onnx.export(..., dynamo=False)` is now passed only when the installed torch accepts the kwarg (torch >= 2.6; the env here is 2.3.1, whose exporter always uses the legacy path), and the scratch dir uses `tempfile.gettempdir()` instead of a hardcoded `/tmp`. Unblocks the three ONNX export/quantization tests (`tests/test_dyn_jetflow_cpu_export.py`, previously 1F + 2E).
- Weak "no mock losses" assertions replaced with recomputation contracts: `tests/test_dyn_l2_open_discovery.py`, `tests/test_dyn_materials_l1.py`, `tests/test_dyn_quantum_sandbox.py`, `tests/test_dyn_jet_flow_sota_path.py` now recompute recorded RMSE/losses/drifts from recorded residual series and require exact agreement (1e-9), require finite strictly-positive losses instead of `>= 0.0`, and require the Helmholtz-Hodge projection to actually reduce mean divergence versus the raw field it was applied to.

### Referee-probe fixes and red-test reconciliation (2026-10-19)
- `_invariance_verdict` (`apedp_dyn/epistemic/rime_repair_engine.py`): dev slices are now contiguous and order-preserving. The previous `RandomState(0).permutation` shuffled rows before slicing, which contradicted the head/tail temporal split the pair evaluation itself performs and destroyed the serial structure that lag-family findings and repairs are about; shuffled slices fabricated mixed-regime subsamples. No test or caller depended on the shuffle.
- `_slice_pair_key` now honors its own docstring contract ("with no gates"): slice-level `generate_repairs` runs with `r5_emit_unidentified=True`. The R5 identifiability gate is a pooled-fit stability criterion, not a reproducibility criterion; with it active, every 96-row slice was judged ill-conditioned a priori and the invariance probe collapsed to NONE with zero information (verified: rel-stderr b ~ 0.19-0.21 per slice vs 0.029 pooled). Identifiability status still travels with each slice candidate.
- Red-test reconciliation for the three tests that predate the cross-slice invariance gate (`test_option_2_epistemic_repair_engine_end_to_end_repair`, `test_options_123_under_2pct_and_10pct_noise`, `test_selective_risk_qec_world`): their configs now run with `invariance_slices=0` (explicit, commented). Under the fixed probe both frames genuinely fail cross-slice structural reproducibility (slices re-propose structurally different originals — state vs lag forms; best 2/5 on the QEC world), so the fail-closed AMBIGUOUS verdict is the correct referee behavior, not a repair-engine regression. The repair mechanics the tests exist to pin are unchanged (verified: with the gate off the same runs reach IMPROVED with sealed R2 0.987-0.9999 and the expected R5/R2 winners). The flagship `test_option_2` additionally asserts the gated refusal (`AMBIGUOUS` + "cross-slice invariance" note), and the gate's own behavior remains covered by `tests/test_dyn_rime_repair_v012_sensitivity.py` (green, including its explicit AMBIGUOUS-when-not-reproducible accounting).
- `tests/test_dyn_real_market_l2_sweep.py::test_single_window_sp500_calm_sweep` re-pinned after audit, anti-spurious guards preserved (out_of_grammar, fidelity margin <= 1, whiteness, no boundary artifact, `ma_dev` in expression): the grammar now yields a pool of ONE adequate candidate for W2_2019 (`vol - 1/ma_dev**(3/2)`, complexity 4), so the hardened sole-survivor contract promotes it with decisive attribution ("no structurally distinct rival survived scoring"). The previous refusal pin relied on a reorder twin (`ma_dev**(-2)+sqrt(ma_dev)`) that is no longer generated. The apparent ~130x train/holdout error gap is a units artifact: `train_loss` is MSE (3.58e-5 -> RMSE ~ 6.0e-3), `holdout_error` is RMSE (4.65e-3); in RMSE terms the survivor generalizes with the same profile as the historically accepted candidate.

## [Unreleased] — Structural-invariant probe: one check per invariant (2026-10-18)

`_StructuralProbesMixin._probe_structural_invariants` was a 382-line method whose
fifteen checks shared one evidence context by closing over fifteen locals. It is now
an orchestrator plus one method per invariant, with the envelope measured once
(`_EnvelopeScan`) and handed to each check (`_StructuralContext`).

### What changed
- `_EnvelopeScan.from_samples()` computes `lowest`, `highest`, `widest`,
  `first_unevaluated` and `boundary_divergence` once. Fifteen checks used to re-derive
  them from the same lists; a check can no longer decide against a different sample
  than the check before it.
- The five magnitude ceilings (`current_density_ceiling`, `velocity_saturation_ceiling`,
  `yield_strength_ceiling`, `stokes_drag_ceiling`, `saturation_ceiling`) now share
  `_envelope_ceiling()`, so a new ceiling cannot quietly measure a different point
  than its siblings.
- Checks run in check-number order and each returns `None` (invariant not declared --
  nothing counted), `[]` (declared and satisfied) or its counterexamples. Counting of
  `invariants_checked` and counterexample ordering are therefore unchanged, including
  the two checks (dimensional, heat capacity) that can record a probe error instead of
  counting.
- Envelope preparation (constant-model handling plus the `lambdify` sweep) moved into
  `_prepare_structural_envelope()`, which returns `None` for a constant with no finite
  real value so the caller must report it. `_probe_structural_invariants` is 80 lines;
  the longest single check is 43. The file
  grows 382 -> 604 lines because each check now carries its own docstring recording
  what was removed from it (variable-name matching, the printed-text magic-constant
  search).
- `_EnvelopeScan` and `_StructuralContext` are re-exported from the facade, matching
  the existing policy that every name defined in a package module is reachable from
  the package -- the policy `tests/test_package_facades.py` enforces.

### Verification
- **Exact old-vs-new equivalence.** The pre-split method (from commit `919d4c8`) and
  the new one were run over 31 cases -- every invariant alone, all fifteen declared at
  once, constant models, `1/x` (non-real at 0), two-symbol envelopes, and three
  dimensional paths including an undefined unit -- comparing the serialised
  counterexamples and the full metrics dict: **0 mismatches**, 40 counterexamples, all
  15 probes exercised. The harness also verified it loaded the pre-split source (no
  `_EnvelopeScan`) rather than re-testing the new code.
- **Behavioural battery unchanged:** `/tmp/falsifier_fingerprint.py` still digests
  **`8c4c4f17a3597525`**.
- `tests/test_structural_invariant_coverage.py`: 38 tests still pass with their
  expectations unchanged (not re-pinned).
- The 11 test files that import the split packages: **186 passed**, exit 0.
  `ruff check apedp/falsification` clean, exit 0.

## [Unreleased] — Structural-invariant coverage: a constant model no longer skips the axis (2026-10-18)

Found by a new per-invariant coverage test, not by a refactor; fixed in the probe
the earlier split created (`apedp/falsification/probes_structural.py`).

### The finding
`_probe_structural_invariants` returned early when the expression had no free
symbols, so a *constant* law was declared with invariants, evaluated by no check,
and reported as `invariants_declared = 1, invariants_checked = 0` with an **empty**
`probe_errors` list. Reproduction: `falsify("-1.0", {"x": (1, 2)}, ["positivity"])`
returned no counterexample and no error -- a declared positivity invariant that the
report neither checked nor mentioned. The candidate was still refused, but by the
numerical axis (`INCONCLUSIVE_PROBE_ERROR`), i.e. for a reason unrelated to the
invariant the caller declared. The module's own stated contract ("a probe that
could not decide is an error on every axis, not a pass") was not met on this path.

### The fix
A model with no free symbols is still a model: the probe now evaluates the constant
once and runs the same fifteen checks over it. A constant with no finite real value
(e.g. `1e400`) is reported as an unresolved `EnvelopeSweep`/`NoRealValues` probe with
`invariants_checked = 0`, matching how a model that returns no real value across its
envelope is already handled.

| Input | Before | After |
|---|---|---|
| `-1.0`, `["positivity"]` | `INCONCLUSIVE_PROBE_ERROR`, checked 0, no error, no counterexample | `REFUSED_CANNOT_REPAIR`, checked 1, `PositivityViolationProbe` |
| `2.0 + 0.0*x`, `["positivity"]` | checked 0, no error | checked 1, no counterexample (invariant satisfied) |
| `1e400`, `["positivity"]` | checked 0, no error | checked 0, `NoRealValues` unresolved error |

### Scope and verification
- **Confined to constant models.** The 12-formula, five-axis behavioural battery
  (`/tmp/falsifier_fingerprint.py`) has digest **`8c4c4f17a3597525`** both before the
  refactor and after this fix, with the payloads comparing equal: no non-constant
  model changed a finding. This is why the fix did not disturb the refactor's
  equivalence evidence.
- **New test file** `tests/test_structural_invariant_coverage.py` (38 tests): every
  declared structural invariant is paired with a model that must violate it and one
  that must not, asserting on the probe that fires; plus symbol-renaming, undeclared-
  invariant silence, and the dimensional probe. Expectations come from the
  invariant's meaning, not from a recorded run.
- **Non-vacuous:** restoring the pre-fix early return makes exactly the four
  constant-model tests fail (`4 failed, 34 passed`); restoring the fixed file
  byte-identically returns `38 passed`.
- 14 affected test files + facade + coverage suites: **495 passed**, with only the
  same 2 pre-existing `test_dimensional_invariance.py` failures (P07, E16).
  `ruff check apedp` clean, exit 0.

## [Unreleased] — Split the three god-files into focused packages (2026-10-18)

`apedp/falsification.py` (2322 lines), `apedp/gated_assurance.py` (2116) and
`apedp/benchmarks.py` (1414) are now packages. Behaviour-preserving refactor: no
threshold, probe, verdict, weight or control-flow change — only code movement
plus re-exports.

### New layout
- `apedp/falsification/`: `constants`, `taxonomy`, `counterexamples`, `constraints`, `report`, `falsifier`, `blind`.
- `apedp/gated_assurance/`: `checks`, `contracts`, `mechanistic`, `operational`, `engine`, `benchmark_contracts`, `feasibility`.
- `apedp/benchmarks/`: `types`, `results`, `pathological`, `harness`.

Each `__init__.py` is a facade re-exporting **every** name the old module exposed,
including the names it merely imported (e.g. `apedp.benchmarks.FalsificationReport`,
`apedp.falsification.LimitTimeout`), so all existing import paths are unchanged.
Consequences recorded honestly:
- `__module__` of moved classes now names the implementation module (e.g.
  `apedp.falsification.counterexamples.Counterexample`). No `pickle` use of these
  types exists in the repository, and nothing pins `__module__`.
- Rebinding a *moved* symbol on the facade no longer changes what the
  implementation calls: `tests/test_flaw_fixes.py::test_boundary_axis_surfaces_a_timed_out_limit`
  patched `falsification.limit_with_timeout` and now patches
  `apedp.falsification.falsifier.limit_with_timeout` instead (the module the probe
  actually reads). The assertion is unchanged. Patching `falsification.sp` still
  works because that is an attribute on the `sympy` module object itself.
- `tests/test_evidence_tiers.py` scanned `apedp/falsification.py` for
  `probe_name="..."` literals; it now globs `apedp/falsification/*.py`. Coverage
  and staleness assertions are unchanged, including `len(declared) >= 20`.

### Verification
- **Source is lossless:** the generator compared every non-blank, non-header
  source line of each original file against the concatenation of its emitted
  modules, in order — exact match (decorators included; comment blocks travel with
  the definition below them).
- **API is identical:** a before/after snapshot of each module's `dir()` — kind,
  signature, base classes, dataclass fields and values, class members — reports
  **0 missing, 0 changed** names for all three. The only additions are the
  submodule attributes a package necessarily exposes (`falsification.taxonomy`, …).
- `ruff check` clean on the three packages (I001/E402/F401/F811 resolved by
  pruning headers to used imports, sorting, and a file-level `# ruff: noqa: F401`
  on each facade).
- The 14 test files that import these modules: **451 passed**, with the same 2
  pre-existing `test_dimensional_invariance.py` failures (P07, E16) present before
  and after the split. No new failure attributable to the refactor.

### Follow-up — the case tables moved out of the benchmark class

`apedp/benchmarks/pathological.py` (980 lines) held the class *and* both case
tables inline. The tables are now `apedp/benchmarks/cases.py` (976 lines, data
only: `_get_25_case_suite`, `_get_100_case_suite`), and
`PathologicalEquationBenchmark` keeps only selection behaviour and delegates.
The class API is unchanged — `get_benchmark_cases`, `_get_25_case_suite` and
`get_100_case_suite` are still classmethods with the same signatures, so a
subclass can still override either suite.

- **Byte-identical tables:** the extracted text was compared character-by-character
  against `git show HEAD:apedp/benchmarks.py` (the pre-split monolith, where both
  tables stated their method-body indentation): `identical=True` for both, 308 and
  646 lines.
- **Identical at runtime:** a SHA-256 fingerprint over every dataclass field of
  every case (factories normalised) is unchanged for all three entry points:
  25-case `026febdf62324e5e`, 100-case `8c7a5c2cf322fd19`, `get_benchmark_cases()`
  (= the 25-case suite) `026febdf62324e5e`.
- **API:** the before/after snapshot reports **0 missing, 0 changed**; the only
  new facade names are `cases` and its two private functions.
- `ruff check apedp/benchmarks` clean; `test_package_facades`,
  `test_falsification_and_benchmarks`, `test_benchmark_integrity`,
  `test_adversarial_science`, `test_evidence_tiers`, `test_phase2`,
  `test_untested_module_smoke` **91 passed, exit 0**.

### Follow-up — the falsifier's five probe axes moved into mixins

`apedp/falsification/falsifier.py` was a 1154-line single class. The probe
implementations now live in focused modules:

| Module | Contents | Lines |
|---|---|---|
| `probe_support.py` | `_ProbeSupportMixin`: `_record_probe_error`, `_sample_envelope` | 105 |
| `probes_mathematical.py` | `_MathematicalProbesMixin` | 133 |
| `probes_boundary.py` | `_BoundaryProbesMixin` | 146 |
| `probes_numerical.py` | `_NumericalProbesMixin` | 194 |
| `probes_structural.py` | `_StructuralProbesMixin` (orchestrator + one check per invariant; 604 lines after the follow-ups below) | 604 |
| `probes_empirical.py` | `_EmpiricalProbesMixin` | 77 |
| `falsifier.py` | `IndependentAdversarialFalsifier.__init__` + `falsify()` | 200 |

Method bodies, signatures and docstrings are unchanged; the class is now
`IndependentAdversarialFalsifier(_ProbeSupportMixin, _MathematicalProbesMixin,
_BoundaryProbesMixin, _NumericalProbesMixin, _StructuralProbesMixin,
_EmpiricalProbesMixin)`.

**Consequence recorded honestly: the class's bases and MRO change.** `bases` went
from `(object,)` to the six mixins, so three test subclasses in
`tests/test_benchmark_integrity.py` (`_BlindFalsifier`, `_AlwaysRefusingFalsifier`,
`_RecordingFalsifier`) now inherit the mixins transitively. Everything callers use
is unaffected: every former method is still reachable on the class, the class's own
constructor signature is byte-identical, and the public facade still exports
`IndependentAdversarialFalsifier` from `apedp.falsification`.

**Verification**
- **Behavioural digest unchanged.** `/tmp/falsifier_fingerprint.py` runs a 12-formula
  battery spanning all five axes and hashes verdicts, `survived`, every
  counterexample's `(probe_name, failure_class)`, axes tested, refusal reason,
  `cannot_repair`, evidence tier and metric keys: digest **8c4c4f17a3597525** before
  and after, with the two full payloads comparing equal (`fingerprint payload
  identical: True`). This is a stronger statement than a green test run: no probe
  changed a finding.
- **API surface:** all previously own members are still reachable (`missing: []`),
  the class signature is unchanged, and the only snapshot difference is the intended
  `bases` change.
- `ruff check apedp` clean; all 169+ `apedp.*` modules import.
- The 14 test files that import these modules plus `test_package_facades`:
  **457 passed**, with only the same 2 pre-existing `test_dimensional_invariance.py`
  failures (P07, E16) seen before any of this work.
- `tests/test_flaw_fixes.py::test_boundary_axis_surfaces_a_timed_out_limit` patched
  `falsification.limit_with_timeout`; the boundary probe now reads it from
  `apedp.falsification.probes_boundary`, so the patch target moved there. Assertions
  unchanged. Same class of adaptation as the earlier `falsifier` patch: rebinding a
  moved symbol on a facade does not reach the implementation module.

Docs: `docs/DOCUMENTATION.md` §5.11 records the layout and the facade guarantee.

## [Unreleased] — Comprehensive Remediation of 7 Open Core System Items (2026-10-17)

> **Revert note (2026-10-18):** Items **1, 3, 8, 9, and 10** below were reverted to committed HEAD. Review found the holdout-priority selection ranked rival candidates by confirmation-holdout RMSE — holdout information leaking into promotion selection — and the ESEDS benchmark's promotion FPR (`fpr_promo`) measured 0.333 under it versus 0.0 at HEAD. `adjudicate.py`, `frozen_spec.py`, and `APEDPX_FROZEN_v1_2.json` are restored to HEAD (referee record `da604e6a…`; `DYNAMIC_EPISTEMIC_VERSION` `bootstrap-v3+solefid-v5`); the ESEDS suite passes 8/8 and the version pin in `tests/test_dyn_epistemic_bootstrap.py` is restored to `bootstrap-v3+solefid-v5`. The frozen-record re-pin (`e93e3307…`) and `bootstrap-v4` stamp described under items 8 and 10 are not in the tree; the pass counts quoted under item 10 describe the reverted state only. Items **2 and 4–7 remain in effect.**

### Fixed & Remediated
1. **`adjudicate()` candidate evaluation and holdout priority (`apedp_dyn/epistemic/adjudicate.py`)**:
   - `adjudicate()` no longer aborts on rank-1 gate-4 failure when other candidates successfully passed gate 4; evaluates and promotes the best holdout candidate among gate-4 survivors.
   - When a rival candidate defeats the emitted candidate on holdout ($\Delta < -\tau$), `adjudicate()` no longer falsely reports `multi_hypothesis` (which was caused by an unchecked $\Delta \le \tau$ comparison on negative $\Delta$); correctly evaluates holdout superiority.
2. **`_one_probe()` overfit probe separation (`apedp_dyn/epistemic/scope_refusal.py`)**:
   - Separates rather than declines an emitted candidate that beats an overfit probe on holdout. Bases `r_best` and `r_rival` on out-of-sample holdout RMSE rather than in-sample train loss.
3. **`bootstrap=True` gate 4 residual fallback (`apedp_dyn/epistemic/adjudicate.py`)**:
   - `adjudicate(bootstrap=True)` now falls back to `_residual_series(best)` and `_residual_series(second)` from `gate4_status` provenance when explicit arguments are omitted, running paired block bootstrap and preserving `EpistemicContract.FULL` rather than demoting to `INCOMPLETE`.
4. **`RimeAdProposer._make_dag_candidate` double scaling fix (`apedp_dyn/proposers/rime_ad_proposer.py`)**:
   - Removed double application of slope and intercept (which previously caused measured max error of 8.97).
5. **RIME implicit minimal relation discovery (`rust/apedp-kernel/src/rime/implicit.rs`)**:
   - Converted implicit relation search to a degree-incremental progression ($1 \dots \text{max\_degree}$). For circles with `max_degree=3`, correctly discovers and returns the minimal quadratic relation $x^2 + y^2 - 1 = 0$ instead of higher-order polynomial multiples $xy(x^2 + y^2 - 1) = 0$.
6. **Hawkes orderflow regular bar dynamics (`apedp_dyn/proposers/hawkes_orderflow_proposer.py`)**:
   - Updated `compute_score` to incorporate order flow activity (volume, trades, price volatility) so that regular timestamped bars produce dynamic, informative intensity signals instead of collapsing to constant zeros `[0.]`.
7. **Flow-head test comments & correlation assertion (`tests/test_dyn_flow_head_rust.py`)**:
   - Removed stale `0.25 / 0.47` chaos comments and added explicit Pearson correlation assertion (`loss_corr > 0.99`, measured 0.9985) verifying lockstep convergence between Rust and PyTorch.
8. **Frozen specification integrity**:
   - Re-computed and updated SHA-256 for `apedp_dyn/epistemic/adjudicate.py` in `APEDPX_FROZEN_v1_2.json` and `PINNED_FROZEN_REGISTRY` in `apedp_dyn/epistemic/frozen_spec.py`. Verified with 15/15 green guard checks.

### Follow-up — two residual adjudication defects found while re-verifying items 1–8 (`apedp_dyn/epistemic/adjudicate.py`)

Both were exposed by, not fixed by, the holdout-priority selection in item 1; both were reproduced before the change and re-verified after.

9. **An unverified candidate could "defeat" a validated winner.** The rival set was every other candidate with a finite holdout RMSE, so a candidate with no gate-4 record at all (holdout 0.02) made a gate-4 passer (holdout 0.50) report `INSUFFICIENT_EVIDENCE` — "defeated on holdout by rival" — for evidence the rival never established. Candidates with no gate-4 record are now excluded from the rival set whenever any candidate passed gate 4; when none did, the selection has already fallen back to plain holdout ranking, so provenance is not required there (no behaviour change on that path). The reproduced case now routes to the existing sole-survivor contract, which refuses the passer for failing the pre-registered absolute fidelity ceiling — the same conservative outcome reached for a stated reason.
    - **Scope decision recorded, not smuggled.** The broader rule (rivals restricted to gate-4-*validated* candidates) was implemented, measured, and then narrowed. Measured on the frozen DUFF/COUP/STAB sweep (36 cells): the rival is refuted in 34/36 cells, so under the broader rule nearly every cell falls to the sole-survivor fidelity ceiling — mean recovery 0.694 → **0.472** (false promotions 0 → 0). Rejecting a refuted rival is a different epistemic contract about what a rival is *for* (the resolution rule detects an indistinguishable rival, which by construction would have passed gate 4), and that trade-off belongs to the preregistration owner rather than to an unannounced side effect of a bug fix. The broader rule is therefore NOT applied and is left as a documented open question.
10. **Residual roles were attributed positionally, not to a candidate.** `best` is now the holdout winner, but a bare residual array carries no identity: `dyn_sweep.py` orders its series by train loss (its comment claimed "Rank mirrors adjudicate()"), so the pair bootstrap ran with the roles swapped and refused a decisive winner (`MULTI_HYPOTHESIS` / branch `rival-favored`) — reproduced: train-best holdout 0.60 vs holdout-best 0.05, both gate-4 validated. Gate 4's recorded signed residuals *are* attributable to a candidate, so they now take precedence on the pair path and in the sole-survivor whiteness guard; the arguments remain the fallback for callers whose provenance records no series. After the change the same call returns `UNIQUE_SUPPORT` for the holdout winner with bootstrap branch `confirm` and contract `FULL`, identical to the no-argument fallback path.
    - Consequence recorded honestly: with `best` always the minimum-holdout candidate among the eligible set and rivals a subset of that set, the `Δ < −τ` "defeated on holdout" branch added in item 1 is now unreachable in both branches. It is left in place as a defensive refusal rather than deleted in the same change.
    - Thresholds unmoved: `RESID_RHO_FLAG`, `SOLE_SURVIVOR_GAMMA`, `tau_epi` sources and every canonical literal untouched; no new threshold literals.
    - Re-minted the frozen referee record (second re-pin of `adjudicate.py`): `70834e26…` → `e93e3307…` in `APEDPX_FROZEN_v1_2.json` and `PINNED_FROZEN_REGISTRY` (an intermediate `b0b2d1d4…` was pinned for the two code fixes and superseded within this same change when the version stamp below was edited; no other file was touched between them). `referee_change` and `tree_state` in the record updated with this rationale. The blind runner refused to score the drifted referee before the re-mint (`FreezeDriftError` / "Mint a new freeze record"), as designed — recorded here because the guard firing is the evidence that the protocol works.
    - `DYNAMIC_EPISTEMIC_VERSION` bumped `bootstrap-v3+solefid-v5` → `bootstrap-v4+solefid-v5`: v4 covers the promotion-SELECTION rules (holdout-priority selection from item 1, rival eligibility, provenance-attributed residual roles), which change which candidate is promoted rather than only how the verdict is printed. The pin in `tests/test_dyn_epistemic_bootstrap.py` was updated in the same change — the assertion still pins one exact value, it is not relaxed.
    - Verification: AGENTS.md dynamic suite **97 passed** (exit 0; 95 before the two regression tests added here); adjudication/bootstrap/sweep/tournament/calibration/uq/hygiene/flow-head/RIME-scope modules **104 passed** (exit 0); `cargo test --lib implicit` **4 passed** (exit 0); freeze-integrity and blind-guard tests 9 passed with the re-minted record. Both new regression tests were confirmed NON-VACUOUS by reverting the two hunks and re-running them: they fail on the pre-fix code with exactly the reported defects (`defeated on holdout by rival no_g4 (Δ=-0.48 < -τ=-0.05)` and `MULTI_HYPOTHESIS` where `UNIQUE_SUPPORT` is required), then pass again after restoring the fixed file (sha256 verified byte-identical).
    - `run_sweep()` re-run after the change against the pre-change baseline (both 36 cells, ~65 s each): **1 cell differs** — STAB/1%/rep0 flips `multi_hypothesis` [branch `rival-favored`, truth not promoted] → `unique_support` [branch `confirm`, truth promoted], which is the item-10 role-swap bug being corrected rather than a re-scored benchmark. Mean recovery 0.694 → **0.722**, mean abstention 0.306 → 0.278, false promotions 0 → 0, and no cell loses recovery. No sweep artifact files were regenerated from this run.


### Added — four frozen preregistrations (frozen BEFORE their confirmation runs; harnesses/artifacts committed separately)
- [`docs/PREREGISTRATION_JETFLOW_VIDEO_V02.md`](docs/PREREGISTRATION_JETFLOW_VIDEO_V02.md) (PREREG-JETFLOW-VIDEO-V02-2026-10-17): fresh-seed {101, 202, 303} matched-benchmark replication (H1 SSIM/smoothness ordering, H2 PSNR ≥ 9.5 dB, H3 training-cost ratio ≤ 0.5, H4 structural pass counts) + architecture invariants (strict 3D causality, SE(3)/Plücker exactness, text-encoder order/negation, Noether idempotence, TT fidelity, periodic parity). Margins derive from the disclosed exploratory seed-0 run.
- [`docs/PREREGISTRATION_HIGHER_ORDER_PDE.md`](docs/PREREGISTRATION_HIGHER_ORDER_PDE.md) (PREREG-HOPDE-2026-10-17): two-sided fresh-seed {810, 811, 812} validation — KdV, biharmonic KS-type, and second-order-temporal Klein–Gordon worlds must confirm; explosive positive-`c₄` ultraviolet and null worlds must be refused.
- [`docs/PREREGISTRATION_CROSS_DOMAIN_ENGINES.md`](docs/PREREGISTRATION_CROSS_DOMAIN_ENGINES.md) (PREREG-XDOMAIN-2026-10-17): fresh-instance confirmation batteries for DMRG (ED parity, thermodynamic limit, field arm), LOB transport (8-combo parameter grid, three recovery paths), audio chirp transport, molecular SE(3) relaxation (invariance, equivariance, mirror), robotics unicycle (convergence orders, Lyapunov descent).
- [`docs/PREREGISTRATION_QUANTUM_PROFILER_HOLDOUT.md`](docs/PREREGISTRATION_QUANTUM_PROFILER_HOLDOUT.md) (PREREG-QPROF-2026-10-17): first out-of-sample confirmation of the quantum profiler/coordinate transmuter (Brus power law, Landauer staircase, Schwinger tunneling) with smooth-polynomial-refusal and null guards on fresh noise seeds.

### Added — JetFlow-Video v0.2 pillars (exploratory evidence)
- **Compositional text encoder** (`packages/jetflow-video/jetflow_video/text_encoder.py`): sequence-level `(B, L_seq, D_text)` contextual encoding, negation gating, order-sensitive modifiers, HuggingFace bridge, prompt→physics steering extraction.
- **3D causal spatio-temporal VAE** (`bridge.py`): strict past-only `CausalConv3d`/`CausalResBlock3D` stack, diagonal-Gaussian KL posterior, end-to-end video-batch training.
- **Depth-aware SE(3)/Plücker transport** (`depth_se3.py`): exact Rodrigues/`V(ω)` exponential map, 6D Plücker ray embedding, learned monocular depth + 3D perspective warp with disocclusion source.
- **DC-STT neural velocity/source predictor** (`apedp_dyn/video/decoupled_transport.py`): `NeuralTransportFlowNet` (streamfunction-curl + compressible-potential decomposition, bounded disocclusion source) trained end-to-end through a differentiable transport step with Rust rollout.
- **Rust periodic warp kernel** (`rust/apedp-kernel/src/video.rs::warp_bilinear_2d_periodic`, PyO3 export, bridge `periodic=True`): generator-exact floor-then-mod backtrace indexing; torch / NumPy fallback / Rust parity pinned at 1e-12 in `tests/test_dyn_periodic_boundary_transport.py`.
- **Higher-order PDE hypothesis engine** (`apedp_dyn/pde/higher_order_engine.py`): spatial orders to 4th + temporal orders {1, 2} with variance-normalized BIC selection, proposer/referee separation, and the ultraviolet Fourier-symbol stability guard (rejects explosive `c₄ > 0` biharmonic terms).

### Committed — RIME §16e negative-result artifacts (prereg was `363e8df`)
- `apedp_dyn/epistemic/scope_refusal_mv.py` (multivariate rival-probe construction), `benchmarks/run_scope_refusal_mv_validation.py`, canonical artifact + disclosed void first run, `docs/RIME_SCOPE_MV_REPORT.md`, registered pins `tests/test_dyn_rime_scope_refusal_mv.py`. Recorded outcome: §16e as preregistered FAILS its two-sided validation (control wrongfully refused via c→0.1 linearization; §16c monomial guards undefined); v2 prereg required before any modification.

### Measured — exploratory JetFlow matched benchmark, post-remediation seed 0 (artifact `benchmarks/jetflow_vs_diffusion_matched_results.json`)
- `jet_tt_2step` 10.74 dB PSNR / 0.514 SSIM / 0.275 temporal consistency at 8 structural passes per clip; SSIM and temporal smoothness beat `diffusion_matched` (0.477 / 0.365) and `diffusion_frontier_K48` (0.492 / 0.372); PSNR remains 0.23 dB below `diffusion_matched`.
- Training at equal 300 steps: 14.40 s (JetFlow) vs 80.75 s (baseline) — ratio 0.178; final flow-matching loss 0.1528 vs 0.1631.
- Harness caveat recorded: the artifact's `model_passes_per_clip` for `jet_tt_2step` (9.0) includes one warmup generation amortized over `n_gen`; the structural count is 8 (the confirmation protocol resets the counter after warmup).
- AGENTS.md dynamic regression suite on the committed tree: **127 passed**.

## [Unreleased] — DC-STT landed honestly: horizon decay + first learned flow head (2026-10-17)

### Added
- **DC-STT stack committed** (`apedp_dyn/video/decoupled_transport.py`, `rust/apedp-kernel/src/video.rs` warp/divergence/disocclusion kernels, parity-verified bridge wrappers, multi-scene benchmark + artifact, spec). Big-tech comparison claims in the spec (§5) and module docstring softened to their measured scope: transport/reconstruction claims **given** a flow field — no learned prior, therefore no generation-quality claim vs any industrial video model.
- **`benchmarks/run_dcstt_horizon_decay.py` + artifact:** the serving-claim instrument the multi-scene benchmark lacked. Measured (artifact-backed): integrator fidelity vs a 10x fine-substep reference 18–43 dB, **but appearance detail (blur proxy) decays monotonically** — fluid-vortex detail energy falls to 11.3% of initial by H=128 (0.060 at H=256), camera-zoom to 25% — washout from nu-Laplacian dissipation + numerical diffusion. Verdict: current evidence supports **short-horizon transport/upsampling**, not unbounded-rollout serving quality.
- **`apedp_dyn/video/learned_flow_head.py` — first learned component in the DC-STT path:** small CNN inferring per-pixel v(x) from 2 frames, self-supervised (advection-consistency + div² + TV), trained **entirely on CPU** under the wall-clock deadline discipline; `warp_frames` pinned to exact parity with the Rust-bridge NumPy fallback warp (max|Δ| < 10⁻¹²; backtrace sign convention tested by point-delta transport). `tests/test_dyn_learned_flow_head.py` 6/6.
- **`benchmarks/run_learned_flow_dcstt_benchmark.py` + artifact:** evaluation on the shared fluid-domain split (n_clips=10 so the matched benchmark's burnable holdout `fluid_clip_0003/0005` is never trained on — an earlier draft's n_clips=20 default leaked those ids into TRAINING and was fixed before any result was accepted). 90 s deadline CPU training: 1367 steps, loss 4.4e-4 → 6e-5. HONEST VERDICT (single-run): the learned flow head does **not yet beat persistence** in aggregate (38.79/36.06/33.42 dB vs 39.54/37.21/35.24 dB at H=4/8/14); it wins only on one of two holdout clips. Recorded as measured; no tuning to green. The head is the scaffold for re-injection: horizon-decay evidence above shows rollout quality, not throughput, is the binding constraint for long rolls.

### Fixed
- **`rust/apedp-kernel/src/flow_head.rs` — `erf_hp` Maclaurin branch used the wrong series constant** (`2/√(2π)` instead of `2/√π`, smaller by exactly √2), corrupting every GELU nonlinearity in the Rust flow-head port: torch parity failed at max|Δv| 2.7–8.0 (saturated, 254/256 pixels wrong) while a zeroed-input bisection masked it (any index bug is invisible when the gathered input is all zeros). Fixed to `FRAC_2_SQRT_PI = 1.128_379_167_095_512_6`; the Lentz `gcf` branch was already exact. Crisp diagnostics first, then repairs: layer-count bisection (n_layers=1 already wrong at 0.977 corr), a numpy im2col/GEMM mirror verified vs torch to 6.4e-15 (conv + layout + channel order exonerated), and a cargo-test erf grid that localized the defect to ±0.28 at x=±1.45 with correct zeros — the constant was the only remaining suspect. Post-fix: |erf_hp − math.erf| ≤ 3.3e-16 on [-6, 6]; forward parity ≤ 1.5e-14 at n_layers=1/2/3 (hidden=8); `tests/test_dyn_flow_head_rust.py` 4/4 incl. 300-step training and rollout parity; full AGENTS.md regression suite 58/58.
- **Permanent pin added:** `flow_head::erf_hp_grid_tests::erf_hp_matches_the_reference_grid` — 397-point grid ([-6, 6] step 0.05, [1.3, 1.7] step 0.005 both signs at the branch crossover, saturation out to ±12) against f64-exact `math.erf` references, tol 8e-16 (~2 ulp of cross-library noise, six orders below the caught defect), plus an odd-symmetry assertion at 4e-16 and a max-deviation diagnostic line so failures report the guilty (x, ref).
- **`numeric.rs` audited against Python reference grids (2026-10-17):** the claims held everywhere; every claim was re-measured, not trusted. `erf_approx` (A&S 7.1.26) errs 1.388e-7 max vs `math.erf` on [-10, 10] — 93% of the documented 1.5e-7 ceiling, pinned at 1.45e-7; `bessel_j0` errs 1.06e-14 absolute on [-8, 8] (doc: ~1e-12), pinned at 1e-12 with the degradation band [8, 16] pinned BOUNDED at 2e-11 (measured 1.48e-11 at x=15.65); one-sided-Jacobi `singular_values` matches `numpy.linalg.svd` on 8 deterministic embedded matrices, pinned at 1e-10 relative. The mirror-set (metrics/conditioning/QR/Cholesky/precision) stays covered differentially by `tests/test_rust_bridge_*.py`. All pins live in `numeric::reference_grid_audit` (Rust tables generated from Python 3.12 `math.erf`/`scipy.special.j0`; nothing is regenerated in Rust). Two stale docstrings claiming the flow-head GELU uses `erf_approx` were corrected to the grid-pinned `erf_hp`.
- **Degenerate-regime extension of the Jacobi SVD pin:** a second embedded reference module (`numeric::jacobi_degenerate_pins`) covers exact rank-1, duplicate + zero columns, a 1e-13 near-singular pair, a diagonal matrix with zero pivots, and collinear column pairs (condition numbers 4.6e13…1e300). Scaled-absolute tolerance 1e-10·σmax deliberately avoids relative comparison of numerically-zero sigmas — the defect class it guards is a garbage large σ or a diverged rotation. Verified against `numpy.linalg.svd`; full rust lib suite 173/173.
- **Learned-flow head: 2× CPU budget rung measured (`benchmarks/learned_flow_dcstt_2x_results.json`, seed 0, no other change):** 90 s → 180 s deadline (1367 → 4721 steps, train loss 6.0e-5 → 2.5e-5). Aggregate learned-flow PSNR 38.79/36.06/33.42 → **39.08/36.65/34.35 dB at H=4/8/14 vs persistence 39.54/37.21/35.24**: the persistence gap shrinks but does NOT close (−0.75→−0.47, −1.15→−0.56, −1.81→−0.89 dB). Per-clip asymmetry persists and slightly strengthens: clip 0005 beats persistence at every horizon (margins grew to +0.56/+0.64/+0.75 dB); clip 0003 still trails (−1.5…−2.5 dB). Verdict: budget alone is not the binding constraint — the single-2-frame-context self-supervised objective with its dye-renormalization loss floor is; the evidence-backed next lever is supervised flow transfer (teacher-generated pseudo-flows / distillation into the head), not more CPU seconds.
- **Stage 3 LANDED — flow-supervised trainer + the 4× budget rung closes the persistence gap.** Two mechanisms, measured under the repo's wall-clock discipline (seed 0, train-ids-only teacher, holdout untouched):
  1. *Stage-3 supervision* (`learned_flow_head.flow_supervision_loss`, `train_flow_head(..., flow_targets=...)`): the fluid-domain generator's own ANALYTIC field, exposed by `make_fluid_clip(..., return_flow=True)` — train clips only, stacked in the trainer's exact concat order, in head-units (vx, vy) = (v_gen, u_gen) with the swap pinned by a hard warp-agreement test. One real bug caught and fixed en route: the supervision target must be `flow[ti+1]` (the generator advects frames[ti] with field[ti]); the first cut used `flow[ti-1]` (stale by two indices — caught by a magnitude probe that REFUTED the tanh-saturation hypothesis first, max|teacher| = 3.66 < the 4.0 bound). Both artifacts kept: `learned_flow_flowsup_180_results_staleindex.json` and the corrected `learned_flow_flowsup_180_results.json`.
  2. *4× rung* (`learned_flow_dcstt_4x_results.json`, re-run on a quiescent machine after the first attempt was CPU-confounded (kept as `…confounded.json`)): **learned 39.50/37.32/35.35 dB vs persistence 39.54/37.21/35.24** — parity at H=4 and ABOVE persistence at H=8/H=14 (+0.10/+0.11 dB).
  Ladder (all quiescent, both axes recorded; NOTE per-step wall-clock varied 0.038–0.28 s/step across runs from concurrent machine load, so the budget axis is wall-clock, not steps — steps: 1367/4721/1492 at 90/180/360 s):
  | H | 90s | 180s | 360s (4×) | flow-sup 180s (correct idx) | persistence |
  |---|---|---|---|---|---|
  | 4 | 38.79 | 39.08 | **39.50** | 39.12 | 39.54 |
  | 8 | 36.06 | 36.65 | **37.32** | 36.96 | 37.21 |
  | 14 | 33.42 | 34.35 | **35.35** | 34.57 | 35.24 |
  Honest verdicts (recorded, not swept): (a) flow supervision beats self-supervision at EQUAL 180 s budget at every horizon (+0.04/+0.31/+0.22) and produces clip-0003's best numbers ever — but regresses clip 0005 below persistence (the head there converges to ≈near-zero flow), so the aggregate does NOT beat persistence at this budget; the 4× self-supervised rung is the better aggregate buy. (b) `flow_supervision_loss` evaluates the head twice per step (consistency + supervised MSE) — 647 steps in 180 s vs 4721 self-supervised; a shared-forward optimization is open and deferred. (c) teacher construction is train-ids-only by construction (holdout ids 0003/0005 never enter `make_fluid_clip(..., return_flow=True)`).
  Tests: `tests/test_dyn_learned_flow_head.py` 9/9 (swap convention, div-free stencil aligned with `test_dyn_fluid_domain.py`, supervised-beats-self on a known target, targets-shape rejection); `test_dyn_fluid_domain.py` + `test_dyn_decoupled_continuous_transport.py` 9/9 green on the merged tree.
- **`DiffusersVAEBridge` added to `apedp_dyn/video/vae_bridge.py` (Stage-2 wiring of the distillation ladder):** HuggingFace `diffusers.AutoencoderKL` behind the exact `VideoVAEBridge` interface surface (encode/decode frame+video+TT+psnr), lazy optional import with actionable guidance, float32-internal/float64-boundary casting, scaling_factor applied symmetrically, and loud channel-count refusal (no silent remap). Config-only (random-init tiny KL — 0.41 needs the explicit per-block tuple types, pinned after reproducing the truncation bug) and pretrained-weights paths are distinct: `pretrained_weights_loaded` records which, so no artifact can claim a prior it has. The pretrained test probes network/cache and SKIPS (never passes) when weights are unreachable — in this env it is skipped. Tests: `tests/test_dyn_diffusers_vae_bridge.py` 8 passed / 1 skipped; `diffusers==0.41.0` installed into the repo venv.
- **Quick follow-up arc (2026-10-17, same day): shared forward + periodic-protocol rerungs + trained-weights kernel parity.** (i) `flow_supervision_loss` now evaluates the head ONCE per step (shared forward between the consistency warp and the supervised MSE); numerically identical to the naive composition — pinned at 1e-12 (`test_flow_supervision_loss_equals_the_naive_composition`, suite now 10/10). (ii) The concurrent session's periodic-warp default in the merged benchmark changed the protocol: the benchmark now trains AND rolls out toroidally (`periodic_warp: true` recorded per artifact, `--clamped-boundary` ablation flag kept). Fair (exclusive-CPU) rerungs under the periodic protocol, new filenames so the committed clamped-warp artifacts survive: flow-supervised 180s lands at **37.21/34.95/32.78** vs persistence 39.54/37.21/35.24 (.AreEqual — WORSE than its clamped-warp 39.12/36.96/34.57: the periodic protocol degrades the supervised head's rollout, and the shifted budget/steps axis is recorded in the artifact); self-supervised 4× periodic lands at **39.50/37.23/35.24** (parity with persistence everywhere — H=4 −0.05, H=8 +0.01, H=14 +0.00). Ladder truth: under PERIODIC semantics the best aggregate equals persistence, no longer beats it; the earlier above-persistence verdict was CLAMPED-warp-specific and is labeled as such. (iii) **Distill parity on the trained weights: the 4×-periodic head was exported (`learned_flow_dcstt_4x_periodic_head.pt`, 98 KB) and drives the kernel — `tests/test_dyn_flow_head_distill_parity.py` passes at max|Δv| ≤ 1e-12 on real train-clip contexts (holdout ids never consumed), weights-pack length asserted against the spec, missing-artifact path SKIPS.** The Rust serving path now provably executes the same weights that earned the rung's numbers.

### Known limitations (recorded, not swept)
- Single-run artifacts; the learned head needs more CPU budget/architecture before it earns a serving claim. Integer-shift baseline collapsed to (0,0) on these clips (velocity ≈ uniform translation), so persistence is the real comparison. Stage-3 adds: clip-0005 regression under flow supervision (open); shared-forward double cost in `flow_supervision_loss` (open); per-step-cost variance between ladder rungs — wall-clock matched, step-mismatched, disclosed in the ladder table.

### Added
- **`benchmarks/run_jetflow_vs_diffusion_matched.py` + `benchmarks/jetflow_vs_diffusion_matched_results.json` + `docs/jetflow_vs_diffusion_matched_benchmark.md`:** replaces the earlier equal-*step-count* protocol (which under-budgeted the slower JetFlow stack) with equal **wall-clock training budget** enforced by an in-loop `deadline_s` stop added to all three trainers (`distill.train_flow_model`, `distill.distill_solution`, `diffusion_baseline.train_flow_model_baseline`): probe-estimated step counts could drift ±33% from real-run per-step cost (first run: jet 81.7 s / baseline 117.8 s for a 90 s budget — baseline overshot, jet under-consumed), so the deadline, not the probe, is the binding budget axis. v2 artifact: jet 3172 steps vs baseline 3478 steps, both exactly 90.0 s (28.38 vs 25.88 ms/step — a near tie, confirming the earlier per-step-cost attribution was wrong and the quality gap is architectural). Model-forward counts are *measured* per clip (not hand-estimated); identical CFG=3.0, trainer-consistent sampler directions, and identical burnable hold-out ids across architectures.
- **`packages/jetflow-video/jetflow_video/diffusion_baseline.py` (new):** the fair textbook baseline — small UNet-style ResBlock denoiser (152k params vs JetFlow's 133k), standard cosine-schedule DDPM v-objective, CFG-capable; trained through the same budget controller, same split, same instruments.
- **Honest verdict (single-run seed 0, artifact-backed):** at matched budget and ~0.41–0.45 s/clip the diffusion baseline wins every measured quality instrument (PSNR 12.59 vs 7.43, SSIM 0.663 vs 0.250, FVD-proxy 0.272 vs 1.053, temporal 0.133 vs 0.642); quality plateaus at K≈32 (K=48 spends 1.75× longer for nothing); the distilled 2-step student is the only sub-50 ms option at 14.2 ms/clip (29× faster, PSNR 9.70) — JetFlow's defensible claim is serving-side latency, not matched-compute quality. Known limitation (recorded, not swept): single-run variance unquantified; no repeats yet.

### Tests
- Full AGENTS.md regression battery plus the video/jetflow suite: 92 passed (one `test_end_to_end_text_to_video_jet2_flow_matching` failure in the first pass did not reproduce in two 13-file suite reruns and passes in isolation — one-off transient, recorded honestly rather than hidden).

## [Unreleased] — mypy strict debt cleared; blind-freeze re-mint v1.2 (2026-10-17)

### Fixed (typing)
- **Both mypy CI gates exit 0:** `mypy apedp_dyn/ --ignore-missing-imports` and `mypy apedp/ apedp_dyn/ --ignore-missing-imports` report no issues (was 122 errors in 38 files). The `apedp/` core gate was already green; this release brings `apedp_dyn/` to parity.
- Honest fixes by category: `isinstance(dict)` guards at every JSON boundary (`canonical_spec`, `frozen_spec`, `ode/dispatch`, `riemann`/`goldbach` rust-binary parsers, lean-provenance loader); value-preserving `np.asarray` wraps at ndarray-declared return boundaries (video stack, execution-cost models, `high_dim_pde` Rust-engine wrappers); union request types widened ONLY where callers legitimately pass ndarrays (`staircase_prior`, `fit_step_family`, `step_family_basis`, `sample_at_timestamps`, shape-inference request forms); `_dp_segment` return annotation corrected to its actual `(list[list[float]], ndarray, dict)` contract; `ProposerBase.fit_and_propose` abstract contract made call-shape-agnostic (`*args/**kwargs`) so heterogeneous concrete proposers conform to AGENTS.md §1.1; scorer factories in `rime_repair_engine` (uniform `Callable[[pd.DataFrame], ndarray]` signatures, bound-at-emit-time closures); Tesla-style conditional nested scorers renamed in `coordinate_transmuter`; missing `Any`/`Callable` imports; integer rank selection via numpy scalar `.sum()` in `tensor_train_video`.
- `CandidateProposal.parameters` widened to `dict[str, Any]` — heterogeneous by contract: numeric fit parameters plus predictive-status strings such as `identifiability_status` (readers cast explicitly); no runtime instance shape changed.
- **No thresholds touched:** per AGENTS.md §1.3 no new threshold literals anywhere; the new `cross_asset_gate.evaluate_cross_asset_gate_matrix` diagnostic adapter only computes within-class/pooled/two-tier Sharpe inputs and delegates all verdicts to the canonical `evaluate_cross_asset_gate` (sandbox probe callers called an API shape that always threw and silently fell back — they now reach the real gate and read `report.passed`).

### Fixed (frozen-record protocol)
- `apedp_dyn/sandbox/grammar_space.py` was type-hardened (explicit `from typing import Any` import; value-preserving `np.asarray` wrap at an ndarray-declared return boundary) — no referee logic, thresholds, or canonicalization changes. Per the runner's own mint-first protocol, **`APEDPX_FROZEN_v1_2.json` supersedes v1_1** (drift: `6f4da69fb15b…` → `aba0f42abb0f…`, sole guard-file change), `blind_benchmark.FROZEN_RECORD` and the `frozen_spec.py` pinned hash were re-pointed, and all 15 referee guard files verify against the new record.

## [Unreleased] — Rust↔Python Bridge Hardening (2026-10-17, `production`)

### Performance
- **Zero-copy NumPy columns in the native kernel:** `extract_columns` deep-copied every dataset column into owned `Vec<f64>` on every FFI crossing, re-copying the same data on each per-candidate call (`evaluate_batch`, `score_candidates_parallel`, `rime_evaluate_json`, …). The new `DataCol` erased-column trait (`rust/apedp-kernel/src/data.rs`) makes owned `Vec<f64>` and borrowed `ndarray::ArrayView1<f64>` interchangable, and `PythonColumns` (a guard-backed view holder) materializes zero-copy views over the Python-side buffers across `allow_threads`. The parser, numeric, genetic, and RIME (graph/eval/search/additive/residual) surfaces now take `&[V: DataCol]`; the JSON-fed internal paths keep their owned `Vec<Vec<f64>>` unchanged.
- **Compile caching at the bridge:** `RustKernelBridge` re-parsed identical expression strings at every FFI crossing. Added `_kernel_compile` (LRU, 4096 entries, keyed on `(expr_str, variable_names)`, owner-guarded so a swapped kernel clears the cache) and a `lambdify` cache for the Python fallback (keyed on expression text).
- **Release profile:** workspace `Cargo.toml` sets `codegen-units = 1` alongside `lto = true`.

### Added
- **Zero-copy 1-D extraction also:** the guarded-views pattern extends past column lists — single-array arguments (targets, `y_true`/`y_pred`) now extract through a `GuardedVector` (pin-once `PyReadonlyArray1` guard; contiguous buffers hand out borrowed `&[f64]` slices via `PyReadonlyArray::as_slice`, strided buffers take one owned fallback copy), so `compute_metrics`, `solve_linear_coefficients`, `evolve_native`, and every RIME entry point stop re-copying their target and series arguments per call.
- **Kernel-path counters + `kernel_path` field:** every `RustKernelBridge` instance exposes `tallies` (rust/fallback call counts) and `last_path`, and every returned result dict (metrics, scoring rows, probes, conditioning, linear solves, precision boundary, evolve rows) carries a `kernel_path` tag. A benchmark run can now *prove* the Rust fast path executed instead of being unable to distinguish it from the silently-correct Python fallback.
- **Loud missing-kernel warning:** constructing a bridge without the native `apedp_kernel` extension logs exactly one WARNING naming the `maturin develop --release` build command (previously the fallback was invisible and "easy to miss", in the bridge's own docstring words).
- **CI wheel artifacts:** `build-kernel-wheel` job builds the manylinux wheel via `PyO3/maturin-action` and uploads it as a workflow artifact on every push/PR, so the kernel is `pip install`able without running maturin locally; a tag-gated `publish-kernel-wheel` job publishes to PyPI only when a `PYPI_API_TOKEN` secret is configured.

### Measured (before/after, min-of-3 interleaved fresh-process runs, 600×4 float64 dataset)
- **Reference engine (Python fallback) repeated eval: 2678.9 → 34.5 µs/call (77×).** The lambdify cache is where most wall-clock in fallback-heavy workloads went.
- **Batch scoring round (15 candidates): 1087.6 → 888.1 µs (−18%); per candidate 72.5 → 59.2 µs.** Compile-free cache hits plus view-based columns.
- **Repeated candidate-evaluation call: 110.8 → 95.6 µs (−14%).** Per-call cost at this dataset size is dominated by Rayon pool dispatch (~85 µs), so the copy-elimination scaling grows with dataset size (the elimination only has to cover the ~19 KB per-call re-copy here).
- **`compute_metrics`: 11.1 → 7.6 µs (−32%).**
- Unique-expression direct eval (no cache effect): 24.9 → 20.7 µs (−17%). Kernel-level `score_candidates_parallel`: 817.7 → 785.2 µs — no kernel-path regression; the only regression risk found in profiling (machine-load drift) was excluded by interleaved min-of-N.
- Artifacts: `benchmarks/run_bridge_fastpath_benchmark.py` (identical driver for both stacks) and `benchmarks/bridge_fastpath_benchmark_results.json` (raw runs + minima). The "before" stack is the git HEAD kernel (owned `Vec<Vec<f64>>` copies, bare `kernel.compile` at every crossing) built from a clean wheel into an isolated venv.

### Tests
- `tests/test_rust_bridge_observability.py` (new, 8 tests): tallies prove the fast path ran, result dicts carry `kernel_path`, failing kernels tally fallback (not vanishing), compile cache serves repeated expressions once, cache is keyed on the variable table (wrong-variable-layout reuse is the dangerous failure), the fallback lambdify cache is consulted, and the missing-kernel warning fires exactly once at WARNING level with the build command.
- Full verification: 179 Rust tests (`cargo test`), 45 bridge tests, 26/26 differential conformance against the rebuilt release kernel, 58/58 AGENTS.md regression battery.

---

## [Unreleased] — JetFlow-Video Honest-Fast-Path Remediation (2026-10-17, `experimental`)

### Remediated (fraud vectors)
- **Fake NumPy trainer removed (`apedp_dyn/video/jet_flow_model.py`):** the NumPy `train_step` was not a trainer — it mutated one bias vector by a single scalar mean-gradient while every other conditioner weight stayed frozen-random. The method is now diagnostics-only (`DeprecationWarning`, NO parameter mutation); real training routes to `packages/jetflow-video JetFlowTrainer` (full-parameter AdamW; grad-flow verified). Pinned by `tests/test_dyn_jet_flow_sota_path.py::test_jet_flow_video_model_train_step` (behavior change justified in the test docstring — the old pin certified fake behavior).
- **Salted-hash prompt embedding replaced (`apedp_dyn/video/semantic_bridge.py`):** `embed_text_prompt` seeded random Gaussians from the Python builtin `hash()`, which is salted per process (PYTHONHASHSEED) — embeddings were non-reproducible across runs and carried zero structure. Replaced with `feature_hash_word_char_trigram_v1` (MD5-indexed hashing trick over word + char-trigram tokens + sentence-context term). Byte-identical across processes (subprocess-pinned); lexical similarity preserved (0.744 shared-vocabulary vs 0.229 unrelated). Honestly labeled as lexical features, not CLIP-scale semantics.
- **Found package bug — sampler direction:** `model.generate_video` integrates t: 0 -> 1 from noise, but `trainer.compute_loss` parameterizes t = 0 as data / t = 1 as noise — the shipped sampler ran the ODE backwards relative to training. The samplers in `first_frame.py` and `distill.py` implement the trainer-consistent direction (integrate t: 1 -> 0); the package inconsistency is documented, unresolved in `generate_video` itself.
- **Benchmark-doc honesty:** `docs/jet_flow_video_benchmark.md` executive-summary numbers (8.7x–24.4x, 13.9x, 0.02 ms/frame) were superseded by its own artifact (75.85x–186.18x, 8.03x, 0.782 ms/frame); treat the JSON artifact as the citable source. The "2-step beats 50-step diffusion" benchmark is a strawman (integrators on a told analytic field) — replaced by genuine teacher->student distillation curves below.

### Added
- **`apedp_dyn/video/eval_gate.py` + `tests/test_dyn_video_eval_gate.py` (5/5):** held-out certification instruments — PSNR/SSIM (video-aware), temporal-consistency (smooth advection beats frame-shuffled), FVD PROXY (sliced Wasserstein-2 over a frozen deterministic featurizer; explicitly NOT I3D-FVD), burnable held-out split (disjoint/deterministic/exclusive), and a certify API with caller-supplied thresholds (no new threshold literals).
- **`apedp_dyn/video/fluid_domain.py` + `tests/test_dyn_fluid_domain.py` (4/4):** domain-scoped dataset — 2D incompressible NS tracer clips (velocity from streamfunction, grid-divergence-free |div| < 1e-10 — the Noether projector finally applies to a physically real object here), semi-Lagrangian dye advection, scope manifest with exclusive holdout ids.
- **`packages/jetflow-video/jetflow_video/first_frame.py` + `tests/test_dyn_jetflow_anchored_and_distill.py` (5/5):** real-first-frame conditioning — anchored corruption (frame 0 stays clean), anchor-excluded loss, and generation that re-imposes the anchor bitwise at every step in the trainer-consistent sampling direction.
- **`packages/jetflow-video/jetflow_video/perceptual_decoder.py` + `tests/test_dyn_jetflow_perceptual_decoder.py` (3/3):** zero-init residual decoder on the Haar base (training STARTS at exact Haar parity — monotone upgrade path, no regression cliff), patch-LSGAN discriminator + gradient loss trained on fluid frames.
- **`packages/jetflow-video/jetflow_video/distill.py` + `benchmarks/run_jetflow_distill_benchmark.py` + `benchmarks/jetflow_distill_curves.json`:** authentic teacher->student solution distillation (trained teacher on the fluid domain, run split-disciplined). Landmark: student 2-step endpoint RMSE vs 16-step teacher 0.808 -> 0.507 measured on the artifact; the honest replacement for the old integrator benchmark.
- **`benchmarks/run_jetflow_cpu_export_benchmark.py` + `benchmarks/jetflow_cpu_export_benchmark_results.json` + `tests/test_dyn_jetflow_cpu_export.py` (3/3):** ONNX export (static traced graph per clip config — traced attention cannot carry a dynamic temporal axis; documented) plus `onnxruntime` dynamic int8 quantization: 173.5 KB -> 104.9 KB with parity max|Δ| = 0.0; measured latencies (eager 8.68 ms / ORT fp32 5.32 ms / ORT int8 9.97 ms) — int8 is a SIZE win at this scale, ORT fp32 the speed win; the artifact reports both rather than marketing one.

### Known limitations (recorded, not swept)
- Synthetic fluid-domain evidence only; no real-data photorealism claim derives from this round.
- FVD proxy is not I3D-FVD; ranked-separation instrument only.
- ONNX graphs are static per clip length (regenerate per export).
- `model.generate_video` sampler-direction bug is documented but NOT fixed in place (deployed samplers route through the corrected `first_frame.py`/`distill.py` paths).

## [Unreleased] — RIME-AD Native Rust Engines & Unified Quantum Benchmark (`experimental`)

### Added
- **`rust/apedp-kernel/src/rime/total_variation.rs` (BV-ROF Engine):** Fast Condat exact $O(N)$ 1D Total Variation filter (`tv1d_condat`) and staircase jump analysis (`tv_denoise_and_analyze`). Recovers quantized plateaus and quantum step heights without polynomial Runge oscillation, robust through 5% observation noise.
- **`rust/apedp-kernel/src/rime/implicit.rs` (Sparse Implicit SVD Null Space):** Symmetric algebraic relation discovery finding sparse null-space vectors of generalized feature tensors $\Theta(x, y) c \approx 0$ via Jacobi eigendecomposition. Discovers Detuned Rabi Lorentzian resonance $(25 + \Delta^2) P_e - 25 = 0$ in $0.31$ ms.
- **`rust/apedp-kernel/src/rime/diff_pade.rs` (Differential Padé & Jet Algebra):** Discovers governing polynomial differential equations $P(x) y' + Q(x) y = 0$ on the Jet bundle without pre-planted exponential primitives. Discovers Schwinger pair production ($x^2 y' - (2x+1)y = 0$, $R^2 = 0.9999$), Landau-Zener ($v^2 y' - 0.75 y = 0$), and BCS energy gap ($g^2 y' - y = 0$), with forward RK4 trajectory integration.
- **`rust/apedp-kernel/src/rime/enumerate.rs` (Shared-Computation Canonical DAG):** Systematic complexity-ordered small-expression enumeration with hash-consed memoization (PSE-style shared computation). Recovers Resonant Rabi harmonic dynamics ($1 - \cos^2(t)$, $R^2 = 1.000000$) and Brus confinement ($R^2 = 0.998414$).
- **`benchmarks/run_quantum_all_equations_benchmark.py` & `docs/quantum_all_equations_benchmark_results.md`:** Unified machine-measured benchmark across all 9 physical and quantum equations tested before.
- **`tests/test_dyn_rust_all_equations_benchmark.py` (5/5 green):** Continuous integration assertions validating all Rust engines against all equations.
- **`benchmarks/run_population_extrapolation_rime_ad_vs_ga.py` & `docs/population_extrapolation_rime_ad_vs_ga.md`:** Multi-population extrapolation benchmark comparing RIME-AD against GA/GP across Pop A-train, Pop A-test, Pop B (fresh seed), Pop C (parameter shift $c_1 = 2.0 \to 2.5, c_2 = 3.0 \to 2.0$), Pop D (heavy-tailed Laplace OOD), and 7-channel distractor noise challenge.
- **`tests/test_dyn_population_extrapolation_benchmark.py` (3/3 green):** Automated regression verifying that RIME-AD matches GA-Large across all populations, exactly recovers true mechanism $+1 \cdot (2.9956 \cdot x_3[t-2] + 1.9966 \cdot (x_1[t-1] x_2[t-1])) + 9.9988$, exhibits zero false discoveries on 7 distractor noise channels, and operates ~246x faster than GA-Large.
- **`rust/apedp-kernel/src/rime/koopman.rs` (Delay-Embedded Spectral Discovery):** Affine order-2 linear recurrence fit by least squares with companion-matrix eigenvalues (complex-pair frequency/decay readout) and forward rollout helper. Recovers harmonic $\omega = 2.0000$ (holdout rollout $R^2 = 1.000000$) and damped $(\omega = 3, \sigma = -0.5)$ spectra. Explicit v1 scope: scalar uniform trajectories, two-mode dynamics.
- **`rust/apedp-kernel/src/rime/lie_symmetry.rs` (Lie Point-Symmetry Discovery):** Exact bivariate polynomial algebra for the linearised symmetry condition $\eta_x + (\eta_y - \xi_x) f - \xi_y f^2 - \xi f_x - \eta f_y = 0$, with Gram/Jacobi nullspace basis. Verifies scaling/x-translation symmetries of $y' = y$, the exact 9-dimensional translation algebra of $y' = 0$, and $y$-translation for $y' = x$. Explicit v1 scope: first-order polynomial ODEs.
- **`tests/test_dyn_rust_koopman_vs_ga.py` (3/3 green) & `tests/test_dyn_rust_lie_symmetry.py` (3/3 green):** Matched-split Koopman vs native-GA comparison (identical train/holdout; GA finds the exact $-2\sin^2(t)+1$ identity at holdout $R^2 = 1.000000$ in 14.71 ms / 1560 evals vs Koopman 0.31 ms) plus Lie symmetry verification. gplearn/PySR unavailable in this environment, so the GP side is the native genetic-programming kernel path.
- **`benchmarks/run_five_gate_validation.py`, `benchmarks/five_gate_validation_results.json`, `docs/five_gate_validation_results.md`, `tests/test_dyn_five_gate_validation.py` (green):** Executed five-gate programme, all gates passed — exact recovery on 8/9 equations with the Planck limitation held; honest noise degradation (Schwinger 2% noise refuses exact claim); 4/4 adversarial abstentions; matched-compute ablation RIME vs native GA (Rabi tie at $R^2 \approx 1.0$; Brus GA overfits to holdout $R^2 = -2.93$ while RIME transfers); sealed confirmation under fresh seed 12345.
- **Planck honesty fix (`tests/test_dyn_planck_limitation.py`, green):** Proved no exact first-order polynomial ODE exists for Planck radiance ($u'/u = 3/v - e^v/(e^v-1) \notin \mathbb{Q}(v)$); the engine correctly returns `is_exact_recovery = False`. Benchmark verdict corrected from `PASSED` to `RECORDED LIMITATION`, with `is_exact` now stored in the JSON artifact.
- **Exponential generation (`enumerate.rs` `include_exp`, `search_rational_pairs`, native-GA `allow_exp`):** `exp()` nodes are now generatable in shared-DAG enumeration and native GA trees (previously evaluable but unreachable). The Padé-style rational-pairing stage (transcendental-denominator shortlist, spike guard, top-k budget) recovers the Planck textbook form $v^3/(e^v-1)$ at $R^2 = 1.000000$ with fresh-domain transfer at $R^2 = 1.000000$. GA with `exp` wired still misses Planck at small budget (grammar access is not discovery). Pinned by `tests/test_dyn_rust_exp_discovery.py`.
- **`RimeAdProposer` univariate fan-out (`max_univariate_features`):** multi-feature frames now run Representations 1-4 per feature instead of OLS-only; on 90-asset dailies all engines engage but every emission fails forward holdout (documented negative result).
- **Extended misc ops (`Neg`/`Log`/`Sqrt`/`Abs`) in enumeration and native GA:** domain-guarded evaluator arms (NaN out-of-domain, skipped by callers) gated behind `include_extended` (enumerator) and `allow_misc` (GA, threaded from `SearchConfig` through the bridge). Enumeration recovers all four forms at R² = 1.0 with the gate provably off by default; GA discovers `abs(x)` and `log(u)` exactly at small budget. General `Pow(x, y)`, `tan`, hyperbolic and inverse-trig deferred (numerical safety + pool explosion). Pinned by `tests/test_dyn_rust_extended_ops.py` (4/4).
- **Screening-gated univariate fan-out (`screen_top_k`, quadratic screen):** `RimeAdProposer` ranks features by [1, x, x²] OLS dev R² (catches even-symmetric dependence invisible to linear correlation) and fans Representations 1-4 over the measured top features instead of a blind 12-cap. Pinned by `tests/test_dyn_rime_ad_screening.py` (3/3).
- **Domain grammar expansion (pi/e, min/max, sign, cbrt, erf, besselj):** parser gains two-argument `min(a,b)`/`max(a,b)` as real `Node` variants (the parser previously rejected all multi-argument forms), `Func::Erf` (Abramowitz-Stegun 7.1.26, |err| ≤ 1.5e-7), `Func::BesselJ0` (power series, ~1e-12 for |x| ≤ 8), `Func::Cbrt`, and exact `pi`/`e` leaf constants. Enumerator gains the same ops behind `include_extended` (pi enters the depth-0 pool so `sin(pi*x)` is reachable; min/max join the commutative binary dedup). GA gains all of it behind `allow_misc` (9-op misc group + 8% pi/e leaf rate). Two real bugs caught by the new pins: `signum(0.0) == 1.0` in Rust made `erf_approx(0)` return ~1e-9 (now special-cased to exactly 0), and `assert_eq!` on NaN always fails (switched to `is_nan()`). Pinned by `tests/test_dyn_rust_domain_ops.py` (6/6) plus Rust accuracy tests against reference values (erf(1)=0.8427007929, J0 first zero 2.4048255577). Deferred: general `Pow(x,y)` (unsafe on negative bases), `tan`/hyperbolics/inverse-trig (pool explosion, low physics priority).
- **Advanced ops (`tan`/`sinh`/`cosh`/`tanh`/`asin`/`acos`/`atan` + general `Pow(x,y)`):** gated behind `include_advanced` (enumerator) and `allow_advanced` (GA). Enumerator recovers all 7 unary forms at R² = 1.0 with the gate provably off by default. GA discovers `atan(x)` and `tanh(x)` exactly and `x^2.5` via `x·x·√x`; general Pow with random exponent fires at 30% rate under the advanced flag. GA formatter bug fixed: advanced Func names previously rendered as `f(x)` (the `_ => "f"` fallback — invisible before because GA never generated them). Pinned by 3 added tests in `tests/test_dyn_rust_domain_ops.py` (9/9 total).
- **Dual-harness adjudication (2026-10-15, measured):** the engine-level and referee-level verdicts coexist without contradiction -- they measure different properties. Schwinger: engine exact-recovery TRUE (traj R² = 0.999903, exact formula) AND referee-whiteness refuse TRUE (residual RMSE 0.019 on scale 6.3 but smoothly drifted, ρ1 = 0.93). BCS: same layered outcome (R² = 0.999999, RMSE 2.8e-4, ρ1 = 0.95). Resolution: adopt LAYERED verdicts (structure gate + whiteness gate) everywhere, per the six-metric design; neither harness is wrong, each is incomplete alone.
- **Planck scope correction (adjudicated 2026-10-15):** the non-existence proof covers first-order LINEAR polynomial ODEs only. A quadratic Riccati relation v^3 u' - (3v^2 - v^3) u + u^2 = 0 holds to ~2e-7 (verified); no current engine discovers second-degree Jet relations -- recorded follow-up. Test docstring corrected; behavioral pin (linear abstention) unchanged and green.
- **Riccati refinement: SR3-lite sparse refinement + rational snapping (`diff_pade.rs`, 2026-10-16):** the differential engine now iteratively re-solves the sparse support (`refine_sparse_system`, hold-fixed STLSQ) and rationalizes coefficients (`snap_rational`, denominators <= 16, 1% tolerance), each behind revert guards on BOTH trajectory R2 and jet RMSE fidelity. New `snapped_terms` result field. The Planck Riccati relation is now recovered EXACTLY: `x^3 y' + (x^3 - 3x^2) y + y^2 = 0` (traj R2 = 0.999999999759, jet RMSE = 8.597e-07, snapped; previously a cubic linear fit with spurious terms at R2 = 0.999965 / RMSE = 0.004906). This resolves the 2026-10-15 "recorded follow-up". Pinned by `test_planck_riccati_ode_recovery` (now asserts exact) + `tests/test_dyn_frontier_9_of_10.py`.
- **Exact-recovery gate fix (integrity, 2026-10-16):** the old `is_exact = traj > 0.99 && jet < 0.05` gate mislabeled fitted approximations as exact once refinement improved them (demonstrated: cubic-linear Planck fit at jet ~1e-4 flagged exact, contradicting the Hermite non-existence proof). New gate: `traj > 0.99 && jet < 0.05 && (snapped_terms || jet < 1e-9)` — exactness now requires structural evidence (rationalized coefficients) or machine precision. Regression pins added: Rust `test_planck_cubic_linear_fit_must_not_claim_exact` + `tests/test_dyn_planck_limitation.py::test_planck_cubic_linear_fit_must_not_claim_exact`; the fresh-domain linear-transfer threshold was updated 0.9 -> 0.99 (measured 0.938) with the abstention pin retained.
- **Frontier credibility fix (`docs/frontier_9_of_10_architectural_resolution.md`, 2026-10-16):** the referenced artifact did not exist on disk (AGENTS.md 1.4 violation); created with machine-regenerated numbers and explicit corrections (Burgers diffusion error 11.2%; heat holdout is a re-fit, not transfer; TV plateau error 1.27%). `DOCUMENTATION.md` PDE engine entry corrected `production` -> `experimental`. New `tests/test_dyn_frontier_claims_integrity.py` (6/6) regenerates every headline number and asserts the artifact states them.
- **Native GA constant polish (`genetic.rs` `polish_candidate_constants`, 2026-10-16):** post-evolution coordinate-descent stage over inner `Const` leaves via golden-section line search (`span = max(|v|, 0.5)` shrinking 0.4/sweep, 8 sweeps in-loop, affine readout refit at every probe; strict-improvement replacements only, so objective-flat constants -- e.g. a lone global multiplier absorbed by the readout -- are left untouched) plus a Powell-style conjugate-direction line search along each sweep's net displacement, which escapes coupled valleys where every single-axis move is locally non-improving (sin(f*x+p) frequency/phase: 9.1e-4 -> 1.3e-6). Config: `polish_constants` (default true) + `polish_top_k` (default 3, top candidates per island), threaded through the `evolve_native` binding and `RustKernelBridge.evolve_native`. Rust mechanism pins: sin(c*x) refines c = 1.8 -> 2.000 with >100x MSE collapse; joint (frequency, phase) converges; constant-free trees cost exactly one eval and are returned structurally identical; same-seed A/B never worsens the pooled best while counting polish evals. Python pins: `tests/test_dyn_native_ga_constant_polish.py` (3/3; never-worsens, top_k=0 == disabled, bridge flag forwarding, x^2 recovery unaffected).
- **GA trig frequency/phase template seeding (`genetic.rs` `trig_template`, 2026-10-16):** top-level random trees now seed the `sin|cos(f*x + p)` template (rate 0.18, gated on `allow_trans` + variables + depth >= 3) with `f ~ U(0.3, 2.8)`, `p ~ U(-pi, pi)` (35% phase-free), directly addressing the frozen N04 forensics (uniform depth-4 trees could not express a phase-shifted oscillation; best fit R2 = 0.04). Measured at default budget (50k-eval class): `sin(2x)`, `2*cos(1.3x)`, `sin(1.5x)` recover EXACTLY (MSE = 0); `3*sin(1.7x+0.4)` reaches the correct family with frequency/phase resolved to ~5e-3 (MSE 1.3e-6) via the Powell-coupled polish; `2.5*exp(-0.5x)` recovers exactly via `2.5*sqrt(exp(-x))` (MSE 5e-32). Pinned by `tests/test_dyn_native_ga_trig_templates.py` (3/3) + Rust `genetic.rs::trig_seed_tests` (2/2; seeding rate band, flag gating). `tests/test_dyn_rust_domain_ops.py::test_native_ga_discovers_advanced_ops` re-pinned numerically (general-Pow path finds exact `x**2.5`, train MSE ~1e-17) instead of the old sqrt-spelling assert; `atan(x)` recovery unchanged.
- **Frozen-coefficient PDE transfer evaluator (`apedp_dyn/epistemic/pde_transfer.py`, 2026-10-16):** the dev/holdout protocol in `pde_discovery.discover_pde` re-runs STRidge on the holdout partition, so its `holdout_r2` is a re-fit metric (documented limitation #1). New `evaluate_frozen_transfer(u_hold, dt, dx, active_terms, coefficients)` scores dev-discovered coefficients with NO refit, using the identical 2nd-order time / 4th-order space central stencils, interior mask, and R2 definition (vs mean u_t) as `rime/pde.rs`; unknown term names raise (never silently dropped). Measured on the Rust Burgers fixture (u_t = -u*u_x + 0.1*u_xx): dev coefficients (-0.9962, 0.1112), frozen holdout R2 = 0.999772 vs re-fit 0.999909 -- the honest transfer number and the measured re-fit inflation are now both on record. Pinned by `tests/test_dyn_pde_transfer.py` (4/4; analytic heat exactness, wrong-coefficient penalty, Burgers partition survival, unknown-term rejection). Wiring into the `discover_pde` adapter itself is deferred to coordinate with the concurrent pde.rs change set.
- **RIME-AD native external-corpus adapter (`benchmarks/external_60_adapters.py::RimeADNativeAdapter`, 2026-10-16):** contract-compliant adapter running the native engine (`evolve_native`) on the frozen 60-problem corpus public grids. The frozen round compared the full `AutonomousModelSynthesizer` stack, so hidden-truth scoring of this engine is a NEW round under protocol section 6.3 (own preregistration/seed/checksums) and is explicitly not performed here; the adapter never reads `hidden/`, abstains on eval failure without shipping numbers, and is pinned by `tests/test_external_60_native_adapter.py` (2/2, including a run against a corpus copy with no hidden directory). Frozen `tournament_results_external_60/` untouched.
- **Reproducibility targets (`Makefile`, 2026-10-16):** `make kernel` (maturin develop), `make wheel` (distributable wheel to `dist/`), `make reproduce` (builds the kernel, then regenerates and verifies every headline number a reader would quote: frontier claims integrity, Planck/Riccati claims, canonical ledger, the frozen external round, trig recovery, PDE frozen transfer -- 30 tests, green). Root `Cargo.lock` was already tracked and pins the kernel's dependency tree.
- **OpenAI-math sandbox (`benchmarks/openai_math_sandbox/`, 2026-10-16, `audit-only`):** mirror of the OpenAI 2026-10-06 math-release pipeline (attempt -> publish claim -> independent verification -> retraction accounting; source release: 722 manuscripts, 3 retractions, 372 result families, 719 listed) applied to APEDP-X. Registry `problems.json` holds 6 supported/conditional machine-checkable tasks and 8 explicit `unsupported` gap entries (Lean formalization, quasi-RH, BSD, Hodge, Kakeya R^4, asymptotic algorithmics, Navier-Stokes, 719-manuscript scale), each with a reason. Measured 2026-10-16: attempted 6, verified 6, failed 0, unavailable 0, unsupported 8 -- Goldbach [4, 2e6] 999,999 evens 0 violations (ceiling C2, universal claim abstains), Riemann 13 zeros from scratch calibrated at 1.15e-9 (tally APPROXIMATE_TALLY_MATCH, S(T) uncertified), Planck Riccati jet 8.60e-07, harmonic jet 6.11e-14, trig MSE 0.0 both targets, Burgers frozen holdout R2 0.999772. Runner enforces in code that no task is ever `verified` without a checker certificate (missed threshold = `failed` = retraction count); README carries the gap analysis. Pinned by `tests/test_openai_math_sandbox.py` (6/6). Companion hardening: `tests/test_dyn_pde_transfer.py` now skips with a reason when the pending `rime_pde_discover_json` binding is absent, so a clean checkout of the commit that shipped `pde_transfer.py` stays green instead of erroring.
- **Full openai/math catalog ingestion (`benchmarks/openai_math_sandbox/ingest_openai_catalog.py` + `catalog/`, 2026-10-16, `audit-only`):** ingests ALL 372 result families / 719 manuscripts from a SHA-256-pinned snapshot of `openai/math` (CONTENTS.md, history.md, overview.pdf, formalization.yaml; `catalog/source_snapshot/`), joins the Lean catalogue, and records the accounting: 3 named withdrawals (722 -> 719; Weil-classes sign error cascading to two dependents), 14 revised + 13 citation-updated, historical family numbering 001-377 with 5 retired numbers, and 5 explicit discrepancies kept on record -- including that the repo-stated 300/719 (~42%) formalization count does not reconcile with formalization.yaml's 173 source papers / 200 main-result declarations (both numbers reported, delta unresolved) and that the yaml is `unchecked`/`agent`-produced (not an independent check; APEDP-X does not run Lean). APEDP-X verdict at catalog scale: 372/372 families `uncheckable_proof_level` (no proof engine; probe tags for 40 finite-numerics and 35 explicit-construction families are illustration only). Pinned by `tests/test_openai_math_catalog.py` (7/7) against the committed snapshot hashes.

- **`apedp_dyn/epistemic/prior_guided_search.py` — the loop-closer.** Consumes the shape
  prior and dataset profile on DEV rows only and selects a search path through a FIXED
  table (stepped -> step basis + extrapolatable rule form; singular/power -> transmutation
  fitter; else abstain). Emits `CandidateProposal` objects; never promotes, never sees
  holdout. No operator-name literals usable for per-equation selection (machine-guarded).
  Owns a deterministic two-stage floor-rule fit (global log-width grid + max-margin
  tie-break + local refine): max-margin phasing was REQUIRED, measured, because several
  (w, off) classify dev rows identically (SSE 1.5e-29 tie) while holdout separates them
  (0.974 vs 1.0).
- **`apedp_dyn/epistemic/report_fidelity.py` — verdicts derive from rows.** `verdict_for_row`
  with explicit args (tau_extrapolation, rho_max, eps_exact_rmse, delta_exact_r2):
  exact-recovery bypass checked FIRST (rho1 of numerical noise must not refuse R2 == 1),
  then gate, then stated refusal reasons. The Landauer harness template now renders this
  function's output verbatim; hardcoded `**PASSED** (Exact Recovery)` / `**REFUSED**`
  labels were removed from `benchmarks/run_landauer_rime_vs_ga_benchmark.py` (4 lines +
  import; NaN rho1 now renders `n/a (exact-zero residuals)` instead of `0.0000`).
- **`tests/test_dyn_prior_guided_unsolicited.py` (7/7 green).** Unsolicited Landauer recovery
  on sealed splits (dev [0.1,4.9] -> holdout [5.0,9.8], R2 == 1.0 both, zero human primitive
  choice); table truth across staircase/singular/power/line (line abstains); noise rung 1
  (2%: holdout >= 0.999); rung 2 labeled semi-synthetic (BTC 5m + injected W=125/OFF=40
  staircase at local SNR=5: width 125.02/125, quantum within 0.4%, offset modulo width,
  holdout 0.8910 vs oracle 0.8917); clean-stream abstention control; low-SNR boundary
  (SNR<=4 abstains) pinned, not hidden; single-threshold edge states its no-holdout limit.
- **`tests/test_dyn_landauer_report_fidelity.py` (4/4 green).** Verdict-rule units, committed
  md == committed JSON cell-for-cell on the main table, no-hardcode guard on the harness,
  end-to-end rendered-cell presence.

### Measured integration gap found and fixed (kept on record)
- The transmutation fitter DECLINES stepped data under noise that the shape prior
  identifies (its level-clustering rule needs <=20 discrete rounded levels; noisy returns
  exceed that). The adapter therefore owns a uniform row-1 floor-fit path instead of
  depending on the other module's internal gating. Two components disagreeing on the same
  data is exactly what the closed loop had to resolve, and did.

### Honest scope restated
- Rung 2 is semi-synthetic by necessity: the repo has no real dataset with known truth and
  >=40 rows (Bertozzi 5, Brown 20). The injection (form, SNR, oracle) is fully labeled in
  the test. Unsolicited discovery on real measured unknowns remains unclaimed.

### Reporting integrity
- `RIME_SUITE_TOTAL` raised 200 -> 211 (whitelisting the two new files), verified by
  re-running `tests/test_dyn_reporting_integrity.py`.

## [Unreleased] — RIME shape inference steps 1-2 (data-driven family prior + step basis, `experimental`)

### Added
- **`apedp_dyn/epistemic/shape_inference.py` — data-driven form-family prior.** The point is to
  stop supplying structure by hand and let the data choose the family.
  - Exact O(k·n²) dynamic-programming segmentation; segment count from an explicit BIC
    argument (no module-level thresholds, per AGENTS.md §1.3).
  - Family verdict among `piecewise_constant` / `smooth` / `no_structure`, decided by how
    many PARAMETERS each family needs to reach a common fit target (a shared
    achievable×(1−sse_target) R²), with the flat and sloped families costed per parameter.
  - `fit_step_family` / `step_family_basis` emit a candidate step form
    `c₀ + Σ cᵢ·sign(x − tᵢ)` with data-estimated thresholds and OLS coefficients. This form
    needs NO new operator node: `Sign`, `Add`, `Const`, `Mul` already exist in the kernel.
  - `staircase_prior(x, y)` chains the two and returns `None` for the seeds whenever the prior
    does not support a step family, so a curve is never handed one.
- **`tests/test_dyn_shape_inference.py` (15/15 green, fixed seeds).** Acceptance is bidirectional:
  - Landauer staircase recovered EXACTLY: R² 1.0, coefficients (2.5, 0.5×5) matching the true
    generating form, 5 change-points estimated within half the local bin gap of truth 1..5.
  - 2% noise still identifies it (R² 0.9997); 0–10% noise all classify `piecewise_constant`.
  - Straight line → `smooth`, 0 change-points, and `staircase_prior` returns no seeds.
  - Saturating Padé curve → `smooth`, 0 change-points. iid noise → `smooth`. BTC forward
    returns (1094 rows) → `no_structure`. No fabricated steps on real market data.
  - Step count is data-estimated: 2-step staircase → 2 change-points, 5-step → 5.
  - <40 rows → `no_structure`.

### Recorded measurement failures (kept on the record, not hidden)
Four statistics were implemented and REJECTED by measurement, each in both directions:
1. **"Piecewise-constant beats a line"** — flags iid noise (+0.14 R²) and a saturating curve
   (+0.63) as stepped while granting only +0.027 on the real staircase, because a line already
   fits a ramp of steps at R² 0.973.
2. **Flat-run fraction on the p90 |Δy| scale** — collapses under noise (0.975 → 0.065 at 1%),
   because >90% flat steps put the 90th percentile inside the plateau, so the tolerance is set
   by noise instead of the jump.
3. **Within-segment drift ratio** — mislabels a COARSE segmentation of a CURVE as plateaus
   (0.70 on a noisy Padé curve, 0.98 on iid noise), because a strongly curved segment looks
   plateau-like.
4. **Separately-BIC-selected k per family** — degenerates, because a piecewise-constant model
   with enough segments drives SSE to exactly 0 and BIC's log-SSE term rewards that without
   bound (a clean straight line was assigned 32 constant segments).

Also fixed on the way: a half-bin threshold bias (DP boundaries are indices, so
emitting `xs[j]` as the edge put every sign flip one half-bin late — R² 0.990 instead of
1.000000), and an inverted sign on the parsimony gap.

### Critical negative control (why the gate, not the R², is the evidence)
`fit_step_family` applied to a STRAIGHT LINE with forced thresholds returns R² 0.979 and
`identifiable=True`, because 6 constants approximate a line over a wide range. So a high R²
from this basis is NOT admissible evidence of steps: only the independent shape prior's
verdict makes it admissible. `staircase_prior` therefore declines to emit seeds on a line,
and the test asserts the raw R² would have been misleading.

### Stated limitations
- Step 2 is still a PROPOSAL. The step basis is not yet fed to the kernel search, so
  multi-term composition and the kernel-side wiring remain unbuilt. No claim of unsolicited
  end-to-end discovery is made.
- Measured boundary: at 20% noise the staircase no longer classifies as
  `piecewise_constant` within `max_segments=32` — the cap, not the statistic, is the limit.
  Not tuned to green; recorded as a boundary.

### Reporting integrity
- `RIME_SUITE_TOTAL` raised 193 → 200 (whitelisting this file), verified by re-running
  `tests/test_dyn_reporting_integrity.py`.

## [Unreleased] — RIME Repair v0.1.x freeze (comparative-validation milestone)

### Added (v0.2.5 round — Autonomous Epistemic Profiler, Coordinate Transmuter & Frontier Quantum Equations, `experimental`)
- **`apedp_dyn/epistemic/autonomous_profiler.py` — Autonomous Diagnostic Pre-Scan:**
  - Profiles target data topologies without human domain prompting:
    - Dynamic range $\mathcal{R}_y = \max(y)/\min(y) > 100 \implies$ selects scale-invariant `"log_mse"`.
    - Arrhenius tunneling singularity detector: $\frac{d \ln y}{d(1/x)} \approx \text{const} < 0$.
    - Power-law scaling detector: $\frac{d \ln y}{d \ln x} \approx \text{const}$.
    - Quantized plateau staircase scanner: flat derivative fraction $> 40\%$ and discrete clustering.
    - Dual-asymptotic crossover scanner: low-$x$ vs. high-$x$ boundary power divergence (Rayleigh-Jeans vs. Wien).
  - Emits immutable `DatasetProfile` with deterministic SHA-256 hash.
- **`apedp_dyn/epistemic/coordinate_transmuter.py` — Manifold Linearization & Parameter Recovery:**
  - Projects non-linear physical manifolds with essential singularities or power laws onto linearized basis spaces:
    - Log-linear singular basis: $\ln y = \beta_0 + \beta_1 \ln x + \beta_2 (1/x)$ recovers $y = C x^k e^{-c/x}$.
    - Power-law basis: $\ln y = \beta_0 + \beta_1 \ln x$ recovers $y = C x^k$.
    - Quantized staircase basis: midpoint linear fit recovers step height $G_0$ and step width $\Delta w$.
  - Emits valid `CandidateProposal` objects with closures compliant with `AGENTS.md` §1.1.
- **Discovery Verified on 3 Frontier Quantum Equations (`tests/test_dyn_autonomous_epistemic_profiler.py`):**
  - **Landau-Zener Non-Adiabatic Transition:** $P_{\mathrm{LZ}} = \exp(-\alpha / v)$ discovered with $R^2 = 1.000000$, $\alpha = 0.75$, power $k = 0.0$.
  - **BCS Superconducting Weak-Coupling Gap:** $\Delta(g) = 3.52 \exp(-1/g)$ non-perturbative essential singularity discovered with $R^2 = 1.000000$.
  - **Integer Quantum Hall Effect Conductance:** $\sigma_{xy} = G_0 \lfloor 1/B \rfloor$ discrete plateau staircase discovered with $R^2 = 1.000000$.

### Added (v0.2.4 round — RIME Quantum Equation Capabilities & Limitation Remediations, `experimental`)
- **Limitation 1 Remediated: Negative Powers in Brus Quantum Confinement (`rust/apedp-kernel`, `RimeProposer`):**
  - Added negative rational exponents $x^{-1}$ and $x^{-2}$ to `rust/apedp-kernel/src/rime/seeds.rs`, added `(-1, 1)` and `(-2, 1)` to `PERMITTED_POWERS` in `rust/apedp-kernel/src/rime/rewrite.rs`, and fixed `Node::Power` sign handling in `rust/apedp-kernel/src/rime/graph.rs` to guarantee even negative powers evaluate strictly positive.
  - Added algebraic transforms `inv` ($1/x$), `inv_sq` ($1/x^2$), and `lor_norm` ($1/(1+(x/s)^2)$) to `_ALGEBRAIC_TRANSFORMS` in `apedp_dyn/proposers/rime_proposer.py`.
  - On Brus quantum dot confinement ($E_g(R) = 1.74 + 2.45 R^{-2} - 0.28 R^{-1}$), native RIME directly recovers negative power scaling laws ($+2.03553 (R^{-2}) + 1.69599$), eliminating bounded proxy $\tanh$ extrapolation collapse and raising extrapolation $R^2$ from $-70.27$ to $+0.9959$.
- **Limitation 2 Remediated: Continuous Denominator Scale & Lorentzian Resonance (`R5_PADE_SATURATION`, `rime_repair_engine.py`):**
  - Expanded `RepairOperator.R5_PADE_SATURATION` in `apedp_dyn/epistemic/rime_repair_engine.py` to fit both Form 1 (rising saturation $a \cdot z / (1 + b |z|)$) and Form 2 (Lorentzian resonance / lineshape $a / (1 + b |z|)$).
  - Added `resonance: bool` parameter to `_wrap_pade_repaired` with covariance condition ($< 5000$) and asymptotic parameter uncertainty ($< 5\%$) identifiability diagnostics.
  - On Detuned Rabi oscillations ($P_{\max}(\Delta) = \frac{\Omega_R^2}{\Omega_R^2 + \Delta^2} = \frac{1}{1 + 0.04 \Delta^2}$), accurately recovers $a = 1.000$, $b = 0.0400 \implies \Omega_R^2 = 25.00$ on clean data ($R^2 = 1.000000$) and under 2% noise recovers $\Omega_R^2 = 24.81$ (0.76% parameter error, holdout $R^2 = 0.999968$, `identifiability_status: IDENTIFIED_STABLE`).
- **Limitation 3 Remediated: Harmonic Frequency Multipliers in Rabi Pulse Dynamics (`rust/apedp-kernel`, `RimeProposer`):**
  - Added `harmonic_seeds: bool` parameter to `SearchConfig`, `small_forms_extended`, and `RimeProposer`.
  - Emits $\cos(x)$, $\sin(x)$, $\cos(x+x) = \cos(2x)$, and $\sin(x+x) = \sin(2x)$ at Depth 0 when `harmonic_seeds` is enabled.
  - Added `cos2` ($\cos(2x)$) and `sin2` ($\sin(2x)$) STEM transforms to `_STEM_TRANSFORMS` in `RimeProposer`.
  - On resonant Rabi pulse dynamics ($P_e(t) = \sin^2(t) = \frac{1 - \cos(2t)}{2}$), native RIME natively discovers `-0.5 * (cos((t + t))) + 0.5` with clean holdout $R^2 = 1.000000$ and 2% noisy holdout $R^2 = 0.999992$.
- **Verification & Pinned Regression Suite (`tests/test_dyn_rime_quantum_equation_fixes.py`, 3/3 green):**
  - Covers negative powers, Lorentzian lineshapes, and harmonic frequency multipliers under clean and noisy regimes. All 58 canonical AGENTS.md tests and all 103 Rust kernel tests pass cleanly.

### Added (v0.2.3 round — RIME native resolution of Seed Catalog Exhaustion & Grammar Constraint, `experimental`)
- **Resolution of Seed Catalog Exhaustion (`rust/apedp-kernel/src/rime/seeds.rs`, `SearchConfig.max_seeds`, `SearchConfig.lagged_products`):**
  - Overcame the fixed 60-seed catalogue cap with configurable `max_seeds` budget parameter.
  - Added Family 6b contemporaneous lagged interaction products `(x_i[t-k] * x_j[t-k])` for $k \in \{1, 2, 5\}$ (`lagged_products: bool = false`).
  - Preserved 100% backward compatibility for all existing callers: default `small_forms_with_all_options` delegates to `small_forms_extended(..., None, false)` maintaining byte-identical catalogue output.
- **Resolution of Grammar Constraint via Native Multi-Term Additive Search (`rust/apedp-kernel/src/rime/additive.rs`, `SearchConfig.additive_terms`, `RimeProposer(additive_terms=...)`):**
  - Wired native Rust forward stepwise OLS engine (`search_additive`) into PyO3 entry points (`rime_search_json`, `rime_search_envs_json`) and Python `RimeProposer`.
  - Performs joint least-squares parameter refitting on common rows, honestly charging each added term for structural edges, regression coefficients, and Bonferroni selection multiplicity ($\lambda_c \cdot (\text{edges} + 1\text{ coefficient} + 2 \ln M)$).
  - Composes winning multi-term expressions into unified executable `Graph` objects evaluated seamlessly in both Rust and Python `compute_score`.
- **Benchmark Equivalence & Parity with GA-Large (`tests/test_dyn_rime_native_additive_and_seeds.py`, 4/4 green):**
  - On the composite population extrapolation benchmark ($y[t] = 2 x_1[t-1] x_2[t-1] + 3 x_3[t-2] + 10$), `RimeProposer(additive_terms=2, lagged_products=True, max_seeds=120)` natively recovers the composite equation with exact structural terms and true parameter values ($2.996 \cdot x_3[t-2] + 1.997 \cdot (x_1[t-1] \cdot x_2[t-1]) + 9.999$).
  - Delivers transfer parity with GA-Large across populations without requiring post-hoc referee repair:
    - Population A (test holdout): $R^2 = 0.999820$ (GA-Large: $0.9998$)
    - Population B (fresh seeds): $R^2 = 0.999767$ (GA-Large: $0.9998$)
    - Population C (distribution shift): $R^2 = 0.895609$ (GA-Large: $0.8961$)
  - Confirmed 7-distractor noise filtering ($R^2 > 0.9999$) and parsimonious single-term retention on pure single-term data.

### Added (v0.2.2 round — rational parameter identifiability & decoupled metrics, `experimental`)
- **Rational Parameter Reconciliation & R5 Scaling Audit:** Reconciled mathematical parameter convention in `R5_PADE_SATURATION` (`apedp_dyn/epistemic/rime_repair_engine.py`). For $y = \frac{1000 p^2}{1 + 2000 p^2}$, explicitly established that when fitting $\text{pred} = \frac{a \cdot z}{1 + b \cdot |z|}$ against un-saturated parent model $z = k_{\text{fit}} \cdot p^2$, the physical numerator scale is $a \cdot k_{\text{fit}} = 1000.00$, the physical denominator scale is $b \cdot k_{\text{fit}} = 2000.00$, and the saturation envelope ratio is $\alpha = b/a = 2.0000$. Resolved prior documentation ambiguity conflating the dimensionless envelope ratio $\alpha=2.0$ with the raw physical denominator coefficient $c_{\text{den}}=2000$.
- **Identifiability & Uncertainty Diagnostics in R5:** Added covariance matrix conditioning analysis (`cov_condition = np.linalg.cond(pcov)`), asymptotic parameter standard errors (`a_stderr`, `b_stderr`), boundary-hit flags, and explicit `identifiability_status` (`IDENTIFIED_STABLE` vs `PREDICTIVE_ONLY_ILL_CONDITIONED`). On narrow/weakly-saturated domains ($p \in [0.001, 0.008]$), flags ill-conditioning when denominator scale drifts (6.5% error) even if holdout $R^2 > 0.999$.
- **Decoupled 5-Metric Evaluation Schema:** Implemented independent scoring in candidate proposal metadata and parameters: `structure_match`, `parameter_recovery`, `prediction_score` ($R^2$, RMSE), `extrapolation_score`, and `identifiability_status`. High predictive fit ($R^2 > 0.99$) is strictly decoupled from mechanism or parameter identification.
- **Negative Controls & Skepticism Gates:** Verified against pure quadratic ($y = 1000 p^2 \implies b \to 0$, no false saturation) and exponential saturation ($y = 0.5(1 - e^{-u/0.5})$ where Padé achieves $R^2 > 0.999$, proving high fit alone does not certify rational mechanism).
- **Pinned Verification Suite (`tests/test_dyn_rime_rational_parameter_identifiability.py`, 5/5 green):** Covers exact recovery, noise robustness ($0.5\%, 2\%, 5\%$), domain range breakdown, negative controls, and decoupled reporting. All 74 canonical and dynamic regression tests pass cleanly.

### Added (v0.2.1 round — flag calibration, staged search, layered FPR, `experimental`)

**0. Null-calibrated whiteness flag (`apedp_dyn/epistemic/whiteness_calibration.py`)** — Guard 2's `|rho1|` flag is derived from the target's own serial dependence by block-demeaning (contiguous blocks resampled then block-demeaned, so the null inherits the target's autocorrelation while level-only fits score ~0 and cannot launder AR residuals through an offset). Replaces the fixed magic 0.5, which miscalibrated in both directions. Measured both fixes: on an AR(2) target a residual with |rho1|=0.723 is refused by the frozen 0.5 and accepted by the calibrated 0.858 (over-refusal fixed); on a white target a residual with |rho1|=0.273 is admitted by the frozen 0.5 and refused by the calibrated 0.226 (under-refusal fixed). Flag is monotone in target persistence (0.114 white → 0.858 AR(2)). Ships `screen_structured_null_worlds`, which reports proposer-level and system-level FPR **separately**: 12 proposer promotions vs 0 system promotions on 24 structured null worlds. Lives in its own module because `adjudicate.py` is a hash-pinned frozen record.
  - **Governance note (recorded, not concealed):** an initial attempt placed the calibration inside `adjudicate.py`. It was rejected by the repo's own freeze-drift guard (`FreezeDriftError` / `REFEREE OR FROZEN RECORD CHANGED`) — correct behaviour — and reverted; the capability ships as an opt-in adopter module. When no holdout target is recorded, the resolver keeps the frozen absolute floor and stamps `absolute_fallback_no_target` in the audit record, because an uncalibrated flag must remain visible as uncalibrated.

**0. Staged search + coverage/budget auditing (`apedp_dyn/epistemic/rime_staged_search.py`)** — two-stage search (cheap family-admit pass at a tiny budget, then full-budget expansion restricted to admitted families) plus `coverage_curve`, which makes coverage a measured quantity instead of an assumption. Measured budget curve: recovery saturates at 176 **evaluated** with identical best R² 0.9616 across budgets 96→2000 while reached families drop 3→2 — i.e. depth/budget claims beyond the cap are budget-bound, now pinned rather than argued. Expanded grammar (all ops ON) spends 200 vs 176 evaluated for the same winning family and the same score: expansion buys reachability, not recall. Staged run is never worse than the straight full-budget run. Structured-null FPR at the new operator count is unchanged at the system level (proposer 4/12 worlds, system 0).

### Added (v0.2.0 round — RIME structural flaw remediation & epistemic robustness, `experimental`)
- **Flaw 1 (Grammar Poverty) Remediated:** Native rational power graph node `Node::Power { src, p_num, p_den }` and structural rewrite op `RewriteOp::AddPower` added to `rust/apedp-kernel/src/rime/`. Admits non-integer rational powers $p \in \{\pm \frac{1}{3}, \pm \frac{1}{2}, \frac{2}{3}, 2, 3\}$ into seed catalogue and structural rewrites (`power_ops=True`). Features pole-safe float evaluation and sign preservation for odd roots. Recovers nuclear surface scaling $A^{2/3}$ with exact $R^2 = 1.000000$ to machine precision without manual feature engineering.
- **Flaw 4 (Small-Sample Paralysis) Remediated:** Configurable `min_rows: int = 30` parameter in `RimeProposer`. Preserves frozen default $N \ge 30$ threshold for noisy financial time-series to prevent overfitting, while allowing high-precision experimental physics datasets (e.g. Bertozzi 1964 $N=5$, $\sigma < 0.1\%$) to execute without unconditional early exits.
- **Flaw 2 (Correlated Proxy Trap) Remediated:** Identifiability banding (`identifiability_band: float = 0.002`) and empirical derivative gradient matching (`check_derivatives: bool = False`) added to proposal assembly in `RimeProposer`. Prevents false-certainty promotion when collinear hypotheses differ by $\Delta R^2 \le 0.002$, reporting `AMBIGUOUS_EQUIVALENCE_SET` and recording `competing_hypotheses` and `equivalence_band_size`. Flags `UNIQUE_WINNER` only when lead margin strictly exceeds the identifiability band.
- **Flaw 3 (Greedy Beam Monoculture) Remediated:** Diverse beam frontier selection (`SearchConfig.diverse_beam: bool = false`) added to native Rust beam search. Ensures round-robin representation across structural archetypes (polynomial, power, rational, bounded, transcendental) in the beam frontier, eliminating greedy beam starvation of non-linear candidates.
- **Verification & Pinned Test Suite (`tests/test_dyn_rime_v02_flaw_remediation.py`, 5/5 green):** Covers exact rational power recovery, small-sample laboratory mode on Bertozzi data, identifiability banding, gradient alignment, and diverse beam frontier. All 58 canonical AGENTS.md tests, all 103 Rust unit tests, and all 33 RIME/STEM regression tests pass cleanly.

### Added (v0.1.13 round — preregistered real-measurement benchmark suite, `experimental`)
- **Dataset 1: AME2020 Real Nuclear Binding Energies (1,159 real nuclides, `data/mass_1.mas20.txt`):** RIME with declared `square_seeds=True` natively discovers Coulomb electrostatic repulsion $Z^2$ and interaction $A \cdot Z$ on binding energy per nucleon without handed columns or rational powers, achieving holdout $R^2 = 0.7049$ (single-term) and $R^2 = 0.7705$ (2-term additive) on 559 unseen nuclides. Evaluated against literature Weizsäcker SEMF which achieves holdout $R^2 = 0.9921$ with physically calibrated parameters ($a_v = 15.52$, $a_s = 17.07$, $a_c = 0.698$, $a_a = 22.40$ MeV).
- **Dataset 2: Bertozzi 1964 Relativistic Electron Experiment (`data/bertozzi_1964.jsonl`):** Experimental linear accelerator measurements of electron kinetic energy $K$ vs speed $v/c$. Epistemic Referee adjudicates Classical Newtonian mechanics vs Relativistic Einstein mechanics on sealed high-energy holdout ($K \in \{4.5, 15.0\}$ MeV). Classical mechanics catastrophically fails ($v = 7.7c$, $\text{RMSE} = 4.88c$), while Relativistic mechanics achieves holdout $\text{RMSE} = 0.0055c$ ($R^2 = 0.9998$) and is promoted with `UNIQUE_SUPPORT` (zero false promotion of superluminal classical physics).
- **Dataset 3: Real Market Orderflow & Daily Asset Returns (Noise / Null Negative Control):** Evaluates factor discovery across 22 real microstructure signals on S&P 500 calm bull and COVID-19 crash regimes. Epistemic Referee strictly refuses false promotion: $\text{FPR} = 0.00\%$, 0 false promotions, $0.0\%$ capital allocated.
- **Pinned verification (`tests/test_dyn_preregistered_real_data_benchmark.py`, 3 pins):** All 3 real-data tests pass cleanly with zero regressions across the canonical suite (58/58 green).

### Added (v0.1.12 round — full STEM grammar expansion: options 1, 2, 3, `experimental`)
- **Option 1: Native Rust Kernel STEM Primitives & Python Adapter (`SearchConfig.stem_ops`, default `false`):** adds elementary scalar STEM primitives `Node::Sin`, `Node::Cos`, `Node::SafeExp` (clamped to $[-60, 60]$ preventing float overflow), `Node::SafeLog` ($\ln(1 + |x|)$ eliminating negative/pole hazards), and `Node::Diff` ($\Delta x_t = x_t - x_{t-1}$) to `rust/apedp-kernel/src/rime/graph.rs`, `rewrite.rs`, `search.rs`, `seeds.rs`, and `lib.rs`. Exposed via `RimeProposer(stem_ops=...)` and expanded STEM transforms (`sin`, `cos`, `exp`, `log`, `diff`) in `_ALL_TRANSFORMS` while preserving `_ALGEBRAIC_TRANSFORMS` byte-identity.
- **Option 2: Modular Multi-Engine STEM Proposers (`apedp_dyn/proposers/stem_proposers.py`, `experimental`):**
  - `HamiltonianProposer`: Discovers energy-conserving mechanical oscillator dynamics ($\dot{q} = p/m$, $\dot{p} = -k q - \alpha q^3$), validating energy conservation $\frac{dH}{dt} = 0$, and rejecting dissipative systems.
  - `SolenoidalFluidProposer`: Discovers incompressible 2D velocity fields via stream function $\psi(x, y)$ analytically guaranteeing zero divergence ($\nabla \cdot \mathbf{u} \equiv 0$).
  - `CompartmentalSirProposer`: Discovers epidemiological compartmental dynamics ($S, I, R$) with exact mass/population conservation ($S + I + R = N$).
- **Option 3: Typed Physical Dimensional Homogeneity Engine (`apedp_dyn/epistemic/dimensional_homogeneity.py`, `experimental`):** AST-based recursive dimensional metrology and validation engine leveraging SI base dimensions from `Dimension`. Enforces Buckingham-$\Pi$ dimensional consistency: strict dimension matching for additions/subtractions, rational power exponents, and dimensionless arguments for transcendental functions (`sin`, `cos`, `exp`, `log`, `tanh`). Exposes `DimensionalHomogeneityGate` for epistemic referee adjudication.
- **Verification & Pinned Test Suite (`tests/test_dyn_stem_grammar_options_123.py`, 10/10 green):** complete regression coverage across native primitives, physical proposers, dimensional gates, and negative controls (rejection of unphysical / dissipative / dimensionally inconsistent models). Zero regressions across all suites (58/58 canonical AGENTS.md suite, 27/27 RIME/STEM suite).

### Added (v0.1.11 round — rational operations & Padé saturation, options 1, 2, 3, `experimental`)
- **Option 1: Native Rust SafeDiv rational operations (`rust/apedp-kernel/src/rime/`, `SearchConfig.rational_ops`, `RimeProposer(rational_ops=...)`, default `false`):** adds pole-free `Node::SafeDiv(a, b)` evaluating $x / (1 + |y|)$ to the kernel rewrite operators and search graph. Enables symbolic discovery of Padé rational approximants directly in mechanism space. Achieves $R^2 = 1.0000$ on rational targets like $y = x / (1 + x)$. Zero poles or zero-division hazards on $\mathbb{R}$.
- **Option 2: Epistemic repair engine Padé saturation operator (`RepairOperator.R5_PADE_SATURATION`, `apedp_dyn/epistemic/rime_repair_engine.py`):** bounded non-linear envelope fit $g(z) = \frac{a \cdot z}{1 + b \cdot |z|}$ targeted at `FINDING_CURVATURE_SUSPECT`. Demonstrates machine-zero algebraic exactness on clean plants ($R^2 = 1.0000$, $\text{RMSE} = 3.8 \times 10^{-17}$), and robust $R^2 = 0.9068$ under 2% realistic measurement noise ($R^2 = 0.2024$ under 10% severe distortion).
- **Option 3: Normalized saturating algebraic transforms (`_ALGEBRAIC_TRANSFORMS`, `apedp_dyn/proposers/rime_proposer.py`):** added `sat`, `sat_sq`, `sat_norm`, and `sat_sq_norm`. Normalizes inputs by frozen training standard deviation $\sigma = \text{std}(x)$ so that $(p/\sigma)^2 / (1 + (p/\sigma)^2)$ captures physical saturation at threshold, maintaining stable discovery performance under noise ($R^2 = 0.9753$ clean, $0.9750$ at 2% noise, $0.9752$ at 10% noise).
- **Pinned verification (`tests/test_dyn_rime_rational_and_pade.py`, 6 pins):** native kernel discovery, default grammar preservation, Padé repair candidate generation, sealed holdout repair pipeline, normalized transform discovery, and independent 0%/2%/10% noise calibration table. Zero regressions across the 58-test canonical suite and 118 RIME tests.

### Added (v0.1.10 round — contemporaneous powers & reachability remediation, `experimental`)
- **Opt-in contemporaneous square seeds (`rust/apedp-kernel/src/rime/seeds.rs`, `SearchConfig.square_seeds`, `RimeProposer(square_seeds=...)`, default `false`):** admits same-time self-multiplication `(x_i * x_i)` into seed catalogue family 5 at depth 0. Closes the measured QEC reachability failure (previously 0/1217 candidates were the true contemporaneous law, forcing search to emit lagged proxies or rely on Python repair). Raw `[p]` discovers `p_L = 1000*p^2` directly at depth 0 with holdout R² > 0.9995.
- **Discrimination against grid-proxy fitting (`tests/test_dyn_rime_reachability_and_contemporaneous_qec.py`):** physical law `(p * p)` is 100% reversal-invariant (|R2_fwd − R2_rev| < 1e-12, holdout R² 0.9995), whereas legacy lagged proxy `(p * p[t-5])` error inflates by >10× under row reversal.
- **Systematic curvature screen against wrong-family emissions:** on d=5 (planted cubic law p^3), a quadratic fit `(p * p)` triggers `FINDING_CURVATURE_SUSPECT` in `classify_findings`, whereas on d=3 true quadratic truth curvature is cleared (`no_reliable_structure`).

### Added (v0.1.9 round — relative-gain criterion, rational-power family, identifiability band)
- **Relative SSE-reduction criterion (`RepairConfig.min_sse_reduction`, default 0 = frozen):** G = (SSE_base − SSE_cand)/SSE_base on the same validation rows, reported alongside absolute R² headroom on every gate. Preregistered companion for high-baseline problems where an absolute R² bar fails by construction (measured on AME2020 below). Caller chooses; neither criterion is inferred.
- **Domain-informed rational-power term family (`RepairConfig.rational_powers`, default empty = frozen):** sign-safe x^p terms of declared dependencies appended to the R2 catalogue, enabled ONLY through the explicit config field — Layer-C priors enter declared, never as hidden corrections. Scored by the same gates.
- **Identifiability band (`RepairConfig.equivalence_r2_tol`, default 0 = frozen single-winner behaviour):** passing candidates within the band of the best are observationally indistinguishable, and the run resolves to `AMBIGUOUS` (MULTI_HYPOTHESIS territory) instead of crowning the marginally-highest scorer. Sealed split is never opened on AMBIGUOUS.
- **Decisive real-data measurement (`tests/test_dyn_real_data_ame_repair.py`, 6 pins):** with the physics-motivated family admitted, the AME2020 ranking on validation is proxy (A·Z) 0.99960 > (A·A) 0.99930 > **surface term (A^0.667) 0.99915** > (A^0.333) 0.99898 > (Z·Z) 0.99842. On the sealed split the proxy and the mechanism differ by <1e-3 R². With the band declared the engine returns `AMBIGUOUS` (set size 4) rather than a unique winner — predictive success is reported separately from identification, which is the SEMF's job, not the search's. SEMF literature coefficients still reach R² 0.9999 on the same frame.
- Control-design note recorded: a candidate cap of 10 was measured to be exactly consumed by R1 + the 9 interaction probes, silently excluding the power terms; the pinned run uses cap 14 (caught by an assertion that failed for the right reason).


### Added (v0.1.8 round — REAL-data repair benchmark, AME2020 masses, no planting)
- **`tests/test_dyn_real_data_ame_repair.py` (3 pins):** answers the standing methodological criticism that every demonstrated repair ran on planted synthetic worlds. Runs discovery → diagnosis → bounded repair on genuinely measured data (AME2020 atomic-mass evaluation, `data/mass_1.mas20.txt`, parsed to binding energies; 1159 usable nuclides).
  - Discovery recovers the semi-empirical mass formula **volume term**: `+7426.14*A + 93729.3` at dev R² 0.994.
  - Diagnosis on that original fires `curvature_suspect` (the missing surface term A^(2/3)).
  - **Threshold calibration exposed by real data:** at the preregistered absolute bar (0.02) the engine returns `NO_IMPROVEMENT` — the bar is mis-scaled for a real dataset whose original already explains 99.4% (achievable gain ≈ 0.005). At a scale-appropriate bar (≤0.005) it returns `IMPROVED` with the `(A*Z)` term at sealed R² **0.9995** vs original 0.9947 — the first sealed-validated repair on measured data in this repository. Both outcomes pinned; the frozen default is untouched (a relative-gain criterion is proposed as a preregistered follow-up, not applied unilaterally).
  - Literature reference on the same frame: SEMF with standard coefficients reaches R² 0.9999 (units keV), so the data support the known law — the boundary is grammar reachability (A^(2/3) absent from the transform set) and threshold calibration, not data quality.
  - Honesty note pinned in the test docstring: the added `(A*Z)` is a *correlated proxy* for the Coulomb/surface structure, not the SEMF term — predictive gain without unique identification (MULTI_HYPOTHESIS territory, deferred to the APEDP-X track); nothing is promoted.
  - Control-design bug caught during the round: reusing one `RepairPartitions` across a threshold sweep raised `HoldoutContaminationError` on the second run — the guard working as designed; the sweep now builds fresh partitions per run.


### Added (v0.1.7 round — intraday crypto sweep on fetched minute bars, audit-only)
- **`tools/fetch_intraday_crypto.py` + `data/market/intraday/` (new, clearly-labeled corpus):** Binance public klines, 1000 bars per symbol per interval at 1m/5m/15m/30m/1h, over the crypto legs of the repo's OWN universes (15/30/45 lists from `multi_asset_signals_sandbox.py`; 14/26/39 fetchable — MATIC delisted, MKR/EOS/XMR/FTM/ICX not currently listed). Own PROVENANCE.json; the frozen daily corpus is untouched.
- **Measured (`tests/test_dyn_intraday_universe_sweep.py`, 5 pins):** real basket forward returns abstain in **15/15** universe×interval cells under both seed orders (no grammar-expressible edge at any tested resolution); null worlds 0 proposals; planted lag-1 control recovered at dev R²>0.98 at every interval (pipeline verified on minute bars); per-position coverage on the 39-input frame follows the 60-form arithmetic exactly — `coverage_first` recovers inputs 0–20 (39 levels + 21 lag-1) while 21+ remain generation gaps.
- **Scope limits recorded, not papered over:** Binance has no 45m interval (requested 45-minute frame does not exist as an exchange bar); equity legs unfetchable (Yahoo Finance intraday HTTP 429 at fetch time) so 60/90-asset intraday universes are out of scope; equal bar counts confound interval with calendar span (1000 1m ≈ 17h vs 1000 1h ≈ 42d).
- **Control-design bug caught and fixed during the round (recorded for audit):** the first planted-position control propagated a NaN std into the planted target (one symbol carries an extra timestamp → NaN row), so the finite-row guard abstained — misread as a RIME limitation until isolated by a solo-frame test. Frame construction now `dropna()`s the pivoted bars first.


### Fixed (v0.1.6 round — generation gap closed, opt-in `seed_order="coverage_first"`)
- **Measured gap (v0.1.6 instrumentation):** the legacy input-major family-2 (delayed levels) truncates the 60-form seed catalogue at high input counts and dropped whole inputs' shortest lag — 3/22 positions (H-RV, H-A, H-OBV_MOM) could never be recovered on the 22-signal frame while a solo-frame control recovered the identical dependence at R² 0.96–0.9995.
- **Fix (`rust/apedp-kernel/src/rime/seeds.rs`):** opt-in lag-major family-2 ordering (`SeedOrder::CoverageFirst`, config key `seed_order`, `RimeProposer(seed_order=...)`, default `"legacy"`). Same budget, same families, same forms whenever the legacy order fits inside the budget (Rust-pinned for n≤6); different WHICH forms survive truncation at many inputs. No threshold, gate, or frozen-spec default changed; `static`/`additive` paths unaffected.
- **Re-measured under the new order:** the 3 gap positions recover the planted column at their solo-frame R² (H-RV 0.9937, H-A 0.9995, H-OBV_MOM 0.9597 — previously 0.19 / 0.68 / 0.09); 19/22 exact + those 3 = **22/22 name the planted column** (the only non-exact recoveries remain the 2 collinear substitutes H-OFI→H-CVD, H-STOCH→H-WILLR, a non-identifiability property, not generation). Null-world false promotions **0/50, unchanged**. Pinned by `tests/test_dyn_rime_seed_order_coverage.py` (3) + 3 new Rust catalogue tests (99/99).


### Added (v0.1.6 round — search-funnel instrumentation + per-position coverage, audit-only)
- **Search-funnel counters (`rust/apedp-kernel/src/rime/search.rs`, `SearchOutcome`):** additive-only `rejected_score` / `dropped_dedup` / `dropped_scope` alongside the existing `evaluated` / `accepted`, plus per-proposal `inputs_used` (`Graph::inputs_used()`, output-reachable arena walk). No score, gate, selection or default changed. Invariants pinned: `rejected_score + accepted == evaluated`, `evaluated <= budget`. Passed through to proposal metadata (`inputs_used_names`, counter keys) in `apedp_dyn/proposers/rime_proposer.py`; `rime_usage_log.py` now prefers the structural list over string mining.
- **Per-position coverage challenge (`tests/test_dyn_rime_coverage_funnel.py`):** planted lag-1 dependence at each of the 22 trading-signal positions (y = 2·x[t-1] + 10 + noise, fixed seeds). **Measured: 17/22 exact recovery; 2/22 collinear substitute** (H-OFI→H-CVD, H-STOCH→H-WILLR — in-grammar twins fitting identically, MULTI rationale); **3/22 generation gap** (H-RV, H-A, H-OBV_MOM — the solo-frame control recovers the identical dependence at R² 0.96–0.9995 while the 22-input frame lands at R² 0.09–0.68, proving the loss is seed-catalogue truncation at 22 inputs, not scoring). Null worlds abstain 2/2. This is the reviewer-directed generation-vs-qualification separation: what is never generated (truncation) vs generated-but-not-selected.
- Fixed during this round: a prefix-collision bug in the kernel's equation-name substitution (`"v13"` was corrupted to `"H-CVD3"` by ascending `String::replace`) — replaced with an exact two-pass control-character-placeholder substitution; rendered equations now match structural `inputs_used_names` verbatim (pinned).


### Added (v0.1.5 round — power calibration + additive refinement, `experimental`, kernel level)
- **`rust/apedp-kernel/tests/rime_power_calibration.rs` (`audit-only`):** recovery / promotion / multi-term rates with Wilson 95% intervals, functional recovery on fresh independent series. Measured: single-term recovery follows $n \gtrsim 21/\rho^2$ (near the ≈$19/\rho^2$ Bonferroni limit for ~200 candidates), null 1/120 promoted; 0% recovery at IC 0.02 below $n=15000$ is the information limit, not a proposer defect.
- **`rust/apedp-kernel/src/rime/additive.rs` (`search_additive`) + kernel config key `additive_terms`:** opt-in multi-term refinement after the frozen gate. Closes a measured representation gap (`x[t-1]+x[t-2]` recovered 0/40 in every cell before; up to 40/40 after for two-term laws, with 0/40 spurious extra terms on single-term laws and identical promotion counts everywhere). Added terms pay edges + 1 coefficient + `2 ln M` (pool size); `Const` coefficients are charged explicitly because `edge_count` treats them as free. Default OFF; `search` is untouched; `seeds::shift` became `pub(crate)`. Not wired into the Python `RimeProposer` (needs pool-size multiplicity in the trial ledger first).
- No frozen spec file, threshold or `canonical_preregistered_spec_v1.json` entry was modified.


### Added (v0.1.4 round, engine tag `rime_v0.1.4` — refusal wiring + repeat bounds)
- **Referee-side out-of-grammar scope refusal (`apedp_dyn/epistemic/scope_refusal.py` + `--adjudicated` in `benchmarks/known_truth_challenge.py`, `experimental`, spec §16d):** closes the §16c negative-control gap (scope refusal must come from adjudication). The referee fits a preregistered family-distinct probe pair (exp family, rate grid [1, 50], dev-normalized; log family, raw axis) from dev data ONLY, and adjudicates each emitted candidate pairwise via the standard `adjudicate()` contract; promotion requires UNIQUE_SUPPORT against EVERY probe. Measured: both methods' exp-world spurious emissions refused (rival-favored bootstrap / indistinguishable rival); all 22 in-grammar true recoveries survive un-declined; `out_of_grammar_emitted_promoted` = 0 everywhere measured. Audit transparency: the first adjudicated run promoted the OLS emission — diagnosed as a referee-side probe defect (axis clipped to [0,1], degrading the exponential probe on extrapolating log grids — the recorded OLS extrapolation failure mode in referee form) and fixed BEFORE any citation, with the strict-AND semantics untouched.
- **Repeat bounds (preregistered, `--repeats`):** 5 observation-noise seeds (phase = 23.0 + 1009·seed; same design points, fresh draws; seed 0 byte-reproduces the pinned v0.1.3 frames), full world set, adjudicated. Artifact `benchmarks/known_truth_results_repeats.json`: recovery 1.0 ± 0.0 per method per seed (11/11 in-grammar every seed), per-world holdout R² std ≤ 5e-5, coefficient accuracy stable (|Ĉ−C|/C ≲ 5e-4 RIME), exp-world emissions refused 5/5 with zero wrongful in-grammar refusals (bounds are noise-driven only). Pinned by `tests/test_dyn_rime_known_truth_scope_repeats.py` (7 pins: referee blindness on all planted exponents, true-monomial UNIQUE_SUPPORT + wrong-exponent refusals, adjudicated single-run refusal without §16c regression, 3-seed repeat bounds, seed-rule frame determinism); `tests/test_dyn_rime_known_truth.py` (§16c) unchanged and green.
- **Population-extrapolation extensions (`tests/test_dyn_population_extrapolation_v2.py`, pinned and re-executed this round):** RIME depths 1–6 byte-identical (`x3[t-2]` on every population), +sq transform axis moves nothing, GA basis ladder isolates products (linear+lag / +squares / +products → only products close the gap), repair-mediated RIME seals 0.9998 with B 0.9998 / C 0.8952 transfer matching GA-large end-to-end — no spec change required.

### Added (v0.1.2 static-grammar round, engine tag `rime_v0.1.2`)
- **Matched-compute harness (`benchmarks/matched_compute.py`, audit-only):** equal-budget comparison of RIME (temporal + static grammars), the existing proposers, the algebraic OLS-basis control, and **PySR 2.7 (head-to-head, installed with Julia 1.11.9)** — scored on a preregistered **grammar-coverage axis**: out-of-coverage abstention is correct behaviour, an emitted-but-failed proposal is a failure on any family, and static worlds require holdout **reversal-invariance** so grid-order tracking is scored `index_fit`. Fairness fixes from the measured audit: 5 repeats with un-stably-verdict screening, full environment metadata, per-cell wall-budget utilization (equal allowance ≠ equal utilized compute — RIME uses ~0.3–0.7% of the allowance, DLI ~8%), PySR runs serial-deterministic with a BINDING 2-s timeout (one-time Julia warmup excluded) and lag columns engineered for information parity (its operator set has no time-shift; its attempted coverage excludes input-side IIR laws). Citable conclusion at equal 2-second serial budgets: **complementarity, not dominance** — rime_temporal 4/5 (fails static_log), PySR 3/5 (fails static_log at this budget, out of scope on the state law).
- **Static search mode (kernel `config["static"]`, default OFF):** temporal primitives (lags, memory cells) are removed from seeds AND rewrites — grid-order fitting becomes unrepresentable for order-structured static data. No threshold or gate changes; the default grammar is byte-identical to v0.1.1.
- **Algebraic input transforms (`RimeProposer(algebraic_transforms=[...])`):** canned `sq/cub/ssqrt/slog` transform columns appended at the input layer, recomputed identically inside `compute_score` so proposals stay causally usable on unseen frames; NaN-preserving, unknown kinds rejected at construction.

### Fixed (v0.1.2 — this is also a CORRECTION of a prior conclusion)
- **Known-truth recovery challenge (`benchmarks/known_truth_challenge.py` + pinned wrapper `tests/test_dyn_rime_known_truth.py`, audit-only):** blind structural-recovery test answering the reviewer-directed open question. 11 in-grammar planted monomial worlds (d ∈ {2,3,6} × grid {log,uniform,irregular} × noise 1×/3× relative-to-signal × coefficient/offset variation) + 1 exp() negative control. **Measured: RIME static grammar recovers the planted structure on 11/11 (max |Ĉ−C|/C ≈ 5e-4, tighter than the BIC-OLS control), 0 false promotions in-grammar; on the out-of-grammar control RIME does NOT abstain (emits a spurious fit) — scope refusal must come from adjudication, not the search.** Harness numerics fixes recorded before citation: dev-slice-relative noise σ, standardized structure regression / OLS design, BIC-selected sparse OLS as the correctly-specified baseline (a full-degree polynomial OLS is documented as failing on extrapolating grids: holdout R² −33). Abstention never counts toward recovery; per preregistration.
- **Kernel phase profile (`rust/apedp-kernel/tests/rime_profile.rs` → `benchmarks/kernel_profile_results.json`, audit-only):** per-candidate decomposition into graph evaluation / readout fit / search bookkeeping (remainder estimate, median of 7). Measured on the planted lag-2 world with the release profile: at n=150 the per-candidate cost is ~19 µs of which ~90% is search bookkeeping (enumeration, dedup, gate arithmetic), while at n=12 000 graph evaluation dominates (~181 µs vs ~54 µs readout fit). Operational consequence for v0.2 planning: residual-guided evolution should first target the enumerative/bookkeeping layer at small n, not the arithmetic.
- **Scale-dependent ridge in `fit_readout` (engine numerics, `rust/apedp-kernel/src/rime/eval.rs`):** the fixed `1e-9` literal added to `Σ(o−ō)²` exceeded the raw second moment of algebraic-scale regressors (a p² column ≈1e-8) by ~30× and silently zeroed every such fit — this, not the grammar, was the primary cause of the measured QEC static-law failure. The ridge is now relative to the regressor's raw second moment (identical for unit-scale data; the QEC attribution "RIME's grammar is the wrong instrument for static scaling laws" is retired: the temporal grammar reaches the law via its self-interaction rewrite, pinned by `test_dyn_ga_qec.py` at holdout R²>0.99 both directions).
- **`tests/rime_regression.rs` was stale against the v0.1.2 typed API** (referenced the removed `SearchPacket` surface and did not compile); rewritten as equivalents against `SearchOutcome` with true null-abstention, positive-control and budget assertions; `rime_stage1.rs` assertions recalibrated to commit 30e73bf's planted amplitude (previously internally inconsistent: even a perfect fit could not clear the control's own bar).

### Added (this round, earlier)
- **RIME Repair diagnosis v0.1.0 (`apedp_dyn/epistemic/rime_repair_diagnosis.py`, experimental):** frozen `ResidualReport` + finding vocabulary on real proposals; all cutoffs explicit caller args.
- **RIME Repair engine v0.1.1–v0.1.4 (`apedp_dyn/epistemic/rime_repair_engine.py`, experimental):** bounded R1/R2/R3 operators, chronological 4-role partitions with single-use sealed split (`HoldoutContaminationError` on reopen), five explicit gates, prediction-equivalence clustering. Promotion structurally deferred to the APEDP-X track (no CandidateProposal→ScoredCandidate bridge fabricated).
- **Sensitivity/recall milestone (v0.1.2–v0.1.3):** recall 9/9 strong planted defects with structure recovery; weak-defect curve never harms; BTC abstention regression; 5/5 walk-forward abstain; adversarial stability-block.

### Fixed
- **RIME `residual.rs` stub:** removed hardcoded threshold literal; residuals now use the fitted readout (consistent with `eval.rs` scoring); threshold is an explicit caller argument with regression tests.

### Model-class closure (experimental engine only — no frozen-spec change)
- Interaction-probe screen (`interaction_suspect`) + lagged square/product R2 terms mirrored 1:1 (probe == term by construction); lag choices cite the frozen LAG_SET's two shortest entries.
- Planted composite (linear part + lagged product of independent drivers) completes through diagnosis→R2→gates→sealed; QEC-conservative refusal preserved (3 refused rows).
- Flaw-closure round: product recall 0/5 budgets (penalty-bound); recall 18/18 with 95% lower bound; single-round termination pinned (multi-round chaining documented as unreachable-by-construction under veto semantics); wall-time resource table; read-only whiteness-guard sweep (operating point 0.3–0.5, default untouched).
- #11 selective risk/coverage (post-hoc refusal policy, no harness changes): GA/OLS attempt signal worlds and abstain noise worlds; QEC thirds split roles (OLS refuses flat fit, repair upgrades to sealed 0.987).
- #7 floor sweep via explicit tau_epi (default untouched): near-tie flips across bracket monotonically; decisive UNIQUE at 0.05+0.1; identical-predictions MULTI at all floors; anchor artifact verified on disk.

### Documented limitations (capability boundary, not a defect)
- Same-driver omitted lags are usually absorbed by OLS into ~white residuals (4/6 sensitivity corners silent); firing corners yield gated rejection. R3's demonstrated domain is emergent lag from an independent driver. Whiteness alone is insufficient for structural identification.
- Whiteness gate assumes stationarity across evaluation rows: on log-spaced QEC data, quadratic extrapolation misfit trips the temporal screen and vetoes an R²=0.95 repair (pinned conservative behavior; refusal is the safe direction).

### Frozen scope
- No further grammar expansion. Next milestone is comparative validation (matched-compute repair vs proposer, independent inspection), not new operators.

## [2.0.0] - 2026-10-12

### Added
- **Full-Lineage Experiment & Candidate Ledger (`FullLineageExperimentLedger`):** Complete lineage tracking recording candidate ancestry, source proposer, feature sets, hyperparameters, random seeds, and specific dataset slice exposures.
- **Anti-Snooping Holdout Guard (`HoldoutContaminationError`):** Permanent holdout slice burning. When an empirical hypothesis fails Stage 4 out-of-sample confirmation, the evaluated slice `(dataset_id, slice_id)` is permanently blacklisted. Any iterative tweak attempting to reuse it triggers `HoldoutContaminationError`.
- **Six-Stage Scoped Promotion State Machine (`PromotionTracker`):** Replaced monolithic boolean promotion flags with six non-fungible sequential stages:
  1. `MATHEMATICAL_CONSISTENCY` (Lean 4 kernel + dimensional invariants)
  2. `PREDICTIVE_RECOVERY` (Cross-validated out-of-sample $R^2 > 0.05$, $p < 0.01$)
  3. `UNIQUE_IDENTIFICATION` (Diebold-Mariano test $p < 0.01$; literature projection $R^2 < 0.85$)
  4. `OUT_OF_SAMPLE_TRANSFER` (Un-snooped temporal regimes and cross-asset transfer retention $\ge 50\%$)
  5. `ECONOMIC_VALIDITY` (Net Sharpe after fees, breakeven fee $\ge 3.0$ bps, $DSR \ge 0.95$)
  6. `REAL_MARKET_PROMOTION` (Live sandbox / paper-trading execution stability)
- **Calibrated Three-Tier Benchmark Suite (`calibrated_benchmark_suite.py`):**
  - Tier 1: Referee discipline measuring FPR ($\le 0.05$), CAR ($\ge 0.70$), and VRR ($\ge 0.75$) with exact Wilson score confidence intervals.
  - Tier 2: Discovery statistical power ($1 - \beta$) across a calibrated grid of effect sizes ($IC \in [0.02, 0.05, 0.15]$), fat-tailed Student-$t$ noise ($\nu=4$), and structural regime switches.
  - Tier 3: Transfer generalization measuring retention decay across distinct out-of-distribution market regimes.
- **Canonical Preregistered Specification (`specifications/canonical_preregistered_spec_v1.json`):** Single authoritative source of truth for all preregistered constants, guarded by SHA-256 hash verification in `apedp_dyn/conf/canonical_spec.py`.
- **Documentation & Configuration Integrity Suite (`tests/test_documentation_and_spec_integrity.py`):** CI tests that fail if runtime code, test suites, or documentation drift from frozen canonical specifications.
- **Discovery Power Calibration Curves (`generate_power_curve_matrix`):** Power calibration evaluating theoretical vs empirical detection thresholds ($\Phi(\sqrt{N} \cdot IC - z_\alpha)$) with Wilson CIs across sample sizes and noise structures.
- **Expanded Proposer Suite:**
  - `DistributedLagInteractionProposer`: Distributed lag kernels ($k \in \{1, 2, 5, 10\}$) and bilinear feature interactions.
  - `RegimeSwitchingProposer`: Piecewise threshold models adaptive to market volatility states.
  - `HawkesOrderFlowProposer`: Self-exciting point processes modeling order arrival clustering and quote replenishment.
- **Regime Robustness Diagnostic (`evaluate_regime_robustness_diagnostic`):** Environment-stratified predictive incrementality evaluating cross-regime forecast survival without rigid residual homoskedasticity vetoes.
- **Decoupled Execution Cost Models (`execution_cost_models.py`):** Separated cost models: `FixedPlusSlippageCostModel` for slower factors, `AlmgrenChrissImpactCostModel` for temporary/permanent market impact, and `LimitOrderQueueCostModel` for queue fills and adverse selection.

### RIME v0.1.2 — Reliability & Evidence Integrity (completing the review specification)

The evidence round was extended with the reliability guarantees the review checklist required (no grammar growth, rules 1–3 unchanged):

- **Frozen environment handling.** `MIN_ENV_FINITE_ROWS = 15` and three explicit `EnvStatus` values (`Scored` / `TooFewRows` / `Malformed`): short environments are excluded from every aggregation, ragged ones fail loudly, all-unusable input abstains with an explicitly infinite reference and zero evaluated candidates, NaN gaps are skip-and-count (there is a gapped-environment recovery test), and per-env statuses ship in the search packet. Regression tests cover heterogeneous 3-env worlds, exclusion, and degenerate abstention.
- **PruneEdge semantics proven, not assumed.** New `rime_prune_semantics.rs`: exactly-zero contribution; complexity/DoF accounting agrees everywhere (zeroing, not removal — stated in the packet); content address changes so pruned ≠ parent; byte-reproducible predictions across serialization; no collateral edits; a fully zeroed graph yields an intercept-only readout that cannot outscore the null.
- **Explicit search limits.** Structural rewrites provably never CREATE a `Sign`/`Abs` node (depth-3 expansion over the whole universe; wrapping an inherited one is composition and is allowed from seeds only, reachability-tested); an off-grid 0.87 decay is never rendered as an exact `state_0.8700` and its in-sample R² is honestly < 1; budget-1 and empty-seed searches abstain explicitly.
- **Benchmark provenance & determinism.** The artifact records benchmark version, git revision (+dirty flag), kernel version, config, seed policy, timestamp, the feature-availability rule, and a rime-determinism self-check; recovery/accuracy/cost/complexity remain separate fields.
- **APEDP-X boundary tests.** Graph identity is checked end-to-end (Rust id = proposal id = packet id, 64-hex), malformed graphs (self-referencing arena, out-of-frozen lag) raise explicitly at the kernel, the referee contract field list is asserted complete, and the no-confirmation-feedback property was verified structurally (no RIME references in epistemic/, no referee imports in RIME).

### RIME v0.1.2 — Evidence Round (multi-environment scoring, stress benchmark, terminology)

**Added.** Addressing the v0.1.1 review's three next steps, without adding structure beyond the frozen primitives.

- **Multi-environment selection (`search_envs`, `rime_search_envs_json`, `RimeProposer(per_frame_envs=True)`).** Environments are scored separately under a rule frozen before scoring and reported in the packet: `MeanPerEnv` (equal weight per environment — the literal §8 reading, fixing the pooled path's length weighting) and `MedianWorstGated` (median across environments plus a **worst-environment improvement gate**: the candidate must beat its parent by more than that environment's own noise scale $\gamma\,\mathrm{var}(y)/n$, with the judged environment selected by the *candidate's* lowest relative improvement — selecting by the parent's MSE would hand the decision to whichever environment has the largest variance, a property of the data, not the mechanism). A dominance-trap regression test at both layers pins the behaviour: one long stream's wins cannot outvote a short stream where the mechanism is absent; the consistent-law control passes the same gate.
- **Broader benchmark (`benchmarks/rime_vs_neural.py --stress`, artifact `stress` key).** Noise sweeps (0.02, 0.5, 1.0), amplitude sweeps (0.5, 3.0), on-grid (0.80) and deliberately off-grid (0.87) memory decays, each with and without a ≈0.9-correlated distractor input — 18 cells. RIME reported the right structure in 18/18 cells; the artifact records the honest details (e.g. `lag a=0.5, noise=1.0` → OOS $R^2$ 0.154; off-grid `state d=0.87` recovered via the nearest grid member — the documented discrete-decay approximation). Not done here, deliberately: PySR and existing-proposer matched-compute runs remain pending their own environment.
- **Terminology:** `compute_score` is documented and tested as **past-information-only** — a temporal (no look-ahead) property of the signal path, explicitly **not** a causal claim; the proposal metadata carries `signal_semantics`, `per_frame_envs` and `aggregation`.

### RIME v0.1.1 — Stage-1 Engine Integration

**Added.** The RIME symbolic-mechanism engine (`rust/apedp-kernel/src/rime/`) is now correct, searchable, and wired into the proposer boundary.
  - `Node::State`: a genuine persistent memory cell, $h_t = d\,h_{t-1} + u_t$ with $h_{-1} = \text{init}$, replacing the `AddState` encoding that was not a recursion and carried the input term with the wrong sign.
  - `rime/eval.rs` fits a ridge least-squares readout $a\,o_t + b$ before scoring (spec §7 required fitting; it was never implemented) and charges the two readout degrees of freedom plus an EBIC-style selection charge $2\ln M$ for the search width (spec §10, "Bonferroni over grammar size").
  - `rime/seeds.rs`: deterministic 60-form seed catalogue (levels, delayed levels, differences, momentum, both lead-lag interaction orientations, memory cells on a discrete decay grid, wrappers). Without it the five frozen rewrites cannot reach difference or momentum forms at all.
  - `rime/search.rs`: deterministic budgeted beam search anchored on the mean predictor, with expression-level deduplication and per-candidate referee packets (graph id, equation, dev evidence).
  - `apedp_dyn/proposers/rime_proposer.py`: RIME as a `CandidateProposal` producer with a past-information-only `compute_score` (temporal hygiene, explicitly not a causal claim) that re-evaluates the graph on unseen frames; abstains on null worlds.
  - PyO3 entry points `rime_search_json` and `rime_evaluate_series_json` (batch-oriented FFI, spec §12).

### Fixed & Remediated
- **RIME per-node memoisation returned wrong values (Stage-1 audit defect 1):** the evaluator cached one value per *node* instead of per *(node, time step)*, so any node read both directly and through `Lag` returned the current row in both places; `x[t] - x[t-1]` evaluated to `[NaN, 0, 0, 0]` instead of `[NaN, 1, 1, 1]`. Evaluation is now series-at-a-time in arena (topological) order, releasing each node's series once its last consumer has read it, with unreachable nodes skipped and non-topological arenas rejected by `Graph::validate` instead of silently producing numbers. Regression: `tests/rime_regression.rs::difference_term_keeps_its_values`, `rust/apedp-kernel/src/rime/graph.rs::repeated_operand_is_not_memoised_across_time`.
- **`AddState` was not a state (defect 2):** all-ones input at $d = 0.5$ returned `[NaN, -0.5, -0.5, -0.5]`; a true decaying memory returns `[1, 1.5, 1.75, 1.875]`. Regression: `tests/rime_regression.rs::add_state_is_a_recursive_decay`.
- **Candidates were scored on their raw output (defect 3):** a correctly structured candidate with a small amplitude and a large offset was invisible (improvement $\approx 0$ against $\mathrm{var}(y)$ in the thousands). Regression: `tests/rime_regression.rs::scale_and_intercept_are_fitted_before_scoring`.
- **The bundled null test proved nothing (defect 4):** it charged 1.0 per edge while the world under test had MSE $\approx 0.08$, so no candidate could ever be promoted. The control now runs on the sampling-noise scale, is paired with a positive control through the identical scoring path, and bounds the false-promotion rate over 24 independent null worlds. A short-series arm (`n = 150`, 60 candidates) additionally pins the multiplicity charge; without it a fraction-of-variance penalty promotes fit noise.
- **Tactical Active Fraction Harmonization:** Resolved historical documentation drift where documentation stated $[5\%, 95\%]$ while code enforced $[5\%, 90\%]$. The range is now pinned strictly to $[5\%, 90\%]$ across code, documentation, and regression tests.
- **Audit Documentation Location:** Moved engineering audit report to [`docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md`](docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md) within the project repository workspace.
- **Wilson CI Boundary Handling:** Fixed numerical precision at boundaries ($k=0$ and $k=n$) in `wilson_ci` ensuring exact $[0.0, 1.0]$ bounds.

---

## [1.2.0] - 2026-10-11

### Added
- **Native Lean 4 Verification Pipeline (`lean_universal_bridge.py`):** Zero-disk in-memory verification using Lean 4 compiler's native `--stdin` streaming, reducing theorem verification time from hundreds of milliseconds of disk I/O to sub-50ms kernel checks.
- **Deflated Sharpe Ratio (DSR):** Closed-form implementation of Bailey & López de Prado (2014) accounting for selection bias, number of trials, non-normality, and extreme-value asymptotic expected max Sharpe ratio.
- **Decoupled Novelty Adjudication (`novelty_adjudication.py`):**
  - Predictive incrementality evaluated via Diebold-Mariano tests with HAC standard errors ($p < 0.01$).
  - Structural novelty evaluated against canonical literature registry (Kyle 1985, Amihud 2002, Biais-Large-Bouchaud resiliency, Carhart 1997 momentum) with $R^2 < 0.85$ novelty ceiling.
- **Stock-Split Scale Invariance Gate:** Replaced unphysical price-scaling checks with correct joint price/volume conservation: $P' = P/k$, $V' = kV$, testing $S(P', V') == S(P, V)$.
- **Competing Multi-Proposer Implementations:** Added `SparseMicrostructureProposer` (standardized ElasticNet) and `OrderBookResiliencyProposer` (exponential decay recovery).

---

## [1.1.0] - 2026-10-09

### Fixed
- **Rank Ablation Live Path:** Re-implemented missing rank ablation calculation in `canonical_ledger.py` verifying incremental signal contribution.
- **Hardcoded Sharpe/DD Display Literals:** Eliminated all hardcoded display literals; all metrics computed dynamically from per-observation ledger rows.
- **Rust GA Index-0 Bug:** Fixed genetic algorithm best-individual selection in Rust kernel (`src/genetic.rs`) to correctly select minimum-MSE individual rather than index-0 individual.
- **Decorative Gate Suppression:** Added `GATE_NON_BINDING` handling in `canonical_ledger.py` ensuring non-binding thresholds suppress Sharpe ratios and abstain.
- **Mask-First Imputation:** Enforced validity masks prior to any nan-to-num or imputation operations.
