## 2026-10-28 — Three metric placeholders now measure or refuse

- **FIXED, `EnsembleSearcher.get_diversity_score`:** returned a constant 0.8 with a
  "would analyze equation structures" comment. It now reads the solutions out of the
  results (`best` / `expression` / `expressions` / `equations` / `solutions`) and scores
  `(distinct - 1) / (total - 1)`: 0.0 when every member agrees, 1.0 when no two do. Fewer
  than two reported solutions — including its own `search_ensemble` plan — raises, because
  a plan has no diversity to measure.
- **FIXED, `DeepEnsemble.predict`:** reported `aleatoric=jnp.zeros_like(std)`, presenting an
  unmodelled quantity as a measured zero. These regressors train deterministically, so the
  field is `None`; the monkeypatch that previously masked the same placeholder after import
  was removed with it.
- **FIXED, `PhysicsConstraintEngine.dimensional_analysis_score`:** returned 0.5 with no
  target and 1.0 with one, regardless of any candidate. It now takes the dimensions of the
  candidate's terms and returns the fraction that match the target, raising when there are
  no terms or no target. The Rust kernel had the identical placeholder and now implements
  the same measurement over a JSON term list (rebuilt extension); the Python fallback and
  the kernel agree, and both `check_dimensional_consistency` paths now compare all seven
  exponents instead of mass/length/time only.
- **Verified:** `tests/test_placeholder_metrics_are_measured_or_refused.py` 9 passed (both
  implementations of the dimensional score, the diversity endpoints, the refusals, and
  `aleatoric is None`); `tests/test_search_engines_real.py` 13 passed.

## 2026-10-28 — MR arm correctness tested locally: four levers, and `r` fails as it should

- **BUILT:** `apedp_dyn/quantum_compilation/local_mr_sim.py`, a Z-basis (value + sign)
simulator for the DQI decoder with `mr` (Z measure-and-reset, phase-preserving), `r`
(clears, randomizes the phase, as `kickmix` does) and `hmr` (X basis). It implements the
builder subset the decoder uses (`unary_iteration`, `init_and`, `iadd`/`isub`,
`flip_if_*`, rotations, GF(2^b) arithmetic, `gf2_phase_by_product`) with semantics
pinned against `kickmix`, including its primitive polynomials.
- **VALIDATED:** `scripts/verify_local_mr_sim.py` builds the reference decoder with both
engines and feeds both the same shots. At m=15/n=6/GF(2⁴): syndromes identical, phase
identical, syndromes restored, every other qubit `|0>`, 0 phase errors — the local engine
reproduces `kickmix` exactly (qubit *ids* differ because freed ids are not reused here).
- **MEASURED:** `scripts/run_mr_gated_arms.py` runs the four gated levers with their
predicates read by a real `mr` through `cx(control, ancilla); mr(ancilla, bit)`, not the
oracle registers the probe used. Every arm reproduces the reference's values AND phase
at 32 shots: `swap` (c3), `sweep` (c0 ∨ c3), `forney` (the `sigma == 0` zero-test),
`clear` (measured clear of the EEA instead of its adjoint, with the syndromes re-prepared
from a saved readout), and all four together. Record:
`benchmarks/quantum_compilation/local_mr_arms.json`.
- **FALSIFIED, by construction:** the same swap arm read with `r` instead of `mr` fails on
phase and leaves junk behind while its syndromes still match — so the pass is about the
measurement, not the circuit. This is the mechanism-level version of the 11/16 phase
defect the probe recorded.
- **STILL UNSCORED:** the arms' cost. `local_mr_sim` computes values and phases, not gate
counts, and `contracts.py` pins the official `kickmix` build and verifiers, so the 4.142×
stays a Toffoli bound until `MR` exists upstream.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed (pinned primitives, the MR/R/HMR
contract, and the arm record).

## 2026-10-28 — Search engines score on data, and NGTS runs UCT or nothing

- **MEASURED, not defaulted:** `ValidationPipeline.validate` now reports
  `generalization_gap` (held-out minus cross-validation error) and derives
  `generalization_score` from it as `1/(1+gap)`. The old score was a lookup — 0.9 below
  a threshold, 0.7 above — so it reported a number the data could not move.
- **No fabricated evidence:** `NGTSSearch.run_simulations` runs UCT and calls a reward
  function for each selected leaf. With no reward function it runs **zero** simulations
  and returns 0; it previously returned `num_sims` and backpropagated
  `random.random()*0.5`, i.e. reported search work it had not done. The Rust kernel had
  the same defect in `ngts.rs` (`rand_simple` values) and now takes the reward callable
  in `run_simulations(num_sims, reward=None)`.
- **Hierarchical phases score what they propose:** `_mid_search` / `_fine_search` take
  `data`/`target`, evaluate each expansion on the data, and return nothing when there is
  no data. `HierarchicalSearch.search` attaches a measured `r2` to every candidate and
  **drops** candidates it cannot evaluate; it used to validate every candidate against
  default `train_error=0.1 / test_error=0.15`, so an unmeasured expression carried a
  validation score.
- **Ensemble reports what it cannot run:** a proposer outside `ProposerRouter`'s map is
  `status: error` instead of a planned member, and `best` is present and `None` until a
  member returns a scored candidate.
- **Determinism:** `beam_search_expressions` no longer consumes `seed` for sampling; two
  seeds return the same list, and level 1 pairs atoms with each other so multi-variable
  forms appear immediately.
- **Verified:** `tests/test_search_engines_real.py` 13 passed (was 12 of 13 failing);
  related suites 58 passed / 2 skipped / 1 xfailed.

## 2026-10-28 — Scratch pool measured dead: 25-qubit ceiling, and the allocator claim withdrawn

- **MEASURED** on the pinned Benchmark II instance (`scripts/measure_dqi_scratch_pool.py`,
  record `benchmarks/quantum_compilation/dqi_scratch_pool.json`): the circuit holds
  **1,828 qubits concurrently live against 1,893 distinct ids**; the peak is reached
  inside `_chien_search_and_forney_mbuc` and the innermost frame there holds **25**
  qubits. A shared scratch pool can therefore redistribute at most 25 — the −104 the V6
  row claims is 4× that ceiling and is not reachable by pooling.
- **CORRECTED, committed claim:** §5.5 of the boundary doc said the live peak was EEA
  round scratch in `_multiply_and_add_or_sub` and that a shared pool was "the only
  mechanism that could move" the transients. Both are withdrawn on measurement: the peak
  is in Chien/Forney, and the pool is dead.
- **Left OPEN, not answered:** the probe's trace replay returns 1,821 ids against 1,893
  shipped and observes only 1,873 ids, so allocations made inside the extension are
  missing; a lowest-hole free list replays identically to the shipped LIFO stack. An
  earlier draft of the script asserted the 65 stranded ids were `alloc_qubit` LIFO
  stranding and named a free list as the fix. That verdict was not supported by the
  replay and has been removed rather than shipped.
- **Verified:** `tests/test_dqi_scratch_pool.py` pins the artifact; the probe runs under
  CPython 3.13 only, since `kickmix.so` is a 3.13 build.

## 2026-10-28 — Post-`_update_flag` probe collected: 4.142× measured

- **MEASURED:** the shift gate the previous entry had to abandon is now correct. The
  wrong arm read `flag` at the top of the round; `_update_flag` toggles it first, so the
  probe now carries six bits per round and the liveness is
  `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with the post-update flag. Official instance:
  `T_mean` 1,457,103.45 → **1,382,207.08, a 4.142× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, 1,332 round bits (6 × 222).
  Scaled: 3.709× at m=255 (512 shots), 2.998× at m=15.
- **Corrected prediction:** the prize was bounded at ~5% from the failed arm's cost and
  measures at 5.15% (1,457,103 → 1,382,207). Dead shifts are 31.59% of rounds, not the
  35–40% guessed before the fix.
- **Running total:** Forney gating 1.332×, MR clear 2.652×, sweep 3.599×, swap 3.929×,
  shift 4.142×. Receipt arithmetic 24.439T → 5.900T, portfolio 2.361982 → 3.79; still
  3.10× short of the 12.86× target.
- **Verified:** `tests/test_dqi_predicate_gating.py` 12 passed / 1 skipped.

## 2026-10-28 — Dead register swaps gated: 3.929× measured, and the shift gate falsified

- **MEASURED:** `_swap_a_and_b` exchanges `reg_a`/`reg_b` wholesale — 2 × 73 × 12 = 1,752
  qubits — every round, but its control is `swap_ctrl = c3_hc ∧ flag`, so it is dead in
  any round without `c3` (**67.57%** of rounds). Gating it on `c3` alone: official
  `T_mean` 1,590,938.28 → **1,457,103.45, a 3.929× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, for 1,110 round bits (5 × 222).
  Scaled: 3.507× at m=255 (70.4% dead), 2.813× at m=15. Receipt arithmetic 24.439T →
  6.22T, portfolio 2.361982 → 3.7271; still 3.27× short of 12.86×.
- **FALSIFIED (mine, one hour old):** gating `_shift_b_array_right` on the complementary
  pair `c1 ∨ c2 ∨ ¬flag`. It looked right and fails on **every shot** (512/512 value
  errors at m=255). Cause: `flag` is toggled by `_update_flag`, which runs *before* the
  shift in the same round, so a flag read at the top of the round is stale. The arm's
  cost (62,434 at m=255 against 65,698) bounds the remaining prize at ~5%, and collecting
  it needs a second probe point after `_update_flag`. Not taken; pinned as a negative in
  `tests/test_dqi_predicate_gating.py`.
- **Cost of the whole design so far:** 1,803 classical bits for the clear, 1,110 for the
  round predicates, zero extra qubits, zero extra Toffolis.

## 2026-10-28 — Dead multiply sweeps gated: 3.599× measured

- **MEASURED:** the synchronized EEA emits all five steps every round, but
  `_multiply_and_add_or_sub` touches `ctrl` only through `c0_hc` and `c3_hc`. In a round
  where both are zero — the two shift rounds of each four-round counter cycle — its
  4,536-Toffoli sweep is emitted and does nothing. Reading `(c0_hc, c3_hc)` per round
  and gating the sweep on `c0 ∨ c3` removes **35.14%** of the forward EEA at the official
  instance: `T_mean` 2,158,651.45 → **1,590,938.28, a 3.599× over the reference**, with
  `Q` unchanged at 1,893 and **0 value / 0 phase errors**. Scaled: 3.205× at m=255
  (40.79% dead), 2.586× at m=15 (46.98% dead).
- **Cost:** one classical bit per EEA round — 222 bits at the official instance, 54 at
  m=255, 30 at m=15 — no qubits, no Toffolis. Still an upper bound: the round bits are
  read from the same circuit in a probe pass.
- **Not taken:** `_shift_b_array_right` is live in the `c1`/`c2` rounds *and* in the
  `c0`/`c3` rounds when `flag` is clear (`ctrl_nB = c0_nf ⊕ c1_hc`), so it cannot be
  gated on the complementary pair without also reading `flag`; left alone.
- **Verified:** `tests/test_dqi_predicate_gating.py` 9 passed / 1 skipped.
- With the Forney gating and the MR clear this is 3.599× of the 12.86× the target needs:
  `T_mean` 1.591M against 445,229 leaves a 3.57× gap, and the receipt moves 24.439T →
  6.791T (portfolio 2.361982 → 3.6198), arithmetic on `R·Q²·T`, not a receipt.

## 2026-10-28 — MR clear halves the EEA: 2.652× measured, and `R` is not a substitute

- **MEASURED (the better design):** the reversible EEA is run twice — forward, then
  its adjoint to uncompute — so half of Benchmark II's 75% EEA share is uncomputation.
  Clearing the EEA scratch and the syndromes and re-preparing the syndromes from a
  classical readout replaces that half. `scripts/probe_dqi_predicate_gating.py`
  measures it with both oracle registers (syndrome readout + root bits): official
  instance `T_mean` **5,725,071.61 → 2,158,651.45, a 2.652×**, `Q` unchanged at 1,893,
  **0 value and 0 phase errors** on 64 shots, needing 1,803 classical bits and 1,803
  classically controlled X's. Scaled: 2.351× at m=255 (512 shots), 2.012× at m=15.
  Because `Q` does not move, the score falls by the same 2.652×: 24.439T → **9.215T**,
  i.e. the portfolio score 2.361982 → **3.2695** (arithmetic on `R·Q²·T`, not a receipt).
  Still 4.85× short of the 12.86× target.
- **FALSIFIED (mine, from four hours earlier):** "`R` can stand in for uncomputation."
  A reset clear costs exactly what the reference spends (2,158,651.45 either way — CX
  is not Toffoli-like) but it is **wrong**: 35/64 shots at the official instance and
  254/512 at m=255 have a flipped phase. Cause, measured: a circuit with **no phase in
  it at all** — `x(q); reset(q)` — reports a random `phase_flipped` (11/16 shots), while
  the same circuit without the reset reports 0/16. `R` randomizes the tracked phase, so
  it can only be used where the answer does not depend on it. Measure-then-X on the
  measured value (i.e. `MR`) gives 0/0.
- **BOUND (corrected):** with MR, both levers together are worth 2.652×, not the 1.332×
  the Forney-gating entry below states: that entry bounded *one* phase of the decoder and
  its "even granted MR, T lands at 4.30M" bullet is superseded here. `T_mean` 2.159M is
  still 4.85× above the 445,229 target, so the §5.5 conclusion stands — the target needs
  a decoder, not an optimisation.
- **Still blocked in the IR:** `MR` does not exist. `push_condition` takes a classical
  bit; `hmr` is X-basis; `push_condition(qubit)`/`bit_store(control=qubit)` are rejected;
  `R` is phase-unsafe. The instruction that would make 2.652× real is Z-basis
  measure-and-reset, and this design needs it twice.
- **Verified:** `tests/test_dqi_predicate_gating.py` 7 passed / 1 skipped;
  the skipped re-measurement was run by hand under the kickmix interpreter and reproduces
  the record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Predicate-gated Forney body: 1.332× measured, and inexpressible in this IR

- **MEASURED:** `scripts/probe_dqi_predicate_gating.py` gates the Forney body on the
  truth table of `sigma(alpha^j) == 0` (supplied as an oracle classical register, so
  this is an upper bound, not a submission). At the official instance `T_mean` falls
  **5,725,082.58 → 4,298,955.87 (1.332×)** with `Q` = 1,893 and the static count
  unchanged, and with **0 value and 0 phase errors** on 128 shots: the body is a
  no-op wherever `located = is_sigma_zero & omega` is zero, so skipping it is free
  in correctness. This is the 1.335× "if the phase were free" row of
  `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5, measured instead of estimated.
  Scaled checks: 1.185× at `m=255, GF(2^8)` and 1.011× at `m=15, GF(2^4)`, where the
  root density (2.03%, 12.7%) is much higher.
- **CORRECTED (null optimisation):** the committed `is_error` gate at
  `dqi_reference_circuit.py:573`/`:596` buys **0.006%** — 353 of 5,725,461 Toffoli per
  shot — because `P(M_j = 0) = 2^-12`. The gap between `toffoli_count` and
  `max_toffoli_count` in the receipt is that gate firing about once per shot, not a
  conditional-execution saving worth pursuing.
- **BLOCKED (IR):** the predicate cannot reach the condition stack. `push_condition`
  requires a `km.b`; the only readout is `hmr`, which is **X-basis** (on `|1>` it
  returned 31/64 ones); `push_condition(qubit)` and `bit_store(control=qubit)` are
  both rejected by the binding; and the instruction set `{NEG, BIT_INVERT,
  BIT_STORE0/1, X, Z, R, HMR, CX, CZ, SWAP, CSWAP, CCX, CCZ, Z_POW, INFO_TAG,
  DEBUG_PRINT, PUSH/POP_CONDITION}` is closed under the Z-diagonal group, so no
  Hadamard can be synthesized to conjugate the predicate into the X basis. One
  instruction — Z-basis measure-and-reset (`MR`) — is the entire difference.
- **BOUND (superseded):** 1.332× leaves `T_mean` at 4.30M against the 445,229 target
  (9.7× short) and the receipt at 18.35T — arithmetic on `R·Q²·T`, not a receipt. The
  Forney body is 24.9% of `T`; the other 75% is EEA iterations, which never skip. See
  the entry above: clearing instead of uncomputing that EEA reaches 2.652×, so this
  1.332× prices only the Forney phase.
- **Verified:** `tests/test_dqi_predicate_gating.py` 4 passed / 1 skipped; the skipped
  re-measurement was run by hand under the kickmix interpreter and reproduces the
  record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Benchmark II cost structure measured; two of my own optimisations falsified

- **MEASURED:** `scripts/measure_dqi_primitive_costs.py` recovers the DQI Toffoli
  breakdown, since `kickmix` exposes none. Per-call at GF(2¹²): `gf2_imul` and
  `gf2_idiv` **0** (CNOT-only linear maps), `ixor_gf2_mul` 63, Itoh–Tsujii
  inversion 315 = 5 × 63, scaffold division 378. Slopes at the benchmark widths:
  17,208 Toffoli per EEA iteration, 351 per Chien/Forney position.
- **FALSIFIED (mine):** "composite-field/Karatsuba cuts constant-multiplier cost
  30–50%" — there is nothing to cut. Multiply and divide by a constant are already
  0 Toffoli. The only multiplier that costs anything is quantum × quantum, and 63
  at degree 12 is already better than a naive schoolbook count.
- **FALSIFIED (mine):** Montgomery batch inversion of the Forney denominators. It
  needs 3 multiplications per element, so it can recover at most 2/5 of an
  inversion; at this repo's marginal qubit price 2T/Q = 6,048.7 Toffoli-equivalents
  per qubit, and 12 qubits per batched element, **every block size loses** (best
  B=4: 193,489 saved vs 290,338 spent, score ratio 1.021 — worse).
- **BOUND (new):** the Forney step inverts once per position
  (`dqi_reference_circuit.py:579`), so 4,095 × 315 = **1,289,925 Toffolis = 22.5%
  of `T_mean`, and 2.90× the entire 445,229 target** before any EEA iteration runs.
  Ceilings with `Q` unchanged: 1.259× if a position cost one multiply, 1.335× if
  the phase were free — against 12.86× required. The target is unreachable for this
  architecture, not merely unreceipted.
- **Q floor:** 1,887 measured = 1,752 persistent (`2 × 73 × 12`, both frozen) + 135
  transient. The claimed −104 is 77% of all transient qubits, and the live peak is
  EEA round scratch inside `_multiply_and_add_or_sub`; the `n_a`/`n_b` recycling
  the V6 row names frees 16.
- **Verified:** `tests/test_dqi_primitive_costs.py` 7 passed / 1 skipped; the
  skipped re-measurement was run by hand under the kickmix interpreter and
  reproduces the record exactly. Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5.

## 2026-10-28 — Riemann off-line scan scoping fixed; August batch screened for Benchmark II

- **FIXED (integrity):** `rust/riemann_verifier` built its off-line critical-strip grid
  as `(100..400) * 0.1` — `t in [10, 40]` — **regardless of `max_t`**, then printed
  `OFF_LINE_GRID_SCAN_PASS`. A run asked about `T=300` certified a window 260 units
  shorter than the question; the same defect class as the `PASS` removed from the DQI
  Gluzdov V6 projection the same day. The window is now `[10, max_t]` and is never
  truncated to fit the sample budget: the budget (20,000 samples) degrades resolution
  instead, and a coarsened scan reports `OFF_LINE_GRID_SCAN_PASS_COARSENED` rather than
  a bare PASS. JSON gains `grid_scan_window`, `grid_scan_step`, `grid_scan_samples`,
  `grid_scan_coarsened`, `grid_scan_covers_requested_height`.
- **Verified:** `T=60 → [10,60] step 0.1 / 501 samples`; `T=300 → [10,300] step 0.1 /
  2,901 samples`; `T=5000 → [10,5000] step 0.2495 / 20,000 samples / PASS_COARSENED`.
  `tests/test_riemann_verifier_grid_scan.py` 3/3, including a source-level revert pin.
  `probe_wave1.json` / `PROBE_WAVE1.md` regenerated against the fixed binary; verdicts
  unchanged (2 consistent_finite_range, 1 no_refutation, 1 not_probeable).
- **RECORDED (unfixed, separate):** root discovery at a fixed 0.05 step still misses
  zeros with height — 0 % at T≤100, 14.5 % at T=300, 84 % at T=5000 (721/4520). That is
  a discovery-budget problem, not the scan window, and it is now stated as such.
- **NEW:** `catalog/august_ten_advances_screen.json` +
  `catalog/AUGUST_TEN_ADVANCES_BENCHMARK_II.md` — the August 1 *Ten advances* batch is
  the only OpenAI release whose disciplines intersect Benchmark II (coding theory,
  arithmetic circuit complexity, quantum complexity, lattice cryptography), so it is
  screened against the DQI cost model. **0 of 10 rows admits a cost path**: three are
  lower bounds or hardness results (a barrier is not a tool), six have no circuit content,
  and the single coding-theory row is blocked by the frozen instance
  `(m=4095, n=70, degree=12, r=2016)` that `ProblemSpecDQI` fixes. Its only use is to put
  a floor under `R(eps_plus)`, not to lower the score.
- **Disclosed:** recomputing the target gain from the two workload rows gives 14.3973×,
  against the quoted 14.396× — a +0.009 (0.06 %) rounding difference recorded rather than
  absorbed. `T_mean` must fall to 7.78 % of reference (12.86×); `Q` only to 94.5 % (1.058×).
- **Verified:** `tests/test_august_ten_advances_benchmark_ii.py` 7/7, which cross-checks
  the frozen instance against the verifier's own `DEFAULT_M/N/DEGREE/R` and the cost model
  against the pinned workloads in `apedp_dyn.quantum.dqi_photonic_resource`.

## 2026-10-28 — Audit artifacts for the negative results, and the PDE honesty note

- **COMMITTED:** the seven `scripts/` audit instruments that back numbers already
  recorded in `CHANGELOG.md` and `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` had been
  left untracked, so the claims cited them from outside the repository. Two were
  load-bearing: the committed `run_halt_recycling_scale.py` imports
  `RecycledFlagDecoder` from `scripts/test_recycled_flags_small.py`, and the
  restored documentation cites `scripts/test_viable_optimization_path.py`.
  Verified by running the committed driver end to end (−240 counter-only at
  m=15/1023, +600/+360 flag, +240/+480 combined).
- **DISCLOSED:** importing `RecycledFlagDecoder` is safe; running it is not. Its
  toy circuit fails the official verifier (effective error rate 0.9155 ≥ 1/16,
  198/256 value errors). The flags prototype is a broken circuit, not merely a
  slower one.
- **NEW:** `docs/pde_discovery_honest_limits.md` plus
  `tests/test_dyn_pde_discovery_noise_robustness.py` — states that the default PDE
  discovery config is brittle to small additive noise and that the clean pinned
  demos are not evidence of general noise robustness, with the first fixed-config
  multi-seed sweep as the backing test (2 passed).
- **NEW:** `tests/test_dyn_dqi_photonic_resource.py` (14 passed) pins the shipped
  `dqi_photonic_resource` mapper: platform invariance of `R·Q²·T`, the fault-tolerance
  verdict, and the caller-supplied superconducting baseline having no defaults.
- **NEW:** `tests/test_dyn_rime_ad_input_contract.py` pins the `RimeAdProposer`
  input contract (`DataFrame | dict[str, DataFrame]`, unsupported shapes fail
  loudly rather than returning a candidate).

## 2026-10-28 — Portfolio synthesis names the real Benchmark I incumbent

- **CORRECTED:** `docs/invsqrt_frontier_and_portfolio_synthesis.md` still named the
  retired 90Q affine build ($157,752.008809$) as champion and carried a Benchmark I
  factor of 13.164804×. The pinned incumbent is the 89Q shared-PDDD circuit
  (`PASS`, 200,540,928 shots, `ce04c9d5…`) at $C_1 = 157,601.080829$, i.e.
  **13.177411×**. The retired rows (90Q affine, SABLE-90) are kept on the record
  with the reason each was retired rather than deleted, so the champion's margin
  stays checkable.
- **Recorded:** the FAROS-R1/R2/R3 rejection margins are re-stated against the
  89Q basis (R2 15-bit now +1,590.87 / +1.01 %, R1 +41,576.84 / +26.4 %); every
  rejection conclusion is unchanged and every margin grows.
- **Disclosed:** Benchmark I's worst error/tolerance is 0.999623, so 0.0377 % of
  the phase budget is unused — less than a single rotation is worth (see the
  rotation ledger in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4).

## 2026-10-28 — Gluzdov V6 stops claiming a result it never had

- **FABRICATION REMOVED:** `dqi_gluzdov_v6.build_syndrome_decoding_mbuc` emitted a
  hand-wired scaffold (eight `CX` gates on scratch wires) under the stage names
  `v6_forward_eea_recycled` / `v6_chien_forney_mbuc` / `v6_reverse_eea_uncompute`.
  It was not a decoder: no EEA, no Chien search, no Forney evaluation. The module
  docstring nevertheless announced a 14.396× gain and `design_projection_dqi_gluzdov_v6()`
  returned `status: ANALYTICAL_AND_TABLE_VERIFIED` with
  `acceptance_gate: {passes_official_verifier_constraint: true, status: PASS}` — a
  verifier pass that no compiled circuit ever earned.
- **Now:** the circuit builder delegates to the real
  `ProosAndZalkaDecoder.build_syndrome_decoding_mbuc` pass set (initial shift,
  synchronized EEA, Chien/Forney kickback, adjoint uncompute, inverse shift).
  `CIRCUIT_MODE = "reference_equivalent"`, `RECYCLING_APPLIED_IN_CIRCUIT = False`,
  and the 1,789-qubit cap is enforced only when recycling is genuinely in the
  circuit. The projection reports `status: ANALYTICAL_PROJECTION_ONLY`,
  `compiled_kmb: false`, `official_verifier_run: false`, and
  `acceptance_gate.status: HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT` with the four
  pieces of evidence required to close it. The 14.396× factor is a target.
- **NEW:** `dqi_builder.build_candidate(n_iters, architecture=...)` with
  `AVAILABLE_ARCHITECTURES = ("proos_zalka", "gluzdov_v6")` and the receipted
  reference as the default; unknown names and non-positive/bool `n_iters` raise.
- **Verified:** `tests/test_dqi_gluzdov_v6.py` 10 passed / 1 skipped (the skipped
  one needs the kickmix extension).

## 2026-10-28 — Restore the deleted module registry in `docs/DOCUMENTATION.md`

- **INTEGRITY REPAIR:** commit `c08acbe` ("Document stability selection") replaced
  the 2,134-line `docs/DOCUMENTATION.md` with a 35-line stub, deleting the module
  registry (310 of 312 `apedp_dyn` modules lost their index entry, 429 module
  links, and the `[5 %, 90 %]` tactical-bound discussion). Nothing flagged it
  because the four firewalls in `tests/test_documentation_and_spec_integrity.py`
  were left failing and reported as pre-existing. They were not pre-existing —
  they were the regression being detected.
- **Restored** the registry from `444c1fa` (2,347 lines, 523 repository-relative
  links, 312/312 modules indexed) and re-applied the additions made since
  (stability selection, DQI photonic resource mapping, 89Q champion
  reproduction, mixed polarity, photonic boundary) as §23. The restored text
  carries the numbering defect it had at `444c1fa` (two `## 18` and two `## 19`
  headings, sections 16/17/20/21/22 headingless); it is left as-is rather than
  renumbered, because renumbering breaks every `#N-...` anchor the TOC uses.
- **Added** the one genuinely unindexed module, `apedp_dyn/video/collision_domain.py`.
- **Verified:** `tests/test_documentation_and_spec_integrity.py` 6/6 pass (link
  coverage ≥ 200, zero orphan modules, zero pinned debt, registry status tags
  agree with code `Status:` lines).

## 2026-10-28 — Rotation overhead closed, halting-counter saving falsified, photonic artifact

- **MEASURED (negative):** the 44 priced rotations in the 89Q invsqrt champion
  cost 315.299429 of the 1770.798661 Toffoli-equivalent (17.81 % of the score)
  and cannot be reduced. `scripts/search_rotation_merges.py`: 0 exact mergeable
  pairs (all 44 sit on distinct qubits; the 7 qubits carrying a second rotation
  have 75–192 non-commuting gates between the two), and 0 affordable drops (the
  smallest rotation, 2⁻³⁵ = 2.998e-11 turns, is 7.5× the entire 3.975e-12-turn
  phase slack). Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4.
- **FALSIFIED:** halting-counter recycling does not hold at scale.
  `scripts/run_halt_recycling_scale.py` measures −240 Toffolis at every instance
  (constant, because one fixed-width comparison per call site is removed):
  −0.722 % on the toy m=15/n=6/GF(2⁴), −0.0118 % at the benchmark instance
  m=4095/n=70/GF(2¹²) (2,033,345 → 2,033,105). The shipped prototype (flag +
  counter) is a **regression** at every size measured (+480 at full scale). The
  sFFCC table in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.2 is corrected accordingly.
- **NEW (non-submission):** `scripts/report_photonic_non_submission.py` prices all
  three champions on the sFFCC lattice at L=16 and writes
  `artifacts/quantum_compilation/photonic_non_submission.json` with
  `submission_status: NON_SUBMISSION`, `score_delta: 0.0`, and the reasons: the
  verifiers consume a compiled `.kmb`/`.kmx` artifact, and none of the three
  score formulas contains a time, photon, fusion or detector term.
- **NEW:** `docs/BENCHMARK_BREAKING_LITERATURE.md` — published techniques priced
  against each benchmark's measured exchange rate (I: 89.0 units/Toffoli vs
  1770.8/qubit; II: `R·Q²` vs `2RQT`; III: 35,990.0/Toffoli vs 4,596,711.8/qubit),
  including the two techniques that provably cannot help (better Clifford+T
  synthesis cannot move Benchmark I, whose rotation cost is the benchmark's own
  formula of ε alone).

## 2026-10-27 — Stability selection utility + blind-benchmark PDE work

- **NEW:** `apedp/stability_selection.py` — general stability-selection helper
  (`stability_select`) for repeated sparse fits with different seeds/subsamples:
  keeps terms with selection frequency >= pi, averages coefficients, returns empty
  dict on all-failure. Honest caveats: only helps stochastic fitters; deterministic
  fitters see no benefit.
- **WORKSPACE (not committed to repo):** blind-benchmark weak-form PDE fitter
  and stability-fit variant in `/tmp/buffy/blind/`. Results: stability selection
  (PI=0.9 chosen on VALID, frozen, THRESH=0.02, K=20, 5 seeds/BENCH)
  improves exact recovery from 50/100 to 66/100 on BENCH heat/burgers/allen_cahn/
  fisher_kpp. Heat still fails at noise 0.05/0.1 (0/5 each). Full artifacts:
  `/tmp/buffy/blind/bench_stability.json`, `benchmarks/mutation_battery_report.json`
  and the other `/tmp/buffy/blind/*.json` still on disk (no process delete).
- **Caveats recorded on disk:** `/tmp/buffy/blind/limits.txt` (reproducibility,
  seed count, noise model, synthetic data, heat failure, K=20 choice).

### Honest limitations (not fixed, disclosed)
- The utility is NOT a general PDE-discovery fix. Repo PDE path
  (`apedp_dyn/epistemic/pde_discovery`, Rust STRidge) is deterministic and was
  NOT improved by this; on a sample heat+noise case it still returned many
  spurious terms. Stability selection applies only to fitters that randomize
  their structure (here: weak-form test-function windows).
- Heat remains the hard case under noise; this is a partial reduction of spurious
  terms, not a complete solution. Root cause points to library/sparsity, not
  threshold/selection tuning alone.
- Validation-BENCH split: PI=0.9 was picked on VALID only and frozen before BENCH,
  so there is no benchmark tuning. But the result still depends on a single
  frozen pi and a single threshold; different noise models / domains may need
  re-tuning on a held-out split.

### Non-deleted working processes on disk
The following working directories / artifacts exist and were NOT deleted:
- `/tmp/buffy/blind/` — blind PDE benchmark scripts + results + logs.
- `benchmarks/mutation_battery_report.json` — mutation battery report (still
  referenced by the repo).
- `scripts/search_claims_pddd.c` — search-claims probe source (new, not deleted).
- Other existing `/tmp/` working artifacts remain as-is; this commit does not
  touch them.
