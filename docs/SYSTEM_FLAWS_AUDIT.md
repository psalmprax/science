# APEDP System Flaws Audit

Date: 2026-11-08 (deep architectural & code remediation round — Categories 1–5)
Basis: independent adversarial audit over the epistemic referee layers,
proposers, discovery engines, the Rust kernel (`apedp-kernel`), formal
verification bridges, ledgers, API/UI trust boundaries, CI/Docker packaging,
and physics/robotics modules, verified by execution.

## 2026-11-16 follow-up — the first full run's failures, each traced to a cause

Basis: `stage` from `96c4e60` to the commit that carries this section. F-05
below reported the first full local run: 53 failed and 6 errors. F-03, F-04 and
F-07 accounted for 15 of those 59 test ids. This round takes the other 44 one
at a time. 43 now pass for a stated reason and one needs the trustees (FLAW-01).
F-05's sentence that the thirteen unexplained failures "may be real
regressions, or differences between this environment and the one their pins
were recorded in" is superseded: each is now one or the other, below.

Two facts shaped the work and outlast it.

- **J-01 there is no passing version to compare against.** `6a92eaa`
  (2026-10-24) is a root commit: the repository's history starts there, and
  ten of the thirteen unexplained tests already fail at it. `git bisect` found
  the cause of one of the other three (the 225/64 count, J-03); the rest were
  diagnosed from the code and the records.
- **J-02 the changelog describes work the tree does not contain — OPEN.**
  Commit `a32bfa5` (2026-11-03, "Publish the accumulated CHANGELOG entries")
  added 1,870 lines to `CHANGELOG.md` and touched no other file. Three of the
  failures were fixes described in those lines, or in
  `docs/SCAFFOLD_INVENTORY.md`, and absent from the repository:
  `drop_collinear_columns` (J-03), the corrected H-A coupling pin with its new
  test and the corrected §76 (J-04), and the 225-of-225 count (J-03). They are
  now implemented as described. *Not checked:* the rest of those 1,870 lines.
  Some of what they describe is present (`apedp/model_synthesis_helpers.py`,
  the renamed creep test); whether all of it is has not been audited, and until
  it is, a changelog entry from that commit is not evidence that the change
  exists.

Each entry: cause, what changed, the pin for it.

- **J-03 grammar scope, two tests — FIXED (code missing; one stale count).**
  `LibraryProposer` selected `{one, abs_x, log_abs_x, sqrt_abs_x}` on the
  Duffing trajectory instead of `{u, x, x3, v}`: `sin_x` and `tanh_x`
  correlate with `x` at 0.99999 there and `cos_x` with `x2` at −1.00000, and
  the documented filter for that did not exist. `drop_collinear_columns` and
  `COLLINEARITY_LIMIT = 0.999` are now in `apedp_dyn/grammar/proposer.py` as
  the changelog specifies, and `last_skipped` reports each dropped column.
  Recovery is exact (1, −4, −2, −0.2). `test_coup_restricted_counts` pinned 64
  survivors of 225; commit `99f15f2` fixed the operand formatting that had
  rejected every candidate carrying a coupling term, and the inventory records
  225 of 225. The assertion had not been updated.
- **J-04 H-A coupling rank sequence — FIXED (stale pin, corrected against an
  independent record).** The pin `H-RV,H-RV,H-A,H-P×8` predates FLAW-A01. The
  corrected sequence, `H-RV,H-RV,H-A,H-A,H-A,H-RV×6`, is not a snapshot of
  today's output: the re-derivation table in
  `docs/TRADING_EQUATION_FORENSICS.md` already carried the post-fix H-A counts
  3/3/5 for BTC/ETH/SOL, and a new parametrised test reproduces all three.
  §76 carries a correction note; its "persistence absorbs the rotation"
  reading described the leaked sequence and is withdrawn (H-P is never rank-1
  on BTC once the leak is removed).
- **J-05 the DQI patch record — FIXED (this audit's tooling had altered
  evidence).** Ten files under
  `sandbox/google-verifiable-quantum-circuit-compilation/verifiers` were
  reformatted by a repository-wide lint pass. They are a third party's
  verifier, and the record pins the exact patch applied to them. Restored from
  `6a92eaa`; the directory is excluded from ruff and mypy by provenance so a
  later pass cannot rewrite it. This was the test the CI matrix stopped on in
  F-08.
- **J-06 the synthesizer, three tests — FIXED (two defects, one test that
  could not fail).**
  (a) *Scale invariance at 1e9.* An amplitude that starts at `median(|y|)`
  shares a parameter vector with rates of order 1. Measured on the test's own
  law with the 24 `rational_transition` templates: identical to unit scale
  through `max|y| = 2.8e4`, 4 of 24 degraded at 2.8e5, all 24 failed from
  2.8e6 (best validation error 0.58 of the variance against 1.6e-5), and the
  family left the ranking. Fits now run on `y / max|y|` above 1e4, as they
  already did below 1e-2; inside the band nothing changes.
  (b) *Planck, `TIER_B_01`.* Stage D refined on `x / max|x|` only. There the
  in-bounds envelope optimum needs `A = 1731`, `B = 1481` against caps of 126;
  every start railed, the refinement was rejected, and the synthesizer
  abstained at R² 0.940 on a problem
  `docs/APEDP_V4_SYNTHESIS_BENCHMARK_REPORT.md` records above 0.999. Stage D
  now also refines on `x / geometric-mean|x|` and keeps the better accepted
  result (`_stage_d_x_scales`). Planck returns R² 0.9991 in three x units.
  Checked on the 60 EXT-BENCH and four Tier B/C problems: no validation R²
  fell; three selections changed (C42 to its true family, R24's label to its
  true family, `TIER_B_01` from abstention). The scan costs 26 % more time.
  (c) *Near-OOD veto.* The only behavioural test ran the whole synthesizer on
  P05 and required the bare veto to pick the wrong family there. A
  two-parameter logarithmic candidate now leads P05, the bare veto no longer
  fires on it or on any of the 60 problems, and the test failed on its own
  guard. The rule moved unchanged into `_parsimony_tiebreak` and is tested on
  the recorded pair of candidates, in both directions.
  Pins: `tests/test_stage_d_scales.py` (17), the rewritten
  `TestNearOodVetoIsNoiseFloored`, `tests/test_scale_invariance.py`,
  `tests/test_tier_bc_benchmark.py`.
- **J-07 FLAW-07 was never carried into the sandboxes — FIXED, and the
  "verified" solver was not.** FLAW-07 made the apex tier require rival
  rejection and interventional invariance; the ladder's own fixture was
  updated and the two sandboxes that submit C6 claims were not, so their valid
  cases were downgraded (three tests; the meta scorecard read 50 % retention).
  Granting the evidence by case number would have been a fiat, so it is
  measured. In the PDE sandbox, `audit_interventional_invariance` re-runs the
  simulator under eight single-argument changes. By that measure the verified
  scheme failed: its time step came from the advective term alone, the default
  run sat at a stability number of 0.995 by coincidence, and doubling `nu`,
  refining to 200 cells or asking for `cfl = 0.8` grew its energy by a factor
  of thousands. The step now respects advection and diffusion together and the
  scheme holds under all eight. In the meta-sandbox, the clean Kepler case
  measures invariance of the FCC density over nine size and radius changes and
  that no feasible rival lattice exceeds it. Withholding either measurement
  downgrades the claim.
- **J-08 the tournament trace is not portable — pin narrowed, with the
  evidence; one finding withdrawn.** `tests/test_dyn_tournament.py` pinned one
  machine's whole trace. Same code and seeds under scipy 1.14.1, 1.15.3,
  1.17.1 and 1.18.1 give three different traces for C1, C4 and C8, and the
  recorded pin reproduces under none of them (table in the test's docstring).
  Those candidates sit at the gate-4 tolerance, where the layer they end at
  follows the optimizer through an unconverged fit: the fitted truth's holdout
  error on rep 0 ranges from 0.085 to 0.344 of the observed RMS. Pinned now is
  what all five traces share (C5 at gate 1, C6 at gate 3, C2/C3/C7 at gate 4;
  no adversary promoted; status implied by the survivor count). This is a
  narrower pin, not a fix: exact attribution of the three boundary candidates
  is no longer asserted anywhere. The earlier docstring's finding D1, that gate
  4 rejects the fitted truth at 3 % noise, describes where one optimizer
  stopped and should not be cited. `test_dyn_pointer` read a git-ignored
  artefact nothing writes; it now writes one from a live run.
- **J-09 one "flaky" test was a 2.7 GB allocation in the kernel — FIXED.**
  `test_fan_out_emits_candidates_on_signal_feature` failed only in the
  parallel run because its worker was killed. `rime_rational_search_json`
  pairs each numerator with each denominator; the evaluation cache kept every
  quotient's n-vector for the life of the search, and one call on a 200-row,
  one-feature problem took the process from 262 MB to 2,992 MB. In this round's
  full run the same allocation took four other workers' tests with it. The
  quotient is now evicted once scored: one call peaks at 377 MB, the test file
  at 604 MB instead of 3,247, in 22 s instead of 30. The 201 kernel unit tests
  pass; `tests/test_kernel_rational_search_memory.py` measures the call in a
  child process against a 1,200 MB limit.
- **J-10 two ratchets and the CI command.** `test_D_collinear_pair_tied`
  required a tie class in a fixed order; the two fits differ by 1.4e-17 and
  the order belongs to the LAPACK build. It asserts the class, and a second
  test keeps a distinct rival out. The exception-hygiene size budget was
  re-recorded with its own tool: commit `531f8e5` grew
  `epistemic/adjudicate.py` from 1,026 to 1,127 lines for the null-model guard
  and re-minted the freeze, but left the budget, so the ratchet had failed
  since. The re-record admits that growth, tightens three ceilings and budgets
  `promotion_lifecycle.py` for the first time. *This is a raised ceiling.* The
  alternative, splitting a freeze-pinned module and moving mutation anchors, is
  the owner's decision. The CI test job now installs the pinned Lean release
  (it sets `APEDP_REQUIRE_NATIVE`, under which a missing toolchain fails) and
  deselects the governance test, whose verdict the governance-manifest job
  already reports and enforces; three contract tests hold that list at one
  entry and fail once the manifest is signed again.
- **J-11 dependency-bound tests, 27 — FIXED (skip policy), and two real
  defects among them.** A test that needs an absent optional dependency now
  skips with the reason, and fails instead where `APEDP_REQUIRE_NATIVE` says
  the environment must provide it (`tests/toolchain_availability.py`,
  `tests/kickmix_availability.py`). Two groups were defects in the tests. The
  notebook-adapter tests created `/kaggle` and `/content` on the host, after
  which platform detection reported Kaggle to every later run and six tests
  failed permanently; the adapter's platform roots are now module constants
  the tests redirect. Two peak-memory tests assumed freshly allocated pages
  were new to the resident set, which depended on what had run earlier in the
  same worker; they allocate with an anonymous mapping and touch each page.

- **J-12 two of this round's fixes edited freeze-pinned files, and the guard
  caught it — re-minted, with outcomes changed.**
  `pde_verification_sandbox.py` and `apedp_x_meta_sandbox.py` are among the
  fifteen files `APEDPX_FROZEN_v1_18.json` hashes. The guard list was not
  checked before J-07 edited them; the full run then failed four tests with
  "REFEREE OR FROZEN RECORD CHANGED", which was the guard working. The record
  carries the two new digests and a `remint_log` entry each. Unlike the
  2026-11-14 re-mint these change B_internal outcomes, and the entries say so:
  PDE trap 6 and the clean Kepler case, both downgraded since FLAW-07, are
  promoted, and the PDE time step changes every trap's trajectory numbers. No
  claim-ladder rule, evidence requirement, threshold or adjudication call
  changed. Whether a sandbox change that moves internal benchmark outcomes
  should be re-minted in place or as a new record version is the owner's
  convention to set; this followed the in-place precedent of `531f8e5`.
- **J-13 the second full local run** (Python 3.13, numpy 2.5.3, scipy 1.18.1,
  kernel built with Rust 1.97.0, Lean on `PATH`, two workers, 23 minutes):
  5,868 passed, 123 skipped, 1 xfailed, 10 failed. The ten: four freeze-guard
  tests (J-12, now passing); five whose worker was killed for memory (J-09),
  each of which passes when run alone; and `test_dyn_governance` (FLAW-01).
  The run was not repeated after J-09 and J-12; the affected tests were re-run
  on their own, and the AGENTS.md regression suite gives 283 passed, 1
  skipped. Today's fixed tests also pass under the two library sets the CI
  matrix installs (numpy 2.2.6 with scipy 1.15.3, numpy 2.4.6 with scipy
  1.17.1), checked for nine test files. The hosted runner's verdict is in
  J-15.
- **J-14 the owner's commit of the same morning, merged — three stale tests
  and the size budget again.** `101b9c3` ("Harden Rust referee bindings,
  numerical solvers, Lean bridge, …") reached `stage` while this round was in
  progress. It merges cleanly. Two things in it fail where a Lean toolchain is
  present, which the CI test job now has. (a) It rewrote
  `epistemic/lean_universal_bridge.py` so that a conservation claim is proved
  from the candidate's own expression or refused, and removed the canned
  Pythagorean theorem that used to be attached to any such claim; its new
  `tests/test_lean_universal_bridge_soundness.py` pins that. Three older tests
  in `tests/test_dyn_lean_universal_bridge.py` still relied on the canned
  theorem and failed on that commit alone. They now use a candidate the bridge
  really proves (a sum of squares) for the "synthesized but not kernel-checked
  is P1" contract, a new test pins the refusal of the old oscillator case, and
  one message assertion follows the new wording. No bridge code was changed.
  (b) The same rewrite grew the module from 1,498 to 1,970 lines, past the
  size budget re-recorded in J-10 an hour earlier. It is re-recorded again.
  That is the second owner commit in six days to outgrow this budget
  (`adjudicate.py` +101, this +472). A ceiling that is raised each time it is
  reached constrains nothing: either these modules are split or the size
  budget should be dropped. That is the owner's decision.

- **J-16 the fitter's verdict on the true model depends on the scipy release
  — test fixed; the fitter itself OPEN.** After the merge, CI's 3.11 leg ran
  1,007 tests and stopped at
  `test_dyn_adjudicate_regression.py::TestProvenanceFallbackFiresEndToEnd::test_fallback_abstains_without_any_explicit_series`,
  which passes here. Reproduced under scipy 1.17.1. The fixture fits the true
  Duffing model at 5 % noise from the fitter's default start. The cubic term
  is barely exercised at the training amplitude, the cost is flat along its
  coefficient, and `least_squares` stops with status 3 and optimality ~1e7:
  the coefficient comes out at −2.62 (scipy 1.18.1) or −2.70 (1.17.1) against
  a generating −2.0, with the same training cost to three figures. The holdout
  verdict on the *true model* follows the stopping point: gate 4 passes on
  1.18 (margin 0.94) and fails on 1.17 (1.03), and the residuals left by the
  wrong coefficient are structured (|rho1| 0.71–0.75), so the referee refuses
  the truth either way. Started from the generating parameters, the same
  fitter returns −2.0007 under scipy 1.15.3, 1.17.1 and 1.18.1 alike, gate-4
  margin 0.40, |rho1| 0.058, and the referee promotes it at the production
  flag. The test now fits from the generating parameters, sets the flag just
  below and just above the recorded series' |rho1|, and requires the reason
  to quote that value; that is a sharper test of the provenance fallback than
  the one it replaces, and it passes under all three releases. The companion
  test compared the fallback with an explicit `holdout_sq_error` series, the
  squared-error series `_residual_series` stopped using; it compares signed
  residuals at three flags.
  *Resolved by J-18:* the default-start fit, which production uses, stopped 30 %
  off on a well-posed problem, and the referee's verdict on the true model at
  5 % noise is decided by that stopping point. It is the same mechanism as
  J-08's tournament boundary. `fit_candidate_multistart` exists and is not
  used by these paths; whether to use it, rescale the problem, or tighten the
  fit-time integrator tolerance (`simulate` defaults to `rtol=1e-3`, chosen
  "for fitter throughput") changes every recorded fit and is the owner's
  decision.
- **J-17 four more, from the hosted runner and a full run under its
  libraries — FIXED.** With J-16 pushed, CI's 3.10 leg ran 2,155 tests and
  stopped at `test_dyn_notebook_adapter.py::test_detect_platform_cluster`: a
  GitHub runner sets `GITHUB_ACTIONS=true`, `detect_platform()` checks CI
  before Slurm, and the test blanked `CI` but not `GITHUB_ACTIONS`. The
  adapter tests now start with every variable the adapter reads unset (a test
  enforces that the list covers the source), and reproduce the runner's
  environment explicitly. A full local run under numpy 2.4.6 and scipy 1.17.1,
  CI's 3.11/3.12 libraries, then gave 5,843 passed and 2 failed. (a)
  `test_kernel_rational_search_memory` (J-09) read `ru_maxrss` in a child
  process; on Linux that survives fork and exec, so the child of a 3.2 GB
  pytest worker reported 3.2 GB for a search that peaks at 376 MB. It reads
  the child's `VmHWM`. (b)
  `test_vqc_stabilizer::test_an_exact_distribution_is_refused_past_the_enumerated_width`
  asserted the Rust tier's refusal message unconditionally; the CI test job
  does not build `apedp_vqc`, so there is no Rust tier there to refuse. That
  half is now conditional on the extension, as
  `tests/test_vqc_native_parity.py` already is, and the other branch checks
  the call was answered by Python. (c) The rustup step failed twice on
  2026-11-16 (`detected conflict: 'bin/cargo-clippy'`) after finding a
  half-installed 1.99.0, with `rust-docs` among its components, in a home that
  had been isolated a minute earlier and that the minimal profile never fills
  that way. What wrote it is not established from the logs. The toolchain is
  now installed in the step right after the isolation, into a home that is
  emptied first; if it is ever found non-empty, a warning lists what was there.
  (d) CI's 3.11 leg then ran 3,207 tests and stopped at
  `test_dyn_tournament.py::TestRmseRel::test_present_iff_holdout_reached`: there,
  C1 on rep 2 was fitted to parameters whose holdout run overflowed, so it
  stopped at gate 4 with no RMSE ("holdout simulation invalid"). The tournament
  already records that case as `None`; the test assumed it could not happen. It
  now requires such a row to stop at gate 4 and say why. Same cause as J-16, a
  third library- and interpreter-dependent stopping point.
- **J-18 the fitter converges — FIXED, and a recorded referee failure was the
  fitter's.** J-16's open half. The fit simulated at `simulate()`'s default
  `rtol=1e-3` and differentiated with `least_squares`' default step (~1.5e-8
  relative), so its Jacobian was the integrator's step selection, not the
  model: on the true Duffing model at 5 % noise it stopped with status 3 and
  first-order optimality up to 1e10, at a cubic coefficient of −2.60 (scipy
  1.18.1) or −2.73 (1.17.1), short of the minimum at −1.63. It now integrates
  at `FIT_RTOL = 1e-7`, `FIT_ATOL = 1e-9` and differentiates with
  `FIT_DIFF_STEP = 1e-5` and `x_scale="jac"`
  (`apedp_dyn/fitting/parameter_fit.py`). Measured on four noise seeds under
  scipy 1.15.3, 1.17.1 and 1.18.1: training losses agree to seven figures, the
  lowest-loss structure is the truth in all twelve fits (the old fitter chose
  the linear rival on seed 22 under one release), and the gate-4 verdict
  agrees across releases on every seed. A fit costs about twice as long; the
  full local suite went from 23 to 32 minutes, and the CI test job's limit is
  75 minutes, with the measurement in the workflow.
  What it moved. A full run gave 7 failures; each is explained and none is
  re-pinned to a new snapshot.
  (a) *STAB/10/rep2, the "frozen failure" in
  `outputs/dyn_epistemic_freeze.json`:* the cubic truth "blew up on holdout"
  and the linear rival was promoted unopposed. The blow-up was the fitter's
  stopping point. Converged, the truth passes gate 4 (0.0416 against 0.0863)
  and is promoted over the rival by 0.0339 against a tie band of 0.0154. Three
  guard tests used the cell as their sole-survivor case; they now take the
  rival alone, which is the case the guards are about, and a test pins that
  the frozen failure no longer happens. The flag-sensitivity sweep on the same
  residual series is judged on the rival alone for the same reason.
  (b) *Bootstrap parameter bands* (`epistemic/bootstrap.py`) centred the
  resampled problems on a trajectory integrated at 1e-3 and refitted them at
  the new tolerance; the band no longer contained the estimate it was built
  around. The centre is integrated at the fitter's tolerances.
  (c) *Noiseless two-DOF saturation test:* both models now reach a training
  loss of ~2e-14 and holdout errors of 3e-7 to 7e-7; its 5 % and 1 %
  comparisons were integrator noise there (CI's 3.10 leg had already failed it
  under the old fitter: 5.65e-7 against 6.37e-7). Both comparisons hold above a
  stated floor.
  (d) *The provenance-fallback tests* (J-16) return to the production route
  (default start, lowest training loss); the generating-parameter start is
  gone. The converged truth carries |rho1| 0.548 raw and 0.447 on
  standardized innovations here, so at the production flag of 0.5 the
  innovation fallback admits it.
  *Addendum (2026-11-17):* that statistic is not reproducible, and the tests
  no longer pin it. CI's 3.10 leg on `c43c972` measured the same recorded
  series at raw 0.304 (1 failed, 1011 passed); a local Python 3.10.20 with
  CI's numpy 2.2.6 and scipy 1.15.3 measured 0.548 again. The training loss
  agrees to seven figures; the residual structure does not, because the cubic
  coefficient sits on a flat direction of the training loss and where along it
  the fit stops moves the holdout residuals. The tests now pin the guard's
  behaviour on whatever series gate 4 recorded: refusal at a flag just below
  its |rho1| (quoted in the reason), admission just above it, and promotion of
  the truth at the production flag. The flag sweep on the rival alone keeps its
  fixed pins: the rival's |rho1| is 0.5934 under both local environments, and
  the truth's moves in the fourth figure (0.2268 against 0.2261) on the same
  cell. What this leaves open: on this cell the guard's verdict is stable, but
  the margin it is decided by is a property of the machine as well as the
  data.
- **J-19 a memory test asserted an ordering the kernel does not keep — FIXED
  in the test.** CI's 3.11 leg on `6d20a3f` ran 5,626 tests and failed one:
  `tests/test_vqc_memory.py`, window peak 158,822,400 bytes against a floor of
  158,826,496, one 4 KiB page apart. The floor is VmRSS read just after
  resetting the peak counter; the peak is VmHWM, which Linux raises lazily
  (before an unmap, not on every fault), so a page faulted in after the reset
  and released without an unmap leaves the peak a page under the floor. The
  three orderings the tests asserted (peak ≥ floor, whole-life peak ≥ floor,
  growth ≥ 0) now allow four pages. The measuring code is unchanged: it is
  pinned by `benchmarks/` through the external-comparison record, its
  reported growth is window peak minus floor as documented, and a one-page
  negative growth is within the resolution of what it measures.
  The scientific drift guardrail and the known-truth recovery gate pass
  unchanged. Not re-derived: benchmark records written by scripts that call the
  fitter; no test flags them, and they describe the old fitter until re-run.
Still open after this round: the governance signature (FLAW-01); J-02;
`test_market_discovery_covid_promotes`, which kills the interpreter on
this machine and has no verdict; the committed mutation report, still the
2026-10-23 file; the four tracked result files that test runs rewrite (F-05);
and the subject line of commit `f67e019`, which says "after its first week of
running" of a workflow that had run for hours.

---

## 2026-11-16 round — the first time the workflow ran, and what it had been hiding

Basis: `stage` at `87d89b6`, GitHub Actions run 43805790159 — the first run of
`.github/workflows/test.yml` on this branch to start a job (E-07 below explains
why none had). Its verdict: audit-artefact drift, the Lean core and the kernel
wheel build passed; Python quality, Rust quality and the mutation battery
failed; the governance job failed as designed (FLAW-01); the test matrix was
skipped because its prerequisites were red. None of the three unexpected
failures was introduced by the commit that made the workflow valid. Each was a
standing defect that a running CI would have reported weeks earlier.

A second fact changed what this audit can check. The native kernel builds here
with the locally available Rust 1.97.0 (the pin is 1.99.0, which this
environment cannot download), and torch, jax and the rest of the CI
dependencies install. Every statement in earlier rounds of the form "could not
be observed here without the kernel" is superseded by F-05.

- **F-01 the type-check baseline was pinned in an environment CI does not
  have — FIXED, and re-pinned from CI's own count.** CI counted 32 mypy errors under
  `apedp_dyn/` against a pin of 29. Reproduced with the same tool and torch
  2.9.1 present: 31 at the commit that pinned 29 (`49fed02`), so the pin was
  two lower than the gate's own environment could ever measure, and one error
  entered afterwards (`sandbox/eseds_dynamical.py:548`, 2026-11-10). None of
  the 32 is in a file this audit changed. Seven are removed by two
  annotation-only edits (`quantum_compilation/candidates.py`,
  `local_mr_sim.py`), leaving 25 here with and without torch. The next run
  (43813711672) measured the same 25 on the hosted runner, and every other
  count matched the local one as well, so the limits are lowered to what CI
  measured (`apedp_dyn/` mypy 29→25, `apedp/` mypy 938→897, `scripts/` ruff
  604→603, `benchmarks/` ruff 217→202) and `quality_baseline.json` now
  records the environment they were measured in.
  `scripts/check_quality_baseline.py` now prints the findings when a path
  regresses. It printed "+3" and nothing else, which is why this took a
  reproduction to diagnose.
- **F-02 the mutation battery could not run, in two independent ways — FIXED;
  34 of 37 mutants not re-verified here.** (a) The CI job called
  `maturin develop` with no virtualenv, which maturin refuses; the job now
  creates one. (b) Had it got further it would still have stopped: the anchors
  of MUT-C1 (Hawkes standardisation leak) and MUT-H2 (native-failure re-raise)
  stopped matching the source on 2026-11-08 and 2026-10-25 respectively, and
  the battery rightly raises on an anchor that does not apply exactly once. The committed
  `benchmarks/mutation_battery_report.json` says 37 of 37 killed and is dated
  2026-10-23; it describes a tree that no longer exists. Both anchors are
  repaired, and MUT-E6's is updated for the formatting in F-03. Verified here:
  all nine pristine pin groups pass, and the three repaired mutants are each
  killed with byte-identical restoration. The other 34 were not re-run here
  and the committed report is left as it was; the CI job is the full run.
  `tests/test_mutation_battery_anchors.py` now checks, in the ordinary suite
  and in under a second, that every registered anchor still applies exactly
  once, so the next drift is caught at the commit that causes it.
- **F-03 the Rust gates had never passed — FIXED for the kernel crate, one
  crate excluded for a stated reason.** `cargo fmt --all --check` reported 119
  differences in 22 files, and the Clippy gate (`-D warnings`) stopped on six
  compiler warnings: five unused imports and one needless `mut`. The kernel
  crate is now formatted (15 files) and the six warnings fixed with the
  compiler's own suggestions. `rust/apedp-vqc` is **not** formatted: its seven
  source files are pinned byte-for-byte by the machine-and-timing record that
  `tests/test_vqc_native_parity_record.py` checks, so reformatting them would
  make that record claim it measured bytes it did not. `make lint-rust`
  therefore checks format for the kernel crate only and says why; the crate's
  43 differences wait for a re-measurement. Clippy and the tests cover the
  whole workspace: 201 kernel unit tests, 19 kernel integration tests and 31
  VQC tests pass here, on Rust 1.97.0.
- **F-04 two defects introduced by this audit, found by its first full run —
  FIXED.** (a) `87d89b6` removed eighteen `# noqa: BLE001 - reason` comments
  because the pinned ruff reports them unused. They are not only suppressions:
  `scripts/audit_exception_hygiene.py` reads that exact marker as the record
  that a blanket handler is deliberate, and
  `tests/test_dyn_exception_hygiene.py` ratchets on it, so the removal turned
  eighteen annotated handlers into "silent" ones under the repository's own
  gate. The commit message said the files were AST-identical, which was true
  and beside the point. Restored; `BLE001` is declared in
  `[tool.ruff.lint] external`, which keeps the marker valid for ruff's
  unused-suppression check while the rule itself still fires on an unannotated
  blind `except`. (b) The parser pre-check added in `9c9c8bf` (H-09) caught
  `Exception` without the marker, in `apedp/safety.py` and
  `apedp_dyn/safety.py`. Annotated. Both went unseen for the reason E-07
  gives, and because the hygiene test was not in the subset re-run after
  those commits.
- **F-05 first full-suite run with the kernel and torch present — 5,639
  passed, 53 failed, 6 errors, 154 skipped (5,858 collected, 24 minutes, two
  workers).** Each failing test was then re-run alone on this tree and on
  `3d556eb`, the commit before this audit's first change. The verdicts are
  identical except for the tests F-03 and F-04 explain, so the remaining
  failures predate this audit. They are not all the same kind:
  - *A dependency the test needs is absent* (about two thirds), each cause
    checked where it could be: no `lean` on the test's `PATH` (three
    universal-bridge tests; all ten pass once Lean 4.24.0 is on it); the
    native VQC extension not built (`test_vqc_stabilizer`; passes once
    `rust/apedp-vqc` is built and installed); the pinned Rust toolchain not
    downloadable here (the Goldbach and Riemann sandbox tests; both pass with a
    usable toolchain); no `onnx` (JetFlow CPU export, three, and a Hugging
    Face streaming test that needs the network); no built `kickmix` extension
    (the DQI, DFTHC-SOS and FAROS-ISQ tests and the kickmix build pin,
    nineteen — ten of which skip when run alone and fail in the full run with
    `module 'kickmix' has no attribute 'CircuitBuilder'`, an order dependence
    between tests); no `stim` (`test_vqc_memory`, with one peak-RSS assertion
    that also fails on this machine); a generated `outputs/`
    artefact that is not in the repository (`test_dyn_pointer`); and platform
    detection that reads this container as a hosted notebook
    (`test_dyn_notebook_adapter`, six). These tests fail rather than skip when
    the dependency is missing. The CI test job installs neither Lean, nor the
    VQC extension, nor `onnx`, `stim` or `kickmix`, so most of this group
    should be expected to fail there as well.
  - *Expected*: `test_dyn_governance` (FLAW-01), and
    `test_dyn_exception_hygiene`, which after F-04 fails only on
    `epistemic/adjudicate.py` being 1,127 lines against a ceiling of 1,026 —
    already true at `3d556eb`, and a freeze-pinned file this audit has not
    edited.
  - *Unexplained, and the owner's to triage* (thirteen tests in ten files;
    **each is resolved in J-03 to J-10 above, 2026-11-16**):
    `test_dyn_grammar_scope` (an enumeration count of 225 where 64 is pinned;
    a recovered library that is not the Duffing truth), `test_dyn_ha_coupling`,
    `test_dyn_pde` (two), `test_dyn_term_necessity`, `test_dyn_tournament`
    (two), `test_dyn_meta_sandbox` (scorecard 50 where 100 is pinned),
    `test_dqi_submission_score` (a recorded patch hash that does not match),
    `test_scale_invariance[1e9]` (classified `exponential_envelope` where
    `rational_transition` is expected), `test_tier_bc_benchmark` (Planck R²
    0.94) and `test_external_60_adapter_contract`. These may be real
    regressions, or differences between this environment (Python 3.13, numpy
    2.5, Rust 1.97.0) and the one their pins were recorded in. Nothing here
    distinguishes the two.
  - `tests/test_dyn_ui_studio.py::test_market_discovery_covid_promotes` kills
    the interpreter on this 7 GB machine, alone or under xdist; no verdict.
  - Three more failed only in the parallel run and pass alone on both trees
    (`test_jetflow_trainer_and_dataset`, the notebook tutorial test,
    `test_fan_out_emits_candidates_on_signal_feature`).
  CI runs the suite with `-x`, so on a hosted runner the first of these that
  fails there will be the only one reported per run.
  Running the suites also rewrote four tracked result files
  (`benchmarks/five_gate_validation_results.json`,
  `docs/five_gate_validation_results.md`, and from the Rust tests
  `benchmarks/kernel_profile_results.json` and
  `benchmarks/sre_tree_vs_arena_baseline.json`). E-02 fixed three such paths;
  these four remain and were restored by hand.

- **F-06 the package could not be imported on Python 3.10, which it declares
  and CI tests — FIXED.** Run 43813711672 was the first in which the quality
  jobs passed and the test matrix started: Python quality and Rust quality
  green (format, Clippy and the Rust tests on the pinned 1.99.0, so the
  formatting done here with 1.97.0 matches the pin). The 3.10 leg then failed
  at its second step: `apedp/ingestion/ingestion_academic_apis.py` imported
  `datetime.UTC`, which exists from 3.11, and `apedp/__init__.py` imports that
  module, so `import apedp` raised `ImportError` on the interpreter
  `requires-python = ">=3.10"` promises. The matrix has `fail-fast`, so the
  3.11 and 3.12 legs were cancelled and gave no verdict. Two scripts had the
  same import. All three use `timezone.utc` now.
  `tests/test_minimum_python_version.py` checks, on any interpreter, that
  every source file parses under the declared minimum's grammar and uses no
  standard-library name from a curated list of later additions unless it is
  guarded; the 3.10 CI leg remains the real check, and what it finds after
  the import is not yet known.

- **F-07 six tests mistook an empty directory for the kickmix extension —
  FIXED; this is what stopped the first test-matrix run.** Run 43815370584
  (after F-06) is the first in which a test leg reached the suite: on 3.11 the
  kernel import check, the drift guardrail, the known-truth gate, the smoke
  tier and collection (5,942 tests) passed, and the regression run stopped at
  its first failure after 849 passes,
  `test_dqi_gluzdov_v6.py::test_circuit_builder_candidate_when_kickmix_available`
  — one of the order-dependent failures F-05 lists. The cause is the guard,
  not the order: these tests skip unless `import kickmix` succeeds, and the
  extension's directory exists in every checkout whether or not the extension
  has been built into it, so once its parent is on `sys.path` (other tests put
  it there) the import returns an empty namespace package and the guard
  passes. Reproduced deterministically by putting that directory on the path:
  six failures and four errors before, none after. The six guards now ask
  `tests/kickmix_availability.py` whether kickmix can build circuits, not
  whether a name imports.
  Not addressed: `test_faros_isq` (five) and `test_dyn_kickmix_build_pin`
  (four) require the built extension outright and fail in any checkout
  without it, CI included; whether those should skip or the build should be
  provisioned in CI is the owner's decision. With `-x`, the matrix will stop
  at whichever of the F-05 failures comes next.

- **F-08 where CI stands at the end of this round (run 43817593950, `stage`
  at `cc50e3b`).** Passing, each for the first time on this branch: Python
  quality (lint, both baselines, the CI-contract tests), Rust quality (format,
  Clippy and the Rust tests on the pinned 1.99.0), audit-artefact drift, the
  Lean core, the kernel wheel build, workflow validity, and the mutation
  battery — which answers what F-02 left open: on the hosted runner, with the
  pinned toolchain, all 37 mutants were verified fresh and killed, none from
  cache, in three consecutive runs. The committed
  `benchmarks/mutation_battery_report.json` is still the stale 2026-10-23
  file; CI uploads the current report as a workflow artefact and it has not
  been copied into the repository.
  Failing: the governance-manifest job, as designed, until the trustees
  re-sign (FLAW-01); and the test matrix, whose 3.10 leg now gets through the
  import, the kernel check, the drift guardrail, the known-truth gate, the
  smoke tier and collection, and stops after 955 passes at
  `tests/test_dqi_submission_score.py::test_the_recorded_patch_is_the_exact_patch_that_was_applied`
  — a recorded patch hash that does not match the file, one of the thirteen
  F-05 lists as unexplained, and now known to fail on the hosted runner as
  well as here. `fail-fast` then cancels the 3.11 and 3.12 legs. That test
  compares a research record with its source; changing either to make them
  agree is not something this audit can do honestly. The same holds for the
  rest of that list, which the matrix will reach one at a time.
  *Superseded by J-05: the mismatch was this audit's own lint pass having
  rewritten the vendored verifier. The source is restored to the recorded
  bytes; neither the record nor the test changed.*
  So the build job, which needs every gate, cannot go green on this branch
  until two things happen that are outside this audit: the trustees sign, and
  the owner triages those tests.
- **G-01 the referee could not be loaded without what it referees — FIXED as
  a dependency boundary; the files have not moved.** This is the first step of
  the "repository split" earlier rounds listed as not done, and it is the part
  that changes behaviour. Measured on `stage` in a fresh interpreter:
  `import apedp_dyn.epistemic.promotion_lifecycle` executed 74 repository
  modules (96 for the whole core), among them the native kernel, SymPy, pint
  and the discovery engines, because importing a submodule runs its parent
  packages and `apedp_dyn/__init__.py` imported everything. The module's own
  dependencies are numpy and scipy.
  Three changes: (a) `apedp_dyn`, `apedp_dyn.epistemic` and `apedp_dyn.market`
  re-export lazily (`apedp_dyn/_lazy.py`, PEP 562): same names, same
  `__all__`, each resolved on first access. (b) The evidence types stage 1
  judges (`InvariantKind`, `LeanTheoremVerification`,
  `LeanVerificationResult`) move to `epistemic/proof_evidence.py`, standard
  library only, and are re-exported from their old modules as the same class
  objects. (c) `tests/referee_core_manifest.py` declares the eight core
  modules and what they may depend on — numpy, scipy, pandas and three named
  optional paths — and `tests/test_referee_core_boundary.py` enforces it
  statically, at import and under test, with a planted-leak check.
  After: the same import executes 6 repository modules and loads numpy; the
  whole core is 13 (8 core modules, their 4 parent packages and `_lazy.py`)
  with numpy, scipy and pandas. In an environment with
  only those and pytest — no torch, no SymPy, no kernel, the repository not
  installed — the core's tests and the boundary tests pass (36 passed, 1
  skipped: the planted leak cannot even be imported there). CI's
  `referee-core` job repeats that on every push.
  Two things the equivalence test (`tests/test_lazy_package_exports.py`)
  caught while this was being written, both of which would have shipped
  otherwise: `apedp_dyn.epistemic.adjudicate` is both an exported function
  and a submodule, and a naive lazy package hands back the *module* once
  anything imports the submodule (handled; pinned from four different first
  touches); and `from apedp_dyn import ode, pde` exports subpackages, which a
  name-to-module table sends into infinite recursion (handled).
  A third was caught by the full suite, not by the new tests: the coverage
  audit classifies pure re-export modules as façades
  (`tools/coverage_audit.py`) and `tests/test_facade_contracts.py` pins
  `apedp_dyn` among them; a lazy `__init__` is not "imports plus `__all__`",
  so the package silently left the set whose exports that test checks. The
  classifier now recognises the lazy shape, and only that shape — a
  `TYPE_CHECKING` block of imports, string-literal tables, and the one
  `lazy_exports(...)` call — with ten near-miss variants pinned as *not* a
  façade (`tests/test_coverage_audit.py`).
  And one in the boundary check itself, raised in review of the pull
  request: the static rule recorded only the package a `from` import names,
  so `from apedp_dyn.epistemic import agnostic_adapter` passed as an import
  of the allowed parent, and inside a function no test calls neither
  runtime check would have seen it. It now resolves each imported name to
  the submodule or lazy re-export it loads; ten ways of writing an
  undeclared import are pinned as detected, the first three of which the
  earlier version let through.
  One behaviour is not reproduced, and is stated in that test: a submodule of
  a package that is not lazy is an attribute only once imported, as in any
  Python package. Eager imports used to make every such chain work by
  accident. No use of one was found in the repository.
  What this is not: a second distribution. There is no `apedp-referee`
  package and no file changed path, because paths are load-bearing here —
  AGENTS.md names `apedp_dyn.conf.canonical_spec`, the governance drift script
  loads it by file path, and fifteen files are hash-pinned by the freeze
  records. Extracting a package is now a mechanical move behind an enforced
  boundary, and a decision for whoever owns those pins.
  *Addendum (2026-11-17): extracted.* The eight modules moved, with history,
  to `apedp_referee/`, which is its own distribution (`apedp-referee` 0.1.0,
  `apedp_referee/pyproject.toml`) as well as part of the full one. Each old
  path is a four-line alias that installs the package's module object under
  the old name in `sys.modules`, with a `.pyi` stub for type checkers, so
  `apedp_dyn.conf.canonical_spec` (AGENTS.md's name), string monkeypatch
  targets and every existing import resolve to the same objects. None of the
  eight files was among those the freeze records pin (searched: the only
  path-keyed record naming one was the exception-hygiene size budget, carried
  over). The governance drift script loads `apedp_referee/canonical_spec.py`
  by file path. The package ships a copy of the canonical specification;
  `tests/test_referee_core_boundary.py` requires it to be byte-identical to
  `specifications/canonical_preregistered_spec_v1.json`, which the trustees
  sign, and both to hash to the pin. The governance manifest stays a
  repository record: installed alone, the referee reports it absent. Two edges
  back into `apedp_dyn` remain, both in `apedp_referee/_host.py`, both inside
  functions, both reached only for host conveniences (default ledger location,
  storage backend from the environment), and both raise a message saying what
  to pass instead when `apedp_dyn` is not installed; the third, a
  type-checking import of `BaseStorageBackend`, is replaced by a three-method
  `LedgerStorage` protocol. One latent defect surfaced: the full distribution
  never declared pandas, which `market/universe.py` imports; it does now.
  Verified: in a fresh Python 3.12 environment holding numpy, scipy, pandas
  and pytest, `pip install ./apedp_referee` builds a wheel of exactly the ten
  modules, the specification and `py.typed`, and the package's tests pass from
  a directory outside the checkout, including the test that asserts the
  module came from site-packages and that `apedp_dyn`, `apedp`, the kernel,
  torch and SymPy are not importable. CI's `referee-core` job repeats that.

Limits of this round: the local kernel is built with Rust 1.97.0, not the
pinned 1.99.0, so "passes here" is not "passes on the pin" for format, Clippy
or numerics; and a full local run is one machine, once.

---

## 2026-11-15 round — measured error rates, and an abstention that was a missing build

Basis: `stage` at `beff10d`. Pins: `tests/test_engineering_hardening_2026_11_15.py`.

- **E-01 without the native kernel, RIME reported "nothing found" — FIXED.**
  `RimeProposer.fit_and_propose` began with `if not _RIME_AVAILABLE: return []`.
  An empty list is also what RIME returns when it searches a null world and
  correctly abstains, so the two were indistinguishable: on a machine without
  the build, a planted `y = 3x` came back as no mechanism, and every test that
  pins an abstention (`tests/test_dyn_trading_universe_sweep.py` among them)
  passed without a search having run. It now raises
  `NativeKernelUnavailableError`. Consequence for anything written here
  earlier: a "RIME abstains" result is evidence only where the kernel was
  loaded. CI builds it; the review environment for these rounds did not have it.
- **E-02 running the tests rewrote committed results — FIXED.** Three paths:
  the API's lifecycle store (`apedp_persistence/`), the known-truth benchmark's
  results file, and the Lorenz benchmark's registered results file and ledger,
  which every smoke test overwrote or appended to. On a machine without the
  kernel the known-truth file was replaced by a sweep in which RIME had not
  run. Tests now write to scratch directories (`APEDP_PERSISTENCE_DIR`,
  `--out-dir`).
- **E-03 a missing build was a failure or a pass, never a skip — FIXED.**
  `tests/conftest.py` reports a test that stops on
  `NativeKernelUnavailableError` or on a missing `apedp_kernel` / `torch` as
  skipped with the reason. CI sets `APEDP_REQUIRE_NATIVE=1`, under which nothing
  is converted and such a test fails.
- **E-04 the gates' error rates had never been measured — MEASURED.**
  `benchmarks/run_promotion_gate_calibration.py` simulates worlds with known
  truth and records how often each evidence-taking gate passes them
  (`benchmarks/promotion_gate_calibration.json`, seed 20261115,
  1000 simulations per stage-2 cell, 95% Wilson
  intervals). The record carries the canonical-spec hash and a test fails when
  the spec moves without a new run.

  Stage 2, share of candidates passed:

  | n | true IC | gate (threshold and five positive folds) | threshold `IC >= 0.02` alone |
  |---|---|---|---|
| 250 | 0.0 | 3.4% [2.4, 4.7] | 40.5% [37.5, 43.6] |
| 250 | 0.05 | 9.3% [7.7, 11.3] | 67.4% [64.4, 70.2] |
| 250 | 0.1 | 24.4% [21.8, 27.2] | 89.1% [87.0, 90.9] |
| 600 | 0.0 | 3.7% [2.7, 5.1] | 33.2% [30.3, 36.2] |
| 600 | 0.05 | 16.1% [14.0, 18.5] | 74.9% [72.1, 77.5] |
| 600 | 0.1 | 45.3% [42.2, 48.4] | 95.7% [94.3, 96.8] |
| 2000 | 0.0 | 2.4% [1.6, 3.5] | 16.9% [14.7, 19.3] |
| 2000 | 0.05 | 38.5% [35.5, 41.6] | 88.4% [86.3, 90.2] |
| 2000 | 0.1 | 86.8% [84.6, 88.8] | 100.0% [99.6, 100.0] |

  The preregistered threshold on its own passes 17-41% of pure noise: at these
  sample sizes an IC of 0.02 is inside the noise. The fold requirement added on
  2026-11-14 is what makes stage 2 a test (2-4% false passes), and it is paid
  for in power: a true IC of 0.05 passes 39% of the time at n=2000 and 16% at
  n=600. Anyone using the numeric `verify_stage_2` with `cv_stable=True` typed
  in gets the right-hand column.

  Stage 4, a candidate with no signal that passed stage 2 by luck:
  3.8% [2.4, 5.9] with independent targets;
  5.2% [3.6, 7.5] with
  21-bar overlapping targets and the horizon *not* declared;
  2.4% [1.4, 4.1] with it
  declared.

  Stage 5, the best of K strategies searched (504 daily observations, 3 bps):

  | true annual Sharpe | K = 1 | K = 20 | K = 200 |
  |---|---|---|---|
| 0.0 | 2.5% [1.1, 5.7] | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.9] |
| 1.0 | 37.0% [30.6, 43.9] | 18.5% [13.7, 24.5] | 18.5% [13.7, 24.5] |
| 2.0 | 83.0% [77.2, 87.6] | 99.5% [97.2, 99.9] | 100.0% [98.1, 100.0] |
| 3.0 | 100.0% [98.1, 100.0] | 100.0% [98.1, 100.0] | 100.0% [98.1, 100.0] |

  Stages 2-5 end to end, share of candidates reaching each stage:

  | world | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|
| null (IC 0.0) | 1.8% | 0.0% | 0.0% | 0.0% |
| planted (IC 0.1) | 68.4% | 27.8% | 22.0% | 9.6% |
| planted_strong (IC 0.25) | 100.0% | 97.6% | 97.4% | 96.0% |

  The referee is safe and not very sensitive: nothing from a null world got
  past stage 3 in 500 runs, and a real but modest signal (IC 0.10, traded daily
  at 3 bps) reaches stage 5 about one time in ten.
- **E-05 stage 4 trusted a declared horizon — FIXED.** The first calibration
  run showed that unrelated 21-bar overlapping series passed stage 4 in 33% of
  trials when the caller left `horizon` at 1. `effective_sample_size` now
  measures the serial dependence in the two series (Bartlett's formula on
  ranks) and stage 4 uses the smaller of that and `n / horizon`. The table
  above is after the fix.
- **E-06 nothing in the Lean tree was machine-checked — PARTLY FIXED.** The
  557-file `formal/` tree declares its own axioms or is never elaborated at
  all; its measured machine-checked fraction is 0.0
  (`docs/LEAN_FORMAL_STATUS.md`), and that has not changed. What is new is a
  core that *is* checked: `lean_checked/` (Lean 4.24.0, toolchain fetched from
  the Lean GitHub release, which the review environment's proxy does permit)
  holds 39 theorems elaborated by the Lean kernel — dimensional-algebra
  consistency and refutations (momentum is not an energy, force is not an
  energy), the promotion machine's no-shortcut and failure-halts-permanently
  properties, and the holdout budget's burn-permanence and single-use
  properties, proved on small models of the Python referees, not on the
  Python itself. `scripts/check_lean_checked.py` re-measures every
  theorem's `#print axioms` against the committed `lean_checked/axiom_audit.json`
  (only `propext`/`Quot.sound`/`Classical.choice`; 12 theorems axiom-free),
  forbids `sorry`/`axiom`/`native_decide` statically, and runs in CI with the
  pinned toolchain (`lean-checked` job, required by `build`). Honest scope: a
  theorem about a model constrains the design, not the implementation. The
  bridge is a test, not a proof: `tests/test_lean_model_conformance.py`
  transcribes each model's transition function and runs it beside
  `PromotionTracker` and `FullLineageExperimentLedger` on every operation
  sequence up to a small length (1,884 promotion sequences plus every
  single deviation from the full ladder; 1,554 holdout sequences under each
  of three budgets), requiring identical refusals and identical state.
  Seeded defects in the real code — stage skipping, a forgotten burn, a
  budget off by one, a burner allowed back, a failed stage re-run — are each
  caught; "halt not enforced" alone is not, because in the Python the
  single-shot rule already implies it. Outside that bound, and for slice
  identity, persistence and concurrency, nothing is claimed.
  `Dimensions.lean` has no counterpart test.

- **E-07 the test workflow had never run a job on `stage` — FIXED, and CI's
  first real verdict is still to come.** GitHub's own record, read through the
  API on 2026-11-15: of the 304 runs of `.github/workflows/test.yml` since
  2026-10-17, 268 failed in 0 seconds with "workflow file issue" — 102 of them
  on `stage`, 82 on `master` — and no run on either branch ever started a job.
  The 36 runs that did execute were on two CI-repair branches, and none
  finished green. Two successive causes, both a context used where GitHub does
  not provide it: `secrets` in a job-level `if:` (already present on
  2026-10-17, since removed), then `RUSTUP_HOME: ${{ runner.temp }}/rustup` in
  the `test` job's `env:` (commit `1bf397b`, 2026-10-20). The `runner` context
  exists only inside steps, so the file was rejected whole. Consequences:
  every sentence in this repository of the form "CI enforces X" described an
  intention for those four weeks — the governance-manifest job, the
  audit-artifact drift job, the mutation battery ("CI-gated, 5/5 killed") and
  the kernel-backed test matrix did not run here — and
  `tests/test_ci_rust_toolchain.py` *required* the invalid line, so the one
  test about it pinned the defect in place. The three commits of 2026-11-12
  to 2026-11-15 in this audit were pushed into the same silence; their
  "CI must confirm" notes were confirmable by nobody.
  Fix: the variable is set in a step through `$GITHUB_ENV`; the test now pins
  that form. Because a workflow file cannot report its own invalidity, the
  check lives in a second file, `.github/workflows/workflow-lint.yml`, which
  runs a checksum-pinned actionlint over every workflow on every push, and
  `tests/test_ci_workflow_validity.py` checks context availability and
  dangling `needs` in pure Python (including a replay of the exact line).
  Found alongside, with the pinned `ruff==0.17.3` that CI installs: `make lint`
  reported 43 findings on a clean checkout where `docs/QUALITY_BASELINE.md`
  pins 0 — 21 import-order findings that depended on a symlink into one
  developer's `.venv` (`apedp_kernel` is now declared first-party in
  `pyproject.toml`), and 22 suppressions the pinned linter no longer needs
  (removed; their written reasons are kept as comments).
  What this does **not** establish: that the workflow is green. It will now
  run for the first time. The governance-manifest job is expected to fail
  until the trustees re-sign (FLAW-01) — that is the gate working — and the
  kernel-backed suites have never been observed on this tree by this audit.

Still not done, and why: the repository split (it moves most files and could
not be verified here without the kernel); elaboration of the `formal/`
transcription tree itself (E-06 adds a checked core beside it, it does not
redeem those files); the data rebuild (needs a download that includes delisted
assets); the trustees' signatures; and the prospective run.

---

## 2026-11-14 follow-up — the items the 2026-11-12 round left open

Basis: `stage` at `9c9c8bf`. Pins: `tests/test_integrity_followup_2026_11_14.py`.
The H-numbers are the ones assigned in the 2026-11-12 section below, whose
entries are kept as written; this section is their current status.

- **H-02 stages took their metrics as numbers — FIXED.** Every stage now has a
  `verify_stage_N_from_*` entry point that computes what it judges: stage 1
  from a `LeanVerificationResult` (kernel exit code, no self-declared axioms, no
  `sorry`, a proven dimensional and a proven bound theorem); stage 2 from
  predictions and targets (Spearman IC, positive in each of five contiguous
  folds); stage 3 from forecasts and literature features (Diebold-Mariano
  against every incumbent, the largest p-value judged; every supplied
  literature feature must be evaluable); stage 4 from holdout predictions and
  targets, claimed through a persistent experiment ledger before scoring;
  stage 5 from gross returns, turnover and timestamps (net Sharpe, breakeven
  fee and annualisation computed here, DSR from a persistent non-empty trial
  ledger); stage 6 from returns whose timestamps fall after the stage-5 record
  and not in the future. The numeric `verify_stage_N` entry points raise unless
  the tracker is built with `allow_reported_metrics=True`; a stage passed that
  way is `caller_reported` and labelled `..._ON_REPORTED_METRICS`.
  `VerifiedModuleLibrary` now admits a module only on an evidence-backed
  promotion and checks the tracker's flags against its records.
  *Limit:* this moves the trust boundary from numbers to evidence objects. A
  caller in the same process can still hand in a fabricated series or a
  hand-built Lean result; what it can no longer do is skip the computation.
- **H-04 slice identity was a name — FIXED, with stated limits.**
  `fingerprint_holdout_slice(data, index=...)` gives a slice a content hash,
  hashes of its non-constant columns, a sketch of its `(index, row)` pairs and
  its index range. A burned or spent slice is recognised under another name
  when the content is identical, when at least half the narrower slice's
  columns are identical, when any indexed row is shared (a sub-window or an
  overlapping window), or when the index ranges intersect within the same
  dataset id. *Not caught:* data that was both rescaled and moved to a
  different dataset id; a row-subset whose index was also relabelled; an
  overlap of under roughly 1-2% between slices of more than 256 rows; and any
  exposure recorded without a fingerprint, unless the ledger was built with
  `require_holdout_fingerprint=True`.
- **H-06 / H-07 the trading sandbox forgot its trials and never registered its
  holdout — FIXED for callers that supply or enable persistent ledgers;
  DISCLOSED otherwise.** `run_epistemic_trading_adjudication` takes
  `trial_ledger` and `experiment_ledger`, or opens the deployment's when
  `APEDP_SANDBOX_PERSISTENT_LEDGERS=1`. It claims the holdout by content before
  scoring, writes the outcome back, and does not charge a replayed search
  twice. Measured: a second seed on the same rows raises
  `HoldoutContaminationError`, so does the same panel under a new dataset name,
  and the refused seed's 80 trials are still charged. The freeze record was
  re-minted for the file (`APEDPX_FROZEN_v1_18.json::remint_log`).
  *Still the default:* with no ledger passed and the flag unset, both ledgers
  are in-memory, because the sandbox's ordinary use -- one dataset, many seeds
  -- is the pattern a persistent single-use holdout refuses. The output now says
  so (`ledger_persistent: false`) instead of leaving it to be inferred.
  `multi_asset_signals_sandbox.py` and the other sandboxes are unchanged.
- **H-12 hindsight-selected universes — MEASURED AND GATED; THE DATA IS NOT
  REBUILT.** `apedp_dyn/market/universe.py` audits a panel, and
  `data/market/PROVENANCE.json` now records `universe_selection` for all six
  multi-asset files (each list was fixed in October 2026 for data starting in
  2021). `verify_stage_5_from_returns` requires a `UniverseAudit` and refuses a
  hindsight-selected one. `tools/build_point_in_time_universe.py` ranks a
  candidate pool using only earlier bars. None of this adds the assets a 2026
  list left out: that needs a pool that includes the names which later died,
  which has to be downloaded and was not available in the review environment.
- **H-13 (new) the 90-asset panel contains bars that never traded — FOUND,
  RECORDED, NOT REPAIRED IN PLACE.** In
  `data/market/multi_asset_90_5yr_orderflow.csv` five crypto pairs end early and
  their last bar is repeated to the end of the file: XMR from 2024-02-21 (468
  bars), MATIC from 2024-09-11 (328), FTM from 2025-01-14 (243), EOS from
  2025-05-27 (152), MKR from 2025-09-16 (75) -- 1,266 asset-days in which close,
  high, low, volume, taker-buy and OFI are constants. The build script's
  `ffill().bfill()` is the cause. The 30- and 60-asset panels and the top-15
  files have no such runs. The file itself is unchanged because
  `tests/test_dyn_trading_universe_sweep.py` and the frozen
  `multi_asset_signals_sandbox.py` read it directly and average across assets
  without NaN handling; `mask_stale_bars` is the loader-side fix, and a test
  pins the five runs so the defect cannot be forgotten. Any result computed on
  this panel past February 2024 includes those bars.
- **H-14 (new) the two quorum signatures share one timestamp — FOR THE OWNER TO
  CONFIRM.** In `specifications/governance_manifest.json` both signature
  records and both transparency-log entries are stamped
  `2026-10-18T16:31:16.664147+00:00`, identical to the microsecond. That is
  what one process assembling both entries produces. If both private keys were
  in that process, the 2-of-3 quorum was one party signing twice, and re-signing
  the current digest the same way would restore a green check without restoring
  what the check is for. FLAW-01 itself is unchanged and cannot be closed from
  inside the repository: the re-signing request for digest `a8177b25…` is
  current (`scripts/run_governance_remint_ceremony.py status`), and no change in
  this round touched the canonical specification.
- **Repository residue — PARTLY REMOVED.** Out of the index: `apedp_kernel` (a
  symlink into one developer's `.venv/lib/python3.12`, dangling in every
  clone), `Crash` (empty), and `apedp/discovery.py` (a three-line shim that the
  `apedp/discovery/` package shadows, so it could never be imported). Tracked
  binaries were inventoried and left: see `docs/REPOSITORY_RESIDUE.md`.

Not run in this follow-up either: suites that need the native kernel or torch,
and the mutation battery. CI has to confirm those.

---

## 2026-11-12 round — integrity safeguards trusted the caller they police

Basis: external review of `stage` at `3d556eb`, each item reproduced by driving
the public API (no monkeypatching). Pins:
`tests/test_integrity_hardening_2026_11_12.py`.

*Items marked OPEN here were revisited on 2026-11-14; the section above gives
their current status. The entries below are left as they were written.*

- **H-01 promotion gates accepted caller-supplied thresholds — FIXED.** A
  candidate with Sharpe −5 and DSR 0 reached `REAL_MARKET_PROMOTION_GRANTED` by
  passing `min_ic=0.0`, `dm_alpha=1.0`, `min_dsr_hurdle=0.0`, … Threshold
  arguments may now only tighten the canonical value.
- **H-02 tracker state was writable and forgettable — FIXED (in-process), OPEN
  (wiring).** Assigning `is_halted = False` un-halted a candidate; a stage could be
  re-run with new inputs; a new tracker forgot the halt. State is now checked
  against the stage records, stages are single-shot, and `state_path` journals
  outcomes. *Still open:* the only non-test caller
  (`sandbox/trading_signals_sandbox.py`, freeze-pinned) hard-codes a stage-1
  refusal, so stages 2–6 are exercised by tests only; and stages 1–4 and 6 still
  take their metrics as numbers. Only stage 5 has a derive-it-here variant
  (`verify_stage_5_from_ledger`).
- **H-03 a holdout was only protected after a self-reported failure — FIXED.**
  Fifty candidates inspected one slice with nothing burned. Confirmation slices
  now carry an inspection budget (default one candidate).
- **H-04 slice identity was the raw string — FIXED (spelling), OPEN (content).**
  A trailing space or a case change produced a fresh slice. Identities are now
  normalised. *Still open:* identity is a name, not the data: the same rows under a
  genuinely different name, or an overlapping date range, are not detected. That
  needs slices registered by content hash and row/date range.
- **H-05 the ledger file could be edited without detection — FIXED (evidence).**
  Deleting the burn line made a burned slice reusable. Entries are hash-chained
  with a head marker; damage raises `LedgerIntegrityError`. This is tamper
  *evidence*: someone who rewrites the ledger and its head marker together is not
  stopped. Anchor `head_hash` outside the writer's reach for that.
- **H-06 trial multiplicity restarts with every in-memory ledger — PARTLY FIXED.**
  The best of 1,000 noise strategies scored DSR 0.537 on the ledger that saw the
  search and 1.000 on a fresh one. Persistence is now disclosed and required by
  `verify_stage_5_from_ledger`. *Still open:* every non-test
  `CandidateTrialLedger()` construction is still in-memory (the sandboxes are
  freeze-pinned), and `selection_pool_size` is declared by the proposer.
- **H-07 the trading sandbox never registers its holdout — OPEN.**
  `trading_signals_sandbox.py` scores on `X_ho` without a
  `record_data_exposure` call, so the guard above does not see it. The file is
  freeze-pinned; fixing it is a re-mint.
- **H-08 formula parser hang — FIXED.** `9**9**9**9`, `zeta(10**7)`.
- **H-09 governance gate crashed before comparing hashes; deploy did not depend on
  it — FIXED.** FLAW-01 itself (the missing trustee signatures) is unchanged.
- **H-10 README claimed Lean formal verification and shipped a broken example —
  FIXED** (documentation only; the Lean layer is as `LEAN_FORMAL_STATUS.md`
  measures it).
- **H-11 keyless UI API reachable off-loopback when launched via uvicorn — FIXED.**
- **H-12 hindsight-selected market universe — OPEN.** The 90 assets in
  `scratch/fetch_and_build_90_assets_5yr.py` are names that survived 2021–2025.
  Not a code defect; backtests on that file carry survivorship bias until the
  universe is rebuilt point-in-time.

Not run in this round: anything requiring the native kernel or torch (neither
could be built or installed in the review environment). Kernel-dependent suites
and the mutation battery must be confirmed by CI.

---

## 2026-11-08 round — status matrix (Categories 1–5)

- **Category 1 (Governance & Spec Synchronization):**
  - `ARCHITECTURE.md` stale SHA-256 reference updated from `a6d026f3…` to canonical `dbf67c5ac443073721421eba456519a0eda09ae07680d638c26f1d019b0b2343`.
  - `specifications/governance_manifest.json` remains fail-closed pending external trustee Ed25519 signature ceremony via `scripts/run_governance_remint_ceremony.py` (never forged locally per `AGENTS.md` §1.3).
- **Category 2 (Epistemic Referee, Holdout & Ledger Integrity):**
  - `apedp_dyn/epistemic/adjudicate.py`: `_identity(sc)` now canonicalizes `_ProposalReference` and duck-typed proposal objects via SymPy (`_canon_expr`); holdout re-anchor selects the best distinct holdout rival (`min(_rivals_ho, key=lambda s: s.holdout_rmse)`); `equiv` in the bootstrap `MULTI_HYPOTHESIS` branch anchors on `best_rmse`; `selection_reuse_flag` is attached across all post-selection pair verdicts.
  - `apedp_dyn/epistemic/caller_data_protocol.py`: `SealedConfirmationSlice.consume()` now derives `outcome_status` (`"PASSED"` vs `"FAILED"`) from the referee verdict so rejected confirmation slices are permanently burned.
  - `apedp_dyn/epistemic/experiment_ledger.py`: `_persist_append` holds `fcntl.flock(LOCK_EX)` across `_refresh_burned_holdouts_locked`, holdout contamination check, JSONL write, `flush()`, and `os.fsync()`, and forces immediate `sync_to_remote()` on holdout burn; added `flush()`, `close()`, and context-manager protocol.
  - `apedp_dyn/epistemic/trial_ledger.py`: `record_trial` uses `fcntl.flock(LOCK_EX)` + `os.fsync()`; `get_sharpe_distribution()` filters non-finite values and excludes equation-discovery dummy `0.0` Sharpe records; added `get_selection_distribution()` and guarded negative/non-finite variance in `compute_expected_max_sharpe`.
  - `apedp_dyn/epistemic/promotion_lifecycle.py` & `apedp_dyn/conf/canonical_spec.py`: Stage 2/4/5/6 thresholds exported from `canonical_spec.py`; `verify_stage_4` enforces Stage 2 `in_sample_ic` lower bound (`PromotionViolationError` on deflation spoofing) and rejects non-positive/non-finite `in_sample_ic`; `verify_stage_6` bounds `0.0 <= max_drawdown <= max_dd_ceiling`.
  - `apedp_dyn/epistemic/lean_universal_bridge.py`: removed top-level `sys.path.insert(0, ".")`; `_ast_detect_damping_factor` detects N-ary `sp.Mul` damping factors; `infer_invariants` uses identifier tokenization; `BOUND_POSITIVITY` numerical probe evaluates over all free variables on a multi-dimensional grid and fails closed on exception.
- **Category 3 (API, UI Server & Security Hardening):**
  - `apedp/api/__init__.py`: `_trust_boundary` authenticates `X-API-Key` before rate-limiting, exempts `AUTH_EXEMPT_PATHS` from rate-limiting, evicts stale IP deques in `_ClientRateLimiter`, and enforces `MAX_REQUEST_BODY_SIZE` on streaming request chunks.
  - `apedp/api/routes/pipeline.py`: `_validate_safe_pipeline_path` blocks path traversal (`..`) and restricts `csv_path`/`output_dir` to allowed roots.
  - `apedp/api/routes/discovery.py`: `_finite_or_none` sanitizes `NaN`/`±Inf` in `symbolic_regression` and `bayesian_comparison`.
  - `apedp_dyn/ui/server.py`: `_require_api_key` uses `hmac.compare_digest` and supports `Authorization: Bearer`; `ext60_run` uses `tempfile.TemporaryDirectory`; `get_ladder` marks static reference rows with `"is_static_reference_fixture": True`.
- **Category 4 (Build, Packaging, Docker & CI Parity):**
  - `Dockerfile`: multi-stage build compiling `rust/apedp-kernel` wheel, copying `apedp/`, `apedp_dyn/`, and `specifications/`, running as non-root `appuser`.
  - `pyproject.toml`: aligned `maturin>=1.7,<2.0`, `torch>=2.5`, and removed non-existent `optax-contrib>=0.1`.
  - `.github/workflows/test.yml`: expanded Bandit scan to `apedp/` and `apedp_dyn/`, and added Rust toolchain `1.99.0` + `maturin develop --locked` to `mutation-battery`.
  - `scripts/run_mutation_battery.py`: normalized `pin_argv[0]` in `_cache_key()` and added `("round3", PIN_ROUND3)` to the pristine sanity check.
- **Category 5 (Scientific & Algorithmic Module Correctness):**
  - `apedp/discovery/_bayes.py`: `score_candidate` counts non-trivial numeric constant atoms in SymPy expressions when `num_params is None`.
  - `apedp/rust_bridge.py` & `apedp_dyn/rust_bridge.py`: `enforce_native_search` synchronizes `_ENFORCE_NATIVE_SEARCH` across both modules, exports `is_native_search_enforced`, and routes `score_candidates_batch` through `self._kernel_compile`.
  - `apedp_dyn/proposers/sparse_microstructure_proposer.py`: respects `target_col` and includes unstandardized `raw_intercept` in `_score_fn`.
  - `apedp_dyn/gates/scaling_invariance.py`: `check_with_pi_groups` validates `pi_groups` exponent dimensions against `alpha`.
  - `apedp_dyn/epistemic/shape_inference.py`: removed dead `_prefix_sums`/`_flat_run_stats` calls and duplicate `_dp_segment` invocation.
  - `apedp_dyn/robotics/cbf_certificate.py`: `unicycle_commands_from_certified` and `model_check_unicycle_horizon` perform heading-aware pairwise barrier braking and accept `nominal_velocities`.

---

## 2026-10-21 round — status matrix

- FLAW-01 governance manifest SHA desync — **OPEN (trustee-blocked; ceremony
  mechanism now UNBLOCKED, 2026-10-22)**: the Ed25519 quorum in
  `specifications/governance_manifest.json` still seals the deprecated spec
  hash `a6d026…` while disk computes a later hash. Re-signing requires the
  authorized trustee PRIVATE keys, which live outside this repository; per
  AGENTS §1.2/§1.3 the panel must never bypass or re-mint the manifest
  locally. Earlier root-cause diagnosis stands: post-certification spec edits
  without re-certification (`pruning_holdout_*` block). The FLAW-13
  spec-centralization edit below extended the desync scope with full §1.3
  procedural compliance (new pinned hash + CHANGELOG rationale).
  **2026-10-22 structural gap closed (mechanism only):** the re-certification
  PROCEDURE now exists and is pinned —
  `scripts/run_governance_remint_ceremony.py` (audit-only, fail-closed:
  `--status` / `--request` / `--collect` / `--mint`), verified against the
  REGISTERED public keys only, chaining `PREREGISTRATION_RECERTIFICATION`
  entries through the production transparency-log API and writing the
  manifest only after full production verification + quorum. It never
  generates keys, never forges signatures, never touches the spec JSON, and
  refuses to run on pin disagreement. Pinned in
  `tests/test_dyn_governance_remint_ceremony.py` (5 pins: sandbox end-to-end
  re-certification, forgery/unregistered refusals, below-quorum refusal,
  pin-drift fail-closed, and the live-repo guard proving the repo CANNOT be
  minted without real trustees). Only the external signature ceremony with
  the two registered trustees remains to flip the red.
- FLAW-02 coupled ODE proposer discarded discovered RHS — **FIXED**:
  `apedp_dyn/ode/proposer.py` now evaluates `sum(coeff · basis(x))` over the
  kernel's term encoding; verified RMSE 0.0000 vs the true derivative
  (identity previously had RMSE 1.67 on the same probe).
- FLAW-03 non-atomic S3 mutex — **FIXED**: conditional put `IfNoneMatch="*"`
  (412 → contested) with read-back token verification as a universal fallback;
  the parameter-rejection fallback now also issues the unconditional put
  before the read-back (v2, 2026-10-21: the first draft skipped it and every
  acquire on old-botocore backends deadlocked until timeout); exactly one
  worker can win. Mutation pins `MUT-D1`/`MUT-D2`
  (`scripts/run_mutation_battery.py`, killed).
- FLAW-04 non-uniform-grid FD bias — **FIXED**: uniformity gate (1e-4 relative)
  with Fornberg arbitrary-grid weights fallback in
  `rime/diff_pade.rs::compute_derivatives/compute_second_derivatives`.
  Remediation v2 (2026-10-21, from the mutation-pin run): the first Fornberg
  translation was structurally wrong — the returned weights no longer
  annihilated constants (3-node probe {-0.435, 1.769, -0.0015} vs the exact
  {-478.26, +589.74, -111.48}), so the fallback route was numerically ~1e3x
  off at the stencil edges; `fornberg_weights` now mirrors the proven Python
  mirror `_fornberg_centered_weights` term-for-term, and
  `test_nonuniform_grid_derivatives_use_fornberg_fallback` holds the
  exactness contract (quadratics differentiated exactly on a 1.3x-per-step
  non-uniform grid; MUT-E1/MUT-E2 kill the sign and gate mutants).
- FLAW-05 constant-field R² = 1.0 tautology + STRidge pruning defeat —
  **FIXED**: zero residual may claim 1.0, any real residual on a flat target
  scores 0.0 (`rime/pde.rs`, `pde_discovery.py`, `pde_transfer.py`); a failed
  linear solve (se = 0) disables the significance rescue instead of
  unconditionally defeating pruning.
- FLAW-06 pole masking + uncentered Koopman R² — **FIXED**: pole evaluation
  now sets `pole_singularity_hit` on the result and disqualifies exact
  recovery; `koopman.rs` centers ss_tot and scores 0.0 on zero variance.
  Mutation pins (2026-10-21): the pole-flag report contract is pinned on BOTH
  rollout branches — Bessel-J0 forces the 2nd-order branch (S(x)=x vanishes
  at the rollout start; `test_pole_masked_second_order_is_not_exact_recovery`)
  and the parabola with `max_diff_order=1` forces the 1st-order branch
  (`test_pole_masked_first_order_reported`); MUT-E3/E3b kill the flag-drop
  mutants on each branch (the earlier only-2nd-order pin missed the branch
  the sparse selection actually picks — the battery exposed it, the branch
  pair now covers both sites).
- FLAW-07 C6 leapfrogging — **FIXED**: C6 requires `EQUIVALENCE_RIVALS_REJECTED`
  and `INTERVENTIONAL_INVARIANCE` (monotone strictness); ladder test fixture
  updated to the strengthened apex contract.
- FLAW-08 silent bootstrap disable on re-anchor — **FIXED**: on re-anchor the
  referee recovers the paired gate-4 residual series for the new head/rival and
  runs the block-bootstrap tiebreak; unpairable pairs keep the previously
  disclosed honest point-estimate contract (marked INCOMPLETE).
- FLAW-09 Hawkes silent return-type mutation — **FIXED**: an eval-frame binary
  collapse now returns the honest collapsed binary (zeros), preserving the
  declared {0,1} contract; the fit-time declared-continuous path is unchanged
  (pinned contract).
- FLAW-10 circular axiom injection in the Lean core bridge — **FIXED**:
  core-mode axiomatic theorems (`lean_ring_tactic`, `axiomatic_*`) are labeled
  `syntactic_schema_check::…` with an explicit no-Mathlib-independence note in
  the diagnostic; kernel-checked core Int/Nat lemmas keep honest names.
- FLAW-11 odd-pole limit cancellation — **FIXED**: opposed signed one-sided
  probes with non-negligible magnitude raise `LimitTimeout` instead of
  averaging a fake finite limit; the previously pinned canceling-average 1/x
  assertion was updated to the sound contract.
- FLAW-12 inactive columns inflating the exact-RMSE ceiling — **FIXED**:
  `rime/implicit.rs` measures term scale over ACTIVE non-constant columns only.
- FLAW-13 ODE referee hardcoded thresholds + misleading verdict — **FIXED**:
  defaults now bind to preregistered spec keys `ode_referee_min_mean_r2=0.95` /
  `ode_referee_min_worst_r2=0.85` (spec re-pinned to `5bb42f03…` per §1.3);
  `UNSTABLE_DIVERGENT` covers mean holdout R² < 0.
- FLAW-14 toroidal Nyquist→DC wrapping — **FIXED**: one-sided upwind stencils
  with zero-flux Neumann boundaries in `spectrogram_transport.py`.
- FLAW-15 unnormalized SVD tolerance — **FIXED**: nullspace gate is relative
  (`|λ| ≤ tol · λ_max`) in `rime/lie_symmetry.rs`. Remediation v2 (2026-10-21,
  from the mutation-pin run): the v1 relative gate left the EIGENSOLVE itself
  scale-blind — on an O(1e6)-amplitude ODE the Gram matrix spans strata up to
  O(1e24), the absolute Jacobi convergence test never fires, the rotation
  budget depletes, and a structural zero was returned as eigenvalue ~1.0
  with nullspace basis residual ~7e5 (rel-gate then counted 9 instead of 4
  genuine directions). Fix: `discover_point_symmetries` now solves the
  CORRELATION matrix (D⁻¹GD⁻¹, diag exactly 1; congruence preserves the
  kernel with generators mapping back as w = D⁻¹v) and `jacobi_eigen`
  pre-normalizes to unit spectral scale (eigenvalues rescaled back), so eps-
  relative accuracy is meaningful at every stratum; corr-spectrum probe:
  structural nonzeros sit at O(0.5) for both O(1) and O(1e6) amplitudes
  (4 genuine directions in both — the scaled system now reports the same
  symmetry verdict as the unscaled one; MUT-E6 kills the raw-Gram revert).
  Honesty trail: the absolute-tolerance-gate mutant (MUT-E5 candidate) was
  registered and survived with NO natural kill — the v2 well-scaled eigensolve
  dissolved the defect expression it targeted (nothing lands in the ≤1e-8
  band), so it was removed with the rationale recorded in
  `scripts/run_mutation_battery.py` rather than shipped as a silent survivor.
- FLAW-16 Hamiltonian archetype mismatch — **FIXED**:
  `HamiltonianProposer` emits `CONTINUOUS_FACTOR` to match the pinned
  acceleration-law contract.
- FLAW-17 CBF post-correction norm violation — **FIXED**: a final norm-ball
  projection closes the alternating scheme before the fail-closed checks.
  Remediation v2 (2026-10-21, from the mutation-pin run): the global
  re-scale is provably insufficient — scaling the corrected endpoint shrinks
  the deflection that repaired its pair and re-breaks the barrier — so the
  close-out now distributes each remaining deficit BETWEEN ITS ENDPOINTS,
  each share bounded by the endpoint's direction-aware norm-ball headroom
  (`budget(v,d) = -(v·d) + sqrt((v·d)² − (|v|² − box²))`, equalities
  preserved, |v| ≤ v_max exactly). The remedy runs ONLY when the alternating
  loop exhausted with pending corrections; `max_iter=0` still reaches the
  fail-closed verdict refused (pin in `tests/test_dyn_cbf_certificate.py`).
  Mutation pin `MUT-D3` (killed).
- FLAW-18 JSONL concurrency + O(N²) sync — **FIXED**: `fcntl.flock(LOCK_EX)`
  wraps every append (graceful degradation where fcntl is absent); remote
  sync is amortized (batch of 16 pending appends) and flushed before snapshots.
- FLAW-19 dead gradient + internal variable leak — **FIXED**: the unassigned
  `np.gradient(I_arr, …)` call is removed and the SIR formula uses physical
  notation (`i_col`).

Verification: Rust-side fixes pass `cargo test` (186/186) and every affected
Python suite re-run is recorded in the 2026-10-21 CHANGELOG entry.

---

## 2026-10-21 follow-through round (Set A sweep + FLAW-STRAITA)

Independent verification audit of the items above plus the historical Set A
catalog, with line-by-line confirmation. Final remaining items addressed (or
honestly re-documented) the same day, execution-verified:

- FLAW-A01 active 5-day lookahead in 4 train-block consumers — **FIXED**:
  `ensemble_gated.py`, `ensemble_m.py`, `incremental_info.py` and `e5_market.py`
  built train blocks as `np.intersect1d(_blocks(len(dates)), tr)` with the raw
  `dates < origin` mask, while the `_d5` target forward-shifts the
  5-day rolling mean (`shift(-5)`); blocks landing within 5 days of the origin
  boundary `[origin-5, origin-1]` read forward rows `[origin, origin+4]` —
  look-ahead contamination of every fit in those modules. Fixed by the shared
  helper `trading_e4t1_v2::_train_blocks_no_leak` (mirrors `run_origin`'s
  original corrected mask `tr[tr + 5 < origin_idx]`), applied at every train
  block call site in the four modules. Pins:
  `tests/test_dyn_external_audit_2026_10_21_pins.py::
  test_flaw_a01_corrected_mask_excludes_five_day_boundary_blocks`
  (behavioral, with the pre-fix leak demonstrated inline) and
  `…::test_flaw_a01_all_four_consumers_use_corrected_mask` (text pin).
- FLAW-A09 sibling lag-term buffer overwrite in streaming scoring — **FIXED**:
  `apedp_dyn/proposers/distributed_lag_interaction_proposer.py::compute_score`
  wrote `lag_state["buffers"][base_col]` on the FIRST lag term of a call, so a
  second lag term on the same base column (`lag_1_x` + `lag_2_x` both selected)
  read that same call's own tail as prior history and prepended future batch
  data to the head of the array (both on the full frame and in every chunk).
  The buffer is now snapshotted pre-call per (call, base column) and written
  once per (call, base column). Pin:
  `tests/test_dyn_expanded_proposers_and_calibrated_power.py::
  test_distributed_lag_multi_lag_same_column_streaming_safe`
  (chunked == single-pass with two lag terms sharing one column).
- FLAW-A13 zero-denominator blind metrics masked as certified 0.0 — **FIXED**:
  `apedp_dyn/sandbox/blind_benchmark.py` per-domain breakdowns reported
  `fpr: 0.0` when a domain had zero negative controls (and likewise
  `fdr`/`vrr` on empty denominators) — falsely certifying a 0% error rate
  where nothing was measured. The per-domain metrics now report `None` on a
  zero denominator, matching the headline metrics' existing convention; the
  affected pins updated to the honest contract. Pins:
  `tests/test_dyn_blind.py::test_zero_denominator_domains_report_none_not_certified_zero`
  plus the repaired `test_frr_coverage_and_domains`.
- FLAW-B01 governance quorum desync — **STILL OPEN (trustee-blocked)**:
  unchanged from FLAW-01 above: the manifest still seals the deprecated spec
  hash while the disk spec hashes differently; re-signing requires trustee
  private keys outside this repository. Documented, not bypassed.
- FLAW-STRAITA (exact-RMSE gate audit, this session)
  — **FIXED**: auditing MUT-E4's survival in
  `scripts/run_mutation_battery.py` exposed that `rime/implicit.rs` still
  carried two COMPENSATING anchored gauges in raw-coefficient space: (1) a
  `max_c = max|c_j|` rescale applied AFTER the strata-coherent
  `c_j = v_j/norm_j` mapping (raw coefficients span strata: const ~O(1/√n) vs
  x² ~O(1/norm); at O(1e6)-scale strata the gauge
  annihilated the small-coefficient large-stratum terms and the pure 1e6
  circle degenerated to the constant-only fallback `-1 = 0` with rmse 1.0 —
  refusing a genuinely exact relation), and (2) a raw `|c_j| > 1e-6`
  active-count recount that the max_c gauge had been accidentally inflating
  above threshold. Removing only (1) exposed (2)
  (`test_large_scale_exact_relation_certifiable` failed with rmse 3.6e-17);
  both gauges were replaced by contribution-space equivalents inside one
  coherent rule: no further normalization after `v/norm` (the Jacobi
  eigenvector is already 2-normalized), and membership by contribution ratio
  (`|c_j|·norm_j > 1e-6 · max_j(|c_j|·norm_j)`). Post-fix probes (rebuilt
  extension): pure 1e6 circle certifies at rmse 4.1e-17, near-zero-scale pure
  circle 1.6e-18, unit circle 5.7e-16 unchanged; MUT-E4's kill is restored on
  the natural near-miss fixture (pristine refuses the 3.16e-12 sin(2θ) misfit
  at rmse 1.58e-8 > contracted ceiling; the mutant's inflated ceiling
  certifies it, the conditional pin rejects the leaked certificate).
  Pins: `test_flaw_straita_implicit_rs_raw_gauges_removed` (text) and
  `test_flaw_straita_extreme_scale_circle_certifies_not_degenerate` (bridge).

Verification round: `cargo test --lib` 193/193; mutation battery
ALL CONFIGURED MUTANTS KILLED (MUT-E4 back among them);
`scripts/check.sh` full tier green end-to-end (tier-1 pytest 125 passed,
battery all-killed, Rust 193/193); pins 27/27; blind + expanded-proposers
suites 24 passed.

### Code-only external rating (8.1/10) — blocker-list closure mapping

A subsequent code-only external assessment rated the system 8.1/10 and
listed "what prevents 9.0+" as four verified code flaws. Same-day closure
mapping (source-verified, 2026-10-21):

1. "Active Lookahead Leakage in 4 Modules" — RESOLVED above (FLAW-A01):
   the raw `np.intersect1d(_blocks(len(dates)), tr)` pattern has ZERO hits
   in the package; all four modules go through
   `trading_e4t1_v2::_train_blocks_no_leak`, pinned by
   `test_flaw_a01_corrected_mask_excludes_five_day_boundary_blocks` and
   `test_flaw_a01_all_four_consumers_use_corrected_mask`.
2. "Distributed Lag Buffer Keying" — RESOLVED above (FLAW-A09): buffers are
   snapshotted pre-call and written once per (call, base column); pinned by
   `test_distributed_lag_multi_lag_same_column_streaming_safe`.
3. "Benchmark Division-by-Zero Semantics" — RESOLVED above (FLAW-A13):
   per-domain fpr/fdr/vrr report `None` on zero denominators; pinned by
   `test_zero_denominator_domains_report_none_not_certified_zero`.
4. "Governance Manifest Re-Signing" — REMAINS OPEN (trustee-blocked):
   re-verified live — `tests/test_dyn_governance.py` runs 3 passed / 1
   failed with `GovernanceTamperingError: Specification SHA-256 mismatch:
   manifest specifies a6d026…, but actual disk file computes to 5bb42f…`
   (the disk hash matches `canonical_spec.py`'s import pin, per FLAW-01
   above). Resolution requires renewed trustee Ed25519 signatures held
   outside this repository; per AGENTS §1.2/§1.3 the manifest is never
   re-minted or bypassed locally.

---

# Previous round (2026-09-18, second pass)

Date: 2026-09-18 (second remediation pass)
Basis: code + test execution only. No documentation was used as an evaluation source.

## Test / Quality Baseline

- Tests: `288 passed, 0 failed` in ~48s [was `257 in ~55s`].
- Numeric noise: `0` floating-point `RuntimeWarning` in a passing run [was `125`].
  The two remaining warnings are third-party deprecation notices from
  `starlette.testclient` / `httpx`, not this project's numerics.
- `ruff check .` = `All checks passed` across the whole repository.
- `mypy apedp/ --ignore-missing-imports` = **0 errors in 35 files** [was 5
  `no-any-return` in 2 files]. The type check is a CI gate
  (`.github/workflows/test.yml`), so it is guarded by the pipeline step rather
  than by a pytest test.

> **2026-09-23 re-baseline — the three claims above are STALE.** Measured on the
> current tree: `ruff check apedp tests` = **416 errors**, `mypy apedp
> --ignore-missing-imports` = **266 errors in 23 files**, and the suite is **474
> collected / 451 passed / 19 failed / 4 skipped** (the 19 are missing optional
> extras: `jax`, `sklearn`, `scipy`). The snapshot above describes a 35-file
> tree; the package now has 108 source files. It was not maintained as code was
> added.
>
> The CI gate has also been corrected as of 2026-09-23: `.github/workflows/test.yml`
> now triggers on every branch (`branches: ['**']`, widened again after this note
> was written) instead of a branch that does not exist, so the
> `pytest`/`ruff`/`mypy`/`bandit` steps can actually fire.
> Full analysis: `docs/INDEPENDENT_REVIEW_2026-09-23.md` §2.
- Frozen benchmark metrics are **unchanged by both passes**: 25-case suite
  detection `1.000`, refusal `0.5833`, repair success `0.9231`, false repair
  `0.000`, trust `98.077`; 100-case suite `1.000` / `0.6190` / `0.9747` / `0.000` /
  `99.367`, infeasibility refusal `1.000`.

## Remediation Status

Every finding below is listed with what it was and where it now stands. Statuses
are deliberately narrow: "fixed" means a test in `tests/test_system_flaws_audit.py`
(plus, for the falsification findings, `tests/test_evidence_tiers.py`) fails if the
defect returns. Those guards were mutation-tested — the defect was reintroduced and
the test was required to fail — which is recorded per finding.

### Critical — Breaks Use

1. **Docker boot failure — FIXED**
   `Dockerfile` invoked `uvicorn apedp.api:create_app` without `--factory`, so the
   container exited on boot having tried to serve the factory function itself.
   The CMD now passes `--factory`. Also fixed in the same file: the
   `pip-compile ... || true` step (which hid a failing resolve while producing a
   `requirements.txt` nothing consumed) is gone, and `COPY tests/` no longer ships
   the suite in the runtime image.

2. **API trust boundary — FIXED, but opt-in by default**
   CORS was `allow_origins=["*"]` with `allow_credentials=True`, a pair browsers
   reject outright. Origins now come from `APEDP_CORS_ORIGINS` (comma-separated),
   the default is same-origin, and a wildcard is honoured only with credentials
   disabled.
   *Second pass:* the missing authentication and throttling are now implemented
   rather than deferred. `APEDP_API_KEY` requires a constant-time-compared
   (``hmac.compare_digest``) `X-API-Key` header on every non-exempt route, and
   `APEDP_RATE_LIMIT_PER_MINUTE` applies a sliding-window limiter that answers
   `429` with `Retry-After`. `/health`, `/docs`, `/redoc` and `/openapi.json` stay
   reachable, because a health check that needs a credential cannot gate traffic.
   **Still open as a posture, not as a capability:** both are opt-*in*, so an
   unconfigured deployment is still an open, unthrottled server — which is the
   right default for a single-user local install and is now stated plainly (startup
   logs a warning when no key is set) instead of left implicit. The limiter is
   in-process, so it bounds one worker and is not a substitute for an edge limiter;
   that limitation is documented in the module docstring.

3. **Silent unit bypass — FIXED (disclosed, not removed)**
   `_normalize_variable_value` accepted a bare `float` for a unit-bearing variable
   and interpreted it in the registered unit without saying so. The interpretation
   is now explicit: the assumption is logged once per variable and recorded in
   `ClosedLoopAssuranceSystem.unit_assumptions`, and `strict_units=True` refuses
   the bare number with a message showing how to pass a `pint` quantity.
   The permissive default remains by design — every internal call site passes bare
   numbers — but a wrong-unit evaluation now leaves a trace instead of a silent
   result.

4. **Dimension vector overflow — FIXED**
   Dimension vectors were `np.int8`, so exponents wrapped silently
   (`np.int8(100) * np.int8(2) == -56`): a large exponent could make an
   inconsistent model pass the homogeneity check. Vectors are now `np.int64`;
   `200 * DIM_MAP['meter'] == 200`.

### Major — Wrong / Misleading Science

5. **Stagnated search still persisted — FIXED**
   A search that never improved on its seeded population reported `PERSISTED`,
   filing an initialisation artefact as a search result. `DiscoveryPipeline.run`
   now refuses with `status="REFUSED"` and a reason naming the stagnation, and
   nothing reaches the model store.

6. **Limited constant discovery — OPEN (capability, not defect)**
   ERCs remain dimensionless and `inner_fit` is still off by default, so
   `exp(19.5*x)`-style inner scales need the caller to enable the flags. The
   limitation is real and documented; it is a capability gap rather than a
   correctness bug.

7. **Over-broad error swallowing — FIXED on both surfaces**
   The falsification axes were fixed in the first pass: all five axes publish a
   uniform `probe_errors` channel, seven silent `except: pass` sites became
   recorded errors with narrowed exception types, and a verdict with an unresolved
   probe cannot claim `SURVIVED_PROVISIONAL` (`INCONCLUSIVE_PROBE_ERROR`).
   *Second pass:* `repair.py` is closed too. Its `PROBE_SWALLOWED_ERRORS` boundary
   (7 types, 8 sites) now records every swallowed probe into a `probe_errors` list
   with the symbol, the probed point, the exception and whether the question was
   left **unanswered**. The audit's concern — that a real defect could read as
   *unknown* and be filed as clean — was exactly right: the boundary audit used to
   assume "not singular" whenever a probe raised, and returned `APPROVED`. An
   unanswered probe now yields the distinct status `UNDETERMINED`, and
   `DiscoveryPipeline.run` treats that as a kill signal (it refuses, and nothing is
   persisted). A crashed probe that a *different* measurement resolves (a limit
   that times out but whose denominator vanishes under direct substitution) is
   recorded and marked resolved rather than blocking, so recording every crash did
   not become failing every crash: on the repository's own repair fixtures the
   statuses and metrics are unchanged (`APPROVED` for `1/(x**2 + 1)`, `REPAIRED`
   for `1/x`).
   *Mutation-guarded:* reverting `UNDETERMINED` to `APPROVED` fails
   `test_an_unanswered_probe_cannot_read_as_approved` and
   `test_pipeline_refuses_a_repair_it_could_not_establish`.

8. **Safety / timeout tradeoffs — PARTIALLY FIXED**
   The substring block was fixed: forbidden *identifiers* now match on word
   boundaries, so `importance`, `lambda_eff`, `deflection`, `getattr_index` and
   `compile_time` parse while `__import__`, `import os`, `eval(`, `lambda x: x` and
   `a;b` are still rejected. `limit_with_timeout` still abandons its daemon thread
   on timeout; that is inherent to bounding wall-clock latency in-process and is
   documented, not silently absorbed. That abandoned thread is also the source of
   most of the former warning noise, which is now contained (see the noise item
   below) rather than suppressed globally.

9. **Persistence loss — FIXED**
   Fixed in the first pass: `list_models()` reads the store directory, so a fresh
   process sees models saved earlier; model IDs use microsecond resolution plus a
   collision guard; `ExperimentLog` timestamps are timezone-aware UTC.
   *Second pass:* durability and shape. Saves are now atomic — write to a sibling
   temp file, `flush`, `os.fsync`, then `os.replace` — so an interrupted save can
   no longer destroy the previous model or expose a half-written record
   (`json.dump` wrote in place, and the earlier file was already gone by then).
   Log rotation uses `os.replace` instead of read-then-`write_text("")`, which
   truncated the live log in place, and every appended entry is flushed and
   fsynced. `ModelStore.load_model_expr()` returns a real `sympy.Expr` by parsing
   the stored expression through the sandboxed parser, so a caller no longer has to
   reconstruct a model from the record dict (whose `sympy_expr` field is LaTeX,
   not an expression); `load_model()` keeps its record-shaped contract and says so.
   A corrupt record returns `None` and logs, rather than raising out of a read path.
   *Mutation-guarded:* making `os.replace` raise mid-save fails
   `test_a_crash_during_save_leaves_the_previous_file_intact`.

10. **Unitless ingest — PARTIALLY FIXED**
    The CSV path still carries no units and defaults to `dimensionless`, so
    dimensional assurance is void unless the caller supplies `variable_units`.
    *Second pass on the arbitrary threshold:* the `>20%` bad-row rule is now the
    named constant `DEFAULT_MAX_BAD_ROW_FRACTION`, documented as **chosen, not
    derived** (with the reasoning: a file more than a fifth unparseable is likelier
    to be the wrong file than a clean table with sporadic blanks), and overridable
    per pipeline via `DiscoveryPipeline(max_bad_row_fraction=...)`, and the error
    message names the knob.

### Minor — Correctness / Maintainability

| Finding | Status |
|---|---|
| `core.py` `base_unit ** float(exponent)` loses exact rationals | **RESOLVED as intentional, now pinned.** The proposed fix was implemented and *measured*: pint stores an exact sympy rational (`1/3`) but compares it **unequal** to the equivalent float, so returning the exact rational makes `sqrt(x)` and `x**0.5` disagree about one dimension — a false dimensional error on legitimate models. The float normalisation is what keeps `sqrt(x)`, `x**0.5` and `x**Rational(1,2)` consistent, and `test_sqrt_a_half_power_and_a_rational_exponent_agree` pins both facts so a well-meant "exactness" change cannot silently break homogeneity |
| `core.py` variable exponent raises `NotImplementedError` | OPEN (explicit refusal, not silent wrongness) |
| `BuckinghamPiEngine` returns formatted strings, not symbolic groups | OPEN |
| `mypy` `Returning Any` | **FIXED — 0 errors.** The root cause is that `float ** float` is typed `Any` by mypy; `applied_defects.py` now coerces the scalar, and the four benchmark evaluators return `np.asarray(..., dtype=float)`, matching the project's existing convention |
| `metrology.py` singularity warning keyed only on `Pow -1/-2`, `log`/`sqrt` | **FIXED.** Replaced by a structural rule in `metrology.has_singularity_risk`: any power whose exponent is not a positive integer is a pole or branch cut, plus `log`/`tan`/`asin`/`acos`/`acot`. `1/x**3`, `x**Rational(1,3)`, `x**Rational(-1,2)` and `tan(x)` were all classified safe before. Advisory only — it decides whether to attach a warning, never whether propagation is allowed |
| God-files: `gated_assurance.py`, `falsification.py`, `benchmarks.py` | OPEN (all three grew again in this remediation) |
| Suite ~48s, 125 numeric warnings emitted by passing tests | **FIXED — 0 numeric warnings.** The cause was real work, not sloppiness: probes deliberately evaluate models at their singular points, and numpy reported every one of those. Silencing numpy globally would have hidden genuine numerical problems, so the suppression is scoped to the probe with `apedp.safety.quiet_float_warnings` (a context manager *and* decorator) applied at the five falsification axes, the symbolic-regression candidate evaluator, the outer-coefficient fit and the numeric limit fallback. `test_deliberate_singular_evaluations_are_silent_but_real_ones_are_not` requires the suppression to be *scoped*: a warning raised outside the probe is still reported. Separately, `is_degenerate` now names the empty-input case instead of letting `np.std` warn over nothing |

## What Was Not Changed For This Audit

The remediation after the first audit touched `Dockerfile`, `apedp/api/__init__.py`,
`apedp/core.py`, `apedp/safety.py`, `apedp/pipeline.py`,
`apedp/persistence/model_store.py`, `apedp/falsification.py`, and added
`apedp/evidence.py` plus `tests/test_system_flaws_audit.py` and
`tests/test_evidence_tiers.py`.

The second pass touched `apedp/applied_defects.py`,
`apedp/ingestion/external_benchmarks.py`, `apedp/repair.py`, `apedp/pipeline.py`,
`apedp/persistence/model_store.py`, `apedp/metrology.py`,
`apedp/api/routes/metrology.py`, `apedp/api/routes/repair.py`, `apedp/api/__init__.py`,
`apedp/symbolic_regression.py`, `apedp/discovery.py`, `apedp/falsification.py`,
`apedp/safety.py`, `apedp/gated_assurance.py`, `apedp/benchmarks.py` and
`tests/test_system_flaws_audit.py`, `tests/test_evidence_tiers.py`.

Findings **6** (capability) and the remaining minor-table rows (`BuckinghamPiEngine`
groups, variable exponents, god-files) are **not** closed. Two things are fixed but
deliberately keep a permissive default, and say so: unit assumptions (#3) and the
unauthenticated server (#2).

---

## Outstanding requirements register (post-review, 2026-10-19)

Checkpoint `47594e7` is frozen as the remediation milestone (tag
`remediation-checkpoint-47594e7`). Commit `715437b` (the follow-through commit
on top) records subsequent work — the F17 extreme-scale robustness amendment
and this register — and does not replace or re-badge the frozen checkpoint:
the tag continues to anchor `47594e7`, and the register references it as the
canonical review state. The following requirements are explicitly
**unresolved** and must not be treated as completed; each carries its exact
acceptance criterion:

| ID | Requirement | Source | Acceptance criterion |
| --- | --- | --- | --- |
| REG-F14 | Caller-layer three-role data protocol: discovery candidates evaluated on selection data; referee pair separation on selection; the archived promotion is evaluated ONCE on a distinct sealed confirmation slice, and the confirmed verdict supersedes the selection verdict | External review (epistemic sign-off) | End-to-end integration test: a candidate pool selected on slice A, promoted head confirmed on untouched slice B; verdict recorded from slice B only; no shared slice |
| REG-F14b | Multiplicity accounting documentation: how `selection_reused_confirmation_holdout` maps onto the DSR ledger charge for each caller | Same | Every production caller either (a) holds a sealed confirmation split, or (b) exposes the DSR-adjusted multiplicity in its published record |
| REG-F16 | Formal collision-avoidance guarantee: CBF-QP synthesis (or equivalently rigorous verification) establishing `dot(h) + alpha(h) >= 0` under explicit assumptions for `batch_multi_agent_cbf_step` | External review (safety sign-off) | Formal certificate or model-checking artifact reviewed against the stepper; until then the subsystem is documented as a *collision-deadlock remediation*, not a verified safety guarantee |
| REG-MUT | Targeted mutation testing of referee and leakage paths (adjudicate anchor, frozen-library term resolution, hawkes standardization constants) | Same | Mutation run report in CI or as a scripted artifact demonstrating each pin fails under its mutation; must not be marked complete silently |
| REG-LEAN | Independent Lean verification run under a pinned, externally established Lean toolchain (outside this environment) | Same | Reproduce the 323-record sweep exit codes on an independently provisioned toolchain with recorded version hash |

Sources that document the current (non-claiming) boundary: `CHANGELOG.md`
("Still open", external-review rounds), `tests/test_dyn_blind_audit_round2_remediation.py` docstrings,
`apedp_dyn/cuda/batch_physics_steppers.py::batch_multi_agent_cbf_step`
docstring (not-a-certified-CBF-QP disclosure, test-pinned).

### Register status updates (2026-10-20, evidence-linked)

| ID | Status | Evidence / remaining work |
| --- | --- | --- |
| REG-MUT | **CLOSED-WITH-EVIDENCE** | `scripts/run_mutation_battery.py` (dependency-free battery, byte-identical-restoration invariant, hard-fail exit), report `benchmarks/mutation_battery_report.json` (status ALL_KILLED, 15/15 mutants, sanity-verified pristine pins), CI gate `mutation-battery` job in `.github/workflows/test.yml` + `scripts/check.sh` full tier. **2026-10-21 extension:** the external 19-flaw remediation pins (`tests/test_dyn_external_audit_2026_10_21_pins.py`, 23 tests, each carrying an inline old-semantics mirror) are wired into the battery as MUT-D1..D4 — S3 conditional-put take-over (FLAW-03), fallback-put deadlock (FLAW-03), CBF budget-aware deficit split (FLAW-17), re-anchor residual recovery (FLAW-08). **2026-10-21 rust-kernel extension:** the battery's pin runner now accepts literal argv (`cargo test --lib` recompiles from mutated source, so pins see the mutation without touching the installed maturin extension) and the cargo pins cover MUT-E1 (Fornberg weight sign, FLAW-04), MUT-E2 (uniformity-gate route, FLAW-04), MUT-E3/E3b (pole-mask flag drop on the 2nd-/1st-order rollout branches, FLAW-06), MUT-E4 (constant column back into the exact-RMSE ceiling, FLAW-12), MUT-E6 (raw multi-stratum Gram eigensolve, FLAW-15); battery re-run kills all fifteen. **Honesty trail #1:** MUT-C1 (hawkes eval-frame-leak mutant) SURVIVED the first run because the inherited dynamic pin scored a fit/eval-identical frame, exposing a real coverage gap; the pin was SHARPENED (`tests/test_dyn_hawkes_standardization_pins.py`), never weakened, and the battery re-run kills it. **Honesty trail #2:** the pin runs surfaced three REAL defects inside the v1 remediations themselves — the wrong-structured Fornberg recurrence (FLAW-04, fixed by mirroring the proven Python reference), the provably-insufficient global re-scale in the CBF close-out (FLAW-17, fixed by endpoint-headroom deficit distribution), and the scale-blind eigensolve behind the FLAW-15 relative gate (fixed by correlation-space congruence + spectral normalization). **Honesty trail #3:** an absolute-tolerance-gate mutant (MUT-E5 candidate) was registered and GREW no natural kill after the v2 rime fixes — the defect expression it targeted is dissolved by the corrected eigensolve — so it was removed with its rationale recorded in the battery rather than shipped as a silent survivor. **2026-10-23 extension (Pow-exponent and enforcement paths):** ten mutants were added for the exponent/birth-fit code and the native-search enforcement surface — MUT-H1 (enforcement switch reports off), MUT-H2 (bridge failure swallowed under enforcement), MUT-H3 (requested-fallback suspension removed), MUT-H4 (`strict_native_ga` without `use_native_ga` runs Python), MUT-H5 (strict mode falls back on a missing kernel), MUT-H6 (fallback tagged `kernel_path="rust"`), MUT-H7 (fixed exponent menu reintroduced), MUT-H8 (birth fit skipped at the call site), MUT-H9 (birth fit with zero sweeps), MUT-H10 (polish window collapsed). The battery now also runs the enforcement pin group and `cargo test --lib` as pristine sanity groups, and reports ALL_KILLED, 34/34, all restorations byte-identical. **Honesty trail #4:** MUT-H7 SURVIVED the first run — not because the generator was menu-bound but because the test sampled trees with the general-Pow arm enabled, whose shallow exponent SUBTREE is itself a constant leaf drawing from the continuous grid, masking the constant-exponent arm under test. The pin was SHARPENED (`allow_advanced=false` sampling, plus a minimum-sample guard), never weakened; the re-run kills all 34. **2026-10-23 follow-up (predicate + harness entry points, cargo cache):** three mutants were added — MUT-H11 mutates the `has_pow_const` birth-fit predicate itself (a constant exponent classified as a non-tunable variable exponent; killed by `test_birth_fit_applies_to_the_initial_population`, 195 passed / 1 failed), and MUT-H12/MUT-H13 remove the `enforce_native_search(True)` call from `run_synthesis_benchmark.run_benchmark()` / `Synthesis50TournamentRunner.run_all()`, killed by the new AST pin `test_benchmark_entry_points_arm_process_wide_enforcement` (both harnesses build bridges internally, so a `SearchConfig` flag cannot reach them). Battery reports ALL_KILLED, 37/37. Mutant results are now reused from `benchmarks/mutation_battery_cache.json` only when the mutation bytes, pin argv, content hash of the pin-input tree (every `.py` source, `rust/**` compiled sources, `specifications/**`, pytest config) and `rustc --version` all match; each reuse is flagged `cached: true` in the report and the pristine sanity pins still run fresh (measured: ~7.5 min fresh vs ~2 min warm), while `--no-cache` forces a full re-verification. |
| REG-F14/F14b | **CLOSED-WITH-EVIDENCE (mode (b) wiring landed; mode (a) NOW operationally wired — blind_benchmark first census caller)** | Reference implementation (`docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md`, `apedp_dyn/epistemic/caller_data_protocol.py`, `tests/test_dyn_caller_data_protocol.py`, 7 pins incl. the register's end-to-end sealed-slice-B letter) PLUS the per-caller wiring gate the register demanded: the production-caller census `apedp_dyn/epistemic/caller_registry.py` (6 callers: dyn_sweep, multi_asset_signals, real_market_l2_sweep, blind_benchmark, calibrated_benchmark_suite, canonical_ledger), each wired with a `publish_caller_compliance_record` publisher in its module and validated at publish time; selection-stage exposure hook (`record_selection_exposure`) makes the derived multiplicity LIVE. Pins `tests/test_dyn_caller_compliance_wiring.py` (12): census closure/publishers, per-caller mode-(b) validation, missing/unknown/duplicate-refusal, reuse charge flow (`mark_selection_reuse` → effective n inflates), hand-written multiplicity refusal, end-to-end wiring letter — PLUS four mode-(a) operational pins: sealed split issued over the caller's OWN real slice ids (`issue_sealed_confirmation_lead`, census/dataset-family-checked), blind confirmation slice consumed EXACTLY ONCE through the real `evaluate_confirmation_cases` scoring path (frozen-referee-checked, unknown ids refused), publication REFUSED while the slice is unconsumed or with foreign provenance, and the spent slice cannot be re-watched by a second candidate; the published sealed record (mode (a), one-shot provenance, `archived_verdict_source = "confirmation"`, selection superseded) validates under the same F14b validator. **Honest boundary:** the five remaining census callers stay published in mode (b) by default and the blind suite's mode-(a) record is demonstrated at wiring level through the pinned registration/consumption/archive/publish path — a LIVE production promotion claim through the sealed slice is still pending and is disclosed as such, not silently claimed. External epistemic sign-off (register source) remains the outstanding step. **Restart durability (2026-10-20):** sealed-slice consumption is now durable across process restarts — issued leads and consumption provenance persist to an atomic state file (`caller_lead_protocol_state.json`, path explicit or derived from the ledger's persistence directory), the slice is claimed spent BEFORE evaluation so a mid-evaluation crash cannot resurrect it, re-issue/re-consume after restart is refused from the durable record, and a corrupt store fails CLOSED (`CallerProtocolStateError`, no silent memory-only fallback). Ledger exposure writes are FAIL-CLOSED (the prior silent except-and-pass is removed: write failures propagate; an unregistered candidate is refused before the sealed slice is spent). The mode-(b) `dsr_shrink` disclosure imports the preregistered `DSR_THRESHOLD` from the hash-verified canonical spec (spec JSON unmodified — it is quorum-signed; a value change requires a re-signed revision). The one-shot guarantee remains in-memory-only ONLY where no durable state path resolves (in-memory unit pins), disclosed as a unit-use limitation. **Pre-existing, retroactively disclosed (outside this register's mandated suite):** `tests/test_dyn_governance.py::test_canonical_repository_governance_manifest_valid` fails at the committed tip independent of this work — `specifications/governance_manifest.json` pins spec SHA `a6d026…` while the spec on disk hashes `a11fd957…` (matching `canonical_spec.py`'s import-time pin); resolution requires renewed trustee certification, not a code edit, and is disclosed rather than claimed. |
| REG-F16 | **CLOSED-WITH-EVIDENCE (formal certificate artifact; external safety sign-off still outstanding)** | `apedp_dyn/robotics/cbf_certificate.py` + `tests/test_dyn_cbf_certificate.py` (8 pins) + artifact `benchmarks/cbf_certificate_report.json` (`benchmarks/run_cbf_certificate_battery.py`, exit 0). FORMAL scope (machine-checked under the explicit assumption object, which refuses γ·dt > 0.5 etc.): pairwise barrier h = dist − r_safe; certificate-by-construction for every released command (every release re-checked fail-closed), closed-form factory feasibility lemma (identical velocities ⇒ the QP set is never empty ⇒ alternating projections converge), projection-optimality containment vs the feasible zero command, discrete forward invariance h_{k+1} ≥ (1−γdt)h_k > 0 over certified Euler horizons (4 envs × 120 steps pinned), 64 certified releases verified with worst slack −3.3e-16. SCALED-VERIFICATION layer (same formal checks re-verified at larger grids under the same assumption object with `max_agents=128`, in-scope): 16/32/64/128-agent deterministic jittered-lattice grids (A3 margin assertion-checked) — 1440 additional releases fail-closed-verified (worst slack −9.91e-10 ≥ −tol at 128 agents), Euler forward invariance h > 0 at every step on every scale, and the REALIZED guard (certify → `unicycle_commands_from_certified` → PRODUCTION `batch_unicycle_arc_step`) collision-free at every scale under the physically-consistent realized-velocity-feedback nominal discipline (128-agent realized min barrier 0.3919). Guard latencies recorded as WALL-CLOCK measurements (measurement of the environment that ran the battery, explicitly NOT guarantees): mean guard 0.32/0.48/0.99/2.97 ms and mean guard+stepper step 0.60/0.77/1.31/3.39 ms at 16/32/64/128 agents. DISCLOSED SCOPE PROBE (recorded verbatim, not exit-affecting): fresh full-range random nominals redrawn every step at 128 agents are OUTSIDE the realized layer's evidence (deterministic breach: min barrier −0.1102, 17 collision steps) — the realized layer remains model-checked EVIDENCE scenario-scoped (deadlock / realized-feedback disciplines), never a formal claim. MODEL-CHECKED baseline kept: adversarial symmetric head-on sweeps — min barrier 0.3588, 0 collisions. `batch_multi_agent_cbf_step` is UNTOUCHED and its not-a-certified-CBF disclosure stays pinned; `certified_cbf_guard` is additive. External safety sign-off (register source) remains the outstanding review step. |
| REG-LEAN | **CLOSED-WITH-EVIDENCE (pinned-toolchain reproduction; physically-external provision still outstanding)** | `scripts/run_lean_pinned_reproduction.py` + artifact `benchmarks/lean_pinned_toolchain_reproduction.json`: ALL 323/323 records reproduced (proof_claim_status + kernel_exit_code per candidate) under the independently provisioned pinned toolchain `leanprover--lean4---v4.34.1` (commit `5045d0056413266e57c625dcd7c365b10e377c52`) vs the reference run's v4.35.0 (`eb6cafb7…`); counts match (323/315/8/0). Byte-identical restoration of the reference artifact AND the committed `.lean` evidence verified (the script restores HEAD-committed bytes — a bare regeneration silently corrupts the remediated u15 honest disclosure back into a vacuous theorem, caught and fixed). **Root-cause fix landed:** the reference sweep itself was re-produceable only after raising the bridge's 15/25 s compile subprocess timeouts to 240 s — under 8-way contention the two heaviest fluid-cell files were being wall-clock-killed to P1 (exit −1) while compiling clean standalone; disclosed in `CHANGELOG.md`. The register's "outside this environment" EXTERNAL provisioning/sign-off remains the outstanding step and is not claimed here. |

---

## Round-3 external assessment record (2026-10-19)

Post-round-3 external review scored update: **6.5/10 overall** (up from ~6/10).
Reported scorecard retains the distinction between verified remediation state and
scientific-capability maturity:

| Dimension | Score |
| --- | --- |
| Engineering and audit discipline | 8.5/10 |
| Epistemic safeguards | 8/10 |
| Robustness and regression evidence | 8/10 |
| Structural discovery capability | 5.5/10 |
| Scientific validation and generalization | 4.5/10 |
| Demonstrated advantage over alternatives | 3/10 |

Reviewer-stated criteria for reaching **8/10** (recorded here as forward
requirements, not claimed progress):
1. Completion of the outstanding requirements above (REG-F14/F14b, REG-F16,
   REG-MUT, REG-LEAN) with per-item acceptance evidence.
2. Independent review of the mathematical certificates and critical code paths.
3. A reproducible external evaluation showing APEDP-X reliably distinguishes
   valid discoveries from false or non-identifiable hypotheses.

For **9/10 or higher**, additionally: measurable advantage over established
alternatives across multiple independent problem families, not just successful
individual demonstrations.

Matching the reviewer's framing, this register treats the requirement-closure
path as: individual acceptance evidence per requirement first, then
independent external validation of scientific performance. Already-green
checks (46-pin suite, Rust lib 186/186, branch alignment, frozen checkpoint
`47594e7` vs follow-through record) are not reopened without a relevant
change or new failure signal.
