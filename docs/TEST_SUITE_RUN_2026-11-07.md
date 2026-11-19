# Test suite run — 2026-11-07

**Code under test:** commit `72e3224` (the collection, tokenizer and PDE fixes).
The run was executed on the worktree that produced this commit; the worktree
also carried **52 pre-existing uncommitted modifications** that are not part of
this commit.

## Collection

```bash
.venv/bin/python -m pytest tests --collect-only -q
```

**5,657 tests collected, 0 errors, 48.66 s.** Before this change, collection
aborted with two errors: `BPEMMLMTokenizer` did not exist and
`apedp_continuum` had been deleted.

## Regression run

```bash
.venv/bin/python -m pytest tests -n 6 \
  -m "not integration and not scientific_regression" -v --tb=short
```

The selection contains 5,650 tests. The run was stopped at **31% after 18:11**
because nine failures had already appeared, all in subsystems untouched by this
change. **This is not a green run.**

### Failures observed

- `tests/test_dyn_agnostic_adapter.py::test_3arm_ablation_experiment_scorecard`
  — Arm A measures 2/4 valid retained (test and whitepaper claim 4/4); Arm B
  measures 2/4 traps promoted (claim 4/4).
- `tests/test_dyn_eedr.py::test_independent_proposer_benchmark_pooled_execution`
  — pooled VRR measures 50.0% (test and whitepaper claim 100.0%; P2 and P4
  retain 0/5 valid cases). The benchmark nevertheless prints `PASSED` because
  its pass criterion is FPR-only.
- `tests/test_dfthcsos_arm2.py::test_auditor_promotes_only_verified_isolated_improvement`
- `tests/test_discovery_benchmark.py::test_tier_a1_recognizes_known_laws`
- `tests/test_dyn_d2b.py::test_coupled_enumeration_and_prune`
- `tests/test_dyn_d3.py::TestD3KeptCounts::test_ode_mode_kept`
- `tests/test_dyn_d4.py::TestD4KeptCounts::test_ode_mode_kept`
- `tests/test_dyn_d5.py::TestD5STABKeptCounts::test_ode_mode_kept`
- `tests/test_dyn_deepmind_challenge.py::test_alphatensor_and_graphcast_direct_promotion`

The two benchmark failures are deterministic when run alone and are not caused
by this change; the benchmark modules are unmodified at HEAD. Making the suite
green requires either repairing those benchmark semantics or correcting the
published 100% VRR claims — a scientific-claim decision, not an assertion to
loosen silently.

## Checks that do pass

| Check | Result |
|---|---|
| `tests/test_apedp_mmlm_architecture.py tests/test_dyn_pde_ode_packages.py` | **26 passed** |
| `tests/test_documentation_and_spec_integrity.py` | **16 passed** |
| `tests/test_dyn_governance.py` | **4 passed, 1 failed** — the expected FLAW-01 manifest-drift failure |
| `ruff check` on the three touched Python files | **All checks passed** |
