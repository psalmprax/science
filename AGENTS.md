# AGENTS.md — Instructions & Integrity Protocol for AI Coding Agents

Welcome, agent. This repository (`APEDP-X / CPMD`) is an advanced scientific discovery and epistemic referee platform. It enforces extreme scientific rigor, formal mathematical proofs, and data-integrity firewalls. 

When modifying, auditing, or extending this codebase, you **MUST** follow these repository-specific protocols at all times.

---

## 1. Core Operating Principles

### 1.1 Strict Proposer vs. Referee Separation
- Proposers (in `apedp_dyn/proposers/`) generate candidate hypotheses.
- Referees (in `apedp_dyn/epistemic/`) adjudicate hypotheses. The promotion referee itself — thresholds, the six-stage lifecycle, the holdout and trial ledgers, the universe audit — is the `apedp_referee` package, which must import nothing from the rest of the repository (`tests/referee_core_manifest.py`).
- **Rule:** A proposer must never decide its own acceptance, bypass gates, or inspect confirmation holdout data. Proposers only emit standardized `CandidateProposal` objects.

### 1.2 Anti-Snooping Holdout Discipline
- Holdout slices are consumable, scarce resources.
- If a candidate hypothesis fails out-of-sample confirmation, that `(dataset_id, slice_id)` slice is permanently burned in `FullLineageExperimentLedger`.
- **Rule:** Never bypass `HoldoutContaminationError`, rename slices, or reset the burned holdout registry to force a failing candidate to pass.

### 1.3 Canonical Specification Integrity
- All preregistered thresholds (e.g. tactical active window $[5\%, 90\%]$, $DSR \ge 0.95$, Diebold-Mariano $p < 0.01$, literature $R^2 < 0.85$, referee FPR/CAR/VRR) are defined in [`specifications/canonical_preregistered_spec_v1.json`](specifications/canonical_preregistered_spec_v1.json).
- `apedp_referee/canonical_spec.py` (also importable as `apedp_dyn.conf.canonical_spec`, the same module) enforces a SHA-256 hash check upon import. The referee package ships a byte-identical copy of the specification at `apedp_referee/canonical_preregistered_spec_v1.json`.
- **Rule:** Never define arbitrary hardcoded threshold literals. Always import from `apedp_dyn.conf.canonical_spec` (or `apedp_referee.canonical_spec` inside the referee package). Never modify `canonical_preregistered_spec_v1.json` without updating its pinned hash, copying it into `apedp_referee/`, and documenting the rationale in `CHANGELOG.md`.
- The discovery routes' thresholds are named in [`specifications/discovery_checks_v1.json`](specifications/discovery_checks_v1.json) and pinned by `apedp_dyn/conf/discovery_checks.py`: the accuracy line that defines a wrong law, the last checks' level and power, and (since 2026-11-25) the limit on the share of wrong laws among those reported, with the upper bound that share must also meet (`apedp_dyn/discovery_l2/report_share.py`). The owner decided on 2026-11-25 that the trustees sign for this file together with the canonical specification; until they have, the signed governance manifest does not cover it (`docs/GOVERNANCE_RESIGNING_REQUEST.md`, section 8).
- **Rule:** Import a threshold by what it means. Never borrow an entry defined for another test because its value fits; if no named threshold means what is needed, add one, with a new pin, a `CHANGELOG.md` entry and the owner's word.

### 1.4 Documentary & Repository Integrity
- **Rule:** Never claim a documentation file exists unless it is verified on disk within the repository workspace (e.g. `docs/`, `specifications/`, or the repo root). Do not confuse temporary IDE/CLI artifact folders (`~/.gemini/antigravity-cli/brain/...`) with the git repository workspace.
- **Rule:** Keep `docs/DOCUMENTATION.md` updated whenever new modules, proposers, or epistemic gates are introduced.

### 1.5 The Owner Decides What Is in Force
- A preregistration fixes, before its run, what each outcome *proposes*: adopt, keep on request, not adopt. Fixing it in advance stops the result from being read to suit; it does not give the agent who wrote the plan the authority to change what the system runs.
- **Rule:** A change to what is in force — a route, a default, a check added to or taken out of a path, a threshold — waits for the owner's explicit word, even where a preregistration fixed the consequence. Record the result, write the proposed change (as a module or a branch if useful), state it to the owner, and leave the default unchanged until they agree. An owner's decision is not reversed by a consequence written before or after it.
- **Rule:** Where an agent has already changed what is in force without that word, it says so plainly and offers the switch that undoes it.

### 1.6 Independent Review and Blind Tests the Builder Did Not Shape
- The agent that builds a check also writes its benchmark, its expectations and its record. A preregistration stops goalposts moving after a run, not easy goalposts chosen before it.
- **Rule:** Before a preregistration is frozen, and before its result is recorded, a separate reviewer that did not take part in the work checks it cold: the plan's denominators, limits and consequences against what the runner computes, and every number of the record against the raw outputs. Its findings, and what was done about each, go into the record.
- **Rule:** A blind test of a check includes traps the check was not developed against: at least one arm drawn from a red-team search aimed at the frozen check after freezing, or from a generator other than the one used in development. A test whose traps all come from the development generator measures in-distribution performance and says so.

### 1.7 Branches: `stage` First
- All work is pushed to `stage`. `master` receives nothing directly.
- **Rule:** `master` changes only by merging `stage` into it, on the owner's word, after `stage`'s checks have run. A run branch (`*-run-*`) starts a workflow and is never merged.

---

## 2. Standard Testing Commands

Always run the full dynamic regression suite to verify changes before presenting results:

```bash
pytest tests/test_dyn_competing_proposers_and_dsr.py \
       tests/test_audit_open_flaws_remediated.py \
       tests/test_dyn_blind.py \
       tests/test_dyn_canonical_ledger.py \
       tests/test_dyn_native_ga_best.py \
       tests/test_dyn_experiment_ledger_and_benchmarks.py \
       tests/test_documentation_and_spec_integrity.py \
       tests/test_dyn_expanded_proposers_and_calibrated_power.py \
       tests/test_dyn_dmrg_quantum.py \
       tests/test_dyn_lob_transport.py \
       tests/test_dyn_audio_field_transport.py \
       tests/test_dyn_molecular_se3_flow.py \
       tests/test_dyn_robotics_trajectory.py \
       tests/test_dyn_temporal_rate_modulation.py \
       tests/test_dyn_temporal_rate_io.py \
       tests/test_gf2_wsp.py \
       tests/test_gf2_rank_search.py \
       tests/test_field_tensor_search.py \
       tests/test_gf2_mul_proposer.py \
       tests/test_quantum_verdict_tristate.py \
       tests/test_quantum_compilation.py \
       tests/test_mmlm_tracks.py \
       tests/test_tracked_import_closure.py \
       tests/test_dyn_sandbox.py \
       tests/test_dyn_ui_sandbox.py \
       tests/test_discovery_envelope.py -v
```

---

## 3. Module Status Labels

When documenting or registering modules, use these standard status tags:
- `production`: Fully tested, stable, actively used in core pipelines.
- `experimental`: In active development or evaluation; not yet certified for live decisions.
- `audit-only`: Specialized harness, verification script, or forensic test module used to audit system integrity.
- `deprecated`: Retained only for historical replication; superseded by newer implementations.
