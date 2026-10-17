# AGENTS.md — Instructions & Integrity Protocol for AI Coding Agents

Welcome, agent. This repository (`APEDP-X / CPMD`) is an advanced scientific discovery and epistemic referee platform. It enforces extreme scientific rigor, formal mathematical proofs, and data-integrity firewalls. 

When modifying, auditing, or extending this codebase, you **MUST** follow these repository-specific protocols at all times.

---

## 1. Core Operating Principles

### 1.1 Strict Proposer vs. Referee Separation
- Proposers (in `apedp_dyn/proposers/`) generate candidate hypotheses.
- Referees (in `apedp_dyn/epistemic/`) adjudicate hypotheses.
- **Rule:** A proposer must never decide its own acceptance, bypass gates, or inspect confirmation holdout data. Proposers only emit standardized `CandidateProposal` objects.

### 1.2 Anti-Snooping Holdout Discipline
- Holdout slices are consumable, scarce resources.
- If a candidate hypothesis fails out-of-sample confirmation, that `(dataset_id, slice_id)` slice is permanently burned in `FullLineageExperimentLedger`.
- **Rule:** Never bypass `HoldoutContaminationError`, rename slices, or reset the burned holdout registry to force a failing candidate to pass.

### 1.3 Canonical Specification Integrity
- All preregistered thresholds (e.g. tactical active window $[5\%, 90\%]$, $DSR \ge 0.95$, Diebold-Mariano $p < 0.01$, literature $R^2 < 0.85$, referee FPR/CAR/VRR) are defined in [`specifications/canonical_preregistered_spec_v1.json`](specifications/canonical_preregistered_spec_v1.json).
- `apedp_dyn/conf/canonical_spec.py` enforces a SHA-256 hash check upon import.
- **Rule:** Never define arbitrary hardcoded threshold literals. Always import from `apedp_dyn.conf.canonical_spec`. Never modify `canonical_preregistered_spec_v1.json` without updating its pinned hash and documenting the rationale in `CHANGELOG.md`.

### 1.4 Documentary & Repository Integrity
- **Rule:** Never claim a documentation file exists unless it is verified on disk within the repository workspace (e.g. `docs/`, `specifications/`, or the repo root). Do not confuse temporary IDE/CLI artifact folders (`~/.gemini/antigravity-cli/brain/...`) with the git repository workspace.
- **Rule:** Keep `docs/DOCUMENTATION.md` updated whenever new modules, proposers, or epistemic gates are introduced.

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
       tests/test_dyn_robotics_trajectory.py -v
```

---

## 3. Module Status Labels

When documenting or registering modules, use these standard status tags:
- `production`: Fully tested, stable, actively used in core pipelines.
- `experimental`: In active development or evaluation; not yet certified for live decisions.
- `audit-only`: Specialized harness, verification script, or forensic test module used to audit system integrity.
- `deprecated`: Retained only for historical replication; superseded by newer implementations.
