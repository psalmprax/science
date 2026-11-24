# Documentation Claims Audit — 2026-11-07

Scope: current-status and entry-point documents (`README.md`,
`docs/DOCUMENTATION.md`, `docs/GAP_ANALYSIS.md`, `docs/MAINTAINABILITY_SCORE.md`,
`docs/SCAFFOLD_INVENTORY.md`, `docs/SYSTEM_FLAWS_AUDIT.md`,
`docs/ASSESSMENT.md`, `docs/INDEPENDENT_REVIEW_2026-09-23.md`) plus every
document touched by the two collection failures. This is not a row-by-row audit
of all 3,073 lines of `DOCUMENTATION.md`; the unchecked remainder is stated at
the end.

Method: each claim was compared with the file on disk, a live import, or a
freshly executed test. Historical documents were not rewritten; where a claim
was true when written but is false now, a dated marker was added.

## Claims that were stale, and what changed

| Document | Claim | Live finding | Disposition |
|---|---|---|---|
| `docs/DOCUMENTATION.md` (APEDP-MMLM table) | `apedp_mmlm/engines.py` contains `BPEMMLMTokenizer` with UTF-8 byte fallback | No such class existed; the test that imported it could not collect | Implemented `BPEMMLMTokenizer` in `apedp_mmlm/engines.py`, exported it from `apedp_mmlm/__init__.py`, and pinned it with the existing 15-test module |
| `docs/DOCUMENTATION.md` (APEDP-MMLM table) | Engine 2 has "DMRG two-site variational sweeping, SVD Schmidt spectrum" | No `dmrg_two_site_sweep` method existed; the engine stores one bond-space matrix per token, not per-position cores, so a variational ground-state solver is not what the data structure supports | Implemented a two-site SVD compression sweep with operator-Schmidt entropy, and corrected the wording from "variational DMRG" to the operation actually performed |
| `docs/DOCUMENTATION.md` (governance row) | "Verified (4/4 unit and integration tests passing)" | Live: **4 passed, 1 failed**; `test_canonical_repository_governance_manifest_valid` fails because the manifest seals `a6d026f3…` while the spec and pin are `dbf67c5a…` | Row now says 4 of 5, names FLAW-01, and points to `docs/GOVERNANCE_RESIGNING_REQUEST.md` |
| `tests/test_dyn_pde_ode_packages.py` | Imports `apedp_continuum` from `packages/apedp-continuum` | The six `packages/apedp-*` shims were deleted on 2026-11-06 and `tests/test_scaffold_refusals.py` pins their absence | Removed the shim import and the two parity assertions; the remaining 11 PDE/ODE tests now run against `apedp_dyn`, where the behaviour lives |
| `docs/SYSTEM_FLAWS_AUDIT.md` (2026-09-23 re-baseline note) | CI "now triggers on `master` (plus `main`/`develop`)" | `.github/workflows/test.yml` now triggers on `branches: ['**']` | Note updated to say every branch, and that it was widened again |
| `docs/ASSESSMENT.md` §2.6 | "No version control: the working tree is not a git repo" | This is a git repository on branch `stage` | Added a dated historical-snapshot header; the body is retained as written rather than silently rewritten |

## Claims checked and found current

| Document | Claim | Evidence |
|---|---|---|
| `docs/GAP_ANALYSIS.md` | Governance manifest sealed to an older hash; fail-closed red; re-signing is trustee-only | Manifest `a6d026f3…`, disk/pin `dbf67c5a…`; `pytest tests/test_dyn_governance.py -q` = 4 passed, 1 failed |
| `docs/SCAFFOLD_INVENTORY.md` | `packages/apedp-{audio,continuum,distributed,molecular,quantum,robotics}` and `crates/apedp_kernel` are deleted and pinned absent | `packages/` contains only `jetflow-video`; `crates/` does not exist; `tests/test_scaffold_refusals.py` passes |
| `docs/MAINTAINABILITY_SCORE.md` | Collection is the remaining ergonomics bottleneck | 5,657 tests collect in 48.66 s; full run is the long pole |
| `README.md` | CI covers Python 3.10–3.12; platform does **not** claim live profitability or domain-agnostic discovery | Workflow matrix is `["3.10", "3.11", "3.12"]`; the "does NOT establish" section is accurate |
| `docs/INDEPENDENT_REVIEW_2026-09-23.md` | Dated snapshot with a §10 remediation table | Dated, internally consistent, and explicitly a snapshot; left unchanged |

## Deliberately not "corrected"

- `docs/SYSTEM_FLAWS_AUDIT.md` 2026-09-18 baseline (`288 passed`, `ruff
  all checks passed`, `mypy 0`) is immediately followed by its own 2026-09-23
  "these claims are STALE" blockquote. Rewriting the baseline would destroy the
  record of what was claimed; the correction already sits beside it.
- `docs/INDEPENDENT_REVIEW_2026-09-23.md` and `docs/ASSESSMENT.md` are dated
  reviews. Only the one claim that is now factually impossible ("not a git
  repo") received a marker; the rest is preserved as the reviewer wrote it.

## Unchecked remainder

`docs/DOCUMENTATION.md` is a 3,073-line registry. This audit covered the rows
relevant to the failures and the current-status documents above, not every
"Verified (N/N)" row. A full row-by-row audit would need to execute each cited
test file and is a separate task.
