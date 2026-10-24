# Caller-Layer Three-Role Data Protocol (REG-F14 / REG-F14b design)

**Status:** design + acceptance-tested reference implementation
(`apedp_dyn/epistemic/caller_data_protocol.py`, `experimental`; additive — no
existing caller is rewired silently; wiring is per-caller and auditable).
**Register anchor:** `docs/SYSTEM_FLAWS_AUDIT.md` — REG-F14 and REG-F14b.

## 1. The problem (in one sentence)

A design-time dataset can leak into a promotion claim: candidates are picked
on data the promotion verdict later re-watches, so "confirmed" can mean
"selected AGAIN" rather than "new evidence."

## 2. The three roles (non-fungible, disjoint per lead)

| Role | Symbol | Who may see it | When it is spent |
|---|---|---|---|
| SELECTION | slice A | discovery/search and referee pair separation | any number of times while the lead is open |
| CONFIRMATION | slice B | the SEALED single-use promotion evaluation | exactly ONCE, by the archived promotion head, via `SealedConfirmationSlice.consume()` |
| PROSPECTIVE | unassigned at design time | live forward data as it arrives | collected post-promotion; never folded back into A or B |

Issuance contract (`CallerSplitProtocol.issue_caller_split`):

1. **Disjointness is mandatory** — selection ids ∩ confirmation ids = ∅,
   verified at issue time; overlap raises `CallerRoleOverlapError`.
2. **The confirmation slice is sealed** — wrapped in
   `SealedConfirmationSlice`; ids are readable but the evaluation verdict can
   only be obtained through `consume()`, which records provenance (slice ids,
   one-shot flag, timestamp) and refuses a second `consume()` with
   `ConfirmationAlreadyConsumedError` (F14's letter: evaluated ONCE).
3. **Ledger integration** — every evaluation is recorded as a
   `DatasetExposureRecord` on the candidate's `FullLineageExperimentLedger`
   node with the correct `DatasetRole`, so the existing anti-snooping burn
   discipline keeps ownership of slice lifecycle (a CONFIRMATION exposure that
   fails state transitions burns the slice for successors).
4. **The confirmed verdict supersedes selection** — the promotion record
   stores BOTH verdicts, with `archived_verdict_source = "confirmation"`; a
   caller cannot archive a selection-stage verdict as the promoted claim.

## 3. REG-F14b — multiplicity mapping (how re-use becomes a charge)

Every production caller must satisfy one of exactly two modes:

* **Mode (a) — sealed confirmation split held:** the caller's published record
  carries `confirmation_mode = "sealed"` and the one-shot verdict provenance.
* **Mode (b) — published DSR-adjusted multiplicity:** where a sealed split is
  not operationally possible, the caller publishes
  `multiplicity = {n_selection_variants, n_confirmation_reuses,
  dsr_adjusted_effective_n}`, computed from LEDGER exposure counts (`n =
  number of DISTINCT candidates charged against this caller's
  selection-family exposures`), never hand-written, and verified at publish
  time by `validate_caller_compliance` — a record with
  `selection_reused_confirmation_holdout = true` must expose
  `dsr_adjusted_effective_n` increase over the nominal n, deflated by the
  confirmed confirmation re-count, or the record is refused.

The mapping rules live in code (`multiplicity_from_ledger`), are monotone in
variants/reuses, and are pinned so the deflation cannot be edited by hand.

## 4. Acceptance test (the register's letter, automated)

`tests/test_dyn_caller_data_protocol.py`:

* `test_end_to_end_pool_selected_on_A_confirmed_once_on_sealed_B` — pool
  scored on slice A → archived head consumes slice B exactly once → recorded
  verdict carries B ids and `one_shot` provenance; A∩B = ∅ asserted; second
  consume raises.
* `test_overlapping_roles_refused_at_issue` — role assignment is not
  renegotiable after the fact.
* `test_verdict_supersedence_and_source_label` — archived record labels
  confirmation as the verdict source; selection verdict retained but marked
  superseded.
* `test_mode_a_and_b_compliance_paths` — a sealed-split record validates; a
  leaked/no-split record validates ONLY with a ledger-derived multiplicity
  exposure and is refused otherwise.
* `test_multiplicity_monotone_and_ledger_derived` — the charge grows with
  variants/reuses and equals the ledger-computed value (no hand-tuning).

**Out of scope for this design (stated honestly):** automatic *wiring* of
every existing production caller — MODE-A wiring is per-caller engineering
with migration notes; the register's criterion is satisfied when each caller
is either wired or publishes its multiplicity, which the compliance validator
will check per caller as wiring lands.
