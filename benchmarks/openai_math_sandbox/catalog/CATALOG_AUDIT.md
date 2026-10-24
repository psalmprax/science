# OpenAI math catalog — full ingestion audit

Source: `openai/math` @ main, retrieved 2026-10-16. Raw files pinned by SHA-256 in `manuscripts_manifest.json`.

## Accounting

- Families: **372**, manuscripts: **719**, disciplines: **17**.
- Withdrawn: **3** (722 -> 719 reconciliation), revised: **14**, citation-updated: **13**.
- Lean (yaml catalogue): 173 source papers, 200 main-result declarations, 173 catalogue manuscripts joined, 136/372 families covered.
- Repo-stated formalization: 300/719 (~42%).

## Unresolved discrepancies (recorded, not hidden)

- Headline count moved 722 -> 719 by withdrawing 3 manuscripts (history.md, Oct 7 2026); all three are absent from CONTENTS.md.
- Family numbering is historical, not contiguous: 372 live families spread over 001-377 with retired numbers [45, 61, 70, 123, 163]; catalogue order is preserved rather than renumbered after removals.
- history.md states 300/719 top-line results formalized (~42%), but lean/formalization.yaml enumerates only 173 source papers and 200 main-result declarations (173 of them join to catalogue manuscripts). The delta is unresolved upstream; manifest reports both numbers rather than assuming a mapping.
- form is not an independent check: formalization.yaml review.status = 'unchecked', method = 'agent' (agent-produced entries; APEDP-X does not run Lean or the Comparator).
- 1 formalization.yaml source path(s) do not resolve directly against the catalogue; each is joined-by-title or recorded unresolved in lean_join_notes rather than silently dropped.

## APEDP-X checkability over all families

Every family is `uncheckable_proof_level`: 372/372. No manuscript is marked verified by this system — the sandbox rule (no certificate, no `verified`) applies at catalog scale too.

Probe tags (illustrative finite checks only, never verification):

- `none`: 297 families
- `finite_range_numbers`: 40 families
- `explicit_construction`: 35 families

## Discipline coverage

| Discipline | Families |
|---|---|
| Number theory | 31 |
| Algebraic and complex geometry | 36 |
| Real and complex analysis | 16 |
| Convex and metric geometry | 15 |
| Theoretical computer science | 40 |
| Dynamical systems and ergodic theory | 12 |
| Combinatorics | 37 |
| Algebra | 18 |
| Probability and statistical mechanics | 29 |
| Mathematical logic | 6 |
| Group theory | 14 |
| Mathematical physics | 25 |
| Operator algebras | 19 |
| Topology | 18 |
| Functional analysis | 11 |
| Differential geometry | 29 |
| Partial differential equations | 16 |
