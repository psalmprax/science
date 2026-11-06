# Maintainability re-score: 6.0 -> 7.2 / 10, from measurement

An external review scored maintainability **6.0 / 10** on four grounds: the dual
`apedp` / `apedp_dyn` package roots, "fragile adapters", "tight freeze coupling", and the
460-file collection bottleneck. Each ground was re-measured on 2026-11-02 rather than
re-argued. The score moves to **7.2**: the coupling and import-order grounds do not hold
at their stated size, and the collection bottleneck does.

## What was measured

| claim | measurement | verdict |
|---|---|---|
| "two full packages side by side" | `apedp/` 207 files / 60,617 lines; `apedp_dyn/` 1,230 files / 266,929 lines; **no module implemented twice** (each tree owns different subsystems) | true as a description, not as duplication |
| "bridged by fragile adapters" | import-time edges `apedp_dyn` -> `apedp`: **0**. The parser, the proof-claim enum and finally the Rust bridge moved into `apedp_dyn`; the bridge's Python-GA fallback is now a provider registration the legacy tree makes, not an import the canonical tree performs | the coupling claim no longer holds |
| reverse direction (intended) | `apedp` -> `apedp_dyn`: 12 import lines, of which 3 modules are compatibility façades (`apedp/pde`, `apedp/ode`, `apedp/dyn`) and 7 are direct consumers of canonical grammar/PDE code | correct direction |
| "fragile" in practice | 8 modules imported as the *first* module of a fresh interpreter (`apedp`, `apedp.dyn`, `apedp.pde`, `apedp_dyn`, `apedp_dyn.safety`, `apedp_dyn.epistemic.proof_claim`, `apedp_dyn.epistemic.lean_universal_bridge`, `apedp_dyn.proposers.rime_ad_proposer`): **0 failures**; `tests/test_facade_contracts.py` pins all of them | no observable fragility |
| façades are maintained | `apedp/dyn` re-exports 75 names, `apedp/pde` 25, `apedp/ode` 10, `apedp/safety` 6; the contract test fails if any name in `__all__` stops resolving | guarded |
| "tight freeze coupling" | `frozen_spec.py` pins 4 artefacts (3 referee files + 1 benchmark JSON) plus 5 generated-output hashes that are skipped when absent; re-mint to `APEDPX_FROZEN_v1_12.json` is documented in `ARCHITECTURE.md` | bounded and documented |
| "460-file collection bottleneck" | `pytest --collect-only` = **53.3 s**, ~735 MB peak, dominated by top-level torch/jax/sympy/pint imports in test modules | **true, unchanged** |

## Score

| sub-dimension | score | reason |
|---|---|---|
| module layout, absence of duplication | 8 / 10 | two roots, 0 duplicated implementations, each with a stated role |
| dependency direction and coupling | 9 / 10 | 0 import-time edges; the legacy tree registers its GA with `apedp_dyn` instead of being imported by it |
| boundary documentation | 8 / 10 | ADR-0001, façade contract test, CHANGELOG entries with the reason for every move |
| import-time hygiene | 7 / 10 | 8/8 first-import cases pass and are pinned; `apedp/dyn/__init__.py` is still a wholesale wildcard |
| test-suite ergonomics | 4 / 10 | 53.3 s just to collect; 24 pre-existing red cases in `tests/test_facade_contracts.py` from vendored `_qualtran_src` being classified as façades |
| **average** | **7.2 / 10** | up 1.2 from the review, with the remaining deduction named |

## What is still weak (and would move the score again)

1. **Collection latency.** 53.3 s and ~735 MB before a single test runs. Worth splitting
   heavy imports out of collection; nothing has been done here yet.
2. **Vendored code inside the façade audit.** `apedp_dyn/sandbox/.../kickmix/_qualtran_src`
   is an upstream checkout whose modules are classified as re-export façades and then fail
   `test_facade_is_not_merely_empty`. Measured on the pre-change file too, so it is
   pre-existing, not caused by the `safety`/`ProofClaimStatus` move.
3. **The vendored-tree exclusion itself.** `tools/coverage_audit.py` now prunes nested
   git checkouts, `site-packages` and any `.venv*`, which is what turned 24 red façade
   cases green. That rule is broader than one directory, so it is worth re-checking
   whenever a sandbox vendors another tree.

Status: audit-only.
