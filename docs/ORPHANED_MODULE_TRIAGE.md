# Orphaned module triage

Twenty-six `apedp` modules, seventeen `tests/` files, and thirteen
`benchmarks/` files exist in this repository only as `.pyc` bytecode. They
appear in no commit on any ref (`git log --all --diff-filter=A` confirms),
so they have never been in version control. They were invisible to review,
to `git diff`, to CI, and to anyone restoring the tree.

This records what each one is, so the next person does not rediscover it, and
records the decision not to reconstruct some of them.

## Why this was not detected

Nothing in the repository asserts that every module is tracked. A file can
be deleted, work continues because Python imports from `__pycache__`, and
the only symptom is a module that imports on a machine with warm bytecode and
fails on a fresh clone. `apedp.dyn` currently imports as an empty namespace
package: the package directory still exists, so `import apedp.dyn` succeeds
and returns nothing.

## Why there is no decompiler

No decompiler supports CPython 3.12. `decompyle3` 3.9.3 and `uncompyle6`
3.9.3 both stop well below it, and the `pycdc` distribution on PyPI is
"Please Your Customer Direct Connect", an unrelated CRM search tool, not
Decompyle++.

Consequence: nothing here was *recovered*. A `.pyc` retains docstrings,
literal constants, keyword names, and attribute access, which is enough to
re-author a faithful test. It does not retain comments, formatting, or local
variable names, and for a *module* it is not enough to reconstruct
behaviour without inventing it. Any module reconstructed this way is a
rewrite wearing the original's filename, and is labelled as such.

## Recovered

| File | Status |
| --- | --- |
| `tests/test_singularity_probing.py` | Re-authored. Verified verdict-identical to the bytecode original across 13 mutants by `tools/orphan_equivalence.py`. |
| `tests/test_physics_constraints.py` | Re-authored against the current API. Could not be equivalence-verified: the subject API drifted and the original raises `TypeError` on six of eight tests. Four negative assertions added beyond the original. |
| `apedp/constitution.py` | Re-authored. Verified over the auditor's entire distinguishable input space — 1027/1027 identical, including every `sha256` — by `tools/constitution_equivalence.py`. |
| `apedp/self_improvement.py` | Re-authored. Verified 24/24 by `tools/self_improvement_equivalence.py`, covering all detector rules, all gatekeeper paths, and all dataclass shapes. |
| `tests/test_constitution.py` | Written new, to pin the recovered governance layer. Found a real defect on the way in; see below. |

## Unrecoverable against the current API — decided, not pending

Three orphaned test files remain, and none of them is a re-authoring job.
Each exercises an entry point that no longer exists, or a feature whose
module is in the lost list above. Writing "recovered" versions against the
replacements would produce tests that pass while asserting something other
than what the originals asserted — lineage the file would not have.

| File | Tests | Missing subject |
| --- | --- | --- |
| `test_evolutionary_integration` | 6 | `SearchConfig` no longer has `use_typed_grammar`, `use_varpro`, `in_loop_epistemic_guard`, or `use_separability`. These are the flags for `apedp/typed_grammar`, `apedp/varpro`, `apedp/epistemic_pruning`, and `apedp/separability` — all four lost. This file is the test suite for exactly the capability that is gone, and is the clearest single statement of what was lost. |
| `test_tri_hypothesis_pipeline` | 2 | `DiscoveryPipeline.run_tri_hypothesis` no longer exists. The pipeline exposes only `run` and `run_prospective`. |
| `test_decisive_benchmark_protocol` | 12 | `DiscoveryPipeline.run_scientific_investigation` no longer exists; the method is now `run`, with a different signature. This is the most valuable artifact in the set — the system's own 12 adversarial cases, with full docstrings surviving in the bytecode — and it is worth recovering as a design document even if it cannot be recovered as a test. |

Recovering any of them is a capability-restoration task, not a bookkeeping
one. The four modules behind `test_evolutionary_integration` would have to
come back first.

## Scratch files — not recoverable by intent

`test_zzdebug_tmp`, `test_zzexact_tmp`, `test_zzwhich_tmp` contain one
throwaway function each (`test_debug_counts`, `test_exact_copy`,
`test_which`) with no module docstring. They are debugging leftovers. There
is nothing to recover and nothing worth recovering; they are recorded here
so the next triage pass does not count them as a loss. Their `__pycache__`
entries were left in place rather than deleted — the tree is shared with
concurrent sessions, and removing cache files for tidiness is not a trade
worth making.

## Superseded — deliberately not reconstructed

Reconstructing these would put a second copy of working code into the tree.
The functionality is live; the original was deleted when its contents were
absorbed elsewhere.

| Orphan | Live home | Evidence |
| --- | --- | --- |
| `apedp/family_detection` | `apedp/discovery.py`, `apedp/discovery_types.py` | 9 of 11 public names present in live source |
| `apedp/bayesian_refinement` | `apedp/epistemic.py` | 17 of 18 |
| `apedp/neural_surrogate` | `apedp/discovery.py`, `apedp/neural_discovery/integration.py` | 3 of 4 |
| `apedp/persistence/experiment` | — | 3 of 4 |
| `apedp/dyn/` (4 modules) | `apedp_dyn/` | Full parallel successor, CI-gated. The orphan tests targeting it fail on *renamed entry points* (`DiscoveryPipeline.run_scientific_investigation`, `SearchConfig(use_typed_grammar=...)`), not on missing capability. The successor API is stricter. |

## Lost — no live equivalent

Real capability gaps. Nothing imports them, so nothing is currently broken;
the capability is simply absent.

| Cluster | Modules | Absent |
| --- | --- | --- |
| Self-improvement governance | `constitution`, `self_improvement` | **RECOVERED** — see above. Found a real defect on the way in: the gate is inert. |
| RH / identifiability frontier | `rh_adversarial`, `rh_dataset`, `rh_evaluator`, `rh_precision`, `precision_scaling_experiment` | ~40 public names, ~4 live |
| Multi-variable decomposability | `separability`, `separability_calibration` | 1 of 9 and **0 of 8** — the only module in the set with no surviving name at all |
| Active discrimination | `evidence_acquisition` | `EvidenceAcquisitionEngine`, `find_discriminating_regime`, `find_bayesian_eig_regime`. 2 of 12. Note `apedp/decision/stage_c.py` still emits `discrimination_required`: the concept survives in the decision layer, the engine that answered it does not. |
| Dimensional grammar | `typed_grammar` | `TypedProductionGrammar`, `DimVector`, `mutate_expression`. 1 of 9 |
| Forensics / evidence reporting | `domain_forensics`, `evidence_report` | `domain_report`, `build_evidence_report`. 1 of 7, 2 of 5 |
| Recovery classification | `canonical_compiler` | `RecoveryTier`, `classify_recovery`, `DimensionalNullSpaceEngine`. 2 of 12 |
| In-loop guard | `epistemic_pruning` | `InLoopEpistemicGuard`. 1 of 5 |
| VarPro | `varpro` | `VarProOptimizer`. 3 of 5 |
| Guided search | `guided_search` | `CandidateArchive`, `grow_at_level`, `level_of`. 10 of 24 — the search-economics half is gone |

The seven remaining binding orphan test files
(`test_decisive_benchmark_protocol`, `test_evolutionary_integration`,
`test_tri_hypothesis_pipeline`, and the `test_zz*_tmp` scratch files) are
dispositioned above: the first three are unrecoverable against the current
API because the entry points they exercise no longer exist, and the `test_zz*`
files are debugging scratch with nothing to recover.

## Separate finding, not a recovery target

An earlier draft of this document described
`apedp_dyn/grammar/coupled.py:117` as "a gap in current, CI-gated code."
That was wrong, and the code says so itself. `dae_mode='constrained'`
raises `NotImplementedError` deliberately, with a 25-line comment
explaining two independent reasons: `_TERM_BUILDERS` maps `c1`/`c2` to
`X**0`, so a constraint reads `con * 1 = 0` and relates nothing; and
nothing performs the index-1 reduction. The comment then records what the
earlier behaviour actually did — the materializer summed the constraint
into the acceleration RHS, gate 1 rejected all 1539 constraint-carrying
candidates, and the 225 that survived were plain ODEs wearing a DAE label,
"which is worse than an error because it looks like a result."

That is the best-documented refusal in the repository, and it is the
opposite of a defect: it is a measured decision to raise rather than ship
something that looks like a result and is not. Implementing it needs
`M(q) qdd = f + grad(g) * lambda` solved and substituted back, which
`PhysicalModel` ("one explicit first-order ODE per state") has nowhere to
record. The DAE provenance fields and the gate 1.5 checks are the
specification for that work. Left alone deliberately.

## A defect found by recovering the governance layer

Worth recording here rather than only in a commit message, because it is the
argument for having done the recovery.

`ConstitutionalAuditor` reads seven boolean flags off a proposal.
`ModificationProposal` can express two of them. The other five always take
their default, and one of those defaults is the violating value:
`supports_ambiguity_preservation` defaults to `False`, which trips INV-9.
So every proposal is discarded, the gate never promotes, and the
self-improvement architecture is inert — a constitutionally clean proposal
that passes the full regression suite is rejected for the single reason that
it did not affirm something it has no field to affirm.

This is not a reconstruction artifact: the equivalence harness compares it
against the orphaned bytecode across every reachable input and finds the
behaviour identical, so the original discarded for the same reason.

The gate is not supposed to be open unconditionally — INV-9 exists so a
proposal that forces a winner is refused, and that is correct. The defect is
the mismatch between the two types: the constitution demands a property the
proposal cannot declare, so compliance is unreachable rather than merely
uncommon. Pinned by `test_the_proposal_type_cannot_currently_satisfy_inv_9`,
which asserts the current behaviour and names both repairs. Neither repair
has been applied, because both are design changes to constitutional
semantics rather than bug fixes.

## In progress elsewhere

`apedp/dyn/__init__.py` reappeared in the working tree at 10:31 on
2026-09-30, untracked, written by a concurrent session that is rebuilding
that package. The `apedp/dyn/` row in the superseded table above may
therefore be resolved by other hands; do not duplicate the work. Check
`git status --porcelain -- apedp/dyn/` before starting it.

## Cluster audit: is any lost cluster still referenced by live code?

The "Lost — no live equivalent" table above asserts that nothing imports these
modules. That was believed, not measured. The audit: extract every public name
from each orphan's bytecode, then grep every *tracked* source file for it.

**Zero of the nine clusters is referenced by live code.** No latent import
break; all nine are capability gaps, as documented.

Two apparent hits, both false positives worth recording because the naive grep
reports them:

* `CandidateEvidence` (from the lost `evidence_report`) appears in
  `apedp/decision/models.py`, `apedp/hybrid.py` and `tests/test_decision.py` —
  but it is **defined** in `apedp/decision/models.py:42`. The orphan's
  identically-named class is a separate, earlier copy, not a dependency.
* `IdentifiabilityReport` (from the lost `varpro`) appears in
  `apedp/identifiability.py`, where it is **defined** at line 27. Same shape:
  name collision, not a reference.
* `tests/test_rh_benchmark.py` mentions `generate_or_load_zeta_zeros` and
  `RiemannAdversarialEngine`, but reaches them through
  `pytest.importorskip("apedp.mathematical.rh_dataset", reason=...)` — a
  *loud skip* with a stated data-loss reason, not a live reference. That is the
  correct pattern for a test whose subject is gone, and it is why the grep
  reports a hit at all.

A name appearing in tracked source is therefore not evidence of a dependency.
The distinguishing question is whether the tracked file *defines* it or merely
mentions it, and a substring grep cannot tell those apart.

| cluster | public names | referenced by tracked source |
| --- | --- | --- |
| RH / identifiability frontier | 11 | no |
| separability | 11 | no |
| active discrimination | 5 | no |
| dimensional grammar | 3 | no |
| forensics / evidence reporting | 9 | no (3 name collisions) |
| recovery classification | 4 | no |
| in-loop guard | 2 | no |
| varpro | 3 | no (1 name collision) |
| guided search | 12 | no |

## Re-run of EXT-BENCH-60 against current code

`benchmarks/run_external_60_apedp.py --out-dir /tmp/rerun60` re-ran all 60
problems at commit `689740e` and later. **This is a new measurement of current
code, not a correction to the frozen round.** The frozen record in
`tournament_results_external_60/` is protocol-frozen and was never written to:
19 JSON files were sha256'd before the re-run and verified byte-identical
afterwards (19/19 OK).

| | frozen | re-run |
| --- | --- | --- |
| abstentions | 4 | 4 |
| identical expression | — | 40/60 |
| families changed | — | 11 |
| family correct (27 named) | 24/27 | 22/27 |

The near-OOD wiring (`689740e`) changed 10 of 60 recorded families and
*lowered* named-family accuracy by 2. Investigating that produced two
committed fixes and one open defect; see `docs/NEAR_OOD_PROMOTION.md` and
`docs/SCALE_INVARIANCE_DEFECTS.md`. One of the 11 changes is a correction in
the engine's favour: A54 now abstains as `INSUFFICIENT_EVIDENCE(underspecified)`
rather than `NO_VALID_CANDIDATE`, because the record is exactly constant
(IQR 0) and the span-is-zero rule in `_regime_identifiability` is the accurate
criterion. The frozen record predates that rule.

## Reproducing this triage

```bash
# modules with bytecode but no source
python3 - <<'PY'
import pathlib
srcs = {str(p.with_suffix("")) for p in pathlib.Path(".").rglob("*.py")
        if ".venv" not in p.parts}
for c in sorted(pathlib.Path(".").rglob("__pycache__/*.pyc")):
    mod = c.parent.parent / c.name.split(".")[0]
    if str(mod) not in srcs:
        print(mod)
PY
```

Note that once a re-authored test is collected, pytest writes a *second*
`.pyc` for it under its own version tag. A "largest file wins" heuristic
will then pick the reconstruction and compare it against itself. Pin the
bytecode tag; `tools/orphan_equivalence.py` does.
