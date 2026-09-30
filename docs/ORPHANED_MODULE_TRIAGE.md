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
| Self-improvement governance | `constitution`, `self_improvement` | `ConstitutionalAuditor`, `ImprovementLedger`, `IndependentGatekeeper`, `ModificationProposal`. 1 of 13 public names live. `self_improvement` imports `constitution`, so these are one cluster. |
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
unrecovered. The first three target superseded APIs and would need
rewriting against current entry points; the `test_zz*` files are debugging
scratch and should be discarded.

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

## In progress elsewhere

`apedp/dyn/__init__.py` reappeared in the working tree at 10:31 on
2026-09-30, untracked, written by a concurrent session that is rebuilding
that package. The `apedp/dyn/` row in the superseded table above may
therefore be resolved by other hands; do not duplicate the work. Check
`git status --porcelain -- apedp/dyn/` before starting it.

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
