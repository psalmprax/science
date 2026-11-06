# Repository residue: what is deliberately not in the index

Measured on `stage` on 2026-11-06 with `git status --porcelain`, after the tracking
pass that added 193 files. Everything a tracked file names is now in the index: the
patent corpus, the quantum-compilation benchmark records, the decoder tests' stage
accounting inputs, the audit scripts, and the modules `DOCUMENTATION.md` describes.

What is left out was left out for a reason, and this page is the record. The point of
listing it is that an untracked file a reader cannot see is worse than one they can:
the next sweep starts from this table instead of from `git status`.

## Not in the index

| path | size | files | why |
|---|---|---|---|
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/` | 32 MB | 1,563 | Vendored third-party Qualtran source, not this repository's code. `docs/SCAFFOLD_INVENTORY.md` already names the five modules inside it that the static-table census would otherwise report. |
| `submission/` | 19 MB | 9 | Generated submission archives, not source. The scripts that produce them (`scripts/build_packed_decoder_submission.py`, `scripts/assemble_packed_decoder.py`) are tracked, so the artifacts stay derivable rather than authoritative. |
| `unused/` | 24 KB | 1 | Not imported by any tracked module, which is what `tests/test_tracked_import_closure.py` checks. Keeping it out of the index keeps it out of the import graph. |
| `data/skills/` | — | 0 | Empty directory. Nothing to track; git does not store empty directories anyway. |

## The compiled extension a clone cannot have

`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/kickmix.so` is a
built extension. The index carries the directory's `LICENSE.txt` and the vendored tree,
not the binary, so a clone that has not run `scripts/setup_verifier_env.py` cannot load
the pinned contract.

That used to surface as **18 failures** in `tests/test_quantum_compilation.py` on a clean
clone — the same `FileNotFoundError` raised by `contracts.load_contract`. The module now
skips with the build command when the binary is absent, so the clone reports what is
actually true (this test needs a build step) rather than a red suite. Where the
extension exists, nothing changes: the tests run.

## The sixteen test modules that do not pass

These are real tests, and they are **not** in the index because a tracked test is a
claim that it passes. Measured together on 2026-11-06 against current source:
**57 failed, 80 passed, 1 skipped, 3 collection errors**. They are named here so the
next sweep starts from a list rather than from a red run it has to rediscover.

| module | state |
|---|---|
| `tests/test_scaffold_inventory.py` | collection error — imports `SCOPE` from `apedp.applied_synthetic_bio`, which no longer defines it |
| `tests/test_patent_corpus_real.py` | collection error |
| `tests/test_dfthcsos_arm3.py` | collection error |
| `tests/test_dyn_drug_discovery_pipeline.py` | 2 failing of its own |
| `tests/test_scaffold_refusals.py`, `tests/test_silent_scaffolds_refuse.py`, `tests/test_placeholder_kernels_refuse.py` | failing |
| `tests/test_sum_named_column_binding.py`, `tests/test_term_builder_structure_audit.py` | failing |
| `tests/test_dyn_exception_hygiene.py`, `tests/test_dyn_discovery_module_split.py` | failing |
| `tests/test_dyn_two_dof_saturation_fit.py`, `tests/test_external_benchmark_source_verification.py` | failing |
| `tests/test_formal_and_applied_audits.py`, `tests/test_patent_bulk_sources.py`, `tests/test_genome_region_fetch.py` | failing |

Two further tests were removed from the index rather than named here, because they
cannot run anywhere except the machine that holds the artifacts:
`tests/test_region_mask_photo_audit.py` (ten licensed photographs) and
`tests/test_dyn_dfthcsos_sub128_family.py` (archived receipts). Both stay on disk.

## One defect this pass found

Tracking `apedp_dyn/molecular/equivariant_pocket_flow.py` exposed that it imported
`_signed_volume` from `apedp_dyn/molecular/se3_flow.py`, a name that no longer existed,
so neither it nor `apedp_dyn/biophysics/` nor `apedp_dyn/epistemic/drug_discovery_pipeline.py`
could be imported at all. The helper is restored as `se3_flow.signed_volume` (the
unnormalised determinant the chirality gradient needs, **six times** the tetrahedron
volume), with its convention pinned by two tests in
`tests/test_dyn_molecular_se3_flow.py`: one on a tetrahedron, its mirror and a coplanar
quartet, one comparing the analytic cross-product gradient against finite differences.

The lesson is the reason this file exists: an untracked module is an unmeasured module,
and three of the files in this sweep were importing a symbol that had been deleted
months earlier while every tracked test stayed green.
