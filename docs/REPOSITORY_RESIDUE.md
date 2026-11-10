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
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/` | 32 MB | 1,563 | Vendored third-party Qualtran source, not this repository's code. It was tracked as a gitlink to `https://github.com/kickmix/kickmix.git`, which returns `Repository not found`, so no clone could materialise it and `test_every_census_module_in_apedp_dyn_is_named_in_the_inventory` counted five vendored modules that only one working tree had. The gitlink is removed and the path ignored; nothing tracked imports `qualtran`. `docs/SCAFFOLD_INVENTORY.md` still names the four modules inside it, so a local vendored copy cannot hide a table. |
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

## The eighteen formerly-unindexed modules, re-measured and triaged 2026-11-06

These are real tests. A tracked test is a claim that it passes, so the ones that
still fail are named here rather than added to the index. Re-measured one module at
a time (`-q`, no cache) against current source after this session's repairs:
**1 failed, 249 passed, 1 skipped** across the eighteen modules the earlier
register named.

The earlier snapshot was stale in both directions, which is why this section is a
measurement rather than a list: modules it recorded as failing now pass, and one it
recorded as a collection error now runs (and fails on its assertions, which is the
honest state).

| module | in index | measured | why it is red |
|---|---|---|---|
| `tests/test_scaffold_inventory.py` | yes | 65 passed | Every assertion is implemented: the applied modules, the web-equation extractor, the bundled patent corpus, the academic-endpoint configuration contract, cost-penalised position sizing, the Anscombe statistics return, the publication-text readiness reason and the census/scope baselines. Green, so it is added to the index in this pass. |
| `tests/test_patent_corpus_real.py` | yes | 14 passed | Green after `BUNDLED_CORPUS_PATH`, `corpus_status()` and the operand-only refusal tier landed; the corpus, raw artefacts and provenance records were already on disk and are tracked, so the module is added to the index in this pass. |
| `tests/test_formal_and_applied_audits.py` | no | 1 failed, 2 passed | The applied-basis audit now passes: every returned dictionary in all eight `applied_*` modules carries a basis label, and `benchmarks/applied_basis_report.json` is regenerated from that code. The Lean audit still refuses to certify itself — no file is `checked`. |
| `tests/test_genome_region_fetch.py` | yes | 8 passed | Green after the scan gained `reference_length` and an empty-reference refusal; added to the index in the same pass. |
| `tests/test_patent_bulk_sources.py` | yes | 14 passed, 1 skipped | Green: every logged attempt carries the split it left unchanged, and `grow_patent_corpus.corpus_split()` agrees with the corpus (29 records, 15 searchable, 8 decisive, 7 operand-only, refusals named per record). The one skip is the authenticated OPS claims replay, which needs a recording that only live credentials can produce. |
| `tests/test_dyn_exception_hygiene.py` | yes | 12 passed | Green after this pass. The four breached size ceilings were met by consolidating code — an inlined single-use wrapper, a shared out-of-scope record, a moved seed builder, two never-taken DAG branches — not by re-pinning the baseline. Candidate for the index. |
| `tests/test_dfthcsos_arm3.py` | yes | 8 passed | |
| `tests/test_dyn_drug_discovery_pipeline.py` | yes | 7 passed | |
| `tests/test_scaffold_refusals.py` | yes | 18 passed | |
| `tests/test_dyn_discovery_module_split.py` | yes | 15 passed | |
| `tests/test_dyn_two_dof_saturation_fit.py` | yes | 11 passed | |
| `tests/test_silent_scaffolds_refuse.py` | yes | 7 passed | |
| `tests/test_placeholder_kernels_refuse.py` | yes | 6 passed | |
| `tests/test_sum_named_column_binding.py` | yes | 5 passed | |
| `tests/test_term_builder_structure_audit.py` | yes | 22 passed | |
| `tests/test_external_benchmark_source_verification.py` | yes | 9 passed | |
| `tests/test_region_mask_photo_audit.py` | yes | 11 passed | Ten licensed photographs are present on this machine; the module skips when they or Pillow are absent. |
| `tests/test_dyn_dfthcsos_sub128_family.py` | yes | 15 passed | Archived receipts are present on this machine; the module skips when the pinned interpreter/extension is absent. |

Seventeen of the eighteen now pass. Every module that passed while untracked has
been added to the index, which is what a green run licenses. The two artifact-dependent
modules carry module-level skips for their inputs (Pillow or the licensed
photographs for the mask audit; the pinned interpreter and extension for the
DFTHCSOS family), so tracking them claims a pass where the inputs exist rather than
a guarantee that a bare clone can run them.

The scaffold-inventory row is the one worth reading twice. Its import error was
real — `SCOPE`, `EXTRACTION_IS_LOSSY`, `SUBMITS_JOBS` and `describe` were genuinely
missing, and are now declared with the tiers they actually have. The
`applied_synthetic_bio` half is no longer a stub, the displaystyle/nested-brace
extractor returns whole formulas, the patent corpus is reachable from the package
with its operand-only refusals listed, and the static-table/scope-scan records match
the tree, and the academic endpoint table is configuration-only with
`CONNECTS_TO_SERVICES = False` and an anonymous Crossref metadata route. Every
scaffold assertion is now implemented: the finance demo sizes from a
cost-penalised mean-variance rule over a forecast, covariance and fee, the
Anscombe scenario returns the statistics it computed, and the publication-text
gate states why it never opens. No assertion in that file is still waiting on code.

## The AGENTS.md dynamic suite, run whole (2026-11-07)

The 24 modules listed in `AGENTS.md` §2, one process, no cache, with
`APEDP_Q_KICKMIX_PYTHON` pointing at `build/qvcc-venv313/bin/python`:
**275 passed, 0 failed, 0 skipped**. Without that environment variable the
kickmix screen test skips by design (274 passed, 1 skipped), because the test
needs a CPython 3.13 interpreter that can load the pinned extension.

The frozen-tree guard now passes. The failure recorded earlier in this file was
the guard working: `apedp_dyn/ui/server.py` changed in `c38a172` (2026-11-06)
after the active record `APEDPX_FROZEN_v1_10.json` was committed in `7fb2082`
(2026-10-26), and the guard refused to score blind cases until a record covered
the change. The repair was to mint `APEDPX_FROZEN_v1_13.json` through the same
review as the refactor, superseding v1.12 (the newest on disk) and carrying all
fifteen file hashes plus a `remint_log` entry for `server.py`. The review found
that `c38a172` moved the registry discovery route body into
`apedp_dyn/ui/registry_endpoint.py`: an AST-normalised diff of the moved body
against the old one is a single added import of the shared helpers, and
`server.discover_registry` now delegates to `run_registry_discovery(req)`. No
route, response shape, adjudication rule, threshold or blind case changed, and
no blind case imports either module, so the record states "No referee change".
`blind_benchmark.FROZEN_RECORD` was still pointing at v1.10 despite v1.11 and
v1.12 existing; it now names v1.13. `tests/test_dyn_blind.py` is 10 passed.

The kickmix skip is also gone where the 3.13 environment exists: `python3.13
scripts/setup_verifier_env.py` built `build/qvcc-venv313`, and the runtime
dependencies the subprocess imports (`sympy`, `pint`, `pypdf`, `python-docx`,
`matplotlib`, `seaborn`, `networkx`, `fastapi`, `uvicorn`, `scikit-learn`,
`cryptography`, `pandas`) were installed into it. The screen test then runs the
verifier in that interpreter instead of skipping.

This is a different population from the eighteen-module table above: that table
includes `tests/test_patent_bulk_sources.py` and the Lean audit pin and excludes
`tests/test_dyn_blind.py`, so the two counts are reported separately rather than
summed.

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

## Optional interpreters (local only, rebuilt on demand)

Two virtual environments served the quantum-compilation and region-mask analysis arms.
Neither is tracked, neither is needed by the test suite, and together they were 247 MiB
of local residue, so both were deleted on 2026-11-09. The pins below are the recipe;
neither environment is a source file, and deleting the directories removed bytes rather
than knowledge. The scripts print the same setup hint when the import fails.

`.venv-kickmix` is the interpreter that can load the shipped `kickmix.so`. The project
`.venv` is CPython 3.12 and cannot load a 3.13 extension, so this environment is 3.13
with numpy and nothing else: `kickmix` is never installed, it stays in the competition
bundle and is reached through `PYTHONPATH`.

```bash
uv venv .venv-kickmix --python python3.13
uv pip install --python .venv-kickmix/bin/python numpy
```

It is used by `scripts/run_mr_gated_arms.py`, `scripts/prototype_bitserial_sqrt.py`,
`scripts/build_invsqrt_chunked.py`, `scripts/measure_dqi_liveness.py` and the rest of the
`invsqrt`/`dqi` probes; `APEDP_Q_KICKMIX_PYTHON` points at it wherever a script has to
re-exec itself.

`.venv-region-models` is CPython 3.12 created with `--system-site-packages`, carrying the
pins the two region-mask comparison arms were measured on. `torch` 2.3.1+cpu is not
installed by these lines at all: it comes from the host's system site-packages, which is
why the environment is created with `--system-site-packages` and the torchvision line is
`--no-deps`. Every pin is load-bearing: transformers 5.x requires a torch newer than 2.3,
and 4.44 requires `tokenizers<0.20`.

```bash
python3 -m venv --system-site-packages .venv-region-models
.venv-region-models/bin/pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cpu "torchvision==0.18.1+cpu"
.venv-region-models/bin/pip install --no-deps "timm==0.9.16"
.venv-region-models/bin/pip install --no-deps "transformers==4.44.2"
.venv-region-models/bin/pip install --no-deps "tokenizers==0.19.1"
.venv-region-models/bin/pip install --no-deps "huggingface-hub==0.25.2"
```

The recorded provenance in `data/region_mask_audit/model_comparison.json` and
`data/region_mask_audit/semantic_comparison.json` names torch 2.3.1+cpu,
torchvision 0.18.1+cpu, timm 0.9.16 (and, for the semantics arm, transformers 4.44.2)
under CPython 3.12.3, with this environment's interpreter path in the `executable`
field. A rebuilt environment is therefore comparable to the published measurements
rather than merely similar.

`scripts/compare_region_split_to_depth.py`, `scripts/compare_region_split_to_semantics.py`
and `docs/REGION_MASK_AUDIT.md` carry the same commands; this section exists so the
recipe survives the deletion of the directories it builds.
