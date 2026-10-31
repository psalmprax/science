# Which checkout is canonical: the decision

**Status:** decision record, 2026-10-31 · **Decides:** `science` is canonical, the sibling is
a read-only archive until the port list in §4 is drained · **Evidence:** a git-level audit of
both checkouts, reproducible with the commands in §7

Two checkouts of this project exist on this host:

| | path | HEAD | last commit |
|---|---|---|---|
| **canonical** | `/home/psalmprax/ALL_PROJECTS/science` | `75ef0cc` | 2026-10-30 |
| archive | `/home/psalmprax/ALL_PROJECTS/science-quantum-discovery` | `ffa7b54` | 2026-10-27 |

They are not unrelated repositories. Their object stores overlap and they share a real
common ancestor — `444c1fa` — so a port between them is a `git` operation, not a
copy-and-pray. From that ancestor, **the archive has 5 commits this checkout does not**, and
**this checkout has 35 the archive does not**.

## 1. The ruling

**`science` is canonical and is the only writer.** All new work lands here. The archive is
frozen as an archive: no commits, no edits, no new records, until §4 is empty.

## 2. Why this one, on evidence rather than preference

1. **The VQC designer exists only here.** `apedp_dyn/vqc/` (24 modules), its
   eight records, its seven producers and its five documents have no counterpart in the
   archive: `apedp_dyn/vqc` is absent there entirely, as are
   `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`
   and `docs/VQC_SYNTHESIS.md`. A checkout that cannot run the designer cannot be the
   canonical home of it.
2. **It is 35 commits ahead and 5 behind**, and the 5 are almost entirely already here:
   of the 9 files those commits touched, **8 are byte-identical** in this checkout. The
   archive is not carrying a parallel body of work; it is carrying one file's worth of it.
3. **This is where the recent lines live**: the five quantum-signal tranches (2026-10-31),
   the holdout seal, the power/programme record, and the DQI campaign records are all here
   and none of them are in the archive.
4. **Test surface follows**: 1,031 test files here, and the VQC pins (`test_vqc_*.py`, 15
   files) exist only here.

## 3. What the archive is not

It is **not** an independent confirmation of anything. It shares history with this checkout,
so agreement between the two is agreement with itself. Nothing in either checkout becomes
externally validated by the existence of the other, and this decision does not create a
second witness.

## 4. The port list — everything stranded in the archive

Four items. Two are committed there and exist in this object store, so they port by hash;
two never were committed and must be reviewed and copied by hand.

| # | item | size | disposition |
|---|---|---|---|
| 1 | **`ae02214` — `dfthcsos_signed_4bank.py`: `PROMOTION_STATUS_REJECTED` + `REJECTION_FAILURE_EVIDENCE`** | +39 lines | **PORT THIS FIRST.** See §5. |
| 2 | `verify_submission_invsqrt.py`: uncommitted `min_margin_x` diagnostic (returns the worst-margin angle with the margin, not just the margin) | uncommitted diff | review, then port |
| 3 | `scripts/search_davio_exact.c` (270 lines), `scripts/search_shared_davio.c` (271), `scripts/test_davio_fast.c` (148) — OpenMP Davio-order searches | untracked | **check for supersession before porting.** This checkout has `scripts/search_claims_pddd.c`, `scripts/search_mixed_polarity_pddd.py` and `src/shared_davio_results.json`; the Davio plans cited in [`PHOTONIC_ACCELERATOR_BOUNDARY.md`](PHOTONIC_ACCELERATOR_BOUNDARY.md) §5 were produced by those. If the earlier C searches were the path to the same result, they are superseded and should be discarded with the reason recorded, not ported. |
| 4 | `scripts/measure_89q_executed_cost.py` (73 lines) | untracked | port **only after fixing its shebang**: it hard-codes `#!/dev/shm/qvcc-venv313/bin/python`, a machine-local interpreter path. The repository's `test_no_host_specific_home_paths_in_runtime_code` guard matches `/home/...` only, so this path would slip past it — the fix is to invoke it with an explicit interpreter, not to widen the guard. |

Everything else in the archive's working tree was checked and is **already here**: the six
other modified or untracked paths (`instrument_calibrated_trace.py`,
`instrument_invsqrt_trace.py`, `search_claims_pddd.c`, `test_shared_pddd_circuit.py`,
`test_viable_optimization_path.py`, and the verifier file itself) all exist in this checkout.

## 5. Why item 1 is the one that matters

The archive formally **rejected** the DFTHCSOS 10-bit signed 4-bank candidate and preserved
the arithmetic: `error_1` 134.854 mHa against a 20 mHa threshold (6.74×), `error_2` 44.889
mHa (2.24×), peak angle qubits **132 against a 128 cap** (headroom −4), and a 4,363-Toffoli
gap between the literal formula (69,607) and the claim (65,244). This checkout's copy of the
same file still reads `PROMOTION_STATUS_HOLD — HOLD_FOR_ARITHMETIC_AND_RECEIPT_RECONCILIATION`,
i.e. **undecided**.

So this is not a style difference. Read from this checkout alone, a candidate that was
disproved — on accuracy, on a hard qubit cap, and on arithmetic — looks like it is still
waiting for a reconciliation that has already happened. That is a false open question, and
it is the only stranded item that can mislead a reader about what the platform knows.

Port it with the commit, not by hand, so the failure evidence keeps its provenance:

```bash
git -C /home/psalmprax/ALL_PROJECTS/science cherry-pick ae02214
```

*(Not executed by this audit: porting is a decision for the owner, and the audit's job is to
name the item and the command.)*

## 5a. The canonical checkout has the same problem, in a worse form

This audit found something it was not looking for, and it changes what "canonical" is worth
today. `git status` in **this** checkout lists **89 untracked paths**, and they include the
things the ruling in §1 rests on:

```
?? apedp_dyn/vqc/                                   # the whole designer, 24 modules, 11,653 lines
?? apedp_dyn/epistemic/holdout_seal.py               # the burn-log seal
?? apedp_dyn/quantum_compilation/quantum_signal.py   # the signal line
?? apedp_dyn/qed/
?? benchmarks/quantum_compilation/vqc_*.json         # the eight designer records
?? benchmarks/quantum_compilation/quantum_signal_ledger*.json
?? benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl{,.seal}
?? docs/VQC_SYNTHESIS.md, docs/QUANTUM_SIGNAL_LEDGER.md, this file
```

So the designer is not "only in the canonical checkout" in the sense that matters: it is
only in the canonical checkout's **working tree**. It appears in no commit, therefore in no
history, therefore in no clone — including the archive's, which is why the archive has no
`apedp_dyn/vqc` at all. The two checkouts are not two copies of one project; they are a
committed history plus one uncommitted working tree, and the uncommitted part is the more
valuable half.

The same applies to the sealed burn registry: the ledger and its `.seal` are untracked, so
the seal's guarantee is bounded by a file that a `git clean -xdf` would delete. The 282-line
registry is the only record that 88 slices are spent.

**Disposition.** One commit, on a branch, before any further work — and it is deliberately
not made by this audit, because committing is a write the owner has not authorised. Nothing
in §4 is worth porting until the tree this ruling protects actually exists in a commit; a
canonical checkout whose designer is untracked is one `git clean` away from having no
designer at all.

## 6. What must not happen

* **Do not merge the archive branch wholesale.** It is 5 commits ahead and 35 behind; a merge
  would either reintroduce 2026-10-27 state or produce a conflict surface far larger than the
  one file that actually differs.
* **Do not copy artifacts, records or benchmarks across.** A record is only meaningful with
  the code and data that produced it; copying `benchmarks/*.json` between checkouts would
  move a measurement into a tree where it cannot be regenerated.
* **Do not treat the archive as a backup.** Two divergent copies of the same history is how
  a fix gets lost; once §4 is drained, delete it or convert it to a read-only tarball.

## 7. Reproduce

```bash
S=/home/psalmprax/ALL_PROJECTS/science
T=/home/psalmprax/ALL_PROJECTS/science-quantum-discovery
git -C $S merge-base "$(git -C $T rev-parse HEAD)" HEAD      # 444c1fa
git -C $S log --oneline "$(git -C $T rev-parse HEAD)" --not HEAD   # 5 commits, archive only
git -C $S rev-list --count HEAD --not "$(git -C $T rev-parse HEAD)" # 35 commits, here only
for f in $(git -C $S diff --name-only 444c1fa "$(git -C $T rev-parse HEAD)"); do
  diff -q "$S/$f" "$T/$f" >/dev/null 2>&1 && echo "identical $f" || echo "differs   $f"
done
```

## 8. What would reopen this

* A commit in the archive that is not in this checkout's object store — it would not, since
  the stores overlap, so this is checked by `git -C $S cat-file -e` rather than assumed.
* A second host or collaborator writing to the archive; at that point the ruling is not
  "freeze it" but "pick one and revoke the other's write access".
* An external replication, which neither checkout currently has and which this decision
  deliberately does not simulate.
