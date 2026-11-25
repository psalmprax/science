# Governance Manifest Re-Signing Request (FLAW-01)

**Date:** 2026-11-07
**Status:** OPEN — blocked on the registered trustees. This is the only step in
FLAW-01 that repository code and CI cannot perform.

**Read section 8 first.** On 2026-11-25 the owner decided that the trustees sign
once for two files. Section 8 gives the one message to sign and replaces the
message of section 3 and the assembly of section 4. Sections 1, 2, 5 and 6 still
hold.

## 1. What is wrong

The canonical specification and the signed governance manifest cover different
revisions:

| Item | SHA-256 |
|---|---|
| `specifications/canonical_preregistered_spec_v1.json` on disk | `a8177b252d81bf5a60fb58916dc850f323810c11434536dd2c5ab0f2a8f37329` |
| Python pin in `apedp_dyn/conf/canonical_spec.py` | `a8177b252d81bf5a60fb58916dc850f323810c11434536dd2c5ab0f2a8f37329` |
| `spec_sha256` sealed by `specifications/governance_manifest.json` | `a6d026f303a77cbb2f1dd5f8a74c394dd09f8139d811a9bfe4142c82063f6d47` |

The Python pin and the file agree. The Ed25519 manifest does not cover the file,
so `tests/test_dyn_governance.py::test_canonical_repository_governance_manifest_valid`
fails (4 passed, 1 failed) and `python scripts/check_governance_manifest_drift.py
--check` exits 1. This is the intended fail-closed behaviour, not a verification
bug: the existing signatures are cryptographically valid, but they sign the old
hash.

The spec was revised three times after the 2026-10-18 signature ceremony, each
time adding a threshold rather than changing an existing value (pin history in
`apedp_dyn/conf/canonical_spec.py`):

- 2026-10-21: ODE referee holdout thresholds `ode_referee_min_mean_r2=0.95`,
  `ode_referee_min_worst_r2=0.85`, moved out of hardcoded literals.
- 2026-10-28: temporal-rate modulation null-test thresholds.
- 2026-10-29: `temporal_rate_min_compositions=2`.

No teammate or agent may re-seal this themselves: that would make the trust
anchor worthless. **Do not** generate replacement keys, add a trustee, lower the
quorum, edit the spec to match the old hash, or reuse the old signatures.

## 2. Who must act

Quorum is **2 of 3** (`quorum_threshold: 2`). Any two of the registered
trustees may sign; the two current signers are the natural choice for continuity.

| Trustee ID | Role | Ed25519 public key (hex) |
|---|---|---|
| `referee_lead_01` | Chief Referee | `9cb876ccb44b63fc8bad3b580ea86a3e14bb81c90361e801ab0a8675a8bf9dfd` |
| `referee_stat_02` | Statistical Referee | `05fd93155bdf784c5eac8581866521c027cbb9b366c6c52732d42084adae2680` |
| `auditor_formal_03` | Formal Verification Auditor | `27924bd7c02fd38f738d06ced27f4d95825c3dcf803012d628ea106f760b39e6` |

Current signature records (both over `a6d026f3…`, 2026-10-18T16:31:16.664147+00:00):

- `referee_lead_01`: `5a8a03767b1352c3d749246f8dc40c67986d6f1a5c4fcf3df8bfe6a0af4ad6318cf199cb1dacad0ba4b864a6503435c6dc7857b7a364ba5746ef6b289db4f103`
- `referee_stat_02`: `c7ef865bd3b1d83006b0755c250e2faf5abfcf0229e55e8e042f252e685da27894cedffc5707355f0d3094bcc2ac0f5bd94d2b471c8d7940b46db92dad3e980b`

These signatures are **not reusable**. Each trustee must produce a new signature
over the new digest.

## 3. Exactly what each trustee must send

Sign the **ASCII hex string** below (not the file bytes, not a JSON fragment):

```
a8177b252d81bf5a60fb58916dc850f323810c11434536dd2c5ab0f2a8f37329
```

With the repository's own signer:

```bash
python - <<'PY'
from apedp_dyn.epistemic.governance import sign_spec_sha256

# Read the private key from the trustee's own secret store. Never commit it,
# never paste it into this repository, and never send it to the maintainer.
private_key_hex = "<from trustee secret store>"
new_digest = "a8177b252d81bf5a60fb58916dc850f323810c11434536dd2c5ab0f2a8f37329"
print(sign_spec_sha256(private_key_hex, new_digest))
PY
```

Send back only:

1. `trustee_id`
2. `signature_hex` — exactly 128 lowercase hex characters (64-byte Ed25519 signature)
3. `signed_at` — ISO-8601 UTC, e.g. `2026-11-07T14:05:00+00:00`

Before sending, the trustee can verify their own signature against their public
key above:

```python
from apedp_dyn.epistemic.governance import verify_ed25519_signature
assert verify_ed25519_signature(public_key_hex, new_digest, signature_hex)
```

Never send the private key. A signature is the only artifact required.

## 4. How the maintainer assembles the manifest

Once at least two signatures arrive, replace `spec_sha256` and `signatures`,
keep the trustee roster and quorum unchanged, and **append** two transparency-log
entries. Do not rewrite entries 0 and 1: the log is append-only and its current
head is `12873a9066959ec19bb645e00ec9e91a0fa4b9192f3de17cdb2395762ce9fae9`.

The entry hash is:

```python
sha256(f"{index}|{timestamp}|{action}|{spec_sha256}|{trustee_id}|{prev_entry_hash}")
```

Assembly reference:

```python
import datetime
from apedp_dyn.epistemic.governance import (
    GovernanceManifest,
    SignatureRecord,
    TransparencyLogEntry,
    verify_governance_manifest,
)

NEW = "a8177b252d81bf5a60fb58916dc850f323810c11434536dd2c5ab0f2a8f37329"
manifest = GovernanceManifest.from_file("specifications/governance_manifest.json")

# 1. New signatures, supplied by the trustees.
manifest.spec_sha256 = NEW
manifest.signatures = [
    SignatureRecord("referee_lead_01", "<lead signature>", "<lead signed_at>"),
    SignatureRecord("referee_stat_02", "<stat signature>", "<stat signed_at>"),
]

# 2. Append the revision entries, chaining from the existing head.
prev = manifest.transparency_log[-1].entry_hash
for index, (trustee, timestamp, action) in enumerate(
    [
        ("referee_lead_01", "<lead signed_at>", "SPEC_REVISION"),
        ("referee_stat_02", "<stat signed_at>", "QUORUM_CO_SIGNATURE"),
    ],
    start=len(manifest.transparency_log),
):
    entry_hash = TransparencyLogEntry.compute_hash(
        index, timestamp, action, NEW, trustee, prev
    )
    manifest.transparency_log.append(
        TransparencyLogEntry(
            index=index,
            timestamp=timestamp,
            action=action,
            spec_sha256=NEW,
            trustee_id=trustee,
            prev_entry_hash=prev,
            entry_hash=entry_hash,
        )
    )
    prev = entry_hash

# 3. Refuse to write a manifest that does not verify.
result = verify_governance_manifest(
    manifest, "specifications/canonical_preregistered_spec_v1.json"
)
assert result.is_valid, result.errors
manifest.save_to_file("specifications/governance_manifest.json")
```

If a third trustee signs, append a third entry (`action="QUORUM_CO_SIGNATURE"`)
and include the signature; the quorum stays 2.

## 5. Acceptance criteria

All three must pass on the commit that carries the re-signed manifest:

```bash
python scripts/check_governance_manifest_drift.py --check
python -m pytest tests/test_dyn_governance.py -q
python -m pytest tests/test_documentation_and_spec_integrity.py -q
```

Expected: the drift script exits 0, and the governance suite is **5 passed**
instead of 4 passed / 1 failed. The spec file itself must not change; its digest
must still be `a8177b25…` after the ceremony.

## 6. What is explicitly out of scope

- Editing `specifications/canonical_preregistered_spec_v1.json`.
- Changing `PINNED_SPEC_SHA256` in `apedp_dyn/conf/canonical_spec.py`.
- Changing the trustee roster or `quorum_threshold`.
- Generating keys inside this repository or CI.
- Committing a manifest with placeholder or agent-generated signatures.

## 7. Checked again on 2026-11-24

The request was checked against the tree: the specification on disk and its pin
are both `a8177b25…f37329`, the manifest still seals `a6d026f3…3f6d47`. The second
specification, `specifications/discovery_checks_v1.json`, held the thresholds
that decide whether a found law is reported and was pinned by hash in code only.
Whether the trustees should seal it too was put to the owner.

## 8. The ceremony as decided on 2026-11-25: one signature for both files

The owner decided: the trustees sign once, for both specifications.

**What is sealed.**

| File | SHA-256 |
|---|---|
| `specifications/canonical_preregistered_spec_v1.json` | `a8177b252d81bf5a60fb58916dc850f323810c11434536dd2c5ab0f2a8f37329` |
| `specifications/discovery_checks_v1.json` | `60eabf58a23c9f2d366d45875e427999bb34c4086cb74107caf9dabd7bb220d0` |

The second file changed on 2026-11-25, before this ceremony and by the same
decision: the limit on the share of wrong laws among those reported got its own
name there, with a second condition (see `CHANGELOG.md`). The digest above is the
file with that change.

**The message each trustee signs** is these 64 characters, as ASCII text:

```
2c3f7268c1698e71bede3417e4180e845c1fbc7b32f61b87dd32cea148ab73bb
```

It is the SHA-256 of a fixed text naming both files and their digests
(`apedp_dyn.epistemic.governance.combined_digest`). Anyone can print it again
from the tree, and should before signing:

```bash
python scripts/governance_ceremony.py message
```

If either file changes before the ceremony, the message changes and this section
must be rewritten; a test fails until it is
(`tests/test_governance_two_files.py`).

**How a trustee signs**, on their own machine, with their own key:

```bash
python -c "from apedp_dyn.epistemic.governance import sign_spec_sha256; print(sign_spec_sha256(input('private key hex: ').strip(), '2c3f7268c1698e71bede3417e4180e845c1fbc7b32f61b87dd32cea148ab73bb'))"
```

They send back three things and nothing else: their trustee id, the 128 hex
characters printed, and the time of signing (ISO-8601, UTC). The private key is
never sent, never pasted into this repository, and never given to an agent.

**How the manifest is assembled**, once two of the three have signed:

```bash
python scripts/governance_ceremony.py assemble \
    --signature referee_lead_01:<128 hex>:<signed at> \
    --signature referee_stat_02:<128 hex>:<signed at>
```

The script keeps the trustees and the quorum as they are, appends to the
transparency log without rewriting it, checks every signature against its
trustee's registered public key over the message above, and writes the manifest
only if the whole of it verifies. It cannot sign and cannot make a key. An agent
may run it with signatures the trustees supply.

**After it**, all of these pass on the commit that carries the new manifest, and
the `--deselect` of the governance test in `.github/workflows/test.yml` is removed:

```bash
python scripts/check_governance_manifest_drift.py --check
python -m pytest tests/test_dyn_governance.py tests/test_governance_two_files.py -q
```

**What is still out of scope:** everything in section 6. The old signatures over
`a6d026f3…` are not signatures over this message and cannot be reused; a test
checks that they do not verify over it.

**One thing to know before signing.** If the proposal for signed evidence
receipts (`docs/PROPOSAL_EVIDENCE_RECEIPTS_2026-11-24.md`, approved in principle
on 2026-11-25, not built) is built later, it adds a public key to the canonical
specification, which changes its digest and needs this ceremony again.
