# Proposal: evidence the proposer cannot make up

**Status: a proposal. Nothing here is built or in force.** It changes what the referee's first
stage accepts, so it waits for the owner's word (AGENTS.md 1.5).

## The gap

`PromotionTracker.verify_stage_1_from_lean` (`apedp_referee/promotion_lifecycle.py`) passes a
candidate when a `LeanVerificationResult` says `all_theorems_proven`, `kernel_exit_code == 0`, a
proof-claim status that claims an independent proof, no theorem resting on a declared axiom, and
a proven dimensional and bound theorem. `LeanVerificationResult`
(`apedp_referee/proof_evidence.py`) is an ordinary record: any caller can construct one with those
values. The module says so itself: "Nothing in this module verifies anything." The outside audit
of 24 November named this, and it is true.

So the first stage checks what a record *says* about a Lean run. It does not check that a Lean
run happened, on this candidate, with this source.

## What is proposed

A receipt written by the checker, not by whoever proposes the candidate.

| Field | What it binds |
|---|---|
| `candidate_id`, `formula_sha256` | The receipt to one candidate and one formula |
| `lean_source_sha256` | The exact Lean file that was checked |
| `toolchain` | The Lean version and the hash of the checker's own code |
| `theorems` | For each: its identifier, its statement's hash, proven or not, whether it rests on an axiom the module declared |
| `kernel_exit_code`, `transcript_sha256` | What the kernel returned and the hash of its full output |
| `issued_utc`, `nonce` | When, and a value that makes a replay of an old receipt detectable |
| `signature` | An Ed25519 signature over all of the above, in a fixed serialisation |

Three rules make it worth something:

1. **The key is not where the research code runs.** The checker runs as its own job (on GitHub: a
   job that alone is given the signing secret, as the trustees' keys are held by the trustees).
   The process that proposes candidates never sees it.
2. **The first stage verifies the receipt, not the record.** It recomputes the candidate's and the
   formula's hashes, checks the signature against a public key pinned in the specification, and
   refuses a receipt whose toolchain is not the pinned one. A `LeanVerificationResult` without a
   valid receipt is judged as today's code judges a schema check: capped at P1.
3. **The transcript is kept.** The receipt's hash points at a stored transcript, so a second
   party can run the same Lean file and compare.

The signing and verifying code already exists for the governance manifest
(`apedp_dyn/epistemic/governance.py`: `sign_spec_sha256`, `verify_ed25519_signature`); the referee
package would carry its own copy of the verifier, since it must import nothing from the rest of
the repository.

## What it costs and what it does not solve

- One new record type in `apedp_referee`, one new check in the first stage, one pinned public key
  in the canonical specification. The last is a change to the specification: a new pin, a
  CHANGELOG entry, and the trustees' signature over the new digest (the manifest is already
  waiting for one).
- Every existing caller that builds a `LeanVerificationResult` by hand, tests included, would be
  capped at P1 until it goes through the checker. That is the point, and it will turn some tests
  red until they are given receipts from a test key.
- It does not make a proof meaningful. A theorem that says little is still proven. It only
  establishes that the kernel accepted this file for this candidate.
- The same shape would serve the other checkers (the circuit checker, the exact tensor check of
  MMLM-POC-001). Those are not part of this proposal.

## What the owner is asked

Whether to build it. If yes, it can be done in the same ceremony as the manifest's re-signing,
since both change what the trustees sign.
