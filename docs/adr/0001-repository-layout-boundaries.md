# ADR-0001: Preserve repository layout and active Rust workspace boundaries

- **Status:** Accepted
- **Date:** 2026-10-24
- **Owners/reviewers:** Repository maintainers

## Context

The repository contains runtime packages, research evidence, benchmarks,
experiments, and multiple Rust manifests with different roles. Existing paths
are referenced by imports, tests, persisted state, and scientific provenance.
In particular, the root Cargo workspace contains `rust/apedp-kernel`; the
separate `crates/apedp_kernel` tree is not a root workspace member.

## Decision

Preserve established paths and the active workspace membership. Use the zones
documented in [CONTRIBUTING.md](../../CONTRIBUTING.md#repository-structure-policy)
for new files. A path or manifest that resembles a runtime package is not by
itself permission to migrate it or change its role.

## Alternatives considered

- Consolidate all packages and Rust sources under a new canonical tree. Rejected
  because the migration would affect imports, packaging, provenance, and
  compatibility without a scoped migration plan.
- Treat every Cargo manifest as part of the root workspace. Rejected because
  the current workspace deliberately selects only the active kernel.

## Consequences

New runtime code must follow the documented ownership map. Any future move
requires an independently reviewed migration plan, updated readers and tests,
and preserved compatibility where needed. This decision does not preclude a
future migration.

## References

- [Repository structure policy](../../CONTRIBUTING.md#repository-structure-policy)
- [Root Cargo workspace](../../Cargo.toml)
