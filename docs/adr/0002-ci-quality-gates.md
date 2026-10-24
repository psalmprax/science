# ADR-0002: Gate Python regression tests on Python and Rust quality

- **Status:** Accepted
- **Date:** 2026-10-24
- **Owners/reviewers:** Repository maintainers

## Context

The Python runtime includes a native Rust kernel, so a passing Python test suite
alone does not establish that the Rust workspace is formatted, linted, or
tested. The CI workflow also separates quality jobs from the Python-version
test matrix and has independent distribution and deployment jobs.

## Decision

Keep Python quality (`make lint`, `make typecheck`) and Rust quality
(`make lint-rust`, locked workspace tests) as prerequisites of the Python test
matrix in [APEDP CI/CD](../../.github/workflows/test.yml). Treat the Python
matrix and mutation battery as distinct pull-request checks. Document branch
protection expectations, but configure required statuses in repository
settings; workflow YAML cannot enforce merge policy. Keep wheel checks scoped to
the paths and release events that trigger [Cross-Platform Native Wheels](../../.github/workflows/wheels.yml).

## Alternatives considered

- Run all checks in one job. Rejected because independent failures would be
  harder to diagnose, and quality checks would not be an explicit prerequisite.
- Treat workflow dependencies as branch protection. Rejected because job
  dependencies control execution order, not which checks GitHub requires to
  merge.

## Consequences

Contributors can identify the required quality and test signals before review.
Repository administrators must keep branch-protection selections aligned with
the current job names. Conditional wheel checks should not be described as
universal documentation-only PR gates.

## References

- [CI status matrix](../../README.md#required-checks-and-ci-status-matrix)
- [Checks matching CI](../../CONTRIBUTING.md#checks-matching-ci)
