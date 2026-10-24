# Architecture Decision Records

Architecture Decision Records (ADRs) capture significant, durable choices about
the repository's structure, interfaces, and engineering process. They explain
the context and consequences so future contributors can revisit a decision
without treating current conventions as unexplained constraints.

## Process

1. Copy [`0000-adr-template.md`](0000-adr-template.md) to the next sequential
   number and use a short, descriptive filename.
2. Submit proposed ADRs with the change that needs the decision. Keep the scope
   focused and document alternatives and consequences, including trade-offs.
3. Review with affected code owners. Accepted records are immutable history:
   supersede an old decision with a new ADR and cross-link both records rather
   than silently rewriting the rationale.
4. Link relevant ADRs from contributor documentation or the affected technical
   guide.

Use `Proposed` while under review, `Accepted` once agreed, and `Superseded` when
a later accepted ADR replaces the decision.

## Index

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-repository-layout-boundaries.md) | Preserve repository zones and active Rust workspace boundaries | Accepted |
| [0002](0002-ci-quality-gates.md) | Gate Python regression tests on Python and Rust quality checks | Accepted |
