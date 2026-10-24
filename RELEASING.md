# Releasing

Releases are prepared from `main` and use the changelog and version metadata for
the artifact being released. The repository does not have one shared version:
for example, the root Python project and native Rust kernel currently declare
different versions. Apply SemVer independently to each published artifact, and
do not change unrelated package versions merely to make them match.

## Release checklist

1. **Choose the scope and version.** Decide which artifact(s) are being released
   and choose a SemVer increment appropriate to their public API changes. Check
   the relevant `pyproject.toml` or `Cargo.toml` version and confirm the intended
   tag does not already exist.
2. **Preflight the source.** Start from an up-to-date `main` checkout with a
   clean working tree. Review open release blockers and verify the applicable
   checks are green for the exact commit:
   - `APEDP CI/CD`: `Python quality`, `Rust quality`, every `test` Python-version
     matrix leg, and `mutation-battery`.
   - `Cross-Platform Native Wheels` when Rust, Cargo metadata, or packaged
     subprojects are in release scope. This workflow has PR path filters and
     may not run for documentation-only changes.
   Branch protection is configured in GitHub repository settings, not by these
   workflow files; see [required checks](README.md#required-checks-and-ci-status-matrix).
3. **Finalize the changelog.** Move the released changes from `[Unreleased]`
   into a new `## [X.Y.Z] - YYYY-MM-DD` section, grouped under the Keep a
   Changelog headings. Leave a fresh `[Unreleased]` scaffold at the top. Do not
   turn the pre-governance development notes into fabricated release history.
4. **Verify version consistency.** Ensure the selected artifact's metadata,
   changelog section, and `vX.Y.Z` tag agree. For a release containing multiple
   artifacts, verify each artifact's own version; equal versions are not
   required. Commit the changelog and version changes, then wait for CI on that
   commit.
5. **Tag and publish.** Create an annotated `vX.Y.Z` tag at the verified commit
   and push it using the repository's approved release permissions. Create a
   GitHub Release for that tag, summarize the changelog, and attach or link the
   relevant build artifacts. The wheel workflow builds on `v*` tags; PyPI
   publication of the native kernel occurs only when the repository's
   `PYPI_API_TOKEN` is configured.
6. **Validate after release.** Confirm the GitHub Release points to the intended
   commit, expected artifacts exist and can be installed or consumed in a clean
   environment, and any applicable service health checks pass. Record any
   release-specific validation or known limitations in the release notes.

## Rollback and corrections

If a release causes a problem, stop or roll back any affected deployment using
the deployment platform's established procedure, and communicate the impact.
Do not move or silently reuse a published tag. Where supported, withdraw or
mark a defective artifact appropriately, then prepare a corrected patch release
with a changelog entry and fresh tag. Consider data and compatibility effects
before reverting; coordinate security issues through [SECURITY.md](SECURITY.md).
