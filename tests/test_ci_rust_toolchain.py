"""Regression guards for the CI Rust toolchain's isolation and lint availability."""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "test.yml"


def test_rustup_home_is_isolated_before_any_rust_step():
    """RUSTUP_HOME must be job-local for every Rust step, and must be set in a
    step. This test used to require ``RUSTUP_HOME: ${{ runner.temp }}/rustup``
    in the job-level ``env`` block. The ``runner`` context does not exist
    there, so that line made GitHub reject the whole workflow file: no job in
    it ran on ``stage`` from 2026-10-20 to 2026-11-15, and this test, which
    could only run inside that workflow, was pinning the defect in place."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    test_job = workflow.split("\n  test:\n", 1)[1].split("\n  build:\n", 1)[0]
    job_config, steps = test_job.split("\n    steps:\n", 1)

    assert "RUSTUP_HOME" not in job_config
    isolation_step = steps.index("      - name: Isolate Rust toolchain home\n")
    rust_setup_step = steps.index("      - name: Set up Rust toolchain\n")
    assert isolation_step < rust_setup_step
    isolation = steps[isolation_step:rust_setup_step]
    assert 'echo "RUSTUP_HOME=$RUNNER_TEMP/rustup" >> "$GITHUB_ENV"' in isolation


def test_the_toolchain_is_installed_into_an_empty_home_before_anything_else_runs():
    """Twice on 2026-11-16 rustup found a half-installed 1.99.0 in the isolated
    home before the install step and failed on a clippy file conflict. The home
    is now emptied (with a warning naming what was there) and the toolchain is
    installed in the very next step, so no other step can run a Rust proxy
    against it first."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    test_job = workflow.split("\n  test:\n", 1)[1].split("\n  build:\n", 1)[0]
    names = [line.strip() for line in test_job.splitlines() if line.startswith("      - name:")]
    isolation = names.index("- name: Isolate Rust toolchain home")
    assert names[isolation + 1] == "- name: Set up Rust toolchain"

    step = test_job.split("      - name: Isolate Rust toolchain home\n", 1)[1].split("\n      - name:", 1)[0]
    removal = step.index('rm -rf "$RUNNER_TEMP/rustup"')
    assert removal < step.index('echo "RUSTUP_HOME=$RUNNER_TEMP/rustup" >> "$GITHUB_ENV"')
    assert "::warning::" in step[:removal]


def test_rust_lint_components_remain_installed():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    setup = workflow.split("      - name: Set up Rust toolchain\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    assert "uses: dtolnay/rust-toolchain@" in setup
    assert "toolchain: '1.99.0'" in setup
    assert "components: rustfmt, clippy" in setup
