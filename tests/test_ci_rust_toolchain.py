"""Regression guards for the CI Rust toolchain's isolation and lint availability."""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "test.yml"


def test_rustup_home_is_set_before_rust_toolchain_setup():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    test_job = workflow.split("\n  test:\n", 1)[1].split("\n  build:\n", 1)[0]
    isolation_step = test_job.index("      - name: Isolate Rust toolchain home\n")
    rust_setup_step = test_job.index("      - name: Set up Rust toolchain\n")

    assert isolation_step < rust_setup_step
    assert 'run: echo "RUSTUP_HOME=$RUNNER_TEMP/rustup" >> "$GITHUB_ENV"' in test_job[
        isolation_step:rust_setup_step
    ]


def test_rust_lint_components_remain_installed():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    setup = workflow.split("      - name: Set up Rust toolchain\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    assert "uses: dtolnay/rust-toolchain@" in setup
    assert "toolchain: '1.99.0'" in setup
    assert "components: rustfmt, clippy" in setup
