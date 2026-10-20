"""Regression guards for the CI Rust toolchain's isolation and lint availability."""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "test.yml"


def test_rustup_home_is_isolated_for_the_entire_test_job():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    test_job = workflow.split("\n  test:\n", 1)[1].split("\n  build:\n", 1)[0]
    job_config, steps = test_job.split("\n    steps:\n", 1)
    env = job_config.split("\n    env:\n", 1)[1].split("\n    strategy:\n", 1)[0]

    assert "      RUSTUP_HOME: ${{ runner.temp }}/rustup" in env.splitlines()
    assert "RUSTUP_HOME" not in steps


def test_rust_lint_components_remain_installed():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    setup = workflow.split("      - name: Set up Rust toolchain\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    assert "uses: dtolnay/rust-toolchain@" in setup
    assert "toolchain: '1.99.0'" in setup
    assert "components: rustfmt, clippy" in setup
