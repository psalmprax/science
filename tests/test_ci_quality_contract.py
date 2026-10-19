"""Regression guards for CI environment and blocking quality gates."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def test_quality_jobs_block_tests_and_run_parity_commands():
    jobs = _workflow("test.yml")["jobs"]
    assert set(jobs["test"]["needs"]) == {"python-quality", "rust-quality"}
    for job, commands in (
        ("python-quality", ("make lint", "make typecheck")),
        ("rust-quality", ("make lint-rust", "cargo test --workspace --locked")),
    ):
        steps = jobs[job]["steps"]
        script = "\n".join(step.get("run", "") for step in steps)
        for command in commands:
            assert command in script
        assert jobs[job].get("continue-on-error", "false") == "false"
        assert all(step.get("continue-on-error", "false") == "false" for step in steps)


def test_python_test_jobs_install_test_dependencies_before_pytest():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job in _workflow(path.name)["jobs"].values():
            dependencies_installed = False
            for step in job.get("steps", []):
                script = step.get("run", "")
                if re.search(r'pip install -e "\.\[[^"]*\btest\b[^"]*\]"', script):
                    dependencies_installed = True
                if "python -m pytest" in script:
                    assert dependencies_installed, (path.name, step.get("name"))


def test_rust_builds_use_the_repository_toolchain_and_locked_dependencies():
    toolchain = re.search(
        r'channel = "([^"]+)"', (ROOT / "rust-toolchain.toml").read_text()
    ).group(1)
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job in _workflow(path.name)["jobs"].values():
            for step in job.get("steps", []):
                action = step.get("uses", "")
                if action.startswith("dtolnay/rust-toolchain@"):
                    assert step["with"]["toolchain"] == toolchain
                if action.startswith("PyO3/maturin-action@"):
                    options = step["with"]
                    assert options["rust-toolchain"] == toolchain
                    assert options["maturin-version"] == "1.14.1"
                    if options.get("command", "build") == "build":
                        assert "--locked" in options["args"]


def test_workflows_have_no_conflicts_or_job_level_secret_conditions():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text()
        assert not re.search(r"(?m)^(<<<<<<<|=======|>>>>>>>)", text)
        for job in _workflow(path.name)["jobs"].values():
            assert "secrets." not in job.get("if", "")
