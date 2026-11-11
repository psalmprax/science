"""Regression guards for CI environment and blocking quality gates."""
from __future__ import annotations

import re
import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    # BaseLoader never instantiates objects (strings only), so the S506
    # arbitrary-objects-instantiation concern does not apply here.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)  # noqa: S506


def test_quality_jobs_block_tests_and_run_parity_commands():
    jobs = _workflow("test.yml")["jobs"]
    assert set(jobs["test"]["needs"]) == {"python-quality", "rust-quality", "research-models"}
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
        for job_name, job in _workflow(path.name)["jobs"].items():
            if job_name == "python-quality" and path.name == "test.yml":
                continue
            if job_name == "research-models" and path.name == "test.yml":
                continue
            dependencies_installed = False
            for step in job.get("steps", []):
                script = step.get("run", "")
                if re.search(r'pip install -e "\.\[[^"]*\btest\b[^"]*\]"', script):
                    dependencies_installed = True
                if "python -m pytest" in script:
                    assert dependencies_installed, (path.name, step.get("name"))
    script = "\n".join(
        step.get("run", "") for step in _workflow("test.yml")["jobs"]["test"]["steps"]
    )
    assert "pip install -e packages/jetflow-video" in script


def test_python_quality_job_avoids_scientific_test_extras():
    script = "\n".join(
        step.get("run", "")
        for step in _workflow("test.yml")["jobs"]["python-quality"]["steps"]
    )
    assert "pip install --require-hashes -r requirements-quality.lock" in script
    assert not re.search(r'pip install -e "\.\[', script)


def test_quality_lock_has_pinned_packages_and_hashes():
    lock = (ROOT / "requirements-quality.lock").read_text()
    assert "--hash=sha256:" in lock
    requirements = [
        line for line in lock.splitlines()
        if line and not line.startswith(("#", " ", "-"))
    ]
    assert requirements
    assert all("==" in line for line in requirements)


def test_expression_evaluator_branch_coverage_is_a_ci_gate():
    script = "\n".join(
        step.get("run", "")
        for step in _workflow("test.yml")["jobs"]["test"]["steps"]
    )
    assert "tests/test_expression_evaluator.py" in script
    assert "--cov-branch" in script
    assert "--include='*/apedp/expression_evaluator.py' --fail-under=95" in script


def test_ci_torch_version_matches_project_test_extra():
    project = (ROOT / "pyproject.toml").read_text()
    pinned_torch = re.search(r'"torch==([^"]+)"', project).group(1)
    test_job = _workflow("test.yml")["jobs"]["test"]
    script = "\n".join(step.get("run", "") for step in test_job["steps"])
    assert f"torch=={pinned_torch}" in script
    assert "https://download.pytorch.org/whl/cpu" in script


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


def test_security_scans_cover_dynamic_package_for_high_severity_findings():
    script = "\n".join(
        step.get("run", "")
        for step in _workflow("test.yml")["jobs"]["test"]["steps"]
    )
    assert "bandit -r apedp/ -ll" in script
    assert "bandit -r apedp_dyn/ -lll" in script


def test_strict_mypy_scope_includes_the_d5_conf_benchmark():
    project = (ROOT / "pyproject.toml").read_text()
    files = re.search(r"(?ms)^files = \[(.*?)^\]", project).group(1)
    assert '"apedp_dyn/conf/paths.py"' in files
    assert '"apedp_dyn/conf/canonical_spec.py"' in files
    assert '"apedp_dyn/conf/benchmark.py"' in files
    assert '"apedp_dyn/quantum_compilation/faros_runtime.py"' in files
    assert '"apedp_dyn/quantum_compilation/kickmix_builds.py"' in files


def test_research_models_are_gated_without_native_runtime():
    steps = _workflow("test.yml")["jobs"]["research-models"]["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    assert "pip install --require-hashes -r requirements-research.lock" in script
    for model in ("normalized_invsqrt", "horner_invsqrt", "parity_phase"):
        assert f"scripts/check_{model}.py" in script
    assert '-k "not native"' in script


def test_rust_home_is_set_in_a_step_not_invalid_job_context():
    job = _workflow("test.yml")["jobs"]["test"]
    assert "RUSTUP_HOME" not in job.get("env", {})
    assert any('RUSTUP_HOME=$RUNNER_TEMP/rustup' in step.get("run", "") for step in job["steps"])


def test_mutation_battery_installs_its_native_prerequisite():
    job = _workflow("test.yml")["jobs"]["mutation-battery"]
    assert job["needs"] == "build-kernel-wheel"
    steps = job["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    assert "python -m pip install wheelhouse/*.whl" in script
    assert "assert is_rust_kernel_available()" in script
    assert any(step.get("uses", "").startswith("actions/download-artifact@") for step in steps)


def test_mutation_anchors_match_the_pristine_sources():
    battery = runpy.run_path(str(ROOT / "scripts" / "run_mutation_battery.py"))
    for mutant in battery["MUTANTS"]:
        source = (ROOT / mutant["file"]).read_text()
        assert source.count(mutant["find"]) == 1, mutant["id"]
        assert mutant["replace"] != mutant["find"], mutant["id"]


def test_workflows_have_no_conflicts_or_job_level_secret_conditions():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text()
        assert not re.search(r"(?m)^(<<<<<<<|=======|>>>>>>>)", text)
        for job in _workflow(path.name)["jobs"].values():
            assert "secrets." not in job.get("if", "")
