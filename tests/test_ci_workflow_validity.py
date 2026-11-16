"""The workflow files are ones GitHub will accept.

``.github/workflows/test.yml`` ran zero jobs on ``stage`` from 2026-10-20 to
2026-11-15: 268 consecutive runs failed in 0 seconds with "workflow file
issue", because one job-level ``env`` entry read ``${{ runner.temp }}`` and the
``runner`` context does not exist at job level. Every gate described as
"CI-enforced" during those four weeks was not running, and nothing inside the
workflow could say so, because nothing inside it started.

Two independent guards:

* ``.github/workflows/workflow-lint.yml`` runs actionlint from a *separate*
  workflow file, so an invalid ``test.yml`` is reported by something that
  still starts.
* These tests check the same class of error in pure Python, so ``make test``
  and a pre-push run catch it before GitHub does.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_CONTEXT = re.compile(
    r"(?<![\w.])(github|env|vars|job|jobs|steps|runner|secrets|strategy|matrix|needs|inputs)\b(?=\s*[.\[])"
)

# https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability
# Only the keys this repository uses, plus the ones most easily got wrong.
_JOB_LEVEL_CONTEXTS: dict[str, frozenset[str]] = {
    "env": frozenset({"github", "needs", "strategy", "matrix", "vars", "secrets", "inputs"}),
    "if": frozenset({"github", "needs", "vars", "inputs"}),
    "name": frozenset({"github", "needs", "strategy", "matrix", "vars", "inputs"}),
    "runs-on": frozenset({"github", "needs", "strategy", "matrix", "vars", "inputs"}),
    "timeout-minutes": frozenset({"github", "needs", "strategy", "matrix", "vars", "inputs"}),
    "environment": frozenset({"github", "needs", "strategy", "matrix", "vars", "inputs"}),
    "concurrency": frozenset({"github", "needs", "strategy", "matrix", "inputs", "vars"}),
    "strategy": frozenset({"github", "needs", "vars", "inputs"}),
    "services": frozenset(
        {"github", "needs", "strategy", "matrix", "vars", "inputs", "env", "secrets"}
    ),
    "container": frozenset(
        {"github", "needs", "strategy", "matrix", "vars", "inputs", "env", "secrets"}
    ),
}
_WORKFLOW_ENV_CONTEXTS = frozenset({"github", "secrets", "inputs", "vars"})


def _strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for value in node.values() for s in _strings(value)]
    if isinstance(node, list):
        return [s for value in node for s in _strings(value)]
    return []


def _contexts_used(node: Any, *, bare_expression: bool = False) -> set[str]:
    used: set[str] = set()
    for text in _strings(node):
        bodies = _EXPRESSION.findall(text)
        if bare_expression and not bodies:
            bodies = [text]  # `if:` may omit the ${{ }} wrapper
        for body in bodies:
            used.update(_CONTEXT.findall(body))
    return used


def unavailable_contexts(workflow: dict[str, Any]) -> list[str]:
    """Every use of a context in a place where GitHub does not provide it."""
    problems: list[str] = []
    bad = _contexts_used(workflow.get("env")) - _WORKFLOW_ENV_CONTEXTS
    if bad:
        problems.append(f"workflow env uses unavailable context(s): {sorted(bad)}")
    for job_id, job in (workflow.get("jobs") or {}).items():
        for key, allowed in _JOB_LEVEL_CONTEXTS.items():
            if key not in job:
                continue
            bad = _contexts_used(job[key], bare_expression=key == "if") - allowed
            if bad:
                problems.append(f"jobs.{job_id}.{key} uses unavailable context(s): {sorted(bad)}")
    return problems


def dangling_needs(workflow: dict[str, Any]) -> list[str]:
    jobs = workflow.get("jobs") or {}
    problems = []
    for job_id, job in jobs.items():
        needs = job.get("needs", [])
        for needed in [needs] if isinstance(needs, str) else needs:
            if needed not in jobs:
                problems.append(f"jobs.{job_id}.needs names a job that does not exist: {needed}")
    return problems


def _load(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} is not a mapping"
    return loaded


def test_there_are_workflows_to_check() -> None:
    names = {path.name for path in WORKFLOWS}
    assert {"test.yml", "workflow-lint.yml"} <= names


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_context_is_used_where_github_does_not_provide_it(path: Path) -> None:
    assert unavailable_contexts(_load(path)) == []


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_needs_names_a_real_job(path: Path) -> None:
    assert dangling_needs(_load(path)) == []


def test_the_guard_detects_the_defect_that_disabled_ci() -> None:
    """A checker that cannot fail is not a check: replay the exact line."""
    broken = yaml.safe_load(
        """
        jobs:
          test:
            runs-on: ubuntu-latest
            env:
              RUSTUP_HOME: ${{ runner.temp }}/rustup
            steps:
              - run: echo "${{ runner.temp }}"
        """
    )
    assert unavailable_contexts(broken) == ["jobs.test.env uses unavailable context(s): ['runner']"]
    # The same context inside a step is legal and must not be flagged.
    broken["jobs"]["test"]["env"] = {"RUSTUP_HOME": "rustup-home"}
    assert unavailable_contexts(broken) == []
    assert dangling_needs({"jobs": {"a": {"needs": ["b"]}}}) == [
        "jobs.a.needs names a job that does not exist: b"
    ]


def test_validity_is_checked_from_a_separate_workflow_file() -> None:
    """test.yml cannot report its own invalidity; a different file must."""
    lint = _load(ROOT / ".github" / "workflows" / "workflow-lint.yml")
    triggers = lint.get("on", lint.get(True))  # PyYAML reads a bare `on` as True
    assert "push" in triggers and "pull_request" in triggers
    script = "\n".join(
        step.get("run", "") for job in lint["jobs"].values() for step in job["steps"]
    )
    assert "actionlint" in script
    assert "sha256sum --check" in script


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not on PATH")
def test_actionlint_accepts_every_workflow() -> None:
    proc = subprocess.run(
        ["actionlint", "-shellcheck=", "-pyflakes=", "-oneline"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
