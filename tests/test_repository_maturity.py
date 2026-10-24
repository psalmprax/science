"""Contributor documentation and review ownership contracts."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", ["README.md", "CONTRIBUTING.md"])
def test_contributor_documentation_links_exist(relative: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    assert "file:///" not in text
    links = re.findall(r"\]\(([^)\s]+)\)", text)
    assert links
    for link in links:
        if link.startswith(("https://", "http://", "mailto:", "#")):
            continue
        assert (path.parent / link.split("#", 1)[0]).exists(), link


def test_codeowners_covers_contributor_boundaries() -> None:
    lines = (ROOT / ".github" / "CODEOWNERS").read_text().splitlines()
    entries = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert entries[0] == ["*", "@psalmprax"]
    for entry in entries:
        assert len(entry) == 2
        pattern, owner = entry
        assert owner == "@psalmprax"
        assert pattern == "*" or pattern.startswith("/")
        if pattern != "*":
            assert list(ROOT.glob(pattern.lstrip("/"))), pattern
    patterns = {entry[0] for entry in entries}
    assert {
        "/packages/", "/crates/", "/rust/", "/apedp*", "/docs/",
        "/specifications/", "/.github/workflows/", "/.github/CODEOWNERS",
    } <= patterns


def test_readme_quickstart_uses_existing_verification_tests() -> None:
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## Getting Started in 5 minutes\n", 1)[1].split("\n## ", 1)[0]
    assert 'python -m pip install -e ".[dev]"' in section
    command = next(line for line in section.splitlines() if line.startswith("python -m pytest "))
    for node in command.split()[3:]:
        if node.startswith("-"):
            continue
        path, _, test = node.partition("::")
        source = (ROOT / path).read_text()
        if test:
            assert f"def {test}(" in source


def test_readme_ci_matrix_tracks_workflows_and_make_targets() -> None:
    readme = (ROOT / "README.md").read_text()
    matrix = readme.split("## CI status matrix\n", 1)[1].split("\n## ", 1)[0]
    makefile = (ROOT / "Makefile").read_text()
    for target in re.findall(r"`make ([a-z-]+)`", matrix):
        assert re.search(rf"(?m)^{re.escape(target)}:", makefile), target
    for filename in ("test.yml", "wheels.yml"):
        path = ROOT / ".github" / "workflows" / filename
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506
        assert f"[{workflow['name']}](.github/workflows/{filename})" in matrix
        for job in (
            ("python-quality", "rust-quality", "test")
            if filename == "test.yml"
            else ("linux-wheels", "macos-wheels", "windows-wheels", "sdist", "python-packages")
        ):
            assert job in workflow["jobs"]
            assert f"`{job}`" in matrix
