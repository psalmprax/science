"""Every module that tracked code imports must itself be tracked.

A module can import cleanly on the machine that wrote it while a fresh checkout fails,
because the imported file was never added to the index.  That happened here: a
committed module imported ``apedp_dyn.quantum.gf2_rank_search``, which was untracked, so
the published branch could not import the module at all.  This test closes that class of
gap by comparing the imports of every tracked file against the set of tracked files.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"apedp", "apedp_dyn"}


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("git is unavailable; the import closure is enforced in CI")
    return result.stdout


def _tracked() -> set[str]:
    return {line for line in _git("ls-files", "*.py").splitlines() if line}


def _module_name(relative: str) -> str:
    parts = relative[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str) -> list[Path]:
    if not module or module.split(".")[0] not in PACKAGES:
        return []
    base = ROOT.joinpath(*module.split("."))
    return [path for path in (base.with_suffix(".py"), base / "__init__.py") if path.exists()]


def _imports(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > len(parts):
                    continue
                base = ".".join(parts[: len(parts) - node.level + 1])
                names.add(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                names.add(node.module)
    return names


def test_tracked_modules_only_import_tracked_modules():
    tracked = _tracked()
    missing: dict[str, list[str]] = {}
    for relative in sorted(tracked):
        path = ROOT / relative
        if not path.is_file():
            continue
        for imported in _imports(path, _module_name(relative)):
            for candidate in _resolve(imported):
                target = candidate.relative_to(ROOT).as_posix()
                if target not in tracked:
                    missing.setdefault(relative, []).append(target)

    assert not missing, f"tracked code imports untracked modules: {missing}"
