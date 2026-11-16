"""The referee core depends on what its manifest says, and nothing else.

Before this boundary existed, ``import apedp_dyn.epistemic.promotion_lifecycle``
executed 74 repository modules (96 for the whole core), among them the native
kernel, SymPy and the discovery engines, because importing any submodule runs
its parent packages and ``apedp_dyn/__init__.py`` imported everything eagerly.
The thing that judges candidates could not be loaded without the things that
propose them. It is 6 modules now, and 13 for the whole core.

``tests/referee_core_manifest.py`` declares the core and its allowed
dependencies. This file enforces the declaration three ways:

1. *statically* -- every ``import`` statement in a core module, at any depth,
   is the standard library, an allowed third-party package, another core
   module, or a declared optional edge that is not at module level;
2. *at import* -- a fresh interpreter that imports the core has loaded exactly
   the declared repository modules;
3. *under test* -- a fresh pytest that runs the core's own tests has loaded
   nothing outside the declaration.

All three run in the full environment and in CI's ``referee-core`` job, where
only the core's requirements are installed; there, (2) and (3) fail by
construction if the core reaches for anything else.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.referee_core_manifest import (
    CORE_MODULES,
    CORE_REQUIREMENTS,
    CORE_TEST_SUPPORT,
    CORE_TESTS,
    FORBIDDEN_AT_RUNTIME,
    OPTIONAL_EDGES,
    SUPPORT_MODULES,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_PACKAGES = ("apedp", "apedp_dyn", "apedp_kernel", "apedp_vqc", "jetflow_video")
STDLIB = set(sys.stdlib_module_names)


def _path(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def _imports(module: str) -> list[tuple[str, bool, bool, int]]:
    """``(imported module, at module level, type-checking only, line)`` for every import."""
    tree = ast.parse(_path(module).read_text(encoding="utf-8"))
    type_only: set[int] = set()
    nested: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            type_only.update(id(child) for child in ast.walk(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            nested.update(id(child) for child in ast.walk(node) if child is not node)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = module.split(".")[:-1]
                package = package[: len(package) - (node.level - 1)]
                base = ".".join(package + ([node.module] if node.module else []))
            names = [base]
        else:
            continue
        for name in names:
            found.append((name, id(node) not in nested, id(node) in type_only, node.lineno))
    return found


def _run(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    # A fresh interpreter with the repository root first on the path and no
    # inherited PYTHONPATH: what this process has already imported is irrelevant.
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    child_env.update(env or {})
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _repository_modules(loaded: list[str]) -> set[str]:
    return {name for name in loaded if name.split(".")[0] in REPO_PACKAGES}


def _forbidden(loaded: list[str]) -> set[str]:
    tops = {name.split(".")[0] for name in loaded}
    return tops & set(FORBIDDEN_AT_RUNTIME)


# --------------------------------------------------------------------- the manifest


def test_the_manifest_names_real_modules_and_real_tests() -> None:
    for module in (*CORE_MODULES, "apedp_dyn._lazy"):
        assert _path(module).is_file(), module
    for test in CORE_TESTS:
        assert (ROOT / test).is_file(), test
    assert set(CORE_MODULES).isdisjoint(SUPPORT_MODULES)
    for (source, target), reason in OPTIONAL_EDGES.items():
        assert source in CORE_MODULES and target not in CORE_MODULES
        assert reason.strip()


# -------------------------------------------------------------------- 1. statically


@pytest.mark.parametrize("module", [*CORE_MODULES, "apedp_dyn._lazy"])
def test_every_import_in_a_core_module_is_declared(module: str) -> None:
    allowed_repo = {*CORE_MODULES, *SUPPORT_MODULES}
    problems = []
    for name, at_module_level, type_only, line in _imports(module):
        top = name.split(".")[0]
        if top in STDLIB or top == "__future__" or top in CORE_REQUIREMENTS:
            continue
        if name in allowed_repo:
            continue
        if (module, name) in OPTIONAL_EDGES:
            if at_module_level and not type_only:
                problems.append(
                    f"line {line}: optional edge to {name} is imported at module level; "
                    "it must stay inside the function that needs it"
                )
            continue
        problems.append(f"line {line}: imports {name}, which the manifest does not allow")
    assert not problems, f"{module}:\n  " + "\n  ".join(problems)


def test_every_declared_optional_edge_still_exists() -> None:
    """A stale exception is an open door nobody is watching."""
    for source, target in OPTIONAL_EDGES:
        assert any(name == target for name, *_ in _imports(source)), (
            f"{source} no longer imports {target}; remove the edge from the manifest"
        )


def test_the_lazy_helper_is_standard_library_only() -> None:
    for name, *_ in _imports("apedp_dyn._lazy"):
        assert name.split(".")[0] in STDLIB or name == "__future__", name


# --------------------------------------------------------------------- 2. at import


def test_importing_the_core_loads_exactly_the_declared_modules() -> None:
    code = (
        "import importlib, json, sys\n"
        f"for name in {list(CORE_MODULES)!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert _repository_modules(loaded) == {*CORE_MODULES, *SUPPORT_MODULES}
    assert not _forbidden(loaded)


def test_the_detector_sees_a_leak() -> None:
    """A check that cannot fail is not a check: importing the bridge beside the
    core must show up as undeclared repository modules and forbidden packages.
    Skipped where the leak cannot even be imported, which is the stronger form
    of the same statement."""
    code = (
        "import importlib, json, sys\n"
        "try:\n"
        "    importlib.import_module('apedp_dyn.epistemic.lean_universal_bridge')\n"
        "except ImportError as exc:\n"
        "    print(json.dumps({'unimportable': str(exc)}))\n"
        "else:\n"
        "    print(json.dumps(sorted(sys.modules)))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    if isinstance(loaded, dict):
        pytest.skip(f"the bridge is not importable here: {loaded['unimportable']}")
    assert _repository_modules(loaded) - {*CORE_MODULES, *SUPPORT_MODULES}
    assert _forbidden(loaded) >= {"apedp", "sympy"}


# -------------------------------------------------------------------- 3. under test


def test_running_the_core_tests_loads_nothing_outside_the_declaration(tmp_path: Path) -> None:
    record = tmp_path / "loaded_modules.txt"
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    child_env["APEDP_RECORD_LOADED_MODULES"] = str(record)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *CORE_TESTS,
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-p",
            "no:randomly",
        ],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    loaded = record.read_text(encoding="utf-8").split()
    undeclared = _repository_modules(loaded) - {*CORE_MODULES, *SUPPORT_MODULES}
    assert not undeclared, (
        f"the core's tests loaded undeclared repository modules: {sorted(undeclared)}"
    )
    assert not _forbidden(loaded), sorted(_forbidden(loaded))
    test_side = {name for name in loaded if name == "tests" or name.startswith("tests.")}
    assert test_side <= set(CORE_TEST_SUPPORT), sorted(test_side - set(CORE_TEST_SUPPORT))
    assert {"tests.test_referee_core_standalone", "tests.test_lean_model_conformance"} <= test_side
