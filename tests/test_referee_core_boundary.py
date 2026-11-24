"""The referee core depends on what its manifest says, and nothing else.

Before 2026-11-16, ``import apedp_dyn.epistemic.promotion_lifecycle`` executed
74 repository modules (96 for the whole core), among them the native kernel,
SymPy and the discovery engines. The thing that judges candidates could not be
loaded without the things that propose them. Since 2026-11-17 the core is its
own package, ``apedp_referee``, and importing all of it loads the package and
its nine modules; the old names are aliases that add the five lazy
``apedp_dyn`` package modules and nothing else.

``tests/referee_core_manifest.py`` declares the core and its allowed
dependencies. This file enforces the declaration four ways:

1. *statically* -- every ``import`` statement in a core module, at any depth,
   is the standard library, an allowed third-party package, another core
   module, or a declared optional edge that is not at module level;
2. *at import* -- a fresh interpreter that imports the core has loaded exactly
   the declared repository modules, and one that imports the old names has
   loaded those plus the declared alias support, with every old name the same
   object as its new one;
3. *under test* -- a fresh pytest that runs the core's own tests has loaded
   nothing outside the declaration;
4. *as a distribution* -- the package's ``pyproject.toml`` declares exactly
   the allowed requirements, ships the specification, and the specification it
   ships is byte-identical to the one the trustees sign.

CI's ``referee-core`` job runs this file with only the core's requirements
installed, and separately installs the package and runs its tests from
outside the checkout.
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
    ALIAS_SUPPORT,
    ALIASES,
    CORE_MODULES,
    CORE_REQUIREMENTS,
    CORE_TEST_SUPPORT,
    CORE_TESTS,
    FORBIDDEN_AT_RUNTIME,
    OPTIONAL_EDGES,
    SUPPORT_MODULES,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_PACKAGES = ("apedp", "apedp_dyn", "apedp_referee", "apedp_kernel", "apedp_vqc", "jetflow_video")
PACKAGE_DIR = ROOT / "apedp_referee"
STDLIB = set(sys.stdlib_module_names)


def _path(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def _is_repository_module(name: str) -> bool:
    relative = ROOT / name.replace(".", "/")
    return relative.with_suffix(".py").is_file() or relative.is_dir()


def _lazy_export_targets(package: str) -> dict[str, str]:
    """``name -> defining module`` from a lazy package's ``_EXPORTS`` table, read
    from its source so that looking does not import anything."""
    init = ROOT / package.replace(".", "/") / "__init__.py"
    if not init.is_file():
        return {}
    for node in ast.parse(init.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_EXPORTS"
        ):
            table = ast.literal_eval(node.value)
            return {name: target.partition(":")[0] for name, target in table.items()}
    return {}


def _imports(module: str, source: str | None = None) -> list[tuple[str, bool, bool, int]]:
    """``(imported module, at module level, type-checking only, line)`` for every import.

    ``from package import name`` is resolved to what it actually loads: the
    submodule ``package.name`` when there is one, or the module a lazy package
    re-exports ``name`` from. Recording only ``package`` would let
    ``from apedp_dyn.epistemic import agnostic_adapter`` through on the strength
    of ``apedp_dyn.epistemic`` being an allowed parent, and inside a function
    no test happens to call, neither runtime check would see it either.
    """
    if source is None:
        source = _path(module).read_text(encoding="utf-8")
    tree = ast.parse(source)
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
            exports = _lazy_export_targets(base)
            names = []
            for alias in node.names:
                child = f"{base}.{alias.name}"
                if alias.name in exports:
                    names.append(exports[alias.name])  # a lazy re-export: its defining module
                elif _is_repository_module(child):
                    names.append(child)  # a submodule imported as a name
                else:
                    names.append(base)  # an ordinary name defined in `base`
        else:
            continue
        for name in dict.fromkeys(names):
            found.append((name, id(node) not in nested, id(node) in type_only, node.lineno))
    return found


def undeclared_imports(module: str, source: str | None = None) -> list[str]:
    """Every import in ``module`` (or in ``source``, judged as if it were that
    module) that the manifest does not allow."""
    allowed_repo = {*CORE_MODULES, *SUPPORT_MODULES}
    problems = []
    for name, at_module_level, type_only, line in _imports(module, source):
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
    return problems


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
    for module in (*CORE_MODULES, *ALIASES, "apedp_dyn._lazy"):
        assert _path(module).is_file(), module
    for test in CORE_TESTS:
        assert (ROOT / test).is_file(), test
    assert set(CORE_MODULES).isdisjoint(SUPPORT_MODULES)
    assert set(ALIASES.values()) <= set(CORE_MODULES)
    for (source, target), reason in OPTIONAL_EDGES.items():
        assert source in CORE_MODULES and target not in CORE_MODULES
        assert reason.strip()


def test_every_module_in_the_package_is_declared() -> None:
    """A module dropped into apedp_referee/ is part of the referee, so it is
    either in the manifest or the manifest is wrong."""
    on_disk = {f"apedp_referee.{p.stem}" for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__"}
    assert on_disk == set(CORE_MODULES), on_disk ^ set(CORE_MODULES)


@pytest.mark.parametrize(("old", "new"), sorted(ALIASES.items()))
def test_each_old_name_is_only_an_alias(old: str, new: str) -> None:
    """The old file does one thing: hand back the package's module. Anything
    else in it would be code that runs under one name and not the other."""
    tree = ast.parse(_path(old).read_text(encoding="utf-8"))
    body = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    statements = [ast.unparse(node) for node in body]
    package, _, name = new.rpartition(".")
    assert statements == [
        "import sys",
        f"from {package} import {name} as _module",
        "sys.modules[__name__] = _module",
    ], statements


# -------------------------------------------------------------------- 1. statically


@pytest.mark.parametrize("module", CORE_MODULES)
def test_every_import_in_a_core_module_is_declared(module: str) -> None:
    problems = undeclared_imports(module)
    assert not problems, f"{module}:\n  " + "\n  ".join(problems)


_TRACKER = "apedp_referee.promotion_lifecycle"


@pytest.mark.parametrize(
    ("source", "leaked"),
    [
        # A submodule imported as a name from an allowed parent package.
        (
            "from apedp_dyn.epistemic import agnostic_adapter\n",
            "apedp_dyn.epistemic.agnostic_adapter",
        ),
        # The same, hidden in a function no test may ever call: only the
        # static check can see this one.
        (
            "def rarely_called():\n    from apedp_dyn.epistemic import agnostic_adapter\n",
            "apedp_dyn.epistemic.agnostic_adapter",
        ),
        # The old names are aliases into apedp_dyn: a core module that used
        # one would load apedp_dyn to reach a module of its own package.
        ("from apedp_dyn.conf import canonical_spec\n", "apedp_dyn.conf.canonical_spec"),
        (
            "from apedp_dyn.epistemic.experiment_ledger import DatasetRole\n",
            "apedp_dyn.epistemic.experiment_ledger",
        ),
        # A lazy re-export: the name comes from an allowed package, the module
        # that defines it is not allowed.
        ("from apedp_dyn.epistemic import adjudicate\n", "apedp_dyn.epistemic.adjudicate"),
        ("from apedp_dyn import fit_candidate\n", "apedp_dyn.fitting"),
        ("from apedp_dyn.market import WINDOWS\n", "apedp_dyn.market.windows"),
        # (Every module named in these strings is one that real tests already
        # exercise. benchmarks/untested_modules_register.json counts a module
        # as tested when a file under tests/ names its dotted path, so naming
        # an untested module here would take it off that register for nothing.)
        # The plain forms.
        (
            "import apedp_dyn.epistemic.lean_universal_bridge\n",
            "apedp_dyn.epistemic.lean_universal_bridge",
        ),
        ("from apedp_dyn.gates.dimensional import check\n", "apedp_dyn.gates.dimensional"),
        ("import sympy\n", "sympy"),
        ("def f():\n    import torch\n", "torch"),
    ],
)
def test_the_static_check_sees_an_undeclared_import_however_it_is_written(
    source: str, leaked: str
) -> None:
    """A check that cannot fail is not a check. The first three forms got
    through an earlier version of this file, which recorded only the package a
    ``from`` import names and not what it loads."""
    problems = undeclared_imports(_TRACKER, source)
    assert len(problems) == 1 and leaked in problems[0], problems


@pytest.mark.parametrize(
    "source",
    [
        "from apedp_referee import canonical_spec\n",
        "from apedp_referee import proof_claim, proof_evidence\n",
        "from apedp_referee.experiment_ledger import DatasetRole\n",
        "from . import trial_ledger\n",
        "import numpy as np\nfrom scipy import stats\nimport pandas\nimport json\n",
    ],
)
def test_the_static_check_accepts_declared_imports(source: str) -> None:
    assert undeclared_imports(_TRACKER, source) == []


def test_an_optional_edge_at_module_level_is_refused() -> None:
    ledger = "apedp_referee._host"
    inside = "def default_path():\n    from apedp_dyn.env import get_ledger_path\n"
    assert undeclared_imports(ledger, inside) == []
    problems = undeclared_imports(ledger, "from apedp_dyn.env import get_ledger_path\n")
    assert len(problems) == 1 and "module level" in problems[0]
    # The edge belongs to the module it was declared for, not to the core at large.
    assert undeclared_imports(_TRACKER, inside)


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


def test_the_old_names_are_the_same_modules_and_cost_only_their_parents() -> None:
    code = (
        "import importlib, json, sys\n"
        f"aliases = {dict(ALIASES)!r}\n"
        "same = {old: importlib.import_module(old) is importlib.import_module(new) for old, new in aliases.items()}\n"
        "parent = {old: getattr(importlib.import_module(old.rpartition('.')[0]), old.rpartition('.')[2]) is sys.modules[new]\n"
        "          for old, new in aliases.items()}\n"
        "print(json.dumps({'same': same, 'parent': parent, 'loaded': sorted(sys.modules)}))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert all(out["same"].values()), out["same"]
    assert all(out["parent"].values()), out["parent"]
    expected = {*CORE_MODULES, *SUPPORT_MODULES, *ALIASES, *ALIAS_SUPPORT} - {"apedp_referee._host"}
    loaded = _repository_modules(out["loaded"])
    assert loaded - {"apedp_referee._host"} == expected, loaded ^ expected
    assert not _forbidden(out["loaded"]) - {"apedp_dyn"}


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
    assert {"test_apedp_referee", "referee_evidence", "tests.test_lean_model_conformance"} <= set(loaded)


# ------------------------------------------------------------- 4. as a distribution


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _requirement_name(requirement: str) -> str:
    import re

    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower()


def test_the_package_declares_exactly_the_allowed_requirements() -> None:
    project = _toml(PACKAGE_DIR / "pyproject.toml")
    declared = {_requirement_name(r) for r in project["project"]["dependencies"]}
    assert declared == set(CORE_REQUIREMENTS)
    setuptools = project["tool"]["setuptools"]
    assert setuptools["packages"] == ["apedp_referee"]
    assert "canonical_preregistered_spec_v1.json" in setuptools["package-data"]["apedp_referee"]


def test_the_full_distribution_declares_the_core_requirements_too() -> None:
    """The full distribution carries apedp_referee, so it needs what it needs.
    pandas was missing from it until the split made the question visible."""
    project = _toml(ROOT / "pyproject.toml")
    declared = {_requirement_name(r) for r in project["project"]["dependencies"]}
    assert set(CORE_REQUIREMENTS) <= declared
    assert "canonical_preregistered_spec_v1.json" in project["tool"]["setuptools"]["package-data"]["apedp_referee"]


def test_the_shipped_specification_is_the_signed_one() -> None:
    """Two copies of one file, each pinned: the package's is checked at every
    import, the repository's is the one the trustees sign. A change to either
    without the other, or to both without the pin, fails here."""
    import hashlib

    from apedp_referee.canonical_spec import PINNED_SPEC_SHA256, SPEC_PATH

    shipped = SPEC_PATH.read_bytes()
    signed = (ROOT / "specifications" / "canonical_preregistered_spec_v1.json").read_bytes()
    assert SPEC_PATH.resolve().parent == PACKAGE_DIR.resolve()
    assert shipped == signed, "copy specifications/canonical_preregistered_spec_v1.json into apedp_referee/"
    assert hashlib.sha256(signed).hexdigest() == PINNED_SPEC_SHA256
