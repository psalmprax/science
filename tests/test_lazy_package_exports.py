"""Lazy package ``__init__`` modules export exactly what the eager ones did.

``apedp_dyn``, ``apedp_dyn.epistemic`` and ``apedp_dyn.market`` re-export their
public names on first access (``apedp_dyn/_lazy.py``) instead of importing
everything when the package is first touched. That is what lets the referee
core load without the rest of the tree
(``tests/test_referee_core_boundary.py``); it must not change what any existing
``from apedp_dyn import X`` receives.

Three views of each package have to agree, and nothing but a test keeps them
in step: the runtime table (``_EXPORTS``), the public list (``__all__``), and
the ``if TYPE_CHECKING:`` imports that type checkers and readers see. A name in
one and not another is either an ``AttributeError`` at runtime that mypy
cannot see, or a public name mypy has never heard of.

This file imports the whole tree, so it belongs to the full environment and is
not part of the referee core's own tests.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAZY_PACKAGES = ("apedp_dyn", "apedp_dyn.epistemic", "apedp_dyn.market")
#: Public names each package exported when its ``__init__`` was eager
#: (``stage`` at 6e4d814). A count going down is an API removal and needs a
#: reason; going up is an addition and needs this number moved with it.
EXPORT_COUNTS = {"apedp_dyn": 68, "apedp_dyn.epistemic": 34, "apedp_dyn.market": 24}


def _init_path(package: str) -> Path:
    return ROOT / package.replace(".", "/") / "__init__.py"


def _type_checking_imports(package: str) -> dict[str, str]:
    """``public name -> "module" or "module:attribute"`` from the TYPE_CHECKING block."""
    tree = ast.parse(_init_path(package).read_text(encoding="utf-8"))
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
    )
    found: dict[str, str] = {}
    for node in ast.walk(block):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module, "use absolute imports in a lazy __init__"
            for alias in node.names:
                public = alias.asname or alias.name
                found[public] = (
                    node.module if public == alias.name else f"{node.module}:{alias.name}"
                )
    return found


def _submodules(module: object) -> tuple[str, ...]:
    return tuple(getattr(module, "_SUBMODULES", ()))


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_the_three_views_of_the_public_api_agree(package: str) -> None:
    module = importlib.import_module(package)
    exports = dict(module._EXPORTS)
    submodules = _submodules(module)
    assert set(exports).isdisjoint(submodules)
    assert set(module.__all__) == set(exports) | set(submodules)
    assert len(module.__all__) == len(set(module.__all__))
    # A subpackage exported as a name is written `from <package> import <name>`.
    assert _type_checking_imports(package) == {**exports, **dict.fromkeys(submodules, package)}
    assert len(module.__all__) == EXPORT_COUNTS[package]


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_every_export_is_the_object_its_module_defines(package: str) -> None:
    module = importlib.import_module(package)
    for name, target in module._EXPORTS.items():
        source, _, attribute = target.partition(":")
        defined = getattr(importlib.import_module(source), attribute or name)
        assert getattr(module, name) is defined, f"{package}.{name}"
        assert vars(module)[name] is defined, f"{package}.{name} was not cached on the package"
    for name in _submodules(module):
        assert getattr(module, name) is importlib.import_module(f"{package}.{name}")
    assert set(module.__all__) <= set(dir(module))


@pytest.mark.parametrize(
    "first_touch",
    [
        "import apedp_dyn.epistemic.adjudicate",  # the submodule, before the name is asked for
        "from apedp_dyn.epistemic.adjudicate import EpistemicStatus",
        "from apedp_dyn.epistemic import adjudicate",  # the name, before the submodule exists
        "import apedp_dyn.epistemic as e; e.adjudicate",
    ],
)
def test_an_export_named_like_its_submodule_is_the_export(first_touch: str) -> None:
    """``apedp_dyn.epistemic.adjudicate`` is a function and a submodule. The
    eager package always ended up holding the function; a naive lazy package
    holds the *module* as soon as anything imports the submodule, and
    ``from apedp_dyn.epistemic import adjudicate`` then returns something that
    cannot be called. Whatever touches it first, in a fresh interpreter, the
    package attribute must be the function."""
    code = (
        f"{first_touch}\n"
        "import sys, types\n"
        "import apedp_dyn.epistemic as package\n"
        "submodule = sys.modules['apedp_dyn.epistemic.adjudicate']\n"
        "assert isinstance(submodule, types.ModuleType)\n"
        "assert package.adjudicate is submodule.adjudicate, type(package.adjudicate)\n"
        "from apedp_dyn.epistemic import adjudicate as imported\n"
        "assert imported is submodule.adjudicate and callable(imported)\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-1500:]


def test_only_the_known_collision_exists() -> None:
    """A new export that shares a name with a submodule needs the same care;
    this makes adding one a visible decision."""
    collisions = {}
    for package in LAZY_PACKAGES:
        module = importlib.import_module(package)
        directory = _init_path(package).parent
        children = {p.stem for p in directory.glob("*.py")} | {
            p.name for p in directory.iterdir() if p.is_dir()
        }
        collisions[package] = sorted(set(module._EXPORTS) & children)
    assert collisions == {
        "apedp_dyn": [],
        "apedp_dyn.epistemic": ["adjudicate"],
        "apedp_dyn.market": [],
    }
    assert type(importlib.import_module("apedp_dyn.epistemic")).__name__ == "_LazyPackage"


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_importing_the_package_imports_none_of_its_exports(package: str) -> None:
    """The point of the change, per package: touching it costs nothing."""
    code = (
        "import importlib, json, sys\n"
        f"module = importlib.import_module({package!r})\n"
        "targets = {t.partition(':')[0] for t in module._EXPORTS.values()}\n"
        "print(json.dumps(sorted(targets & set(sys.modules))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert proc.stdout.strip().splitlines()[-1] == "[]"


def test_submodules_remain_attributes_of_their_package() -> None:
    """``import apedp_dyn; apedp_dyn.epistemic.x`` worked because the eager
    imports loaded every subpackage as a side effect. It still has to, for the
    lazy packages. One thing is not reproduced: a submodule of a package that
    is *not* lazy (``apedp_dyn.conf.canonical_spec`` -- ``conf`` has no
    ``__init__``) is an attribute only once something has imported it, as in
    any Python package. Eagerly importing everything used to make every such
    chain work by accident; nothing in the repository relies on one."""
    code = (
        "import apedp_dyn\n"
        "assert apedp_dyn.epistemic.__name__ == 'apedp_dyn.epistemic'\n"
        "assert apedp_dyn.epistemic.proof_claim.ProofClaimStatus.P2_VERIFIED.value == 'P2'\n"
        "assert apedp_dyn.market.universe.UniverseAudit\n"
        "assert apedp_dyn.ode.__name__ == 'apedp_dyn.ode'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-1500:]


def test_a_name_that_does_not_exist_is_still_an_attribute_error() -> None:
    import apedp_dyn

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_exported'"):
        _ = apedp_dyn.definitely_not_exported
    with pytest.raises(AttributeError):
        _ = apedp_dyn._private_and_absent
    with pytest.raises(ImportError):
        exec("from apedp_dyn import definitely_not_exported", {})  # noqa: S102 - the import statement is the thing under test


def test_star_import_and_patching_behave_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace: dict[str, object] = {}
    exec("from apedp_dyn.market import *", namespace)  # noqa: S102 - the import statement is the thing under test
    market = importlib.import_module("apedp_dyn.market")
    assert set(market.__all__) <= set(namespace)

    sentinel = object()
    monkeypatch.setattr("apedp_dyn.market.WINDOWS", sentinel)
    assert market.WINDOWS is sentinel  # a patch is not undone by the lazy lookup
    monkeypatch.undo()
    assert market.WINDOWS is importlib.import_module("apedp_dyn.market.windows").WINDOWS
