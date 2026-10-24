"""Every re-export façade in the tree must actually re-export something.

Why this exists rather than a coverage number
---------------------------------------------
``apedp/dyn/__init__.py`` is a backwards-compatibility shim: commit 4cb17d2
moved the implementation into ``apedp_dyn`` and left this file re-exporting it
so old ``apedp.dyn`` imports keep working. In the same commit every in-tree
caller moved to the canonical package, so nothing imports the shim -- which is
correct, not a defect -- and it scores 0 of 6 statements in every coverage run
forever.

That permanent 0 was the last remaining finding of an otherwise clean coverage
audit, and it was worthless: the shim contains no logic, so "how much of it ran"
cannot distinguish a working façade from a broken one. tools/coverage_audit.py
now classifies these as REEXPORT_FACADE for exactly that reason.

The check that *can* fail is the one this file makes. A re-export shim has
exactly one job -- make ``__all__`` resolve -- and it fails silently and
completely when the target package renames or drops a name. ``apedp/dyn``
re-exports 75 names from ``apedp_dyn``; if ``apedp_dyn`` drops one, every
legacy ``from apedp.dyn import X`` in downstream code breaks at import time with
an ImportError that names ``apedp_dyn``, not the shim that failed to track it.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tools.coverage_audit import (
    _iter_python_files,
    is_reexport_facade,
    is_test_file,
    module_name_for,
)

ROOT = Path(__file__).resolve().parent.parent


def _facade_modules() -> list[str]:
    """Every re-export façade in the tree, discovered by the same rule the audit uses."""
    out = []
    for path in _iter_python_files(ROOT):
        rel = str(path.relative_to(ROOT))
        if is_test_file(rel) or not is_reexport_facade(rel, ROOT):
            continue
        out.append(module_name_for(rel))
    return sorted(out)


FACADES = _facade_modules()


def test_the_facade_detector_finds_the_known_facades():
    """A detector that finds nothing is a detector that passes.

    Same failure mode as the tools these tests guard: an empty result set here
    would make every test below vacuously true, because ``@pytest.mark.parametrize``
    over an empty list collects zero tests and reports success. So the set is
    pinned first, and it includes apedp.dyn by name.
    """
    assert "apedp.dyn" in FACADES, (
        f"apedp/dyn/__init__.py is a pure re-export shim and must be detected; "
        f"found {FACADES}")
    assert "apedp_dyn" in FACADES
    assert len(FACADES) >= 10, (
        f"only {len(FACADES)} façades detected ({FACADES}); the detector has "
        "probably regressed to matching almost nothing")


@pytest.mark.parametrize("module_name", FACADES)
def test_facade_imports_and_every_exported_name_resolves(module_name: str):
    """The whole contract: importable, and ``__all__`` is not a lie.

    Two failure modes, both silent from the outside:
      * the shim does not import at all (target package reorganised);
      * it imports, but a name in ``__all__`` is absent -- which makes
        ``from facade import name`` raise, and makes ``import *`` quietly skip
        it. The second is the dangerous one, because the façade still imports
        fine and still looks healthy to any check that only tries importing it.
    """
    module = importlib.import_module(module_name)

    declared = getattr(module, "__all__", None)
    assert declared is not None, (
        f"{module_name} is classified as a re-export façade so it must declare "
        "__all__; without it `from x import *` exports whatever happens to be "
        "in the namespace, which is not a contract")
    assert declared, f"{module_name} declares an empty __all__"

    missing = [name for name in declared if not hasattr(module, name)]
    assert not missing, (
        f"{module_name}.__all__ promises {missing} but the module does not "
        "define them -- a legacy `from apedp.dyn import <name>` would raise "
        "ImportError, and the traceback would blame the target package "
        "rather than the shim that stopped tracking it")


@pytest.mark.parametrize("module_name", FACADES)
def test_facade_is_not_merely_empty(module_name: str):
    """A façade that exports nothing is an empty package wearing a shim's name."""
    module = importlib.import_module(module_name)
    assert len(module.__all__) > 0, f"{module_name} re-exports nothing"


def test_the_detector_rejects_a_module_that_contains_logic():
    """The classification must not be able to hide real code.

    The failure this guards against is the tool quietly classifying a module
    with actual behaviour as inert, which would remove it from the audit
    entirely. So the negatives are pinned here, on the real tree rather than on
    a fixture -- a detector tested only on fixtures has been tested against the
    fixtures' shapes, not against this repository.
    """
    for rel in ("apedp/model_synthesis.py", "apedp/closed_loop.py",
                "apedp/decision/policy.py", "tools/coverage_audit.py"):
        assert not is_reexport_facade(rel, ROOT), (
            f"{rel} contains real logic and must never be classified as a "
            "re-export façade")


def test_the_detector_rejects_a_module_with_bare_imports_and_no_dunder_all():
    """No declared exports means the module may still run setup at import time.

    Treating it as inert on the strength of its imports alone is the mistake
    this rule avoids: ``__all__`` is the declaration that the module's body is
    nothing but a namespace.
    """
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pkg").mkdir()
        (root / "pkg" / "__init__.py").write_text("")
        (root / "pkg" / "bare.py").write_text("import os\nimport sys\n")
        (root / "pkg" / "with_all.py").write_text(
            textwrap.dedent("""
                import os

                __all__ = ["os"]
            """))
        (root / "pkg" / "has_function.py").write_text(
            textwrap.dedent("""
                import os

                __all__ = ["os"]


                def helper():
                    return os.sep
            """))
        (root / "pkg" / "has_conditional.py").write_text(
            textwrap.dedent("""
                import os

                __all__ = ["os"]

                if os.name == "nt":
                    SEP = "\\\\"
                else:
                    SEP = "/"
            """))

        assert not is_reexport_facade("pkg/bare.py", root), (
            "bare imports with no __all__ are ambiguous, not inert")
        assert is_reexport_facade("pkg/with_all.py", root)
        assert not is_reexport_facade("pkg/has_function.py", root), (
            "a module defining a function is not a re-export façade")
        assert not is_reexport_facade("pkg/has_conditional.py", root), (
            "a module with a conditional at import time does work of its own")