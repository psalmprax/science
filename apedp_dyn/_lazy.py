"""Deferred re-exports for package ``__init__`` modules (PEP 562).

``apedp_dyn/__init__.py`` used to import every public name eagerly. That made
``import apedp_dyn.epistemic.promotion_lifecycle`` -- a module whose own
dependencies are numpy and scipy -- execute 74 repository modules and load
the native kernel, SymPy and pint first, because importing any submodule runs
its parent packages. The referee could not be loaded, tested or audited apart
from everything it referees. With lazy packages the same import executes 6.

A package that uses :func:`lazy_exports` keeps the same public names and the
same ``__all__``; each name is imported from its defining module the first time
it is asked for, and then cached on the package so later access is an ordinary
attribute lookup. ``from package import name``, ``package.name``,
``from package import *`` and ``unittest.mock.patch("package.name")`` behave as
before. Submodules stay reachable as attributes (``apedp_dyn.epistemic``)
whether or not something else imported them first, which the eager imports
used to guarantee as a side effect.

This module must stay dependency-free: standard library only, and no import
from the rest of the repository. ``tests/test_referee_core_boundary.py``
enforces that.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["lazy_exports"]


class _LazyPackage(types.ModuleType):
    """A package one of whose exports has the same name as one of its submodules.

    ``apedp_dyn.epistemic`` exports the function ``adjudicate``, defined in the
    submodule ``apedp_dyn.epistemic.adjudicate``. With eager imports the
    package attribute ended up as the function: the import system first bound
    the submodule under that name, then ``from .adjudicate import adjudicate``
    rebound it. A lazy package has no second step, so whenever anything
    imported the submodule, ``package.adjudicate`` would silently become the
    module -- and ``from apedp_dyn.epistemic import adjudicate`` would hand
    callers something they cannot call.

    The import system binds a submodule with ``setattr(package, name, module)``
    after the submodule has executed. This intercepts exactly that assignment,
    for exactly those names, and stores the export instead, which is what the
    eager package always ended up holding.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        collisions = vars(self).get("_lazy_collisions")
        if (
            collisions
            and name in collisions
            and isinstance(value, types.ModuleType)
            and value.__name__ == f"{self.__name__}.{name}"
        ):
            # A submodule still executing (an import cycle) does not have the
            # attribute yet; leave the module in place, as the eager package
            # did at that same moment, and the next lookup resolves it.
            value = getattr(value, collisions[name], value)
        super().__setattr__(name, value)


def lazy_exports(
    package: str, exports: Mapping[str, str], submodules: Sequence[str] = ()
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build ``__getattr__`` and ``__dir__`` for the package named ``package``.

    ``exports`` maps each public name to the module that defines it, as
    ``"pkg.module"`` or, when the defining name differs, ``"pkg.module:name"``.
    ``submodules`` are subpackages the package lists in ``__all__`` in their
    own right (``from apedp_dyn import ode``).
    """
    module = sys.modules[package]
    collisions = {
        name: (target.partition(":")[2] or name)
        for name, target in exports.items()
        if target.partition(":")[0] == f"{package}.{name}"
    }
    if collisions:
        vars(module)["_lazy_collisions"] = collisions
        module.__class__ = _LazyPackage

    def __getattr__(name: str) -> Any:
        target = exports.get(name)
        if target is None:
            # Not a re-export. A submodule of that name is still an attribute
            # of its package; anything else is a genuine AttributeError.
            # Private and dunder names are never submodule lookups, and the
            # import machinery probes several of them while a package loads.
            if not name.startswith("_"):
                qualified = f"{package}.{name}"
                if qualified in sys.modules:
                    return sys.modules[qualified]
                if importlib.util.find_spec(qualified) is not None:
                    return importlib.import_module(qualified)
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module_name, _, attribute = target.partition(":")
        value = getattr(importlib.import_module(module_name), attribute or name)
        # Cache on the package: the next access never reaches __getattr__, and
        # a monkeypatch that replaces the attribute is not silently undone.
        vars(module)[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(vars(module)) | set(exports) | set(submodules))

    return __getattr__, __dir__
