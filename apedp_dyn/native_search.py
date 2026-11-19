"""Python-search fallback contract, owned by ``apedp_dyn``.

The native GA lives in ``apedp_kernel``; its Python reference implementation lives in
``apedp.evolutionary_search``, which is part of the legacy tree and, by ADR-0001, not
something the canonical package imports. So the legacy tree *registers* its provider
here instead: the bridge asks for the API through this module and never names the
legacy package. If nothing registered a provider, the bridge refuses with
``NativeSearchUnavailableError`` rather than silently substituting a search the caller
did not ask to measure.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class NativeSearchUnavailableError(RuntimeError):
    """Raised when ``strict_native_ga`` refuses the silently slower Python path.

    A native search and its Python fallback are not interchangeable: the
    fallback carries per-candidate constant tuning the kernel does not, and the
    measured cost difference on a 2,000-row problem is orders of magnitude
    (0.08 s native vs >180 s Python for the same 1,629 evaluations). Callers
    that requested the kernel get an exception rather than a result they did
    not measure for.
    """


_PythonSearchProvider = Callable[[], tuple[Any, Callable[..., Any]]]

_provider: _PythonSearchProvider | None = None


def register_python_search_provider(provider: _PythonSearchProvider) -> None:
    """Register the legacy Python GA entry points (``SearchConfig``, ``evolve``)."""
    global _provider
    _provider = provider


def has_python_search_provider() -> bool:
    return _provider is not None


def python_search_api() -> tuple[Any, Callable[..., Any]]:
    """Return ``(SearchConfig, evolve)`` or refuse naming the missing registration."""
    if _provider is None:
        raise NativeSearchUnavailableError(
            "the Python GA reference implementation has not been registered; import "
            "apedp.evolutionary_search (or anything that imports apedp.rust_bridge) "
            "before requesting the Python search fallback. Native search is unaffected."
        )
    return _provider()


__all__ = [
    "NativeSearchUnavailableError",
    "has_python_search_provider",
    "python_search_api",
    "register_python_search_provider",
]
