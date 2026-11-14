"""Native-kernel availability shim for module-level ``import apedp_kernel`` users.

The native PyO3/Rust extension ships as its own ``apedp-kernel`` wheel
(built by CI from ``rust/apedp-kernel``; cross-platform wheels are attached as
artifacts, with a tag-gated PyPI publish job). It is deliberately **not** part
of the pure-Python ``apedp`` wheel, so installations may legitimately run
without it while the pure-Python/NumPy engines cover most surfaces.

A few engines (differential-Padé ODE discovery, coupled-system discovery,
PDE/discovery variants, RIME AD proposal) are kernel-only: they have no
pure-Python implementation. Historically those modules imported
``apedp_kernel`` unconditionally, so ``import apedp_dyn`` exploded with a bare
``ModuleNotFoundError`` on any machine without the extension — contradicting
the documented pure-Python fallback story (and masked in development because
the kernel is always installed there).

This module makes the failure mode honest and import-safe:

    try:
        import apedp_kernel as ak
    except ImportError:  # kernel genuinely absent
        from apedp_dyn.kernel_shim import missing_kernel as ak

``import apedp_dyn`` then always succeeds, and any *call* into a kernel-only
engine raises an actionable ``RuntimeError`` naming the missing install step
instead of a bare ``AttributeError`` on ``None``.
"""
from __future__ import annotations


class NativeKernelUnavailableError(RuntimeError):
    """An engine that exists only in the native kernel was asked to run without it.

    This is a different fact from "the engine ran and found nothing", and code
    that needs the kernel raises it instead of returning an empty result: an
    empty result is how a search reports a null world, and a missing build must
    not be able to pass for one. It is a ``RuntimeError`` so existing handlers
    of the shim's error keep working.
    """


_INSTALL_HINT = (
    "The native apedp-kernel extension is not installed. This engine has no "
    "pure-Python implementation. Install a prebuilt wheel from the CI "
    "'Cross-Platform Native Wheels' workflow artifacts, or build it from "
    "source with `maturin develop --release` inside rust/apedp-kernel."
)


class _MissingKernel:
    """Sentinel replacing ``apedp_kernel`` when the native extension is absent.

    Attribute access for real entry points raises an actionable RuntimeError;
    dunder lookups raise AttributeError so probing utilities such as
    ``hasattr(ak, "__file__")`` and ``repr()`` keep working as expected.
    """

    def __getattr__(self, name: str) -> object:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise NativeKernelUnavailableError(f"apedp_kernel.{name} is unavailable: {_INSTALL_HINT}")


missing_kernel = _MissingKernel()
