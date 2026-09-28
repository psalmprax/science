"""Fast smoke tier: syntax + import surface. Seconds, not minutes.

A committed IndentationError in ``apedp_dyn/grammar/candidate_ir.py`` once
blocked ``from apedp_dyn.grammar import TERM_FAMILIES`` outright, and nothing
noticed until something imported the package: the slow physics tests never
touch that module first, and CI ran only the ten-minute suite. The cheapest
possible check had no home.

Tier 1 (syntax) compiles every project source file. It needs no imports, so an
absent optional dependency cannot make it pass or fail by accident.
Tier 2 (import) imports every ``apedp_dyn`` module, catching import-cycle and
import-time NameError/AttributeError breakage.

Gated by the ``smoke`` marker so CI can run just this tier: ``pytest -m smoke``.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SYNTAX_PACKAGES = ("apedp", "apedp_dyn")
IMPORT_PACKAGE = "apedp_dyn"

pytestmark = pytest.mark.smoke


def _sources(package: str) -> list[Path]:
    base = ROOT / package
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _module_names(package: str) -> list[str]:
    """Dotted module names derived from files.

    Deliberately not ``pkgutil.walk_packages``: that swallows ImportError from
    a subpackage when no ``onerror`` is supplied, which is precisely the
    failure this tier exists to surface.
    """
    names: list[str] = []
    for path in _sources(package):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts:
            names.append(".".join(parts))
    return sorted(set(names))


@pytest.mark.parametrize("package", SYNTAX_PACKAGES)
def test_every_source_file_parses(package: str) -> None:
    paths = _sources(package)
    # A vacuous pass is worse than no gate: a wrong ROOT would silently
    # compile zero files and report success.
    assert len(paths) > 5, f"only {len(paths)} sources found under {package}"
    failures = []
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not failures, "source files that do not parse:\n" + "\n".join(failures)


def test_every_apedp_dyn_module_imports() -> None:
    names = _module_names(IMPORT_PACKAGE)
    assert len(names) > 10, f"only {len(names)} modules found; check looks vacuous"
    failures = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the failure IS the report
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules that fail to import:\n" + "\n".join(failures)
