"""The fixed judge of the recursion: what counts as a solved problem, and the seal that shows it did not change.

Status: experimental

**Solved.** A problem is solved when the route *reported* a law (the referee
confirmed it on measured rows and the last checks in force vouched for it) and
that law is accurate on clean rows the route never saw: R² at or above
``DISCOVERY_ACCURACY_R2``, the line that defines a wrong law
(``apedp_dyn.conf.discovery_checks``). A reported law below that line is
**wrong** and counted against the searcher. A tentative law is neither.
``same_form`` adds, for information only, whether rewrites show the reported
law equal to the true one with its constants rounded.

**The seal.** ``seal()`` takes the SHA-256 of every file that decides the
above: the referee package, both threshold specifications and their loaders,
the route, the last checks, the measured range, the learning gate, this file,
the curriculum (which fixes the measured distribution), the progress test and
the benchmark's fingerprints. ``check(seal)`` raises ``JudgeChanged`` if any
differs. The levels call it at the start of a run and at every generation, so
a run in which the judge moved cannot report a gain.

**Settings.** A searcher's settings are checked against ``SETTINGS``: only the
names listed there, inside their bounds. No setting names a threshold, a test
level or anything the referee reads, so a search over settings cannot reach them.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.conf.discovery_checks import DISCOVERY_ACCURACY_R2
from apedp_dyn.recursion import egraph

ROOT = Path(__file__).resolve().parents[2]

SEALED_FILES = (
    "specifications/canonical_preregistered_spec_v1.json",
    "specifications/discovery_checks_v1.json",
    "apedp_dyn/conf/discovery_checks.py",
    "apedp_dyn/discovery_l2/refined_search.py",
    "apedp_dyn/discovery_l2/default_route.py",
    "apedp_dyn/discovery_l2/vouching_v2.py",
    "apedp_dyn/discovery_l2/vouched_route_v2.py",
    "apedp_dyn/discovery_l2/measured_range.py",
    "apedp_dyn/discovery_l2/learning_loop.py",
    "apedp_dyn/recursion/judge.py",
    "apedp_dyn/recursion/curriculum.py",
    "apedp_dyn/recursion/progress.py",
    "benchmarks/invented_laws_fingerprints.json",
)
SEALED_TREES = ("apedp_referee",)

#: the only settings a searcher has: name -> (lowest, highest, whole number)
SETTINGS: dict[str, tuple[float, float, bool]] = {
    "memory_seconds": (0.0, 120.0, False),
    "library_seconds": (0.0, 120.0, False),
    "library_blocks": (0, 48, True),
    "library_top": (2, 10, True),
    "library_proposals": (1, 6, True),
}


class JudgeChanged(RuntimeError):
    """A sealed file differs from what it was when the run started."""


class SettingRefused(ValueError):
    """A setting that is not a searcher's to set."""


def sealed_paths() -> list[Path]:
    paths = [ROOT / name for name in SEALED_FILES]
    for tree in SEALED_TREES:
        paths.extend(p for p in sorted((ROOT / tree).rglob("*")) if p.is_file() and p.suffix in (".py", ".json")
                     and "__pycache__" not in p.parts and "tests" not in p.parts)
    return paths


def seal() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sealed_paths()}


def check(sealed: Mapping[str, str]) -> None:
    now = seal()
    changed = sorted(name for name in set(now) | set(sealed) if now.get(name) != sealed.get(name))
    if changed:
        raise JudgeChanged(f"sealed files changed during the run: {changed}")


def check_settings(settings: Mapping[str, Any]) -> dict[str, float | int]:
    """The settings, each inside its bounds; anything else is refused."""
    out: dict[str, float | int] = {}
    for name, value in settings.items():
        if name not in SETTINGS:
            raise SettingRefused(f"{name!r} is not a searcher's setting; the settings are {sorted(SETTINGS)}")
        low, high, whole = SETTINGS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise SettingRefused(f"{name} = {value!r} is outside [{low}, {high}]")
        out[name] = int(value) if whole else float(value)
    return out


def r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    if not np.all(np.isfinite(pred)):
        return None
    total = float(np.sum((y - np.mean(y)) ** 2))
    return None if total <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / total


def same_form(law: sp.Expr, truth: sp.Expr, digits: int = 3) -> bool | None:
    """Whether rewrites show the law equal to the true one, both with constants rounded. ``None``: not shown."""
    def rounded(expr: sp.Expr) -> sp.Expr:
        return expr.xreplace({f: sp.Float(float(f"{float(f):.{digits}g}")) for f in expr.atoms(sp.Float)})
    shown = egraph.equal(rounded(sp.expand(law)), rounded(sp.expand(truth)), iterations=4, node_limit=1500)["equal"]
    return True if shown else None


def verdict(record: Mapping[str, Any], names: list[str], truth: sp.Expr, test_X: np.ndarray, test_clean: np.ndarray
            ) -> dict[str, Any]:
    """Solved, wrong or neither, for one route record."""
    law = record.get("law") if record.get("reported") else None
    if not law:
        return {"reported": False, "solved": False, "wrong": False, "r2": None, "same_form": None, "law": None}
    symbols = [sp.Symbol(n) for n in names]
    expr = sp.sympify(str(law), locals=dict(zip(names, symbols, strict=True)))
    with np.errstate(all="ignore"):
        pred = np.asarray(sp.lambdify(symbols, expr, "numpy")(*test_X.T), dtype=float) * np.ones(len(test_X))
    score = r2(test_clean, pred)
    accurate = bool(score is not None and score >= DISCOVERY_ACCURACY_R2)
    try:
        form = same_form(expr, truth)
    except (ValueError, TypeError, RecursionError):
        form = None
    return {"reported": True, "solved": accurate, "wrong": not accurate, "r2": None if score is None else round(score, 6),
            "same_form": form, "law": str(law)}
