"""The last checks, third version: the second version, and no law with a pole among the rows.

Status: experimental

**Not in force; development only.** Nothing calls it but its tests and
``scripts/replay_vouching_v3.py``. Putting it in force needs a blind test and
the owner's word (AGENTS.md 1.5).

**Why.** Above 5 % noise the second version let more wrong laws through than
the first (audit M-36). Read case by case, 66 of the 70 wrong laws it passed
there have a pole inside the range of the rows: a denominator that is positive
for some rows and negative for others. So does nearly every wrong law it
passed at lower noise in the second red-team campaign (83 of 98). Such a
law's values are unbounded among the rows. If its pole sits even slightly off
the true one, the error on rows near either pole is as large as one likes. A
finite sample therefore says little about its accuracy, at any noise. That is
audit M-34. The first version did better above 5 % noise only because it
refused nearly every law there (it kept 9 of 25 right laws, against 20).

**The rule.** After the second version has chosen a law, the law is withheld
if any denominator in it (the base of a negative power), or the argument of a
logarithm, takes both signs, or zero, on the rows it was fitted on and the
pool's inputs. The rows are used, not a box around them, so a pole in a
corner the data never reaches does not count. It only withholds. Law and
reason are those of the second version otherwise.

**What it cannot see.** A denominator that touches zero without changing sign,
such as ``(x - 2)**2``. A pole between two rows that lie on the same side of
it, which cannot happen for a sign change across the data's range. Poles of
functions other than a power or a logarithm (``tan``, ``gamma``).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import vouching_v2 as second
from apedp_dyn.discovery_l2.vouching_v2 import Vouched

POLE = ("its values are unbounded among the rows (a pole lies inside their range), so how far off its pole is "
        "cannot be bounded from them")


def singular_parts(expr: sp.Expr) -> list[sp.Expr]:
    """The denominators (bases of negative powers) and logarithm arguments in the expression, each once."""
    parts: list[sp.Expr] = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp.is_number and bool(node.exp < 0):
            parts.append(node.base)
        elif isinstance(node, sp.log):
            parts.append(node.args[0])
    return [p for p in dict.fromkeys(parts) if p.free_symbols]


def pole_among(text: str, names: Sequence[str], rows: np.ndarray) -> str | None:
    """The first singular part that takes both signs, or zero, on the rows; ``None`` if there is none."""
    symbols = {name: sp.Symbol(name) for name in names}
    expr = sp.sympify(text, locals=symbols)
    order = [symbols[name] for name in names]
    for part in singular_parts(expr):
        fn = sp.lambdify(order, part, "numpy")
        with np.errstate(all="ignore"):
            values = np.asarray(fn(*rows.T), dtype=float) * np.ones(len(rows))
        values = values[np.isfinite(values)]
        if len(values) and (float(values.min()) <= 0.0 <= float(values.max())):
            return str(part)
    return None


def after_second(done: Vouched, names: Sequence[str], X: np.ndarray, X_more: np.ndarray | None = None) -> Vouched:
    """The second version's outcome with the pole rule applied: unchanged, or withheld."""
    if done.law is None:
        return done
    rows = np.asarray(X, dtype=float)
    if X_more is not None and len(X_more):
        rows = np.vstack([rows, np.asarray(X_more, dtype=float)])
    try:
        part = pole_among(done.law, names, rows)
    except (sp.SympifyError, TypeError, ValueError, ArithmeticError, NameError, AttributeError) as exc:
        return Vouched(None, f"the pole rule could not read the law: {type(exc).__name__}", members=done.members,
                       record={**done.record, "pole_rule": "unreadable"})
    if part is None:
        return Vouched(done.law, done.reason, done.switched, done.members, done.power,
                       {**done.record, "pole_rule": "no pole among the rows"})
    return Vouched(None, f"{POLE} ({part} changes sign)", members=done.members, power=done.power,
                   record={**done.record, "pole_rule": part})


def vouch(law: str, others: Sequence[str], names: Sequence[str], X: np.ndarray, y: np.ndarray,
          X_more: np.ndarray | None = None, **options: Any) -> Vouched:
    """``vouching_v2.vouch``, then the pole rule. Arguments as the second version's."""
    return after_second(second.vouch(law, others, names, X, y, X_more, **options), names, X, X_more)
