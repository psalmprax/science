"""Structural invariance: rank by whether the SAME form wins, not by fit.

The ceiling on discovery in this system is not search coverage -- it is that
MSE ranking cannot identify a mechanism when candidates are observationally
close (measured: on AME2020 the correct surface term A^(2/3) and the
correlated proxy A*Z sit ~3e-4 apart, and the marginally-better proxy
outranked the truth).

`invariant_structure_verdict` implements the minimal honest fix: split the
development frame into K disjoint slices, run the search independently on
each, and ask whether the SAME canonical structure is the winner in at least
`min_slices` of them. Uniqueness is earned by agreement across independent
slices rather than by a thin margin in one pooled fit.

This is deliberately NOT "average the fits": a candidate that wins on 1/K
slices with a huge margin scores below a candidate that wins on every slice
with a small one. When no structure clears the slice threshold, the verdict
is MULTI by construction -- not because a band was fitted after the fact.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SliceResult:
    """One slice's winner, with the fields needed to audit the agreement."""

    canonical_key: str
    equation: str
    dev_r2: float
    n_rows: int


@dataclass(frozen=True)
class InvarianceVerdict:
    verdict: str                 # "UNIQUE" | "MULTI" | "NONE"
    winner_key: str | None
    slice_wins: int
    n_slices: int
    agreement: float
    slice_results: tuple[SliceResult, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "winner_key": self.winner_key,
            "slice_wins": self.slice_wins,
            "n_slices": self.n_slices,
            "agreement": self.agreement,
            "slice_results": [s.__dict__ for s in self.slice_results],
            "note": self.note,
        }


_IDENT = r"[A-Za-z_][A-Za-z_0-9]*"
_ARITH = ("*",)


def _strip_additive_zeros(eq: str) -> str:
    """Drop additive-zero decorations: ``(-0.000000[t-20])`` contributes nothing.

    Measured cause: RIME's ``PruneEdge`` parks a ``Const(0)`` that nothing
    reads, so one mechanism is spelled ``A_23`` or
    ``(A_23 + (-0.000000[t-10]))`` depending on which seed won. Without this
    the spellings fragment cross-slice agreement (measured 2/5 slices on the
    AME surface term, from syntax alone).
    """
    import re

    out = eq
    for _ in range(6):
        new = re.sub(r"\(\s*-0+(?:\.0+)?(?:\[t-\d+\])?\s*\)", "ZERO", out)
        new = new.replace("+ ZERO", "").replace("ZERO +", "")
        if new == out:
            break
        out = new
    return out


def canonical_key(proposal: Any) -> str:
    """Structural identity: which variables are combined, and how.

    Two properties, both measured:

      * the FITTED READOUT is excluded (scale/intercept) -- refitting the same
        structure on another slice must not change its key;
      * identity is a multiset of the decomposed variables (with their lag
        decorations) plus the count of arithmetical operators, so ``A_23`` and
        ``(A_23 + (-0.000000[t-10]))`` compare equal while
        ``(x1[t-1] * x2[t-1])`` and ``x1[t-1]`` stay distinct.

    ``metadata["equation"]`` is the kernel's graph rendering without the
    readout, which is exactly the object of an identity claim.
    """
    import hashlib
    import re

    eq = (getattr(proposal, "metadata", {}) or {}).get("equation") or getattr(proposal, "formula", "")
    eq = _strip_additive_zeros(str(eq))
    # Lag decoration travels with its variable: "v0[t-10]" is one token.
    eq = re.sub(r"\[t-(\d+)\]", r"[t-\1]", eq)
    idents = re.findall(_IDENT + r"(?:\[t-\d+\])?", eq)
    # Drop zero/number-ish tokens that are literals rather than variables.
    idents = [t for t in idents if not t.startswith("0")]
    ops = sum(eq.count(o) for o in _ARITH)
    sig = ("|".join(sorted(idents))) + "#" + str(ops)
    return "g:" + hashlib.sha256(sig.encode()).hexdigest()[:16]


def _probe_frame(frame: pd.DataFrame, target_col: str, rows: int = 97) -> pd.DataFrame:
    """Deterministic probe slice (order-independent) for auditing slice winners."""
    if len(frame) == 0:
        return frame
    order_col = [c for c in frame.columns if c != target_col]
    fr = frame.sort_values(order_col[0], kind="stable").reset_index(drop=True) if order_col else frame
    return fr.iloc[:rows].reset_index(drop=True)


def invariant_structure_verdict(
    frame: pd.DataFrame,
    search: Callable[[pd.DataFrame], Sequence[Any]],
    *,
    n_slices: int = 5,
    min_slices: int = 4,
    target_col: str = "target",
) -> InvarianceVerdict:
    """Run `search` on K disjoint slices of `frame` and score agreement.

    `search` must accept a DataFrame (slice) and return proposals ranked by
    dev performance. No slice is reused, so nothing can be tuned across them.
    """
    if n_slices < 2:
        raise ValueError("need at least 2 slices for an invariance claim")
    if min_slices > n_slices:
        raise ValueError("min_slices cannot exceed n_slices")

    shuffled = frame.iloc[np.random.RandomState(0).permutation(len(frame))].reset_index(drop=True)
    bounds = np.linspace(0, len(shuffled), n_slices + 1).astype(int)
    results: list[SliceResult] = []
    for k in range(n_slices):
        sl = shuffled.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True)
        props = list(search(sl)) if len(sl) else []
        if not props:
            results.append(SliceResult("", "", float("nan"), len(sl)))
            continue
        best = max(props, key=lambda p: p.parameters.get("dev_r2", float("-inf")))
        results.append(SliceResult(
            canonical_key(best),
            (best.metadata or {}).get("equation") or getattr(best, "formula", ""),
            float(best.parameters.get("dev_r2", float("nan"))),
            len(sl),
        ))

    wins: dict[str, int] = {}
    for r in results:
        if r.canonical_key:
            wins[r.canonical_key] = wins.get(r.canonical_key, 0) + 1

    if not wins:
        return InvarianceVerdict(
            "NONE", None, 0, n_slices, 0.0, tuple(results),
            "no slice produced a winner; nothing claimed")

    best_key, best_wins = max(wins.items(), key=lambda kv: kv[1])
    agreement = best_wins / n_slices
    if best_wins >= min_slices:
        verdict = "UNIQUE"
        note = (f"{best_key[:12]} won {best_wins}/{n_slices} independent slices "
                f"(threshold {min_slices})")
    else:
        verdict = "MULTI"
        note = (f"no structure won >= {min_slices}/{n_slices} slices "
                f"(best {best_wins}: {best_key[:12]}); ambiguity preserved")
    return InvarianceVerdict(verdict, best_key, best_wins, n_slices, agreement,
                             tuple(results), note)
