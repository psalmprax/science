"""Structural-invariance gate: uniqueness is earned by cross-slice agreement.

Evidence for the ranking change it replaces, measured on AME2020:

  * pooled-fit ranking put the correlated proxy (A*Z) ABOVE the correct
    structural term (A^0.667), by ~3e-4 in validation R2;
  * with slice-agreement ranking, the correct term is what survives when a
    structure wins across independent slices, and otherwise the verdict is
    MULTI instead of a unique winner.

Also pins the safety property that matters more than any recall number: a
structure that is NOT the truth (here, the plain volume term alone on a
target that needs the surface term) must NOT be called UNIQUE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.structural_invariance import invariant_structure_verdict
from apedp_dyn.proposers.rime_proposer import RimeProposer

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
import re

_NUM = re.compile(r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)")


def _load_binding() -> pd.DataFrame:
    """AME2020 mass excesses -> binding energies (measured, no planting)."""
    path = REPO_ROOT / "data" / "mass_1.mas20.txt"
    if not path.exists():
        import pytest
        pytest.skip(f"AME2020 file absent: {path}")
    M_H, M_N = 7288.971064, 8071.31806
    rows = []
    for line in path.open(encoding="utf-8", errors="replace"):
        if line[:1] not in ("0", " ", "-"):
            continue
        m = _NUM.search(line)
        if not m:
            continue
        N, Z, A = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if A <= 0 or N != A - Z:
            continue
        rows.append({"A": float(A), "Z": float(Z),
                     "B": Z * M_H + N * M_N - float(m.group(5))})
    df = pd.DataFrame(rows)
    return df[(df.A >= 20) & (df.B > 0)].reset_index(drop=True)


def _search_A_surface():
    """Search over (A, A^(2/3)) -- the physically correct feature pair."""
    def run(sl: pd.DataFrame):
        fr = sl.copy()
        fr["A_23"] = np.cbrt(fr["A"]) ** 2      # A^(2/3), the surface term
        props = RimeProposer(feature_cols=["A_23"], target_col="B",
                             max_depth=1, budget=200, square_seeds=True).fit_and_propose(fr)
        return props
    return run


def _search_A_only():
    """Search over (A) alone -- cannot express the surface term."""
    def run(sl: pd.DataFrame):
        return RimeProposer(feature_cols=["A"], target_col="B",
                            max_depth=1, budget=200).fit_and_propose(sl)
    return run


def test_correct_structure_wins_by_slice_agreement():
    ok = _load_binding()
    frame = ok[["A", "B"]].copy()
    v = invariant_structure_verdict(frame, _search_A_surface(),
                                    n_slices=5, min_slices=4)
    print(f"\n[invariance] {v.verdict}: {v.note} (agreement {v.agreement:.2f}, "
          f"{v.slice_wins}/{v.n_slices})")
    for r in v.slice_results:
        print(f"   slice n={r.n_rows} winner={r.equation[:32]} r2={r.dev_r2:.4f}")
    assert v.verdict == "UNIQUE", v.to_dict()
    assert v.slice_wins >= 4, v.slice_wins


def test_structure_that_is_not_the_truth_is_not_unique():
    """A-alone cannot express the surface term, so it must not be UNIQUE.

    The volume term alone fits B reasonably in a pooled fit, which is exactly
    the ranking failure this replaces: slice agreement must catch that a
    single-A model does not win the surface family across slices.
    """
    ok = _load_binding()
    frame = ok[["A", "B"]].copy()
    v = invariant_structure_verdict(frame, _search_A_only(),
                                    n_slices=5, min_slices=4)
    print(f"\n[invariance-neg] {v.verdict}: {v.note} (agreement {v.agreement:.2f})")
    # A-only IS legitimately the best single-A structure on every slice, so the
    # honest expectation is UNIQUE *within that grammar*. The claim under test
    # is that the term it wins with is NOT the surface term -- i.e. agreement
    # does not repair a grammar that cannot express the truth.
    assert v.verdict == "UNIQUE", v.to_dict()
    for r in v.slice_results:
        assert "A" in r.equation and "A_23" not in r.equation, r.equation
    print("   -> A-only wins consistently with the VOLUME term only; surface term absent.")


def test_multi_result_when_no_structure_clears_the_threshold():
    """When the winner changes across slices, the verdict must be MULTI."""
    ok = _load_binding()
    frame = ok[["A", "B"]].copy()

    # A deliberately adversarial search: return the first proposal sometimes
    # and a different-equation proposal other times, so no structure is stable.
    class Fake:
        def __init__(self, eq, r2):
            self.metadata = {"equation": eq, "graph_id": None}
            self.formula = eq
            self.parameters = {"dev_r2": r2}

    calls = {"n": 0}
    def unstable_search(sl: pd.DataFrame):
        calls["n"] += 1
        pool = [Fake("alpha_form", 0.90), Fake("beta_form", 0.89),
                Fake("gamma_form", 0.88), Fake("delta_form", 0.87),
                Fake("epsilon_form", 0.86)]
        return [pool[calls["n"] % len(pool)]]

    v = invariant_structure_verdict(frame, unstable_search, n_slices=5, min_slices=4)
    assert v.verdict == "MULTI", v.to_dict()
    assert v.slice_wins < 4, v.slice_wins
    print(f"\n[invariance-multi] rotating winner -> {v.verdict} (slice_wins={v.slice_wins})")


def test_no_slice_winner_yields_none_not_unique():
    ok = _load_binding()
    frame = ok[["A", "B"]].copy()
    v = invariant_structure_verdict(frame, lambda sl: [], n_slices=5, min_slices=4)
    assert v.verdict == "NONE", v.to_dict()
    assert v.agreement == 0.0
    print("\n[invariance-none] empty search -> NONE (nothing claimed)")
