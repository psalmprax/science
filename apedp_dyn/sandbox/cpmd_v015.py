"""CPMD v0.15 — narrow-claim confirmation (Option A).

Prereg: docs/CPMD_PREREGISTRATION_v015.md (committed separately BEFORE
this file). Frozen OLS-T, 8 cells, disjoint seed block per cell.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_calibration import gen
from cpmd_effective_slope import LAGS
from cpmd_v014 import finwin_truth, ols_t_cell, sl

CELLS = [
    # (name, lags, tcrit, wide, nl, seed0)
    ("base4-12 nl=0.05", LAGS, 2.3646, False, 0.05, 9000),
    ("base4-12 nl=0.10", LAGS, 2.3646, False, 0.10, 9200),
    ("base4-12 nl=0.15", LAGS, 2.3646, False, 0.15, 9400),
    ("base4-12 nl=0.20", LAGS, 2.3646, False, 0.20, 9600),
    ("lags3-10 nl=0.10", np.arange(3, 11, dtype=float), 2.5706, False, 0.10, 9800),
    ("lags3-10 nl=0.20", np.arange(3, 11, dtype=float), 2.5706, False, 0.20, 10000),
    ("lags6-15 nl=0.10", np.arange(6, 16, dtype=float), 2.3060, True, 0.10, 10200),
    ("lags6-15 nl=0.20", np.arange(6, 16, dtype=float), 2.3060, True, 0.20, 10400),
]


def run():
    ok = 0
    for name, lags, tc, wide, nl, seed0 in CELLS:
        truth = finwin_truth(lags, "power")
        s, good = ols_t_cell(lags, tc, truth,
                             lambda k, nl=nl, wd=wide, lg=lags, s0=seed0:
                             sl(lg, gen("power", nl, s0 + k, wide=wd)[1]))
        print(f"{name}: {s}")
        ok += good
    print(f"v0.15 Option A: {ok}/8 {'CONFIRMED' if ok == 8 else 'ENVELOPE SHRINKS'}")


if __name__ == "__main__":
    run()
