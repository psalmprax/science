"""Bound what the archive's remaining 83,809 Toffolis can be spent on.

Every input here is measured, not assumed:

* key-equation stage 313,110 T (``scripts/emit_bm_iteration.py``);
* gated Forney block 315 T (``scripts/emit_chien_forney_stage.py``), executed only where the
  locator vanishes, which is at most ``t = 35`` of the 4,095 positions;
* adjoint uncompute 313,110 T, or 0 T for a Z-basis clear (``scripts/emit_uncompute_pass.py``);
* Montgomery batch inversion 774,081 T (``scripts/emit_chien_variants.py``).

So the archive's budget is the line ``313,110 + 315·w + u ≤ 396,919`` over executed blocks
``w`` and everything else ``u``, and this script reports its vertices. It does not decide
which design the archive used -- it says which are arithmetically possible.

    python3 scripts/bound_archive_overhead.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

KEY_EQUATION_TOFFOLIS = 313110.0        # measured, 70 iterations
FORNEY_PER_BLOCK = 315.0                # measured, per executed position
ADJOINT_UNCOMPUTE_TOFFOLIS = 313110.0   # measured, equal to the forward pass
POSITIONS = 4095                        # 2^12 - 1
MAX_LIVE_BLOCKS = 35                    # deg(sigma) <= t = n/2
ARCHIVE_TOFFOLIS = 396919.33            # sealed artifact
REMAINDER = ARCHIVE_TOFFOLIS - KEY_EQUATION_TOFFOLIS


def bound() -> dict:
    max_blocks_at_zero_overhead = REMAINDER / FORNEY_PER_BLOCK
    overhead_at_worst_case_weight = REMAINDER - FORNEY_PER_BLOCK * MAX_LIVE_BLOCKS
    return {
        "script": "scripts/bound_archive_overhead.py",
        "inputs": {
            "key_equation_toffolis": KEY_EQUATION_TOFFOLIS,
            "forney_toffolis_per_block": FORNEY_PER_BLOCK,
            "adjoint_uncompute_toffolis": ADJOINT_UNCOMPUTE_TOFFOLIS,
            "positions": POSITIONS,
            "max_live_blocks": MAX_LIVE_BLOCKS,
            "archive_toffolis": ARCHIVE_TOFFOLIS,
        },
        "feasible_region": {
            "inequality": "313,110 + 315*w + u <= 396,919.33",
            "remainder": REMAINDER,
            "max_blocks_at_zero_overhead": max_blocks_at_zero_overhead,
            "overhead_left_at_worst_case_weight": overhead_at_worst_case_weight,
            "overhead_left_fraction_of_remainder": overhead_at_worst_case_weight / REMAINDER,
            "per_measurement_round_at_worst_case_weight": (
                overhead_at_worst_case_weight / POSITIONS
            ),
            "adjoint_uncompute_fits": ADJOINT_UNCOMPUTE_TOFFOLIS <= REMAINDER,
            "shared_inversion_infeasible_by": 774081.0 - REMAINDER,
            "unconditional_blocks_infeasible_by": FORNEY_PER_BLOCK * POSITIONS - REMAINDER,
        },
        "readings": {
            "static_gated": {
                "total": KEY_EQUATION_TOFFOLIS + FORNEY_PER_BLOCK * POSITIONS
                         + ADJOINT_UNCOMPUTE_TOFFOLIS,
                "fits": False,
            },
            "runtime_gated": {
                "total": KEY_EQUATION_TOFFOLIS + FORNEY_PER_BLOCK * MAX_LIVE_BLOCKS,
                "fits": True,
                "slack": overhead_at_worst_case_weight,
            },
            "runtime_gated_with_adjoint": {
                "total": KEY_EQUATION_TOFFOLIS + FORNEY_PER_BLOCK * MAX_LIVE_BLOCKS
                         + ADJOINT_UNCOMPUTE_TOFFOLIS,
                "fits": False,
                "exceeds_by": (KEY_EQUATION_TOFFOLIS + FORNEY_PER_BLOCK * MAX_LIVE_BLOCKS
                               + ADJOINT_UNCOMPUTE_TOFFOLIS - ARCHIVE_TOFFOLIS),
            },
        },
        "conclusion": (
            f"the remainder of {REMAINDER:,.0f} T covers the gated Forney block at worst-case "
            f"weight {FORNEY_PER_BLOCK * MAX_LIVE_BLOCKS:,.0f} T with "
            f"{overhead_at_worst_case_weight:,.0f} T ({100 * overhead_at_worst_case_weight / REMAINDER:.0f}%) "
            f"left over -- {overhead_at_worst_case_weight / POSITIONS:.1f} T per measurement "
            f"round across {POSITIONS:,} rounds, or up to {max_blocks_at_zero_overhead:.0f} "
            f"executed blocks if nothing else is paid for. The adjoint uncompute does NOT fit "
            f"({ADJOINT_UNCOMPUTE_TOFFOLIS:,.0f} T against {REMAINDER:,.0f} T), so the archive's "
            "uncompute is measurement-based, and neither sharing scheme is affordable: the "
            "Montgomery batch inversion misses by "
            f"{774081.0 - REMAINDER:,.0f} T and unconditional blocks by "
            f"{FORNEY_PER_BLOCK * POSITIONS - REMAINDER:,.0f} T"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=pathlib.Path, help="write the receipt here")
    args = parser.parse_args()

    receipt = bound()
    f = receipt["feasible_region"]
    assert abs(f["remainder"] - 83809.33) < 0.01, f["remainder"]
    assert abs(f["max_blocks_at_zero_overhead"] - 266.06) < 0.01
    assert abs(f["overhead_left_at_worst_case_weight"] - 72784.33) < 0.01
    assert f["adjoint_uncompute_fits"] is False
    assert receipt["readings"]["runtime_gated"]["fits"] is True
    assert receipt["readings"]["static_gated"]["fits"] is False

    print(f"remainder {f['remainder']:,.0f} T = 315*w + u, w <= {f['max_blocks_at_zero_overhead']:.2f}")
    print(f"at w = {MAX_LIVE_BLOCKS}: Forney {FORNEY_PER_BLOCK * MAX_LIVE_BLOCKS:,.0f} T, "
          f"overhead left {f['overhead_left_at_worst_case_weight']:,.0f} T "
          f"({f['per_measurement_round_at_worst_case_weight']:.1f} T per round)")
    print(f"adjoint fits: {f['adjoint_uncompute_fits']} | Montgomery misses by "
          f"{f['shared_inversion_infeasible_by']:,.0f} T | unconditional by "
          f"{f['unconditional_blocks_infeasible_by']:,.0f} T")
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
