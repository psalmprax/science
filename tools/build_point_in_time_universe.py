"""Build a point-in-time universe from a wide market panel.

Reads a panel in the repository's wide format (one row per bar, a date column,
and ``<ASSET>_CLOSE`` / ``<ASSET>_VOL`` columns), and writes:

  * ``<out>/panel_masked.csv``   -- the panel with forward-filled bars removed
    (an asset that stopped trading ends; it is not held flat);
  * ``<out>/membership.csv``     -- for every bar, which assets are in the
    universe, ranked on traded value over the bars strictly before it;
  * ``<out>/universe_audit.json`` -- what was found, and whether the result
    supports an economic claim.

This tool does not download anything. A point-in-time universe is only as
honest as the candidate pool it is ranked from, so the input must contain the
assets that were tradable during the window and later were not -- delisted
pairs, renamed tickers, failed projects -- each up to its last real bar. Ranking
the shipped ``multi_asset_90_5yr_orderflow.csv`` removes the bars that never
traded and the selection inside those 90 names; it does not add back the names
a 2026 list left out, and the audit says so.

Usage:
    python tools/build_point_in_time_universe.py \\
        --panel data/market/multi_asset_90_5yr_orderflow.csv \\
        --pool-fixed-as-of 2026-10-08 --top-n 30 --out /tmp/universe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.market.universe import (  # noqa: E402 - repo root must be importable first
    DEFAULT_DATE_COLUMN,
    audit_universe,
    find_stale_runs,
    mask_stale_bars,
    point_in_time_membership,
)


def build(
    panel_path: Path,
    out_dir: Path,
    *,
    pool_fixed_as_of: str | None,
    top_n: int,
    lookback_bars: int,
    rebalance_every: int,
    date_column: str = DEFAULT_DATE_COLUMN,
) -> dict[str, Any]:
    panel = pd.read_csv(panel_path)
    if date_column not in panel.columns:
        raise SystemExit(f"{panel_path} has no {date_column!r} column")
    stale = find_stale_runs(panel, date_column=date_column)
    masked = mask_stale_bars(panel, date_column=date_column)
    membership = point_in_time_membership(
        panel,
        top_n=top_n,
        lookback_bars=lookback_bars,
        rebalance_every=rebalance_every,
        date_column=date_column,
    )
    # The audit that matters is of the candidate pool: ranking it point-in-time
    # does not change when the pool itself was drawn up.
    pool_audit = audit_universe(
        masked, selected_as_of=pool_fixed_as_of, selection_basis="point_in_time_from_pool",
        date_column=date_column,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    masked.to_csv(out_dir / "panel_masked.csv", index=False)
    membership.astype(int).to_csv(out_dir / "membership.csv")
    report: dict[str, Any] = {
        "source_panel": str(panel_path),
        "top_n": top_n,
        "lookback_bars": lookback_bars,
        "rebalance_every": rebalance_every,
        "bars": len(panel),
        "bars_with_a_universe": int(membership.any(axis=1).sum()),
        "assets_ever_in_universe": int(membership.any(axis=0).sum()),
        "forward_filled_runs_removed": [run.to_dict() for run in stale],
        "candidate_pool_audit": pool_audit.to_dict(),
    }
    (out_dir / "universe_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True, help="wide panel CSV")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--pool-fixed-as-of",
        default=None,
        help="date the candidate pool's asset list was fixed (YYYY-MM-DD); omit if unknown",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--lookback-bars", type=int, default=60)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--date-column", default=DEFAULT_DATE_COLUMN)
    args = parser.parse_args(argv)
    report = build(
        args.panel,
        args.out,
        pool_fixed_as_of=args.pool_fixed_as_of,
        top_n=args.top_n,
        lookback_bars=args.lookback_bars,
        rebalance_every=args.rebalance_every,
        date_column=args.date_column,
    )
    pool = report["candidate_pool_audit"]
    print(f"wrote {args.out}/panel_masked.csv, membership.csv, universe_audit.json")
    print(f"forward-filled runs removed: {len(report['forward_filled_runs_removed'])}")
    print(f"candidate pool fit for economic claims: {pool['fit_for_economic_claims']}")
    for reason in pool["reasons"]:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
