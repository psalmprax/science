#!/usr/bin/env python3
"""Six further published regularities on the survivorship-free crypto panel, for PREREG-LAW-AUDIT-FINANCE-MORE-2026-11.

The claims are stated, with their sources, in the preregistration. The numbers and the verdict rule
are ``apedp_dyn.market.law_audit_more``, built around the first batch's frozen module.

    python scripts/run_law_audit_finance_more.py calibrate --out DIR [--manifest M.json] [--shard k/N]
    python scripts/run_law_audit_finance_more.py stretch   --out DIR
    python scripts/run_law_audit_finance_more.py power     --out DIR [--manifest M.json] [--shard k/N]
    python scripts/run_law_audit_finance_more.py summarise --out DIR
    python scripts/run_law_audit_finance_more.py run       --panel-dir DIR --out DIR

``calibrate`` and ``power`` run on simulated panels (``apedp_dyn.market.law_audit_more_worlds``) with
the real panel's shape, taken from its manifest: the first and last bar of each asset, and no price.
``calibrate`` measures how far each number's resampling interval has to be widened to cover the
number's long-run value as often as it claims; ``stretch`` turns that into the factors that
``law_audit_more.STRETCH`` must carry. ``power`` gives every claim its verdict in worlds where it is
true, where it is false and where it is reversed, on panels the calibration never saw; ``summarise``
counts the verdicts.

``run`` is the one preregistered run. It reads ``panel_development.csv.gz`` only, checks its SHA-256
against the digest pinned by the first batch's runner, and writes ``results.json``. The holdout
panel is not opened by anything in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_law_audit_finance as first  # noqa: E402

from apedp_dyn.market import law_audit as la  # noqa: E402
from apedp_dyn.market import law_audit_more as lm  # noqa: E402
from apedp_dyn.market import law_audit_more_worlds as mw  # noqa: E402

PREREGISTRATION_ID = "PREREG-LAW-AUDIT-FINANCE-MORE-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-LAW-AUDIT-FINANCE-MORE.json"
POWER_DIR = "preregistration/POWER-LAW-AUDIT-FINANCE-MORE-2026-11"
FROZEN = ("preregistration/PREREG-LAW-AUDIT-FINANCE-MORE-2026-11.md", "scripts/run_law_audit_finance_more.py",
          "apedp_dyn/market/law_audit_more.py", "apedp_dyn/market/law_audit_more_worlds.py",
          "apedp_dyn/market/law_audit.py", "apedp_dyn/market/law_audit_worlds.py",
          "scripts/run_law_audit_finance.py", f"{POWER_DIR}/calibration.json", f"{POWER_DIR}/summary.json")

CALIBRATION_REPLICATES = 300
CALIBRATION_SEED = 155_000
POWER_REPLICATES = 150
POWER_SEED = 177_000


def frozen_digests() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN}


def check_frozen() -> None:
    if not DIGEST.exists():
        raise SystemExit(f"{DIGEST.name} is missing: the preregistration is not frozen")
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    changed = sorted(f for f, digest in frozen_digests().items() if recorded.get(f) != digest)
    if changed:
        raise SystemExit(f"frozen files differ from {DIGEST.name}: {changed}")


# --- simulated panels ---------------------------------------------------------------------------

def world_in(worlds: tuple[mw.MoreWorld, ...], name: str) -> tuple[int, mw.MoreWorld]:
    for index, world in enumerate(worlds):
        if world.name == name:
            return index, world
    raise SystemExit(f"no world named {name!r}")


def simulated(world: mw.MoreWorld, index: int, replicate: int, seed: int, shape: tuple[Any, Any, Any]) -> lm.FlowPanel:
    dates, assets, mask = shape
    return mw.simulate(world, dates, assets, mask, np.random.default_rng(seed + 10_000 * index + replicate))


def jobs_of(worlds: tuple[mw.MoreWorld, ...], replicates: int, only: str | None) -> list[tuple[int, mw.MoreWorld, int]]:
    names = only.split(",") if only else [w.name for w in worlds]
    return [(*world_in(worlds, name), r) for name in names for r in range(replicates)]


# --- how far the intervals have to be widened ------------------------------------------------------

def run_calibrate(out: Path, manifest: Path | None, replicates: int, shard: str | None, only: str | None) -> int:
    shape = first.shape_of_the_panel(manifest)
    jobs, name = first.shard_of(jobs_of(mw.CALIBRATION_WORLDS, replicates, only), shard, "calibrate")

    def rows() -> Iterable[dict[str, Any]]:
        for index, world, replicate in jobs:
            measured = lm.measure(lm.prepare(simulated(world, index, replicate, CALIBRATION_SEED, shape)))
            yield {"world": world.name, "replicate": replicate,
                   "numbers": {n: [measured[n]["estimate"], *measured[n]["percentile"]] for n in lm.NUMBERS}}

    first.write_rows(out, name, rows(), len(jobs))
    return 0


def calibration_table(rows: list[dict[str, Any]], level: float = la.LEVEL) -> dict[str, Any]:
    """Per world and number: its spread over panels, the interval's width, and the widening it needs."""
    table: dict[str, Any] = {}
    for world in dict.fromkeys(r["world"] for r in rows):
        chosen = [r for r in rows if r["world"] == world]
        cell: dict[str, Any] = {"replicates": len(chosen), "numbers": {}}
        for number in lm.NUMBERS:
            values = np.array([r["numbers"][number] for r in chosen], dtype=float)
            usable = values[np.isfinite(values).all(axis=1)]
            if len(usable) < 0.9 * len(values):
                cell["numbers"][number] = {"no_interval": len(values) - len(usable)}
                continue
            centre = float(usable[:, 0].mean())
            needed = first.needed_stretch(usable, centre)
            finite = np.where(np.isfinite(needed), needed, np.inf)
            cell["numbers"][number] = {
                "mean": centre, "sd": float(usable[:, 0].std()),
                "mean_half_width": float((usable[:, 2] - usable[:, 1]).mean() / 2.0),
                "covered_unwidened": float((finite <= 1.0).mean()),
                "stretch_needed": float(np.quantile(finite, 1.0 - level)),
                "no_interval": len(values) - len(usable)}
        table[world] = cell
    return table


def stretch_from(table: dict[str, Any]) -> dict[str, float]:
    """The factor per claim: the largest need over the calibration worlds, rounded up to a tenth."""
    factors: dict[str, float] = {}
    for claim in lm.CLAIMS:
        needs = [cell["numbers"][claim].get("stretch_needed", math.inf) for cell in table.values()]
        factors[claim] = max(1.0, math.ceil(max(needs) * 10.0 - 1e-9) / 10.0)
    return factors


def run_stretch(out: Path) -> int:
    rows = first.read_rows(out, "calibrate")
    table = calibration_table(rows)
    record: dict[str, Any] = {
        "worlds": {name: world_dials(world_in(mw.CALIBRATION_WORLDS, name)[1]) for name in table},
        "level": la.LEVEL, "rows": len(rows), "seed": CALIBRATION_SEED, "table": table,
        "stretch": stretch_from(table)}
    (out / "calibration.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for world, cell in table.items():
        print(f"{world} ({cell['replicates']})")
        for number, c in cell["numbers"].items():
            if "mean" in c:
                print(f"   {number:18s} mean {c['mean']:+.4f} sd {c['sd']:.4f} half-width {c['mean_half_width']:.4f}"
                      f" covered {c['covered_unwidened']:.3f} needs {c['stretch_needed']:.2f}")
            else:
                print(f"   {number:18s} no interval in {c['no_interval']} panels")
    print("stretch:", record["stretch"], "| law_audit_more.STRETCH:", dict(lm.STRETCH))
    return 0


def world_dials(world: mw.MoreWorld) -> dict[str, Any]:
    return {**{k: v for k, v in world.__dict__.items() if k != "base"}, "base": dict(world.base.__dict__)}


# --- the power check ---------------------------------------------------------------------------

def one_power_row(world: mw.MoreWorld, index: int, replicate: int, shape: tuple[Any, Any, Any]) -> dict[str, Any]:
    audit = lm.audit(simulated(world, index, replicate, POWER_SEED, shape))
    return {"world": world.name, "replicate": replicate,
            "readings": {c: [r["estimate"], r["low"], r["high"], r["verdict"]] for c, r in audit["readings"].items()}}


def run_power(out: Path, manifest: Path | None, replicates: int, shard: str | None, only: str | None) -> int:
    shape = first.shape_of_the_panel(manifest)
    jobs, name = first.shard_of(jobs_of(mw.WORLDS, replicates, only), shard, "power")
    first.write_rows(out, name, (one_power_row(world, index, replicate, shape) for index, world, replicate in jobs),
                     len(jobs))
    return 0


def summarise_power(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per world and claim: whether the claim is true there, how often each verdict, and the number's spread."""
    summary: dict[str, Any] = {}
    for name in dict.fromkeys(r["world"] for r in rows):
        world = world_in(mw.WORLDS, name)[1]
        chosen = [r for r in rows if r["world"] == name]
        cell: dict[str, Any] = {"replicates": len(chosen), "readings": {}}
        for claim in lm.CLAIMS:
            values = np.array([r["readings"][claim][:3] for r in chosen], dtype=float)
            verdicts = [r["readings"][claim][3] for r in chosen]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # a number with no value at all
                cell["readings"][claim] = {
                    "holds": mw.holds(world, claim),
                    **{v: verdicts.count(v) for v in (la.FAILS, la.SURVIVES, la.NOT_REJECTED)},
                    "mean": float(np.nanmean(values[:, 0])), "sd": float(np.nanstd(values[:, 0])),
                    "mean_half_width": float(np.nanmean(values[:, 2] - values[:, 1]) / 2.0),
                    "no_interval": int(np.isnan(values[:, 1]).sum())}
        true_claims = [claim for claim in lm.CLAIMS if cell["readings"][claim]["holds"]]
        cell["claims_that_hold"] = true_claims
        cell["panels_failing_a_claim_that_holds"] = sum(
            any(r["readings"][claim][3] == la.FAILS for claim in true_claims) for r in chosen)
        cell["panels_passing_a_claim_that_does_not_hold"] = sum(
            any(r["readings"][claim][3] == la.SURVIVES for claim in lm.CLAIMS if claim not in true_claims)
            for r in chosen)
        summary[name] = cell
    return summary


def totals(summary: dict[str, Any]) -> dict[str, Any]:
    """Per claim, over all worlds: verdicts where it is true and where it is not."""
    out: dict[str, Any] = {}
    for claim in lm.CLAIMS:
        true = [cell["readings"][claim] | {"n": cell["replicates"]} for cell in summary.values() if cell["readings"][claim]["holds"]]
        false = [cell["readings"][claim] | {"n": cell["replicates"]} for cell in summary.values() if not cell["readings"][claim]["holds"]]
        out[claim] = {
            "panels_where_true": sum(c["n"] for c in true), "failed_where_true": sum(c[la.FAILS] for c in true),
            "survived_where_true": sum(c[la.SURVIVES] for c in true),
            "panels_where_not_true": sum(c["n"] for c in false), "survived_where_not_true": sum(c[la.SURVIVES] for c in false),
            "failed_where_not_true": sum(c[la.FAILS] for c in false)}
    return out


def run_summarise(out: Path) -> int:
    rows = first.read_rows(out, "power")
    summary = summarise_power(rows)
    record: dict[str, Any] = {
        "worlds": {w.name: world_dials(w) for w in mw.WORLDS}, "level": la.LEVEL, "rows": len(rows),
        "seed": POWER_SEED, "stretch": dict(lm.STRETCH), "summary": summary, "totals": totals(summary)}
    (out / "summary.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for world, cell in summary.items():
        print(f"{world} ({cell['replicates']})")
        for claim, c in cell["readings"].items():
            print(f"   {claim:12s} {'true ' if c['holds'] else 'false'} fails {c[la.FAILS]:4d}  survives "
                  f"{c[la.SURVIVES]:4d}  not rejected {c[la.NOT_REJECTED]:4d} | mean {c['mean']:+.4f} "
                  f"sd {c['sd']:.4f} half-width {c['mean_half_width']:.4f}")
        print(f"   panels failing a claim that holds: {cell['panels_failing_a_claim_that_holds']}"
              f" | passing one that does not: {cell['panels_passing_a_claim_that_does_not_hold']}")
    for claim, t in record["totals"].items():
        print(f"{claim:12s} true in {t['panels_where_true']:5d} panels: failed {t['failed_where_true']:4d}, survived "
              f"{t['survived_where_true']:5d} | not true in {t['panels_where_not_true']:5d}: survived "
              f"{t['survived_where_not_true']:4d}, failed {t['failed_where_not_true']:5d}")
    return 0


# --- the preregistered run ----------------------------------------------------------------------

def subgroups(flows: lm.FlowPanel) -> dict[str, lm.FlowPanel]:
    """The parts of the panel the first batch reports on separately, without a criterion."""
    panel = flows.panel
    present = np.isfinite(panel.close)
    last = np.where(present.any(axis=0), len(panel.dates) - 1 - np.argmax(present[::-1], axis=0), -1)
    stopped = last < len(panel.dates) - 1 - first.STOPPED_GAP_DAYS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # an asset with no bar at all
        value = np.nanmedian(panel.close * panel.volume, axis=0)
    larger = value >= np.nanmedian(value)
    return {"fit_period": flows.between(*first.FIT), "test_period": flows.between(*first.TEST),
            "stopped_trading": flows.columns(stopped), "still_trading": flows.columns(~stopped),
            "larger_half": flows.columns(larger), "smaller_half": flows.columns(~larger & np.isfinite(value))}


def run_once(panel_dir: Path, out: Path) -> int:
    check_frozen()
    path = panel_dir / first.PANEL_FILE
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != first.PANEL_SHA256:
        raise SystemExit(f"{first.PANEL_FILE} has SHA-256 {digest}; the preregistered panel is {first.PANEL_SHA256}")
    flows = lm.load_flow_panel(path).between(*first.PERIOD)
    if not bool((np.diff(flows.panel.dates) == np.timedelta64(1, "D")).all()):
        raise SystemExit("the panel's rows are not consecutive calendar days; lags would not be days")
    results: dict[str, Any] = {
        "preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(),
        "panel": {"commit": first.PANEL_COMMIT, "file": first.PANEL_FILE, "sha256": digest,
                  "period": list(first.PERIOD), "days": len(flows.panel.dates), "assets_in_file": len(flows.panel.assets)},
        "audit": lm.audit(flows), "subgroups": {},
    }
    for name, part in subgroups(flows).items():
        try:
            reading = lm.audit(part)
            results["subgroups"][name] = {"assets": reading["assets"], "asset_days": reading["asset_days"],
                                          "readings": reading["readings"],
                                          "numbers": {n: reading["numbers"][n]["estimate"] for n in lm.NUMBERS}}
        except (ValueError, ZeroDivisionError, IndexError) as exc:   # a part too small to measure
            results["subgroups"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"assets {results['audit']['assets']}  asset-days {results['audit']['asset_days']}")
    for claim, r in results["audit"]["readings"].items():
        print(f"   {claim:12s} {r['estimate']:+.4f}  [{r['low']:+.4f}, {r['high']:+.4f}]  published "
              f"{r['published']}  -> {r['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("calibrate", "stretch", "power", "summarise", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="the panel's manifest.json (shape only)")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--shard", default=None, help="k/N")
    parser.add_argument("--worlds", default=None, help="comma-separated world names")
    parser.add_argument("--panel-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        return run_calibrate(args.out, args.manifest, args.replicates or CALIBRATION_REPLICATES, args.shard,
                             args.worlds)
    if args.command == "stretch":
        return run_stretch(args.out)
    if args.command == "power":
        return run_power(args.out, args.manifest, args.replicates or POWER_REPLICATES, args.shard, args.worlds)
    if args.command == "summarise":
        return run_summarise(args.out)
    if args.panel_dir is None:
        raise SystemExit("run needs --panel-dir")
    return run_once(args.panel_dir, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
