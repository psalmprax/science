#!/usr/bin/env python3
"""Five published laws of asset returns on the survivorship-free crypto panel, for PREREG-LAW-AUDIT-FINANCE-2026-11.

The laws are stated, with their published ranges and sources, in the preregistration. The numbers
and the verdict rule are ``apedp_dyn.market.law_audit``.

    python scripts/run_law_audit_finance.py calibrate --out DIR [--manifest M.json] [--shard k/N]
    python scripts/run_law_audit_finance.py stretch   --out DIR
    python scripts/run_law_audit_finance.py power     --out DIR [--manifest M.json] [--shard k/N]
    python scripts/run_law_audit_finance.py summarise --out DIR
    python scripts/run_law_audit_finance.py run       --panel-dir DIR --out DIR

``calibrate`` and ``power`` run on simulated panels (``apedp_dyn.market.law_audit_worlds``) with the
real panel's shape, taken from its manifest: the first and last bar of each asset, and no price.
``calibrate`` measures how far each number's resampling interval has to be widened to cover the
number's long-run value as often as it claims; ``stretch`` turns that into the factors that
``law_audit.STRETCH`` must carry. ``power`` then gives every claim its verdict in worlds where the
law holds and in worlds where it is broken by a known amount, on panels the calibration never saw;
``summarise`` counts the verdicts.

``run`` is the one preregistered run. It reads ``panel_development.csv.gz`` only, checks its SHA-256
against the digest pinned below, and writes ``results.json``. The holdout panel is not opened by
anything in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.market import law_audit as la  # noqa: E402
from apedp_dyn.market import law_audit_worlds as lw  # noqa: E402

PREREGISTRATION_ID = "PREREG-LAW-AUDIT-FINANCE-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-LAW-AUDIT-FINANCE.json"
POWER_DIR = "preregistration/POWER-LAW-AUDIT-FINANCE-2026-11"
FROZEN = ("preregistration/PREREG-LAW-AUDIT-FINANCE-2026-11.md", "scripts/run_law_audit_finance.py",
          "apedp_dyn/market/law_audit.py", "apedp_dyn/market/law_audit_worlds.py",
          f"{POWER_DIR}/calibration.json", f"{POWER_DIR}/summary.json")

#: The panel: branch ``market-data``, built by ``scripts/build_point_in_time_crypto_panel.py``.
PANEL_COMMIT = "a598b83"
PANEL_FILE = "panel_development.csv.gz"
PANEL_SHA256 = "96b5ee0b88c4dd5c004565f2582c8683b440513ad5dc9f8d511a0fb49fa4644e"
PERIOD = ("2021-01-01", "2025-12-31")
FIT = ("2021-01-01", "2023-12-31")      # the form tests fit their shapes here ...
TEST = ("2024-01-01", "2025-12-31")     # ... and are scored here
STOPPED_GAP_DAYS = 30                   # an asset "stopped" if its last bar is this long before the end

#: Worlds the widening factors are measured in: every one keeps all five laws, at the middle of
#: the published ranges and towards their edges (heavier and lighter tails, slower and faster decay).
CALIBRATION_WORLDS = ("all_hold", "tail_2.2", "tail_4", "memory_beta_0.2", "memory_beta_0.4")
CALIBRATION_REPLICATES = 300
CALIBRATION_SEED = 55_000
POWER_REPLICATES = 150
POWER_SEED = 77_000


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

def shape_of_the_panel(manifest: Path | None) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Dates, assets and presence. From the real manifest, or a small stand-in when none is given."""
    if manifest is None:
        dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2024-01-01"))
        assets = tuple(f"A{i:03d}" for i in range(60))
        mask: np.ndarray = np.ones((len(dates), len(assets)), dtype=bool)
        mask[:400, :15] = False      # some start late
        mask[-300:, 15:25] = False   # some stop early
        return dates, assets, mask
    record = json.loads(manifest.read_text(encoding="utf-8"))
    shape: tuple[np.ndarray, tuple[str, ...], np.ndarray] = lw.presence(
        record["first_bar"], record["last_bar"], *PERIOD)
    return shape


def world_named(name: str) -> tuple[int, lw.World]:
    for index, world in enumerate(lw.WORLDS):
        if world.name == name:
            return index, world
    raise SystemExit(f"no world named {name!r}")


def simulated(world: lw.World, index: int, replicate: int, seed: int, shape: tuple[Any, Any, Any]) -> la.Panel:
    dates, assets, mask = shape
    return lw.simulate(world, dates, assets, mask, np.random.default_rng(seed + 10_000 * index + replicate))


def shard_of(jobs: list[Any], shard: str | None, stem: str) -> tuple[list[Any], str]:
    if not shard:
        return jobs, f"{stem}.jsonl"
    k, n = (int(v) for v in shard.split("/"))
    return [job for position, job in enumerate(jobs) if position % n == k], f"{stem}-{k}-of-{n}.jsonl"


def write_rows(out: Path, name: str, rows: Iterable[dict[str, Any]], count: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    print(f"{count} simulated panels in {time.time() - started:.0f} s -> {out / name}")


def read_rows(out: Path, stem: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for path in sorted(out.glob(f"{stem}*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no {stem} rows in {out}")
    return rows


# --- how far the intervals have to be widened ------------------------------------------------------

def run_calibrate(out: Path, manifest: Path | None, replicates: int, shard: str | None,
                  only: str | None) -> int:
    shape = shape_of_the_panel(manifest)
    names = only.split(",") if only else list(CALIBRATION_WORLDS)
    jobs, name = shard_of([(*world_named(w), r) for w in names for r in range(replicates)], shard, "calibrate")

    def rows() -> Iterable[dict[str, Any]]:
        for index, world, replicate in jobs:
            measured = la.measure(la.prepare(simulated(world, index, replicate, CALIBRATION_SEED, shape)))
            yield {"world": world.name, "replicate": replicate,
                   "numbers": {n: [measured[n]["estimate"], *measured[n]["percentile"]] for n in la.NUMBERS}}

    write_rows(out, name, rows(), len(jobs))
    return 0


def needed_stretch(values: np.ndarray, centre: float) -> np.ndarray:
    """For each (estimate, low, high): the widening at which the interval reaches ``centre``."""
    estimate, low, high = values[:, 0], values[:, 1], values[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        up = (centre - estimate) / np.where(high > estimate, high - estimate, np.nan)
        down = (estimate - centre) / np.where(low < estimate, estimate - low, np.nan)
    needed: np.ndarray = np.where(estimate <= centre, up, down)
    return needed


def calibration_table(rows: list[dict[str, Any]], level: float = la.LEVEL) -> dict[str, Any]:
    """Per world and number: its spread over panels, the interval's width, and the widening it needs."""
    table: dict[str, Any] = {}
    for world in dict.fromkeys(r["world"] for r in rows):
        chosen = [r for r in rows if r["world"] == world]
        cell: dict[str, Any] = {"replicates": len(chosen), "numbers": {}}
        for number in la.NUMBERS:
            values = np.array([r["numbers"][number] for r in chosen], dtype=float)
            usable = values[np.isfinite(values).all(axis=1)]
            if len(usable) < 0.9 * len(values):
                cell["numbers"][number] = {"no_interval": len(values) - len(usable)}
                continue
            centre = float(usable[:, 0].mean())
            needed = needed_stretch(usable, centre)
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
    """The factor per number: the largest need over the calibration worlds, rounded up to a tenth."""
    factors: dict[str, float] = {}
    for number in dict.fromkeys(la.NUMBER_OF.values()):
        needs = [cell["numbers"][number].get("stretch_needed", math.inf) for cell in table.values()]
        factors[number] = max(1.0, math.ceil(max(needs) * 10.0 - 1e-9) / 10.0)
    return factors


def run_stretch(out: Path) -> int:
    rows = read_rows(out, "calibrate")
    table = calibration_table(rows)
    record: dict[str, Any] = {
        "worlds": {name: world_named(name)[1].__dict__ for name in table}, "level": la.LEVEL,
        "rows": len(rows), "seed": CALIBRATION_SEED, "table": table, "stretch": stretch_from(table)}
    (out / "calibration.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for world, cell in table.items():
        print(f"{world} ({cell['replicates']})")
        for number, c in cell["numbers"].items():
            if "mean" in c:
                print(f"   {number:12s} mean {c['mean']:+.4f} sd {c['sd']:.4f} half-width {c['mean_half_width']:.4f}"
                      f" covered {c['covered_unwidened']:.3f} needs {c['stretch_needed']:.2f}")
            else:
                print(f"   {number:12s} no interval in {c['no_interval']} panels")
    print("stretch:", record["stretch"], "| law_audit.STRETCH:", dict(la.STRETCH))
    return 0


# --- the power check ---------------------------------------------------------------------------

def split(panel: la.Panel) -> tuple[la.Panel, la.Panel]:
    """The fit and test parts: the recorded dates, or the first three fifths of a shorter panel."""
    if panel.dates[0] <= np.datetime64(FIT[0]) and panel.dates[-1] >= np.datetime64(TEST[1]):
        return panel.between(*FIT), panel.between(*TEST)
    cut = panel.dates[int(0.6 * len(panel.dates))]
    before = cut - np.timedelta64(1, "D")
    return panel.between(str(panel.dates[0]), str(before)), panel.between(str(cut), str(panel.dates[-1]))


def one_power_row(world: lw.World, index: int, replicate: int, shape: tuple[Any, Any, Any]) -> dict[str, Any]:
    panel = simulated(world, index, replicate, POWER_SEED, shape)
    audit = la.audit(panel)
    forms = la.form_tests(*split(panel))
    return {"world": world.name, "replicate": replicate,
            "readings": {c: [r["estimate"], r["low"], r["high"], r["verdict"]] for c, r in audit["readings"].items()},
            "forms": {c: [f["estimate"], f["low"], f["high"], f["favours"]] for c, f in forms.items()}}


def run_power(out: Path, manifest: Path | None, replicates: int, shard: str | None, only: str | None) -> int:
    shape = shape_of_the_panel(manifest)
    jobs, name = shard_of([(i, w, r) for i, w in enumerate(lw.WORLDS) for r in range(replicates)
                           if only is None or w.name in only.split(",")], shard, "power")
    write_rows(out, name, (one_power_row(world, index, replicate, shape) for index, world, replicate in jobs),
               len(jobs))
    return 0


#: The five laws. ``tail_cubic`` is the strict reading of the first; ``independent`` is the control.
LAWS = ("tail", "random", "memory", "leverage", "volume")


def holds(world: lw.World, claim: str, long_run: float) -> bool:
    """Whether ``claim`` is true in ``world``.

    For the claims about a dependence, by the dial that makes it. For the tail index and the decay
    exponent, by whether the number's long-run value (its mean over the simulated panels) lies in
    the published set: the number measured is what the claim is about, and for the tail index it
    is not the dial (a Hill index over the largest 5 % reads lower than the limiting exponent).
    The decay exponent also needs the decay to be a power law at all.
    """
    moving = bool(world.vol_scale > 0)
    if claim == "random":
        return bool(world.autocorrelation == 0)
    if claim == "independent":
        return not moving
    if claim == "leverage":
        return moving and bool(world.leverage > 0)
    if claim == "volume":
        return moving and bool(world.volume_link > 0)
    low, high = la.PUBLISHED[claim]
    inside = bool(low <= long_run <= high)
    return inside and (claim != "memory" or world.memory == "power")     # the decay has to be a power law


def summarise_power(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per world and claim: whether the claim holds there, how often each verdict, and the number's spread."""
    summary: dict[str, Any] = {}
    for name in dict.fromkeys(r["world"] for r in rows):
        world = world_named(name)[1]
        chosen = [r for r in rows if r["world"] == name]
        cell: dict[str, Any] = {"replicates": len(chosen), "readings": {}, "forms": {}}
        for claim in chosen[0]["readings"]:
            values = np.array([r["readings"][claim][:3] for r in chosen], dtype=float)
            verdicts = [r["readings"][claim][3] for r in chosen]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # a number with no value at all
                mean = float(np.nanmean(values[:, 0]))
                cell["readings"][claim] = {
                    "holds": holds(world, claim, mean),
                    **{v: verdicts.count(v) for v in (la.FAILS, la.SURVIVES, la.NOT_REJECTED)},
                    "mean": mean, "sd": float(np.nanstd(values[:, 0])),
                    "mean_half_width": float(np.nanmean(values[:, 2] - values[:, 1]) / 2.0),
                    "no_interval": int(np.isnan(values[:, 1]).sum())}
        true_laws = [claim for claim in LAWS if cell["readings"][claim]["holds"]]
        cell["laws_that_hold"] = true_laws
        cell["panels_failing_a_law_that_holds"] = sum(
            any(r["readings"][claim][3] == la.FAILS for claim in true_laws) for r in chosen)
        for claim in chosen[0]["forms"]:
            favours = [r["forms"][claim][3] for r in chosen]
            cell["forms"][claim] = {v: favours.count(v) for v in (la.PUBLISHED_FORM, la.RIVAL_FORM, la.NOT_DECIDED)}
        summary[name] = cell
    return summary


def run_summarise(out: Path) -> int:
    rows = read_rows(out, "power")
    summary: dict[str, Any] = {
        "worlds": {w.name: w.__dict__ for w in lw.WORLDS}, "level": la.LEVEL, "rows": len(rows),
        "seed": POWER_SEED, "settings": la.audit_settings(), "summary": summarise_power(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for world, cell in summary["summary"].items():
        print(f"{world} ({cell['replicates']})")
        for claim, c in cell["readings"].items():
            print(f"   {claim:12s} {'holds ' if c['holds'] else 'broken'} fails {c[la.FAILS]:4d}  survives "
                  f"{c[la.SURVIVES]:4d}  not rejected {c[la.NOT_REJECTED]:4d} | mean {c['mean']:+.4f} "
                  f"sd {c['sd']:.4f} half-width {c['mean_half_width']:.4f}")
        print(f"   panels failing a law that holds: {cell['panels_failing_a_law_that_holds']}"
              f" | forms: {dict(cell['forms'])}")
    return 0


# --- the preregistered run ----------------------------------------------------------------------

def subgroups(panel: la.Panel) -> dict[str, la.Panel]:
    """Parts of the panel the numbers are reported on separately, without a criterion."""
    present = np.isfinite(panel.close)
    last = np.where(present.any(axis=0), len(panel.dates) - 1 - np.argmax(present[::-1], axis=0), -1)
    stopped = last < len(panel.dates) - 1 - STOPPED_GAP_DAYS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # an asset with no bar at all
        value = np.nanmedian(panel.close * panel.volume, axis=0)
    larger = value >= np.nanmedian(value)
    return {"fit_period": panel.between(*FIT), "test_period": panel.between(*TEST),
            "stopped_trading": panel.columns(stopped), "still_trading": panel.columns(~stopped),
            "larger_half": panel.columns(larger), "smaller_half": panel.columns(~larger & np.isfinite(value))}


def run_once(panel_dir: Path, out: Path) -> int:
    check_frozen()
    path = panel_dir / PANEL_FILE
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PANEL_SHA256:
        raise SystemExit(f"{PANEL_FILE} has SHA-256 {digest}; the preregistered panel is {PANEL_SHA256}")
    panel = la.load_panel(path).between(*PERIOD)
    if not bool((np.diff(panel.dates) == np.timedelta64(1, "D")).all()):
        raise SystemExit("the panel's rows are not consecutive calendar days; lags would not be days")
    results: dict[str, Any] = {
        "preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(),
        "panel": {"commit": PANEL_COMMIT, "file": PANEL_FILE, "sha256": digest, "period": list(PERIOD),
                  "days": len(panel.dates), "assets_in_file": len(panel.assets)},
        "audit": la.audit(panel),
        "forms": la.form_tests(panel.between(*FIT), panel.between(*TEST)),
        "subgroups": {},
    }
    for name, part in subgroups(panel).items():
        try:
            reading = la.audit(part)
            results["subgroups"][name] = {"assets": reading["assets"], "asset_days": reading["asset_days"],
                                          "readings": reading["readings"]}
        except (ValueError, ZeroDivisionError, IndexError) as exc:   # a part too small to measure
            results["subgroups"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"assets {results['audit']['assets']}  asset-days {results['audit']['asset_days']}")
    for claim, r in results["audit"]["readings"].items():
        print(f"   {claim:12s} {r['estimate']:+.4f}  [{r['low']:+.4f}, {r['high']:+.4f}]  published "
              f"{r['published']}  -> {r['verdict']}")
    for claim, f in results["forms"].items():
        print(f"   form of {claim:8s} {f['estimate']:+.5f}  [{f['low']:+.5f}, {f['high']:+.5f}]  -> {f['favours']}")
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
