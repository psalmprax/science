"""What the quantum-signal study could have detected, and what it actually cost.

Two numbers the five tranche records never reported, both computed from records
alone -- no panel is read and no holdout slice is touched:

1. **Statistical power.** Every tranche reports how many rows cleared the 0.95
   floor. None reports the smallest *true* Sharpe the design could have detected
   at that floor, given the bars it had and the trials it charged. Without that
   number a null result cannot be told apart from a blunt test: the design may be
   rejecting the hypothesis, or rejecting everything.

   `minimum_detectable_sharpe` inverts the Deflated Sharpe Ratio the records
   already use. `compute_deflated_sharpe_ratio` maps a true Sharpe to a DSR;
   the MDS is the Sharpe at which that map reaches 0.95. Power at a candidate
   true Sharpe `s` is `P(SR_hat >= MDS)` under `SR_hat ~ N(s, se(s))`, using the
   same standard error the DSR itself uses.

2. **The programme charge.** Each tranche charges its own trials and builds its
   own hurdle, which is correct within a tranche and wrong across a research
   programme: five tranches, five asset sets, five resolution choices and a
   handful of mid-course method changes were all decided by a researcher who had
   already seen the previous tranche. This recomputes every row's hurdle against
   the trials the whole programme spent, and reports what moved.

The output is `benchmarks/quantum_compilation/quantum_signal_power.json`. The
script refuses to write it unless every check passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD  # noqa: E402
from apedp_dyn.epistemic.trial_ledger import (  # noqa: E402
    compute_deflated_sharpe_ratio,
)

RECORD_DIR = ROOT / "benchmarks" / "quantum_compilation"
#: The five records, in the order they were run. Tranche 1 is the un-suffixed file.
RECORDS = {
    "tranche1": RECORD_DIR / "quantum_signal_ledger.json",
    "tranche2": RECORD_DIR / "quantum_signal_ledger_tranche2.json",
    "tranche3": RECORD_DIR / "quantum_signal_ledger_tranche3.json",
    "tranche4": RECORD_DIR / "quantum_signal_ledger_tranche4.json",
    "tranche5": RECORD_DIR / "quantum_signal_ledger_tranche5.json",
}
#: The reference true Sharpes power is quoted at. 1.0 is the informal bar for a
#: tradeable edge; 0.5 is a good one; 3.0 is the kind of number the floor is
#: actually sized for.
REFERENCE_SHARPES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
#: The floor the DSR has to reach, imported rather than typed.
FLOOR = DSR_THRESHOLD
Z_FLOOR = 1.6448536269514722  # Φ^-1(0.95), used only for the closed-form note


def _dsr(observed: float, *, n_bars: int, n_trials: int, var_trials: float,
         skewness: float, kurtosis: float, annualization: float) -> dict:
    return compute_deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_trials=n_trials,
        var_trials=var_trials,
        n_bars=n_bars,
        skewness=skewness,
        kurtosis=kurtosis,
        annualization_factor=annualization,
    )


def minimum_detectable_sharpe(*, n_bars: int, n_trials: int, var_trials: float,
                              skewness: float, kurtosis: float,
                              annualization: float,
                              floor: float = FLOOR) -> float:
    """The true annualised Sharpe at which this design's DSR reaches the floor.

    Monotone in the true Sharpe (both the hurdle and the standard error are
    fixed given the sample and the trial count), so it is solved rather than
    searched. `scipy.optimize.brentq` needs a bracket where the DSR straddles
    the floor; the upper end is widened until it does, which also reports a
    design whose floor can never be met (a NaN-handling case that does not
    arise here but is not assumed away).
    """

    def gap(share: float) -> float:
        return _dsr(share, n_bars=n_bars, n_trials=n_trials,
                    var_trials=var_trials, skewness=skewness,
                    kurtosis=kurtosis, annualization=annualization)["dsr"] - floor

    low, high = 0.0, 1.0
    if gap(low) >= 0.0:
        return 0.0
    while gap(high) < 0.0:
        high *= 2.0
        if high > 1e6:
            raise SystemExit("the floor is unreachable even at Sharpe 1e6")
    return float(optimize.brentq(gap, low, high, xtol=1e-10, rtol=1e-12))


def power_at(true_sharpe: float, *, n_bars: int, n_trials: int,
             var_trials: float, skewness: float, kurtosis: float,
             annualization: float, floor: float = FLOOR) -> dict:
    """P(this design reports a pass | the true Sharpe is `true_sharpe`)."""
    mds = minimum_detectable_sharpe(
        n_bars=n_bars, n_trials=n_trials, var_trials=var_trials,
        skewness=skewness, kurtosis=kurtosis, annualization=annualization,
        floor=floor)
    se = _dsr(true_sharpe, n_bars=n_bars, n_trials=n_trials,
              var_trials=var_trials, skewness=skewness, kurtosis=kurtosis,
              annualization=annualization)["se_sharpe"]
    z = (mds - true_sharpe) / max(se, 1e-15)
    return {
        "true_sharpe": float(true_sharpe),
        "detectable_at_floor": float(mds),
        "standard_error": float(se),
        "power": float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))),
    }


def _row_summary(rows: list[dict]) -> dict[str, float]:
    """Median sample length and shape across the rows at one resolution."""
    return {
        "rows": len(rows),
        "n_bars_median": int(median(int(row["n_obs"]) for row in rows)),
        "n_bars_min": int(min(int(row["n_obs"]) for row in rows)),
        "n_bars_max": int(max(int(row["n_obs"]) for row in rows)),
        "skewness_median": float(median(float(row["skewness"]) for row in rows)),
        "kurtosis_median": float(median(float(row["kurtosis"]) for row in rows)),
        "observed_sharpe_max": float(max(float(row["sharpe"]) for row in rows)),
        "observed_dsr_max": float(max(float(row["dsr"]["dsr"]) for row in rows)),
        "observed_hurdle_max": float(max(float(row["dsr"]["expected_max_sr"])
                                         for row in rows)),
    }


def per_record_power(record: dict) -> dict[str, dict]:
    """MDS and the power curve, per resolution, for one tranche as run."""
    rows = [*record["rows"], *record["baselines"]]
    by_resolution: dict[str, list[dict]] = {}
    for row in rows:
        by_resolution.setdefault(row["resolution"], []).append(row)
    out: dict[str, dict] = {}
    for resolution, subset in sorted(by_resolution.items()):
        summary = _row_summary(subset)
        arguments = dict(
            n_bars=summary["n_bars_median"],
            n_trials=int(record["n_trials"]),
            var_trials=float(record["var_trials"]["by_resolution"][resolution]),
            skewness=summary["skewness_median"],
            kurtosis=summary["kurtosis_median"],
            annualization=float(subset[0]["annualization"]),
        )
        mds = minimum_detectable_sharpe(**arguments)
        out[resolution] = {
            **summary,
            "n_trials": int(record["n_trials"]),
            "var_trials": arguments["var_trials"],
            "annualization": arguments["annualization"],
            "minimum_detectable_sharpe": mds,
            "power_curve": [power_at(value, **arguments) for value in REFERENCE_SHARPES],
        }
    return out


def programme_charge(records: dict[str, dict]) -> dict:
    """Re-price every row against every trial the five tranches spent."""
    total_trials = int(sum(int(record["n_trials"]) for record in records.values()))
    sharpes: dict[str, list[float]] = {}
    rows: list[tuple[str, dict]] = []
    for name, record in records.items():
        for row in [*record["rows"], *record["baselines"]]:
            rows.append((name, row))
            sharpes.setdefault(row["resolution"], []).append(float(row["sharpe"]))
        for fit in record.get("parameterised", {}).get("fits", []):
            sharpes.setdefault(fit["resolution"], []).extend(
                float(candidate["sharpe"]) for candidate in fit["candidates"])
    dispersion = {resolution: (float(np.var(values)) if len(values) > 1 else 0.0)
                  for resolution, values in sharpes.items()}

    repriced: list[dict] = []
    for name, row in rows:
        # The charge is the trial count, held at the dispersion the tranche itself used.
        # The expected maximum is increasing in the trial count at fixed sigma, so this
        # isolates what the programme spent from how the dispersion was estimated.
        own = float(records[name]["var_trials"]["by_resolution"][row["resolution"]])
        new = _dsr(float(row["sharpe"]), n_bars=int(row["n_obs"]),
                   n_trials=total_trials,
                   var_trials=own,
                   skewness=float(row["skewness"]),
                   kurtosis=float(row["kurtosis"]),
                   annualization=float(row["annualization"]))
        # Pooling the dispersion across tranches is a second, independent change that
        # can move a hurdle either way; it is reported beside the charge, never as it.
        pooled = _dsr(float(row["sharpe"]), n_bars=int(row["n_obs"]),
                      n_trials=total_trials,
                      var_trials=dispersion[row["resolution"]],
                      skewness=float(row["skewness"]),
                      kurtosis=float(row["kurtosis"]),
                      annualization=float(row["annualization"]))
        repriced.append({
            "record": name,
            "dataset_id": row["dataset_id"],
            "resolution": row["resolution"],
            "features": row["features"],
            "threshold": row["threshold"],
            "sharpe": float(row["sharpe"]),
            "n_bars": int(row["n_obs"]),
            "skewness": float(row["skewness"]),
            "kurtosis": float(row["kurtosis"]),
            "annualization": float(row["annualization"]),
            "hurdle_as_run": float(row["dsr"]["expected_max_sr"]),
            "hurdle_programme": float(new["expected_max_sr"]),
            "hurdle_programme_pooled_dispersion": float(pooled["expected_max_sr"]),
            "dsr_as_run": float(row["dsr"]["dsr"]),
            "dsr_programme": float(new["dsr"]),
            "dsr_programme_pooled_dispersion": float(pooled["dsr"]),
            "significant_as_run": bool(row["dsr"]["is_significant"]),
            "significant_programme": bool(new["is_significant"]),
        })

    by_resolution = {}
    for resolution, values in sorted(sharpes.items()):
        subset = [item for item in repriced if item["resolution"] == resolution]
        arguments = dict(
            n_bars=int(median(int(item["n_bars"]) for item in subset)),
            n_trials=total_trials,
            var_trials=float(median(
                records[item["record"]]["var_trials"]["by_resolution"][resolution]
                for item in subset)) if subset else 0.0,
            skewness=float(median(item["skewness"] for item in subset)) if subset else 0.0,
            kurtosis=float(median(item["kurtosis"] for item in subset)) if subset else 3.0,
            annualization=float(subset[0]["annualization"]) if subset else 1.0,
        )
        by_resolution[resolution] = {
            "rows": len(subset),
            "var_trials": arguments["var_trials"],
            "var_trials_pooled_across_tranches": dispersion[resolution],
            "annualization": arguments["annualization"],
            "minimum_detectable_sharpe": minimum_detectable_sharpe(**arguments),
            "power_curve": [power_at(value, **arguments) for value in REFERENCE_SHARPES],
        }

    became_significant = [item for item in repriced
                          if item["significant_programme"] and not item["significant_as_run"]]
    return {
        "total_trials": total_trials,
        "trials_by_record": {name: int(record["n_trials"])
                             for name, record in records.items()},
        "dispersion_by_resolution": dispersion,
        "by_resolution": by_resolution,
        "rows": repriced,
        "rows_clearing_floor_as_run": sum(1 for item in repriced
                                          if item["significant_as_run"]),
        "rows_clearing_floor_programme": sum(1 for item in repriced
                                             if item["significant_programme"]),
        "rows_that_become_significant": len(became_significant),
        "hurdle_uplift": {
            resolution: {
                "as_run_median": float(median(item["hurdle_as_run"] for item in repriced
                                              if item["resolution"] == resolution)),
                "programme_median": float(median(item["hurdle_programme"] for item in repriced
                                                 if item["resolution"] == resolution)),
            }
            for resolution in sorted(sharpes)
        },
        "note": (
            "every row of every tranche re-priced against the programme's whole trial "
            "count, held at the dispersion that row's tranche used, so the number moved "
            "is the multiplicity the tranche structure hid and not a change of "
            "estimator; the pooled-dispersion variant is carried beside it per row as a "
            "sensitivity, because pooling can move a hurdle either way"),
    }


def build(records: dict[str, dict]) -> dict:
    by_record = {name: per_record_power(record) for name, record in records.items()}
    programme = programme_charge(records)

    checks: dict[str, bool] = {}
    # The MDS is defined by the floor, so evaluating the DSR there must land on it.
    checks["the_minimum_detectable_sharpe_meets_the_floor"] = all(
        abs(_dsr(entry["minimum_detectable_sharpe"],
                 n_bars=entry["n_bars_median"], n_trials=entry["n_trials"],
                 var_trials=entry["var_trials"],
                 skewness=entry["skewness_median"],
                 kurtosis=entry["kurtosis_median"],
                 annualization=entry["annualization"])["dsr"] - FLOOR) < 1e-6
        for per_resolution in by_record.values()
        for entry in per_resolution.values()
    )
    checks["power_is_monotone_in_the_true_sharpe"] = all(
        all(a["power"] <= b["power"] + 1e-12
            for a, b in zip(curve, curve[1:]))
        for per_resolution in by_record.values()
        for entry in per_resolution.values()
        for curve in [entry["power_curve"]]
    )
    checks["the_programme_charge_covers_every_tranche"] = (
        programme["total_trials"] == sum(programme["trials_by_record"].values())
        and set(programme["trials_by_record"]) == set(records)
    )
    checks["the_programme_hurdle_is_never_lower_than_the_tranche_hurdle"] = all(
        item["hurdle_programme"] >= item["hurdle_as_run"] - 1e-9
        for item in programme["rows"]
    )
    checks["pooling_dispersion_across_tranches_is_reported_separately"] = all(
        "hurdle_programme_pooled_dispersion" in item
        and "dsr_programme_pooled_dispersion" in item
        for item in programme["rows"]
    )
    checks["no_row_becomes_significant_under_the_stricter_charge"] = (
        programme["rows_that_become_significant"] == 0
    )
    checks["every_recorded_row_is_reported"] = (
        len(programme["rows"])
        == sum(len(record["rows"]) + len(record["baselines"]) for record in records.values())
    )
    checks["the_record_reproduces_its_own_recorded_dsr"] = all(
        abs(_dsr(float(row["sharpe"]), n_bars=int(row["n_obs"]),
                 n_trials=int(record["n_trials"]),
                 var_trials=float(record["var_trials"]["by_resolution"][row["resolution"]]),
                 skewness=float(row["skewness"]), kurtosis=float(row["kurtosis"]),
                 annualization=float(row["annualization"]))["dsr"]
            - float(row["dsr"]["dsr"])) < 1e-9
        for record in records.values()
        for row in [*record["rows"], *record["baselines"]]
    )

    if not all(checks.values()):
        raise SystemExit("refusing to write a power record with failed checks: "
                         + ", ".join(name for name, ok in checks.items() if not ok))

    return {
        "schema": "apedp.quantum_signal_power.v1",
        "floor": FLOOR,
        "reference_true_sharpes": list(REFERENCE_SHARPES),
        "method": {
            "minimum_detectable_sharpe": (
                "the true annualised Sharpe at which this design's Deflated Sharpe "
                "Ratio reaches the 0.95 floor, solved by root-finding on the same "
                "compute_deflated_sharpe_ratio the records use"),
            "power": (
                "P(the record would report a pass) when the true Sharpe is s, taking "
                "the observed Sharpe as N(s, se(s)) with the DSR's own standard error"),
            "inputs_read": "the five tranche records only; no panel, no holdout slice",
        },
        "by_record": by_record,
        "programme": programme,
        "checks": checks,
        "headline": {
            name: {resolution: {
                "minimum_detectable_sharpe": entry["minimum_detectable_sharpe"],
                "observed_dsr_max": entry["observed_dsr_max"],
                "power_at_sharpe_1": next(item["power"] for item in entry["power_curve"]
                                          if abs(item["true_sharpe"] - 1.0) < 1e-12),
            } for resolution, entry in per_resolution.items()}
            for name, per_resolution in by_record.items()
        },
        "reading": (
            "a row that fails the floor is evidence against an effect larger than its "
            "minimum detectable Sharpe, and no evidence at all about an effect smaller "
            "than it; the headline block carries both numbers so a null cannot be quoted "
            "without the threshold it was actually tested at"),
        "limitations": [
            "the MDS is computed from the median sample length and median return shape "
            "at each resolution, not per row; individual rows differ by a few percent",
            "power assumes the observed Sharpe is approximately normal around the true "
            "one, which is the approximation the DSR itself is built on",
            "the programme charge re-prices trials and dispersion but still cannot "
            "charge the choice of assets, resolutions or when each method change was "
            "made; it is a floor on the programme's multiplicity, not a complete account",
            f"Z_FLOOR = {Z_FLOOR:.6f} is quoted only for the closed-form reading; the "
            f"MDS is solved numerically from the DSR",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=RECORD_DIR / "quantum_signal_power.json")
    args = parser.parse_args()

    records = {}
    for name, path in RECORDS.items():
        if not path.is_file():
            raise SystemExit(f"missing tranche record {path}")
        records[name] = json.loads(path.read_text(encoding="utf-8"))

    record = build(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print(f"floor {record['floor']:.2f}   programme trials {record['programme']['total_trials']}")
    for name, per_resolution in record["by_record"].items():
        for resolution, entry in sorted(per_resolution.items()):
            print(f"{name:9s} {resolution:3s}  bars {entry['n_bars_median']:5d}"
                  f"  trials {entry['n_trials']:4d}"
                  f"  MDS {entry['minimum_detectable_sharpe']:6.2f}"
                  f"  best SR {entry['observed_sharpe_max']:7.2f}"
                  f"  best DSR {entry['observed_dsr_max']:.3f}"
                  f"  P(pass|SR=1) {entry['power_curve'][2]['power']:.4f}")
    print(f"rows clearing floor: as run "
          f"{record['programme']['rows_clearing_floor_as_run']}"
          f"  programme charge {record['programme']['rows_clearing_floor_programme']}"
          f"  newly significant {record['programme']['rows_that_become_significant']}")
    print(f"checks: {sum(record['checks'].values())}/{len(record['checks'])}")
    print(f"written: {args.output.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
