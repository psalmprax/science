"""Run the Hypothesis B temporal-rate null-test demonstration.

The script generates synthetic clock runs, adjudicates a null, a common-mode
injection, and an oscillator-specific injection, and optionally measures the
detector's false-positive rate and power. It is a detector demonstration, not
a measurement of nature.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.conf.canonical_spec import (  # noqa: E402
    TEMPORAL_RATE_CALIBRATION_TRIALS,
    TEMPORAL_RATE_SURROGATE_DRAWS,
)
from apedp_dyn.epistemic.temporal_rate_io import (  # noqa: E402
    adjudicate_preregistered_temporal_rate,
    dataset_sha256,
    load_clock_dataset,
    load_preregistration,
    preregister_datasets,
    save_preregistration,
)
from apedp_dyn.epistemic.temporal_rate_modulation import (  # noqa: E402
    TemporalRateProtocolError,
    adjudicate_temporal_rate_modulation,
    run_detector_calibration,
    simulate_clock_dataset,
)


def _emit(payload: dict[str, object], json_out: Path | None) -> None:
    text = json.dumps(payload, indent=2)
    print(text)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-frequency-hz", type=float, default=1.0)
    parser.add_argument("--calibration-frequency-hz", type=float, default=0.83)
    parser.add_argument("--duration-s", type=float, default=128.0)
    parser.add_argument("--sample-rate-hz", type=float, default=16.0)
    parser.add_argument("--noise-sigma", type=float, default=1e-16)
    parser.add_argument("--injection-amplitude", type=float, default=1e-15)
    parser.add_argument(
        "--harmonics", type=int, default=None, help="drive harmonics tested"
    )
    parser.add_argument(
        "--surrogates", type=int, default=TEMPORAL_RATE_SURROGATE_DRAWS
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--calibration-trials",
        type=int,
        default=TEMPORAL_RATE_CALIBRATION_TRIALS,
    )
    parser.add_argument(
        "--calibration-surrogates",
        type=int,
        default=TEMPORAL_RATE_SURROGATE_DRAWS,
    )
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--confirmation", type=Path, default=None)
    parser.add_argument("--calibration-metadata", type=Path, default=None)
    parser.add_argument("--confirmation-metadata", type=Path, default=None)
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--write-preregistration", type=Path, default=None)
    parser.add_argument("--preregistration-notes", type=str, default="")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if (args.calibration is None) != (args.confirmation is None):
        parser.error("--calibration and --confirmation must be supplied together")
    if (
        args.preregistration is not None
        and args.write_preregistration is not None
    ):
        parser.error(
            "--preregistration and --write-preregistration are mutually "
            "exclusive"
        )
    if (
        args.preregistration is not None
        or args.write_preregistration is not None
    ) and args.calibration is None:
        parser.error(
            "--preregistration options require --calibration and "
            "--confirmation"
        )
    if args.calibration is not None:
        calibration = load_clock_dataset(
            args.calibration, metadata_path=args.calibration_metadata
        )
        confirmation = load_clock_dataset(
            args.confirmation, metadata_path=args.confirmation_metadata
        )
        if args.write_preregistration is not None:
            preregistration = preregister_datasets(
                calibration,
                confirmation,
                harmonics=args.harmonics,
                n_surrogates=args.surrogates,
                seed=args.seed,
                notes=args.preregistration_notes,
            )
            save_preregistration(
                preregistration, args.write_preregistration
            )
            _emit(
                {
                    "note": (
                        "Preregistered dataset fingerprints; no adjudication "
                        "was run."
                    ),
                    "calibration_sha256": dataset_sha256(calibration),
                    "confirmation_sha256": dataset_sha256(confirmation),
                    "preregistration": preregistration.to_dict(),
                },
                args.json_out,
            )
            return

        preregistration = None
        if args.preregistration is not None:
            preregistration = load_preregistration(args.preregistration)
            result = adjudicate_preregistered_temporal_rate(
                preregistration,
                calibration,
                confirmation,
            )
        else:
            result = adjudicate_temporal_rate_modulation(
                calibration,
                confirmation,
                harmonics=args.harmonics,
                n_surrogates=args.surrogates,
                seed=args.seed,
            )
        payload: dict[str, object] = {
            "note": (
                "Adjudication of supplied clock datasets; a positive "
                "verdict is not a time-travel claim."
            ),
            "calibration_sha256": dataset_sha256(calibration),
            "confirmation_sha256": dataset_sha256(confirmation),
            "result": result.to_dict(),
        }
        if preregistration is not None:
            payload["preregistration"] = preregistration.to_dict()
        _emit(payload, args.json_out)
        return

    calibration = simulate_clock_dataset(
        dataset_id="calibration",
        drive_frequency_hz=args.calibration_frequency_hz,
        duration_s=args.duration_s,
        sample_rate_hz=args.sample_rate_hz,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    cases = {
        "null": simulate_clock_dataset(
            dataset_id="null",
            drive_frequency_hz=args.drive_frequency_hz,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            noise_sigma=args.noise_sigma,
            seed=args.seed + 1,
        ),
        "common_mode_injection": simulate_clock_dataset(
            dataset_id="common-mode-injection",
            drive_frequency_hz=args.drive_frequency_hz,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            anomaly_amplitude=args.injection_amplitude,
            anomaly_phase_rad=1.1,
            noise_sigma=args.noise_sigma,
            seed=args.seed + 2,
        ),
        "oscillator_specific_injection": simulate_clock_dataset(
            dataset_id="oscillator-specific-injection",
            drive_frequency_hz=args.drive_frequency_hz,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            anomaly_amplitude=args.injection_amplitude,
            anomaly_common_mode=False,
            anomaly_clock_index=0,
            noise_sigma=args.noise_sigma,
            seed=args.seed + 3,
        ),
    }

    payload: dict[str, object] = {
        "note": "Synthetic detector demonstration; not a measurement of nature.",
        "results": {
            name: adjudicate_temporal_rate_modulation(
                calibration,
                dataset,
                harmonics=args.harmonics,
                n_surrogates=args.surrogates,
                seed=args.seed,
            ).to_dict()
            for name, dataset in cases.items()
        },
    }
    if args.calibrate:
        payload["detector_calibration"] = run_detector_calibration(
            n_trials=args.calibration_trials,
            n_surrogates=args.calibration_surrogates,
            injection_amplitude=args.injection_amplitude,
            seed=args.seed,
            drive_frequency_hz=args.drive_frequency_hz,
            calibration_frequency_hz=args.calibration_frequency_hz,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            noise_sigma=args.noise_sigma,
        ).to_dict()

    _emit(payload, args.json_out)


if __name__ == "__main__":
    try:
        main()
    except TemporalRateProtocolError as exc:
        raise SystemExit(f"temporal-rate protocol error: {exc}") from exc
