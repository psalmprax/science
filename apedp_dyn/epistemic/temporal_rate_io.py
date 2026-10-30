"""Clock-dataset ingestion for the temporal-rate modulation null test.

Status: experimental

Two interchange formats are supported:

* CSV plus a JSON sidecar metadata file. The CSV has ``time_s``, ``drive``,
  and one fractional-frequency column per clock. The sidecar names the dataset,
  the drive frequency, and each clock's id, composition, column, and
  systematic-uncertainty budget.
* A compressed NPZ archive whose arrays carry the same metadata. NPZ loading
  uses ``allow_pickle=False`` so a dataset file cannot execute code.

Both formats feed the same ``ClockDataset`` validation boundary: finite
values, strictly increasing time, matched lengths, and at least two samples
per clock. ``dataset_sha256`` gives calibration and confirmation runs a
canonical provenance fingerprint independent of file formatting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.conf.canonical_spec import (
    PINNED_SPEC_SHA256,
    TEMPORAL_RATE_FAMILYWISE_ALPHA,
    TEMPORAL_RATE_HARMONICS,
    TEMPORAL_RATE_HETEROGENEITY_ALPHA,
    TEMPORAL_RATE_MIN_CLOCKS,
    TEMPORAL_RATE_MIN_COMPOSITIONS,
    TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE,
    TEMPORAL_RATE_SURROGATE_DRAWS,
)
from apedp_dyn.epistemic.temporal_rate_modulation import (
    ClockChannel,
    ClockDataset,
    TemporalRateAdjudication,
    TemporalRateProtocolError,
    adjudicate_temporal_rate_modulation,
    validate_temporal_rate_pair,
)

_METADATA_KEYS = {"dataset_id", "drive_frequency_hz", "clocks"}
_CLOCK_KEYS = {
    "clock_id",
    "composition",
    "column",
    "systematic_susceptibility_sigma",
}
_NPZ_KEYS = {
    "time_s",
    "drive",
    "drive_frequency_hz",
    "dataset_id",
    "clock_ids",
    "compositions",
    "fractional_frequency",
    "systematic_susceptibility_sigma",
}
_PROTOCOL_VERSION = "temporal-rate-v2"
_PREREGISTRATION_KEYS = {
    "protocol_version",
    "created_utc",
    "calibration_sha256",
    "confirmation_sha256",
    "calibration_dataset_id",
    "confirmation_dataset_id",
    "calibration_drive_frequency_hz",
    "confirmation_drive_frequency_hz",
    "clock_compositions",
    "harmonics",
    "n_surrogates",
    "seed",
    "familywise_alpha",
    "heterogeneity_alpha",
    "min_fractional_amplitude",
    "canonical_spec_sha256",
    "notes",
}


@dataclass(frozen=True)
class TemporalRatePreregistration:
    """Frozen identity of a calibration/confirmation pair before adjudication."""

    protocol_version: str
    created_utc: str
    calibration_sha256: str
    confirmation_sha256: str
    calibration_dataset_id: str
    confirmation_dataset_id: str
    calibration_drive_frequency_hz: float
    confirmation_drive_frequency_hz: float
    clock_compositions: dict[str, str]
    harmonics: int
    n_surrogates: int
    seed: int
    familywise_alpha: float
    heterogeneity_alpha: float
    min_fractional_amplitude: float
    canonical_spec_sha256: str
    notes: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.protocol_version, str)
            or not self.protocol_version.strip()
        ):
            raise TemporalRateProtocolError(
                "preregistration protocol_version must be a non-empty string"
            )
        if not isinstance(self.created_utc, str) or not self.created_utc.strip():
            raise TemporalRateProtocolError(
                "preregistration created_utc must be a non-empty string"
            )
        for name in ("calibration_sha256", "confirmation_sha256"):
            value = getattr(self, name)
            if not _is_sha256(value):
                raise TemporalRateProtocolError(
                    f"preregistration {name} must be a 64-character hex digest"
                )
        if self.calibration_sha256 == self.confirmation_sha256:
            raise TemporalRateProtocolError(
                "preregistration calibration and confirmation fingerprints are identical"
            )
        for name in ("calibration_dataset_id", "confirmation_dataset_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TemporalRateProtocolError(
                    f"preregistration {name} must be a non-empty string"
                )
        if self.calibration_dataset_id == self.confirmation_dataset_id:
            raise TemporalRateProtocolError(
                "preregistration calibration and confirmation dataset ids are identical"
            )
        for name in (
            "calibration_drive_frequency_hz",
            "confirmation_drive_frequency_hz",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise TemporalRateProtocolError(
                    f"preregistration {name} must be finite and positive"
                )
        if (
            self.calibration_drive_frequency_hz
            == self.confirmation_drive_frequency_hz
        ):
            raise TemporalRateProtocolError(
                "preregistration calibration and confirmation drive frequencies "
                "are identical"
            )
        if not isinstance(self.clock_compositions, dict):
            raise TemporalRateProtocolError(
                "preregistration clock_compositions must be an object"
            )
        if len(self.clock_compositions) < TEMPORAL_RATE_MIN_CLOCKS:
            raise TemporalRateProtocolError(
                "preregistration clock_compositions must contain at least "
                f"{TEMPORAL_RATE_MIN_CLOCKS} clocks"
            )
        for clock_id, composition in self.clock_compositions.items():
            if (
                not isinstance(clock_id, str)
                or not clock_id.strip()
                or not isinstance(composition, str)
                or not composition.strip()
            ):
                raise TemporalRateProtocolError(
                    "preregistration clock_compositions must map non-empty "
                    "clock ids to non-empty compositions"
                )
        if (
            len(set(self.clock_compositions.values()))
            < TEMPORAL_RATE_MIN_COMPOSITIONS
        ):
            raise TemporalRateProtocolError(
                "preregistration must contain at least "
                f"{TEMPORAL_RATE_MIN_COMPOSITIONS} physically distinct clock "
                "compositions"
            )
        for name in ("harmonics", "n_surrogates", "seed"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise TemporalRateProtocolError(
                    f"preregistration {name} must be a non-negative integer"
                )
        if self.harmonics < 1:
            raise TemporalRateProtocolError(
                "preregistration harmonics must be at least 1"
            )
        if self.n_surrogates < 1:
            raise TemporalRateProtocolError(
                "preregistration n_surrogates must be at least 1"
            )
        for name in (
            "familywise_alpha",
            "heterogeneity_alpha",
            "min_fractional_amplitude",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise TemporalRateProtocolError(
                    f"preregistration {name} must be finite and positive"
                )
        for name in ("familywise_alpha", "heterogeneity_alpha"):
            if getattr(self, name) >= 1.0:
                raise TemporalRateProtocolError(
                    f"preregistration {name} must be less than 1"
                )
        if not _is_sha256(self.canonical_spec_sha256):
            raise TemporalRateProtocolError(
                "preregistration canonical_spec_sha256 must be a 64-character "
                "hex digest"
            )
        if not isinstance(self.notes, str):
            raise TemporalRateProtocolError(
                "preregistration notes must be a string"
            )
        object.__setattr__(
            self, "clock_compositions", dict(self.clock_compositions)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "created_utc": self.created_utc,
            "calibration_sha256": self.calibration_sha256,
            "confirmation_sha256": self.confirmation_sha256,
            "calibration_dataset_id": self.calibration_dataset_id,
            "confirmation_dataset_id": self.confirmation_dataset_id,
            "calibration_drive_frequency_hz": (
                self.calibration_drive_frequency_hz
            ),
            "confirmation_drive_frequency_hz": (
                self.confirmation_drive_frequency_hz
            ),
            "clock_compositions": dict(self.clock_compositions),
            "harmonics": self.harmonics,
            "n_surrogates": self.n_surrogates,
            "seed": self.seed,
            "familywise_alpha": self.familywise_alpha,
            "heterogeneity_alpha": self.heterogeneity_alpha,
            "min_fractional_amplitude": self.min_fractional_amplitude,
            "canonical_spec_sha256": self.canonical_spec_sha256,
            "notes": self.notes,
        }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def load_clock_dataset(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> ClockDataset:
    """Load a clock dataset from ``.csv`` plus sidecar JSON or ``.npz``.

    For CSV, the sidecar defaults to the same stem with a ``.json`` suffix.
    For NPZ, metadata is embedded and ``metadata_path`` must be omitted.
    """
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".npz":
        if metadata_path is not None:
            raise TemporalRateProtocolError(
                "NPZ datasets embed their metadata; metadata_path is not allowed"
            )
        return _load_npz(dataset_path)
    if suffix == ".csv":
        sidecar = (
            Path(metadata_path)
            if metadata_path is not None
            else dataset_path.with_suffix(".json")
        )
        return _load_csv(dataset_path, sidecar)
    raise TemporalRateProtocolError(
        f"unsupported clock dataset format '{suffix}' (expected .csv or .npz)"
    )


def save_clock_dataset(
    dataset: ClockDataset,
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> None:
    """Write a clock dataset as CSV plus sidecar JSON or as NPZ."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".npz":
        if metadata_path is not None:
            raise TemporalRateProtocolError(
                "NPZ datasets embed their metadata; metadata_path is not allowed"
            )
        _save_npz(dataset, dataset_path)
        return
    if suffix == ".csv":
        sidecar = (
            Path(metadata_path)
            if metadata_path is not None
            else dataset_path.with_suffix(".json")
        )
        _save_csv(dataset, dataset_path, sidecar)
        return
    raise TemporalRateProtocolError(
        f"unsupported clock dataset format '{suffix}' (expected .csv or .npz)"
    )


def dataset_sha256(dataset: ClockDataset) -> str:
    """Return a canonical SHA-256 fingerprint of a clock dataset."""
    digest = hashlib.sha256()
    _update_text(digest, "temporal-rate-dataset-v1")
    _update_text(digest, dataset.dataset_id)
    _update_array(digest, np.array([dataset.drive_frequency_hz]))
    _update_array(digest, dataset.time_s)
    _update_array(digest, dataset.drive)
    _update_array(digest, np.array([len(dataset.channels)]))
    for channel in dataset.channels:
        _update_text(digest, channel.clock_id)
        _update_text(digest, channel.composition)
        _update_array(
            digest, np.array([channel.systematic_susceptibility_sigma])
        )
        _update_array(digest, channel.fractional_frequency)
    return digest.hexdigest()


def preregister_datasets(
    calibration: ClockDataset,
    confirmation: ClockDataset,
    *,
    harmonics: int | None = None,
    n_surrogates: int | None = None,
    seed: int = 0,
    notes: str = "",
    created_utc: str | None = None,
) -> TemporalRatePreregistration:
    """Freeze data identity and the full analysis configuration.

    ``harmonics``, ``n_surrogates``, and ``seed`` are recorded in the
    artifact and can no longer be changed at adjudication time. The remaining
    analysis thresholds are canonical and are bound by the canonical-spec
    SHA-256, so a later re-pin invalidates the preregistration.
    """
    harmonics = (
        TEMPORAL_RATE_HARMONICS if harmonics is None else int(harmonics)
    )
    n_surrogates = (
        TEMPORAL_RATE_SURROGATE_DRAWS
        if n_surrogates is None
        else int(n_surrogates)
    )
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    validate_temporal_rate_pair(
        calibration, confirmation, harmonics=harmonics
    )
    return TemporalRatePreregistration(
        protocol_version=_PROTOCOL_VERSION,
        created_utc=(
            created_utc
            if created_utc is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        calibration_sha256=dataset_sha256(calibration),
        confirmation_sha256=dataset_sha256(confirmation),
        calibration_dataset_id=calibration.dataset_id,
        confirmation_dataset_id=confirmation.dataset_id,
        calibration_drive_frequency_hz=float(calibration.drive_frequency_hz),
        confirmation_drive_frequency_hz=float(confirmation.drive_frequency_hz),
        clock_compositions={
            channel.clock_id: channel.composition
            for channel in confirmation.channels
        },
        harmonics=harmonics,
        n_surrogates=n_surrogates,
        seed=seed,
        familywise_alpha=TEMPORAL_RATE_FAMILYWISE_ALPHA,
        heterogeneity_alpha=TEMPORAL_RATE_HETEROGENEITY_ALPHA,
        min_fractional_amplitude=TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE,
        canonical_spec_sha256=PINNED_SPEC_SHA256,
        notes=notes,
    )


def save_preregistration(
    preregistration: TemporalRatePreregistration,
    path: str | Path,
) -> None:
    """Write a preregistration artifact as deterministic JSON."""
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(preregistration.to_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise TemporalRateProtocolError(
            f"could not write preregistration '{destination}': {exc}"
        ) from exc


def load_preregistration(path: str | Path) -> TemporalRatePreregistration:
    """Load and validate a preregistration artifact."""
    source = Path(path)
    payload = _read_json(source, "preregistration")
    if not isinstance(payload, dict):
        raise TemporalRateProtocolError(
            "preregistration must be a JSON object"
        )
    keys = set(payload)
    missing = _PREREGISTRATION_KEYS - keys
    unknown = keys - _PREREGISTRATION_KEYS
    if missing or unknown:
        raise TemporalRateProtocolError(
            "preregistration must contain exactly "
            f"{sorted(_PREREGISTRATION_KEYS)}; missing={sorted(missing)} "
            f"unknown={sorted(unknown)}"
        )
    try:
        return TemporalRatePreregistration(
            protocol_version=payload["protocol_version"],
            created_utc=payload["created_utc"],
            calibration_sha256=payload["calibration_sha256"],
            confirmation_sha256=payload["confirmation_sha256"],
            calibration_dataset_id=payload["calibration_dataset_id"],
            confirmation_dataset_id=payload["confirmation_dataset_id"],
            calibration_drive_frequency_hz=float(
                payload["calibration_drive_frequency_hz"]
            ),
            confirmation_drive_frequency_hz=float(
                payload["confirmation_drive_frequency_hz"]
            ),
            clock_compositions=payload["clock_compositions"],
            harmonics=payload["harmonics"],
            n_surrogates=payload["n_surrogates"],
            seed=payload["seed"],
            familywise_alpha=payload["familywise_alpha"],
            heterogeneity_alpha=payload["heterogeneity_alpha"],
            min_fractional_amplitude=payload["min_fractional_amplitude"],
            canonical_spec_sha256=payload["canonical_spec_sha256"],
            notes=payload["notes"],
        )
    except TemporalRateProtocolError:
        raise
    except (TypeError, ValueError) as exc:
        raise TemporalRateProtocolError(
            f"invalid preregistration '{source}': {exc}"
        ) from exc


def verify_preregistration(
    preregistration: TemporalRatePreregistration,
    calibration: ClockDataset,
    confirmation: ClockDataset,
) -> None:
    """Fail closed if either dataset no longer matches the preregistration."""
    if preregistration.protocol_version != _PROTOCOL_VERSION:
        raise TemporalRateProtocolError(
            "preregistration protocol_version "
            f"'{preregistration.protocol_version}' is not supported; expected "
            f"'{_PROTOCOL_VERSION}'"
        )
    mismatches: list[str] = []
    if preregistration.calibration_dataset_id != calibration.dataset_id:
        mismatches.append("calibration_dataset_id")
    if preregistration.confirmation_dataset_id != confirmation.dataset_id:
        mismatches.append("confirmation_dataset_id")
    if (
        preregistration.calibration_drive_frequency_hz
        != calibration.drive_frequency_hz
    ):
        mismatches.append("calibration_drive_frequency_hz")
    if (
        preregistration.confirmation_drive_frequency_hz
        != confirmation.drive_frequency_hz
    ):
        mismatches.append("confirmation_drive_frequency_hz")
    compositions = {
        channel.clock_id: channel.composition
        for channel in confirmation.channels
    }
    if preregistration.clock_compositions != compositions:
        mismatches.append("clock_compositions")
    if preregistration.calibration_sha256 != dataset_sha256(calibration):
        mismatches.append("calibration_sha256")
    if preregistration.confirmation_sha256 != dataset_sha256(confirmation):
        mismatches.append("confirmation_sha256")
    if preregistration.canonical_spec_sha256 != PINNED_SPEC_SHA256:
        mismatches.append("canonical_spec_sha256")
    if preregistration.familywise_alpha != TEMPORAL_RATE_FAMILYWISE_ALPHA:
        mismatches.append("familywise_alpha")
    if preregistration.heterogeneity_alpha != TEMPORAL_RATE_HETEROGENEITY_ALPHA:
        mismatches.append("heterogeneity_alpha")
    if (
        preregistration.min_fractional_amplitude
        != TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE
    ):
        mismatches.append("min_fractional_amplitude")
    if mismatches:
        raise TemporalRateProtocolError(
            "preregistration mismatch for " + ", ".join(mismatches)
        )


def adjudicate_preregistered_temporal_rate(
    preregistration: TemporalRatePreregistration,
    calibration: ClockDataset,
    confirmation: ClockDataset,
) -> TemporalRateAdjudication:
    """Verify the preregistered fingerprints, then adjudicate.

    The analysis parameters are taken from the artifact. They are deliberately
    not overridable here: changing the seed, surrogate count, or harmonic count
    after seeing the confirmation data would reintroduce the researcher degree
    of freedom the preregistration exists to remove. To change them, create a
    new preregistration before adjudication.
    """
    verify_preregistration(preregistration, calibration, confirmation)
    return adjudicate_temporal_rate_modulation(
        calibration,
        confirmation,
        harmonics=preregistration.harmonics,
        n_surrogates=preregistration.n_surrogates,
        seed=preregistration.seed,
    )


def _update_text(digest: Any, text: str) -> None:
    encoded = text.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def _update_array(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest.update(struct.pack("<Q", array.ndim))
    for dimension in array.shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(array.tobytes(order="C"))


def _load_csv(dataset_path: Path, metadata_path: Path) -> ClockDataset:
    metadata = _read_json(metadata_path, "metadata")
    dataset_id, drive_frequency_hz, clock_specs = _parse_metadata(metadata)
    columns = [spec["column"] for spec in clock_specs]

    try:
        with dataset_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            if not fieldnames:
                raise TemporalRateProtocolError(
                    f"CSV dataset '{dataset_path}' has no header"
                )
            if len(set(fieldnames)) != len(fieldnames):
                raise TemporalRateProtocolError(
                    f"CSV dataset '{dataset_path}' has duplicate column names"
                )
            required = ["time_s", "drive", *columns]
            missing = [name for name in required if name not in fieldnames]
            if missing:
                raise TemporalRateProtocolError(
                    f"CSV dataset '{dataset_path}' is missing columns {missing}"
                )
            rows = list(reader)
    except OSError as exc:
        raise TemporalRateProtocolError(
            f"could not read CSV dataset '{dataset_path}': {exc}"
        ) from exc
    except csv.Error as exc:
        raise TemporalRateProtocolError(
            f"CSV dataset '{dataset_path}' is malformed: {exc}"
        ) from exc

    if len(rows) < 2:
        raise TemporalRateProtocolError(
            f"CSV dataset '{dataset_path}' has fewer than two rows"
        )

    time_s = np.array(
        [_cell(row, "time_s", row_index, dataset_path) for row_index, row in enumerate(rows)],
        dtype=float,
    )
    drive = np.array(
        [_cell(row, "drive", row_index, dataset_path) for row_index, row in enumerate(rows)],
        dtype=float,
    )
    channels = []
    for spec in clock_specs:
        column = str(spec["column"])
        values = np.array(
            [
                _cell(row, column, row_index, dataset_path)
                for row_index, row in enumerate(rows)
            ],
            dtype=float,
        )
        channels.append(
            ClockChannel(
                clock_id=str(spec["clock_id"]),
                composition=str(spec["composition"]),
                fractional_frequency=values,
                systematic_susceptibility_sigma=float(
                    spec["systematic_susceptibility_sigma"]
                ),
            )
        )
    return ClockDataset(
        dataset_id=dataset_id,
        time_s=time_s,
        drive=drive,
        channels=tuple(channels),
        drive_frequency_hz=drive_frequency_hz,
    )


def _cell(
    row: dict[str, str | None],
    column: str,
    row_index: int,
    dataset_path: Path,
) -> float:
    raw = row.get(column)
    if raw is None or raw.strip() == "":
        raise TemporalRateProtocolError(
            f"CSV dataset '{dataset_path}' row {row_index + 2} column "
            f"'{column}' is empty"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise TemporalRateProtocolError(
            f"CSV dataset '{dataset_path}' row {row_index + 2} column "
            f"'{column}' is not a number"
        ) from exc
    if not np.isfinite(value):
        raise TemporalRateProtocolError(
            f"CSV dataset '{dataset_path}' row {row_index + 2} column "
            f"'{column}' is not finite"
        )
    return value


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TemporalRateProtocolError(
            f"could not read {label} file '{path}': {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TemporalRateProtocolError(
            f"{label} file '{path}' is not valid JSON: {exc}"
        ) from exc
    except UnicodeError as exc:
        raise TemporalRateProtocolError(
            f"{label} file '{path}' is not UTF-8 text: {exc}"
        ) from exc


def _parse_metadata(
    metadata: Any,
) -> tuple[str, float, list[dict[str, Any]]]:
    if not isinstance(metadata, dict):
        raise TemporalRateProtocolError("metadata must be a JSON object")
    keys = set(metadata)
    missing = _METADATA_KEYS - keys
    unknown = keys - _METADATA_KEYS
    if missing or unknown:
        raise TemporalRateProtocolError(
            "metadata must contain exactly "
            f"{sorted(_METADATA_KEYS)}; missing={sorted(missing)} "
            f"unknown={sorted(unknown)}"
        )

    dataset_id = metadata["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise TemporalRateProtocolError("metadata 'dataset_id' must be a non-empty string")
    dataset_id = dataset_id.strip()

    try:
        drive_frequency_hz = float(metadata["drive_frequency_hz"])
    except (TypeError, ValueError) as exc:
        raise TemporalRateProtocolError(
            "metadata 'drive_frequency_hz' must be a number"
        ) from exc
    if not np.isfinite(drive_frequency_hz) or drive_frequency_hz <= 0.0:
        raise TemporalRateProtocolError(
            "metadata 'drive_frequency_hz' must be finite and positive"
        )

    clocks = metadata["clocks"]
    if not isinstance(clocks, list) or not clocks:
        raise TemporalRateProtocolError("metadata 'clocks' must be a non-empty list")

    clock_specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_columns: set[str] = set()
    for index, entry in enumerate(clocks):
        if not isinstance(entry, dict):
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}] must be an object"
            )
        entry_keys = set(entry)
        missing = _CLOCK_KEYS - entry_keys
        unknown = entry_keys - _CLOCK_KEYS
        if missing or unknown:
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}] must contain exactly "
                f"{sorted(_CLOCK_KEYS)}; missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        clock_id = entry["clock_id"]
        composition = entry["composition"]
        column = entry["column"]
        if not isinstance(clock_id, str) or not clock_id.strip():
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}].clock_id must be a non-empty string"
            )
        if not isinstance(composition, str) or not composition.strip():
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}].composition must be a non-empty string"
            )
        if not isinstance(column, str) or not column.strip():
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}].column must be a non-empty string"
            )
        clock_id = clock_id.strip()
        composition = composition.strip()
        column = column.strip()
        if column in {"time_s", "drive"}:
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}].column '{column}' is reserved"
            )
        if clock_id in seen_ids:
            raise TemporalRateProtocolError(
                f"metadata contains duplicate clock_id '{clock_id}'"
            )
        if column in seen_columns:
            raise TemporalRateProtocolError(
                f"metadata contains duplicate column '{column}'"
            )
        try:
            systematic = float(entry["systematic_susceptibility_sigma"])
        except (TypeError, ValueError) as exc:
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}].systematic_susceptibility_sigma "
                "must be a number"
            ) from exc
        if not np.isfinite(systematic) or systematic < 0.0:
            raise TemporalRateProtocolError(
                f"metadata clocks[{index}].systematic_susceptibility_sigma "
                "must be finite and non-negative"
            )
        seen_ids.add(clock_id)
        seen_columns.add(column)
        clock_specs.append(
            {
                "clock_id": clock_id,
                "composition": composition,
                "column": column,
                "systematic_susceptibility_sigma": systematic,
            }
        )
    return dataset_id, drive_frequency_hz, clock_specs


def _save_csv(dataset: ClockDataset, dataset_path: Path, metadata_path: Path) -> None:
    columns = [
        f"{channel.clock_id}_fractional_frequency"
        for channel in dataset.channels
    ]
    metadata = {
        "dataset_id": dataset.dataset_id,
        "drive_frequency_hz": dataset.drive_frequency_hz,
        "clocks": [
            {
                "clock_id": channel.clock_id,
                "composition": channel.composition,
                "column": column,
                "systematic_susceptibility_sigma": (
                    channel.systematic_susceptibility_sigma
                ),
            }
            for channel, column in zip(dataset.channels, columns, strict=True)
        ],
    }
    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "drive", *columns])
            for row in range(dataset.time_s.size):
                writer.writerow(
                    [
                        repr(float(dataset.time_s[row])),
                        repr(float(dataset.drive[row])),
                        *(
                            repr(float(channel.fractional_frequency[row]))
                            for channel in dataset.channels
                        ),
                    ]
                )
    except OSError as exc:
        raise TemporalRateProtocolError(
            f"could not write CSV dataset '{dataset_path}': {exc}"
        ) from exc


def _save_npz(dataset: ClockDataset, dataset_path: Path) -> None:
    try:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            time_s=dataset.time_s,
            drive=dataset.drive,
            drive_frequency_hz=np.float64(dataset.drive_frequency_hz),
            dataset_id=np.array(dataset.dataset_id),
            clock_ids=np.array([channel.clock_id for channel in dataset.channels]),
            compositions=np.array(
                [channel.composition for channel in dataset.channels]
            ),
            fractional_frequency=np.vstack(
                [channel.fractional_frequency for channel in dataset.channels]
            ),
            systematic_susceptibility_sigma=np.array(
                [
                    channel.systematic_susceptibility_sigma
                    for channel in dataset.channels
                ],
                dtype=float,
            ),
        )
    except OSError as exc:
        raise TemporalRateProtocolError(
            f"could not write NPZ dataset '{dataset_path}': {exc}"
        ) from exc


def _load_npz(dataset_path: Path) -> ClockDataset:
    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = _NPZ_KEYS - keys
            unknown = keys - _NPZ_KEYS
            if missing or unknown:
                raise TemporalRateProtocolError(
                    f"NPZ dataset '{dataset_path}' must contain exactly "
                    f"{sorted(_NPZ_KEYS)}; missing={sorted(missing)} "
                    f"unknown={sorted(unknown)}"
                )
            arrays = {
                key: np.array(archive[key], copy=True) for key in _NPZ_KEYS
            }
    except TemporalRateProtocolError:
        raise
    except OSError as exc:
        raise TemporalRateProtocolError(
            f"could not read NPZ dataset '{dataset_path}': {exc}"
        ) from exc
    except ValueError as exc:
        raise TemporalRateProtocolError(
            f"could not read NPZ dataset '{dataset_path}' without pickle: {exc}"
        ) from exc

    time_s = _as_1d(arrays["time_s"], "time_s", dataset_path)
    drive = _as_1d(arrays["drive"], "drive", dataset_path)
    if time_s.size != drive.size:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' time_s and drive lengths differ"
        )
    fractional = np.asarray(arrays["fractional_frequency"], dtype=float)
    if fractional.ndim != 2:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' fractional_frequency must be 2-D"
        )
    systematics = _as_float_vector(
        arrays["systematic_susceptibility_sigma"],
        "systematic_susceptibility_sigma",
        dataset_path,
    )
    clock_ids = _as_strings(arrays["clock_ids"], "clock_ids", dataset_path)
    compositions = _as_strings(
        arrays["compositions"], "compositions", dataset_path
    )
    n_clocks = clock_ids.size
    if (
        compositions.size != n_clocks
        or systematics.size != n_clocks
        or fractional.shape[0] != n_clocks
        or fractional.shape[1] != time_s.size
    ):
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' clock arrays have inconsistent shapes"
        )
    if len(set(clock_ids.tolist())) != n_clocks:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' contains duplicate clock ids"
        )
    channels = tuple(
        ClockChannel(
            clock_id=str(clock_ids[index]),
            composition=str(compositions[index]),
            fractional_frequency=fractional[index],
            systematic_susceptibility_sigma=float(systematics[index]),
        )
        for index in range(n_clocks)
    )
    return ClockDataset(
        dataset_id=_as_string(arrays["dataset_id"], "dataset_id", dataset_path),
        time_s=time_s,
        drive=drive,
        channels=channels,
        drive_frequency_hz=_as_scalar(
            arrays["drive_frequency_hz"], "drive_frequency_hz", dataset_path
        ),
    )


def _as_scalar(values: np.ndarray, name: str, dataset_path: Path) -> float:
    array = np.asarray(values)
    if array.size != 1:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be a scalar"
        )
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be numeric"
        ) from exc


def _as_1d(values: np.ndarray, name: str, dataset_path: Path) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be 1-D"
        )
    array = np.asarray(array, dtype=float)
    if array.size < 2:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} has fewer than two samples"
        )
    return array


def _as_float_vector(
    values: np.ndarray, name: str, dataset_path: Path
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be a non-empty 1-D array"
        )
    return np.asarray(array, dtype=float)


def _as_strings(values: np.ndarray, name: str, dataset_path: Path) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be a non-empty 1-D array"
        )
    if array.dtype.kind not in {"U", "S"}:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be strings"
        )
    if array.dtype.kind == "S":
        text = np.array(
            [item.decode("utf-8") for item in array], dtype=str
        )
    else:
        text = array.astype(str)
    if any(not item.strip() for item in text):
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} contains an empty string"
        )
    return text


def _as_string(values: np.ndarray, name: str, dataset_path: Path) -> str:
    array = np.asarray(values)
    if array.size != 1:
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be a scalar string"
        )
    item = array.reshape(-1)[0]
    if not isinstance(item, (str, np.str_, np.bytes_)):
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be a string"
        )
    text = item.decode("utf-8") if isinstance(item, np.bytes_) else str(item)
    if not text.strip():
        raise TemporalRateProtocolError(
            f"NPZ dataset '{dataset_path}' {name} must be non-empty"
        )
    return text.strip()
